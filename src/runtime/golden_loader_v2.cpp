// CUDALM — CUDLMG02 golden container loader implementation. No PyTorch.
// See include/cudalm/golden_loader_v2.h (layout) and
// tools/common/golden_v2.py (the Python writer mirror).

#include "cudalm/golden_loader_v2.h"

#include <cctype>
#include <cstdio>
#include <set>

namespace cudalm {

namespace {

bool name_charset_ok(const char* s, std::size_t len) {
  for (std::size_t i = 0; i < len; ++i) {
    const char c = s[i];
    const bool ok = std::isalnum(static_cast<unsigned char>(c)) != 0 ||
                    c == '.' || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

bool align_is_power_of_two(std::uint8_t a) {
  return a != 0 && (a & (a - 1)) == 0 && a <= 16;
}

Dtype dtype_from_u8(std::uint8_t d) {
  // CUDALMG02 tensor dtype allowlist (all Qwen3.5 goldens are bf16; the
  // parser is written against the full v2 allowlist for uniformity).
  switch (d) {
    case static_cast<std::uint8_t>(Dtype::kFp16):
    case static_cast<std::uint8_t>(Dtype::kInt4Packed):
    case static_cast<std::uint8_t>(Dtype::kFp16Scale):
    case static_cast<std::uint8_t>(Dtype::kFp32):
    case static_cast<std::uint8_t>(Dtype::kBf16):
      return static_cast<Dtype>(d);
    default:
      return Dtype::kFp16;  // invalid sentinel; checked by caller
  }
}

}  // namespace

Status GoldenFileV2::load(const std::string& path, GoldenFileV2* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return Status::error("cannot open " + path);
  if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return Status::error("seek failed"); }
  const long sz = std::ftell(f);
  if (sz < 0) { std::fclose(f); return Status::error("ftell failed"); }
  std::rewind(f);
  std::vector<std::uint8_t> buf(static_cast<std::size_t>(sz));
  if (sz > 0 && std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
    std::fclose(f);
    return Status::error("short read on " + path);
  }
  std::fclose(f);

  const std::uint8_t* p = buf.data();
  const std::size_t n = buf.size();

  if (n < gfmt2::kHeaderBytes)
    return Status::error("file too small for CUDLMG02 header");
  if (std::memcmp(p, gfmt2::kMagic, 8) != 0)
    return Status::error("bad magic (not CUDLMG02)");
  const std::uint32_t version = wfmt2::rd_u32(p + 8);
  if (version != gfmt2::kVersion)
    return Status::error("unsupported version " + std::to_string(version));
  if (wfmt2::rd_u32(p + 12) != 0)
    return Status::error("reserved flags must be 0");
  const std::uint32_t num_tensors = wfmt2::rd_u32(p + 16);
  if (wfmt2::rd_u32(p + 20) != 0)
    return Status::error("reserved pad must be 0");

  // Fixed header words.
  const Qwen35Config cfg = wfmt2::config_from_blob(p + 24);
  if (!wfmt2::config_blob_reserved_ok(p + 24))
    return Status::error("config blob reserved word must be 0");
  if (!cfg.valid())
    return Status::error("Qwen35Config blob failed validation");
  const int position = wfmt2::rd_i32(p + gfmt2::kPositionOffset);
  const int layer_idx = wfmt2::rd_i32(p + gfmt2::kLayerIdxOffset);
  const std::int32_t input_seed = wfmt2::rd_i32(p + gfmt2::kInputSeedOffset);
  if (wfmt2::rd_i32(p + gfmt2::kReservedOffset) != 0)
    return Status::error("reserved header word must be 0");
  if (position < 0 ||
      position >= static_cast<int>(cfg.max_seq_len))
    return Status::error("position out of bounds [0, max_seq_len)");
  if (layer_idx < 0 || layer_idx >= cfg.num_hidden_layers)
    return Status::error("layer_idx out of bounds");
  if (!cfg.is_full_attention(layer_idx) &&
      !cfg.is_linear_attention(layer_idx))
    return Status::error("layer_idx is not a valid layer type "
                         "(Phase B full-attention or Phase C Gated DeltaNet)");

  const std::uint64_t table_offset = wfmt2::rd_u64(p + gfmt2::kTableOffsetField);
  const std::uint64_t payload_offset =
      wfmt2::rd_u64(p + gfmt2::kPayloadOffsetField);
  if (table_offset != gfmt2::kHeaderBytes)
    return Status::error("table_offset must be " +
                         std::to_string(gfmt2::kHeaderBytes));
  const std::uint64_t file_size = n;
  if (payload_offset > file_size)
    return Status::error("payload_offset out of bounds");
  const std::uint64_t payload_size = file_size - payload_offset;
  if (payload_offset % wfmt2::kPayloadAlign != 0)
    return Status::error("payload_offset must be 16B-aligned");
  if (num_tensors > wfmt2::kMaxNumTensors)
    return Status::error("num_tensors exceeds limit");

  // ---- tensor table (same record format as the .cudalm v2 container) ------
  std::vector<GoldenV2TensorInfo> tensors;
  std::set<std::string> seen;
  {
    std::size_t pos = static_cast<std::size_t>(table_offset);
    const std::size_t end = static_cast<std::size_t>(payload_offset);
    for (std::uint32_t i = 0; i < num_tensors; ++i) {
      GoldenV2TensorInfo info;
      if (pos + 1 > end) return Status::error("tensor table: name_len out of bounds");
      const std::size_t nlen = p[pos];
      pos += 1;
      if (nlen == 0 || nlen > wfmt2::kMaxNameLen)
        return Status::error("tensor table: bad name_len");
      // kEntryFixedBytes includes the name_len byte already consumed above.
      if (pos + nlen + (wfmt2::kEntryFixedBytes - 1) > end)
        return Status::error("tensor table: entry out of bounds");
      info.name.assign(reinterpret_cast<const char*>(p + pos), nlen);
      pos += nlen;
      if (!name_charset_ok(info.name.data(), nlen))
        return Status::error("tensor table: bad name charset in '" +
                             info.name + "'");
      if (!seen.insert(info.name).second)
        return Status::error("duplicate tensor name '" + info.name + "'");
      const std::uint8_t dtype_u = p[pos];
      pos += 1;
      const Dtype dt = dtype_from_u8(dtype_u);
      if (dt == Dtype::kFp16 && dtype_u != static_cast<std::uint8_t>(Dtype::kFp16))
        return Status::error("bad dtype in tensor '" + info.name + "'");
      info.dtype = dt;
      const std::uint8_t ndim = p[pos];
      pos += 1;
      if (ndim < 1 || ndim > 8)
        return Status::error("bad ndim in tensor '" + info.name + "'");
      if (wfmt2::rd_u16(p + pos) != 0)
        return Status::error("tensor '" + info.name +
                             "': reserved pad must be 0");
      pos += 2;
      for (int d = 0; d < 8; ++d) {
        const std::int64_t dv = wfmt2::rd_i64(p + pos + 8 * d);
        if (d < ndim) {
          if (dv <= 0 || dv > static_cast<std::int64_t>(wfmt2::kMaxDimValue))
            return Status::error("bad dim in tensor '" + info.name + "'");
          info.dims.push_back(dv);
        } else if (dv != 0) {
          return Status::error("tensor '" + info.name +
                               "': unused dim must be 0");
        }
      }
      pos += 8 * 8;
      info.offset = wfmt2::rd_u64(p + pos);
      pos += 8;
      info.byte_size = wfmt2::rd_u64(p + pos);
      pos += 8;
      info.align = p[pos];
      pos += 1;
      if (p[pos] != 0)
        return Status::error("tensor '" + info.name +
                             "': reserved pad must be 0");
      pos += 1;
      if (!align_is_power_of_two(info.align))
        return Status::error("tensor '" + info.name + "': bad align");
      if (info.offset > payload_size ||
          info.byte_size > payload_size - info.offset)
        return Status::error("tensor '" + info.name + "' out of payload bounds");
      if ((payload_offset + info.offset) % info.align != 0)
        return Status::error("tensor '" + info.name + "' not " +
                             std::to_string(info.align) + "B-aligned");
      std::vector<std::int64_t> elem_dims = info.dims;
      if (info.dtype == Dtype::kInt4Packed) {
        // Shape is expressed in PACKED bytes (last dim = K/2), as in v1/v2.
      }
      const std::uint64_t expect = wfmt::expected_bytes(info.dtype, elem_dims);
      if (info.byte_size != expect)
        return Status::error("tensor '" + info.name +
                             "': byte_size != expected for dtype/shape");
      info.host_bytes = p + payload_offset + info.offset;
      tensors.push_back(std::move(info));
    }
    // The table region is [table_offset, payload_offset); the n records
    // occupy its front and the rest is zero padding to the 16B-aligned
    // payload_offset (the CUDLMG02 header carries no table_size field).
    for (std::size_t i = pos; i < end; ++i)
      if (p[i] != 0)
        return Status::error("tensor table: non-zero padding before "
                             "payload_offset");
    if (tensors.size() != num_tensors)
      return Status::error("tensor table count mismatch");
  }

  out->pool_ = std::move(buf);
  out->payload_offset_ = payload_offset;
  out->payload_size_ = payload_size;
  out->config_ = cfg;
  out->position_ = position;
  out->layer_idx_ = layer_idx;
  out->input_seed_ = input_seed;
  out->tensors_ = std::move(tensors);
  return Status::ok_status();
}

const GoldenV2TensorInfo* GoldenFileV2::find(const std::string& name) const {
  for (const auto& t : tensors_)
    if (t.name == name) return &t;
  return nullptr;
}

namespace {

Status expect_tensor(const GoldenFileV2& f, const std::string& name,
                     Dtype dtype, const std::vector<std::int64_t>& dims) {
  const GoldenV2TensorInfo* t = f.find(name);
  if (!t) return Status::error("missing tensor '" + name + "'");
  if (t->dtype != dtype)
    return Status::error("dtype mismatch for '" + name + "' (expected " +
                         dtype_name(dtype) + ")");
  if (t->dims != dims)
    return Status::error("shape mismatch for '" + name +
                         "' (expected " + std::to_string(dims.size()) + "-D)");
  return Status::ok_status();
}

}  // namespace

Status GoldenFileV2::validate_golden_tensors() const {
  const Qwen35Config& c = config_;
  const int H = c.hidden_size;
  const int inter = c.intermediate_size;

  auto expect = [&](const std::string& n, const std::vector<std::int64_t>& d) {
    return expect_tensor(*this, n, Dtype::kBf16, d);
  };
  auto expect_f32 = [&](const std::string& n,
                        const std::vector<std::int64_t>& d) {
    return expect_tensor(*this, n, Dtype::kFp32, d);
  };

  if (c.is_linear_attention(layer_idx_)) {
    // Phase C: Gated DeltaNet decode stages (bf16 except stage.g fp32) + the
    // persistent-state hard gate (conv bf16 [conv_dim,3], recurrent fp32
    // [n_heads, key_dim, value_dim], each captured before/after the step).
    const int conv_dim = c.linear_conv_dim();
    const int key_dim = c.linear_key_dim();
    const int value_dim = c.linear_value_dim();
    const int n_heads = c.lin_num_v_heads;
    const int hd_k = c.lin_key_head_dim;
    const int hd_v = c.lin_value_head_dim;
    Status s = expect("stage.input", {1, H});
    if (!s.ok) return s;
    s = expect("stage.rmsnorm1", {1, H});
    if (!s.ok) return s;
    s = expect("stage.in_proj_qkv", {1, conv_dim});
    if (!s.ok) return s;
    s = expect("stage.in_proj_z", {1, value_dim});
    if (!s.ok) return s;
    s = expect("stage.in_proj_b", {1, n_heads});
    if (!s.ok) return s;
    s = expect("stage.in_proj_a", {1, n_heads});
    if (!s.ok) return s;
    s = expect("stage.conv_out", {1, conv_dim});
    if (!s.ok) return s;
    s = expect("stage.conv_silu", {1, conv_dim});
    if (!s.ok) return s;
    s = expect("stage.q", {1, key_dim});
    if (!s.ok) return s;
    s = expect("stage.k", {1, key_dim});
    if (!s.ok) return s;
    s = expect("stage.v", {1, value_dim});
    if (!s.ok) return s;
    s = expect("stage.beta", {1, n_heads});
    if (!s.ok) return s;
    s = expect_f32("stage.g", {1, n_heads});
    if (!s.ok) return s;
    s = expect("stage.core_out", {1, value_dim});
    if (!s.ok) return s;
    s = expect("stage.gated_norm", {1, value_dim});
    if (!s.ok) return s;
    s = expect("stage.out_proj", {1, H});
    if (!s.ok) return s;
    s = expect("stage.residual1", {1, H});
    if (!s.ok) return s;
    s = expect("stage.rmsnorm2", {1, H});
    if (!s.ok) return s;
    s = expect("stage.mlp_gate", {1, inter});
    if (!s.ok) return s;
    s = expect("stage.mlp_up", {1, inter});
    if (!s.ok) return s;
    s = expect("stage.silu_mul", {1, inter});
    if (!s.ok) return s;
    s = expect("stage.mlp_down", {1, H});
    if (!s.ok) return s;
    s = expect("stage.final_output", {1, H});
    if (!s.ok) return s;
    s = expect("state.conv_before", {conv_dim, 3});
    if (!s.ok) return s;
    s = expect("state.conv_after", {conv_dim, 3});
    if (!s.ok) return s;
    s = expect_f32("state.recurrent_before", {n_heads, hd_k, hd_v});
    if (!s.ok) return s;
    return expect_f32("state.recurrent_after", {n_heads, hd_k, hd_v});
  }

  // Full-attention (Phase B) tensor set.
  const int qo = c.n_heads * c.head_dim;   // per-head-count flat width
  const int kvo = c.n_kv_heads * c.head_dim;
  const int rows = c.n_kv_heads * (position_ + 1);
  Status s = expect("stage.input", {1, H});
  if (!s.ok) return s;
  s = expect("stage.rmsnorm1", {1, H});
  if (!s.ok) return s;
  s = expect("stage.q_gate", {1, 2 * qo});
  if (!s.ok) return s;
  s = expect("stage.q", {1, qo});
  if (!s.ok) return s;
  s = expect("stage.att_gate", {1, qo});
  if (!s.ok) return s;
  s = expect("stage.k", {1, kvo});
  if (!s.ok) return s;
  s = expect("stage.v", {1, kvo});
  if (!s.ok) return s;
  s = expect("stage.q_norm", {1, qo});
  if (!s.ok) return s;
  s = expect("stage.k_norm", {1, kvo});
  if (!s.ok) return s;
  s = expect("stage.rope_q", {1, qo});
  if (!s.ok) return s;
  s = expect("stage.rope_k", {1, kvo});
  if (!s.ok) return s;
  s = expect("stage.attention_raw", {1, qo});
  if (!s.ok) return s;
  s = expect("stage.attention_gated", {1, qo});
  if (!s.ok) return s;
  s = expect("stage.o_proj", {1, H});
  if (!s.ok) return s;
  s = expect("stage.residual1", {1, H});
  if (!s.ok) return s;
  s = expect("stage.rmsnorm2", {1, H});
  if (!s.ok) return s;
  s = expect("stage.mlp_gate", {1, inter});
  if (!s.ok) return s;
  s = expect("stage.mlp_up", {1, inter});
  if (!s.ok) return s;
  s = expect("stage.silu_mul", {1, inter});
  if (!s.ok) return s;
  s = expect("stage.mlp_down", {1, H});
  if (!s.ok) return s;
  s = expect("stage.final_output", {1, H});
  if (!s.ok) return s;
  s = expect("kv.k_state", {rows, c.head_dim});
  if (!s.ok) return s;
  return expect("kv.v_state", {rows, c.head_dim});
}

}  // namespace cudalm
