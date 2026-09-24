// CUDALM — .cudalm v2 loader implementation. No PyTorch.
// See include/cudalm/weight_loader_v2.h and weight_format_v2.h.

#include "cudalm/weight_loader_v2.h"

#include <algorithm>
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
  // v2 file dtype allowlist (v1 files may NOT contain bf16; v2 may).
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

// ---------------------------------------------------------------------------
// WeightFileV2
// ---------------------------------------------------------------------------

Status WeightFileV2::load(const std::string& path, WeightFileV2* out) {
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

  if (n < wfmt2::kHeaderBytes)
    return Status::error("file too small for v2 header");
  if (std::memcmp(p, wfmt2::kMagic, 8) != 0)
    return Status::error("bad magic (not CUDLMW02)");
  const std::uint32_t version = wfmt2::rd_u32(p + 8);
  if (version != wfmt2::kVersion)
    return Status::error("unsupported version " + std::to_string(version));
  if (wfmt2::rd_u32(p + 12) != 0)
    return Status::error("reserved flags must be 0");
  const std::uint32_t num_tensors = wfmt2::rd_u32(p + 16);
  const std::uint16_t num_metadata = wfmt2::rd_u16(p + 20);
  if (wfmt2::rd_u16(p + 22) != 0)
    return Status::error("reserved pad must be 0");
  const std::uint64_t config_offset = wfmt2::rd_u64(p + 24);
  const std::uint64_t config_size = wfmt2::rd_u64(p + 32);
  const std::uint64_t meta_offset = wfmt2::rd_u64(p + 40);
  const std::uint64_t meta_size = wfmt2::rd_u64(p + 48);
  const std::uint64_t table_offset = wfmt2::rd_u64(p + 56);
  const std::uint64_t table_size = wfmt2::rd_u64(p + 64);
  const std::uint64_t payload_offset = wfmt2::rd_u64(p + 72);
  const std::uint64_t payload_size = wfmt2::rd_u64(p + 80);

  const std::uint64_t file_size = n;
  auto within = [&](std::uint64_t off, std::uint64_t size) {
    return off <= file_size && size <= file_size - off;
  };
  if (num_tensors > wfmt2::kMaxNumTensors)
    return Status::error("num_tensors exceeds limit");
  if (num_metadata > wfmt2::kMaxMetadataEntries)
    return Status::error("num_metadata exceeds limit");
  if (config_size != wfmt2::kConfigBytes)
    return Status::error("config_size must be " + std::to_string(wfmt2::kConfigBytes));
  if (!within(config_offset, config_size))
    return Status::error("config section out of bounds");
  if (!within(meta_offset, meta_size))
    return Status::error("metadata section out of bounds");
  if (!within(table_offset, table_size))
    return Status::error("tensor table out of bounds");
  if (!within(payload_offset, payload_size))
    return Status::error("payload out of bounds");
  if (payload_offset % wfmt2::kPayloadAlign != 0)
    return Status::error("payload_offset must be 16B-aligned");
  // Sections must be non-overlapping.
  const std::uint64_t sections[4][2] = {
      {config_offset, config_size}, {meta_offset, meta_size},
      {table_offset, table_size}, {payload_offset, payload_size}};
  for (int a = 0; a < 4; ++a)
    for (int b = a + 1; b < 4; ++b) {
      const std::uint64_t as = sections[a][0], ae = as + sections[a][1];
      const std::uint64_t bs = sections[b][0], be = bs + sections[b][1];
      if (as < be && bs < ae) return Status::error("sections overlap");
    }

  // ---- config blob --------------------------------------------------------
  const Qwen35Config cfg = wfmt2::config_from_blob(p + config_offset);
  if (!wfmt2::config_blob_reserved_ok(p + config_offset))
    return Status::error("config blob reserved word must be 0");
  if (!cfg.valid())
    return Status::error("Qwen35Config blob failed validation");

  // ---- metadata (TLV strings) ---------------------------------------------
  std::vector<std::pair<std::string, std::string>> metadata;
  {
    std::size_t pos = static_cast<std::size_t>(meta_offset);
    const std::size_t end = static_cast<std::size_t>(meta_offset + meta_size);
    for (std::uint16_t i = 0; i < num_metadata; ++i) {
      if (pos + 1 > end) return Status::error("metadata: key_len out of bounds");
      const std::size_t klen = p[pos];
      pos += 1;
      if (pos + klen + 2 > end) return Status::error("metadata: key/value_len out of bounds");
      std::string key(reinterpret_cast<const char*>(p + pos), klen);
      pos += klen;
      const std::size_t vlen = wfmt2::rd_u16(p + pos);
      pos += 2;
      if (vlen > wfmt2::kMaxMetaValueLen)
        return Status::error("metadata: value too long for key '" + key + "'");
      if (pos + vlen > end) return Status::error("metadata: value out of bounds for '" + key + "'");
      std::string val(reinterpret_cast<const char*>(p + pos), vlen);
      pos += vlen;
      metadata.emplace_back(std::move(key), std::move(val));
    }
    if (pos != end) return Status::error("metadata: trailing bytes");
  }

  // ---- arch id: stored as metadata-free fixed field? NO — it lives in the
  // metadata under the conventional key "arch". (Readers use meta("arch").)
  // ---- tensor table ---------------------------------------------------------
  std::vector<Qwen35TensorInfo> tensors;
  std::set<std::string> seen;
  {
    std::size_t pos = static_cast<std::size_t>(table_offset);
    const std::size_t end = static_cast<std::size_t>(table_offset + table_size);
    for (std::uint32_t i = 0; i < num_tensors; ++i) {
      Qwen35TensorInfo info;
      if (pos + 1 > end) return Status::error("tensor table: name_len out of bounds");
      const std::size_t nlen = p[pos];
      pos += 1;
      if (nlen == 0 || nlen > wfmt2::kMaxNameLen)
        return Status::error("tensor table: bad name_len");
      // kEntryFixedBytes includes the name_len byte already consumed above,
      // so the fixed tail after `pos` is kEntryFixedBytes - 1.
      if (pos + nlen + (wfmt2::kEntryFixedBytes - 1) > end)
        return Status::error("tensor table: entry out of bounds");
      info.name.assign(reinterpret_cast<const char*>(p + pos), nlen);
      pos += nlen;
      if (!name_charset_ok(info.name.data(), nlen))
        return Status::error("tensor table: bad name charset in '" + info.name + "'");
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
        return Status::error("tensor '" + info.name + "': reserved pad must be 0");
      pos += 2;
      for (int d = 0; d < 8; ++d) {
        const std::int64_t dv = wfmt2::rd_i64(p + pos + 8 * d);
        if (d < ndim) {
          if (dv <= 0 || dv > static_cast<std::int64_t>(wfmt2::kMaxDimValue))
            return Status::error("bad dim in tensor '" + info.name + "'");
          info.dims.push_back(dv);
        } else if (dv != 0) {
          return Status::error("tensor '" + info.name + "': unused dim must be 0");
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
        return Status::error("tensor '" + info.name + "': reserved pad must be 0");
      pos += 1;
      if (!align_is_power_of_two(info.align))
        return Status::error("tensor '" + info.name + "': bad align");
      // Bounds within payload.
      if (info.offset > payload_size || info.byte_size > payload_size - info.offset)
        return Status::error("tensor '" + info.name + "' out of payload bounds");
      // File-absolute alignment.
      if ((payload_offset + info.offset) % info.align != 0)
        return Status::error("tensor '" + info.name + "' not " +
                             std::to_string(info.align) + "B-aligned");
      // Byte-size/shape consistency.
      std::vector<std::int64_t> elem_dims = info.dims;
      if (info.dtype == Dtype::kInt4Packed) {
        // Shape is expressed in PACKED bytes (last dim = K/2), as in v1.
      }
      const std::uint64_t expect = wfmt::expected_bytes(info.dtype, elem_dims);
      if (info.byte_size != expect)
        return Status::error("tensor '" + info.name + "': byte_size != expected for dtype/shape");
      info.host_bytes = p + payload_offset + info.offset;
      tensors.push_back(std::move(info));
    }
    if (pos != end) return Status::error("tensor table: trailing bytes");
    if (tensors.size() != num_tensors)
      return Status::error("tensor table count mismatch");
  }

  out->pool_ = std::move(buf);
  out->payload_offset_ = payload_offset;
  out->payload_size_ = payload_size;
  out->config_ = cfg;
  out->metadata_ = std::move(metadata);
  out->tensors_ = std::move(tensors);
  // arch id is metadata key "arch" (may be absent for non-model files).
  if (const std::string* a = out->meta("arch")) out->arch_ = *a;
  return Status::ok_status();
}

const std::string* WeightFileV2::meta(const std::string& key) const {
  for (const auto& kv : metadata_)
    if (kv.first == key) return &kv.second;
  return nullptr;
}

const Qwen35TensorInfo* WeightFileV2::find(const std::string& name) const {
  for (const auto& t : tensors_)
    if (t.name == name) return &t;
  return nullptr;
}

namespace {

Status expect_tensor(const WeightFileV2& f, const std::string& name,
                     Dtype dtype, const std::vector<std::int64_t>& dims) {
  const Qwen35TensorInfo* t = f.find(name);
  if (!t) return Status::error("missing tensor '" + name + "'");
  if (t->dtype != dtype)
    return Status::error("dtype mismatch for '" + name + "' (expected " +
                         dtype_name(dtype) + ")");
  if (t->dims != dims)
    return Status::error("shape mismatch for '" + name + "' (expected " +
                         std::to_string(dims.size()) + "-D)");
  return Status::ok_status();
}

}  // namespace

Status WeightFileV2::validate_layer(int layer_idx) const {
  const Qwen35Config& c = config_;
  if (layer_idx < 0 || layer_idx >= c.num_hidden_layers)
    return Status::error("layer_idx out of range");
  const std::string L = "layers." + std::to_string(layer_idx) + ".";
  auto expect = [&](const std::string& n, Dtype dt,
                    const std::vector<std::int64_t>& d) {
    Status s = expect_tensor(*this, L + n, dt, d);
    return s;
  };

  const int H = c.hidden_size;
  // Shared per-layer set (both types): layernorms + SwiGLU.
  for (const char* norm : {"input_layernorm.weight", "post_attention_layernorm.weight"}) {
    Status s = expect(norm, Dtype::kBf16, {H});
    if (!s.ok) return s;
  }
  {
    Status s = expect("mlp.gate_proj.weight", Dtype::kInt4Packed,
                      {c.intermediate_size, H / 2});
    if (!s.ok) return s;
    s = expect("mlp.up_proj.weight", Dtype::kInt4Packed,
               {c.intermediate_size, H / 2});
    if (!s.ok) return s;
    s = expect("mlp.gate_proj.scale", Dtype::kFp16Scale,
               {c.intermediate_size, H / c.group_size});
    if (!s.ok) return s;
    s = expect("mlp.up_proj.scale", Dtype::kFp16Scale,
               {c.intermediate_size, H / c.group_size});
    if (!s.ok) return s;
    s = expect("mlp.down_proj.weight", Dtype::kInt4Packed, {H, c.intermediate_size / 2});
    if (!s.ok) return s;
    s = expect("mlp.down_proj.scale", Dtype::kFp16Scale,
               {H, c.intermediate_size / c.group_size});
    if (!s.ok) return s;
  }

  if (c.is_full_attention(layer_idx)) {
    const int qo = c.q_proj_out();      // 4096
    const int kvo = c.kv_proj_out();    // 512
    const int oin = c.o_proj_in();      // 2048
    Status s = expect("self_attn.q_proj.weight", Dtype::kInt4Packed, {qo, H / 2});
    if (!s.ok) return s;
    s = expect("self_attn.q_proj.scale", Dtype::kFp16Scale, {qo, H / c.group_size});
    if (!s.ok) return s;
    s = expect("self_attn.k_proj.weight", Dtype::kInt4Packed, {kvo, H / 2});
    if (!s.ok) return s;
    s = expect("self_attn.k_proj.scale", Dtype::kFp16Scale, {kvo, H / c.group_size});
    if (!s.ok) return s;
    s = expect("self_attn.v_proj.weight", Dtype::kInt4Packed, {kvo, H / 2});
    if (!s.ok) return s;
    s = expect("self_attn.v_proj.scale", Dtype::kFp16Scale, {kvo, H / c.group_size});
    if (!s.ok) return s;
    s = expect("self_attn.o_proj.weight", Dtype::kInt4Packed, {H, oin / 2});
    if (!s.ok) return s;
    s = expect("self_attn.o_proj.scale", Dtype::kFp16Scale, {H, oin / c.group_size});
    if (!s.ok) return s;
    s = expect("self_attn.q_norm.weight", Dtype::kBf16, {c.head_dim});
    if (!s.ok) return s;
    s = expect("self_attn.k_norm.weight", Dtype::kBf16, {c.head_dim});
    return s;
  }

  // Gated DeltaNet layer.
  const int cv = c.linear_value_dim();   // 2048
  const int cd = c.linear_conv_dim();    // 6144
  Status s = expect("linear_attn.in_proj_qkv.weight", Dtype::kInt4Packed, {cd, H / 2});
  if (!s.ok) return s;
  s = expect("linear_attn.in_proj_qkv.scale", Dtype::kFp16Scale, {cd, H / c.group_size});
  if (!s.ok) return s;
  s = expect("linear_attn.in_proj_z.weight", Dtype::kInt4Packed, {cv, H / 2});
  if (!s.ok) return s;
  s = expect("linear_attn.in_proj_z.scale", Dtype::kFp16Scale, {cv, H / c.group_size});
  if (!s.ok) return s;
  s = expect("linear_attn.in_proj_b.weight", Dtype::kInt4Packed,
             {c.lin_num_v_heads, H / 2});
  if (!s.ok) return s;
  s = expect("linear_attn.in_proj_b.scale", Dtype::kFp16Scale,
             {c.lin_num_v_heads, H / c.group_size});
  if (!s.ok) return s;
  s = expect("linear_attn.in_proj_a.weight", Dtype::kInt4Packed,
             {c.lin_num_v_heads, H / 2});
  if (!s.ok) return s;
  s = expect("linear_attn.in_proj_a.scale", Dtype::kFp16Scale,
             {c.lin_num_v_heads, H / c.group_size});
  if (!s.ok) return s;
  s = expect("linear_attn.out_proj.weight", Dtype::kInt4Packed, {H, cv / 2});
  if (!s.ok) return s;
  s = expect("linear_attn.out_proj.scale", Dtype::kFp16Scale, {H, cv / c.group_size});
  if (!s.ok) return s;
  s = expect("linear_attn.conv1d.weight", Dtype::kBf16,
             {cd, 1, c.lin_conv_kernel_dim});
  if (!s.ok) return s;
  s = expect("linear_attn.dt_bias", Dtype::kBf16, {c.lin_num_v_heads});
  if (!s.ok) return s;
  s = expect("linear_attn.A_log", Dtype::kFp32, {c.lin_num_v_heads});
  if (!s.ok) return s;
  s = expect("linear_attn.norm.weight", Dtype::kFp32, {c.lin_value_head_dim});
  return s;
}

Status WeightFileV2::validate_model_norm() const {
  return expect_tensor(*this, "norm.weight", Dtype::kBf16, {config_.hidden_size});
}

Status WeightFileV2::validate_model_embedding() const {
  return expect_tensor(*this, "embed_tokens.weight", Dtype::kBf16,
                       {config_.vocab_size, config_.hidden_size});
}

Status WeightFileV2::validate_full_model() const {
  // Embedding + final norm.
  Status s = validate_model_embedding();
  if (!s.ok) return s;
  s = validate_model_norm();
  if (!s.ok) return s;
  // Every decoder layer (0..num_hidden_layers-1), validated against the
  // config's per-layer tensor set + hybrid type.
  for (int i = 0; i < config_.num_hidden_layers; ++i) {
    s = validate_layer(i);
    if (!s.ok) {
      s.message = "layer " + std::to_string(i) + ": " + s.message;
      return s;
    }
  }
  // Weight tying: the pinned Qwen3.5-0.8B ties the LM head to the embedding
  // (config tie_word_embeddings=true, source _tied_weights_keys), so the
  // metadata must be present and "true". A non-tied model (a separate
  // lm_head.weight tensor) is out of scope for this phase.
  const std::string* tie = meta("tie_word_embeddings");
  if (!tie || *tie != "true")
    return Status::error(
        "full model: tie_word_embeddings metadata must be \"true\" (the pinned "
        "Qwen3.5-0.8B ties the LM head to the embedding)");
  return Status::ok_status();
}

// ---------------------------------------------------------------------------
// Qwen35LayerWeights
// ---------------------------------------------------------------------------

Qwen35LayerWeights::~Qwen35LayerWeights() = default;

Status Qwen35LayerWeights::load(const WeightFileV2& file, int layer_idx,
                                cudaStream_t stream, Qwen35LayerWeights* out) {
  Status s = file.validate_layer(layer_idx);
  if (!s.ok) return s;

  const Qwen35Config& c = file.config();
  const std::string L = "layers." + std::to_string(layer_idx) + ".";
  out->cfg_ = c;
  out->idx_ = layer_idx;

  struct Slot {
    const char* name;
    DeviceBuffer* buf;
    TensorView* view;
  };
  // Fixed stable upload order. All GEMVs upload weight + scale.
  // 0..3: layernorms + q/k norm (zero); 4..: GEMV pairs; then deltanet params.
  std::vector<DeviceBuffer> bufs;
  bufs.resize(24);
  auto upload = [&](const char* name, int slot, TensorView* view) {
    const Qwen35TensorInfo* info = file.find(L + name);
    DeviceBuffer b(info->byte_size, stream);
    b.copy_from_host(info->host_bytes, info->byte_size, stream);
    *view = TensorView(b.data(), info->dtype, info->dims);
    bufs[slot] = std::move(b);
  };
  upload("input_layernorm.weight", 0, &out->input_layernorm);
  upload("post_attention_layernorm.weight", 1, &out->post_attention_ln);
  if (c.is_full_attention(layer_idx)) {
    upload("self_attn.q_norm.weight", 2, &out->q_norm);
    upload("self_attn.k_norm.weight", 3, &out->k_norm);
    upload("self_attn.q_proj.weight", 4, &out->q_proj.weight);
    upload("self_attn.q_proj.scale", 5, &out->q_proj.scale);
    upload("self_attn.k_proj.weight", 6, &out->k_proj.weight);
    upload("self_attn.k_proj.scale", 7, &out->k_proj.scale);
    upload("self_attn.v_proj.weight", 8, &out->v_proj.weight);
    upload("self_attn.v_proj.scale", 9, &out->v_proj.scale);
    upload("self_attn.o_proj.weight", 10, &out->o_proj.weight);
    upload("self_attn.o_proj.scale", 11, &out->o_proj.scale);
  } else {
    upload("linear_attn.in_proj_qkv.weight", 4, &out->in_proj_qkv.weight);
    upload("linear_attn.in_proj_qkv.scale", 5, &out->in_proj_qkv.scale);
    upload("linear_attn.in_proj_z.weight", 6, &out->in_proj_z.weight);
    upload("linear_attn.in_proj_z.scale", 7, &out->in_proj_z.scale);
    upload("linear_attn.in_proj_b.weight", 8, &out->in_proj_b.weight);
    upload("linear_attn.in_proj_b.scale", 9, &out->in_proj_b.scale);
    upload("linear_attn.in_proj_a.weight", 10, &out->in_proj_a.weight);
    upload("linear_attn.in_proj_a.scale", 11, &out->in_proj_a.scale);
    upload("linear_attn.out_proj.weight", 12, &out->out_proj.weight);
    upload("linear_attn.out_proj.scale", 13, &out->out_proj.scale);
    upload("linear_attn.conv1d.weight", 14, &out->conv1d_weight);
    upload("linear_attn.dt_bias", 15, &out->dt_bias);
    upload("linear_attn.A_log", 16, &out->A_log);
    upload("linear_attn.norm.weight", 17, &out->linear_norm);
  }
  upload("mlp.gate_proj.weight", 18, &out->gate_proj.weight);
  upload("mlp.gate_proj.scale", 19, &out->gate_proj.scale);
  upload("mlp.up_proj.weight", 20, &out->up_proj.weight);
  upload("mlp.up_proj.scale", 21, &out->up_proj.scale);
  upload("mlp.down_proj.weight", 22, &out->down_proj.weight);
  upload("mlp.down_proj.scale", 23, &out->down_proj.scale);
  out->bufs_ = std::move(bufs);

  auto fill = [](W4A16Linear& lin) {
    lin.N = static_cast<int>(lin.weight.dim(0));
    lin.K = static_cast<int>(2 * lin.weight.dim(1));
  };
  fill(out->gate_proj);
  fill(out->up_proj);
  fill(out->down_proj);
  if (c.is_full_attention(layer_idx)) {
    fill(out->q_proj);
    fill(out->k_proj);
    fill(out->v_proj);
    fill(out->o_proj);
  } else {
    fill(out->in_proj_qkv);
    fill(out->in_proj_z);
    fill(out->in_proj_b);
    fill(out->in_proj_a);
    fill(out->out_proj);
  }
  return Status::ok_status();
}

}  // namespace cudalm
