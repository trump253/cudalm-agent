// CUDALM — weight-file loader implementation.
//
// Host-side parse is strictly validating: the first violation produces a
// descriptive Status; a partially-populated object is never returned as ok.

#include "cudalm/weight_loader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>

namespace cudalm {

using wfmt::kConfigBytes;
using wfmt::kHeaderBytes;
using wfmt::kMagic;
using wfmt::kMaxDimValue;
using wfmt::kMaxNameLen;
using wfmt::kMaxNumTensors;
using wfmt::kPayloadAlign;
using wfmt::kVersion;

namespace {

bool ascii_printable(const std::uint8_t* p, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (p[i] < 0x21 || p[i] > 0x7E) return false;
  }
  return true;
}

}  // namespace

Status WeightFile::load(const std::string& path, WeightFile* out) {
  out->pool_.clear();
  out->tensors_.clear();
  out->config_ = ModelConfig{};
  out->payload_offset_ = 0;
  out->payload_size_ = 0;

  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return Status::error("cannot open file: " + path);
  if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return Status::error("seek failed"); }
  long sz = std::ftell(f);
  if (sz < 0) { std::fclose(f); return Status::error("ftell failed"); }
  std::rewind(f);
  out->pool_.resize(static_cast<std::size_t>(sz));
  if (sz > 0 && std::fread(out->pool_.data(), 1, out->pool_.size(), f) != out->pool_.size()) {
    std::fclose(f);
    return Status::error("short read (corrupt file?)");
  }
  std::fclose(f);

  const std::uint8_t* p = out->pool_.data();
  const std::size_t n = out->pool_.size();

  if (n < kHeaderBytes) return Status::error("file smaller than header (72 bytes)");
  if (std::memcmp(p, kMagic, 8) != 0) return Status::error("bad magic");
  const std::uint32_t version = wfmt::rd_u32(p + 8);
  if (version != kVersion) return Status::error("unsupported version");
  if (wfmt::rd_u32(p + 12) != 0) return Status::error("flags must be 0");
  const std::uint32_t nt = wfmt::rd_u32(p + 16);
  if (nt < 1 || nt > kMaxNumTensors) return Status::error("n_tensors out of range");

  out->config_ = wfmt::config_from_blob(p + 20);
  if (!out->config_.valid()) return Status::error("config blob is invalid");

  const std::uint64_t table_off = wfmt::rd_u64(p + 56);
  const std::uint64_t payload_off = wfmt::rd_u64(p + 64);
  if (table_off != kHeaderBytes) return Status::error("table_offset must be 72");
  if (payload_off < table_off || payload_off > n)
    return Status::error("payload_offset out of range");
  if (payload_off % kPayloadAlign != 0)
    return Status::error("payload_offset not 16B aligned");
  out->payload_offset_ = payload_off;
  out->payload_size_ = n - payload_off;

  std::set<std::string> seen;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> regions;  // (offset,size)
  std::uint64_t pos = table_off;

  for (std::uint32_t i = 0; i < nt; ++i) {
    // name_len
    if (pos + 4 > payload_off) return Status::error("table runs past payload");
    const std::uint32_t name_len = wfmt::rd_u32(p + pos);
    pos += 4;
    if (name_len < 1 || name_len > kMaxNameLen)
      return Status::error("tensor name_len out of range");
    if (pos + name_len > payload_off) return Status::error("table runs past payload");
    WeightTensorInfo info;
    info.name.assign(reinterpret_cast<const char*>(p + pos), name_len);
    if (!ascii_printable(p + pos, name_len))
      return Status::error("tensor name not printable ASCII");
    if (!seen.insert(info.name).second)
      return Status::error("duplicate tensor name: " + info.name);
    pos += name_len;

    if (pos + 18 > payload_off) return Status::error("table runs past payload");
    const std::uint8_t dtype_u = p[pos];
    const std::uint8_t ndim = p[pos + 1];
    pos += 2;
    if (dtype_u != static_cast<std::uint8_t>(Dtype::kFp16) &&
        dtype_u != static_cast<std::uint8_t>(Dtype::kInt4Packed) &&
        dtype_u != static_cast<std::uint8_t>(Dtype::kFp16Scale))
      return Status::error("bad dtype in record for " + info.name);
    info.dtype = static_cast<Dtype>(dtype_u);
    if (ndim != 1 && ndim != 2)
      return Status::error("bad ndim in record for " + info.name);

    for (std::uint8_t d = 0; d < ndim; ++d) {
      if (pos + 4 > payload_off) return Status::error("table runs past payload");
      const std::uint32_t dv = wfmt::rd_u32(p + pos);
      pos += 4;
      if (dv < 1 || dv > kMaxDimValue)
        return Status::error("bad dim value in record for " + info.name);
      info.dims.push_back(static_cast<std::int64_t>(dv));
    }
    if (pos + 17 > payload_off) return Status::error("table runs past payload");
    info.offset = wfmt::rd_u64(p + pos);
    info.byte_size = wfmt::rd_u64(p + pos + 8);
    info.align = p[pos + 16];
    pos += 17;
    // 7 reserved bytes must be zero (determinism guard)
    for (int r = 0; r < 7; ++r) {
      if (p[pos + r] != 0) return Status::error("reserved bytes nonzero in " + info.name);
    }
    pos += 7;

    if (info.byte_size != wfmt::expected_bytes(info.dtype, info.dims))
      return Status::error("byte_size/shape mismatch for " + info.name);
    if (info.offset + info.byte_size > out->payload_size_)
      return Status::error("payload region out of bounds for " + info.name);
    const std::uint64_t region_addr = payload_off + info.offset;
    if (region_addr % kPayloadAlign != 0)
      return Status::error("payload region not 16B aligned for " + info.name);
    if (info.align < 1 || info.align > 16 || (info.align & (info.align - 1)) != 0)
      return Status::error("bad align field for " + info.name);
    if (region_addr % info.align != 0)
      return Status::error("align field violated for " + info.name);

    info.host_bytes = p + region_addr;
    regions.emplace_back(info.offset, info.byte_size);
    out->tensors_.push_back(std::move(info));
  }

  if (pos > payload_off) return Status::error("table runs past payload");

  // Overlap check (sort regions by offset; adjacent must not overlap).
  std::sort(regions.begin(), regions.end());
  for (std::size_t i = 1; i < regions.size(); ++i) {
    const std::uint64_t prev_end = regions[i - 1].first + regions[i - 1].second;
    if (regions[i].first < prev_end)
      return Status::error("payload regions overlap");
  }
  return Status::ok_status();
}

const WeightTensorInfo* WeightFile::find(const std::string& name) const {
  for (auto& t : tensors_) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

Status WeightFile::validate_block_tensors() const {
  const ModelConfig& c = config_;
  if (c.group_size != 128)
    return Status::error("block requires group_size == 128");
  const int H = c.hidden_size;
  const int qo = c.q_proj_out();
  const int kv = c.kv_proj_out();
  const int hd = c.head_dim;
  const int inter = c.intermediate_size;

  // Shape contract (docs/weight_format.md): q_proj (q_proj_out, H),
  // k/v_proj (kv_proj_out, H), o_proj (H, q_proj_out) — Q width is
  // independent of hidden_size (v0.1.1).
  struct Expect { const char* name; Dtype dtype; std::int64_t d0; std::int64_t d1; };
  const Expect expects[] = {
      {"attn_norm.weight",  Dtype::kFp16,        H, -1},
      {"ffn_norm.weight",   Dtype::kFp16,        H, -1},
      {"attn.rope_cos",     Dtype::kFp16,        c.max_seq_len, hd / 2},
      {"attn.rope_sin",     Dtype::kFp16,        c.max_seq_len, hd / 2},
      {"attn.q_proj.weight", Dtype::kInt4Packed, qo, H / 2},
      {"attn.q_proj.scale",  Dtype::kFp16Scale,  qo, H / 128},
      {"attn.k_proj.weight", Dtype::kInt4Packed, kv, H / 2},
      {"attn.k_proj.scale",  Dtype::kFp16Scale,  kv, H / 128},
      {"attn.v_proj.weight", Dtype::kInt4Packed, kv, H / 2},
      {"attn.v_proj.scale",  Dtype::kFp16Scale,  kv, H / 128},
      {"attn.o_proj.weight", Dtype::kInt4Packed, H, qo / 2},
      {"attn.o_proj.scale",  Dtype::kFp16Scale,  H, qo / 128},
      {"mlp.gate_proj.weight", Dtype::kInt4Packed, inter, H / 2},
      {"mlp.gate_proj.scale",  Dtype::kFp16Scale,  inter, H / 128},
      {"mlp.up_proj.weight",   Dtype::kInt4Packed, inter, H / 2},
      {"mlp.up_proj.scale",    Dtype::kFp16Scale,  inter, H / 128},
      {"mlp.down_proj.weight", Dtype::kInt4Packed, H, inter / 2},
      {"mlp.down_proj.scale",  Dtype::kFp16Scale,  H, inter / 128},
  };
  for (const Expect& e : expects) {
    const WeightTensorInfo* t = find(e.name);
    if (!t) return Status::error("missing tensor: " + std::string(e.name));
    if (t->dtype != e.dtype)
      return Status::error("dtype mismatch for " + std::string(e.name));
    if (e.d1 < 0) {
      if (t->dims.size() != 1 || t->dims[0] != e.d0)
        return Status::error("shape mismatch for " + std::string(e.name));
    } else {
      if (t->dims.size() != 2 || t->dims[0] != e.d0 || t->dims[1] != e.d1)
        return Status::error("shape mismatch for " + std::string(e.name));
    }
  }
  return Status::ok_status();
}

// ---------------------------------------------------------------------------
// BlockWeights
// ---------------------------------------------------------------------------

BlockWeights::~BlockWeights() = default;

Status BlockWeights::load(const std::string& path, cudaStream_t stream,
                          BlockWeights* out, const ModelConfig* expect) {
  WeightFile wf;
  Status s = WeightFile::load(path, &wf);
  if (!s.ok) return s;
  if (expect && expect->valid() && *expect != wf.config()) {
    return Status::error("file config does not match expected config");
  }
  s = wf.validate_block_tensors();
  if (!s.ok) return s;

  out->config_ = wf.config();
  out->bufs_.resize(18);  // fixed order, see below (DeviceBuffer is move-only)

  struct Slot {
    const char* name;
    TensorView* view;
    int buf;
  };
  // Upload order (fixed, stable): 0 attn_norm, 1 ffn_norm, 2 rope_cos,
  // 3 rope_sin, then w/scale pairs for q,k,v,o,gate,up,down (4..17).
  const Slot slots[] = {
      {"attn_norm.weight",    &out->attn_norm, 0},
      {"ffn_norm.weight",     &out->ffn_norm,  1},
      {"attn.rope_cos",       &out->rope_cos,  2},
      {"attn.rope_sin",       &out->rope_sin,  3},
      {"attn.q_proj.weight",  &out->q_proj.weight, 4},
      {"attn.q_proj.scale",   &out->q_proj.scale,   5},
      {"attn.k_proj.weight",  &out->k_proj.weight, 6},
      {"attn.k_proj.scale",   &out->k_proj.scale,   7},
      {"attn.v_proj.weight",  &out->v_proj.weight, 8},
      {"attn.v_proj.scale",   &out->v_proj.scale,   9},
      {"attn.o_proj.weight",  &out->o_proj.weight, 10},
      {"attn.o_proj.scale",   &out->o_proj.scale,   11},
      {"mlp.gate_proj.weight",&out->gate_proj.weight, 12},
      {"mlp.gate_proj.scale", &out->gate_proj.scale,  13},
      {"mlp.up_proj.weight",  &out->up_proj.weight, 14},
      {"mlp.up_proj.scale",   &out->up_proj.scale,   15},
      {"mlp.down_proj.weight",&out->down_proj.weight, 16},
      {"mlp.down_proj.scale", &out->down_proj.scale,  17},
  };

  for (const Slot& slot : slots) {
    const WeightTensorInfo* info = wf.find(slot.name);
    DeviceBuffer buf(info->byte_size, stream);
    buf.copy_from_host(info->host_bytes, info->byte_size, stream);
    *slot.view = TensorView(buf.data(), info->dtype, info->dims);
    out->bufs_[slot.buf] = std::move(buf);
  }

  // Fill N/K for each W4A16 linear (K = 2 * packed cols).
  auto fill = [](W4A16Linear& lin) {
    lin.N = static_cast<int>(lin.weight.dim(0));
    lin.K = static_cast<int>(2 * lin.weight.dim(1));
  };
  fill(out->q_proj);
  fill(out->k_proj);
  fill(out->v_proj);
  fill(out->o_proj);
  fill(out->gate_proj);
  fill(out->up_proj);
  fill(out->down_proj);
  return Status::ok_status();
}

}  // namespace cudalm
