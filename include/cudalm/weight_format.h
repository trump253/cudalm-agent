// CUDALM — binary weight-file format (`.cudalm`), version 1.
//
// Spec: docs/weight_format.md (single source of truth). Little-endian,
// deterministic, bounds-checked. Written offline by tools/convert_weights.py,
// read by src/runtime/weight_loader.cpp (no PyTorch).
//
// Provenance: CUDALM-native. The INT4 quantization contract (symmetric
// G=128, q in [-7,7], fp16 scale, low nibble = k=2b) is carried from CUDALab
// cb6a6a9 (cudalab/int4gemv_quantize.py); see docs/provenance.md.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#include "cudalm/model_config.h"
#include "cudalm/tensor.h"

namespace cudalm {

// Simple non-exceptional status for host-side file operations.
// (Shared by the weight and golden loaders.)
struct Status {
  bool ok = true;
  std::string message;
  static Status ok_status() { return Status{true, ""}; }
  static Status error(std::string msg) { return Status{false, std::move(msg)}; }
};

namespace wfmt {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr char kMagic[8] = {'C', 'U', 'D', 'L', 'M', 'W', '0', '1'};
constexpr std::uint32_t kVersion = 1;

constexpr std::size_t kConfigBytes = 36;   // 7 x i32 + 2 x f32
constexpr std::size_t kHeaderBytes = 72;   // 8+4+4+4+36+8+8
constexpr std::size_t kPayloadAlign = 16;
constexpr std::size_t kMaxNameLen = 255;
constexpr std::size_t kMaxDimValue = 10000000;
constexpr std::size_t kMaxNumTensors = 10000;

// ---------------------------------------------------------------------------
// Little-endian read/write helpers (host-side only).
// ---------------------------------------------------------------------------
inline std::uint16_t rd_u16(const std::uint8_t* p) {
  std::uint16_t v; std::memcpy(&v, p, 2); return v;
}
inline std::uint32_t rd_u32(const std::uint8_t* p) {
  std::uint32_t v; std::memcpy(&v, p, 4); return v;
}
inline std::uint64_t rd_u64(const std::uint8_t* p) {
  std::uint64_t v; std::memcpy(&v, p, 8); return v;
}
inline std::int32_t rd_i32(const std::uint8_t* p) {
  std::uint32_t v; std::memcpy(&v, p, 4); return static_cast<std::int32_t>(v);
}
inline float rd_f32(const std::uint8_t* p) {
  std::uint32_t v; std::memcpy(&v, p, 4);
  float f; std::memcpy(&f, &v, 4); return f;
}

inline void wr_u16(std::uint8_t* p, std::uint16_t v) { std::memcpy(p, &v, 2); }
inline void wr_u32(std::uint8_t* p, std::uint32_t v) { std::memcpy(p, &v, 4); }
inline void wr_u64(std::uint8_t* p, std::uint64_t v) { std::memcpy(p, &v, 8); }
inline void wr_i32(std::uint8_t* p, std::int32_t v) {
  std::uint32_t u; std::memcpy(&u, &v, 4); std::memcpy(p, &u, 4);
}
inline void wr_f32(std::uint8_t* p, float v) {
  std::uint32_t u; std::memcpy(&u, &v, 4); std::memcpy(p, &u, 4);
}

// ---------------------------------------------------------------------------
// ModelConfig <-> 36-byte blob (fixed field order, see spec).
// ---------------------------------------------------------------------------
inline void config_to_blob(const ModelConfig& c, std::uint8_t* out) {
  std::size_t o = 0;
  wr_i32(out + o, c.hidden_size); o += 4;
  wr_i32(out + o, c.n_heads); o += 4;
  wr_i32(out + o, c.n_kv_heads); o += 4;
  wr_i32(out + o, c.head_dim); o += 4;
  wr_i32(out + o, c.intermediate_size); o += 4;
  wr_i32(out + o, c.group_size); o += 4;
  wr_i32(out + o, c.max_seq_len); o += 4;
  wr_f32(out + o, c.eps); o += 4;
  wr_f32(out + o, c.rope_theta); o += 4;
  (void)o;
}

inline ModelConfig config_from_blob(const std::uint8_t* p) {
  ModelConfig c;
  c.hidden_size = rd_i32(p + 0);
  c.n_heads = rd_i32(p + 4);
  c.n_kv_heads = rd_i32(p + 8);
  c.head_dim = rd_i32(p + 12);
  c.intermediate_size = rd_i32(p + 16);
  c.group_size = rd_i32(p + 20);
  c.max_seq_len = rd_i32(p + 24);
  c.eps = rd_f32(p + 28);
  c.rope_theta = rd_f32(p + 32);
  return c;
}

// ---------------------------------------------------------------------------
// Expected payload size for a dtype/shape (0 on invalid).
// ---------------------------------------------------------------------------
inline std::uint64_t expected_bytes(Dtype dtype, const std::vector<std::int64_t>& dims) {
  std::int64_t numel = 1;
  for (auto d : dims) numel *= d;
  if (numel <= 0) return 0;
  return static_cast<std::uint64_t>(numel) * dtype_element_bytes(dtype);
}

}  // namespace wfmt
}  // namespace cudalm
