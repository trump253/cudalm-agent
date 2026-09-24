// CUDALM — binary weight-file format (`.cudalm`), version 2.
//
// Spec: docs/qwen35_architecture.md §10 (this header is the normative layout).
// Little-endian, deterministic, bounds-checked. Written offline by
// tools/convert_qwen35.py (Python), read by src/runtime/weight_loader_v2.cpp
// (no PyTorch).
//
// v1 (`CUDLMW01`, weight_format.h) is IMMUTABLE: v2 is a separate container
// with a separate magic. Differences vs v1:
//   * architecture id + provenance metadata (TLV strings)
//   * Qwen35Config blob instead of ModelConfig
//   * arbitrary named tensor table (no fixed decoder-block tensor set)
//   * new dtype: bf16 (Dtype::kBf16, additive — v1 behavior unchanged)
//   * no RoPE tables (Qwen3.5 cos/sin are computed on the fly)
//
// Provenance: CUDALM-native. INT4 quantization contract carried unchanged
// from v1 (symmetric G=128, q in [-7,7], fp16 scale, low nibble = k=2b).

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "cudalm/qwen35_config.h"
#include "cudalm/tensor.h"

namespace cudalm {
namespace wfmt2 {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr char kMagic[8] = {'C', 'U', 'D', 'L', 'M', 'W', '0', '2'};
constexpr std::uint32_t kVersion = 2;

// Fixed 88-byte file header:
//   magic[8]
//   version      u32   (= 2)
//   flags        u32   (reserved, must be 0)
//   num_tensors  u32
//   num_metadata u16
//   _pad         u16   (0)
//   config_offset u64  (file-absolute)
//   config_size   u64  (== kConfigBytes)
//   meta_offset   u64
//   meta_size     u64
//   table_offset  u64
//   table_size    u64
//   payload_offset u64 (must be 16B-aligned file-absolute)
//   payload_size   u64
constexpr std::size_t kHeaderBytes = 88;  // 24 fixed + 8 x u64 section fields
constexpr std::size_t kConfigBytes = 88;   // see config_to_blob below
constexpr std::size_t kPayloadAlign = 16;
constexpr std::size_t kMaxNameLen = 255;
constexpr std::size_t kMaxDimValue = 100000000;
constexpr std::size_t kMaxNumTensors = 100000;
constexpr std::size_t kMaxMetadataEntries = 256;

// Metadata value length limit (sha256 hex = 64; generous headroom).
constexpr std::size_t kMaxMetaValueLen = 4096;

// Canonical metadata keys (order in the file; readers must not rely on
// order but writers emit them in this order for determinism).
constexpr char kMetaArch[] = "arch";
constexpr char kMetaModelRepo[] = "model_repo";
constexpr char kMetaModelRevision[] = "model_revision";
constexpr char kMetaConfigSha256[] = "config_sha256";
constexpr char kMetaCheckpointSha256[] = "checkpoint_sha256";
constexpr char kMetaTransformersCommit[] = "transformers_commit";
constexpr char kMetaTransformersVersion[] = "transformers_version";
constexpr char kMetaSourceDtype[] = "source_dtype";
constexpr char kMetaGenerator[] = "generator";

// Tensor-table entry layout (variable):
//   name_len  u8   (1..255)
//   name      [name_len]  (charset: [A-Za-z0-9._-])
//   dtype     u8   (cudalm::Dtype)
//   ndim      u8   (1..8)
//   _pad      u16  (0)
//   dims      i64[8]  (first ndim used, rest must be 0)
//   offset    u64  (from payload start)
//   byte_size u64
//   align     u8   (1,2,4,8,16)
//   _pad      u8   (0)
// Fixed part size (excluding name):
constexpr std::size_t kEntryFixedBytes = 1 + 1 + 1 + 2 + 8 * 8 + 8 + 8 + 1 + 1;

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
inline std::int64_t rd_i64(const std::uint8_t* p) {
  std::uint64_t v; std::memcpy(&v, p, 8); return static_cast<std::int64_t>(v);
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
inline void wr_i64(std::uint8_t* p, std::int64_t v) {
  std::uint64_t u; std::memcpy(&u, &v, 8); std::memcpy(p, &u, 8);
}
inline void wr_i32(std::uint8_t* p, std::int32_t v) {
  std::uint32_t u; std::memcpy(&u, &v, 4); std::memcpy(p, &u, 4);
}
inline void wr_f32(std::uint8_t* p, float v) {
  std::uint32_t u; std::memcpy(&u, &v, 4); std::memcpy(p, &u, 4);
}

// ---------------------------------------------------------------------------
// Qwen35Config <-> 88-byte blob (fixed field order).
// ---------------------------------------------------------------------------
inline void config_to_blob(const Qwen35Config& c, std::uint8_t* out) {
  std::size_t o = 0;
  wr_i32(out + o, c.hidden_size); o += 4;
  wr_i32(out + o, c.num_hidden_layers); o += 4;
  wr_i32(out + o, c.intermediate_size); o += 4;
  wr_i32(out + o, c.vocab_size); o += 4;
  wr_i32(out + o, c.n_heads); o += 4;
  wr_i32(out + o, c.n_kv_heads); o += 4;
  wr_i32(out + o, c.head_dim); o += 4;
  wr_i32(out + o, c.lin_num_k_heads); o += 4;
  wr_i32(out + o, c.lin_num_v_heads); o += 4;
  wr_i32(out + o, c.lin_key_head_dim); o += 4;
  wr_i32(out + o, c.lin_value_head_dim); o += 4;
  wr_i32(out + o, c.lin_conv_kernel_dim); o += 4;
  wr_i32(out + o, c.full_attention_interval); o += 4;
  wr_i32(out + o, c.group_size); o += 4;
  wr_i32(out + o, c.max_seq_len); o += 4;
  wr_f32(out + o, c.eps); o += 4;
  wr_f32(out + o, c.rope_theta); o += 4;
  wr_f32(out + o, c.partial_rotary_factor); o += 4;
  wr_i32(out + o, c.mrope_section[0]); o += 4;
  wr_i32(out + o, c.mrope_section[1]); o += 4;
  wr_i32(out + o, c.mrope_section[2]); o += 4;
  wr_i32(out + o, 0); o += 4;  // reserved
  (void)o;
}

// The reserved u32 of the 88-byte config blob (offset 84..87) must be 0.
// The Python parser (tools/common/cudalm_v2.py Qwen35Config.from_blob)
// rejects non-zero; the C++ v2 loaders (weight + golden) enforce the same
// rule so both paths behave identically.
constexpr std::size_t kConfigReservedOffset = 84;  // within the 88-B blob
inline bool config_blob_reserved_ok(const std::uint8_t* p) {
  return rd_u32(p + kConfigReservedOffset) == 0;
}

inline Qwen35Config config_from_blob(const std::uint8_t* p) {
  Qwen35Config c;
  c.hidden_size = rd_i32(p + 0);
  c.num_hidden_layers = rd_i32(p + 4);
  c.intermediate_size = rd_i32(p + 8);
  c.vocab_size = rd_i32(p + 12);
  c.n_heads = rd_i32(p + 16);
  c.n_kv_heads = rd_i32(p + 20);
  c.head_dim = rd_i32(p + 24);
  c.lin_num_k_heads = rd_i32(p + 28);
  c.lin_num_v_heads = rd_i32(p + 32);
  c.lin_key_head_dim = rd_i32(p + 36);
  c.lin_value_head_dim = rd_i32(p + 40);
  c.lin_conv_kernel_dim = rd_i32(p + 44);
  c.full_attention_interval = rd_i32(p + 48);
  c.group_size = rd_i32(p + 52);
  c.max_seq_len = rd_i32(p + 56);
  c.eps = rd_f32(p + 60);
  c.rope_theta = rd_f32(p + 64);
  c.partial_rotary_factor = rd_f32(p + 68);
  c.mrope_section[0] = rd_i32(p + 72);
  c.mrope_section[1] = rd_i32(p + 76);
  c.mrope_section[2] = rd_i32(p + 80);
  // reserved at p+84..p+87
  return c;
}

}  // namespace wfmt2
}  // namespace cudalm
