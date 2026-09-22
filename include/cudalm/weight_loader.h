// CUDALM — weight-file loader (host parse + device upload). No PyTorch.
//
// WeightFile:  parses and validates a .cudalm file entirely on the host
//              (whole-file read; v0.1 files are small). Deterministic.
// BlockWeights: uploads the decoder-block tensor set to the device and
//              exposes raw TensorViews for the ops.
//
// Provenance: CUDALM-native.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/model_config.h"
#include "cudalm/tensor.h"
#include "cudalm/weight_format.h"

namespace cudalm {

struct WeightTensorInfo {
  std::string name;
  Dtype dtype = Dtype::kFp16;
  std::vector<std::int64_t> dims;
  std::uint64_t offset = 0;    // from payload start
  std::uint64_t byte_size = 0;
  std::uint8_t align = 16;
  const std::uint8_t* host_bytes = nullptr;  // into WeightFile's pool
};

class WeightFile {
 public:
  // Parse + validate the whole file (host only). On failure, `message`
  // describes the first violation and the object is left empty.
  static Status load(const std::string& path, WeightFile* out);

  const ModelConfig& config() const { return config_; }
  std::size_t num_tensors() const { return tensors_.size(); }
  const std::vector<WeightTensorInfo>& tensors() const { return tensors_; }
  const WeightTensorInfo* find(const std::string& name) const;
  std::uint64_t payload_size() const { return payload_size_; }

  // Check the full decoder-block tensor set (names + shapes) against a config.
  // Returns a descriptive error Status when something is missing/mismatched.
  Status validate_block_tensors() const;

 private:
  ModelConfig config_{};
  std::vector<std::uint8_t> pool_;  // whole file (owns host_bytes pointers)
  std::vector<WeightTensorInfo> tensors_;
  std::uint64_t payload_offset_ = 0;
  std::uint64_t payload_size_ = 0;
};

// A single W4A16 projection: packed int4 weight + fp16 group scales.
struct W4A16Linear {
  TensorView weight;   // kInt4Packed  [N, K/2]
  TensorView scale;    // kFp16Scale   [N, K/128]
  int N = 0;
  int K = 0;
};

// Decoder-block weights resident on the device. Buffers are owned by the
// object in a fixed, stable order (declared order below).
class BlockWeights {
 public:
  // Upload the decoder-block tensor set from `path`. Validates the file
  // config + tensor set; if `expect` is non-null and valid, the file config
  // must match it exactly.
  static Status load(const std::string& path, cudaStream_t stream,
                     BlockWeights* out, const ModelConfig* expect = nullptr);

  // ---- fp16 tensors -----------------------------------------------------
  TensorView attn_norm;   // [H]
  TensorView ffn_norm;    // [H]
  TensorView rope_cos;    // [max_seq_len, head_dim/2]
  TensorView rope_sin;    // [max_seq_len, head_dim/2]
  // ---- W4A16 projections ------------------------------------------------
  W4A16Linear q_proj;
  W4A16Linear k_proj;
  W4A16Linear v_proj;
  W4A16Linear o_proj;
  W4A16Linear gate_proj;
  W4A16Linear up_proj;
  W4A16Linear down_proj;
  const ModelConfig& config() const { return config_; }

  ~BlockWeights();

 private:
  std::vector<DeviceBuffer> bufs_;  // stable ownership
  ModelConfig config_{};
};

// ---------------------------------------------------------------------------
// INT4 pack/unpack (host-side reference). Mirrors CUDALab
// cudalab/int4gemv_quantize.py pack_q/unpack_w and the device unpack exactly.
// Used by CPU tests and by the golden cross-checks.
// ---------------------------------------------------------------------------
namespace int4 {

// Pack two INT4 values (two's complement, domain [-8, 7]) into one byte:
// low nibble = value for k=2b (lo), high nibble = value for k=2b+1 (hi).
inline std::uint8_t pack_byte(int lo, int hi) {
  return static_cast<std::uint8_t>(((hi & 0xF) << 4) | (lo & 0xF));
}

// Sign-extend a 4-bit two's-complement nibble to int.
inline int unpack_nibble(std::uint8_t b, bool high) {
  if (!high) return static_cast<std::int8_t>((b & 0x0Fu) << 4) >> 4;
  return static_cast<std::int8_t>(b) >> 4;
}

inline int unpack_lo(std::uint8_t b) { return unpack_nibble(b, false); }
inline int unpack_hi(std::uint8_t b) { return unpack_nibble(b, true); }

}  // namespace int4
}  // namespace cudalm
