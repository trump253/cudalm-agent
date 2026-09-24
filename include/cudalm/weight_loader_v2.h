// CUDALM — .cudalm v2 weight-file loader (host parse + validation).
// No PyTorch. Layout: weight_format_v2.h (normative), spec + provenance:
// docs/qwen35_architecture.md.
//
// WeightFileV2: parses and validates a v2 file entirely on the host
// (whole-file read; v0.2 files are small to ~GB). Deterministic.
//
// Provenance: CUDALM-native.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/tensor.h"
#include "cudalm/weight_format_v2.h"
#include "cudalm/weight_loader.h"  // Status, W4A16Linear (shared v1 types)

namespace cudalm {

struct Qwen35TensorInfo {
  std::string name;
  Dtype dtype = Dtype::kBf16;
  std::vector<std::int64_t> dims;
  std::uint64_t offset = 0;     // from payload start
  std::uint64_t byte_size = 0;
  std::uint8_t align = 16;
  const std::uint8_t* host_bytes = nullptr;  // into WeightFileV2's pool
};

class WeightFileV2 {
 public:
  // Parse + validate the whole file (host only). On failure, `message`
  // describes the first violation and the object is left empty.
  static Status load(const std::string& path, WeightFileV2* out);

  const Qwen35Config& config() const { return config_; }
  const std::string& arch() const { return arch_; }
  const std::vector<std::pair<std::string, std::string>>& metadata() const {
    return metadata_;
  }
  const std::string* meta(const std::string& key) const;

  std::size_t num_tensors() const { return tensors_.size(); }
  const std::vector<Qwen35TensorInfo>& tensors() const { return tensors_; }
  const Qwen35TensorInfo* find(const std::string& name) const;
  std::uint64_t payload_size() const { return payload_size_; }

  // ---- Qwen3.5 tensor-set validation -------------------------------------
  // Checks the exact name/dtype/shape set of one decoder layer against the
  // file (layer type taken from config_). Errors are descriptive.
  Status validate_layer(int layer_idx) const;
  // Model-level tensor: `norm.weight` [hidden_size] bf16.
  Status validate_model_norm() const;

 private:
  Qwen35Config config_{};
  std::string arch_;
  std::vector<std::pair<std::string, std::string>> metadata_;
  std::vector<std::uint8_t> pool_;  // whole file (owns host_bytes pointers)
  std::vector<Qwen35TensorInfo> tensors_;
  std::uint64_t payload_offset_ = 0;
  std::uint64_t payload_size_ = 0;
};

// Device-resident weight set for ONE Qwen3.5 decoder layer (v0.2). Buffers
// are owned in a fixed stable order. Non-GEMV tensors are stored bf16.
class Qwen35LayerWeights {
 public:
  // Upload one layer's tensor set from a parsed v2 file. Validates the
  // layer tensor set first; on failure `*out` is left empty.
  static Status load(const WeightFileV2& file, int layer_idx,
                     cudaStream_t stream, Qwen35LayerWeights* out);

  bool is_full_attention() const { return cfg_.is_full_attention(idx_); }
  int layer_idx() const { return idx_; }
  const Qwen35Config& config() const { return cfg_; }

  // ---- fp16-scaled W4A16 projections (name -> GEMV) -----------------------
  // Full-attention layers:
  W4A16Linear q_proj;    // [4096, 1024] fused [q; gate] (rows h*512..h*512+511)
  W4A16Linear k_proj;    // [512, 1024]
  W4A16Linear v_proj;    // [512, 1024]
  W4A16Linear o_proj;    // [1024, 2048]
  // Gated DeltaNet layers:
  W4A16Linear in_proj_qkv;  // [6144, 1024]
  W4A16Linear in_proj_z;    // [2048, 1024]
  W4A16Linear in_proj_b;    // [16, 1024]
  W4A16Linear in_proj_a;    // [16, 1024]
  W4A16Linear out_proj;     // [1024, 2048]
  // Shared (both types):
  W4A16Linear gate_proj;    // [3584, 1024]
  W4A16Linear up_proj;      // [3584, 1024]
  W4A16Linear down_proj;    // [1024, 3584]

  // ---- bf16 parameters ----------------------------------------------------
  TensorView input_layernorm;    // [H] zero-centered
  TensorView post_attention_ln;  // [H] zero-centered
  // Full-attention only:
  TensorView q_norm;             // [head_dim] zero-centered
  TensorView k_norm;             // [head_dim] zero-centered
  // DeltaNet only:
  TensorView conv1d_weight;      // [conv_dim, 1, kernel] bf16 (depthwise)
  TensorView dt_bias;            // [lin_num_v_heads] bf16
  TensorView A_log;              // [lin_num_v_heads] fp32
  TensorView linear_norm;        // [lin_value_head_dim] fp32 (gated norm)

  ~Qwen35LayerWeights();

 private:
  std::vector<DeviceBuffer> bufs_;  // stable ownership
  Qwen35Config cfg_{};
  int idx_ = -1;
};

}  // namespace cudalm
