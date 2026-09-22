// CUDALM — golden-reference file loader (CUDLMG01, v1). Host-only, no PyTorch.
//
// GoldenFile: parses and validates a golden container entirely on the host
// (whole-file read). A golden file embeds the 18 decoder-block weight tensors
// (identical to the .cudalm weight file for the same seed), the model config,
// the decode position, all 16 stage tensors of one block evaluation, and the
// full KV-cache state after the write at that position.
//
// Layout (spec: docs/weight_format.md, "Golden container"):
//   Header 80 B = magic(8) "CUDLMG01" | version u32 (=1) | flags u32 (=0)
//                 | n_tensors u32 | config blob (36 B)
//                 | position i32 (>= 0, < max_seq_len) | reserved i32 (=0)
//                 | table_offset u64 (=80) | payload_offset u64 (16B-aligned)
//   Table   n_tensors x TensorRecord  (bit-identical record format to the
//           weight container; wfmt:: helpers apply)
//   Payload tensor blobs in table order, each region 16B-aligned
//
// Tensor set (36 = 18 weights + 18 golden tensors; see
// golden_tensor_expectations below and binfmt.py for the mirror):
//   stage.input, stage.rmsnorm1, stage.q, stage.k, stage.v, stage.rope_q,
//   stage.rope_k, stage.attention_output, stage.output_projection,
//   stage.residual1, stage.rmsnorm2, stage.gate, stage.up,
//   stage.silu_gate_mul_up, stage.down, stage.final_output   (all FP16, [1, X])
//   kv.k_state, kv.v_state   (FP16, [n_kv_heads*(position+1), head_dim],
//                              row order (kv_head, position) row-major)
//
// Provenance: CUDALM-native (shares the wfmt record format with .cudalm).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cudalm/model_config.h"
#include "cudalm/weight_format.h"

namespace cudalm {

namespace gfmt {

constexpr char kMagic[8] = {'C', 'U', 'D', 'L', 'M', 'G', '0', '1'};
constexpr std::uint32_t kVersion = 1;

// Header field offsets.
constexpr std::size_t kHeaderBytes = 80;          // 8+4+4+4+36+4+4+8+8
constexpr std::size_t kPositionOffset = 56;       // i32, right after config
constexpr std::size_t kReservedOffset = 60;       // i32, must be zero
constexpr std::size_t kTableOffsetField = 64;     // u64, must be 80
constexpr std::size_t kPayloadOffsetField = 72;   // u64, 16B-aligned

}  // namespace gfmt

struct GoldenTensorInfo {
  std::string name;
  Dtype dtype = Dtype::kFp16;
  std::vector<std::int64_t> dims;
  std::uint64_t offset = 0;    // from payload start
  std::uint64_t byte_size = 0;
  std::uint8_t align = 16;
  const std::uint8_t* host_bytes = nullptr;  // into GoldenFile's pool
};

class GoldenFile {
 public:
  // Parse + validate the whole file (host only). On failure, `message`
  // describes the first violation and the object is left empty.
  static Status load(const std::string& path, GoldenFile* out);

  const ModelConfig& config() const { return config_; }
  int position() const { return position_; }
  std::size_t num_tensors() const { return tensors_.size(); }
  const std::vector<GoldenTensorInfo>& tensors() const { return tensors_; }
  const GoldenTensorInfo* find(const std::string& name) const;
  std::uint64_t payload_size() const { return payload_size_; }

  // Check the 18 block weight tensors (same set as WeightFile::validate_block_tensors).
  Status validate_block_tensors() const;
  // Check weights + all 18 golden tensors for position() (stages + KV state).
  Status validate_golden_tensors() const;

 private:
  ModelConfig config_{};
  int position_ = 0;
  std::vector<std::uint8_t> pool_;  // whole file (owns host_bytes pointers)
  std::vector<GoldenTensorInfo> tensors_;
  std::uint64_t payload_offset_ = 0;
  std::uint64_t payload_size_ = 0;
};

}  // namespace cudalm
