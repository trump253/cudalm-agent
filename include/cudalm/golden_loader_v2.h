// CUDALM — golden-reference file loader (CUDLMG02, v0.2 Qwen3.5 path).
// Host-only, no PyTorch. CUDALM-native (shares the tensor-record format
// with .cudalm v2 and CUDLMG01).
//
// GoldenFileV2: parses and validates a CUDLMG02 container entirely on the
// host (whole-file read). A golden file embeds the Qwen35Config, the decode
// state (position + layer index + input seed), all 21 stage tensors of one
// full-attention-layer decode step (bf16), and the full KV-cache state
// (bf16) after the write at that position.
//
// Layout (spec: docs/qwen35_architecture.md §14; the Python mirror is
// tools/common/golden_v2.py):
//   Header 144 B:
//     magic(8) "CUDLMG02" | version u32 (=1) | flags u32 (=0)
//     | n_tensors u32 | _pad u32 (=0)
//     | Qwen35Config blob (88 B, weight_format_v2.h field order)
//     | position i32 (0 <= position < max_seq_len)
//     | layer_idx  i32 (a full-attention layer index per the config)
//     | input_seed i32 (the seeded random hidden-state stream seed)
//     | _reserved  i32 (=0)
//     | table_offset u64 (=144) | payload_offset u64 (16B-aligned)
//   Table   n_tensors x TensorRecord (bit-identical record format to the
//           .cudalm v2 weight container)
//   Payload tensor blobs in table order, each region 16B-aligned
//
// Tensor set (23 = 21 stages + 2 KV states, all bf16; shapes for a
// full-attention layer of config c at position p):
//   stage.input            [1, H]
//   stage.rmsnorm1         [1, H]
//   stage.q_gate           [1, 2*n_heads*head_dim]  (fused q_proj output)
//   stage.q                [1, n_heads*head_dim]
//   stage.att_gate         [1, n_heads*head_dim]
//   stage.k                [1, n_kv*head_dim]
//   stage.v                [1, n_kv*head_dim]
//   stage.q_norm           [1, n_heads*head_dim]
//   stage.k_norm           [1, n_kv*head_dim]
//   stage.rope_q           [1, n_heads*head_dim]
//   stage.rope_k           [1, n_kv*head_dim]
//   stage.attention_raw    [1, n_heads*head_dim]
//   stage.attention_gated  [1, n_heads*head_dim]
//   stage.o_proj           [1, H]
//   stage.residual1        [1, H]
//   stage.rmsnorm2         [1, H]
//   stage.mlp_gate         [1, inter]
//   stage.mlp_up           [1, inter]
//   stage.silu_mul         [1, inter]
//   stage.mlp_down         [1, H]
//   stage.final_output     [1, H]
//   kv.k_state             [n_kv*(position+1), head_dim]  row order
//   kv.v_state             [n_kv*(position+1), head_dim]  (kv_head, position)
//
// Provenance: CUDALM-native.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cudalm/qwen35_config.h"
#include "cudalm/weight_format.h"      // Status (shared with v1)
#include "cudalm/weight_format_v2.h"   // Qwen35Config blob, tensor helpers

namespace cudalm {

namespace gfmt2 {

constexpr char kMagic[8] = {'C', 'U', 'D', 'L', 'M', 'G', '0', '2'};
constexpr std::uint32_t kVersion = 1;

// Header field offsets (144 B total).
constexpr std::size_t kHeaderBytes = 144;
constexpr std::size_t kPositionOffset = 112;  // i32
constexpr std::size_t kLayerIdxOffset = 116;  // i32
constexpr std::size_t kInputSeedOffset = 120; // i32
constexpr std::size_t kReservedOffset = 124;  // i32, must be zero
constexpr std::size_t kTableOffsetField = 128;    // u64, must be 144
constexpr std::size_t kPayloadOffsetField = 136;  // u64, 16B-aligned

}  // namespace gfmt2

struct GoldenV2TensorInfo {
  std::string name;
  Dtype dtype = Dtype::kBf16;
  std::vector<std::int64_t> dims;
  std::uint64_t offset = 0;     // from payload start
  std::uint64_t byte_size = 0;
  std::uint8_t align = 16;
  const std::uint8_t* host_bytes = nullptr;  // into GoldenFileV2's pool
};

class GoldenFileV2 {
 public:
  // Parse + validate the whole file (host only). On failure, `message`
  // describes the first violation and the object is left empty.
  static Status load(const std::string& path, GoldenFileV2* out);

  const Qwen35Config& config() const { return config_; }
  int position() const { return position_; }
  int layer_idx() const { return layer_idx_; }
  std::int32_t input_seed() const { return input_seed_; }
  std::size_t num_tensors() const { return tensors_.size(); }
  const std::vector<GoldenV2TensorInfo>& tensors() const { return tensors_; }
  const GoldenV2TensorInfo* find(const std::string& name) const;
  std::uint64_t payload_size() const { return payload_size_; }

  // Check the 23 golden tensors (21 stages + 2 KV states) for
  // (config_, layer_idx_, position_).
  Status validate_golden_tensors() const;

 private:
  Qwen35Config config_{};
  int position_ = 0;
  int layer_idx_ = -1;
  std::int32_t input_seed_ = 0;
  std::vector<std::uint8_t> pool_;  // whole file (owns host_bytes pointers)
  std::vector<GoldenV2TensorInfo> tensors_;
  std::uint64_t payload_offset_ = 0;
  std::uint64_t payload_size_ = 0;
};

}  // namespace cudalm
