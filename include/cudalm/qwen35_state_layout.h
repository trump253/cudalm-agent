// CUDALM — v0.5 Phase A hybrid state-layout formulas.
//
// Pure config-derived arithmetic for the multi-sequence state pools: the
// hybrid layer counts and the per-page / per-slot byte sizes. NO magic
// constants: everything is computed from the Qwen35Config hybrid schedule
// (docs/qwen35_architecture.md §22). Header-only, CPU-only.
//
// Pinned by tests/cpu/test_state_pool_formulas.cpp (0.8B: 6 full / 18
// linear; per-page and per-slot byte sizes for page_tokens 4/8/16).

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>

#include "cudalm/qwen35_config.h"

namespace cudalm {

// Number of full-attention layers = count of i with is_full_attention(i),
// i.e. the layers at indices (interval-1, 2*interval-1, ...). For the
// pinned 0.8B config (24 layers, interval 4): 6 (3,7,11,15,19,23).
inline std::int32_t qwen35_num_full_layers(const Qwen35Config& cfg) {
  int n = 0;
  for (int i = 0; i < cfg.num_hidden_layers; ++i)
    if (cfg.is_full_attention(i)) ++n;
  return n;
}

// Number of Gated DeltaNet layers = num_hidden_layers - full layers.
// For the pinned 0.8B config: 18.
inline std::int32_t qwen35_num_linear_layers(const Qwen35Config& cfg) {
  return cfg.num_hidden_layers - qwen35_num_full_layers(cfg);
}

// Byte size of ONE physical KV page across ALL full-attention layers.
// Page layout per layer (docs §22): K and V each
// [n_kv_heads][page_tokens][head_dim] bf16; all layers share the same
// physical page id space (one block table per sequence).
//   bytes_per_page = 2 (K+V) * num_full_layers * n_kv_heads * page_tokens
//                    * head_dim * sizeof(bf16)
inline std::size_t qwen35_kv_page_bytes(const Qwen35Config& cfg,
                                        int page_tokens) {
  return static_cast<std::size_t>(2) *
         static_cast<std::size_t>(qwen35_num_full_layers(cfg)) *
         static_cast<std::size_t>(cfg.n_kv_heads) *
         static_cast<std::size_t>(page_tokens) *
         static_cast<std::size_t>(cfg.head_dim) * sizeof(__nv_bfloat16);
}

// Byte size of ONE sequence's DeltaNet state slot across ALL linear-
// attention layers:
//   per layer: conv bf16 [linear_conv_dim, conv_state_len]
//             + recurrent fp32 [lin_num_k_heads, lin_key_head_dim,
//                                lin_value_head_dim]
//   bytes_per_slot = num_linear_layers *
//                    (conv_dim*3*sizeof(bf16) +
//                     nk*kd*vd*sizeof(fp32))
// (conv_state_len == lin_conv_kernel_dim - 1; pinned kernel-4 -> 3.)
inline std::size_t qwen35_delta_slot_bytes(const Qwen35Config& cfg) {
  const std::size_t conv_bytes =
      static_cast<std::size_t>(cfg.linear_conv_dim()) *
      static_cast<std::size_t>(cfg.linear_conv_state_len()) *
      sizeof(__nv_bfloat16);
  const std::size_t rec_bytes =
      static_cast<std::size_t>(cfg.lin_num_k_heads) *
      static_cast<std::size_t>(cfg.lin_key_head_dim) *
      static_cast<std::size_t>(cfg.lin_value_head_dim) * sizeof(float);
  return static_cast<std::size_t>(qwen35_num_linear_layers(cfg)) *
         (conv_bytes + rec_bytes);
}

}  // namespace cudalm
