// CUDALM — Qwen3.5 decode KV cache (bf16, RAII). CUDALM-native.
//
// Flat layout contract (same spirit as the v0.1 KvCache, bf16 storage —
// docs/qwen35_architecture.md §7 "State"):
//   K, V each: bf16 [n_kv_heads][max_seq_len][head_dim] row-major
//   flat offset of row (n, t) = ((n * max_seq_len) + t) * head_dim
//   (t = 0-based position, n = kv head, d = head dim)
//
// The K cache stores ROPE'd K rows; the V cache stores raw V rows.
// Qwen35KvCache is independent from (and does not modify) the v0.1
// KvCache class.

#pragma once

#include <cstddef>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cudalm/device_buffer.h"
#include "cudalm/qwen35_config.h"

namespace cudalm {

// RAII decode KV cache for Qwen3.5 full-attention layers. Move-only.
class Qwen35KvCache {
 public:
  // Pre-allocate K and V to [n_kv_heads][max_seq_len][head_dim] bf16
  // (zero-initialized). The config's KV dims must be positive
  // (precondition-checked).
  explicit Qwen35KvCache(const Qwen35Config& cfg, cudaStream_t stream = 0);

  Qwen35KvCache(const Qwen35KvCache&) = delete;
  Qwen35KvCache& operator=(const Qwen35KvCache&) = delete;
  Qwen35KvCache(Qwen35KvCache&& other) noexcept = default;
  Qwen35KvCache& operator=(Qwen35KvCache&& other) noexcept = default;
  ~Qwen35KvCache() = default;

  // Write the current token's K/V rows at `position` (all kv heads).
  //   k : bf16 [n_kv_heads, head_dim] (post-RoPE K rows)
  //   v : bf16 [n_kv_heads, head_dim]
  // Precondition (host-checked, abort on violation): 0 <= position
  // < max_seq_len. All other cache rows are left untouched.
  void write(int position, const __nv_bfloat16* k, const __nv_bfloat16* v,
             cudaStream_t stream);

  // ---- Read-only accessors --------------------------------------------
  const __nv_bfloat16* k() const { return k_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* v() const { return v_.data<__nv_bfloat16>(); }
  // Non-const accessors (tests/seeders that own the cache).
  __nv_bfloat16* k_mut() { return k_.data<__nv_bfloat16>(); }
  __nv_bfloat16* v_mut() { return v_.data<__nv_bfloat16>(); }
  int n_kv_heads() const { return n_kv_; }
  int max_seq_len() const { return max_seq_; }
  int head_dim() const { return head_dim_; }
  // Total bf16 elements per tensor (K or V).
  std::size_t numel() const {
    return static_cast<std::size_t>(n_kv_) * max_seq_ * head_dim_;
  }
  // Byte size of one tensor (K or V).
  std::size_t bytes() const { return numel() * sizeof(__nv_bfloat16); }
  // Flat element offset of row (n, t) within K (or V).
  std::size_t row_offset(int n, int t) const {
    return (static_cast<std::size_t>(n) * max_seq_ + t) * head_dim_;
  }

 private:
  DeviceBuffer k_;
  DeviceBuffer v_;
  int n_kv_ = 0;
  int max_seq_ = 0;
  int head_dim_ = 0;
};

}  // namespace cudalm
