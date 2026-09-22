// CUDALM — decode KV cache (RAII). CUDALM-native (no upstream port).
//
// Flat layout contract (docs/bootstrap_plan_v0.1.md §5):
//   K, V each: fp16 [n_kv_heads][max_seq_len][head_dim] row-major
//   flat offset of row (n, t) = ((n * max_seq_len) + t) * head_dim
//   (t = 0-based position, n = kv head, d = head dim)
//
// KvCache owns two pre-allocated DeviceBuffers (K and V) sized to
// max_seq_len, and exposes the current-token write plus read-only accessors.
// The position is bounds-checked host-side (0 <= position < max_seq_len);
// a violation is a fatal precondition abort (v0.1 has no error-returning
// path).

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cudalm/device_buffer.h"
#include "cudalm/model_config.h"

namespace cudalm {

// RAII decode KV cache. Move-only (owns two device allocations).
class KvCache {
 public:
  // Pre-allocate K and V to [n_kv_heads][max_seq_len][head_dim] fp16.
  // The config's KV dims must be positive (precondition-checked).
  explicit KvCache(const ModelConfig& cfg, cudaStream_t stream = 0);

  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;
  KvCache(KvCache&& other) noexcept = default;
  KvCache& operator=(KvCache&& other) noexcept = default;
  ~KvCache() = default;

  // Write the current token's K/V rows at `position` (all kv heads).
  //   k : fp16 [n_kv_heads, head_dim] (post-RoPE K rows)
  //   v : fp16 [n_kv_heads, head_dim]
  // Precondition (host-checked, abort on violation): 0 <= position
  // < max_seq_len. All other cache rows are left untouched.
  void write(int position, const __half* k, const __half* v,
             cudaStream_t stream);

  // ---- Read-only accessors --------------------------------------------
  const __half* k() const { return k_.data<__half>(); }
  const __half* v() const { return v_.data<__half>(); }
  int n_kv_heads() const { return n_kv_; }
  int max_seq_len() const { return max_seq_; }
  int head_dim() const { return head_dim_; }
  // Total fp16 elements per tensor (K or V).
  std::size_t numel() const {
    return static_cast<std::size_t>(n_kv_) * max_seq_ * head_dim_;
  }
  // Byte size of one tensor (K or V).
  std::size_t bytes() const { return numel() * sizeof(__half); }
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
