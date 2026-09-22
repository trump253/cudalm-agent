// CUDALM — decode KV cache (RAII) implementation. CUDALM-native.

#include "cudalm/kv_cache.h"

#include "cudalm/cuda_check.h"
#include "cudalm/kernels/kv.h"

namespace cudalm {

KvCache::KvCache(const ModelConfig& cfg, cudaStream_t stream) {
  CUDALM_PRECONDITION(
      cfg.n_kv_heads >= 1 && cfg.max_seq_len >= 1 && cfg.head_dim >= 1,
      "KvCache: requires n_kv_heads>=1, max_seq_len>=1, head_dim>=1");
  n_kv_ = cfg.n_kv_heads;
  max_seq_ = cfg.max_seq_len;
  head_dim_ = cfg.head_dim;
  const std::size_t bytes = numel() * sizeof(__half);
  k_.allocate(bytes, stream);
  v_.allocate(bytes, stream);
  // Zero-init so untouched rows have a deterministic value (tests pin
  // "all rows except written ones are zero").
  k_.clear(bytes, stream);
  v_.clear(bytes, stream);
}

void KvCache::write(int position, const __half* k, const __half* v,
                    cudaStream_t stream) {
  CUDALM_PRECONDITION(
      position >= 0 && position < max_seq_,
      "KvCache::write: position out of bounds [0, max_seq_len)");
  kernels::kv_write_fp16(k_.data<__half>(), v_.data<__half>(), k, v, position,
                         n_kv_, max_seq_, head_dim_, stream);
}

}  // namespace cudalm
