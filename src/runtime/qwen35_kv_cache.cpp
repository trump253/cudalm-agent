// CUDALM — Qwen3.5 bf16 KV cache implementation. CUDALM-native.

#include "cudalm/qwen35_kv_cache.h"

#include "cudalm/cuda_check.h"
#include "cudalm/kernels/qwen35_kernels.h"

namespace cudalm {

Qwen35KvCache::Qwen35KvCache(const Qwen35Config& cfg, cudaStream_t stream)
    : n_kv_(cfg.n_kv_heads), max_seq_(cfg.max_seq_len),
      head_dim_(cfg.head_dim) {
  CUDALM_PRECONDITION(n_kv_ >= 1 && max_seq_ >= 1 && head_dim_ >= 1,
                      "Qwen35KvCache: config KV dims must be positive");
  const std::size_t bytes =
      static_cast<std::size_t>(n_kv_) * max_seq_ * head_dim_ *
      sizeof(__nv_bfloat16);
  k_.allocate(bytes, stream);
  v_.allocate(bytes, stream);
  k_.clear(bytes, stream);
  v_.clear(bytes, stream);
}

void Qwen35KvCache::write(int position, const __nv_bfloat16* k,
                          const __nv_bfloat16* v, cudaStream_t stream) {
  CUDALM_PRECONDITION(
      position >= 0 && position < max_seq_,
      "Qwen35KvCache::write: position out of bounds [0, max_seq_len)");
  kernels::qwen35_kv_write_bf16(
      k_.data<__nv_bfloat16>(), v_.data<__nv_bfloat16>(), k, v, position,
      n_kv_, max_seq_, head_dim_, stream);
}

}  // namespace cudalm
