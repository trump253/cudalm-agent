// CUDALM — decode KV-cache write kernel (CUDALM-native, no upstream port).
//
// Flat layout contract (docs/bootstrap_plan_v0.1.md §5):
//   K, V each: fp16 [n_kv_heads][max_seq_len][head_dim] row-major
//   flat offset of row (n, t) = ((n * max_seq_len) + t) * head_dim
//
// One thread per (n, d) element of the current token: copies the row
// element from the packed [n_kv_heads, head_dim] source into the cache row
// at `position`. Plain fp16 store (no math), so a round trip is bit-exact.
//
// Provenance: CUDALM-native (see docs/provenance.md, "CUDALM-native").

#include "cudalm/kernels/kv.h"

#include "cudalm/cuda_check.h"

namespace cudalm {
namespace kernels {
namespace {

constexpr int kBlock = 128;

__global__ void kv_write_fp16_kernel(const __half* __restrict__ k_src,
                                     const __half* __restrict__ v_src,
                                     __half* __restrict__ k_dst,
                                     __half* __restrict__ v_dst,
                                     int position, int max_seq_len,
                                     int head_dim) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int n = idx / head_dim;
  const int d = idx - n * head_dim;
  const std::size_t dst =
      (static_cast<std::size_t>(n) * max_seq_len + position) * head_dim + d;
  k_dst[dst] = k_src[idx];
  v_dst[dst] = v_src[idx];
}

}  // namespace

void kv_write_fp16(__half* k_cache, __half* v_cache, const __half* k,
                   const __half* v, int position, int n_kv_heads,
                   int max_seq_len, int head_dim, cudaStream_t stream) {
  CUDALM_PRECONDITION(
      n_kv_heads >= 1 && head_dim >= 1 && max_seq_len >= 1 &&
          position >= 0 && position < max_seq_len,
      "kv_write_fp16: requires n_kv_heads>=1, head_dim>=1, max_seq_len>=1, "
      "0<=position<max_seq_len");

  const int total = n_kv_heads * head_dim;
  const int grid = (total + kBlock - 1) / kBlock;
  kv_write_fp16_kernel<<<grid, kBlock, 0, stream>>>(k, v, k_cache, v_cache,
                                                    position, max_seq_len,
                                                    head_dim);
  CUDA_CHECK_LAUNCH();
}

}  // namespace kernels
}  // namespace cudalm
