// CUDALM — plain BF16 GEMV (tied-LM-head matmul), v0.3 Phase B.
//
// Adapted from CUDALab commit cb6a6a9, kernels/gemv/gemv_vec4_row.cu (the
// 16B-vectorized proven variant) + kernels/gemv/gemv_common.h (the shared
// gemv_scalar_kernel, used as the fallback). The changes from the fp16
// CUDALab source are mechanical:
//   * __half -> __nv_bfloat16 (el_to_float -> __bfloat162float, el_from_float
//     -> __float2bfloat16_rn — the same single-RNE-at-the-boundary rounding
//     the rest of the Qwen3.5 bf16 kernel family uses);
//   * PyTorch bindings (at::Tensor / getCurrentCUDAStream /
//     C10_CUDA_KERNEL_LAUNCH_CHECK) stripped to raw pointers + cudaStream_t +
//     CUDA_CHECK_LAUNCH;
//   * N = vocab_size (248320), K = hidden_size (1024) for the LM head; the
//     kernel is general (any N, K) so it is reusable.
// Math / control flow are otherwise unchanged: one block per output row,
// 256 threads, FP32 accumulation, warp-shuffle + shared + warp-0 reduction,
// single BF16 RNE store. See docs/provenance.md.

#include "cudalm/kernels/bf16_gemv.h"

#include <cstddef>

#include "cudalm/cuda_check.h"

namespace cudalm {

namespace {

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;  // 8

// 16B unit: 8 __nv_bfloat16 (16B).
union U16 {
  uint4 v;
  __nv_bfloat16 h[8];
};

// Vectorized: 16B load (8 bf16/thread/iter), one block per row. Adapted from
// CUDALab gemv_vec4_row_kernel (T = __nv_bfloat16).
__global__ void bf16_gemv_vec4_row_kernel(const U16* __restrict__ W,
                                          const U16* __restrict__ x,
                                          __nv_bfloat16* __restrict__ out,
                                          int N, int K) {
  const int row = blockIdx.x;
  const U16* __restrict__ wrow = W + static_cast<std::size_t>(row) * (K / 8);
  const int nvec = K / 8;  // 16B vectors per row

  float acc = 0.f;
#pragma unroll 4
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const U16 w = wrow[i];
    const U16 xv = x[i];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      acc += __bfloat162float(w.h[j]) * __bfloat162float(xv.h[j]);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  __shared__ float warp_sums[kWarps];
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = (lane < kWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int off = kWarps / 2; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = __float2bfloat16_rn(acc);
  }
}

// Scalar fallback: one block per row, 256 threads, strided scalar loads.
// Adapted from CUDALab gemv_scalar_kernel (T = __nv_bfloat16).
__global__ void bf16_gemv_scalar_kernel(const __nv_bfloat16* __restrict__ W,
                                        const __nv_bfloat16* __restrict__ x,
                                        __nv_bfloat16* __restrict__ out, int N,
                                        int K) {
  const int row = blockIdx.x;
  const __nv_bfloat16* __restrict__ wrow =
      W + static_cast<std::size_t>(row) * K;
  float acc = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    acc += __bfloat162float(wrow[k]) * __bfloat162float(x[k]);

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  __shared__ float warp_sums[kWarps];
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = (lane < kWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int off = kWarps / 2; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = __float2bfloat16_rn(acc);
  }
}

bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

}  // namespace

void bf16_gemv(const __nv_bfloat16* weight, const __nv_bfloat16* x,
               __nv_bfloat16* y, int N, int K, cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "bf16_gemv requires N >= 1");
  CUDALM_PRECONDITION(K >= 1, "bf16_gemv requires K >= 1");

  // Vectorized path: W base 16B ∧ x base 16B ∧ K % 8 == 0 (epv = 8);
  // otherwise the scalar fallback (legal inputs must never be rejected).
  if (K % 8 == 0 && aligned16(weight) && aligned16(x)) {
    bf16_gemv_vec4_row_kernel<<<N, kBlock, 0, stream>>>(
        reinterpret_cast<const U16*>(weight), reinterpret_cast<const U16*>(x),
        y, N, K);
    CUDA_CHECK_LAUNCH();
  } else {
    bf16_gemv_scalar_kernel<<<N, kBlock, 0, stream>>>(weight, x, y, N, K);
    CUDA_CHECK_LAUNCH();
  }
}

}  // namespace cudalm
