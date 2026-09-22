// CUDALM — fp16 elementwise kernels (residual add, SiLU·mul).
//
// 16-byte vectorized (8 halves per thread via uint4); fp32 math, single fp16
// RNE store per element. Provenance: CUDALM-native (docs/provenance.md).

#include "cudalm/kernels/elementwise.h"

#include <cstdint>

#include "cudalm/cuda_check.h"

namespace cudalm {

namespace {

constexpr int kBlock = 256;

bool ptr_aligned(const void* p, std::size_t align) {
  return (reinterpret_cast<std::uintptr_t>(p) & (align - 1)) == 0;
}

__global__ void add_fp16_kernel(const uint4* __restrict__ a,
                                const uint4* __restrict__ b,
                                uint4* __restrict__ y, int nvec) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nvec) return;
  uint4 va = a[i];
  uint4 vb = b[i];
  uint4 vo;
  const __half2* ha = reinterpret_cast<const __half2*>(&va);
  const __half2* hb = reinterpret_cast<const __half2*>(&vb);
  __half2* ho = reinterpret_cast<__half2*>(&vo);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 fa = __half22float2(ha[k]);
    const float2 fb = __half22float2(hb[k]);
    ho[k] = __floats2half2_rn(fa.x + fb.x, fa.y + fb.y);
  }
  y[i] = vo;
}

__global__ void silu_mul_fp16_kernel(const uint4* __restrict__ gate,
                                     const uint4* __restrict__ up,
                                     uint4* __restrict__ y, int nvec) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nvec) return;
  uint4 vg = gate[i];
  uint4 vu = up[i];
  uint4 vo;
  const __half2* hg = reinterpret_cast<const __half2*>(&vg);
  const __half2* hu = reinterpret_cast<const __half2*>(&vu);
  __half2* ho = reinterpret_cast<__half2*>(&vo);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 g = __half22float2(hg[k]);
    const float2 u = __half22float2(hu[k]);
    // silu(x) = x / (1 + exp(-x)), in fp32; then the fp16 RNE store.
    const float sgx = g.x / (1.0f + expf(-g.x));
    const float sgy = g.y / (1.0f + expf(-g.y));
    ho[k] = __floats2half2_rn(sgx * u.x, sgy * u.y);
  }
  y[i] = vo;
}

}  // namespace

void add_fp16(const __half* a, const __half* b, __half* y, std::size_t n,
              cudaStream_t stream) {
  CUDALM_PRECONDITION(n % 8 == 0, "add_fp16 requires n % 8 == 0");
  CUDALM_PRECONDITION(
      ptr_aligned(a, 16) && ptr_aligned(b, 16) && ptr_aligned(y, 16),
      "add_fp16 requires 16B-aligned a/b/y");
  const int nvec = static_cast<int>(n / 8);
  const int grid = (nvec + kBlock - 1) / kBlock;
  add_fp16_kernel<<<grid, kBlock, 0, stream>>>(
      reinterpret_cast<const uint4*>(a), reinterpret_cast<const uint4*>(b),
      reinterpret_cast<uint4*>(y), nvec);
  CUDA_CHECK_LAUNCH();
}

void silu_mul_fp16(const __half* gate, const __half* up, __half* y,
                   std::size_t n, cudaStream_t stream) {
  CUDALM_PRECONDITION(n % 8 == 0, "silu_mul_fp16 requires n % 8 == 0");
  CUDALM_PRECONDITION(ptr_aligned(gate, 16) && ptr_aligned(up, 16) &&
                          ptr_aligned(y, 16),
                      "silu_mul_fp16 requires 16B-aligned gate/up/y");
  const int nvec = static_cast<int>(n / 8);
  const int grid = (nvec + kBlock - 1) / kBlock;
  silu_mul_fp16_kernel<<<grid, kBlock, 0, stream>>>(
      reinterpret_cast<const uint4*>(gate),
      reinterpret_cast<const uint4*>(up), reinterpret_cast<uint4*>(y), nvec);
  CUDA_CHECK_LAUNCH();
}

}  // namespace cudalm
