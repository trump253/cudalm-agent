// CUDALM — RMSNorm fp16, port of CUDALab rmsnorm_v4 (fp16 specialization).
//
// Upstream: commit cb6a6a9, kernels/rmsnorm/rmsnorm_v4.cu
// (rmsnorm_v4_half_kernel + launch_half + v4_precheck, is_fp16 path).
// Math and control flow are preserved 1:1 (register-resident x, 256-thread
// block, warp shfl reduction, rsqrtf(v/H + eps), fp16 RNE store). The only
// deviations are mechanical:
//   * PyTorch host layer (at::Tensor / TORCH_CHECK / getCurrentCUDAStream /
//     C10_CUDA_KERNEL_LAUNCH_CHECK) replaced by raw pointers + cudaStream_t
//     + CUDALM_PRECONDITION + CUDA_CHECK_LAUNCH;
//   * the fp32 specialization is not ported (v0.1 is fp16-only).
// See docs/provenance.md.

#include "cudalm/kernels/rmsnorm.h"

#include <cstddef>
#include <cstdint>

#include "cudalm/cuda_check.h"

namespace cudalm {

namespace {

constexpr int kV4Block = 256;

// 16B-aligned base pointer check (row stride H*2B is a multiple of the
// required alignment for all supported PER, so base alignment suffices).
bool ptr_aligned(const void* p, std::size_t align) {
  return (reinterpret_cast<std::uintptr_t>(p) & (align - 1)) == 0;
}

// Exact port of rmsnorm_v4_half_kernel (CUDALab cb6a6a9).
template <int PER>
__global__ void rmsnorm_v4_half_kernel(const __half* __restrict__ x,
                                       const __half* __restrict__ w,
                                       __half* __restrict__ y, int H,
                                       float eps) {
  static_assert(PER == 4 || PER == 8 || PER == 16 || PER == 32, "PER");
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const __half* __restrict__ xrow = x + static_cast<std::size_t>(row) * H;
  __half* __restrict__ yrow = y + static_cast<std::size_t>(row) * H;
  const int base = tid * PER;  // first element owned by this thread

  __half2 buf[PER / 2];
  float ss = 0.f;

  if (PER % 8 == 0) {
    // 8 consecutive halves = one 16B float4.
    const float4* p = reinterpret_cast<const float4*>(xrow + base);
    const int nvec = PER / 8;
#pragma unroll
    for (int i = 0; i < nvec; i++) {
      float4 v = p[i];
      const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        buf[i * 4 + k] = h[k];
        float2 f = __half22float2(h[k]);
        ss += f.x * f.x + f.y * f.y;
      }
    }
  } else {
    // PER == 4: two 4B half2 loads.
    const __half2* p = reinterpret_cast<const __half2*>(xrow + base);
#pragma unroll
    for (int i = 0; i < PER / 2; i++) {
      __half2 h = p[i];
      buf[i] = h;
      float2 f = __half22float2(h);
      ss += f.x * f.x + f.y * f.y;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    ss += __shfl_down_sync(0xffffffffu, ss, offset);

  const int nwarp = (nthreads + 31) >> 5;
  __shared__ float warp_sums[32];
  __shared__ float s_inv_rms;
  if ((tid & 31) == 0) warp_sums[tid >> 5] = ss;
  __syncthreads();
  if (tid < 32) {
    float v = (tid < nwarp) ? warp_sums[tid] : 0.f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, offset);
    if (tid == 0) s_inv_rms = rsqrtf(v / static_cast<float>(H) + eps);
  }
  __syncthreads();
  const float inv_rms = s_inv_rms;

  if (PER % 8 == 0) {
    const int nvec = PER / 8;
    const float4* wp = reinterpret_cast<const float4*>(w + base);
    float4* q = reinterpret_cast<float4*>(yrow + base);
#pragma unroll
    for (int i = 0; i < nvec; i++) {
      float4 wv = wp[i];
      const __half2* hw = reinterpret_cast<const __half2*>(&wv);
      float4 o;
      __half2* ho = reinterpret_cast<__half2*>(&o);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        float2 f = __half22float2(buf[i * 4 + k]);
        float2 fw = __half22float2(hw[k]);
        ho[k] = __floats2half2_rn(f.x * inv_rms * fw.x, f.y * inv_rms * fw.y);
      }
      q[i] = o;
    }
  } else {
    const __half2* wp = reinterpret_cast<const __half2*>(w + base);
    __half2* q = reinterpret_cast<__half2*>(yrow + base);
#pragma unroll
    for (int i = 0; i < PER / 2; i++) {
      float2 f = __half22float2(buf[i]);
      float2 fw = __half22float2(wp[i]);
      q[i] = __floats2half2_rn(f.x * inv_rms * fw.x, f.y * inv_rms * fw.y);
    }
  }
}

}  // namespace

void rmsnorm_fp16(const __half* x, const __half* w, __half* y, int M, int H,
                  float eps, cudaStream_t stream) {
  // Upstream v4_precheck (fp16 branch), translated to CUDALM.
  CUDALM_PRECONDITION(H % kV4Block == 0,
                      "rmsnorm_fp16 requires H % 256 == 0");
  const int per = H / kV4Block;
  CUDALM_PRECONDITION(per == 4 || per == 8 || per == 16 || per == 32,
                      "rmsnorm_fp16 requires H/256 in {4,8,16,32} "
                      "(H in {1024,2048,4096,8192})");
  const std::size_t align = (per % 8 == 0) ? 16 : 4;
  CUDALM_PRECONDITION(ptr_aligned(x, align) && ptr_aligned(w, align) &&
                          ptr_aligned(y, align),
                      "rmsnorm_fp16 requires x/w/y base pointers "
                      "16B-aligned (PER%8==0) or 4B-aligned (PER==4)");

  dim3 grid(M), block(kV4Block);
  switch (per) {
    case 4:
      rmsnorm_v4_half_kernel<4><<<grid, block, 0, stream>>>(x, w, y, H, eps);
      CUDA_CHECK_LAUNCH();
      return;
    case 8:
      rmsnorm_v4_half_kernel<8><<<grid, block, 0, stream>>>(x, w, y, H, eps);
      CUDA_CHECK_LAUNCH();
      return;
    case 16:
      rmsnorm_v4_half_kernel<16><<<grid, block, 0, stream>>>(x, w, y, H, eps);
      CUDA_CHECK_LAUNCH();
      return;
    case 32:
      rmsnorm_v4_half_kernel<32><<<grid, block, 0, stream>>>(x, w, y, H, eps);
      CUDA_CHECK_LAUNCH();
      return;
  }
  // Unreachable: the preconditions above enforce per in {4,8,16,32}.
  CUDALM_PRECONDITION(false, "rmsnorm_fp16: PER out of range");
}

}  // namespace cudalm
