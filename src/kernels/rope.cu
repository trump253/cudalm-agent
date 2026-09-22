// RoPE v3 — fp16 implementation (packed __half2 + scalar fallback).
//
// Ported 1:1 (math + control flow preserved) from CUDALab
// kernels/rope/rope_v3_half2.cu @ cb6a6a9 (tag v0.7.1), operator contract
// from kernels/rope/rope_common.h. Mechanical translation only:
//   1) namespace cudalab::rope -> cudalm::kernels;
//   2) host entry: the PyTorch ATen dispatcher rope_v3_half2_fwd becomes the
//      raw-pointer rope_fp16(..., cudaStream_t) with the same alignment
//      contract (v0.4.1) and the same scalar-fallback selection;
//   3) the fp32 template branch (T = float) is not ported — the CUDALM v0.1
//      operator is fp16-only (same as the golden container contract), so
//      rope_v3_half2_kernel is de-templated to its __half branch verbatim;
//   4) grid size: int (upstream int64_t n_pairs fits v0.1 M*D comfortably).
//
// v0.4.1 alignment contract (host-checked, never rejects): the base
// pointers of x and out must be 4-byte aligned for the __half2 packed path;
// otherwise the scalar fallback path is selected (same math, slightly
// slower). cos/sin are loaded scalar (2B) and impose no alignment
// requirement. No data-dependent checks (no D2H sync).
//
// Provenance: see docs/provenance.md (rope row).

#include "cudalm/kernels/rope.h"

#include "cudalm/cuda_check.h"

namespace cudalm {
namespace kernels {
namespace {

constexpr int kBlock = 128;

// Element conversion helpers (from rope_common.h; fp16 specializations).
inline __device__ float el_to_float(__half v) { return __half2float(v); }
inline __device__ __half el_from_float(float f) { return __float2half_rn(f); }

// v0.4.1 fallback kernel: x/out base pointers not 4B-aligned. All 2B
// accesses; math statement-for-statement identical to the packed path
// (el_to_float promotion + FP32 rotation + per-value RNE rounding).
__global__ void rope_v3_half2_scalar_kernel(const __half* __restrict__ x,
                                            const std::int64_t* __restrict__ positions,
                                            const __half* __restrict__ cos_t,
                                            const __half* __restrict__ sin_t,
                                            __half* __restrict__ out,
                                            std::int64_t n_pairs, std::int64_t d2) {
  const std::int64_t t =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= n_pairs) return;
  const std::int64_t m = t / d2;
  const std::int64_t i = t - m * d2;
  const std::int64_t pos = positions[m];

  const __half* xp = x + m * (2 * d2) + 2 * i;
  const float a = el_to_float(xp[0]);
  const float b = el_to_float(xp[1]);
  const float c = el_to_float(cos_t[pos * d2 + i]);
  const float s = el_to_float(sin_t[pos * d2 + i]);

  __half* yp = out + m * (2 * d2) + 2 * i;
  yp[0] = el_from_float(a * c - b * s);
  yp[1] = el_from_float(a * s + b * c);
}

// Packed path: one 4B __half2 load of the (a, b) pair, FP32 rotation, one
// packed 4B store. Requires 4B-aligned base pointers of x and out.
__global__ void rope_v3_half2_kernel(const __half* __restrict__ x,
                                     const std::int64_t* __restrict__ positions,
                                     const __half* __restrict__ cos_t,
                                     const __half* __restrict__ sin_t,
                                     __half* __restrict__ out,
                                     std::int64_t n_pairs, std::int64_t d2) {
  const std::int64_t t =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= n_pairs) return;
  const std::int64_t m = t / d2;
  const std::int64_t i = t - m * d2;
  const std::int64_t pos = positions[m];

  const __half* xp = x + m * (2 * d2) + 2 * i;
  const float c = el_to_float(cos_t[pos * d2 + i]);
  const float s = el_to_float(sin_t[pos * d2 + i]);

  __half* yp = out + m * (2 * d2) + 2 * i;
  const __half2 ab = *reinterpret_cast<const __half2*>(xp);
  const float a = __half2float(ab.x);
  const float b = __half2float(ab.y);
  const __half2 y = __floats2half2_rn(a * c - b * s, a * s + b * c);
  *reinterpret_cast<__half2*>(yp) = y;
}

constexpr bool aligned4(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 3u) == 0u;
}

}  // namespace

void rope_fp16(const __half* x, const std::int64_t* positions,
               const __half* cos_t, const __half* sin_t, __half* y, int M,
               int D, cudaStream_t stream) {
  CUDALM_PRECONDITION(M >= 1 && D >= 2 && (D % 2) == 0,
                      "rope_fp16: requires M>=1, D>=2, D%2==0");

  const std::int64_t n_pairs = static_cast<std::int64_t>(M) * (D / 2);
  const std::int64_t d2 = D / 2;
  const int grid = static_cast<int>((n_pairs + kBlock - 1) / kBlock);

  // v0.4.1: both base pointers 4B-aligned -> packed __half2 kernel;
  // otherwise the scalar fallback (baseline-compatible math, same rounding).
  if (aligned4(x) && aligned4(y)) {
    rope_v3_half2_kernel<<<grid, kBlock, 0, stream>>>(x, positions, cos_t,
                                                      sin_t, y, n_pairs, d2);
  } else {
    rope_v3_half2_scalar_kernel<<<grid, kBlock, 0, stream>>>(
        x, positions, cos_t, sin_t, y, n_pairs, d2);
  }
  CUDA_CHECK_LAUNCH();
}

}  // namespace kernels
}  // namespace cudalm
