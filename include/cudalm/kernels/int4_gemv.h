// CUDALM — W4A16 GEMV (weight-only INT4 × FP16 activation), port of
// CUDALab int4gemv_rowtile4_hx with its scalar fallback. Raw pointers +
// cudaStream_t; no PyTorch.
//
// Math (the stored fp16 scale is the contract — docs/weight_format.md):
//   y[n] = Σ_k unpack(W_packed)[n,k] · scale[n, k/128] · x[k]
//   (FP32 accumulate, single FP16 round-to-nearest-even store per output)
//
// Provenance: CUDALab commit cb6a6a9, kernels/int4gemv/
// int4gemv_rowtile4_hx.cu + int4gemv_common.h (unpack helpers, scalar
// fallback kernel, alignment contract). See docs/provenance.md.

#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudalm {

// y[n] = Σ_k unpack(weight)[n,k] * scales[n, k/128] * x[k]
//
//   weight : packed INT4 [N, K/2] uint8 row-major; low nibble = k = 2b,
//            high nibble = k = 2b+1 (4-bit two's complement)
//   scales : fp16 [N, K/128] row-major
//   x      : fp16 [K]
//   y      : fp16 [N]
//
// Precondition (host-checked, abort on violation): N >= 1, K % 128 == 0.
//
// Alignment contract (host-checked, never rejects): the 16B-vectorized
// path (upstream rowtile4_hx) requires 16B-aligned `weight` and `x` base
// pointers (K % 32 == 0, implied by K % 128 == 0); otherwise the scalar
// fallback kernel (upstream int4gemv_baseline, same code source as the
// upstream per-variant fallback) runs instead. Legal inputs must never be
// rejected (DeviceBuffer allocations are 256B aligned, so all runtime
// buffers take the vectorized path).
void int4_gemv(const std::uint8_t* weight, const __half* scales,
               const __half* x, __half* y, int N, int K, cudaStream_t stream);

}  // namespace cudalm
