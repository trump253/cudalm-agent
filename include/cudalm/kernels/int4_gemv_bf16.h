// CUDALM — W4A16 GEMV with BFLOAT16 activation/output (v0.2, Qwen3.5 path).
// Port of CUDALab int4gemv_rowtile4_hx (same upstream source and structure
// as the v0.1 fp16 specialization in int4_gemv.h), with __half replaced by
// __nv_bfloat16 at the activation/output boundaries only.
//
// Math (the stored fp16 scale is the contract — docs/weight_format.md §4,
// extended for bf16 by docs/qwen35_architecture.md §4/§3-row-8):
//   y[n] = Σ_k unpack(W_packed)[n,k] · scale[n, k/128] · x[k]
//   (FP32 accumulation, single BF16 round-to-nearest-even store per output)
//
// The quantization contract is unchanged from v1 (G=128 symmetric, q in
// [-7,7], fp16 scale, low nibble = k=2b); only the activation/output dtype
// is bf16. This is the Qwen3.5 "W4A16" path: bf16 activation, bf16 output.
//
// Provenance: CUDALab commit cb6a6a9, kernels/int4gemv/
// int4gemv_rowtile4_hx.cu (the fp16 port record in int4_gemv.h applies
// 1:1; see docs/provenance.md, "int4gemv_rowtile4_hx (bf16 specialization)").

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudalm {

// y[n] = Σ_k unpack(weight)[n,k] * scales[n, k/128] * x[k]
//
//   weight : packed INT4 [N, K/2] uint8 row-major; low nibble = k = 2b,
//            high nibble = k = 2b+1 (4-bit two's complement)
//   scales : fp16 [N, K/128] row-major (UNCHANGED: the fp16 scale is the
//            contract in both the fp16 and the bf16 specializations)
//   x      : bf16 [K]
//   y      : bf16 [N]
//
// Precondition (host-checked, abort on violation): N >= 1, K % 128 == 0.
//
// Alignment contract (host-checked, never rejects — same as the fp16 path):
// the 16B-vectorized path requires 16B-aligned `weight` and `x` base
// pointers (K % 32 == 0, implied by K % 128 == 0); otherwise the scalar
// fallback kernel runs instead. Legal inputs must never be rejected
// (DeviceBuffer allocations are 256B aligned, so all runtime buffers take
// the vectorized path).
void int4_gemv_bf16(const std::uint8_t* weight, const __half* scales,
                    const __nv_bfloat16* x, __nv_bfloat16* y, int N, int K,
                    cudaStream_t stream);

}  // namespace cudalm
