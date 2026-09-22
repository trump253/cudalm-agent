// CUDALM — rotary position embedding (RoPE), port of CUDALab
// kernels/rope/rope_v3_half2.cu (operator definition in
// kernels/rope/rope_common.h). Raw pointers + cudaStream_t; no PyTorch.
//
// Interleaved-pair math (fp32 intermediates, one FP16 round-to-nearest-even
// store per output element):
//   y[m, 2*i]   = x[m, 2*i]   * cos[pos[m], i] - x[m, 2*i+1] * sin[pos[m], i]
//   y[m, 2*i+1] = x[m, 2*i+1] * cos[pos[m], i] + x[m, 2*i]   * sin[pos[m], i]
//
// Provenance: CUDALab commit cb6a6a9 (tag v0.7.1). See docs/provenance.md.

#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudalm {
namespace kernels {

// Interleaved-pair RoPE over M (token, head) rows of head_dim D.
//
//   x         : fp16 [M, D] row-major (D % 2 == 0)
//   positions : int64 [M] device memory; row m uses position positions[m]
//   cos_t     : fp16 [L, D/2] row-major, L = max_seq_len (table from the
//               weight file, `attn.rope_cos`)
//   sin_t     : fp16 [L, D/2] row-major (table from the weight file,
//               `attn.rope_sin`)
//   y         : fp16 [M, D] row-major output; must not alias any input
//
// All arithmetic is fp32; each output element is rounded to fp16 (RNE)
// exactly once. No data-dependent checks (no D2H sync); the range
// positions[m] < L is the caller's responsibility (garbage in, garbage
// out — same semantics as the upstream operator).
//
// v0.4.1 alignment contract (host-checked, never rejects): the base
// pointers of x and y must be 4-byte aligned for the __half2 packed path;
// otherwise the scalar fallback path runs instead (same math, slightly
// slower). Alignment of cos_t/sin_t does not matter (scalar 2B loads).
//
// Precondition (host-checked, abort on violation): M >= 1, D >= 2,
// D % 2 == 0.
void rope_fp16(const __half* x, const std::int64_t* positions,
               const __half* cos_t, const __half* sin_t, __half* y, int M,
               int D, cudaStream_t stream);

}  // namespace kernels
}  // namespace cudalm
