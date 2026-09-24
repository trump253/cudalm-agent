// CUDALM — plain BF16 GEMV (BF16 weight × BF16 input → BF16 output), the
// tied-LM-head matmul for the v0.3 full single-token forward.
//
// Math (docs/qwen35_architecture.md §18):
//   y[n] = Σ_k W[n,k] · x[k]   (FP32 accumulation, single BF16 RNE store)
// with W bf16 [N, K] row-major, x bf16 [K], y bf16 [N]. For the LM head:
// N = vocab_size = 248320, K = hidden_size = 1024, W = embed_tokens.weight
// (the tied weight), x = the final-normed hidden, y = logits.
//
// Provenance: adapted from CUDALab commit cb6a6a9, kernels/gemv/
// (gemv_vec4_row.cu — the 16B-vectorized proven variant — + the shared
// gemv_scalar_kernel in gemv_common.h as the fallback), with __half replaced
// by __nv_bfloat16 and the PyTorch bindings stripped to raw pointers +
// cudaStream_t. See docs/provenance.md for the port record.

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cudalm {

// y[n] = Σ_k W[n,k] * x[k]  (FP32 accumulation, one BF16 RNE per output).
//
//   weight : bf16 [N, K] row-major
//   x      : bf16 [K]
//   y      : bf16 [N]
//
// Precondition (host-checked, abort on violation): N >= 1, K >= 1.
//
// Vectorization contract (host-checked, never rejects — same policy as the
// W4A16 GEMV): the 16B-vectorized path requires 16B-aligned `weight` and `x`
// base pointers and K % 8 == 0 (epv = 16/sizeof(bf16) = 8); otherwise the
// scalar fallback kernel runs instead. Legal inputs must never be rejected
// (DeviceBuffer allocations are 256B aligned, so all runtime buffers take the
// vectorized path; K = 1024 is a multiple of 8).
void bf16_gemv(const __nv_bfloat16* weight, const __nv_bfloat16* x,
               __nv_bfloat16* y, int N, int K, cudaStream_t stream);

}  // namespace cudalm
