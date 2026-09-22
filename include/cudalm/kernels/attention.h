// CUDALM — causal decode attention (CUDALM-native, no upstream port).
//
// Small 3-kernel pipeline (no FlashAttention, no persistent kernels, no
// online-softmax — clarity first, per docs/bootstrap_plan_v0.1.md §6-D/E):
//   1. scores : one thread per (query-head, t); q·K dot over head_dim with
//               fp32 accumulation, scaled by 1/sqrt(head_dim);
//   2. softmax: block per query head over [0..position], fp32, max-subtracted;
//   3. PV     : one thread per (query-head, d); fp32 accumulation over
//               t ∈ [0..position], one fp16 RNE store per output element.
//
// All intermediate values stay fp32 (device scratch); only the final out
// tensor is fp16 — matching the golden math contract (fp32 math, one fp16
// RNE at the stage boundary).
//
// Math (t = 0..position, kh(h) = h * n_kv_heads / n_heads):
//   scores[h, t] = dot(q[h, :], K[kh(h)][t, :]) / sqrt(head_dim)
//   probs[h, t]  = softmax over t of scores[h, :]  (subtract row max)
//   out[h, d]    = Σ_t probs[h, t] · V[kh(h)][t, d]

#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudalm {
namespace kernels {

void attention_decode_fp16(const __half* q, const __half* k_cache,
                           const __half* v_cache, int position, __half* out,
                           int n_heads, int n_kv_heads, int head_dim,
                           int max_seq_len, float* scratch,
                           cudaStream_t stream);

}  // namespace kernels
}  // namespace cudalm
