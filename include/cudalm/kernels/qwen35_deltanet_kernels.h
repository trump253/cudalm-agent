// CUDALM — Qwen3.5 Gated DeltaNet decode kernels (v0.2, Phase C).
//
// Single-GPU autoregressive decode step (batch-1, one token). Math contract:
// docs/qwen35_architecture.md §8 (Gated DeltaNet) + §6.2 (Gated RMSNorm),
// pinned official source transformers fc91372 (torch fallback path). Raw
// pointers + cudaStream_t; no PyTorch.
//
// Persistent state (owned by Qwen35DeltaNetLayer, docs §3-row-10):
//   conv_state      bf16 [conv_dim, 3]        (updated in place each step)
//   recurrent_state fp32 [n_heads, head_dim, head_dim]  (S[h,k,v], in place)
// The update ordering (golden-verified, docs §8) is:
//   decay S = S*exp(g) -> delta update S += k (x) d -> output from UPDATED S;
//   conv state is updated BEFORE the conv output is consumed.
// g is computed in fp32 (A_log stays fp32; dt_bias bf16 -> fp32); beta is
// bf16 (sigmoid rounded); the recurrent state is ALWAYS fp32.

#pragma once

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cudalm {
namespace kernels {

// Depthwise causal conv1d DECODE update with persistent conv state + SiLU.
// Mirrors the pinned torch_causal_conv1d_update rounding boundaries exactly:
//   buf = [cs(3), new(1)] (bf16);  conv_state := [cs1, cs2, new] (in place);
//   c = fp32-acc of conv over buf, ONE bf16 RNE;  out = SiLU(c) with SiLU in
//   fp32 of the bf16 c, ONE bf16 RNE.
//   conv_state : bf16 [conv_dim, 3]   (READ + UPDATED IN-PLACE)
//   new_mixed  : bf16 [conv_dim]      (in_proj_qkv output at this step)
//   conv_w     : bf16 [conv_dim, 4]   (conv1d.weight [conv_dim,1,4] flattened)
//   conv_out   : bf16 [conv_dim]      (c, before SiLU)
//   conv_silu  : bf16 [conv_dim]      (SiLU(c))
void qwen35_deltanet_conv_decode_bf16(
    __nv_bfloat16* conv_state, const __nv_bfloat16* new_mixed,
    const __nv_bfloat16* conv_w, __nv_bfloat16* conv_out,
    __nv_bfloat16* conv_silu, int conv_dim, cudaStream_t stream);

// g / beta preparation (docs §8). One thread per head.
//   beta[h] = bf16( 1 / (1 + exp(-f32(b[h]))) )           (sigmoid, 1 RNE)
//   g[h]    = -expf(A_log[h]) * softplus(f32(a[h]) + f32(dt_bias[h]))
//            (A_log fp32 [n_heads]; a, dt_bias bf16 [n_heads]; g fp32 out)
void qwen35_deltanet_gbeta_bf16(
    const __nv_bfloat16* b, const __nv_bfloat16* a, const float* A_log,
    const __nv_bfloat16* dt_bias, __nv_bfloat16* beta, float* g, int n_heads,
    cudaStream_t stream);

// Gated delta-rule recurrent DECODE update (docs §8). q/k/v enter as bf16;
// the FLA-aligned l2norm on q/k runs with bf16 rounding semantics (bf16
// products, fp32 sum, bf16 rsqrt — NOT a pure-fp32 l2norm), and q is
// additionally scaled by 1/sqrt(head_dim); the NORMALIZED bf16 q/k then feed
// the fp32 recurrent delta-rule / state math (state S is fp32 and updated IN
// PLACE). One block per head (128 threads = value index); the ordering is
// decay -> delta -> output from UPDATED S. num_v_heads == num_key_heads is
// required (pinned 0.8B: 16 == 16, no repeat_interleave).
//   q, k, v   : bf16 [n_heads, head_dim]   (post-conv split)
//   g         : fp32 [n_heads];  beta : bf16 [n_heads]
//   S         : fp32 [n_heads, head_dim, head_dim]  (READ + UPDATED IN PLACE)
//   core_out  : bf16 [n_heads, head_dim]   (fp32 core, one bf16 RNE)
void qwen35_deltanet_delta_rule_fp32(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const __nv_bfloat16* beta, float* S,
    __nv_bfloat16* core_out, int n_heads, int head_dim, float eps,
    cudaStream_t stream);

// Gated RMSNorm (docs §6.2) — the EXACT two-bf16-rounding official flow, NOT
// an fp32 ideal:
//   n  = bf16( x_f32 * rsqrt(mean(x_f32^2) + eps) )   [ROUNDING 1]
//   a  = w * f32(n)                                    (w fp32 [dim], no round)
//   y  = bf16( a * silu(f32(gate)) )                   [ROUNDING 2]
// One block per row (dim threads). x, gate : bf16 [n, dim]; w : fp32 [dim];
// y : bf16 [n, dim].
void qwen35_rmsnorm_gated_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gate, const float* w,
    __nv_bfloat16* y, int n, int dim, float eps, cudaStream_t stream);

}  // namespace kernels
}  // namespace cudalm
