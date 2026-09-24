// CUDALM — Qwen3.5 BF16 kernel family (v0.2, Phase B full-attention path).
// CUDALM-native (no upstream port). All ops mirror the pinned official
// model's dtype flow (transformers fc91372, docs/qwen35_architecture.md
// §5-§7) so the golden (pinned-transformers reference with the W4A16 GEMV
// contract) and the runtime round at the SAME stage boundaries:
//
//   * zero-centered RMSNorm: fp32 math, ONE bf16 RNE at the boundary
//   * partial RoPE: cos/sin fp32 (device table) -> bf16 before the multiply;
//     per element: bf16(x*cos_bf16), bf16(rot*sin_bf16), bf16(sum of the
//     two bf16 products); pass-through dims are copied untouched
//   * decode attention: bf16 QK matmul (fp32 acc -> bf16), bf16 × scaling,
//     softmax in fp32 -> bf16, bf16 PV matmul (fp32 acc -> bf16)
//   * elementwise: fp32 math inside, one bf16 RNE at the boundary
//     (residual add; silu(gate)*up with silu rounded to bf16 FIRST;
//     sigmoid(gate) rounded to bf16 FIRST, then the multiply)
//
// Raw pointers + cudaStream_t; no PyTorch.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cudalm {
namespace kernels {

// y[r] = bf16( f32(x[r]) * rsqrt(mean(f32(x[r])^2) + eps) * (1 + f32(w)) )
// for r = 0..M-1, w shared over all M rows (the zero-centered weight).
// fp32 accumulation (per-thread partials + warp/block reduction); the
// official rounding is a SINGLE bf16 RNE after the full fp32 chain
// (pinned Qwen3_5RMSNorm.forward).
//
// Precondition (host-checked, abort on violation): M >= 1, H % 256 == 0
// and H/256 in {1, 4} (H in {256, 1024} — the Qwen3.5 layer-norm and
// per-head q/k-norm widths); x, w, y base pointers 4B-aligned (8B for the
// H=1024 path; DeviceBuffer allocations are 256B aligned, so this holds
// for all runtime buffers).
void qwen35_rmsnorm_zc_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                            __nv_bfloat16* y, int M, int H, float eps,
                            cudaStream_t stream);

// Split the fused q_proj output (rows h*2*head_dim .. h*2*head_dim+head_dim-1
// = query head h, rows h*2*head_dim+head_dim .. h*2*head_dim+2*head_dim-1
// = gate head h, h = 0..n_heads-1 — the official `chunk(2, dim=-1)` layout)
// into interleaved-free per-head rows:
//   q[h*head_dim + d]      = fused[h*2*head_dim + d]
//   gate[h*head_dim + d]   = fused[h*2*head_dim + head_dim + d]
// Plain copies (bit-exact); no math.
void qwen35_split_q_gate_bf16(const __nv_bfloat16* fused, __nv_bfloat16* q,
                              __nv_bfloat16* gate, int n_heads, int head_dim,
                              cudaStream_t stream);

// Partial rotary (Qwen3.5 rotate-half, first rotary_dim dims of each row;
// the remaining head_dim - rotary_dim dims pass through unchanged).
//
//   x, y : bf16 [M, head_dim] row-major (y must not alias x)
//   cos_t: fp32 [rotary_dim/2] device; cos_t[j] = cos(p * inv_freq[j])
//          computed HOST-side in fp32 from rope_theta (official
//          inv_freq[j] = 1/(theta^(2j/rotary_dim)))
//   sin_t: fp32 [rotary_dim/2] device, same construction
//
// Per row m, per d < rotary_dim (j = d/2; rotate-half partner):
//   c = bf16(cos_t[j]); s = bf16(sin_t[j])
//   r1 = bf16(f32(x[m,d]) * f32(c))
//   r2 = bf16(f32(sign * x[m, partner(d)]) * f32(s))
//        partner: d < rd/2 -> (d + rd/2, sign -1); else (d - rd/2, sign +1)
//   y[m,d] = bf16(f32(r1) + f32(r2))
// d >= rotary_dim: y[m,d] = x[m,d].
//
// Precondition (host-checked, abort on violation): M >= 1, head_dim >= 2,
// rotary_dim in {2,4,...,head_dim} (even), head_dim % 128 == 0 (runtime
// sizes: head_dim=256, rotary_dim=64).
void qwen35_partial_rope_bf16(const __nv_bfloat16* x, __nv_bfloat16* y,
                              int M, int head_dim, int rotary_dim,
                              const float* cos_t, const float* sin_t,
                              cudaStream_t stream);

// Scatter the current token's K/V rows into the bf16 cache at `position`.
//   k_cache/v_cache: bf16 [n_kv_heads][max_seq_len][head_dim] row-major
//   k, v           : bf16 [n_kv_heads, head_dim] (current rows)
// Plain bf16 copies (bit-exact round trip).
//
// Precondition (host-checked, abort on violation): n_kv_heads >= 1,
// head_dim >= 1, 0 <= position < max_seq_len.
void qwen35_kv_write_bf16(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                          const __nv_bfloat16* k, const __nv_bfloat16* v,
                          int position, int n_kv_heads, int max_seq_len,
                          int head_dim, cudaStream_t stream);

// Causal decode attention over cache rows [0..position] (GQA: query head h
// reads kv head h * n_kv_heads / n_heads). 3-kernel pipeline mirroring the
// official eager bf16 dtype flow (docs/qwen35_architecture.md §7):
//   1. scores: s1 = bf16(fp32 dot(q[h], K[kh][t])); s2 = bf16(f32(s1)*scale)
//      (scale = head_dim^-0.5, passed in; the official `matmul(...)*scaling`
//      on bf16)
//   2. softmax over t in fp32 (max-subtracted), probs = bf16(e/sum)
//      (official softmax(dtype=fp32).to(bf16))
//   3. out[h,d] = bf16( Σ_t f32(probs[t]) * f32(V[kh][t,d]) )
//      (official bf16 matmul: fp32 accumulation, one bf16 RNE)
//
//   q       : bf16 [n_heads, head_dim] (post-RoPE)
//   k_cache : bf16 [n_kv_heads][max_seq_len][head_dim]
//   v_cache : bf16 [n_kv_heads][max_seq_len][head_dim]
//   out     : bf16 [n_heads, head_dim]
//   scratch : bf16 [2 * n_heads * max_seq_len] (caller-owned; layout
//             [scores2 | probs], each n_heads*max_seq_len)
//
// Precondition (host-checked, abort on violation): n_heads % n_kv_heads ==
// 0, all dims >= 1, 0 <= position < max_seq_len.
void qwen35_attention_decode_bf16(const __nv_bfloat16* q,
                                  const __nv_bfloat16* k_cache,
                                  const __nv_bfloat16* v_cache,
                                  int position, __nv_bfloat16* out,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  int max_seq_len, __nv_bfloat16* scratch,
                                  cudaStream_t stream);

// y[i] = bf16(f32(a[i]) + f32(b[i])) — the Qwen3.5 residual contract
// (official `residual + hidden_states` on bf16 tensors: fp32 add of the two
// bf16 values, one bf16 RNE).
//
// Precondition (host-checked, abort on violation): n % 8 == 0; a, b, y
// 16-byte aligned (DeviceBuffer allocations are 256B aligned).
void qwen35_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                     __nv_bfloat16* y, std::size_t n, cudaStream_t stream);

// y[i] = bf16(f32(silu_bf16) * f32(up[i])) with
// silu_bf16 = bf16(f32(gate) / (1 + exp(-f32(gate)))) — the official
// SwiGLU gate (act_fn(gate_proj(x)) rounded to bf16 FIRST, then the bf16
// multiply with up_proj(x)).
//
// Precondition (host-checked, abort on violation): n % 8 == 0; gate, up, y
// 16-byte aligned.
void qwen35_silu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                          __nv_bfloat16* y, std::size_t n,
                          cudaStream_t stream);

// y[i] = bf16(f32(attn[i]) * f32(sig_bf16)) with
// sig_bf16 = bf16(1 / (1 + exp(-f32(gate[i])))) — the official attention
// gate (torch.sigmoid(gate) rounded to bf16 FIRST, then the bf16 multiply
// with the attention output).
//
// Precondition (host-checked, abort on violation): n % 8 == 0; attn, gate,
// y 16-byte aligned.
void qwen35_gate_mul_bf16(const __nv_bfloat16* attn, const __nv_bfloat16* gate,
                          __nv_bfloat16* y, std::size_t n,
                          cudaStream_t stream);

}  // namespace kernels
}  // namespace cudalm
