// CUDALM — v0.6 Phase B: TRUE BATCHED DECODE kernel family.
//
// Each function is the batch (B rows) counterpart of ONE frozen single-row
// kernel from the v0.2/v0.5 kernel family. The per-row math, accumulation
// order, and bf16 rounding boundaries are IDENTICAL to the frozen kernel —
// a batch kernel applied to B rows must produce, for every row, the SAME
// bits as the frozen single kernel applied to that row alone (the
// row-parity contract, pinned and tested in tests/cuda/
// test_qwen35_batch_kernels.cpp).
//
// Upstream check (per the Phase B mandate): the frozen CUDALab tree at
// cb6a6a9ef76394cc66d272c99aa8697db0a34f1e contains NO batched kernels
// (single-token GEMV / paged decode / DeltaNet only — the same kernels the
// frozen single-row family was ported from). These batch kernels therefore
// EXTEND the frozen single-kernel math with a batch dimension (grid gets a
// B axis or a B-prefixed linear index); no new math is introduced. See
// docs/provenance.md (v0.6 Phase B).
//
// Row addressing for stateful/paged ops is HETEROGENEOUS by design: each
// row b may carry its own position (d_positions[b]), its own Delta slot
// (d_slots[b] — indexed into the pool base), and its own block table
// (block_table + b*row_stride, a DEVICE block table; per-row metadata H2D
// is the caller's job, exactly like the frozen single path's block-table
// H2D). Row b may only touch its own slot/pages/logical KV prefix
// [0..position[b]] — no cross-row access.
//
// Raw pointers + cudaStream_t; no PyTorch.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudalm {
namespace kernels {

// ---------------------------------------------------------------------------
// Embedding gather: out[b*H + i] = embed[token_ids[b]*H + i].
// Plain bf16 copies (bit-exact round trip — same as the frozen single
// path's D2D row copy).
// ---------------------------------------------------------------------------
void batch_embed_gather_bf16(const __nv_bfloat16* embed, const int* token_ids,
                             __nv_bfloat16* out, int B, int H,
                             cudaStream_t stream);

// ---------------------------------------------------------------------------
// W4A16 GEMV, B rows: y_b = W * x_b for b = 0..B-1.
// Batched counterpart of int4gemv_rowtile4_bf16_kernel (frozen
// int4_gemv_bf16): same R=4 row tile, same 128-thread block, same strided
// v loop, same 4x LDG.128 x fragment, same single-group lemma, same shfl +
// shared reduction — the x fragment base is offset by b*K and the output
// row base by b*N. Per row: bit-identical to the frozen single call.
// Precondition: K % 128 == 0, 16B-aligned bases (the same contract as the
// frozen vectorized path; the frozen scalar fallback is NOT batched — the
// runtime shapes always satisfy the vectorized contract).
// ---------------------------------------------------------------------------
void batch_int4_gemv_bf16(const std::uint8_t* weight, const __half* scale,
                          const __nv_bfloat16* x, __nv_bfloat16* y, int N,
                          int K, int B, cudaStream_t stream);

// ---------------------------------------------------------------------------
// BF16 GEMV (tied LM head), B rows: y_b = W * x_b.
// Batched counterpart of bf16_gemv_vec4_row_kernel (frozen bf16_gemv):
// one block per (output row, batch row), 256 threads, fp32 accumulation,
// the same warp/shared reduction, one bf16 RNE store — x base offset by
// b*K, out base by b*N. Per row: bit-identical to the frozen single call.
// Precondition: K % 8 == 0, 16B-aligned bases (frozen vectorized contract).
// ---------------------------------------------------------------------------
void batch_bf16_gemv(const __nv_bfloat16* weight, const __nv_bfloat16* x,
                     __nv_bfloat16* y, int N, int K, int B,
                     cudaStream_t stream);

// ---------------------------------------------------------------------------
// Partial rotate-half RoPE, B x unit_rows rows, HETEROGENEOUS positions:
// x/y : bf16 [B*unit_rows, head_dim]; row m belongs to batch row
//       b = m / unit_rows (unit_rows = n_heads for Q, n_kv for K).
// cos_t/sin_t: fp32 [B][rotary_dim/2] device tables (host-computed in fp32
//       with the SAME rope_table_host construction as the frozen path —
//       one table per row's position).
// Per (row, d): the frozen elementwise math VERBATIM (bf16 cos/sin cast,
// r1 = bf16(x*c), r2 = bf16(sign*x_partner*s), y = bf16(r1+r2);
// pass-through dims copied) — only the table index gains the b prefix.
// Per row: bit-identical to the frozen single call at that row's position.
// ---------------------------------------------------------------------------
void batch_partial_rope_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int B,
                             int unit_rows, int head_dim, int rotary_dim,
                             const float* cos_t, const float* sin_t,
                             cudaStream_t stream);

// ---------------------------------------------------------------------------
// Gated DeltaNet, B rows (state addressed by per-row slot):
//
// conv decode: one thread per (channel, row); conv state base =
//   conv_base + d_slots[b] * conv_dim * 3 (the pool's per-ordinal conv
//   buffer, slot stride = conv_dim*3 bf16 — see Qwen35DeltaStatePool);
//   the frozen conv math verbatim (read old state, shift+insert, depthwise
//   conv fp32 acc, one RNE, SiLU one RNE).
// ---------------------------------------------------------------------------
void batch_deltanet_conv_decode_bf16(__nv_bfloat16* conv_base,
                                     const int* d_slots,
                                     const __nv_bfloat16* new_mixed,
                                     const __nv_bfloat16* conv_w,
                                     __nv_bfloat16* conv_out,
                                     __nv_bfloat16* conv_silu, int conv_dim,
                                     int B, cudaStream_t stream);

// conv_silu [B][conv_dim] -> q [B][key_dim] / k [B][key_dim] / v
// [B][value_dim] (the frozen single path's three D2D splits, one kernel;
// plain copies).
void batch_deltanet_conv_split_bf16(const __nv_bfloat16* conv_silu,
                                    __nv_bfloat16* q, __nv_bfloat16* k,
                                    __nv_bfloat16* v, int key_dim,
                                    int value_dim, int B, cudaStream_t stream);

// g/beta preparation, B x n_heads rows: the frozen per-(b,h) math verbatim
// (beta = bf16 sigmoid; g fp32 = -exp(A_log) * softplus(a + dt_bias)).
void batch_deltanet_gbeta_bf16(const __nv_bfloat16* b, const __nv_bfloat16* a,
                               const float* A_log,
                               const __nv_bfloat16* dt_bias,
                               __nv_bfloat16* beta, float* g, int n_heads,
                               int B, cudaStream_t stream);

// Gated delta-rule recurrent decode update, B x n_heads heads:
// rec state base = rec_base + d_slots[b] * n_heads * hd * hd
// (the pool's per-ordinal recurrent buffer, slot stride = rec_elems fp32);
// the frozen block-per-head math VERBATIM (bf16 l2norm chain, block sums,
// A/B column accumulations, in-place fp32 state update) — only the state
// base and the q/k/v/g/beta row offsets gain the b prefix.
void batch_deltanet_delta_rule_fp32(const __nv_bfloat16* q,
                                    const __nv_bfloat16* k,
                                    const __nv_bfloat16* v, const float* g,
                                    const __nv_bfloat16* beta, float* rec_base,
                                    const int* d_slots, __nv_bfloat16* core_out,
                                    int n_heads, int head_dim, float eps,
                                    int B, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Paged KV, B rows (heterogeneous positions, per-row block tables):
// block_table : device int [B][row_stride] (row b's table =
//               block_table + b*row_stride; row_stride = the manager's
//               max_blocks; only [0, position[b]/page_tokens + 1] entries
//               are read — same prefix contract as the frozen path);
// d_positions : device int [B] (row b's position).
//
// write: k_src/v_src [B][n_kv*head_dim] -> scatter into the pages at each
// row's position (the frozen write math verbatim; plain copies).
// ---------------------------------------------------------------------------
void batch_paged_kv_write_bf16(const __nv_bfloat16* k_src,
                               const __nv_bfloat16* v_src,
                               __nv_bfloat16* k_pages, __nv_bfloat16* v_pages,
                               const int* block_table, const int* d_positions,
                               int row_stride, int page_tokens, int n_kv_heads,
                               int head_dim, std::size_t page_stride, int B,
                               cudaStream_t stream);

// Paged causal DECODE attention, B rows:
// q/out : bf16 [B][n_heads][head_dim]; row b attends to its OWN logical
// KV prefix [0..d_positions[b]] (T_b = d_positions[b] + 1) through its OWN
// block table. scratch: bf16 [scores2 | probs], each [B][n_heads][T_max]
// with T_max = max_b(T_b) (rows are PADDED to T_max; row b only reads/writes
// [0..T_b) — the padding is never touched).
// The frozen 3-kernel pipeline (scores dot / three-pass softmax / PV)
// verbatim per (b,h) row: same d-ascending dot, same strided softmax passes
// over [0..T_b), same t-ascending PV, same bf16 RNE boundaries.
void batch_paged_attention_decode_bf16(const __nv_bfloat16* q,
                                       __nv_bfloat16* k_pages,
                                       __nv_bfloat16* v_pages,
                                       const int* block_table,
                                       const int* d_positions, int row_stride,
                                       int page_tokens, __nv_bfloat16* out,
                                       int n_heads, int n_kv_heads,
                                       int head_dim, std::size_t page_stride,
                                       int T_max, int B,
                                       __nv_bfloat16* scratch,
                                       cudaStream_t stream);

}  // namespace kernels
}  // namespace cudalm
