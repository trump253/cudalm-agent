// CUDALM — Qwen3.5 paged-KV decode kernels (v0.5 Phase B). CUDALM-native.
//
// The multi-sequence replacement for the contiguous decode cache
// [n_kv][max_seq_len][head_dim] (qwen35_kernels.h). ONE full-attention
// layer's page storage is a flat page array
//
//   bf16 [num_pages][n_kv_heads][page_tokens][head_dim]   (K and V each)
//
// and a DEVICE block table (int32, logical block -> physical page id)
// resolves token t to
//
//   page    = block_table[t / page_tokens]
//   offset  = t % page_tokens
//   row(n,t) = pages[page * page_stride + (n * page_tokens + offset) * head_dim]
//
// where `page_stride` (in bf16 elements) is the distance between two
// consecutive pages of the SAME layer ordinal. In the Phase-A pool layout
// (row-major [n_full][num_pages][n_kv_heads][page_tokens][head_dim]) the
// pages of one ordinal are ADJACENT, so it is EXACTLY
// `n_kv_heads*page_tokens*head_dim` (== Qwen35KvPagePool::page_elems() ==
// Qwen35KvPagePool::page_stride_elems()); ordinals are
// `num_pages * n_kv_heads * page_tokens * head_dim` apart, and a call
// selects its ordinal by passing `k_page(ord, 0)` as the page-array base.
// The ordinal stride is NOT the kernel page_stride (passing it as one
// would land "page p" in ordinal (ord+p)'s page 0).
//
// So physical pages may be completely non-contiguous while the attention
// still reads tokens 0..position in LOGICAL order. There is NO host-side
// gather and NO materialization of a contiguous cache: every kernel reads
// the page indirection directly (that is the point of this gate).
//
// BIT-EXACTNESS CONTRACT (pinned by tests/cuda/test_paged_kv.cpp): for the
// same logical K/V rows and the same q, the paged pipeline produces
// BIT-IDENTICAL output to the frozen contiguous pipeline
// (qwen35_kv_write_bf16 + qwen35_attention_decode_bf16). The kernels below
// keep the frozen accumulation order EXACTLY (scores: d = 0..head_dim per
// (h,t); PV: t = 0..T-1 per (h,d); softmax is unchanged — it operates on
// the same scores2/probs buffers), so only the row ADDRESS changes.
//
// STALE-BLOCK-TABLE CONTRACT: the kernels read ONLY block-table entries
// [0, position / page_tokens]. Entries beyond that (stale/garbage/out-of-
// range sentinels) must never be dereferenced — test_paged_kv plants
// out-of-range sentinels in the stale region and runs under
// compute-sanitizer to prove it.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cudalm {
namespace kernels {

// Scatter the current token's K/V rows into the paged cache at `position`.
//   k_src/v_src : bf16 [n_kv_heads, head_dim] (post-RoPE K, raw V)
//   k_pages/v_pages: base of the layer's page storage; page p slice is at
//                 k_pages + p * page_stride, a bf16
//                 [n_kv_heads, page_tokens, head_dim] block
//   page_stride : bf16 elements between consecutive pages (see header)
//   block_table : int32 [num_blocks] DEVICE (physical page per logical
//                 block; only block_table[position / page_tokens] is read)
// Plain bf16 copies (bit-exact round trip, same as the contiguous write).
// Precondition (host-checked, abort): n_kv_heads >= 1, head_dim >= 1,
// page_tokens >= 1, page_stride >= n_kv_heads*page_tokens*head_dim,
// 0 <= position, position/page_tokens < num_blocks.
void qwen35_paged_kv_write_bf16(const __nv_bfloat16* k_src,
                                const __nv_bfloat16* v_src,
                                __nv_bfloat16* k_pages,
                                __nv_bfloat16* v_pages,
                                const int* block_table, int position,
                                int page_tokens, int n_kv_heads, int head_dim,
                                std::size_t page_stride, cudaStream_t stream);

// Paged causal decode attention over logical tokens [0..position] (GQA:
// query head h reads kv head h * n_kv_heads / n_heads). Same 3-kernel
// pipeline and the SAME bf16 rounding boundaries as
// qwen35_attention_decode_bf16 (docs §7):
//   1. s1 = bf16(fp32 dot(q[h], K_row(h,t))); s2 = bf16(f32(s1)*scale)
//   2. softmax over t in fp32 (max-subtracted) -> bf16 probs
//   3. out[h,d] = bf16( Sum_t f32(probs[t]) * f32(V_row(h,t,d)) )
// with K_row/V_row resolved through the device block table (see header).
//
//   q       : bf16 [n_heads, head_dim] (post-RoPE)
//   k_pages : base of the layer's K page storage (see qwen35_paged_kv_write_bf16)
//   v_pages : same for V
//   page_stride: bf16 elements between consecutive pages of this layer
//   block_table: int32 [num_blocks] DEVICE
//   out     : bf16 [n_heads, head_dim]
//   scratch : bf16 [2 * n_heads * (position+1)] caller-owned; layout
//             [scores2 | probs], each n_heads*(position+1) — SAME layout as
//             the contiguous kernel with max_seq_len = position+1.
// Precondition (host-checked, abort): n_heads % n_kv_heads == 0, all dims
// >= 1, page_stride >= n_kv_heads*page_tokens*head_dim, 0 <= position,
// (position/page_tokens)+1 <= num_blocks.
void qwen35_paged_attention_decode_bf16(const __nv_bfloat16* q,
                                        const __nv_bfloat16* k_pages,
                                        const __nv_bfloat16* v_pages,
                                        const int* block_table, int position,
                                        int page_tokens, __nv_bfloat16* out,
                                        int n_heads, int n_kv_heads,
                                        int head_dim, std::size_t page_stride,
                                        __nv_bfloat16* scratch,
                                        cudaStream_t stream);

}  // namespace kernels
}  // namespace cudalm
