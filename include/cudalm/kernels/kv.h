// CUDALM — decode KV-cache write kernel.
//
// Flat layout contract (docs/bootstrap_plan_v0.1.md §5, CUDALM-native —
// no upstream port):
//   K, V each: fp16 [n_kv_heads][max_seq_len][head_dim] row-major
//   flat offset of row (n, t) = ((n * max_seq_len) + t) * head_dim
//   (t = 0-based position, n = kv head, d = head dim)
//
// At decode position `position` the caller writes the current
// k[n, position, :] / v[n, position, :] for ALL n_kv_heads; attention
// later reads [0..position] inclusive for each kv head it uses.

#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudalm {
namespace kernels {

// Scatter the current token's K/V rows into the cache at `position`.
//
//   k_cache : fp16 [n_kv_heads][max_seq_len][head_dim] (K storage, modified)
//   v_cache : fp16 [n_kv_heads][max_seq_len][head_dim] (V storage, modified)
//   k       : fp16 [n_kv_heads, head_dim] current post-RoPE K rows
//   v       : fp16 [n_kv_heads, head_dim] current V rows
//   position: 0 <= position < max_seq_len
//
// All other cache rows are left untouched.
//
// Precondition (host-checked, abort on violation): n_kv_heads >= 1,
// head_dim >= 1, 0 <= position < max_seq_len.
void kv_write_fp16(__half* k_cache, __half* v_cache, const __half* k,
                   const __half* v, int position, int n_kv_heads,
                   int max_seq_len, int head_dim, cudaStream_t stream);

}  // namespace kernels
}  // namespace cudalm
