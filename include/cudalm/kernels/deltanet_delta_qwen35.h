// CUDALM — v0.7 Phase C: DeltaNet delta-rule state-update variants +
// EXPERIMENTAL / measured dispatcher for the Qwen3.5-0.8B serving config
// (n_heads = 16, head_dim = 128; B = 1/2/3).
//
// STATUS: benchmark/test/profiling infrastructure only. The production
// Qwen3.5 runtime calls the FROZEN baseline directly
// (kernels::qwen35_deltanet_delta_rule_fp32 /
// kernels::batch_deltanet_delta_rule_fp32).
//
// The frozen baselines (src/kernels/qwen35_deltanet_kernels.cu
// deltanet_delta_kernel, grid = n_heads, block = 128; and
// src/kernels/batch_decode.cu batch_deltanet_delta_kernel, grid =
// n_heads*B) are UNTOUCHED and remain directly callable (paired A/B
// benchmark + oracle).
//
// All candidates change ONLY the PARALLELIZATION / data movement, never
// the per-element math. Each candidate keeps, for every state/output
// element, the SAME fp32 operation sequence as the frozen kernel:
//   * the l2norm / qk block reductions execute the SAME 128-thread tree
//     (4 x warp-shfl 5-step + smem[4] + ((s0+s1)+s2)+s3) with the SAME
//     element-to-thread mapping (thread t owns element t);
//   * the per-value chains A_t = sum_k S[h,k,t]*exp_g*k_s[k] and
//     B_t = sum_k S[h,k,t]*exp_g*q_s[k] are accumulated sequentially in
//     k-ascending order in fp32 by ONE thread per value (no split of the
//     accumulation, no atomics, no re-association);
//   * d_t = (v_t - A_t)*beta, o_t = B_t + d_t*qk (ONE bf16 RNE) and
//     S[h,k,t] = S[h,k,t]*exp_g + k_s[k]*d_t are term-for-term identical;
//   * the BF16 rounding boundaries (bf16 square products in l2norm,
//     bf16 t/sqrt/reciprocal/scale, bf16 core_out) are at the same
//     program points.
// Therefore every candidate's core_out AND its updated recurrent state
// are BIT-IDENTICAL to the frozen baseline on the same input (verified:
// tests/cuda/test_deltanet_delta_qwen35_optimized.cpp, B=1/2/3, multi-
// seed, multi-step state chains).

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cudalm {
namespace kernels {

// ---------------------------------------------------------------------------
// Explicit per-variant launchers (benchmark / parity-test primitives).
// Same argument contract as the frozen entry points (head_dim == 128
// precondition; legal inputs are never rejected; the B=1 state pointer is
// S[h,k,v] fp32 [n_heads,128,128], the batch state base is
// rec_base + d_slots[b]*n_heads*hd*hd).
// ---------------------------------------------------------------------------

// B=1 candidates (one token):
//   vreg   : 1 block/head, 128 threads; the 128-deep state column is kept
//            REGISTER-RESIDENT across the A/B pass and the in-place update
//            pass (no S reload in the update pass).
//   vvec   : 1 block/head, 128 threads; the A/B + update passes run on
//            threads 0..31 with 4 values per thread and float4 state loads
//            (4 independent value chains per thread; the update pass
//            re-reads the column, L1-hot).
//   vchunk : 2-kernel: (1) the exact-tree reduction pass (q_s/k_s/qk per
//            head to a small fp32 scratch) in 128-thread blocks; (2) the
//            state pass in grid (n_heads,4) blocks of 32 threads, one value
//            per thread, register-resident column.
void deltanet_delta_vreg(const __nv_bfloat16* q, const __nv_bfloat16* k,
                         const __nv_bfloat16* v, const float* g,
                         const __nv_bfloat16* beta, float* S,
                         __nv_bfloat16* core_out, int n_heads, int head_dim,
                         float eps, cudaStream_t stream);
void deltanet_delta_vvec(const __nv_bfloat16* q, const __nv_bfloat16* k,
                         const __nv_bfloat16* v, const float* g,
                         const __nv_bfloat16* beta, float* S,
                         __nv_bfloat16* core_out, int n_heads, int head_dim,
                         float eps, cudaStream_t stream);
void deltanet_delta_vchunk(const __nv_bfloat16* q, const __nv_bfloat16* k,
                           const __nv_bfloat16* v, const float* g,
                           const __nv_bfloat16* beta, float* S,
                           __nv_bfloat16* core_out, int n_heads, int head_dim,
                           float eps, cudaStream_t stream);

// Batch candidates (B tokens; same row-parity contract as the frozen batch
// kernel: row b is bit-identical to the frozen single call on row b).
void batch_deltanet_delta_vreg(const __nv_bfloat16* q,
                               const __nv_bfloat16* k,
                               const __nv_bfloat16* v, const float* g,
                               const __nv_bfloat16* beta, float* rec_base,
                               const int* d_slots, __nv_bfloat16* core_out,
                               int n_heads, int head_dim, float eps, int B,
                               cudaStream_t stream);
void batch_deltanet_delta_vvec(const __nv_bfloat16* q,
                               const __nv_bfloat16* k,
                               const __nv_bfloat16* v, const float* g,
                               const __nv_bfloat16* beta, float* rec_base,
                               const int* d_slots, __nv_bfloat16* core_out,
                               int n_heads, int head_dim, float eps, int B,
                               cudaStream_t stream);
void batch_deltanet_delta_vchunk(const __nv_bfloat16* q,
                                 const __nv_bfloat16* k,
                                 const __nv_bfloat16* v, const float* g,
                                 const __nv_bfloat16* beta, float* rec_base,
                                 const int* d_slots,
                                 __nv_bfloat16* core_out, int n_heads,
                                 int head_dim, float eps, int B,
                                 cudaStream_t stream);

// ---------------------------------------------------------------------------
// EXPERIMENTAL / measured dispatcher (benchmark/test/profiling
// infrastructure — NOT a production entry point). Selects the variant per
// (n_heads[, B]) from the measured table in the .cu; any unrecognized legal
// shape falls back to the frozen baseline path. NOT used by the production
// runtime.
// ---------------------------------------------------------------------------
void deltanet_delta_rule_fp32_qwen35_experimental(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const __nv_bfloat16* beta, float* S,
    __nv_bfloat16* core_out, int n_heads, int head_dim, float eps,
    cudaStream_t stream);

void batch_deltanet_delta_rule_fp32_qwen35_experimental(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const __nv_bfloat16* beta, float* rec_base,
    const int* d_slots, __nv_bfloat16* core_out, int n_heads, int head_dim,
    float eps, int B, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Register-count queries for the variant kernels (microbench reporting).
// -1 if unavailable.
// ---------------------------------------------------------------------------
int deltanet_delta_vreg_regs();
int deltanet_delta_vvec_regs();
int deltanet_delta_vchunk_regs();

}  // namespace kernels
}  // namespace cudalm
