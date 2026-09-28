// CUDALM — v0.7 Phase B: W4A16 GEMV adaptive-N-row-tile variants +
// EXPERIMENTAL / measured dispatcher for the Qwen3.5-0.8B serving shapes.
//
// STATUS: the experimental dispatcher is BENCHMARK / TEST / PROFILING
// INFRASTRUCTURE only. The production Qwen3.5 runtime calls the FROZEN
// R=4 baseline (int4_gemv_bf16 / kernels::batch_int4_gemv_bf16) directly;
// the measured-table candidate was measured (see the candidate evidence)
// and REJECTED for production because the canonical E2E improvement was
// within noise. The measured table is retained here so the candidate
// can be re-benchmarked/reproduced; it is NOT a production entry point.
//
// The frozen v0.6/v0.7A baseline (src/kernels/int4_gemv_bf16.cu
// int4gemv_rowtile4_bf16_kernel, R=4 row tile, 128-thread block) is
// UNTOUCHED and remains directly callable (paired A/B benchmark + oracle).
//
// The variants here change ONE parameter only — how many output N rows a
// block processes (R = 1 / 2 / 8, grid = ceil(N/R)) — while keeping,
// per output row:
//   * the SAME 128-thread block and threadIdx.x -> v (K) mapping,
//   * the SAME per-thread K-loop order (v ascending),
//   * the SAME in-vector term order (k ascending, lo-before-hi),
//   * the SAME warp reduction (shfl 5-step) and cross-warp reduction
//     (shared[R][4] + shfl 2-step) trees,
//   * the SAME single BF16 round-to-nearest-even store.
// Therefore each variant's output for a given (row, B-row) is BIT-IDENTICAL
// to the frozen R=4 baseline on that row (verified, not assumed:
// tests/cuda/test_w4a16_qwen35_optimized.cpp, all production shapes x
// B=1/2/3). The frozen row-parity contract (batch row b == frozen single
// call on row b, bit-identical) is inherited by every variant.
//
// Experimental / measured dispatcher (int4_gemv_bf16_qwen35_experimental /
// batch_int4_gemv_bf16_qwen35_experimental): selects R per (N, K[, B]) from
// the measured-shape table in the .cu (docs/v07_w4a16_optimization.md). NO
// runtime autotuning. Any legal shape not in the table takes the frozen R=4
// baseline path (bit-compatible generic fallback, including the scalar
// fallback when the 16B alignment contract is not met). NOT used by the
// production runtime.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudalm {

// ---------------------------------------------------------------------------
// Explicit per-variant launchers (benchmark / parity-test primitives).
// Same argument contract as int4_gemv_bf16 (weight [N,K/2] packed INT4,
// scales fp16 [N,K/128], x bf16 [K] (B=1) / [B,K] (batch), y bf16 [N] /
// [B,N]); K % 128 == 0 precondition. When the 16B vectorization contract
// (aligned weight/x bases) is not met, the call falls through to the
// FROZEN int4_gemv_bf16 / batch path (scalar fallback included) — legal
// inputs are never rejected.
// ---------------------------------------------------------------------------

// B=1 variants (grid = ceil(N/R), block 128).
void int4_gemv_bf16_rowtile1(const std::uint8_t* weight, const __half* scales,
                             const __nv_bfloat16* x, __nv_bfloat16* y, int N,
                             int K, cudaStream_t stream);
void int4_gemv_bf16_rowtile2(const std::uint8_t* weight, const __half* scales,
                             const __nv_bfloat16* x, __nv_bfloat16* y, int N,
                             int K, cudaStream_t stream);
void int4_gemv_bf16_rowtile8(const std::uint8_t* weight, const __half* scales,
                             const __nv_bfloat16* x, __nv_bfloat16* y, int N,
                             int K, cudaStream_t stream);

// Batch variants (grid = (ceil(N/R), B), block 128; x bf16 [B,K],
// y bf16 [B,N]; row b output is bit-identical to the frozen single call on
// row b).
void batch_int4_gemv_bf16_rowtile1(const std::uint8_t* weight,
                                   const __half* scales,
                                   const __nv_bfloat16* x, __nv_bfloat16* y,
                                   int N, int K, int B, cudaStream_t stream);
void batch_int4_gemv_bf16_rowtile2(const std::uint8_t* weight,
                                   const __half* scales,
                                   const __nv_bfloat16* x, __nv_bfloat16* y,
                                   int N, int K, int B, cudaStream_t stream);
void batch_int4_gemv_bf16_rowtile8(const std::uint8_t* weight,
                                   const __half* scales,
                                   const __nv_bfloat16* x, __nv_bfloat16* y,
                                   int N, int K, int B, cudaStream_t stream);

// ---------------------------------------------------------------------------
// EXPERIMENTAL / measured dispatcher (benchmark/test/profiling
// infrastructure — NOT a production entry point). R is chosen from the
// measured-shape table (N, K[, B]); any unrecognized legal shape falls back
// to the frozen R=4 baseline path.
// ---------------------------------------------------------------------------
void int4_gemv_bf16_qwen35_experimental(const std::uint8_t* weight,
                                        const __half* scales,
                                        const __nv_bfloat16* x,
                                        __nv_bfloat16* y, int N, int K,
                                        cudaStream_t stream);

void batch_int4_gemv_bf16_qwen35_experimental(const std::uint8_t* weight,
                                              const __half* scales,
                                 const __nv_bfloat16* x, __nv_bfloat16* y,
                                 int N, int K, int B, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Register-count queries for the R=1/2/8 variant kernels (microbench
// reporting; no behavior change). -1 if unavailable.
// ---------------------------------------------------------------------------
int int4_gemv_bf16_rowtile1_regs();
int int4_gemv_bf16_rowtile2_regs();
int int4_gemv_bf16_rowtile8_regs();

}  // namespace cudalm
