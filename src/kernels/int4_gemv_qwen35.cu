// CUDALM — v0.7 Phase B: W4A16 GEMV adaptive-N-row-tile variants +
// EXPERIMENTAL / measured dispatcher (Qwen3.5-0.8B serving shapes).
//
// STATUS: benchmark/test/profiling infrastructure. The production Qwen3.5
// runtime calls the FROZEN R=4 baseline directly (int4_gemv_bf16 /
// kernels::batch_int4_gemv_bf16). The measured-table candidate was
// benchmarked at V07B_CANDIDATE_SHA and REJECTED for production (canonical
// E2E improvement within noise; docs/v07_w4a16_optimization.md); the table
// is retained so the candidate can be re-benchmarked/reproduced.
//
// STRUCTURE (identical to the frozen rowtile4 baseline — the ONLY change
// is R, the number of output N rows per block):
//
//   grid  = ceil(N/R) [batch: (ceil(N/R), B)], block = 128 threads
//   per thread, strided v loop v = tid, tid+128, ... (v < K/32):
//     4x LDG.128 x fragment loaded ONCE into registers (xh[16]), reused
//     across the R rows;
//     inner unroll R: row = R*blockIdx.x + r (guard row < N):
//       1x LDG.128 W vector + 1x fp16 scale (single-group lemma g = v>>2)
//       + 32 nibble unpack + 32x ((q*s)*x) into acc[r];
//   reduction: per acc[r] warp shfl 5-step (16->1) + shared[R][4] +
//   warp-0 shfl 2-step; lane 0 stores out[row] = bf16_rn(t).
//
// The batch axis mirrors the frozen batch kernel exactly: blockIdx.y = b
// offsets the activation row (b*K bf16 = b*4*nvec uint4) and the output
// row (b*N); the B=1 launch is a 1-D grid (blockIdx.y = 0).
//
// NUMERICS (the EXACT contract): per output row, the per-thread term
// sequence (v ascending, in-vector k ascending, lo-before-hi) and BOTH
// reduction trees are term-for-term identical to the frozen
// int4gemv_rowtile4_bf16_kernel / batch_int4gemv_rowtile4_bf16_kernel, so
// each row's BF16 output is BIT-IDENTICAL to the frozen baseline on that
// row (and the frozen row-parity contract — batch row b == frozen single
// call on row b — is inherited by every variant). No K partitioning, no
// split-K, no atomics, no changed blockDim: the v-mapping, K-loop order,
// reduction tree and rounding boundary are all frozen. Pinned by
// tests/cuda/test_w4a16_qwen35_optimized.cpp (all production shapes x
// B=1/2/3, edge inputs) and the full-model EXACT gates.
//
// PROVENANCE: the frozen baseline is a port of CUDALab cb6a6a9
// int4gemv_rowtile4_hx (docs/provenance.md); the R=8 structure mirrors the
// CUDALab int4gemv_rowtile8 experiment (INT4GEMV-0003) with the same
// bit-identity argument. The R=1/2 small-tile variants and the Qwen3.5
// shape-keyed dispatcher are CUDALM-specific (the CUDALab incumbent
// targets larger GEMV shapes and has no batch axis).

#include "cudalm/kernels/int4_gemv_qwen35.h"

#include <cstddef>

#include "cudalm/cuda_check.h"
#include "cudalm/kernels/batch_decode.h"
#include "cudalm/kernels/int4_gemv_bf16.h"

namespace cudalm {
namespace {

constexpr int kTileBlock = 128;
constexpr int kTileWarps = kTileBlock / 32;  // 4

// 16B unit: 16 packed bytes = 32 INT4 (uint4) on the weight side;
// 8 __nv_bfloat16 (16B) on the x side. (Same union as the frozen TUs.)
union U32I4 {
  uint4 v;
  std::uint8_t b[16];
  __nv_bfloat16 h[8];
};

// CUDA 11.8 has no __bfloat1622float2; split by hand (same result).
__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 h) {
  return make_float2(__bfloat162float(h.x), __bfloat162float(h.y));
}

// Nibble unpack for the 16B weight fragment (same code as the frozen TUs).
__device__ __forceinline__ void int4gemv_vec_acc_unpack(const U32I4& w,
                                                        int (&q)[32]) {
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    int lo = static_cast<std::int8_t>((w.b[j] & 0x0Fu) << 4) >> 4;
    int hi = static_cast<std::int8_t>(w.b[j]) >> 4;
    q[2 * j] = lo;
    q[2 * j + 1] = hi;
  }
}

// ---------------------------------------------------------------------------
// The R-parameterized kernel. Same math/control flow as the frozen
// int4gemv_rowtile4_bf16_kernel with R rows per block instead of 4; the
// batch axis (blockIdx.y = b) mirrors the frozen batch kernel.
// ---------------------------------------------------------------------------
template <int R>
__global__ void int4gemv_rowtile_bf16_kernel(
    const uint4* __restrict__ Wp, const __half* __restrict__ scale,
    const uint4* __restrict__ x, __nv_bfloat16* __restrict__ out, int N,
    int K) {
  const int nvec = K / 32;  // uint4 / weight row (K int4 = K/32 x 16B)
  const int ngroup = K / 128;
  const int b = blockIdx.y;
  // Activation x is bf16 [B][K]: one row = K/8 uint4 = 4*nvec (NOT nvec —
  // nvec is the INT4 weight row stride; the activation row is 4x wider).
  const uint4* __restrict__ xb =
      x + static_cast<std::size_t>(b) * (4 * nvec);
  __nv_bfloat16* __restrict__ outb = out + static_cast<std::size_t>(b) * N;
  float acc[R];
#pragma unroll
  for (int i = 0; i < R; ++i) acc[i] = 0.f;

  for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
    const uint4* xv = xb + 4 * v;
    U32I4 xa;
    xa.v = xv[0];
    U32I4 xb2;
    xb2.v = xv[1];
    U32I4 xc;
    xc.v = xv[2];
    U32I4 xd;
    xd.v = xv[3];  // 4x LDG.128
    const __nv_bfloat162* ha =
        reinterpret_cast<const __nv_bfloat162*>(&xa);
    const __nv_bfloat162* hb =
        reinterpret_cast<const __nv_bfloat162*>(&xb2);
    const __nv_bfloat162* hc =
        reinterpret_cast<const __nv_bfloat162*>(&xc);
    const __nv_bfloat162* hd =
        reinterpret_cast<const __nv_bfloat162*>(&xd);
    __nv_bfloat162 xh[16];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      xh[p] = ha[p];
      xh[4 + p] = hb[p];
      xh[8 + p] = hc[p];
      xh[12 + p] = hd[p];
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = R * blockIdx.x + r;
      if (row >= N) continue;  // last-block guard
      const uint4* __restrict__ wrow = Wp + row * nvec;
      const __half* __restrict__ srow = scale + row * ngroup;
      U32I4 w;
      w.v = wrow[v];  // 1x LDG.128
      // Single-group lemma: g = v >> 2
      const float s = __half2float(srow[v >> 2]);  // 1x 2B load
      int q[32];
      int4gemv_vec_acc_unpack(w, q);  // 32 nibbles
#pragma unroll
      for (int p = 0; p < 8; ++p) {
        const float2 fa = bf162_to_float2(xh[p]);
        const float2 fb = bf162_to_float2(xh[8 + p]);
        acc[r] += (static_cast<float>(q[2 * p]) * s) * fa.x;
        acc[r] += (static_cast<float>(q[2 * p + 1]) * s) * fa.y;
        acc[r] += (static_cast<float>(q[2 * p + 16]) * s) * fb.x;
        acc[r] += (static_cast<float>(q[2 * p + 17]) * s) * fb.y;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    }
  }

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  __shared__ float warp_sums[R][kTileWarps];
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) warp_sums[r][wid] = acc[r];
  }
  __syncthreads();

  if (wid == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = R * blockIdx.x + r;
      if (row >= N) continue;
      float t = (lane < kTileWarps) ? warp_sums[r][lane] : 0.f;
#pragma unroll
      for (int off = kTileWarps / 2; off > 0; off >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, off);
      }
      if (lane == 0) outb[row] = __float2bfloat16_rn(t);
    }
  }
}

bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

// Vectorization-contract gate (identical to the frozen int4_gemv_bf16):
// true when the 16B path is legal.
bool vec_contract_ok(const std::uint8_t* weight, const __nv_bfloat16* x,
                     int K) {
  return K % 32 == 0 && aligned16(weight) && aligned16(x);
}

template <int R>
void launch_b1(const std::uint8_t* weight, const __half* scales,
               const __nv_bfloat16* x, __nv_bfloat16* y, int N, int K,
               cudaStream_t stream) {
  const int grid = (N + R - 1) / R;
  int4gemv_rowtile_bf16_kernel<R><<<grid, kTileBlock, 0, stream>>>(
      reinterpret_cast<const uint4*>(weight), scales,
      reinterpret_cast<const uint4*>(x), y, N, K);
  CUDA_CHECK_LAUNCH();
}

template <int R>
void launch_batch(const std::uint8_t* weight, const __half* scales,
                  const __nv_bfloat16* x, __nv_bfloat16* y, int N, int K,
                  int B, cudaStream_t stream) {
  dim3 grid((N + R - 1) / R, B);
  int4gemv_rowtile_bf16_kernel<R><<<grid, kTileBlock, 0, stream>>>(
      reinterpret_cast<const uint4*>(weight), scales,
      reinterpret_cast<const uint4*>(x), y, N, K);
  CUDA_CHECK_LAUNCH();
}

}  // namespace

// ---------------------------------------------------------------------------
// Explicit per-variant launchers.
// ---------------------------------------------------------------------------

void int4_gemv_bf16_rowtile1(const std::uint8_t* weight, const __half* scales,
                             const __nv_bfloat16* x, __nv_bfloat16* y, int N,
                             int K, cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "int4_gemv_bf16_rowtile1 requires N >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "int4_gemv_bf16_rowtile1 requires K % 128 == 0");
  if (vec_contract_ok(weight, x, K)) {
    launch_b1<1>(weight, scales, x, y, N, K, stream);
  } else {
    int4_gemv_bf16(weight, scales, x, y, N, K, stream);  // frozen (scalar fb)
  }
}

void int4_gemv_bf16_rowtile2(const std::uint8_t* weight, const __half* scales,
                             const __nv_bfloat16* x, __nv_bfloat16* y, int N,
                             int K, cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "int4_gemv_bf16_rowtile2 requires N >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "int4_gemv_bf16_rowtile2 requires K % 128 == 0");
  if (vec_contract_ok(weight, x, K)) {
    launch_b1<2>(weight, scales, x, y, N, K, stream);
  } else {
    int4_gemv_bf16(weight, scales, x, y, N, K, stream);  // frozen (scalar fb)
  }
}

void int4_gemv_bf16_rowtile8(const std::uint8_t* weight, const __half* scales,
                             const __nv_bfloat16* x, __nv_bfloat16* y, int N,
                             int K, cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "int4_gemv_bf16_rowtile8 requires N >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "int4_gemv_bf16_rowtile8 requires K % 128 == 0");
  if (vec_contract_ok(weight, x, K)) {
    launch_b1<8>(weight, scales, x, y, N, K, stream);
  } else {
    int4_gemv_bf16(weight, scales, x, y, N, K, stream);  // frozen (scalar fb)
  }
}

void batch_int4_gemv_bf16_rowtile1(const std::uint8_t* weight,
                                   const __half* scales,
                                   const __nv_bfloat16* x, __nv_bfloat16* y,
                                   int N, int K, int B, cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "batch_int4_gemv_bf16_rowtile1 N >= 1");
  CUDALM_PRECONDITION(B >= 1, "batch_int4_gemv_bf16_rowtile1 B >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "batch_int4_gemv_bf16_rowtile1 K % 128 == 0");
  if (vec_contract_ok(weight, x, K)) {
    launch_batch<1>(weight, scales, x, y, N, K, B, stream);
  } else {
    kernels::batch_int4_gemv_bf16(weight, scales, x, y, N, K, B, stream);
  }
}

void batch_int4_gemv_bf16_rowtile2(const std::uint8_t* weight,
                                   const __half* scales,
                                   const __nv_bfloat16* x, __nv_bfloat16* y,
                                   int N, int K, int B, cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "batch_int4_gemv_bf16_rowtile2 N >= 1");
  CUDALM_PRECONDITION(B >= 1, "batch_int4_gemv_bf16_rowtile2 B >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "batch_int4_gemv_bf16_rowtile2 K % 128 == 0");
  if (vec_contract_ok(weight, x, K)) {
    launch_batch<2>(weight, scales, x, y, N, K, B, stream);
  } else {
    kernels::batch_int4_gemv_bf16(weight, scales, x, y, N, K, B, stream);
  }
}

void batch_int4_gemv_bf16_rowtile8(const std::uint8_t* weight,
                                   const __half* scales,
                                   const __nv_bfloat16* x, __nv_bfloat16* y,
                                   int N, int K, int B, cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "batch_int4_gemv_bf16_rowtile8 N >= 1");
  CUDALM_PRECONDITION(B >= 1, "batch_int4_gemv_bf16_rowtile8 B >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "batch_int4_gemv_bf16_rowtile8 K % 128 == 0");
  if (vec_contract_ok(weight, x, K)) {
    launch_batch<8>(weight, scales, x, y, N, K, B, stream);
  } else {
    kernels::batch_int4_gemv_bf16(weight, scales, x, y, N, K, B, stream);
  }
}

// ---------------------------------------------------------------------------
// Production dispatcher.
//
// The R choice is a STATIC measured-shape table (docs/
// v07_w4a16_optimization.md, microbench + NCU evidence), keyed on
// (N, K[, B]) for the Qwen3.5-0.8B production census
// (benchmarks/profiling/v07b_w4a16_shape_census.txt). No runtime
// autotuning. Anything not in the table takes the FROZEN R=4 baseline
// path — the bit-compatible generic fallback (with its scalar fallback
// for non-16B-aligned inputs).
//
// MEASURED ADAPTIVE TABLE (candidate, measured at V07B_CANDIDATE_SHA;
// REJECTED for production — retained for reproducible candidate bench).
//
// The R choice is a STATIC measured-shape table (no runtime autotuning),
// keyed on the Qwen3.5-0.8B production census (benchmarks/profiling/
// v07b_w4a16_shape_census.txt), from the Phase-B microbenchmark
// (benchmarks/bench_w4a16_qwen35_shapes, exact-SHA runs in benchmarks/
// v07b_w4a16_microbench_run*.txt), NCU (benchmarks/profiling/
// v07b_ncu_*.txt) and the candidate batched-only nsys (benchmarks/
// profiling/v07b_candidate_batched_nsys_*). Anything not in the table
// takes the FROZEN R4 baseline — the bit-compatible generic fallback
// (with its scalar fallback for non-16B-aligned inputs).
//
//   B=1:  N=16,K=1024  -> R1   (1.38-1.43x; grid 4 -> 16 blocks,
//                                     waves 0.01 -> 0.03)
//         N=512,K=1024 -> R2   (1.43-1.47x; grid 128 -> 256, waves
//                                     0.24 -> 0.47, occ 21 -> 42%)
//         else         -> frozen R4
//   B>1:  N=16,K=1024              -> R1   (1.38-1.42x per call)
//         N=512,K=1024 && B==2     -> R2   (1.09-1.11x per call)
//         N=512,K=1024 && B==3     -> frozen R4 (R2 measured 0.94x)
//         else                     -> frozen R4
//
// FINAL STATE: the production runtime does NOT use this dispatcher (the
// candidate was REJECTED after the exact-SHA E2E acceptance check: kernel
// level -5.9% W4A16 time, E2E wall within noise — docs/
// v07_w4a16_optimization.md). The measured table is retained so the
// candidate can be re-benchmarked/reproduced (it is what
// V07B_CANDIDATE_SHA ran in the full-model candidate benchmark).
// ---------------------------------------------------------------------------

namespace {

// Measured R for a B=1 shape; 4 = frozen baseline (default/fallback).
int pick_rowtile_b1(int N, int K) {
  (void)K;
  switch (N) {
    case 16:
      return 1;  // in_proj_b/a: 1.38~1.43x (degenerate small-N, grid 4)
    case 512:
      return 2;  // k/v_proj: 1.43~1.47x (occupancy-limited, grid 128)
    default:
      return 4;  // frozen R4 baseline (generic fallback)
  }
}

// Measured R for a batch shape; 4 = frozen baseline (default/fallback).
int pick_rowtile_batch(int N, int K, int B) {
  (void)K;
  if (N == 16) {
    return 1;  // in_proj_b/a: 1.38~1.42x per call for B=2 and B=3
  }
  if (N == 512) {
    // k/v_proj: R2 wins only at B=2 (1.09~1.11x); R2 LOSES at B=3
    // (0.94x) and beyond — keep the frozen baseline there.
    return (B == 2) ? 2 : 4;
  }
  return 4;  // frozen R4 baseline (generic fallback)
}

}  // namespace

void int4_gemv_bf16_qwen35_experimental(const std::uint8_t* weight, const __half* scales,
                           const __nv_bfloat16* x, __nv_bfloat16* y, int N,
                           int K, cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "int4_gemv_bf16_qwen35_experimental requires N >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "int4_gemv_bf16_qwen35_experimental requires K % 128 == 0");
  if (!vec_contract_ok(weight, x, K)) {
    int4_gemv_bf16(weight, scales, x, y, N, K, stream);  // frozen (scalar fb)
    return;
  }
  switch (pick_rowtile_b1(N, K)) {
    case 1:
      launch_b1<1>(weight, scales, x, y, N, K, stream);
      break;
    case 2:
      launch_b1<2>(weight, scales, x, y, N, K, stream);
      break;
    case 8:
      launch_b1<8>(weight, scales, x, y, N, K, stream);
      break;
    default:
      int4_gemv_bf16(weight, scales, x, y, N, K, stream);  // frozen R4
      break;
  }
}

void batch_int4_gemv_bf16_qwen35_experimental(const std::uint8_t* weight,
                                 const __half* scales,
                                 const __nv_bfloat16* x, __nv_bfloat16* y,
                                 int N, int K, int B, cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "batch_int4_gemv_bf16_qwen35_experimental requires N >= 1");
  CUDALM_PRECONDITION(B >= 1, "batch_int4_gemv_bf16_qwen35_experimental requires B >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "batch_int4_gemv_bf16_qwen35_experimental requires K % 128 == 0");
  if (!vec_contract_ok(weight, x, K)) {
    kernels::batch_int4_gemv_bf16(weight, scales, x, y, N, K, B, stream);
    return;
  }
  switch (pick_rowtile_batch(N, K, B)) {
    case 1:
      launch_batch<1>(weight, scales, x, y, N, K, B, stream);
      break;
    case 2:
      launch_batch<2>(weight, scales, x, y, N, K, B, stream);
      break;
    case 8:
      launch_batch<8>(weight, scales, x, y, N, K, B, stream);
      break;
    default:
      kernels::batch_int4_gemv_bf16(weight, scales, x, y, N, K, B, stream);
      break;
  }
}

// ---------------------------------------------------------------------------
// Register-count queries (microbench reporting; no behavior change).
// ---------------------------------------------------------------------------

namespace {
template <int R>
int rowtile_regs() {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, int4gemv_rowtile_bf16_kernel<R>) !=
      cudaSuccess)
    return -1;
  return attr.numRegs;
}
}  // namespace

int int4_gemv_bf16_rowtile1_regs() { return rowtile_regs<1>(); }
int int4_gemv_bf16_rowtile2_regs() { return rowtile_regs<2>(); }
int int4_gemv_bf16_rowtile8_regs() { return rowtile_regs<8>(); }

}  // namespace cudalm
