// CUDALM — W4A16 GEMV (bf16 activation/output), port of CUDALab
// int4gemv_rowtile4_hx + scalar fallback.
//
// Upstream: commit cb6a6a9, kernels/int4gemv/int4gemv_rowtile4_hx.cu
// (int4gemv_rowtile4_hx_kernel + int4gemv_rowtile4_hx_fwd) and
// kernels/int4gemv/int4gemv_common.h (int4gemv_unpack_byte, U32I4,
// int4gemv_vec_acc_unpack, int4gemv_scalar_kernel, launch_int4gemv_scalar,
// int4gemv_vec_contract_ok) — the SAME source the v0.1 fp16 port in
// int4_gemv.cu was derived from.
//
// This is the bf16 specialization for Qwen3.5 (v0.2): kernel math and
// control flow are identical to the fp16 port (R=4 row tile, 128-thread
// block, strided v loop, 4× LDG.128 x fragments, single-group lemma
// g = v>>2, 32 nibble unpack + 32 (MUL+FFMA) per row, shfl 5-step +
// shared[4][4] + shfl 2-step reduction). The only changes are mechanical:
//   * __half activation/output -> __nv_bfloat16 (the fp16 SCALE is
//     unchanged: the stored fp16 scale is the contract);
//   * x fragments kept as __nv_bfloat162[16]; conversions via the local
//     bf162_to_float2 helper (CUDA 11.8 has no __bfloat1622float2);
//     store via __float2bfloat16_rn.
// See docs/provenance.md.

#include "cudalm/kernels/int4_gemv_bf16.h"

#include <cstddef>

#include "cudalm/cuda_check.h"

namespace cudalm {

namespace {

constexpr int kRowTile4BfBlock = 128;
constexpr int kRowTile4BfWarps = kRowTile4BfBlock / 32;  // 4
constexpr int kRowTile4BfRows = 4;
constexpr int kInt4BfScalarBlock = 256;
constexpr int kInt4BfScalarWarps = kInt4BfScalarBlock / 32;  // 8

// Nibble unpack (same code as the fp16 port / CUDALab
// int4gemv_unpack_byte, verbatim): low nibble = element 2b, high nibble =
// element 2b+1, 4-bit two's complement, both sign-extended.
__device__ __forceinline__ void int4gemv_unpack_byte(std::uint8_t b,
                                                     int& lo, int& hi) {
  lo = static_cast<std::int8_t>((b & 0x0Fu) << 4) >> 4;
  hi = static_cast<std::int8_t>(b) >> 4;
}

// CUDA 11.8 has no __bfloat1622float2; split by hand (same result).
__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 h) {
  return make_float2(__bfloat162float(h.x), __bfloat162float(h.y));
}

// 16B unit: 16 packed bytes = 32 INT4 (uint4) on the weight side;
// 8 __nv_bfloat16 (16B) on the x side.
union U32I4 {
  uint4 v;
  std::uint8_t b[16];
  __nv_bfloat16 h[8];
};

// Nibble unpack for the 16B weight fragment (same code as the fp16 port).
__device__ __forceinline__ void int4gemv_vec_acc_unpack(const U32I4& w,
                                                        int (&q)[32]) {
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    int lo, hi;
    int4gemv_unpack_byte(w.b[j], lo, hi);
    q[2 * j] = lo;
    q[2 * j + 1] = hi;
  }
}

// bf16 specialization of int4gemv_rowtile4_hx_kernel (math/control flow
// verbatim from the fp16 port).
__global__ void int4gemv_rowtile4_bf16_kernel(const uint4* __restrict__ Wp,
                                              const __half* __restrict__ scale,
                                              const uint4* __restrict__ x,
                                              __nv_bfloat16* __restrict__ out,
                                              int N, int K) {
  const int nvec = K / 32;  // uint4 / row
  const int ngroup = K / 128;
  float acc[kRowTile4BfRows];
#pragma unroll
  for (int i = 0; i < kRowTile4BfRows; ++i) acc[i] = 0.f;

  for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
    // Load the x fragment once and keep the raw __nv_bfloat162
    // representation (16 registers), reused across the R=4 rows.
    // xh[j] = x[2j..2j+1]: xa/xb -> j=0..7, xc/xd -> j=8..15
    const uint4* xv = x + 4 * v;
    U32I4 xa;
    xa.v = xv[0];
    U32I4 xb;
    xb.v = xv[1];
    U32I4 xc;
    xc.v = xv[2];
    U32I4 xd;
    xd.v = xv[3];  // 4× LDG.128
    const __nv_bfloat162* ha =
        reinterpret_cast<const __nv_bfloat162*>(&xa);
    const __nv_bfloat162* hb =
        reinterpret_cast<const __nv_bfloat162*>(&xb);
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
    for (int r = 0; r < kRowTile4BfRows; ++r) {
      const int row = kRowTile4BfRows * blockIdx.x + r;
      if (row >= N) continue;  // last-block guard
      const uint4* __restrict__ wrow = Wp + row * nvec;
      const __half* __restrict__ srow = scale + row * ngroup;
      U32I4 w;
      w.v = wrow[v];  // 1× LDG.128
      // Single-group lemma: g = v >> 2
      const float s = __half2float(srow[v >> 2]);  // 1× 2B load
      int q[32];
      int4gemv_vec_acc_unpack(w, q);  // 32 nibbles
      // Per-row conversion: xh[p] -> k = 2p, 2p+1; xh[8+p] -> k = 16+2p, ..+1
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
  for (int r = 0; r < kRowTile4BfRows; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    }
  }

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  __shared__ float warp_sums[kRowTile4BfRows][kRowTile4BfWarps];
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < kRowTile4BfRows; ++r) warp_sums[r][wid] = acc[r];
  }
  __syncthreads();

  if (wid == 0) {
#pragma unroll
    for (int r = 0; r < kRowTile4BfRows; ++r) {
      const int row = kRowTile4BfRows * blockIdx.x + r;
      if (row >= N) continue;
      float t = (lane < kRowTile4BfWarps) ? warp_sums[r][lane] : 0.f;
#pragma unroll
      for (int off = kRowTile4BfWarps / 2; off > 0; off >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, off);
      }
      if (lane == 0) out[row] = __float2bfloat16_rn(t);
    }
  }
}

// bf16 specialization of int4gemv_scalar_kernel (same code as the fp16
// port): one block per output row, 256 threads, strided packed-byte loads.
__global__ void int4gemv_scalar_bf16_kernel(const std::uint8_t* __restrict__ Wp,
                                            const __half* __restrict__ scale,
                                            const __nv_bfloat16* __restrict__ x,
                                            __nv_bfloat16* __restrict__ out,
                                            int N, int K) {
  const int row = blockIdx.x;
  const std::uint8_t* __restrict__ wrow = Wp + row * (K / 2);
  const __half* __restrict__ srow = scale + row * (K / 128);
  float acc = 0.f;
  for (int b = threadIdx.x; b < K / 2; b += blockDim.x) {
    const std::uint8_t byte = wrow[b];
    // 1 byte = 2 consecutive k, always the same group: g = (2b)/128 = b/64
    const float s = __half2float(srow[b >> 6]);
    int lo, hi;
    int4gemv_unpack_byte(byte, lo, hi);
    // Per-element dequant: q→fp32, ×scale(fp16→fp32), ×bf16 activation
    acc += (static_cast<float>(lo) * s) * __bfloat162float(x[2 * b]);
    acc += (static_cast<float>(hi) * s) * __bfloat162float(x[2 * b + 1]);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  __shared__ float warp_sums[kInt4BfScalarWarps];
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();

  if (wid == 0) {
    acc = (lane < kInt4BfScalarWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int off = kInt4BfScalarWarps / 2; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) out[row] = __float2bfloat16_rn(acc);
  }
}

bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

}  // namespace

void int4_gemv_bf16(const std::uint8_t* weight, const __half* scales,
                    const __nv_bfloat16* x, __nv_bfloat16* y, int N, int K,
                    cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "int4_gemv_bf16 requires N >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "int4_gemv_bf16 requires K % 128 == 0");

  // Same vectorization contract as the fp16 path: W base 16B ∧ x base 16B
  // ∧ K%32==0 (implied by K%128==0); scale/out are per-group/per-row
  // accesses — no alignment requirement.
  if (K % 32 == 0 && aligned16(weight) && aligned16(x)) {
    const int grid = (N + kRowTile4BfRows - 1) / kRowTile4BfRows;
    int4gemv_rowtile4_bf16_kernel<<<grid, kRowTile4BfBlock, 0, stream>>>(
        reinterpret_cast<const uint4*>(weight), scales,
        reinterpret_cast<const uint4*>(x), y, N, K);
    CUDA_CHECK_LAUNCH();
  } else {
    // Scalar fallback: legal inputs must never be rejected.
    int4gemv_scalar_bf16_kernel<<<N, kInt4BfScalarBlock, 0, stream>>>(
        weight, scales, x, y, N, K);
    CUDA_CHECK_LAUNCH();
  }
}

}  // namespace cudalm
