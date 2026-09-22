// CUDALM — W4A16 GEMV, port of CUDALab int4gemv_rowtile4_hx + scalar
// fallback (int4gemv_baseline).
//
// Upstream: commit cb6a6a9, kernels/int4gemv/int4gemv_rowtile4_hx.cu
// (int4gemv_rowtile4_hx_kernel + int4gemv_rowtile4_hx_fwd) and
// kernels/int4gemv/int4gemv_common.h (int4gemv_unpack_byte, U32I4,
// int4gemv_vec_acc_unpack, int4gemv_scalar_kernel,
// launch_int4gemv_scalar, int4gemv_vec_contract_ok).
//
// Kernel math and control flow are preserved 1:1 (R=4 row tile, 128-thread
// block, strided v loop, 4× LDG.128 x fragments kept as __half2[16],
// single-group lemma g = v>>2, 32 nibble unpack + 32 (MUL+FFMA) per row,
// shfl 5-step + shared[4][4] + shfl 2-step reduction). Deviations are
// mechanical:
//   * PyTorch host layer replaced by raw pointers + cudaStream_t +
//     CUDALM_PRECONDITION + CUDA_CHECK_LAUNCH;
//   * int64_t dimensions replaced by int (v0.1 sizes; loop bounds only,
//     no math effect);
//   * int4gemv_vec_contract_ok inlined into the host entry.
// See docs/provenance.md.

#include "cudalm/kernels/int4_gemv.h"

#include <cstddef>

#include "cudalm/cuda_check.h"

namespace cudalm {

namespace {

constexpr int kRowTile4HxBlock = 128;
constexpr int kRowTile4HxWarps = kRowTile4HxBlock / 32;  // 4
constexpr int kRowTile4HxRows = 4;
constexpr int kInt4ScalarBlock = 256;
constexpr int kInt4ScalarWarps = kInt4ScalarBlock / 32;  // 8

// Port of int4gemv_unpack_byte (verbatim): low nibble = element 2b, high
// nibble = element 2b+1, 4-bit two's complement, both sign-extended.
//   lo: (b & 0xF) << 4 places the 4-bit value in bits 4-7 (bit 7 = lo sign);
//       the int8_t cast makes 8..15 negative, the arithmetic >>4 sign-extends.
//   hi: the byte's bit 7 IS the hi nibble's sign bit; the int8_t cast
//       sign-extends from bit 7, the arithmetic >>4 extracts hi.
__device__ __forceinline__ void int4gemv_unpack_byte(std::uint8_t b,
                                                     int& lo, int& hi) {
  lo = static_cast<std::int8_t>((b & 0x0Fu) << 4) >> 4;
  hi = static_cast<std::int8_t>(b) >> 4;
}

// 16B unit: 16 packed bytes = 32 INT4 (uint4) on the weight side;
// 8 __half (16B) on the x side.
union U32I4 {
  uint4 v;
  std::uint8_t b[16];
  __half h[8];
};

// Port of int4gemv_vec_acc_unpack (verbatim).
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

// Port of int4gemv_rowtile4_hx_kernel (math/control flow verbatim).
__global__ void int4gemv_rowtile4_hx_kernel(const uint4* __restrict__ Wp,
                                            const __half* __restrict__ scale,
                                            const uint4* __restrict__ x,
                                            __half* __restrict__ out, int N,
                                            int K) {
  const int nvec = K / 32;  // uint4 / row
  const int ngroup = K / 128;
  float acc[kRowTile4HxRows];
#pragma unroll
  for (int i = 0; i < kRowTile4HxRows; ++i) acc[i] = 0.f;

  for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
    // Load the x fragment once and keep the raw __half2 representation
    // (16 registers), reused across the R=4 rows.
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
    const __half2* ha = reinterpret_cast<const __half2*>(&xa);
    const __half2* hb = reinterpret_cast<const __half2*>(&xb);
    const __half2* hc = reinterpret_cast<const __half2*>(&xc);
    const __half2* hd = reinterpret_cast<const __half2*>(&xd);
    __half2 xh[16];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      xh[p] = ha[p];
      xh[4 + p] = hb[p];
      xh[8 + p] = hc[p];
      xh[12 + p] = hd[p];
    }
#pragma unroll
    for (int r = 0; r < kRowTile4HxRows; ++r) {
      const int row = kRowTile4HxRows * blockIdx.x + r;
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
        const float2 fa = __half22float2(xh[p]);
        const float2 fb = __half22float2(xh[8 + p]);
        acc[r] += (static_cast<float>(q[2 * p]) * s) * fa.x;
        acc[r] += (static_cast<float>(q[2 * p + 1]) * s) * fa.y;
        acc[r] += (static_cast<float>(q[2 * p + 16]) * s) * fb.x;
        acc[r] += (static_cast<float>(q[2 * p + 17]) * s) * fb.y;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < kRowTile4HxRows; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    }
  }

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  __shared__ float warp_sums[kRowTile4HxRows][kRowTile4HxWarps];
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < kRowTile4HxRows; ++r) warp_sums[r][wid] = acc[r];
  }
  __syncthreads();

  if (wid == 0) {
#pragma unroll
    for (int r = 0; r < kRowTile4HxRows; ++r) {
      const int row = kRowTile4HxRows * blockIdx.x + r;
      if (row >= N) continue;
      float t = (lane < kRowTile4HxWarps) ? warp_sums[r][lane] : 0.f;
#pragma unroll
      for (int off = kRowTile4HxWarps / 2; off > 0; off >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, off);
      }
      if (lane == 0) out[row] = __float2half_rn(t);
    }
  }
}

// Port of int4gemv_scalar_kernel (INT4GEMV-0000 compute core, verbatim):
// one block per output row, 256 threads, strided packed-byte loads.
__global__ void int4gemv_scalar_kernel(const std::uint8_t* __restrict__ Wp,
                                       const __half* __restrict__ scale,
                                       const __half* __restrict__ x,
                                       __half* __restrict__ out, int N,
                                       int K) {
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
    // Per-element dequant (verbatim): q→fp32, ×scale, ×FP16 activation
    acc += (static_cast<float>(lo) * s) * __half2float(x[2 * b]);
    acc += (static_cast<float>(hi) * s) * __half2float(x[2 * b + 1]);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  __shared__ float warp_sums[kInt4ScalarWarps];
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();

  if (wid == 0) {
    acc = (lane < kInt4ScalarWarps) ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int off = kInt4ScalarWarps / 2; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) out[row] = __float2half_rn(acc);
  }
}

bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

}  // namespace

void int4_gemv(const std::uint8_t* weight, const __half* scales,
               const __half* x, __half* y, int N, int K,
               cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "int4_gemv requires N >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "int4_gemv requires K % 128 == 0");

  // Upstream int4gemv_vec_contract_ok: W base 16B ∧ x base 16B ∧ K%32==0
  // (the last is implied by K%128==0 and kept as self-documentation).
  // scale/out are per-group 2B accesses — no alignment requirement.
  if (K % 32 == 0 && aligned16(weight) && aligned16(x)) {
    const int grid = (N + kRowTile4HxRows - 1) / kRowTile4HxRows;
    int4gemv_rowtile4_hx_kernel<<<grid, kRowTile4HxBlock, 0, stream>>>(
        reinterpret_cast<const uint4*>(weight), scales,
        reinterpret_cast<const uint4*>(x), y, N, K);
    CUDA_CHECK_LAUNCH();
  } else {
    // Scalar fallback: legal inputs must never be rejected.
    int4gemv_scalar_kernel<<<N, kInt4ScalarBlock, 0, stream>>>(weight, scales,
                                                               x, y, N, K);
    CUDA_CHECK_LAUNCH();
  }
}

}  // namespace cudalm
