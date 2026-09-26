// CUDALM — v0.6 Phase B: TRUE BATCHED DECODE kernel implementations.
//
// Every kernel below MIRRORS one frozen single-row kernel from this repo
// (the v0.2/v0.5 family) — same math, same accumulation order, same bf16
// rounding boundaries — extended with a batch axis. The row-parity contract
// (batch row b == frozen single call on row b, BIT-IDENTICAL) is pinned by
// tests/cuda/test_qwen35_batch_kernels.cpp. Frozen sources mirrored:
//
//   batch_int4_gemv_bf16      <- int4gemv_rowtile4_bf16_kernel
//                                (src/kernels/int4_gemv_bf16.cu)
//   batch_bf16_gemv           <- bf16_gemv_vec4_row_kernel
//                                (src/kernels/bf16_gemv.cu)
//   batch_partial_rope_bf16   <- qwen35_partial_rope_kernel
//                                (src/kernels/qwen35_kernels.cu)
//   batch_deltanet_conv_*     <- deltanet_conv_kernel
//                                (src/kernels/qwen35_deltanet_kernels.cu)
//   batch_deltanet_gbeta_*    <- deltanet_gbeta_kernel (same file)
//   batch_deltanet_delta_*    <- deltanet_delta_kernel (same file)
//   batch_paged_kv_write_*    <- qwen35_paged_kv_write_kernel
//                                (src/kernels/paged_kv.cu)
//   batch_paged_attention_*   <- qwen35_paged_attention_{scores,softmax,
//                                pv}_kernel (same file)
//
// Upstream check: the frozen CUDALab cb6a6a9 tree has NO batched kernels
// (single-token only), so this file extends the frozen single-kernel math
// with a batch dimension — no new math. Provenance: docs/provenance.md
// (v0.6 Phase B).

#include "cudalm/kernels/batch_decode.h"

#include <cfloat>
#include <cmath>

#include "cudalm/cuda_check.h"

namespace cudalm {
namespace kernels {

namespace {

// 16B unit: 16 packed bytes = 32 INT4 (uint4) on the weight side;
// 8 __nv_bfloat16 (16B) on the x side. (Same union as the frozen int4
// bf16 GEMV TU.)
union U32I4 {
  uint4 v;
  std::uint8_t b[16];
  __nv_bfloat16 h[8];
};

union U16 {
  uint4 v;
  __nv_bfloat16 h[8];
};

// CUDA 11.8 has no __bfloat1622float2; split by hand (same result).
__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 h) {
  return make_float2(__bfloat162float(h.x), __bfloat162float(h.y));
}

// Nibble unpack for the 16B weight fragment (same code as the frozen
// int4gemv_vec_acc_unpack).
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
// Embedding gather (plain bf16 copies — bit-exact round trip).
// ---------------------------------------------------------------------------
__global__ void batch_embed_gather_kernel(const __nv_bfloat16* __restrict__ W,
                                          const int* __restrict__ token_ids,
                                          __nv_bfloat16* __restrict__ out,
                                          int H) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= H) return;
  const int tok = token_ids[blockIdx.y];
  out[static_cast<std::size_t>(blockIdx.y) * H + idx] =
      W[static_cast<std::size_t>(tok) * H + idx];
}

// ---------------------------------------------------------------------------
// W4A16 GEMV, B rows (mirror of int4gemv_rowtile4_bf16_kernel; the batch
// axis is blockIdx.y: x fragment base + b*K, output row base + b*N).
// ---------------------------------------------------------------------------
constexpr int kBatchInt4Block = 128;
constexpr int kBatchInt4Warps = kBatchInt4Block / 32;  // 4
constexpr int kBatchInt4Rows = 4;

__global__ void batch_int4gemv_rowtile4_bf16_kernel(
    const uint4* __restrict__ Wp, const __half* __restrict__ scale,
    const uint4* __restrict__ x, __nv_bfloat16* __restrict__ out, int N,
    int K) {
  const int nvec = K / 32;  // uint4 / weight row (K int4 = K/32 x 16B)
  const int ngroup = K / 128;
  const int b = blockIdx.y;
  // Activation x is bf16 [B][K]: one row = K/8 uint4 = 4*nvec (NOT nvec —
  // nvec is the INT4 weight row stride; the activation row is 4x wider).
  const uint4* __restrict__ xb = x + static_cast<std::size_t>(b) * (4 * nvec);
  __nv_bfloat16* __restrict__ outb = out + static_cast<std::size_t>(b) * N;
  float acc[kBatchInt4Rows];
#pragma unroll
  for (int i = 0; i < kBatchInt4Rows; ++i) acc[i] = 0.f;

  for (int v = threadIdx.x; v < nvec; v += blockDim.x) {
    const uint4* xv = xb + 4 * v;
    U32I4 xa;
    xa.v = xv[0];
    U32I4 xbf;
    xbf.v = xv[1];
    U32I4 xc;
    xc.v = xv[2];
    U32I4 xd;
    xd.v = xv[3];  // 4x LDG.128
    const __nv_bfloat162* ha =
        reinterpret_cast<const __nv_bfloat162*>(&xa);
    const __nv_bfloat162* hb =
        reinterpret_cast<const __nv_bfloat162*>(&xbf);
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
    for (int r = 0; r < kBatchInt4Rows; ++r) {
      const int row = kBatchInt4Rows * blockIdx.x + r;
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
  for (int r = 0; r < kBatchInt4Rows; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    }
  }

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  __shared__ float warp_sums[kBatchInt4Rows][kBatchInt4Warps];
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < kBatchInt4Rows; ++r) warp_sums[r][wid] = acc[r];
  }
  __syncthreads();

  if (wid == 0) {
#pragma unroll
    for (int r = 0; r < kBatchInt4Rows; ++r) {
      const int row = kBatchInt4Rows * blockIdx.x + r;
      if (row >= N) continue;
      float t = (lane < kBatchInt4Warps) ? warp_sums[r][lane] : 0.f;
#pragma unroll
      for (int off = kBatchInt4Warps / 2; off > 0; off >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, off);
      }
      if (lane == 0) outb[row] = __float2bfloat16_rn(t);
    }
  }
}

// ---------------------------------------------------------------------------
// BF16 GEMV, B rows (mirror of bf16_gemv_vec4_row_kernel; grid (N, B)).
// ---------------------------------------------------------------------------
__global__ void batch_bf16_gemv_vec4_row_kernel(const U16* __restrict__ W,
                                                const U16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ out,
                                                int N, int K) {
  const int row = blockIdx.x;
  const int b = blockIdx.y;
  const U16* __restrict__ wrow = W + static_cast<std::size_t>(row) * (K / 8);
  const U16* __restrict__ xb = x + static_cast<std::size_t>(b) * (K / 8);
  const int nvec = K / 8;

  float acc = 0.f;
#pragma unroll 4
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const U16 w = wrow[i];
    const U16 xv = xb[i];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      acc += __bfloat162float(w.h[j]) * __bfloat162float(xv.h[j]);
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  __shared__ float warp_sums[8];
  if (lane == 0) warp_sums[wid] = acc;
  __syncthreads();
  if (wid == 0) {
    acc = (lane < 8) ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int off = 4; off > 0; off >>= 1)
      acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0)
      out[static_cast<std::size_t>(b) * N + row] = __float2bfloat16_rn(acc);
  }
}

// ---------------------------------------------------------------------------
// Partial RoPE, B x unit_rows rows, per-row fp32 cos/sin tables (mirror of
// qwen35_partial_rope_kernel; the table index gains the b = m/unit_rows
// prefix — everything else verbatim).
// ---------------------------------------------------------------------------
__global__ void batch_partial_rope_kernel(const __nv_bfloat16* __restrict__ x,
                                          __nv_bfloat16* __restrict__ y,
                                          int nrows, int unit_rows,
                                          int head_dim, int rotary_dim,
                                          const float* __restrict__ cos_t,
                                          const float* __restrict__ sin_t) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = idx / head_dim;
  const int d = idx - m * head_dim;
  if (m >= nrows) return;  // ceiling-rounded grid tail
  const int b = m / unit_rows;
  const __nv_bfloat16* __restrict__ xrow =
      x + static_cast<std::size_t>(m) * head_dim;
  __nv_bfloat16* __restrict__ yrow = y + static_cast<std::size_t>(m) * head_dim;

  if (d >= rotary_dim) {
    yrow[d] = xrow[d];  // pass-through (no rounding: it is a copy)
    return;
  }
  const int j = d % (rotary_dim / 2);
  const float* cb_t = cos_t + static_cast<std::size_t>(b) * (rotary_dim / 2);
  const float* sb_t = sin_t + static_cast<std::size_t>(b) * (rotary_dim / 2);
  const __nv_bfloat16 cb = __float2bfloat16_rn(cb_t[j]);
  const __nv_bfloat16 sb = __float2bfloat16_rn(sb_t[j]);
  const bool first_half = (d < rotary_dim / 2);
  const int pd = first_half ? (d + rotary_dim / 2) : (d - rotary_dim / 2);
  const float pv = __bfloat162float(xrow[pd]);
  // rotate_half(z) = [-z[rd/2:], z[:rd/2]]
  const float r2f = (first_half ? -pv : pv) * __bfloat162float(sb);
  // Official bf16 chain: bf16(x*cos_bf16) + bf16(rot*sin_bf16), each of
  // the three ops rounded to bf16.
  const __nv_bfloat16 r1 =
      __float2bfloat16_rn(__bfloat162float(xrow[d]) * __bfloat162float(cb));
  const __nv_bfloat16 r2 = __float2bfloat16_rn(r2f);
  yrow[d] = __float2bfloat16_rn(__bfloat162float(r1) + __bfloat162float(r2));
}

// ---------------------------------------------------------------------------
// DeltaNet conv decode, B rows (mirror of deltanet_conv_kernel; one thread
// per (channel, row); the conv state base gains the per-row slot prefix).
// ---------------------------------------------------------------------------
__global__ void batch_deltanet_conv_kernel(
    __nv_bfloat16* __restrict__ conv_base, const int* __restrict__ d_slots,
    const __nv_bfloat16* __restrict__ new_mixed,
    const __nv_bfloat16* __restrict__ conv_w,
    __nv_bfloat16* __restrict__ conv_out, __nv_bfloat16* __restrict__ conv_silu,
    int conv_dim) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= conv_dim) return;
  const int b = blockIdx.y;
  __nv_bfloat16* __restrict__ conv_state =
      conv_base +
      static_cast<std::size_t>(d_slots[b]) * static_cast<std::size_t>(conv_dim) *
          3;
  const __nv_bfloat16* __restrict__ nm = new_mixed + b * conv_dim;
  // buf = [cs0, cs1, cs2, new]; read the OLD state first (the conv consumes
  // the OLD state values; the state is then shifted to [cs1, cs2, new]).
  const __nv_bfloat16 cs0 = conv_state[ch * 3 + 0];
  const __nv_bfloat16 cs1 = conv_state[ch * 3 + 1];
  const __nv_bfloat16 cs2 = conv_state[ch * 3 + 2];
  const float buf[4] = {__bfloat162float(cs0), __bfloat162float(cs1),
                        __bfloat162float(cs2), __bfloat162float(nm[ch])};
  // In-place state update (bf16 shift + insert of the new token — bit-exact).
  conv_state[ch * 3 + 0] = cs1;
  conv_state[ch * 3 + 1] = cs2;
  conv_state[ch * 3 + 2] = nm[ch];
  // Depthwise conv: c = sum_j W[ch, j] * buf[j]; fp32 accumulation, one RNE.
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    acc += __bfloat162float(conv_w[ch * 4 + j]) * buf[j];
  const __nv_bfloat16 c = __float2bfloat16_rn(acc);
  conv_out[b * conv_dim + ch] = c;
  // SiLU on the bf16 c: fp32 silu, one bf16 RNE.
  const float cf = __bfloat162float(c);
  conv_silu[b * conv_dim + ch] = __float2bfloat16_rn(cf / (1.0f + expf(-cf)));
}

// conv_silu [B][conv_dim] -> q/k/v (plain copies — the frozen single path's
// three D2D splits in one kernel).
__global__ void batch_deltanet_conv_split_kernel(
    const __nv_bfloat16* __restrict__ conv_silu,
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v, int conv_dim, int key_dim, int value_dim,
    int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;  // ceiling-rounded grid tail
  const int b = idx / conv_dim;
  const int ch = idx - b * conv_dim;
  const __nv_bfloat16 s = conv_silu[idx];
  if (ch < key_dim) {
    q[b * key_dim + ch] = s;
  } else if (ch < 2 * key_dim) {
    k[b * key_dim + (ch - key_dim)] = s;
  } else {
    v[b * value_dim + (ch - 2 * key_dim)] = s;
  }
}

// g/beta, B x n_heads (mirror of deltanet_gbeta_kernel; per (b,h) math
// verbatim — A_log/dt_bias indexed by h, rows indexed by (b,h)).
__global__ void batch_deltanet_gbeta_kernel(
    const __nv_bfloat16* __restrict__ b, const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ beta, float* __restrict__ g, int total,
    int n_heads) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int h = idx % n_heads;
  // beta = sigmoid(b) — fp32 opmath, one bf16 RNE (official b.sigmoid()).
  const float bf = __bfloat162float(b[idx]);
  beta[idx] = __float2bfloat16_rn(1.0f / (1.0f + expf(-bf)));
  // g = -exp(A_log) * softplus(a_f32 + dt_bias_f32) — all fp32 (frozen
  // softplus_f32: log1p(exp(x)) for x <= 20, else x).
  const float af = __bfloat162float(a[idx]);
  const float dtf = __bfloat162float(dt_bias[h]);
  const float x = af + dtf;
  const float sp = (x > 20.0f) ? x : log1pf(expf(x));
  g[idx] = -expf(A_log[h]) * sp;
}

// Delta rule, B x n_heads heads (mirror of deltanet_delta_kernel; one block
// per (b,h) — 128 threads = value index t; the rec state base gains the
// per-row slot prefix; q/k/v/g/beta/core rows indexed by (b,h)).
__global__ void batch_deltanet_delta_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, float* __restrict__ rec_base,
    const int* __restrict__ d_slots, __nv_bfloat16* __restrict__ core_out,
    int n_heads, int hd, float eps) {
  const int b = blockIdx.x / n_heads;
  const int h = blockIdx.x - b * n_heads;
  const int t = threadIdx.x;  // value index
  const __nv_bfloat16* qh =
      q + (static_cast<std::size_t>(b) * n_heads + h) * hd;
  const __nv_bfloat16* kh =
      k + (static_cast<std::size_t>(b) * n_heads + h) * hd;
  const __nv_bfloat16* vh =
      v + (static_cast<std::size_t>(b) * n_heads + h) * hd;

  __shared__ float q_s[128];
  __shared__ float k_s[128];
  __shared__ float smem_red[4];
  q_s[t] = __bfloat162float(qh[t]);
  k_s[t] = __bfloat162float(kh[t]);
  const float v_reg = __bfloat162float(vh[t]);
  __syncthreads();

  // block sum over exactly 128 threads (frozen block_sum_128 replicated;
  // the leading __syncthreads keeps consecutive calls race-free).
  const auto block_sum_128 = [&](float val) -> float {
    __syncthreads();
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      val += __shfl_down_sync(0xffffffffu, val, o);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) smem_red[warp] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
      smem_red[0] = smem_red[0] + smem_red[1] + smem_red[2] + smem_red[3];
    }
    __syncthreads();
    return smem_red[0];
  };

  // l2norm (frozen chain: bf16 square products, fp32 sum, bf16 t,
  // bf16 sqrt, bf16 reciprocal, bf16 scale; q gets the 1/sqrt(hd) fp32
  // scale AFTER the bf16 round).
  const float qprod =
      __bfloat162float(__float2bfloat16_rn(q_s[t] * q_s[t]));
  const float kprod =
      __bfloat162float(__float2bfloat16_rn(k_s[t] * k_s[t]));
  const float qn = block_sum_128(qprod);
  const float kn = block_sum_128(kprod);
  __syncthreads();
  const float q_t = __bfloat162float(__float2bfloat16_rn(qn + eps));
  const float k_t = __bfloat162float(__float2bfloat16_rn(kn + eps));
  const float q_sqrt =
      __bfloat162float(__float2bfloat16_rn(sqrtf(q_t)));
  const float k_sqrt =
      __bfloat162float(__float2bfloat16_rn(sqrtf(k_t)));
  const float q_inv =
      __bfloat162float(__float2bfloat16_rn(1.0f / q_sqrt));
  const float k_inv =
      __bfloat162float(__float2bfloat16_rn(1.0f / k_sqrt));
  q_s[t] = __bfloat162float(__float2bfloat16_rn(q_s[t] * q_inv));
  k_s[t] = __bfloat162float(__float2bfloat16_rn(k_s[t] * k_inv));
  q_s[t] = q_s[t] * (1.0f / sqrtf(static_cast<float>(hd)));
  __syncthreads();

  const int bh = static_cast<std::size_t>(b) * n_heads + h;
  const float exp_g = expf(g[bh]);
  const float beta_h = __bfloat162float(beta[bh]);
  const float qk = block_sum_128(k_s[t] * q_s[t]);
  __syncthreads();

  // Per value index t (frozen math verbatim):
  //   A = sum_k S[k,t]*exp_g*k[k]; B = sum_k S[k,t]*exp_g*q[k]
  float A = 0.0f, B = 0.0f;
  float* S_col =
      rec_base +
      static_cast<std::size_t>(d_slots[b]) *
          static_cast<std::size_t>(n_heads) * hd * hd +
      static_cast<std::size_t>(h) * hd * hd + t;
  for (int kk = 0; kk < hd; ++kk) {
    const float s = S_col[static_cast<std::size_t>(kk) * hd];
    A += s * exp_g * k_s[kk];
    B += s * exp_g * q_s[kk];
  }
  const float d = (v_reg - A) * beta_h;
  const float o = B + d * qk;  // output from the UPDATED state
  core_out[bh * hd + t] = __float2bfloat16_rn(o);

  // Update the column: S[k,t] = S[k,t]*exp_g + k[k]*d.
  for (int kk = 0; kk < hd; ++kk) {
    float* sp = S_col + static_cast<std::size_t>(kk) * hd;
    *sp = (*sp) * exp_g + k_s[kk] * d;
  }
}

// ---------------------------------------------------------------------------
// Paged KV, B rows (heterogeneous positions, per-row block tables).
// ---------------------------------------------------------------------------
__device__ inline void paged_row_b(int t, int page_tokens,
                                   const int* __restrict__ block_table,
                                   int* page, int* off) {
  *page = block_table[t / page_tokens];
  *off = t - (t / page_tokens) * page_tokens;
}

__device__ inline std::size_t paged_row_offset_b(int page, int off, int n,
                                                 int page_tokens, int head_dim,
                                                 std::size_t page_stride) {
  return static_cast<std::size_t>(page) * page_stride +
         (static_cast<std::size_t>(n) * page_tokens + off) * head_dim;
}

// Write (mirror of qwen35_paged_kv_write_kernel; idx over B*total).
__global__ void batch_paged_kv_write_kernel(
    const __nv_bfloat16* __restrict__ k_src,
    const __nv_bfloat16* __restrict__ v_src,
    __nv_bfloat16* __restrict__ k_pages, __nv_bfloat16* __restrict__ v_pages,
    const int* __restrict__ block_table, const int* __restrict__ d_positions,
    int row_stride, int page_tokens, int n_kv_heads, int head_dim,
    std::size_t page_stride, int B) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_kv_heads * head_dim;
  if (idx >= B * total) return;
  const int b = idx / total;
  const int r = idx - b * total;
  const int n = r / head_dim;
  const int d = r - n * head_dim;
  const int position = d_positions[b];
  const int* __restrict__ bt =
      block_table + static_cast<std::size_t>(b) * row_stride;
  int page, off;
  paged_row_b(position, page_tokens, bt, &page, &off);
  const std::size_t dst =
      paged_row_offset_b(page, off, n, page_tokens, head_dim, page_stride) + d;
  k_pages[dst] = k_src[idx];
  v_pages[dst] = v_src[idx];
}

// 1) scores (mirror of qwen35_paged_attention_scores_kernel; idx over
//    B*n_heads*T_max; row b touches only its [0..T_b) prefix).
__global__ void batch_paged_scores_kernel(
    const __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_pages, const int* __restrict__ block_table,
    const int* __restrict__ d_positions, int row_stride,
    __nv_bfloat16* __restrict__ scores2, int T_max, int n_heads,
    int n_kv_heads, int page_tokens, int head_dim, std::size_t page_stride,
    float scale, int B) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int per_b = n_heads * T_max;
  if (idx >= B * per_b) return;
  const int b = idx / per_b;
  const int rem = idx - b * per_b;
  const int h = rem / T_max;
  const int t = rem - h * T_max;
  const int T_b = d_positions[b] + 1;
  if (t >= T_b) return;  // padded tail of row b's T_max row: never touched
  const int kh = h * n_kv_heads / n_heads;

  const __nv_bfloat16* qh =
      q + (static_cast<std::size_t>(b) * n_heads + h) * head_dim;
  const int* __restrict__ bt =
      block_table + static_cast<std::size_t>(b) * row_stride;
  int page, off;
  paged_row_b(t, page_tokens, bt, &page, &off);
  const __nv_bfloat16* kr = k_pages +
      paged_row_offset_b(page, off, kh, page_tokens, head_dim, page_stride);

  float acc = 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    acc += __bfloat162float(qh[d]) * __bfloat162float(kr[d]);
  }
  const __nv_bfloat16 s1 = __float2bfloat16_rn(acc);
  scores2[(static_cast<std::size_t>(b) * n_heads + h) * T_max + t] =
      __float2bfloat16_rn(__bfloat162float(s1) * scale);
}

// 2) softmax over row [0..T_b) (mirror of the frozen paged softmax kernel —
//    same strided three-pass structure; grid (n_heads, B)).
__global__ void batch_paged_softmax_kernel(
    const __nv_bfloat16* __restrict__ scores2,
    __nv_bfloat16* __restrict__ probs, const int* __restrict__ d_positions,
    int T_max, int n_heads) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int T = d_positions[b] + 1;
  const __nv_bfloat16* srow =
      scores2 + (static_cast<std::size_t>(b) * n_heads + h) * T_max;
  __nv_bfloat16* prow =
      probs + (static_cast<std::size_t>(b) * n_heads + h) * T_max;
  __shared__ float sh_red[32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;

  float m = -FLT_MAX;
  for (int t = tid; t < T; t += blockDim.x)
    m = fmaxf(m, __bfloat162float(srow[t]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
  if (lane == 0) sh_red[warp] = m;
  __syncthreads();
  if (warp == 0) {
    m = (lane < nwarps) ? sh_red[lane] : -FLT_MAX;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (lane == 0) sh_red[0] = m;
  }
  __syncthreads();
  m = sh_red[0];

  float acc = 0.0f;
  for (int t = tid; t < T; t += blockDim.x)
    acc += expf(__bfloat162float(srow[t]) - m);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) sh_red[warp] = acc;
  __syncthreads();
  if (warp == 0) {
    acc = (lane < nwarps) ? sh_red[lane] : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) sh_red[0] = acc;
  }
  __syncthreads();
  const float sum = sh_red[0];

  for (int t = tid; t < T; t += blockDim.x)
    prow[t] = __float2bfloat16_rn(expf(__bfloat162float(srow[t]) - m) / sum);
}

// 3) PV (mirror of qwen35_paged_attention_pv_kernel; idx over
//    B*n_heads*head_dim; per (b,h,d) the t loop runs [0..T_b)).
__global__ void batch_paged_pv_kernel(
    const __nv_bfloat16* __restrict__ probs,
    __nv_bfloat16* __restrict__ v_pages, const int* __restrict__ block_table,
    const int* __restrict__ d_positions, int row_stride,
    __nv_bfloat16* __restrict__ out, int T_max, int n_heads, int n_kv_heads,
    int page_tokens, int head_dim, std::size_t page_stride, int B) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int per_b = n_heads * head_dim;
  if (idx >= B * per_b) return;
  const int b = idx / per_b;
  const int rem = idx - b * per_b;
  const int h = rem / head_dim;
  const int d = rem - h * head_dim;
  const int kh = h * n_kv_heads / n_heads;
  const int T = d_positions[b] + 1;
  const int* __restrict__ bt =
      block_table + static_cast<std::size_t>(b) * row_stride;

  const __nv_bfloat16* ph =
      probs + (static_cast<std::size_t>(b) * n_heads + h) * T_max;

  float acc = 0.0f;
  for (int t = 0; t < T; ++t) {
    int page, off;
    paged_row_b(t, page_tokens, bt, &page, &off);
    const __nv_bfloat16* vrow = v_pages +
        paged_row_offset_b(page, off, kh, page_tokens, head_dim, page_stride);
    acc += __bfloat162float(ph[t]) * __bfloat162float(vrow[d]);
  }
  out[idx] = __float2bfloat16_rn(acc);
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entries
// ---------------------------------------------------------------------------

void batch_embed_gather_bf16(const __nv_bfloat16* embed, const int* token_ids,
                             __nv_bfloat16* out, int B, int H,
                             cudaStream_t stream) {
  CUDALM_PRECONDITION(B >= 1, "batch_embed_gather_bf16 requires B >= 1");
  CUDALM_PRECONDITION(H > 0, "batch_embed_gather_bf16 requires H > 0");
  dim3 grid((H + 127) / 128, B);
  batch_embed_gather_kernel<<<grid, 128, 0, stream>>>(embed, token_ids, out, H);
  CUDA_CHECK(cudaGetLastError());
}

void batch_int4_gemv_bf16(const std::uint8_t* weight, const __half* scale,
                          const __nv_bfloat16* x, __nv_bfloat16* y, int N,
                          int K, int B, cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "batch_int4_gemv_bf16 requires N >= 1");
  CUDALM_PRECONDITION(B >= 1, "batch_int4_gemv_bf16 requires B >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 128 == 0,
                      "batch_int4_gemv_bf16 requires K % 128 == 0");
  dim3 grid((N + kBatchInt4Rows - 1) / kBatchInt4Rows, B);
  batch_int4gemv_rowtile4_bf16_kernel<<<grid, kBatchInt4Block, 0, stream>>>(
      reinterpret_cast<const uint4*>(weight), scale,
      reinterpret_cast<const uint4*>(x), y, N, K);
  CUDA_CHECK_LAUNCH();
}

void batch_bf16_gemv(const __nv_bfloat16* weight, const __nv_bfloat16* x,
                     __nv_bfloat16* y, int N, int K, int B,
                     cudaStream_t stream) {
  CUDALM_PRECONDITION(N >= 1, "batch_bf16_gemv requires N >= 1");
  CUDALM_PRECONDITION(B >= 1, "batch_bf16_gemv requires B >= 1");
  CUDALM_PRECONDITION(K > 0 && K % 8 == 0,
                      "batch_bf16_gemv requires K % 8 == 0");
  dim3 grid(N, B);
  batch_bf16_gemv_vec4_row_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<const U16*>(weight), reinterpret_cast<const U16*>(x),
      y, N, K);
  CUDA_CHECK_LAUNCH();
}

void batch_partial_rope_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int B,
                             int unit_rows, int head_dim, int rotary_dim,
                             const float* cos_t, const float* sin_t,
                             cudaStream_t stream) {
  CUDALM_PRECONDITION(B >= 1, "batch_partial_rope_bf16 requires B >= 1");
  CUDALM_PRECONDITION(
      unit_rows >= 1 && head_dim >= 2 && rotary_dim >= 2 &&
          rotary_dim % 2 == 0 && rotary_dim <= head_dim,
      "batch_partial_rope_bf16: bad shape");
  const int nrows = B * unit_rows;
  const int total = nrows * head_dim;
  const int grid = (total + 127) / 128;
  batch_partial_rope_kernel<<<grid, 128, 0, stream>>>(x, y, nrows, unit_rows,
                                                      head_dim, rotary_dim,
                                                      cos_t, sin_t);
  CUDA_CHECK_LAUNCH();
}

void batch_deltanet_conv_decode_bf16(__nv_bfloat16* conv_base,
                                     const int* d_slots,
                                     const __nv_bfloat16* new_mixed,
                                     const __nv_bfloat16* conv_w,
                                     __nv_bfloat16* conv_out,
                                     __nv_bfloat16* conv_silu, int conv_dim,
                                     int B, cudaStream_t stream) {
  CUDALM_PRECONDITION(conv_dim > 0, "batch deltanet conv: conv_dim > 0");
  CUDALM_PRECONDITION(B >= 1, "batch deltanet conv: B >= 1");
  dim3 grid((conv_dim + 255) / 256, B);
  batch_deltanet_conv_kernel<<<grid, 256, 0, stream>>>(
      conv_base, d_slots, new_mixed, conv_w, conv_out, conv_silu, conv_dim);
  CUDA_CHECK(cudaGetLastError());
}

void batch_deltanet_conv_split_bf16(const __nv_bfloat16* conv_silu,
                                    __nv_bfloat16* q, __nv_bfloat16* k,
                                    __nv_bfloat16* v, int key_dim,
                                    int value_dim, int B, cudaStream_t stream) {
  CUDALM_PRECONDITION(key_dim > 0 && value_dim > 0,
                      "batch deltanet conv split: dims > 0");
  CUDALM_PRECONDITION(B >= 1, "batch deltanet conv split: B >= 1");
  const int conv_dim = 2 * key_dim + value_dim;
  const int total = B * conv_dim;
  const int grid = (total + 255) / 256;
  batch_deltanet_conv_split_kernel<<<grid, 256, 0, stream>>>(
      conv_silu, q, k, v, conv_dim, key_dim, value_dim, total);
  CUDA_CHECK(cudaGetLastError());
}

void batch_deltanet_gbeta_bf16(const __nv_bfloat16* b, const __nv_bfloat16* a,
                               const float* A_log,
                               const __nv_bfloat16* dt_bias,
                               __nv_bfloat16* beta, float* g, int n_heads,
                               int B, cudaStream_t stream) {
  CUDALM_PRECONDITION(n_heads > 0, "batch gbeta: n_heads > 0");
  CUDALM_PRECONDITION(B >= 1, "batch gbeta: B >= 1");
  const int total = n_heads * B;
  const int grid = (total + 31) / 32;
  batch_deltanet_gbeta_kernel<<<grid, 32, 0, stream>>>(
      b, a, A_log, dt_bias, beta, g, total, n_heads);
  CUDA_CHECK(cudaGetLastError());
}

void batch_deltanet_delta_rule_fp32(const __nv_bfloat16* q,
                                    const __nv_bfloat16* k,
                                    const __nv_bfloat16* v, const float* g,
                                    const __nv_bfloat16* beta, float* rec_base,
                                    const int* d_slots, __nv_bfloat16* core_out,
                                    int n_heads, int head_dim, float eps,
                                    int B, cudaStream_t stream) {
  CUDALM_PRECONDITION(
      head_dim == 128,
      "batch deltanet delta rule: head_dim == 128 (pinned 0.8B)");
  CUDALM_PRECONDITION(n_heads > 0, "batch delta rule: n_heads > 0");
  CUDALM_PRECONDITION(B >= 1, "batch delta rule: B >= 1");
  batch_deltanet_delta_kernel<<<n_heads * B, 128, 0, stream>>>(
      q, k, v, g, beta, rec_base, d_slots, core_out, n_heads, head_dim, eps);
  CUDA_CHECK(cudaGetLastError());
}

void batch_paged_kv_write_bf16(const __nv_bfloat16* k_src,
                               const __nv_bfloat16* v_src,
                               __nv_bfloat16* k_pages, __nv_bfloat16* v_pages,
                               const int* block_table, const int* d_positions,
                               int row_stride, int page_tokens, int n_kv_heads,
                               int head_dim, std::size_t page_stride, int B,
                               cudaStream_t stream) {
  CUDALM_PRECONDITION(
      n_kv_heads >= 1 && head_dim >= 1 && page_tokens >= 1 && B >= 1 &&
          row_stride >= 1 &&
          page_stride >=
              static_cast<std::size_t>(n_kv_heads) * page_tokens * head_dim,
      "batch_paged_kv_write_bf16: bad shape");
  const int total = n_kv_heads * head_dim;
  const int grid = (B * total + 127) / 128;
  batch_paged_kv_write_kernel<<<grid, 128, 0, stream>>>(
      k_src, v_src, k_pages, v_pages, block_table, d_positions, row_stride,
      page_tokens, n_kv_heads, head_dim, page_stride, B);
  CUDA_CHECK_LAUNCH();
}

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
                                       cudaStream_t stream) {
  CUDALM_PRECONDITION(
      n_heads % n_kv_heads == 0 && n_heads >= 1 && n_kv_heads >= 1 &&
          head_dim >= 1 && page_tokens >= 1 && T_max >= 1 && B >= 1 &&
          row_stride >= 1 &&
          page_stride >=
              static_cast<std::size_t>(n_kv_heads) * page_tokens * head_dim,
      "batch_paged_attention_decode_bf16: bad shape");
  // Same scale construction as the frozen paged decode wrapper.
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  __nv_bfloat16* scores2 = scratch;
  __nv_bfloat16* probs =
      scratch + static_cast<std::size_t>(B) * n_heads * T_max;

  {
    const int total = B * n_heads * T_max;
    const int grid = (total + 127) / 128;
    batch_paged_scores_kernel<<<grid, 128, 0, stream>>>(
        q, k_pages, block_table, d_positions, row_stride, scores2, T_max,
        n_heads, n_kv_heads, page_tokens, head_dim, page_stride, scale, B);
    CUDA_CHECK_LAUNCH();
  }
  {
    dim3 grid_sm(n_heads, B);
    batch_paged_softmax_kernel<<<grid_sm, 256, 0, stream>>>(
        scores2, probs, d_positions, T_max, n_heads);
    CUDA_CHECK_LAUNCH();
  }
  {
    const int total = B * n_heads * head_dim;
    const int grid = (total + 127) / 128;
    batch_paged_pv_kernel<<<grid, 128, 0, stream>>>(
        probs, v_pages, block_table, d_positions, row_stride, out, T_max,
        n_heads, n_kv_heads, page_tokens, head_dim, page_stride, B);
    CUDA_CHECK_LAUNCH();
  }
}

}  // namespace kernels
}  // namespace cudalm
