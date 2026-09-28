// CUDALM — v0.7 Phase C: DeltaNet delta-rule state-update variants
// (Qwen3.5-0.8B: n_heads = 16, head_dim = 128).
//
// The frozen baseline (deltanet_delta_kernel in
// src/kernels/qwen35_deltanet_kernels.cu) is: grid = n_heads, block = 128,
// one thread per value index t; per head it runs the l2norm/qk reductions
// (exact 128-thread tree) and then, per thread, TWO 128-iteration passes
// over the state column S[h,k,t] (k-ascending, fp32, one thread per
// value): the A/B pass (loads) and the in-place update pass (which RELOADS
// the column from L1/L2 — the baseline compiles at 64 regs and cannot keep
// the 128-deep column in registers).
//
// Every candidate here keeps the per-element fp32 operation sequence
// IDENTICAL to the frozen kernel (same reduction tree, same k-ascending
// accumulation by one thread per value, same bf16 rounding points) and
// changes only parallelism / data movement:
//
//   vreg   : the 128-deep column is kept in a REGISTER array across both
//            passes — the update pass issues no state loads.
//   vvec   : threads 0..31 of the 128-thread block each process 4 values
//            (t = 4l..4l+3) with float4 state loads: 4 independent value
//            chains per thread (ILP), 4x fewer load instructions; the
//            update pass re-reads the (L1-hot) column as float4.
//   vchunk : 2-kernel split — (1) a 128-thread block per head runs the
//            exact-tree reduction pass and writes the normalized q_s/k_s +
//            qk to a small fp32 scratch; (2) the state pass runs in grid
//            (n_heads, 4) blocks of 32 threads (one value per thread,
//            register-resident column), so the state work spreads over 4x
//            more blocks/SMs with no redundant reduction work.
//
// The batch variants mirror the frozen batch kernel (one (b,h) unit each;
// state base = rec_base + d_slots[b]*n_heads*hd*hd).

#include "cudalm/kernels/deltanet_delta_qwen35.h"

#include "cudalm/cuda_check.h"
#include "cudalm/kernels/batch_decode.h"
#include "cudalm/kernels/qwen35_deltanet_kernels.h"

namespace cudalm {
namespace kernels {

namespace {

constexpr int kHeadDim = 128;  // pinned 0.8B linear_key/value_head_dim

// Exact-tree 128-thread block sum (frozen block_sum_128 replicated
// verbatim: warp-shfl 5-step + smem[4] + ((s0+s1)+s2)+s3).
__device__ __forceinline__ float block_sum_128(float val, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __syncthreads();
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    val += __shfl_down_sync(0xffffffffu, val, o);
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  if (threadIdx.x == 0) {
    smem[0] = smem[0] + smem[1] + smem[2] + smem[3];
  }
  __syncthreads();
  return smem[0];
}

// Frozen l2norm + qk prologue (128 threads): on entry q_s/k_s smem hold the
// bf16->f32 q/k; on exit q_s/k_s hold the normalized + q-scaled fp32 values
// (q additionally scaled by 1/sqrt(hd) in fp32 AFTER the bf16 round, exactly
// like the frozen kernel) and qk is the broadcast dot. The rounding
// boundaries (bf16 square, bf16 t, bf16 sqrt, bf16 reciprocal, bf16 scale)
// are at the identical program points of the frozen kernel.
__device__ __forceinline__ void delta_prologue(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ g, const __nv_bfloat16* __restrict__ beta,
    float* q_s, float* k_s, float* smem_red, float* v_reg, float* qk_out,
    float* exp_g_out, float* beta_out, int h, int hd, float eps) {
  const int t = threadIdx.x;
  q_s[t] = __bfloat162float(q[h * hd + t]);
  k_s[t] = __bfloat162float(k[h * hd + t]);
  *v_reg = __bfloat162float(v[h * hd + t]);
  __syncthreads();

  const float qprod =
      __bfloat162float(__float2bfloat16_rn(q_s[t] * q_s[t]));
  const float kprod =
      __bfloat162float(__float2bfloat16_rn(k_s[t] * k_s[t]));
  const float qn = block_sum_128(qprod, smem_red);
  const float kn = block_sum_128(kprod, smem_red);
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

  *exp_g_out = expf(g[h]);
  *beta_out = __bfloat162float(beta[h]);
  *qk_out = block_sum_128(k_s[t] * q_s[t], smem_red);
  __syncthreads();
}

// ---------------------------------------------------------------------------
// vreg: register-resident column (B=1)
// ---------------------------------------------------------------------------
__global__ void deltanet_delta_vreg_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, float* __restrict__ S,
    __nv_bfloat16* __restrict__ core_out, int n_heads, int hd, float eps) {
  const int h = blockIdx.x;
  if (h >= n_heads) return;
  const int t = threadIdx.x;
  __shared__ float q_s[kHeadDim];
  __shared__ float k_s[kHeadDim];
  __shared__ float smem_red[4];
  float v_reg, qk, exp_g, beta_h;
  delta_prologue(q, k, v, g, beta, q_s, k_s, smem_red, &v_reg, &qk, &exp_g,
                 &beta_h, h, hd, eps);

  // A/B pass: load the column into registers while accumulating.
  float A = 0.0f, B = 0.0f;
  float col[kHeadDim];
  float* S_col = S + static_cast<std::size_t>(h) * hd * hd + t;
  #pragma unroll
  for (int kk = 0; kk < hd; ++kk) {
    const float s = S_col[static_cast<std::size_t>(kk) * hd];
    col[kk] = s;
    A += s * exp_g * k_s[kk];
    B += s * exp_g * q_s[kk];
  }
  const float d = (v_reg - A) * beta_h;
  const float o = B + d * qk;  // output from the UPDATED state
  core_out[h * hd + t] = __float2bfloat16_rn(o);

  // Update pass from the REGISTER-resident column (no state reload).
  #pragma unroll
  for (int kk = 0; kk < hd; ++kk) {
    col[kk] = col[kk] * exp_g + k_s[kk] * d;
    S_col[static_cast<std::size_t>(kk) * hd] = col[kk];
  }
}

// vreg batch (B rows; one block per (b,h)).
__global__ void batch_deltanet_delta_vreg_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, float* __restrict__ rec_base,
    const int* __restrict__ d_slots, __nv_bfloat16* __restrict__ core_out,
    int n_heads, int hd, float eps) {
  const int b = blockIdx.x / n_heads;
  const int h = blockIdx.x - b * n_heads;
  const int bh = b * n_heads + h;
  const int t = threadIdx.x;
  __shared__ float q_s[kHeadDim];
  __shared__ float k_s[kHeadDim];
  __shared__ float smem_red[4];
  // (b,h)-indexed prologue: the frozen batch kernel indexes q/k/v/g/beta by
  // (b,h); the math is per-row identical, so run the same prologue on the
  // (b,h) row.
  q_s[t] = __bfloat162float(q[bh * hd + t]);
  k_s[t] = __bfloat162float(k[bh * hd + t]);
  const float v_reg = __bfloat162float(v[bh * hd + t]);
  __syncthreads();
  const float qprod =
      __bfloat162float(__float2bfloat16_rn(q_s[t] * q_s[t]));
  const float kprod =
      __bfloat162float(__float2bfloat16_rn(k_s[t] * k_s[t]));
  const float qn = block_sum_128(qprod, smem_red);
  const float kn = block_sum_128(kprod, smem_red);
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
  const float exp_g = expf(g[bh]);
  const float beta_h = __bfloat162float(beta[bh]);
  const float qk = block_sum_128(k_s[t] * q_s[t], smem_red);
  __syncthreads();

  float A = 0.0f, B = 0.0f;
  float col[kHeadDim];
  float* S_col =
      rec_base +
      static_cast<std::size_t>(d_slots[b]) * static_cast<std::size_t>(n_heads) *
          hd * hd +
      static_cast<std::size_t>(h) * hd * hd + t;
  #pragma unroll
  for (int kk = 0; kk < hd; ++kk) {
    const float s = S_col[static_cast<std::size_t>(kk) * hd];
    col[kk] = s;
    A += s * exp_g * k_s[kk];
    B += s * exp_g * q_s[kk];
  }
  const float d = (v_reg - A) * beta_h;
  const float o = B + d * qk;
  core_out[bh * hd + t] = __float2bfloat16_rn(o);
  #pragma unroll
  for (int kk = 0; kk < hd; ++kk) {
    col[kk] = col[kk] * exp_g + k_s[kk] * d;
    S_col[static_cast<std::size_t>(kk) * hd] = col[kk];
  }
}

// ---------------------------------------------------------------------------
// vvec: float4 x 4 values per thread (B=1; threads 0..31 active in the
// state passes, all 128 threads run the frozen prologue)
// ---------------------------------------------------------------------------
__global__ void deltanet_delta_vvec_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, float* __restrict__ S,
    __nv_bfloat16* __restrict__ core_out, int n_heads, int hd, float eps) {
  const int h = blockIdx.x;
  if (h >= n_heads) return;
  const int t = threadIdx.x;
  __shared__ float q_s[kHeadDim];
  __shared__ float k_s[kHeadDim];
  __shared__ float smem_red[4];
  float v_reg, qk, exp_g, beta_h;
  delta_prologue(q, k, v, g, beta, q_s, k_s, smem_red, &v_reg, &qk, &exp_g,
                 &beta_h, h, hd, eps);

  if (t < 32) {
    const int l = t;  // value group: values 4l..4l+3
    const float4* Srow =
        reinterpret_cast<const float4*>(S + static_cast<std::size_t>(h) * hd *
                                                hd + 4 * l);
    float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f;
    float B0 = 0.0f, B1 = 0.0f, B2 = 0.0f, B3 = 0.0f;
    #pragma unroll
    for (int kk = 0; kk < hd; ++kk) {
      const float4 s4 = Srow[kk * (kHeadDim / 4)];
      A0 += s4.x * exp_g * k_s[kk];
      A1 += s4.y * exp_g * k_s[kk];
      A2 += s4.z * exp_g * k_s[kk];
      A3 += s4.w * exp_g * k_s[kk];
      B0 += s4.x * exp_g * q_s[kk];
      B1 += s4.y * exp_g * q_s[kk];
      B2 += s4.z * exp_g * q_s[kk];
      B3 += s4.w * exp_g * q_s[kk];
    }
    const float v0 = __bfloat162float(v[h * hd + 4 * l + 0]);
    const float v1 = __bfloat162float(v[h * hd + 4 * l + 1]);
    const float v2 = __bfloat162float(v[h * hd + 4 * l + 2]);
    const float v3 = __bfloat162float(v[h * hd + 4 * l + 3]);
    const float d0 = (v0 - A0) * beta_h;
    const float d1 = (v1 - A1) * beta_h;
    const float d2 = (v2 - A2) * beta_h;
    const float d3 = (v3 - A3) * beta_h;
    const float o0 = B0 + d0 * qk;
    const float o1 = B1 + d1 * qk;
    const float o2 = B2 + d2 * qk;
    const float o3 = B3 + d3 * qk;
    core_out[h * hd + 4 * l + 0] = __float2bfloat16_rn(o0);
    core_out[h * hd + 4 * l + 1] = __float2bfloat16_rn(o1);
    core_out[h * hd + 4 * l + 2] = __float2bfloat16_rn(o2);
    core_out[h * hd + 4 * l + 3] = __float2bfloat16_rn(o3);
    float4* Srow_w =
        reinterpret_cast<float4*>(S + static_cast<std::size_t>(h) * hd * hd +
                                  4 * l);
    #pragma unroll
    for (int kk = 0; kk < hd; ++kk) {
      const float4 s4 = Srow_w[kk * (kHeadDim / 4)];
      float4 u;
      u.x = s4.x * exp_g + k_s[kk] * d0;
      u.y = s4.y * exp_g + k_s[kk] * d1;
      u.z = s4.z * exp_g + k_s[kk] * d2;
      u.w = s4.w * exp_g + k_s[kk] * d3;
      Srow_w[kk * (kHeadDim / 4)] = u;
    }
  }
}

// vvec batch.
__global__ void batch_deltanet_delta_vvec_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, float* __restrict__ rec_base,
    const int* __restrict__ d_slots, __nv_bfloat16* __restrict__ core_out,
    int n_heads, int hd, float eps) {
  const int b = blockIdx.x / n_heads;
  const int h = blockIdx.x - b * n_heads;
  const int bh = b * n_heads + h;
  const int t = threadIdx.x;
  __shared__ float q_s[kHeadDim];
  __shared__ float k_s[kHeadDim];
  __shared__ float smem_red[4];
  q_s[t] = __bfloat162float(q[bh * hd + t]);
  k_s[t] = __bfloat162float(k[bh * hd + t]);
  const float v_reg = __bfloat162float(v[bh * hd + t]);
  __syncthreads();
  const float qprod =
      __bfloat162float(__float2bfloat16_rn(q_s[t] * q_s[t]));
  const float kprod =
      __bfloat162float(__float2bfloat16_rn(k_s[t] * k_s[t]));
  const float qn = block_sum_128(qprod, smem_red);
  const float kn = block_sum_128(kprod, smem_red);
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
  const float exp_g = expf(g[bh]);
  const float beta_h = __bfloat162float(beta[bh]);
  const float qk = block_sum_128(k_s[t] * q_s[t], smem_red);
  __syncthreads();

  if (t < 32) {
    const int l = t;
    float* S_base =
        rec_base +
        static_cast<std::size_t>(d_slots[b]) * static_cast<std::size_t>(n_heads) *
            hd * hd +
        static_cast<std::size_t>(h) * hd * hd;
    const float4* Srow =
        reinterpret_cast<const float4*>(S_base + 4 * l);
    float A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f;
    float B0 = 0.0f, B1 = 0.0f, B2 = 0.0f, B3 = 0.0f;
    #pragma unroll
    for (int kk = 0; kk < hd; ++kk) {
      const float4 s4 = Srow[kk * (kHeadDim / 4)];
      A0 += s4.x * exp_g * k_s[kk];
      A1 += s4.y * exp_g * k_s[kk];
      A2 += s4.z * exp_g * k_s[kk];
      A3 += s4.w * exp_g * k_s[kk];
      B0 += s4.x * exp_g * q_s[kk];
      B1 += s4.y * exp_g * q_s[kk];
      B2 += s4.z * exp_g * q_s[kk];
      B3 += s4.w * exp_g * q_s[kk];
    }
    const float v0 = __bfloat162float(v[bh * hd + 4 * l + 0]);
    const float v1 = __bfloat162float(v[bh * hd + 4 * l + 1]);
    const float v2 = __bfloat162float(v[bh * hd + 4 * l + 2]);
    const float v3 = __bfloat162float(v[bh * hd + 4 * l + 3]);
    const float d0 = (v0 - A0) * beta_h;
    const float d1 = (v1 - A1) * beta_h;
    const float d2 = (v2 - A2) * beta_h;
    const float d3 = (v3 - A3) * beta_h;
    core_out[bh * hd + 4 * l + 0] = __float2bfloat16_rn(B0 + d0 * qk);
    core_out[bh * hd + 4 * l + 1] = __float2bfloat16_rn(B1 + d1 * qk);
    core_out[bh * hd + 4 * l + 2] = __float2bfloat16_rn(B2 + d2 * qk);
    core_out[bh * hd + 4 * l + 3] = __float2bfloat16_rn(B3 + d3 * qk);
    float4* Srow_w = reinterpret_cast<float4*>(S_base + 4 * l);
    #pragma unroll
    for (int kk = 0; kk < hd; ++kk) {
      const float4 s4 = Srow_w[kk * (kHeadDim / 4)];
      float4 u;
      u.x = s4.x * exp_g + k_s[kk] * d0;
      u.y = s4.y * exp_g + k_s[kk] * d1;
      u.z = s4.z * exp_g + k_s[kk] * d2;
      u.w = s4.w * exp_g + k_s[kk] * d3;
      Srow_w[kk * (kHeadDim / 4)] = u;
    }
  }
}

// ---------------------------------------------------------------------------
// vchunk: 2-kernel (reduction pass + 32-thread state pass)
// ---------------------------------------------------------------------------
// Reduction pass: exact-tree prologue per head; publishes the normalized
// q_s/k_s (fp32) and qk to a small scratch buffer.
__global__ void deltanet_delta_vchunk_red_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, float* __restrict__ qk_out,
    float* __restrict__ norm_out, int n_heads, int hd, float eps) {
  const int h = blockIdx.x;
  if (h >= n_heads) return;
  const int t = threadIdx.x;
  __shared__ float q_s[kHeadDim];
  __shared__ float k_s[kHeadDim];
  __shared__ float smem_red[4];
  float qk;
  (void)v;
  (void)g;
  (void)beta;
  // The prologue reads v/g/beta through the shared helper; vchunk only needs
  // q_s/k_s/qk, so run the same prologue math directly (identical sequence;
  // the v/g/beta outputs are unused here).
  q_s[t] = __bfloat162float(q[h * hd + t]);
  k_s[t] = __bfloat162float(k[h * hd + t]);
  __syncthreads();
  const float qprod =
      __bfloat162float(__float2bfloat16_rn(q_s[t] * q_s[t]));
  const float kprod =
      __bfloat162float(__float2bfloat16_rn(k_s[t] * k_s[t]));
  const float qn = block_sum_128(qprod, smem_red);
  const float kn = block_sum_128(kprod, smem_red);
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
  qk = block_sum_128(k_s[t] * q_s[t], smem_red);
  __syncthreads();
  if (t == 0) qk_out[h] = qk;
  // Publish normalized values (fp32 round-trip is exact).
  norm_out[h * 2 * hd + t] = q_s[t];
  norm_out[h * 2 * hd + hd + t] = k_s[t];
}

// vchunk batch reduction pass (per (b,h)).
__global__ void batch_deltanet_delta_vchunk_red_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    float* __restrict__ qk_out, float* __restrict__ norm_out, int n_heads,
    int hd, float eps) {
  const int b = blockIdx.x / n_heads;
  const int h = blockIdx.x - b * n_heads;
  const int bh = b * n_heads + h;
  const int t = threadIdx.x;
  __shared__ float q_s[kHeadDim];
  __shared__ float k_s[kHeadDim];
  __shared__ float smem_red[4];
  q_s[t] = __bfloat162float(q[bh * hd + t]);
  k_s[t] = __bfloat162float(k[bh * hd + t]);
  __syncthreads();
  const float qprod =
      __bfloat162float(__float2bfloat16_rn(q_s[t] * q_s[t]));
  const float kprod =
      __bfloat162float(__float2bfloat16_rn(k_s[t] * k_s[t]));
  const float qn = block_sum_128(qprod, smem_red);
  const float kn = block_sum_128(kprod, smem_red);
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
  const float qk = block_sum_128(k_s[t] * q_s[t], smem_red);
  __syncthreads();
  if (t == 0) qk_out[bh] = qk;
  norm_out[bh * 2 * hd + t] = q_s[t];
  norm_out[bh * 2 * hd + hd + t] = k_s[t];
}

// State pass: grid (n_heads, 4) — 32-thread blocks, one value per thread,
// register-resident column. norm_out layout: [head, 2*hd] = [q_s; k_s].
__global__ void deltanet_delta_vchunk_s_kernel(
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, float* __restrict__ S,
    const float* __restrict__ qk_in, const float* __restrict__ norm_in,
    __nv_bfloat16* __restrict__ core_out, int n_heads, int hd, float eps) {
  const int h = blockIdx.x;
  const int c = blockIdx.y;  // value chunk 0..3
  const int l = threadIdx.x;  // 0..31
  const int t = c * 32 + l;
  const float* norm = norm_in + static_cast<std::size_t>(h) * 2 * hd;
  const float* q_s = norm;
  const float* k_s = norm + hd;
  const float exp_g = expf(g[h]);
  const float beta_h = __bfloat162float(beta[h]);
  const float qk = qk_in[h];
  const float v_reg = __bfloat162float(v[h * hd + t]);

  float A = 0.0f, B = 0.0f;
  float col[kHeadDim];
  float* S_col = S + static_cast<std::size_t>(h) * hd * hd + t;
  #pragma unroll
  for (int kk = 0; kk < hd; ++kk) {
    const float s = S_col[static_cast<std::size_t>(kk) * hd];
    col[kk] = s;
    A += s * exp_g * k_s[kk];
    B += s * exp_g * q_s[kk];
  }
  const float d = (v_reg - A) * beta_h;
  const float o = B + d * qk;
  core_out[h * hd + t] = __float2bfloat16_rn(o);
  #pragma unroll
  for (int kk = 0; kk < hd; ++kk) {
    col[kk] = col[kk] * exp_g + k_s[kk] * d;
    S_col[static_cast<std::size_t>(kk) * hd] = col[kk];
  }
}

// vchunk batch state pass: grid (n_heads, 4, B).
__global__ void batch_deltanet_delta_vchunk_s_kernel(
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, float* __restrict__ rec_base,
    const int* __restrict__ d_slots, const float* __restrict__ qk_in,
    const float* __restrict__ norm_in, __nv_bfloat16* __restrict__ core_out,
    int n_heads, int hd, float eps) {
  const int h = blockIdx.x;
  const int c = blockIdx.y;
  const int b = blockIdx.z;
  const int bh = b * n_heads + h;
  const int t = c * 32 + threadIdx.x;
  const float* norm = norm_in + static_cast<std::size_t>(bh) * 2 * hd;
  const float* q_s = norm;
  const float* k_s = norm + hd;
  const float exp_g = expf(g[bh]);
  const float beta_h = __bfloat162float(beta[bh]);
  const float qk = qk_in[bh];
  const float v_reg = __bfloat162float(v[bh * hd + t]);

  float A = 0.0f, B = 0.0f;
  float col[kHeadDim];
  float* S_col =
      rec_base +
      static_cast<std::size_t>(d_slots[b]) * static_cast<std::size_t>(n_heads) *
          hd * hd +
      static_cast<std::size_t>(h) * hd * hd + t;
  #pragma unroll
  for (int kk = 0; kk < hd; ++kk) {
    const float s = S_col[static_cast<std::size_t>(kk) * hd];
    col[kk] = s;
    A += s * exp_g * k_s[kk];
    B += s * exp_g * q_s[kk];
  }
  const float d = (v_reg - A) * beta_h;
  const float o = B + d * qk;
  core_out[bh * hd + t] = __float2bfloat16_rn(o);
  #pragma unroll
  for (int kk = 0; kk < hd; ++kk) {
    col[kk] = col[kk] * exp_g + k_s[kk] * d;
    S_col[static_cast<std::size_t>(kk) * hd] = col[kk];
  }
}

// vchunk scratch: qk [n_heads*B] + norm [n_heads*B, 2*hd] fp32. Lazily
// allocated (this module is benchmark/test infrastructure; the production
// runtime never calls it).
struct VchunkScratch {
  float* qk = nullptr;
  float* norm = nullptr;
  int capacity = 0;  // in (n_heads*B) units
};
VchunkScratch& vchunk_scratch() {
  static VchunkScratch s;
  return s;
}
void vchunk_ensure_scratch(int units, cudaStream_t stream) {
  VchunkScratch& s = vchunk_scratch();
  (void)stream;
  if (units <= s.capacity) return;
  if (s.qk) {
    cudaFree(s.qk);
    cudaFree(s.norm);
    s.qk = nullptr;
    s.norm = nullptr;
    s.capacity = 0;
  }
  CUDA_CHECK(cudaMalloc(&s.qk, sizeof(float) * units));
  CUDA_CHECK(cudaMalloc(&s.norm, sizeof(float) * units * 2 * kHeadDim));
  s.capacity = units;
}

// Alignment guard for the float4 variant (16B-aligned state base).
inline bool state_16b_aligned(const float* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0u;
}

// ---------------------------------------------------------------------------
// EXPERIMENTAL / measured dispatcher table.
//
// The variant choice is a STATIC measured-shape table (no runtime
// autotuning), keyed on (n_heads[, B]) for the pinned Qwen3.5-0.8B config
// (n_heads = 16, head_dim = 128), from the Phase-C microbenchmark
// (benchmarks/bench_deltanet_delta_qwen35, exact-SHA runs in benchmarks/
// v07c_deltanet_microbench_run*.txt), NCU (benchmarks/profiling/
// v07c_ncu_*.txt) and the candidate batched-only nsys (benchmarks/
// profiling/v07c_candidate_batched_nsys_*). Anything not in the table
// takes the FROZEN baseline path — the bit-compatible fallback.
//
// MEASURED ADAPTIVE TABLE — FINAL (V07C_FINAL_FUNCTIONAL_SHA). Production
// uses this table; the KEEP decision and the full exact-SHA evidence chain
// (parity 1047 checks, microbench x2, NCU before/after, batched-only nsys,
// canonical E2E 150v150 Welch p=0.012 / Mann-Whitney p<0.0001) are recorded
// in docs/v07_deltanet_optimization.md.
//
//   B=1:  n_heads=16 -> vchunk   (1.79~1.88x; 2-kernel value-chunk split,
//                                    s pass 50 regs; nsys 24.5 -> 18.1us/call)
//   B=2:  n_heads=16 -> vvec     (1.32~1.33x; float4 x4 values/thread)
//   B=3:  n_heads=16 -> vvec     (1.31~1.33x; vreg measured 0.73x, vchunk
//                                    0.66x at B=3 -> frozen would lose)
//   else  -> frozen baseline (no measured data; bit-compatible fallback)
//
// Weighted production score (18 calls/traversal, canonical B mix 19:1:2):
// mixed table ~ -42% delta-kernel time per traversal vs frozen; nsys
// batched-only delta-rule kernel time 58.70 -> 44.44 ms (-24.3%).
// ---------------------------------------------------------------------------

namespace {

// Variant ids: 0 = frozen baseline, 1 = vreg, 2 = vvec, 3 = vchunk.
int pick_delta_variant_b1(int n_heads, int head_dim) {
  (void)head_dim;
  if (n_heads == 16) return 3;  // vchunk: 1.87~1.88x (B=1 microbench)
  return 0;  // frozen baseline (no measured data)
}

int pick_delta_variant_batch(int n_heads, int head_dim, int B) {
  (void)head_dim;
  if (n_heads == 16) {
    if (B == 2 || B == 3) return 2;  // vvec: 1.31~1.33x (B=2/3 microbench)
    return 0;  // B>=4: no measured data -> frozen
  }
  return 0;  // frozen baseline (no measured data)
}

}  // namespace

}  // namespace

// ---------------------------------------------------------------------------
// Host entries
// ---------------------------------------------------------------------------
void deltanet_delta_vreg(const __nv_bfloat16* q, const __nv_bfloat16* k,
                         const __nv_bfloat16* v, const float* g,
                         const __nv_bfloat16* beta, float* S,
                         __nv_bfloat16* core_out, int n_heads, int head_dim,
                         float eps, cudaStream_t stream) {
  CUDALM_PRECONDITION(head_dim == kHeadDim,
                      "deltanet_delta_vreg requires head_dim == 128");
  CUDALM_PRECONDITION(n_heads > 0, "deltanet_delta_vreg: n_heads > 0");
  deltanet_delta_vreg_kernel<<<n_heads, kHeadDim, 0, stream>>>(
      q, k, v, g, beta, S, core_out, n_heads, head_dim, eps);
  CUDA_CHECK(cudaGetLastError());
}

void batch_deltanet_delta_vreg(const __nv_bfloat16* q,
                               const __nv_bfloat16* k,
                               const __nv_bfloat16* v, const float* g,
                               const __nv_bfloat16* beta, float* rec_base,
                               const int* d_slots, __nv_bfloat16* core_out,
                               int n_heads, int head_dim, float eps, int B,
                               cudaStream_t stream) {
  CUDALM_PRECONDITION(head_dim == kHeadDim,
                      "batch_deltanet_delta_vreg requires head_dim == 128");
  CUDALM_PRECONDITION(n_heads > 0, "batch_deltanet_delta_vreg: n_heads > 0");
  CUDALM_PRECONDITION(B >= 1, "batch_deltanet_delta_vreg: B >= 1");
  batch_deltanet_delta_vreg_kernel<<<n_heads * B, kHeadDim, 0, stream>>>(
      q, k, v, g, beta, rec_base, d_slots, core_out, n_heads, head_dim, eps);
  CUDA_CHECK(cudaGetLastError());
}

void deltanet_delta_vvec(const __nv_bfloat16* q, const __nv_bfloat16* k,
                         const __nv_bfloat16* v, const float* g,
                         const __nv_bfloat16* beta, float* S,
                         __nv_bfloat16* core_out, int n_heads, int head_dim,
                         float eps, cudaStream_t stream) {
  CUDALM_PRECONDITION(head_dim == kHeadDim,
                      "deltanet_delta_vvec requires head_dim == 128");
  CUDALM_PRECONDITION(n_heads > 0, "deltanet_delta_vvec: n_heads > 0");
  // 16B alignment contract: the float4 state loads require a 16B-aligned
  // head base (n_heads*hd*hd floats are 64KB-aligned offsets; the base must
  // be too). Fall through to the frozen path otherwise (bit-compatible).
  if (!state_16b_aligned(S)) {
    qwen35_deltanet_delta_rule_fp32(q, k, v, g, beta, S, core_out, n_heads,
                                    head_dim, eps, stream);
    return;
  }
  deltanet_delta_vvec_kernel<<<n_heads, kHeadDim, 0, stream>>>(
      q, k, v, g, beta, S, core_out, n_heads, head_dim, eps);
  CUDA_CHECK(cudaGetLastError());
}

void batch_deltanet_delta_vvec(const __nv_bfloat16* q,
                               const __nv_bfloat16* k,
                               const __nv_bfloat16* v, const float* g,
                               const __nv_bfloat16* beta, float* rec_base,
                               const int* d_slots, __nv_bfloat16* core_out,
                               int n_heads, int head_dim, float eps, int B,
                               cudaStream_t stream) {
  CUDALM_PRECONDITION(head_dim == kHeadDim,
                      "batch_deltanet_delta_vvec requires head_dim == 128");
  CUDALM_PRECONDITION(n_heads > 0, "batch_deltanet_delta_vvec: n_heads > 0");
  CUDALM_PRECONDITION(B >= 1, "batch_deltanet_delta_vvec: B >= 1");
  // d_slots strides are n_heads*hd*hd floats (64KB) — aligned when rec_base
  // is; check per-row bases is impossible host-side without d_slots, so the
  // base check is the contract (the pool is 256B-aligned).
  if (!state_16b_aligned(rec_base)) {
    batch_deltanet_delta_rule_fp32(q, k, v, g, beta, rec_base, d_slots,
                                   core_out, n_heads, head_dim, eps, B, stream);
    return;
  }
  batch_deltanet_delta_vvec_kernel<<<n_heads * B, kHeadDim, 0, stream>>>(
      q, k, v, g, beta, rec_base, d_slots, core_out, n_heads, head_dim, eps);
  CUDA_CHECK(cudaGetLastError());
}

void deltanet_delta_vchunk(const __nv_bfloat16* q, const __nv_bfloat16* k,
                           const __nv_bfloat16* v, const float* g,
                           const __nv_bfloat16* beta, float* S,
                           __nv_bfloat16* core_out, int n_heads, int head_dim,
                           float eps, cudaStream_t stream) {
  CUDALM_PRECONDITION(head_dim == kHeadDim,
                      "deltanet_delta_vchunk requires head_dim == 128");
  CUDALM_PRECONDITION(n_heads > 0, "deltanet_delta_vchunk: n_heads > 0");
  vchunk_ensure_scratch(n_heads, stream);
  dim3 grid(n_heads, 4);
  deltanet_delta_vchunk_red_kernel<<<n_heads, kHeadDim, 0, stream>>>(
      q, k, v, g, beta, vchunk_scratch().qk, vchunk_scratch().norm, n_heads,
      head_dim, eps);
  deltanet_delta_vchunk_s_kernel<<<grid, 32, 0, stream>>>(
      v, g, beta, S, vchunk_scratch().qk, vchunk_scratch().norm, core_out,
      n_heads, head_dim, eps);
  CUDA_CHECK(cudaGetLastError());
}

void batch_deltanet_delta_vchunk(const __nv_bfloat16* q,
                                 const __nv_bfloat16* k,
                                 const __nv_bfloat16* v, const float* g,
                                 const __nv_bfloat16* beta, float* rec_base,
                                 const int* d_slots,
                                 __nv_bfloat16* core_out, int n_heads,
                                 int head_dim, float eps, int B,
                                 cudaStream_t stream) {
  CUDALM_PRECONDITION(head_dim == kHeadDim,
                      "batch_deltanet_delta_vchunk requires head_dim == 128");
  CUDALM_PRECONDITION(n_heads > 0, "batch_deltanet_delta_vchunk: n_heads > 0");
  CUDALM_PRECONDITION(B >= 1, "batch_deltanet_delta_vchunk: B >= 1");
  vchunk_ensure_scratch(n_heads * B, stream);
  dim3 grid(n_heads, 4, B);
  batch_deltanet_delta_vchunk_red_kernel<<<n_heads * B, kHeadDim, 0, stream>>>(
      q, k, vchunk_scratch().qk, vchunk_scratch().norm, n_heads, head_dim,
      eps);
  batch_deltanet_delta_vchunk_s_kernel<<<grid, 32, 0, stream>>>(
      v, g, beta, rec_base, d_slots, vchunk_scratch().qk,
      vchunk_scratch().norm, core_out, n_heads, head_dim, eps);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Experimental / measured dispatcher (NOT a production entry point).
// ---------------------------------------------------------------------------
void deltanet_delta_rule_fp32_qwen35_experimental(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const __nv_bfloat16* beta, float* S,
    __nv_bfloat16* core_out, int n_heads, int head_dim, float eps,
    cudaStream_t stream) {
  CUDALM_PRECONDITION(n_heads > 0,
                      "deltanet_delta_rule_fp32_qwen35_experimental: "
                      "n_heads > 0");
  CUDALM_PRECONDITION(
      head_dim == kHeadDim,
      "deltanet_delta_rule_fp32_qwen35_experimental requires head_dim == 128");
  switch (pick_delta_variant_b1(n_heads, head_dim)) {
    case 1:
      deltanet_delta_vreg(q, k, v, g, beta, S, core_out, n_heads, head_dim,
                          eps, stream);
      break;
    case 2:
      deltanet_delta_vvec(q, k, v, g, beta, S, core_out, n_heads, head_dim,
                          eps, stream);
      break;
    case 3:
      deltanet_delta_vchunk(q, k, v, g, beta, S, core_out, n_heads, head_dim,
                            eps, stream);
      break;
    default:
      qwen35_deltanet_delta_rule_fp32(q, k, v, g, beta, S, core_out, n_heads,
                                      head_dim, eps, stream);
      break;
  }
}

void batch_deltanet_delta_rule_fp32_qwen35_experimental(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const __nv_bfloat16* beta, float* rec_base,
    const int* d_slots, __nv_bfloat16* core_out, int n_heads, int head_dim,
    float eps, int B, cudaStream_t stream) {
  CUDALM_PRECONDITION(n_heads > 0,
                      "batch_deltanet_delta_rule_fp32_qwen35_experimental: "
                      "n_heads > 0");
  CUDALM_PRECONDITION(
      head_dim == kHeadDim,
      "batch_deltanet_delta_rule_fp32_qwen35_experimental requires "
      "head_dim == 128");
  CUDALM_PRECONDITION(B >= 1,
                      "batch_deltanet_delta_rule_fp32_qwen35_experimental: "
                      "B >= 1");
  switch (pick_delta_variant_batch(n_heads, head_dim, B)) {
    case 1:
      batch_deltanet_delta_vreg(q, k, v, g, beta, rec_base, d_slots, core_out,
                                n_heads, head_dim, eps, B, stream);
      break;
    case 2:
      batch_deltanet_delta_vvec(q, k, v, g, beta, rec_base, d_slots, core_out,
                                n_heads, head_dim, eps, B, stream);
      break;
    case 3:
      batch_deltanet_delta_vchunk(q, k, v, g, beta, rec_base, d_slots,
                                  core_out, n_heads, head_dim, eps, B, stream);
      break;
    default:
      batch_deltanet_delta_rule_fp32(q, k, v, g, beta, rec_base, d_slots,
                                     core_out, n_heads, head_dim, eps, B,
                                     stream);
      break;
  }
}

// ---------------------------------------------------------------------------
// Register-count queries
// ---------------------------------------------------------------------------
int deltanet_delta_vreg_regs() {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, deltanet_delta_vreg_kernel) !=
      cudaSuccess)
    return -1;
  return attr.numRegs;
}

int deltanet_delta_vvec_regs() {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, deltanet_delta_vvec_kernel) !=
      cudaSuccess)
    return -1;
  return attr.numRegs;
}

int deltanet_delta_vchunk_regs() {
  cudaFuncAttributes attr;
  if (cudaFuncGetAttributes(&attr, deltanet_delta_vchunk_s_kernel) !=
      cudaSuccess)
    return -1;
  return attr.numRegs;
}

}  // namespace kernels
}  // namespace cudalm
