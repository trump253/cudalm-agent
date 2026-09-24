// CUDALM — Qwen3.5 Gated DeltaNet decode kernels (Phase C).
// Math contract: docs/qwen35_architecture.md §8 + §6.2 (pinned transformers
// fc91372 torch fallback path). The bf16 RNE boundaries mirror the official
// code exactly; the recurrent state is always fp32.

#include "cudalm/kernels/qwen35_deltanet_kernels.h"

#include "cudalm/cuda_check.h"

namespace cudalm {
namespace kernels {

namespace {

constexpr int kHeadDim = 128;  // pinned 0.8B linear_key/value_head_dim

// fp32 softplus matching torch F.softplus (log1p(exp(x)) for x <= 20, else x).
__device__ __forceinline__ float softplus_f32(float x) {
  return (x > 20.0f) ? x : log1pf(expf(x));
}

// Block sum over exactly 128 threads (4 warps) using a warp-shuffle reduce +
// a 4-slot shared partial. A leading __syncthreads makes consecutive calls on
// the same shared scratch race-free (it waits for all threads to have read
// the previous call's result). The result is broadcast in smem[0].
__device__ __forceinline__ float block_sum_128(float val) {
  __shared__ float smem[4];
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

// ---------------------------------------------------------------------------
// 1) depthwise causal conv1d decode update + SiLU (in-place conv state)
// ---------------------------------------------------------------------------
__global__ void deltanet_conv_kernel(__nv_bfloat16* __restrict__ conv_state,
                                     const __nv_bfloat16* __restrict__ new_mixed,
                                     const __nv_bfloat16* __restrict__ conv_w,
                                     __nv_bfloat16* __restrict__ conv_out,
                                     __nv_bfloat16* __restrict__ conv_silu,
                                     int conv_dim) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= conv_dim) return;
  // buf = [cs0, cs1, cs2, new]; read the OLD state first (the conv consumes the
  // OLD state values; the state is then shifted to [cs1, cs2, new]).
  const __nv_bfloat16 cs0 = conv_state[ch * 3 + 0];
  const __nv_bfloat16 cs1 = conv_state[ch * 3 + 1];
  const __nv_bfloat16 cs2 = conv_state[ch * 3 + 2];
  const float buf[4] = {__bfloat162float(cs0), __bfloat162float(cs1),
                        __bfloat162float(cs2), __bfloat162float(new_mixed[ch])};
  // In-place state update (bf16 shift + insert of the new token — bit-exact).
  conv_state[ch * 3 + 0] = cs1;
  conv_state[ch * 3 + 1] = cs2;
  conv_state[ch * 3 + 2] = new_mixed[ch];
  // Depthwise conv: c = sum_j W[ch, j] * buf[j]; fp32 accumulation, one RNE.
  float acc = 0.0f;
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    acc += __bfloat162float(conv_w[ch * 4 + j]) * buf[j];
  const __nv_bfloat16 c = __float2bfloat16_rn(acc);
  conv_out[ch] = c;
  // SiLU on the bf16 c: fp32 silu, one bf16 RNE (official F.silu on bf16).
  const float cf = __bfloat162float(c);
  conv_silu[ch] = __float2bfloat16_rn(cf / (1.0f + expf(-cf)));
}

// ---------------------------------------------------------------------------
// 2) g / beta preparation
// ---------------------------------------------------------------------------
__global__ void deltanet_gbeta_kernel(
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ a, const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ beta, float* __restrict__ g, int n_heads) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= n_heads) return;
  // beta = sigmoid(b) — fp32 opmath, one bf16 RNE (official b.sigmoid()).
  const float bf = __bfloat162float(b[h]);
  beta[h] = __float2bfloat16_rn(1.0f / (1.0f + expf(-bf)));
  // g = -exp(A_log) * softplus(a_f32 + dt_bias_f32) — all fp32.
  const float af = __bfloat162float(a[h]);
  const float dtf = __bfloat162float(dt_bias[h]);
  g[h] = -expf(A_log[h]) * softplus_f32(af + dtf);
}

// ---------------------------------------------------------------------------
// 3) gated delta-rule recurrent decode update (state fp32, in place)
//    One block per head; 128 threads = value index t.
// ---------------------------------------------------------------------------
__global__ void deltanet_delta_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, float* __restrict__ S,
    __nv_bfloat16* __restrict__ core_out, int n_heads, int hd, float eps) {
  const int h = blockIdx.x;
  if (h >= n_heads) return;
  const int t = threadIdx.x;  // value index
  __shared__ float q_s[kHeadDim];
  __shared__ float k_s[kHeadDim];
  q_s[t] = __bfloat162float(q[h * hd + t]);
  k_s[t] = __bfloat162float(k[h * hd + t]);
  const float v_reg = __bfloat162float(v[h * hd + t]);
  __syncthreads();

  // l2norm (pinned torch l2norm is FLA-aligned and operates in bf16): prod =
  // bf16(x^2); ss = sum of the bf16 products (fp32 accumulation, identical to
  // torch's .sum on this data); t = bf16(ss + eps); inv = bf16(rsqrt(t));
  // out = bf16(x * inv). The oracle then casts to fp32 and scales q by
  // 1/sqrt(hd) in fp32 (after the bf16 round).
  const float qprod =
      __bfloat162float(__float2bfloat16_rn(q_s[t] * q_s[t]));
  const float kprod =
      __bfloat162float(__float2bfloat16_rn(k_s[t] * k_s[t]));
  const float qn = block_sum_128(qprod);
  const float kn = block_sum_128(kprod);
  __syncthreads();
  const float q_t = __bfloat162float(__float2bfloat16_rn(qn + eps));
  const float k_t = __bfloat162float(__float2bfloat16_rn(kn + eps));
  // torch.rsqrt on a bf16 tensor does NOT upcast to a single fp32 rsqrt: it is
  // bf16(1.0 / bf16(sqrt(f32(t)))) — an fp32 sqrt rounded to bf16, then an
  // fp32 reciprocal rounded to bf16 (two bf16 roundings). Replicated exactly.
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

  const float exp_g = expf(g[h]);
  const float beta_h = __bfloat162float(beta[h]);
  const float qk = block_sum_128(k_s[t] * q_s[t]);
  __syncthreads();

  // Per value index t:
  //   A = sum_k S[h,k,t]*exp_g*k_s[k]   (= m, decayed state @ k)
  //   B = sum_k S[h,k,t]*exp_g*q_s[k]   (for the output)
  float A = 0.0f, B = 0.0f;
  float* S_col = S + static_cast<std::size_t>(h) * hd * hd + t;
  for (int kk = 0; kk < hd; ++kk) {
    const float s = S_col[static_cast<std::size_t>(kk) * hd];
    A += s * exp_g * k_s[kk];
    B += s * exp_g * q_s[kk];
  }
  const float d = (v_reg - A) * beta_h;
  const float o = B + d * qk;  // output from the UPDATED state (docs §8)
  core_out[h * hd + t] = __float2bfloat16_rn(o);

  // Update the column: S[h,k,t] = S[h,k,t]*exp_g + k_s[k]*d.
  for (int kk = 0; kk < hd; ++kk) {
    float* sp = S_col + static_cast<std::size_t>(kk) * hd;
    *sp = (*sp) * exp_g + k_s[kk] * d;
  }
}

// ---------------------------------------------------------------------------
// 4) gated RMSNorm (docs §6.2, two bf16 roundings at the official points)
//    One block per row; 128 threads = dim index.
// ---------------------------------------------------------------------------
__global__ void deltanet_gated_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate, const float* __restrict__ w,
    __nv_bfloat16* __restrict__ y, int n, int dim, float eps) {
  const int h = blockIdx.x;
  if (h >= n) return;
  const int t = threadIdx.x;
  __shared__ float x_s[kHeadDim];
  x_s[t] = __bfloat162float(x[h * dim + t]);
  __syncthreads();
  const float var = block_sum_128(x_s[t] * x_s[t]) / static_cast<float>(dim);
  __syncthreads();
  // ROUNDING 1: n = bf16(x_f32 * rsqrt(var + eps))
  const __nv_bfloat16 n_bf =
      __float2bfloat16_rn(x_s[t] * rsqrtf(var + eps));
  // a = w * f32(n) — w is fp32, n upcast; no rounding (official step 2).
  const float a = w[t] * __bfloat162float(n_bf);
  const float gf = __bfloat162float(gate[h * dim + t]);
  const float silu = gf / (1.0f + expf(-gf));
  // ROUNDING 2: y = bf16(a * silu)
  y[h * dim + t] = __float2bfloat16_rn(a * silu);
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entries
// ---------------------------------------------------------------------------
void qwen35_deltanet_conv_decode_bf16(
    __nv_bfloat16* conv_state, const __nv_bfloat16* new_mixed,
    const __nv_bfloat16* conv_w, __nv_bfloat16* conv_out,
    __nv_bfloat16* conv_silu, int conv_dim, cudaStream_t stream) {
  CUDALM_PRECONDITION(conv_dim > 0, "deltanet conv: conv_dim > 0");
  const int block = 256;
  const int grid = (conv_dim + block - 1) / block;
  deltanet_conv_kernel<<<grid, block, 0, stream>>>(
      conv_state, new_mixed, conv_w, conv_out, conv_silu, conv_dim);
  CUDA_CHECK(cudaGetLastError());
}

void qwen35_deltanet_gbeta_bf16(
    const __nv_bfloat16* b, const __nv_bfloat16* a, const float* A_log,
    const __nv_bfloat16* dt_bias, __nv_bfloat16* beta, float* g, int n_heads,
    cudaStream_t stream) {
  CUDALM_PRECONDITION(n_heads > 0, "deltanet gbeta: n_heads > 0");
  const int block = 32;
  const int grid = (n_heads + block - 1) / block;
  deltanet_gbeta_kernel<<<grid, block, 0, stream>>>(
      b, a, A_log, dt_bias, beta, g, n_heads);
  CUDA_CHECK(cudaGetLastError());
}

void qwen35_deltanet_delta_rule_fp32(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const float* g, const __nv_bfloat16* beta, float* S,
    __nv_bfloat16* core_out, int n_heads, int head_dim, float eps,
    cudaStream_t stream) {
  CUDALM_PRECONDITION(
      head_dim == kHeadDim,
      "qwen35_deltanet_delta_rule_fp32 requires head_dim == 128 (pinned 0.8B)");
  CUDALM_PRECONDITION(n_heads > 0, "deltanet delta rule: n_heads > 0");
  deltanet_delta_kernel<<<n_heads, kHeadDim, 0, stream>>>(
      q, k, v, g, beta, S, core_out, n_heads, head_dim, eps);
  CUDA_CHECK(cudaGetLastError());
}

void qwen35_rmsnorm_gated_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gate, const float* w,
    __nv_bfloat16* y, int n, int dim, float eps, cudaStream_t stream) {
  CUDALM_PRECONDITION(
      dim == kHeadDim,
      "qwen35_rmsnorm_gated_bf16 requires dim == 128 (pinned 0.8B)");
  CUDALM_PRECONDITION(n > 0, "deltanet gated rmsnorm: n > 0");
  deltanet_gated_rmsnorm_kernel<<<n, kHeadDim, 0, stream>>>(
      x, gate, w, y, n, dim, eps);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace kernels
}  // namespace cudalm
