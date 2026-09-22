// CUDALM — causal decode attention (CUDALM-native, no upstream port).
//
// 3-kernel pipeline (docs/bootstrap_plan_v0.1.md §6-D/E), clarity-first:
//   1. attention_scores_kernel : one thread per (query-head, t); q·K dot over
//      head_dim with fp32 accumulation, scaled by 1/sqrt(head_dim).
//   2. attention_softmax_kernel: block per query head over t ∈ [0..position];
//      fp32, numerically stable (subtract row max), one pass into scratch.
//   3. attention_pv_kernel     : one thread per (query-head, d); fp32
//      accumulation over t ∈ [0..position], one fp16 RNE store per output.
//
// All intermediate values are fp32 (device scratch); only the final out is
// fp16 — matching the golden math contract (fp32 math, one fp16 RNE at the
// stage boundary). GQA mapping kh(h) = h * n_kv_heads / n_heads (integer
// division, identical to the golden generator).
//
// Provenance: CUDALM-native (see docs/provenance.md, "CUDALM-native").

#include "cudalm/kernels/attention.h"

#include <cmath>
#include <cfloat>

#include "cudalm/cuda_check.h"

namespace cudalm {
namespace kernels {
namespace {

constexpr int kScoresBlock = 128;
constexpr int kSoftmaxBlock = 256;
constexpr int kPvBlock = 128;

// ---------------------------------------------------------------------------
// 1) scores[h, t] = dot(q[h, :], K[kh(h)][t, :]) * scale   (t = 0..position)
// ---------------------------------------------------------------------------
__global__ void attention_scores_kernel(const __half* __restrict__ q,
                                        const __half* __restrict__ k_cache,
                                        float* __restrict__ scores, int T,
                                        int n_heads, int n_kv_heads,
                                        int head_dim, int max_seq_len,
                                        float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_heads * T;
  if (idx >= total) return;
  const int h = idx / T;
  const int t = idx - h * T;
  const int kh = h * n_kv_heads / n_heads;

  const __half* qh = q + static_cast<std::size_t>(h) * head_dim;
  const __half* kr =
      k_cache + (static_cast<std::size_t>(kh) * max_seq_len + t) * head_dim;

  float acc = 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    acc += __half2float(qh[d]) * __half2float(kr[d]);
  }
  scores[idx] = acc * scale;
}

// Warp-level max reduction (in place, result in every lane's `v`).
__device__ inline float block_max_reduce(float v, float* smem, int tid) {
  // intra-warp
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  // inter-warp (only warp 0)
  const int nwarps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    v = (lane < nwarps) ? smem[lane] : -FLT_MAX;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

__device__ inline float block_sum_reduce(float v, float* smem, int tid) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int nwarps = (blockDim.x + 31) >> 5;
  if (warp == 0) {
    v = (lane < nwarps) ? smem[lane] : 0.0f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------------------
// 2) softmax over row [0..T): probs = exp(s - max) / sum(exp(s - max))
//    One block per query head. Dynamic smem: [T] exp values + [32] warps.
// ---------------------------------------------------------------------------
__global__ void attention_softmax_kernel(const float* __restrict__ scores,
                                         float* __restrict__ probs, int T) {
  const int h = blockIdx.x;
  const float* s = scores + static_cast<std::size_t>(h) * T;
  float* p = probs + static_cast<std::size_t>(h) * T;
  extern __shared__ float sh[];  // sh[0..T-1] exp, sh[T..] warp partials
  float* sh_exp = sh;
  float* sh_red = sh + T;
  const int tid = threadIdx.x;

  float m = -FLT_MAX;
  for (int t = tid; t < T; t += blockDim.x) m = fmaxf(m, s[t]);
  m = block_max_reduce(m, sh_red, tid);

  float acc = 0.0f;
  for (int t = tid; t < T; t += blockDim.x) {
    const float e = expf(s[t] - m);
    sh_exp[t] = e;
    acc += e;
  }
  __syncthreads();
  const float sum = block_sum_reduce(acc, sh_red, tid);

  for (int t = tid; t < T; t += blockDim.x) p[t] = sh_exp[t] / sum;
}

// ---------------------------------------------------------------------------
// 3) out[h, d] = sum_t probs[h, t] * V[kh(h)][t, d]  (fp32, one fp16 RNE)
// ---------------------------------------------------------------------------
__global__ void attention_pv_kernel(const float* __restrict__ probs,
                                    const __half* __restrict__ v_cache,
                                    __half* __restrict__ out, int T, int n_heads,
                                    int n_kv_heads, int head_dim,
                                    int max_seq_len) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_heads * head_dim;
  if (idx >= total) return;
  const int h = idx / head_dim;
  const int d = idx - h * head_dim;
  const int kh = h * n_kv_heads / n_heads;

  const float* ph = probs + static_cast<std::size_t>(h) * T;
  const __half* vb =
      v_cache + static_cast<std::size_t>(kh) * max_seq_len * head_dim;

  float acc = 0.0f;
  for (int t = 0; t < T; ++t) {
    acc += ph[t] * __half2float(vb[static_cast<std::size_t>(t) * head_dim + d]);
  }
  out[idx] = __float2half_rn(acc);
}

}  // namespace

void attention_decode_fp16(const __half* q, const __half* k_cache,
                           const __half* v_cache, int position, __half* out,
                           int n_heads, int n_kv_heads, int head_dim,
                           int max_seq_len, float* scratch,
                           cudaStream_t stream) {
  CUDALM_PRECONDITION(
      n_heads >= 1 && n_kv_heads >= 1 && head_dim >= 1 && max_seq_len >= 1 &&
          position >= 0 && position < max_seq_len &&
          (n_heads % n_kv_heads) == 0,
      "attention_decode_fp16: requires n_heads% n_kv_heads==0, 0<=position<max_seq_len");

  const int T = position + 1;
  const int total_scores = n_heads * T;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // scratch layout: [scores | probs], each n_heads*T fp32.
  float* scores = scratch;
  float* probs = scratch + total_scores;

  const int scores_grid = (total_scores + kScoresBlock - 1) / kScoresBlock;
  attention_scores_kernel<<<scores_grid, kScoresBlock, 0, stream>>>(
      q, k_cache, scores, T, n_heads, n_kv_heads, head_dim, max_seq_len, scale);
  CUDA_CHECK_LAUNCH();

  const std::size_t smem = (static_cast<std::size_t>(T) + 32) * sizeof(float);
  attention_softmax_kernel<<<n_heads, kSoftmaxBlock, smem, stream>>>(
      scores, probs, T);
  CUDA_CHECK_LAUNCH();

  const int total_pv = n_heads * head_dim;
  const int pv_grid = (total_pv + kPvBlock - 1) / kPvBlock;
  attention_pv_kernel<<<pv_grid, kPvBlock, 0, stream>>>(probs, v_cache, out, T,
                                                        n_heads, n_kv_heads,
                                                        head_dim, max_seq_len);
  CUDA_CHECK_LAUNCH();
}

}  // namespace kernels
}  // namespace cudalm
