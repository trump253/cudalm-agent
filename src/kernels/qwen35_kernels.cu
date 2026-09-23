// CUDALM — Qwen3.5 BF16 kernel family implementation (v0.2, Phase B).
// CUDALM-native. Math contract + provenance: see
// include/cudalm/kernels/qwen35_kernels.h and
// docs/qwen35_architecture.md §5-§7 (pinned official rounding boundaries).

#include "cudalm/kernels/qwen35_kernels.h"

#include <cfloat>
#include <cmath>

#include "cudalm/cuda_check.h"

namespace cudalm {
namespace kernels {
namespace {

// 16B unit of 8 bf16 (elementwise kernels).
union Bf16x8 {
  uint4 v;
  __nv_bfloat16 h[8];
};

bool ptr_aligned(const void* p, std::size_t align) {
  return (reinterpret_cast<std::uintptr_t>(p) & (align - 1)) == 0;
}

// ---------------------------------------------------------------------------
// Zero-centered RMSNorm (pinned Qwen3_5RMSNorm):
//   y = bf16( (f32(x) * rsqrt(mean(f32(x)^2) + eps)) * (1 + f32(w)) )
// One block per row, 256 threads, PER = H/256 elements per thread.
// ---------------------------------------------------------------------------
constexpr int kZcBlock = 256;

// CUDA 11.8 has no __bfloat1622float2; split by hand (same result).
__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 h) {
  return make_float2(__bfloat162float(h.x), __bfloat162float(h.y));
}

template <int PER>
__global__ void qwen35_rmsnorm_zc_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ w,
                                         __nv_bfloat16* __restrict__ y, int H,
                                         float eps) {
  static_assert(PER == 1 || PER == 4, "PER");
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const __nv_bfloat16* __restrict__ xrow =
      x + static_cast<std::size_t>(row) * H;
  __nv_bfloat16* __restrict__ yrow = y + static_cast<std::size_t>(row) * H;
  const int base = tid * PER;  // first element owned by this thread

  float vals[PER];
  float ss = 0.f;
  if (PER == 1) {
    vals[0] = __bfloat162float(xrow[base]);
    ss = vals[0] * vals[0];
  } else {
    // PER == 4: one 8B load of 4 bf16 (two __nv_bfloat162).
    const __nv_bfloat162* p =
        reinterpret_cast<const __nv_bfloat162*>(xrow + base);
#pragma unroll
    for (int i = 0; i < PER / 2; ++i) {
      const float2 f = bf162_to_float2(p[i]);
      vals[2 * i] = f.x;
      vals[2 * i + 1] = f.y;
      ss += f.x * f.x + f.y * f.y;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    ss += __shfl_down_sync(0xffffffffu, ss, offset);

  const int nwarp = (blockDim.x + 31) >> 5;
  __shared__ float warp_sums[32];
  __shared__ float s_inv_rms;
  if ((tid & 31) == 0) warp_sums[tid >> 5] = ss;
  __syncthreads();
  if (tid < 32) {
    float v = (tid < nwarp) ? warp_sums[tid] : 0.f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, offset);
    // Official: rsqrt(mean(x^2) + eps) with mean = sum/H (fp32 divide).
    if (tid == 0) s_inv_rms = rsqrtf(v / static_cast<float>(H) + eps);
  }
  __syncthreads();
  const float inv_rms = s_inv_rms;

  if (PER == 1) {
    const float w1 = 1.0f + __bfloat162float(w[base]);
    // Official order: (x * inv_rms) * (1 + w) — all fp32, one bf16 RNE.
    yrow[base] = __float2bfloat16_rn(vals[0] * inv_rms * w1);
  } else {
    const __nv_bfloat162* wp =
        reinterpret_cast<const __nv_bfloat162*>(w + base);
    __nv_bfloat162* q = reinterpret_cast<__nv_bfloat162*>(yrow + base);
#pragma unroll
    for (int i = 0; i < PER / 2; ++i) {
      const float2 fw = bf162_to_float2(wp[i]);
      const float w0 = 1.0f + fw.x;
      const float w1 = 1.0f + fw.y;
      const float o0 = vals[2 * i] * inv_rms * w0;
      const float o1 = vals[2 * i + 1] * inv_rms * w1;
      q[i] = __floats2bfloat162_rn(o0, o1);
    }
  }
}

// ---------------------------------------------------------------------------
// Fused [q;gate] split (plain gather — bit-exact copies).
// ---------------------------------------------------------------------------
__global__ void qwen35_split_q_gate_kernel(const __nv_bfloat16* __restrict__ fused,
                                           __nv_bfloat16* __restrict__ q,
                                           __nv_bfloat16* __restrict__ gate,
                                           int head_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int h = idx / head_dim;
  const int d = idx - h * head_dim;
  const std::size_t src = static_cast<std::size_t>(h) * (2 * head_dim) + d;
  q[idx] = fused[src];
  gate[idx] = fused[src + head_dim];
}

// ---------------------------------------------------------------------------
// Partial rotary (rotate-half, first rotary_dim dims). One thread per
// output element; cos/sin arrive as a device fp32 table of rotary_dim/2
// values (host-computed, fp32) and are cast to bf16 here — the official
// `cos.to(dtype=x.dtype)` before the multiply.
// ---------------------------------------------------------------------------
__global__ void qwen35_partial_rope_kernel(const __nv_bfloat16* __restrict__ x,
                                           __nv_bfloat16* __restrict__ y,
                                           int head_dim, int rotary_dim,
                                           const float* __restrict__ cos_t,
                                           const float* __restrict__ sin_t) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = idx / head_dim;
  const int d = idx - m * head_dim;
  const __nv_bfloat16* __restrict__ xrow =
      x + static_cast<std::size_t>(m) * head_dim;
  __nv_bfloat16* __restrict__ yrow = y + static_cast<std::size_t>(m) * head_dim;

  if (d >= rotary_dim) {
    yrow[d] = xrow[d];  // pass-through (no rounding: it is a copy)
    return;
  }
  // Official partial RoPE (pinned apply_rotary_pos_emb + the rope
  // embedding's emb = cat(freqs, freqs)): element d of the first
  // rotary_dim uses frequency f_{d % (rotary_dim/2)} and pairs with the
  // element across the halves (d +/- rotary_dim/2).
  const int j = d % (rotary_dim / 2);
  const __nv_bfloat16 cb = __float2bfloat16_rn(cos_t[j]);
  const __nv_bfloat16 sb = __float2bfloat16_rn(sin_t[j]);
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
// bf16 KV-cache write (plain copies — bit-exact round trip).
// ---------------------------------------------------------------------------
__global__ void qwen35_kv_write_kernel(const __nv_bfloat16* __restrict__ k_src,
                                       const __nv_bfloat16* __restrict__ v_src,
                                       __nv_bfloat16* __restrict__ k_dst,
                                       __nv_bfloat16* __restrict__ v_dst,
                                       int position, int max_seq_len,
                                       int head_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = idx / head_dim;
  const int d = idx - n * head_dim;
  const std::size_t dst =
      (static_cast<std::size_t>(n) * max_seq_len + position) * head_dim + d;
  k_dst[dst] = k_src[idx];
  v_dst[dst] = v_src[idx];
}

// ---------------------------------------------------------------------------
// Decode attention (3-kernel pipeline, official bf16 rounding boundaries).
// scratch: bf16 [scores2 | probs], each n_heads * max_seq_len.
// ---------------------------------------------------------------------------
constexpr int kScoresBfBlock = 128;
constexpr int kSoftmaxBfBlock = 256;
constexpr int kPvBfBlock = 128;

// 1) s1 = bf16(fp32 dot); s2 = bf16(f32(s1) * scale)
__global__ void qwen35_attention_scores_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ scores2, int T, int n_heads, int n_kv_heads,
    int head_dim, int max_seq_len, float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_heads * T;
  if (idx >= total) return;
  const int h = idx / T;
  const int t = idx - h * T;
  const int kh = h * n_kv_heads / n_heads;

  const __nv_bfloat16* qh = q + static_cast<std::size_t>(h) * head_dim;
  const __nv_bfloat16* kr =
      k_cache + (static_cast<std::size_t>(kh) * max_seq_len + t) * head_dim;

  float acc = 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    acc += __bfloat162float(qh[d]) * __bfloat162float(kr[d]);
  }
  // Official: bf16 matmul output (one bf16 RNE), then bf16 x scaling
  // (fp32 multiply of the bf16 values, one more bf16 RNE).
  const __nv_bfloat16 s1 = __float2bfloat16_rn(acc);
  scores2[idx] = __float2bfloat16_rn(__bfloat162float(s1) * scale);
}

__device__ inline float block_max_reduce(float v, float* smem, int tid) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) smem[warp] = v;
  __syncthreads();
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

// 2) softmax over row [0..T) in fp32 (max-subtracted) -> bf16 probs
//    (official F.softmax(dtype=fp32).to(bf16)).
__global__ void qwen35_attention_softmax_kernel(
    const __nv_bfloat16* __restrict__ scores2,
    __nv_bfloat16* __restrict__ probs, int T) {
  const int h = blockIdx.x;
  // scores2 is n_heads*max_seq_len contiguous; row h starts at h*T (T =
  // position+1 is the row stride for this decode step).
  const __nv_bfloat16* srow =
      scores2 + static_cast<std::size_t>(h) * static_cast<std::size_t>(T);
  __nv_bfloat16* prow =
      probs + static_cast<std::size_t>(h) * static_cast<std::size_t>(T);
  extern __shared__ float sh[];  // sh[0..T-1] exp, sh[T..] warp partials
  float* sh_exp = sh;
  float* sh_red = sh + T;
  const int tid = threadIdx.x;

  float m = -FLT_MAX;
  for (int t = tid; t < T; t += blockDim.x)
    m = fmaxf(m, __bfloat162float(srow[t]));
  m = block_max_reduce(m, sh_red, tid);

  float acc = 0.0f;
  for (int t = tid; t < T; t += blockDim.x) {
    const float e = expf(__bfloat162float(srow[t]) - m);
    sh_exp[t] = e;
    acc += e;
  }
  __syncthreads();
  const float sum = block_sum_reduce(acc, sh_red, tid);

  for (int t = tid; t < T; t += blockDim.x)
    prow[t] = __float2bfloat16_rn(sh_exp[t] / sum);
}

// 3) out[h,d] = bf16(Σ_t f32(probs[t]) * f32(V[kh][t,d]))
__global__ void qwen35_attention_pv_kernel(const __nv_bfloat16* __restrict__ probs,
                                           const __nv_bfloat16* __restrict__ v_cache,
                                           __nv_bfloat16* __restrict__ out,
                                           int T, int n_heads, int n_kv_heads,
                                           int head_dim, int max_seq_len) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_heads * head_dim;
  if (idx >= total) return;
  const int h = idx / head_dim;
  const int d = idx - h * head_dim;
  const int kh = h * n_kv_heads / n_heads;

  const __nv_bfloat16* ph = probs + static_cast<std::size_t>(h) * T;
  const __nv_bfloat16* vb =
      v_cache + static_cast<std::size_t>(kh) * max_seq_len * head_dim;

  float acc = 0.0f;
  for (int t = 0; t < T; ++t) {
    acc += __bfloat162float(ph[t]) *
           __bfloat162float(vb[static_cast<std::size_t>(t) * head_dim + d]);
  }
  out[idx] = __float2bfloat16_rn(acc);
}

// ---------------------------------------------------------------------------
// Elementwise (16B vectorized: 8 bf16 per uint4).
// ---------------------------------------------------------------------------
__global__ void qwen35_add_kernel(const Bf16x8* __restrict__ a,
                                  const Bf16x8* __restrict__ b,
                                  Bf16x8* __restrict__ y, int nvec) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nvec) return;
  const Bf16x8 av = a[i];
  const Bf16x8 bv = b[i];
  Bf16x8 ov;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    ov.h[j] = __float2bfloat16_rn(__bfloat162float(av.h[j]) +
                                  __bfloat162float(bv.h[j]));
  }
  y[i] = ov;
}

__global__ void qwen35_silu_mul_kernel(const Bf16x8* __restrict__ gate,
                                       const Bf16x8* __restrict__ up,
                                       Bf16x8* __restrict__ y, int nvec) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nvec) return;
  const Bf16x8 gv = gate[i];
  const Bf16x8 uv = up[i];
  Bf16x8 ov;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float g = __bfloat162float(gv.h[j]);
    // Official SiLU (opmath fp32): x / (1 + exp(-x)), rounded to bf16
    // FIRST (the act_fn output is a bf16 tensor), then the bf16 multiply
    // with up.
    const __nv_bfloat16 s = __float2bfloat16_rn(g / (1.0f + expf(-g)));
    ov.h[j] = __float2bfloat16_rn(__bfloat162float(s) *
                                  __bfloat162float(uv.h[j]));
  }
  y[i] = ov;
}

__global__ void qwen35_gate_mul_kernel(const Bf16x8* __restrict__ attn,
                                       const Bf16x8* __restrict__ gate,
                                       Bf16x8* __restrict__ y, int nvec) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nvec) return;
  const Bf16x8 av = attn[i];
  const Bf16x8 gv = gate[i];
  Bf16x8 ov;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float g = __bfloat162float(gv.h[j]);
    // Official: torch.sigmoid(gate) is a bf16 tensor (fp32 opmath, one
    // bf16 RNE), then the bf16 multiply with the attention output.
    const __nv_bfloat16 sg =
        __float2bfloat16_rn(1.0f / (1.0f + expf(-g)));
    ov.h[j] = __float2bfloat16_rn(__bfloat162float(av.h[j]) *
                                  __bfloat162float(sg));
  }
  y[i] = ov;
}

constexpr int kElemBlock = 128;

}  // namespace

// ---------------------------------------------------------------------------
// Host entries
// ---------------------------------------------------------------------------

void qwen35_rmsnorm_zc_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w,
                            __nv_bfloat16* y, int M, int H, float eps,
                            cudaStream_t stream) {
  CUDALM_PRECONDITION(M >= 1, "qwen35_rmsnorm_zc_bf16 requires M >= 1");
  CUDALM_PRECONDITION(H % kZcBlock == 0,
                      "qwen35_rmsnorm_zc_bf16 requires H % 256 == 0");
  const int per = H / kZcBlock;
  CUDALM_PRECONDITION(per == 1 || per == 4,
                      "qwen35_rmsnorm_zc_bf16 requires H in {256, 1024}");
  const std::size_t align = (per == 4) ? 8 : 4;
  CUDALM_PRECONDITION(
      ptr_aligned(x, align) && ptr_aligned(w, align) && ptr_aligned(y, align),
      "qwen35_rmsnorm_zc_bf16 requires 4B-aligned (H=256) or 8B-aligned "
      "(H=1024) base pointers");

  dim3 grid(M), block(kZcBlock);
  if (per == 1) {
    qwen35_rmsnorm_zc_kernel<1><<<grid, block, 0, stream>>>(x, w, y, H, eps);
    CUDA_CHECK_LAUNCH();
  } else {
    qwen35_rmsnorm_zc_kernel<4><<<grid, block, 0, stream>>>(x, w, y, H, eps);
    CUDA_CHECK_LAUNCH();
  }
}

void qwen35_split_q_gate_bf16(const __nv_bfloat16* fused, __nv_bfloat16* q,
                              __nv_bfloat16* gate, int n_heads, int head_dim,
                              cudaStream_t stream) {
  CUDALM_PRECONDITION(n_heads >= 1 && head_dim >= 1,
                      "qwen35_split_q_gate_bf16: requires positive dims");
  const int total = n_heads * head_dim;
  const int grid = (total + kElemBlock - 1) / kElemBlock;
  qwen35_split_q_gate_kernel<<<grid, kElemBlock, 0, stream>>>(
      fused, q, gate, head_dim);
  CUDA_CHECK_LAUNCH();
}

void qwen35_partial_rope_bf16(const __nv_bfloat16* x, __nv_bfloat16* y,
                              int M, int head_dim, int rotary_dim,
                              const float* cos_t, const float* sin_t,
                              cudaStream_t stream) {
  CUDALM_PRECONDITION(M >= 1 && head_dim >= 2 && rotary_dim >= 2 &&
                          rotary_dim <= head_dim && rotary_dim % 2 == 0,
                      "qwen35_partial_rope_bf16: bad dims");
  const int total = M * head_dim;
  const int grid = (total + kElemBlock - 1) / kElemBlock;
  qwen35_partial_rope_kernel<<<grid, kElemBlock, 0, stream>>>(
      x, y, head_dim, rotary_dim, cos_t, sin_t);
  CUDA_CHECK_LAUNCH();
}

void qwen35_kv_write_bf16(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                          const __nv_bfloat16* k, const __nv_bfloat16* v,
                          int position, int n_kv_heads, int max_seq_len,
                          int head_dim, cudaStream_t stream) {
  CUDALM_PRECONDITION(
      n_kv_heads >= 1 && head_dim >= 1 && max_seq_len >= 1 &&
          position >= 0 && position < max_seq_len,
      "qwen35_kv_write_bf16: requires n_kv_heads>=1, head_dim>=1, "
      "0<=position<max_seq_len");
  const int total = n_kv_heads * head_dim;
  const int grid = (total + kElemBlock - 1) / kElemBlock;
  qwen35_kv_write_kernel<<<grid, kElemBlock, 0, stream>>>(
      k, v, k_cache, v_cache, position, max_seq_len, head_dim);
  CUDA_CHECK_LAUNCH();
}

void qwen35_attention_decode_bf16(const __nv_bfloat16* q,
                                  const __nv_bfloat16* k_cache,
                                  const __nv_bfloat16* v_cache,
                                  int position, __nv_bfloat16* out,
                                  int n_heads, int n_kv_heads, int head_dim,
                                  int max_seq_len, __nv_bfloat16* scratch,
                                  cudaStream_t stream) {
  CUDALM_PRECONDITION(
      n_heads >= 1 && n_kv_heads >= 1 && head_dim >= 1 && max_seq_len >= 1 &&
          position >= 0 && position < max_seq_len &&
          (n_heads % n_kv_heads) == 0,
      "qwen35_attention_decode_bf16: requires n_heads % n_kv_heads == 0, "
      "0<=position<max_seq_len");

  const int T = position + 1;
  const int total_scores = n_heads * T;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // scratch layout: [scores2 | probs], each n_heads*T bf16.
  __nv_bfloat16* scores2 = scratch;
  __nv_bfloat16* probs = scratch + n_heads * max_seq_len;

  const int scores_grid = (total_scores + kScoresBfBlock - 1) / kScoresBfBlock;
  qwen35_attention_scores_kernel<<<scores_grid, kScoresBfBlock, 0, stream>>>(
      q, k_cache, scores2, T, n_heads, n_kv_heads, head_dim, max_seq_len,
      scale);
  CUDA_CHECK_LAUNCH();

  const std::size_t smem = (static_cast<std::size_t>(T) + 32) * sizeof(float);
  qwen35_attention_softmax_kernel<<<n_heads, kSoftmaxBfBlock, smem, stream>>>(
      scores2, probs, T);
  CUDA_CHECK_LAUNCH();

  const int total_pv = n_heads * head_dim;
  const int pv_grid = (total_pv + kPvBfBlock - 1) / kPvBfBlock;
  qwen35_attention_pv_kernel<<<pv_grid, kPvBfBlock, 0, stream>>>(
      probs, v_cache, out, T, n_heads, n_kv_heads, head_dim, max_seq_len);
  CUDA_CHECK_LAUNCH();
}

void qwen35_add_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                     __nv_bfloat16* y, std::size_t n, cudaStream_t stream) {
  CUDALM_PRECONDITION(n % 8 == 0, "qwen35_add_bf16 requires n % 8 == 0");
  CUDALM_PRECONDITION(
      ptr_aligned(a, 16) && ptr_aligned(b, 16) && ptr_aligned(y, 16),
      "qwen35_add_bf16 requires 16B-aligned base pointers");
  const int nvec = static_cast<int>(n / 8);
  const int grid = (nvec + kElemBlock - 1) / kElemBlock;
  qwen35_add_kernel<<<grid, kElemBlock, 0, stream>>>(
      reinterpret_cast<const Bf16x8*>(a), reinterpret_cast<const Bf16x8*>(b),
      reinterpret_cast<Bf16x8*>(y), nvec);
  CUDA_CHECK_LAUNCH();
}

void qwen35_silu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                          __nv_bfloat16* y, std::size_t n,
                          cudaStream_t stream) {
  CUDALM_PRECONDITION(n % 8 == 0, "qwen35_silu_mul_bf16 requires n % 8 == 0");
  CUDALM_PRECONDITION(
      ptr_aligned(gate, 16) && ptr_aligned(up, 16) && ptr_aligned(y, 16),
      "qwen35_silu_mul_bf16 requires 16B-aligned base pointers");
  const int nvec = static_cast<int>(n / 8);
  const int grid = (nvec + kElemBlock - 1) / kElemBlock;
  qwen35_silu_mul_kernel<<<grid, kElemBlock, 0, stream>>>(
      reinterpret_cast<const Bf16x8*>(gate),
      reinterpret_cast<const Bf16x8*>(up), reinterpret_cast<Bf16x8*>(y),
      nvec);
  CUDA_CHECK_LAUNCH();
}

void qwen35_gate_mul_bf16(const __nv_bfloat16* attn, const __nv_bfloat16* gate,
                          __nv_bfloat16* y, std::size_t n,
                          cudaStream_t stream) {
  CUDALM_PRECONDITION(n % 8 == 0, "qwen35_gate_mul_bf16 requires n % 8 == 0");
  CUDALM_PRECONDITION(
      ptr_aligned(attn, 16) && ptr_aligned(gate, 16) && ptr_aligned(y, 16),
      "qwen35_gate_mul_bf16 requires 16B-aligned base pointers");
  const int nvec = static_cast<int>(n / 8);
  const int grid = (nvec + kElemBlock - 1) / kElemBlock;
  qwen35_gate_mul_kernel<<<grid, kElemBlock, 0, stream>>>(
      reinterpret_cast<const Bf16x8*>(attn),
      reinterpret_cast<const Bf16x8*>(gate), reinterpret_cast<Bf16x8*>(y),
      nvec);
  CUDA_CHECK_LAUNCH();
}

}  // namespace kernels
}  // namespace cudalm
