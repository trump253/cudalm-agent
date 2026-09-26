// CUDALM — Qwen3.5 paged-KV decode kernels implementation (v0.5 Phase B).
//
// Row addressing goes through the DEVICE block table on every access (no
// host gather, no contiguous materialization). The accumulation order is
// IDENTICAL to the frozen contiguous pipeline (src/kernels/qwen35_kernels.cu)
// so, for the same logical rows, the output is bit-identical:
//   * scores: acc over d = 0..head_dim-1 for each (h, t);
//   * softmax: three global passes (max / exp+sum / normalize) over
//     [0..T) in the same strided order (mirrors the frozen kernel 1:1 —
//     O(num_warps) shared memory, not O(T), so T up to max_seq_len fits
//     sm_75);
//   * PV: acc over t = 0..T-1 for each (h, d).
// The bf16 rounding boundaries are the official ones (docs §7): one RNE on
// the fp32 dot, one RNE on the bf16*scale product, fp32 softmax, one RNE on
// the probs, one RNE on the PV dot.

#include "cudalm/kernels/paged_kv.h"

#include <cfloat>
#include <cmath>

#include "cudalm/cuda_check.h"

namespace cudalm {
namespace kernels {

namespace {

// Resolve the page+offset of logical token `t` from the device block table.
__device__ inline void paged_row(int t, int page_tokens,
                                 const int* __restrict__ block_table,
                                 int* page, int* off) {
  *page = block_table[t / page_tokens];
  *off = t - (t / page_tokens) * page_tokens;
}

// Element offset of row (kv head n, logical token page `page` at offset
// `off`) from the layer's page-storage base: pages of one ordinal are
// `page_stride` bf16 elements apart. In the Phase-A pool layout that is
// exactly n_kv*pt*hd (pages of one ordinal adjacent; ordinals are
// capacity_pages*n_kv*pt*hd apart and selected via the k_page(ord, 0)
// base) — the ordinal stride is NOT the page_stride.
__device__ inline std::size_t paged_row_offset(int page, int off, int n,
                                               int page_tokens, int head_dim,
                                               std::size_t page_stride) {
  return static_cast<std::size_t>(page) * page_stride +
         (static_cast<std::size_t>(n) * page_tokens + off) * head_dim;
}

// ---------------------------------------------------------------------------
// Paged KV write (plain copies — bit-exact round trip).
// ---------------------------------------------------------------------------
__global__ void qwen35_paged_kv_write_kernel(
    const __nv_bfloat16* __restrict__ k_src,
    const __nv_bfloat16* __restrict__ v_src,
    __nv_bfloat16* __restrict__ k_pages,
    __nv_bfloat16* __restrict__ v_pages,
    const int* __restrict__ block_table, int position, int page_tokens,
    int n_kv_heads, int head_dim, std::size_t page_stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_kv_heads * head_dim;
  if (idx >= total) return;  // the frozen kernel never runs a partial block
                             // (its total is always a multiple of 128); this
                             // one must guard its ceiling-rounded grid
  const int n = idx / head_dim;
  const int d = idx - n * head_dim;
  int page, off;
  paged_row(position, page_tokens, block_table, &page, &off);
  const std::size_t dst =
      paged_row_offset(page, off, n, page_tokens, head_dim, page_stride) + d;
  k_pages[dst] = k_src[idx];
  v_pages[dst] = v_src[idx];
}

// ---------------------------------------------------------------------------
// Paged decode attention (3-kernel pipeline).
// scratch: bf16 [scores2 | probs], each n_heads * T (T = position + 1).
// ---------------------------------------------------------------------------
constexpr int kPagedScoresBfBlock = 128;
constexpr int kPagedSoftmaxBfBlock = 256;
constexpr int kPagedPvBfBlock = 128;

// 1) s1 = bf16(fp32 dot); s2 = bf16(f32(s1) * scale). SAME accumulation
//    order as the frozen contiguous scores kernel (d ascending).
__global__ void qwen35_paged_attention_scores_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pages,
    const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ scores2, int T, int n_heads, int n_kv_heads,
    int page_tokens, int head_dim, std::size_t page_stride, float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_heads * T;
  if (idx >= total) return;
  const int h = idx / T;
  const int t = idx - h * T;
  const int kh = h * n_kv_heads / n_heads;

  const __nv_bfloat16* qh = q + static_cast<std::size_t>(h) * head_dim;
  int page, off;
  paged_row(t, page_tokens, block_table, &page, &off);
  const __nv_bfloat16* kr = k_pages +
      paged_row_offset(page, off, kh, page_tokens, head_dim, page_stride);

  float acc = 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    acc += __bfloat162float(qh[d]) * __bfloat162float(kr[d]);
  }
  const __nv_bfloat16 s1 = __float2bfloat16_rn(acc);
  scores2[idx] = __float2bfloat16_rn(__bfloat162float(s1) * scale);
}

__device__ inline float paged_block_max_reduce(float v, float* smem, int tid) {
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

__device__ inline float paged_block_sum_reduce(float v, float* smem, int tid) {
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

// 2) softmax over row [0..T) — MIRRORS the frozen contiguous softmax kernel
//    1:1 (same strided order, same O(num_warps) shared-memory three-pass
//    structure, same expf usage) so the bf16 probs are bit-identical.
__global__ void qwen35_paged_attention_softmax_kernel(
    const __nv_bfloat16* __restrict__ scores2,
    __nv_bfloat16* __restrict__ probs, int T) {
  const int h = blockIdx.x;
  const __nv_bfloat16* srow =
      scores2 + static_cast<std::size_t>(h) * static_cast<std::size_t>(T);
  __nv_bfloat16* prow =
      probs + static_cast<std::size_t>(h) * static_cast<std::size_t>(T);
  __shared__ float sh_red[32];
  const int tid = threadIdx.x;

  float m = -FLT_MAX;
  for (int t = tid; t < T; t += blockDim.x)
    m = fmaxf(m, __bfloat162float(srow[t]));
  m = paged_block_max_reduce(m, sh_red, tid);

  float acc = 0.0f;
  for (int t = tid; t < T; t += blockDim.x)
    acc += expf(__bfloat162float(srow[t]) - m);
  const float sum = paged_block_sum_reduce(acc, sh_red, tid);

  for (int t = tid; t < T; t += blockDim.x)
    prow[t] = __float2bfloat16_rn(expf(__bfloat162float(srow[t]) - m) / sum);
}

// 3) out[h,d] = bf16(Sum_t f32(probs[t]) * f32(V_row(h,t,d))). SAME
//    accumulation order as the frozen contiguous PV kernel (t ascending).
__global__ void qwen35_paged_attention_pv_kernel(
    const __nv_bfloat16* __restrict__ probs,
    const __nv_bfloat16* __restrict__ v_pages,
    const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ out, int T, int n_heads, int n_kv_heads,
    int page_tokens, int head_dim, std::size_t page_stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_heads * head_dim;
  if (idx >= total) return;
  const int h = idx / head_dim;
  const int d = idx - h * head_dim;
  const int kh = h * n_kv_heads / n_heads;

  const __nv_bfloat16* ph = probs + static_cast<std::size_t>(h) * T;

  float acc = 0.0f;
  for (int t = 0; t < T; ++t) {
    int page, off;
    paged_row(t, page_tokens, block_table, &page, &off);
    const __nv_bfloat16* vrow = v_pages +
        paged_row_offset(page, off, kh, page_tokens, head_dim, page_stride);
    acc += __bfloat162float(ph[t]) * __bfloat162float(vrow[d]);
  }
  // The frozen contiguous PV kernel has NO bounds guard (its grid is
  // n_heads*head_dim / 128, exact at the real 0.8B runtime shape
  // 8 heads * 256 head_dim); this ceiling-rounded grid must guard the
  // partial tail block.
  if (idx < total) out[idx] = __float2bfloat16_rn(acc);
}

}  // namespace

void qwen35_paged_kv_write_bf16(const __nv_bfloat16* k_src,
                                const __nv_bfloat16* v_src,
                                __nv_bfloat16* k_pages,
                                __nv_bfloat16* v_pages,
                                const int* block_table, int position,
                                int page_tokens, int n_kv_heads, int head_dim,
                                std::size_t page_stride, cudaStream_t stream) {
  CUDALM_PRECONDITION(
      n_kv_heads >= 1 && head_dim >= 1 && page_tokens >= 1 && position >= 0 &&
          page_stride >=
              static_cast<std::size_t>(n_kv_heads) * page_tokens * head_dim,
      "qwen35_paged_kv_write_bf16: requires n_kv_heads>=1, head_dim>=1, "
      "page_tokens>=1, position>=0, page_stride>=n_kv*pt*hd");
  const int total = n_kv_heads * head_dim;
  const int grid = (total + 127) / 128;
  qwen35_paged_kv_write_kernel<<<grid, 128, 0, stream>>>(
      k_src, v_src, k_pages, v_pages, block_table, position, page_tokens,
      n_kv_heads, head_dim, page_stride);
  CUDA_CHECK_LAUNCH();
}

void qwen35_paged_attention_decode_bf16(const __nv_bfloat16* q,
                                        const __nv_bfloat16* k_pages,
                                        const __nv_bfloat16* v_pages,
                                        const int* block_table, int position,
                                        int page_tokens, __nv_bfloat16* out,
                                        int n_heads, int n_kv_heads,
                                        int head_dim, std::size_t page_stride,
                                        __nv_bfloat16* scratch,
                                        cudaStream_t stream) {
  CUDALM_PRECONDITION(
      n_heads % n_kv_heads == 0 && n_heads >= 1 && n_kv_heads >= 1 &&
          head_dim >= 1 && page_tokens >= 1 && position >= 0 &&
          page_stride >=
              static_cast<std::size_t>(n_kv_heads) * page_tokens * head_dim,
      "qwen35_paged_attention_decode_bf16: requires n_heads % n_kv_heads == "
      "0, all dims >= 1, page_stride>=n_kv*pt*hd, position >= 0");
  const int T = position + 1;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  __nv_bfloat16* scores2 = scratch;
  __nv_bfloat16* probs = scratch + static_cast<std::size_t>(n_heads) * T;

  {
    const int total = n_heads * T;
    const int grid = (total + kPagedScoresBfBlock - 1) / kPagedScoresBfBlock;
    qwen35_paged_attention_scores_kernel
        <<<grid, kPagedScoresBfBlock, 0, stream>>>(
            q, k_pages, block_table, scores2, T, n_heads, n_kv_heads,
            page_tokens, head_dim, page_stride, scale);
    CUDA_CHECK_LAUNCH();
  }
  qwen35_paged_attention_softmax_kernel
      <<<n_heads, kPagedSoftmaxBfBlock, 0, stream>>>(scores2, probs, T);
  CUDA_CHECK_LAUNCH();
  {
    const int total = n_heads * head_dim;
    const int grid = (total + kPagedPvBfBlock - 1) / kPagedPvBfBlock;
    qwen35_paged_attention_pv_kernel<<<grid, kPagedPvBfBlock, 0, stream>>>(
        probs, v_pages, block_table, out, T, n_heads, n_kv_heads, page_tokens,
        head_dim, page_stride);
    CUDA_CHECK_LAUNCH();
  }
}

}  // namespace kernels
}  // namespace cudalm
