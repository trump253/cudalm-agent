// CUDALM — Qwen3.5 full-attention layer wiring implementation.
// CUDALM-native. Math contract: docs/qwen35_architecture.md §7 (pinned
// official bf16 rounding boundaries; the kernels mirror them).
//
// The K cache stores ROPE'd K rows and the V cache stores raw V rows (the
// official cache-update convention); attention reads [0..position]
// inclusive, so the KV write happens before the attention call.
//
// forwardTimed() records one CUDA-event pair per stage (21 stages) for the
// layer latency breakdown, plus one pair (events[2*kNumStages] /
// [2*kNumStages+1]) around the entire forward's GPU work — the true
// whole-layer GPU time, which includes work no stage covers (the RoPE
// cos/sin table construct + H2D copy enqueued between the input and
// rmsnorm stages). forward() is the same pipeline with events == nullptr.

#include "cudalm/qwen35_full_attention.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "cudalm/cuda_check.h"
#include "cudalm/kernels/int4_gemv_bf16.h"
#include "cudalm/kernels/batch_decode.h"
#include "cudalm/kernels/paged_kv.h"
#include "cudalm/kernels/qwen35_kernels.h"

namespace cudalm {

using kernels::batch_int4_gemv_bf16;  // v0.6 Phase B batch GEMV (kernels namespace)

namespace {
std::size_t bf_bytes(std::size_t n) { return n * sizeof(__nv_bfloat16); }

// Record event #idx if timing is active. Stage s runs between events[2s]
// (recorded before) and events[2s+1] (after). The whole-layer pair uses
// idx = 2*kNumStages (start) and 2*kNumStages+1 (end).
inline void rec(cudaEvent_t* events, int idx, cudaStream_t stream) {
  if (events != nullptr) CUDA_CHECK(cudaEventRecord(events[idx], stream));
}

// Host-side partial-RoPE cos/sin table for decode position `position`
// (fp32, one value per rotary frequency j = 0..rotary_dim/2-1):
//   inv_freq[j] = 1 / (theta ^ (2j / rotary_dim))     (official formula)
//   cos[j] = cos(position * inv_freq[j]); sin[j] = sin(...)
// All fp32; the kernel casts each value to bf16 at the multiply boundary
// (the official `cos.to(dtype=x.dtype)`). Computed with the host libm so
// the table is bit-identical to the CPU golden's (same libm, same fp32
// ops).
void rope_table_host(int position, int rotary_dim, float theta,
                     float* cos_out, float* sin_out) {
  const int half = rotary_dim / 2;
  for (int j = 0; j < half; ++j) {
    const float inv_freq =
        1.0f / powf(theta, (2.0f * static_cast<float>(j)) /
                               static_cast<float>(rotary_dim));
    const float freq = static_cast<float>(position) * inv_freq;
    cos_out[j] = cosf(freq);
    sin_out[j] = sinf(freq);
  }
}
}  // namespace

const char* const* Qwen35FullAttentionLayer::stage_names() {
  static const char* const names[kNumStages] = {
      "input",         "rmsnorm1",   "q_gate_proj", "q_gate_split",
      "k_proj",        "v_proj",     "q_norm",      "k_norm",
      "rope_q",        "rope_k",     "kv_write",    "attention",
      "gate_mul",      "o_proj",     "residual1",   "rmsnorm2",
      "gate_proj",     "up_proj",    "silu_mul",    "down_proj",
      "final_output"};
  return names;
}

Qwen35FullAttentionLayer::Qwen35FullAttentionLayer(
    const Qwen35LayerWeights& weights, cudaStream_t stream)
    : w_(&weights), cfg_(weights.config()) {
  CUDALM_PRECONDITION(
      weights.is_full_attention(),
      "Qwen35FullAttentionLayer: weights must be a full-attention layer");
  kv_.reset(new Qwen35KvCache(cfg_, stream));

  const std::size_t H = static_cast<std::size_t>(cfg_.hidden_size);
  const std::size_t qo =
      static_cast<std::size_t>(cfg_.n_heads) * cfg_.head_dim;
  const std::size_t kvo =
      static_cast<std::size_t>(cfg_.n_kv_heads) * cfg_.head_dim;
  const std::size_t inter = static_cast<std::size_t>(cfg_.intermediate_size);

  input_.allocate(bf_bytes(H), stream);
  rms1_.allocate(bf_bytes(H), stream);
  q_gate_.allocate(bf_bytes(2 * qo), stream);
  q_.allocate(bf_bytes(qo), stream);
  att_gate_.allocate(bf_bytes(qo), stream);
  k_.allocate(bf_bytes(kvo), stream);
  v_.allocate(bf_bytes(kvo), stream);
  q_norm_.allocate(bf_bytes(qo), stream);
  k_norm_.allocate(bf_bytes(kvo), stream);
  rope_q_.allocate(bf_bytes(qo), stream);
  rope_k_.allocate(bf_bytes(kvo), stream);
  attn_raw_.allocate(bf_bytes(qo), stream);
  attn_gated_.allocate(bf_bytes(qo), stream);
  o_proj_.allocate(bf_bytes(H), stream);
  res1_.allocate(bf_bytes(H), stream);
  rms2_.allocate(bf_bytes(H), stream);
  mlp_gate_.allocate(bf_bytes(inter), stream);
  mlp_up_.allocate(bf_bytes(inter), stream);
  silu_mul_.allocate(bf_bytes(inter), stream);
  mlp_down_.allocate(bf_bytes(H), stream);
  final_.allocate(bf_bytes(H), stream);
  attn_scratch_.allocate(2 * static_cast<std::size_t>(cfg_.n_heads) *
                             cfg_.max_seq_len * sizeof(__nv_bfloat16),
                         stream);
  rope_tables_.allocate(2 * static_cast<std::size_t>(cfg_.rotary_dim()) *
                            sizeof(float),
                         stream);
}

void Qwen35FullAttentionLayer::forward(int position,
                                       const __nv_bfloat16* x_in,
                                       cudaStream_t stream) {
  forwardImpl(position, x_in, stream, nullptr);
}

void Qwen35FullAttentionLayer::forwardTimed(int position,
                                            const __nv_bfloat16* x_in,
                                            cudaStream_t stream,
                                            cudaEvent_t* events,
                                            float* stage_us,
                                            float* whole_block_us) {
  forwardImpl(position, x_in, stream, events);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < kNumStages; ++i) {
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, events[2 * i], events[2 * i + 1]));
    stage_us[i] = ms * 1000.0f;  // cudaEventElapsedTime is ms; API is us.
  }
  // Whole-layer pair: spans all GPU work enqueued by this forward,
  // including the inter-stage RoPE-table H2D copy the stage pairs miss.
  float wms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&wms, events[2 * kNumStages],
                                  events[2 * kNumStages + 1]));
  *whole_block_us = wms * 1000.0f;
}

void Qwen35FullAttentionLayer::forward_with_paged_state(
    int position, const __nv_bfloat16* x_in, const PagedStateRef& ps,
    cudaStream_t stream) {
  CUDALM_PRECONDITION(
      ps.k_pages != nullptr && ps.v_pages != nullptr &&
          ps.block_table != nullptr && ps.page_tokens >= 1 &&
          ps.page_stride >=
              static_cast<std::size_t>(cfg_.n_kv_heads) *
                  static_cast<std::size_t>(ps.page_tokens) *
                  static_cast<std::size_t>(cfg_.head_dim),
      "Qwen35FullAttentionLayer::forward_with_paged_state: incomplete "
      "PagedStateRef (null pointers, page_tokens < 1, or page_stride too "
      "small)");
  forwardImpl(position, x_in, stream, nullptr, &ps);
}

void Qwen35FullAttentionLayer::forwardImpl(int position,
                                           const __nv_bfloat16* x_in,
                                           cudaStream_t stream,
                                           cudaEvent_t* events,
                                           const PagedStateRef* paged) {
  CUDALM_PRECONDITION(
      position >= 0 && position < cfg_.max_seq_len,
      "Qwen35FullAttentionLayer::forward: position out of bounds "
      "[0, max_seq_len)");

  const int H = cfg_.hidden_size;
  const int hd = cfg_.head_dim;
  const int n_heads = cfg_.n_heads;
  const int n_kv = cfg_.n_kv_heads;
  const int inter = cfg_.intermediate_size;
  const int rd = cfg_.rotary_dim();

  // Whole-layer timing start: everything this forward enqueues on the
  // stream (all 21 stages + the inter-stage RoPE-table H2D copy) falls
  // between this event and the end event recorded at the bottom.
  rec(events, 2 * kNumStages, stream);

  // 0) stage.input
  rec(events, 0, stream);
  CUDA_CHECK(cudaMemcpyAsync(input_.data(), x_in, bf_bytes(H),
                             cudaMemcpyDeviceToDevice, stream));
  rec(events, 1, stream);

  // Partial-RoPE cos/sin table for this position (fp32, host libm).
  {
    std::vector<float> host(2 * rd);
    rope_table_host(position, rd, cfg_.rope_theta, host.data(),
                    host.data() + rd);
    rope_tables_.copy_from_host(host.data(), host.size() * sizeof(float),
                                stream);
  }
  const float* cos_t = rope_tables_.data<float>();
  const float* sin_t = cos_t + rd;

  // 1) zero-centered RMSNorm 1
  rec(events, 2, stream);
  kernels::qwen35_rmsnorm_zc_bf16(
      input_.data<__nv_bfloat16>(),
      w_->input_layernorm.data<__nv_bfloat16>(), rms1_.data<__nv_bfloat16>(),
      1, H, cfg_.eps, stream);
  rec(events, 3, stream);

  // 2) fused q_proj [q; gate] + split
  rec(events, 4, stream);
  int4_gemv_bf16(w_->q_proj.weight.data<std::uint8_t>(),
                 w_->q_proj.scale.data<__half>(), rms1_.data<__nv_bfloat16>(),
                 q_gate_.data<__nv_bfloat16>(), w_->q_proj.N, w_->q_proj.K,
                 stream);
  rec(events, 5, stream);
  rec(events, 6, stream);
  kernels::qwen35_split_q_gate_bf16(
      q_gate_.data<__nv_bfloat16>(), q_.data<__nv_bfloat16>(),
      att_gate_.data<__nv_bfloat16>(), n_heads, hd, stream);
  rec(events, 7, stream);

  // 3) k / v projections (W4A16)
  rec(events, 8, stream);
  int4_gemv_bf16(w_->k_proj.weight.data<std::uint8_t>(),
                 w_->k_proj.scale.data<__half>(), rms1_.data<__nv_bfloat16>(),
                 k_.data<__nv_bfloat16>(), w_->k_proj.N, w_->k_proj.K, stream);
  rec(events, 9, stream);
  rec(events, 10, stream);
  int4_gemv_bf16(w_->v_proj.weight.data<std::uint8_t>(),
                 w_->v_proj.scale.data<__half>(), rms1_.data<__nv_bfloat16>(),
                 v_.data<__nv_bfloat16>(), w_->v_proj.N, w_->v_proj.K, stream);
  rec(events, 11, stream);

  // 4) per-head zero-centered q/k norms (over head_dim)
  rec(events, 12, stream);
  kernels::qwen35_rmsnorm_zc_bf16(
      q_.data<__nv_bfloat16>(), w_->q_norm.data<__nv_bfloat16>(),
      q_norm_.data<__nv_bfloat16>(), n_heads, hd, cfg_.eps, stream);
  rec(events, 13, stream);
  rec(events, 14, stream);
  kernels::qwen35_rmsnorm_zc_bf16(
      k_.data<__nv_bfloat16>(), w_->k_norm.data<__nv_bfloat16>(),
      k_norm_.data<__nv_bfloat16>(), n_kv, hd, cfg_.eps, stream);
  rec(events, 15, stream);

  // 5) partial rotate-half RoPE (first rd dims of each head)
  rec(events, 16, stream);
  kernels::qwen35_partial_rope_bf16(
      q_norm_.data<__nv_bfloat16>(), rope_q_.data<__nv_bfloat16>(), n_heads,
      hd, rd, cos_t, sin_t, stream);
  rec(events, 17, stream);
  rec(events, 18, stream);
  kernels::qwen35_partial_rope_bf16(
      k_norm_.data<__nv_bfloat16>(), rope_k_.data<__nv_bfloat16>(), n_kv, hd,
      rd, cos_t, sin_t, stream);
  rec(events, 19, stream);

  // 6) KV write at `position` (K cache <- rope_k, V cache <- v). Paged:
  //    scatter into the external page array via the device block table
  //    (no host gather); legacy: the layer's own contiguous cache.
  rec(events, 20, stream);
  if (paged) {
    kernels::qwen35_paged_kv_write_bf16(
        rope_k_.data<__nv_bfloat16>(), v_.data<__nv_bfloat16>(),
        paged->k_pages, paged->v_pages, paged->block_table, position,
        paged->page_tokens, n_kv, hd, paged->page_stride, stream);
  } else {
    kv_->write(position, rope_k_.data<__nv_bfloat16>(),
               v_.data<__nv_bfloat16>(), stream);
  }
  rec(events, 21, stream);

  // 7) causal attention over [0..position] (paged: rows resolved through
  //    the device block table; legacy: contiguous cache).
  rec(events, 22, stream);
  if (paged) {
    kernels::qwen35_paged_attention_decode_bf16(
        rope_q_.data<__nv_bfloat16>(), paged->k_pages, paged->v_pages,
        paged->block_table, position, paged->page_tokens,
        attn_raw_.data<__nv_bfloat16>(), n_heads, n_kv, hd,
        paged->page_stride, attn_scratch_.data<__nv_bfloat16>(), stream);
  } else {
    kernels::qwen35_attention_decode_bf16(
        rope_q_.data<__nv_bfloat16>(), kv_->k(), kv_->v(), position,
        attn_raw_.data<__nv_bfloat16>(), n_heads, n_kv, hd, cfg_.max_seq_len,
        attn_scratch_.data<__nv_bfloat16>(), stream);
  }
  rec(events, 23, stream);

  // 8) attention gate: attn_raw * sigmoid(att_gate)
  rec(events, 24, stream);
  kernels::qwen35_gate_mul_bf16(
      attn_raw_.data<__nv_bfloat16>(), att_gate_.data<__nv_bfloat16>(),
      attn_gated_.data<__nv_bfloat16>(),
      static_cast<std::size_t>(n_heads) * hd, stream);
  rec(events, 25, stream);

  // 9) O projection + residual 1
  rec(events, 26, stream);
  int4_gemv_bf16(w_->o_proj.weight.data<std::uint8_t>(),
                 w_->o_proj.scale.data<__half>(),
                 attn_gated_.data<__nv_bfloat16>(),
                 o_proj_.data<__nv_bfloat16>(), w_->o_proj.N, w_->o_proj.K,
                 stream);
  rec(events, 27, stream);
  rec(events, 28, stream);
  kernels::qwen35_add_bf16(
      input_.data<__nv_bfloat16>(), o_proj_.data<__nv_bfloat16>(),
      res1_.data<__nv_bfloat16>(), H, stream);
  rec(events, 29, stream);

  // 10) zero-centered RMSNorm 2
  rec(events, 30, stream);
  kernels::qwen35_rmsnorm_zc_bf16(
      res1_.data<__nv_bfloat16>(),
      w_->post_attention_ln.data<__nv_bfloat16>(),
      rms2_.data<__nv_bfloat16>(), 1, H, cfg_.eps, stream);
  rec(events, 31, stream);

  // 11) gate / up (W4A16) + SiLU(gate) * up
  rec(events, 32, stream);
  int4_gemv_bf16(w_->gate_proj.weight.data<std::uint8_t>(),
                 w_->gate_proj.scale.data<__half>(),
                 rms2_.data<__nv_bfloat16>(), mlp_gate_.data<__nv_bfloat16>(),
                 w_->gate_proj.N, w_->gate_proj.K, stream);
  rec(events, 33, stream);
  rec(events, 34, stream);
  int4_gemv_bf16(w_->up_proj.weight.data<std::uint8_t>(),
                 w_->up_proj.scale.data<__half>(),
                 rms2_.data<__nv_bfloat16>(), mlp_up_.data<__nv_bfloat16>(),
                 w_->up_proj.N, w_->up_proj.K, stream);
  rec(events, 35, stream);
  rec(events, 36, stream);
  kernels::qwen35_silu_mul_bf16(
      mlp_gate_.data<__nv_bfloat16>(), mlp_up_.data<__nv_bfloat16>(),
      silu_mul_.data<__nv_bfloat16>(), inter, stream);
  rec(events, 37, stream);

  // 12) down (W4A16) + residual 2
  rec(events, 38, stream);
  int4_gemv_bf16(w_->down_proj.weight.data<std::uint8_t>(),
                 w_->down_proj.scale.data<__half>(),
                 silu_mul_.data<__nv_bfloat16>(),
                 mlp_down_.data<__nv_bfloat16>(), w_->down_proj.N,
                 w_->down_proj.K, stream);
  rec(events, 39, stream);
  rec(events, 40, stream);
  kernels::qwen35_add_bf16(
      res1_.data<__nv_bfloat16>(), mlp_down_.data<__nv_bfloat16>(),
      final_.data<__nv_bfloat16>(), H, stream);
  rec(events, 41, stream);

  // Whole-layer timing end (matches the start event above).
  rec(events, 2 * kNumStages + 1, stream);
}


// ---------------------------------------------------------------------------
// v0.6 Phase B: TRUE BATCHED decode over external paged state (B rows, one
// launch per stage; per row BIT-IDENTICAL to forward_with_paged_state() —
// see include/cudalm/kernels/batch_decode.h and docs §26).
// ---------------------------------------------------------------------------
void Qwen35FullAttentionLayer::grow_batch(int B, int t_max,
                                          cudaStream_t stream) {
  const bool bigger = (B > batch_cap_) || (t_max > batch_t_max_);
  if (!bigger) return;
  batch_cap_ = std::max(batch_cap_, B);
  batch_t_max_ = std::max(batch_t_max_, t_max);
  const std::size_t b = static_cast<std::size_t>(batch_cap_);
  const std::size_t t = static_cast<std::size_t>(batch_t_max_);
  const std::size_t H = static_cast<std::size_t>(cfg_.hidden_size);
  const std::size_t qo = static_cast<std::size_t>(cfg_.n_heads) * cfg_.head_dim;
  const std::size_t kvo = static_cast<std::size_t>(cfg_.n_kv_heads) * cfg_.head_dim;
  const std::size_t inter = static_cast<std::size_t>(cfg_.intermediate_size);
  const std::size_t rd = static_cast<std::size_t>(cfg_.rotary_dim());
  b_input_.allocate(bf_bytes(b * H), stream);
  b_rms1_.allocate(bf_bytes(b * H), stream);
  b_q_gate_.allocate(bf_bytes(b * 2 * qo), stream);
  b_q_.allocate(bf_bytes(b * qo), stream);
  b_att_gate_.allocate(bf_bytes(b * qo), stream);
  b_k_.allocate(bf_bytes(b * kvo), stream);
  b_v_.allocate(bf_bytes(b * kvo), stream);
  b_q_norm_.allocate(bf_bytes(b * qo), stream);
  b_k_norm_.allocate(bf_bytes(b * kvo), stream);
  b_rope_q_.allocate(bf_bytes(b * qo), stream);
  b_rope_k_.allocate(bf_bytes(b * kvo), stream);
  b_attn_raw_.allocate(bf_bytes(b * qo), stream);
  b_attn_gated_.allocate(bf_bytes(b * qo), stream);
  b_o_proj_.allocate(bf_bytes(b * H), stream);
  b_res1_.allocate(bf_bytes(b * H), stream);
  b_rms2_.allocate(bf_bytes(b * H), stream);
  b_mlp_gate_.allocate(bf_bytes(b * inter), stream);
  b_mlp_up_.allocate(bf_bytes(b * inter), stream);
  b_silu_mul_.allocate(bf_bytes(b * inter), stream);
  b_mlp_down_.allocate(bf_bytes(b * H), stream);
  b_final_.allocate(bf_bytes(b * H), stream);
  b_attn_scratch_.allocate(2 * b * static_cast<std::size_t>(cfg_.n_heads) * t *
                                  sizeof(__nv_bfloat16),
                           stream);
  b_rope_tables_.allocate(b * rd * sizeof(float), stream);
  b_positions_.allocate(b * sizeof(int), stream);
}

void Qwen35FullAttentionLayer::forward_batch_with_paged_state(
    int B, const int* positions_host, const __nv_bfloat16* x_in,
    const PagedStateRefBatch& ps, cudaStream_t stream) {
  CUDALM_PRECONDITION(
      B >= 1 && positions_host != nullptr && x_in != nullptr &&
          ps.k_pages != nullptr && ps.v_pages != nullptr &&
          ps.block_table != nullptr && ps.d_positions != nullptr &&
          ps.page_tokens >= 1 && ps.row_stride >= 1 &&
          ps.page_stride >=
              static_cast<std::size_t>(cfg_.n_kv_heads) *
                  static_cast<std::size_t>(ps.page_tokens) *
                  static_cast<std::size_t>(cfg_.head_dim),
      "forward_batch_with_paged_state: incomplete PagedStateRefBatch");

  const int H = cfg_.hidden_size;
  const int hd = cfg_.head_dim;
  const int n_heads = cfg_.n_heads;
  const int n_kv = cfg_.n_kv_heads;
  const int inter = cfg_.intermediate_size;
  const int rd = cfg_.rotary_dim();

  int t_max = 1;
  for (int b = 0; b < B; ++b) {
    const int p = positions_host[b];
    CUDALM_PRECONDITION(
        p >= 0 && p < cfg_.max_seq_len,
        "forward_batch_with_paged_state: position out of bounds");
    t_max = std::max(t_max, p + 1);
  }
  grow_batch(B, t_max, stream);

  // H2D the per-row positions (per-batch metadata, stream-ordered before
  // the paged kernels that read them).
  CUDA_CHECK(cudaMemcpyAsync(b_positions_.data(), positions_host,
                             static_cast<std::size_t>(B) * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  const int* d_positions = static_cast<int*>(b_positions_.data());

  // 0) stage.input: plain D2D copy of [B][H].
  CUDA_CHECK(cudaMemcpyAsync(b_input_.data(), x_in,
                             bf_bytes(static_cast<std::size_t>(B) * H),
                             cudaMemcpyDeviceToDevice, stream));

  // Partial-RoPE cos/sin tables, ONE per row's position (fp32, host libm —
  // the same construction as the frozen single path's rope_table_host).
  {
    std::vector<float> host(static_cast<std::size_t>(B) * rd);
    for (int b = 0; b < B; ++b) {
      rope_table_host(positions_host[b], rd, cfg_.rope_theta,
                      host.data() + static_cast<std::size_t>(b) * (rd / 2),
                      host.data() + static_cast<std::size_t>(B) * (rd / 2) +
                          static_cast<std::size_t>(b) * (rd / 2));
    }
    b_rope_tables_.copy_from_host(host.data(), host.size() * sizeof(float),
                                  stream);
  }
  const float* cos_t = b_rope_tables_.data<float>();
  const float* sin_t = cos_t + static_cast<std::size_t>(B) * (rd / 2);

  // 1) zero-centered RMSNorm 1 (frozen kernel, M = B rows).
  kernels::qwen35_rmsnorm_zc_bf16(
      b_input_.data<__nv_bfloat16>(),
      w_->input_layernorm.data<__nv_bfloat16>(),
      b_rms1_.data<__nv_bfloat16>(), B, H, cfg_.eps, stream);

  // 2) fused q_proj [q; gate] (W4A16, batched) + split (frozen kernel;
  //    the [B][n_heads][2*hd] layout splits per (b,h) head row).
  batch_int4_gemv_bf16(w_->q_proj.weight.data<std::uint8_t>(),
                       w_->q_proj.scale.data<__half>(),
                       b_rms1_.data<__nv_bfloat16>(),
                       b_q_gate_.data<__nv_bfloat16>(), w_->q_proj.N,
                       w_->q_proj.K, B, stream);
  kernels::qwen35_split_q_gate_bf16(
      b_q_gate_.data<__nv_bfloat16>(), b_q_.data<__nv_bfloat16>(),
      b_att_gate_.data<__nv_bfloat16>(), B * n_heads, hd, stream);

  // 3) k / v projections (W4A16, batched).
  batch_int4_gemv_bf16(w_->k_proj.weight.data<std::uint8_t>(),
                       w_->k_proj.scale.data<__half>(),
                       b_rms1_.data<__nv_bfloat16>(),
                       b_k_.data<__nv_bfloat16>(), w_->k_proj.N, w_->k_proj.K,
                       B, stream);
  batch_int4_gemv_bf16(w_->v_proj.weight.data<std::uint8_t>(),
                       w_->v_proj.scale.data<__half>(),
                       b_rms1_.data<__nv_bfloat16>(),
                       b_v_.data<__nv_bfloat16>(), w_->v_proj.N, w_->v_proj.K,
                       B, stream);

  // 4) per-head zero-centered q/k norms (frozen kernel, M = B*n_heads rows).
  kernels::qwen35_rmsnorm_zc_bf16(
      b_q_.data<__nv_bfloat16>(), w_->q_norm.data<__nv_bfloat16>(),
      b_q_norm_.data<__nv_bfloat16>(), B * n_heads, hd, cfg_.eps, stream);
  kernels::qwen35_rmsnorm_zc_bf16(
      b_k_.data<__nv_bfloat16>(), w_->k_norm.data<__nv_bfloat16>(),
      b_k_norm_.data<__nv_bfloat16>(), B * n_kv, hd, cfg_.eps, stream);

  // 5) partial rotate-half RoPE with HETEROGENEOUS per-row positions.
  kernels::batch_partial_rope_bf16(
      b_q_norm_.data<__nv_bfloat16>(), b_rope_q_.data<__nv_bfloat16>(), B,
      n_heads, hd, rd, cos_t, sin_t, stream);
  kernels::batch_partial_rope_bf16(
      b_k_norm_.data<__nv_bfloat16>(), b_rope_k_.data<__nv_bfloat16>(), B,
      n_kv, hd, rd, cos_t, sin_t, stream);

  // 6) paged KV write at each row's position (row b through its OWN block
  //    table row; K cache <- rope_k, V cache <- v).
  kernels::batch_paged_kv_write_bf16(
      b_rope_k_.data<__nv_bfloat16>(), b_v_.data<__nv_bfloat16>(),
      ps.k_pages, ps.v_pages, ps.block_table, d_positions, ps.row_stride,
      ps.page_tokens, n_kv, hd, ps.page_stride, B, stream);

  // 7) causal attention over each row's OWN [0..position[b]] (paged, batched).
  kernels::batch_paged_attention_decode_bf16(
      b_rope_q_.data<__nv_bfloat16>(), ps.k_pages, ps.v_pages,
      ps.block_table, d_positions, ps.row_stride, ps.page_tokens,
      b_attn_raw_.data<__nv_bfloat16>(), n_heads, n_kv, hd, ps.page_stride,
      t_max, B, b_attn_scratch_.data<__nv_bfloat16>(), stream);

  // 8) attention gate: attn_raw * sigmoid(att_gate) (frozen kernel, flat).
  kernels::qwen35_gate_mul_bf16(
      b_attn_raw_.data<__nv_bfloat16>(), b_att_gate_.data<__nv_bfloat16>(),
      b_attn_gated_.data<__nv_bfloat16>(),
      static_cast<std::size_t>(B) * n_heads * hd, stream);

  // 9) O projection (W4A16, batched) + residual 1 (frozen add, flat).
  batch_int4_gemv_bf16(w_->o_proj.weight.data<std::uint8_t>(),
                       w_->o_proj.scale.data<__half>(),
                       b_attn_gated_.data<__nv_bfloat16>(),
                       b_o_proj_.data<__nv_bfloat16>(), w_->o_proj.N,
                       w_->o_proj.K, B, stream);
  kernels::qwen35_add_bf16(
      b_input_.data<__nv_bfloat16>(), b_o_proj_.data<__nv_bfloat16>(),
      b_res1_.data<__nv_bfloat16>(), static_cast<std::size_t>(B) * H,
      stream);

  // 10) zero-centered RMSNorm 2 (M = B rows).
  kernels::qwen35_rmsnorm_zc_bf16(
      b_res1_.data<__nv_bfloat16>(),
      w_->post_attention_ln.data<__nv_bfloat16>(),
      b_rms2_.data<__nv_bfloat16>(), B, H, cfg_.eps, stream);

  // 11) gate / up (W4A16, batched) + silu_mul (frozen, flat).
  batch_int4_gemv_bf16(w_->gate_proj.weight.data<std::uint8_t>(),
                       w_->gate_proj.scale.data<__half>(),
                       b_rms2_.data<__nv_bfloat16>(),
                       b_mlp_gate_.data<__nv_bfloat16>(), w_->gate_proj.N,
                       w_->gate_proj.K, B, stream);
  batch_int4_gemv_bf16(w_->up_proj.weight.data<std::uint8_t>(),
                       w_->up_proj.scale.data<__half>(),
                       b_rms2_.data<__nv_bfloat16>(),
                       b_mlp_up_.data<__nv_bfloat16>(), w_->up_proj.N,
                       w_->up_proj.K, B, stream);
  kernels::qwen35_silu_mul_bf16(
      b_mlp_gate_.data<__nv_bfloat16>(), b_mlp_up_.data<__nv_bfloat16>(),
      b_silu_mul_.data<__nv_bfloat16>(),
      static_cast<std::size_t>(B) * inter, stream);

  // 12) down (W4A16, batched) + residual 2.
  batch_int4_gemv_bf16(w_->down_proj.weight.data<std::uint8_t>(),
                       w_->down_proj.scale.data<__half>(),
                       b_silu_mul_.data<__nv_bfloat16>(),
                       b_mlp_down_.data<__nv_bfloat16>(), w_->down_proj.N,
                       w_->down_proj.K, B, stream);
  kernels::qwen35_add_bf16(
      b_res1_.data<__nv_bfloat16>(), b_mlp_down_.data<__nv_bfloat16>(),
      b_final_.data<__nv_bfloat16>(), static_cast<std::size_t>(B) * H,
      stream);
}

}  // namespace cudalm
