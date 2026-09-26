// CUDALM — Qwen3.5 full-attention decoder layer (v0.2, Phase B), single-GPU
// autoregressive decode step. CUDALM-native wiring of the bf16 Qwen3.5
// kernel family + the W4A16 GEMV (bf16 specialization).
//
// Pipeline (docs/qwen35_architecture.md §7, one decode step at position p,
// batch-1):
//
//   x -> zero-centered RMSNorm(input_layernorm)
//       -> q_gate = q_proj(rms1)                      (W4A16, [4096])
//       -> split: q [2048] / att_gate [2048] (per-head fused layout)
//       -> k = k_proj(rms1), v = v_proj(rms1)         (W4A16, [512] each)
//       -> q_n = per-head zero-centered RMSNorm(q, q_norm)   (over 256)
//       -> k_n = per-head zero-centered RMSNorm(k, k_norm)   (over 256)
//       -> rope_q = partial rotate-half RoPE(q_n, p)  (rotary dim 64)
//       -> rope_k = partial rotate-half RoPE(k_n, p)
//       -> KV write at p (K <- rope_k, V <- v)
//       -> attn_raw = causal GQA attention(rope_q, KV[0..p])
//       -> attn_gated = attn_raw * sigmoid(att_gate)   (gate is bf16)
//       -> o = o_proj(attn_gated)                      (W4A16)
//       -> res1 = x + o                                (bf16 add)
//       -> rms2 = zero-centered RMSNorm(res1, post_attention_layernorm)
//       -> g = gate_proj(rms2), u = up_proj(rms2)      (W4A16)
//       -> sgmu = silu(g) * u                           (SwiGLU, bf16)
//       -> d = down_proj(sgmu)                          (W4A16)
//       -> y = res1 + d
//
// All stage buffers are owned device memory (bf16), valid from forward()
// until the next forward() or destruction — the golden test compares them
// stage-by-stage against a CUDLMG02 golden file (runtime-vs-quantized-
// reference HARD GATE; tolerance in stage_compare.h).
//
// The layer is driven by a Phase-A Qwen35LayerWeights (v2 .cudalm layer
// tensor set). This class does NOT modify any v0.1.1 type.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cudalm/device_buffer.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_kv_cache.h"
#include "cudalm/weight_loader_v2.h"

namespace cudalm {

class Qwen35FullAttentionLayer {
 public:
  // `weights` must outlive the layer (non-owning) and must have been loaded
  // for a full-attention layer index (is_full_attention() == true).
  explicit Qwen35FullAttentionLayer(const Qwen35LayerWeights& weights,
                                    cudaStream_t stream = 0);

  // Number of timed pipeline stages (latency breakdown; see stage_names).
  static constexpr int kNumStages = 21;

  // Events required by forwardTimed(): 2 per stage plus one start/stop pair
  // around the ENTIRE forward (whole-layer timing). That pair measures a
  // CUDA-event DEVICE-TIMELINE interval: it spans every unit of GPU work
  // enqueued between the two records (including the inter-stage RoPE-table
  // H2D copy) and also any device idle while the host is enqueuing work
  // (e.g. the host-side RoPE-table build between stages). It is NOT a direct
  // measurement of host CPU work — that lives in host_api_wall_us.
  static constexpr int kNumTimingEvents = 2 * kNumStages + 2;

  // One decode step at 0-based `position`. `x_in` is a device bf16
  // [hidden_size] buffer (the current token's hidden state).
  //
  // Precondition (host-checked, abort on violation):
  // 0 <= position < max_seq_len.
  void forward(int position, const __nv_bfloat16* x_in,
               cudaStream_t stream);

  // Same pipeline as forward(), but records CUDA events and stores elapsed
  // times in microseconds:
  //   * one event pair per stage -> stage_us[kNumStages]
  //   * one event pair around the entire forward -> *whole_block_us: a
  //     CUDA-event device-timeline interval covering the whole forward. It
  //     includes the GPU work enqueued within it (notably the RoPE-table
  //     H2D copy between the input and rmsnorm stages that no stage pair
  //     covers — so summing stage_us[] alone UNDERCOUNTS the layer) and may
  //     include device idle while the host enqueues work. It is not a
  //     direct measurement of host CPU time (see host_api_wall_us).
  // `events` must hold kNumTimingEvents caller-created events.
  // Synchronizes the stream so all event pairs are complete on return.
  void forwardTimed(int position, const __nv_bfloat16* x_in,
                    cudaStream_t stream, cudaEvent_t* events,
                    float* stage_us, float* whole_block_us);

  // v0.5 Phase B: the SAME 21-stage pipeline as forward(), but only the two
  // cache-touching stages differ:
  //   * stage 6 (KV write)    -> paged write into the external page array
  //   * stage 7 (attention)   -> paged causal decode attention
  // through the DEVICE block table. Everything else (RMSNorm, projections,
  // RoPE, gate, MLP, residual) REUSES the exact frozen implementation.
  // The layer's OWN contiguous Qwen35KvCache (kv_) is NOT touched — the
  // legacy forward() keeps using it.
  //
  // PagedStateRef describes one full-attention layer's slice of a Phase-A
  // Qwen35KvPagePool:
  //   k_pages/v_pages: base of this layer's page storage in the pool:
  //     pool.k_page_mut(full_layer_ordinal, 0) / v_page_mut(...) (non-const:
  //     the paged KV WRITE stage writes into the pages)
  //   page_stride    : bf16 elements between consecutive pages of this
  //     layer (= pool.page_stride_elems() for the Phase-A pool layout)
  //   block_table    : DEVICE int32 [num_blocks], logical block -> physical
  //     page id (entries beyond position/page_tokens may be stale — the
  //     kernels never read them)
  //   page_tokens    : the pool's page size (must match the pool)
  // Precondition (host-checked, abort on violation): 0 <= position
  // < max_seq_len, page_tokens >= 1,
  // page_stride >= n_kv_heads*page_tokens*head_dim, non-null pointers.
  struct PagedStateRef {
    __nv_bfloat16* k_pages = nullptr;
    __nv_bfloat16* v_pages = nullptr;
    const int* block_table = nullptr;
    int page_tokens = 0;
    std::size_t page_stride = 0;
  };
  void forward_with_paged_state(int position, const __nv_bfloat16* x_in,
                                const PagedStateRef& ps, cudaStream_t stream);

  // Names of the 21 timed stages, in order (shared with the benchmark).
  static const char* const* stage_names();

  // ---- Stage buffers (device bf16; valid after forward) -----------------
  const __nv_bfloat16* stage_input() const { return input_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_rmsnorm1() const { return rms1_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_q_gate() const { return q_gate_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_q() const { return q_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_att_gate() const { return att_gate_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_k() const { return k_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_v() const { return v_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_q_norm() const { return q_norm_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_k_norm() const { return k_norm_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_rope_q() const { return rope_q_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_rope_k() const { return rope_k_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_attention_raw() const {
    return attn_raw_.data<__nv_bfloat16>();
  }
  const __nv_bfloat16* stage_attention_gated() const {
    return attn_gated_.data<__nv_bfloat16>();
  }
  const __nv_bfloat16* stage_o_proj() const { return o_proj_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_residual1() const { return res1_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_rmsnorm2() const { return rms2_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_mlp_gate() const { return mlp_gate_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_mlp_up() const { return mlp_up_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_silu_mul() const { return silu_mul_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_mlp_down() const { return mlp_down_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_final_output() const {
    return final_.data<__nv_bfloat16>();
  }

  Qwen35KvCache& kv_cache() { return *kv_; }
  const Qwen35KvCache& kv_cache() const { return *kv_; }
  const Qwen35Config& config() const { return cfg_; }

 private:
  // Shared pipeline (frozen v0.2 math, unchanged): when `paged` is null the
  // KV write + attention use the layer's OWN contiguous cache (legacy);
  // otherwise they use the external paged state (v0.5 Phase B).
  void forwardImpl(int position, const __nv_bfloat16* x_in,
                   cudaStream_t stream, cudaEvent_t* events,
                   const PagedStateRef* paged = nullptr);

  const Qwen35LayerWeights* w_ = nullptr;
  Qwen35Config cfg_{};
  std::unique_ptr<Qwen35KvCache> kv_;  // move-only (no default ctor)

  // Stage storage, in pipeline order (bf16).
  DeviceBuffer input_;     // [H]
  DeviceBuffer rms1_;      // [H]
  DeviceBuffer q_gate_;    // [n_heads*head_dim*2] fused q_proj output
  DeviceBuffer q_;         // [n_heads*head_dim]
  DeviceBuffer att_gate_;  // [n_heads*head_dim]
  DeviceBuffer k_;         // [n_kv*head_dim]
  DeviceBuffer v_;         // [n_kv*head_dim]
  DeviceBuffer q_norm_;    // [n_heads*head_dim]
  DeviceBuffer k_norm_;    // [n_kv*head_dim]
  DeviceBuffer rope_q_;    // [n_heads*head_dim]
  DeviceBuffer rope_k_;    // [n_kv*head_dim]
  DeviceBuffer attn_raw_;  // [n_heads*head_dim]
  DeviceBuffer attn_gated_;// [n_heads*head_dim]
  DeviceBuffer o_proj_;    // [H]
  DeviceBuffer res1_;      // [H]
  DeviceBuffer rms2_;      // [H]
  DeviceBuffer mlp_gate_;  // [inter]
  DeviceBuffer mlp_up_;    // [inter]
  DeviceBuffer silu_mul_;  // [inter]
  DeviceBuffer mlp_down_;  // [H]
  DeviceBuffer final_;     // [H]

  DeviceBuffer attn_scratch_;  // bf16 [2 * n_heads * max_seq_len]
  DeviceBuffer rope_tables_;   // fp32 [2 * (rotary_dim/2)] (cos_t | sin_t)
};

}  // namespace cudalm
