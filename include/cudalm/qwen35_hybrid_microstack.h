// CUDALM — Qwen3.5 hybrid decoder micro-stack (v0.2, Phase D), the real
// checkpoint layers 0-3 (3x Gated DeltaNet + 1x full attention), single-GPU
// autoregressive decode (batch-1, one token). CUDALM-native wiring that REUSES
// the frozen Phase B (Qwen35FullAttentionLayer) and Phase C (Qwen35DeltaNetLayer)
// runtimes — no single-layer kernel is copied or re-derived.
//
// Layer order (pinned config hybrid schedule, docs/qwen35_architecture.md §9):
// layer 0/1/2 = Gated DeltaNet, layer 3 = full attention. Each decode step
// takes one device bf16 hidden [hidden_size] and runs layer 0 -> 1 -> 2 -> 3,
// feeding each layer's final output into the next; the micro-stack output is
// layer 3's final output (a device bf16 [hidden_size]).
//
// State ownership: each layer owns its OWN persistent state, never aliased or
// reused across layers:
//   DeltaNet 0/1/2 : conv_state bf16 [6144,3] + recurrent_state fp32 [16,128,128]
//   full-attention 3: the bf16 KV cache [n_kv][max_seq][head_dim] (K, V)
// reset_state() resets every layer independently. A decode step at position p
// therefore advances ALL layers' state by exactly one token.
//
// A micro-stack decode step runs four layers but is NOT a full-model token
// (the model has 24 layers); latency is reported as micro-stack steps, not
// tokens/s. See the Phase D benchmark + docs §16.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_deltanet.h"
#include "cudalm/qwen35_full_attention.h"
#include "cudalm/weight_loader_v2.h"

namespace cudalm {

class Qwen35HybridMicroStack {
 public:
  static constexpr int kNumLayers = 4;
  // Timing events required by forwardTimed(): one pair per layer (4) plus one
  // start/stop pair around the ENTIRE forward.
  static constexpr int kNumTimingEvents = 2 * kNumLayers + 2;

  // Move-only (owns the weight sets + the per-layer runtimes). Default
  // constructible so load() can populate it.
  Qwen35HybridMicroStack() = default;
  Qwen35HybridMicroStack(Qwen35HybridMicroStack&&) = default;
  Qwen35HybridMicroStack& operator=(Qwen35HybridMicroStack&&) = default;
  Qwen35HybridMicroStack(const Qwen35HybridMicroStack&) = delete;
  Qwen35HybridMicroStack& operator=(const Qwen35HybridMicroStack&) = delete;

  // Load the real checkpoint layers 0-3 from a parsed v2 file (which must
  // contain all four; each is validated) and construct the per-layer runtime,
  // dispatching Gated DeltaNet vs full attention by the config's hybrid
  // schedule. On failure `*out` is left unloaded (loaded() == false).
  static Status load(const WeightFileV2& file, cudaStream_t stream,
                     Qwen35HybridMicroStack* out);

  bool loaded() const { return loaded_; }
  const Qwen35Config& config() const { return cfg_; }
  bool is_delta(int i) const;
  bool is_att(int i) const;

  // Reset every layer's persistent state (DeltaNet: zero conv_state +
  // recurrent_state; full attention: zero the KV cache). Per-layer, no
  // cross-layer aliasing.
  void reset_state(cudaStream_t stream);

  // One decode step at 0-based `position`. `x_in` is a device bf16
  // [hidden_size] buffer (the token hidden state fed to layer 0). Runs layer
  // 0 -> 1 -> 2 -> 3, chaining each layer's final output into the next.
  // Precondition (abort on violation): 0 <= position < max_seq_len, loaded().
  void forward(int position, const __nv_bfloat16* x_in, cudaStream_t stream);

  // Like forward(), but records CUDA events and stores elapsed times in
  // microseconds: one pair per layer -> layer_us[kNumLayers], one pair around
  // the entire forward -> *whole_us. The whole_us pair is a CUDA-event
  // DEVICE-TIMELINE interval spanning the GPU work enqueued within it (all
  // four layers) plus any device idle while the host enqueues; it is NOT a
  // direct measurement of host CPU time (that is host_api_wall_us in the
  // benchmark). `events` must hold kNumTimingEvents caller-created events.
  // Synchronizes the stream so all pairs are complete on return.
  void forwardTimed(int position, const __nv_bfloat16* x_in,
                    cudaStream_t stream, cudaEvent_t* events,
                    float* layer_us, float* whole_us);

  // Micro-stack final output = layer 3's final output (device bf16 [H]).
  const __nv_bfloat16* final_output() const;

  // Per-layer accessors (the golden hard gate compares each layer's stages and
  // persistent state). Return nullptr if the layer is of the other type.
  Qwen35DeltaNetLayer* delta(int i) const;            // non-null iff is_delta(i)
  Qwen35FullAttentionLayer* attention(int i) const;   // non-null iff is_att(i)

 private:
  using LayerObj =
      std::variant<std::unique_ptr<Qwen35DeltaNetLayer>,
                   std::unique_ptr<Qwen35FullAttentionLayer>>;

  // Run layer `L` on `x` and return its final-output device pointer.
  const __nv_bfloat16* dispatch(LayerObj& L, int position,
                                const __nv_bfloat16* x, cudaStream_t stream) const;

  Qwen35Config cfg_{};
  // Weights declared BEFORE layers_ so (reverse) destruction tears down the
  // runtimes before the weight sets they reference.
  std::array<std::unique_ptr<Qwen35LayerWeights>, kNumLayers> weights_{};
  std::array<LayerObj, kNumLayers> layers_{};
  bool loaded_ = false;
};

}  // namespace cudalm
