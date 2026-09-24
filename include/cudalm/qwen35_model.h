// CUDALM — Qwen3.5 full model runtime (v0.3: Phase A skeleton + Phase B full
// single-token forward).
//
// Owns the COMPLETE 24-layer Qwen3.5-0.8B text model: the token embedding,
// every decoder layer (dispatched Gated DeltaNet vs full attention by the
// config's hybrid schedule — full attention at layers 3,7,11,15,19,23), the
// final RMSNorm, and the (tied) LM head. CUDALM-native wiring that REUSES the
// frozen v0.2 single-layer runtimes (Qwen35FullAttentionLayer /
// Qwen35DeltaNetLayer) — no single-layer kernel is copied or re-derived.
//
// v0.3 scope:
//   * Phase A = structure + weight ownership + state lifecycle: load() uploads
//     the weights, reset_state() resets every layer's persistent state, and
//     the accessors expose the owned tensors.
//   * Phase B = the COMPLETE single-token forward (forward_token): embedding
//     -> 24 layers -> final RMSNorm -> tied LM head -> logits [vocab] (docs
//     §18). It reuses the v0.2 layer runtimes, threads the runtime's OWN
//     persistent state (a sequential forward threads state), and exposes the
//     embedding output / each layer final / final-norm output / full logits.
//     No generation / sampling / Paged KV / batching (out of scope).
//
// Weight tying (pinned 0.8B, config tie_word_embeddings=true): the LM head has
// NO separate tensor; it ALIASES the embedding (lm_head weight ==
// embed_tokens.weight, used transposed for logits = hidden @ W^T). The
// embedding buffer is therefore the single owner of both.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cudalm/device_buffer.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_deltanet.h"
#include "cudalm/qwen35_full_attention.h"
#include "cudalm/tensor.h"
#include "cudalm/weight_loader_v2.h"

namespace cudalm {

class Qwen35Model {
 public:
  // Move-only (owns the embedding/final-norm buffers + the per-layer weight
  // sets + the per-layer runtimes). Default constructible so load() can
  // populate it.
  Qwen35Model() = default;
  Qwen35Model(Qwen35Model&&) = default;
  Qwen35Model& operator=(Qwen35Model&&) = default;
  Qwen35Model(const Qwen35Model&) = delete;
  Qwen35Model& operator=(const Qwen35Model&) = delete;

  // Load the FULL model (embedding + all num_hidden_layers decoder layers +
  // final norm) from a parsed v2 file. validate_full_model() runs first
  // (embedding + norm + every layer + tie metadata); on failure `*out` is left
  // unloaded (loaded() == false). Each layer's runtime is constructed from its
  // own weight set, dispatched Gated DeltaNet vs full attention by the config.
  static Status load(const WeightFileV2& file, cudaStream_t stream,
                     Qwen35Model* out);

  bool loaded() const { return loaded_; }
  const Qwen35Config& config() const { return cfg_; }
  int num_layers() const { return cfg_.num_hidden_layers; }
  bool tie_word_embeddings() const { return tie_; }
  bool is_full_attention(int i) const { return cfg_.is_full_attention(i); }
  bool is_linear_attention(int i) const { return cfg_.is_linear_attention(i); }

  // ---- Model-level owned tensors (device) --------------------------------
  // Token embedding [vocab_size, hidden_size] bf16 (row-major; row = token).
  const TensorView& embedding() const { return embed_; }
  // Final RMSNorm weight [hidden_size] bf16 (zero-centered (1+w), docs §6).
  const TensorView& final_norm() const { return norm_; }
  // LM head: when tied (pinned 0.8B) this ALIASES the embedding — the SAME
  // device buffer, interpreted as a [vocab_size, hidden_size] weight used
  // transposed (logits = hidden @ lm_head^T). No separate allocation.
  const TensorView& lm_head() const { return lm_head_; }

  // ---- Per-layer accessors -------------------------------------------------
  const Qwen35LayerWeights* layer_weights(int i) const;  // null if i invalid
  Qwen35DeltaNetLayer* delta(int i) const;               // non-null iff linear
  Qwen35FullAttentionLayer* attention(int i) const;      // non-null iff full

  // Reset EVERY decoder layer's persistent state (DeltaNet: zero conv_state +
  // recurrent_state; full attention: zero the KV cache). Per-layer, no
  // cross-layer aliasing. The embedding/final-norm/LM head are weights (not
  // state) and are unaffected.
  void reset_state(cudaStream_t stream);

  // ---- Full single-token forward (v0.3 Phase B; docs §18) ----------------
  // Run the COMPLETE model on ONE token at `position`:
  //   embedding[token_id] -> layer 0 -> ... -> layer 23 -> final RMSNorm
  //   -> tied LM head (embedding^T) -> logits [vocab_size].
  // Uses the layers' OWN persistent state (DeltaNet conv/recurrent, full-
  // attention KV) — it is updated in place, so a sequential forward
  // (p0, p1, p2) threads each layer's state across the steps (call
  // reset_state() first for a fresh-state run). Precondition (host-checked,
  // abort on violation): loaded, 0 <= token_id < vocab_size,
  // 0 <= position < max_seq_len.
  void forward_token(int token_id, int position, cudaStream_t stream);

  // ---- Forward outputs (device; valid from forward_token until the next) --
  // Embedding output [hidden_size] bf16 (= embed_tokens.weight[token_id]).
  const __nv_bfloat16* embedding_output() const {
    return embed_out_.data<__nv_bfloat16>();
  }
  // Final-norm output [hidden_size] bf16 (the model's last_hidden_state).
  const __nv_bfloat16* final_norm_output() const {
    return norm_out_.data<__nv_bfloat16>();
  }
  // FULL logits [vocab_size] bf16 (tied LM head over the final-norm output).
  const __nv_bfloat16* logits() const {
    return logits_buf_.data<__nv_bfloat16>();
  }
  // Each layer's final output [hidden_size] bf16 (dispatches on the layer
  // type; null if `i` invalid or not loaded).
  const __nv_bfloat16* layer_final_output(int i) const;

 private:
  using LayerObj =
      std::variant<std::unique_ptr<Qwen35DeltaNetLayer>,
                   std::unique_ptr<Qwen35FullAttentionLayer>>;

  Qwen35Config cfg_{};
  // Embedding + final-norm device buffers, declared before the TensorViews
  // that alias them (reverse destruction tears down views before buffers).
  DeviceBuffer embed_buf_;
  DeviceBuffer norm_buf_;
  // Layer weight sets declared BEFORE the layer runtimes so (reverse)
  // destruction tears down the runtimes before the weight sets they reference.
  std::vector<std::unique_ptr<Qwen35LayerWeights>> weights_{};
  std::vector<LayerObj> layers_{};
  // Non-owning views over the buffers above; lm_head_ aliases the embedding
  // (weight tying). All trivially destructed before the buffers.
  TensorView embed_{};
  TensorView norm_{};
  TensorView lm_head_{};
  // Forward scratch + output buffers (allocated in load(); sized by config):
  // embedding output [hidden], final-norm output [hidden], logits [vocab].
  DeviceBuffer embed_out_{};
  DeviceBuffer norm_out_{};
  DeviceBuffer logits_buf_{};
  bool tie_ = false;
  bool loaded_ = false;
};

}  // namespace cudalm
