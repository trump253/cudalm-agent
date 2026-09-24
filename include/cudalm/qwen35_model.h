// CUDALM — Qwen3.5 full model runtime skeleton (v0.3, Phase A).
//
// Owns the COMPLETE 24-layer Qwen3.5-0.8B text model: the token embedding,
// every decoder layer (dispatched Gated DeltaNet vs full attention by the
// config's hybrid schedule — full attention at layers 3,7,11,15,19,23), the
// final RMSNorm, and the (tied) LM head. CUDALM-native wiring that REUSES the
// frozen v0.2 single-layer runtimes (Qwen35FullAttentionLayer /
// Qwen35DeltaNetLayer) — no single-layer kernel is copied or re-derived.
//
// v0.3 Phase A scope = structure + weight ownership + state lifecycle only.
// This class does NOT run a full forward / compute logits (that is v0.3 Phase
// B); load() uploads the weights, reset_state() resets every layer's
// persistent state, and the accessors expose the owned tensors.
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
  bool tie_ = false;
  bool loaded_ = false;
};

}  // namespace cudalm
