// CUDALM — Qwen3.5 full model runtime skeleton implementation (v0.3 Phase A).
// Reuses the frozen v0.2 single-layer runtimes (Qwen35FullAttentionLayer /
// Qwen35DeltaNetLayer); adds only the model-level ownership (embedding, final
// norm, tied LM head), the per-layer dispatch over all 24 layers, and the
// whole-model state reset. No kernel is copied.

#include "cudalm/qwen35_model.h"

#include "cudalm/cuda_check.h"
#include "cudalm/qwen35_kv_cache.h"

namespace cudalm {

Status Qwen35Model::load(const WeightFileV2& file, cudaStream_t stream,
                         Qwen35Model* out) {
  out->cfg_ = file.config();
  out->loaded_ = false;
  out->tie_ = false;
  out->embed_ = TensorView();
  out->norm_ = TensorView();
  out->lm_head_ = TensorView();
  out->weights_.clear();
  out->layers_.clear();
  out->embed_buf_.reset();
  out->norm_buf_.reset();

  // Full-model tensor set: embedding + final norm + every layer + tie.
  Status s = file.validate_full_model();
  if (!s.ok) return s;
  // validate_full_model() guarantees tie_word_embeddings == "true", so the LM
  // head aliases the embedding (no separate tensor).
  out->tie_ = true;

  // Embedding: [vocab_size, hidden_size] bf16.
  {
    const Qwen35TensorInfo* info = file.find("embed_tokens.weight");
    out->embed_buf_.allocate(info->byte_size, stream);
    out->embed_buf_.copy_from_host(info->host_bytes, info->byte_size, stream);
    out->embed_ = TensorView(out->embed_buf_.data(), info->dtype, info->dims);
  }
  // Final norm: [hidden_size] bf16.
  {
    const Qwen35TensorInfo* info = file.find("norm.weight");
    out->norm_buf_.allocate(info->byte_size, stream);
    out->norm_buf_.copy_from_host(info->host_bytes, info->byte_size, stream);
    out->norm_ = TensorView(out->norm_buf_.data(), info->dtype, info->dims);
  }
  // LM head: TIED to the embedding — alias the same buffer (no separate
  // allocation). Used transposed: logits = hidden @ lm_head^T.
  out->lm_head_ = out->embed_;

  // Per-layer weight sets + runtimes, dispatched by the hybrid schedule.
  const int n = out->cfg_.num_hidden_layers;
  out->weights_.reserve(n);
  out->layers_.reserve(n);
  for (int i = 0; i < n; ++i) {
    auto w = std::make_unique<Qwen35LayerWeights>();
    s = Qwen35LayerWeights::load(file, i, stream, w.get());
    if (!s.ok) {
      s.message = "layer " + std::to_string(i) + ": " + s.message;
      return s;
    }
    out->weights_.push_back(std::move(w));
    // *weights_.back() is stable: the Qwen35LayerWeights object is heap
    // owned; push_back moves only the unique_ptr (and we reserved n).
    if (out->cfg_.is_linear_attention(i)) {
      out->layers_.emplace_back(
          std::unique_ptr<Qwen35DeltaNetLayer>(
              new Qwen35DeltaNetLayer(*out->weights_.back(), stream)));
    } else {
      out->layers_.emplace_back(
          std::unique_ptr<Qwen35FullAttentionLayer>(
              new Qwen35FullAttentionLayer(*out->weights_.back(), stream)));
    }
  }
  out->loaded_ = true;
  return Status::ok_status();
}

const Qwen35LayerWeights* Qwen35Model::layer_weights(int i) const {
  if (i < 0 || i >= num_layers()) return nullptr;
  return weights_[static_cast<std::size_t>(i)].get();
}

Qwen35DeltaNetLayer* Qwen35Model::delta(int i) const {
  if (i < 0 || i >= num_layers() || !is_linear_attention(i)) return nullptr;
  return std::get<0>(layers_[static_cast<std::size_t>(i)]).get();
}

Qwen35FullAttentionLayer* Qwen35Model::attention(int i) const {
  if (i < 0 || i >= num_layers() || !is_full_attention(i)) return nullptr;
  return std::get<1>(layers_[static_cast<std::size_t>(i)]).get();
}

void Qwen35Model::reset_state(cudaStream_t stream) {
  CUDALM_PRECONDITION(loaded_, "Qwen35Model: not loaded");
  for (int i = 0; i < num_layers(); ++i) {
    if (is_linear_attention(i)) {
      // DeltaNet: zero conv_state + recurrent_state (the layer owns them).
      delta(i)->reset_state(stream);
    } else {
      // Full attention: zero the KV cache (both tensors). The cache is
      // [n_kv][max_seq][head_dim]; bytes() is ONE tensor's size.
      Qwen35KvCache& kv = attention(i)->kv_cache();
      CUDA_CHECK(cudaMemsetAsync(kv.k_mut(), 0, kv.bytes(), stream));
      CUDA_CHECK(cudaMemsetAsync(kv.v_mut(), 0, kv.bytes(), stream));
    }
  }
}

}  // namespace cudalm
