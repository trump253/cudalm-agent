// CUDALM — Qwen3.5 hybrid decoder micro-stack implementation (v0.2, Phase D).
// Reuses the frozen Phase B/C layer runtimes; adds only the top-level
// dispatch, per-layer state reset, and the whole-stack timing hook.

#include "cudalm/qwen35_hybrid_microstack.h"

#include "cudalm/cuda_check.h"
#include "cudalm/qwen35_kv_cache.h"

namespace cudalm {

namespace {

// Both layer types expose the same forward(stage) surface; dispatch on it.
template <typename LayerT>
const __nv_bfloat16* run_one(LayerT& layer, int position,
                             const __nv_bfloat16* x, cudaStream_t stream) {
  layer.forward(position, x, stream);
  return layer.stage_final_output();
}

}  // namespace

Status Qwen35HybridMicroStack::load(const WeightFileV2& file, cudaStream_t stream,
                                    Qwen35HybridMicroStack* out) {
  out->cfg_ = file.config();
  out->loaded_ = false;
  out->weights_ = {};
  out->layers_ = {};
  for (int i = 0; i < kNumLayers; ++i) {
    Status s = file.validate_layer(i);
    if (!s.ok) {
      s.message = "layer " + std::to_string(i) + ": " + s.message;
      return s;
    }
    auto w = std::make_unique<Qwen35LayerWeights>();
    s = Qwen35LayerWeights::load(file, i, stream, w.get());
    if (!s.ok) {
      s.message = "layer " + std::to_string(i) + ": " + s.message;
      return s;
    }
    out->weights_[i] = std::move(w);
    if (out->cfg_.is_linear_attention(i)) {
      out->layers_[i] =
          std::unique_ptr<Qwen35DeltaNetLayer>(new Qwen35DeltaNetLayer(
              *out->weights_[i], stream));
    } else {
      out->layers_[i] =
          std::unique_ptr<Qwen35FullAttentionLayer>(new Qwen35FullAttentionLayer(
              *out->weights_[i], stream));
    }
  }
  out->loaded_ = true;
  return Status::ok_status();
}

bool Qwen35HybridMicroStack::is_delta(int i) const {
  return i >= 0 && i < kNumLayers && cfg_.is_linear_attention(i);
}

bool Qwen35HybridMicroStack::is_att(int i) const {
  return i >= 0 && i < kNumLayers && cfg_.is_full_attention(i);
}

const __nv_bfloat16* Qwen35HybridMicroStack::dispatch(
    LayerObj& L, int position, const __nv_bfloat16* x,
    cudaStream_t stream) const {
  return std::visit(
      [position, x, stream](auto& uptr) {
        return run_one(*uptr, position, x, stream);
      },
      L);
}

void Qwen35HybridMicroStack::reset_state(cudaStream_t stream) {
  CUDALM_PRECONDITION(loaded_, "Qwen35HybridMicroStack: not loaded");
  for (int i = 0; i < kNumLayers; ++i) {
    if (is_delta(i)) {
      // DeltaNet: zero conv_state + recurrent_state (the layer owns them).
      delta(i)->reset_state(stream);
    } else {
      // Full attention: zero the KV cache (both tensors). The cache is
      // allocated [n_kv][max_seq][head_dim]; bytes() is ONE tensor's size.
      Qwen35KvCache& kv = attention(i)->kv_cache();
      CUDA_CHECK(cudaMemsetAsync(kv.k_mut(), 0, kv.bytes(), stream));
      CUDA_CHECK(cudaMemsetAsync(kv.v_mut(), 0, kv.bytes(), stream));
    }
  }
}

void Qwen35HybridMicroStack::forward(int position, const __nv_bfloat16* x_in,
                                     cudaStream_t stream) {
  CUDALM_PRECONDITION(loaded_, "Qwen35HybridMicroStack: not loaded");
  CUDALM_PRECONDITION(
      position >= 0 && position < cfg_.max_seq_len,
      "Qwen35HybridMicroStack::forward: position out of bounds");
  const __nv_bfloat16* x = x_in;
  for (int i = 0; i < kNumLayers; ++i) {
    x = dispatch(layers_[i], position, x, stream);
  }
}

void Qwen35HybridMicroStack::forwardTimed(int position,
                                          const __nv_bfloat16* x_in,
                                          cudaStream_t stream,
                                          cudaEvent_t* events, float* layer_us,
                                          float* whole_us) {
  CUDALM_PRECONDITION(loaded_, "Qwen35HybridMicroStack: not loaded");
  CUDALM_PRECONDITION(
      position >= 0 && position < cfg_.max_seq_len,
      "Qwen35HybridMicroStack::forwardTimed: position out of bounds");
  // Event layout: [0]=whole start; layer i -> [1+2i]=start,[2+2i]=stop;
  // [1+2*kNumLayers]=whole stop. Total 2*kNumLayers+2 events.
  CUDA_CHECK(cudaEventRecord(events[0], stream));
  const __nv_bfloat16* x = x_in;
  for (int i = 0; i < kNumLayers; ++i) {
    CUDA_CHECK(cudaEventRecord(events[1 + 2 * i], stream));
    x = dispatch(layers_[i], position, x, stream);
    CUDA_CHECK(cudaEventRecord(events[2 + 2 * i], stream));
  }
  CUDA_CHECK(cudaEventRecord(events[1 + 2 * kNumLayers], stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float ms = 0.f;
  for (int i = 0; i < kNumLayers; ++i) {
    CUDA_CHECK(cudaEventElapsedTime(&ms, events[1 + 2 * i],
                                    events[2 + 2 * i]));
    layer_us[i] = ms * 1000.f;
  }
  CUDA_CHECK(cudaEventElapsedTime(&ms, events[0],
                                  events[1 + 2 * kNumLayers]));
  *whole_us = ms * 1000.f;
}

const __nv_bfloat16* Qwen35HybridMicroStack::final_output() const {
  CUDALM_PRECONDITION(loaded_, "Qwen35HybridMicroStack: not loaded");
  const LayerObj& L = layers_.back();
  if (std::holds_alternative<std::unique_ptr<Qwen35DeltaNetLayer>>(L)) {
    return std::get<0>(L)->stage_final_output();
  }
  return std::get<1>(L)->stage_final_output();
}

Qwen35DeltaNetLayer* Qwen35HybridMicroStack::delta(int i) const {
  if (!is_delta(i)) return nullptr;
  return std::get<0>(layers_[i]).get();
}

Qwen35FullAttentionLayer* Qwen35HybridMicroStack::attention(int i) const {
  if (!is_att(i)) return nullptr;
  return std::get<1>(layers_[i]).get();
}

}  // namespace cudalm
