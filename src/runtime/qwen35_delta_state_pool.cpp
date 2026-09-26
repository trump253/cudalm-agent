// CUDALM — Qwen3.5 DeltaNet state pool implementation (v0.5 Phase A).

#include "cudalm/qwen35_delta_state_pool.h"

#include "cudalm/cuda_check.h"

namespace cudalm {

Qwen35DeltaStatePool::Qwen35DeltaStatePool(const Qwen35Config& cfg,
                                           int capacity, cudaStream_t stream)
    : cfg_(cfg),
      n_linear_(qwen35_num_linear_layers(cfg)),
      pool_stream_(stream),
      // capacity is precondition-checked in the body (fatal abort for
      // negative); clamp here so member init itself is always safe.
      ids_(capacity >= 0 ? capacity : 0) {
  CUDALM_PRECONDITION(capacity >= 0,
                      "Qwen35DeltaStatePool: capacity must be >= 0");
  CUDALM_PRECONDITION(
      n_linear_ >= 1 && cfg.linear_conv_dim() >= 1 &&
          cfg.lin_num_k_heads >= 1 && cfg.lin_key_head_dim >= 1 &&
          cfg.lin_value_head_dim >= 1,
      "Qwen35DeltaStatePool: config must have positive DeltaNet dims");
  linear_layer_idxs_.reserve(static_cast<std::size_t>(n_linear_));
  for (int i = 0; i < cfg.num_hidden_layers; ++i)
    if (cfg.is_linear_attention(i)) linear_layer_idxs_.push_back(i);

  // Per-layer slot slices; zero the WHOLE pool at construction (the
  // zero-on-release invariant then holds for every acquire/reset).
  const std::size_t conv_bytes = conv_elems() * sizeof(__nv_bfloat16);
  const std::size_t rec_bytes = rec_elems() * sizeof(float);
  conv_.resize(static_cast<std::size_t>(n_linear_));
  rec_.resize(static_cast<std::size_t>(n_linear_));
  for (int l = 0; l < n_linear_; ++l) {
    conv_[static_cast<std::size_t>(l)].allocate(
        conv_bytes * static_cast<std::size_t>(capacity), stream);
    rec_[static_cast<std::size_t>(l)].allocate(
        rec_bytes * static_cast<std::size_t>(capacity), stream);
    conv_[static_cast<std::size_t>(l)].clear(
        conv_bytes * static_cast<std::size_t>(capacity), stream);
    rec_[static_cast<std::size_t>(l)].clear(
        rec_bytes * static_cast<std::size_t>(capacity), stream);
  }
}

Status Qwen35DeltaStatePool::acquire_slot(int* out_slot) {
  return ids_.acquire(out_slot);
}

Status Qwen35DeltaStatePool::release_slot(int slot) {
  if (slot < 0 || slot >= ids_.capacity())
    return Status::error("delta state pool: slot " + std::to_string(slot) +
                         " out of range (capacity=" +
                         std::to_string(ids_.capacity()) + ")");
  if (!ids_.is_live(slot))
    return Status::error("delta state pool: slot " + std::to_string(slot) +
                         " is not live (double release or invalid)");
  zero_slot(slot);  // ordered on the pool's stream
  return ids_.release(slot);
}

Status Qwen35DeltaStatePool::reset_slot(int slot) {
  if (slot < 0 || slot >= ids_.capacity())
    return Status::error("delta state pool: slot " + std::to_string(slot) +
                         " out of range (capacity=" +
                         std::to_string(ids_.capacity()) + ")");
  if (!ids_.is_live(slot))
    return Status::error("delta state pool: reset_slot: slot " +
                         std::to_string(slot) + " is not live");
  zero_slot(slot);
  return Status::ok_status();
}

void Qwen35DeltaStatePool::reset() {
  const std::vector<int> live = ids_.live_ids();
  for (int slot : live) zero_slot(slot);
  ids_.reset();
}

void Qwen35DeltaStatePool::zero_slot(int slot) {
  const std::size_t conv_bytes = conv_elems() * sizeof(__nv_bfloat16);
  const std::size_t rec_bytes = rec_elems() * sizeof(float);
  for (int l = 0; l < n_linear_; ++l) {
    const std::size_t off = static_cast<std::size_t>(slot);
    CUDA_CHECK(cudaMemsetAsync(
        static_cast<char*>(conv_[static_cast<std::size_t>(l)].data()) +
            off * conv_bytes,
        0, conv_bytes, pool_stream_));
    CUDA_CHECK(cudaMemsetAsync(
        static_cast<char*>(rec_[static_cast<std::size_t>(l)].data()) +
            off * rec_bytes,
        0, rec_bytes, pool_stream_));
  }
}

int Qwen35DeltaStatePool::linear_layer_ordinal(int layer_idx) const {
  if (!cfg_.is_linear_attention(layer_idx)) return -1;
  for (int o = 0; o < static_cast<int>(linear_layer_idxs_.size()); ++o)
    if (linear_layer_idxs_[static_cast<std::size_t>(o)] == layer_idx)
      return o;
  return -1;  // unreachable
}

int Qwen35DeltaStatePool::layer_of_linear_ordinal(int ordinal) const {
  if (ordinal < 0 || ordinal >= static_cast<int>(linear_layer_idxs_.size()))
    return -1;
  return linear_layer_idxs_[static_cast<std::size_t>(ordinal)];
}

namespace {
inline bool slot_valid(int n, int capacity, int ordinal, int slot) {
  return ordinal >= 0 && ordinal < n && slot >= 0 && slot < capacity;
}
}  // namespace

const __nv_bfloat16* Qwen35DeltaStatePool::conv(int ordinal, int slot) const {
  CUDALM_PRECONDITION(slot_valid(n_linear_, ids_.capacity(), ordinal, slot),
                      "Qwen35DeltaStatePool: conv (ordinal, slot) out of "
                      "range");
  return conv_[static_cast<std::size_t>(ordinal)].data<__nv_bfloat16>() +
         static_cast<std::size_t>(slot) * conv_elems();
}

const float* Qwen35DeltaStatePool::recurrent(int ordinal, int slot) const {
  CUDALM_PRECONDITION(slot_valid(n_linear_, ids_.capacity(), ordinal, slot),
                      "Qwen35DeltaStatePool: recurrent (ordinal, slot) out "
                      "of range");
  return rec_[static_cast<std::size_t>(ordinal)].data<float>() +
         static_cast<std::size_t>(slot) * rec_elems();
}

__nv_bfloat16* Qwen35DeltaStatePool::conv_mut(int ordinal, int slot) {
  return const_cast<__nv_bfloat16*>(conv(ordinal, slot));
}

float* Qwen35DeltaStatePool::recurrent_mut(int ordinal, int slot) {
  return const_cast<float*>(recurrent(ordinal, slot));
}

}  // namespace cudalm
