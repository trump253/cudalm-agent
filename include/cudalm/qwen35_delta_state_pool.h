// CUDALM — Qwen3.5 DeltaNet per-sequence state pool (v0.5 Phase A).
// CUDALM-native.
//
// The multi-sequence replacement for the per-layer persistent DeltaNet
// state (conv_state bf16 [conv_dim, 3] + recurrent_state fp32
// [n_heads, head_dim, head_dim], currently owned by each
// Qwen35DeltaNetLayer object). NOT wired into the model yet (Phase B).
//
// One sequence gets ONE state slot; the same slot id addresses the
// per-layer states of ALL linear-attention layers, so multi-sequence
// execution needs one slot per sequence — no per-layer slot tables.
//
// Device layout (per pool; the state stays ON DEVICE — never copied to
// host):
//   per linear-attention layer ordinal l (0..n_linear_layers()-1):
//     conv:      bf16 [capacity][linear_conv_dim, 3]
//     recurrent: fp32 [capacity][lin_num_k_heads, lin_key_head_dim,
//                                  lin_value_head_dim]
// A slot's per-layer state is a contiguous slice; the slot as a whole is
// NOT contiguous (it spans the per-layer tensors), which is exactly what
// the future kernels need (one base pointer per layer + slot offset).
//
// Allocator contract (delegates to FixedIdPool; pinned by
// tests/cuda/test_qwen35_state_manager.cpp):
//   * live slots never alias;
//   * double-release / invalid slot -> Status error, no state change;
//   * pool exhausted -> OOM Status (never aborts);
//   * accounting exact: capacity_slots == used + free, used_bytes ==
//     used_slots * bytes_per_slot.
//
// ZERO SEMANTS (explicit, tested on-device): storage is zeroed at
// construction; release_slot() and reset_slot() ZERO the slot's conv +
// recurrent state in every layer (zero-on-release / zero-on-reset) — an
// acquired or reset slot is guaranteed all-zero regardless of what the
// previous owner wrote. Correctness-first: no memset optimization here.
//
// Provenance: CUDALM-native (v0.5 Phase A).

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cudalm/device_buffer.h"
#include "cudalm/fixed_id_pool.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_state_layout.h"
#include "cudalm/weight_format.h"  // Status

namespace cudalm {

// RAII DeltaNet state pool for the Qwen3.5 linear-attention layers.
// Move-only.
class Qwen35DeltaStatePool {
 public:
  // cfg: Qwen35Config (the hybrid schedule gives the layer count);
  // capacity: number of sequence state slots (>= 0; 0 -> every acquire is
  // OOM). Allocates + zeroes the conv/recurrent storage on `stream`.
  Qwen35DeltaStatePool(const Qwen35Config& cfg, int capacity,
                       cudaStream_t stream = 0);

  Qwen35DeltaStatePool(const Qwen35DeltaStatePool&) = delete;
  Qwen35DeltaStatePool& operator=(const Qwen35DeltaStatePool&) = delete;
  Qwen35DeltaStatePool(Qwen35DeltaStatePool&&) noexcept = default;
  Qwen35DeltaStatePool& operator=(Qwen35DeltaStatePool&&) noexcept = default;
  ~Qwen35DeltaStatePool() = default;

  // ---- allocator ------------------------------------------------------------
  // Acquire one state slot (guaranteed all-zero by zero-on-release).
  // OOM Status when exhausted; no partial state on failure.
  Status acquire_slot(int* out_slot);
  // Release a live slot and ZERO its state in every layer (ordered on the
  // pool's stream). Double-release / invalid -> Status error.
  Status release_slot(int slot);
  // ZERO a LIVE slot in place (sequence reset without changing slot
  // ownership). Invalid / not-live slot -> Status error.
  Status reset_slot(int slot);
  // Release every live slot (each zeroed). No-op when nothing is live.
  void reset();
  // The pool's stream: all internal zeroing is ordered on it; callers must
  // access slot storage on this stream (single-stream resource, like the
  // KV page pool).
  cudaStream_t stream() const { return pool_stream_; }

  // ---- accounting (exact) -----------------------------------------------------
  int capacity_slots() const { return ids_.capacity(); }
  int used_slots() const { return ids_.used(); }
  int free_slots() const { return ids_.free_count(); }
  // Bytes of ONE slot across all linear-attention layers, from the config
  // schedule (qwen35_delta_slot_bytes) — no magic constants.
  std::size_t bytes_per_slot() const { return qwen35_delta_slot_bytes(cfg_); }
  std::size_t total_bytes() const {
    return static_cast<std::size_t>(capacity_slots()) * bytes_per_slot();
  }
  std::size_t used_bytes() const {
    return static_cast<std::size_t>(used_slots()) * bytes_per_slot();
  }

  // ---- layer mapping -----------------------------------------------------------
  // Linear-attention layer ordinal: 0..n_linear_layers()-1 in ascending
  // layer index order (0.8B: layers 0,1,2,4,5,6,... i.e. all except
  // 3,7,11,15,19,23).
  int n_linear_layers() const { return n_linear_; }
  int linear_layer_ordinal(int layer_idx) const;   // -1 if full-attention
  int layer_of_linear_ordinal(int ordinal) const;  // -1 if ordinal invalid

  // ---- device access (Phase B kernels + tests) ---------------------------------
  // Slot's conv state of linear layer ordinal: bf16 [linear_conv_dim, 3].
  const __nv_bfloat16* conv(int ordinal, int slot) const;
  // Slot's recurrent state of linear layer ordinal:
  // fp32 [lin_num_k_heads, lin_key_head_dim, lin_value_head_dim].
  const float* recurrent(int ordinal, int slot) const;
  // Non-const (tests write patterns; Phase B updates in place).
  __nv_bfloat16* conv_mut(int ordinal, int slot);
  float* recurrent_mut(int ordinal, int slot);
  // Element counts of one slot's per-layer state.
  std::size_t conv_elems() const {
    return static_cast<std::size_t>(cfg_.linear_conv_dim()) *
           static_cast<std::size_t>(cfg_.linear_conv_state_len());
  }
  std::size_t rec_elems() const {
    return static_cast<std::size_t>(cfg_.lin_num_k_heads) *
           static_cast<std::size_t>(cfg_.lin_key_head_dim) *
           static_cast<std::size_t>(cfg_.lin_value_head_dim);
  }

  const Qwen35Config& config() const { return cfg_; }

 private:
  void zero_slot(int slot);

  Qwen35Config cfg_{};
  int n_linear_ = 0;
  cudaStream_t pool_stream_ = 0;
  FixedIdPool ids_;
  std::vector<DeviceBuffer> conv_;  // per ordinal: bf16 [capacity][conv_dim,3]
  std::vector<DeviceBuffer> rec_;   // per ordinal: fp32 [capacity][k][d][d]
  std::vector<int> linear_layer_idxs_;  // ascending linear-attention layers
};

}  // namespace cudalm
