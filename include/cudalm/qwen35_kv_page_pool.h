// CUDALM — Qwen3.5 paged-KV resource pool (v0.5 Phase A). CUDALM-native.
//
// The multi-sequence replacement for the per-layer contiguous
// Qwen35KvCache [n_kv_heads][max_seq_len][head_dim] layout (NOT wired into
// Qwen35Model yet — that is Phase B). One physical page id addresses the
// SAME logical token block in EVERY full-attention layer, so a sequence
// needs exactly ONE KvBlockTable (kv_block_table.h) instead of one table
// per layer. The future paged-attention kernel will locate data by
// (layer ordinal, physical page, token offset).
//
// Device layout (per pool; all full-attention layers share the id space):
//   K: bf16 [n_full_layers][num_pages][n_kv_heads][page_tokens][head_dim]
//   V: bf16 [n_full_layers][num_pages][n_kv_heads][page_tokens][head_dim]
// row-major, so page p of layer ordinal l is a contiguous slice of
// n_kv_heads * page_tokens * head_dim bf16.
//
// page_tokens is EXPLICIT configuration (never hardcoded); tests use 4/8/16.
//
// Allocator contract (delegates to FixedIdPool; pinned by
// tests/cuda/test_qwen35_state_manager.cpp):
//   * live page ids are never handed out twice;
//   * free_page makes a page allocatable again (LIFO free order);
//   * double-free / out-of-range id -> Status error, no state change;
//   * pool exhausted -> OOM Status (never aborts);
//   * accounting is exact: capacity_pages == used_pages + free_pages,
//     used_bytes == used_pages * bytes_per_page.
//
// ZERO SEMANTS (explicit, tested on-device): the pool storage is zeroed at
// construction; free_page() and reset() ZERO the released page in all
// layers (zero-on-release) — a page acquired after a release is guaranteed
// all-zero regardless of what the previous owner wrote. Correctness-first:
// no optimization of the memset cost in this phase.
//
// Provenance: CUDALM-native (v0.5 Phase A; replaces nothing — the old
// per-layer Qwen35KvCache ownership is untouched until Phase B).

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cudalm/device_buffer.h"
#include "cudalm/fixed_id_pool.h"
#include "cudalm/kv_block_table.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_state_layout.h"
#include "cudalm/weight_format.h"  // Status

namespace cudalm {

// RAII paged-KV pool for the Qwen3.5 full-attention layers. Move-only.
class Qwen35KvPagePool : public KvPageSource {
 public:
  // cfg: Qwen35Config (the hybrid schedule gives the layer count);
  // page_tokens: tokens per physical page (>= 1, explicit);
  // num_pages: physical page capacity (>= 0; 0 -> every acquire is OOM).
  // Allocates + zeroes the K/V storage on `stream`.
  Qwen35KvPagePool(const Qwen35Config& cfg, int page_tokens, int num_pages,
                   cudaStream_t stream = 0);

  Qwen35KvPagePool(const Qwen35KvPagePool&) = delete;
  Qwen35KvPagePool& operator=(const Qwen35KvPagePool&) = delete;
  Qwen35KvPagePool(Qwen35KvPagePool&&) noexcept = default;
  Qwen35KvPagePool& operator=(Qwen35KvPagePool&&) noexcept = default;
  ~Qwen35KvPagePool() = default;

  // ---- allocator (KvPageSource) -------------------------------------------
  // Acquire one physical page (zeroed on (re)use by zero-on-release).
  // OOM Status when exhausted; no partial state on failure.
  Status allocate_page(int* out_page_id) override;
  // Release a live physical page and ZERO it in every layer (K and V),
  // ordered on the pool's stream (see STREAM CONTRACT). Double-free /
  // invalid -> Status error, no state change.
  Status free_page(int page_id) override;
  // Release every live page (each zeroed on the pool's stream).
  // No-op when nothing is live.
  void reset();
  // The pool's stream: all internal zeroing is ordered on it. Callers must
  // access pool pages on this stream (or one ordered after it) — the pool
  // is a single-stream resource (the Phase B model will pass its stream at
  // construction).
  cudaStream_t stream() const { return pool_stream_; }

  // ---- accounting (exact) ---------------------------------------------------
  int capacity_pages() const { return ids_.capacity(); }
  int used_pages() const { return ids_.used(); }
  int free_pages() const { return ids_.free_count(); }
  int page_tokens() const { return page_tokens_; }
  // Bytes of ONE page across all full-attention layers (K + V), from the
  // config schedule (qwen35_kv_page_bytes) — no magic constants.
  std::size_t bytes_per_page() const {
    return qwen35_kv_page_bytes(cfg_, page_tokens_);
  }
  std::size_t total_bytes() const {
    return static_cast<std::size_t>(capacity_pages()) * bytes_per_page();
  }
  std::size_t used_bytes() const {
    return static_cast<std::size_t>(used_pages()) * bytes_per_page();
  }

  // ---- layer mapping --------------------------------------------------------
  // Full-attention layer ordinal: 0..n_full_layers()-1 in ascending layer
  // index order (for the 0.8B config: ordinal 0..5 = layers 3,7,11,15,19,23).
  int n_full_layers() const { return n_full_; }
  int full_layer_ordinal(int layer_idx) const;   // -1 if not full-attention
  int layer_of_full_ordinal(int ordinal) const;  // -1 if ordinal out of range

  // ---- device access (Phase B kernels + tests) ------------------------------
  // K (or V) page slice: bf16 [n_kv_heads][page_tokens][head_dim], the
  // contiguous page-`page_id` slice of full-attention layer ordinal
  // `ordinal`. Precondition (host-checked): valid ordinal + live-or-not
  // page id in [0, capacity).
  const __nv_bfloat16* k_page(int ordinal, int page_id) const;
  const __nv_bfloat16* v_page(int ordinal, int page_id) const;
  // Non-const (tests write patterns; Phase B writes rows).
  __nv_bfloat16* k_page_mut(int ordinal, int page_id);
  __nv_bfloat16* v_page_mut(int ordinal, int page_id);
  // Element count of one page slice (n_kv_heads * page_tokens * head_dim).
  std::size_t page_elems() const {
    return static_cast<std::size_t>(cfg_.n_kv_heads) *
           static_cast<std::size_t>(page_tokens_) *
           static_cast<std::size_t>(cfg_.head_dim);
  }

  const Qwen35Config& config() const { return cfg_; }

 private:
  void zero_page(int page_id);

  Qwen35Config cfg_{};
  int page_tokens_ = 0;
  int n_full_ = 0;
  cudaStream_t pool_stream_ = 0;
  FixedIdPool ids_;
  DeviceBuffer k_;  // bf16 [n_full][num_pages][n_kv][page_tokens][head_dim]
  DeviceBuffer v_;  // same shape
  std::vector<int> full_layer_idxs_;  // ascending full-attention layer ids
};

}  // namespace cudalm
