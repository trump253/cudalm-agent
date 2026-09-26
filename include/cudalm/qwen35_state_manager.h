// CUDALM — Qwen3.5 unified sequence state / state manager (v0.5 Phase A).
// CUDALM-native.
//
// The host-side control plane over the two device state pools:
//   * Qwen35KvPagePool    — paged KV storage for the full-attention layers
//   * Qwen35DeltaStatePool— conv + recurrent storage for the DeltaNet layers
// Each live sequence is a SequenceState record (a host metadata object):
//   * a UNIQUE, monotonically increasing SequenceId (never reused, even
//     after retire — stale handles can never silently point at a new
//     request);
//   * ONE DeltaNet state slot (shared across all DeltaNet layers);
//   * ONE KvBlockTable (shared across all full-attention layers);
//   * the current sequence length (tokens placed so far).
//
// Phase A scope: state OWNERSHIP only. The manager is NOT wired into
// Qwen35Model forward (Phase B) and there is NO scheduler / admission
// policy (a later phase).
//
// LIFECYCLE CONTRACT (pinned by tests/cuda/test_qwen35_state_manager.cpp):
//
//   create_sequence(&id):
//     fresh unique SequenceId; a FRESH ZEROED DeltaNet slot (zero-on-
//     release guarantees this); an EMPTY KV block table (no pages yet);
//     length = 0. If the Delta slot acquisition OOMs, NOTHING is
//     registered: no id is issued, no half-record exists (the next
//     create_sequence succeeds on a recovered pool and the id sequence
//     stays monotone).
//
//   ensure_kv_capacity(id, position):
//     makes the block containing `position` (and every earlier block)
//     physically allocated, exactly KvBlockTable::ensure_capacity against
//     the pool. TRANSACTIONAL: on OOM the block table, the pool
//     accounting, and the used/free counts are exactly unchanged (no
//     leaked page). position outside [0, max_seq_len) -> Status error.
//
//   set_length(id, new_length) / advance(id, n):
//     PURE METADATA length updates (0 <= new_length <= max_seq_len;
//     advance: n >= 0, new_length = length + n). They do NOT allocate KV
//     pages — page coverage is the caller's job via ensure_kv_capacity
//     (Phase B will ensure before writing token p). Shrinking is allowed
//     (pages are NOT released by a shrink; only reset/retire release).
//
//   reset_sequence(id):
//     the sequence stays LIVE under the SAME id: length = 0, all KV pages
//     released (zeroed on release), the Delta slot ZEROED IN PLACE (same
//     slot id, no release/acquire — so a reset can never OOM). Afterwards
//     the sequence behaves exactly like a fresh one.
//
//   retire_sequence(id):
//     releases ALL KV pages (zeroed) and the Delta slot (zeroed); the
//     record is removed. The id is NEVER reissued. Any later operation on
//     a retired id (lookup included) is a Status error / nullptr.
//
//   lookup(id):
//     pointer to the live record, or nullptr if the id was never created
//     or was retired (a query, not an error).
//
// MEMORY ACCOUNTING: manager-level exact byte accounting is the SUM of
// the pools' exact accounting (total_state_bytes = KV total + Delta
// total; used_state_bytes = KV used + Delta used). All formulas derive
// from the Qwen35Config hybrid schedule (qwen35_state_layout.h).
//
// Provenance: CUDALM-native (v0.5 Phase A).

#pragma once

#include <cstdint>
#include <cstddef>
#include <map>

#include <cuda_runtime.h>

#include "cudalm/kv_block_table.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_delta_state_pool.h"
#include "cudalm/qwen35_kv_page_pool.h"
#include "cudalm/weight_format.h"  // Status

namespace cudalm {

// Monotonically increasing, never-reused sequence identifier.
using SequenceId = std::uint64_t;
// A DeltaNet state slot id (pool-local; -1 = none).
using DeltaStateSlot = int;

// One live sequence's host-side state record.
struct SequenceState {
  SequenceId id = 0;
  DeltaStateSlot delta_slot = -1;
  KvBlockTable block_table;  // logical block -> physical page
  int length = 0;            // tokens placed so far
};

class Qwen35StateManager {
 public:
  // cfg: the model config (hybrid schedule drives ALL counts/formulas);
  // page_tokens: KV page size (>= 1, explicit configuration);
  // kv_capacity_pages / delta_capacity_slots: pool capacities (>= 0).
  // Allocates + zeroes the pool storage on `stream`.
  Qwen35StateManager(const Qwen35Config& cfg, int page_tokens,
                     int kv_capacity_pages, int delta_capacity_slots,
                     cudaStream_t stream = 0);

  Qwen35StateManager(const Qwen35StateManager&) = delete;
  Qwen35StateManager& operator=(const Qwen35StateManager&) = delete;
  Qwen35StateManager(Qwen35StateManager&&) = delete;
  Qwen35StateManager& operator=(Qwen35StateManager&&) = delete;
  ~Qwen35StateManager() = default;

  // ---- lifecycle (see the header contract) ----------------------------------
  Status create_sequence(SequenceId* out_id);
  Status ensure_kv_capacity(SequenceId id, int position);
  Status set_length(SequenceId id, int new_length);
  Status advance(SequenceId id, int n_tokens);
  Status reset_sequence(SequenceId id);
  Status retire_sequence(SequenceId id);

  // nullptr when the id is unknown or retired (query, not an error).
  const SequenceState* lookup(SequenceId id) const;

  // ---- capacity / accounting --------------------------------------------------
  int num_live_sequences() const {
    return static_cast<int>(sequences_.size());
  }
  // The NEXT SequenceId to be issued (1 when nothing has ever been
  // created). Ids are monotone increasing and never reused, so this is
  // always one greater than the highest id issued so far.
  SequenceId next_sequence_id() const { return next_id_; }

  const Qwen35KvPagePool& kv_pool() const { return kv_; }
  const Qwen35DeltaStatePool& delta_pool() const { return delta_; }
  // Non-const pool views (Phase B: the model's external-state forward
  // WRITES into the pool's pages / delta slots; the pool API itself is
  // unchanged).
  Qwen35KvPagePool& kv_pool_mut() { return kv_; }
  Qwen35DeltaStatePool& delta_pool_mut() { return delta_; }
  // The pool's KV page size (the paged kernels' `page_tokens`).
  int page_tokens() const { return kv_.page_tokens(); }
  const Qwen35Config& config() const { return cfg_; }

  // Exact manager-level byte accounting (sum of the exact pool
  // accountings; formulas from the config schedule).
  std::size_t total_state_bytes() const {
    return kv_.total_bytes() + delta_.total_bytes();
  }
  std::size_t used_state_bytes() const {
    return kv_.used_bytes() + delta_.used_bytes();
  }

 private:
  // Status error for an id that is not a live sequence.
  Status not_live(SequenceId id) const;

  Qwen35Config cfg_{};
  Qwen35KvPagePool kv_;
  Qwen35DeltaStatePool delta_;
  std::map<SequenceId, SequenceState> sequences_;
  SequenceId next_id_ = 1;  // ids start at 1; 0 means "no sequence"
};

}  // namespace cudalm
