// CUDALM — fixed-capacity unique-ID freelist (CPU-only, header-only).
//
// The allocator core shared by the v0.5 Phase A state pools (KV pages,
// DeltaNet state slots). Pure host metadata: no CUDA, no model.
//
// Contract (pinned by tests/cpu/test_fixed_id_pool.cpp):
//   * capacity is fixed at construction (capacity >= 0; 0 -> every acquire
//     is OOM);
//   * IDs are [0, capacity); a LIVE id is NEVER handed out twice — no live
//     aliasing is possible;
//   * release returns the id to the free set; a released id MAY be handed
//     out again (LIFO order: the most recently released id is the first one
//     reacquired — deterministic, test-pinned);
//   * double-release (release of a non-live id) -> Status error, no state
//     change;
//   * out-of-range id -> Status error;
//   * acquire with no free id -> OOM Status error, no state change;
//   * reset() releases ALL live ids (the storage/content belonging to them
//     is the OWNER's job — the pools zero on release/reset);
//   * accounting is EXACT at every step: capacity == used + free.
//
// Provenance: CUDALM-native (v0.5 Phase A).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cudalm/weight_format.h"  // Status

namespace cudalm {

class FixedIdPool {
 public:
  // Precondition: capacity >= 0 (negative -> UB; callers validate).
  explicit FixedIdPool(std::int32_t capacity)
      : capacity_(capacity),
        live_(static_cast<std::size_t>(capacity), 0) {
    free_ids_.reserve(static_cast<std::size_t>(capacity_));
    for (int i = capacity_ - 1; i >= 0; --i) free_ids_.push_back(i);
  }

  // Acquire one free id. OOM (clear Status) when the pool is exhausted;
  // on failure the pool state is unchanged.
  Status acquire(int* out_id) {
    if (out_id == nullptr)
      return Status::error("FixedIdPool::acquire: null output pointer");
    if (free_ids_.empty())
      return Status::error("FixedIdPool: OOM — pool exhausted "
                           "(capacity=" + std::to_string(capacity_) + ")");
    const int id = static_cast<int>(free_ids_.back());
    free_ids_.pop_back();
    live_[static_cast<std::size_t>(id)] = 1;
    *out_id = id;
    return Status::ok_status();
  }

  // Release a live id. Invalid (out of range) or double-release (not live)
  // -> Status error, no state change. A released id becomes allocatable
  // again (the owner must guarantee the storage it represents is clean —
  // the device pools zero on release).
  Status release(int id) {
    if (id < 0 || id >= capacity_)
      return Status::error("FixedIdPool::release: id " + std::to_string(id) +
                           " out of range (capacity=" +
                           std::to_string(capacity_) + ")");
    if (!live_[static_cast<std::size_t>(id)])
      return Status::error("FixedIdPool::release: id " + std::to_string(id) +
                           " is not live (double release or invalid)");
    live_[static_cast<std::size_t>(id)] = 0;
    free_ids_.push_back(id);
    return Status::ok_status();
  }

  bool is_live(int id) const {
    return id >= 0 && id < capacity_ &&
           live_[static_cast<std::size_t>(id)] != 0;
  }

  // Release every live id. No-op when nothing is live.
  void reset() {
    for (int id = 0; id < capacity_; ++id) {
      if (live_[static_cast<std::size_t>(id)]) {
        live_[static_cast<std::size_t>(id)] = 0;
        free_ids_.push_back(id);
      }
    }
  }

  // ---- exact accounting ----------------------------------------------------
  int capacity() const { return capacity_; }
  int used() const {
    int n = 0;
    for (int id = 0; id < capacity_; ++id)
      if (live_[static_cast<std::size_t>(id)]) ++n;
    return n;
  }
  int free_count() const { return capacity() - used(); }

  // Live id list in ascending order (tests / diagnostics).
  std::vector<int> live_ids() const {
    std::vector<int> v;
    for (int id = 0; id < capacity_; ++id)
      if (live_[static_cast<std::size_t>(id)]) v.push_back(id);
    return v;
  }

 private:
  int capacity_ = 0;
  std::vector<unsigned char> live_;   // 0 = free, 1 = live
  std::vector<int> free_ids_;         // LIFO free list
};

}  // namespace cudalm
