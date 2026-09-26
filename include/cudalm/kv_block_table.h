// CUDALM — per-sequence logical KV block table (v0.5 Phase A, CPU-only).
//
// Maps logical token blocks to physical KV page ids. A SEQUENCE has exactly
// ONE block table shared by all full-attention layers (the same physical
// page id addresses the same logical token block in every layer — see
// Qwen35KvPagePool). Header-only, CPU-only (the physical page allocation is
// delegated to a KvPageSource; the device pool implements it, tests use a
// fake).
//
// Contract (pinned by tests/cpu/test_kv_block_table.cpp):
//   * position p (0-based) belongs to logical block p / page_tokens at
//     offset p % page_tokens (exact, for every position);
//   * the table is always a PREFIX: blocks [0, num_blocks()) are allocated,
//     blocks >= num_blocks() are not (append-only growth);
//   * ensure_capacity(position, src): the smallest no-op/growth that makes
//     the block containing `position` allocated. Allocates only the new
//     tail blocks. TRANSACTIONAL: if any tail acquisition fails, every page
//     acquired during THIS call is released again and the table is exactly
//     unchanged (num_blocks, accounting) — a failed ensure leaks nothing;
//   * the same logical block NEVER acquires a second physical page (its
//     page is allocated exactly once, at the moment the block enters the
//     prefix);
//   * position outside [0, max_seq_len) -> Status error (fail loud);
//   * lookup(b): page id for an allocated block, -1 otherwise (a query —
//     it never allocates);
//   * clear(src): release every allocated page, table becomes empty.
//
// Provenance: CUDALM-native (v0.5 Phase A).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cudalm/weight_format.h"  // Status

namespace cudalm {

// The physical page provider the block table grows against.
// Qwen35KvPagePool implements it; CPU tests use a counting fake.
class KvPageSource {
 public:
  virtual ~KvPageSource() = default;
  // Acquire one physical page id. OOM Status when exhausted.
  virtual Status allocate_page(int* out_page_id) = 0;
  // Release a live physical page id.
  virtual Status free_page(int page_id) = 0;
};

class KvBlockTable {
 public:
  // max_seq_len and page_tokens must be >= 1.
  KvBlockTable() = default;
  KvBlockTable(int max_seq_len, int page_tokens)
      : max_seq_len_(max_seq_len), page_tokens_(page_tokens) {}

  // Maximum number of logical blocks: ceil(max_seq_len / page_tokens).
  int max_blocks() const {
    return (max_seq_len_ + page_tokens_ - 1) / page_tokens_;
  }

  int max_seq_len() const { return max_seq_len_; }
  int page_tokens() const { return page_tokens_; }
  int num_blocks() const { return static_cast<int>(pages_.size()); }

  // Exact position decomposition (valid domain 0 <= position < max_seq_len).
  int block_of_position(int position) const {
    return position / page_tokens_;
  }
  int offset_of_position(int position) const {
    return position % page_tokens_;
  }
  // First position of the block containing `position` (its base).
  int block_base_position(int logical_block) const {
    return logical_block * page_tokens_;
  }

  // Host-side view of the current prefix (logical block -> physical page),
  // one entry per allocated block. This is the array a caller copies to a
  // device scratch buffer for the paged kernels (v0.5 Phase B: per-token
  // H2D of the current page IDs; the kernels read only entries
  // [0, position/page_tokens], so stale device content beyond the copied
  // prefix is never touched).
  const int* page_ids() const { return pages_.data(); }

  // Query: physical page id of an ALLOCATED logical block, else -1.
  int lookup(int logical_block) const {
    return logical_block >= 0 && logical_block < num_blocks()
               ? pages_[static_cast<std::size_t>(logical_block)]
               : -1;
  }

  // Allocate physical pages for every block covering positions
  // [0, position] (i.e. blocks [0, position/page_tokens]). No-op when
  // already covered. TRANSACTIONAL on partial failure (see header).
  // `position` outside [0, max_seq_len) -> Status error.
  Status ensure_capacity(int position, KvPageSource& src) {
    if (position < 0 || position >= max_seq_len_)
      return Status::error("KvBlockTable::ensure_capacity: position " +
                           std::to_string(position) + " outside [0, " +
                           std::to_string(max_seq_len_) + ")");
    const int target = block_of_position(position);
    if (target < num_blocks()) return Status::ok_status();  // covered
    const int needed = target + 1 - num_blocks();
    std::vector<int> acquired;
    acquired.reserve(static_cast<std::size_t>(needed));
    for (int i = 0; i < needed; ++i) {
      int page_id = -1;
      Status s = src.allocate_page(&page_id);
      if (!s.ok) {
        // Roll back THIS call's acquisitions (in reverse); the existing
        // prefix of the table is untouched by construction.
        for (std::size_t j = acquired.size(); j-- > 0;)
          src.free_page(acquired[j]);
        return Status::error(
            "KvBlockTable::ensure_capacity: page allocation failed at "
            "logical block " + std::to_string(num_blocks() + i) +
            " (" + s.message + "); table unchanged");
      }
      acquired.push_back(page_id);
    }
    // All acquired: commit the tail blocks.
    pages_.insert(pages_.end(), acquired.begin(), acquired.end());
    return Status::ok_status();
  }

  // Release every allocated page (the table becomes empty).
  Status clear(KvPageSource& src) {
    Status last = Status::ok_status();
    for (std::size_t i = 0; i < pages_.size(); ++i) {
      Status s = src.free_page(pages_[i]);
      if (!s.ok) last = std::move(s);  // keep releasing; report the failure
    }
    pages_.clear();
    return last;
  }

 private:
  int max_seq_len_ = 0;
  int page_tokens_ = 0;
  std::vector<int> pages_;  // index = logical block, value = physical page
};

}  // namespace cudalm
