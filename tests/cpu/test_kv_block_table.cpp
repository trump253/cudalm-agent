// CUDALM — v0.5 Phase A: KvBlockTable gate (CPU-only, fake page source).
//
// Pinned contract (see include/cudalm/kv_block_table.h):
//   * position -> block/offset exact for the boundary matrix
//     (0, page_tokens-1, page_tokens, page_tokens+1, last valid, max);
//   * ensure_capacity grows the prefix minimally and is idempotent;
//   * the same logical block never acquires a second physical page;
//   * TRANSACTIONAL OOM: a failed ensure (even mid-call) leaves the table
//     and the source accounting exactly unchanged — no leaked page;
//   * position outside [0, max_seq_len) -> Status error, no change;
//   * clear releases every page (exact count);
//   * released pages are reusable and the LIFO identity is observable.

#include "../../tests/common/check.h"
#include "cudalm/kv_block_table.h"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

using namespace cudalm;

namespace {

// A counting fixed-capacity page source (LIFO free order, like the pool).
class FakePages : public KvPageSource {
 public:
  explicit FakePages(int capacity)
      : capacity_(capacity), live_(static_cast<std::size_t>(capacity), 0) {
    for (int i = capacity - 1; i >= 0; --i) free_ids_.push_back(i);
  }
  Status allocate_page(int* out) override {
    if (free_ids_.empty()) return Status::error("fake pool OOM");
    const int id = static_cast<int>(free_ids_.back());
    free_ids_.pop_back();
    live_[static_cast<std::size_t>(id)] = 1;
    *out = id;
    return Status::ok_status();
  }
  Status free_page(int id) override {
    if (id < 0 || id >= capacity_ || !live_[static_cast<std::size_t>(id)])
      return Status::error("fake pool: not live");
    live_[static_cast<std::size_t>(id)] = 0;
    free_ids_.push_back(id);
    return Status::ok_status();
  }
  int used() const {
    int n = 0;
    for (int i = 0; i < capacity_; ++i)
      if (live_[static_cast<std::size_t>(i)]) ++n;
    return n;
  }
  int capacity() const { return capacity_; }

 private:
  int capacity_;
  std::vector<unsigned char> live_;
  std::vector<int> free_ids_;
};

// 1) Pure position decomposition on the boundary matrix.
int test_position_decomposition() {
  const int MAX_SEQ = 64, PT = 8;
  KvBlockTable t(MAX_SEQ, PT);
  CHECK_EQ(t.max_blocks(), 8);  // ceil(64/8)
  CHECK_EQ(t.block_of_position(0), 0);
  CHECK_EQ(t.offset_of_position(0), 0);
  CHECK_EQ(t.block_of_position(PT - 1), 0);
  CHECK_EQ(t.offset_of_position(PT - 1), PT - 1);
  CHECK_EQ(t.block_of_position(PT), 1);
  CHECK_EQ(t.offset_of_position(PT), 0);
  CHECK_EQ(t.block_of_position(PT + 1), 1);
  CHECK_EQ(t.offset_of_position(PT + 1), 1);
  CHECK_EQ(t.block_of_position(MAX_SEQ - 1), 7);
  CHECK_EQ(t.offset_of_position(MAX_SEQ - 1), PT - 1);
  // Non-divisible: MAX_SEQ = 64, PT = 10 -> 7 blocks, last partial.
  KvBlockTable u(64, 10);
  CHECK_EQ(u.max_blocks(), 7);
  CHECK_EQ(u.block_of_position(63), 6);
  CHECK_EQ(u.offset_of_position(63), 3);
  std::printf("  [ok] position -> block/offset exact on the boundary matrix\n");
  return 0;
}

// 2) Minimal prefix growth, idempotency, one physical page per block.
int test_ensure_grows_prefix_minimally() {
  const int MAX_SEQ = 64, PT = 8;
  FakePages src(8);
  KvBlockTable t(MAX_SEQ, PT);
  CHECK_EQ(t.num_blocks(), 0);
  CHECK(t.ensure_capacity(PT - 1, src).ok);  // block 0 only
  CHECK_EQ(t.num_blocks(), 1);
  CHECK_EQ(src.used(), 1);
  const int page0 = t.lookup(0);
  CHECK(page0 >= 0);
  CHECK(t.ensure_capacity(PT, src).ok);  // crosses the boundary: block 1
  CHECK_EQ(t.num_blocks(), 2);
  CHECK_EQ(src.used(), 2);
  const int page1 = t.lookup(1);
  CHECK(page1 >= 0);
  CHECK(page1 != page0);  // no physical aliasing
  // Idempotent: re-ensuring inside covered blocks allocates nothing.
  CHECK(t.ensure_capacity(PT + 1, src).ok);
  CHECK(t.ensure_capacity(PT - 1, src).ok);
  CHECK(t.ensure_capacity(0, src).ok);
  CHECK_EQ(t.num_blocks(), 2);
  CHECK_EQ(src.used(), 2);
  CHECK_EQ(t.lookup(0), page0);
  CHECK_EQ(t.lookup(1), page1);
  // Grow to the end: blocks 2..7 (six more pages), last position MAX_SEQ-1.
  CHECK(t.ensure_capacity(MAX_SEQ - 1, src).ok);
  CHECK_EQ(t.num_blocks(), 8);
  CHECK_EQ(src.used(), 8);
  for (int b = 0; b < 8; ++b) CHECK(t.lookup(b) >= 0);
  std::printf("  [ok] minimal prefix growth + idempotency + no aliasing\n");
  return 0;
}

// 3) A block's page is allocated exactly once (no repeated allocation for
//    the same logical block, no matter how many times it is re-ensured).
int test_same_block_never_reallocates() {
  const int MAX_SEQ = 32, PT = 4;
  FakePages src(8);
  KvBlockTable t(MAX_SEQ, PT);
  CHECK(t.ensure_capacity(PT + 1, src).ok);  // blocks 0,1
  const int page1 = t.lookup(1);
  for (int p = PT; p < 2 * PT; ++p) {
    CHECK(t.ensure_capacity(p, src).ok);
    CHECK_EQ(t.lookup(1), page1);  // stable under repeated ensure
  }
  CHECK_EQ(src.used(), 2);  // nothing extra was ever allocated
  std::printf("  [ok] a logical block acquires exactly one physical page\n");
  return 0;
}

// 4) TRANSACTIONAL OOM: a failed ensure — including one that fails MID-CALL
//    after acquiring some tail pages — leaves the table and the source
//    accounting exactly unchanged (no leaked page).
int test_oom_transactional() {
  {
    // Exhaustion: the table already holds all free pages.
    const int MAX_SEQ = 64, PT = 8;
    FakePages src(2);
    KvBlockTable t(MAX_SEQ, PT);
    CHECK(t.ensure_capacity(PT, src).ok);  // blocks 0,1 = all pages
    CHECK_EQ(t.num_blocks(), 2);
    const int used_before = src.used();
    Status s = t.ensure_capacity(2 * PT, src);  // needs block 2: OOM
    CHECK(!s.ok);
    CHECK(s.message.find("failed") != std::string::npos);
    CHECK_EQ(t.num_blocks(), 2);
    CHECK_EQ(src.used(), used_before);  // no leak
  }
  {
    // Mid-call failure: empty table, 1 free page, target needs 3 blocks.
    const int MAX_SEQ = 64, PT = 8;
    FakePages src(1);
    KvBlockTable t(MAX_SEQ, PT);
    Status s = t.ensure_capacity(2 * PT, src);  // blocks 0,1,2
    CHECK(!s.ok);
    CHECK_EQ(t.num_blocks(), 0);  // the partial acquisition was rolled back
    CHECK_EQ(src.used(), 0);      // ... and released
    // The rolled-back page is usable again (LIFO -> the same id).
    CHECK(t.ensure_capacity(PT - 1, src).ok);
    CHECK_EQ(t.num_blocks(), 1);
    CHECK_EQ(src.used(), 1);
    std::printf("  [ok] mid-call OOM rolls back: no leaked page\n");
  }
  return 0;
}

// 5) Position bounds fail loud and change nothing.
int test_position_bounds() {
  const int MAX_SEQ = 64, PT = 8;
  FakePages src(8);  // enough to grow to the end at the last step
  KvBlockTable t(MAX_SEQ, PT);
  CHECK(t.ensure_capacity(PT, src).ok);
  const int nb = t.num_blocks();
  Status s = t.ensure_capacity(-1, src);
  CHECK(!s.ok);
  s = t.ensure_capacity(MAX_SEQ, src);  // one past the end
  CHECK(!s.ok);
  s = t.ensure_capacity(1000000, src);
  CHECK(!s.ok);
  CHECK_EQ(t.num_blocks(), nb);
  CHECK_EQ(src.used(), 2);
  // The LAST valid position is accepted.
  CHECK(t.ensure_capacity(MAX_SEQ - 1, src).ok);
  std::printf("  [ok] position outside [0, max_seq_len) -> Status error\n");
  return 0;
}

// 6) clear releases every page (exact count) and empties the table.
int test_clear_releases_all() {
  const int MAX_SEQ = 64, PT = 8;
  FakePages src(8);
  KvBlockTable t(MAX_SEQ, PT);
  CHECK(t.ensure_capacity(MAX_SEQ - 1, src).ok);
  CHECK_EQ(t.num_blocks(), 8);
  CHECK_EQ(src.used(), 8);
  CHECK(t.clear(src).ok);
  CHECK_EQ(t.num_blocks(), 0);
  CHECK_EQ(src.used(), 0);
  for (int b = 0; b < 8; ++b) CHECK_EQ(t.lookup(b), -1);
  // clear on an empty table is a no-op.
  CHECK(t.clear(src).ok);
  CHECK_EQ(src.used(), 0);
  std::printf("  [ok] clear releases every page (exact count)\n");
  return 0;
}

// 7) Reuse observability: released pages come back LIFO through the source.
int test_reuse_observability() {
  const int MAX_SEQ = 64, PT = 8;
  FakePages src(2);
  KvBlockTable t(MAX_SEQ, PT);
  CHECK(t.ensure_capacity(PT, src).ok);  // acquires pages 1 then 0 (LIFO)
  const int p0 = t.lookup(0), p1 = t.lookup(1);
  CHECK(p0 >= 0 && p1 >= 0 && p0 != p1);
  CHECK(t.clear(src).ok);
  // Fresh growth reuses the freed pages (LIFO: p1's page first).
  CHECK(t.ensure_capacity(PT - 1, src).ok);
  CHECK_EQ(t.lookup(0), p1);
  std::printf("  [ok] released pages are reusable (LIFO identity)\n");
  return 0;
}

}  // namespace

int main() {
  std::printf("test_kv_block_table: CUDALM v0.5 Phase A block table gate\n");
  int rc = 0;
  rc |= test_position_decomposition();
  rc |= test_ensure_grows_prefix_minimally();
  rc |= test_same_block_never_reallocates();
  rc |= test_oom_transactional();
  rc |= test_position_bounds();
  rc |= test_clear_releases_all();
  rc |= test_reuse_observability();
  if (rc != 0) {
    std::fprintf(stderr, "test_kv_block_table: FAIL\n");
    return rc;
  }
  std::printf("test_kv_block_table: PASS\n");
  return 0;
}
