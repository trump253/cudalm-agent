// CUDALM — v0.5 Phase A: FixedIdPool allocator-core gate (CPU-only).
//
// Pinned contract (see include/cudalm/fixed_id_pool.h):
//   * allocate-all -> all ids unique, exactly [0, capacity);
//   * capacity+1 acquires -> OOM, no state change;
//   * free -> the id is reusable (LIFO: most recently freed first);
//   * double-free / out-of-range -> Status error, no state change;
//   * FIXED-SEED MIXED allocate/free stress (uniform op coin flip,
//     live-biased releases so the pool drains and climbs across the
//     whole occupancy range) vs an independent host live-set model:
//     unique live ids, full live-set equality, exact accounting every
//     step, OOM no state change, valid release exact, invalid/double
//     rejected, released ids actually reused, occupancy sweep covered;
//   * accounting exact at every step (capacity == used + free).

#include "../../tests/common/check.h"
#include "cudalm/fixed_id_pool.h"

#include <cstdio>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace cudalm;

namespace {

int test_allocate_all_unique() {
  const int CAP = 8;
  FixedIdPool p(CAP);
  std::set<int> seen;
  for (int i = 0; i < CAP; ++i) {
    int id = -1;
    Status s = p.acquire(&id);
    CHECK(s.ok);
    CHECK(id >= 0 && id < CAP);
    CHECK(seen.insert(id).second);  // unique
    CHECK(p.used() == i + 1);
    CHECK(p.free_count() == CAP - (i + 1));
  }
  CHECK(p.used() == CAP);
  CHECK(p.free_count() == 0);
  // OOM on capacity+1, no state change.
  int id = -1;
  Status s = p.acquire(&id);
  CHECK(!s.ok);
  CHECK(s.message.find("OOM") != std::string::npos);
  CHECK(p.used() == CAP);
  CHECK(p.free_count() == 0);
  std::printf("  [ok] allocate-all unique + OOM on capacity+1\n");
  return 0;
}

int test_reuse_and_lifo_order() {
  FixedIdPool p(4);
  int a, b, c;
  CHECK(p.acquire(&a).ok);
  CHECK(p.acquire(&b).ok);
  CHECK(p.acquire(&c).ok);
  CHECK(p.release(a).ok);
  CHECK(p.release(b).ok);
  // LIFO: b (freed most recently) comes back first, then a.
  int r1, r2;
  CHECK(p.acquire(&r1).ok);
  CHECK_EQ(r1, b);
  CHECK(p.acquire(&r2).ok);
  CHECK_EQ(r2, a);
  // A released id may be reused repeatedly.
  CHECK(p.release(c).ok);
  int r3;
  CHECK(p.acquire(&r3).ok);
  CHECK_EQ(r3, c);
  std::printf("  [ok] reuse + LIFO free order\n");
  return 0;
}

int test_double_free_and_invalid() {
  FixedIdPool p(4);
  int x;
  CHECK(p.acquire(&x).ok);
  const int used_before = p.used();
  // Double-free: x already released.
  CHECK(p.release(x).ok);
  Status s = p.release(x);
  CHECK(!s.ok);
  CHECK(s.message.find("not live") != std::string::npos);
  CHECK(p.used() == used_before - 1);
  // Out-of-range ids.
  CHECK(!p.release(-1).ok);
  CHECK(!p.release(4).ok);
  CHECK(!p.release(1000000).ok);
  CHECK(!p.is_live(-1));
  CHECK(!p.is_live(4));
  std::printf("  [ok] double-free / out-of-range rejected (no state change)\n");
  return 0;
}

int test_reset() {
  FixedIdPool p(5);
  std::vector<int> ids;
  for (int i = 0; i < 4; ++i) {
    int x;
    CHECK(p.acquire(&x).ok);
    ids.push_back(x);
  }
  CHECK(p.used() == 4);
  p.reset();
  CHECK(p.used() == 0);
  CHECK(p.free_count() == 5);
  for (int x : ids) CHECK(!p.is_live(x));
  // After reset everything is reacquirable.
  for (int i = 0; i < 5; ++i) {
    int x;
    CHECK(p.acquire(&x).ok);
  }
  CHECK(p.used() == 5);
  // A second reset releases those 5 again; a third reset (nothing live)
  // is a no-op.
  p.reset();
  CHECK(p.used() == 0);
  CHECK(p.free_count() == 5);
  p.reset();
  CHECK(p.used() == 0);
  std::printf("  [ok] reset releases all live ids\n");
  return 0;
}

int test_zero_capacity() {
  FixedIdPool p(0);
  int x;
  CHECK(!p.acquire(&x).ok);
  CHECK(p.used() == 0);
  CHECK(p.free_count() == 0);
  std::printf("  [ok] zero-capacity pool: every acquire is OOM\n");
  return 0;
}

// Fixed-seed MIXED allocate/free stress: the op choice is a uniform
// coin flip (NO bias toward acquire), so the pool wanders across all
// occupancy levels and releases happen at low/mid/high occupancy alike.
// An independent host live-set model must mirror the pool at every step:
//   * live ids unique (model.insert must succeed);
//   * pool live set == model live set (full equality checked periodically
//     and at the end, not just the size);
//   * used/free/capacity exact at EVERY step;
//   * acquire at full pool -> OOM, no state change;
//   * valid release exact (ok + model updated);
//   * invalid / out-of-range / double release rejected, no state change;
//   * released ids ARE reused (reuse counter must be > 0 at the end).
int test_random_stress() {
  const int CAP = 64;
  const int OPS = 200000;
  const std::uint32_t SEED = 20260925u;
  FixedIdPool p(CAP);
  std::mt19937 rng(SEED);
  std::uniform_int_distribution<int> op_dist(0, 1);    // 0 = acquire, 1 = release
  std::uniform_int_distribution<int> cand_dist(0, 9);  // release-id choice
  std::uniform_int_distribution<int> id_dist(0, CAP + 5);  // includes invalid
  std::set<int> model;       // the independent live-set model
  std::set<int> ever_issued; // every id ever handed out (reuse tracking)
  long n_acquire_ok = 0, n_oom = 0, n_release_ok = 0,
       n_release_rejected = 0, n_reuse = 0;
  int min_occ = CAP, max_occ = 0;
  for (int i = 0; i < OPS; ++i) {
    if (op_dist(rng) == 0) {
      // ACQUIRE (no occupancy bias).
      int got = -1;
      Status s = p.acquire(&got);
      if (model.size() < CAP) {
        CHECK(s.ok);
        CHECK(got >= 0 && got < CAP);
        CHECK(model.insert(got).second);  // unique
        if (!ever_issued.insert(got).second) ++n_reuse;  // reissued id
        ++n_acquire_ok;
      } else {
        CHECK(!s.ok);  // OOM: no state change (model untouched by design)
        ++n_oom;
      }
    } else {
      // RELEASE: usually a LIVE id (deterministic pick from the model, so
      // the pool actually drains across the occupancy range); sometimes a
      // raw random candidate (released / out-of-range -> must be rejected).
      int vid = -1;
      if (cand_dist(rng) < 9 && !model.empty()) {
        auto it = model.begin();
        std::advance(it, static_cast<long>(rng()) % model.size());
        vid = *it;
      } else {
        vid = id_dist(rng);
      }
      Status s = p.release(vid);
      if (vid >= 0 && vid < CAP && model.count(vid)) {
        CHECK(s.ok);
        model.erase(vid);
        ++n_release_ok;
      } else {
        CHECK(!s.ok);  // invalid, out-of-range, or double release
        ++n_release_rejected;
      }
    }
    // Exact accounting at EVERY step + occupancy-range tracking.
    const int occ = p.used();
    if (occ < min_occ) min_occ = occ;
    if (occ > max_occ) max_occ = occ;
    CHECK(occ == static_cast<int>(model.size()));
    CHECK(p.free_count() == CAP - occ);
    CHECK(p.capacity() == occ + p.free_count());
    // Full live-set equality (not just size) at checkpoints.
    if (i % 5000 == 0 || i == OPS - 1) {
      const std::vector<int> live = p.live_ids();
      CHECK(live.size() == model.size());
      std::set<int> live_set(live.begin(), live.end());
      CHECK(live_set == model);
    }
  }
  // The mixed workload must have exercised EVERY branch (coverage proof):
  // normal acquires, OOM-at-full, valid releases, rejected releases,
  // actual reuse of previously issued (released) ids, and a real
  // occupancy sweep (down below half and up to full, not just full-only).
  CHECK(n_acquire_ok > 0);
  CHECK(n_oom > 0);
  CHECK(n_release_ok > 0);
  CHECK(n_release_rejected > 0);
  CHECK(n_reuse > 0);
  CHECK(min_occ * 2 < CAP);  // low occupancy was actually visited
  CHECK(max_occ == CAP);     // ... and so was full
  std::printf("  [ok] fixed-seed mixed stress (%d ops, cap %d: %d live at "
              "end, occupancy range [%d, %d]; %ld acquire-ok, %ld OOM, "
              "%ld release-ok, %ld rejected, %ld reuses)\n",
              OPS, CAP, static_cast<int>(model.size()), min_occ, max_occ,
              n_acquire_ok, n_oom, n_release_ok, n_release_rejected, n_reuse);
  return 0;
}


}  // namespace

int main() {
  std::printf("test_fixed_id_pool: CUDALM v0.5 Phase A allocator core gate\n");
  int rc = 0;
  rc |= test_allocate_all_unique();
  rc |= test_reuse_and_lifo_order();
  rc |= test_double_free_and_invalid();
  rc |= test_reset();
  rc |= test_zero_capacity();
  rc |= test_random_stress();
  if (rc != 0) {
    std::fprintf(stderr, "test_fixed_id_pool: FAIL\n");
    return rc;
  }
  std::printf("test_fixed_id_pool: PASS\n");
  return 0;
}
