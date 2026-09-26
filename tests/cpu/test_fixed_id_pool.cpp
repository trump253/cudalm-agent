// CUDALM — v0.5 Phase A: FixedIdPool allocator-core gate (CPU-only).
//
// Pinned contract (see include/cudalm/fixed_id_pool.h):
//   * allocate-all -> all ids unique, exactly [0, capacity);
//   * capacity+1 acquires -> OOM, no state change;
//   * free -> the id is reusable (LIFO: most recently freed first);
//   * double-free / out-of-range -> Status error, no state change;
//   * random fixed-seed stress with an independent live-set model;
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

// Fixed-seed stress: an independent live-set model must mirror the pool at
// every step (uniqueness + exact accounting invariants).
int test_random_stress() {
  const int CAP = 64;
  const int OPS = 200000;
  const std::uint32_t SEED = 20260925u;
  FixedIdPool p(CAP);
  std::mt19937 rng(SEED);
  std::uniform_int_distribution<int> op_dist(0, 1);   // 0 = acquire, 1 = release
  std::uniform_int_distribution<int> id_dist(0, CAP + 5);  // includes invalid ids
  std::set<int> model;  // the independent live-set model
  for (int i = 0; i < OPS; ++i) {
    if (op_dist(rng) == 0 || model.size() < CAP) {
      int got = -1;
      Status s = p.acquire(&got);
      if (model.size() < CAP) {
        CHECK(s.ok);
        CHECK(got >= 0 && got < CAP);
        CHECK(model.insert(got).second);  // unique
      } else {
        CHECK(!s.ok);
      }
    } else {
      const int vid = id_dist(rng);
      Status s = p.release(vid);
      if (vid >= 0 && vid < CAP && model.count(vid)) {
        CHECK(s.ok);
        model.erase(vid);
      } else {
        CHECK(!s.ok);  // invalid or double-free
      }
    }
    if (i % 10000 == 0) {
      CHECK(p.used() == static_cast<int>(model.size()));
      CHECK(p.capacity() == p.used() + p.free_count());
    }
  }
  CHECK(p.used() == static_cast<int>(model.size()));
  CHECK(p.capacity() == p.used() + p.free_count());
  std::printf("  [ok] fixed-seed stress (%d ops, cap %d: %d live at end)\n",
              OPS, CAP, static_cast<int>(model.size()));
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
