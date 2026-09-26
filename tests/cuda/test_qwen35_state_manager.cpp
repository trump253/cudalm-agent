// CUDALM — v0.5 Phase A: hybrid state-manager device gate (CUDA).
//
// The device-side correctness gates for the new state control plane
// (no checkpoint needed — a small synthetic valid() config):
//
//   * KV page pool: allocate-all unique, OOM on capacity+1, free -> reuse,
//     double-free/invalid rejected, fixed-seed stress, exact accounting;
//   * Delta state pool: acquire-all unique, OOM, double-release, exact
//     accounting;
//   * REAL on-device reuse contamination (not just allocator metadata):
//       - Delta: write a non-zero pattern to a slot's conv + recurrent
//         state in EVERY layer, release, the same physical slot is
//         re-acquired, and the new owner reads back ALL ZEROS (conv AND
//         recurrent, every layer);
//       - KV: write a non-zero pattern to a physical page in EVERY layer
//         (K AND V), release, the page is reused, the new owner reads
//         back ALL ZEROS;
//   * sequence lifecycle via Qwen35StateManager: create A/B -> resources
//     never alias; A reset -> B untouched, A fresh; A retire -> resources
//     reclaimed, the id never valid again and never reissued; C create ->
//     physical resources reusable, SequenceId != A's;
//   * OOM transactional behavior: a failed ensure_kv_capacity (pool
//     exhausted, incl. the cross-page-boundary case) leaves the block
//     table and the pool accounting exactly unchanged; a failed
//     create_sequence (delta pool exhausted) registers nothing.
//
// Provenance: CUDALM-native (v0.5 Phase A).

#include "../../tests/common/check.h"
#include "cudalm/qwen35_state_manager.h"

#include <cstdio>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using namespace cudalm;

namespace {

// Small synthetic hybrid config (valid(); MUST match
// tests/cpu/test_state_pool_formulas.cpp): 8 layers, interval 4 ->
// 2 full (layers 3,7) + 6 linear; max_seq_len 64.
Qwen35Config small_config() {
  Qwen35Config c;
  c.hidden_size = 256;
  c.num_hidden_layers = 8;
  c.intermediate_size = 512;
  c.vocab_size = 1024;
  c.n_heads = 4;
  c.n_kv_heads = 2;
  c.head_dim = 64;
  c.lin_num_k_heads = 4;
  c.lin_num_v_heads = 4;
  c.lin_key_head_dim = 32;
  c.lin_value_head_dim = 32;
  c.lin_conv_kernel_dim = 4;
  c.full_attention_interval = 4;
  c.group_size = 128;
  c.max_seq_len = 64;
  c.eps = 1e-6f;
  c.rope_theta = 1e7f;
  c.partial_rotary_factor = 0.25f;
  c.mrope_section[0] = 4;
  c.mrope_section[1] = 2;
  c.mrope_section[2] = 2;
  return c;
}

// Copy `n` bf16 device elements to host; returns them.
std::vector<__nv_bfloat16> host_copy_bf16(const __nv_bfloat16* dev,
                                          std::size_t n, cudaStream_t s) {
  std::vector<__nv_bfloat16> h(n);
  if (n > 0)
    CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return h;
}

// Write `pattern` (as bf16) to every element of the slice.
void fill_bf16(__nv_bfloat16* dev, std::size_t n, float pattern,
               cudaStream_t s) {
  std::vector<__nv_bfloat16> h(n, __float2bfloat16(pattern));
  if (n > 0)
    CUDA_CHECK(cudaMemcpyAsync(dev, h.data(), n * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
}

void fill_f32(float* dev, std::size_t n, float pattern, cudaStream_t s) {
  std::vector<float> h(n, pattern);
  if (n > 0)
    CUDA_CHECK(cudaMemcpyAsync(dev, h.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
}

bool all_zero_bf16(const std::vector<__nv_bfloat16>& h) {
  for (__nv_bfloat16 x : h)
    if (__bfloat162float(x) != 0.0f) return false;
  return true;
}

bool all_zero_f32(const float* dev, std::size_t n, cudaStream_t s) {
  std::vector<float> h(n, 1.0f);
  if (n > 0)
    CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(float),
                               cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  for (float x : h)
    if (x != 0.0f) return false;
  return true;
}

// Write a non-zero pattern into EVERY layer's state for one slot.
void dirty_delta_slot(Qwen35DeltaStatePool& pool, int slot, cudaStream_t s) {
  for (int l = 0; l < pool.n_linear_layers(); ++l) {
    fill_bf16(pool.conv_mut(l, slot), pool.conv_elems(), 1.5f, s);
    fill_f32(pool.recurrent_mut(l, slot), pool.rec_elems(), 0.25f, s);
  }
}

// True iff the slot reads back ALL ZEROS in every layer (conv + recurrent).
bool delta_slot_all_zero(const Qwen35DeltaStatePool& pool, int slot,
                         cudaStream_t s) {
  for (int l = 0; l < pool.n_linear_layers(); ++l) {
    const std::vector<__nv_bfloat16> conv =
        host_copy_bf16(pool.conv(l, slot), pool.conv_elems(), s);
    if (!all_zero_bf16(conv)) return false;
    if (!all_zero_f32(pool.recurrent(l, slot), pool.rec_elems(), s))
      return false;
  }
  return true;
}

// Write a non-zero pattern into EVERY full-attention layer's K and V for a
// physical page.
void dirty_kv_page(Qwen35KvPagePool& pool, int page, cudaStream_t s) {
  for (int l = 0; l < pool.n_full_layers(); ++l) {
    fill_bf16(pool.k_page_mut(l, page), pool.page_elems(), 2.5f, s);
    fill_bf16(pool.v_page_mut(l, page), pool.page_elems(), -1.5f, s);
  }
}

bool kv_page_all_zero(const Qwen35KvPagePool& pool, int page, cudaStream_t s) {
  for (int l = 0; l < pool.n_full_layers(); ++l) {
    const std::vector<__nv_bfloat16> k =
        host_copy_bf16(pool.k_page(l, page), pool.page_elems(), s);
    if (!all_zero_bf16(k)) return false;
    const std::vector<__nv_bfloat16> v =
        host_copy_bf16(pool.v_page(l, page), pool.page_elems(), s);
    if (!all_zero_bf16(v)) return false;
  }
  return true;
}

// ---- KV page pool ------------------------------------------------------------
int test_kv_pool_allocator() {
  const Qwen35Config cfg = small_config();
  const int PT = 4, CAP = 6;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35KvPagePool pool(cfg, PT, CAP, s);
    // Accounting identity + config-derived bytes (no magic constants).
    CHECK_EQ(pool.capacity_pages(), CAP);
    CHECK_EQ(pool.used_pages(), 0);
    CHECK_EQ(pool.free_pages(), CAP);
    CHECK(pool.bytes_per_page() == qwen35_kv_page_bytes(cfg, PT));
    CHECK_EQ(pool.bytes_per_page(), 4096ull);
    CHECK(pool.total_bytes() == static_cast<std::size_t>(CAP) * 4096);
    CHECK_EQ(pool.used_bytes(), 0ull);
    CHECK_EQ(pool.n_full_layers(), 2);
    CHECK_EQ(pool.full_layer_ordinal(3), 0);
    CHECK_EQ(pool.full_layer_ordinal(7), 1);
    CHECK_EQ(pool.full_layer_ordinal(0), -1);  // linear layer
    CHECK_EQ(pool.layer_of_full_ordinal(0), 3);
    CHECK_EQ(pool.layer_of_full_ordinal(1), 7);
    // Allocate all -> unique.
    std::set<int> seen;
    for (int i = 0; i < CAP; ++i) {
      int id = -1;
      CHECK(pool.allocate_page(&id).ok);
      CHECK(seen.insert(id).second);
      CHECK_EQ(pool.used_pages(), i + 1);
      CHECK_EQ(pool.free_pages(), CAP - i - 1);
      CHECK(pool.used_bytes() == static_cast<std::size_t>(i + 1) * 4096);
    }
    // OOM on capacity+1, no state change.
    int id = -1;
    Status st = pool.allocate_page(&id);
    CHECK(!st.ok);
    CHECK(st.message.find("OOM") != std::string::npos);
    CHECK_EQ(pool.used_pages(), CAP);
    // Free -> reuse (LIFO: the last allocated id comes back first).
    const int last = *seen.rbegin();
    CHECK(pool.free_page(last).ok);
    CHECK_EQ(pool.used_pages(), CAP - 1);
    int again = -1;
    CHECK(pool.allocate_page(&again).ok);
    CHECK_EQ(again, last);
    // Double-free / invalid -> rejected, no state change.
    const int used_now = pool.used_pages();
    CHECK(pool.free_page(again).ok);   // the one legitimate free
    CHECK(!pool.free_page(again).ok);  // double-free: rejected
    CHECK(!pool.free_page(-1).ok);     // negative id: rejected
    CHECK(!pool.free_page(CAP).ok);    // out of range: rejected
    CHECK_EQ(pool.used_pages(), used_now - 1);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] KV pool: unique/OOM/reuse/double-free/accounting\n");
  return 0;
}

int test_kv_pool_stress() {
  const Qwen35Config cfg = small_config();
  const int PT = 4, CAP = 32;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35KvPagePool pool(cfg, PT, CAP, s);
    std::mt19937 rng(20260925u);
    std::uniform_int_distribution<int> op_dist(0, 1);
    std::uniform_int_distribution<int> id_dist(0, CAP + 3);
    std::set<int> model;
    for (int i = 0; i < 100000; ++i) {
      if (op_dist(rng) == 0 || model.size() < CAP) {
        int id = -1;
        Status st = pool.allocate_page(&id);
        if (model.size() < CAP) {
          CHECK(st.ok);
          CHECK(model.insert(id).second);
        } else {
          CHECK(!st.ok);
        }
      } else {
        const int vid = id_dist(rng);
        const bool expect_ok =
            vid >= 0 && vid < CAP && model.count(vid) > 0;
        CHECK(pool.free_page(vid).ok == expect_ok);
        if (expect_ok) model.erase(vid);
      }
      CHECK_EQ(pool.used_pages(), static_cast<int>(model.size()));
      CHECK_EQ(pool.free_pages(), CAP - static_cast<int>(model.size()));
    }
    pool.reset();
    CHECK_EQ(pool.used_pages(), 0);
    CHECK_EQ(pool.free_pages(), CAP);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] KV pool: fixed-seed stress (100k ops, cap 32)\n");
  return 0;
}

// ---- Delta state pool ----------------------------------------------------------
int test_delta_pool_allocator() {
  const Qwen35Config cfg = small_config();
  const int CAP = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35DeltaStatePool pool(cfg, CAP, s);
    CHECK_EQ(pool.capacity_slots(), CAP);
    CHECK_EQ(pool.used_slots(), 0);
    CHECK_EQ(pool.free_slots(), CAP);
    CHECK(pool.bytes_per_slot() == qwen35_delta_slot_bytes(cfg));
    CHECK_EQ(pool.bytes_per_slot(), 112128ull);
    CHECK(pool.total_bytes() == static_cast<std::size_t>(CAP) * 112128);
    CHECK_EQ(pool.n_linear_layers(), 6);
    CHECK_EQ(pool.linear_layer_ordinal(0), 0);
    CHECK_EQ(pool.linear_layer_ordinal(3), -1);  // full layer
    // linear ordinals: 0->0, 1->1, 2->2, 4->3, 5->4, 6->5 (3 and 7 are full)
    CHECK_EQ(pool.layer_of_linear_ordinal(5), 6);
    CHECK_EQ(pool.linear_layer_ordinal(6), 5);
    // Acquire all -> unique.
    std::set<int> seen;
    for (int i = 0; i < CAP; ++i) {
      int sl = -1;
      CHECK(pool.acquire_slot(&sl).ok);
      CHECK(seen.insert(sl).second);
      CHECK_EQ(pool.used_slots(), i + 1);
    }
    int sl = -1;
    Status st = pool.acquire_slot(&sl);
    CHECK(!st.ok);
    CHECK(st.message.find("OOM") != std::string::npos);
    // Release -> reuse LIFO; double-release / invalid rejected.
    const int last = *seen.rbegin();
    CHECK(pool.release_slot(last).ok);
    CHECK(!pool.release_slot(last).ok);  // double release
    CHECK(!pool.release_slot(-1).ok);
    CHECK(!pool.release_slot(CAP).ok);
    CHECK_EQ(pool.used_slots(), CAP - 1);
    int again = -1;
    CHECK(pool.acquire_slot(&again).ok);
    CHECK_EQ(again, last);
    // reset_slot on a live slot is accepted; on a dead slot rejected.
    CHECK(pool.release_slot(again).ok);
    CHECK(!pool.reset_slot(again).ok);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] Delta pool: unique/OOM/reuse/double-release/"
              "accounting\n");
  return 0;
}

// ---- REAL on-device reuse contamination ----------------------------------------
int test_delta_reuse_contamination() {
  const Qwen35Config cfg = small_config();
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35DeltaStatePool pool(cfg, 2, s);
    int a = -1, b = -1;
    CHECK(pool.acquire_slot(&a).ok);
    CHECK(pool.acquire_slot(&b).ok);
    // Owner A writes a non-zero pattern into EVERY layer of its slot.
    dirty_delta_slot(pool, a, s);
    CHECK(!delta_slot_all_zero(pool, a, s));  // sanity: the pattern landed
    // Release A; B (untouched) must still be zero — no cross-slot leak.
    CHECK(pool.release_slot(a).ok);
    CHECK(delta_slot_all_zero(pool, b, s));
    // The SAME physical slot is re-acquired by owner C (LIFO -> a).
    int c = -1;
    CHECK(pool.acquire_slot(&c).ok);
    CHECK_EQ(c, a);
    // C must read back ALL ZEROS in conv AND recurrent, every layer.
    CHECK(delta_slot_all_zero(pool, c, s));
    // reset_slot in-place: dirty, reset, zero again.
    dirty_delta_slot(pool, c, s);
    CHECK(pool.reset_slot(c).ok);
    CHECK(delta_slot_all_zero(pool, c, s));
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] Delta reuse contamination: reused slot reads back "
              "ALL ZEROS (conv+recurrent, all layers)\n");
  return 0;
}

int test_kv_reuse_contamination() {
  const Qwen35Config cfg = small_config();
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35KvPagePool pool(cfg, PT, 2, s);
    int a = -1, b = -1;
    CHECK(pool.allocate_page(&a).ok);
    CHECK(pool.allocate_page(&b).ok);
    // Owner A dirties EVERY layer (K and V) of its page.
    dirty_kv_page(pool, a, s);
    CHECK(!kv_page_all_zero(pool, a, s));  // sanity: pattern landed
    CHECK(kv_page_all_zero(pool, b, s));   // no cross-page leak
    CHECK(pool.free_page(a).ok);
    // The SAME physical page is re-acquired (LIFO -> a) and must be clean.
    int c = -1;
    CHECK(pool.allocate_page(&c).ok);
    CHECK_EQ(c, a);
    CHECK(kv_page_all_zero(pool, c, s));
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] KV reuse contamination: reused page reads back ALL "
              "ZEROS (K+V, all layers)\n");
  return 0;
}

// ---- block table on the REAL pool ------------------------------------------------
int test_block_table_on_real_pool() {
  const Qwen35Config cfg = small_config();  // max_seq_len = 64
  const int PT = 4;  // 16 logical blocks max
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35KvPagePool pool(cfg, PT, 16, s);
    KvBlockTable t(cfg.max_seq_len, PT);
    CHECK_EQ(t.max_blocks(), 16);
    // Boundary matrix: 0, PT-1, PT, PT+1, last valid, max.
    CHECK(t.ensure_capacity(0, pool).ok);
    CHECK_EQ(t.num_blocks(), 1);
    CHECK_EQ(pool.used_pages(), 1);
    CHECK(t.ensure_capacity(PT - 1, pool).ok);  // still block 0
    CHECK_EQ(pool.used_pages(), 1);
    CHECK(t.ensure_capacity(PT, pool).ok);      // crosses into block 1
    CHECK_EQ(t.num_blocks(), 2);
    CHECK_EQ(pool.used_pages(), 2);
    CHECK(t.ensure_capacity(PT + 1, pool).ok);
    CHECK_EQ(pool.used_pages(), 2);
    // The LAST valid position is accepted and exhausts the pool exactly
    // (16 blocks = 16 pages).
    CHECK(t.ensure_capacity(cfg.max_seq_len - 1, pool).ok);
    CHECK_EQ(t.num_blocks(), 16);
    CHECK_EQ(pool.used_pages(), 16);
    CHECK_EQ(pool.free_pages(), 0);
    // Position == max_seq_len is invalid (one past the end) — and nothing
    // is allocated by a failing call.
    CHECK(!t.ensure_capacity(cfg.max_seq_len, pool).ok);
    CHECK(!t.ensure_capacity(-1, pool).ok);
    CHECK_EQ(pool.used_pages(), 16);
    // One shared table across ALL layers: the same page id in the table
    // addresses the same block in every full-attention layer (by pool
    // construction — the pool has ONE page id space).
    const int page5 = t.lookup(5);
    CHECK(page5 >= 0);
    // clear releases all 16 pages.
    CHECK(t.clear(pool).ok);
    CHECK_EQ(t.num_blocks(), 0);
    CHECK_EQ(pool.used_pages(), 0);
    // Mid-growth OOM on a TIGHT pool: 3 pages, growth to block 3 needs a
    // fourth. Table and accounting stay exactly unchanged, no leak.
    Qwen35KvPagePool tight(cfg, PT, 3, s);
    KvBlockTable t2(cfg.max_seq_len, PT);
    CHECK(t2.ensure_capacity(3 * PT - 1, tight).ok);  // blocks 0,1,2
    CHECK_EQ(t2.num_blocks(), 3);
    CHECK_EQ(tight.used_pages(), 3);
    CHECK_EQ(tight.free_pages(), 0);
    Status st = t2.ensure_capacity(3 * PT, tight);    // needs block 3
    CHECK(!st.ok);
    CHECK_EQ(t2.num_blocks(), 3);   // unchanged
    CHECK_EQ(tight.used_pages(), 3);  // no leak
    CHECK_EQ(tight.free_pages(), 0);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] block table on real pool: boundaries, OOM no-leak, "
              "single shared table\n");
  return 0;
}

// ---- sequence lifecycle -----------------------------------------------------------
int test_sequence_lifecycle() {
  const Qwen35Config cfg = small_config();
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/4, /*delta_slots=*/2, s);
    CHECK_EQ(mgr.num_live_sequences(), 0);
    CHECK_EQ(mgr.next_sequence_id(), static_cast<SequenceId>(1));
    const std::size_t total = mgr.total_state_bytes();
    CHECK(total ==
          static_cast<std::size_t>(4) * 4096 +
          static_cast<std::size_t>(2) * 112128);
    CHECK_EQ(mgr.used_state_bytes(), 0ull);

    SequenceId a = 0, b = 0;
    CHECK(mgr.create_sequence(&a).ok);
    CHECK(mgr.create_sequence(&b).ok);
    CHECK_EQ(a, static_cast<SequenceId>(1));
    CHECK_EQ(b, static_cast<SequenceId>(2));  // monotone
    CHECK_EQ(mgr.num_live_sequences(), 2);
    const SequenceState* ra = mgr.lookup(a);
    const SequenceState* rb = mgr.lookup(b);
    CHECK(ra != nullptr && rb != nullptr);
    CHECK_EQ(ra->length, 0);
    CHECK_EQ(rb->length, 0);
    CHECK_EQ(ra->block_table.num_blocks(), 0);
    // Resources never alias: distinct delta slots; (no pages yet).
    CHECK(ra->delta_slot != rb->delta_slot);
    CHECK(ra->delta_slot >= 0 && ra->delta_slot < mgr.delta_pool().capacity_slots());
    CHECK(rb->delta_slot >= 0 && rb->delta_slot < mgr.delta_pool().capacity_slots());
    // used_state_bytes = 2 delta slots (no pages yet).
    CHECK(mgr.used_state_bytes() == 2ull * 112128);

    // Grow A across a page boundary; B stays untouched.
    CHECK(mgr.ensure_kv_capacity(a, PT).ok);  // blocks 0,1 -> 2 pages
    CHECK_EQ(ra->block_table.num_blocks(), 2);
    CHECK(mgr.delta_pool().used_slots() == 2);
    // B: write a pattern into its delta slot; A's slot must be untouched.
    {
      Qwen35DeltaStatePool& dp =
          const_cast<Qwen35DeltaStatePool&>(mgr.delta_pool());
      for (int l = 0; l < dp.n_linear_layers(); ++l)
        fill_bf16(dp.conv_mut(l, rb->delta_slot), dp.conv_elems(), 3.5f, s);
    }
    CHECK(mgr.reset_sequence(a).ok);  // A: length 0, pages released, slot
                                      // zeroed; B MUST be unchanged
    ra = mgr.lookup(a);
    CHECK_EQ(ra->length, 0);
    CHECK_EQ(ra->block_table.num_blocks(), 0);
    CHECK_EQ(mgr.kv_pool().used_pages(), 0);
    {
      const Qwen35DeltaStatePool& dp = mgr.delta_pool();
      CHECK(delta_slot_all_zero(dp, ra->delta_slot, s));
      // B's pattern survived A's reset (no cross-sequence zeroing).
      bool b_dirty = false;
      for (int l = 0; l < dp.n_linear_layers(); ++l) {
        const std::vector<__nv_bfloat16> h =
            host_copy_bf16(dp.conv(l, rb->delta_slot), dp.conv_elems(), s);
        if (!all_zero_bf16(h)) b_dirty = true;
      }
      CHECK(b_dirty);
    }
    CHECK_EQ(ra->id, a);  // same id, still live

    // Retire A: resources reclaimed; the id is dead.
    CHECK(mgr.retire_sequence(a).ok);
    CHECK(mgr.lookup(a) == nullptr);
    CHECK(!mgr.ensure_kv_capacity(a, 0).ok);
    CHECK(!mgr.set_length(a, 1).ok);
    CHECK(!mgr.advance(a, 1).ok);
    CHECK(!mgr.reset_sequence(a).ok);
    CHECK(!mgr.retire_sequence(a).ok);  // double retire
    CHECK_EQ(mgr.kv_pool().used_pages(), 0);
    CHECK_EQ(mgr.delta_pool().used_slots(), 1);
    CHECK(mgr.used_state_bytes() == 1ull * 112128);

    // C create: physical resources ARE reusable (the freed slot comes
    // back LIFO) but the SequenceId is NOT A's (monotone, non-reusing).
    SequenceId c = 0;
    CHECK(mgr.create_sequence(&c).ok);
    CHECK_EQ(c, static_cast<SequenceId>(3));
    CHECK(mgr.lookup(c)->delta_slot == ra->delta_slot);  // slot reused
    CHECK(c != a);
    CHECK_EQ(mgr.delta_pool().used_slots(), 2);

    // Length metadata: set/advance contract + bounds.
    CHECK(mgr.set_length(c, 10).ok);
    CHECK(mgr.lookup(c)->length == 10);
    CHECK(mgr.advance(c, 5).ok);
    CHECK(mgr.lookup(c)->length == 15);
    CHECK(!mgr.advance(c, cfg.max_seq_len).ok);  // would exceed max
    CHECK(mgr.lookup(c)->length == 15);
    CHECK(!mgr.set_length(c, -1).ok);
    CHECK(!mgr.set_length(c, cfg.max_seq_len + 1).ok);
    CHECK(mgr.lookup(c)->length == 15);
    // ensure at position beyond max_seq_len -> error, table unchanged.
    CHECK(!mgr.ensure_kv_capacity(c, cfg.max_seq_len).ok);
    CHECK_EQ(mgr.lookup(c)->block_table.num_blocks(), 0);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] lifecycle: create/no-alias/reset/retire/id "
              "non-reuse/length contract\n");
  return 0;
}

// ---- OOM transactional behavior -----------------------------------------------------
// 1) KV OOM mid-growth: the pool (2 pages) is exhausted by A and B (one
//    page each); A crossing the page boundary needs a page that does not
//    exist. The failed ensure must leave the block table AND the pool
//    accounting exactly unchanged (no leaked page, no partial growth).
int test_oom_kv_transactional() {
  const Qwen35Config cfg = small_config();  // max_seq_len = 64
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/2, /*delta_slots=*/2, s);
    SequenceId a = 0, b = 0;
    CHECK(mgr.create_sequence(&a).ok);
    CHECK(mgr.create_sequence(&b).ok);
    // Both sequences take one page each: the pool is now exhausted.
    CHECK(mgr.ensure_kv_capacity(a, PT - 1).ok);
    CHECK(mgr.ensure_kv_capacity(b, PT - 1).ok);
    CHECK_EQ(mgr.kv_pool().used_pages(), 2);
    CHECK_EQ(mgr.kv_pool().free_pages(), 0);
    const std::size_t used_before = mgr.used_state_bytes();
    // A crosses the boundary: block 1 needs a page that does not exist.
    Status st = mgr.ensure_kv_capacity(a, PT);
    CHECK(!st.ok);
    CHECK(st.message.find("failed") != std::string::npos);
    // Exactly unchanged: A's table, the pool accounting, manager bytes.
    CHECK_EQ(mgr.lookup(a)->block_table.num_blocks(), 1);
    CHECK_EQ(mgr.kv_pool().used_pages(), 2);
    CHECK_EQ(mgr.kv_pool().free_pages(), 0);
    CHECK(mgr.used_state_bytes() == used_before);
    // No corruption: A's owned block still resolves; B is untouched.
    CHECK(mgr.lookup(a)->block_table.lookup(0) >= 0);
    CHECK_EQ(mgr.lookup(b)->block_table.num_blocks(), 1);
    // After B retires (reclaiming its page), the same growth succeeds.
    CHECK(mgr.retire_sequence(b).ok);
    CHECK(mgr.ensure_kv_capacity(a, PT).ok);
    CHECK_EQ(mgr.lookup(a)->block_table.num_blocks(), 2);
    CHECK_EQ(mgr.kv_pool().used_pages(), 2);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] KV OOM transactional: failed cross-boundary ensure "
              "leaves everything unchanged\n");
  return 0;
}

// 2) Delta OOM on create: the delta pool is exhausted; a failed
//    create_sequence must register NOTHING (no id issued, no half-record,
//    no leaked slot) and the next create must succeed monotonically.
int test_oom_create_transactional() {
  const Qwen35Config cfg = small_config();
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/2, /*delta_slots=*/1, s);
    SequenceId a = 0;
    CHECK(mgr.create_sequence(&a).ok);
    CHECK_EQ(a, static_cast<SequenceId>(1));
    CHECK_EQ(mgr.num_live_sequences(), 1);
    // Second create: no free slot.
    SequenceId b = 0;
    Status st = mgr.create_sequence(&b);
    CHECK(!st.ok);
    CHECK(st.message.find("OOM") != std::string::npos);
    CHECK_EQ(b, static_cast<SequenceId>(0));  // nothing written on failure
    CHECK_EQ(mgr.num_live_sequences(), 1);  // no half-record
    CHECK(mgr.lookup(2) == nullptr);        // the would-be id is unissued
    CHECK_EQ(mgr.delta_pool().used_slots(), 1);  // no leaked slot
    CHECK_EQ(mgr.delta_pool().free_slots(), 0);
    CHECK(mgr.used_state_bytes() == 1ull * 112128);
    // Recovery: retire A, then B (well: the next create) succeeds and the
    // id sequence stays monotone (never 1 again).
    CHECK(mgr.retire_sequence(a).ok);
    SequenceId c = 0;
    CHECK(mgr.create_sequence(&c).ok);
    CHECK_EQ(c, static_cast<SequenceId>(2));  // monotone, non-reusing
    CHECK_EQ(mgr.num_live_sequences(), 1);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] create OOM transactional: nothing registered, ids "
              "stay monotone\n");
  return 0;
}

}  // namespace

int main() {
  std::printf("test_qwen35_state_manager: CUDALM v0.5 Phase A device gate "
              "(small synthetic config)\n");
  int rc = 0;
  rc |= test_kv_pool_allocator();
  rc |= test_kv_pool_stress();
  rc |= test_delta_pool_allocator();
  rc |= test_delta_reuse_contamination();
  rc |= test_kv_reuse_contamination();
  rc |= test_block_table_on_real_pool();
  rc |= test_sequence_lifecycle();
  rc |= test_oom_kv_transactional();
  rc |= test_oom_create_transactional();
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_state_manager: FAIL\n");
    return rc;
  }
  std::printf("test_qwen35_state_manager: PASS\n");
  return 0;
}
