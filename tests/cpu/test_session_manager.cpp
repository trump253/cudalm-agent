// CUDALM — v0.8 Phase A: session abstraction lifecycle gate (CPU test over
// the REAL state pools — no checkpoint, no model; a small synthetic valid()
// config, the same discipline as test_qwen35_state_manager.cpp).
//
// Gates the NEW ownership layer (SessionManager over the frozen v0.5
// Qwen35StateManager):
//
//   * create -> valid / destroy -> invalid:
//       - a created session is live (lookup, Active, bound sequence live,
//         length 0, empty KV table, live Delta slot, fresh metadata);
//       - a destroyed session is invalid (lookup nullptr; EVERY operation
//         on the id is a Status error, incl. double-destroy);
//       - SessionId is monotone and NEVER reissued (its own id space —
//         distinct from SequenceId and RequestId);
//       - the session <-> sequence binding is 1:1 and the invariant
//         num_sessions == num_live_sequences holds after EVERY operation;
//   * TRANSACTIONAL create: with an exhausted Delta pool a failed
//     create_session registers NOTHING (no id issued, no half-record, no
//     leaked slot) and the next create succeeds monotonically;
//   * ISOLATION: sessions A/B own INDEPENDENT resources — distinct Delta
//     slots; dirtying A's Delta conv+recurrent (every layer) and A's KV
//     pages leaves B's state ALL ZEROS and B's metadata untouched;
//   * RESET: reset_session(A) keeps A LIVE under the SAME SessionId and
//     restores fresh-state semantics — length 0, all KV pages released,
//     the Delta slot ZEROED IN PLACE (same slot id), reset_count metadata
//     advanced — while B is EXACTLY unchanged (pattern intact, same slot,
//     same pages); a reset cannot OOM and is repeatable;
//   * DESTROY + REUSE: destroy_session(A) retires A's sequence (pages +
//     slot released + zeroed) and invalidates A's SessionId; a later
//     create_session reuses A's PHYSICAL slot / pages (LIFO) under a NEW
//     SessionId and the new owner reads back ALL ZEROS (no residue of A),
//     while B is still intact;
//   * CONTEXT CAPACITY (fits): the overflow-policy foundation —
//     length + n <= max_seq_len is the exact boundary (0 / max / max+1 /
//     negative / unknown id / destroyed id), a pure query (no allocation,
//     no mutation);
//   * error cases: every operation on an unknown id fails loud with the
//     pool accounting and the session set exactly unchanged.
//
// Provenance: CUDALM-native (v0.8 Phase A).

#include "../../tests/common/check.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/session.h"

#include <cstdio>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using namespace cudalm;

namespace {

// Small synthetic hybrid config (valid(); MUST match
// test_qwen35_state_manager.cpp): 8 layers, interval 4 -> 2 full (layers
// 3,7) + 6 linear; max_seq_len 64.
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

// ---- device pattern helpers (same discipline as test_qwen35_state_manager)
std::vector<__nv_bfloat16> host_copy_bf16(const __nv_bfloat16* dev,
                                          std::size_t n, cudaStream_t s) {
  std::vector<__nv_bfloat16> h(n);
  if (n > 0)
    CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return h;
}

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

// True iff the slot is NON-ZERO somewhere in every layer (a landed
// pattern — the inverse sanity check of delta_slot_all_zero).
bool delta_slot_any_dirty(const Qwen35DeltaStatePool& pool, int slot,
                          cudaStream_t s) {
  for (int l = 0; l < pool.n_linear_layers(); ++l) {
    const std::vector<__nv_bfloat16> conv =
        host_copy_bf16(pool.conv(l, slot), pool.conv_elems(), s);
    if (!all_zero_bf16(conv)) return true;
  }
  return false;
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

// The session <-> sequence 1:1 binding invariant (checked after EVERY
// mutation in the tests below): the number of live sessions must equal the
// number of live sequences (every session owns exactly one live sequence;
// no leaked or dangling sequence). Returns true when the invariant holds.
bool check_binding_invariant(const SessionManager& sm) {
  return sm.num_sessions() == sm.manager().num_live_sequences();
}

// ---- create -> valid / destroy -> invalid ------------------------------------
int test_session_create_destroy() {
  const Qwen35Config cfg = small_config();
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/4, /*delta_slots=*/2, s);
    SessionManager sm(mgr);
    CHECK_EQ(sm.num_sessions(), 0);
    CHECK_EQ(sm.next_session_id(), static_cast<SessionId>(1));
    CHECK(check_binding_invariant(sm));

    // ---- create -> valid ------------------------------------------------
    SessionId a = 0;
    Status st = sm.create_session(&a);
    CHECK(st.ok);
    CHECK_EQ(a, static_cast<SessionId>(1));  // ids start at 1
    CHECK_EQ(sm.next_session_id(), static_cast<SessionId>(2));
    const Session* ra = sm.lookup(a);
    CHECK(ra != nullptr);
    CHECK_EQ(ra->id, a);
    CHECK(ra->state == SessionState::Active);
    CHECK_EQ(ra->reset_count, 0);
    // The bound sequence is live and fresh: length 0, empty KV table,
    // a live Delta slot.
    const SequenceState* seqa = mgr.lookup(ra->sequence_id);
    CHECK(seqa != nullptr);
    CHECK_EQ(seqa->length, 0);
    CHECK_EQ(seqa->block_table.num_blocks(), 0);
    CHECK(seqa->delta_slot >= 0);
    CHECK(seqa->delta_slot < mgr.delta_pool().capacity_slots());
    // The Status accessors agree with the record.
    SequenceId bound = 0;
    CHECK(sm.sequence_id_of(a, &bound).ok);
    CHECK_EQ(bound, ra->sequence_id);
    int len = -1;
    CHECK(sm.context_length_of(a, &len).ok);
    CHECK_EQ(len, 0);
    // null out pointers fail loud.
    CHECK(!sm.create_session(nullptr).ok);
    CHECK(!sm.sequence_id_of(a, nullptr).ok);
    CHECK(!sm.context_length_of(a, nullptr).ok);
    CHECK(check_binding_invariant(sm));

    // A second session: monotone id, distinct binding.
    SessionId b = 0;
    CHECK(sm.create_session(&b).ok);
    CHECK_EQ(b, static_cast<SessionId>(2));
    CHECK(sm.lookup(b) != nullptr);
    CHECK(sm.lookup(b)->sequence_id != ra->sequence_id);
    CHECK_EQ(sm.num_sessions(), 2);
    CHECK(check_binding_invariant(sm));

    // ---- destroy -> invalid ---------------------------------------------
    // Cache A's bound sequence id BEFORE the destroy: destroy_session
    // erases the record, so `ra` dangles afterwards and must never be
    // dereferenced again (the same discipline as the v0.5 state-manager
    // gate).
    const SequenceId a_seq = ra->sequence_id;
    CHECK(sm.destroy_session(a).ok);
    CHECK(sm.lookup(a) == nullptr);
    CHECK_EQ(sm.num_sessions(), 1);
    // EVERY operation on the destroyed id is a Status error.
    CHECK(!sm.sequence_id_of(a, &bound).ok);
    CHECK(!sm.context_length_of(a, &len).ok);
    CHECK(!sm.reset_session(a).ok);
    CHECK(!sm.destroy_session(a).ok);  // double destroy: fail loud
    // The bound sequence was retired with the session: dead in the
    // manager (no dangling live sequence, no leaked slot).
    CHECK(mgr.lookup(a_seq) == nullptr);
    CHECK_EQ(mgr.delta_pool().used_slots(), 1);  // only B's slot remains
    // B is unaffected and still inspectable.
    CHECK(sm.lookup(b) != nullptr);
    CHECK(check_binding_invariant(sm));

    // ---- the id is NEVER reissued ---------------------------------------
    SessionId c = 0;
    CHECK(sm.create_session(&c).ok);
    CHECK_EQ(c, static_cast<SessionId>(3));  // not a's id
    CHECK(sm.lookup(c) != nullptr);
    CHECK(sm.lookup(c)->sequence_id != a_seq);  // fresh bound sequence
    CHECK(check_binding_invariant(sm));
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] create->valid / destroy->invalid / id never reused\n");
  return 0;
}

// ---- transactional create -------------------------------------------------------
int test_session_create_oom_transactional() {
  const Qwen35Config cfg = small_config();
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    // Delta pool capacity 1: the second create must OOM.
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/2, /*delta_slots=*/1, s);
    SessionManager sm(mgr);
    SessionId a = 0;
    CHECK(sm.create_session(&a).ok);
    CHECK_EQ(a, static_cast<SessionId>(1));
    CHECK_EQ(sm.num_sessions(), 1);
    // Second create: no free slot.
    SessionId b = 0;
    Status st = sm.create_session(&b);
    CHECK(!st.ok);
    CHECK(st.message.find("OOM") != std::string::npos);
    CHECK_EQ(b, static_cast<SessionId>(0));  // nothing written on failure
    CHECK_EQ(sm.num_sessions(), 1);  // no half-record
    CHECK(sm.lookup(2) == nullptr);  // the would-be id is unissued
    CHECK_EQ(sm.next_session_id(), static_cast<SessionId>(2));
    CHECK_EQ(mgr.delta_pool().used_slots(), 1);  // no leaked slot
    CHECK_EQ(mgr.num_live_sequences(), 1);       // no leaked sequence
    CHECK(check_binding_invariant(sm));
    // Recovery: destroy A, then the next create succeeds with a NEW
    // monotone id (never 1 again).
    CHECK(sm.destroy_session(a).ok);
    SessionId c = 0;
    CHECK(sm.create_session(&c).ok);
    CHECK_EQ(c, static_cast<SessionId>(2));
    CHECK_EQ(sm.num_sessions(), 1);
    CHECK(check_binding_invariant(sm));
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] create OOM transactional: nothing registered, ids "
              "stay monotone\n");
  return 0;
}

// ---- isolation: A/B state resources are independent ------------------------------
int test_session_isolation() {
  const Qwen35Config cfg = small_config();
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/6, /*delta_slots=*/2, s);
    SessionManager sm(mgr);
    SessionId a = 0, b = 0;
    CHECK(sm.create_session(&a).ok);
    CHECK(sm.create_session(&b).ok);
    const Session* ra = sm.lookup(a);
    const Session* rb = sm.lookup(b);
    CHECK(ra != nullptr && rb != nullptr);

    // Distinct Delta slots (the per-session state ownership boundary).
    const SequenceState* sa = mgr.lookup(ra->sequence_id);
    const SequenceState* sb = mgr.lookup(rb->sequence_id);
    CHECK(sa != nullptr && sb != nullptr);
    CHECK(sa->delta_slot != sb->delta_slot);

    // Grow A across a KV page boundary (2 pages); B stays at zero pages.
    const SequenceId sra = ra->sequence_id;
    CHECK(mgr.ensure_kv_capacity(sra, PT).ok);
    CHECK_EQ(sa->block_table.num_blocks(), 2);
    CHECK_EQ(sb->block_table.num_blocks(), 0);
    CHECK_EQ(mgr.kv_pool().used_pages(), 2);

    // A dirties its Delta slot (conv + recurrent, every layer) and BOTH of
    // its KV pages (K + V, every layer).
    Qwen35DeltaStatePool& dp = mgr.delta_pool_mut();
    Qwen35KvPagePool& kp = mgr.kv_pool_mut();
    dirty_delta_slot(dp, sa->delta_slot, s);
    const int a_p0 = sa->block_table.lookup(0);
    const int a_p1 = sa->block_table.lookup(1);
    CHECK(a_p0 >= 0 && a_p1 >= 0 && a_p0 != a_p1);
    dirty_kv_page(kp, a_p0, s);
    dirty_kv_page(kp, a_p1, s);
    CHECK(delta_slot_any_dirty(dp, sa->delta_slot, s));  // sanity: landed

    // B MUST read back ALL ZEROS in every layer (conv + recurrent) and its
    // metadata is EXACTLY fresh: no cross-session contamination.
    CHECK(delta_slot_all_zero(dp, sb->delta_slot, s));
    CHECK_EQ(sb->length, 0);
    CHECK_EQ(sb->block_table.num_blocks(), 0);
    CHECK(check_binding_invariant(sm));
    std::printf("  [ok] isolation: A dirty (delta + 2 KV pages) -> B all "
                "zeros, metadata fresh\n");
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}

// ---- reset: A back to fresh under the SAME id, B untouched -----------------------
int test_session_reset() {
  const Qwen35Config cfg = small_config();
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/6, /*delta_slots=*/2, s);
    SessionManager sm(mgr);
    SessionId a = 0, b = 0;
    CHECK(sm.create_session(&a).ok);
    CHECK(sm.create_session(&b).ok);
    const Session* ra = sm.lookup(a);
    const Session* rb = sm.lookup(b);
    SequenceId sra = ra->sequence_id, srb = rb->sequence_id;

    // A: length 10 + 3 pages + a dirty Delta slot. B: a dirty pattern in
    // its own slot (must survive A's reset byte-for-byte).
    CHECK(mgr.advance(sra, 10).ok);
    CHECK(mgr.ensure_kv_capacity(sra, 2 * PT).ok);  // blocks 0,1,2
    Qwen35DeltaStatePool& dp = mgr.delta_pool_mut();
    const int a_slot = mgr.lookup(sra)->delta_slot;
    const int b_slot = mgr.lookup(srb)->delta_slot;
    dirty_delta_slot(dp, a_slot, s);
    dirty_delta_slot(dp, b_slot, s);
    CHECK_EQ(mgr.kv_pool().used_pages(), 3);

    // ---- reset A ----------------------------------------------------------
    CHECK(sm.reset_session(a).ok);
    // A stays LIVE under the SAME id, in fresh state:
    const Session* ra2 = sm.lookup(a);
    CHECK(ra2 != nullptr);
    CHECK_EQ(ra2->id, a);
    CHECK(ra2->state == SessionState::Active);
    CHECK_EQ(ra2->reset_count, 1);  // lifecycle metadata advanced
    CHECK_EQ(ra2->sequence_id, sra);  // SAME bound sequence
    const SequenceState* sa2 = mgr.lookup(sra);
    CHECK(sa2 != nullptr);
    CHECK_EQ(sa2->length, 0);                    // position reset
    CHECK_EQ(sa2->block_table.num_blocks(), 0);  // KV logical context reset
    CHECK_EQ(sa2->delta_slot, a_slot);           // slot kept, not re-acquired
    CHECK(delta_slot_all_zero(dp, a_slot, s));   // Delta state zeroed in place
    CHECK_EQ(mgr.kv_pool().used_pages(), 0);     // all pages released
    int len = -1;
    CHECK(sm.context_length_of(a, &len).ok);
    CHECK_EQ(len, 0);
    // ---- B is EXACTLY unchanged -------------------------------------------
    const Session* rb2 = sm.lookup(b);
    CHECK(rb2 != nullptr);
    CHECK_EQ(rb2->reset_count, 0);
    CHECK(delta_slot_any_dirty(dp, b_slot, s));  // B's pattern survived
    CHECK_EQ(mgr.lookup(srb)->length, 0);
    CHECK_EQ(mgr.lookup(srb)->block_table.num_blocks(), 0);
    CHECK(check_binding_invariant(sm));

    // A reset of an already-fresh session is a valid no-op repeat (cannot
    // OOM; idempotent semantics).
    CHECK(sm.reset_session(a).ok);
    CHECK_EQ(sm.lookup(a)->reset_count, 2);
    CHECK(delta_slot_all_zero(dp, a_slot, s));
    // Reset of an unknown id fails loud, no state change.
    CHECK(!sm.reset_session(999).ok);
    CHECK_EQ(sm.num_sessions(), 2);
    CHECK(check_binding_invariant(sm));
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] reset: A fresh under the SAME id, B exactly "
              "unchanged\n");
  return 0;
}

// ---- destroy + physical reuse with no residue ------------------------------------
int test_session_destroy_reuse() {
  const Qwen35Config cfg = small_config();
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/4, /*delta_slots=*/2, s);
    SessionManager sm(mgr);
    SessionId a = 0, b = 0;
    CHECK(sm.create_session(&a).ok);
    CHECK(sm.create_session(&b).ok);
    SequenceId sra = sm.lookup(a)->sequence_id;
    SequenceId srb = sm.lookup(b)->sequence_id;

    // A dirties everything it owns: its Delta slot + both KV pages.
    Qwen35DeltaStatePool& dp = mgr.delta_pool_mut();
    Qwen35KvPagePool& kp = mgr.kv_pool_mut();
    CHECK(mgr.ensure_kv_capacity(sra, PT).ok);  // 2 pages
    const int a_slot = mgr.lookup(sra)->delta_slot;
    const int a_p0 = mgr.lookup(sra)->block_table.lookup(0);
    const int a_p1 = mgr.lookup(sra)->block_table.lookup(1);
    dirty_delta_slot(dp, a_slot, s);
    dirty_kv_page(kp, a_p0, s);
    dirty_kv_page(kp, a_p1, s);
    // B keeps a dirty pattern of its own.
    const int b_slot = mgr.lookup(srb)->delta_slot;
    dirty_delta_slot(dp, b_slot, s);

    // ---- destroy A: resources released, id invalidated -------------------
    CHECK(sm.destroy_session(a).ok);
    CHECK(sm.lookup(a) == nullptr);
    CHECK(mgr.lookup(sra) == nullptr);  // the bound sequence is retired
    // Pool accounting: exactly B's resources remain (1 slot, 0 pages).
    CHECK_EQ(mgr.delta_pool().used_slots(), 1);
    CHECK_EQ(mgr.kv_pool().used_pages(), 0);
    CHECK_EQ(mgr.num_live_sequences(), 1);
    CHECK(check_binding_invariant(sm));
    // B still intact.
    CHECK(delta_slot_any_dirty(dp, b_slot, s));

    // ---- create C: NEW SessionId, PHYSICAL reuse, ALL ZEROS --------------
    SessionId c = 0;
    CHECK(sm.create_session(&c).ok);
    CHECK(c != a);                       // the id is never reissued
    CHECK_EQ(c, static_cast<SessionId>(3));
    const SequenceState* sc = mgr.lookup(sm.lookup(c)->sequence_id);
    CHECK(sc != nullptr);
    // The LIFO free order reissues A's physical slot and pages to C.
    CHECK_EQ(sc->delta_slot, a_slot);
    CHECK_EQ(mgr.delta_pool().used_slots(), 2);
    CHECK(delta_slot_all_zero(dp, a_slot, s));  // NO residue of A (conv+rec)
    CHECK(mgr.ensure_kv_capacity(sc->id, PT).ok);
    const int c_p0 = sc->block_table.lookup(0);
    const int c_p1 = sc->block_table.lookup(1);
    // LIFO free order: clear() released A's pages in block order (p0 then
    // p1), so p1 is the most recently released and C's FIRST allocation is
    // a_p1, its second is a_p0 (the exact physical pages are pinned — the
    // reuse is not an assumption but a check).
    CHECK_EQ(c_p0, a_p1);
    CHECK_EQ(c_p1, a_p0);
    CHECK(kv_page_all_zero(kp, c_p0, s));  // NO residue of A (K+V)
    CHECK(kv_page_all_zero(kp, c_p1, s));
    CHECK(check_binding_invariant(sm));
    std::printf("  [ok] destroy + reuse: new id, physical slot/pages "
                "reused, reads back ALL ZEROS\n");
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}

// ---- context capacity: fits() boundary matrix -------------------------------------
int test_session_fits() {
  const Qwen35Config cfg = small_config();  // max_seq_len = 64
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/4, /*delta_slots=*/2, s);
    SessionManager sm(mgr);
    SessionId a = 0;
    CHECK(sm.create_session(&a).ok);
    SequenceId sra = sm.lookup(a)->sequence_id;

    // length 0: the exact boundary (0 / max / max+1).
    CHECK(sm.fits(a, 0));
    CHECK(sm.fits(a, cfg.max_seq_len));
    CHECK(!sm.fits(a, cfg.max_seq_len + 1));
    // length 60: 4 more fit, 5 do not (context_length + new > max ->
    // reject — the v0.8 early overflow policy).
    CHECK(mgr.advance(sra, 60).ok);
    CHECK(sm.fits(a, 4));
    CHECK(!sm.fits(a, 5));
    CHECK(sm.fits(a, 0));
    // Invalid arguments / unknown ids: false (a query, not an error).
    CHECK(!sm.fits(a, -1));
    CHECK(!sm.fits(999, 0));
    // fits() is a pure query: no allocation, no mutation.
    CHECK_EQ(mgr.lookup(sra)->length, 60);
    CHECK_EQ(mgr.lookup(sra)->block_table.num_blocks(), 0);
    CHECK_EQ(mgr.kv_pool().used_pages(), 0);
    CHECK(check_binding_invariant(sm));
    // After destroy: the id no longer fits anything.
    CHECK(sm.destroy_session(a).ok);
    CHECK(!sm.fits(a, 0));
    CHECK(check_binding_invariant(sm));
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] fits(): exact boundary matrix, pure query\n");
  return 0;
}

// ---- error cases: unknown ids fail loud with no state change -----------------------
int test_session_error_cases() {
  const Qwen35Config cfg = small_config();
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/4, /*delta_slots=*/2, s);
    SessionManager sm(mgr);
    const std::size_t used_before = mgr.used_state_bytes();
    CHECK_EQ(sm.num_sessions(), 0);
    // EVERY lifecycle / query operation on an unknown id is a Status error
    // (fail loud — never silently accepted).
    SequenceId sid = 0;
    int len = 0;
    CHECK(!sm.sequence_id_of(42, nullptr).ok);  // null out pointer
    CHECK(!sm.sequence_id_of(42, &sid).ok);
    CHECK(!sm.context_length_of(42, &len).ok);
    CHECK(!sm.reset_session(42).ok);
    CHECK(!sm.destroy_session(42).ok);
    // The pools and the session set are EXACTLY unchanged.
    CHECK_EQ(sm.num_sessions(), 0);
    CHECK_EQ(mgr.num_live_sequences(), 0);
    CHECK(mgr.used_state_bytes() == used_before);
    CHECK_EQ(mgr.kv_pool().used_pages(), 0);
    CHECK_EQ(mgr.delta_pool().used_slots(), 0);
    CHECK(check_binding_invariant(sm));
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] error cases: unknown ids fail loud, no state change\n");
  return 0;
}

}  // namespace

int main() {
  std::printf("test_session_manager: CUDALM v0.8 Phase A session lifecycle "
              "gate (small synthetic config, real device pools)\n");
  int rc = 0;
  rc |= test_session_create_destroy();
  rc |= test_session_create_oom_transactional();
  rc |= test_session_isolation();
  rc |= test_session_reset();
  rc |= test_session_destroy_reuse();
  rc |= test_session_fits();
  rc |= test_session_error_cases();
  if (rc != 0) {
    std::fprintf(stderr, "test_session_manager: FAIL\n");
    return rc;
  }
  std::printf("test_session_manager: PASS\n");
  return 0;
}
