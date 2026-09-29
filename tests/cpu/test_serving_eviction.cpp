// CUDALM — v0.9 Phase C: session TTL / LRU eviction contract gate
// (CPU test — NO checkpoint, NO model: the deterministic fake forwarder
// (a SUCCESSFUL forward commits real sequence metadata) + real pools +
// real SessionManager + real Scheduler + the ServingController Phase C
// eviction API with a FAKE monotonic clock).
//
// Proves the eviction contract (EVICT == DESTROY THE WHOLE SESSION —
// never context truncation):
//   * A. TTL: B (older last activity) expires, A (recent activity)
//     survives; B's SessionId is invalidated forever; the KV/Delta
//     accounting releases; a new session reusing B's Delta slot sees a
//     ZEROED slot (the slot is patterned BEFORE the eviction);
//   * B. BUSY protection: a TTL-expired session with a LIVE request is
//     never evicted (no cancel, no mutation);
//   * C. TERMINAL-BUT-UNDRAINED protection (the Phase B lifecycle): a
//     session whose request is terminal but whose stream events are not
//     yet drained is PROTECTED from the TTL sweep — only AFTER the
//     events are fully drained does the expired session become
//     evictable (the Phase C hard contract);
//   * D. LRU-on-session-pressure: max_sessions reached -> ONE eligible
//     idle LRU session (oldest activity) is evicted, the new session is
//     created (a fresh monotonic SessionId), num_sessions stays at the
//     limit;
//   * E. NO CANDIDATE: busy + terminal-undrained (and unmanaged)
//     sessions -> the pressure create REJECTS (no SessionId consumed,
//     no pool mutation, both sessions untouched — the pressure safety
//     gate);
//   * F. DEFAULT POLICY = EXACTLY PHASE A: with no eviction enabled,
//     max_sessions reached still REJECTS;
//   * G. UNMANAGED sessions (created directly on the manager, outside
//     the controller) are never guessed, never auto-evicted — they
//     still occupy the session limit;
//   * H. REVIEW FIX: a DEADLINE terminal refreshes the session's
//     activity (the TTL is re-anchored to the deadline terminal, not
//     to the old admission/drive time) — the request is
//     deadline-cancelled on the first step_stream() after a long
//     undriven silence; a sweep BEFORE the TTL from that terminal
//     must NOT evict; at idle_age == TTL the sweep evicts;
//   * I. REVIEW FIX (over-limit fail-safe): pressure eviction
//     triggers ONLY at live_sessions == max_sessions — when the
//     controller is ALREADY OVER the limit (unmanaged sessions took
//     the capacity) the create REJECTS without evicting anything
//     (no SessionId consumed, no pool mutation, the eligible
//     managed session survives).
//
// Provenance: CUDALM-native (v0.9 Phase C).

#include "../../tests/common/check.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/scheduler.h"
#include "cudalm/session.h"
#include "cudalm/serving_controller.h"

using namespace cudalm;

namespace {

Qwen35Config small_config() {
  Qwen35Config c;
  c.hidden_size = 256;
  c.num_hidden_layers = 8;
  c.intermediate_size = 512;
  c.vocab_size = 512;
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

// Deterministic CPU forwarder (same pattern as the Phase B gate).
class FakeForwarder : public SequenceForwarder {
 public:
  int vocab = 512;

  Status forward_token(int /*token_id*/, SequenceId sequence_id,
                       Qwen35StateManager& mgr,
                       cudaStream_t /*stream*/) override {
    last_step_ = count_[sequence_id]++;
    const SequenceState* st = mgr.lookup(sequence_id);
    if (st == nullptr) {
      return Status::error("fake forward: unknown sequence");
    }
    const int pos = st->length;
    Status s = mgr.ensure_kv_capacity(sequence_id, pos);
    if (!s.ok) return s;
    return mgr.set_length(sequence_id, pos + 1);
  }

  Status logits_to_host(std::vector<__nv_bfloat16>* out,
                        cudaStream_t /*stream*/) const override {
    out->assign(static_cast<std::size_t>(vocab),
                __float2bfloat16(-1.0f));
    for (int t = 0; t < 8; ++t) {
      (*out)[static_cast<std::size_t>(t)] =
          __float2bfloat16(5.0f - 0.5f * static_cast<float>(t));
    }
    const int pick = static_cast<int>(
        (static_cast<std::uint64_t>(3) *
             static_cast<std::uint64_t>(last_step_) +
         1) %
        static_cast<std::uint64_t>(vocab));
    (*out)[static_cast<std::size_t>(pick)] = __float2bfloat16(10.0f);
    return Status::ok_status();
  }

  int vocab_size() const override { return vocab; }
  int forwards(SequenceId sid) const {
    auto it = count_.find(sid);
    return it == count_.end() ? 0 : it->second;
  }

 private:
  mutable int last_step_ = -1;
  mutable std::map<SequenceId, int> count_;
};

// A deterministic FAKE monotonic clock (manual advance).
class FakeClock : public MonotonicClock {
 public:
  std::chrono::steady_clock::time_point t0 =
      std::chrono::steady_clock::now();
  int ms = 0;

  std::chrono::steady_clock::time_point now() const override {
    return t0 + std::chrono::milliseconds(ms);
  }
  void advance_ms(int n) { ms += n; }
};

// Number of linear (DeltaNet) layer ordinals in the config.
int linear_ordinals(Qwen35StateManager& mgr) {
  int n = 0;
  while (mgr.delta_pool().layer_of_linear_ordinal(n) >= 0) ++n;
  return n;
}

// Write a NON-ZERO pattern into a Delta slot (conv + recurrent, every
// linear layer) — so the zero-on-release can be OBSERVED.
int pattern_slot(Qwen35StateManager& mgr, int slot, cudaStream_t stream) {
  const int n = linear_ordinals(mgr);
  auto& pool = mgr.delta_pool_mut();
  std::vector<__nv_bfloat16> hc(pool.conv_elems(),
                                __float2bfloat16(1.0f));
  std::vector<float> hr(pool.rec_elems(), 1.0f);
  for (int o = 0; o < n; ++o) {
    CUDA_CHECK(cudaMemcpyAsync(pool.conv_mut(o, slot), hc.data(),
                          hc.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(pool.recurrent_mut(o, slot), hr.data(),
                          hr.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

inline bool bf16_is_zero(__nv_bfloat16 v) {
  const std::uint16_t bits = *reinterpret_cast<std::uint16_t*>(&v);
  return bits == 0;
}

// Check a Delta slot is ALL ZERO (conv + recurrent, every linear layer)
// — proves zero-on-release / fresh reuse (no stale contamination).
int check_slot_zeroed(Qwen35StateManager& mgr, int slot,
                      cudaStream_t stream) {
  const int n = linear_ordinals(mgr);
  const auto& pool = mgr.delta_pool();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> hc;
  std::vector<float> hr;
  for (int o = 0; o < n; ++o) {
    hc.assign(pool.conv_elems(), __float2bfloat16(-1.0f));
    CUDA_CHECK(cudaMemcpyAsync(hc.data(), pool.conv(o, slot),
                          hc.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost, stream));
    for (__nv_bfloat16 v : hc) {
      CHECK(bf16_is_zero(v));
    }
    hr.assign(pool.rec_elems(), -1.0f);
    CUDA_CHECK(cudaMemcpyAsync(hr.data(), pool.recurrent(o, slot),
                          hr.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    for (float v : hr) {
      CHECK(v == 0.0f);
    }
  }
  return 0;
}

}  // namespace

int main() {
  std::printf("test_serving_eviction: CUDALM v0.9 Phase C session "
              "TTL / LRU eviction contract gate (fake forwarder, real "
              "pools)\n");

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const Qwen35Config cfg = small_config();
  const SamplingConfig kGreedy = SamplingConfig::greedy();

  // =========================================================================
  // A. TTL: the older idle session expires; the recently-active one
  //    survives; the evicted slot is released ZEROED (no stale reuse)
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    FakeClock clk;
    Scheduler sched(fwd, mgr, stream, &sm);
    const SessionEvictionPolicy pol =
        SessionEvictionPolicy{}.with_idle_ttl(std::chrono::milliseconds(1000));
    ServingController ctrl(sched, sm, ServingLimits{-1, -1, 0}, &clk, pol);

    // t = 0: create B (the OLDER activity) and run ONE small turn on
    // it (2 input + 1 gen = length 3: B owns real KV pages + its Delta
    // slot is used); t = 100: create A (the NEWER activity):
    SessionId B = 0, A = 0;
    CHECK(ctrl.create_session(&B).ok);
    RequestId rb0 = 0;
    CHECK(ctrl.admit_turn(B, {1, 2}, 1, -1, kGreedy, &rb0).ok);
    ctrl.run_stream();  // fully drained (terminal is NOT undrained)
    CHECK_EQ(mgr.lookup(sm.lookup(B)->sequence_id)->length, 3);
    clk.advance_ms(100);
    CHECK(ctrl.create_session(&A).ok);

    // Pattern B's Delta slot (so zero-on-release is OBSERVABLE):
    const int slot_B = mgr.lookup(sm.lookup(B)->sequence_id)->delta_slot;
    CHECK_EQ(pattern_slot(mgr, slot_B, stream), 0);

    const int kv_before = mgr.kv_pool().used_pages();  // B's 1 page
    const int dl_before = mgr.delta_pool().used_slots();
    const int seq_before = mgr.num_live_sequences();

    // t = 1100: A gets RECENT activity (a successful reset_session
    // refreshes it); t = 1150: at the sweep, B's idle is 1150ms >=
    // 1000 (EXPIRED) while A's idle is only 50ms (SURVIVES):
    clk.advance_ms(1000);
    CHECK(ctrl.reset_session(A).ok);
    clk.advance_ms(50);
    const std::vector<SessionId> evicted = ctrl.evict_expired_sessions();
    CHECK(static_cast<int>(evicted.size()) == 1 && evicted[0] == B);
    // B is INVALIDATED FOREVER; A survives:
    CHECK(sm.lookup(B) == nullptr);
    CHECK(sm.lookup(A) != nullptr);
    CHECK_EQ(ctrl.stats().evicted_sessions_ttl, static_cast<std::uint64_t>(1));
    // KV / Delta / sequence accounting released (B's 3 tokens = 1
    // page of 4; its Delta slot + its sequence are gone; A's state
    // is untouched — A has no turn yet, so the pool is empty):
    CHECK_EQ(mgr.kv_pool().used_pages(), kv_before - 1);
    CHECK_EQ(mgr.delta_pool().used_slots(), dl_before - 1);
    CHECK_EQ(mgr.num_live_sequences(), seq_before - 1);
    std::printf("  [ok] TTL: B expired + evicted, A (recent activity) "
                "survives, accounting released\n");

    // A's state / logical length is untouched (and still usable):
    RequestId ra = 0;
    CHECK(ctrl.admit_turn(A, {11}, 1, -1, kGreedy, &ra).ok);
    ctrl.run_stream();
    CHECK_EQ(mgr.lookup(sm.lookup(A)->sequence_id)->length, 2);
    std::printf("  [ok] surviving A: state + logical length intact\n");

    // Create C: a NEW monotonic SessionId (B's id is never reused); if
    // the Delta slot is reused it must be ZEROED (no stale
    // contamination):
    SessionId C = 0;
    CHECK(ctrl.create_session(&C).ok);
    CHECK(C > B);
    const int slot_C = mgr.lookup(sm.lookup(C)->sequence_id)->delta_slot;
    if (slot_C == slot_B) {
      CHECK_EQ(check_slot_zeroed(mgr, slot_C, stream), 0);
    }
    CHECK_EQ(mgr.delta_pool().used_slots(), dl_before);
    std::printf("  [ok] C: fresh monotonic SessionId, reused slot %s\n",
                slot_C == slot_B ? "verified zeroed" : "not reused");
  }

  // =========================================================================
  // B. BUSY protection: TTL-expired but LIVE-request sessions survive
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    FakeClock clk;
    Scheduler sched(fwd, mgr, stream, &sm);
    const SessionEvictionPolicy pol =
        SessionEvictionPolicy{}.with_idle_ttl(std::chrono::milliseconds(1000));
    ServingController ctrl(sched, sm, ServingLimits{-1, -1, 0}, &clk, pol);
    SessionId B = 0;
    CHECK(ctrl.create_session(&B).ok);
    // A LIVE (Waiting) request on B:
    RequestId rb = 0;
    CHECK(ctrl.admit_turn(B, {21, 22}, 2, -1, kGreedy, &rb).ok);
    CHECK(sched.session_busy(B));
    clk.advance_ms(5000);  // WAY past the TTL
    const std::vector<SessionId> evicted =
        ctrl.evict_expired_sessions();
    CHECK(evicted.empty());
    CHECK(sm.lookup(B) != nullptr);  // PROTECTED (busy)
    CHECK_EQ(ctrl.stats().evicted_sessions_ttl, static_cast<std::uint64_t>(0));
    // (The request is still exactly where it was — no cancel, no step:)
    CHECK(sched.get(rb)->status == RequestStatus::Waiting);
    std::printf("  [ok] busy session: TTL-expired but PROTECTED (no "
                "cancel, no mutation)\n");
  }

  // =========================================================================
  // C. TERMINAL-BUT-UNDRAINED protection (the Phase B lifecycle — the
  //    Phase C hard contract)
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    FakeClock clk;
    Scheduler sched(fwd, mgr, stream, &sm);
    const SessionEvictionPolicy pol =
        SessionEvictionPolicy{}.with_idle_ttl(std::chrono::milliseconds(1000));
    ServingController ctrl(sched, sm, ServingLimits{-1, -1, 0}, &clk, pol);
    SessionId B = 0;
    CHECK(ctrl.create_session(&B).ok);
    // 1 input + max_new 1; drive with the plain step() (NO drain) to
    // terminal — the stream events are NOT yet consumed:
    RequestId rb = 0;
    CHECK(ctrl.admit_turn(B, {31}, 1, -1, kGreedy, &rb).ok);
    CHECK(ctrl.step().ok);
    CHECK(ctrl.step().ok);
    CHECK(sched.get(rb)->status == RequestStatus::Finished);
    CHECK(!sched.session_busy(B));  // scheduler: not busy...
    CHECK(!ctrl.is_eviction_eligible(B));  // ...but PROTECTED (undrained)
    clk.advance_ms(5000);  // WAY past the TTL
    const std::vector<SessionId> evicted =
        ctrl.evict_expired_sessions();
    CHECK(evicted.empty());  // NOT evicted: the stream is not drained
    CHECK(sm.lookup(B) != nullptr);
    // Drain the stream (poll = fully drained -> reaped):
    std::vector<ServingEvent> ev;
    CHECK(ctrl.poll(rb, &ev).ok);
    CHECK_EQ(static_cast<int>(ev.size()), 2);  // the committed token + the terminal
    // Now idle + expired + drained: evictable:
    const std::vector<SessionId> evicted2 =
        ctrl.evict_expired_sessions();
    CHECK(static_cast<int>(evicted2.size()) == 1 && evicted2[0] == B);
    CHECK(sm.lookup(B) == nullptr);
    CHECK_EQ(ctrl.stats().evicted_sessions_ttl, static_cast<std::uint64_t>(1));
    std::printf("  [ok] terminal-but-undrained: PROTECTED until fully "
                "drained, then evictable\n");
  }

  // =========================================================================
  // D. LRU-on-session-pressure: at the limit, ONE eligible idle LRU
  //    session (oldest activity) is evicted; the new session is created
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    FakeClock clk;
    Scheduler sched(fwd, mgr, stream, &sm);
    const SessionEvictionPolicy pol =
        SessionEvictionPolicy{}.with_lru_on_session_pressure();
    ServingController ctrl(sched, sm, ServingLimits{2, -1, 0}, &clk, pol);
    // t = 0: create B (oldest activity); t = 100: create A (newer):
    SessionId B = 0, A = 0;
    CHECK(ctrl.create_session(&B).ok);
    clk.advance_ms(100);
    CHECK(ctrl.create_session(&A).ok);
    CHECK_EQ(sm.num_sessions(), 2);
    // t = 200: create C at the LIMIT -> the LRU eligible (B) is
    // evicted, C is created:
    clk.advance_ms(100);
    SessionId C = 0;
    CHECK(ctrl.create_session(&C).ok);
    CHECK(sm.lookup(B) == nullptr);  // B evicted (oldest activity)
    CHECK(sm.lookup(A) != nullptr);  // A kept (newer activity)
    CHECK(sm.lookup(C) != nullptr);
    CHECK(C > B);  // a NEW monotonic SessionId
    CHECK_EQ(sm.num_sessions(), 2);  // still exactly at the limit
    CHECK_EQ(ctrl.stats().evicted_sessions_lru, static_cast<std::uint64_t>(1));
    CHECK_EQ(ctrl.stats().evicted_sessions_ttl, static_cast<std::uint64_t>(0));
    std::printf("  [ok] LRU pressure: B (oldest) evicted, A kept, C "
                "created with a fresh SessionId\n");
  }

  // =========================================================================
  // E. NO CANDIDATE: busy + terminal-undrained -> the pressure create
  //    REJECTS (no SessionId consumed, no pool mutation, both
  //    untouched) — the pressure safety gate
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    FakeClock clk;
    Scheduler sched(fwd, mgr, stream, &sm);
    const SessionEvictionPolicy pol =
        SessionEvictionPolicy{}.with_lru_on_session_pressure();
    ServingController ctrl(sched, sm, ServingLimits{2, -1, 0}, &clk, pol);
    SessionId A = 0, B = 0;
    CHECK(ctrl.create_session(&A).ok);
    CHECK(ctrl.create_session(&B).ok);
    // A: BUSY (a live Waiting request):
    RequestId ra = 0;
    CHECK(ctrl.admit_turn(A, {41}, 2, -1, kGreedy, &ra).ok);
    CHECK(sched.session_busy(A));
    // B: TERMINAL-BUT-UNDRAINED (1 input + max_new 1, plain step()):
    RequestId rb = 0;
    CHECK(ctrl.admit_turn(B, {51}, 1, -1, kGreedy, &rb).ok);
    CHECK(ctrl.step().ok);
    CHECK(ctrl.step().ok);
    CHECK(sched.get(rb)->status == RequestStatus::Finished);
    CHECK(!ctrl.is_eviction_eligible(A));
    CHECK(!ctrl.is_eviction_eligible(B));
    const SequenceId next_seq_before = mgr.next_sequence_id();
    const int kv_before = mgr.kv_pool().used_pages();
    const int dl_before = mgr.delta_pool().used_slots();
    // Create C at the limit: NO eligible candidate -> Phase A reject:
    SessionId C = 123;
    Status s = ctrl.create_session(&C);
    CHECK(!s.ok);
    CHECK_EQ(C, static_cast<SessionId>(123));  // untouched
    CHECK_EQ(ctrl.stats().eviction_no_candidate, static_cast<std::uint64_t>(1));
    CHECK_EQ(ctrl.stats().rejected_session_limit, static_cast<std::uint64_t>(1));
    // No SessionId consumed (no partial mutation):
    CHECK_EQ(mgr.next_sequence_id(), next_seq_before);
    CHECK_EQ(sm.num_sessions(), 2);
    // No pool mutation:
    CHECK_EQ(mgr.kv_pool().used_pages(), kv_before);
    CHECK_EQ(mgr.delta_pool().used_slots(), dl_before);
    // A/B untouched: A's request still live (RUNNING — the B-steps
    // drove it too: 1 input + g0 committed, g1 pending — never
    // cancelled / evicted); B's stream still fully pollable:
    CHECK(sched.get(ra)->status == RequestStatus::Running);
    CHECK_EQ(mgr.lookup(sm.lookup(A)->sequence_id)->length, 2);  // 1 + g0
    std::vector<ServingEvent> ev;
    CHECK(ctrl.poll(rb, &ev).ok);
    CHECK_EQ(static_cast<int>(ev.size()), 2);
    std::printf("  [ok] no candidate: pressure create REJECTED (no id "
                "consumed, no pool mutation, A/B untouched)\n");
  }

  // =========================================================================
  // F. DEFAULT policy = EXACTLY PHASE A: no eviction without an explicit
  //    policy; max_sessions reached still REJECTS
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    FakeClock clk;
    Scheduler sched(fwd, mgr, stream, &sm);
    // The DEFAULT policy (both features disabled):
    ServingController ctrl(sched, sm, ServingLimits{2, -1, 0}, &clk);
    SessionId A = 0, B = 0;
    CHECK(ctrl.create_session(&A).ok);
    CHECK(ctrl.create_session(&B).ok);
    clk.advance_ms(100000);  // absurdly idle — no TTL is enabled anyway
    SessionId C = 0;
    Status s = ctrl.create_session(&C);
    CHECK(!s.ok);  // Phase A rejection (no silent eviction)
    CHECK_EQ(ctrl.stats().rejected_session_limit, static_cast<std::uint64_t>(1));
    CHECK_EQ(ctrl.stats().evicted_sessions_ttl, static_cast<std::uint64_t>(0));
    CHECK_EQ(ctrl.stats().evicted_sessions_lru, static_cast<std::uint64_t>(0));
    CHECK_EQ(sm.num_sessions(), 2);
    // The explicit sweep is a no-op without an enabled TTL:
    const std::vector<SessionId> evicted =
        ctrl.evict_expired_sessions();
    CHECK(evicted.empty());
    std::printf("  [ok] default policy: EXACTLY Phase A (no eviction "
                "without an explicit policy)\n");
  }

  // =========================================================================
  // G. UNMANAGED sessions (created directly on the manager) are never
  //    auto-evicted — they still occupy the session limit
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    FakeClock clk;
    Scheduler sched(fwd, mgr, stream, &sm);
    const SessionEvictionPolicy pol =
        SessionEvictionPolicy{}.with_lru_on_session_pressure();
    ServingController ctrl(sched, sm, ServingLimits{2, -1, 0}, &clk, pol);
    // U: created DIRECTLY on the manager (unmanaged):
    SessionId U = 0;
    CHECK(sm.create_session(&U).ok);
    SessionId A = 0;
    CHECK(ctrl.create_session(&A).ok);
    // A: BUSY (the only managed session — and it is not eligible):
    RequestId ra = 0;
    CHECK(ctrl.admit_turn(A, {61}, 2, -1, kGreedy, &ra).ok);
    CHECK(sched.session_busy(A));
    // Create C at the limit (U + A): U is unmanaged (never a
    // candidate), A is busy -> REJECT; U is untouched:
    SessionId C = 0;
    Status s = ctrl.create_session(&C);
    CHECK(!s.ok);
    CHECK(sm.lookup(U) != nullptr);  // the unmanaged session survives
    CHECK(sm.lookup(A) != nullptr);
    CHECK_EQ(ctrl.stats().eviction_no_candidate, static_cast<std::uint64_t>(1));
    CHECK_EQ(ctrl.stats().evicted_sessions_lru, static_cast<std::uint64_t>(0));
    std::printf("  [ok] unmanaged session: never guessed, never "
                "auto-evicted (still occupies the limit)\n");
  }

  // =========================================================================
  // H. REVIEW FIX: a DEADLINE terminal refreshes the session activity
  //    (the TTL is re-anchored to the deadline terminal)
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    FakeClock clk;
    Scheduler sched(fwd, mgr, stream, &sm);
    const SessionEvictionPolicy pol =
        SessionEvictionPolicy{}.with_idle_ttl(std::chrono::milliseconds(1000));
    ServingController ctrl(sched, sm, ServingLimits{-1, -1, 0}, &clk, pol);
    SessionId S = 0;
    CHECK(ctrl.create_session(&S).ok);  // activity t = 0
    // Admit a request with a deadline at t = 500, then do NOT drive:
    const std::chrono::steady_clock::time_point dl =
        clk.t0 + std::chrono::milliseconds(500);
    RequestId r = 0;
    CHECK(ctrl.admit_turn(S, {71}, 2, -1, kGreedy, &r, dl).ok);
    // t = 1000 (a long undriven silence): the FIRST step_stream:
    clk.advance_ms(1000);
    std::vector<ServingEvent> ev = ctrl.step_stream();
    // The deadline fired BEFORE the next forward (no additional
    // commit — nothing was ever forwarded):
    CHECK_EQ(fwd.forwards(sm.lookup(S)->sequence_id), 0);
    CHECK_EQ(static_cast<int>(ev.size()), 1);  // only the terminal
    CHECK(ev[0].kind == ServingEventKind::RequestTerminal);
    CHECK(ev[0].status == RequestStatus::Cancelled);
    CHECK(ev[0].deadline_exceeded);
    CHECK(sched.get(r)->status == RequestStatus::Cancelled);
    std::printf("  [ok] deadline terminal: cancelled before the next "
                "forward, no commit\n");

    // The TTL must be anchored to the TERMINAL (t = 1000), not to the
    // admission (t = 0): at t = 1900 (idle 900ms < 1000) the sweep
    // must NOT evict...
    clk.advance_ms(900);
    const std::vector<SessionId> kept = ctrl.evict_expired_sessions();
    CHECK(kept.empty());
    CHECK(sm.lookup(S) != nullptr);
    // ...and at idle_age == TTL (t = 2000) the sweep evicts:
    clk.advance_ms(100);
    const std::vector<SessionId> evicted =
        ctrl.evict_expired_sessions();
    CHECK(static_cast<int>(evicted.size()) == 1 && evicted[0] == S);
    CHECK(sm.lookup(S) == nullptr);
    CHECK_EQ(ctrl.stats().evicted_sessions_ttl, static_cast<std::uint64_t>(1));
    std::printf("  [ok] TTL re-anchored to the deadline terminal "
                "(no evict before the TTL, evict at idle_age == TTL)\n");
  }

  // =========================================================================
  // I. REVIEW FIX (over-limit fail-safe): pressure eviction triggers
  //    ONLY at live_sessions == max_sessions — already over the limit
  //    -> reject WITHOUT evicting anything
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    FakeClock clk;
    Scheduler sched(fwd, mgr, stream, &sm);
    const SessionEvictionPolicy pol =
        SessionEvictionPolicy{}.with_lru_on_session_pressure();
    ServingController ctrl(sched, sm, ServingLimits{2, -1, 0}, &clk, pol);
    // A: a MANAGED, eligible, IDLE session:
    SessionId A = 0;
    CHECK(ctrl.create_session(&A).ok);
    // U1 / U2: UNMANAGED (directly on the manager) -> the controller
    // is ALREADY OVER the limit (3 live > 2):
    SessionId U1 = 0, U2 = 0;
    CHECK(sm.create_session(&U1).ok);
    CHECK(sm.create_session(&U2).ok);
    CHECK_EQ(sm.num_sessions(), 3);
    const SequenceId next_seq_before = mgr.next_sequence_id();
    const int kv_before = mgr.kv_pool().used_pages();
    const int dl_before = mgr.delta_pool().used_slots();
    // Create C while OVER the limit: NO eviction, straight to the
    // Phase A session-limit rejection:
    SessionId C = 123;
    Status s = ctrl.create_session(&C);
    CHECK(!s.ok);
    CHECK_EQ(C, static_cast<SessionId>(123));
    // A was NOT evicted (even though it is the LRU eligible one);
    // U1/U2 are untouched:
    CHECK(sm.lookup(A) != nullptr);
    CHECK(sm.lookup(U1) != nullptr);
    CHECK(sm.lookup(U2) != nullptr);
    CHECK_EQ(sm.num_sessions(), 3);
    // No new SessionId / sequence consumed, no pool mutation:
    CHECK_EQ(mgr.next_sequence_id(), next_seq_before);
    CHECK_EQ(mgr.kv_pool().used_pages(), kv_before);
    CHECK_EQ(mgr.delta_pool().used_slots(), dl_before);
    // No automatic eviction happened at all:
    CHECK_EQ(ctrl.stats().evicted_sessions_lru, static_cast<std::uint64_t>(0));
    CHECK_EQ(ctrl.stats().evicted_sessions_ttl, static_cast<std::uint64_t>(0));
    CHECK_EQ(ctrl.stats().rejected_session_limit, static_cast<std::uint64_t>(1));
    std::printf("  [ok] over-limit fail-safe: rejected WITHOUT evicting "
                "(no id consumed, no pool mutation, A/U1/U2 untouched)\n");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("test_serving_eviction: PASS\n");
  return 0;
}
