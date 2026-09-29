// CUDALM — v0.9 Phase A: serving admission / backpressure contract gate
// (CPU test — NO checkpoint, NO model: the deterministic fake forwarder
// (the Phase C/D pattern — a SUCCESSFUL forward commits real sequence
// metadata, a FAILED one commits nothing) + real state pools + real
// SessionManager + real Scheduler + the NEW ServingController).
//
// Proves the serving admission contract:
//   * max_sessions: create allowed below the limit, REJECTED at the
//     limit (zero mutation, no SessionId consumed); destroy releases
//     the session quota (reuse after the limit);
//   * max_live_requests: turn admission allowed below the limit,
//     REJECTED at the limit (zero mutation, no RequestId consumed);
//     a TERMINAL request (Finished) releases the request quota — the
//     reuse case: limit reached -> reject -> existing request finishes
//     -> new admission succeeds (no permanent counting deadlock);
//   * max_context_tokens_per_session: the policy cap rejects a turn
//     whose PROJECTED length exceeds it (while the model's max_seq_len
//     would allow it), allows the exact boundary (==), and is clamped
//     to the model's max_seq_len in the constructor;
//   * CANCEL and FORWARD FAILURE release the request quota; the same
//     terminal request is never double-decremented (repeated sync() is
//     a no-op);
//   * reset_session does NOT release the session quota;
//   * stats counters carry the pinned semantics;
//   * QUOTA TRI-STATE (review fix): a quota of 0 = ZERO CAPACITY (the
//     very first create_session / admit_turn is rejected, zero
//     mutation), while -1 STILL means UNLIMITED (previously 0 was
//     misread as unlimited).
//
// Provenance: CUDALM-native (v0.9 Phase A).

#include "../../tests/common/check.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/scheduler.h"
#include "cudalm/session.h"
#include "cudalm/serving_controller.h"

using namespace cudalm;

namespace {

// Small synthetic hybrid config (the Phase C/D CPU gate's shape).
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

// Deterministic CPU forwarder (Phase C pattern).
class FakeForwarder : public SequenceForwarder {
 public:
  int vocab = 512;
  int fail_attempt = -1;  // fail when the per-sequence ATTEMPT INDEX
                          // equals this (deterministic fault injection)

  Status forward_token(int /*token_id*/, SequenceId sequence_id,
                       Qwen35StateManager& mgr,
                       cudaStream_t /*stream*/) override {
    last_step_ = count_[sequence_id]++;
    if (last_step_ == fail_attempt) {
      return Status::error("fake forward fault (injected)");
    }
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
    const int pick =
        static_cast<int>((static_cast<std::uint64_t>(3) *
                             static_cast<std::uint64_t>(last_step_) +
                         1) %
                         static_cast<std::uint64_t>(vocab));
    (*out)[static_cast<std::size_t>(pick)] = __float2bfloat16(10.0f);
    return Status::ok_status();
  }

  int vocab_size() const override { return vocab; }

 private:
  mutable int last_step_ = -1;
  mutable std::map<SequenceId, int> count_;
};

struct PoolSnapshot {
  int pages;
  int slots;

  bool operator==(const PoolSnapshot& o) const {
    return pages == o.pages && slots == o.slots;
  }
};

PoolSnapshot snap(const Qwen35StateManager& mgr) {
  return {mgr.kv_pool().used_pages(), mgr.delta_pool().used_slots()};
}

}  // namespace

int main() {
  std::printf("test_serving_admission: CUDALM v0.9 Phase A serving "
              "admission contract gate (fake forwarder, real pools)\n");

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const Qwen35Config cfg = small_config();
  const SamplingConfig kGreedy = SamplingConfig::greedy();

  // =========================================================================
  // 1. max_sessions: limit enforcement + zero mutation + destroy-release
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm,
                           ServingLimits{/*max_sessions=*/2,
                                         /*max_live_requests=*/-1,
                                         /*max_context=*/0});

    SessionId s1 = 0, s2 = 0, s3 = 0;
    CHECK(ctrl.create_session(&s1).ok);
    CHECK(ctrl.create_session(&s2).ok);
    const PoolSnapshot sp = snap(mgr);
    Status s = ctrl.create_session(&s3);  // limit reached (2 == 2)
    CHECK(!s.ok);
    CHECK(s.message.find("session limit") != std::string::npos);
    CHECK_EQ(s3, static_cast<SessionId>(0));  // NO SessionId consumed
    CHECK_EQ(sm.num_sessions(), 2);  // zero mutation
    CHECK(snap(mgr) == sp);
    {
      const ServingStats st = ctrl.stats();
      CHECK_EQ(st.live_sessions, 2);
      CHECK_EQ(static_cast<int>(st.total_admitted_sessions), 2);
      CHECK_EQ(static_cast<int>(st.rejected_session_limit), 1);
    }
    std::printf("  [ok] max_sessions: rejected at the limit, zero "
                "mutation, no SessionId consumed\n");

    // reset does NOT release the session quota:
    CHECK(ctrl.reset_session(s1).ok);
    s = ctrl.create_session(&s3);
    CHECK(!s.ok);
    CHECK(s.message.find("session limit") != std::string::npos);
    std::printf("  [ok] reset: session quota unchanged\n");

    // destroy releases the session quota (reuse after the limit):
    CHECK(ctrl.destroy_session(s1).ok);
    CHECK_EQ(ctrl.stats().live_sessions, 1);
    CHECK(ctrl.create_session(&s3).ok);  // succeeds after the release
    CHECK_EQ(ctrl.stats().live_sessions, 2);
    CHECK_EQ(static_cast<int>(ctrl.stats().total_admitted_sessions), 3);
    std::printf("  [ok] destroy: session quota released, reuse succeeds\n");
  }

  // =========================================================================
  // 2. max_live_requests: limit + zero mutation + TERMINAL-RELEASE + reuse
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm,
                           ServingLimits{-1, /*max_live_requests=*/1, 0});

    SessionId a = 0, b = 0;
    CHECK(ctrl.create_session(&a).ok);
    CHECK(ctrl.create_session(&b).ok);

    RequestId r1 = 0;
    Status s = ctrl.admit_turn(a, {10, 20}, 2, -1, kGreedy, &r1);
    CHECK(s.ok);
    CHECK_EQ(ctrl.stats().live_requests, 1);

    // Limit reached (1 == 1): reject with ZERO MUTATION.
    RequestId r2 = 0;
    const int reqs = sched.num_requests();
    const RequestId next = sched.next_request_id();
    const int lenA = mgr.lookup(sm.lookup(a)->sequence_id)->length;
    const PoolSnapshot sp = snap(mgr);
    s = ctrl.admit_turn(b, {30, 40}, 2, -1, kGreedy, &r2);
    CHECK(!s.ok);
    CHECK(s.message.find("live request limit") != std::string::npos);
    CHECK_EQ(r2, static_cast<RequestId>(0));  // NO RequestId consumed
    CHECK_EQ(sched.num_requests(), reqs);
    CHECK_EQ(sched.next_request_id(), next);
    CHECK_EQ(mgr.lookup(sm.lookup(a)->sequence_id)->length, lenA);
    CHECK(snap(mgr) == sp);
    CHECK_EQ(static_cast<int>(ctrl.stats().rejected_request_limit), 1);
    std::printf("  [ok] max_live_requests: rejected at the limit, zero "
                "mutation, no RequestId consumed\n");

    // REUSE: the existing request finishes -> the quota is released ->
    // the new admission succeeds (no permanent counting deadlock).
    CHECK(ctrl.run().ok);
    CHECK_EQ(ctrl.stats().live_requests, 0);
    CHECK(sched.get(r1)->status == RequestStatus::Finished);
    // Repeated sync() is a no-op (no double-decrement path):
    ctrl.sync();
    ctrl.sync();
    CHECK_EQ(ctrl.stats().live_requests, 0);
    CHECK_EQ(static_cast<int>(ctrl.stats().total_admitted_requests), 1);
    RequestId r3 = 0;
    s = ctrl.admit_turn(b, {30, 40}, 2, -1, kGreedy, &r3);
    CHECK(s.ok);  // the previously-rejected session can now be admitted
    CHECK_EQ(ctrl.stats().live_requests, 1);
    CHECK(ctrl.run().ok);
    CHECK_EQ(ctrl.stats().live_requests, 0);
    std::printf("  [ok] terminal request releases the quota; reuse after "
                "the limit succeeds\n");
  }

  // =========================================================================
  // 3. max_context_tokens_per_session: the POLICY cap (below the model
  //    limit) + exact boundary + constructor clamp
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);
    // The model's max_seq_len is 64; the policy cap is 12 — the cap
    // RESTRICTS below the model limit.
    ServingController ctrl(sched, sm,
                           ServingLimits{-1, -1, /*max_context=*/12});

    // Constructor clamp: a cap above max_seq_len is clamped to it.
    ServingController clamped(sched, sm,
                              ServingLimits{-1, -1, /*max_context=*/999999});
    CHECK_EQ(clamped.limits().max_context_tokens_per_session, 64);

    SessionId s = 0;
    CHECK(ctrl.create_session(&s).ok);
    const SequenceId seq = sm.lookup(s)->sequence_id;

    // length 0 + 9 input + 4 max_new = 13 > 12 -> REJECTED (the model
    // limit of 64 would allow this).
    RequestId r = 0;
    const int reqs = sched.num_requests();
    const PoolSnapshot sp = snap(mgr);
    Status st = ctrl.admit_turn(s, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 4, -1,
                                kGreedy, &r);
    CHECK(!st.ok);
    CHECK(st.message.find("context limit") != std::string::npos);
    CHECK_EQ(r, static_cast<RequestId>(0));
    CHECK_EQ(mgr.lookup(seq)->length, 0);  // zero mutation
    CHECK_EQ(sched.num_requests(), reqs);
    CHECK(snap(mgr) == sp);
    CHECK_EQ(static_cast<int>(ctrl.stats().rejected_context_limit), 1);
    std::printf("  [ok] context policy cap: rejects below the model "
                "limit, zero mutation\n");

    // Exact boundary: 0 + 8 + 4 == 12 -> ALLOWED (== is accepted, the
    // scheduler's own boundary semantics).
    st = ctrl.admit_turn(s, {1, 2, 3, 4, 5, 6, 7, 8}, 4, -1, kGreedy, &r);
    CHECK(st.ok);
    CHECK(ctrl.run().ok);
    CHECK_EQ(mgr.lookup(seq)->length, 12);
    std::printf("  [ok] context policy cap: exact boundary (==) accepted\n");
  }

  // =========================================================================
  // 4. CANCEL and FORWARD FAILURE release the request quota
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm,
                           ServingLimits{-1, /*max_live_requests=*/1, 0});
    SessionId a = 0, b = 0;
    CHECK(ctrl.create_session(&a).ok);
    CHECK(ctrl.create_session(&b).ok);

    // CANCEL (before driving): the Waiting request is cancelled ->
    // capacity released immediately (synced inside cancel()).
    RequestId r1 = 0;
    CHECK(ctrl.admit_turn(a, {10, 20}, 8, -1, kGreedy, &r1).ok);
    CHECK_EQ(ctrl.stats().live_requests, 1);
    CHECK(ctrl.cancel(r1).ok);
    CHECK(sched.get(r1)->status == RequestStatus::Cancelled);
    CHECK_EQ(ctrl.stats().live_requests, 0);  // released immediately
    std::printf("  [ok] cancel releases the request quota\n");

    // FORWARD FAILURE: g0's commit fails (attempt index 2 = after the
    // 2-input prefill) -> the turn is Failed -> capacity released.
    RequestId r2 = 0;
    CHECK(ctrl.admit_turn(b, {30, 40}, 2, -1, kGreedy, &r2).ok);
    fwd.fail_attempt = 2;
    ctrl.run();
    fwd.fail_attempt = -1;
    CHECK(sched.get(r2)->status == RequestStatus::Failed);
    CHECK_EQ(ctrl.stats().live_requests, 0);  // released exactly once
    ctrl.sync();
    ctrl.sync();
    CHECK_EQ(ctrl.stats().live_requests, 0);
    // The session is LIVE at its committed boundary and can retry:
    CHECK(sm.lookup(b) != nullptr);
    RequestId r3 = 0;
    CHECK(ctrl.admit_turn(b, {50, 60}, 2, -1, kGreedy, &r3).ok);
    CHECK(ctrl.run().ok);
    std::printf("  [ok] forward failure releases the quota; the session "
                "retries from its committed boundary\n");
  }

  // =========================================================================
  // 5. Quota tri-state boundary (review fix): 0 = ZERO CAPACITY, -1 =
  //    UNLIMITED (previously a quota of 0 was misread as unlimited)
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);

    // max_sessions = 0: the FIRST create_session is REJECTED.
    ServingController ctrl(sched, sm,
                           ServingLimits{/*max_sessions=*/0, -1, 0});
    SessionId s = 0;
    const PoolSnapshot sp = snap(mgr);
    Status st = ctrl.create_session(&s);
    CHECK(!st.ok);
    CHECK(st.message.find("session limit") != std::string::npos);
    CHECK_EQ(s, static_cast<SessionId>(0));  // NO SessionId consumed
    CHECK_EQ(sm.num_sessions(), 0);
    CHECK(snap(mgr) == sp);  // no sequence / slot / page mutation
    CHECK_EQ(static_cast<int>(ctrl.stats().rejected_session_limit), 1);
    std::printf("  [ok] max_sessions = 0: zero capacity — first create "
                "rejected, zero mutation\n");

    // max_live_requests = 0: session creation is ALLOWED, but the
    // FIRST admit_turn is REJECTED.
    ServingController ctrl2(sched, sm,
                            ServingLimits{-1, /*max_live_requests=*/0, 0});
    SessionId s2 = 0;
    CHECK(ctrl2.create_session(&s2).ok);  // unlimited sessions
    // (session creation legitimately allocates a Delta slot — the
    // zero-mutation baseline for the admit rejection is AFTER it)
    const PoolSnapshot sp2 = snap(mgr);
    RequestId r = 0;
    const int reqs = sched.num_requests();
    const RequestId next = sched.next_request_id();
    const int len = mgr.lookup(sm.lookup(s2)->sequence_id)->length;
    st = ctrl2.admit_turn(s2, {10, 20}, 2, -1, kGreedy, &r);
    CHECK(!st.ok);
    CHECK(st.message.find("live request limit") != std::string::npos);
    CHECK_EQ(r, static_cast<RequestId>(0));  // NO RequestId consumed
    CHECK_EQ(sched.num_requests(), reqs);
    CHECK_EQ(sched.next_request_id(), next);
    CHECK_EQ(mgr.lookup(sm.lookup(s2)->sequence_id)->length, len);
    CHECK(snap(mgr) == sp2);  // no logical / KV / Delta mutation
    CHECK_EQ(static_cast<int>(ctrl2.stats().rejected_request_limit), 1);
    std::printf("  [ok] max_live_requests = 0: zero capacity — session "
                "created, first admit rejected, zero mutation\n");
  }

  {
    // -1 still means UNLIMITED (the tri-state fix did not change it).
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm,
                           ServingLimits{-1, -1, /*max_context=*/0});
    SessionId s1 = 0, s2 = 0, s3 = 0;
    CHECK(ctrl.create_session(&s1).ok);
    CHECK(ctrl.create_session(&s2).ok);
    CHECK(ctrl.create_session(&s3).ok);
    RequestId r1 = 0, r2 = 0;
    CHECK(ctrl.admit_turn(s1, {1, 2}, 2, -1, kGreedy, &r1).ok);
    CHECK(ctrl.admit_turn(s2, {3, 4}, 2, -1, kGreedy, &r2).ok);
    CHECK_EQ(ctrl.stats().live_requests, 2);  // two live at once, freely
    CHECK_EQ(static_cast<int>(ctrl.stats().rejected_session_limit), 0);
    CHECK_EQ(static_cast<int>(ctrl.stats().rejected_request_limit), 0);
    std::printf("  [ok] -1 still unlimited (sessions + concurrent live "
                "requests admitted freely)\n");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("test_serving_admission: PASS\n");
  return 0;
}
