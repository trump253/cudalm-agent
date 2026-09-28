// CUDALM — v0.9 Phase B: committed-token streaming contract gate
// (CPU test — NO checkpoint, NO model: the deterministic fake forwarder
// (Phase C/D pattern — a SUCCESSFUL forward commits real sequence
// metadata) + real pools + real SessionManager + real Scheduler + the
// ServingController streaming / cancel / deadline API).
//
// Proves the streaming contract (COMMIT-BEFORE-VISIBLE):
//   * the PENDING g0 (sampled after the last input forward) is NOT
//     stream-visible; g0 is emitted only AFTER the forward that
//     commits it; g1 (pending) stays invisible until its forward;
//   * exactly-once + in order (a later poll of the same request emits
//     NOTHING new); the final max-new / EOS token is emitted BEFORE
//     the RequestTerminal event;
//   * CANCEL: the committed prefix is drainable, the PENDING token is
//     never emitted, the session stays live at the committed boundary
//     and the NEXT turn continues from it, the quota is released;
//   * DEADLINE (fake monotonic clock): progress before the deadline,
//     cancel BEFORE the next forward once expired (no additional token
//     commit), the terminal event carries deadline_exceeded = true;
//   * A/B INTERLEAVING: one step can commit for both requests; each
//     request's event stream concatenates to exactly its own final
//     committed generated ids (no cross-request contamination).
//
// Provenance: CUDALM-native (v0.9 Phase B).

#include "../../tests/common/check.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
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

// Deterministic CPU forwarder (Phase C pattern). Token picked at
// forward index s is (3*s + 1) % vocab — per SEQUENCE.
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
    const int pick =
        static_cast<int>((static_cast<std::uint64_t>(3) *
                             static_cast<std::uint64_t>(last_step_) +
                         1) %
                         static_cast<std::uint64_t>(vocab));
    (*out)[static_cast<std::size_t>(pick)] = __float2bfloat16(10.0f);
    return Status::ok_status();
  }

  int vocab_size() const override { return vocab; }
  // Per-sequence forward count (commit accounting checks).
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

// Collect the token ids + the terminal event of ONE request from an
// event stream (asserts per-request order + exactly one terminal).
struct Collected {
  std::vector<int> tokens;
  bool terminal = false;
  RequestStatus status = RequestStatus::Waiting;
  FinishReason reason = FinishReason::None;
  bool deadline_exceeded = false;
  int terminal_position = -1;  // index in the merged stream (-1 = none)
};

// Returns 0 on success (CHECK helpers must return int).
int collect(const std::vector<ServingEvent>& events, RequestId rid,
            Collected* c) {
  for (std::size_t i = 0; i < events.size(); ++i) {
    const ServingEvent& e = events[i];
    if (e.request_id != rid) continue;
    if (e.kind == ServingEventKind::Token) {
      CHECK(!c->terminal);  // a Token never follows its terminal event
      c->tokens.push_back(e.token_id);
    } else {
      CHECK(!c->terminal);  // exactly ONE terminal event
      c->terminal = true;
      c->status = e.status;
      c->reason = e.finish_reason;
      c->deadline_exceeded = e.deadline_exceeded;
      c->terminal_position = static_cast<int>(i);
    }
  }
  return 0;
}

}  // namespace

int main() {
  std::printf("test_serving_streaming: CUDALM v0.9 Phase B committed-"
              "token streaming contract gate (fake forwarder, real pools)\n");

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const Qwen35Config cfg = small_config();
  const SamplingConfig kGreedy = SamplingConfig::greedy();

  // =========================================================================
  // 1. COMMIT-BEFORE-VISIBLE: pending g0 invisible, emitted only after its
  //    committing forward; exactly-once; final token before terminal
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm, ServingLimits{-1, -1, 0});

    SessionId s = 0;
    CHECK(ctrl.create_session(&s).ok);
    const SequenceId seq = sm.lookup(s)->sequence_id;
    // Turn: 2 input tokens, max_new 2. Fake picks: g0 = 4 (after the
    // 2nd input forward), g1 = 7 (after the g0 decode forward).
    RequestId r = 0;
    CHECK(ctrl.admit_turn(s, {10, 20}, 2, -1, kGreedy, &r).ok);

    std::vector<ServingEvent> ev = ctrl.step_stream();  // prefill t0
    Collected c;
    CHECK_EQ(collect(ev, r, &c), 0);
    CHECK(c.tokens.empty());  // nothing committed yet
    CHECK(!c.terminal);
    CHECK_EQ(fwd.forwards(seq), 1);

    ev = ctrl.step_stream();  // prefill t1 + SAMPLE g0 (pending)
    CHECK_EQ(collect(ev, r, &c), 0);
    CHECK(c.tokens.empty());  // g0 is PENDING — NOT stream-visible
    CHECK(!c.terminal);
    CHECK_EQ(fwd.forwards(seq), 2);

    ev = ctrl.step_stream();  // decode: COMMIT g0 + sample g1 (pending)
    CHECK_EQ(collect(ev, r, &c), 0);
    CHECK(c.tokens.size() == 1 && c.tokens[0] == 4);  // g0 emitted NOW
    CHECK(!c.terminal);  // g1 still pending — the turn is not over
    CHECK_EQ(fwd.forwards(seq), 3);

    // EXACTLY-ONCE: a poll WITHOUT new forwards emits nothing new.
    std::vector<ServingEvent> pol;
    CHECK(ctrl.poll(r, &pol).ok);
    CHECK(pol.empty());

    ev = ctrl.step_stream();  // decode: COMMIT g1 (= max_new) -> terminal
    CHECK_EQ(collect(ev, r, &c), 0);
    // The FINAL max-new token is emitted BEFORE the terminal report:
    CHECK(c.tokens.size() == 2 && c.tokens[1] == 7);
    CHECK(c.terminal);
    CHECK(c.status == RequestStatus::Finished);
    CHECK(c.reason == FinishReason::MaxNewTokens);
    CHECK_EQ(mgr.lookup(seq)->length, 4);  // 2 input + 2 committed
    std::printf("  [ok] pending g0 invisible; emitted after commit; "
                "exactly-once; final token before terminal\n");
  }

  // =========================================================================
  // 2. EOS: the EOS token is committed (emitted) before the terminal
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm, ServingLimits{-1, -1, 0});
    SessionId s = 0;
    CHECK(ctrl.create_session(&s).ok);
    const SequenceId seq = sm.lookup(s)->sequence_id;
    // eos = 7: g1 (the token committed by the 3rd forward) is EOS.
    RequestId r = 0;
    CHECK(ctrl.admit_turn(s, {10, 20}, 5, 7, kGreedy, &r).ok);
    std::vector<ServingEvent> all = ctrl.run_stream();
    Collected c;
    CHECK_EQ(collect(all, r, &c), 0);
    CHECK(c.tokens.size() == 2 && c.tokens[0] == 4 && c.tokens[1] == 7);
    CHECK(c.terminal);
    CHECK(c.status == RequestStatus::Finished);
    CHECK(c.reason == FinishReason::Eos);
    CHECK_EQ(fwd.forwards(seq), 4);  // 2 input + g0 decode + EOS(g1) decode
    CHECK_EQ(mgr.lookup(seq)->length, 4);
    std::printf("  [ok] EOS token emitted before the terminal report\n");
  }

  // =========================================================================
  // 3. CANCEL: committed prefix drainable, pending never emitted, the
  //    session continues from the committed boundary
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm,
                           ServingLimits{-1, /*max_live=*/1, 0});
    SessionId s = 0;
    CHECK(ctrl.create_session(&s).ok);
    const SequenceId seq = sm.lookup(s)->sequence_id;
    RequestId r = 0;
    CHECK(ctrl.admit_turn(s, {10, 20}, 2, -1, kGreedy, &r).ok);
    std::vector<ServingEvent> all;  // the events of the three pre-cancel
                                     // steps (drained by each step_stream)
    for (int i = 0; i < 3; ++i) {
      std::vector<ServingEvent> ev = ctrl.step_stream();
      all.insert(all.end(), ev.begin(), ev.end());
    }
    // (step 3 COMMITted g0 (=4) — already drained above — and g1 is
    // PENDING)

    // Cancel with g1 still PENDING:
    CHECK(ctrl.cancel(r).ok);
    CHECK(sched.get(r)->status == RequestStatus::Cancelled);
    CHECK_EQ(ctrl.stats().live_requests, 0);  // quota released immediately
    // The committed prefix is already drained; the PENDING token never
    // emits; the terminal event arrives via poll:
    std::vector<ServingEvent> pol;
    CHECK(ctrl.poll(r, &pol).ok);
    all.insert(all.end(), pol.begin(), pol.end());
    Collected c;
    CHECK_EQ(collect(all, r, &c), 0);
    CHECK(c.tokens.size() == 1 && c.tokens[0] == 4);  // only the committed
    CHECK(c.terminal);
    CHECK(c.status == RequestStatus::Cancelled);
    CHECK(c.reason == FinishReason::Cancelled);
    CHECK(!c.deadline_exceeded);
    CHECK_EQ(fwd.forwards(seq), 3);  // NO forward after the cancel
    CHECK_EQ(mgr.lookup(seq)->length, 3);  // the committed boundary

    // The session is LIVE and the next turn CONTINUES from the boundary:
    RequestId r2 = 0;
    CHECK(ctrl.admit_turn(s, {30, 40}, 2, -1, kGreedy, &r2).ok);
    std::vector<ServingEvent> all2 = ctrl.run_stream();
    Collected c2;
    CHECK_EQ(collect(all2, r2, &c2), 0);
    CHECK(c2.tokens.size() == 2);
    CHECK(c2.terminal);
    CHECK_EQ(mgr.lookup(seq)->length, 3 + 2 + 2);  // boundary + new turn
    std::printf("  [ok] cancel: committed prefix drainable, pending never "
                "emitted, next turn continues from the boundary\n");
  }

  // =========================================================================
  // 4. DEADLINE (fake monotonic clock): cooperative boundary between
  //    scheduler steps; no additional token commit after expiry
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);
    FakeClock clock;
    ServingController ctrl(sched, sm, ServingLimits{-1, -1, 0}, &clock);
    SessionId s = 0;
    CHECK(ctrl.create_session(&s).ok);
    const SequenceId seq = sm.lookup(s)->sequence_id;
    // Deadline at t0 + 2 ms; a step is taken at ms 0 and ms 1 (progress
    // before the deadline), the ms-2 step must CANCEL before forwarding.
    RequestId r = 0;
    CHECK(ctrl.admit_turn(s, {10, 20}, 2, -1, kGreedy, &r,
                          clock.now() + std::chrono::milliseconds(2))
              .ok);

    ctrl.step_stream();  // ms 0: not expired -> prefill t0
    clock.advance_ms(1);
    ctrl.step_stream();  // ms 1: not expired -> prefill t1 + g0 pending
    CHECK_EQ(fwd.forwards(seq), 2);
    CHECK(sched.get(r)->status != RequestStatus::Cancelled);
    clock.advance_ms(1);  // ms 2: EXPIRED
    std::vector<ServingEvent> ev = ctrl.step_stream();
    // Cancelled BEFORE the next forward: no additional token commit.
    CHECK_EQ(fwd.forwards(seq), 2);
    CHECK_EQ(mgr.lookup(seq)->length, 2);
    Collected c;
    CHECK_EQ(collect(ev, r, &c), 0);
    CHECK(c.tokens.empty());  // g0 was PENDING — never committed/emitted
    CHECK(c.terminal);
    CHECK(c.status == RequestStatus::Cancelled);
    CHECK(c.deadline_exceeded);  // the serving-layer termination reason
    CHECK_EQ(ctrl.stats().live_requests, 0);  // quota released
    // The session is live; the next turn continues from the boundary:
    RequestId r2 = 0;
    CHECK(ctrl.admit_turn(s, {10, 20}, 2, -1, kGreedy, &r2).ok);
    CHECK(ctrl.run_stream().size() >= 2);
    CHECK_EQ(mgr.lookup(seq)->length, 2 + 2 + 2);
    std::printf("  [ok] deadline: cancel before the next forward, no extra "
                "commit, deadline_exceeded reported\n");
  }

  // =========================================================================
  // 5. A/B INTERLEAVING: one step can commit for both; per-request event
  //    streams are exactly their own committed tokens (no contamination)
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, 4, 8, 4, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm,
                           ServingLimits{-1, /*max_live=*/2, 0});
    SessionId a = 0, b = 0;
    CHECK(ctrl.create_session(&a).ok);
    CHECK(ctrl.create_session(&b).ok);
    const SequenceId sa = sm.lookup(a)->sequence_id;
    const SequenceId sb = sm.lookup(b)->sequence_id;
    RequestId ra = 0, rb = 0;
    // A: 2 input + 2 gen (fake picks from sa's sequence); B: 1 input +
    // 2 gen (its own sequence — the per-sequence fake makes A's and B's
    // token streams different prefixes of (3s+1)).
    CHECK(ctrl.admit_turn(a, {10, 20}, 2, -1, kGreedy, &ra).ok);
    CHECK(ctrl.admit_turn(b, {30}, 2, -1, kGreedy, &rb).ok);

    std::vector<ServingEvent> all;
    for (int i = 0; i < 8 && ctrl.stats().live_requests > 0; ++i) {
      std::vector<ServingEvent> ev = ctrl.step_stream();
      all.insert(all.end(), ev.begin(), ev.end());
    }
    Collected ca, cb;
    CHECK_EQ(collect(all, ra, &ca), 0);
    CHECK_EQ(collect(all, rb, &cb), 0);
    CHECK(ca.terminal && cb.terminal);
    CHECK(ca.status == RequestStatus::Finished);
    CHECK(cb.status == RequestStatus::Finished);
    // Each request's event stream == its OWN final committed generated
    // ids (concatenated streamed ids == final generated ids):
    CHECK(ca.tokens == sched.get(ra)->generated);
    CHECK(cb.tokens == sched.get(rb)->generated);
    CHECK(ca.tokens.size() == 2 && cb.tokens.size() == 2);
    // No cross-request contamination: every event carries one of the two
    // ids, and the per-request order in the MERGED stream is preserved
    // (collect would have failed a terminal-before-token ordering).
    for (const ServingEvent& e : all) {
      CHECK(e.request_id == ra || e.request_id == rb);
    }
    CHECK(ca.tokens != cb.tokens);  // the two streams are distinct
    CHECK_EQ(mgr.lookup(sa)->length, 4);
    CHECK_EQ(mgr.lookup(sb)->length, 3);
    std::printf("  [ok] A/B interleaving: per-request streams exact, no "
                "cross-request contamination\n");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("test_serving_streaming: PASS\n");
  return 0;
}
