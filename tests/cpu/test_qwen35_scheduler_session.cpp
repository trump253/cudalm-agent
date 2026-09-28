// CUDALM — v0.8 Phase C: scheduler + session integration contract gate
// (CPU test, NO checkpoint, NO model — the same deterministic fake
// forwarder discipline as test_qwen35_scheduler.cpp, over the REAL
// Qwen35StateManager pools + a real SessionManager + a real Scheduler
// (with a session manager attached)).
//
// Proves the Phase C CONTROL-PLANE semantics:
//   * ADMISSION FAILURES = ZERO MUTATION: invalid SessionId, invalid
//     token / eos / sampling / max_new, empty input, CONTEXT OVERFLOW
//     (length + input + max_new > max_seq_len — exact boundary accepted),
//     and BUSY session (one live request per session) all fail loud with
//     no half-request, no consumed RequestId, no session state change;
//   * INSTANCE IDENTITY GATE: a SessionManager bound to a DIFFERENT
//     Qwen35StateManager than the scheduler's is rejected (fail loud,
//     zero mutation on BOTH managers) — a numerical SequenceId
//     coincidence across managers must never be silently driven;
//   * NO SEQUENCE IS CREATED at session-turn admission (the request is
//     bound to the session's existing bound sequence);
//   * TERMINAL == NOT RETIRED: after Finished / Cancelled / Failed the
//     session's sequence stays LIVE and the session state is preserved
//     (only the Session lifecycle can reset/release it);
//   * COMMIT CONTRACT (sampled != committed): a session turn that
//     commits m generated tokens forwards input + m tokens (forward_count
//     and the session length include the stop-triggering token — the EOS
//     token IS committed); a CANCELLED / FAILED request's PENDING token
//     is NOT part of the session history (the session stays at the last
//     committed token boundary);
//   * NEXT TURN CONTINUES from the committed length (no replay — the
//     fake's per-sequence step counter proves the position derivation);
//   * max_new_tokens == 0 = an input-only turn (no sampling);
//   * the LEGACY path is untouched (frozen N + m - 1 + retire-once).
//
// Provenance: CUDALM-native (v0.8 Phase C).

#include "../../tests/common/check.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/scheduler.h"
#include "cudalm/session.h"

using namespace cudalm;

namespace {

// Small synthetic hybrid config (valid(); the same shape as
// test_qwen35_scheduler.cpp): 8 layers, interval 4 -> 2 full (layers 3,7)
// + 6 linear; max_seq_len 64.
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

// Deterministic CPU forwarder (the same contract as
// test_qwen35_scheduler.cpp's FakeForwarder): the logits of a forward
// depend ONLY on THAT SEQUENCE'S OWN forward (attempt) index — a
// sequence's token stream is a pure function of its own history,
// regardless of interleaving.
//
//   pick(step) = (3*step + 1) % vocab      (the greedy argmax)
//
// Fault injection: forward_token fails when the token == fail_token.
// (A failed attempt still advances the attempt counter — the counter
// models the FAKE's internal step index; the REAL committed count is the
// manager's sequence length.)
class FakeForwarder : public SequenceForwarder {
 public:
  int vocab = 1024;  // must match small_config().vocab_size
  int fail_token = -1;
  std::vector<std::pair<SequenceId, int>> log;  // (sid, token) attempt order

  Status forward_token(int token_id, SequenceId sequence_id,
                       Qwen35StateManager& mgr,
                       cudaStream_t /*stream*/) override {
    log.push_back({sequence_id, token_id});
    last_step_ = count_[sequence_id]++;
    if (token_id == fail_token) {
      // A FAILED forward commits NOTHING: no length / page mutation.
      return Status::error("fake forward fault (injected)");
    }
    // A SUCCESSFUL forward commits the token's metadata exactly as a
    // real committed forward would (page coverage for the position +
    // length + 1); the model math is faked.
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
    const int pick = pick_token(last_step_);
    (*out)[static_cast<std::size_t>(pick)] = __float2bfloat16(10.0f);
    return Status::ok_status();
  }

  int vocab_size() const override { return vocab; }

  int pick_token(int step) const {
    return static_cast<int>(
        (static_cast<std::uint64_t>(3) * static_cast<std::uint64_t>(step) + 1) %
        static_cast<std::uint64_t>(vocab));
  }

  // The tokens this fake forwarded (attempt order) for one sequence.
  std::vector<int> tokens_of(SequenceId sid) const {
    std::vector<int> out;
    for (const auto& p : log) {
      if (p.first == sid) out.push_back(p.second);
    }
    return out;
  }
  int attempts_of(SequenceId sid) const {
    return count_.count(sid) ? count_[sid] : 0;
  }

 private:
  mutable int last_step_ = -1;
  mutable std::map<SequenceId, int> count_;
};

// The zero-mutation snapshot (scheduler + manager + pool accounting).
struct Snapshot {
  int num_requests = 0;
  int next_request_id = 0;
  int num_sessions = 0;
  int num_live_sequences = 0;
  int used_pages = 0;
  int used_slots = 0;
  int session_length = -1;  // -1 = not applicable
};

Snapshot snap(const Scheduler& s, const SessionManager& sm,
              const Qwen35StateManager& m, SessionId sid) {
  Snapshot x;
  x.num_requests = s.num_requests();
  x.next_request_id = s.next_request_id();
  x.num_sessions = sm.num_sessions();
  x.num_live_sequences = m.num_live_sequences();
  x.used_pages = m.kv_pool().used_pages();
  x.used_slots = m.delta_pool().used_slots();
  if (sid != 0) {
    const Session* rec = sm.lookup(sid);
    if (rec != nullptr) {
      const SequenceState* seq = m.lookup(rec->sequence_id);
      if (seq != nullptr) x.session_length = seq->length;
    }
  }
  return x;
}

int check_snapshot(const char* name, const Snapshot& a, const Snapshot& b) {
  CHECK_EQ(a.num_requests, b.num_requests);
  CHECK_EQ(a.next_request_id, b.next_request_id);
  CHECK_EQ(a.num_sessions, b.num_sessions);
  CHECK_EQ(a.num_live_sequences, b.num_live_sequences);
  CHECK_EQ(a.used_pages, b.used_pages);
  CHECK_EQ(a.used_slots, b.used_slots);
  CHECK_EQ(a.session_length, b.session_length);
  std::printf("  [ok] %s: zero mutation\n", name);
  return 0;
}

const SamplingConfig kGreedy = SamplingConfig::greedy();

}  // namespace

int main() {
  std::printf("test_qwen35_scheduler_session: CUDALM v0.8 Phase C "
              "scheduler+session contract gate (fake forwarder, real pools)\n");
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const Qwen35Config cfg = small_config();

  {
    // ---- env: 8 pages / 6 slots (5 sessions + one legacy sequence) ----
    Qwen35StateManager mgr(cfg, /*pt=*/4, /*pages=*/8, /*slots=*/6, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Scheduler sched(fwd, mgr, stream, &sm);

    // ===================================================================
    // 1. ADMISSION FAILURES — ZERO MUTATION
    // ===================================================================
    {
      SessionId a = 0;
      CHECK(sm.create_session(&a).ok);
      const SequenceId sid = sm.lookup(a)->sequence_id;

      const Snapshot b0 = snap(sched, sm, mgr, a);

      // (a) unknown SessionId
      RequestId r = 0;
      Status s = sched.admit_session_turn(999, {50, 99}, 2, -1, kGreedy, &r);
      CHECK(!s.ok);
      CHECK(s.message.find("not live") != std::string::npos);
if (      check_snapshot("unknown SessionId", b0, snap(sched, sm, mgr, a))) return 1;

      // (b) empty input
      s = sched.admit_session_turn(a, {}, 2, -1, kGreedy, &r);
      CHECK(!s.ok);
      CHECK(s.message.find("empty") != std::string::npos);
if (      check_snapshot("empty input", b0, snap(sched, sm, mgr, a))) return 1;

      // (c) max_new_tokens < 0
      s = sched.admit_session_turn(a, {50}, -1, -1, kGreedy, &r);
      CHECK(!s.ok);
      CHECK(s.message.find("max_new_tokens < 0") != std::string::npos);
if (      check_snapshot("max_new_tokens < 0", b0, snap(sched, sm, mgr, a))) return 1;

      // (d) invalid eos (== vocab)
      s = sched.admit_session_turn(a, {50}, 1, fwd.vocab, kGreedy, &r);
      CHECK(!s.ok);
      CHECK(s.message.find("eos_token_id") != std::string::npos);
if (      check_snapshot("invalid eos", b0, snap(sched, sm, mgr, a))) return 1;

      // (e) invalid input token (== vocab)
      s = sched.admit_session_turn(a, {fwd.vocab}, 1, -1, kGreedy, &r);
      CHECK(!s.ok);
      CHECK(s.message.find("invalid input token id") != std::string::npos);
if (      check_snapshot("invalid input token", b0, snap(sched, sm, mgr, a))) return 1;

      // (f) invalid sampling config
      SamplingConfig bad = SamplingConfig{};
      bad.temperature = 1.0f;
      bad.top_k = -1;
      s = sched.admit_session_turn(a, {50}, 1, -1, bad, &r);
      CHECK(!s.ok);
      CHECK(s.message.find("sampling") != std::string::npos);
if (      check_snapshot("invalid sampling config", b0, snap(sched, sm, mgr, a))) return 1;

      // (g) CONTEXT OVERFLOW: length 62 + input 2 + max_new 1 = 65 > 64
      CHECK(mgr.set_length(sid, 62).ok);
      const Snapshot b62 = snap(sched, sm, mgr, a);
      s = sched.admit_session_turn(a, {50, 99}, 1, -1, kGreedy, &r);
      CHECK(!s.ok);
      CHECK(s.message.find("context overflow") != std::string::npos);
if (      check_snapshot("context overflow (62+2+1 > 64)", b62,
                     snap(sched, sm, mgr, a))) return 1;

      // (h) EXACT BOUNDARY ACCEPTED: length 61 + 2 + 1 == 64 -> ok
      CHECK(mgr.set_length(sid, 61).ok);
      s = sched.admit_session_turn(a, {50, 99}, 1, -1, kGreedy, &r);
      CHECK(s.ok);
      CHECK(r != 0);
      // (cancel it again — the boundary request is never executed here)
      CHECK(sched.cancel(r).ok);
      CHECK_EQ(mgr.lookup(sid)->length, 61);  // cancel: no mutation
      CHECK(!sched.session_busy(a));
      std::printf("  [ok] exact boundary (61+2+1 == 64) accepted\n");
    }
    {
      // (i) a scheduler WITHOUT a session manager fails loud
      FakeForwarder fwd2;
      Scheduler legacy(fwd2, mgr, stream);  // no SessionManager
      RequestId r = 0;
      Status s = legacy.admit_session_turn(1, {50}, 1, -1, kGreedy, &r);
      CHECK(!s.ok);
      CHECK(s.message.find("no session manager") != std::string::npos);
      std::printf("  [ok] scheduler without session manager: fail loud\n");
    }

    // ===================================================================
    // 1b. INSTANCE IDENTITY GATE: a SessionManager bound to a DIFFERENT
    //     Qwen35StateManager must be rejected (zero mutation on both).
    // ===================================================================
    {
      // Two independent managers; the FIRST session on each is bound to
      // the SAME numeric SequenceId (1) — exactly the collision the gate
      // guards: without it, driving sb through a scheduler over mgrA
      // would silently operate on mgrA's pools (SequenceId 1 there is a
      // DIFFERENT sequence).
      Qwen35StateManager mgrA(cfg, 4, 4, 4, stream);
      Qwen35StateManager mgrB(cfg, 4, 4, 4, stream);
      SessionManager smA(mgrA);
      SessionManager smB(mgrB);
      SessionId sa = 0, sb = 0;
      CHECK(smA.create_session(&sa).ok);
      CHECK(smB.create_session(&sb).ok);
      const SequenceId seqA_id = smA.lookup(sa)->sequence_id;
      const SequenceId seqB_id = smB.lookup(sb)->sequence_id;
      CHECK_EQ(seqA_id, seqB_id);  // the numerical coincidence

      // A scheduler over mgrA given a SessionManager bound to mgrB:
      FakeForwarder fwdX;
      Scheduler wrong(fwdX, mgrA, stream, &smB);
      // Snapshot BOTH managers + the (mismatched) scheduler BEFORE the
      // (rejected) admission:
      const int a_pages = mgrA.kv_pool().used_pages();
      const int a_slots = mgrA.delta_pool().used_slots();
      const int a_len = mgrA.lookup(seqA_id)->length;
      const int b_pages = mgrB.kv_pool().used_pages();
      const int b_slots = mgrB.delta_pool().used_slots();
      const int b_len = mgrB.lookup(seqB_id)->length;
      const int nreq = wrong.num_requests();
      const RequestId nid = wrong.next_request_id();

      RequestId r = 0;
      Status s = wrong.admit_session_turn(sb, {50}, 1, -1, kGreedy, &r);
      CHECK(!s.ok);
      CHECK(s.message.find("DIFFERENT") != std::string::npos);
      // ZERO MUTATION: no request registered, no RequestId consumed, and
      // NEITHER manager changed (length / KV pages / Delta slots):
      CHECK_EQ(wrong.num_requests(), nreq);
      CHECK_EQ(wrong.next_request_id(), nid);
      CHECK_EQ(mgrA.kv_pool().used_pages(), a_pages);
      CHECK_EQ(mgrA.delta_pool().used_slots(), a_slots);
      CHECK_EQ(mgrA.lookup(seqA_id)->length, a_len);
      CHECK_EQ(mgrB.kv_pool().used_pages(), b_pages);
      CHECK_EQ(mgrB.delta_pool().used_slots(), b_slots);
      CHECK_EQ(mgrB.lookup(seqB_id)->length, b_len);
      std::printf("  [ok] instance identity: wrong-manager session manager "
                  "rejected, zero mutation on both\n");
    }

    // ===================================================================
    // 2. TURN LIFECYCLE: no sequence created, terminal != retired,
    //    commit contract (input + m forwards), next turn continues
    // ===================================================================
    {
      SessionId a = 0;
      CHECK(sm.create_session(&a).ok);
      const SequenceId sid = sm.lookup(a)->sequence_id;
      const int base_live = mgr.num_live_sequences();  // + this session

      RequestId r1 = 0;
      CHECK(sched.admit_session_turn(a, {50, 99}, 2, -1, kGreedy, &r1).ok);
      // NO new sequence at admission (bound to the existing one):
      CHECK_EQ(mgr.num_live_sequences(), base_live);
      CHECK_EQ(mgr.lookup(sid)->length, 0);  // nothing forwarded yet

      // (busy: a second live turn on the same session is rejected)
      const Snapshot b1 = snap(sched, sm, mgr, a);
      RequestId rX = 0;
      Status s = sched.admit_session_turn(a, {7}, 1, -1, kGreedy, &rX);
      CHECK(!s.ok);
      CHECK(s.message.find("live request") != std::string::npos);
      CHECK(sched.session_busy(a));
if (      check_snapshot("busy session (second live turn rejected)", b1,
                     snap(sched, sm, mgr, a))) return 1;

      CHECK(sched.run().ok);
      const Request* p1 = sched.get(r1);
      CHECK(p1 != nullptr);
      CHECK(p1->status == RequestStatus::Finished);
      CHECK(p1->finish_reason == FinishReason::MaxNewTokens);
      // COMMIT CONTRACT: 2 inputs + 2 generated, ALL committed.
      CHECK_EQ(p1->forward_count, 4);
      CHECK_EQ(p1->committed_generated, 2);
      CHECK(p1->generated ==
            (std::vector<int>{fwd.pick_token(1), fwd.pick_token(2)}));
      CHECK_EQ(mgr.lookup(sid)->length, 4);  // input + generated committed
      // TERMINAL != RETIRED: the sequence (and the session) stay live.
      CHECK_EQ(mgr.num_live_sequences(), base_live);
      CHECK(sm.lookup(a) != nullptr);
      CHECK(!sched.session_busy(a));  // the session is free for turn 2

      // NEXT TURN CONTINUES from the committed length (no replay):
      RequestId r2 = 0;
      CHECK(sched.admit_session_turn(a, {9}, 1, -1, kGreedy, &r2).ok);
      CHECK(sched.run().ok);
      const Request* p2 = sched.get(r2);
      CHECK(p2->status == RequestStatus::Finished);
      CHECK_EQ(p2->forward_count, 2);  // 1 input + 1 generated
      CHECK_EQ(p2->committed_generated, 1);
      // The fake's per-sequence step counter proves continuation: A's
      // sequence had already made 4 forwards (steps 0..3) in turn 1, so
      // turn 2's input is forwarded at step 4 and g0 = pick(4). A REPLAY
      // would have produced pick(0) = 1 instead of pick(4) = 13.
      CHECK(p2->generated == (std::vector<int>{fwd.pick_token(4)}));
      CHECK_EQ(mgr.lookup(sid)->length, 6);
      // The sequence's full committed token history (fwd log):
      const std::vector<int> hist = fwd.tokens_of(sid);
      CHECK(hist == (std::vector<int>{50, 99, fwd.pick_token(1),
                                      fwd.pick_token(2), 9, fwd.pick_token(4)}));
      std::printf("  [ok] turn lifecycle: no create / not retired / commit "
                  "contract / next turn continues (no replay)\n");
      CHECK(sched.cancel(sched.next_request_id() + 1000).ok == false);
    }

    // ===================================================================
    // 3. CANCEL: session live, committed kept, PENDING token not history
    // ===================================================================
    {
      SessionId b = 0;
      CHECK(sm.create_session(&b).ok);
      const SequenceId sid = sm.lookup(b)->sequence_id;
      RequestId r3 = 0;
      CHECK(sched.admit_session_turn(b, {10, 20}, 4, -1, kGreedy, &r3).ok);
      CHECK(sched.step().ok);  // input 10 (step 0)
      CHECK(sched.step().ok);  // input 20 (step 1) -> prefill done, g0
                               // = pick(1) = 4 sampled (PENDING)
      const Request* p3 = sched.get(r3);
      CHECK(p3->status == RequestStatus::Running);
      CHECK_EQ(mgr.lookup(sid)->length, 2);
      CHECK(p3->generated == (std::vector<int>{fwd.pick_token(1)}));
      CHECK_EQ(p3->committed_generated, 0);  // g0 sampled, NOT committed

      CHECK(sched.cancel(r3).ok);
      p3 = sched.get(r3);
      CHECK(p3->status == RequestStatus::Cancelled);
      CHECK(p3->finish_reason == FinishReason::Cancelled);
      // The PENDING token is NOT part of the session history:
      CHECK_EQ(mgr.lookup(sid)->length, 2);  // only the 2 committed inputs
      CHECK_EQ(p3->committed_generated, 0);
      // The session (and its sequence) stay live (the earlier scenarios'
      // sessions are live too):
      CHECK(sm.lookup(b) != nullptr);
      CHECK_EQ(mgr.num_live_sequences(), 3);  // scenarios 1, 2, 3
      CHECK(!sched.session_busy(b));

      // The session can take its NEXT turn, from the committed boundary:
      RequestId r4 = 0;
      CHECK(sched.admit_session_turn(b, {9}, 1, -1, kGreedy, &r4).ok);
      CHECK(sched.run().ok);
      const Request* p4 = sched.get(r4);
      CHECK(p4->status == RequestStatus::Finished);
      CHECK_EQ(mgr.lookup(sid)->length, 4);  // 2 + 1 + 1
      std::printf("  [ok] cancel: session live, committed kept, pending "
                  "token not history, next turn continues\n");
    }

    // ===================================================================
    // 4. FORWARD FAILURE: session live, committed kept, failed token not
    //    committed (a fresh fake/manager — the fake's attempt counter is
    //    local to it)
    // ===================================================================
    {
      Qwen35StateManager m2(cfg, 4, 4, 4, stream);
      SessionManager s2(m2);
      FakeForwarder f2;
      Scheduler s2sched(f2, m2, stream, &s2);
      SessionId c = 0;
      CHECK(s2.create_session(&c).ok);
      const SequenceId sid = s2.lookup(c)->sequence_id;
      // Input {50, 99}: g0 = pick(1) = 4 -> make the forward of 4 FAIL.
      f2.fail_token = f2.pick_token(1);
      RequestId r5 = 0;
      CHECK(s2sched.admit_session_turn(c, {50, 99}, 3, -1, kGreedy, &r5).ok);
      Status s = s2sched.run();
      CHECK(!s.ok);  // the injected fault
      const Request* p5 = s2sched.get(r5);
      CHECK(p5->status == RequestStatus::Failed);
      CHECK(p5->finish_reason == FinishReason::Failed);
      CHECK(p5->generated == (std::vector<int>{f2.pick_token(1)}));
      CHECK_EQ(p5->committed_generated, 0);  // g0 sampled, NOT committed
      CHECK_EQ(p5->forward_count, 2);  // the 2 inputs only
      CHECK_EQ(m2.lookup(sid)->length, 2);  // the last committed boundary
      // The session stays live — its next turn continues from length 2:
      f2.fail_token = -1;  // clear the fault
      RequestId r6 = 0;
      CHECK(s2sched.admit_session_turn(c, {9}, 1, -1, kGreedy, &r6).ok);
      CHECK(s2sched.run().ok);
      const Request* p6 = s2sched.get(r6);
      CHECK(p6->status == RequestStatus::Finished);
      CHECK_EQ(m2.lookup(sid)->length, 4);  // 2 + 1 + 1
      CHECK(s2.lookup(c) != nullptr);
      std::printf("  [ok] forward failure: session live, committed kept, "
                  "failed token not committed, next turn continues\n");
    }

    // ===================================================================
    // 5. max_new_tokens == 0: an INPUT-ONLY turn (no sampling)
    // ===================================================================
    {
      SessionId d = 0;
      CHECK(sm.create_session(&d).ok);
      const SequenceId sid = sm.lookup(d)->sequence_id;
      RequestId r7 = 0;
      CHECK(sched.admit_session_turn(d, {1, 2, 3}, 0, -1, kGreedy, &r7).ok);
      CHECK(sched.run().ok);
      const Request* p7 = sched.get(r7);
      CHECK(p7->status == RequestStatus::Finished);
      CHECK(p7->finish_reason == FinishReason::MaxNewTokens);
      CHECK(p7->generated.empty());
      CHECK_EQ(p7->forward_count, 3);  // inputs only
      CHECK_EQ(mgr.lookup(sid)->length, 3);
      std::printf("  [ok] max_new_tokens == 0: input-only turn\n");
    }

    // ===================================================================
    // 6. EOS via the scheduler: the EOS token IS committed, then terminal
    // ===================================================================
    {
      SessionId e = 0;
      CHECK(sm.create_session(&e).ok);
      const SequenceId sid = sm.lookup(e)->sequence_id;
      // Input {50, 99}: g0 = pick(1) = 4, g1 = pick(2) = 7. Gate on 7.
      const int eos = fwd.pick_token(2);
      RequestId r8 = 0;
      CHECK(sched.admit_session_turn(e, {50, 99}, 5, eos, kGreedy, &r8).ok);
      CHECK(sched.run().ok);
      const Request* p8 = sched.get(r8);
      CHECK(p8->status == RequestStatus::Finished);
      CHECK(p8->finish_reason == FinishReason::Eos);
      CHECK(p8->generated == (std::vector<int>{fwd.pick_token(1), eos}));
      CHECK_EQ(p8->committed_generated, 2);  // EOS committed like the rest
      CHECK_EQ(p8->forward_count, 4);  // 2 inputs + 2 generated (NO lag)
      CHECK_EQ(mgr.lookup(sid)->length, 4);
      std::printf("  [ok] EOS: committed then terminal (forward_count 4, "
                  "length 4)\n");
    }

    // ===================================================================
    // 7. LEGACY PATH UNTOUCHED (frozen N + m - 1 + retire-once)
    // ===================================================================
    {
      Scheduler::Spec spec;
      spec.prompt = {50, 99};
      spec.max_new_tokens = 2;
      spec.eos_token_id = -1;
      spec.sampling = kGreedy;
      RequestId r9 = 0;
      CHECK(sched.admit(spec, &r9).ok);
      CHECK(sched.run().ok);
      const Request* p9 = sched.get(r9);
      CHECK(p9->status == RequestStatus::Finished);
      CHECK(p9->finish_reason == FinishReason::MaxNewTokens);
      CHECK_EQ(p9->forward_count, 3);  // N + m - 1 (frozen legacy)
      CHECK_EQ(p9->committed_generated, 0);
      CHECK(p9->ownership == RequestOwnership::SequenceOwned);
      // Its sequence was retired exactly once on the terminal transition
      // (the 5 sessions' bound sequences are all still live):
      CHECK_EQ(mgr.num_live_sequences(), 5);
      std::printf("  [ok] legacy path untouched (N+m-1, retire-once)\n");
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("test_qwen35_scheduler_session: PASS\n");
  return 0;
}
