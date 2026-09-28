// CUDALM — v0.8 Phase D: text-level session facade contract gate (CPU
// test — NO model / NO GPU: the REAL native tokenizer artifact + the
// deterministic fake forwarder (the Phase C pattern, extended so a
// SUCCESSFUL forward commits real sequence metadata and a FAILED one
// commits nothing) + real state pools + real SessionManager + real
// Scheduler, driven through the Qwen35SessionTextGenerator facade).
//
// Proves the Phase D TEXT-SESSION CONTRACT end to end at the control
// plane:
//   * ENCODE -> SESSION-BOUND REQUEST -> DECODE: a text turn produces
//     input ids == the native encode, generated ids == the (deterministic
//     fake) continuation, generated text == the native decode of those
//     ids, and the context length == input + committed generated;
//   * INCREMENTAL MULTI-TURN: turn 2 encodes ONLY the new text and
//     CONTINUES from the committed length — the fake's per-sequence step
//     counter proves no replay of turn 1;
//   * RESET restarts the session (length 0, same SessionId) and the next
//     turn works;
//   * ERRORS PRODUCE NO HALF-TURN: invalid UTF-8, empty text, unknown
//     session and CONTEXT OVERFLOW all fail loud with zero mutation (no
//     registered request, no consumed RequestId, session length and pool
//     accounting exactly unchanged, session live).
//
// Self-skips (77) when the tokenizer artifact is absent.
//
// Provenance: CUDALM-native (v0.8 Phase D).

#include "../../tests/common/check.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/qwen35_tokenizer.h"
#include "cudalm/scheduler.h"
#include "cudalm/session.h"
#include "cudalm/session_text_generator.h"

using namespace cudalm;

namespace {

bool file_exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

// Small synthetic hybrid config (the Phase C CPU gate's shape):
// max_seq_len 64 (the overflow arithmetic below), vocab irrelevant here
// (the fake forwarder carries the model's real vocab).
Qwen35Config small_config() {
  Qwen35Config c;
  c.hidden_size = 256;
  c.num_hidden_layers = 8;
  c.intermediate_size = 512;
  c.vocab_size = Qwen35Tokenizer::kModelVocabSize;
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

// Deterministic CPU forwarder (the Phase C CPU gate's fake, same pick
// function — the greedy argmax at attempt index t is pick(t) =
// (3t + 1) % vocab). A SUCCESSFUL forward commits real sequence metadata
// (page coverage + length + 1); a FAILED one commits NOTHING.
class FakeForwarder : public SequenceForwarder {
 public:
  int vocab = static_cast<int>(Qwen35Tokenizer::kModelVocabSize);
  int fail_token = -1;  // fail when the TOKEN equals this (token-based)
  int fail_attempt = -1;  // fail when the per-sequence ATTEMPT INDEX
                          // equals this (deterministic — cannot collide
                          // with an input token id)
  std::vector<std::pair<SequenceId, int>> log;  // (sid, token) attempts

  Status forward_token(int token_id, SequenceId sequence_id,
                       Qwen35StateManager& mgr,
                       cudaStream_t /*stream*/) override {
    log.push_back({sequence_id, token_id});
    last_step_ = count_[sequence_id]++;
    if (token_id == fail_token || last_step_ == fail_attempt) {
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <tokenizer.cudaltk>\n", argv[0]);
    return 2;
  }
  if (!file_exists(argv[1])) {
    std::fprintf(stderr,
                 "[SKIP] qwen35 session text: tokenizer artifact absent at "
                 "%s\n",
                 argv[1]);
    return 77;
  }

  std::printf("test_qwen35_session_text: CUDALM v0.8 Phase D text-session "
              "facade contract gate (real tokenizer, fake forwarder, real "
              "pools)\n");

  std::unique_ptr<Qwen35Tokenizer> tok;
  Status s = Qwen35Tokenizer::load(argv[1], &tok);
  CHECK(s.ok);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const Qwen35Config cfg = small_config();
  const SamplingConfig kGreedy = SamplingConfig::greedy();

  {
    Qwen35StateManager mgr(cfg, /*pt=*/4, /*pages=*/8, /*slots=*/3, stream);
    SessionManager sm(mgr);
    FakeForwarder fwd;
    Qwen35SessionTextGenerator gen(fwd, *tok, mgr, sm, stream);

    // ===================================================================
    // 1. ENCODE -> SESSION-BOUND REQUEST -> DECODE (turn 1)
    // ===================================================================
    SessionId sid = 0;
    CHECK(gen.create_session(&sid).ok);
    const SequenceId seq = sm.lookup(sid)->sequence_id;
    const RequestId base_next = gen.scheduler().next_request_id();
    int n1 = 0;  // turn 1's input size (used by turn 2's continuity check)
    {
      const std::string t1 = "The capital of France is";
      std::vector<std::uint32_t> exp_in;
      CHECK(tok->encode(t1, &exp_in).ok);
      const int n = static_cast<int>(exp_in.size());
      n1 = n;
      CHECK(n > 0);

      SessionTextTurnResult r =
          gen.generate_turn(sid, t1, 2, kGreedy, -1);  // no EOS gate
      CHECK(r.ok);
      CHECK_EQ(static_cast<int>(r.input_token_ids.size()), n);
      for (int i = 0; i < n; ++i) {
        CHECK_EQ(r.input_token_ids[static_cast<std::size_t>(i)],
                 static_cast<int>(exp_in[static_cast<std::size_t>(i)]));
      }
      // Deterministic continuation: g0 = pick(n-1), g1 = pick(n).
      CHECK(r.generated_token_ids ==
            (std::vector<int>{fwd.pick_token(n - 1), fwd.pick_token(n)}));
      CHECK_EQ(r.forward_count, n + 2);  // ALL committed (no lag)
      std::string exp_text;
      {
        std::vector<std::uint32_t> g(r.generated_token_ids.begin(),
                                     r.generated_token_ids.end());
        CHECK(tok->decode(g.data(), g.size(), false, &exp_text).ok);
      }
      CHECK(r.generated_text == exp_text);  // decode == native decode
      CHECK_EQ(r.context_length, n + 2);  // input + generated, committed
      CHECK(r.stop_reason == FinishReason::MaxNewTokens);
      CHECK_EQ(r.request_id, base_next);  // the turn's request id
      CHECK_EQ(mgr.lookup(seq)->length, n + 2);
      std::printf("  [ok] turn 1: encode -> session-bound request -> decode "
                  "(n=%d, +2 generated, all committed)\n", n);
    }

    // ===================================================================
    // 2. TURN 2: encodes ONLY the new text, CONTINUES (no replay)
    // ===================================================================
    {
      const std::string t2 = "and the weather is";
      std::vector<std::uint32_t> exp_in;
      CHECK(tok->encode(t2, &exp_in).ok);
      const int n2 = static_cast<int>(exp_in.size());
      CHECK(n2 > 0);

      SessionTextTurnResult r =
          gen.generate_turn(sid, t2, 2, kGreedy, -1);
      CHECK(r.ok);
      // Input ids == ONLY turn 2's text (turn 1 is NOT re-encoded):
      for (int i = 0; i < n2; ++i) {
        CHECK_EQ(r.input_token_ids[static_cast<std::size_t>(i)],
                 static_cast<int>(exp_in[static_cast<std::size_t>(i)]));
      }
      // Continuation (NOT a restart): after turn 1 the fake's per-
      // sequence step counter is at n1 + 2, so turn 2's input is
      // forwarded at attempts n1+2 .. n1+1+n2 and its picks continue
      // from THERE (g0 = pick(n1+1+n2), g1 = pick(n1+2+n2)). A REPLAY
      // of turn 1 would restart at attempt 0 and produce pick(n2-1) —
      // a different token.
      const std::vector<int> hist = fwd.tokens_of(seq);
      CHECK_EQ(static_cast<int>(hist.size()), n1 + 2 + n2 + 2);
      std::vector<int> tail(hist.end() - (n2 + 2), hist.end());
      std::vector<int> exp_tail;
      for (int i = 0; i < n2; ++i) {
        exp_tail.push_back(static_cast<int>(
            exp_in[static_cast<std::size_t>(i)]));
      }
      exp_tail.push_back(fwd.pick_token(n1 + 1 + n2));
      exp_tail.push_back(fwd.pick_token(n1 + 2 + n2));
      CHECK(tail == exp_tail);
      CHECK(r.generated_token_ids ==
            (std::vector<int>{fwd.pick_token(n1 + 1 + n2),
                              fwd.pick_token(n1 + 2 + n2)}));
      CHECK_EQ(r.context_length, n1 + 2 + n2 + 2);  // continues from n1+2
      std::printf("  [ok] turn 2: only the new text encoded, continues from "
                  "the committed length (no replay)\n");
    }

    // ===================================================================
    // 3. RESET: state back to zero under the SAME SessionId; next turn
    //    works from length 0
    // ===================================================================
    {
      CHECK(gen.reset_session(sid).ok);
      CHECK_EQ(mgr.lookup(seq)->length, 0);
      std::vector<std::uint32_t> exp_in;
      CHECK(tok->encode("Hello again", &exp_in).ok);
      const int n3 = static_cast<int>(exp_in.size());
      SessionTextTurnResult r = gen.generate_turn(sid, "Hello again", 1,
                                                  kGreedy, -1);
      CHECK(r.ok);
      CHECK_EQ(r.context_length, n3 + 1);  // fresh from 0 (no turn-1/2
                                           // residue in the state)
      CHECK(r.stop_reason == FinishReason::MaxNewTokens);
      CHECK_EQ(sm.lookup(sid)->reset_count, 1);
      std::printf("  [ok] reset: length 0 under the same session, next turn "
                  "works fresh\n");
    }

    // ===================================================================
    // 4. ERRORS PRODUCE NO HALF-TURN (zero mutation, session live)
    // ===================================================================
    {
      const int reqs = gen.scheduler().num_requests();
      const RequestId next_id = gen.scheduler().next_request_id();
      const int len = mgr.lookup(seq)->length;
      const int pages = mgr.kv_pool().used_pages();
      const int slots = mgr.delta_pool().used_slots();
      auto check_zero = [&](const char* name) {
        if (gen.scheduler().num_requests() != reqs ||
            gen.scheduler().next_request_id() != next_id ||
            mgr.lookup(seq)->length != len ||
            mgr.kv_pool().used_pages() != pages ||
            mgr.delta_pool().used_slots() != slots ||
            sm.lookup(sid) == nullptr) {
          std::fprintf(stderr, "  [FAIL] %s: mutation detected\n", name);
          return 1;
        }
        std::printf("  [ok] %s: zero mutation, session live\n", name);
        return 0;
      };

      // (a) invalid UTF-8: encode fails BEFORE any admission.
      SessionTextTurnResult r = gen.generate_turn(sid, "\xff\xfe\xfd", 2,
                                                  kGreedy, -1);
      CHECK(!r.ok);
      CHECK_EQ(r.request_id, static_cast<RequestId>(0));
      if (check_zero("invalid UTF-8")) return 1;

      // (b) empty text: encodes to 0 ids -> admission rejects.
      r = gen.generate_turn(sid, "", 2, kGreedy, -1);
      CHECK(!r.ok);
      CHECK_EQ(r.request_id, static_cast<RequestId>(0));
      if (check_zero("empty text")) return 1;

      // (c) unknown session.
      r = gen.generate_turn(999, "hello", 2, kGreedy, -1);
      CHECK(!r.ok);
      CHECK_EQ(r.request_id, static_cast<RequestId>(0));
      if (check_zero("unknown session")) return 1;

      // (d) CONTEXT OVERFLOW: fill the session near max_seq_len, then a
      // turn whose input + max_new does not fit.
      SessionId sid2 = 0;
      CHECK(gen.create_session(&sid2).ok);
      const SequenceId seq2 = sm.lookup(sid2)->sequence_id;
      std::vector<std::uint32_t> in4;
      CHECK(tok->encode("overflow probe", &in4).ok);
      const int n4 = static_cast<int>(in4.size());
      const int max_seq = cfg.max_seq_len;  // 64
      CHECK(mgr.set_length(seq2, max_seq - n4).ok);  // L + n4 + 1 > max_seq
      const int len2 = mgr.lookup(seq2)->length;
      const int pages2 = mgr.kv_pool().used_pages();
      const int slots2 = mgr.delta_pool().used_slots();
      r = gen.generate_turn(sid2, "overflow probe", 1, kGreedy, -1);
      CHECK(!r.ok);
      CHECK(r.error.find("context overflow") != std::string::npos);
      CHECK_EQ(r.request_id, static_cast<RequestId>(0));
      CHECK_EQ(mgr.lookup(seq2)->length, len2);
      CHECK_EQ(mgr.kv_pool().used_pages(), pages2);
      CHECK_EQ(mgr.delta_pool().used_slots(), slots2);
      CHECK_EQ(gen.scheduler().num_requests(), reqs);
      CHECK_EQ(gen.scheduler().next_request_id(), next_id);
      std::printf("  [ok] context overflow: rejected, zero mutation\n");
      CHECK(gen.destroy_session(sid2).ok);
      // After (a)-(d): the admission-failure cases changed NOTHING
      // (sid2's set_length allocates no pages; it is destroyed again).
      if (check_zero("after all admission-failure cases")) return 1;

      // (e) a FAILED FORWARD: no half-turn — the session stays at its
      //     committed boundary (a new session; the fault is injected at
      //     ATTEMPT INDEX n5 = g0's commit forward — deterministic, no
      //     input-token coincidence). (e) legitimately REGISTERS its two
      //     turn requests, so it snapshots its own expectations.
      const int pages_e = mgr.kv_pool().used_pages();
      const int slots_e = mgr.delta_pool().used_slots();
      const int len_e = mgr.lookup(seq)->length;
      SessionId sid3 = 0;
      CHECK(gen.create_session(&sid3).ok);
      std::vector<std::uint32_t> in5;
      CHECK(tok->encode("fault probe", &in5).ok);
      const int n5 = static_cast<int>(in5.size());
      fwd.fail_attempt = n5;  // g0's commit will fail
      r = gen.generate_turn(sid3, "fault probe", 2, kGreedy, -1);
      fwd.fail_attempt = -1;
      CHECK(!r.ok);
      CHECK(r.stop_reason == FinishReason::Failed);
      CHECK_EQ(r.context_length, n5);  // only the input is committed
      CHECK(r.generated_token_ids.empty());  // no pending token reported
      CHECK(sm.lookup(sid3) != nullptr);  // the session stays live
      // The session can take a next turn from the committed boundary:
      std::vector<std::uint32_t> in6;
      CHECK(tok->encode("next", &in6).ok);
      const int n6 = static_cast<int>(in6.size());
      SessionTextTurnResult r2 = gen.generate_turn(sid3, "next", 1, kGreedy,
                                                   -1);
      CHECK(r2.ok);
      CHECK_EQ(r2.context_length, n5 + n6 + 1);
      std::printf("  [ok] forward failure: no half-turn, committed boundary "
                  "kept, session live\n");
      CHECK(gen.destroy_session(sid3).ok);
      // After (e)'s teardown: the pool is back to (e)'s snapshot and the
      // main session is exactly as before (sid3 is gone).
      CHECK_EQ(mgr.kv_pool().used_pages(), pages_e);
      CHECK_EQ(mgr.delta_pool().used_slots(), slots_e);
      CHECK_EQ(mgr.lookup(seq)->length, len_e);
      CHECK(sm.lookup(sid) != nullptr);
    }

    CHECK(gen.destroy_session(sid).ok);
    CHECK_EQ(mgr.kv_pool().used_pages(), 0);
    CHECK_EQ(mgr.delta_pool().used_slots(), 0);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("test_qwen35_session_text: PASS\n");
  return 0;
}
