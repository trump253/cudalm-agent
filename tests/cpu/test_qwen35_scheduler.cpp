// CUDALM — v0.6 Phase A: request scheduler / control-plane gate (CPU test,
// NO checkpoint, NO model — deterministic fake forwarder over the REAL
// Qwen35StateManager pools).
//
// Proves the scheduler's CONTROL-PLANE SEMANTICS in isolation from GPU
// numerics (the real-checkpoint parity is test_qwen35_scheduler_
// integration): request lifecycle, monotonic RequestId, transactional
// admission, deterministic FIFO / round-robin iteration (snapshot
// semantics), prefill/decode coexistence, EOS / max_new_tokens completion,
// cancel + retire, capacity / re-admission, fatal-Status handling, stale-
// request safety, and per-request sampling RNG isolation — all against
// the real v0.5 state pools (create / retire / capacity are the real
// ones; only the forward + logits are faked deterministically).

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

using namespace cudalm;

namespace {

// Small synthetic hybrid config (valid(); the same shape as
// tests/cuda/test_qwen35_state_manager.cpp's small_config): 8 layers,
// interval 4 -> 2 full (layers 3,7) + 6 linear; max_seq_len 64.
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

// Deterministic CPU forwarder: the logits of a forward depend ONLY on
// THAT SEQUENCE'S OWN forward index (its step) — never on the sequence
// id, the wall clock, or the other sequences (a real model's logits
// depend only on the forwarded tokens). So a sequence's token stream is
// a pure function of (its prompt, its config, its seed) regardless of
// interleaving or of which physical SequenceId it got.
//
//   logits(t) = 10.0          if t == pick(step)
//              = 5.0 - 0.5*t  if t < 8  (a small ramp: gives
//                                        top-k sampling a spread)
//              = -1.0         otherwise
//   pick(step) = (3*step + 1) % vocab
//
// Greedy mode therefore emits the pick token at every sample event;
// seeded sampling mode (temperature > 0, top_k = 3) draws over the pick +
// tokens {0, 1} with a fixed distribution. Fault injection: forward_token
// fails when the forwarded token == fail_token (default: no fault).
class FakeForwarder : public SequenceForwarder {
 public:
  int vocab = 1024;  // must match small_config().vocab_size
  int fail_token = -1;
  std::vector<std::pair<SequenceId, int>> log;  // (sid, token) call order

  Status forward_token(int token_id, SequenceId sequence_id,
                       Qwen35StateManager& /*mgr*/,
                       cudaStream_t /*stream*/) override {
    log.push_back({sequence_id, token_id});
    last_step_ = count_[sequence_id]++;
    if (token_id == fail_token) {
      return Status::error("fake forward fault (injected)");
    }
    return Status::ok_status();
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

 private:
  mutable int last_step_ = -1;
  mutable std::map<SequenceId, int> count_;
};

// The greedy stream a sequence produces under the fake forwarder: the
// pick token at each SAMPLE EVENT (the last prompt forward + each decode
// forward). n = prompt length, m = number of generated tokens (m sample
// events: steps n-1, n, ..., n+m-2).
std::vector<int> expected_greedy(FakeForwarder* fwd, int n, int m) {
  std::vector<int> out;
  for (int k = 0; k < m; ++k) {
    out.push_back(fwd->pick_token(n - 1 + k));
  }
  return out;
}

// Expected total forwards for a completed request (v0.4 semantics):
// N prefill forwards + (m - 1) decode forwards.
int expected_forwards(int n, int m) { return n + m - 1; }

}  // namespace

// ---- 1. request lifecycle -------------------------------------------------
int test_lifecycle() {
  std::fprintf(stderr, "[lifecycle]\n");
  Qwen35Config cfg = small_config();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  FakeForwarder fwd;
  {
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0;
    Scheduler::Spec spec;
    spec.prompt = {10, 11, 12};
    spec.max_new_tokens = 2;
    spec.sampling = SamplingConfig::greedy();
    Status s = sched.admit(spec, &a);
    CHECK(s.ok);
    CHECK_EQ(a, 1u);  // ids start at 1
    const Request* ra = sched.get(a);
    CHECK(ra != nullptr);
    CHECK(ra->status == RequestStatus::Waiting);
    CHECK_EQ(ra->sequence_id, 1u);  // manager's first sequence
    CHECK_EQ(ra->forward_count, 0);
    CHECK_EQ(mgr.num_live_sequences(), 1);
    CHECK_EQ(sched.num_requests(), 1);
    CHECK_EQ(sched.num_live(), 1);
    CHECK(sched.get(99) == nullptr);

    // Run to completion: prompt 3 + max_new 2 -> 3 + 1 = 4 forwards,
    // generated = [pick(1,2), pick(1,3)] = [14, 17].
    s = sched.run();
    CHECK(s.ok);
    ra = sched.get(a);
    CHECK(ra != nullptr);
    CHECK(ra->status == RequestStatus::Finished);
    CHECK(ra->finish_reason == FinishReason::MaxNewTokens);
    CHECK_EQ(ra->prefill_pos, 3);
    CHECK_EQ(ra->forward_count, expected_forwards(3, 2));
    CHECK_EQ(static_cast<int>(ra->generated.size()), 2);
    std::vector<int> exp = expected_greedy(&fwd, 3, 2);
    CHECK(ra->generated == exp);
    CHECK_EQ(static_cast<int>(exp.size()), 2);
    CHECK_EQ(exp[0], 7);   // pick(2)
    CHECK_EQ(exp[1], 10);  // pick(3)
    // The sequence was retired exactly once: no live sequences remain.
    CHECK_EQ(mgr.num_live_sequences(), 0);
    CHECK_EQ(sched.num_live(), 0);
    CHECK_EQ(sched.num_requests(), 1);  // terminal records stay inspectable
    // Stale request never runs again: extra steps are no-ops.
    s = sched.step();
    CHECK(s.ok);
    CHECK(sched.step().ok);
    ra = sched.get(a);
    CHECK_EQ(ra->forward_count, expected_forwards(3, 2));
    CHECK_EQ(mgr.num_live_sequences(), 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

// ---- 2. monotonic, never-reused RequestId ----------------------------------
int test_request_id_monotonic() {
  std::fprintf(stderr, "[request-id-monotonic]\n");
  Qwen35Config cfg = small_config();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  FakeForwarder fwd;
  {
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0, c = 0, d = 0;
    Scheduler::Spec spec;
    spec.prompt = {5};
    spec.max_new_tokens = 1;
    spec.sampling = SamplingConfig::greedy();
    CHECK(sched.admit(spec, &a).ok);
    CHECK(sched.admit(spec, &b).ok);
    CHECK(sched.admit(spec, &c).ok);
    CHECK_EQ(a, 1u);
    CHECK_EQ(b, 2u);
    CHECK_EQ(c, 3u);
    // Cancel + finish + fail everything; ids must NOT be reused.
    CHECK(sched.cancel(a).ok);
    CHECK(sched.run().ok);  // b, c finish
    spec.prompt = {6};
    CHECK(sched.admit(spec, &d).ok);
    CHECK_EQ(d, 4u);  // NOT 1 (A's cancelled id), NOT 2/3
    CHECK_EQ(sched.next_request_id(), 5u);
    CHECK_EQ(sched.get(a)->status, RequestStatus::Cancelled);
    CHECK_EQ(sched.get(b)->status, RequestStatus::Finished);
    CHECK_EQ(sched.get(c)->status, RequestStatus::Finished);
    CHECK_EQ(sched.get(d)->status, RequestStatus::Waiting);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

// ---- 3. transactional admission (capacity + validation) --------------------
int test_admission_transactional() {
  std::fprintf(stderr, "[admission-transactional]\n");
  Qwen35Config cfg = small_config();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  FakeForwarder fwd;
  {
    // Delta slots = 2: A + B fit, C must fail at admission.
    Qwen35StateManager mgr(cfg, 2, 8, 2, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0, c = 0;
    Scheduler::Spec spec;
    spec.prompt = {10, 11};
    spec.max_new_tokens = 1;
    spec.sampling = SamplingConfig::greedy();
    CHECK(sched.admit(spec, &a).ok);
    CHECK(sched.admit(spec, &b).ok);
    Status s = sched.admit(spec, &c);
    CHECK(!s.ok);
    CHECK_EQ(mgr.num_live_sequences(), 2);  // no leaked sequence
    CHECK_EQ(sched.num_requests(), 2);  // no half-registered request
    CHECK_EQ(sched.next_request_id(), 3u);  // no consumed RequestId
    // Existing A/B unchanged by the failed admission.
    CHECK_EQ(sched.get(a)->status, RequestStatus::Waiting);
    CHECK_EQ(sched.get(b)->status, RequestStatus::Waiting);
    CHECK_EQ(sched.get(a)->forward_count, 0);

    // Re-admission after A finishes (its sequence is reclaimed).
    CHECK(sched.run().ok);  // a, b finish (A first? FIFO: a then b per step)
    CHECK_EQ(sched.get(a)->status, RequestStatus::Finished);
    CHECK_EQ(mgr.num_live_sequences(), 0);
    spec.prompt = {20, 21};
    CHECK(sched.admit(spec, &c).ok);  // C now admitted successfully
    CHECK_EQ(c, 3u);  // the first id ever issued to C
    CHECK_EQ(mgr.num_live_sequences(), 1);
    CHECK(sched.run().ok);
    CHECK_EQ(sched.get(c)->status, RequestStatus::Finished);
    CHECK_EQ(mgr.num_live_sequences(), 0);

    // Cancelled requests free their sequence for re-admission too.
    spec.prompt = {30};
    CHECK(sched.admit(spec, &a).ok);  // id 4 (never reused)
    CHECK(sched.cancel(a).ok);
    spec.prompt = {31};
    RequestId e = 0;
    CHECK(sched.admit(spec, &e).ok);  // reuses A's freed sequence state
    CHECK_EQ(e, 5u);
    CHECK_EQ(mgr.num_live_sequences(), 1);
    CHECK(sched.run().ok);
    CHECK_EQ(mgr.num_live_sequences(), 0);
  }
  {
    // Input-validation failures: nothing registered, nothing consumed.
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId r = 123;
    Scheduler::Spec spec;
    spec.prompt = {10};
    spec.max_new_tokens = 1;
    spec.sampling = SamplingConfig::greedy();
    spec.prompt.clear();
    CHECK(!sched.admit(spec, &r).ok);
    spec.prompt = {10};
    spec.max_new_tokens = 0;
    CHECK(!sched.admit(spec, &r).ok);
    spec.max_new_tokens = 1;
    spec.sampling.top_k = -1;
    CHECK(!sched.admit(spec, &r).ok);
    spec.sampling = SamplingConfig::greedy();
    spec.eos_token_id = fwd.vocab;
    CHECK(!sched.admit(spec, &r).ok);
    spec.eos_token_id = -1;
    spec.prompt = {fwd.vocab};
    CHECK(!sched.admit(spec, &r).ok);
    spec.prompt = {10};
    spec.max_new_tokens = small_config().max_seq_len + 1;  // 1 + 65 - 1 > 64
    CHECK(!sched.admit(spec, &r).ok);
    CHECK_EQ(mgr.num_live_sequences(), 0);
    CHECK_EQ(sched.num_requests(), 0);
    CHECK_EQ(sched.next_request_id(), 1u);  // zero ids consumed
    r = 0;
    spec.max_new_tokens = 1;
    CHECK(sched.admit(spec, &r).ok);
    CHECK_EQ(r, 1u);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

// ---- 4. FIFO / round-robin iteration + snapshot semantics ------------------
int test_fifo_iteration() {
  std::fprintf(stderr, "[fifo-iteration]\n");
  Qwen35Config cfg = small_config();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  FakeForwarder fwd;
  {
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0, c = 0;
    Scheduler::Spec sa;
    sa.prompt = {100, 101, 102};  // A: 3 prefill tokens
    sa.max_new_tokens = 2;
    sa.sampling = SamplingConfig::greedy();
    Scheduler::Spec sb;
    sb.prompt = {200, 201};  // B: 2 prefill tokens
    sb.max_new_tokens = 2;
    sb.sampling = SamplingConfig::greedy();
    CHECK(sched.admit(sa, &a).ok);
    CHECK(sched.admit(sb, &b).ok);
    CHECK_EQ(a, 1u);
    CHECK_EQ(b, 2u);

    // Step 1: A p0, B p0 — admission (id) order.
    fwd.log.clear();
    CHECK(sched.step().ok);
    CHECK_EQ(static_cast<int>(fwd.log.size()), 2);
    CHECK_EQ(fwd.log[0].first, 1u);
    CHECK_EQ(fwd.log[0].second, 100);
    CHECK_EQ(fwd.log[1].first, 2u);
    CHECK_EQ(fwd.log[1].second, 200);

    // Step 2: A p1, B p1 (B's prefill completes -> B samples g0).
    fwd.log.clear();
    CHECK(sched.step().ok);
    CHECK_EQ(fwd.log[0].first, 1u);
    CHECK_EQ(fwd.log[0].second, 101);
    CHECK_EQ(fwd.log[1].first, 2u);
    CHECK_EQ(fwd.log[1].second, 201);
    CHECK_EQ(static_cast<int>(sched.get(b)->generated.size()), 1);
    CHECK_EQ(sched.get(b)->generated[0], fwd.pick_token(1));

    // C admitted BETWEEN step 2 and step 3 -> it first participates in
    // step 3 (never in step 2); order A, B, C (ascending id).
    Scheduler::Spec sc;
    sc.prompt = {300, 301, 302, 303};
    sc.max_new_tokens = 1;
    sc.sampling = SamplingConfig::greedy();
    CHECK(sched.admit(sc, &c).ok);
    CHECK_EQ(c, 3u);
    fwd.log.clear();
    CHECK(sched.step().ok);
    CHECK_EQ(static_cast<int>(fwd.log.size()), 3);
    CHECK_EQ(fwd.log[0].first, 1u);
    CHECK_EQ(fwd.log[0].second, 102);  // A prefill done -> samples g0
    CHECK_EQ(fwd.log[1].first, 2u);
    CHECK_EQ(fwd.log[1].second, fwd.pick_token(1));  // B decodes its g0
    CHECK_EQ(fwd.log[2].first, 3u);
    CHECK_EQ(fwd.log[2].second, 300);  // C's first token
    CHECK_EQ(sched.get(a)->generated[0], fwd.pick_token(2));
    CHECK_EQ(sched.get(b)->status, RequestStatus::Finished);  // max_new 2
    CHECK_EQ(sched.get(b)->finish_reason, FinishReason::MaxNewTokens);
    CHECK_EQ(sched.get(b)->forward_count, expected_forwards(2, 2));

    // B finished: it no longer advances; A (decoding) + C (prefilling) do.
    fwd.log.clear();
    CHECK(sched.step().ok);
    CHECK_EQ(static_cast<int>(fwd.log.size()), 2);
    CHECK_EQ(fwd.log[0].first, 1u);  // A decodes g0
    CHECK_EQ(fwd.log[1].first, 3u);  // C p1
    CHECK_EQ(sched.get(a)->status, RequestStatus::Finished);
    CHECK(sched.run().ok);
    CHECK_EQ(sched.num_live(), 0);
    CHECK_EQ(sched.get(c)->forward_count, expected_forwards(4, 1));
    CHECK_EQ(mgr.num_live_sequences(), 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

// ---- 5. prefill / decode coexistence ---------------------------------------
int test_prefill_decode_coexist() {
  std::fprintf(stderr, "[prefill-decode-coexist]\n");
  Qwen35Config cfg = small_config();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  FakeForwarder fwd;
  {
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0;
    Scheduler::Spec sa;
    sa.prompt = {10, 11};  // A: short prompt
    sa.max_new_tokens = 2;
    sa.sampling = SamplingConfig::greedy();
    Scheduler::Spec sb;
    sb.prompt = {20, 21, 22, 23, 24};  // B: long prompt
    sb.max_new_tokens = 1;
    sb.sampling = SamplingConfig::greedy();
    CHECK(sched.admit(sa, &a).ok);
    CHECK(sched.admit(sb, &b).ok);

    // step 1: A p0, B p0 (both prefill)
    CHECK(sched.step().ok);
    // step 2: A p1 (prefill done -> samples g0), B p1 (still prefill)
    CHECK(sched.step().ok);
    // step 3: A decodes g0 (samples g1 -> A Finished), B p2 (prefill)
    CHECK(sched.step().ok);
    CHECK_EQ(sched.get(a)->status, RequestStatus::Finished);
    CHECK_EQ(sched.get(a)->finish_reason, FinishReason::MaxNewTokens);
    CHECK_EQ(sched.get(b)->status, RequestStatus::Running);
    CHECK_EQ(sched.get(b)->prefill_pos, 3);  // still mid-prefill
    // step 4: B p3 — A (terminal) must not be touched
    fwd.log.clear();
    CHECK(sched.step().ok);
    CHECK_EQ(static_cast<int>(fwd.log.size()), 1);
    CHECK_EQ(fwd.log[0].first, 2u);
    CHECK_EQ(fwd.log[0].second, 23);
    CHECK_EQ(sched.get(b)->prefill_pos, 4);
    // step 5: B p4 (prefill done -> samples g0 -> B Finished)
    CHECK(sched.step().ok);
    CHECK_EQ(sched.get(b)->status, RequestStatus::Finished);
    CHECK_EQ(sched.get(b)->finish_reason, FinishReason::MaxNewTokens);
    CHECK_EQ(sched.get(b)->forward_count, expected_forwards(5, 1));
    std::vector<int> exp_b = expected_greedy(&fwd, 5, 1);
    CHECK(sched.get(b)->generated == exp_b);
    CHECK(sched.run().ok);  // nothing left
    CHECK_EQ(mgr.num_live_sequences(), 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

// ---- 6. EOS completion ------------------------------------------------------
int test_eos_completion() {
  std::fprintf(stderr, "[eos-completion]\n");
  Qwen35Config cfg = small_config();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  FakeForwarder fwd;
  {
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0;
    // A: prompt 2, max_new 5, but EOS at its FIRST sample event
    // (step index 1 -> pick(1, 1) = 11).
    Scheduler::Spec sa;
    sa.prompt = {40, 41};
    sa.max_new_tokens = 5;
    sa.eos_token_id = fwd.pick_token(1);  // == 4
    sa.sampling = SamplingConfig::greedy();
    // B: no EOS gate, same prompt shape -> completes by max_new_tokens.
    Scheduler::Spec sb;
    sb.prompt = {40, 41};
    sb.max_new_tokens = 2;
    sb.eos_token_id = -1;
    sb.sampling = SamplingConfig::greedy();
    CHECK(sched.admit(sa, &a).ok);
    CHECK(sched.admit(sb, &b).ok);

    CHECK(sched.step().ok);  // A p0, B p0
    CHECK(sched.step().ok);  // A p1 -> samples 11 == EOS; B p1 -> samples
    CHECK_EQ(sched.get(a)->status, RequestStatus::Finished);
    CHECK_EQ(sched.get(a)->finish_reason, FinishReason::Eos);
    CHECK_EQ(static_cast<int>(sched.get(a)->generated.size()), 1);
    CHECK_EQ(sched.get(a)->generated[0], 4);  // the EOS token IS included
    CHECK_EQ(sched.get(a)->forward_count, expected_forwards(2, 1));
    CHECK_EQ(sched.get(a)->prefill_pos, 2);
    CHECK_EQ(mgr.num_live_sequences(), 1);  // A retired, B live
    // B is unaffected by A's EOS retirement.
    CHECK(sched.run().ok);
    CHECK_EQ(sched.get(b)->status, RequestStatus::Finished);
    CHECK_EQ(sched.get(b)->finish_reason, FinishReason::MaxNewTokens);
    std::vector<int> exp_b = expected_greedy(&fwd, 2, 2);
    CHECK(sched.get(b)->generated == exp_b);
    CHECK_EQ(mgr.num_live_sequences(), 0);
    // Stale (terminal) A is never advanced again.
    const int a_fc = sched.get(a)->forward_count;
    fwd.log.clear();
    CHECK(sched.step().ok);
    CHECK_EQ(static_cast<int>(fwd.log.size()), 0);
    CHECK_EQ(sched.get(a)->forward_count, a_fc);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

// ---- 7. cancel + retire -----------------------------------------------------
int test_cancel_retire() {
  std::fprintf(stderr, "[cancel-retire]\n");
  Qwen35Config cfg = small_config();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  FakeForwarder fwd;
  {
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0;
    Scheduler::Spec spec;
    spec.prompt = {10, 11};
    spec.max_new_tokens = 4;
    spec.sampling = SamplingConfig::greedy();
    CHECK(sched.admit(spec, &a).ok);
    CHECK(sched.admit(spec, &b).ok);
    CHECK_EQ(mgr.num_live_sequences(), 2);

    // Cancel A while Waiting (no forwards yet).
    CHECK(sched.cancel(a).ok);
    CHECK_EQ(sched.get(a)->status, RequestStatus::Cancelled);
    CHECK_EQ(sched.get(a)->finish_reason, FinishReason::Cancelled);
    CHECK_EQ(sched.get(a)->forward_count, 0);
    CHECK_EQ(mgr.num_live_sequences(), 1);  // A's sequence reclaimed
    // A is never advanced again (stale), B proceeds.
    fwd.log.clear();
    CHECK(sched.step().ok);
    CHECK_EQ(static_cast<int>(fwd.log.size()), 1);
    CHECK_EQ(fwd.log[0].first, 2u);
    // Repeat cancel: IDEMPOTENT ok (pinned contract).
    CHECK(sched.cancel(a).ok);
    CHECK_EQ(sched.get(a)->status, RequestStatus::Cancelled);
    CHECK_EQ(mgr.num_live_sequences(), 1);
    // Unknown id: fail loud.
    CHECK(!sched.cancel(999).ok);
    // Cancel B mid-run (Running): also terminal + retired.
    CHECK(sched.cancel(b).ok);
    CHECK_EQ(sched.get(b)->status, RequestStatus::Cancelled);
    CHECK_EQ(mgr.num_live_sequences(), 0);
    CHECK(sched.run().ok);  // nothing left to do
    CHECK_EQ(mgr.num_live_sequences(), 0);
    // A request cancelled after some forwards keeps its partial record.
    RequestId c = 0;
    spec.max_new_tokens = 4;
    CHECK(sched.admit(spec, &c).ok);
    CHECK(sched.step().ok);  // c: p0
    CHECK(sched.cancel(c).ok);
    CHECK_EQ(sched.get(c)->status, RequestStatus::Cancelled);
    CHECK_EQ(sched.get(c)->forward_count, 1);
    CHECK_EQ(mgr.num_live_sequences(), 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

// ---- 8. sampling RNG isolation (per-request Sampler) ------------------------
int test_sampling_isolation() {
  std::fprintf(stderr, "[sampling-isolation]\n");
  Qwen35Config cfg = small_config();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  // temperature 1.0 + top_k 3: a real (non-greedy) sampling path over the
  // fake's fixed 3-way spread; seeded per request.
  SamplingConfig s42 = SamplingConfig{1.0f, 3, 1.0f, 42u};
  SamplingConfig s123 = SamplingConfig{1.0f, 3, 1.0f, 123u};
  std::vector<int> prompt_a = {10, 11, 12};
  std::vector<int> prompt_b = {20, 21};

  // Interleaved run: A (seed 42) + B (seed 123), one scheduler.
  std::vector<int> ai_ids, bi_ids;
  {
    FakeForwarder fwd;
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0;
    Scheduler::Spec sa;
    sa.prompt = prompt_a;
    sa.max_new_tokens = 3;
    sa.sampling = s42;
    Scheduler::Spec sb;
    sb.prompt = prompt_b;
    sb.max_new_tokens = 2;
    sb.sampling = s123;
    CHECK(sched.admit(sa, &a).ok);
    CHECK(sched.admit(sb, &b).ok);
    CHECK(sched.run().ok);
    ai_ids = sched.get(a)->generated;
    bi_ids = sched.get(b)->generated;
    CHECK_EQ(static_cast<int>(ai_ids.size()), 3);
    CHECK_EQ(static_cast<int>(bi_ids.size()), 2);
  }
  // Standalone runs (fresh scheduler + manager, one request each) — the
  // interleaving must not change ANY request's token stream.
  {
    FakeForwarder fwd;
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0;
    Scheduler::Spec sa;
    sa.prompt = prompt_a;
    sa.max_new_tokens = 3;
    sa.sampling = s42;
    CHECK(sched.admit(sa, &a).ok);
    CHECK(sched.run().ok);
    CHECK(sched.get(a)->generated == ai_ids);  // A interleaved == A alone
  }
  {
    FakeForwarder fwd;
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId b = 0;
    Scheduler::Spec sb;
    sb.prompt = prompt_b;
    sb.max_new_tokens = 2;
    sb.sampling = s123;
    CHECK(sched.admit(sb, &b).ok);
    CHECK(sched.run().ok);
    CHECK(sched.get(b)->generated == bi_ids);  // B interleaved == B alone
  }
  // Same prompt + same seed => same stream, regardless of how many other
  // same-seed requests share the scheduler (admission order must not
  // change the RNG).
  std::vector<int> x_ids, y_ids, xy_x, xy_y;
  {
    FakeForwarder fwd;
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId x = 0;
    Scheduler::Spec sx;
    sx.prompt = prompt_a;
    sx.max_new_tokens = 3;
    sx.sampling = s42;
    CHECK(sched.admit(sx, &x).ok);
    CHECK(sched.run().ok);
    x_ids = sched.get(x)->generated;
  }
  {
    FakeForwarder fwd;
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId y = 0;
    Scheduler::Spec sy;
    sy.prompt = prompt_a;
    sy.max_new_tokens = 3;
    sy.sampling = s42;
    CHECK(sched.admit(sy, &y).ok);
    CHECK(sched.run().ok);
    y_ids = sched.get(y)->generated;
  }
  {
    FakeForwarder fwd;
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId x = 0, y = 0;
    Scheduler::Spec sx;
    sx.prompt = prompt_a;
    sx.max_new_tokens = 3;
    sx.sampling = s42;
    Scheduler::Spec sy;
    sy.prompt = prompt_a;
    sy.max_new_tokens = 3;
    sy.sampling = s42;
    CHECK(sched.admit(sx, &x).ok);
    CHECK(sched.admit(sy, &y).ok);
    CHECK(sched.run().ok);
    xy_x = sched.get(x)->generated;
    xy_y = sched.get(y)->generated;
  }
  CHECK(x_ids == y_ids);       // same prompt+seed => same stream
  CHECK(xy_x == x_ids);        // interleaving with a same-seed twin: X unchanged
  CHECK(xy_y == x_ids);        // ... and the twin gets the same stream too
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

// ---- 9. fatal Status -> Failed + retire, iteration continues ---------------
// ---- 9. fatal Status: failed-prefill progress + run() isolation ---------
int test_fatal_status() {
  std::fprintf(stderr, "[fatal-status]\n");
  Qwen35Config cfg = small_config();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // (a) HARD GATE: run() failure isolation — a request that fatals during
  // run() must NOT stop the other live requests: A -> Failed + retired,
  // B keeps advancing across later iterations until it finishes, and
  // run() returns the FIRST error. The failed forward is NOT committed
  // to the progress: forward_count and prefill_pos exclude it (prompt
  // {50, 99} with 50 ok / 99 failing -> forward_count = 1, prefill_pos
  // = 1, NOT 2), and the sequence is retired exactly once.
  {
    FakeForwarder fwd;
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0;
    Scheduler::Spec sa;
    sa.prompt = {50, 99};  // A: the 2nd prompt token faults inside run()
    sa.max_new_tokens = 2;
    sa.sampling = SamplingConfig::greedy();
    Scheduler::Spec sb;
    sb.prompt = {60, 61};
    sb.max_new_tokens = 2;
    sb.sampling = SamplingConfig::greedy();
    CHECK(sched.admit(sa, &a).ok);
    CHECK(sched.admit(sb, &b).ok);
    fwd.fail_token = 99;
    Status s = sched.run();
    CHECK(!s.ok);  // the first error is reported...
    CHECK_EQ(sched.get(a)->status, RequestStatus::Failed);
    CHECK_EQ(sched.get(a)->finish_reason, FinishReason::Failed);
    CHECK_EQ(sched.get(a)->forward_count, 1);  // ... but the failed
                                               // forward is NOT counted
    CHECK_EQ(sched.get(a)->prefill_pos, 1);    // ... nor committed to the
                                               // progress (NOT 2)
    // ... while B was never stopped: it ran to completion afterwards.
    CHECK_EQ(sched.get(b)->status, RequestStatus::Finished);
    CHECK_EQ(sched.get(b)->finish_reason, FinishReason::MaxNewTokens);
    CHECK_EQ(sched.get(b)->forward_count, expected_forwards(2, 2));
    CHECK_EQ(sched.num_live(), 0);
    CHECK_EQ(mgr.num_live_sequences(), 0);  // both sequences retired exactly
                                            // once (A on failure, B on finish)
    // Stale (failed) request is never advanced again; cancel is the
    // idempotent no-op (already terminal).
    CHECK(sched.cancel(a).ok);
    CHECK_EQ(sched.get(a)->status, RequestStatus::Failed);
    CHECK_EQ(sched.get(a)->forward_count, 1);
    CHECK_EQ(sched.get(a)->prefill_pos, 1);
  }

  // (b) step-level pins: the SAME failure observed one step at a time —
  // the failed forward is not committed (prefill_pos stays 1), and the
  // remaining snapshot ids of the FAILING step still advance (B does its
  // p1 in the very step where A's p1 faults).
  {
    FakeForwarder fwd;
    Qwen35StateManager mgr(cfg, 2, 8, 4, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0;
    Scheduler::Spec sa;
    sa.prompt = {50, 99};
    sa.max_new_tokens = 2;
    sa.sampling = SamplingConfig::greedy();
    Scheduler::Spec sb;
    sb.prompt = {60, 61};
    sb.max_new_tokens = 2;
    sb.sampling = SamplingConfig::greedy();
    CHECK(sched.admit(sa, &a).ok);
    CHECK(sched.admit(sb, &b).ok);
    fwd.fail_token = 99;
    // Step 1: A p0 (50) ok, B p0 ok — both commit progress.
    CHECK(sched.step().ok);
    CHECK_EQ(sched.get(a)->forward_count, 1);
    CHECK_EQ(sched.get(a)->prefill_pos, 1);
    CHECK_EQ(sched.get(b)->forward_count, 1);
    CHECK_EQ(sched.get(b)->prefill_pos, 1);
    // Step 2: A p1 (99) FAILS -> A Failed + retired, prefill_pos stays 1
    // (the failed token is NOT committed); B p1 STILL advances in this
    // same step (the remaining snapshot ids continue); step() reports
    // the first error.
    Status s = sched.step();
    CHECK(!s.ok);
    CHECK_EQ(sched.get(a)->status, RequestStatus::Failed);
    CHECK_EQ(sched.get(a)->finish_reason, FinishReason::Failed);
    CHECK_EQ(sched.get(a)->forward_count, 1);  // failed forward not counted
    CHECK_EQ(sched.get(a)->prefill_pos, 1);    // failed token not committed
    CHECK_EQ(sched.get(b)->forward_count, 2);  // B unaffected by A's failure
    CHECK_EQ(sched.get(b)->prefill_pos, 2);
    CHECK_EQ(mgr.num_live_sequences(), 1);  // A retired, B live
    // Step 3: only B advances (A is terminal); B decodes g0 and finishes.
    fwd.fail_token = -1;
    CHECK(sched.step().ok);
    CHECK_EQ(sched.get(b)->status, RequestStatus::Finished);
    CHECK_EQ(sched.get(b)->forward_count, expected_forwards(2, 2));
    CHECK_EQ(sched.get(a)->status, RequestStatus::Failed);  // stale: untouched
    CHECK_EQ(sched.get(a)->forward_count, 1);
    CHECK_EQ(sched.get(a)->prefill_pos, 1);
    CHECK_EQ(mgr.num_live_sequences(), 0);
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

int main() {
  int rc = 0;
  rc |= test_lifecycle();
  rc |= test_request_id_monotonic();
  rc |= test_admission_transactional();
  rc |= test_fifo_iteration();
  rc |= test_prefill_decode_coexist();
  rc |= test_eos_completion();
  rc |= test_cancel_retire();
  rc |= test_sampling_isolation();
  rc |= test_fatal_status();
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_scheduler: FAIL\n");
    return rc;
  }
  std::printf(
      "test_qwen35_scheduler: PASS (lifecycle + monotonic RequestId + "
      "transactional admission + FIFO/round-robin snapshot + "
      "prefill/decode coexistence + EOS/max_new_tokens + cancel/retire + "
      "per-request sampling RNG isolation + fatal Status (failed forward "
      "not committed to progress + run() failure isolation))\n");
  return 0;
}
