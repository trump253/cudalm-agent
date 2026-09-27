// CUDALM — v0.6 Phase C: COMBINED CONTROL-PLANE STRESS gate (CPU, NO
// checkpoint, NO model — deterministic fake forwarder over the REAL
// Qwen35StateManager pools).
//
// Two parts:
//
//   Part 1 (failure isolation in a DYNAMIC set): a dynamically-arriving set
//   of requests in which ONE request's forwards always fail. Proves:
//     * the failing request -> Failed + retired EXACTLY ONCE, its failed
//       forward NOT committed (forward_count stays 0), and it is NEVER
//       advanced again;
//     * the OTHER live requests CONTINUE and complete (Finished, correct
//       forward count) — the failure does not stall or corrupt them;
//     * run() returns the FIRST error and its step/run semantics do not
//       degrade (execution continues past the failure);
//     * final resource accounting is exact: num_live == 0 and
//       mgr.num_live_sequences == 0 (no leaked sequence, no leak).
//
//   Part 2 (deterministic lifecycle STRESS, fixed seed): several dozen
//   requests driven by a FIXED-SEED deterministic script (admit / step /
//   late admit / finish / cancel / continue), fully reproducible (no
//   infinite / flaky randomness). After EVERY step AND admit AND cancel,
//   the invariants are checked:
//     * scheduler live count == StateManager live sequence count;
//     * a TERMINAL request is NEVER advanced again (its forward_count is
//       frozen at the moment it goes terminal);
//     * each LIVE request owns EXACTLY ONE live SequenceId (lookup non-null,
//       ids distinct across live requests);
//   and at the END: num_live == 0 and mgr.num_live_sequences == 0.
//
// This gate exercises ONLY the control plane (lifecycle / admission /
// cancellation / failure isolation / accounting) — no GPU numerics, no
// batched path (the fake forwarder is single-only, so every advance is the
// frozen serial path). It does NOT redefine any Phase A/B GPU failure
// semantics; it only stresses the shared control plane that carries them.

#include "../../tests/common/check.h"

#include <algorithm>
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

using namespace cudalm;

namespace {

// Small synthetic hybrid config (valid(); the same shape as
// test_qwen35_scheduler.cpp's small_config): 8 layers, interval 4 -> 2 full
// (layers 3,7) + 6 linear; max_seq_len 64.
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

// Deterministic CPU forwarder (single-only, so the frozen serial path is
// always taken). Logits are CONSTANT (greedy always picks token kPick), so
// every request generates the same deterministic token stream regardless of
// interleaving — the control-plane semantics are what is under stress, not
// numerics. Fault injection: forward_token fails for any sequence whose id
// is in `fail_sids` (every attempt), and succeeds otherwise.
class StressForwarder : public SequenceForwarder {
 public:
  int vocab = 1024;  // must match small_config().vocab_size
  int pick = 7;      // the greedy token (logits[pick] is the unique max)
  std::set<SequenceId> fail_sids;  // sequences whose forwards always fail

  Status forward_token(int /*token_id*/, SequenceId sequence_id,
                       Qwen35StateManager& /*mgr*/,
                       cudaStream_t /*stream*/) override {
    attempts_[sequence_id]++;
    if (fail_sids.count(sequence_id)) {
      return Status::error("injected fault for sid " +
                           std::to_string(sequence_id));
    }
    ok_[sequence_id]++;
    return Status::ok_status();
  }
  Status logits_to_host(std::vector<__nv_bfloat16>* out,
                        cudaStream_t /*stream*/) const override {
    out->assign(static_cast<std::size_t>(vocab), __float2bfloat16(-1.0f));
    (*out)[static_cast<std::size_t>(pick)] = __float2bfloat16(10.0f);
    return Status::ok_status();
  }
  int vocab_size() const override { return vocab; }

  int ok_forwards(SequenceId sid) const {
    auto it = ok_.find(sid);
    return it == ok_.end() ? 0 : it->second;
  }
  int attempts(SequenceId sid) const {
    auto it = attempts_.find(sid);
    return it == attempts_.end() ? 0 : it->second;
  }

 private:
  std::map<SequenceId, int> ok_;
  std::map<SequenceId, int> attempts_;
};

// A SplitMix64 (portable, fixed-seed) for the deterministic workload.
struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  int range(int lo, int hi) {  // [lo, hi)
    return lo + static_cast<int>(next() % static_cast<std::uint64_t>(hi - lo));
  }
};

// N prefill + (m - 1) decode forwards for m generated tokens.
int expected_forwards(int n, int m) { return n + m - 1; }

}  // namespace

int main() {
  Qwen35Config cfg = small_config();
  int rc = 0;

  // =========================================================================
  // Part 1: FAILURE ISOLATION in a DYNAMIC request set (CPU fake forwarder).
  // =========================================================================
  {
    // Pool sized for the peak live set (a handful of concurrent sequences).
    Qwen35StateManager mgr(cfg, /*page_tokens=*/2, /*kv_pages=*/32,
                           /*delta_slots=*/8, /*stream=*/0);
    StressForwarder fwd;
    Scheduler sched(fwd, mgr, 0);

    // Admit a dynamically-arriving set; ONE of them (admitted late) will
    // always fail on its first forward.
    RequestId good1 = 0, good2 = 0, bad = 0, good3 = 0;
    std::vector<RequestId> order;
    auto admit = [&](const std::vector<int>& prompt, int max_new,
                     RequestId* out) -> int {
      Scheduler::Spec sp;
      sp.prompt = prompt;
      sp.max_new_tokens = max_new;
      sp.sampling = SamplingConfig::greedy();
      Status s = sched.admit(sp, out);
      if (!s.ok) {
        std::fprintf(stderr, "admit failed: %s\n", s.message.c_str());
        return 1;
      }
      order.push_back(*out);
      return 0;
    };

    CHECK_EQ(admit({10, 20, 30}, 3, &good1), 0);  // early
    CHECK_EQ(admit({40, 50}, 2, &good2), 0);      // early
    // Make the LATE request the failing one: its sequence id is known only
    // after admission, so admit it, then register its sid as a fault.
    CHECK_EQ(admit({60, 70, 80, 90}, 4, &bad), 0);
    fwd.fail_sids.insert(sched.get(bad)->sequence_id);
    CHECK_EQ(admit({100, 110}, 3, &good3), 0);  // even later

    // RequestIds are monotonic, never reused.
    CHECK(order[1] > order[0]);
    CHECK(order[2] > order[1]);
    CHECK(order[3] > order[2]);

    // Drive to completion via run() (continues past the first error).
    Status r = sched.run();
    CHECK(!r.ok);  // run() reports the FIRST error (the failing request)

    // The failing request: Failed + retired exactly once; its failed forward
    // is NOT committed (forward_count 0); it is never advanced again.
    const Request* rb = sched.get(bad);
    CHECK(rb != nullptr);
    CHECK_EQ(rb->status, RequestStatus::Failed);
    CHECK_EQ(rb->finish_reason, FinishReason::Failed);
    CHECK_EQ(rb->forward_count, 0);  // the failing forward did not commit
    CHECK(mgr.lookup(rb->sequence_id) == nullptr);  // retired exactly once
    CHECK_EQ(fwd.ok_forwards(rb->sequence_id), 0);  // never advanced (ok)

    // The OTHER live requests CONTINUED and completed (correct forward
    // count = N + m - 1; all generated the greedy pick token).
    const Request* r1 = sched.get(good1);
    const Request* r2 = sched.get(good2);
    const Request* r3 = sched.get(good3);
    CHECK(r1 && r2 && r3);
    CHECK_EQ(r1->status, RequestStatus::Finished);
    CHECK_EQ(r2->status, RequestStatus::Finished);
    CHECK_EQ(r3->status, RequestStatus::Finished);
    CHECK_EQ(r1->forward_count, expected_forwards(3, 3));
    CHECK_EQ(r2->forward_count, expected_forwards(2, 2));
    CHECK_EQ(r3->forward_count, expected_forwards(2, 3));
    for (int t : r1->generated) CHECK_EQ(t, fwd.pick);

    // Final resource accounting is exact (no leaked sequence, no leak).
    CHECK_EQ(sched.num_live(), 0);
    CHECK_EQ(mgr.num_live_sequences(), 0);
    // Metrics reflect the isolated failure (observability only).
    const SchedulerStats st = sched.stats();
    CHECK_EQ(st.requests_admitted, 4);
    CHECK_EQ(st.requests_failed, 1);
    CHECK_EQ(st.requests_finished, 3);
    CHECK_EQ(st.requests_live, 0);
    // ---- attempt vs committed FAILURE-PATH gate -------------------------
    // The failing request ISSUED exactly one single attempt (its first
    // forward failed) but COMMITTED nothing: forward_count == 0 (checked
    // above) and the failed attempt must NOT leak into the COMPLETED
    // traversal or the COMMITTED logical-token metrics (it may have failed
    // at model preflight without ever executing the layer traversal).
    //   completed successful singles = good1 5 + good2 3 + good3 4 = 12
    //   issued single attempts       = 12 + 1 (the failed one)   = 13
    CHECK_EQ(st.single_forward_calls, 13);                // ISSUED (incl. failed)
    CHECK_EQ(st.successful_single_forward_calls, 12);     // successful only
    CHECK_EQ(st.logical_token_forwards, 12);              // failed attempt = ZERO
    CHECK_EQ(st.model_traversal_calls, 12);  // failed preflight/fake call is NOT a traversal
    CHECK_EQ(st.batch_forward_calls, 0);
    CHECK_EQ(st.batched_sequence_tokens, 0);
    std::fprintf(stderr,
                 "[stress-part1] failure isolation in a dynamic set: bad "
                 "request Failed+retired once (forward not committed, "
                 "forward_count=0), the other 3 continued to Finished; "
                 "run() reported the first error; final num_live=0, mgr "
                 "live=0; attempt-vs-committed metrics: issued=13 "
                 "successful=12 logical=12 traversals=12 (failed attempt "
                 "is neither a traversal nor committed logical)\n");
  }

  // =========================================================================
  // Part 2: DETERMINISTIC LIFECYCLE STRESS (fixed seed, fully reproducible).
  // =========================================================================
  {
    const std::uint64_t kSeed = 20260209;  // fixed: no flakiness
    Rng rng(kSeed);
    // Peak live is bounded (<= 8) so the delta pool (10 slots) never OOMs.
    Qwen35StateManager mgr(cfg, /*page_tokens=*/2, /*kv_pages=*/64,
                           /*delta_slots=*/10, /*stream=*/0);
    StressForwarder fwd;
    Scheduler sched(fwd, mgr, 0);

    const int kTotalAdmits = 40;  // several dozen
    const int kMaxLive = 8;
    const int kMaxSteps = 5000;   // a generous bound (should be far below)

    int admitted = 0;
    int steps = 0;
    RequestId next_id_seen = 1;
    // Per-request terminal bookkeeping for the "never advanced again" check.
    std::map<RequestId, bool> was_terminal;
    std::map<RequestId, int> terminal_fc;

    auto admit_one = [&]() -> int {
      Scheduler::Spec sp;
      const int plen = rng.range(1, 6);  // 1..5
      sp.prompt.assign(plen, 0);
      for (int i = 0; i < plen; ++i)
        sp.prompt[static_cast<std::size_t>(i)] = rng.range(0, cfg.vocab_size);
      sp.max_new_tokens = rng.range(1, 5);  // 1..4
      sp.sampling = SamplingConfig::greedy();
      RequestId id = 0;
      Status s = sched.admit(sp, &id);
      if (!s.ok) {
        std::fprintf(stderr, "admit failed: %s\n", s.message.c_str());
        return 1;
      }
      if (id != next_id_seen) {
        std::fprintf(stderr, "RequestId not monotonic: %llu != %llu\n",
                     static_cast<unsigned long long>(id),
                     static_cast<unsigned long long>(next_id_seen));
        return 1;
      }
      next_id_seen++;  // monotonic, never reused
      admitted++;
      return 0;
    };

    // The invariant check (run after EVERY step / admit / cancel).
    auto invariants = [&]() -> int {
      // (a) scheduler live count == StateManager live sequence count.
      if (sched.num_live() != mgr.num_live_sequences()) {
        std::fprintf(stderr,
                     "  INVARIANT FAIL: sched.num_live()=%d != "
                     "mgr.num_live_sequences()=%d\n",
                     sched.num_live(), mgr.num_live_sequences());
        return 1;
      }
      std::set<SequenceId> live_seqs;
      for (RequestId id = 1; id < sched.next_request_id(); ++id) {
        const Request* r = sched.get(id);
        if (r == nullptr) continue;
        if (is_terminal(r->status)) {
          if (!was_terminal[static_cast<std::size_t>(id)]) {
            was_terminal[static_cast<std::size_t>(id)] = true;
            terminal_fc[static_cast<std::size_t>(id)] = r->forward_count;
          } else if (r->forward_count !=
                     terminal_fc[static_cast<std::size_t>(id)]) {
            std::fprintf(stderr,
                         "  INVARIANT FAIL: terminal request %llu advanced "
                         "again (fc %d != %d)\n",
                         static_cast<unsigned long long>(id), r->forward_count,
                         terminal_fc[static_cast<std::size_t>(id)]);
            return 1;
          }
        } else {
          // live: owns EXACTLY ONE live SequenceId (lookup non-null).
          if (mgr.lookup(r->sequence_id) == nullptr) {
            std::fprintf(stderr,
                         "  INVARIANT FAIL: live request %llu's sequence %llu "
                         "is not live\n",
                         static_cast<unsigned long long>(id),
                         static_cast<unsigned long long>(r->sequence_id));
            return 1;
          }
          if (!live_seqs.insert(r->sequence_id).second) {
            std::fprintf(stderr,
                         "  INVARIANT FAIL: two live requests share sequence "
                         "%llu\n",
                         static_cast<unsigned long long>(r->sequence_id));
            return 1;
          }
        }
      }
      // the number of distinct live sequences == num_live (implied by (a) +
      // distinctness, but assert it explicitly).
      if (static_cast<int>(live_seqs.size()) != sched.num_live()) {
        std::fprintf(stderr,
                     "  INVARIANT FAIL: distinct live sequences %zu != "
                     "num_live %d\n",
                     live_seqs.size(), sched.num_live());
        return 1;
      }
      return 0;
    };

    rc |= invariants();  // initial (nothing admitted yet)

    // Admit the first wave.
    for (int i = 0; i < 4 && admitted < kTotalAdmits; ++i) {
      if (admit_one()) return 1;
      rc |= invariants();
    }

    // The deterministic script: step; late-admit when the set is busy;
    // occasionally cancel a live request (deterministically the lowest-id
    // live one); stop when all kTotalAdmits are admitted and none live.
    while (admitted < kTotalAdmits || sched.num_live() > 0) {
      if (steps >= kMaxSteps) {
        std::fprintf(stderr, "  STRESS FAIL: step bound exceeded\n");
        return 1;
      }
      // Step once (the dynamic set advances; some may finish).
      sched.step();
      steps++;
      rc |= invariants();
      if (rc) return rc;

      // Late admit as soon as there is room (keeps arrivals DYNAMIC — never
      // all at once, at most kMaxLive live — and guarantees we reach
      // kTotalAdmits), bounded by kMaxLive so the delta pool never OOMs.
      if (admitted < kTotalAdmits && sched.num_live() < kMaxLive) {
        if (admit_one()) return 1;
        rc |= invariants();
        if (rc) return rc;
      }
      // Deterministically cancel the lowest-id LIVE request sometimes
      // (driven by the fixed seed), to stress cancellation under load.
      if (sched.num_live() >= 3 && (rng.next() % 7) == 0) {
        RequestId victim = 0;
        for (RequestId id = 1; id < sched.next_request_id(); ++id) {
          const Request* r = sched.get(id);
          if (r && !is_terminal(r->status)) {
            victim = id;
            break;
          }
        }
        if (victim != 0) {
          CHECK(sched.cancel(victim).ok);
          rc |= invariants();
          if (rc) return rc;
        }
      }
    }

    // FINAL: everything terminal; resources fully reclaimed.
    CHECK_EQ(sched.num_live(), 0);
    CHECK_EQ(mgr.num_live_sequences(), 0);
    rc |= invariants();
    const SchedulerStats st = sched.stats();
    CHECK_EQ(st.requests_admitted, kTotalAdmits);
    CHECK_EQ(st.requests_live, 0);
    CHECK_EQ(st.requests_finished + st.requests_cancelled + st.requests_failed,
             kTotalAdmits);
    std::fprintf(stderr,
                 "[stress-part2] deterministic lifecycle stress (seed "
                 "%llu): %d requests, %d steps; invariants held after every "
                 "step/admit/cancel (live count matches, terminals never "
                 "re-advanced, each live request owns one live sequence); "
                 "final num_live=0, mgr live=0; finished=%d cancelled=%d "
                 "failed=%d\n",
                 static_cast<unsigned long long>(kSeed), kTotalAdmits, steps,
                 st.requests_finished, st.requests_cancelled,
                 st.requests_failed);
  }

  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_scheduler_continuous_stress: FAIL\n");
    return rc;
  }
  std::printf(
      "test_qwen35_scheduler_continuous_stress: PASS (combined control-plane "
      "stress: dynamic-set failure isolation [one Failed+retired-once, others "
      "continue, run() reports first error, exact final accounting] + "
      "fixed-seed deterministic lifecycle stress [40 requests, invariants "
      "held after every step/admit/cancel, terminals never re-advanced, "
      "each live request owns one live sequence, final num_live=0])\n");
  return 0;
}
