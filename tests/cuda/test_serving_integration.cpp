// CUDALM — v0.9 Phase A: serving layer INTEGRATION gate (CUDA, REAL
// Qwen3.5-0.8B-Base checkpoint).
//
// Confirms that the NEW serving layer (ServingController) driving the
// FROZEN Session/Scheduler changes NOTHING about token/state
// correctness, and that the limits enforce (and release) correctly on
// the real runtime:
//
//   * PART 1 — CORRECTNESS: sessions A (greedy) and B (seeded
//     sampling), TWO turns each, admitted and driven THROUGH the
//     controller (two live turns at a time — the batched path), vs the
//     SAME token streams driven directly through a raw Scheduler on an
//     independent manager: the generated token IDs per turn, the turn-
//     boundary lengths and the final lengths are IDENTICAL (the serving
//     layer is policy-only; it does not touch the token path).
//   * PART 2 — LIMITS ON THE REAL RUNTIME (max_sessions = 1,
//     max_live_requests = 1): session create rejected at the limit
//     (zero mutation), turn admission rejected at the limit (zero
//     mutation, no RequestId consumed), then REUSE — after the live
//     request finishes and after the session is destroyed, new
//     admissions succeed.
//   * PART 3 — CONTEXT POLICY CAP (cap 8 << the model's max_seq_len
//     262144): the exact boundary (length + input + max_new == cap) is
//     ALLOWED; over the cap is REJECTED (zero mutation) — the cap is a
//     pure policy restriction below the model limit.
//   * CLEAN TEARDOWN: every manager's pool accounting back to zero.
//
// Self-skips (77) when the checkpoint is absent.
//
// Provenance: CUDALM-native (v0.9 Phase A).

#include "../../tests/common/check.h"

#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/scheduler.h"
#include "cudalm/session.h"
#include "cudalm/serving_controller.h"
#include "cudalm/weight_loader_v2.h"

using namespace cudalm;

namespace {

bool file_exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

const int kPageTokens = 2;
const int kPoolPages = 48;
const int kDeltaSlots = 2;

// The Phase C integration gate's token streams (arbitrary ids inside
// the real vocab): A (greedy) / B (seeded), two turns each.
const std::vector<int> kA_T1 = {1024, 2048, 3073};
const std::vector<int> kB_T1 = {100, 200};
const std::vector<int> kA_T2 = {15, 16};
const std::vector<int> kB_T2 = {7, 8};
const int kA_T1_M = 3;
const int kB_T1_M = 3;
const int kA_T2_M = 2;
const int kB_T2_M = 2;

SamplingConfig sampling_cfg(std::uint64_t seed) {
  return SamplingConfig{0.7f, 50, 1.0f, seed};
}

int run_turn(Scheduler& sched, SessionId sid,
             const std::vector<int>& input, int max_new,
             const SamplingConfig& sampling) {
  RequestId rid = 0;
  Status s = sched.admit_session_turn(sid, input, max_new, -1, sampling,
                                      &rid);
  if (!s.ok) {
    std::fprintf(stderr, "  [FAIL] admit: %s\n", s.message.c_str());
    std::exit(1);
  }
  sched.run();
  const Request* r = sched.get(rid);
  if (r == nullptr || r->status != RequestStatus::Finished) {
    std::fprintf(stderr, "  [FAIL] turn not Finished\n");
    std::exit(1);
  }
  return static_cast<int>(r->generated.size());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <full_model.cudalm> <checkpoint_dir> <python> "
                 "<src_dir> [--no-convert]\n",
                 argv[0]);
    return 2;
  }
  const std::string out = argv[1];
  const std::string ckpt = argv[2];
  const std::string py = argv[3];
  const std::string src = argv[4];
  const bool no_convert = argc >= 6 && std::string(argv[5]) == "--no-convert";

  if (no_convert && !file_exists(out)) {
    std::fprintf(stderr, "[SKIP] serving integration: no preconverted "
                         "model at %s and --no-convert given\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] serving integration: checkpoint absent at "
                         "%s\n",
                 ckpt.c_str());
    return 77;
  }
  if (!file_exists(out)) {
    CHECK_EQ(system(("python3 " + src + "/tools/convert_qwen35.py" +
                     " --full-model --checkpoint-dir " + ckpt +
                     " --out " + out).c_str()),
             0);
  }

  WeightFileV2 file;
  Status s = WeightFileV2::load(out, &file);
  CHECK(s.ok);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Qwen35Model model;
  s = Qwen35Model::load(file, stream, &model);
  CHECK(s.ok);
  CHECK(model.loaded());
  const Qwen35Config& cfg = model.config();
  std::fprintf(stderr,
               "[serving] model loaded: %d layers, vocab %d, max_seq_len "
               "%d\n",
               model.num_layers(), cfg.vocab_size, cfg.max_seq_len);

  const SamplingConfig kGreedy = SamplingConfig::greedy();

  // =========================================================================
  // PART 1 — CORRECTNESS: controller-driven A/B (two turns each) vs the
  // raw-scheduler reference on an independent manager
  // =========================================================================
  {
    // --- reference (no serving layer) ---
    Qwen35StateManager ref_mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots,
                               stream);
    SessionManager ref_sm(ref_mgr);
    ModelForwarder ref_fwd(model);
    Scheduler ref_sched(ref_fwd, ref_mgr, stream, &ref_sm);
    SessionId ra = 0, rb = 0;
    CHECK(ref_sm.create_session(&ra).ok);
    CHECK(ref_sm.create_session(&rb).ok);
    const SequenceId rsa = ref_sm.lookup(ra)->sequence_id;
    const SequenceId rsb = ref_sm.lookup(rb)->sequence_id;
    std::vector<int> ref_gen;  // A t1, A t2, B t1, B t2 (in run order)
    run_turn(ref_sched, ra, kA_T1, kA_T1_M, kGreedy);
    run_turn(ref_sched, rb, kB_T1, kB_T1_M, sampling_cfg(42));
    run_turn(ref_sched, ra, kA_T2, kA_T2_M, kGreedy);
    run_turn(ref_sched, rb, kB_T2, kB_T2_M, sampling_cfg(123));
    std::vector<int> ref_A, ref_B;
    for (int i = 0; i < 2; ++i) {
      // requests: 1=A t1, 2=B t1, 3=A t2, 4=B t2
      ref_A.insert(ref_A.end(), ref_sched.get(static_cast<RequestId>(2 * i + 1))
                                   ->generated.begin(),
                   ref_sched.get(static_cast<RequestId>(2 * i + 1))
                       ->generated.end());
      ref_B.insert(ref_B.end(), ref_sched.get(static_cast<RequestId>(2 * i + 2))
                                   ->generated.begin(),
                   ref_sched.get(static_cast<RequestId>(2 * i + 2))
                       ->generated.end());
    }
    const int ref_lenA = ref_mgr.lookup(rsa)->length;
    const int ref_lenB = ref_mgr.lookup(rsb)->length;

    // --- serving layer (the NEW path) ---
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    SessionManager sm(mgr);
    ModelForwarder fwd(model);
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm,
                           ServingLimits{/*max_sessions=*/2,
                                         /*max_live_requests=*/2, 0});
    SessionId a = 0, b = 0;
    CHECK(ctrl.create_session(&a).ok);
    CHECK(ctrl.create_session(&b).ok);
    const SequenceId sa = sm.lookup(a)->sequence_id;
    const SequenceId sb = sm.lookup(b)->sequence_id;

    // Turn 1 pair: BOTH admitted (two live) and driven in one run() —
    // the batched path THROUGH the controller.
    RequestId rA1 = 0, rB1 = 0;
    CHECK(ctrl.admit_turn(a, kA_T1, kA_T1_M, -1, kGreedy, &rA1).ok);
    CHECK(ctrl.admit_turn(b, kB_T1, kB_T1_M, -1, sampling_cfg(42), &rB1).ok);
    CHECK_EQ(ctrl.stats().live_requests, 2);
    CHECK(ctrl.run().ok);
    CHECK_EQ(ctrl.stats().live_requests, 0);

    // Turn 2 pair: admitted again (the quota was RELEASED by the
    // terminal turn-1 requests — reuse) and driven.
    RequestId rA2 = 0, rB2 = 0;
    CHECK(ctrl.admit_turn(a, kA_T2, kA_T2_M, -1, kGreedy, &rA2).ok);
    CHECK(ctrl.admit_turn(b, kB_T2, kB_T2_M, -1, sampling_cfg(123), &rB2)
              .ok);
    CHECK(ctrl.run().ok);

    // TOKEN PARITY + LENGTH PARITY (the serving layer changed nothing):
    CHECK(sched.get(rA1)->generated == ref_sched.get(1)->generated);
    CHECK(sched.get(rA2)->generated == ref_sched.get(3)->generated);
    CHECK(sched.get(rB1)->generated == ref_sched.get(2)->generated);
    CHECK(sched.get(rB2)->generated == ref_sched.get(4)->generated);
    CHECK_EQ(mgr.lookup(sa)->length, ref_lenA);
    CHECK_EQ(mgr.lookup(sb)->length, ref_lenB);
    const ServingStats st = ctrl.stats();
    CHECK_EQ(st.live_sessions, 2);
    CHECK_EQ(st.live_requests, 0);
    CHECK_EQ(static_cast<int>(st.total_admitted_requests), 4);
    CHECK_EQ(static_cast<int>(st.rejected_session_limit), 0);
    CHECK_EQ(static_cast<int>(st.rejected_request_limit), 0);
    CHECK_EQ(static_cast<int>(st.rejected_context_limit), 0);
    std::printf("  [ok] part 1: A/B through the serving layer == the raw "
                "token-level path (generated ids + lengths identical; "
                "final A %d / B %d)\n",
                ref_lenA, ref_lenB);

    CHECK(ctrl.destroy_session(a).ok);
    CHECK(ctrl.destroy_session(b).ok);
    CHECK(ref_sm.destroy_session(ra).ok);
    CHECK(ref_sm.destroy_session(rb).ok);
    CHECK_EQ(mgr.kv_pool().used_pages(), 0);
    CHECK_EQ(mgr.delta_pool().used_slots(), 0);
    CHECK_EQ(ref_mgr.kv_pool().used_pages(), 0);
    CHECK_EQ(ref_mgr.delta_pool().used_slots(), 0);
  }

  // =========================================================================
  // PART 2 — LIMITS ON THE REAL RUNTIME (max_sessions = 1,
  // max_live_requests = 1) + reuse
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    SessionManager sm(mgr);
    ModelForwarder fwd(model);
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm,
                           ServingLimits{/*max_sessions=*/1,
                                         /*max_live_requests=*/1, 0});
    SessionId s1 = 0, s2 = 0;
    CHECK(ctrl.create_session(&s1).ok);
    Status cs = ctrl.create_session(&s2);  // session limit (1 == 1)
    CHECK(!cs.ok);
    CHECK(cs.message.find("session limit") != std::string::npos);
    CHECK_EQ(s2, static_cast<SessionId>(0));
    CHECK_EQ(sm.num_sessions(), 1);

    RequestId r1 = 0;
    CHECK(ctrl.admit_turn(s1, {5, 6, 7}, 4, -1, kGreedy, &r1).ok);
    RequestId r2 = 0;
    const RequestId next = sched.next_request_id();
    const int len1 = mgr.lookup(sm.lookup(s1)->sequence_id)->length;
    Status ts = ctrl.admit_turn(s1, {8, 9}, 2, -1, kGreedy, &r2);
    // REJECTED by the SERVING live-request limit (before the runtime's
    // busy check) — zero mutation.
    CHECK(!ts.ok);
    CHECK(ts.message.find("live request limit") != std::string::npos);
    CHECK_EQ(r2, static_cast<RequestId>(0));
    CHECK_EQ(sched.next_request_id(), next);
    CHECK_EQ(mgr.lookup(sm.lookup(s1)->sequence_id)->length, len1);

    CHECK(ctrl.run().ok);  // the turn commits on the real model
    CHECK_EQ(ctrl.stats().live_requests, 0);

    // REUSE after the terminal release:
    RequestId r3 = 0;
    CHECK(ctrl.admit_turn(s1, {8, 9}, 4, -1, kGreedy, &r3).ok);
    CHECK(ctrl.run().ok);
    CHECK(sched.get(r3)->status == RequestStatus::Finished);

    // REUSE after the session release:
    CHECK(ctrl.destroy_session(s1).ok);
    CHECK_EQ(ctrl.stats().live_sessions, 0);
    CHECK(ctrl.create_session(&s2).ok);
    std::printf("  [ok] part 2: session/request limits enforced + reused "
                "on the real runtime\n");
    CHECK(ctrl.destroy_session(s2).ok);
    CHECK_EQ(mgr.kv_pool().used_pages(), 0);
    CHECK_EQ(mgr.delta_pool().used_slots(), 0);
  }

  // =========================================================================
  // PART 3 — CONTEXT POLICY CAP (cap 8 << max_seq_len 262144)
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    SessionManager sm(mgr);
    ModelForwarder fwd(model);
    Scheduler sched(fwd, mgr, stream, &sm);
    ServingController ctrl(sched, sm,
                           ServingLimits{-1, -1, /*max_context=*/8});
    SessionId s1 = 0;
    CHECK(ctrl.create_session(&s1).ok);
    const SequenceId seq = sm.lookup(s1)->sequence_id;

    // Exact boundary: 0 + 4 + 4 == 8 -> ALLOWED (and runs on the real
    // model — the cap does not alter the token path).
    RequestId r1 = 0;
    CHECK(ctrl.admit_turn(s1, {1, 2, 3, 4}, 4, -1, kGreedy, &r1).ok);
    CHECK(ctrl.run().ok);
    CHECK_EQ(mgr.lookup(seq)->length, 8);

    // Over the cap: 8 + 1 + 1 = 10 > 8 -> REJECTED (zero mutation),
    // while the model limit (262144) would allow it.
    RequestId r2 = 0;
    const int len = mgr.lookup(seq)->length;
    Status ts = ctrl.admit_turn(s1, {5}, 1, -1, kGreedy, &r2);
    CHECK(!ts.ok);
    CHECK(ts.message.find("context limit") != std::string::npos);
    CHECK_EQ(r2, static_cast<RequestId>(0));
    CHECK_EQ(mgr.lookup(seq)->length, len);
    std::printf("  [ok] part 3: context policy cap (8 << %d) enforced on "
                "the real runtime; boundary accepted, over-cap rejected\n",
                cfg.max_seq_len);
    CHECK(ctrl.destroy_session(s1).ok);
    CHECK_EQ(mgr.kv_pool().used_pages(), 0);
    CHECK_EQ(mgr.delta_pool().used_slots(), 0);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("test_serving_integration: PASS\n");
  return 0;
}
