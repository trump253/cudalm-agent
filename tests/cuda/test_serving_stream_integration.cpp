// CUDALM — v0.9 Phase B: committed-token streaming INTEGRATION gate
// (CUDA, REAL Qwen3.5-0.8B-Base checkpoint).
//
// Confirms the streaming contract on the real runtime: two sessions
// (A greedy / B seeded) run TOGETHER through the ServingController
// (step_stream drives the batched decode path — one Scheduler::step()
// commits for both live requests; the serving layer drains them
// separately). For every turn:
//
//   * the CONCATENATED streamed token ids == the request's FINAL
//     committed generated ids (exactly once, in order — the hard
//     commit-before-visible constraint on real forwards),
//   * the final context length is correct (n + m exactly),
//   * the session CONTINUES to the next turn from the committed
//     boundary (turn 3 on session A after streaming turns 1-2).
//
// No full KV/Delta re-memcmp (the v0.8 hard gates are frozen); the
// per-turn token parity is the streaming-specific check.
//
// Self-skips (77) when the checkpoint is absent.
//
// Provenance: CUDALM-native (v0.9 Phase B).

#include "../../tests/common/check.h"

#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
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

SamplingConfig sampling_cfg(std::uint64_t seed) {
  return SamplingConfig{0.7f, 50, 1.0f, seed};
}

// Collect one request's streamed token ids from an event stream.
std::vector<int> streamed_tokens(const std::vector<ServingEvent>& events,
                                 RequestId rid) {
  std::vector<int> ids;
  for (const ServingEvent& e : events) {
    if (e.request_id == rid && e.kind == ServingEventKind::Token) {
      ids.push_back(e.token_id);
    }
  }
  return ids;
}

// Drive to quiescence via step_stream (the batched path), collecting
// ALL events. Returns true when every tracked request is terminal.
std::vector<ServingEvent> drive_all(ServingController& ctrl) {
  std::vector<ServingEvent> all;
  for (int i = 0; i < 400; ++i) {
    if (ctrl.stats().live_requests == 0) break;
    std::vector<ServingEvent> ev = ctrl.step_stream();
    all.insert(all.end(), ev.begin(), ev.end());
  }
  return all;
}

// Returns 0 on success (CHECK helpers must return int).
int check_stream_parity(const std::vector<ServingEvent>& all,
                        const Scheduler& sched, RequestId rid, int expect) {
  // The concatenated streamed ids == the request's final committed
  // generated ids (exactly once, in order):
  const std::vector<int> streamed = streamed_tokens(all, rid);
  const std::vector<int> final_ids = sched.get(rid)->generated;
  CHECK(streamed == final_ids);
  CHECK_EQ(static_cast<int>(streamed.size()), expect);
  return 0;
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
    std::fprintf(stderr, "[SKIP] serving stream integration: no "
                         "preconverted model at %s and --no-convert given\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] serving stream integration: checkpoint "
                         "absent at %s\n",
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
               "[serving-stream] model loaded: %d layers, vocab %d\n",
               model.num_layers(), cfg.vocab_size);

  const SamplingConfig kGreedy = SamplingConfig::greedy();
  Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  SessionManager sm(mgr);
  ModelForwarder fwd(model);
  Scheduler sched(fwd, mgr, stream, &sm);
  // Two live turns at a time — the batched decode path THROUGH the
  // serving layer.
  ServingController ctrl(sched, sm,
                         ServingLimits{-1, /*max_live_requests=*/2, 0});
  SessionId a = 0, b = 0;
  CHECK(ctrl.create_session(&a).ok);
  CHECK(ctrl.create_session(&b).ok);
  const SequenceId sa = sm.lookup(a)->sequence_id;
  const SequenceId sb = sm.lookup(b)->sequence_id;

  // ---- turn 1: A (3 in + 3 gen, greedy) / B (2 in + 3 gen, seed 42) ----
  RequestId rA1 = 0, rB1 = 0;
  CHECK(ctrl.admit_turn(a, {1024, 2048, 3073}, 3, -1, kGreedy, &rA1).ok);
  CHECK(ctrl.admit_turn(b, {100, 200}, 3, -1, sampling_cfg(42), &rB1).ok);
  std::vector<ServingEvent> all = drive_all(ctrl);
  CHECK(ctrl.stats().live_requests == 0);
  CHECK_EQ(check_stream_parity(all, sched, rA1, 3), 0);
  CHECK_EQ(check_stream_parity(all, sched, rB1, 3), 0);
  CHECK_EQ(mgr.lookup(sa)->length, 6);  // 3 + 3
  CHECK_EQ(mgr.lookup(sb)->length, 5);  // 2 + 3
  std::printf("  [ok] turn 1: streamed ids == final generated (A 3 / B 3);"
              " lengths 6 / 5\n");

  // ---- turn 2 (CONTINUE from the committed boundary): A (2+2) / B
  //      (2+2, seed 123) ----
  RequestId rA2 = 0, rB2 = 0;
  CHECK(ctrl.admit_turn(a, {15, 16}, 2, -1, kGreedy, &rA2).ok);
  CHECK(ctrl.admit_turn(b, {7, 8}, 2, -1, sampling_cfg(123), &rB2).ok);
  all = drive_all(ctrl);
  CHECK(ctrl.stats().live_requests == 0);
  CHECK_EQ(check_stream_parity(all, sched, rA2, 2), 0);
  CHECK_EQ(check_stream_parity(all, sched, rB2, 2), 0);
  CHECK_EQ(mgr.lookup(sa)->length, 10);  // 6 + 2 + 2
  CHECK_EQ(mgr.lookup(sb)->length, 9);  // 5 + 2 + 2
  std::printf("  [ok] turn 2: continued from the committed boundary; "
              "streamed ids exact; lengths 10 / 9\n");

  // ---- turn 3: the session CONTINUES after streaming (A only, 1+2) ----
  RequestId rA3 = 0;
  CHECK(ctrl.admit_turn(a, {42}, 2, -1, kGreedy, &rA3).ok);
  all = ctrl.run_stream();
  CHECK(ctrl.stats().live_requests == 0);
  CHECK_EQ(check_stream_parity(all, sched, rA3, 2), 0);
  CHECK_EQ(mgr.lookup(sa)->length, 13);  // 10 + 1 + 2
  std::printf("  [ok] turn 3: session continues after streaming; final "
              "length A 13\n");

  // ---- stats sanity ----
  const ServingStats st = ctrl.stats();
  CHECK_EQ(st.live_sessions, 2);
  CHECK_EQ(st.live_requests, 0);
  CHECK_EQ(static_cast<int>(st.total_admitted_requests), 5);
  CHECK_EQ(static_cast<int>(st.rejected_session_limit), 0);
  CHECK_EQ(static_cast<int>(st.rejected_request_limit), 0);

  // ---- clean teardown: pool accounting to zero ----
  CHECK(ctrl.destroy_session(a).ok);
  CHECK(ctrl.destroy_session(b).ok);
  CHECK_EQ(mgr.kv_pool().used_pages(), 0);
  CHECK_EQ(mgr.delta_pool().used_slots(), 0);

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("test_serving_stream_integration: PASS\n");
  return 0;
}
