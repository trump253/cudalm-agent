// CUDALM — v0.9 Phase C: session TTL / LRU eviction INTEGRATION gate
// (CUDA, REAL Qwen3.5-0.8B-Base checkpoint).
//
// A SMALL real-runtime gate (no deep KV/Delta re-memcmp — the v0.8
// hard gates are frozen): the eviction policy and persistent sessions
// COEXIST on the real model:
//
//   * create A / B; each runs ONE real turn through the controller
//     (B first, then A — A keeps the more RECENT activity),
//   * a FAKE-CLOCK TTL sweep evicts ONLY the idle B (A survives),
//   * the evicted B CANNOT continue (SessionId invalidated forever),
//   * A CONTINUES to the next turn from its committed state (the
//     streamed ids == the final generated ids; the length is exact —
//     no contamination from the evicted neighbor),
//   * a fresh C generates fresh (a new monotonic SessionId; the
//     released Delta slot is reused zeroed),
//   * clean teardown: pool accounting back to zero.
//
// Self-skips (77) when the checkpoint is absent.
//
// Provenance: CUDALM-native (v0.9 Phase C).

#include "../../tests/common/check.h"

#include <sys/stat.h>

#include <chrono>
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
const int kDeltaSlots = 2;  // exactly A + B (C reuses B's released slot)

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
    std::fprintf(stderr, "[SKIP] serving eviction integration: no "
                         "preconverted model at %s and --no-convert given\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] serving eviction integration: checkpoint "
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
  std::fprintf(stderr, "[serving-eviction] model loaded: %d layers, vocab "
                       "%d\n",
               model.num_layers(), cfg.vocab_size);

  const SamplingConfig kGreedy = SamplingConfig::greedy();
  Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  SessionManager sm(mgr);
  ModelForwarder fwd(model);
  Scheduler sched(fwd, mgr, stream, &sm);
  // Phase C policy: a 1s idle TTL (fake clock) — no background thread;
  // the sweep happens only at the explicit maintenance calls below.
  FakeClock clk;
  const SessionEvictionPolicy pol =
      SessionEvictionPolicy{}.with_idle_ttl(std::chrono::milliseconds(1000));
  ServingController ctrl(sched, sm,
                         ServingLimits{-1, /*max_live_requests=*/2, 0}, &clk,
                         pol);
  SessionId a = 0, b = 0;
  CHECK(ctrl.create_session(&a).ok);
  CHECK(ctrl.create_session(&b).ok);
  const SequenceId sa = sm.lookup(a)->sequence_id;
  const SequenceId sb = sm.lookup(b)->sequence_id;
  const int dl_before = mgr.delta_pool().used_slots();

  // ---- turn 1: B FIRST (2 in + 2 gen) — the OLDER activity ----
  RequestId rB1 = 0;
  CHECK(ctrl.admit_turn(b, {100, 200}, 2, -1, kGreedy, &rB1).ok);
  std::vector<ServingEvent> all = ctrl.run_stream();
  CHECK(ctrl.stats().live_requests == 0);
  CHECK_EQ(check_stream_parity(all, sched, rB1, 2), 0);
  CHECK_EQ(mgr.lookup(sb)->length, 4);  // 2 + 2

  // ---- A runs SECOND (3 in + 2 gen) — the RECENT activity ----
  clk.advance_ms(1000);  // t = 1000: A's activity anchors here
  RequestId rA1 = 0;
  CHECK(ctrl.admit_turn(a, {1024, 2048, 3073}, 2, -1, kGreedy, &rA1).ok);
  all = ctrl.run_stream();
  CHECK(ctrl.stats().live_requests == 0);
  CHECK_EQ(check_stream_parity(all, sched, rA1, 2), 0);
  CHECK_EQ(mgr.lookup(sa)->length, 5);  // 3 + 2
  std::printf("  [ok] turn 1: B (2+2) then A (3+2); lengths 4 / 5\n");

  // ---- the TTL sweep at t = 1050: B idle 1050ms >= 1000 (EXPIRED);
  //      A idle 50ms (SURVIVES) ----
  clk.advance_ms(50);
  const std::vector<SessionId> evicted = ctrl.evict_expired_sessions();
  CHECK(static_cast<int>(evicted.size()) == 1 && evicted[0] == b);
  CHECK(sm.lookup(b) == nullptr);  // B INVALIDATED FOREVER
  CHECK(sm.lookup(a) != nullptr);  // A survives
  CHECK_EQ(ctrl.stats().evicted_sessions_ttl, static_cast<std::uint64_t>(1));
  CHECK_EQ(mgr.delta_pool().used_slots(), dl_before - 1);
  // The evicted B CANNOT continue:
  RequestId rB2 = 0;
  CHECK(!ctrl.admit_turn(b, {1}, 1, -1, kGreedy, &rB2).ok);
  std::printf("  [ok] TTL sweep: B evicted (invalidated forever, "
              "accounting released); A survives\n");

  // ---- A CONTINUES to the next turn from its committed state (2+2) ----
  RequestId rA2 = 0;
  CHECK(ctrl.admit_turn(a, {15, 16}, 2, -1, kGreedy, &rA2).ok);
  all = ctrl.run_stream();
  CHECK(ctrl.stats().live_requests == 0);
  CHECK_EQ(check_stream_parity(all, sched, rA2, 2), 0);
  CHECK_EQ(mgr.lookup(sa)->length, 9);  // 5 + 2 + 2
  std::printf("  [ok] A continues after B's eviction (length 9); no "
              "contamination\n");

  // ---- a fresh C generates fresh (2 in + 2 gen) ----
  SessionId c = 0;
  CHECK(ctrl.create_session(&c).ok);
  CHECK(c > b);  // a NEW monotonic SessionId (B's id never reused)
  const SequenceId sc = sm.lookup(c)->sequence_id;
  RequestId rC1 = 0;
  CHECK(ctrl.admit_turn(c, {5, 6}, 2, -1, kGreedy, &rC1).ok);
  all = ctrl.run_stream();
  CHECK(ctrl.stats().live_requests == 0);
  CHECK_EQ(check_stream_parity(all, sched, rC1, 2), 0);
  CHECK_EQ(mgr.lookup(sc)->length, 4);  // 2 + 2
  CHECK_EQ(mgr.delta_pool().used_slots(), dl_before);  // A's + C's slots
  std::printf("  [ok] fresh C: new monotonic SessionId, fresh "
              "generation\n");

  // ---- stats sanity ----
  const ServingStats st = ctrl.stats();
  CHECK_EQ(st.live_sessions, 2);
  CHECK_EQ(st.live_requests, 0);
  CHECK_EQ(st.evicted_sessions_ttl, static_cast<std::uint64_t>(1));
  CHECK_EQ(st.evicted_sessions_lru, static_cast<std::uint64_t>(0));

  // ---- clean teardown: pool accounting to zero ----
  CHECK(ctrl.destroy_session(a).ok);
  CHECK(ctrl.destroy_session(c).ok);
  CHECK_EQ(mgr.kv_pool().used_pages(), 0);
  CHECK_EQ(mgr.delta_pool().used_slots(), 0);

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("test_serving_eviction_integration: PASS\n");
  return 0;
}
