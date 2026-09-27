// CUDALM — v0.6 Phase C: DYNAMIC CONTINUOUS-BATCHING reproducible BENCHMARK
// (real Qwen/Qwen3.5-0.8B-Base checkpoint; single stream; prefill stays
// serial; correctness-first). Standalone executable (NOT a ctest).
//
// Compares the SAME request workload (four requests, distinct prompt length
// 2/5/3/4, distinct max_new, 1 greedy + 3 seeded) executed two ways:
//
//   A. INDEPENDENT / SERIAL : each request run ALONE (fresh manager, direct
//      forward_token_with_state, to its max_new), one at a time, summed.
//   B. SCHEDULER CONTINUOUS BATCHED : the four requests admitted DYNAMICALLY
//      (A,B first; C after 1 step; D after 2 steps) and driven by the
//      Phase A/B scheduler, whose decode cohorts take the TRUE batched GPU
//      path (forward_batch_with_state).
//
// Both do the SAME total work (identical prompts / max_new / seeds, all four
// complete, no cancellation in the timed workload), so the comparison is
// apples-to-apples.
//
// Timing (correct CUDA discipline):
//   * kWarmupRuns UN-timed warmup runs of each mode (loads the CUDA context,
//     warms the pools / kernels);
//   * N timed runs of each mode (N = --measured-runs, default kMeasuredRuns
//     = 3 — v0.7 tooling: only the number of timed repetitions changes, the
//     per-run execution is byte-for-byte identical to the v0.6 benchmark);
//     each timed run is bracketed by cudaStreamSynchronize(stream) BEFORE
//     the start timestamp and AFTER the workload (so the wall time includes
//     all enqueued GPU work, and no residual work bleeds into the next run);
//   * std::chrono::steady_clock wall clock (host wall time around the
//     synchronized region).
//
// The report also prints a "profile_totals" section: the completed
// model-traversal / forward counts accumulated over EVERY workload execution
// of the process (warmup + all measured runs, both modes). That is the
// denominator for the v0.7 Nsight-Systems kernels-per-traversal launch-
// overhead analysis.
//
// NO performance pass threshold. This benchmark REPORTS numbers for THIS
// environment; it makes no general "X% speedup" claim (the correctness-first
// batch kernels are not tuned — that is v0.7). Even if the batched mode shows
// no speedup here, that is NOT a failure.
//
// The report explicitly records: GPU, CUDA version, checkpoint, batch
// workload, warmup / measured runs.
//
// Usage:
//   bench_qwen35_continuous_batching <full_model.cudalm> <checkpoint_dir>
//       <python> <src_dir> [out.txt] [--no-convert]
// out.txt defaults to "-" (stdout). Self-skips (77) when the checkpoint is
// absent.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "cudalm/cuda_check.h"
#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/scheduler.h"
#include "cudalm/sampling.h"
#include "cudalm/weight_loader_v2.h"

using namespace cudalm;

namespace {

bool file_exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}
int run_cmd(const std::string& cmd) {
  std::fprintf(stderr, "  $ %s\n", cmd.c_str());
  return std::system(cmd.c_str());
}

// A small host D2H helper (bf16), stream-ordered.
std::vector<__nv_bfloat16> d2h(const __nv_bfloat16* dev, std::size_t n,
                               cudaStream_t s) {
  std::vector<__nv_bfloat16> h(n);
  if (n)
    CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost, s));
  return h;
}

// v0.7 tooling: basic descriptive stats over the measured-run wall times.
struct RunStats {
  double mean = 0.0, median = 0.0, min = 0.0, max = 0.0;
};
RunStats compute_stats(const std::vector<double>& v) {
  RunStats s;
  if (v.empty()) return s;
  std::vector<double> sv = v;
  std::sort(sv.begin(), sv.end());
  double sum = 0.0;
  for (double x : sv) sum += x;
  s.mean = sum / static_cast<double>(sv.size());
  s.median = sv.empty() ? 0.0
                        : (sv.size() % 2 ? sv[sv.size() / 2]
                                         : (sv[sv.size() / 2 - 1] +
                                            sv[sv.size() / 2]) / 2.0);
  s.min = sv.front();
  s.max = sv.back();
  return s;
}

// ---- workload (identical to the Phase C continuous-batching gate) ----------
static const std::vector<int> kPromptA = {1024, 2048};           // 2 tok
static const std::vector<int> kPromptB = {15, 16, 17, 18, 19};   // 5 tok
static const std::vector<int> kPromptC = {3072, 4096, 5120};     // 3 tok
static const std::vector<int> kPromptD = {6, 7, 8, 9};           // 4 tok
static const int kMaxNewA = 3;  // greedy
static const int kMaxNewB = 6;  // seed 100
static const int kMaxNewC = 5;  // seed 200
static const int kMaxNewD = 3;  // seed 300
static const int kPageTokens = 2;
static const int kPoolPages = 16;
static const int kDeltaSlots = 4;
static const int kWarmupRuns = 1;
static const int kMeasuredRuns = 3;

SamplingConfig seeded(std::uint64_t seed) {
  return SamplingConfig{0.8f, 50, 1.0f, seed};
}

struct ReqSpec {
  const std::vector<int>* prompt;
  int max_new;
  SamplingConfig sampling;
};
std::vector<ReqSpec> workload() {
  return {{&kPromptA, kMaxNewA, SamplingConfig::greedy()},
          {&kPromptB, kMaxNewB, seeded(100)},
          {&kPromptC, kMaxNewC, seeded(200)},
          {&kPromptD, kMaxNewD, seeded(300)}};
}
int total_prompt_tokens(const std::vector<ReqSpec>& ws) {
  int tot = 0;
  for (const ReqSpec& w : ws) tot += static_cast<int>(w.prompt->size());
  return tot;
}
int total_generated(const std::vector<ReqSpec>& ws) {
  int tot = 0;
  for (const ReqSpec& w : ws) tot += w.max_new;
  return tot;
}
// Total logical token-forwards = sum over requests of (N + m - 1).
int total_logical_tokens(const std::vector<ReqSpec>& ws) {
  int tot = 0;
  for (const ReqSpec& w : ws)
    tot += static_cast<int>(w.prompt->size()) + w.max_new - 1;
  return tot;
}

// ---- mode A: independent / serial (each request alone, one at a time) ------
struct SerialMetrics {
  int logical_tokens = 0;
  int single_forward_calls = 0;
};
// Runs each request ALONE (fresh manager, direct forward_token_with_state to
// its max_new). Returns 0 on success. Timing is done by the CALLER (the work
// is enqueued on `stream` and fully drained by the per-step logits sync).
int run_serial(Qwen35Model& model, const Qwen35Config& cfg,
               const std::vector<ReqSpec>& ws, cudaStream_t stream,
               SerialMetrics* out) {
  int forwards = 0;
  for (const ReqSpec& w : ws) {
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    SequenceId sid = 0;
    if (!mgr.create_sequence(&sid).ok) return 1;
    Sampler sampler(w.sampling);
    const int N = static_cast<int>(w.prompt->size());
    for (int i = 0; i < N; ++i) {
      if (!model.forward_token_with_state((*w.prompt)[i], sid, mgr, stream)
               .ok) {
        return 1;
      }
      forwards++;
    }
    for (int g = 0; g < w.max_new; ++g) {
      std::vector<__nv_bfloat16> logits =
          d2h(model.logits(), cfg.vocab_size, stream);
      const int tok = sampler.sample(logits.data(), cfg.vocab_size);
      if (tok < 0 || tok >= cfg.vocab_size) return 1;
      if (g + 1 < w.max_new) {
        if (!model.forward_token_with_state(tok, sid, mgr, stream).ok) {
          return 1;
        }
        forwards++;
      }
    }
    mgr.retire_sequence(sid);
  }
  out->logical_tokens = forwards;
  out->single_forward_calls = forwards;
  return 0;
}

// ---- mode B: scheduler continuous batched (dynamic arrivals) ---------------
struct BatchedMetrics {
  SchedulerStats stats;
};
// Admits the workload DYNAMICALLY (A,B first; C after 1 step; D after 2) and
// drives it to completion with the Phase A/B scheduler. Returns 0 on success.
// Timing is done by the CALLER.
int run_batched(Qwen35Model& model, const Qwen35Config& cfg,
                const std::vector<ReqSpec>& ws, cudaStream_t stream,
                BatchedMetrics* out) {
  Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  ModelForwarder fwd(model);
  Scheduler sched(fwd, mgr, stream);
  RequestId ids[4] = {0, 0, 0, 0};
  auto admit = [&](int idx) -> bool {
    Scheduler::Spec sp;
    sp.prompt = *ws[static_cast<std::size_t>(idx)].prompt;
    sp.max_new_tokens = ws[static_cast<std::size_t>(idx)].max_new;
    sp.sampling = ws[static_cast<std::size_t>(idx)].sampling;
    return sched.admit(sp, &ids[idx]).ok;
  };
  if (!admit(0) || !admit(1)) return 1;  // A, B
  if (!sched.step().ok) return 1;        // 1 step
  if (!admit(2)) return 1;               // C (late)
  if (!sched.step().ok) return 1;        // 1 step
  if (!admit(3)) return 1;               // D (later)
  if (!sched.run().ok) return 1;         // drive to completion
  if (sched.num_live() != 0 || mgr.num_live_sequences() != 0) return 1;
  out->stats = sched.stats();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <full_model.cudalm> <checkpoint_dir> <python> "
                 "<src_dir> [out.txt] [--no-convert]\n",
                 argv[0]);
    return 2;
  }
  const std::string out_path = argv[1];
  const std::string ckpt = argv[2];
  const std::string py = argv[3];
  const std::string src = argv[4];
  std::string report_path = "-";
  bool no_convert = false;
  int measured_runs = kMeasuredRuns;  // v0.7 tooling: --measured-runs N
                                      // (default keeps the v0.6 behavior;
                                      // ONLY the number of timed repetitions
                                      // changes — the per-run execution is
                                      // identical)
  for (int i = 5; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--no-convert") no_convert = true;
    else if (a == "--measured-runs") {
      if (i + 1 < argc) measured_runs = std::atoi(argv[++i]);
      if (measured_runs < 1) measured_runs = 1;
    } else if (a.rfind("--measured-runs=", 0) == 0) {
      measured_runs = std::atoi(a.c_str() + 16);
      if (measured_runs < 1) measured_runs = 1;
    } else if (report_path == "-") report_path = a;
  }

  if (no_convert && !file_exists(out_path)) {
    std::fprintf(stderr,
                 "[SKIP] continuous-batching benchmark: no preconverted "
                 "model\n");
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] continuous-batching benchmark: checkpoint "
                         "absent\n");
    return 77;
  }
  if (!file_exists(out_path)) {
    if (run_cmd(py + " " + src + "/tools/convert_qwen35.py" +
                    " --full-model --checkpoint-dir " + ckpt + " --out " +
                    out_path) != 0) {
      return 1;
    }
  }

  WeightFileV2 file;
  Status s = WeightFileV2::load(out_path, &file);
  if (!s.ok) {
    std::fprintf(stderr, "load failed: %s\n", s.message.c_str());
    return 1;
  }
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  Qwen35Model model;
  s = Qwen35Model::load(file, stream, &model);
  if (!s.ok) {
    std::fprintf(stderr, "model load failed: %s\n", s.message.c_str());
    return 1;
  }
  const Qwen35Config& cfg = model.config();

  // Environment identification.
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int cuda_rt = 0, cuda_drv = 0;
  cudaRuntimeGetVersion(&cuda_rt);
  cudaDriverGetVersion(&cuda_drv);

  const std::vector<ReqSpec> ws = workload();
  const int n_req = static_cast<int>(ws.size());
  const int prompt_tok = total_prompt_tokens(ws);
  const int gen_tok = total_generated(ws);
  const int logical = total_logical_tokens(ws);

  // v0.7 tooling: PROFILE TOTALS accumulated over EVERY workload execution
  // of this process (warmup + all measured runs, BOTH modes). This ties the
  // whole process to the Nsight Systems kernel-launch count for the launch-
  // overhead analysis (kernels per completed model traversal).
  struct ProfileTotals {
    int serial_traversals = 0;          // serial mode: every forward is single
    int batched_single = 0;             // issued single attempts (batched mode)
    int batched_successful_single = 0;  // committed singles (batched mode)
    int batched_batch = 0;              // committed batches (batched mode)
    int batched_traversals = 0;         // completed traversals (batched mode)
    int batched_logical = 0;            // committed logical tokens (batched)
    int batched_batched_tokens = 0;     // sum of B over committed batches
  } tot;

  // ---- warmup (un-timed): one run of each mode ----------------------------
  for (int w = 0; w < kWarmupRuns; ++w) {
    SerialMetrics sm;
    if (run_serial(model, cfg, ws, stream, &sm) != 0) return 1;
    tot.serial_traversals += sm.single_forward_calls;
    BatchedMetrics bm;
    if (run_batched(model, cfg, ws, stream, &bm) != 0) return 1;
    tot.batched_single += bm.stats.single_forward_calls;
    tot.batched_successful_single += bm.stats.successful_single_forward_calls;
    tot.batched_batch += bm.stats.batch_forward_calls;
    tot.batched_traversals += bm.stats.model_traversal_calls;
    tot.batched_logical += bm.stats.logical_token_forwards;
    tot.batched_batched_tokens += bm.stats.batched_sequence_tokens;
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // ---- measured runs -------------------------------------------------------
  std::vector<double> serial_times, batched_times;
  SerialMetrics sm_last;
  BatchedMetrics bm_last;
  for (int r = 0; r < measured_runs; ++r) {
    CUDA_CHECK(cudaStreamSynchronize(stream));  // clean start
    const auto t0 = std::chrono::steady_clock::now();
    SerialMetrics sm;
    if (run_serial(model, cfg, ws, stream, &sm) != 0) {
      std::fprintf(stderr, "serial run %d failed\n", r);
      return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));  // include all GPU work
    const auto t1 = std::chrono::steady_clock::now();
    serial_times.push_back(std::chrono::duration<double>(t1 - t0).count());
    sm_last = sm;
    tot.serial_traversals += sm.single_forward_calls;

    CUDA_CHECK(cudaStreamSynchronize(stream));  // clean start
    const auto u0 = std::chrono::steady_clock::now();
    BatchedMetrics bm;
    if (run_batched(model, cfg, ws, stream, &bm) != 0) {
      std::fprintf(stderr, "batched run %d failed\n", r);
      return 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));  // include all GPU work
    const auto u1 = std::chrono::steady_clock::now();
    batched_times.push_back(std::chrono::duration<double>(u1 - u0).count());
    bm_last = bm;
    tot.batched_single += bm.stats.single_forward_calls;
    tot.batched_successful_single += bm.stats.successful_single_forward_calls;
    tot.batched_batch += bm.stats.batch_forward_calls;
    tot.batched_traversals += bm.stats.model_traversal_calls;
    tot.batched_logical += bm.stats.logical_token_forwards;
    tot.batched_batched_tokens += bm.stats.batched_sequence_tokens;
  }
  const RunStats serial_stats = compute_stats(serial_times);
  const RunStats batched_stats = compute_stats(batched_times);
  const double serial_s = serial_stats.mean;
  const double batched_s = batched_stats.mean;

  // ---- build the report ----------------------------------------------------
  std::string rep;
  auto line = [&](const std::string& x) { rep += x + "\n"; };
  line("# CUDALM v0.6 Phase C — dynamic continuous-batching benchmark");
  line("# (reproducible; correctness-first; NO performance pass threshold)");
  line("");
  line("environment:");
  line("  gpu: " + std::string(prop.name));
  char vbuf[64];
  std::snprintf(vbuf, sizeof(vbuf), "%d.%d", cuda_rt / 1000,
                (cuda_rt % 1000) / 10);
  line("  cuda_runtime: " + std::string(vbuf));
  std::snprintf(vbuf, sizeof(vbuf), "%d.%d", cuda_drv / 1000,
                (cuda_drv % 1000) / 10);
  line("  cuda_driver: " + std::string(vbuf));
  line("  checkpoint: " + ckpt);
  line("  model: " + out_path);
  line("  stream: single (prefill stays serial; decode cohort is batched)");
  line("");
  line("workload (SAME for both modes; all four complete, no cancellation in "
       "the timed run):");
  line("  requests: " + std::to_string(n_req) + " (A greedy; B/C/D seeded)");
  line("  prompt_tokens: " + std::to_string(prompt_tok));
  line("  generated_tokens: " + std::to_string(gen_tok));
  line("  logical_sequence_token_forwards: " + std::to_string(logical) +
       "  (sum over requests of N + m - 1)");
  line("  arrival: A,B first; C after 1 step; D after 2 steps (dynamic)");
  line("timing:");
  line("  warmup_runs (un-timed): " + std::to_string(kWarmupRuns));
  line("  measured_runs: " + std::to_string(measured_runs) +
       " (each bracketed by cudaStreamSynchronize; steady_clock wall)");
  line("");
  line("results (per-run metrics are identical across measured runs for this "
       "fixed workload; wall time is summarized):");
  line("");
  line("  mode A — independent / SERIAL (each request alone, one at a time):");
  line("    wall_time_s mean:   " + std::to_string(serial_stats.mean));
  line("    wall_time_s median: " + std::to_string(serial_stats.median));
  line("    wall_time_s min:    " + std::to_string(serial_stats.min));
  line("    wall_time_s max:    " + std::to_string(serial_stats.max));
  line("    logical_tokens: " + std::to_string(sm_last.logical_tokens));
  line("    logical_tokens_per_s (mean): " +
       std::to_string(sm_last.logical_tokens / serial_stats.mean));
  line("    single_forward_calls: " +
       std::to_string(sm_last.single_forward_calls));
  line("    successful_single_forward_calls: " +
       std::to_string(sm_last.single_forward_calls));
  line("    batch_forward_calls: 0");
  line("    model_traversal_calls: " +
       std::to_string(sm_last.single_forward_calls));
  line("    avg_decode_batch_size: 0");
  line("    max_decode_batch_size: 0");
  line("");
  line("  mode B — SCHEDULER CONTINUOUS BATCHED (dynamic arrivals):");
  line("    wall_time_s mean:   " + std::to_string(batched_stats.mean));
  line("    wall_time_s median: " + std::to_string(batched_stats.median));
  line("    wall_time_s min:    " + std::to_string(batched_stats.min));
  line("    wall_time_s max:    " + std::to_string(batched_stats.max));
  line("    logical_tokens: " +
       std::to_string(bm_last.stats.logical_token_forwards));
  line("    logical_tokens_per_s (mean): " +
       std::to_string(bm_last.stats.logical_token_forwards / batched_stats.mean));
  line("    single_forward_calls: " +
       std::to_string(bm_last.stats.single_forward_calls));
  line("    successful_single_forward_calls: " +
       std::to_string(bm_last.stats.successful_single_forward_calls));
  line("    batch_forward_calls: " +
       std::to_string(bm_last.stats.batch_forward_calls));
  line("    model_traversal_calls: " +
       std::to_string(bm_last.stats.model_traversal_calls));
  line("    batched_sequence_tokens: " +
       std::to_string(bm_last.stats.batched_sequence_tokens));
  line("    avg_decode_batch_size: " +
       std::to_string(bm_last.stats.avg_committed_decode_batch_size));
  line("    max_decode_batch_size: " +
       std::to_string(bm_last.stats.max_batch_size));
  line("");
  line("profile_totals (this process: warmup + all measured runs, BOTH "
       "modes;");
  line("ties the whole run to the Nsight Systems kernel-launch count):");
  line("  serial_mode_traversals: " + std::to_string(tot.serial_traversals));
  line("  batched_mode_single_attempts: " +
       std::to_string(tot.batched_single));
  line("  batched_mode_successful_singles: " +
       std::to_string(tot.batched_successful_single));
  line("  batched_mode_committed_batches: " +
       std::to_string(tot.batched_batch));
  line("  batched_mode_traversals: " + std::to_string(tot.batched_traversals));
  line("  batched_mode_logical_tokens: " +
       std::to_string(tot.batched_logical));
  line("  batched_mode_batched_tokens: " +
       std::to_string(tot.batched_batched_tokens));
  line("  total_completed_traversals: " +
       std::to_string(tot.serial_traversals + tot.batched_traversals));
  line("");
  line("notes:");
  line("  * logical_sequence_token_forwards == model-traversal calls only in "
       "the SERIAL mode; in the batched mode a committed batch of B is B "
       "logical tokens but ONE traversal (so batched traversals < logical "
       "tokens).");
  line("  * NO speedup claim is made: the correctness-first batch kernels are "
       "untuned (tuning is v0.7). A batched time >= serial time in THIS "
       "environment is NOT a failure.");
  line("  * Numbers are for THIS environment / checkpoint / workload only; "
       "they are not a general performance conclusion.");
  line("");
  line("per-measured-run wall_time_s:");
  for (int r = 0; r < measured_runs; ++r) {
    line("  run " + std::to_string(r + 1) + ": serial=" +
         std::to_string(serial_times[static_cast<std::size_t>(r)]) +
         "s batched=" +
         std::to_string(batched_times[static_cast<std::size_t>(r)]) + "s");
  }

  if (report_path == "-") {
    std::printf("%s", rep.c_str());
  } else {
    std::ofstream f(report_path);
    f << rep;
    std::fprintf(stderr, "wrote %s\n", report_path.c_str());
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::fprintf(stderr,
               "benchmark complete: serial=%.4fs batched=%.4fs "
               "(logical=%d tokens)\n",
               serial_s, batched_s, logical);
  return 0;
}
