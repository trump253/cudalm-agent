// CUDALM — v0.6 Phase B: scheduler TRUE-BATCHED decode real-checkpoint gate
// (CUDA, REAL Qwen3.5-0.8B-Base checkpoint).
//
// Proves that the v0.6 Phase B scheduler (deterministic FIFO control plane
// over the TRUE BATCHED decode runtime) is semantically IDENTICAL to
// independent execution AND actually uses batched GPU compute:
//
//   * three requests with DIFFERENT prompt lengths (3 / 5 / 2 tokens) and
//     chosen max_new_tokens so the decode cohort forms at B = 3 and then
//     SHRINKS to B = 2 when one request finishes (A max_new 4, B max_new 4,
//     C max_new 6 — A and C are still alive when B's long prefill completes,
//     so all three are decode-ready together for one step, then A finishes
//     leaving a B=2 cohort):
//         A: prompt {1024,2048,3072} (3), max_new 4, seed 42
//         B: prompt {15,16,17,18,19}   (5), max_new 4, seed 123
//         C: prompt {4096,5000}        (2), max_new 6, seed 7
//
//   * BATCHING PROOF (scheduler instrumentation + forwarder observation):
//       - sched.batch_forward_calls() > 0  (batched forwards actually issued);
//       - sched.max_batch_size() == 3      (a 3-cohort was batched);
//       - the per-forward_batch cohort sizes recorded by the forwarder are
//         EXACTLY {3, 2} (a B=3 cohort, then a B=2 cohort after A finishes —
//         the batch-shrink case);
//       - for a decode cohort of B>=2 there is EXACTLY ONE batch forward,
//         never B single forwards (batch_forward_calls < single decode count).
//
//   * PARITY vs INDEPENDENT references (fresh manager, ONE sequence alone,
//     direct forward_token_with_state — NOT the scheduler): per request the
//     generated token IDs, the FULL logits[248320] that produced each
//     generated token, the forward count, the finish reason, and the FINAL
//     hybrid state (18x DeltaNet conv/rec + 6x logical K/V) are all EXACT.
//
// Phase B is CORRECTNESS-first: the GPU batches the decode (one model
// traversal per cohort) but the observable per-request results are IDENTICAL
// to the frozen Phase A single-forward semantics. No throughput claim.
//
// Self-skips (77) when the checkpoint is absent; in the Phase B evidence
// environment the checkpoint IS present and this test actually runs (77 is
// not a sign-off).

#include "../../tests/common/check.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/scheduler.h"
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

// Request specs (see header): different prompt lengths -> heterogeneous
// decode positions; max_new chosen so the cohort is B=3 then shrinks to B=2.
static const std::vector<int> kPromptA = {1024, 2048, 3072};  // 3
static const std::vector<int> kPromptB = {15, 16, 17, 18, 19};  // 5
static const std::vector<int> kPromptC = {4096, 5000};  // 2
// max_new chosen so the cohort is B=3 (A,B,C all decode-ready together) then
// shrinks to B=2 (A finishes mid-run). See the header trace.
static const int kMaxNewA = 4;
static const int kMaxNewB = 4;
static const int kMaxNewC = 6;
static const int kPageTokens = 2;
static const int kPoolPages = 24;  // generous: the batch preflight never
                                   // falls back (capacity is not the gate).
static const int kDeltaSlots = 8;

SamplingConfig sampling_cfg(std::uint64_t seed) {
  return SamplingConfig{0.8f, 50, 1.0f, seed};
}
int expected_forwards(int n, int m) { return n + m - 1; }

// ---- device -> host + bit-exact compare ------------------------------------
std::vector<__nv_bfloat16> d2h_bf16(const __nv_bfloat16* dev, std::size_t n,
                                    cudaStream_t s) {
  std::vector<__nv_bfloat16> h(n);
  if (n)
    CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost, s));
  return h;
}
std::vector<float> d2h_f32(const float* dev, std::size_t n, cudaStream_t s) {
  std::vector<float> h(n);
  if (n)
    CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(float),
                               cudaMemcpyDeviceToHost, s));
  return h;
}
int cmp_exact(const char* name, const std::vector<__nv_bfloat16>& ref,
              const std::vector<__nv_bfloat16>& act) {
  if (ref.size() != act.size()) {
    std::fprintf(stderr, "  %-48s FAIL size %zu != %zu\n", name, ref.size(),
                 act.size());
    return 1;
  }
  const bool ok =
      std::memcmp(ref.data(), act.data(), ref.size() * sizeof(__nv_bfloat16)) ==
      0;
  std::fprintf(stderr, "  %-48s %s (n=%zu)\n", name, ok ? "bit-identical" : "FAIL",
               ref.size());
  return ok ? 0 : 1;
}
int cmp_exact_f32(const char* name, const std::vector<float>& ref,
                  const std::vector<float>& act) {
  if (ref.size() != act.size()) {
    std::fprintf(stderr, "  %-48s FAIL size %zu != %zu\n", name, ref.size(),
                 act.size());
    return 1;
  }
  const bool ok =
      std::memcmp(ref.data(), act.data(), ref.size() * sizeof(float)) == 0;
  std::fprintf(stderr, "  %-48s %s (n=%zu)\n", name, ok ? "bit-identical" : "FAIL",
               ref.size());
  return ok ? 0 : 1;
}

// ---- full hybrid state of ONE sequence (external, through the pool) --------
struct StateSnap {
  std::vector<std::vector<__nv_bfloat16>> conv;  // [L] linear layers
  std::vector<std::vector<float>> rec;           // [L]
  std::vector<std::vector<__nv_bfloat16>> kvk;   // [L] logical rows 0..n-1
  std::vector<std::vector<__nv_bfloat16>> kvv;   // [L]
  int length = 0;
};

int capture_state(const Qwen35StateManager& mgr, const SequenceState& rec,
                  const Qwen35Config& cfg, int n_tokens, cudaStream_t stream,
                  StateSnap* out) {
  const int L = cfg.num_hidden_layers;
  const int nkv = cfg.n_kv_heads;
  const int hd = cfg.head_dim;
  const int pt = mgr.page_tokens();
  const int slot = rec.delta_slot;
  out->length = static_cast<int>(rec.length);
  out->conv.assign(L, {});
  out->rec.assign(L, {});
  out->kvk.assign(L, {});
  out->kvv.assign(L, {});
  for (int i = 0; i < L; ++i) {
    if (cfg.is_linear_attention(i)) {
      const int ord = mgr.delta_pool().linear_layer_ordinal(i);
      const std::size_t cn = static_cast<std::size_t>(cfg.linear_conv_dim()) *
                             cfg.linear_conv_state_len();
      const std::size_t rn = static_cast<std::size_t>(cfg.lin_num_v_heads) *
                             cfg.lin_value_head_dim * cfg.lin_value_head_dim;
      out->conv[static_cast<std::size_t>(i)] =
          d2h_bf16(mgr.delta_pool().conv(ord, slot), cn, stream);
      out->rec[static_cast<std::size_t>(i)] =
          d2h_f32(mgr.delta_pool().recurrent(ord, slot), rn, stream);
    } else {
      const int ord = mgr.kv_pool().full_layer_ordinal(i);
      const std::size_t rn = static_cast<std::size_t>(nkv) * n_tokens * hd;
      std::vector<__nv_bfloat16> k(rn), v(rn);
      for (int t = 0; t < n_tokens; ++t) {
        const int page = rec.block_table.lookup(t / pt);
        if (page < 0) {
          std::fprintf(stderr, "  [state] block %d unallocated\n", t / pt);
          return 1;
        }
        const int off = t % pt;
        for (int n = 0; n < nkv; ++n) {
          const std::size_t dst =
              (static_cast<std::size_t>(n) * n_tokens + t) * hd;
          const __nv_bfloat16* kp =
              mgr.kv_pool().k_page(ord, page) +
              (static_cast<std::size_t>(n) * pt + off) * hd;
          const __nv_bfloat16* vp =
              mgr.kv_pool().v_page(ord, page) +
              (static_cast<std::size_t>(n) * pt + off) * hd;
          const std::size_t bytes = hd * sizeof(__nv_bfloat16);
          CUDA_CHECK(cudaMemcpyAsync(k.data() + dst, kp, bytes,
                                     cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaMemcpyAsync(v.data() + dst, vp, bytes,
                                     cudaMemcpyDeviceToHost, stream));
        }
      }
      out->kvk[static_cast<std::size_t>(i)] = std::move(k);
      out->kvv[static_cast<std::size_t>(i)] = std::move(v);
    }
  }
  return 0;
}

int cmp_state(const char* label, const StateSnap& ref, const StateSnap& act,
              int n_tokens, const Qwen35Config& cfg) {
  int rc = 0;
  char name[72];
  std::snprintf(name, sizeof(name), "%s length", label);
  std::fprintf(stderr, "  %-48s %s (%d == %d)\n", name,
               ref.length == act.length ? "bit-identical" : "FAIL", ref.length,
               act.length);
  rc |= (ref.length == act.length) ? 0 : 1;
  const int L = cfg.num_hidden_layers;
  for (int i = 0; i < L; ++i) {
    if (cfg.is_linear_attention(i)) {
      std::snprintf(name, sizeof(name), "%s L%d delta.conv", label, i);
      rc |= cmp_exact(name, ref.conv[static_cast<std::size_t>(i)],
                      act.conv[static_cast<std::size_t>(i)]);
      std::snprintf(name, sizeof(name), "%s L%d delta.rec", label, i);
      rc |= cmp_exact_f32(name, ref.rec[static_cast<std::size_t>(i)],
                          act.rec[static_cast<std::size_t>(i)]);
    } else {
      std::snprintf(name, sizeof(name), "%s L%d kv.K[0..%d]", label, i,
                    n_tokens - 1);
      rc |= cmp_exact(name, ref.kvk[static_cast<std::size_t>(i)],
                      act.kvk[static_cast<std::size_t>(i)]);
      std::snprintf(name, sizeof(name), "%s L%d kv.V[0..%d]", label, i,
                    n_tokens - 1);
      rc |= cmp_exact(name, ref.kvv[static_cast<std::size_t>(i)],
                      act.kvv[static_cast<std::size_t>(i)]);
    }
  }
  return rc;
}

// ---- recording forwarder (logits + batch sizes + per-sequence final state) -
struct LogEntry {
  SequenceId sid;
  int token;
  std::vector<__nv_bfloat16> logits;  // [vocab]
};

// Wraps the real ModelForwarder (which supports_batch). After every
// successful forward it records the FULL logits (per row for a batch) and
// snapshots the FULL hybrid state of every advanced sequence (the LAST
// snapshot per sequence == its final state, captured before the scheduler
// retires it). It also records the size B of every forward_batch call.
class RecordingForwarder : public SequenceForwarder {
 public:
  RecordingForwarder(Qwen35Model& model, const Qwen35Config& cfg)
      : inner_(model), cfg_(cfg), vocab_(model.config().vocab_size) {}

  Status forward_token(int token_id, SequenceId sequence_id,
                       Qwen35StateManager& mgr, cudaStream_t stream) override {
    Status s = inner_.forward_token(token_id, sequence_id, mgr, stream);
    if (s.ok) {
      std::vector<__nv_bfloat16> logits;
      s = inner_.logits_to_host(&logits, stream);
      LogEntry e;
      e.sid = sequence_id;
      e.token = token_id;
      e.logits = std::move(logits);
      log_.push_back(std::move(e));
      snapshot_state(mgr, sequence_id, stream);
    }
    return s;
  }

  Status forward_batch(const int* token_ids, const SequenceId* sids, int B,
                       Qwen35StateManager& mgr,
                       cudaStream_t stream) override {
    Status s = inner_.forward_batch(token_ids, sids, B, mgr, stream);
    if (s.ok) {
      batch_sizes_.push_back(B);
      std::vector<__nv_bfloat16> logits;
      s = inner_.logits_batch_to_host(&logits, B, stream);
      for (int b = 0; b < B; ++b) {
        LogEntry e;
        e.sid = sids[b];
        e.token = token_ids[b];
        e.logits.assign(logits.begin() + static_cast<std::size_t>(b) * vocab_,
                        logits.begin() + static_cast<std::size_t>(b + 1) *
                                            vocab_);
        log_.push_back(std::move(e));
        snapshot_state(mgr, sids[b], stream);
      }
    }
    return s;
  }

  Status logits_to_host(std::vector<__nv_bfloat16>* out,
                        cudaStream_t stream) const override {
    return inner_.logits_to_host(out, stream);
  }
  Status logits_batch_to_host(std::vector<__nv_bfloat16>* out, int B,
                              cudaStream_t stream) const override {
    return inner_.logits_batch_to_host(out, B, stream);
  }
  int vocab_size() const override { return vocab_; }
  bool supports_batch() const override { return inner_.supports_batch(); }

  // The last `m` forwards of `sid`, in order (the forwards that produced its
  // `m` generated tokens).
  std::vector<LogEntry> last_forwards(SequenceId sid, int m) const {
    std::vector<const LogEntry*> mine;
    for (const LogEntry& e : log_)
      if (e.sid == sid) mine.push_back(&e);
    if (static_cast<int>(mine.size()) < m) {
      std::fprintf(stderr, "  [recording] %zu < %d forwards for sid %llu\n",
                   mine.size(), m, static_cast<unsigned long long>(sid));
      abort();
    }
    std::vector<LogEntry> out;
    for (std::size_t i = mine.size() - static_cast<std::size_t>(m);
         i < mine.size(); ++i)
      out.push_back(*mine[i]);
    return out;
  }
  int num_forwards(SequenceId sid) const {
    int n = 0;
    for (const LogEntry& e : log_)
      if (e.sid == sid) n++;
    return n;
  }
  const std::vector<int>& batch_sizes() const { return batch_sizes_; }
  const StateSnap* state_after(SequenceId sid) const {
    auto it = state_after_.find(sid);
    return it == state_after_.end() ? nullptr : &it->second;
  }

 private:
  void snapshot_state(Qwen35StateManager& mgr, SequenceId sid,
                      cudaStream_t stream) {
    const SequenceState* rec = mgr.lookup(sid);
    if (rec == nullptr) return;
    StateSnap snap;
    if (capture_state(mgr, *rec, cfg_, static_cast<int>(rec->length), stream,
                      &snap) == 0)
      state_after_[sid] = std::move(snap);  // last write wins == final state
  }

  ModelForwarder inner_;
  const Qwen35Config& cfg_;
  int vocab_;
  std::vector<LogEntry> log_;
  std::vector<int> batch_sizes_;
  std::map<SequenceId, StateSnap> state_after_;
};

// ---- independent reference: ONE sequence alone, DIRECT forwards ------------
struct RefResult {
  std::vector<int> generated;
  std::vector<std::vector<__nv_bfloat16>> gen_logits;  // [m] full logits
  int forward_count = 0;
  bool eos_hit = false;
  StateSnap final_state;  // captured after the last forward (before retire)
  bool has_state = false;
};

int run_reference_direct(Qwen35Model& model, const Qwen35Config& cfg,
                         const char* label, const std::vector<int>& prompt,
                         int max_new_tokens, int eos_token_id,
                         const SamplingConfig& sampling, cudaStream_t stream,
                         RefResult* out) {
  Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  SequenceId sid = 0;
  Status s = mgr.create_sequence(&sid);
  CHECK(s.ok);
  Sampler sampler(sampling);
  const int N = static_cast<int>(prompt.size());
  for (int i = 0; i < N; ++i) {
    s = model.forward_token_with_state(prompt[i], sid, mgr, stream);
    if (!s.ok) {
      std::fprintf(stderr, "  ref %s forward %d: %s\n", label, i,
                   s.message.c_str());
      return 1;
    }
    out->forward_count++;
  }
  while (static_cast<int>(out->generated.size()) < max_new_tokens) {
    std::vector<__nv_bfloat16> logits =
        d2h_bf16(model.logits(), cfg.vocab_size, stream);
    out->gen_logits.push_back(std::move(logits));
    const int g = sampler.sample(out->gen_logits.back().data(), cfg.vocab_size);
    if (g < 0 || g >= cfg.vocab_size) {
      std::fprintf(stderr, "  ref %s: sampler out of range\n", label);
      return 1;
    }
    out->generated.push_back(g);
    if (g == eos_token_id) {
      out->eos_hit = true;
      break;
    }
    if (static_cast<int>(out->generated.size()) >= max_new_tokens) break;
    s = model.forward_token_with_state(g, sid, mgr, stream);
    if (!s.ok) {
      std::fprintf(stderr, "  ref %s decode: %s\n", label, s.message.c_str());
      return 1;
    }
    out->forward_count++;
  }
  // Final hybrid state (after the last forward, before retire).
  const SequenceState* rec = mgr.lookup(sid);
  CHECK(rec != nullptr);
  CHECK_EQ(capture_state(mgr, *rec, cfg, static_cast<int>(rec->length), stream,
                         &out->final_state),
           0);
  out->has_state = true;
  CHECK(mgr.retire_sequence(sid).ok);
  std::fprintf(stderr, "[reference] %-4s: %d generated, %d forwards, %s\n",
               label, static_cast<int>(out->generated.size()),
               out->forward_count, out->eos_hit ? "eos" : "max_new_tokens");
  return 0;
}

int cmp_request(const char* label, const Request& req, const RefResult& ref,
                const std::vector<LogEntry>& gen_entries,
                const StateSnap* sched_state, const Qwen35Config& cfg) {
  int rc = 0;
  std::fprintf(stderr, "  [parity] %s: generated %zu, forwards %d (expected "
                       "%d), reason %d\n",
               label, req.generated.size(), req.forward_count,
               ref.forward_count, static_cast<int>(req.finish_reason));
  CHECK_EQ(req.generated.size(), ref.generated.size());
  for (std::size_t i = 0; i < req.generated.size(); ++i)
    CHECK_EQ(req.generated[i], ref.generated[i]);
  CHECK_EQ(gen_entries.size(), ref.gen_logits.size());
  for (std::size_t i = 0; i < gen_entries.size(); ++i) {
    char name[72];
    std::snprintf(name, sizeof(name), "%s gen_logits[%zu][vocab]", label, i);
    rc |= cmp_exact(name, ref.gen_logits[i], gen_entries[i].logits);
  }
  CHECK_EQ(req.forward_count, ref.forward_count);
  const FinishReason exp_reason =
      ref.eos_hit ? FinishReason::Eos : FinishReason::MaxNewTokens;
  CHECK_EQ(req.finish_reason, exp_reason);
  // FINAL hybrid state EXACT.
  if (sched_state == nullptr || !ref.has_state) {
    std::fprintf(stderr, "  %-48s FAIL missing final state\n", label);
    return 1;
  }
  rc |= cmp_state(label, ref.final_state, *sched_state, ref.final_state.length,
                  cfg);
  return rc;
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
    std::fprintf(stderr, "[SKIP] scheduler batch: no preconverted model\n");
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] scheduler batch: checkpoint absent\n");
    return 77;
  }
  if (!file_exists(out)) {
    CHECK_EQ(run_cmd(py + " " + src + "/tools/convert_qwen35.py" +
                         " --full-model --checkpoint-dir " + ckpt + " --out " +
                         out),
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
  const Qwen35Config& cfg = model.config();
  std::fprintf(stderr,
               "[scheduler-batch] model loaded: %d layers, vocab %d, "
               "page_tokens %d, pool pages %d, delta slots %d\n",
               model.num_layers(), cfg.vocab_size, kPageTokens, kPoolPages,
               kDeltaSlots);
  int rc = 0;

  // =========================================================================
  // Part 1: independent references (A/B/C alone, DIRECT forwards).
  // =========================================================================
  RefResult refA, refB, refC;
  rc |= run_reference_direct(model, cfg, "A", kPromptA, kMaxNewA, -1,
                             sampling_cfg(42), stream, &refA);
  rc |= run_reference_direct(model, cfg, "B", kPromptB, kMaxNewB, -1,
                             sampling_cfg(123), stream, &refB);
  rc |= run_reference_direct(model, cfg, "C", kPromptC, kMaxNewC, -1,
                             sampling_cfg(7), stream, &refC);
  CHECK_EQ(refA.forward_count, expected_forwards(3, kMaxNewA));  // 3 + 3 = 6
  CHECK_EQ(refB.forward_count, expected_forwards(5, kMaxNewB));  // 5 + 3 = 8
  CHECK_EQ(refC.forward_count, expected_forwards(2, kMaxNewC));  // 2 + 5 = 7

  // =========================================================================
  // Part 2: scheduler run with the TRUE BATCHED forwarder.
  // =========================================================================
  RecordingForwarder fwd(model, cfg);
  {
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0, c = 0;
    Scheduler::Spec sa;
    sa.prompt = kPromptA;
    sa.max_new_tokens = kMaxNewA;
    sa.sampling = sampling_cfg(42);
    Scheduler::Spec sb;
    sb.prompt = kPromptB;
    sb.max_new_tokens = kMaxNewB;
    sb.sampling = sampling_cfg(123);
    Scheduler::Spec sc;
    sc.prompt = kPromptC;
    sc.max_new_tokens = kMaxNewC;
    sc.sampling = sampling_cfg(7);
    CHECK(sched.admit(sa, &a).ok);
    CHECK(sched.admit(sb, &b).ok);
    CHECK(sched.admit(sc, &c).ok);
    CHECK(sched.run().ok);
    CHECK_EQ(sched.num_live(), 0);
    CHECK_EQ(mgr.num_live_sequences(), 0);

    // ---- BATCHING PROOF ----
    std::fprintf(stderr,
                 "[batching] batch_forward_calls=%d single_forward_calls=%d "
                 "max_batch_size=%d batch_fallback_calls=%d forwarder "
                 "cohort sizes={",
                 sched.batch_forward_calls(), sched.single_forward_calls(),
                 sched.max_batch_size(), sched.batch_fallback_calls());
    for (std::size_t i = 0; i < fwd.batch_sizes().size(); ++i)
      std::fprintf(stderr, "%s%d", i ? "," : "", fwd.batch_sizes()[i]);
    std::fprintf(stderr, "}\n");
    CHECK(sched.batch_forward_calls() > 0);  // batched forwards issued
    CHECK_EQ(sched.max_batch_size(), 3);  // a 3-cohort was batched
    CHECK_EQ(sched.batch_fallback_calls(), 0);  // no serial fallback
    // The cohort sizes are EXACTLY {3, 2}: a B=3 cohort, then a B=2 cohort
    // after A finishes (the batch-shrink case).
    CHECK_EQ(fwd.batch_sizes().size(), 2u);
    CHECK_EQ(fwd.batch_sizes()[0], 3);
    CHECK_EQ(fwd.batch_sizes()[1], 2);
    // A decode cohort of B>=2 costs EXACTLY ONE batch forward (never B
    // single forwards): the two batch forwards advanced 3 + 2 = 5 decode
    // tokens, so single_forward_calls counts only prefill + size-1 cohorts.
    CHECK_EQ(sched.batch_forward_calls(), 2);

    // ---- parity vs references (all EXACT, incl. final state) ----
    const Request* ra = sched.get(a);
    const Request* rb = sched.get(b);
    const Request* rcq = sched.get(c);
    CHECK(ra != nullptr && rb != nullptr && rcq != nullptr);
    rc |= cmp_request(
        "A", *ra, refA,
        fwd.last_forwards(ra->sequence_id,
                          static_cast<int>(ra->generated.size())),
        fwd.state_after(ra->sequence_id), cfg);
    rc |= cmp_request(
        "B", *rb, refB,
        fwd.last_forwards(rb->sequence_id,
                          static_cast<int>(rb->generated.size())),
        fwd.state_after(rb->sequence_id), cfg);
    rc |= cmp_request(
        "C", *rcq, refC,
        fwd.last_forwards(rcq->sequence_id,
                          static_cast<int>(rcq->generated.size())),
        fwd.state_after(rcq->sequence_id), cfg);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_scheduler_batch: FAIL\n");
    return rc;
  }
  std::printf(
      "test_qwen35_scheduler_batch: PASS (B=3 cohort then B=2 shrink: "
      "batch_forward_calls=2, max_batch_size=3, cohort sizes {3,2}; "
      "generated IDs + per-step FULL logits + forward count + finish reason + "
      "FINAL hybrid state all EXACT vs independent references)\n");
  return 0;
}
