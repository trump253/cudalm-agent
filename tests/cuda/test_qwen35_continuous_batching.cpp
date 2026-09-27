// CUDALM — v0.6 Phase C: DYNAMIC CONTINUOUS-BATCHING real-checkpoint hard
// gate (CUDA, REAL Qwen/Qwen3.5-0.8B-Base).
//
// Proves the real Qwen3.5-0.8B runtime keeps running under DYNAMIC request
// arrival / completion / cancellation / batch grow-shrink, staying EXACTLY
// consistent with each request's independent execution:
//
//   * FOUR requests, distinct prompt length / max_new / seed:
//       A: prompt 2 tok, max_new 3, GREEDY      (finishes first)
//       B: prompt 5 tok, max_new 6, seed 100    (CANCELLED mid-flight)
//       C: prompt 3 tok, max_new 5, seed 200    (LATE admission)
//       D: prompt 4 tok, max_new 3, seed 300    (LATEST; resource reuse)
//     (>= 1 greedy + 3 seeded; distinct prompt lengths 2/5/3/4.)
//
//   * TRULY DYNAMIC arrival (not all admitted up front):
//       t0: admit A, B.  -> 2 steps.  t2: admit C (A/B running).  A finishes
//       -> t4: admit D (after a departure; reuses A's freed resources).
//     C and D join while existing requests are RUNNING; a late request never
//     enters a pre-admission snapshot (snapshot semantics); RequestIds are
//     monotonic and never reused (A=1, B=2, C=3, D=4).
//
//   * TRUE BATCH GROW / SHRINK (naturally, no scheduler reordering): the
//     decode-cohort-size trace is [1,1,2,2,2,3,1] — it GROWS 1->2 and 2->3
//     and SHRINKS 3->1. batch_forward_calls>0, max_batch_size==3, so the
//     TRUE batched GPU decode path (forward_batch_with_state) is actually
//     exercised (prefill stays serial; the decode cohort is batched).
//
//   * INDEPENDENT-REFERENCE PARITY (fresh manager, DIRECT
//     forward_token_with_state loop, per-request Sampler — NOT the
//     scheduler): for each request the generated token IDs, the FULL
//     logits[248320] that produced each generated token, the forward count,
//     and the finish reason are EXACT. For the CANCELLED B, the prefix
//     produced before cancel (5 tokens) is EXACT vs B's independent prefix.
//     A request's output depends only on its own prompt/state/RNG — never on
//     when others join or leave.
//
//   * MID-FLIGHT CANCELLATION: B is cancelled while live (prefill complete,
//     generated >= 1). B -> Cancelled + its sequence retired EXACTLY ONCE;
//     B is never executed again (no further B forwards; B absent from every
//     later cohort); B's pre-cancel prefix is EXACT vs the reference prefix;
//     cancelling B changes NOTHING in A/C/D (their full parity holds).
//
//   * RESOURCE REUSE AFTER DEPARTURE: D is admitted after A finishes (and B
//     is cancelled later). D reuses A's freed DELTA SLOT (the pool is LIFO,
//     so the most recently released slot is reacquired first — observed
//     explicitly: D.delta_slot == A.delta_slot); A's stale SequenceId stays
//     invalid (lookup nullptr); D's FINAL hybrid state (18x Delta conv/rec +
//     6x logical K/V) is BIT-IDENTICAL to an independent fresh-D reference
//     (fresh-zero logical state, no contamination from A's reused slot);
//     live requests (B/C) are unchanged.
//
//   * SERVING METRICS: the new SchedulerStats (observability only) is
//     checked: admitted/finished/cancelled/failed counts, single vs batch
//     forward calls, model-traversal vs logical-token distinction,
//     max/avg decode batch size, and the batch-size / decode-cohort traces.
//
// Self-skips (77) when the checkpoint is absent; in the Phase C evidence
// environment the checkpoint IS present and this test actually runs (77 is
// not a sign-off). No throughput claim (correctness-first; prefill stays
// serial, single stream).

#include "../../tests/common/check.h"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---- workload (distinct prompt length / max_new / seed) --------------------
static const std::vector<int> kPromptA = {1024, 2048};          // 2 tok
static const std::vector<int> kPromptB = {15, 16, 17, 18, 19};  // 5 tok
static const std::vector<int> kPromptC = {3072, 4096, 5120};    // 3 tok
static const std::vector<int> kPromptD = {6, 7, 8, 9};          // 4 tok
static const int kMaxNewA = 3;  // greedy
static const int kMaxNewB = 6;  // seed 100
static const int kMaxNewC = 5;  // seed 200
static const int kMaxNewD = 3;  // seed 300

static const int kPageTokens = 2;
static const int kPoolPages = 16;   // peak live B(5)+C(4)+D(3) = 12 pages
static const int kDeltaSlots = 4;   // peak live = 3 sequences

SamplingConfig seeded(std::uint64_t seed) {
  return SamplingConfig{0.8f, 50, 1.0f, seed};
}

// N prefill + (m - 1) decode forwards for m generated tokens.
int expected_forwards(int n, int m) { return n + m - 1; }

// ---- device -> host helpers -------------------------------------------------
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
    std::fprintf(stderr, "  %-46s FAIL size %zu != %zu\n", name, ref.size(),
                 act.size());
    return 1;
  }
  const bool ok =
      std::memcmp(ref.data(), act.data(), ref.size() * sizeof(__nv_bfloat16)) ==
      0;
  if (!ok) {
    std::fprintf(stderr, "  %-46s FAIL not bit-identical (n=%zu)\n", name,
                 ref.size());
    return 1;
  }
  std::fprintf(stderr, "  %-46s bit-identical (n=%zu)\n", name, ref.size());
  return 0;
}
int cmp_exact_f32(const char* name, const std::vector<float>& ref,
                  const std::vector<float>& act) {
  if (ref.size() != act.size()) {
    std::fprintf(stderr, "  %-46s FAIL size %zu != %zu\n", name, ref.size(),
                 act.size());
    return 1;
  }
  const bool ok =
      std::memcmp(ref.data(), act.data(), ref.size() * sizeof(float)) == 0;
  if (!ok) {
    std::fprintf(stderr, "  %-46s FAIL not bit-identical (n=%zu)\n", name,
                 ref.size());
    return 1;
  }
  std::fprintf(stderr, "  %-46s bit-identical (n=%zu)\n", name, ref.size());
  return 0;
}

// ---- full hybrid state capture (reuse of the Phase A/B infrastructure) ------
struct StateSnap {
  std::vector<std::vector<__nv_bfloat16>> conv;  // [L] linear layers
  std::vector<std::vector<float>> rec;           // [L]
  std::vector<std::vector<__nv_bfloat16>> kvk;   // [L] rows 0..n-1
  std::vector<std::vector<__nv_bfloat16>> kvv;   // [L]
};
int capture_state_external(const Qwen35StateManager& mgr,
                           const SequenceState& rec, const Qwen35Config& cfg,
                           int n_tokens, cudaStream_t stream, StateSnap* out) {
  const int L = cfg.num_hidden_layers;
  const int nkv = cfg.n_kv_heads;
  const int hd = cfg.head_dim;
  const int pt = mgr.page_tokens();
  const int slot = rec.delta_slot;
  out->conv.assign(L, {});
  out->rec.assign(L, {});
  out->kvk.assign(L, {});
  out->kvv.assign(L, {});
  for (int i = 0; i < L; ++i) {
    if (cfg.is_linear_attention(i)) {
      const int ord = mgr.delta_pool().linear_layer_ordinal(i);
      const std::size_t cn =
          static_cast<std::size_t>(cfg.linear_conv_dim()) *
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
          std::fprintf(stderr, "  [state capture] block %d unallocated\n",
                       t / pt);
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
int cmp_state(const char* label, const StateSnap& a, const StateSnap& b,
              int n_tokens, const Qwen35Config& cfg) {
  int rc = 0;
  const int L = cfg.num_hidden_layers;
  for (int i = 0; i < L; ++i) {
    char name[96];
    if (cfg.is_linear_attention(i)) {
      std::snprintf(name, sizeof(name), "%s L%d delta.conv", label, i);
      rc |= cmp_exact(name, a.conv[static_cast<std::size_t>(i)],
                      b.conv[static_cast<std::size_t>(i)]);
      std::snprintf(name, sizeof(name), "%s L%d delta.rec", label, i);
      rc |= cmp_exact_f32(name, a.rec[static_cast<std::size_t>(i)],
                          b.rec[static_cast<std::size_t>(i)]);
    } else {
      std::snprintf(name, sizeof(name), "%s L%d kv.K[0..%d]", label, i,
                    n_tokens - 1);
      rc |= cmp_exact(name, a.kvk[static_cast<std::size_t>(i)],
                      b.kvk[static_cast<std::size_t>(i)]);
      std::snprintf(name, sizeof(name), "%s L%d kv.V[0..%d]", label, i,
                    n_tokens - 1);
      rc |= cmp_exact(name, a.kvv[static_cast<std::size_t>(i)],
                      b.kvv[static_cast<std::size_t>(i)]);
    }
  }
  return rc;
}

// ---- recording forwarder (single AND per-row batch logits) ------------------
struct LogEntry {
  SequenceId sid;
  int token;
  std::vector<__nv_bfloat16> logits;  // [vocab]
};
// Records the FULL logits of every forward that PRODUCES a generated token:
// the single path (after logits_to_host) and every ROW of a committed batch
// (after logits_batch_to_host, associated with the cohort's sids). The
// scheduler only fetches logits when it samples, so per-sid entry count ==
// that request's generated-token count.
class BatchRecordingForwarder : public SequenceForwarder {
 public:
  explicit BatchRecordingForwarder(Qwen35Model& model)
      : inner_(model), vocab_(model.config().vocab_size) {}

  Status forward_token(int token_id, SequenceId sequence_id,
                       Qwen35StateManager& mgr, cudaStream_t stream) override {
    Status s = inner_.forward_token(token_id, sequence_id, mgr, stream);
    if (s.ok) {
      last_sid_ = sequence_id;
      last_token_ = token_id;
    }
    return s;
  }
  Status logits_to_host(std::vector<__nv_bfloat16>* out,
                        cudaStream_t stream) const override {
    Status s = inner_.logits_to_host(out, stream);
    if (s.ok) {
      LogEntry e;
      e.sid = last_sid_;
      e.token = last_token_;
      e.logits = *out;
      log_.push_back(std::move(e));
    }
    return s;
  }
  int vocab_size() const override { return vocab_; }

  bool supports_batch() const override { return inner_.supports_batch(); }
  Status forward_batch(const int* token_ids, const SequenceId* sids, int B,
                       Qwen35StateManager& mgr, cudaStream_t stream) override {
    Status s = inner_.forward_batch(token_ids, sids, B, mgr, stream);
    if (s.ok) {
      batch_sids_.assign(sids, sids + B);
      batch_tokens_.assign(token_ids, token_ids + B);
    }
    return s;
  }
  Status logits_batch_to_host(std::vector<__nv_bfloat16>* out, int B,
                              cudaStream_t stream) const override {
    Status s = inner_.logits_batch_to_host(out, B, stream);
    if (s.ok) {
      const std::size_t V = static_cast<std::size_t>(vocab_);
      for (int b = 0; b < B; ++b) {
        LogEntry e;
        e.sid = batch_sids_[static_cast<std::size_t>(b)];
        e.token = batch_tokens_[static_cast<std::size_t>(b)];
        e.logits.assign(out->data() + static_cast<std::size_t>(b) * V,
                        out->data() + static_cast<std::size_t>(b + 1) * V);
        log_.push_back(std::move(e));
      }
    }
    return s;
  }

  // The last `m` recorded forwards of `sid` (the `m` forwards that produced
  // its `m` generated tokens), in order.
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

 private:
  ModelForwarder inner_;
  int vocab_;
  mutable std::vector<LogEntry> log_;  // mutable: recorded from the const
                                       // logits fetchers (observer side-channel)
  mutable SequenceId last_sid_ = 0;
  mutable int last_token_ = -1;
  mutable std::vector<SequenceId> batch_sids_;
  mutable std::vector<int> batch_tokens_;
};

// ---- independent reference: ONE sequence alone, DIRECT forwards -------------
struct RefResult {
  std::vector<int> generated;
  std::vector<std::vector<__nv_bfloat16>> gen_logits;  // [m] full logits
  int forward_count = 0;
  bool eos_hit = false;
  StateSnap state_at;   // optional: captured when the length reached state_len
  int state_len = -1;   // -1 = none captured
};
// Runs `prompt` alone on a fresh manager (direct forward_token_with_state,
// per-request Sampler). If `capture_at_len >= 0`, the full hybrid state is
// captured the moment the sequence length reaches that value (a mid-run
// ground truth for the no-contamination / fresh-zero checks).
int run_reference_direct(Qwen35Model& model, const Qwen35Config& cfg,
                         const char* label, const std::vector<int>& prompt,
                         int max_new_tokens, int eos_token_id,
                         const SamplingConfig& sampling, int capture_at_len,
                         cudaStream_t stream, RefResult* out) {
  Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  SequenceId sid = 0;
  Status s = mgr.create_sequence(&sid);
  CHECK(s.ok);
  Sampler sampler(sampling);
  const int N = static_cast<int>(prompt.size());
  bool captured = false;
  auto maybe_capture = [&]() -> int {
    if (captured || capture_at_len < 0) return 0;
    const SequenceState* rec = mgr.lookup(sid);
    if (rec == nullptr || rec->length != capture_at_len) return 0;
    CHECK_EQ(capture_state_external(mgr, *rec, cfg, capture_at_len, stream,
                                    &out->state_at),
             0);
    out->state_len = capture_at_len;
    captured = true;
    return 0;
  };
  for (int i = 0; i < N; ++i) {
    s = model.forward_token_with_state(prompt[i], sid, mgr, stream);
    if (!s.ok) {
      std::fprintf(stderr, "  ref %s forward %d: %s\n", label, i,
                   s.message.c_str());
      return 1;
    }
    out->forward_count++;
    if (maybe_capture()) return 1;
  }
  while (static_cast<int>(out->generated.size()) < max_new_tokens) {
    std::vector<__nv_bfloat16> logits =
        d2h_bf16(model.logits(), cfg.vocab_size, stream);
    out->gen_logits.push_back(std::move(logits));
    const int g =
        sampler.sample(out->gen_logits.back().data(), cfg.vocab_size);
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
    if (maybe_capture()) return 1;
  }
  CHECK(mgr.retire_sequence(sid).ok);
  std::fprintf(stderr,
               "[reference] %-4s: %d generated, %d forwards, %s\n", label,
               static_cast<int>(out->generated.size()), out->forward_count,
               out->eos_hit ? "eos" : "max_new_tokens");
  return 0;
}

// Compare a NORMAL (completed) request vs its full reference.
int cmp_request_full(const char* label, const Request& req,
                     const RefResult& ref, const std::vector<LogEntry>& gen) {
  int rc = 0;
  CHECK_EQ(req.generated.size(), ref.generated.size());
  for (std::size_t i = 0; i < req.generated.size(); ++i)
    CHECK_EQ(req.generated[i], ref.generated[i]);
  CHECK_EQ(gen.size(), ref.gen_logits.size());
  for (std::size_t i = 0; i < gen.size(); ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s gen_logits[%zu][vocab]", label, i);
    rc |= cmp_exact(name, ref.gen_logits[i], gen[i].logits);
  }
  CHECK_EQ(req.forward_count, ref.forward_count);
  const FinishReason exp =
      ref.eos_hit ? FinishReason::Eos : FinishReason::MaxNewTokens;
  CHECK_EQ(req.finish_reason, exp);
  return rc;
}
// Compare a CANCELLED request's PREFIX (m tokens) vs the reference prefix.
int cmp_request_prefix(const char* label, const Request& req,
                       const RefResult& ref, const std::vector<LogEntry>& gen) {
  int rc = 0;
  const int m = static_cast<int>(req.generated.size());
  CHECK(m >= 1);
  CHECK(m < static_cast<int>(ref.generated.size()));  // cancelled early
  for (int i = 0; i < m; ++i) CHECK_EQ(req.generated[i], ref.generated[i]);
  CHECK_EQ(gen.size(), static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s gen_logits[%d][vocab]", label, i);
    rc |= cmp_exact(name, ref.gen_logits[static_cast<std::size_t>(i)],
                    gen[static_cast<std::size_t>(i)].logits);
  }
  // forward_count at the m-token prefix = N + (m - 1).
  const int n = static_cast<int>(req.prompt.size());
  CHECK_EQ(req.forward_count, expected_forwards(n, m));
  CHECK_EQ(req.finish_reason, FinishReason::Cancelled);
  return rc;
}

bool a_delta_slot_read(const Qwen35StateManager& mgr, SequenceId sid,
                       int* out_slot) {
  const SequenceState* rec = mgr.lookup(sid);
  if (rec == nullptr) return false;
  *out_slot = rec->delta_slot;
  return true;
}

std::string trace_str(const std::vector<int>& v) {
  std::string s;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(v[i]);
  }
  return s;
}
bool has_strict_increase(const std::vector<int>& v) {
  for (std::size_t i = 1; i < v.size(); ++i)
    if (v[i] > v[i - 1]) return true;
  return false;
}
bool has_strict_decrease(const std::vector<int>& v) {
  for (std::size_t i = 1; i < v.size(); ++i)
    if (v[i] < v[i - 1]) return true;
  return false;
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
    std::fprintf(stderr,
                 "[SKIP] continuous batching: no preconverted model\n");
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] continuous batching: checkpoint absent\n");
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
  CHECK(model.loaded());
  const Qwen35Config& cfg = model.config();
  std::fprintf(stderr,
               "[continuous] model loaded: %d layers, vocab %d, page_tokens "
               "%d, pool pages %d, delta slots %d\n",
               model.num_layers(), cfg.vocab_size, kPageTokens, kPoolPages,
               kDeltaSlots);
  int rc = 0;

  // =========================================================================
  // Part 1: INDEPENDENT references (A/B/C/D ALONE, fresh manager, direct
  // forwards, per-request Sampler — NOT the scheduler). D's final hybrid
  // state is captured (the no-contamination ground truth for Part 3).
  // =========================================================================
  RefResult refA, refB, refC, refD;
  // D's reference also captures its hybrid state at length 5 (4 prefill +
  // 1 decode) — the ground truth for the "fresh-zero on a reused slot" check
  // (the dynamic D is live and inspectable at exactly length 5, after step 9).
  rc |= run_reference_direct(model, cfg, "A", kPromptA, kMaxNewA, -1,
                             SamplingConfig::greedy(), -1, stream, &refA);
  rc |= run_reference_direct(model, cfg, "B", kPromptB, kMaxNewB, -1,
                             seeded(100), -1, stream, &refB);
  rc |= run_reference_direct(model, cfg, "C", kPromptC, kMaxNewC, -1,
                             seeded(200), -1, stream, &refC);
  rc |= run_reference_direct(model, cfg, "D", kPromptD, kMaxNewD, -1,
                             seeded(300), /*capture_at_len=*/5, stream,
                             &refD);
  CHECK_EQ(refD.state_len, 5);
  CHECK_EQ(refA.forward_count, expected_forwards(2, kMaxNewA));  // 4
  CHECK_EQ(refB.forward_count, expected_forwards(5, kMaxNewB));  // 10
  CHECK_EQ(refC.forward_count, expected_forwards(3, kMaxNewC));  // 7
  CHECK_EQ(refD.forward_count, expected_forwards(4, kMaxNewD));  // 6

  // =========================================================================
  // Part 2: DYNAMIC scheduler run (truly staggered arrival; a real
  // mid-flight cancellation; a late resource-reusing admission).
  // =========================================================================
  BatchRecordingForwarder fwd(model);
  SequenceId a_seq = 0, b_seq = 0, c_seq = 0, d_seq = 0;
  int a_delta_slot = -1;
  {
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    Scheduler sched(fwd, mgr, stream);

    Scheduler::Spec sa;
    sa.prompt = kPromptA;
    sa.max_new_tokens = kMaxNewA;
    sa.sampling = SamplingConfig::greedy();
    Scheduler::Spec sb;
    sb.prompt = kPromptB;
    sb.max_new_tokens = kMaxNewB;
    sb.sampling = seeded(100);
    RequestId a = 0, b = 0, c = 0, d = 0;

    // t0: admit A, B (the only initial arrivals).
    CHECK(sched.admit(sa, &a).ok);
    CHECK(sched.admit(sb, &b).ok);
    CHECK_EQ(a, 1u);
    CHECK_EQ(b, 2u);
    a_seq = sched.get(a)->sequence_id;
    b_seq = sched.get(b)->sequence_id;

    CHECK(sched.step().ok);  // s1: A p0, B p0
    CHECK(sched.step().ok);  // s2: A p1(->g0: A decode-ready), B p1

    // t2: admit C LATE (A/B running). C's first snapshot is the NEXT step;
    // it never entered s1/s2.
    Scheduler::Spec sc;
    sc.prompt = kPromptC;
    sc.max_new_tokens = kMaxNewC;
    sc.sampling = seeded(200);
    CHECK(sched.admit(sc, &c).ok);
    CHECK_EQ(c, 3u);  // monotonic, never reused
    c_seq = sched.get(c)->sequence_id;

    CHECK(sched.step().ok);  // s3: A decode(g0->g1) [cohort 1], B p2, C p0

    // A is still alive after s3 (it finishes DURING s4): capture A's physical
    // delta slot NOW, before its retirement releases it (the reuse ground
    // truth). A was admitted first -> ascending first-come handout -> slot 0.
    CHECK(a_delta_slot_read(mgr, a_seq, &a_delta_slot));
    CHECK_EQ(a_delta_slot, 0);

    CHECK(sched.step().ok);  // s4: A decode(g1->g2: A FINISHED+retired), B p3,
                             //     C p1
    CHECK(mgr.lookup(a_seq) == nullptr);  // A retired (stale id invalid)

    // t4: admit D LATEST, after A's departure (resource reuse candidate).
    Scheduler::Spec sd;
    sd.prompt = kPromptD;
    sd.max_new_tokens = kMaxNewD;
    sd.sampling = seeded(300);
    CHECK(sched.admit(sd, &d).ok);
    CHECK_EQ(d, 4u);
    d_seq = sched.get(d)->sequence_id;
    CHECK(d_seq != a_seq);  // fresh SequenceId (ids never reused)

    // D reuses A's freed DELTA SLOT (pool is LIFO: the most recently released
    // slot is reacquired first; A was the only departure so far, so its slot
    // is at the top of the free list). Observed directly:
    CHECK_EQ(mgr.lookup(d_seq)->delta_slot, a_delta_slot);

    CHECK(sched.step().ok);  // s5: B p4(->g0: B decode-ready), C p2(->g0: C
                             //     decode-ready), D p0
    CHECK(sched.step().ok);  // s6: cohort [B,C] B=2 BATCH, D p1
    CHECK(sched.step().ok);  // s7: cohort [B,C] B=2 BATCH, D p2
    CHECK(sched.step().ok);  // s8: cohort [B,C] B=2 BATCH, D p3(->g0: D
                             //     decode-ready)
    CHECK(sched.step().ok);  // s9: cohort [B,C,D] B=3 BATCH: C finishes at
                             //     max_new; B reaches 5 generated

    // Capture the DYNAMIC D's hybrid state at length 5 (D is live here; it
    // finishes during s10) — the no-contamination ground truth on the REUSED
    // delta slot (a_delta_slot).
    StateSnap d_dyn_state;
    {
      const SequenceState* rec = mgr.lookup(d_seq);
      CHECK(rec != nullptr);
      CHECK_EQ(rec->length, 5);
      CHECK_EQ(capture_state_external(mgr, *rec, cfg, 5, stream, &d_dyn_state),
               0);
    }

    // MID-FLIGHT CANCELLATION of B (live: prefill complete, 5 generated).
    CHECK_EQ(static_cast<int>(sched.get(b)->generated.size()), 5);
    CHECK_EQ(sched.get(b)->status, RequestStatus::Running);
    const int b_forwards_before_cancel = fwd.num_forwards(b_seq);
    CHECK_EQ(sched.cancel(b).ok, true);
    CHECK_EQ(sched.get(b)->status, RequestStatus::Cancelled);
    CHECK_EQ(sched.get(b)->finish_reason, FinishReason::Cancelled);
    CHECK(mgr.lookup(b_seq) == nullptr);  // B's sequence retired exactly once
    CHECK_EQ(mgr.num_live_sequences(), 1);  // only D remains live

    CHECK(sched.step().ok);  // s10: D decode(g1->g2: D FINISHED). B is
                             // terminal -> never advanced again.
    CHECK_EQ(fwd.num_forwards(b_seq), b_forwards_before_cancel);  // B no-ops
    CHECK_EQ(sched.get(d)->status, RequestStatus::Finished);
    CHECK_EQ(sched.num_live(), 0);
    CHECK_EQ(mgr.num_live_sequences(), 0);

    // ---- true batched decode actually observed + grow/shrink -------------
    const SchedulerStats st = sched.stats();
    std::fprintf(stderr,
                 "[batching] batch_forward_calls=%d single_forward_calls=%d "
                 "max_batch_size=%d batch_fallback_calls=%d\n",
                 st.batch_forward_calls, st.single_forward_calls,
                 st.max_batch_size, sched.batch_fallback_calls());
    std::fprintf(stderr, "[batching] batch_size_trace={%s}\n",
                 trace_str(st.batch_size_trace).c_str());
    std::fprintf(stderr, "[batching] decode_cohort_trace={%s}\n",
                 trace_str(st.decode_cohort_trace).c_str());
    CHECK(st.batch_forward_calls > 0);  // true batched decode was used
    CHECK(st.max_batch_size >= 2);
    CHECK_EQ(st.max_batch_size, 3);  // a 3-wide cohort was batched
    CHECK_EQ(sched.batch_fallback_calls(), 0);  // no fallback (happy path)
    // grow AND shrink in the decode-cohort trace
    CHECK(has_strict_increase(st.decode_cohort_trace));  // grow
    CHECK(has_strict_decrease(st.decode_cohort_trace));  // shrink
    // the expected natural trace (A singles, [B,C]x3, [B,C,D], D single)
    const std::vector<int> kExpectedTrace = {1, 1, 2, 2, 2, 3, 1};
    CHECK(st.decode_cohort_trace == kExpectedTrace);
    const std::vector<int> kExpectedBatch = {2, 2, 2, 3};
    CHECK(st.batch_size_trace == kExpectedBatch);

    // ---- serving metrics sanity (lifecycle + logical != traversal) --------
    std::fprintf(stderr,
                 "[metrics] admitted=%d live=%d finished=%d cancelled=%d "
                 "failed=%d | single=%d batch=%d traversals=%d logical=%d "
                 "batched_tokens=%d max_batch=%d avg_batch=%.3f\n",
                 st.requests_admitted, st.requests_live, st.requests_finished,
                 st.requests_cancelled, st.requests_failed,
                 st.single_forward_calls, st.batch_forward_calls,
                 st.model_traversal_calls, st.logical_token_forwards,
                 st.batched_sequence_tokens, st.max_batch_size,
                 st.avg_committed_decode_batch_size);
    CHECK_EQ(st.requests_admitted, 4);
    CHECK_EQ(st.requests_finished, 3);   // A, C, D
    CHECK_EQ(st.requests_cancelled, 1);  // B
    CHECK_EQ(st.requests_failed, 0);
    CHECK_EQ(st.requests_live, 0);
    // model traversal = single + batch (a committed batch of B is ONE
    // traversal); logical token-forwards = single*1 + sum(batch B).
    CHECK_EQ(st.model_traversal_calls,
             st.single_forward_calls + st.batch_forward_calls);
    CHECK_EQ(st.logical_token_forwards,
             st.single_forward_calls + st.batched_sequence_tokens);
    CHECK_EQ(st.batched_sequence_tokens, 2 + 2 + 2 + 3);  // batch trace sum
    // A batch(B=3) is 3 logical token-forwards but 1 traversal, so the
    // logical count strictly exceeds the traversal count.
    CHECK(st.logical_token_forwards > st.model_traversal_calls);
    CHECK(st.avg_committed_decode_batch_size > 2.0);
    CHECK(st.avg_committed_decode_batch_size < 3.0);

    // ---- independent-reference PARITY (all EXACT) -------------------------
    const Request* ra = sched.get(a);
    const Request* rb = sched.get(b);
    const Request* rcq = sched.get(c);
    const Request* rd = sched.get(d);
    CHECK(ra && rb && rcq && rd);
    std::fprintf(stderr, "[parity] A (greedy, finished)\n");
    rc |= cmp_request_full(
        "A", *ra, refA,
        fwd.last_forwards(a_seq, static_cast<int>(ra->generated.size())));
    std::fprintf(stderr, "[parity] B (seed 100, CANCELLED mid-flight)\n");
    rc |= cmp_request_prefix(
        "B", *rb, refB,
        fwd.last_forwards(b_seq, static_cast<int>(rb->generated.size())));
    std::fprintf(stderr, "[parity] C (seed 200, late admission, finished)\n");
    rc |= cmp_request_full(
        "C", *rcq, refC,
        fwd.last_forwards(c_seq, static_cast<int>(rcq->generated.size())));
    std::fprintf(stderr, "[parity] D (seed 300, latest, resource reuse)\n");
    rc |= cmp_request_full(
        "D", *rd, refD,
        fwd.last_forwards(d_seq, static_cast<int>(rd->generated.size())));

    // ---- D's hybrid state on the REUSED slot == fresh-D reference ---------
    // The dynamic D ran on A's REUSED delta slot (a_delta_slot). If A's stale
    // state had leaked, D's conv/rec would differ. Compare the dynamic D's
    // state at length 5 (captured live above) against the independent fresh-D
    // reference's state at length 5: must be BIT-IDENTICAL (fresh-zero
    // logical state, no cross-request contamination).
    std::fprintf(stderr, "[parity] D@len5 (reused slot %d) vs fresh-D ref\n",
                 a_delta_slot);
    rc |= cmp_state("D@len5 (reused slot) vs fresh-D reference", d_dyn_state,
                    refD.state_at, 5, cfg);
    std::fprintf(stderr,
                 "[ok] A/B/C/D dynamic run == independent references (B "
                 "prefix-exact vs cancelled); true batched decode "
                 "grow/shrink; D reuses A's freed delta slot with fresh-zero "
                 "final state\n");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_continuous_batching: FAIL\n");
    return rc;
  }
  std::printf(
      "test_qwen35_continuous_batching: PASS (A/B/C/D dynamic arrival "
      "completes/cancels/reuses under true batched decode: decode-cohort "
      "trace {1,1,2,2,2,3,1} grows and shrinks, max_batch_size=3, "
      "batch_forward_calls>0; A/C/D generated IDs + per-step FULL logits + "
      "forward count + finish reason EXACT vs independent references, "
      "cancelled B prefix EXACT; D reuses A's freed delta slot (fresh-zero "
      "final state); serving metrics sane)\n");
  return 0;
}
