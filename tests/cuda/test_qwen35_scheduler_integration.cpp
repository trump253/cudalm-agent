// CUDALM — v0.6 Phase A: scheduler integration hard gate (CUDA, REAL
// Qwen3.5-0.8B-Base checkpoint).
//
// Proves that the v0.6 Phase A request scheduler (deterministic FIFO /
// round-robin control plane over the frozen v0.5 runtime) is
// semantically IDENTICAL to independent execution:
//
//   * three requests, real checkpoint, single model + single stream:
//       A: prompt {1024, 2048, 3072} (3 tok), max_new 3, seed 42
//       B: prompt {15, 16, 17, 18, 19, 20, 21} (7 tok), max_new 2,
//          seed 123
//       C: prompt {1024, 2048} (2 tok), max_new 3, seed 7
//     — admitted DYNAMICALLY: A + B first, C admitted after 2 scheduler
//     iterations (A/B already started; prefill + decode requests
//     coexist while C prefills/decodes alongside them);
//
//   * INDEPENDENT REFERENCES (fresh manager, ONE sequence alone, direct
//     forward_token_with_state loop — NOT the scheduler): per request the
//     generated token IDs, the FULL logits[248320] that produced each
//     generated token, the forward count, and the finish reason;
//
//   * SCHEDULER RUN vs references: for every request A/B/C all four are
//     EXACT (memcmp; the scheduler advances each request by at most one
//     forward_token_with_state per iteration — no batched GPU compute);
//
//   * ISOLATION GATE: B's FULL hybrid state (18x DeltaNet conv/rec +
//     6x logical K/V rows, read through the block table) in the
//     interleaved run is compared, at length 4 (before A finishes — A
//     completes in scheduler step 5) and at length 5 (after A's finish +
//     retire), against the FULL hybrid state of B run ALONE at the same
//     lengths — bit-identical (another request's live state / finish /
//     retirement changes nothing in a still-live request's state);
//
//   * SAMPLING ISOLATION GATE (real logits, seeded sampling):
//       X alone (seed 42, prompt P)  vs
//       Y alone (seed 42, prompt P)  vs
//       X + Y interleaved in ONE scheduler (both seed 42, prompt P)
//     -> identical generated token streams in all three cases: the
//     per-request Sampler (per-request SplitMix64) means admission order
//     and interleaving can NEVER change a request's sampling behavior.
//
// Phase A is scheduler SEMANTICS ONLY: the GPU still runs one sequence
// forward at a time (the frozen v0.5 runtime). No throughput claim.
//
// Self-skips (77) when the checkpoint is absent; in the Phase A evidence
// environment the checkpoint IS present and this test actually runs
// (77 is not a sign-off).

#include "../../tests/common/check.h"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

// ---- request / pool sizing -------------------------------------------------
static const std::vector<int> kPromptA = {1024, 2048, 3072};  // 3 tokens
static const std::vector<int> kPromptB = {15, 16, 17, 18, 19, 20, 21};  // 7
static const std::vector<int> kPromptC = {1024, 2048};  // 2 tokens
static const int kMaxNewA = 3;
static const int kMaxNewB = 2;
static const int kMaxNewC = 3;
static const int kPageTokens = 2;
static const int kPoolPages = 10;  // peak: A(3) + B(3) + C(2) live = 8
static const int kDeltaSlots = 4;  // peak: A + B + C live = 3

// Seeded sampling (a real, non-greedy v0.4 sampling path).
SamplingConfig sampling_cfg(std::uint64_t seed) {
  return SamplingConfig{0.8f, 50, 1.0f, seed};
}

// Forward counts for completed requests (v0.4 semantics):
// N prefill + (m - 1) decode.
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

// Bit-exact comparison (same model weights + same frozen kernels + same
// op order => any non-zero difference is a real bug).
int cmp_exact(const char* name, const std::vector<__nv_bfloat16>& ref,
              const std::vector<__nv_bfloat16>& act) {
  if (ref.size() != act.size()) {
    std::fprintf(stderr, "  %-44s FAIL size %zu != %zu\n", name, ref.size(),
                 act.size());
    return 1;
  }
  const bool ok =
      std::memcmp(ref.data(), act.data(), ref.size() * sizeof(__nv_bfloat16)) ==
      0;
  if (!ok) {
    std::fprintf(stderr, "  %-44s FAIL not bit-identical (n=%zu)\n", name,
                 ref.size());
    return 1;
  }
  std::fprintf(stderr, "  %-44s bit-identical (n=%zu)\n", name, ref.size());
  return 0;
}
int cmp_exact_f32(const char* name, const std::vector<float>& ref,
                  const std::vector<float>& act) {
  if (ref.size() != act.size()) {
    std::fprintf(stderr, "  %-44s FAIL size %zu != %zu\n", name, ref.size(),
                 act.size());
    return 1;
  }
  const bool ok =
      std::memcmp(ref.data(), act.data(), ref.size() * sizeof(float)) == 0;
  if (!ok) {
    std::fprintf(stderr, "  %-44s FAIL not bit-identical (n=%zu)\n", name,
                 ref.size());
    return 1;
  }
  std::fprintf(stderr, "  %-44s bit-identical (n=%zu)\n", name, ref.size());
  return 0;
}

// Full hybrid state of ONE sequence (external side): 18x DeltaNet conv +
// rec + 6x full-attention LOGICAL K/V rows 0..n_tokens-1 read from the
// pool's physical pages THROUGH the host block table.
struct StateSnap {
  std::vector<std::vector<__nv_bfloat16>> conv;  // [L] (linear layers)
  std::vector<std::vector<float>> rec;           // [L]
  std::vector<std::vector<__nv_bfloat16>> kvk;   // [L] rows 0..n-1, all kv
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
  out->conv.resize(L);
  out->rec.resize(L);
  out->kvk.resize(L);
  out->kvv.resize(L);
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
    char name[64];
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

// ---- recording forwarder (logs the FULL logits of every forward) ----------
struct LogEntry {
  SequenceId sid;
  int token;
  std::vector<__nv_bfloat16> logits;  // [vocab]
};

// Wraps the real ModelForwarder; after every successful forward, copies
// the full logits to host (a snapshot of THAT forward's output).
class RecordingForwarder : public SequenceForwarder {
 public:
  explicit RecordingForwarder(Qwen35Model& model)
      : inner_(model), vocab_(model.config().vocab_size) {}

  Status forward_token(int token_id, SequenceId sequence_id,
                       Qwen35StateManager& mgr, cudaStream_t stream) override {
    Status s = inner_.forward_token(token_id, sequence_id, mgr, stream);
    if (s.ok) {
      LogEntry e;
      e.sid = sequence_id;
      e.token = token_id;
      std::vector<__nv_bfloat16> logits;
      s = inner_.logits_to_host(&logits, stream);
      e.logits = std::move(logits);
      log_.push_back(std::move(e));
    }
    return s;
  }

  Status logits_to_host(std::vector<__nv_bfloat16>* out,
                        cudaStream_t stream) const override {
    return inner_.logits_to_host(out, stream);
  }
  int vocab_size() const override { return vocab_; }

  // The last `m` forwards of `sid`, in order — these produced the `m`
  // generated tokens (the last prompt forward + the decode forwards).
  std::vector<LogEntry> last_forwards(SequenceId sid, int m) const {
    std::vector<const LogEntry*> mine;
    for (const LogEntry& e : log_) {
      if (e.sid == sid) {
        mine.push_back(&e);
      }
    }
    if (static_cast<int>(mine.size()) < m) {
      std::fprintf(stderr, "  [recording] %zu < %d forwards for sid %llu\n",
                   mine.size(), m, static_cast<unsigned long long>(sid));
      abort();
    }
    std::vector<LogEntry> out;
    for (std::size_t i = mine.size() - static_cast<std::size_t>(m);
         i < mine.size(); ++i) {
      out.push_back(*mine[i]);  // copy (the test keeps the reference side
                                // live while `log_` may grow)
    }
    return out;
  }

  int num_forwards(SequenceId sid) const {
    int n = 0;
    for (const LogEntry& e : log_) {
      if (e.sid == sid) {
        n++;
      }
    }
    return n;
  }

 private:
  ModelForwarder inner_;
  int vocab_;
  std::vector<LogEntry> log_;
};

// ---- independent reference: ONE sequence alone, DIRECT forwards ------------
// The v0.4 progression semantics implemented WITHOUT the scheduler:
// prefill p0..pN-1 (one forward each, NO sampling), then the prefill-last
// logits sample g0, then forward g_{k-1} -> logits -> sample g_k.
struct RefResult {
  std::vector<int> generated;
  std::vector<std::vector<__nv_bfloat16>> gen_logits;  // [m] full logits
  int forward_count = 0;
  bool eos_hit = false;
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
    if (static_cast<int>(out->generated.size()) >= max_new_tokens) {
      break;
    }
    s = model.forward_token_with_state(g, sid, mgr, stream);
    if (!s.ok) {
      std::fprintf(stderr, "  ref %s decode: %s\n", label, s.message.c_str());
      return 1;
    }
    out->forward_count++;
  }
  CHECK(mgr.retire_sequence(sid).ok);
  std::fprintf(stderr,
               "[reference] %-4s: %d generated, %d forwards, %s\n", label,
               static_cast<int>(out->generated.size()), out->forward_count,
               out->eos_hit ? "eos" : "max_new_tokens");
  return 0;
}

// Compare one request's scheduler run against its independent reference.
int cmp_request(const char* label, const Request& req, const RefResult& ref,
                const std::vector<LogEntry>& gen_entries) {
  int rc = 0;
  std::fprintf(stderr, "  [parity] %s: generated %zu, forwards %d (expected "
                       "%d), reason %d\n",
               label, req.generated.size(), req.forward_count,
               ref.forward_count, static_cast<int>(req.finish_reason));
  // (1) generated token IDs EXACT
  CHECK_EQ(req.generated.size(), ref.generated.size());
  for (std::size_t i = 0; i < req.generated.size(); ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s generated[%zu]", label, i);
    CHECK_EQ(req.generated[i], ref.generated[i]);
    (void)name;
  }
  // (2) per-generated-step FULL logits EXACT (the m forwards that
  //     produced the m generated tokens)
  CHECK_EQ(gen_entries.size(), ref.gen_logits.size());
  for (std::size_t i = 0; i < gen_entries.size(); ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s gen_logits[%zu][vocab]", label, i);
    rc |= cmp_exact(name, ref.gen_logits[i], gen_entries[i].logits);
  }
  // (3) forward count EXACT
  CHECK_EQ(req.forward_count, ref.forward_count);
  // (4) finish reason EXACT
  const FinishReason exp_reason =
      ref.eos_hit ? FinishReason::Eos : FinishReason::MaxNewTokens;
  CHECK_EQ(req.finish_reason, exp_reason);
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
    std::fprintf(stderr, "[SKIP] qwen35 scheduler integration: no "
                         "preconverted model at %s and --no-convert given\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] qwen35 scheduler integration: checkpoint "
                         "absent at %s\n",
                 ckpt.c_str());
    return 77;
  }
  if (!file_exists(out)) {
    CHECK_EQ(run_cmd(py + " " + src + "/tools/convert_qwen35.py" +
                         " --full-model --checkpoint-dir " + ckpt +
                         " --out " + out),
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
               "[scheduler] model loaded: %d layers, vocab %d, page_tokens "
               "%d, pool pages %d, delta slots %d\n",
               model.num_layers(), cfg.vocab_size, kPageTokens, kPoolPages,
               kDeltaSlots);

  int rc = 0;

  // =========================================================================
  // Part 1: independent references (A-alone, B-alone, C-alone — fresh
  // manager per request, DIRECT forwards, NOT the scheduler).
  // =========================================================================
  RefResult refA, refB, refC;
  rc |= run_reference_direct(model, cfg, "A", kPromptA, kMaxNewA, -1,
                             sampling_cfg(42), stream, &refA);
  rc |= run_reference_direct(model, cfg, "B", kPromptB, kMaxNewB, -1,
                             sampling_cfg(123), stream, &refB);
  rc |= run_reference_direct(model, cfg, "C", kPromptC, kMaxNewC, -1,
                             sampling_cfg(7), stream, &refC);
  CHECK_EQ(refA.forward_count, expected_forwards(3, kMaxNewA));  // 5
  CHECK_EQ(refB.forward_count, expected_forwards(7, kMaxNewB));  // 8
  CHECK_EQ(refC.forward_count, expected_forwards(2, kMaxNewC));  // 4

  // =========================================================================
  // Part 1b: B-partial reference — B ALONE, direct forwards p0..p4, with
  // the FULL hybrid state captured at length 4 and length 5 (no sampling
  // involved: the hybrid state is a pure function of the forwarded
  // tokens). These are the "B without A/C" ground truths for the
  // isolation gate.
  // =========================================================================
  StateSnap refB_len4, refB_len5;
  {
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    SequenceId sid = 0;
    CHECK(mgr.create_sequence(&sid).ok);
    for (int i = 0; i < 4; ++i) {
      s = model.forward_token_with_state(kPromptB[static_cast<std::size_t>(i)],
                                         sid, mgr, stream);
      CHECK(s.ok);
    }
    const SequenceState* rec = mgr.lookup(sid);
    CHECK(rec != nullptr);
    CHECK_EQ(capture_state_external(mgr, *rec, cfg, 4, stream, &refB_len4), 0);
    s = model.forward_token_with_state(kPromptB[4], sid, mgr, stream);
    CHECK(s.ok);
    rec = mgr.lookup(sid);
    CHECK(rec != nullptr);
    CHECK_EQ(capture_state_external(mgr, *rec, cfg, 5, stream, &refB_len5), 0);
    CHECK(mgr.retire_sequence(sid).ok);
  }

  // =========================================================================
  // Part 2: scheduler run — A + B admitted, 2 iterations, then C admitted
  // DYNAMICALLY, then run to completion (deterministic FIFO/round-robin;
  // one forward_token_with_state per advance, single stream).
  //
  // ISOLATION GATE: B's FULL hybrid state in the interleaved run is
  // compared at length 4 (BEFORE A finishes — A completes in scheduler
  // step 5) and at length 5 (AFTER A's finish + retire) against the
  // B-alone reference states at the same lengths: A's live state, its
  // finish, and its retirement must never touch B's state.
  // =========================================================================
  RecordingForwarder fwd(model);
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
    CHECK_EQ(a, 1u);
    CHECK_EQ(b, 2u);
    CHECK(sched.step().ok);  // s1: A p0, B p0
    CHECK(sched.step().ok);  // s2: A p1, B p1
    CHECK(sched.admit(sc, &c).ok);  // C admitted AFTER A/B started
    CHECK_EQ(c, 3u);
    CHECK(sched.step().ok);  // s3: A p2(->g0), B p2, C p0
    CHECK(sched.step().ok);  // s4: A g0(->g1), B p3, C p1(->g0)
    // B has 4 tokens; A (4 forwards done) is still live and will finish
    // in the NEXT step. Capture B's full hybrid state (pre-A-finish).
    const SequenceState* rec_b = mgr.lookup(sched.get(b)->sequence_id);
    CHECK(rec_b != nullptr);
    StateSnap b_len4;
    CHECK_EQ(capture_state_external(mgr, *rec_b, cfg, 4, stream, &b_len4), 0);
    rc |= cmp_state("B@len4 interleaved vs B-alone", b_len4, refB_len4, 4, cfg);
    CHECK(sched.step().ok);  // s5: A g1(->g2: A Finished+retired), B p4,
                             //     C g0(->g1)
    CHECK_EQ(sched.get(a)->status, RequestStatus::Finished);
    CHECK_EQ(sched.get(a)->finish_reason, FinishReason::MaxNewTokens);
    // B has 5 tokens and A just finished + retired: capture B again
    // (post-A-retire) and compare with the B-alone reference.
    CHECK_EQ(capture_state_external(mgr, *rec_b, cfg, 5, stream, &b_len4), 0);
    rc |= cmp_state("B@len5 interleaved vs B-alone (post-A-retire)",
                    b_len4, refB_len5, 5, cfg);
    CHECK(sched.run().ok);  // s6: B p5, C g1(->g2: C Finished);
                            //     s7: B p6(->g0); s8: B g0(->g1: B Finished)
    CHECK_EQ(sched.num_live(), 0);
    CHECK_EQ(mgr.num_live_sequences(), 0);

    // ---- parity vs the independent references (all EXACT) -------------
    const Request* ra = sched.get(a);
    const Request* rb = sched.get(b);
    const Request* rcq = sched.get(c);
    CHECK(ra != nullptr && rb != nullptr && rcq != nullptr);
    rc |= cmp_request("A", *ra, refA,
                      fwd.last_forwards(ra->sequence_id,
                                        static_cast<int>(ra->generated.size())));
    rc |= cmp_request("B", *rb, refB,
                      fwd.last_forwards(rb->sequence_id,
                                        static_cast<int>(rb->generated.size())));
    rc |= cmp_request("C", *rcq, refC,
                      fwd.last_forwards(rcq->sequence_id,
                                        static_cast<int>(rcq->generated.size())));
    std::fprintf(stderr,
                 "[ok] A/B/C scheduler run == independent references "
                 "(generated IDs + per-step FULL logits + forward count + "
                 "finish reason, all EXACT); B state unchanged across A's "
                 "finish/retire\n");
  }

  // =========================================================================
  // Part 3: sampling isolation gate (real logits, same prompt + same seed
  // => identical streams; interleaving / admission order must not change
  // the per-request RNG).
  // =========================================================================
  const std::vector<int> kPromptP = {1024, 2048, 3072};
  const int kMaxNewP = 2;
  std::vector<int> x_ids, y_ids, xy_x, xy_y;
  {
    ModelForwarder mfw(model);
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    Scheduler sched(mfw, mgr, stream);
    RequestId x = 0;
    Scheduler::Spec sx;
    sx.prompt = kPromptP;
    sx.max_new_tokens = kMaxNewP;
    sx.sampling = sampling_cfg(42);
    CHECK(sched.admit(sx, &x).ok);
    CHECK(sched.run().ok);
    x_ids = sched.get(x)->generated;
  }
  {
    ModelForwarder mfw(model);
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    Scheduler sched(mfw, mgr, stream);
    RequestId y = 0;
    Scheduler::Spec sy;
    sy.prompt = kPromptP;
    sy.max_new_tokens = kMaxNewP;
    sy.sampling = sampling_cfg(42);
    CHECK(sched.admit(sy, &y).ok);
    CHECK(sched.run().ok);
    y_ids = sched.get(y)->generated;
  }
  {
    ModelForwarder mfw(model);
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    Scheduler sched(mfw, mgr, stream);
    RequestId x = 0, y = 0;
    Scheduler::Spec sx;
    sx.prompt = kPromptP;
    sx.max_new_tokens = kMaxNewP;
    sx.sampling = sampling_cfg(42);
    Scheduler::Spec sy;
    sy.prompt = kPromptP;
    sy.max_new_tokens = kMaxNewP;
    sy.sampling = sampling_cfg(42);
    CHECK(sched.admit(sx, &x).ok);
    CHECK(sched.admit(sy, &y).ok);
    CHECK(sched.run().ok);
    xy_x = sched.get(x)->generated;
    xy_y = sched.get(y)->generated;
  }
  CHECK(x_ids == y_ids);      // same prompt + same seed => same stream
  CHECK(xy_x == x_ids);       // interleaved with a same-seed twin: unchanged
  CHECK(xy_y == x_ids);       // the twin gets the same stream too
  std::fprintf(stderr,
               "[ok] sampling isolation: same prompt+seed => identical "
               "streams (X alone == Y alone == X+Y interleaved); per-request "
               "Sampler, admission order never changes the RNG\n");

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_scheduler_integration: FAIL\n");
    return rc;
  }
  std::printf(
      "test_qwen35_scheduler_integration: PASS (A/B/C dynamic-admission "
      "scheduler run == independent references: generated IDs + per-step "
      "FULL logits + forward count + finish reason EXACT; B state "
      "unchanged across A's finish/retire; sampling RNG isolation)\n");
  return 0;
}
