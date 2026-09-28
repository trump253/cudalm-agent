// CUDALM — v0.8 Phase C: scheduler + session integration HARD GATE
// (CUDA, REAL Qwen3.5-0.8B-Base checkpoint) — the v0.8 Phase C sign-off
// test.
//
// Proves that SESSION-BOUND turn execution through the v0.6 scheduler is
// numerically IDENTICAL to independent continuous execution, with true
// BATCHED decode cohorts across two live sessions:
//
//   * Session A (greedy) and Session B (seeded sampling), TWO turns each:
//       A t1: input {1024, 2048, 3073} (3 tok), max_new 3, greedy
//       B t1: input {100, 200}          (2 tok), max_new 3, seed 42
//       A t2: input {15, 16}            (2 tok), max_new 2, greedy
//       B t2: input {7, 8}              (2 tok), max_new 2, seed 123
//     — the t1 pair is admitted and run to completion (the sessions STAY
//     LIVE afterwards), the t2 pair is admitted (the sessions freed by
//     their terminal t1 requests) and run to completion;
//
//   * INDEPENDENT REFERENCES (fresh manager, ONE sequence alone, direct
//     forward_token_with_state loop, SAME per-turn sampler configs, no
//     turn boundary — the one-shot continuous execution of the exact same
//     token stream);
//
//   * BIT-IDENTICAL (memcmp) for every session: every committed forward's
//     FULL logits [vocab] in order (A: 6 + 4 = 10, B: 5 + 4 = 9 —
//     including the batched decode rows, recorded from forward_batch +
//     logits_batch_to_host), the generated token IDs per turn, the
//     logical length at the turn boundary (A 6 / B 5) and at the end
//     (A 10 / B 9), and the full hybrid state (18x DeltaNet conv+rec +
//     6x full-attention logical K/V rows through the block table) at the
//     turn boundary AND at the end — proving:
//       - the COMMIT CONTRACT through the scheduler (sampled != committed;
//         every generated token, stop-triggering included, is in the
//         session state when the turn ends);
//       - TURN 2 DOES NOT REPLAY TURN 1 (the per-step logits of the t2
//         forwards match the reference's steps 6..9 / 5..8 — a replay
//         would restart the positions and break the RoPE-dependent logits);
//       - A/B INTERLEAVING CAUSES NO CONTAMINATION (each session matches
//         its OWN solo reference while sharing batched model traversals);
//
//   * BATCH EVIDENCE: the two live sessions' decodes form TRUE batched
//     cohorts — batch_forward_calls == 4 (two per run), max_batch_size
//     == 2, decode_cohort_trace == [1, 2, 2, 1, 2, 2], and each batch
//     row's recorded logits are bit-identical to that session's solo
//     reference row (row parity on session-bound sequences);
//
//   * SESSION PRESERVATION AFTER A TURN TERMINAL: after the t1 run the
//     requests are terminal but both sessions (and their bound
//     sequences) are LIVE with their committed state intact (length,
//     pages, dirty Delta slot — the state == the reference-at-boundary
//     gate above pins the contents);
//
//   * ADMISSION ZERO-MUTATION on the real model: a BUSY session (a
//     second live turn while t1 is live) and CONTEXT OVERFLOW (session C
//     at max_seq_len - 2 + input 2 + max_new 1 > max_seq_len) both fail
//     loud with no half-request / no consumed RequestId / no session
//     state change; the EXACT boundary (max_seq_len - 3 + 2 + 1 ==
//     max_seq_len) is accepted;
//
//   * CLEAN TEARDOWN: destroy_session(A/B) + retiring the reference
//     sequences returns every manager's accounting to EXACTLY ZERO.
//
// Self-skips (77) when the checkpoint is absent; the Phase C evidence
// environment MUST actually run it.

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
#include "cudalm/session.h"
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

// ---- turn / pool sizing ------------------------------------------------------
// Session A (greedy) / B (seeded sampling), two turns each:
static const std::vector<int> kA_T1 = {1024, 2048, 3073};  // 3 tokens
static const std::vector<int> kB_T1 = {100, 200};  // 2 tokens
static const std::vector<int> kA_T2 = {15, 16};  // 2 tokens
static const std::vector<int> kB_T2 = {7, 8};  // 2 tokens
static const int kA_T1_M = 3;
static const int kB_T1_M = 3;
static const int kA_T2_M = 2;
static const int kB_T2_M = 2;
// A final length: 3 + 3 + 2 + 2 = 10; B final: 2 + 3 + 2 + 2 = 9.
static const int kA_LEN1 = 6;
static const int kB_LEN1 = 5;
static const int kA_LEN2 = 10;
static const int kB_LEN2 = 9;
static const int kPageTokens = 2;
static const int kPoolPages = 12;  // scheduler peak: A(5) + B(5) = 10
static const int kDeltaSlots = 4;  // scheduler peak: A + B + C = 3

SamplingConfig sampling_cfg(std::uint64_t seed) {
  return SamplingConfig{0.7f, 50, 1.0f, seed};
}

// ---- device -> host helpers ---------------------------------------------------
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
  const bool ok = std::memcmp(ref.data(), act.data(),
                              ref.size() * sizeof(__nv_bfloat16)) == 0;
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

// ---- recording forwarder (logs FULL logits of every forward) -----------------
struct LogEntry {
  SequenceId sid;
  int token;
  std::vector<__nv_bfloat16> logits;  // [vocab]
};

// Wraps the real ModelForwarder; after every successful forward (single
// OR batch — batch rows come from logits_batch_to_host, exactly the
// logits the scheduler samples from), records that forward's full logits.
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
      s = inner_.logits_to_host(&e.logits, stream);
      if (s.ok) log_.push_back(std::move(e));
    }
    return s;
  }

  Status forward_batch(const int* token_ids, const SequenceId* sids, int B,
                       Qwen35StateManager& mgr, cudaStream_t stream) override {
    Status s = inner_.forward_batch(token_ids, sids, B, mgr, stream);
    if (s.ok) {
      std::vector<__nv_bfloat16> logits;
      s = inner_.logits_batch_to_host(&logits, B, stream);
      if (s.ok) {
        for (int k = 0; k < B; ++k) {
          LogEntry e;
          e.sid = sids[static_cast<std::size_t>(k)];
          e.token = token_ids[static_cast<std::size_t>(k)];
          const std::size_t o =
              static_cast<std::size_t>(k) * static_cast<std::size_t>(vocab_);
          e.logits.assign(logits.begin() + o,
                          logits.begin() + o + static_cast<std::size_t>(vocab_));
          log_.push_back(std::move(e));
        }
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
  bool supports_batch() const override {
    return inner_.supports_batch();  // model loaded -> real batching
  }

  // ALL forwards of `sid`, in forward order (copies).
  std::vector<LogEntry> entries(SequenceId sid) const {
    std::vector<LogEntry> out;
    for (const LogEntry& e : log_) {
      if (e.sid == sid) out.push_back(e);
    }
    return out;
  }

  ModelForwarder& inner() { return inner_; }

 private:
  ModelForwarder inner_;
  int vocab_;
  std::vector<LogEntry> log_;
};

// ---- independent reference: ONE session, continuous multi-turn ---------------
// The Phase B/C commit contract WITHOUT the scheduler: each turn appends
// its input from the sequence's CURRENT length (no (re)initialization),
// samples with a FRESH per-turn Sampler, and commits every generated
// token (the stop-triggering one included) before the turn ends.
struct RefTurn {
  std::vector<int> fwd_tokens;   // every committed forward of this turn
  std::vector<std::vector<__nv_bfloat16>> fwd_logits;  // one per forward
  std::vector<int> generated;
  int forward_count = 0;
  bool eos_hit = false;
};

struct RefSession {
  SequenceId sid = 0;
  RefTurn t1, t2;
  StateSnap state_after_t1;
  StateSnap state_after_t2;
};

int run_ref_turn(Qwen35Model& model, const Qwen35Config& cfg,
                 Qwen35StateManager& mgr, SequenceId sid,
                 const std::vector<int>& input, int max_new, int eos,
                 const SamplingConfig& sampling, cudaStream_t stream,
                 RefTurn* out) {
  // (1) commit the input tokens (append-only continuation):
  for (int t : input) {
    Status s = model.forward_token_with_state(t, sid, mgr, stream);
    if (!s.ok) {
      std::fprintf(stderr, "  ref forward %d: %s\n", t, s.message.c_str());
      return 1;
    }
    out->fwd_tokens.push_back(t);
    out->fwd_logits.push_back(
        d2h_bf16(model.logits(), cfg.vocab_size, stream));
    out->forward_count++;
  }
  if (max_new == 0) {
    return 0;  // input-only turn
  }
  // (2) commit-then-stop loop — sample gk (pending) -> forward gk
  //     (commit) -> the stop check happens ONLY on the committed token:
  Sampler sampler(sampling);  // fresh per turn (== per-request Sampler)
  for (int k = 0; k < max_new; ++k) {
    const std::vector<__nv_bfloat16> logits =
        d2h_bf16(model.logits(), cfg.vocab_size, stream);
    const int g = sampler.sample(logits.data(), cfg.vocab_size);
    if (g < 0 || g >= cfg.vocab_size) {
      std::fprintf(stderr, "  ref: sampler out of range\n");
      return 1;
    }
    out->generated.push_back(g);
    if (g == eos) out->eos_hit = true;
    // COMMIT g (always — terminal only AFTER the commit):
    Status s = model.forward_token_with_state(g, sid, mgr, stream);
    if (!s.ok) {
      std::fprintf(stderr, "  ref commit %d: %s\n", g, s.message.c_str());
      return 1;
    }
    out->fwd_tokens.push_back(g);
    out->fwd_logits.push_back(
        d2h_bf16(model.logits(), cfg.vocab_size, stream));
    out->forward_count++;
    if (g == eos) break;  // terminal AFTER the committed EOS
  }
  return 0;
}

// Compare one session's full scheduler log against its reference turns.
int cmp_logit_streams(const char* label, const std::vector<LogEntry>& act,
                      const RefTurn& t1, const RefTurn& t2) {
  int rc = 0;
  const std::size_t ref_n =
      t1.fwd_logits.size() + t2.fwd_logits.size();
  std::fprintf(stderr, "  [parity] %s: %zu scheduler forwards vs %zu "
                       "reference forwards\n",
               label, act.size(), ref_n);
  CHECK_EQ(act.size(), ref_n);
  for (std::size_t i = 0; i < act.size(); ++i) {
    const std::vector<__nv_bfloat16>& ref_l =
        i < t1.fwd_logits.size() ? t1.fwd_logits[i]
                                 : t2.fwd_logits[i - t1.fwd_logits.size()];
    const int ref_tok =
        i < t1.fwd_tokens.size() ? t1.fwd_tokens[i]
                                 : t2.fwd_tokens[i - t1.fwd_tokens.size()];
    char name[80];
    std::snprintf(name, sizeof(name), "%s forward[%zu].token", label, i);
    CHECK_EQ(act[i].token, ref_tok);
    std::snprintf(name, sizeof(name), "%s forward[%zu].logits[vocab]", label,
                  i);
    rc |= cmp_exact(name, ref_l, act[i].logits);
  }
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
    std::fprintf(stderr, "[SKIP] qwen35 scheduler+session integration: no "
                         "preconverted model at %s and --no-convert given\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] qwen35 scheduler+session integration: "
                         "checkpoint absent at %s\n",
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
               "[sched+session] model loaded: %d layers, vocab %d, "
               "max_seq_len %d\n",
               model.num_layers(), cfg.vocab_size, cfg.max_seq_len);

  int rc = 0;
  const SamplingConfig kGreedy = SamplingConfig::greedy();

  // =========================================================================
  // Part 1: INDEPENDENT REFERENCES — each session's token stream driven
  // continuously on an ALONE sequence (two turns, no turn boundary, same
  // per-turn sampler configs). States captured after each turn.
  // =========================================================================
  Qwen35StateManager refA_mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots,
                              stream);
  Qwen35StateManager refB_mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots,
                              stream);
  RefSession refA, refB;
  CHECK(refA_mgr.create_sequence(&refA.sid).ok);
  CHECK(refB_mgr.create_sequence(&refB.sid).ok);
  rc |= run_ref_turn(model, cfg, refA_mgr, refA.sid, kA_T1, kA_T1_M, -1,
                     kGreedy, stream, &refA.t1);
  {
    const SequenceState* rec = refA_mgr.lookup(refA.sid);
    CHECK(rec != nullptr);
    CHECK_EQ(rec->length, kA_LEN1);
    CHECK_EQ(capture_state_external(refA_mgr, *rec, cfg, kA_LEN1, stream,
                                    &refA.state_after_t1),
             0);
  }
  rc |= run_ref_turn(model, cfg, refA_mgr, refA.sid, kA_T2, kA_T2_M, -1,
                     kGreedy, stream, &refA.t2);
  {
    const SequenceState* rec = refA_mgr.lookup(refA.sid);
    CHECK(rec != nullptr);
    CHECK_EQ(rec->length, kA_LEN2);
    CHECK_EQ(capture_state_external(refA_mgr, *rec, cfg, kA_LEN2, stream,
                                    &refA.state_after_t2),
             0);
  }
  rc |= run_ref_turn(model, cfg, refB_mgr, refB.sid, kB_T1, kB_T1_M, -1,
                     sampling_cfg(42), stream, &refB.t1);
  {
    const SequenceState* rec = refB_mgr.lookup(refB.sid);
    CHECK(rec != nullptr);
    CHECK_EQ(rec->length, kB_LEN1);
    CHECK_EQ(capture_state_external(refB_mgr, *rec, cfg, kB_LEN1, stream,
                                    &refB.state_after_t1),
             0);
  }
  rc |= run_ref_turn(model, cfg, refB_mgr, refB.sid, kB_T2, kB_T2_M, -1,
                     sampling_cfg(123), stream, &refB.t2);
  {
    const SequenceState* rec = refB_mgr.lookup(refB.sid);
    CHECK(rec != nullptr);
    CHECK_EQ(rec->length, kB_LEN2);
    CHECK_EQ(capture_state_external(refB_mgr, *rec, cfg, kB_LEN2, stream,
                                    &refB.state_after_t2),
             0);
  }
  // Reference forward accounting (commit contract: input + m per turn):
  CHECK_EQ(refA.t1.forward_count, static_cast<int>(kA_T1.size()) + kA_T1_M);
  CHECK_EQ(refA.t2.forward_count, static_cast<int>(kA_T2.size()) + kA_T2_M);
  CHECK_EQ(refB.t1.forward_count, static_cast<int>(kB_T1.size()) + kB_T1_M);
  CHECK_EQ(refB.t2.forward_count, static_cast<int>(kB_T2.size()) + kB_T2_M);
  std::fprintf(stderr,
               "[reference] A: t1 gen %zu (%d fws), t2 gen %zu (%d fws) "
               "| B: t1 gen %zu (%d fws), t2 gen %zu (%d fws)\n",
               refA.t1.generated.size(), refA.t1.forward_count,
               refA.t2.generated.size(), refA.t2.forward_count,
               refB.t1.generated.size(), refB.t1.forward_count,
               refB.t2.generated.size(), refB.t2.forward_count);

  // =========================================================================
  // Part 2: SCHEDULER RUN — sessions A + B, two turns, batched cohorts.
  // =========================================================================
  Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  SessionManager sessions(mgr);
  RecordingForwarder rec(model);
  Scheduler sched(rec, mgr, stream, &sessions);

  SessionId sidA = 0, sidB = 0, sidC = 0;
  CHECK(sessions.create_session(&sidA).ok);
  CHECK(sessions.create_session(&sidB).ok);
  const SequenceId seqA = sessions.lookup(sidA)->sequence_id;
  const SequenceId seqB = sessions.lookup(sidB)->sequence_id;
  CHECK_EQ(mgr.num_live_sequences(), 2);  // the two bound sequences only

  // ---- t1 admission (+ the BUSY-session zero-mutation probe) -------------
  RequestId a1 = 0, b1 = 0, busy = 0;
  CHECK(sched.admit_session_turn(sidA, kA_T1, kA_T1_M, -1, kGreedy, &a1).ok);
  CHECK(sched.admit_session_turn(sidB, kB_T1, kB_T1_M, -1,
                                 sampling_cfg(42), &b1).ok);
  {
    const int reqs = sched.num_requests();
    const RequestId next_id = sched.next_request_id();
    const int lenA = mgr.lookup(seqA)->length;
    const int lenB = mgr.lookup(seqB)->length;
    const int pages = mgr.kv_pool().used_pages();
    const int slots = mgr.delta_pool().used_slots();
    Status bs = sched.admit_session_turn(sidA, {99}, 1, -1, kGreedy, &busy);
    CHECK(!bs.ok);
    CHECK(bs.message.find("live request") != std::string::npos);
    CHECK_EQ(sched.num_requests(), reqs);
    CHECK_EQ(sched.next_request_id(), next_id);
    CHECK_EQ(mgr.lookup(seqA)->length, lenA);
    CHECK_EQ(mgr.lookup(seqB)->length, lenB);
    CHECK_EQ(mgr.kv_pool().used_pages(), pages);
    CHECK_EQ(mgr.delta_pool().used_slots(), slots);
    std::fprintf(stderr, "  [ok] busy session: second live turn rejected, "
                         "zero mutation\n");
  }

  // ---- t1 run -------------------------------------------------------------
  CHECK(sched.run().ok);
  {
    const Request* pa = sched.get(a1);
    const Request* pb = sched.get(b1);
    CHECK(pa != nullptr && pb != nullptr);
    // Commit contract: input + ALL generated committed (no lag).
    CHECK(pa->status == RequestStatus::Finished);
    CHECK(pa->finish_reason == FinishReason::MaxNewTokens);
    CHECK_EQ(pa->forward_count, static_cast<int>(kA_T1.size()) + kA_T1_M);
    CHECK_EQ(pa->committed_generated, kA_T1_M);
    CHECK(pb->status == RequestStatus::Finished);
    CHECK(pb->finish_reason == FinishReason::MaxNewTokens);
    CHECK_EQ(pb->forward_count, static_cast<int>(kB_T1.size()) + kB_T1_M);
    CHECK_EQ(pb->committed_generated, kB_T1_M);
    // Generated token IDs EXACT (vs the continuous references):
    CHECK(pa->generated == refA.t1.generated);
    CHECK(pb->generated == refB.t1.generated);
    // SESSION PRESERVATION AFTER A TURN TERMINAL: requests terminal,
    // sessions + bound sequences LIVE, committed state intact.
    CHECK(sessions.lookup(sidA) != nullptr);
    CHECK(sessions.lookup(sidB) != nullptr);
    CHECK_EQ(mgr.lookup(seqA)->length, kA_LEN1);  // 6
    CHECK_EQ(mgr.lookup(seqB)->length, kB_LEN1);  // 5
    CHECK_EQ(mgr.num_live_sequences(), 2);  // NOT retired
    CHECK(!sched.session_busy(sidA));
    CHECK(!sched.session_busy(sidB));  // freed for their next turn
    // TURN-BOUNDARY STATE: the interleaved mid-stream state equals the
    // solo reference's state at the same length.
    StateSnap stA, stB;
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(seqA), cfg, kA_LEN1,
                                    stream, &stA),
             0);
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(seqB), cfg, kB_LEN1,
                                    stream, &stB),
             0);
    rc |= cmp_state("A turn1 boundary", refA.state_after_t1, stA, kA_LEN1,
                    cfg);
    rc |= cmp_state("B turn1 boundary", refB.state_after_t1, stB, kB_LEN1,
                    cfg);
    std::fprintf(stderr, "  [ok] t1 run: committed, preserved, boundary "
                         "states bit-identical\n");
  }

  // ---- t2 admission (the sessions freed by their terminal t1) ------------
  RequestId a2 = 0, b2 = 0;
  CHECK(sched.admit_session_turn(sidA, kA_T2, kA_T2_M, -1, kGreedy, &a2).ok);
  CHECK(sched.admit_session_turn(sidB, kB_T2, kB_T2_M, -1,
                                 sampling_cfg(123), &b2).ok);

  // ---- t2 run -------------------------------------------------------------
  CHECK(sched.run().ok);
  {
    const Request* pa = sched.get(a2);
    const Request* pb = sched.get(b2);
    CHECK(pa != nullptr && pb != nullptr);
    CHECK(pa->status == RequestStatus::Finished);
    CHECK(pa->finish_reason == FinishReason::MaxNewTokens);
    CHECK_EQ(pa->forward_count, static_cast<int>(kA_T2.size()) + kA_T2_M);
    CHECK_EQ(pa->committed_generated, kA_T2_M);
    CHECK(pb->status == RequestStatus::Finished);
    CHECK(pb->finish_reason == FinishReason::MaxNewTokens);
    CHECK_EQ(pb->forward_count, static_cast<int>(kB_T2.size()) + kB_T2_M);
    CHECK_EQ(pb->committed_generated, kB_T2_M);
    CHECK(pa->generated == refA.t2.generated);
    CHECK(pb->generated == refB.t2.generated);
    // FINAL LENGTHS:
    CHECK_EQ(mgr.lookup(seqA)->length, kA_LEN2);  // 10
    CHECK_EQ(mgr.lookup(seqB)->length, kB_LEN2);  // 9
    std::fprintf(stderr, "  [ok] t2 run: continued from the committed "
                         "boundary, final lengths %d / %d\n",
                 kA_LEN2, kB_LEN2);
  }

  // =========================================================================
  // Part 3: FINAL COMPARISON — per-step full logits (ALL forwards of each
  // session, in order, including the batched decode rows), the final
  // hybrid state, and the batch evidence.
  // =========================================================================
  {
    const std::vector<LogEntry> logA = rec.entries(seqA);
    const std::vector<LogEntry> logB = rec.entries(seqB);
    rc |= cmp_logit_streams("A (10 fws)", logA, refA.t1, refA.t2);
    rc |= cmp_logit_streams("B (9 fws)", logB, refB.t1, refB.t2);

    StateSnap stA, stB;
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(seqA), cfg, kA_LEN2,
                                    stream, &stA),
             0);
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(seqB), cfg, kB_LEN2,
                                    stream, &stB),
             0);
    rc |= cmp_state("A final", refA.state_after_t2, stA, kA_LEN2, cfg);
    rc |= cmp_state("B final", refB.state_after_t2, stB, kB_LEN2, cfg);
  }
  {
    // BATCH EVIDENCE: two live sessions formed true batched decode
    // cohorts (the predicted trace, from the scheduler's FIFO walk):
    //   t1: [B serial decode], [A,B]x2 (B finishes), [A serial]
    //   t2: [A,B]x2 (both finish)
    const int batch_calls = sched.batch_forward_calls();
    const int max_batch = sched.max_batch_size();
    std::fprintf(stderr,
                 "  [batch] batch_forward_calls=%d max_batch_size=%d "
                 "single=%d batched_tokens=%d\n",
                 batch_calls, max_batch, sched.single_forward_calls(),
                 sched.stats().batched_sequence_tokens);
    CHECK_EQ(batch_calls, 4);
    CHECK_EQ(max_batch, 2);
    CHECK_EQ(sched.single_forward_calls(), 11);
    CHECK_EQ(sched.stats().batched_sequence_tokens, 8);
    const std::vector<int> expected_trace = {1, 2, 2, 1, 2, 2};
    CHECK(sched.stats().decode_cohort_trace == expected_trace);
  }

  // =========================================================================
  // Part 4: CONTEXT OVERFLOW on the real model (session C) — zero mutation,
  // the exact boundary accepted; then clean teardown to ZERO accounting.
  // =========================================================================
  {
    CHECK(sessions.create_session(&sidC).ok);
    const SequenceId seqC = sessions.lookup(sidC)->sequence_id;
    const int max_seq = cfg.max_seq_len;
    // Overflow: (max_seq - 2) + input 2 + max_new 1 > max_seq.
    CHECK(mgr.set_length(seqC, max_seq - 2).ok);
    const int reqs = sched.num_requests();
    const RequestId next_id = sched.next_request_id();
    const int pages = mgr.kv_pool().used_pages();
    const int slots = mgr.delta_pool().used_slots();
    RequestId overflow = 0;
    Status os = sched.admit_session_turn(sidC, {1, 2}, 1, -1, kGreedy,
                                         &overflow);
    CHECK(!os.ok);
    CHECK(os.message.find("context overflow") != std::string::npos);
    CHECK_EQ(sched.num_requests(), reqs);
    CHECK_EQ(sched.next_request_id(), next_id);
    CHECK_EQ(mgr.kv_pool().used_pages(), pages);
    CHECK_EQ(mgr.delta_pool().used_slots(), slots);
    CHECK_EQ(mgr.lookup(seqC)->length, max_seq - 2);
    // Exact boundary: (max_seq - 3) + 2 + 1 == max_seq -> accepted
    // (then cancelled: zero forwards, zero mutation).
    CHECK(mgr.set_length(seqC, max_seq - 3).ok);
    RequestId boundary = 0;
    CHECK(sched.admit_session_turn(sidC, {1, 2}, 1, -1, kGreedy,
                                   &boundary).ok);
    CHECK(sched.cancel(boundary).ok);
    CHECK_EQ(mgr.lookup(seqC)->length, max_seq - 3);
    std::fprintf(stderr, "  [ok] overflow rejected (zero mutation), exact "
                         "boundary accepted\n");
    CHECK(sessions.destroy_session(sidC).ok);
  }
  // Teardown: sessions A + B destroyed, references retired -> ZERO.
  CHECK(sessions.destroy_session(sidA).ok);
  CHECK(sessions.destroy_session(sidB).ok);
  CHECK(refA_mgr.retire_sequence(refA.sid).ok);
  CHECK(refB_mgr.retire_sequence(refB.sid).ok);
  CHECK_EQ(mgr.kv_pool().used_pages(), 0);
  CHECK_EQ(mgr.delta_pool().used_slots(), 0);
  CHECK_EQ(refA_mgr.kv_pool().used_pages(), 0);
  CHECK_EQ(refA_mgr.delta_pool().used_slots(), 0);
  CHECK_EQ(refB_mgr.kv_pool().used_pages(), 0);
  CHECK_EQ(refB_mgr.delta_pool().used_slots(), 0);
  std::fprintf(stderr, "  [ok] teardown: all accounting back to zero\n");

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr,
                 "test_qwen35_scheduler_session_integration: FAIL (rc=%d)\n",
                 rc);
    return 1;
  }
  std::fprintf(stderr,
               "test_qwen35_scheduler_session_integration: PASS\n");
  return 0;
}
