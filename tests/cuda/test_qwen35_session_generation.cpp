// CUDALM — v0.8 Phase B: incremental multi-turn generation real-checkpoint
// hard gate (CUDA, real Qwen3.5-0.8B-Base checkpoint).
//
// The device-level proof that INCREMENTAL MULTI-TURN EXECUTION on a session
// is numerically INVISIBLE — bit-identical (memcmp) to the equivalent
// ONE-SHOT CONTINUOUS execution of the same token stream:
//
//   turn 1: session A append [a,b,c]  -> generate [d,e]  (greedy, M=2)
//   turn 2: same session A append [f,g] -> generate ...  (seeded sampling, M=2)
//
// vs the reference: ONE fresh v0.5 sequence (NOT a session — the v0.5
// independent-execution semantics) driven manually, token by token, in ONE
// continuous run with the same appends, the same sampling configs at the
// same points (a fresh per-phase Sampler — the same per-turn RNG isolation
// the turn API applies), and NO turn boundary in between.
//
// Gates (all BIT-IDENTICAL unless noted):
//   * per-step FULL logits [248320] of EVERY committed forward (turn 1
//     steps 0..4 + turn 2 steps 0..3 == the reference's 9 forwards, in
//     order) — a turn-boundary bug (state reset / re-prefill / position
//     drift) breaks the first turn-2 step;
//   * generated token IDs (turn 1 == [d,e], turn 2 == the reference's
//     seeded samples);
//   * logical length (5 after turn 1, 9 after turn 2);
//   * the FULL final hybrid state: 18x DeltaNet conv (bf16) + recurrent
//     (fp32) and 6x full-attention K/V logical rows 0..8 read from the
//     pool's physical pages THROUGH the block table — the COMMIT CONTRACT
//     (the turn's LAST generated token is in the session state; no
//     lagging);
//   * pool accounting (A holds exactly ceil(9/2) = 5 pages);
//   * PREFLIGHT on the loaded model (zero forwards): invalid eos_token_id
//     and invalid input token are rejected with the session EXACTLY
//     unchanged;
//   * max_new_tokens == 0: an INPUT-ONLY append (3 input tokens, nothing
//     generated; state == a fresh 3-token reference, bit-identical);
//   * EOS COMMIT semantics: a turn that hits eos==the-first-greedy-token
//     stops (Eos) with exactly [that token] generated, and its final state
//     is BIT-IDENTICAL to the M=1 no-gate turn (the EOS token IS committed);
//   * CONTEXT OVERFLOW on the real model: a session at length
//     max_seq_len - 2 rejects a 2+1 turn (262145 > 262144) with ZERO
//     mutation (no eviction / truncation);
//   * RESET + RE-RUN: reset_session(A) (fresh state under the SAME id,
//     Delta slot ALL ZEROS on device) and re-running the SAME two turns
//     reproduces the ORIGINAL per-step logits / generated ids / final
//     state bit-for-bit (== the one-shot reference);
//   * RUNTIME FAILURE (no rollback, a separate 3-page manager): an
//     input-phase KV OOM keeps the 6 committed inputs (the 7th is not
//     committed); a generation-phase KV OOM keeps the committed g0 (g1 is
//     not committed) — the session stays at the last successful token
//     boundary (context_length exact).
//
// Self-skips (77) when the checkpoint is absent; in the Phase B evidence
// environment the checkpoint IS present and this test actually runs
// (77 is not a sign-off).
//
// Provenance: CUDALM-native (v0.8 Phase B).

#include "../../tests/common/check.h"

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/sampling.h"
#include "cudalm/session.h"
#include "cudalm/session_generator.h"
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

// Deterministic tokens (fixed, no randomness).
static const int kT1Input[] = {1024, 2048, 3072};  // [a,b,c]
static const int kT2Input[] = {15, 16};            // [f,g]
static const int kT1M = 2;  // turn 1: greedy, 2 generated
static const int kT2M = 2;  // turn 2: seeded sampling, 2 generated
static const int kTotalForwards = 3 + kT1M + 2 + kT2M;  // == 9
static const int kMaxTokens = 3 + kT1M + 2 + kT2M;  // final length == 9

static const int kPageTokens = 2;
static const int kPoolPages = 14;
static const int kDeltaSlots = 5;  // A,B,C,D + one independent reference

// Device -> host capture helpers (explicit cudaMemcpyDeviceToHost).
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

// Bit-exact comparison (the Phase B gate: same kernels, same op order).
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
  if (!ok) {
    std::fprintf(stderr, "  %-48s FAIL not bit-identical (n=%zu)\n", name,
                 ref.size());
    return 1;
  }
  return 0;
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
  if (!ok) {
    std::fprintf(stderr, "  %-48s FAIL not bit-identical (n=%zu)\n", name,
                 ref.size());
    return 1;
  }
  return 0;
}

// ---- hybrid state capture (pool slot + pages through the block table) ----
struct StateSnap {
  std::vector<std::vector<__nv_bfloat16>> conv;  // [24] (linear layers)
  std::vector<std::vector<float>> rec;           // [24]
  std::vector<std::vector<__nv_bfloat16>> kvk;   // [24] rows 0..n-1, all kv
  std::vector<std::vector<__nv_bfloat16>> kvv;   // [24]
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
          CUDA_CHECK(cudaMemcpyAsync(
              k.data() + dst, kp, bytes, cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaMemcpyAsync(
              v.data() + dst, vp, bytes, cudaMemcpyDeviceToHost, stream));
        }
      }
      out->kvk[static_cast<std::size_t>(i)] = std::move(k);
      out->kvv[static_cast<std::size_t>(i)] = std::move(v);
    }
  }
  return 0;
}

int cmp_state(const char* label, const StateSnap& a, const StateSnap& b,
              const Qwen35Config& cfg) {
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
      std::snprintf(name, sizeof(name), "%s L%d kv.K/V", label, i);
      rc |= cmp_exact(name, a.kvk[static_cast<std::size_t>(i)],
                      b.kvk[static_cast<std::size_t>(i)]);
      rc |= cmp_exact(name, a.kvv[static_cast<std::size_t>(i)],
                      b.kvv[static_cast<std::size_t>(i)]);
    }
  }
  return rc;
}

bool delta_slot_all_zero(const Qwen35StateManager& mgr, int slot,
                         cudaStream_t stream) {
  const auto& dp = mgr.delta_pool();
  for (int l = 0; l < dp.n_linear_layers(); ++l) {
    const std::vector<__nv_bfloat16> conv =
        d2h_bf16(dp.conv(l, slot), dp.conv_elems(), stream);
    for (__nv_bfloat16 x : conv)
      if (__bfloat162float(x) != 0.0f) return false;
    const std::vector<float> rec =
        d2h_f32(dp.recurrent(l, slot), dp.rec_elems(), stream);
    for (float x : rec)
      if (x != 0.0f) return false;
  }
  return true;
}

// A stateful LogitsObserver: keeps every (step, full host logits) pair.
struct LogitCapture {
  int vocab;
  std::vector<std::vector<__nv_bfloat16>> steps;
  explicit LogitCapture(int v) : vocab(v) {}
  void operator()(const __nv_bfloat16* h, int step) {
    if (step >= static_cast<int>(steps.size()))
      steps.resize(static_cast<std::size_t>(step) + 1);
    steps[static_cast<std::size_t>(step)] =
        std::vector<__nv_bfloat16>(h, h + vocab);
  }
};

// Manual single-token forward + full-logits D2H (the reference driver).
int fwd_cap(Qwen35Model& model, Qwen35StateManager& mgr, SequenceId sid,
            int token, const Qwen35Config& cfg, cudaStream_t stream,
            std::vector<__nv_bfloat16>* cap) {
  Status s = model.forward_token_with_state(token, sid, mgr, stream);
  if (!s.ok) {
    std::fprintf(stderr, "  forward_token_with_state(token %d): %s\n", token,
                 s.message.c_str());
    return 1;
  }
  *cap = d2h_bf16(model.logits(), static_cast<std::size_t>(cfg.vocab_size),
                  stream);
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
    std::fprintf(stderr, "[SKIP] qwen35 session generation: no preconverted "
                         "model at %s and --no-convert given\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] qwen35 session generation: checkpoint absent "
                         "at %s\n",
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
  const int vocab = cfg.vocab_size;
  std::fprintf(stderr,
               "[session-gen] model loaded: %d layers, vocab %d, max_seq %d, "
               "page_tokens %d, pool pages %d, delta slots %d\n",
               model.num_layers(), vocab, cfg.max_seq_len, kPageTokens,
               kPoolPages, kDeltaSlots);

  Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  SessionManager sm(mgr);
  SessionGenerator gen(model, sm);
  int rc = 0;

  const SamplingConfig greedy = SamplingConfig::greedy();
  SamplingConfig sampling_turn2;  // seeded sampling (the per-turn RNG
  sampling_turn2.temperature = 0.7f;
  sampling_turn2.top_k = 50;
  sampling_turn2.top_p = 1.0f;
  sampling_turn2.seed = 1234;

  // ===================================================================
  // REFERENCES FIRST (pool capacity discipline): the one-shot continuous
  // reference on an INDEPENDENT fresh v0.5 sequence (NOT a session) — the
  // same token stream, the same sampling configs at the same points, NO
  // turn boundary. Captured to host, then retired.
  // ===================================================================
  std::vector<std::vector<__nv_bfloat16>> cap(kTotalForwards);
  std::vector<int> ref_generated;
  StateSnap ref_state;
  {
    SequenceId ref = 0;
    CHECK(mgr.create_sequence(&ref).ok);
    int tok = kT1Input[0];
    for (int i = 0; i < 3; ++i)
      CHECK_EQ(fwd_cap(model, mgr, ref, kT1Input[i], cfg, stream, &cap[i]), 0);
    Sampler s1(greedy);  // phase 1: greedy (zero RNG) — same as turn 1
    tok = s1.sample(cap[2].data(), vocab);
    ref_generated.push_back(tok);  // g0
    CHECK_EQ(fwd_cap(model, mgr, ref, tok, cfg, stream, &cap[3]), 0);
    tok = s1.sample(cap[3].data(), vocab);
    ref_generated.push_back(tok);  // g1
    CHECK_EQ(fwd_cap(model, mgr, ref, tok, cfg, stream, &cap[4]), 0);
    for (int i = 0; i < 2; ++i)
      CHECK_EQ(fwd_cap(model, mgr, ref, kT2Input[i], cfg, stream,
                       &cap[5 + i]), 0);
    Sampler s2(sampling_turn2);  // phase 2: SAME seeded config as turn 2
    tok = s2.sample(cap[6].data(), vocab);
    ref_generated.push_back(tok);  // h0
    CHECK_EQ(fwd_cap(model, mgr, ref, tok, cfg, stream, &cap[7]), 0);
    tok = s2.sample(cap[7].data(), vocab);
    ref_generated.push_back(tok);  // h1
    CHECK_EQ(fwd_cap(model, mgr, ref, tok, cfg, stream, &cap[8]), 0);
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(ref), cfg, kMaxTokens,
                                    stream, &ref_state), 0);
    CHECK(mgr.retire_sequence(ref).ok);
  }
  CHECK_EQ(mgr.kv_pool().used_pages(), 0);
  CHECK_EQ(mgr.delta_pool().used_slots(), 0);

  // ===================================================================
  // SESSION A: turn 1 (greedy) + turn 2 (seeded sampling).
  // ===================================================================
  SessionId sa = 0, sb = 0, sc = 0, sd = 0;
  CHECK(sm.create_session(&sa).ok);
  CHECK(sm.create_session(&sb).ok);
  CHECK(sm.create_session(&sc).ok);
  CHECK(sm.create_session(&sd).ok);

  LogitCapture t1(vocab);
  LogitsObserver obs1 =
      [&](const __nv_bfloat16* h, int step) { t1(h, step); };
  TurnResult r1 = gen.generate_turn(
      sa, {kT1Input[0], kT1Input[1], kT1Input[2]}, kT1M, -1, greedy, stream,
      &obs1);
  CHECK(r1.ok);
  CHECK(r1.stop_reason == StopReason::MaxNewTokens);
  CHECK_EQ(r1.input_count, 3);
  CHECK_EQ(r1.forward_count, 3 + kT1M);
  CHECK_EQ(r1.context_length, 3 + kT1M);
  CHECK_EQ(r1.generated.size(), static_cast<std::size_t>(kT1M));
  CHECK(r1.generated[0] == ref_generated[0]);
  CHECK(r1.generated[1] == ref_generated[1]);
  CHECK_EQ(t1.steps.size(), static_cast<std::size_t>(3 + kT1M));

  LogitCapture t2(vocab);
  LogitsObserver obs2 =
      [&](const __nv_bfloat16* h, int step) { t2(h, step); };
  TurnResult r2 = gen.generate_turn(
      sa, {kT2Input[0], kT2Input[1]}, kT2M, -1, sampling_turn2, stream, &obs2);
  CHECK(r2.ok);
  CHECK(r2.stop_reason == StopReason::MaxNewTokens);
  CHECK_EQ(r2.input_count, 2);
  CHECK_EQ(r2.forward_count, 2 + kT2M);
  CHECK_EQ(r2.context_length, kMaxTokens);
  CHECK_EQ(r2.generated.size(), static_cast<std::size_t>(kT2M));
  CHECK(r2.generated[0] == ref_generated[2]);
  CHECK(r2.generated[1] == ref_generated[3]);
  CHECK_EQ(t2.steps.size(), static_cast<std::size_t>(2 + kT2M));

  StateSnap a_state;
  CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sm.lookup(sa)->sequence_id),
                                  cfg, kMaxTokens, stream, &a_state), 0);

  // ---- INCREMENTAL == ONE-SHOT, per committed forward, BIT-IDENTICAL -------
  for (int i = 0; i < 3 + kT1M; ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "turn1 step p%d logits[%d]", i, vocab);
    rc |= cmp_exact(name, cap[static_cast<std::size_t>(i)],
                    t1.steps[static_cast<std::size_t>(i)]);
  }
  for (int j = 0; j < 2 + kT2M; ++j) {
    char name[64];
    std::snprintf(name, sizeof(name), "turn2 step p%d logits[%d]", 5 + j, vocab);
    rc |= cmp_exact(name, cap[static_cast<std::size_t>(5 + j)],
                    t2.steps[static_cast<std::size_t>(j)]);
  }
  rc |= cmp_state("incremental vs one-shot (A final state)", a_state,
                  ref_state, cfg);
  // Pool accounting: A holds exactly ceil(9/2) = 5 pages; 4 slots live.
  CHECK_EQ(mgr.kv_pool().used_pages(), 5);
  CHECK_EQ(mgr.delta_pool().used_slots(), 4);
  std::fprintf(stderr,
               "  [ok] turn1+turn2 == one-shot continuous (logits, ids, "
               "length, paged KV, Delta conv/recurrent — bit-identical)\n");

  // ===================================================================
  // PREFLIGHT on the LOADED MODEL (zero forwards; the session is EXACTLY
  // unchanged).
  // ===================================================================
  {
    const int len_before = mgr.lookup(sm.lookup(sa)->sequence_id)->length;
    const int pages_before = mgr.kv_pool().used_pages();
    TurnResult re = gen.generate_turn(sa, {100}, 1, /*eos_token_id=*/vocab,
                                      greedy, stream);
    CHECK(!re.ok);
    CHECK(re.error.find("invalid eos_token_id") != std::string::npos);
    TurnResult rt =
        gen.generate_turn(sa, {vocab}, 1, -1, greedy, stream);
    CHECK(!rt.ok);
    CHECK(rt.error.find("invalid input token id") != std::string::npos);
    CHECK_EQ(mgr.lookup(sm.lookup(sa)->sequence_id)->length, len_before);
    CHECK_EQ(mgr.kv_pool().used_pages(), pages_before);
    std::fprintf(stderr, "  [ok] preflight on loaded model: invalid eos / "
                         "invalid token rejected, zero mutation\n");
  }

  // ===================================================================
  // max_new_tokens == 0: an INPUT-ONLY append on session B.
  // ===================================================================
  {
    LogitCapture tb(vocab);
    LogitsObserver obsb =
        [&](const __nv_bfloat16* h, int step) { tb(h, step); };
    TurnResult rb = gen.generate_turn(sb, {100, 200, 300}, 0, -1, greedy,
                                      stream, &obsb);
    CHECK(rb.ok);
    CHECK(rb.stop_reason == StopReason::MaxNewTokens);
    CHECK(rb.generated.empty());
    CHECK_EQ(rb.input_count, 3);
    CHECK_EQ(rb.forward_count, 3);
    CHECK_EQ(rb.context_length, 3);
    StateSnap b_state;
    CHECK_EQ(
        capture_state_external(mgr, *mgr.lookup(sm.lookup(sb)->sequence_id),
                               cfg, 3, stream, &b_state), 0);
    SequenceId ref_b = 0;
    CHECK(mgr.create_sequence(&ref_b).ok);
    StateSnap ref_b_state;
    std::vector<__nv_bfloat16> scratch;
    for (int i = 0; i < 3; ++i)
      CHECK_EQ(fwd_cap(model, mgr, ref_b, (i + 1) * 100, cfg, stream,
                       &scratch), 0);
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(ref_b), cfg, 3, stream,
                                    &ref_b_state), 0);
    rc |= cmp_state("B input-only append state", b_state, ref_b_state, cfg);
    CHECK(mgr.retire_sequence(ref_b).ok);
    std::fprintf(stderr,
                 "  [ok] max_new_tokens == 0: input-only append, state == "
                 "fresh 3-token reference\n");
  }

  // ===================================================================
  // EOS COMMIT semantics (sessions C / D): the EOS token IS committed.
  // ===================================================================
  int g_c = -1;
  {
    TurnResult rc_turn =
        gen.generate_turn(sc, {kT1Input[0], kT1Input[1], kT1Input[2]}, 1, -1,
                          greedy, stream);
    CHECK(rc_turn.ok);
    CHECK_EQ(rc_turn.generated.size(), static_cast<std::size_t>(1));
    g_c = rc_turn.generated[0];
    CHECK_EQ(rc_turn.context_length, 4);
    TurnResult rd_turn = gen.generate_turn(sd,
                                           {kT1Input[0], kT1Input[1],
                                            kT1Input[2]},
                                           3, /*eos_token_id=*/g_c, greedy,
                                           stream);
    CHECK(rd_turn.ok);
    CHECK(rd_turn.stop_reason == StopReason::Eos);
    CHECK_EQ(rd_turn.generated.size(), static_cast<std::size_t>(1));
    CHECK(rd_turn.generated[0] == g_c);
    CHECK_EQ(rd_turn.context_length, 4);
    StateSnap c_state, d_state;
    CHECK_EQ(
        capture_state_external(mgr, *mgr.lookup(sm.lookup(sc)->sequence_id),
                               cfg, 4, stream, &c_state), 0);
    CHECK_EQ(
        capture_state_external(mgr, *mgr.lookup(sm.lookup(sd)->sequence_id),
                               cfg, 4, stream, &d_state), 0);
    rc |= cmp_state("EOS committed (D == C)", d_state, c_state, cfg);
    std::fprintf(stderr,
                 "  [ok] EOS: stopped after committing the EOS token; state "
                 "== M=1 no-gate turn\n");
  }

  // ===================================================================
  // CONTEXT OVERFLOW on the real model (session B, metadata length).
  // ===================================================================
  {
    const SequenceId sb_sid = sm.lookup(sb)->sequence_id;
    CHECK(mgr.set_length(sb_sid, cfg.max_seq_len - 2).ok);
    TurnResult ro = gen.generate_turn(sb, {7, 8}, 1, -1, greedy, stream);
    CHECK(!ro.ok);
    CHECK(ro.error.find("context overflow") != std::string::npos);
    // ZERO mutation: length / pages / table exactly unchanged.
    CHECK_EQ(mgr.lookup(sb_sid)->length, cfg.max_seq_len - 2);
    CHECK_EQ(mgr.lookup(sb_sid)->block_table.num_blocks(), 2);
    CHECK_EQ(mgr.kv_pool().used_pages(), 11);  // A(5) + B(2) + C(2) + D(2)
    CHECK(sm.destroy_session(sb).ok);
    std::fprintf(stderr,
                 "  [ok] context overflow: %d + 2 + 1 > max_seq_len %d "
                 "rejected, zero mutation\n",
                 cfg.max_seq_len - 2, cfg.max_seq_len);
  }

  // ===================================================================
  // RESET + RE-RUN on session A: fresh state under the SAME id; the same
  // two turns reproduce the ORIGINAL run bit-for-bit (== one-shot).
  // ===================================================================
  {
    const SequenceId sa_sid = sm.lookup(sa)->sequence_id;
    const int a_slot = mgr.lookup(sa_sid)->delta_slot;
    CHECK(sm.reset_session(sa).ok);
    CHECK_EQ(mgr.lookup(sa_sid)->length, 0);
    CHECK_EQ(mgr.lookup(sa_sid)->block_table.num_blocks(), 0);
    CHECK_EQ(mgr.lookup(sa_sid)->delta_slot, a_slot);
    CHECK(delta_slot_all_zero(mgr, a_slot, stream));
    CHECK_EQ(mgr.kv_pool().used_pages(), 4);  // C(2) + D(2) only
    LogitCapture t1b(vocab), t2b(vocab);
    LogitsObserver obs1b =
        [&](const __nv_bfloat16* h, int step) { t1b(h, step); };
    LogitsObserver obs2b =
        [&](const __nv_bfloat16* h, int step) { t2b(h, step); };
    TurnResult r1b = gen.generate_turn(
        sa, {kT1Input[0], kT1Input[1], kT1Input[2]}, kT1M, -1, greedy, stream,
        &obs1b);
    CHECK(r1b.ok);
    CHECK(r1b.generated == r1.generated);
    TurnResult r2b = gen.generate_turn(sa, {kT2Input[0], kT2Input[1]}, kT2M,
                                       -1, sampling_turn2, stream, &obs2b);
    CHECK(r2b.ok);
    CHECK(r2b.generated == r2.generated);
    CHECK_EQ(r2b.context_length, kMaxTokens);
    for (int i = 0; i < 3 + kT1M; ++i) {
      char name[64];
      std::snprintf(name, sizeof(name), "reset turn1 p%d logits[%d]", i, vocab);
      rc |= cmp_exact(name, t1.steps[static_cast<std::size_t>(i)],
                      t1b.steps[static_cast<std::size_t>(i)]);
    }
    for (int j = 0; j < 2 + kT2M; ++j) {
      char name[64];
      std::snprintf(name, sizeof(name), "reset turn2 p%d logits[%d]", 5 + j,
                    vocab);
      rc |= cmp_exact(name, t2.steps[static_cast<std::size_t>(j)],
                      t2b.steps[static_cast<std::size_t>(j)]);
    }
    StateSnap a_state2;
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sa_sid), cfg, kMaxTokens,
                                    stream, &a_state2), 0);
    rc |= cmp_state("reset re-run final state", a_state2, ref_state, cfg);
    std::fprintf(stderr,
                 "  [ok] reset + re-run: fresh under the SAME id; re-run "
                 "bit-identical to the original (== one-shot)\n");
  }

  // Tear down the main sessions; the accounting must return to zero.
  CHECK(sm.destroy_session(sa).ok);
  CHECK(sm.destroy_session(sc).ok);
  CHECK(sm.destroy_session(sd).ok);
  CHECK_EQ(mgr.kv_pool().used_pages(), 0);
  CHECK_EQ(mgr.delta_pool().used_slots(), 0);
  CHECK_EQ(sm.num_sessions(), 0);

  // ===================================================================
  // RUNTIME FAILURE semantics (no rollback; a separate 3-page manager).
  // pages=3, pt=2 -> blocks 0,1,2 -> positions 0..5 fit; position 6 OOMs.
  // ===================================================================
  {
    Qwen35StateManager mgr2(cfg, kPageTokens, /*pages=*/3, /*slots=*/3,
                            stream);
    SessionManager sm2(mgr2);
    SessionGenerator gen2(model, sm2);
    // F0: 6 input tokens M=0 — the EXACT fit (positions 0..5 == 3 pages).
    {
      SessionId f = 0;
      CHECK(sm2.create_session(&f).ok);
      TurnResult r = gen2.generate_turn(
          f, {7, 8, 9, 10, 11, 12}, 0, -1, greedy, stream);
      CHECK(r.ok);
      CHECK(r.generated.empty());
      CHECK_EQ(r.input_count, 6);
      CHECK_EQ(r.context_length, 6);
      CHECK_EQ(mgr2.kv_pool().used_pages(), 3);
      CHECK(sm2.destroy_session(f).ok);
    }
    // F1: 7 input tokens M=0 — the 7th (position 6) OOMs: the 6 committed
    // inputs STAY, the 7th is NOT committed.
    {
      SessionId f = 0;
      CHECK(sm2.create_session(&f).ok);
      TurnResult r = gen2.generate_turn(f,
                                        {7, 8, 9, 10, 11, 12, 13}, 0, -1,
                                        greedy, stream);
      CHECK(!r.ok);
      CHECK(r.error.find("forward failed at input token 6") !=
            std::string::npos);
      CHECK(r.generated.empty());
      CHECK_EQ(r.input_count, 6);
      CHECK_EQ(r.context_length, 6);  // the last successful token boundary
      CHECK_EQ(mgr2.kv_pool().used_pages(), 3);
      CHECK(sm2.destroy_session(f).ok);
    }
    // F2: 5 input tokens M=2 (greedy) — g0 commits (position 5); g1's
    // forward (position 6) OOMs: g0 STAYS committed, g1 is NOT.
    {
      SessionId f = 0;
      CHECK(sm2.create_session(&f).ok);
      TurnResult r = gen2.generate_turn(f, {7, 8, 9, 10, 11}, 2, -1, greedy,
                                        stream);
      CHECK(!r.ok);
      CHECK(r.error.find("forward failed at generated token 1") !=
            std::string::npos);
      CHECK_EQ(r.input_count, 5);
      CHECK_EQ(r.generated.size(), static_cast<std::size_t>(1));  // g0 committed, g1 not
      CHECK(r.generated[0] >= 0 && r.generated[0] < vocab);
      CHECK_EQ(r.context_length, 6);  // 5 inputs + g0
      CHECK_EQ(mgr2.kv_pool().used_pages(), 3);
      CHECK(sm2.destroy_session(f).ok);
    }
    CHECK_EQ(mgr2.kv_pool().used_pages(), 0);
    CHECK_EQ(mgr2.delta_pool().used_slots(), 0);
    std::fprintf(stderr,
                 "  [ok] runtime failure (KV OOM): committed tokens stay, "
                 "the failing token is not committed (no rollback)\n");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_session_generation: FAIL\n");
    return rc;
  }
  std::fprintf(stderr,
               "test_qwen35_session_generation: PASS (incremental multi-turn "
               "== one-shot bit-identical: per-step FULL logits + ids + "
               "length + paged KV + Delta conv/recurrent; preflight / "
               "input-only / EOS-commit / overflow / reset-re-run / "
               "no-rollback failure gates)\n");
  return 0;
}
