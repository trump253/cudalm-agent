// CUDALM — v0.6 Phase B: TRUE BATCHED GPU decode execution — FULL-MODEL
// real-checkpoint hard gate (CUDA, REAL Qwen3.5-0.8B-Base checkpoint).
//
// The Phase B correctness crown: proves that the new full-model batched
// decode API (Qwen35Model::forward_batch_with_state — ONE traversal of all
// 24 layers with TRUE batched GEMVs / DeltaNet / paged attention over a
// heterogeneous cohort) produces, PER ROW, output BIT-IDENTICAL to the
// frozen single-sequence path (forward_token_with_state) run from an
// EQUIVALENT state — across MULTIPLE decode steps.
//
// Coverage (the pinned hard gate):
//   * B = 3 (and B = 1) rows in ONE forward_batch_with_state;
//   * HETEROGENEOUS positions: rows have DIFFERENT prompt lengths (3 / 5 / 2
//     tokens) so their decode positions, page counts, and block tables differ
//     (small page_tokens = 2 forces page boundaries);
//   * NON-TRIVIAL page mapping: distinct per-row block tables + Delta slots;
//   * MULTIPLE decode steps (two consecutive batched decode steps);
//   * PER-ROW BIT-IDENTICAL parity vs the frozen single path for: the
//     embedding output [H], ALL 24 layer-final outputs [H], the final-norm
//     output [H], and the FULL logits [vocab = 248320];
//   * FINAL per-sequence hybrid state BIT-IDENTICAL: 18x DeltaNet conv +
//     recurrent, 6x full-attention LOGICAL K/V rows (through the block
//     table), and the sequence length;
//   * B = 1 parity (a batch of one == the single path).
//
// The batch and the serial reference start from EQUIVALENT states (each
// sequence's hybrid state is a pure function of its own prompt — per-sequence
// isolation — so a fresh manager prefilling the same prompt is bit-identical).
//
// No throughput claim: Phase B is CORRECTNESS-first. Prefill stays SERIAL;
// only DECODE is batched; single model + single CUDA stream.
//
// Self-skips (77) when the checkpoint is absent; in the Phase B evidence
// environment the checkpoint IS present and this test actually runs (77 is
// not a sign-off).

#include "../../tests/common/check.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_state_manager.h"
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

// Pool sizing (small page_tokens to force page boundaries + non-trivial
// mapping; generous capacity so the batch preflight never falls back).
static const int kPageTokens = 2;
static const int kPoolPages = 24;
static const int kDeltaSlots = 8;

// Heterogeneous prompts (DIFFERENT lengths -> different decode positions,
// page counts, block tables): 3 / 5 / 2 tokens.
static const std::vector<int> kPrompt0 = {1024, 2048, 3072};
static const std::vector<int> kPrompt1 = {15, 16, 17, 18, 19};
static const std::vector<int> kPrompt2 = {4096, 5000};
// Two consecutive decode steps (fixed, deterministic tokens).
static const int kStep1Tok[3] = {777, 888, 999};
static const int kStep2Tok[3] = {111, 222, 333};
// B = 1 case.
static const std::vector<int> kPromptB1 = {7, 8, 9, 10};
static const int kB1Tok = 12345;

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
    std::fprintf(stderr, "  %-52s FAIL size %zu != %zu\n", name, ref.size(),
                 act.size());
    return 1;
  }
  const bool ok =
      std::memcmp(ref.data(), act.data(), ref.size() * sizeof(__nv_bfloat16)) ==
      0;
  std::fprintf(stderr, "  %-52s %s (n=%zu)\n", name, ok ? "bit-identical" : "FAIL",
               ref.size());
  return ok ? 0 : 1;
}
int cmp_exact_f32(const char* name, const std::vector<float>& ref,
                  const std::vector<float>& act) {
  if (ref.size() != act.size()) {
    std::fprintf(stderr, "  %-52s FAIL size %zu != %zu\n", name, ref.size(),
                 act.size());
    return 1;
  }
  const bool ok =
      std::memcmp(ref.data(), act.data(), ref.size() * sizeof(float)) == 0;
  std::fprintf(stderr, "  %-52s %s (n=%zu)\n", name, ok ? "bit-identical" : "FAIL",
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
  char name[80];
  std::snprintf(name, sizeof(name), "%s length", label);
  std::fprintf(stderr, "  %-52s %s (%d == %d)\n", name,
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

// ---- one ROW of model outputs (embedding / 24 layer-finals / norm / logits) -
struct RowOut {
  std::vector<__nv_bfloat16> emb;    // [H]
  std::vector<std::vector<__nv_bfloat16>> layer;  // [L][H]
  std::vector<__nv_bfloat16> norm;   // [H]
  std::vector<__nv_bfloat16> logits; // [vocab]
};

// Row b of the LAST forward_batch_with_state (B rows on device).
int capture_batch_row(const Qwen35Model& model, int B, int b, cudaStream_t st,
                      RowOut* out) {
  const Qwen35Config& cfg = model.config();
  const int H = cfg.hidden_size, V = cfg.vocab_size, L = model.num_layers();
  auto emb = d2h_bf16(model.batch_embedding_output(),
                      static_cast<std::size_t>(B) * H, st);
  auto norm = d2h_bf16(model.batch_final_norm_output(),
                       static_cast<std::size_t>(B) * H, st);
  auto logits = d2h_bf16(model.batch_logits(),
                         static_cast<std::size_t>(B) * V, st);
  out->emb.assign(emb.begin() + b * H, emb.begin() + (b + 1) * H);
  out->norm.assign(norm.begin() + b * H, norm.begin() + (b + 1) * H);
  out->logits.assign(logits.begin() + static_cast<std::size_t>(b) * V,
                     logits.begin() + static_cast<std::size_t>(b + 1) * V);
  out->layer.assign(L, {});
  for (int i = 0; i < L; ++i) {
    auto lf = d2h_bf16(model.batch_layer_final_output(i),
                       static_cast<std::size_t>(B) * H, st);
    out->layer[static_cast<std::size_t>(i)].assign(lf.begin() + b * H,
                                                   lf.begin() + (b + 1) * H);
  }
  return 0;
}

// The LAST forward_token_with_state (single path).
int capture_single(const Qwen35Model& model, cudaStream_t st, RowOut* out) {
  const Qwen35Config& cfg = model.config();
  const int H = cfg.hidden_size, V = cfg.vocab_size, L = model.num_layers();
  out->emb = d2h_bf16(model.embedding_output(), H, st);
  out->norm = d2h_bf16(model.final_norm_output(), H, st);
  out->logits = d2h_bf16(model.logits(), V, st);
  out->layer.assign(L, {});
  for (int i = 0; i < L; ++i)
    out->layer[static_cast<std::size_t>(i)] =
        d2h_bf16(model.layer_final_output(i), H, st);
  return 0;
}

int cmp_row(const char* label, const RowOut& ref, const RowOut& act,
            const Qwen35Config& cfg) {
  int rc = 0;
  char name[80];
  std::snprintf(name, sizeof(name), "%s embedding[H]", label);
  rc |= cmp_exact(name, ref.emb, act.emb);
  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    std::snprintf(name, sizeof(name), "%s layer-final L%d[H]", label, i);
    rc |= cmp_exact(name, ref.layer[static_cast<std::size_t>(i)],
                    act.layer[static_cast<std::size_t>(i)]);
  }
  std::snprintf(name, sizeof(name), "%s final-norm[H]", label);
  rc |= cmp_exact(name, ref.norm, act.norm);
  std::snprintf(name, sizeof(name), "%s FULL logits[vocab=%d]", label,
                cfg.vocab_size);
  rc |= cmp_exact(name, ref.logits, act.logits);
  return rc;
}

// Prefill `prompt` into a fresh sequence of `mgr` via the SERIAL single path
// (the Phase B contract: prefill is serial; the last prompt token produces
// the decode-ready state).
int prefill_serial(Qwen35Model& model, Qwen35StateManager& mgr,
                   const std::vector<int>& prompt, SequenceId* sid,
                   cudaStream_t stream) {
  Status s = mgr.create_sequence(sid);
  if (!s.ok) return 1;
  for (int tok : prompt) {
    s = model.forward_token_with_state(tok, *sid, mgr, stream);
    if (!s.ok) {
      std::fprintf(stderr, "  [prefill] %s\n", s.message.c_str());
      return 1;
    }
  }
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
    std::fprintf(stderr, "[SKIP] batched decode: no preconverted model at %s\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] batched decode: checkpoint absent at %s\n",
                 ckpt.c_str());
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
  const int kB = 3;
  std::fprintf(stderr,
               "[batched-decode] model loaded: %d layers, vocab %d, "
               "page_tokens %d, pool pages %d, delta slots %d\n",
               model.num_layers(), cfg.vocab_size, kPageTokens, kPoolPages,
               kDeltaSlots);
  int rc = 0;

  // =========================================================================
  // Part 1: B = 3, heterogeneous positions, TWO decode steps. Batch side and
  // three independent serial references (one per row), all from equivalent
  // states. Per-row outputs + final hybrid state are BIT-IDENTICAL.
  // =========================================================================
  {
    Qwen35StateManager mgrB(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    SequenceId sB[3];
    CHECK_EQ(prefill_serial(model, mgrB, kPrompt0, &sB[0], stream), 0);
    CHECK_EQ(prefill_serial(model, mgrB, kPrompt1, &sB[1], stream), 0);
    CHECK_EQ(prefill_serial(model, mgrB, kPrompt2, &sB[2], stream), 0);
    const std::vector<int> kPrompt[3] = {kPrompt0, kPrompt1, kPrompt2};

    Qwen35StateManager mgrS0(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    Qwen35StateManager mgrS1(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    Qwen35StateManager mgrS2(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    Qwen35StateManager* mgrS[3] = {&mgrS0, &mgrS1, &mgrS2};
    SequenceId sS[3];
    for (int i = 0; i < 3; ++i)
      CHECK_EQ(prefill_serial(model, *mgrS[i], kPrompt[i], &sS[i], stream), 0);

    for (int step = 0; step < 2; ++step) {
      const int* toks = (step == 0) ? kStep1Tok : kStep2Tok;
      // ---- batch decode (ONE traversal for the whole cohort) ----
      s = model.forward_batch_with_state(toks, sB, kB, mgrB, stream);
      CHECK(s.ok);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      RowOut brow[3];
      for (int i = 0; i < 3; ++i)
        CHECK_EQ(capture_batch_row(model, kB, i, stream, &brow[i]), 0);
      StateSnap bstate[3];
      for (int i = 0; i < 3; ++i) {
        const SequenceState* rec = mgrB.lookup(sB[i]);
        CHECK(rec != nullptr);
        const int ntok = static_cast<int>(kPrompt[i].size()) + (step + 1);
        CHECK_EQ(capture_state(mgrB, *rec, cfg, ntok, stream, &bstate[i]), 0);
      }
      // ---- serial reference (frozen single path) per row ----
      for (int i = 0; i < 3; ++i) {
        s = model.forward_token_with_state(toks[i], sS[i], *mgrS[i], stream);
        CHECK(s.ok);
        RowOut srow;
        CHECK_EQ(capture_single(model, stream, &srow), 0);
        StateSnap sstate;
        const SequenceState* rec = mgrS[i]->lookup(sS[i]);
        CHECK(rec != nullptr);
        const int ntok = static_cast<int>(kPrompt[i].size()) + (step + 1);
        CHECK_EQ(capture_state(*mgrS[i], *rec, cfg, ntok, stream, &sstate), 0);
        char label[40];
        std::snprintf(label, sizeof(label), "step%d row%d", step + 1, i);
        rc |= cmp_row(label, srow, brow[i], cfg);
        rc |= cmp_state(label, sstate, bstate[i], ntok, cfg);
      }
    }
    for (int i = 0; i < 3; ++i) CHECK(mgrS[i]->retire_sequence(sS[i]).ok);
    CHECK(mgrB.retire_sequence(sB[0]).ok);
    CHECK(mgrB.retire_sequence(sB[1]).ok);
    CHECK(mgrB.retire_sequence(sB[2]).ok);
  }

  // =========================================================================
  // Part 2: B = 1 (a batch of one is bit-identical to the single path).
  // =========================================================================
  {
    Qwen35StateManager mgrB(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    SequenceId sB;
    CHECK_EQ(prefill_serial(model, mgrB, kPromptB1, &sB, stream), 0);
    Qwen35StateManager mgrS(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    SequenceId sS;
    CHECK_EQ(prefill_serial(model, mgrS, kPromptB1, &sS, stream), 0);

    int tok = kB1Tok;
    s = model.forward_batch_with_state(&tok, &sB, 1, mgrB, stream);
    CHECK(s.ok);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    RowOut brow;
    CHECK_EQ(capture_batch_row(model, 1, 0, stream, &brow), 0);
    StateSnap bstate;
    const SequenceState* recB = mgrB.lookup(sB);
    CHECK(recB != nullptr);
    CHECK_EQ(capture_state(mgrB, *recB, cfg, static_cast<int>(kPromptB1.size()) + 1,
                           stream, &bstate), 0);

    s = model.forward_token_with_state(kB1Tok, sS, mgrS, stream);
    CHECK(s.ok);
    RowOut srow;
    CHECK_EQ(capture_single(model, stream, &srow), 0);
    StateSnap sstate;
    const SequenceState* recS = mgrS.lookup(sS);
    CHECK(recS != nullptr);
    CHECK_EQ(capture_state(mgrS, *recS, cfg, static_cast<int>(kPromptB1.size()) + 1,
                           stream, &sstate), 0);
    rc |= cmp_row("B1 row0", srow, brow, cfg);
    rc |= cmp_state("B1 row0", sstate, bstate,
                    static_cast<int>(kPromptB1.size()) + 1, cfg);
    CHECK(mgrS.retire_sequence(sS).ok);
    CHECK(mgrB.retire_sequence(sB).ok);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_batched_decode: FAIL\n");
    return rc;
  }
  std::printf(
      "test_qwen35_batched_decode: PASS (B=3 heterogeneous 2-step + B=1: "
      "per-row embedding / 24 layer-finals / final-norm / FULL logits / final "
      "hybrid state / length all BIT-IDENTICAL to the frozen single path)\n");
  return 0;
}
