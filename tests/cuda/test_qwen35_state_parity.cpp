// CUDALM — v0.5 Phase B: single-sequence external-state parity hard gate
// (CUDA, real checkpoint).
//
// Proves, with the REAL Qwen3.5-0.8B-Base checkpoint:
//
//   A: frozen legacy Qwen35Model (owned per-layer state, forward_token)
//   B: fresh Qwen35StateManager sequence + external-state forward
//      (forward_token_with_state: DeltaNet forward_with_state over the
//      DeltaStatePool slot + full-attention forward_with_paged_state over
//      the KvPagePool pages + device block table)
//
// run on the SAME 6 deterministic tokens (p0..p5, crossing two KV page
// boundaries at the small page_tokens = 2) are BIT-IDENTICAL:
//   * every step: embedding output, all 24 layer final outputs, final-norm
//     output, FULL logits [248320];
//   * final hybrid state: 18x DeltaNet conv (bf16) + recurrent (fp32) and
//     6x full-attention active K/V logical rows 0..p5 (read from the pool's
//     physical pages THROUGH the block table — never a host-side gather).
// Bit-exactness is the gate (atol = 0): both paths share the same frozen
// kernels, and the paged kernels keep the frozen accumulation order (see
// paged_kv.h) — any non-zero difference is a real bug.
//
// Plus the Phase B lifecycle gates:
//   * COMPATIBILITY GATES: a manager built for a DIFFERENT config (n_kv
//     heads 4 vs 2) is rejected before ANY mutation (no KV page, no Delta
//     mutation, length unchanged); a forward on a DIFFERENT CUDA stream
//     (v0.5 single-stream contract) is rejected the same way — all 48
//     state items bit-identical + accounting unchanged;
//   * OOM-BEFORE-MUTATION: with a 3-page pool (6 tokens at pt=2 exhaust it),
//     the 7th forward_token_with_state fails with a Status OOM and NOTHING
//     changed: length, block-table num_blocks, pool used/free, every
//     DeltaNet conv/recurrent byte, every active KV row;
//   * RESET PARITY: reset_sequence() + re-running the SAME 6 tokens is
//     bit-identical to the first external run (outputs, state, length) —
//     Phase A lifecycle + Phase B integration leave no contamination;
//   * FAIL LOUD: unknown/retired sequence id and invalid token_id return
//     Status errors with no state change.
//
// Self-skips (77) when the checkpoint is absent; in the Phase B evidence
// environment the checkpoint IS present and this test actually runs
// (77 is not a sign-off).

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

// Deterministic token sequence (p0..p5). page_tokens = 2, so positions 2
// and 4 cross KV page boundaries (blocks 1 and 2 are acquired mid-run).
static const int kTokens[] = {1024, 2048, 3072, 15, 16, 17};
static const int kN = 6;
static const int kPageTokens = 2;
static const int kPoolPages = 3;  // exactly 6 tokens at pt=2 -> 7th OOMs

// Device -> host capture helpers (explicit cudaMemcpyDeviceToHost; the
// targets are HOST std::vector buffers).
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

// One captured model output for a single forward step.
struct StepOut {
  std::vector<__nv_bfloat16> embed;   // [H]
  std::vector<__nv_bfloat16> layers;  // [24][H] in layer order
  std::vector<__nv_bfloat16> norm;    // [H]
  std::vector<__nv_bfloat16> logits;  // [vocab]
};

void capture_step(Qwen35Model& model, const Qwen35Config& cfg,
                  cudaStream_t stream, StepOut* out) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const int H = cfg.hidden_size;
  const int L = model.num_layers();
  out->embed = d2h_bf16(model.embedding_output(), H, stream);
  out->layers.resize(static_cast<std::size_t>(L) * H);
  for (int i = 0; i < L; ++i) {
    std::vector<__nv_bfloat16> row =
        d2h_bf16(model.layer_final_output(i), H, stream);
    std::memcpy(out->layers.data() +
                    static_cast<std::size_t>(i) * H,
                row.data(), H * sizeof(__nv_bfloat16));
  }
  out->norm = d2h_bf16(model.final_norm_output(), H, stream);
  out->logits = d2h_bf16(model.logits(), cfg.vocab_size, stream);
}

// Bit-exact comparison (the Phase B gate: same kernels, same op order).
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

// Capture the full hybrid final state of BOTH paths for bit-exact compare:
//   legacy: layer-owned DeltaNet conv/rec + contiguous KV rows 0..n-1
//   external: pool slot conv/rec + pool pages via the host block table
struct StateSnap {
  std::vector<std::vector<__nv_bfloat16>> conv;  // [24] (linear layers)
  std::vector<std::vector<float>> rec;           // [24]
  std::vector<std::vector<__nv_bfloat16>> kvk;   // [24] rows 0..n-1, all kv
  std::vector<std::vector<__nv_bfloat16>> kvv;   // [24]
};

void capture_state_legacy(Qwen35Model& model, const Qwen35Config& cfg,
                          int n_tokens, cudaStream_t stream, StateSnap* out) {
  const int H = cfg.hidden_size;
  const int L = model.num_layers();
  const int nkv = cfg.n_kv_heads;
  const int hd = cfg.head_dim;
  out->conv.resize(L);
  out->rec.resize(L);
  out->kvk.resize(L);
  out->kvv.resize(L);
  for (int i = 0; i < L; ++i) {
    if (model.is_linear_attention(i)) {
      Qwen35DeltaNetLayer& d = *model.delta(i);
      const std::size_t cn =
          static_cast<std::size_t>(cfg.linear_conv_dim()) *
          cfg.linear_conv_state_len();
      const std::size_t rn = static_cast<std::size_t>(cfg.lin_num_v_heads) *
                             cfg.lin_value_head_dim * cfg.lin_value_head_dim;
      out->conv[static_cast<std::size_t>(i)] = d2h_bf16(d.conv_state(), cn, stream);
      out->rec[static_cast<std::size_t>(i)] = d2h_f32(d.recurrent_state(), rn, stream);
    } else {
      // Contiguous cache layout is [n_kv][max_seq_len][head_dim]: logical
      // row t of head n sits at (n * max_seq_len + t) * head_dim. Gather the
      // rows into the same (n * n_tokens + t) order the external capture
      // produces (direct device -> host per row).
      Qwen35KvCache& kv = model.attention(i)->kv_cache();
      const std::size_t rn =
          static_cast<std::size_t>(nkv) * n_tokens * hd;
      const std::size_t msl = static_cast<std::size_t>(cfg.max_seq_len);
      std::vector<__nv_bfloat16> k(rn), v(rn);
      for (int t = 0; t < n_tokens; ++t)
        for (int n = 0; n < nkv; ++n) {
          const std::size_t dst =
              (static_cast<std::size_t>(n) * n_tokens + t) * hd;
          const std::size_t src =
              (static_cast<std::size_t>(n) * msl + t) * hd;
          const std::size_t bytes = hd * sizeof(__nv_bfloat16);
          CUDA_CHECK(cudaMemcpyAsync(k.data() + dst, kv.k() + src, bytes,
                                     cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaMemcpyAsync(v.data() + dst, kv.v() + src, bytes,
                                     cudaMemcpyDeviceToHost, stream));
        }
      out->kvk[static_cast<std::size_t>(i)] = std::move(k);
      out->kvv[static_cast<std::size_t>(i)] = std::move(v);
    }
  }
  (void)H;
}

// Returns 1 if a block-table lookup came back unallocated (cannot happen for
// a live sequence whose length covers the requested rows).
int capture_state_external(const Qwen35StateManager& mgr,
                           const SequenceState& rec, const Qwen35Config& cfg,
                           int n_tokens, cudaStream_t stream,
                           StateSnap* out) {
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
      const std::size_t rn =
          static_cast<std::size_t>(nkv) * n_tokens * hd;
      std::vector<__nv_bfloat16> k(rn), v(rn);
      // Read the LOGICAL rows 0..n_tokens-1 from the physical pages THROUGH
      // the block table: each row is a direct device -> host copy (the
      // page ids come from the host block table, exactly like the kernels
      // use them — no host-side KV gather, no device staging).
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
              int n_tokens, const Qwen35Config& cfg) {
  int rc = 0;
  const int L = cfg.num_hidden_layers;
  const int nkv = cfg.n_kv_heads;
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
  (void)nkv;
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
    std::fprintf(stderr, "[SKIP] qwen35 state parity: no preconverted model "
                         "at %s and --no-convert given\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] qwen35 state parity: checkpoint absent at "
                         "%s\n",
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
               "[parity] model loaded: %d layers, vocab %d, max_seq %d, "
               "page_tokens %d, pool pages %d\n",
               model.num_layers(), cfg.vocab_size, cfg.max_seq_len,
               kPageTokens, kPoolPages);

  // -------------------------------------------------------------------
  // A: frozen legacy owned-state run (p0..p5).
  // -------------------------------------------------------------------
  std::vector<StepOut> A(kN);
  model.reset_state(stream);
  for (int i = 0; i < kN; ++i) {
    model.forward_token(kTokens[i], i, stream);
    capture_step(model, cfg, stream, &A[static_cast<std::size_t>(i)]);
  }
  StateSnap A_state;
  capture_state_legacy(model, cfg, kN, stream, &A_state);

  // -------------------------------------------------------------------
  // B: external-state run (manager + pools, SAME tokens).
  // -------------------------------------------------------------------
  Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages,
                         /*delta_capacity_slots=*/4, stream);
  SequenceId sid = 0;
  s = mgr.create_sequence(&sid);
  CHECK(s.ok);
  std::fprintf(stderr, "[parity] external sequence id=%llu\n",
               static_cast<unsigned long long>(sid));

  std::vector<StepOut> B(kN);
  for (int i = 0; i < kN; ++i) {
    // position is DERIVED from the sequence length inside the call.
    s = model.forward_token_with_state(kTokens[i], sid, mgr, stream);
    if (!s.ok) {
      std::fprintf(stderr, "  forward_token_with_state step %d: %s\n", i,
                   s.message.c_str());
      return 1;
    }
    capture_step(model, cfg, stream, &B[static_cast<std::size_t>(i)]);
  }
  const SequenceState* rec = mgr.lookup(sid);
  CHECK(rec != nullptr);
  CHECK_EQ(rec->length, kN);
  StateSnap B_state;
  CHECK_EQ(capture_state_external(mgr, *rec, cfg, kN, stream, &B_state), 0);

  // ---- per-step output parity (bit-exact) ------------------------------
  int rc = 0;
  for (int i = 0; i < kN; ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "step p%d embedding", i);
    rc |= cmp_exact(name, A[static_cast<std::size_t>(i)].embed,
                    B[static_cast<std::size_t>(i)].embed);
    for (int l = 0; l < model.num_layers(); ++l) {
      std::snprintf(name, sizeof(name), "step p%d layer %d final", i, l);
      rc |= cmp_exact(
          name,
          std::vector<__nv_bfloat16>(
              A[static_cast<std::size_t>(i)].layers.begin() +
                  static_cast<std::size_t>(l) * cfg.hidden_size,
              A[static_cast<std::size_t>(i)].layers.begin() +
                  static_cast<std::size_t>(l + 1) * cfg.hidden_size),
          std::vector<__nv_bfloat16>(
              B[static_cast<std::size_t>(i)].layers.begin() +
                  static_cast<std::size_t>(l) * cfg.hidden_size,
              B[static_cast<std::size_t>(i)].layers.begin() +
                  static_cast<std::size_t>(l + 1) * cfg.hidden_size));
    }
    std::snprintf(name, sizeof(name), "step p%d final_norm", i);
    rc |= cmp_exact(name, A[static_cast<std::size_t>(i)].norm,
                    B[static_cast<std::size_t>(i)].norm);
    std::snprintf(name, sizeof(name), "step p%d logits[%d]", i,
                  cfg.vocab_size);
    rc |= cmp_exact(name, A[static_cast<std::size_t>(i)].logits,
                    B[static_cast<std::size_t>(i)].logits);
  }

  // ---- final hybrid state parity (bit-exact) ----------------------------
  rc |= cmp_state("state", A_state, B_state, kN, cfg);

  // -------------------------------------------------------------------
  // OOM-BEFORE-MUTATION gate: the pool is exhausted (3 pages = 6 tokens
  // at pt=2); the 7th forward must fail with NO state change.
  // -------------------------------------------------------------------
  {
    StateSnap pre;
    CHECK_EQ(capture_state_external(mgr, *rec, cfg, kN, stream, &pre), 0);
    const int pre_len = rec->length;
    const int pre_blocks = rec->block_table.num_blocks();
    const std::size_t pre_used = mgr.used_state_bytes();
    const std::vector<int> pre_live = mgr.kv_pool().live_pages();

    s = model.forward_token_with_state(1024, sid, mgr, stream);
    CHECK(!s.ok);  // loud OOM
    std::fprintf(stderr, "  [ok] 7th forward rejected: %s\n", s.message.c_str());

    StateSnap post;
    CHECK_EQ(capture_state_external(mgr, *rec, cfg, kN, stream, &post), 0);
    rc |= cmp_state("oom.no-change", pre, post, kN, cfg);
    CHECK_EQ(rec->length, pre_len);
    CHECK_EQ(rec->block_table.num_blocks(), pre_blocks);
    CHECK_EQ(mgr.used_state_bytes(), pre_used);
    CHECK(mgr.kv_pool().live_pages() == pre_live);
    std::fprintf(stderr, "  [ok] OOM-before-mutation: length=%d, blocks=%d, "
                         "used bytes + live pages unchanged\n",
                 pre_len, pre_blocks);
  }

  // ---- fail loud: unknown id + invalid token ----------------------------
  {
    s = model.forward_token_with_state(kTokens[0], 999, mgr, stream);
    CHECK(!s.ok);
    std::fprintf(stderr, "  [ok] unknown sequence rejected: %s\n",
                 s.message.c_str());
    const int len_before = mgr.lookup(sid)->length;
    s = model.forward_token_with_state(cfg.vocab_size + 7, sid, mgr, stream);
    CHECK(!s.ok);
    CHECK_EQ(mgr.lookup(sid)->length, len_before);
    std::fprintf(stderr, "  [ok] invalid token rejected: %s\n",
                 s.message.c_str());
  }

  // -------------------------------------------------------------------
  // COMPATIBILITY GATES (before ANY state mutation): a manager whose
  // CONFIG or whose STREAM does not match the model is rejected fail-loud
  // with ZERO mutation — no KV allocation, no Delta mutation, length
  // unchanged, no layer forward.
  // -------------------------------------------------------------------
  {
    // ---- mismatched manager config -> rejected before mutation ---------
    {
      Qwen35Config bad_cfg = cfg;
      bad_cfg.n_kv_heads = 4;  // the loaded model config is 2 -> mismatch
      Qwen35StateManager bad_mgr(bad_cfg, kPageTokens, kPoolPages,
                                 /*delta_capacity_slots=*/4, stream);
      SequenceId bad_sid = 0;
      s = bad_mgr.create_sequence(&bad_sid);
      CHECK(s.ok);
      const std::size_t pre_used = bad_mgr.used_state_bytes();
      const std::vector<int> pre_live = bad_mgr.kv_pool().live_pages();
      const int pre_delta_slots = bad_mgr.delta_pool().used_slots();

      s = model.forward_token_with_state(kTokens[0], bad_sid, bad_mgr,
                                         stream);
      CHECK(!s.ok);
      std::fprintf(stderr, "  [ok] config-mismatched manager rejected: %s\n",
                   s.message.c_str());

      // Zero mutation on the mismatched manager: no KV page allocated, no
      // Delta slot consumed by the forward, length unchanged.
      CHECK_EQ(bad_mgr.used_state_bytes(), pre_used);
      CHECK(bad_mgr.kv_pool().live_pages() == pre_live);
      CHECK_EQ(bad_mgr.kv_pool().used_pages(), 0);
      CHECK_EQ(bad_mgr.delta_pool().used_slots(), pre_delta_slots);
      CHECK_EQ(bad_mgr.lookup(bad_sid)->length, 0);
    }
    // ---- mismatched CUDA stream -> rejected before mutation ------------
    {
      cudaStream_t other = nullptr;
      CUDA_CHECK(cudaStreamCreate(&other));
      StateSnap pre;
      CHECK_EQ(capture_state_external(mgr, *rec, cfg, kN, stream, &pre), 0);
      const int pre_len = rec->length;
      const int pre_blocks = rec->block_table.num_blocks();
      const std::size_t pre_used = mgr.used_state_bytes();
      const std::vector<int> pre_live = mgr.kv_pool().live_pages();

      s = model.forward_token_with_state(kTokens[0], sid, mgr, other);
      CHECK(!s.ok);
      std::fprintf(stderr, "  [ok] stream-mismatched forward rejected: %s\n",
                   s.message.c_str());

      // Zero mutation on the (config-matching) manager: all 48 state items
      // bit-identical + accounting unchanged.
      StateSnap post;
      CHECK_EQ(capture_state_external(mgr, *rec, cfg, kN, stream, &post), 0);
      rc |= cmp_state("gate.no-change", pre, post, kN, cfg);
      CHECK_EQ(rec->length, pre_len);
      CHECK_EQ(rec->block_table.num_blocks(), pre_blocks);
      CHECK_EQ(mgr.used_state_bytes(), pre_used);
      CHECK(mgr.kv_pool().live_pages() == pre_live);
      CUDA_CHECK(cudaStreamDestroy(other));
    }
  }

  // -------------------------------------------------------------------
  // RESET PARITY: reset_sequence() + SAME 6 tokens == first external run.
  // -------------------------------------------------------------------
  {
    s = mgr.reset_sequence(sid);
    CHECK(s.ok);
    CHECK_EQ(mgr.lookup(sid)->length, 0);
    std::vector<StepOut> R(kN);
    for (int i = 0; i < kN; ++i) {
      s = model.forward_token_with_state(kTokens[i], sid, mgr, stream);
      if (!s.ok) {
        std::fprintf(stderr, "  reset re-run step %d: %s\n", i,
                     s.message.c_str());
        return 1;
      }
      capture_step(model, cfg, stream, &R[static_cast<std::size_t>(i)]);
    }
    CHECK_EQ(mgr.lookup(sid)->length, kN);
    StateSnap R_state;
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sid), cfg, kN, stream, &R_state), 0);
    for (int i = 0; i < kN; ++i) {
      char name[64];
      std::snprintf(name, sizeof(name), "reset p%d logits[%d]", i,
                    cfg.vocab_size);
      rc |= cmp_exact(name, B[static_cast<std::size_t>(i)].logits,
                      R[static_cast<std::size_t>(i)].logits);
      std::snprintf(name, sizeof(name), "reset p%d layer %d final", i,
                    model.num_layers() - 1);
      rc |= cmp_exact(
          name,
          std::vector<__nv_bfloat16>(
              B[static_cast<std::size_t>(i)].layers.begin() +
                  static_cast<std::size_t>(model.num_layers() - 1) *
                      cfg.hidden_size,
              B[static_cast<std::size_t>(i)].layers.begin() +
                  static_cast<std::size_t>(model.num_layers()) *
                      cfg.hidden_size),
          std::vector<__nv_bfloat16>(
              R[static_cast<std::size_t>(i)].layers.begin() +
                  static_cast<std::size_t>(model.num_layers() - 1) *
                      cfg.hidden_size,
              R[static_cast<std::size_t>(i)].layers.begin() +
                  static_cast<std::size_t>(model.num_layers()) *
                      cfg.hidden_size));
    }
    rc |= cmp_state("reset.state", B_state, R_state, kN, cfg);
    std::fprintf(stderr, "  [ok] reset parity: re-run compared vs first "
                         "external run\n");
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_state_parity: FAIL\n");
    return rc;
  }
  std::fprintf(stderr, "test_qwen35_state_parity: PASS (legacy == external "
                       "bit-identical: 6 steps x outputs + hybrid state + "
                       "compat gates + OOM-before-mutation + reset parity)\n");
  return 0;
}
