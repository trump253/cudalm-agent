// CUDALM — v0.5 Phase C: multi-sequence INTERLEAVED external-state hard
// gate (CUDA, real checkpoint) — the v0.5 final sign-off test.
//
// Proves, with the REAL Qwen3.5-0.8B-Base checkpoint, that the
// external-state runtime (Qwen35Model::forward_token_with_state over the
// Qwen35StateManager pools) is strictly correct under INTERLEAVED
// multi-sequence execution:
//
//   * single model, single CUDA stream, ONE forward_token_with_state()
//     per step — only the CALL ORDER interleaves different SequenceIds
//     (explicit test script; NO scheduler, NO batching, NO admission
//     policy — that is v0.6+);
//
//   * TOKEN STREAMS (all crossing the page_tokens = 2 boundaries):
//       A = {1024, 2048, 3072, 4096, 5000, 6000}   (6 tokens -> 3 pages)
//       B = {15, 16, 17, 18, 19}                   (5 tokens -> 3 pages)
//       C = {7, 8, 9, 10, 11, 12}                  (6 tokens -> 3 pages)
//
//   * INDEPENDENT REFERENCES (fresh manager, ONE sequence alone):
//       refA / refB / refC — per step: 24 layer final outputs, final
//       norm, FULL logits[248320]; final: 18x DeltaNet conv (bf16) +
//       rec (fp32) + 6x full-attention LOGICAL K/V rows (read through
//       the block table, never a host gather) + length + block-table
//       shape (num_blocks).
//
//   * INTERLEAVED RUN (ONE manager, A + B live, non-trivial schedule):
//       A0 B0 A1 A2 B1 A3 B2 A4 B3 A5 B4
//       every A step BIT-IDENTICAL to refA, every B step BIT-IDENTICAL
//       to refB; final A/B hybrid states bit-identical to the references.
//
//   * ISOLATION GATE (key steps, FULL hybrid state pre/post — not just
//     length/accounting):
//       before A4: capture B (len 3) -> forward A4 -> B bit-identical;
//       before B2: capture A (len 4) -> forward B2 -> A bit-identical.
//
//   * RETIRE / REUSE CONTAMINATION GATE:
//       retire A -> A's SequenceId PERMANENTLY invalid (lookup nullptr,
//       advance / ensure_kv_capacity rejected), A's 3 KV pages + Delta
//       slot reclaimed, B's state bit-identical; create C -> C's id !=
//       A's id, C ACTUALLY reuses A's freed Delta slot and A's freed KV
//       physical pages (set equality); C runs its stream from fresh-zero
//       state: per-step outputs + final Delta state + logical KV
//       bit-identical to refC; C's run leaves the still-live B
//       bit-identical (full state pre/post, incl. a key step).
//
//   * RESET-ONE-SEQUENCE ISOLATION:
//       reset_sequence(B) clears ONLY B (length 0, all pages released,
//       Delta slot zeroed in place) with the live C state bit-identical.
//
// Everything is runtime-vs-runtime and BIT-EXACT (atol = 0 / memcmp):
// interleaved calls share the SAME model weights and the SAME frozen
// kernels, so any non-zero difference is a real bug.
//
// Self-skips (77) when the checkpoint is absent; in the Phase C evidence
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
#include <set>
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

// ---- token streams + pool sizing ----------------------------------------
static const int kA[] = {1024, 2048, 3072, 4096, 5000, 6000};  // 6 tokens
static const int kB[] = {15, 16, 17, 18, 19};                  // 5 tokens
static const int kC[] = {7, 8, 9, 10, 11, 12};                 // 6 tokens
static const int kNA = 6;
static const int kNB = 5;
static const int kNC = 6;
static const int kPageTokens = 2;  // every stream crosses page boundaries
static const int kPoolPages = 6;   // A + B live simultaneously (3 + 3)
static const int kDeltaSlots = 4;

// Non-trivial A/B interleave schedule (11 steps).
struct SchedStep {
  char seq;  // 'A' or 'B'
  int idx;
};
static const SchedStep kABSchedule[] = {
    {'A', 0}, {'B', 0}, {'A', 1}, {'A', 2}, {'B', 1},
    {'A', 3}, {'B', 2}, {'A', 4}, {'B', 3}, {'A', 5},
    {'B', 4}};
static const int kNAB = 11;

// ---- device -> host capture helpers --------------------------------------
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

// One captured model output for a single forward step (the Phase C gate:
// 24 layer finals + final norm + FULL logits).
struct StepOut {
  std::vector<__nv_bfloat16> layers;  // [24][H] in layer order
  std::vector<__nv_bfloat16> norm;    // [H]
  std::vector<__nv_bfloat16> logits;  // [vocab]
};

void capture_step(Qwen35Model& model, const Qwen35Config& cfg,
                  cudaStream_t stream, StepOut* out) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const int H = cfg.hidden_size;
  const int L = model.num_layers();
  out->layers.resize(static_cast<std::size_t>(L) * H);
  for (int i = 0; i < L; ++i) {
    std::vector<__nv_bfloat16> row =
        d2h_bf16(model.layer_final_output(i), H, stream);
    std::memcpy(out->layers.data() + static_cast<std::size_t>(i) * H,
                row.data(), H * sizeof(__nv_bfloat16));
  }
  out->norm = d2h_bf16(model.final_norm_output(), H, stream);
  out->logits = d2h_bf16(model.logits(), cfg.vocab_size, stream);
}

// Bit-exact comparison (the Phase C gate: same kernels, same op order).
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
  std::vector<std::vector<__nv_bfloat16>> conv;  // [24] (linear layers)
  std::vector<std::vector<float>> rec;           // [24]
  std::vector<std::vector<__nv_bfloat16>> kvk;   // [24] rows 0..n-1, all kv
  std::vector<std::vector<__nv_bfloat16>> kvv;   // [24]
};

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
          const std::size_t dst = (static_cast<std::size_t>(n) * n_tokens +
                                   t) *
                                  hd;
          const __nv_bfloat16* kp = mgr.kv_pool().k_page(ord, page) +
                                    (static_cast<std::size_t>(n) * pt + off) *
                                        hd;
          const __nv_bfloat16* vp = mgr.kv_pool().v_page(ord, page) +
                                    (static_cast<std::size_t>(n) * pt + off) *
                                        hd;
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

// Verify the Delta slot of `slot` is ALL ZERO in every linear layer
// (18 ordinals x conv bf16 + rec fp32). Used for the reset isolation gate.
int check_delta_zero(const Qwen35StateManager& mgr, int slot,
                     const Qwen35Config& cfg, cudaStream_t stream) {
  const __nv_bfloat16 z16 = __float2bfloat16_rn(0.0f);
  for (int i = 0; i < cfg.num_hidden_layers; ++i) {
    if (!cfg.is_linear_attention(i)) continue;
    const int ord = mgr.delta_pool().linear_layer_ordinal(i);
    const std::size_t cn =
        static_cast<std::size_t>(cfg.linear_conv_dim()) *
        cfg.linear_conv_state_len();
    const std::size_t rn = static_cast<std::size_t>(cfg.lin_num_v_heads) *
                           cfg.lin_value_head_dim * cfg.lin_value_head_dim;
    std::vector<__nv_bfloat16> c =
        d2h_bf16(mgr.delta_pool().conv(ord, slot), cn, stream);
    for (std::size_t j = 0; j < c.size(); ++j)
      if (c[j] != z16) {
        std::fprintf(stderr, "  [delta-zero] L%d conv nonzero\n", i);
        return 1;
      }
    std::vector<float> r =
        d2h_f32(mgr.delta_pool().recurrent(ord, slot), rn, stream);
    for (std::size_t j = 0; j < r.size(); ++j)
      if (r[j] != 0.0f) {
        std::fprintf(stderr, "  [delta-zero] L%d rec nonzero\n", i);
        return 1;
      }
  }
  return 0;
}

// Physical page ids of a live sequence's block table (host copy).
std::vector<int> table_pages(const SequenceState& rec) {
  return std::vector<int>(rec.block_table.page_ids(),
                          rec.block_table.page_ids() +
                              rec.block_table.num_blocks());
}

// Sorted multiset equality over page-id vectors (allocation order may
// differ from live_pages()' ascending order).
bool pages_equal_sorted(const std::vector<int>& a, const std::vector<int>& b) {
  if (a.size() != b.size()) return false;
  std::vector<int> x(a), y(b);
  std::sort(x.begin(), x.end());
  std::sort(y.begin(), y.end());
  return x == y;
}

// ---- independent reference run: fresh manager, ONE sequence alone -------
struct RefRun {
  std::vector<StepOut> steps;
  StateSnap state;
  int length = 0;
  int num_blocks = 0;
};

int run_reference(Qwen35Model& model, const Qwen35Config& cfg,
                  const char* label, const int* tokens, int n,
                  cudaStream_t stream, RefRun* out) {
  Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  SequenceId sid = 0;
  Status s = mgr.create_sequence(&sid);
  CHECK(s.ok);
  out->steps.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    s = model.forward_token_with_state(tokens[i], sid, mgr, stream);
    if (!s.ok) {
      std::fprintf(stderr, "  ref %s step %d: %s\n", label, i,
                   s.message.c_str());
      return 1;
    }
    capture_step(model, cfg, stream, &out->steps[static_cast<std::size_t>(i)]);
  }
  const SequenceState* rec = mgr.lookup(sid);
  CHECK(rec != nullptr);
  out->length = rec->length;
  out->num_blocks = rec->block_table.num_blocks();
  CHECK_EQ(capture_state_external(mgr, *rec, cfg, n, stream, &out->state), 0);
  return 0;
}

// Compare one interleaved StepOut stream against its reference (per step:
// 24 layer finals + final norm + FULL logits).
int cmp_steps(const char* label, const std::vector<StepOut>& ref,
              const std::vector<StepOut>& act) {
  int rc = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s p%d layers[24][H]", label,
                  static_cast<int>(i));
    rc |= cmp_exact(name, ref[i].layers, act[i].layers);
    std::snprintf(name, sizeof(name), "%s p%d final_norm", label,
                  static_cast<int>(i));
    rc |= cmp_exact(name, ref[i].norm, act[i].norm);
    std::snprintf(name, sizeof(name), "%s p%d logits[vocab]", label,
                  static_cast<int>(i));
    rc |= cmp_exact(name, ref[i].logits, act[i].logits);
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
    std::fprintf(stderr, "[SKIP] qwen35 state interleave: no preconverted "
                         "model at %s and --no-convert given\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] qwen35 state interleave: checkpoint absent "
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
  std::fprintf(stderr,
               "[interleave] model loaded: %d layers, vocab %d, page_tokens "
               "%d, pool pages %d, delta slots %d\n",
               model.num_layers(), cfg.vocab_size, kPageTokens, kPoolPages,
               kDeltaSlots);

  int rc = 0;

  // -------------------------------------------------------------------
  // INDEPENDENT REFERENCES (fresh manager, one sequence alone).
  // -------------------------------------------------------------------
  RefRun refA, refB, refC;
  CHECK_EQ(run_reference(model, cfg, "A", kA, kNA, stream, &refA), 0);
  CHECK_EQ(run_reference(model, cfg, "B", kB, kNB, stream, &refB), 0);
  CHECK_EQ(run_reference(model, cfg, "C", kC, kNC, stream, &refC), 0);
  CHECK_EQ(refA.length, kNA);
  CHECK_EQ(refB.length, kNB);
  CHECK_EQ(refC.length, kNC);
  std::fprintf(stderr, "[interleave] references done (A %d, B %d, C %d "
                       "steps)\n",
               kNA, kNB, kNC);

  // -------------------------------------------------------------------
  // INTERLEAVED RUN: one manager, A + B live, schedule
  //   A0 B0 A1 A2 B1 A3 B2 A4 B3 A5 B4
  // with ISOLATION checks (full hybrid state pre/post of the OTHER
  // sequence) at the A4 and B2 steps.
  // -------------------------------------------------------------------
  {
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    SequenceId sid_a = 0, sid_b = 0;
    s = mgr.create_sequence(&sid_a);
    CHECK(s.ok);
    s = mgr.create_sequence(&sid_b);
    CHECK(s.ok);
    std::fprintf(stderr, "[interleave] A id=%llu, B id=%llu\n",
                 static_cast<unsigned long long>(sid_a),
                 static_cast<unsigned long long>(sid_b));

    std::vector<StepOut> stepA(kNA), stepB(kNB);
    for (int si = 0; si < kNAB; ++si) {
      const SchedStep& st = kABSchedule[static_cast<std::size_t>(si)];
      const SequenceState* rec_a = mgr.lookup(sid_a);
      const SequenceState* rec_b = mgr.lookup(sid_b);
      CHECK(rec_a != nullptr);
      CHECK(rec_b != nullptr);

      // ISOLATION GATE: full hybrid state of the OTHER sequence pre/post.
      StateSnap other_pre, other_post;
      int other_len = -1;
      const SequenceState* other = nullptr;
      if (st.seq == 'A' && st.idx == 4) {          // before A4: B at len 3
        other = rec_b;
        other_len = rec_b->length;
        CHECK_EQ(capture_state_external(mgr, *other, cfg, other_len, stream,
                                        &other_pre),
                 0);
      } else if (st.seq == 'B' && st.idx == 2) {   // before B2: A at len 4
        other = rec_a;
        other_len = rec_a->length;
        CHECK_EQ(capture_state_external(mgr, *other, cfg, other_len, stream,
                                        &other_pre),
                 0);
      }

      const int token =
          (st.seq == 'A') ? kA[st.idx] : kB[st.idx];
      const SequenceId sid = (st.seq == 'A') ? sid_a : sid_b;
      s = model.forward_token_with_state(token, sid, mgr, stream);
      if (!s.ok) {
        std::fprintf(stderr, "  interleave %c%d: %s\n", st.seq, st.idx,
                     s.message.c_str());
        return 1;
      }
      if (st.seq == 'A')
        capture_step(model, cfg, stream, &stepA[static_cast<std::size_t>(st.idx)]);
      else
        capture_step(model, cfg, stream, &stepB[static_cast<std::size_t>(st.idx)]);

      if (other != nullptr) {
        CHECK_EQ(capture_state_external(mgr, *other, cfg, other_len, stream,
                                        &other_post),
                 0);
        rc |= cmp_state(st.seq == 'A' ? "iso.B-during-A4"
                                      : "iso.A-during-B2",
                        other_pre, other_post, other_len, cfg);
        std::fprintf(stderr, "  [ok] isolation: %s state unchanged across "
                             "forward %c%d\n",
                     st.seq == 'A' ? "B" : "A", st.seq, st.idx);
      }
    }

    // ---- per-step parity vs the independent references ----------------
    rc |= cmp_steps("A.interleave", refA.steps, stepA);
    rc |= cmp_steps("B.interleave", refB.steps, stepB);

    // ---- final hybrid state + block-table shape -----------------------
    const SequenceState* rec_a = mgr.lookup(sid_a);
    const SequenceState* rec_b = mgr.lookup(sid_b);
    CHECK(rec_a != nullptr);
    CHECK(rec_b != nullptr);
    CHECK_EQ(rec_a->length, kNA);
    CHECK_EQ(rec_b->length, kNB);
    CHECK_EQ(rec_a->block_table.num_blocks(), refA.num_blocks);
    CHECK_EQ(rec_b->block_table.num_blocks(), refB.num_blocks);
    // A's and B's physical pages: 3 each, unique within and disjoint
    // across sequences (one shared table id space per pool).
    const std::vector<int> pa = table_pages(*rec_a);
    const std::vector<int> pb = table_pages(*rec_b);
    CHECK_EQ(pa.size(), 3u);
    CHECK_EQ(pb.size(), 3u);
    std::set<int> sa(pa.begin(), pa.end());
    std::set<int> sb(pb.begin(), pb.end());
    CHECK_EQ(sa.size(), pa.size());  // A unique
    CHECK_EQ(sb.size(), pb.size());  // B unique
    for (int p : pa) CHECK(sb.count(p) == 0);  // disjoint
    std::fprintf(stderr, "  [ok] A pages {%d,%d,%d}, B pages {%d,%d,%d} "
                         "unique + disjoint\n",
                 pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]);

    StateSnap stateA, stateB;
    CHECK_EQ(capture_state_external(mgr, *rec_a, cfg, kNA, stream, &stateA), 0);
    CHECK_EQ(capture_state_external(mgr, *rec_b, cfg, kNB, stream, &stateB), 0);
    rc |= cmp_state("A.final", refA.state, stateA, kNA, cfg);
    rc |= cmp_state("B.final", refB.state, stateB, kNB, cfg);

    // -----------------------------------------------------------------
    // RETIRE / REUSE CONTAMINATION GATE.
    // -----------------------------------------------------------------
    const int a_slot = rec_a->delta_slot;
    const int b_slot = rec_b->delta_slot;
    const std::vector<int> a_pages = table_pages(*rec_a);
    StateSnap b_before_retire, b_after_retire;
    CHECK_EQ(capture_state_external(mgr, *rec_b, cfg, kNB, stream,
                                    &b_before_retire),
             0);
    const std::size_t pre_used_bytes = mgr.used_state_bytes();

    s = mgr.retire_sequence(sid_a);
    CHECK(s.ok);
    // A's SequenceId is PERMANENTLY invalid.
    CHECK(mgr.lookup(sid_a) == nullptr);
    s = mgr.advance(sid_a, 0);
    CHECK(!s.ok);
    s = mgr.ensure_kv_capacity(sid_a, 0);
    CHECK(!s.ok);
    // A's resources reclaimed (EXACT): A's 3 pages released + 1 delta
    // slot freed; B's live pages and state untouched.
    CHECK_EQ(mgr.kv_pool().used_pages(), 3);
    CHECK(pages_equal_sorted(mgr.kv_pool().live_pages(), pb));
    CHECK_EQ(mgr.delta_pool().used_slots(), 1);
    CHECK_EQ(mgr.used_state_bytes(),
             pre_used_bytes -
                 3u * mgr.kv_pool().bytes_per_page() -
                 mgr.delta_pool().bytes_per_slot());
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sid_b), cfg, kNB, stream,
                                    &b_after_retire),
             0);
    rc |= cmp_state("retire.B-unchanged", b_before_retire, b_after_retire,
                    kNB, cfg);
    CHECK_EQ(mgr.lookup(sid_b)->length, kNB);
    std::fprintf(stderr, "  [ok] retire A: id permanently invalid, 3 pages + "
                         "slot %d reclaimed, B state unchanged\n",
                 a_slot);

    // Create C: id NEVER reissued; C ACTUALLY reuses A's freed Delta slot
    // and (as the only free pages) A's freed KV physical pages.
    SequenceId sid_c = 0;
    s = mgr.create_sequence(&sid_c);
    CHECK(s.ok);
    CHECK(sid_c != sid_a);
    const SequenceState* rec_c = mgr.lookup(sid_c);
    CHECK(rec_c != nullptr);
    CHECK_EQ(rec_c->delta_slot, a_slot);  // physical slot reuse
    std::fprintf(stderr, "  [ok] C id=%llu != A id=%llu, C reuses A's delta "
                         "slot %d\n",
                 static_cast<unsigned long long>(sid_c),
                 static_cast<unsigned long long>(sid_a), a_slot);

    // C runs its stream; B (still live) must stay bit-identical.
    StateSnap b_before_c, b_after_c;
    CHECK_EQ(capture_state_external(mgr, *rec_b, cfg, kNB, stream,
                                    &b_before_c),
             0);
    std::vector<StepOut> stepC(kNC);
    StateSnap c_step0_pre, c_step0_post;  // key-step isolation during C
    bool step0_isolation_done = false;
    for (int i = 0; i < kNC; ++i) {
      if (i == 0) {
        CHECK_EQ(capture_state_external(mgr, *rec_b, cfg, kNB, stream,
                                        &c_step0_pre),
                 0);
      }
      s = model.forward_token_with_state(kC[i], sid_c, mgr, stream);
      if (!s.ok) {
        std::fprintf(stderr, "  C step %d: %s\n", i, s.message.c_str());
        return 1;
      }
      capture_step(model, cfg, stream, &stepC[static_cast<std::size_t>(i)]);
      if (i == 0) {
        CHECK_EQ(capture_state_external(mgr, *rec_b, cfg, kNB, stream,
                                        &c_step0_post),
                 0);
        rc |= cmp_state("iso.B-during-C0", c_step0_pre, c_step0_post, kNB,
                        cfg);
        step0_isolation_done = true;
      }
    }
    CHECK(step0_isolation_done);
    rec_c = mgr.lookup(sid_c);  // refreshed (same id, length advanced)
    CHECK(rec_c != nullptr);
    CHECK_EQ(rec_c->length, kNC);
    CHECK_EQ(rec_c->block_table.num_blocks(), refC.num_blocks);

    // C per-step outputs vs the fresh-C reference (FULL logits + layers +
    // norm) and final hybrid state.
    rc |= cmp_steps("C.interleave", refC.steps, stepC);
    StateSnap stateC;
    CHECK_EQ(capture_state_external(mgr, *rec_c, cfg, kNC, stream, &stateC), 0);
    rc |= cmp_state("C.final", refC.state, stateC, kNC, cfg);

    // PHYSICAL reuse: C's 3 pages == A's freed 3 pages (set equality),
    // and they are the ONLY free pages that existed (A's exact set).
    const std::vector<int> pc = table_pages(*rec_c);
    std::set<int> sc(pc.begin(), pc.end());
    std::set<int> sa2(a_pages.begin(), a_pages.end());
    CHECK_EQ(sc.size(), 3u);
    CHECK(sc == sa2);
    // Live pages are exactly B's 3 + C's 3 (== A's old set).
    const std::vector<int> live_now = mgr.kv_pool().live_pages();
    std::set<int> live_set(live_now.begin(), live_now.end());
    std::set<int> expect_live = sa2;
    for (int p : pb) expect_live.insert(p);
    CHECK(live_set == expect_live);
    CHECK_EQ(mgr.delta_pool().used_slots(), 2);  // B + C
    std::fprintf(stderr, "  [ok] C pages {%d,%d,%d} == A's freed pages "
                         "{%d,%d,%d} (physical reuse)\n",
                 pc[0], pc[1], pc[2], a_pages[0], a_pages[1], a_pages[2]);

    // C's run left the still-live B bit-identical.
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sid_b), cfg, kNB, stream,
                                    &b_after_c),
             0);
    rc |= cmp_state("c-run.B-unchanged", b_before_c, b_after_c, kNB, cfg);
    std::fprintf(stderr, "  [ok] C run: fresh-state parity vs refC, B "
                         "unchanged\n");

    // -----------------------------------------------------------------
    // RESET-ONE-SEQUENCE ISOLATION: reset B clears ONLY B.
    // -----------------------------------------------------------------
    StateSnap c_before_reset, c_after_reset;
    CHECK_EQ(capture_state_external(mgr, *rec_c, cfg, kNC, stream,
                                    &c_before_reset),
             0);
    s = mgr.reset_sequence(sid_b);
    CHECK(s.ok);
    const SequenceState* rec_b2 = mgr.lookup(sid_b);
    CHECK(rec_b2 != nullptr);
    CHECK_EQ(rec_b2->length, 0);               // B length = 0
    CHECK_EQ(rec_b2->block_table.num_blocks(), 0);  // B pages released
    CHECK_EQ(mgr.kv_pool().used_pages(), 3);   // only C's 3 pages left
    CHECK(pages_equal_sorted(mgr.kv_pool().live_pages(), pc));
    CHECK_EQ(mgr.delta_pool().used_slots(), 2);  // slots not released, ...
    CHECK_EQ(check_delta_zero(mgr, b_slot, cfg, stream), 0);  // ... B zeroed
    CHECK_EQ(capture_state_external(mgr, *rec_c, cfg, kNC, stream,
                                    &c_after_reset),
             0);
    rc |= cmp_state("reset.C-unchanged", c_before_reset, c_after_reset, kNC,
                    cfg);
    std::fprintf(stderr, "  [ok] reset B: length 0, pages released, delta "
                         "slot %d zeroed, C state unchanged\n",
                 b_slot);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_state_interleave: FAIL\n");
    return rc;
  }
  std::fprintf(stderr,
               "test_qwen35_state_interleave: PASS (A/B interleaved == "
               "independent refs bit-identical + isolation + "
               "retire/reuse (C reuses A's slot+pages, id fresh) + C "
               "fresh-state parity + reset-one isolation)\n");
  return 0;
}
