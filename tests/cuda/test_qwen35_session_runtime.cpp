// CUDALM — v0.8 Phase A: session runtime real-checkpoint hard gate (CUDA,
// real Qwen3.5-0.8B-Base checkpoint).
//
// The device-level PROOF of the v0.8 Phase A ownership migration: KV /
// DeltaNet state lifetimes now belong to SESSIONS, not requests. With the
// REAL model (24 layers, 18x Gated DeltaNet + 6x full attention) this gate
// drives "requests" on sessions' bound sequences through the frozen
// forward_token_with_state() — exactly the single-token path a future
// session-aware scheduler (Phase C) would issue per advance — and gates:
//
//   * CREATE + BINDING: sessions A/B are live with fresh bound sequences
//     (length 0, empty KV table, live distinct Delta slots); the 1:1
//     session <-> sequence binding holds;
//   * PERSISTENCE ACROSS REQUEST BOUNDARIES (the core v0.8 semantic):
//       request 1 on session A forwards tokens [0..5] and COMPLETES —
//       by the session contract it releases NOTHING: A's 3 KV pages stay
//       allocated, A's Delta slot stays live, A's position stays at 6, and
//       A's hybrid state is DIRTY (not zeroed);
//       request 2 on the SAME session A forwards tokens [6..7] as an
//       APPEND-ONLY continuation from position 6 (no state
//       re-initialization at the turn boundary);
//       the 8 per-step FULL logits [248320] + the final hybrid state
//       (18x delta conv+rec, 6x full-attention logical K/V rows read
//       through the block table) are BIT-IDENTICAL to a one-shot
//       continuous reference run of the same 8 tokens on an independent
//       sequence — a request-boundary bug (state released / zeroed /
//       position reset) would break steps 6..7;
//   * ISOLATION UNDER REAL USE: B's own 4-token request (driven while A's
//     requests run) produces per-step logits + final hybrid state
//     BIT-IDENTICAL to a fresh independent B run — A's requests never
//     touch B's state;
//   * RESET PARITY (real model): reset_session(A) restores fresh state
//     under the SAME SessionId (length 0, all pages released, the Delta
//     slot ALL ZEROS on device, reset_count advanced); re-running the same
//     6 tokens is BIT-IDENTICAL to the first A run (logits + state) —
//     no contamination survives a reset;
//   * DESTROY + REUSE (real model): destroy_session(A) retires A's bound
//     sequence — the SessionId is invalid (every operation errors), a
//     forward on the retired sequence id is rejected (the binding is
//     torn down), and the pool accounting returns to EXACTLY B's
//     resources; a new session C reuses A's physical resources under a
//     NEW SessionId and its 3-token run is BIT-IDENTICAL to a fresh
//     reference (fresh zeros, no residue of A), while B is still intact.
//
// Bit-exactness is the gate (memcmp, atol = 0): both sides share the same
// frozen kernels and op order — any non-zero difference is a real bug.
//
// Self-skips (77) when the checkpoint is absent; in the Phase A evidence
// environment the checkpoint IS present and this test actually runs
// (77 is not a sign-off).
//
// Provenance: CUDALM-native (v0.8 Phase A).

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

// Deterministic token sequences (fixed, no randomness).
// A: 8 tokens in TWO requests: request 1 = [0..5], request 2 = [6..7].
//    With page_tokens = 2, positions 2, 4, 6 cross KV page boundaries.
static const int kA[] = {1024, 2048, 3072, 15, 16, 17, 18, 19};
static const int kNA = 8;
static const int kAReq1 = 6;  // request 1: tokens [0..5]
// B: 4 tokens in ONE request.
static const int kB[] = {100, 200, 300, 400};
static const int kNB = 4;
// C: 3 tokens in ONE request (after A's destroy — the reuse owner).
static const int kC[] = {7, 8, 9};
static const int kNC = 3;

static const int kPageTokens = 2;
static const int kPoolPages = 8;
static const int kDeltaSlots = 3;

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

// Bit-exact comparison (the Phase A gate: same kernels, same op order).
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
  return 0;
}

// ---- hybrid state capture (pool slot + pages through the block table) ----
struct StateSnap {
  std::vector<std::vector<__nv_bfloat16>> conv;  // [24] (linear layers)
  std::vector<std::vector<float>> rec;           // [24]
  std::vector<std::vector<__nv_bfloat16>> kvk;   // [24] rows 0..n-1, all kv
  std::vector<std::vector<__nv_bfloat16>> kvv;   // [24]
};

// Returns 1 if a block-table lookup came back unallocated (cannot happen
// for a live sequence whose length covers the requested rows).
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
      const std::size_t rn =
          static_cast<std::size_t>(nkv) * n_tokens * hd;
      std::vector<__nv_bfloat16> k(rn), v(rn);
      // Read the LOGICAL rows 0..n_tokens-1 from the physical pages THROUGH
      // the block table (exactly like the kernels use them).
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

// True iff the Delta slot is non-zero somewhere (a sanity check that real
// model execution left the state DIRTY — not zeroed by a stray reset).
bool delta_slot_any_dirty(const Qwen35StateManager& mgr, int slot,
                          cudaStream_t stream) {
  const auto& dp = mgr.delta_pool();
  for (int l = 0; l < dp.n_linear_layers(); ++l) {
    const std::vector<__nv_bfloat16> conv =
        d2h_bf16(dp.conv(l, slot), dp.conv_elems(), stream);
    for (__nv_bfloat16 x : conv)
      if (__bfloat162float(x) != 0.0f) return true;
    const std::vector<float> rec = d2h_f32(dp.recurrent(l, slot), dp.rec_elems(), stream);
    for (float x : rec)
      if (x != 0.0f) return true;
  }
  return false;
}

// True iff the Delta slot reads back ALL ZEROS in every layer.
bool delta_slot_all_zero(const Qwen35StateManager& mgr, int slot,
                         cudaStream_t stream) {
  const auto& dp = mgr.delta_pool();
  for (int l = 0; l < dp.n_linear_layers(); ++l) {
    const std::vector<__nv_bfloat16> conv =
        d2h_bf16(dp.conv(l, slot), dp.conv_elems(), stream);
    for (__nv_bfloat16 x : conv)
      if (__bfloat162float(x) != 0.0f) return false;
    const std::vector<float> rec = d2h_f32(dp.recurrent(l, slot), dp.rec_elems(), stream);
    for (float x : rec)
      if (x != 0.0f) return false;
  }
  return true;
}

// Drive `n` tokens of `tokens` through the LIVE sequence `sid` (the frozen
// external-state forward; position is derived from the sequence length).
// APPENDS one captured StepOut per forward (a session's request 2 continues
// the same `out` vector as its request 1). A "request" in this test is
// exactly such a drive — what a session-aware scheduler would advance per
// request.
int drive(Qwen35Model& model, Qwen35StateManager& mgr, SequenceId sid,
          const int* tokens, int n, const Qwen35Config& cfg,
          cudaStream_t stream, std::vector<StepOut>* out) {
  const std::size_t start = out->size();
  out->resize(start + static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    Status s = model.forward_token_with_state(tokens[i], sid, mgr, stream);
    if (!s.ok) {
      std::fprintf(stderr, "  forward_token_with_state step %d: %s\n", i,
                   s.message.c_str());
      return 1;
    }
    capture_step(model, cfg, stream, &(*out)[start + static_cast<std::size_t>(i)]);
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
    std::fprintf(stderr, "[SKIP] qwen35 session runtime: no preconverted "
                         "model at %s and --no-convert given\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] qwen35 session runtime: checkpoint absent "
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
               "[session] model loaded: %d layers, vocab %d, max_seq %d, "
               "page_tokens %d, pool pages %d, delta slots %d\n",
               model.num_layers(), cfg.vocab_size, cfg.max_seq_len,
               kPageTokens, kPoolPages, kDeltaSlots);

  Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  SessionManager sm(mgr);
  int rc = 0;

  // ===================================================================
  // CREATE + BINDING: sessions A/B are live with fresh bound sequences.
  // ===================================================================
  SessionId sa_id = 0, sb_id = 0;
  CHECK(sm.create_session(&sa_id).ok);
  CHECK(sm.create_session(&sb_id).ok);
  CHECK_EQ(sa_id, static_cast<SessionId>(1));
  CHECK_EQ(sb_id, static_cast<SessionId>(2));
  SequenceId sa = 0, sb = 0;
  CHECK(sm.sequence_id_of(sa_id, &sa).ok);
  CHECK(sm.sequence_id_of(sb_id, &sb).ok);
  CHECK(mgr.lookup(sa) != nullptr && mgr.lookup(sb) != nullptr);
  CHECK(mgr.lookup(sa)->delta_slot != mgr.lookup(sb)->delta_slot);
  CHECK_EQ(mgr.num_live_sequences(), 2);  // the 1:1 binding invariant
  CHECK_EQ(mgr.lookup(sa)->length, 0);
  CHECK_EQ(mgr.lookup(sa)->block_table.num_blocks(), 0);

  // ===================================================================
  // REFERENCES FIRST (pool capacity discipline): the one-shot continuous
  // reference runs on INDEPENDENT fresh v0.5 sequences (NOT sessions —
  // the v0.5 independent-execution semantics; the same tokens, no request
  // boundary). Each reference is driven, its hybrid state captured to
  // HOST memory, and the reference sequence RETIRED immediately — the
  // small pool (8 pages) cannot hold the references and the session
  // requests at the same time.
  // ===================================================================
  std::vector<StepOut> ref_a_steps, ref_b_steps;
  StateSnap ref_a_state, ref_b_state;
  {
    SequenceId ref_a = 0;
    CHECK(mgr.create_sequence(&ref_a).ok);
    CHECK_EQ(drive(model, mgr, ref_a, kA, kNA, cfg, stream, &ref_a_steps), 0);
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(ref_a), cfg, kNA, stream,
                                    &ref_a_state), 0);
    CHECK(mgr.retire_sequence(ref_a).ok);
  }
  {
    SequenceId ref_b = 0;
    CHECK(mgr.create_sequence(&ref_b).ok);
    CHECK_EQ(drive(model, mgr, ref_b, kB, kNB, cfg, stream, &ref_b_steps), 0);
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(ref_b), cfg, kNB, stream,
                                    &ref_b_state), 0);
    CHECK(mgr.retire_sequence(ref_b).ok);
  }
  CHECK_EQ(mgr.kv_pool().used_pages(), 0);   // references fully retired
  CHECK_EQ(mgr.delta_pool().used_slots(), 2);  // only A/B's slots remain

  // ===================================================================
  // PERSISTENCE ACROSS REQUEST BOUNDARIES (the core v0.8 semantic).
  //
  // Session A, request 1: tokens [0..5]. The request COMPLETES — by the
  // session contract it releases NOTHING (no reset, no retire, no zero).
  // ===================================================================
  std::vector<StepOut> a_steps;
  CHECK_EQ(drive(model, mgr, sa, kA, kAReq1, cfg, stream, &a_steps), 0);
  CHECK_EQ(mgr.lookup(sa)->length, kAReq1);
  // The session is still live and its state PERSISTS: 3 KV pages held
  // (6 tokens at pt=2), the Delta slot live and DIRTY (real execution
  // state, not zeroed), the position at 6.
  CHECK(sm.lookup(sa_id) != nullptr);
  CHECK_EQ(mgr.lookup(sa)->block_table.num_blocks(), 3);
  CHECK_EQ(mgr.kv_pool().used_pages(), 3);
  CHECK_EQ(mgr.delta_pool().used_slots(), 2);
  CHECK(delta_slot_any_dirty(mgr, mgr.lookup(sa)->delta_slot, stream));
  int pos = -1;
  CHECK(sm.context_length_of(sa_id, &pos).ok);
  CHECK_EQ(pos, kAReq1);
  // Capture A's hybrid state AT LENGTH 6 (for the later reset-parity gate).
  StateSnap a_state_6;
  CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sa), cfg, kAReq1, stream,
                                  &a_state_6), 0);
  // B is untouched by A's request.
  CHECK_EQ(mgr.lookup(sb)->length, 0);
  CHECK_EQ(mgr.lookup(sb)->block_table.num_blocks(), 0);
  std::fprintf(stderr, "  [ok] request 1 completed: A's state persisted "
                       "(3 pages, live dirty Delta slot, position 6)\n");

  // B's own 4-token request (driven while A's requests have run).
  std::vector<StepOut> b_steps;
  CHECK_EQ(drive(model, mgr, sb, kB, kNB, cfg, stream, &b_steps), 0);
  CHECK_EQ(mgr.lookup(sb)->length, kNB);

  // Session A, request 2: tokens [6..7] — an APPEND-ONLY continuation from
  // position 6 (no state re-initialization at the turn boundary; the
  // position is derived from the session's bound sequence length).
  CHECK_EQ(drive(model, mgr, sa, kA + kAReq1, kNA - kAReq1, cfg, stream,
                 &a_steps), 0);
  CHECK_EQ(mgr.lookup(sa)->length, kNA);
  CHECK_EQ(mgr.lookup(sa)->block_table.num_blocks(), 4);  // positions 6..7
                                                          // cross block 3

  StateSnap a_state, b_state;
  CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sa), cfg, kNA, stream,
                                  &a_state), 0);
  CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sb), cfg, kNB, stream,
                                  &b_state), 0);

  // Chunked-on-session == one-shot, per step (embedding / final norm /
  // FULL logits), BIT-IDENTICAL — for A across the request boundary
  // (steps 6..7 are the boundary proof) and for B (isolation).
  for (int i = 0; i < kNA; ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "A step p%d logits[%d] (chunked==one-shot)",
                  i, cfg.vocab_size);
    rc |= cmp_exact(name, ref_a_steps[static_cast<std::size_t>(i)].logits,
                    a_steps[static_cast<std::size_t>(i)].logits);
    std::snprintf(name, sizeof(name), "A step p%d embed", i);
    rc |= cmp_exact(name, ref_a_steps[static_cast<std::size_t>(i)].embed,
                    a_steps[static_cast<std::size_t>(i)].embed);
    std::snprintf(name, sizeof(name), "A step p%d final_norm", i);
    rc |= cmp_exact(name, ref_a_steps[static_cast<std::size_t>(i)].norm,
                    a_steps[static_cast<std::size_t>(i)].norm);
  }
  for (int l = 0; l < model.num_layers(); ++l) {
    char name[64];
    std::snprintf(name, sizeof(name), "A step p6 layer %d (boundary)", l);
    const __nv_bfloat16* ref_l =
        ref_a_steps[static_cast<std::size_t>(6)].layers.data() +
        static_cast<std::size_t>(l) * cfg.hidden_size;
    const __nv_bfloat16* act_l =
        a_steps[static_cast<std::size_t>(6)].layers.data() +
        static_cast<std::size_t>(l) * cfg.hidden_size;
    rc |= cmp_exact(name,
                    std::vector<__nv_bfloat16>(ref_l, ref_l + cfg.hidden_size),
                    std::vector<__nv_bfloat16>(act_l, act_l + cfg.hidden_size));
  }
  for (int i = 0; i < kNB; ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "B step p%d logits[%d] (isolation)", i,
                  cfg.vocab_size);
    rc |= cmp_exact(name, ref_b_steps[static_cast<std::size_t>(i)].logits,
                    b_steps[static_cast<std::size_t>(i)].logits);
  }
  // Final hybrid state: A (chunked) == one-shot reference; B == fresh
  // reference.
  rc |= cmp_state("A state", a_state, ref_a_state, cfg);
  rc |= cmp_state("B state", b_state, ref_b_state, cfg);
  std::fprintf(stderr, "  [ok] chunked-on-session == one-shot bit-identical "
                       "(A across the request boundary; B isolated)\n");

  // ===================================================================
  // RESET PARITY (real model): reset_session(A) -> fresh state under the
  // SAME SessionId; the same 6 tokens reproduce the FIRST run bit-for-bit.
  // ===================================================================
  const int a_slot_before = mgr.lookup(sa)->delta_slot;
  CHECK(sm.reset_session(sa_id).ok);
  // The session stays LIVE under the SAME id, in fresh state:
  CHECK(sm.lookup(sa_id) != nullptr);
  CHECK_EQ(sm.lookup(sa_id)->reset_count, 1);  // lifecycle metadata
  CHECK_EQ(sm.lookup(sa_id)->sequence_id, sa);
  CHECK_EQ(mgr.lookup(sa)->length, 0);                    // position reset
  CHECK_EQ(mgr.lookup(sa)->block_table.num_blocks(), 0);  // KV reset
  CHECK_EQ(mgr.lookup(sa)->delta_slot, a_slot_before);    // same slot
  CHECK(delta_slot_all_zero(mgr, a_slot_before, stream));  // Delta zeroed
  CHECK_EQ(mgr.kv_pool().used_pages(), 2);  // only B's 2 pages remain
  int len = -1;
  CHECK(sm.context_length_of(sa_id, &len).ok);
  CHECK_EQ(len, 0);
  // B is untouched by A's reset.
  CHECK_EQ(mgr.lookup(sb)->length, kNB);
  CHECK_EQ(mgr.lookup(sb)->block_table.num_blocks(), 2);

  std::vector<StepOut> a_reset_steps;
  CHECK_EQ(drive(model, mgr, sa, kA, kAReq1, cfg, stream, &a_reset_steps), 0);
  for (int i = 0; i < kAReq1; ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "A reset p%d logits[%d]", i,
                  cfg.vocab_size);
    rc |= cmp_exact(name, a_steps[static_cast<std::size_t>(i)].logits,
                    a_reset_steps[static_cast<std::size_t>(i)].logits);
    std::snprintf(name, sizeof(name), "A reset p%d embed", i);
    rc |= cmp_exact(name, a_steps[static_cast<std::size_t>(i)].embed,
                    a_reset_steps[static_cast<std::size_t>(i)].embed);
  }
  {
    StateSnap a_reset_state;
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sa), cfg, kAReq1, stream,
                                    &a_reset_state), 0);
    // Both snapshots are at length 6 (a_state_6 was captured right after
    // request 1): the Delta state compares in full, the KV rows 0..5
    // compare exactly — no contamination survives a reset.
    rc |= cmp_state("A reset.state", a_reset_state, a_state_6, cfg);
  }
  std::fprintf(stderr, "  [ok] reset parity: A fresh under the SAME id, "
                       "re-run bit-identical\n");

  // ===================================================================
  // DESTROY + REUSE (real model).
  // ===================================================================
  // Destroy A (it holds 3 pages + a dirty Delta slot after the reset run).
  CHECK(sm.destroy_session(sa_id).ok);
  CHECK(sm.lookup(sa_id) == nullptr);
  // The SessionId is invalid: EVERY operation errors (fail loud).
  SequenceId dead = 0;
  CHECK(!sm.sequence_id_of(sa_id, &dead).ok);
  CHECK(!sm.context_length_of(sa_id, &len).ok);
  CHECK(!sm.reset_session(sa_id).ok);
  CHECK(!sm.destroy_session(sa_id).ok);  // double destroy
  // The bound sequence was retired: a forward on it is rejected (the
  // binding is torn down — no zombie state handle).
  CHECK(!model.forward_token_with_state(kA[0], sa, mgr, stream).ok);
  CHECK(mgr.lookup(sa) == nullptr);
  // Pool accounting: EXACTLY B's resources remain (2 pages, 1 slot).
  CHECK_EQ(mgr.kv_pool().used_pages(), 2);
  CHECK_EQ(mgr.delta_pool().used_slots(), 1);
  CHECK_EQ(mgr.num_live_sequences(), 1);
  CHECK(sm.num_sessions() == 1);

  // Create C: a NEW SessionId (never sa_id), reusing A's physical
  // resources; its 3-token run must equal a FRESH reference bit-for-bit
  // (fresh zeros — no residue of A's state).
  SessionId sc_id = 0;
  CHECK(sm.create_session(&sc_id).ok);
  CHECK(sc_id != sa_id);
  CHECK_EQ(sc_id, static_cast<SessionId>(3));
  SequenceId sc = 0;
  CHECK(sm.sequence_id_of(sc_id, &sc).ok);
  std::vector<StepOut> c_steps;
  CHECK_EQ(drive(model, mgr, sc, kC, kNC, cfg, stream, &c_steps), 0);
  SequenceId ref_c = 0;
  CHECK(mgr.create_sequence(&ref_c).ok);
  std::vector<StepOut> ref_c_steps;
  CHECK_EQ(drive(model, mgr, ref_c, kC, kNC, cfg, stream, &ref_c_steps), 0);
  StateSnap c_state, ref_c_state;
  CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sc), cfg, kNC, stream,
                                  &c_state), 0);
  CHECK_EQ(capture_state_external(mgr, *mgr.lookup(ref_c), cfg, kNC, stream,
                                  &ref_c_state), 0);
  for (int i = 0; i < kNC; ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "C step p%d logits[%d] (reuse, no residue)",
                  i, cfg.vocab_size);
    rc |= cmp_exact(name, ref_c_steps[static_cast<std::size_t>(i)].logits,
                    c_steps[static_cast<std::size_t>(i)].logits);
  }
  rc |= cmp_state("C state", c_state, ref_c_state, cfg);
  CHECK(mgr.retire_sequence(ref_c).ok);
  // B is still intact after A's destroy + C's run (final isolation check):
  // B's hybrid state still equals its fresh reference (re-captured now —
  // nothing since B's own run may have touched it).
  {
    StateSnap b_state_now;
    CHECK_EQ(capture_state_external(mgr, *mgr.lookup(sb), cfg, kNB, stream,
                                    &b_state_now), 0);
    rc |= cmp_state("B state after destroy/reuse", b_state_now, b_state, cfg);
    CHECK_EQ(mgr.lookup(sb)->length, kNB);
  }
  std::fprintf(stderr, "  [ok] destroy + reuse: A invalid, resources "
                       "reclaimed; C fresh under a new id; B intact\n");

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_session_runtime: FAIL\n");
    return rc;
  }
  std::fprintf(stderr,
               "test_qwen35_session_runtime: PASS (create/binding + "
               "persistence across request boundaries + isolation + reset "
               "parity + destroy/reuse, all bit-identical vs references)\n");
  return 0;
}
