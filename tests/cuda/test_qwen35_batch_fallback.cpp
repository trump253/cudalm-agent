// CUDALM — v0.6 Phase B: ZERO-MUTATION batch preflight + SERIAL FALLBACK
// real-checkpoint gate (CUDA, REAL Qwen3.5-0.8B-Base checkpoint).
//
// The Phase B batch path promises: `forward_batch_with_state` PREFLIGHT is
// ZERO-MUTATION — on ANY preflight failure NOTHING changed (no length, no
// block table, no KV pool allocation/accounting, no Delta state, no
// logical KV, no partial allocation) — and the scheduler then falls back
// to the FROZEN Phase A serial path for the cohort (any state progress
// comes ONLY from the serial fallback, never from the failed batch
// attempt).
//
// This gate EXERCISES the fallback (the other Phase B gates avoid it by
// giving the pool generous capacity). The scenario: pool
// kv_capacity_pages = 3 (page_tokens = 2); two sequences prefilled to
// length 2 use 2 pages -> 1 free; each row's NEXT decode position (2)
// needs ONE new page -> the batch needs 2 > 1 free -> preflight failure.
//
//   Part A (direct API, zero mutation): before/after the FAILED
//   `forward_batch_with_state`, for BOTH rows: length, block tables, KV
//   pool accounting (free/used/capacity/live pages), Delta state (all 18
//   conv + recurrent slots) and logical KV rows are captured and must be
//   BIT-IDENTICAL (exactly unchanged).
//
//   Part B (scheduler, same scenario): the scheduler admits the same two
//   requests; when the decode cohort of 2 is attempted, the batch
//   preflight fails and the scheduler falls back to the frozen serial
//   path, proving:
//       * batch_fallback_calls == 1, batch_forward_calls == 0 (NO batch
//         committed — the failed attempt allocated nothing);
//       * row A (first in snapshot order) advances EXACTLY as the frozen
//         Phase A serial path: after the fallback step A's state is
//         BIT-IDENTICAL (length, Delta state, logical KV) to an
//         independent direct-serial reference from an equivalent state,
//         and A's generated tokens match it exactly — A's progress is
//         exactly ONE serial decode (if the failed batch attempt had
//         mutated A, A would differ);
//       * the pool delta is EXACTLY A's one committed serial page (the
//         failed batch attempt committed nothing);
//       * row B's serial forward ALSO fails (the pool is exhausted by A's
//         committed page) -> B is Failed + retired EXACTLY ONCE, its
//         failed forward NOT committed (forward_count stays 2, not 3) —
//         the frozen advance_one semantics; B's page + Delta slot are
//         released (zero-on-release: all-zero, matching the manager's
//         release contract; B's pre-step state was non-zero, verified).
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

// The capacity pin (see the header): page_tokens=2, kv_capacity_pages=3,
// delta slots 4.
static const int kPageTokens = 2;
static const int kPoolPages = 3;
static const int kDeltaSlots = 4;
static const std::vector<int> kPromptA = {1024, 2048};  // A
static const std::vector<int> kPromptB = {15, 16};  // B
static const int kTokA = 777;  // A's decode token (Part A direct call)
static const int kTokB = 888;  // B's decode token (Part A direct call)

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
bool all_zero_bf16(const std::vector<__nv_bfloat16>& v) {
  for (__nv_bfloat16 x : v)
    if (__bfloat162float(x) != 0.0f) return false;
  return true;
}
bool all_zero_f32(const std::vector<float>& v) {
  for (float x : v)
    if (x != 0.0f) return false;
  return true;
}

// ---- state snapshots --------------------------------------------------------
struct PoolAcc {
  int free_pages = 0;
  int used_pages = 0;
  int capacity_pages = 0;
  std::vector<int> live_pages;
};
struct SeqSnap {
  int length = 0;
  int delta_slot = 0;
  int num_blocks = 0;
  std::vector<int> page_ids;
  std::vector<std::vector<__nv_bfloat16>> conv;  // [L] linear layers
  std::vector<std::vector<float>> rec;           // [L]
  std::vector<std::vector<__nv_bfloat16>> kvk;   // [L] logical rows 0..n-1
  std::vector<std::vector<__nv_bfloat16>> kvv;   // [L]
};

PoolAcc capture_pool(const Qwen35StateManager& mgr) {
  PoolAcc p;
  p.free_pages = mgr.kv_pool().free_pages();
  p.used_pages = mgr.kv_pool().used_pages();
  p.capacity_pages = mgr.kv_pool().capacity_pages();
  p.live_pages = mgr.kv_pool().live_pages();
  return p;
}

int capture_seq(const Qwen35StateManager& mgr, SequenceId sid,
                const Qwen35Config& cfg, cudaStream_t stream, SeqSnap* out) {
  const SequenceState* rec = mgr.lookup(sid);
  if (rec == nullptr) return 1;
  const int L = cfg.num_hidden_layers;
  const int nkv = cfg.n_kv_heads;
  const int hd = cfg.head_dim;
  const int pt = mgr.page_tokens();
  const int slot = rec->delta_slot;
  out->length = static_cast<int>(rec->length);
  out->delta_slot = slot;
  out->num_blocks = rec->block_table.num_blocks();
  out->page_ids.assign(rec->block_table.page_ids(),
                       rec->block_table.page_ids() + out->num_blocks);
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
      const int ntok = out->length;
      const std::size_t rn = static_cast<std::size_t>(nkv) * ntok * hd;
      std::vector<__nv_bfloat16> k(rn), v(rn);
      for (int t = 0; t < ntok; ++t) {
        const int page = rec->block_table.lookup(t / pt);
        if (page < 0) {
          std::fprintf(stderr, "  [state] block %d unallocated\n", t / pt);
          return 1;
        }
        const int off = t % pt;
        for (int n = 0; n < nkv; ++n) {
          const std::size_t dst =
              (static_cast<std::size_t>(n) * ntok + t) * hd;
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

// Compare two snapshots. `compare_ids` = also compare the PHYSICAL page
// ids (page_ids / live pages) — valid for a same-manager before/after
// zero-mutation check; NOT for cross-manager comparisons (allocation
// ORDER is an implementation detail, the logical state is the contract).
int cmp_snap(const char* label, const SeqSnap& ref, const SeqSnap& act,
             const PoolAcc& pref, const PoolAcc& pact, const Qwen35Config& cfg,
             bool compare_ids) {
  int rc = 0;
  char name[72];
  auto check_int = [&](const char* n, int a, int b) {
    std::fprintf(stderr, "  %-52s %s (%d == %d)\n", n,
                 a == b ? "identical" : "FAIL", a, b);
    rc |= (a == b) ? 0 : 1;
  };
  std::snprintf(name, sizeof(name), "%s length", label);
  check_int(name, ref.length, act.length);
  std::snprintf(name, sizeof(name), "%s delta_slot", label);
  check_int(name, ref.delta_slot, act.delta_slot);
  std::snprintf(name, sizeof(name), "%s block_table.num_blocks", label);
  check_int(name, ref.num_blocks, act.num_blocks);
  if (compare_ids) {
    const bool pids_ok = ref.page_ids == act.page_ids;
    std::fprintf(stderr, "  %-52s %s (n=%zu)\n",
                 (std::string(label) + " block_table.page_ids").c_str(),
                 pids_ok ? "identical" : "FAIL", ref.page_ids.size());
    rc |= pids_ok ? 0 : 1;
  }
  std::snprintf(name, sizeof(name), "%s pool.free_pages", label);
  check_int(name, pref.free_pages, pact.free_pages);
  std::snprintf(name, sizeof(name), "%s pool.used_pages", label);
  check_int(name, pref.used_pages, pact.used_pages);
  std::snprintf(name, sizeof(name), "%s pool.capacity_pages", label);
  check_int(name, pref.capacity_pages, pact.capacity_pages);
  if (compare_ids) {
    const bool live_ok = pref.live_pages == pact.live_pages;
    std::fprintf(stderr, "  %-52s %s (n=%zu)\n",
                 (std::string(label) + " pool.live_pages").c_str(),
                 live_ok ? "identical" : "FAIL", pref.live_pages.size());
    rc |= live_ok ? 0 : 1;
  }
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
      std::snprintf(name, sizeof(name), "%s L%d kv.K", label, i);
      rc |= cmp_exact(name, ref.kvk[static_cast<std::size_t>(i)],
                      act.kvk[static_cast<std::size_t>(i)]);
      std::snprintf(name, sizeof(name), "%s L%d kv.V", label, i);
      rc |= cmp_exact(name, ref.kvv[static_cast<std::size_t>(i)],
                      act.kvv[static_cast<std::size_t>(i)]);
    }
  }
  return rc;
}

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
    std::fprintf(stderr, "[SKIP] batch fallback: no preconverted model\n");
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] batch fallback: checkpoint absent\n");
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
               "[batch-fallback] model loaded: page_tokens %d, pool pages "
               "%d, delta slots %d\n",
               kPageTokens, kPoolPages, kDeltaSlots);
  int rc = 0;

  // =========================================================================
  // Part A: direct API — the batch preflight failure is ZERO-MUTATION.
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    SequenceId sA = 0, sB = 0;
    CHECK_EQ(prefill_serial(model, mgr, kPromptA, &sA, stream), 0);
    CHECK_EQ(prefill_serial(model, mgr, kPromptB, &sB, stream), 0);
    // After prefill: A length 2 (1 page), B length 2 (1 page), 1 page free.
    CHECK_EQ(mgr.lookup(sA)->length, 2);
    CHECK_EQ(mgr.lookup(sB)->length, 2);
    CHECK_EQ(mgr.kv_pool().free_pages(), kPoolPages - 2);

    PoolAcc pool_before = capture_pool(mgr);
    SeqSnap a_before, b_before;
    CHECK_EQ(capture_seq(mgr, sA, cfg, stream, &a_before), 0);
    CHECK_EQ(capture_seq(mgr, sB, cfg, stream, &b_before), 0);

    // The aggregate shortfall: each row's decode at position 2 needs ONE
    // new page -> the batch needs 2, only 1 free -> preflight failure.
    const int toks[2] = {kTokA, kTokB};
    const SequenceId sids[2] = {sA, sB};
    Status bs = model.forward_batch_with_state(toks, sids, 2, mgr, stream);
    CHECK(!bs.ok);
    CHECK(bs.message.find("aggregate KV capacity") != std::string::npos);

    // ZERO MUTATION: nothing changed (length / block tables / KV pool
    // accounting / Delta state / logical KV) — no partial allocation.
    PoolAcc pool_after = capture_pool(mgr);
    SeqSnap a_after, b_after;
    CHECK_EQ(capture_seq(mgr, sA, cfg, stream, &a_after), 0);
    CHECK_EQ(capture_seq(mgr, sB, cfg, stream, &b_after), 0);
    std::fprintf(stderr, "[zero-mutation] failed batch: before/after "
                         "(length, block tables, KV pool accounting, Delta "
                         "state, logical KV)\n");
    rc |= cmp_snap("A (after failed batch)", a_before, a_after, pool_before,
                   pool_after, cfg, /*compare_ids=*/true);
    rc |= cmp_snap("B (after failed batch)", b_before, b_after, pool_before,
                   pool_after, cfg, /*compare_ids=*/true);

    CHECK(mgr.retire_sequence(sA).ok);
    CHECK(mgr.retire_sequence(sB).ok);
    CHECK_EQ(mgr.num_live_sequences(), 0);
  }

  // =========================================================================
  // Part B: scheduler — the SAME scenario falls back to the frozen serial
  // path (batch_fallback_calls == 1; all progress from the serial fallback).
  // =========================================================================
  {
    Qwen35StateManager mgr(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
    ModelForwarder fwd(model);
    Scheduler sched(fwd, mgr, stream);
    RequestId a = 0, b = 0;
    Scheduler::Spec sa;
    sa.prompt = kPromptA;
    sa.max_new_tokens = 3;  // A stays live after the fallback step (state
                            // inspectable before its later finish/retire)
    sa.sampling = SamplingConfig{0.8f, 50, 1.0f, 42u};
    Scheduler::Spec sb;
    sb.prompt = kPromptB;
    sb.max_new_tokens = 2;
    sb.sampling = SamplingConfig{0.8f, 50, 1.0f, 123u};
    CHECK(sched.admit(sa, &a).ok);
    CHECK(sched.admit(sb, &b).ok);

    // Step 1: A p0 (page 0), B p0 (page 1) — serial prefill, 1 page free.
    CHECK(sched.step().ok);
    CHECK_EQ(sched.get(a)->forward_count, 1);
    CHECK_EQ(sched.get(b)->forward_count, 1);
    CHECK_EQ(mgr.kv_pool().free_pages(), kPoolPages - 2);
    // Step 2: A p1, B p1 -> both sample g0 -> decode-ready.
    CHECK(sched.step().ok);
    CHECK_EQ(sched.get(a)->forward_count, 2);
    CHECK_EQ(sched.get(b)->forward_count, 2);
    CHECK_EQ(static_cast<int>(sched.get(a)->generated.size()), 1);
    CHECK_EQ(static_cast<int>(sched.get(b)->generated.size()), 1);
    CHECK_EQ(sched.batch_forward_calls(), 0);  // no batch committed yet
    CHECK_EQ(sched.batch_fallback_calls(), 0);

    const SequenceId sA = sched.get(a)->sequence_id;
    const SequenceId sB = sched.get(b)->sequence_id;
    PoolAcc pool_before = capture_pool(mgr);
    SeqSnap b_before;
    CHECK_EQ(capture_seq(mgr, sB, cfg, stream, &b_before), 0);
    // Sanity: B's pre-step state is NON-zero (makes the release-zero check
    // below meaningful). conv[0]/rec[0] is the first linear-attention layer
    // (layer 0); the first FULL-attention layer's logical KV must also be
    // non-zero (layer 0 is linear, so kvk[0] is empty by construction).
    CHECK(!all_zero_bf16(b_before.conv[0]));
    CHECK(!all_zero_f32(b_before.rec[0]));
    int first_fa = -1;
    for (int i = 0; i < cfg.num_hidden_layers; ++i)
      if (!cfg.is_linear_attention(i)) {
        first_fa = i;
        break;
      }
    CHECK(first_fa >= 0);
    CHECK(!all_zero_bf16(b_before.kvk[static_cast<std::size_t>(first_fa)]));

    // Step 3: cohort [A,B] -> batch preflight FAILS (2 needed, 1 free) ->
    // SERIAL FALLBACK: A advances (commits the last page, samples g1,
    // still live at max_new 3); B's serial forward FAILS (pool exhausted
    // by A's committed page) -> B Failed + retired, zero committed
    // progress.
    Status s3 = sched.step();
    CHECK(!s3.ok);  // the first error: B's serial capacity failure
    CHECK_EQ(sched.batch_fallback_calls(), 1);  // the fallback happened
    CHECK_EQ(sched.batch_forward_calls(), 0);  // NO batch committed
    CHECK_EQ(sched.single_forward_calls(), 6);  // 4 prefill + 2 serial
                                                // (A ok, B failed)
    // A: advanced EXACTLY like the frozen serial path (1 decode committed).
    CHECK_EQ(sched.get(a)->forward_count, 3);
    CHECK_EQ(static_cast<int>(sched.get(a)->generated.size()), 2);
    CHECK_EQ(sched.get(a)->status, RequestStatus::Running);
    // B: Failed + retired exactly once, the failed serial forward NOT
    // committed (frozen advance_one semantics).
    CHECK_EQ(sched.get(b)->status, RequestStatus::Failed);
    CHECK_EQ(sched.get(b)->finish_reason, FinishReason::Failed);
    CHECK_EQ(sched.get(b)->forward_count, 2);  // 2 prefill, NOT 3
    CHECK_EQ(static_cast<int>(sched.get(b)->generated.size()), 1);
    CHECK_EQ(mgr.num_live_sequences(), 1);  // only A live (B retired once)
    CHECK(mgr.lookup(sB) == nullptr);  // B's record is gone (retired)

    // B: released with ZERO-ON-RELEASE (page + Delta slot all-zero; B's
    // pre-step state was non-zero above). The failed batch attempt
    // committed NOTHING for B (Part A proved the failed batch API is
    // zero-mutation; B's only other attempt — its serial forward — failed
    // at the capacity preflight, which is also zero-mutation by the frozen
    // v0.5 contract).
    {
      const int slotB = b_before.delta_slot;
      const int pageB = b_before.page_ids[0];
      bool b_page_live = false;
      for (int p : mgr.kv_pool().live_pages())
        if (p == pageB) b_page_live = true;
      CHECK(!b_page_live);  // B's page is free
      bool delta_zero = true, kv_zero = true;
      for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        if (cfg.is_linear_attention(i)) {
          const int ord = mgr.delta_pool().linear_layer_ordinal(i);
          const std::size_t cn = static_cast<std::size_t>(cfg.linear_conv_dim()) *
                                 cfg.linear_conv_state_len();
          const std::size_t rn = static_cast<std::size_t>(cfg.lin_num_v_heads) *
                                 cfg.lin_value_head_dim * cfg.lin_value_head_dim;
          delta_zero &= all_zero_bf16(
              d2h_bf16(mgr.delta_pool().conv(ord, slotB), cn, stream));
          delta_zero &= all_zero_f32(
              d2h_f32(mgr.delta_pool().recurrent(ord, slotB), rn, stream));
        } else {
          const int ord = mgr.kv_pool().full_layer_ordinal(i);
          const std::size_t pe = mgr.kv_pool().page_elems();
          kv_zero &=
              all_zero_bf16(d2h_bf16(mgr.kv_pool().k_page(ord, pageB), pe,
                                     stream));
          kv_zero &=
              all_zero_bf16(d2h_bf16(mgr.kv_pool().v_page(ord, pageB), pe,
                                     stream));
        }
      }
      std::fprintf(stderr, "  %-52s %s\n", "B released Delta slot (zero)",
                   delta_zero ? "all-zero" : "FAIL");
      std::fprintf(stderr, "  %-52s %s\n", "B released KV page (zero)",
                   kv_zero ? "all-zero" : "FAIL");
      rc |= (delta_zero && kv_zero) ? 0 : 1;
    }
    // The pool delta is EXACTLY A's one committed serial page: A committed
    // one page; B's retire released B's page; net: A's two pages live, one
    // free. (The failed batch attempt allocated nothing.)
    PoolAcc pool_after = capture_pool(mgr);
    CHECK_EQ(pool_after.used_pages, 2);  // A's two pages only
    CHECK_EQ(pool_after.free_pages, 1);  // B's released page
    {
      const SequenceState* recA = mgr.lookup(sA);
      CHECK(recA != nullptr);
      CHECK_EQ(recA->block_table.num_blocks(), 2);  // A grew by exactly ONE
                                                    // serial page
    }

    // A: its fallback-step state must be BIT-IDENTICAL to an independent
    // direct-SERIAL reference (frozen Phase A semantics): same prompt,
    // same 2 prefill + 1 decode forwards, same per-request sampler stream.
    SeqSnap a_after;
    CHECK_EQ(capture_seq(mgr, sA, cfg, stream, &a_after), 0);
    std::vector<int> a_gen = sched.get(a)->generated;  // [g0, g1]

    Qwen35StateManager mgr_ref(cfg, kPageTokens, kPoolPages, kDeltaSlots,
                               stream);
    SequenceId sR = 0;
    CHECK(mgr_ref.create_sequence(&sR).ok);
    const SamplingConfig s42 = SamplingConfig{0.8f, 50, 1.0f, 42u};
    Sampler sampler(s42);
    CHECK(model.forward_token_with_state(kPromptA[0], sR, mgr_ref, stream).ok);
    CHECK(model.forward_token_with_state(kPromptA[1], sR, mgr_ref, stream).ok);
    std::vector<__nv_bfloat16> lg =
        d2h_bf16(model.logits(), cfg.vocab_size, stream);
    const int g0 = sampler.sample(lg.data(), cfg.vocab_size);
    CHECK(g0 >= 0 && g0 < cfg.vocab_size);
    CHECK(model.forward_token_with_state(g0, sR, mgr_ref, stream).ok);  // len 3
    SeqSnap ref_state;
    CHECK_EQ(capture_seq(mgr_ref, sR, cfg, stream, &ref_state), 0);
    std::fprintf(stderr, "[fallback] A (serial fallback) vs independent "
                         "direct-serial reference (length, Delta state, "
                         "logical KV, pool counts)\n");
    rc |= cmp_snap("A (fallback vs serial ref)", ref_state, a_after,
                   capture_pool(mgr_ref), capture_pool(mgr), cfg,
                   /*compare_ids=*/false);
    CHECK_EQ(a_gen.size(), 2u);
    CHECK_EQ(a_gen[0], g0);
    // Continue the reference to A's completion (g1, g2) for the full token
    // stream.
    lg = d2h_bf16(model.logits(), cfg.vocab_size, stream);
    const int g1 = sampler.sample(lg.data(), cfg.vocab_size);
    CHECK(g1 >= 0 && g1 < cfg.vocab_size);
    CHECK(model.forward_token_with_state(g1, sR, mgr_ref, stream).ok);  // len 4
    lg = d2h_bf16(model.logits(), cfg.vocab_size, stream);
    const int g2 = sampler.sample(lg.data(), cfg.vocab_size);
    CHECK(g2 >= 0 && g2 < cfg.vocab_size);
    std::vector<int> ref_gen_full = {g0, g1, g2};

    // run() drains A (one more serial decode of g1 at max_new 3 ->
    // Finished). B's capacity error was already surfaced by the explicit
    // step() above (s3); NO error occurs INSIDE run(), so run() returns ok
    // (its failure-isolation contract: errors encountered inside run() are
    // recorded and reported, and execution continues).
    Status srun = sched.run();
    CHECK(srun.ok);
    CHECK_EQ(sched.get(a)->status, RequestStatus::Finished);
    CHECK_EQ(sched.get(a)->finish_reason, FinishReason::MaxNewTokens);
    CHECK_EQ(sched.get(a)->forward_count, 2 + 3 - 1);  // N + m - 1
    CHECK(sched.get(a)->generated == ref_gen_full);  // full stream EXACT
    CHECK_EQ(sched.num_live(), 0);
    CHECK_EQ(mgr.num_live_sequences(), 0);
    CHECK(mgr_ref.retire_sequence(sR).ok);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_batch_fallback: FAIL\n");
    return rc;
  }
  std::printf(
      "test_qwen35_batch_fallback: PASS (aggregate KV shortfall: failed "
      "batch preflight is ZERO-MUTATION (length/block tables/KV pool "
      "accounting/Delta state/logical KV unchanged, no partial allocation); "
      "scheduler fallback batch_fallback_calls=1, batch_forward_calls=0; "
      "row A bit-identical to the frozen serial path (state + tokens), row B "
      "Failed + retired exactly once with zero committed progress)\n");
  return 0;
}
