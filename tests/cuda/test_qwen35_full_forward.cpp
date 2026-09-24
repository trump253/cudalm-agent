// CUDALM — v0.3 Phase B full single-token forward golden hard gate (real
// checkpoint; docs/qwen35_architecture.md §18).
//
// Runs the COMPLETE Qwen3.5-0.8B model (embedding -> 24 layers -> final
// RMSNorm -> tied LM head -> logits) on deterministic tokens and compares
// EVERY intermediate output against the pinned-quantized oracle (the same
// W4A16 weights + tied BF16 embedding the runtime loads):
//   scenario A (fresh p=0): reset state once, forward token 15 at p=0.
//   scenario B (sequential p0->p1->p2): reset once, forward tokens 15,16,17
//     at p=0,1,2 — each step threads the RUNTIME's OWN persistent state (no
//     golden-state backfeed).
//
// Compared per token (the hard gate):
//   * embedding output           [hidden]      bf16 (bit-exact: pure row copy)
//   * 24 x layer final output    [hidden]      bf16
//   * final RMSNorm output       [hidden]      bf16
//   * FULL logits                [vocab=248320] bf16  (full vector, not top-k)
//   * 18 x DeltaNet conv_state   [6144,3]      bf16 + recurrent_state fp32
//   * 6  x FullAttention KV rows 0..p  [n_kv*head_dim] bf16
//
// The runtime B run resets ONCE then threads its own state; the golden oracle
// does the same. Tolerance starts at the v0.2 standard (docs §14) and, where
// the 24-layer bf16 chain needs it, a measured minimal envelope (recorded in
// docs §18.2).
//
// Skips with exit 77 when the checkpoint is absent (it lives in /root/models).

#include "../../tests/common/check.h"

#include "cudalm/golden_loader_v2.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_full_attention.h"
#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_kv_cache.h"
#include "cudalm/qwen35_deltanet.h"
#include "cudalm/stage_compare.h"
#include "cudalm/weight_loader_v2.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace cudalm;

namespace {

// Deterministic token ids (MUST match the golden generator defaults:
// --fullmodel-tokens-a 15, --fullmodel-tokens-b 15,16,17).
static const int kTokenA[1] = {15};
static const int kTokenB[3] = {15, 16, 17};
static const int kNA = 1;
static const int kNB = 3;

int run_cmd(const std::string& cmd) {
  std::fprintf(stderr, "  $ %s\n", cmd.c_str());
  const int rc = std::system(cmd.c_str());
  if (rc != 0) std::fprintf(stderr, "  command failed (rc=%d)\n", rc);
  return rc;
}

bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return static_cast<bool>(f);
}

template <typename T>
std::vector<T> dev_to_host(const T* dev, std::size_t n, cudaStream_t stream) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(T),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h;
}

// Compare a bf16 device buffer against golden host bytes (|a-r| <= atol +
// rtol*|r|). Always prints the measured max_abs_err (so the envelope can be
// tuned to the measured error, not a guess). If `max_abs_out` is non-null, the
// measured max_abs_err is stored there (for the depth-aware envelope summary).
int compare_bf16(const std::string& name, const __nv_bfloat16* dev,
                 const std::uint8_t* ref, std::size_t elems, double atol,
                 double rtol, cudaStream_t stream, double* max_abs_out =
                     nullptr) {
  std::vector<__nv_bfloat16> act = dev_to_host(dev, elems, stream);
  const StageCompareResult r = compare_bf16_stages(
      reinterpret_cast<const std::uint16_t*>(ref),
      reinterpret_cast<const std::uint16_t*>(act.data()), elems, atol, rtol);
  if (max_abs_out) *max_abs_out = r.max_abs_err;
  if (!r.ok) {
    std::fprintf(stderr, "  %-40s FAIL max_abs=%.9g max_rel=%.9g (idx=%zu, "
                         "n=%zu, atol=%.4g rtol=%.4g)\n",
                 name.c_str(), r.max_abs_err, r.max_rel_err, r.max_abs_idx,
                 elems, atol, rtol);
    return 1;
  }
  std::fprintf(stderr, "  %-40s max_abs=%.9g OK\n", name.c_str(), r.max_abs_err);
  return 0;
}

// Compare an fp32 device buffer against golden host bytes.
int compare_fp32(const std::string& name, const float* dev,
                 const std::uint8_t* ref, std::size_t elems, double atol,
                 double rtol, cudaStream_t stream, double* max_abs_out =
                     nullptr) {
  std::vector<float> act = dev_to_host(dev, elems, stream);
  const float* refp = reinterpret_cast<const float*>(ref);
  double max_abs = 0.0, max_rel = 0.0;
  std::size_t worst = 0;
  bool ok = true;
  for (std::size_t i = 0; i < elems; ++i) {
    const double a = act[i], rr = refp[i];
    const double ad = std::fabs(a - rr);
    if (ad > max_abs) max_abs = ad;
    const double rel = ad / std::fmax(std::fabs(rr), 1.0);
    if (rel > max_rel) max_rel = rel;
    if (!(ad <= atol + rtol * std::fabs(rr))) ok = false;
    if (!ok && i > worst) worst = i;
  }
  if (max_abs_out) *max_abs_out = max_abs;
  if (!ok) {
    std::fprintf(stderr, "  %-40s FAIL max_abs=%.9g max_rel=%.9g (n=%zu, "
                         "atol=%.4g rtol=%.4g)\n",
                 name.c_str(), max_abs, max_rel, elems, atol, rtol);
    return 1;
  }
  std::fprintf(stderr, "  %-40s max_abs=%.9g max_rel=%.9g OK\n", name.c_str(),
               max_abs, max_rel);
  return 0;
}

// Per-layer measured worsts (running max across the deterministic A+B run),
// captured so the depth-aware "smooth growth" property can be ENFORCED by the
// test (not just documented). Indexed by layer index L (0..23); entries for a
// layer of the wrong type stay 0.
struct PerLayerMeasured {
  double layer_final[24] = {0};
  double conv[24] = {0};
  double recur[24] = {0};
  double kv[24] = {0};
};

// Depth-aware per-layer envelope (docs §18.2): each entry is the measured
// worst of THAT layer across the deterministic A+B run * 1.3 + 1e-3 (a small
// margin). This replaces the old single shared per-category tolerance: L0 stays
// tight (~1e-3, in the v0.2 1e-2 regime), and each later layer gets its own
// (larger) ceiling from its OWN measured worst — so a late layer's large error
// never relaxes an early layer's tight ceiling. Entries for a layer of the
// wrong type are 0 (never compared). The final-norm / full-logits stay
// model-level (kFnAtol / kLogitsAtol). The test ENFORCES the depth-awareness
// below (check_smooth_growth): the later-depth measured worst must exceed the
// earlier-depth measured worst, so "smooth growth" is a property the test
// validates, not just prose.
static const double kLayerFinalAtol[24] = {
    0.001635, 0.003539, 0.006078, 0.006078, 0.006078, 0.01116, 0.01623,
    0.006078, 0.01116, 0.02131, 0.02131, 0.02131, 0.01623, 0.01623, 0.02131,
    0.02258, 0.02131, 0.02734, 0.06194, 0.06702, 0.09241, 0.09241, 0.08225,
    0.1026};
static const double kConvAtol[24] = {
    0.02131, 0.08225, 0.08225, 0, 0.1635, 0.1635, 0.1, 0, 0.1635, 0.1635,
    0.09748, 0, 0.1635, 0.1635, 0.1635, 0, 0.2448, 0.326, 0.2854, 0, 0.3666,
    0.326, 0.2041, 0};
static const double kRecurAtol[24] = {
    0.002032, 0.006816, 0.007876, 0, 0.01755, 0.0146, 0.007308, 0, 0.02716,
    0.02734, 0.004323, 0, 0.0331, 0.007564, 0.01969, 0, 0.04888, 0.04082,
    0.06235, 0, 0.05033, 0.04124, 0.02864, 0};
static const double kKvAtol[24] = {
    0, 0, 0, 0.08225, 0, 0, 0, 0.08225, 0, 0, 0, 0.1432, 0, 0, 0, 0.1432, 0,
    0, 0, 0.1026, 0, 0, 0, 0.1229};
static const double kFnAtol = 0.5799;    // final norm (measured max 0.4453)
static const double kLogitsAtol = 0.4073;  // FULL logits (measured max 0.3125)

// Enforce the depth-aware "smooth growth" on the MEASURED per-layer worsts:
// the later-depth (L16..L23) measured worst must exceed the earlier-depth
// (L0..L7) measured worst for every category. This catches a regression where
// the error no longer grows with depth (e.g. an early layer suddenly as large
// as a late layer). `used` marks which layers are compared for the category.
bool check_smooth_growth(const char* cat, const double m[24], bool used[24]) {
  double early = 0.0, late = 0.0;
  for (int L = 0; L < 24; ++L) {
    if (!used[L]) continue;
    if (L <= 7) early = std::fmax(early, m[L]);
    if (L >= 16) late = std::fmax(late, m[L]);
  }
  std::fprintf(stderr,
               "  smooth-growth %-10s early(L0..7)=%.6g late(L16..23)=%.6g\n",
               cat, early, late);
  return late > early;
}

// Load + validate one per-(token,layer) CUDLMG02 golden (config + tensor set).
int load_golden(const std::string& path, GoldenFileV2* g) {
  Status s = GoldenFileV2::load(path, g);
  if (!s.ok) {
    std::fprintf(stderr, "  load error %s: %s\n", path.c_str(), s.message.c_str());
    return 1;
  }
  if (!((*g).config() == Qwen35Config::qwen35_08b())) {
    std::fprintf(stderr, "  golden config != pinned 0.8B (%s)\n", path.c_str());
    return 1;
  }
  s = (*g).validate_golden_tensors();
  if (!s.ok) {
    std::fprintf(stderr, "  tensor-set error %s: %s\n", path.c_str(),
                 s.message.c_str());
    return 1;
  }
  return 0;
}

// Compare one token's full-forward output against the golden (scenario `sc`,
// position `p`). `sc` is "A" or "B". Returns 0 on success.
//
// Depth-aware per-layer envelope (docs §18.2): the per-layer comparisons use
// the per-layer tolerances kLayerFinalAtol[L] / kConvAtol[L] / kRecurAtol[L] /
// kKvAtol[L] (each = that layer's measured worst * 1.3 + 1e-3), NOT a single
// shared value — so a late layer's large error never relaxes an early layer's
// tight ceiling. The model-level final-norm / full-logits keep kFnAtol /
// kLogitsAtol. The measured per-layer worsts are captured into `measured` (the
// running max across A+B) so check_smooth_growth can validate the depth trend.
int compare_token(const Qwen35Model& model, const std::string& prefix,
                  const char* sc, int p, const Qwen35Config& cfg,
                  cudaStream_t stream, PerLayerMeasured* measured) {
  int rc = 0;
  std::fprintf(stderr, "[scenario %s, position %d]\n", sc, p);

  // ---- Model-level golden: embedding + final norm + FULL logits ----------
  {
    const std::string path =
        prefix + "_" + sc + "_model_p" + std::to_string(p) + ".cudalm";
    GoldenFileV2 g;
    Status s = GoldenFileV2::load(path, &g);
    CHECK(s.ok);
    if (!s.ok) {
      std::fprintf(stderr, "  load error %s: %s\n", path.c_str(), s.message.c_str());
      return 1;
    }
    const GoldenV2TensorInfo* ge = g.find("model.embedding_output");
    const GoldenV2TensorInfo* gn = g.find("model.final_norm_output");
    const GoldenV2TensorInfo* gl = g.find("model.logits");
    CHECK(ge && gn && gl);
    // Embedding is a pure row copy (no rounding) -> bit-exact.
    rc |= compare_bf16("model.embedding_output", model.embedding_output(),
                       ge->host_bytes,
                       static_cast<std::size_t>(cfg.hidden_size), 0.0, 0.0,
                       stream);
    rc |= compare_bf16("model.final_norm_output", model.final_norm_output(),
                       gn->host_bytes,
                       static_cast<std::size_t>(cfg.hidden_size), kFnAtol,
                       0.0, stream);
    rc |= compare_bf16("model.logits (FULL [vocab])", model.logits(),
                       gl->host_bytes, static_cast<std::size_t>(cfg.vocab_size),
                       kLogitsAtol, 0.0, stream);
  }

  // ---- Per-layer: final output + persistent-state hard gate --------------
  const int H = cfg.hidden_size;
  const int n_kv = cfg.n_kv_heads;
  const int hd = cfg.head_dim;
  const std::size_t rec_elems =
      static_cast<std::size_t>(cfg.lin_num_v_heads) *
      cfg.lin_key_head_dim * cfg.lin_value_head_dim;
  const std::size_t conv_elems =
      static_cast<std::size_t>(cfg.linear_conv_dim()) *
      cfg.linear_conv_state_len();

  for (int L = 0; L < cfg.num_hidden_layers; ++L) {
    const std::string path = prefix + "_" + sc + "_L" + std::to_string(L) +
                             "_p" + std::to_string(p) + ".cudalm";
    GoldenFileV2 g;
    const int lr = load_golden(path, &g);
    if (lr) {
      rc |= lr;  // load failed; skip this layer's comparisons
      continue;
    }
    // Layer final output (depth-aware tolerance kLayerFinalAtol[L]).
    const GoldenV2TensorInfo* gf = g.find("stage.final_output");
    CHECK(gf != nullptr);
    std::string lname = "layer.final_output[L" + std::to_string(L) + "]";
    double mf = 0.0;
    rc |= compare_bf16(lname, model.layer_final_output(L), gf->host_bytes,
                       static_cast<std::size_t>(H), kLayerFinalAtol[L], 0.0,
                       stream, &mf);
    if (measured) measured->layer_final[L] = std::fmax(measured->layer_final[L], mf);
    if (model.is_linear_attention(L)) {
      const GoldenV2TensorInfo* gca = g.find("state.conv_after");
      const GoldenV2TensorInfo* gra = g.find("state.recurrent_after");
      CHECK(gca && gra);
      const Qwen35DeltaNetLayer* d = model.delta(L);
      std::string cn = "DeltaNet.conv_after[L" + std::to_string(L) + "]";
      std::string rn = "DeltaNet.recurrent_after[L" + std::to_string(L) + "]";
      double mc = 0.0, mr = 0.0;
      rc |= compare_bf16(cn, d->conv_state(), gca->host_bytes, conv_elems,
                         kConvAtol[L], 0.0, stream, &mc);
      rc |= compare_fp32(rn, d->recurrent_state(), gra->host_bytes, rec_elems,
                         kRecurAtol[L], 0.0, stream, &mr);
      if (measured) {
        measured->conv[L] = std::fmax(measured->conv[L], mc);
        measured->recur[L] = std::fmax(measured->recur[L], mr);
      }
    } else {
      const Qwen35KvCache& kv = model.attention(L)->kv_cache();
      double mk = 0.0;
      for (int which = 0; which < 2; ++which) {
        const char* tname = which == 0 ? "kv.k_state" : "kv.v_state";
        const GoldenV2TensorInfo* gt = g.find(tname);
        CHECK(gt != nullptr);
        const __nv_bfloat16* cache = which == 0 ? kv.k() : kv.v();
        for (int n = 0; n < n_kv; ++n) {
          for (int t = 0; t <= p; ++t) {
            const std::size_t src_row =
                static_cast<std::size_t>(n) * (p + 1) + t;
            const std::uint8_t* row_ptr =
                reinterpret_cast<const std::uint8_t*>(cache) +
                kv.row_offset(n, t) * sizeof(__nv_bfloat16);
            std::string rn = std::string(tname) + "(L" + std::to_string(L) +
                             "," + std::to_string(n) + "," +
                             std::to_string(t) + ")";
            double mv = 0.0;
            rc |= compare_bf16(
                rn, reinterpret_cast<const __nv_bfloat16*>(row_ptr),
                gt->host_bytes + src_row * hd * sizeof(__nv_bfloat16),
                static_cast<std::size_t>(hd), kKvAtol[L], 0.0, stream, &mv);
            mk = std::fmax(mk, mv);
          }
        }
      }
      if (measured) measured->kv[L] = std::fmax(measured->kv[L], mk);
    }
  }
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: %s <out.cudalm> <checkpoint_dir> <python> <src_dir> "
                 "<golden_prefix> [--no-gen]\n",
                 argv[0]);
    return 2;
  }
  const std::string out = argv[1];
  const std::string ckpt = argv[2];
  const std::string py = argv[3];
  const std::string src = argv[4];
  const std::string gprefix = argv[5];
  const bool no_gen = (argc >= 7 && std::string(argv[6]) == "--no-gen");

  if (!no_gen && !file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr,
                 "[SKIP] qwen35 full forward: checkpoint absent at %s\n",
                 ckpt.c_str());
    return 77;
  }

  if (!no_gen) {
    // 1. Generate the full-model goldens (scenarios A + B).
    int rc = run_cmd(py + " " + src + "/tools/generate_qwen35_golden.py" +
                     " --cudalm " + out + " --checkpoint-dir " + ckpt +
                     " --fullmodel-prefix " + gprefix +
                     " --fullmodel-tokens-a 15" +
                     " --fullmodel-tokens-b 15,16,17");
    CHECK_EQ(rc, 0);
  } else {
    CHECK(file_exists(gprefix + "_A_model_p0.cudalm"));
    CHECK(file_exists(gprefix + "_B_model_p2.cudalm"));
  }

  // 2. Load the full model.
  WeightFileV2 file;
  Status s = WeightFileV2::load(out, &file);
  CHECK(s.ok);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Qwen35Model model;
  s = Qwen35Model::load(file, stream, &model);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "Qwen35Model::load: %s\n", s.message.c_str());
    return 1;
  }
  CHECK(model.loaded());
  const Qwen35Config& cfg = model.config();

  // Per-layer measured worsts (running max across the deterministic A+B run)
  // for the test-enforced depth-aware smooth-growth check (docs §18.2).
  PerLayerMeasured measured;

  int rc = 0;
  // ---- Scenario A: fresh p=0 --------------------------------------------
  model.reset_state(stream);
  model.forward_token(kTokenA[0], 0, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  rc |= compare_token(model, gprefix, "A", 0, cfg, stream, &measured);

  // ---- Scenario B: sequential p0->p1->p2 (runtime threads its own state) -
  model.reset_state(stream);
  for (int i = 0; i < kNB; ++i) {
    model.forward_token(kTokenB[i], i, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    rc |= compare_token(model, gprefix, "B", i, cfg, stream, &measured);
  }

  // ---- Depth-aware smooth-growth check (test-enforced, docs §18.2) --------
  // The bf16 rounding compounds with depth, so the later-depth (L16..23)
  // measured worst must exceed the earlier-depth (L0..7) measured worst for
  // every category. `used` marks the compared layers per category (all layers
  // for layer_final; DeltaNet layers for conv/recurrent; FA layers for kv).
  bool used_lf[24] = {}, used_cn[24] = {}, used_rc[24] = {}, used_kv[24] = {};
  for (int L = 0; L < 24; ++L) {
    used_lf[L] = true;
    if (cfg.is_full_attention(L))
      used_kv[L] = true;
    else {
      used_cn[L] = true;
      used_rc[L] = true;
    }
  }
  std::fprintf(stderr, "[depth-aware smooth-growth check]\n");
  bool sg = true;
  sg &= check_smooth_growth("layer_final", measured.layer_final, used_lf);
  sg &= check_smooth_growth("conv", measured.conv, used_cn);
  sg &= check_smooth_growth("recurrent", measured.recur, used_rc);
  sg &= check_smooth_growth("kv", measured.kv, used_kv);
  if (!sg) {
    std::fprintf(stderr,
                 "qwen35 full forward: SMOOTH-GROWTH VIOLATION (the measured "
                 "error no longer grows with depth)\n");
    rc |= 1;
  }

  if (rc != 0) {
    std::fprintf(stderr, "qwen35 full forward: COMPARISON FAILURES (rc=%d)\n",
                 rc);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 1;
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  TEST_PASS("qwen35_full_forward (A p0 + B p0->p1->p2, 24 layers + full "
            "logits + all states)");
  return 0;
}
