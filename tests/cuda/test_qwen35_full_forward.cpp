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
// tuned to the measured error, not a guess).
int compare_bf16(const std::string& name, const __nv_bfloat16* dev,
                 const std::uint8_t* ref, std::size_t elems, double atol,
                 double rtol, cudaStream_t stream) {
  std::vector<__nv_bfloat16> act = dev_to_host(dev, elems, stream);
  const StageCompareResult r = compare_bf16_stages(
      reinterpret_cast<const std::uint16_t*>(ref),
      reinterpret_cast<const std::uint16_t*>(act.data()), elems, atol, rtol);
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
                 double rtol, cudaStream_t stream) {
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
// Per-category minimal necessary envelope (docs §18.2): each tolerance is the
// measured max error of that category across the deterministic A+B run, with a
// ~10-15% margin. The 24-layer bf16 chain compounds per-layer ~1-ulp GEMV
// re-association noise; the final-norm (1+w) amplification and the [vocab]
// logits GEMV push the model-level outputs higher. No category is a sudden
// jump (the growth is smooth with depth) — this is rounding compounding, not a
// wrong-stage / wrong-weight / wrong-dtype bug (which would be O(1)).
int compare_token(const Qwen35Model& model, const std::string& prefix,
                  const char* sc, int p, const Qwen35Config& cfg,
                  cudaStream_t stream, double fn_atol, double logits_atol,
                  double layer_atol, double conv_atol, double kv_atol,
                  double recur_atol) {
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
                       static_cast<std::size_t>(cfg.hidden_size), fn_atol,
                       0.0, stream);
    rc |= compare_bf16("model.logits (FULL [vocab])", model.logits(),
                       gl->host_bytes, static_cast<std::size_t>(cfg.vocab_size),
                       logits_atol, 0.0, stream);
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
    // Layer final output.
    const GoldenV2TensorInfo* gf = g.find("stage.final_output");
    CHECK(gf != nullptr);
    std::string lname = "layer.final_output[L" + std::to_string(L) + "]";
    rc |= compare_bf16(lname, model.layer_final_output(L), gf->host_bytes,
                       static_cast<std::size_t>(H), layer_atol, 0.0, stream);
    if (model.is_linear_attention(L)) {
      const GoldenV2TensorInfo* gca = g.find("state.conv_after");
      const GoldenV2TensorInfo* gra = g.find("state.recurrent_after");
      CHECK(gca && gra);
      const Qwen35DeltaNetLayer* d = model.delta(L);
      std::string cn = "DeltaNet.conv_after[L" + std::to_string(L) + "]";
      std::string rn = "DeltaNet.recurrent_after[L" + std::to_string(L) + "]";
      rc |= compare_bf16(cn, d->conv_state(), gca->host_bytes, conv_elems,
                         conv_atol, 0.0, stream);
      rc |= compare_fp32(rn, d->recurrent_state(), gra->host_bytes, rec_elems,
                         recur_atol, 0.0, stream);
    } else {
      const Qwen35KvCache& kv = model.attention(L)->kv_cache();
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
            rc |= compare_bf16(
                rn, reinterpret_cast<const __nv_bfloat16*>(row_ptr),
                gt->host_bytes + src_row * hd * sizeof(__nv_bfloat16),
                static_cast<std::size_t>(hd), kv_atol, 0.0, stream);
          }
        }
      }
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

  // Measured minimal necessary envelope (docs §18.2): each value is the
  // measured max error of its category across the deterministic A+B run
  // (below) with a ~10-15% margin. The embedding is bit-exact (0.0).
  const double kFnAtol = 0.5;     // final norm   (measured max 0.4453)
  const double kLogitsAtol = 0.35;  // FULL logits  (measured max 0.3125)
  const double kLayerAtol = 0.09;  // 24x layer finals (measured max 0.0781 @ L23)
  const double kConvAtol = 0.31;   // DeltaNet conv  (measured max 0.2813, scenario B)
  const double kKvAtol = 0.12;     // FA KV rows     (measured max 0.1094 k / 0.0938 v)
  const double kRecurAtol = 0.055; // DeltaNet recur fp32 (measured max 0.0472)

  int rc = 0;
  // ---- Scenario A: fresh p=0 --------------------------------------------
  model.reset_state(stream);
  model.forward_token(kTokenA[0], 0, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  rc |= compare_token(model, gprefix, "A", 0, cfg, stream, kFnAtol, kLogitsAtol,
                      kLayerAtol, kConvAtol, kKvAtol, kRecurAtol);

  // ---- Scenario B: sequential p0->p1->p2 (runtime threads its own state) -
  model.reset_state(stream);
  for (int i = 0; i < kNB; ++i) {
    model.forward_token(kTokenB[i], i, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    rc |= compare_token(model, gprefix, "B", i, cfg, stream, kFnAtol,
                        kLogitsAtol, kLayerAtol, kConvAtol, kKvAtol,
                        kRecurAtol);
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
