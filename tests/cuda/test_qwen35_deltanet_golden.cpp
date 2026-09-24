// CUDALM — Qwen3.5 Gated DeltaNet (layer 0) golden gate (Phase C, HARD GATE:
// native runtime vs quantized PyTorch reference).
//
// The Phase C hard gate is the PERSISTENT STATE in addition to the layer
// output. Three state-transition scenarios, each comparing the final output
// AND conv_state AND recurrent_state:
//
//   A (first token):      zero conv + recurrent state; run p=0.
//   B (consecutive):      runtime chains p=0 -> p=1 from its OWN p=0 state
//                         (continue from the previous token's runtime state).
//   C (seeded non-zero):  seed the runtime with an explicit non-zero conv +
//                         recurrent state, run p=0.
//
// Pipeline (all offline; skips with exit 77 when the real checkpoint is
// absent — it lives in /root/models, never in the repo):
//   1. tools/convert_qwen35.py         checkpoint -> .cudalm v2 (layer 0)
//   2. tools/generate_qwen35_golden.py pinned-transformers oracle with the
//      W4A16-dequantized GEMV weights -> CUDLMG02 goldens for the 3 scenarios
//      (same quantized weights build the state — no bf16/quantized mixing)
//   3. this binary:
//      * C++ parse of the .cudalm v2 (pinned config + layer tensor set)
//      * Qwen35LayerWeights upload (layer 0, a DeltaNet layer)
//      * per scenario: seed/reset the runtime state, run decode step(s),
//        compare all bf16 stages (compare_bf16_stages, 1e-2), stage.g (fp32)
//        and the persistent state: conv_state bf16 [6144,3] + recurrent_state
//        fp32 [16,128,128] (tight fp32 tolerance).
//
// Arguments:
//   <out_l0.cudalm> <checkpoint_dir> <golden_A> <golden_B> <golden_C>
//   <python> <src_dir>

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cudalm/cuda_check.h"
#include "cudalm/golden_loader_v2.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_deltanet.h"
#include "cudalm/stage_compare.h"
#include "cudalm/weight_loader_v2.h"

#include "../../tests/common/check.h"

using namespace cudalm;

namespace {

bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return static_cast<bool>(f);
}

int run_cmd(const std::string& cmd) {
  std::fprintf(stderr, "  $ %s\n", cmd.c_str());
  const int rc = std::system(cmd.c_str());
  if (rc != 0) std::fprintf(stderr, "  command failed (rc=%d)\n", rc);
  return rc;
}

// ---- DeltaNet bf16 stage accessor specs -------------------------------------
struct StageSpec {
  const char* name;
  std::size_t elems;
  const __nv_bfloat16* (*get)(const Qwen35DeltaNetLayer&);
};

#define ACCDN(nm)                                                    \
  const __nv_bfloat16* spec_##nm(const Qwen35DeltaNetLayer& b) {     \
    return b.stage_##nm();                                           \
  }
ACCDN(input)
ACCDN(rmsnorm1)
ACCDN(in_proj_qkv)
ACCDN(in_proj_z)
ACCDN(in_proj_b)
ACCDN(in_proj_a)
ACCDN(conv_out)
ACCDN(conv_silu)
ACCDN(q)
ACCDN(k)
ACCDN(v)
ACCDN(beta)
ACCDN(core_out)
ACCDN(gated_norm)
ACCDN(out_proj)
ACCDN(residual1)
ACCDN(rmsnorm2)
ACCDN(mlp_gate)
ACCDN(mlp_up)
ACCDN(silu_mul)
ACCDN(mlp_down)
ACCDN(final_output)

std::vector<StageSpec> dn_stage_specs(const Qwen35Config& c) {
  const std::size_t H = static_cast<std::size_t>(c.hidden_size);
  const std::size_t conv_dim = static_cast<std::size_t>(c.linear_conv_dim());
  const std::size_t key_dim = static_cast<std::size_t>(c.linear_key_dim());
  const std::size_t value_dim =
      static_cast<std::size_t>(c.linear_value_dim());
  const std::size_t inter = static_cast<std::size_t>(c.intermediate_size);
  const std::size_t n_heads = static_cast<std::size_t>(c.lin_num_v_heads);
  return {
      {"stage.input", H, spec_input},
      {"stage.rmsnorm1", H, spec_rmsnorm1},
      {"stage.in_proj_qkv", conv_dim, spec_in_proj_qkv},
      {"stage.in_proj_z", value_dim, spec_in_proj_z},
      {"stage.in_proj_b", n_heads, spec_in_proj_b},
      {"stage.in_proj_a", n_heads, spec_in_proj_a},
      {"stage.conv_out", conv_dim, spec_conv_out},
      {"stage.conv_silu", conv_dim, spec_conv_silu},
      {"stage.q", key_dim, spec_q},
      {"stage.k", key_dim, spec_k},
      {"stage.v", value_dim, spec_v},
      {"stage.beta", n_heads, spec_beta},
      {"stage.core_out", value_dim, spec_core_out},
      {"stage.gated_norm", value_dim, spec_gated_norm},
      {"stage.out_proj", H, spec_out_proj},
      {"stage.residual1", H, spec_residual1},
      {"stage.rmsnorm2", H, spec_rmsnorm2},
      {"stage.mlp_gate", inter, spec_mlp_gate},
      {"stage.mlp_up", inter, spec_mlp_up},
      {"stage.silu_mul", inter, spec_silu_mul},
      {"stage.mlp_down", H, spec_mlp_down},
      {"stage.final_output", H, spec_final_output},
  };
}

// Device -> host copy of a stage buffer.
template <typename T>
std::vector<T> dev_to_host(const T* dev, std::size_t n, cudaStream_t stream) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(T),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h;
}

// Compare a bf16 device stage against the golden host bytes (tolerance 1e-2).
int compare_bf16(const char* name, const __nv_bfloat16* dev,
                 const std::uint8_t* ref, std::size_t elems,
                 double* max_abs_out, cudaStream_t stream) {
  std::vector<__nv_bfloat16> act = dev_to_host(dev, elems, stream);
  const StageCompareResult r = compare_bf16_stages(
      reinterpret_cast<const std::uint16_t*>(ref),
      reinterpret_cast<const std::uint16_t*>(act.data()), elems);
  if (max_abs_out) *max_abs_out = r.max_abs_err;
  if (!r.ok) {
    std::fprintf(stderr, "  %s: FAIL max_abs_err=%.9g max_rel_err=%.9g "
                         "(idx=%zu, n=%zu)\n",
                 name, r.max_abs_err, r.max_rel_err, r.max_abs_idx, elems);
    return 1;
  }
  return 0;
}

// Compare an fp32 device array against the golden host bytes (rtol/atol).
int compare_fp32(const char* name, const float* dev, const std::uint8_t* ref,
                 std::size_t elems, double rtol, double atol,
                 cudaStream_t stream) {
  std::vector<float> act = dev_to_host(dev, elems, stream);
  const float* refp = reinterpret_cast<const float*>(ref);
  double max_abs = 0.0, max_rel = 0.0;
  std::size_t worst_idx = 0;
  double worst_ad = 0.0;
  bool ok = true;
  for (std::size_t i = 0; i < elems; ++i) {
    const double a = act[i], r = refp[i];
    const double ad = std::fabs(a - r);
    if (ad > max_abs) max_abs = ad;
    const double rel = ad / std::fmax(std::fabs(r), 1.0);
    if (rel > max_rel) max_rel = rel;
    if (ad > atol + rtol * std::fabs(r)) {
      if (ad > worst_ad) { worst_ad = ad; worst_idx = i; }
      ok = false;
    }
  }
  if (!ok) {
    std::fprintf(stderr, "  %s: FAIL max_abs=%.9g max_rel=%.9g "
                         "(worst elem %zu abs=%.3g, atol=%.3g rtol=%.3g, n=%zu)\n",
                 name, max_abs, max_rel, worst_idx, worst_ad, atol, rtol, elems);
    return 1;
  }
  std::fprintf(stderr, "  %-26s max_abs=%.9g max_rel=%.9g OK\n", name, max_abs,
               max_rel);
  return 0;
}

// Compare the persistent state (conv bf16 [conv_dim,3] + recurrent fp32
// [n_heads,head_dim,head_dim]) after a step. The state is the Phase C hard
// gate.
int compare_state(const Qwen35DeltaNetLayer& layer, const GoldenFileV2& g,
                  double rec_rtol, double rec_atol, cudaStream_t stream) {
  int rc = 0;
  const Qwen35Config& c = g.config();
  const std::size_t conv_dim = static_cast<std::size_t>(c.linear_conv_dim());
  const std::size_t n_heads = static_cast<std::size_t>(c.lin_num_v_heads);
  const std::size_t hd = static_cast<std::size_t>(c.lin_value_head_dim);

  const GoldenV2TensorInfo* ca = g.find("state.conv_after");
  CHECK(ca != nullptr);
  double ma = 0.0;
  rc |= compare_bf16("state.conv_after", layer.conv_state(), ca->host_bytes,
                     conv_dim * 3, &ma, stream);
  if (ma > 0.0)
    std::fprintf(stderr, "  state.conv_after           max_abs=%.9g OK\n", ma);

  const GoldenV2TensorInfo* ra = g.find("state.recurrent_after");
  CHECK(ra != nullptr);
  rc |= compare_fp32("state.recurrent_after", layer.recurrent_state(),
                     ra->host_bytes, n_heads * hd * hd, rec_rtol, rec_atol,
                     stream);
  return rc;
}

// Compare all bf16 stages + stage.g (fp32) + the persistent state.
int run_scenario(const char* label, Qwen35DeltaNetLayer& layer,
                 const GoldenFileV2& g, double rec_rtol, double rec_atol,
                 cudaStream_t stream) {
  std::fprintf(stderr, "\n[%s] (layer=%d pos=%d)\n", label, g.layer_idx(),
               g.position());
  int rc = 0;
  double worst = 0.0;
  for (const StageSpec& sp : dn_stage_specs(g.config())) {
    const GoldenV2TensorInfo* gt = g.find(sp.name);
    CHECK(gt != nullptr);
    double max_abs = 0.0;
    rc |= compare_bf16(sp.name, sp.get(layer), gt->host_bytes, sp.elems,
                       &max_abs, stream);
    if (max_abs > worst) worst = max_abs;
    std::fprintf(stderr, "  %-26s max_abs=%.9g OK\n", sp.name, max_abs);
  }
  const GoldenV2TensorInfo* gt_g = g.find("stage.g");
  CHECK(gt_g != nullptr);
  rc |= compare_fp32("stage.g", layer.stage_g(), gt_g->host_bytes,
                     static_cast<std::size_t>(g.config().lin_num_v_heads),
                     rec_rtol, rec_atol, stream);
  std::fprintf(stderr, "  worst bf16 stage max_abs_err = %.9g\n", worst);
  rc |= compare_state(layer, g, rec_rtol, rec_atol, stream);
  return rc;
}

// Load a golden file + run its pinned-config / tensor-set validation.
int load_golden(const std::string& path, GoldenFileV2* g) {
  Status s = GoldenFileV2::load(path, g);
  if (!s.ok) {
    std::fprintf(stderr, "  load error: %s\n", s.message.c_str());
    return 1;
  }
  if (!((*g).config() == Qwen35Config::qwen35_08b())) {
    std::fprintf(stderr, "  golden config != pinned 0.8B\n");
    return 1;
  }
  s = (*g).validate_golden_tensors();
  if (!s.ok) {
    std::fprintf(stderr, "  tensor-set error: %s\n", s.message.c_str());
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr,
                 "usage: %s <out_l0.cudalm> <checkpoint_dir> <golden_A> "
                 "<golden_B> <golden_C> <python> <src_dir>\n",
                 argv[0]);
    return 2;
  }
  const std::string out_l0 = argv[1];
  const std::string ckpt = argv[2];
  const std::string golden_a = argv[3];
  const std::string golden_b = argv[4];
  const std::string golden_c = argv[5];
  const std::string py = argv[6];
  const std::string src = argv[7];
  // Optional 8th arg "--no-gen": skip the offline Python convert + golden
  // generation and assume the .cudalm + the 3 goldens already exist. Used to
  // run compute-sanitizer (memcheck) on the CUDA-only path without spawning
  // child processes (compute-sanitizer's process reaper deadlocks on the
  // std::system Python children).
  const bool no_gen = (argc >= 9) && (std::string(argv[8]) == "--no-gen");

  if (!no_gen && !file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr,
                 "[SKIP] qwen35 delta-net golden: checkpoint not present at "
                 "%s\n",
                 ckpt.c_str());
    return 77;
  }

  // 1-2. Convert layer 0 + generate the 3 scenario goldens (offline Python
  //    toolchain; scenario C uses a deterministic non-zero seed). Skipped
  //    under --no-gen (memcheck) when the artifacts already exist.
  int rc = 0;
  if (!no_gen) {
    rc = run_cmd(py + " " + src + "/tools/convert_qwen35.py" +
                 " --checkpoint-dir " + ckpt + " --out " + out_l0 +
                 " --layers 0 --manifest " + out_l0 + ".manifest.json");
    CHECK_EQ(rc, 0);
    rc = run_cmd(py + " " + src + "/tools/generate_qwen35_golden.py" +
                 " --cudalm " + out_l0 + " --checkpoint-dir " + ckpt +
                 " --layer 0 --out " + golden_a + " --position 0" +
                 " --input-seed 20260209");
    CHECK_EQ(rc, 0);
    rc = run_cmd(py + " " + src + "/tools/generate_qwen35_golden.py" +
                 " --cudalm " + out_l0 + " --checkpoint-dir " + ckpt +
                 " --layer 0 --out " + golden_b + " --position 1" +
                 " --input-seed 20260209");
    CHECK_EQ(rc, 0);
    rc = run_cmd(py + " " + src + "/tools/generate_qwen35_golden.py" +
                 " --cudalm " + out_l0 + " --checkpoint-dir " + ckpt +
                 " --layer 0 --out " + golden_c + " --position 0" +
                 " --input-seed 20260209 --state-seed 20260707");
    CHECK_EQ(rc, 0);
  }

  // 3. C++ parse + pinned checks.
  WeightFileV2 file;
  Status s = WeightFileV2::load(out_l0, &file);
  CHECK(s.ok);
  CHECK(file.config() == Qwen35Config::qwen35_08b());
  s = file.validate_layer(0);
  CHECK(s.ok);

  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  CHECK(n >= 1);
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Qwen35LayerWeights weights;
  s = Qwen35LayerWeights::load(file, 0, stream, &weights);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "layer weight load error: %s\n", s.message.c_str());
    return 1;
  }
  CHECK(!weights.is_full_attention());  // must be a DeltaNet layer

  // Tolerances: bf16 stages use compare_bf16_stages (1e-2). The fp32 state
  // and stage.g use a tight fp32 tolerance (the state is the Phase C hard
  // gate; a real math bug is O(0.01+)). The l2norm and conv are replicated
  // bit-exactly (verified against the pinned torch ops), so the only residual
  // is the fp32 recurrence-contraction add order (my per-thread loop vs
  // torch's .sum) — measured <=6e-8 over these scenarios. atol=1e-5 +
  // rtol=1e-4 absorbs that with >100x headroom while still gating any real
  // bug by ~3+ orders of magnitude.
  const double rec_rtol = 1e-4;
  const double rec_atol = 1e-5;

  // ---- Scenario A: first token (zero conv + recurrent state) ---------------
  {
    GoldenFileV2 g;
    rc |= load_golden(golden_a, &g);
    if (!rc) {
      Qwen35DeltaNetLayer layer(weights, stream);
      layer.reset_state(stream);
      const GoldenV2TensorInfo* in = g.find("stage.input");
      CHECK(in != nullptr);
      const int H = g.config().hidden_size;
      DeviceBuffer x_in(static_cast<std::size_t>(H) * sizeof(__nv_bfloat16),
                        stream);
      x_in.copy_from_host(in->host_bytes, in->byte_size, stream);
      layer.forward(g.position(), x_in.data<__nv_bfloat16>(), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      rc |= run_scenario("scenario A: first token (zero state)", layer, g,
                         rec_rtol, rec_atol, stream);
    }
  }

  // ---- Scenario B: consecutive tokens (runtime p0 -> p1 chain) -------------
  {
    if (!rc) {
      GoldenFileV2 ga, gb;
      rc |= load_golden(golden_a, &ga);  // provides x_0 (p=0 input)
      rc |= load_golden(golden_b, &gb);  // reference (p=1)
      if (!rc) {
        Qwen35DeltaNetLayer layer(weights, stream);
        layer.reset_state(stream);
        const int H = gb.config().hidden_size;
        // p=0 with x_0 (from golden A), building the runtime's own state.
        {
          const GoldenV2TensorInfo* in = ga.find("stage.input");
          CHECK(in != nullptr);
          DeviceBuffer x0(
              static_cast<std::size_t>(H) * sizeof(__nv_bfloat16), stream);
          x0.copy_from_host(in->host_bytes, in->byte_size, stream);
          layer.forward(0, x0.data<__nv_bfloat16>(), stream);
        }
        // p=1 with x_1 (from golden B), from the runtime's own p=0 state.
        {
          const GoldenV2TensorInfo* in = gb.find("stage.input");
          CHECK(in != nullptr);
          DeviceBuffer x1(
              static_cast<std::size_t>(H) * sizeof(__nv_bfloat16), stream);
          x1.copy_from_host(in->host_bytes, in->byte_size, stream);
          layer.forward(gb.position(), x1.data<__nv_bfloat16>(), stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        rc |= run_scenario("scenario B: consecutive (p0->p1 chain)", layer, gb,
                           rec_rtol, rec_atol, stream);
      }
    }
  }

  // ---- Scenario C: seeded non-zero previous state --------------------------
  {
    if (!rc) {
      GoldenFileV2 g;
      rc |= load_golden(golden_c, &g);
      if (!rc) {
        Qwen35DeltaNetLayer layer(weights, stream);
        const Qwen35Config& c = g.config();
        const std::size_t conv_dim =
            static_cast<std::size_t>(c.linear_conv_dim());
        const std::size_t n_heads =
            static_cast<std::size_t>(c.lin_num_v_heads);
        const std::size_t hd =
            static_cast<std::size_t>(c.lin_value_head_dim);
        const GoldenV2TensorInfo* cb = g.find("state.conv_before");
        const GoldenV2TensorInfo* rb = g.find("state.recurrent_before");
        CHECK(cb != nullptr && rb != nullptr);
        std::vector<__nv_bfloat16> conv_host(conv_dim * 3);
        std::memcpy(conv_host.data(), cb->host_bytes, conv_dim * 3 * 2);
        std::vector<float> rec_host(n_heads * hd * hd);
        std::memcpy(rec_host.data(), rb->host_bytes, n_heads * hd * hd * 4);
        layer.seed_state(conv_host.data(), rec_host.data(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const GoldenV2TensorInfo* in = g.find("stage.input");
        CHECK(in != nullptr);
        const int H = c.hidden_size;
        DeviceBuffer x_in(static_cast<std::size_t>(H) * sizeof(__nv_bfloat16),
                          stream);
        x_in.copy_from_host(in->host_bytes, in->byte_size, stream);
        layer.forward(g.position(), x_in.data<__nv_bfloat16>(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        rc |= run_scenario("scenario C: seeded non-zero state", layer, g,
                           rec_rtol, rec_atol, stream);
      }
    }
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc == 0) TEST_PASS("test_qwen35_deltanet_golden");
  else std::fprintf(stderr, "test_qwen35_deltanet_golden FAILED\n");
  return rc;
}
