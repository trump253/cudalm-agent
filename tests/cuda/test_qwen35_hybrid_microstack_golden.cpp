// CUDALM — Qwen3.5 4-layer hybrid decoder micro-stack golden hard gate
// (v0.2, Phase D). CUDALM-native (runtime) + pinned-transformers oracle
// (tools/generate_qwen35_golden.py --microstack-prefix).
//
// Loads the REAL checkpoint layers 0-3 (3x Gated DeltaNet + 1x full
// attention) into the Qwen35HybridMicroStack and runs it as a 4-layer chain
// (layer 0 -> 1 -> 2 -> 3), comparing against a pinned-oracle micro-stack
// golden that runs the same quantized layers in order with the SAME W4A16
// weights (no bf16/quantized mixing). Each layer owns its own persistent
// state; the runtime must advance its OWN prior-step state (scenario B
// chains without a reset).
//
// Per (token, layer) the golden is one CUDLMG02 file (the historical
// per-layer container is reused unchanged). The micro-stack golden is the SET
// of these files; the layer L input == layer L-1 output chain is what the
// runtime must reproduce.
//
// Compared per layer: every bf16 pipeline stage (compare_bf16_stages, 1e-2 —
// docs/qwen35_architecture.md §14), the fp32 stage.g (DeltaNet), and the
// persistent STATE (the Phase D hard gate):
//   DeltaNet 0/1/2 : conv_state (bf16 [6144,3]) + recurrent_state (fp32
//                    [16,128,128], tight fp32 tolerance)
//   full-attention 3: K + V cache rows 0..p (bf16)
// plus the micro-stack final output (layer 3's final_output).
//
// Scenarios (each starts from a reset_state, i.e. zero persistent state):
//   A : p=0              (single first token)
//   B : p=0 -> 1 -> 2    (sequential decode; the runtime uses its own
//                          prior-step state, no reset in between)
//
// Usage:
//   test_qwen35_hybrid_microstack_golden <out_ms.cudalm> <checkpoint_dir>
//       <prefix_A> <prefix_B> <python> <src_dir> [--no-gen]
//
// The Python toolchain (convert_qwen35.py + generate_qwen35_golden.py) runs
// offline to produce <out_ms> (layers 0-3) + the per-(token,layer) goldens
// under prefix_A/prefix_B. --no-gen skips it (compute-sanitizer memcheck).
//
// No PyTorch in this binary. Provenance: CUDALM-native runtime + pinned
// oracle (see docs/provenance.md Phase D).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cudalm/cuda_check.h"
#include "cudalm/device_buffer.h"
#include "cudalm/golden_loader_v2.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_deltanet.h"
#include "cudalm/qwen35_full_attention.h"
#include "cudalm/qwen35_hybrid_microstack.h"
#include "cudalm/qwen35_kv_cache.h"
#include "cudalm/stage_compare.h"
#include "cudalm/weight_loader_v2.h"

#include "../../tests/common/check.h"

using namespace cudalm;

namespace {

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

// Device -> host copy of a buffer.
template <typename T>
std::vector<T> dev_to_host(const T* dev, std::size_t n, cudaStream_t stream) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(T),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h;
}

// Compare a bf16 device buffer against golden host bytes. The tolerance is
// per-token (see run_scenario): the first token uses the single-layer bar
// (1e-2, docs §14) and later tokens a looser bar that covers the 4-layer chain
// compounding (each layer's ~1 ulp GEMV re-association noise propagates through
// the prior layers' output + the persistent state).
int compare_bf16(const std::string& name, const __nv_bfloat16* dev,
                 const std::uint8_t* ref, std::size_t elems,
                 double* max_abs_out, cudaStream_t stream, double atol,
                 double rtol) {
  std::vector<__nv_bfloat16> act = dev_to_host(dev, elems, stream);
  const StageCompareResult r = compare_bf16_stages(
      reinterpret_cast<const std::uint16_t*>(ref),
      reinterpret_cast<const std::uint16_t*>(act.data()), elems, atol, rtol);
  if (max_abs_out) *max_abs_out = r.max_abs_err;
  if (!r.ok) {
    std::fprintf(stderr, "  %s: FAIL max_abs_err=%.9g max_rel_err=%.9g "
                         "(idx=%zu, n=%zu)\n",
                 name.c_str(), r.max_abs_err, r.max_rel_err, r.max_abs_idx,
                 elems);
    return 1;
  }
  std::fprintf(stderr, "  %-30s max_abs=%.9g OK\n", name.c_str(), r.max_abs_err);
  return 0;
}

// Compare an fp32 device buffer against golden host bytes (rtol/atol).
int compare_fp32(const std::string& name, const float* dev,
                 const std::uint8_t* ref, std::size_t elems, double rtol,
                 double atol, cudaStream_t stream) {
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
      if (ad > worst_ad) {
        worst_ad = ad;
        worst_idx = i;
      }
      ok = false;
    }
  }
  if (!ok) {
    std::fprintf(stderr,
                 "  %s: FAIL max_abs=%.9g max_rel=%.9g (worst elem %zu "
                 "abs=%.3g, atol=%.3g rtol=%.3g, n=%zu)\n",
                 name.c_str(), max_abs, max_rel, worst_idx, worst_ad, atol,
                 rtol, elems);
    return 1;
  }
  std::fprintf(stderr, "  %-30s max_abs=%.9g max_rel=%.9g OK\n", name.c_str(),
               max_abs, max_rel);
  return 0;
}

// ---- DeltaNet (layers 0-2) stage specs (22 bf16 stages + fp32 stage.g) ----
struct DeltaStageSpec {
  const char* name;
  std::size_t elems;
  const __nv_bfloat16* (*get)(const Qwen35DeltaNetLayer&);
};

#define ACCDN(nm)                                                          \
  const __nv_bfloat16* spec_dn_##nm(const Qwen35DeltaNetLayer& b) {        \
    return b.stage_##nm();                                                 \
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

std::vector<DeltaStageSpec> delta_specs(const Qwen35Config& c) {
  const std::size_t H = static_cast<std::size_t>(c.hidden_size);
  const std::size_t conv_dim = static_cast<std::size_t>(c.linear_conv_dim());
  const std::size_t key_dim = static_cast<std::size_t>(c.linear_key_dim());
  const std::size_t value_dim =
      static_cast<std::size_t>(c.linear_value_dim());
  const std::size_t inter = static_cast<std::size_t>(c.intermediate_size);
  const std::size_t n_heads = static_cast<std::size_t>(c.lin_num_v_heads);
  return {
      {"stage.input", H, spec_dn_input},
      {"stage.rmsnorm1", H, spec_dn_rmsnorm1},
      {"stage.in_proj_qkv", conv_dim, spec_dn_in_proj_qkv},
      {"stage.in_proj_z", value_dim, spec_dn_in_proj_z},
      {"stage.in_proj_b", n_heads, spec_dn_in_proj_b},
      {"stage.in_proj_a", n_heads, spec_dn_in_proj_a},
      {"stage.conv_out", conv_dim, spec_dn_conv_out},
      {"stage.conv_silu", conv_dim, spec_dn_conv_silu},
      {"stage.q", key_dim, spec_dn_q},
      {"stage.k", key_dim, spec_dn_k},
      {"stage.v", value_dim, spec_dn_v},
      {"stage.beta", n_heads, spec_dn_beta},
      {"stage.core_out", value_dim, spec_dn_core_out},
      {"stage.gated_norm", value_dim, spec_dn_gated_norm},
      {"stage.out_proj", H, spec_dn_out_proj},
      {"stage.residual1", H, spec_dn_residual1},
      {"stage.rmsnorm2", H, spec_dn_rmsnorm2},
      {"stage.mlp_gate", inter, spec_dn_mlp_gate},
      {"stage.mlp_up", inter, spec_dn_mlp_up},
      {"stage.silu_mul", inter, spec_dn_silu_mul},
      {"stage.mlp_down", H, spec_dn_mlp_down},
      {"stage.final_output", H, spec_dn_final_output},
  };
}

// ---- full-attention (layer 3) stage specs (21 bf16 stages) ---------------
struct FaStageSpec {
  const char* name;
  std::size_t elems;
  const __nv_bfloat16* (*get)(const Qwen35FullAttentionLayer&);
};

#define ACCFA(nm)                                                          \
  const __nv_bfloat16* spec_fa_##nm(const Qwen35FullAttentionLayer& b) {   \
    return b.stage_##nm();                                                 \
  }
ACCFA(input)
ACCFA(rmsnorm1)
ACCFA(q_gate)
ACCFA(q)
ACCFA(att_gate)
ACCFA(k)
ACCFA(v)
ACCFA(q_norm)
ACCFA(k_norm)
ACCFA(rope_q)
ACCFA(rope_k)
ACCFA(attention_raw)
ACCFA(attention_gated)
ACCFA(o_proj)
ACCFA(residual1)
ACCFA(rmsnorm2)
ACCFA(mlp_gate)
ACCFA(mlp_up)
ACCFA(silu_mul)
ACCFA(mlp_down)
ACCFA(final_output)

std::vector<FaStageSpec> fa_specs(const Qwen35Config& c) {
  const std::size_t H = static_cast<std::size_t>(c.hidden_size);
  const std::size_t qo =
      static_cast<std::size_t>(c.n_heads) * static_cast<std::size_t>(c.head_dim);
  const std::size_t kvo = static_cast<std::size_t>(c.n_kv_heads) *
                          static_cast<std::size_t>(c.head_dim);
  const std::size_t inter = static_cast<std::size_t>(c.intermediate_size);
  return {
      {"stage.input", H, spec_fa_input},
      {"stage.rmsnorm1", H, spec_fa_rmsnorm1},
      {"stage.q_gate", 2 * qo, spec_fa_q_gate},
      {"stage.q", qo, spec_fa_q},
      {"stage.att_gate", qo, spec_fa_att_gate},
      {"stage.k", kvo, spec_fa_k},
      {"stage.v", kvo, spec_fa_v},
      {"stage.q_norm", qo, spec_fa_q_norm},
      {"stage.k_norm", kvo, spec_fa_k_norm},
      {"stage.rope_q", qo, spec_fa_rope_q},
      {"stage.rope_k", kvo, spec_fa_rope_k},
      {"stage.attention_raw", qo, spec_fa_attention_raw},
      {"stage.attention_gated", qo, spec_fa_attention_gated},
      {"stage.o_proj", H, spec_fa_o_proj},
      {"stage.residual1", H, spec_fa_residual1},
      {"stage.rmsnorm2", H, spec_fa_rmsnorm2},
      {"stage.mlp_gate", inter, spec_fa_mlp_gate},
      {"stage.mlp_up", inter, spec_fa_mlp_up},
      {"stage.silu_mul", inter, spec_fa_silu_mul},
      {"stage.mlp_down", H, spec_fa_mlp_down},
      {"stage.final_output", H, spec_fa_final_output},
  };
}

// Compare one DeltaNet layer: all bf16 stages + stage.g (fp32) + the
// persistent state (conv_state bf16 + recurrent_state fp32).
int compare_delta_layer(const Qwen35HybridMicroStack& stack, int L,
                        const GoldenFileV2& g, double bf_atol, double bf_rtol,
                        double fp_rtol, double fp_atol, cudaStream_t stream) {
  Qwen35DeltaNetLayer* d = stack.delta(L);
  CHECK(d != nullptr);
  std::fprintf(stderr, "  [layer %d = DeltaNet] pos=%d\n", L, g.position());
  int rc = 0;
  for (const DeltaStageSpec& sp : delta_specs(g.config())) {
    const GoldenV2TensorInfo* gt = g.find(sp.name);
    CHECK(gt != nullptr);
    rc |= compare_bf16(sp.name, sp.get(*d), gt->host_bytes, sp.elems, nullptr,
                       stream, bf_atol, bf_rtol);
  }
  const GoldenV2TensorInfo* gt_g = g.find("stage.g");
  CHECK(gt_g != nullptr);
  rc |= compare_fp32("stage.g", d->stage_g(), gt_g->host_bytes,
                     static_cast<std::size_t>(g.config().lin_num_v_heads),
                     fp_rtol, fp_atol, stream);
  // Persistent state (the Phase D hard gate for this layer).
  const Qwen35Config& c = g.config();
  const std::size_t conv_dim = static_cast<std::size_t>(c.linear_conv_dim());
  const std::size_t n_heads = static_cast<std::size_t>(c.lin_num_v_heads);
  const std::size_t hd = static_cast<std::size_t>(c.lin_value_head_dim);
  const GoldenV2TensorInfo* ca = g.find("state.conv_after");
  CHECK(ca != nullptr);
  rc |= compare_bf16("state.conv_after", d->conv_state(), ca->host_bytes,
                     conv_dim * 3, nullptr, stream, bf_atol, bf_rtol);
  const GoldenV2TensorInfo* ra = g.find("state.recurrent_after");
  CHECK(ra != nullptr);
  rc |= compare_fp32("state.recurrent_after", d->recurrent_state(),
                     ra->host_bytes, n_heads * hd * hd, fp_rtol, fp_atol,
                     stream);
  return rc;
}

// Compare one full-attention layer: all bf16 stages + the KV cache rows 0..p.
int compare_fa_layer(const Qwen35HybridMicroStack& stack, int L,
                     const GoldenFileV2& g, double bf_atol, double bf_rtol,
                     cudaStream_t stream) {
  Qwen35FullAttentionLayer* a = stack.attention(L);
  CHECK(a != nullptr);
  std::fprintf(stderr, "  [layer %d = full-attention] pos=%d\n", L,
               g.position());
  int rc = 0;
  for (const FaStageSpec& sp : fa_specs(g.config())) {
    const GoldenV2TensorInfo* gt = g.find(sp.name);
    CHECK(gt != nullptr);
    rc |= compare_bf16(sp.name, sp.get(*a), gt->host_bytes, sp.elems, nullptr,
                       stream, bf_atol, bf_rtol);
  }
  // KV cache state (the Phase D hard gate for this layer): rows 0..position.
  const Qwen35Config& c = g.config();
  const int pos = g.position();
  const int n_kv = c.n_kv_heads;
  const int hd = c.head_dim;
  Qwen35KvCache& kv = a->kv_cache();
  for (int which = 0; which < 2; ++which) {
    const char* tname = which == 0 ? "kv.k_state" : "kv.v_state";
    const GoldenV2TensorInfo* gt = g.find(tname);
    CHECK(gt != nullptr);
    const __nv_bfloat16* cache =
        which == 0 ? kv.k() : kv.v();
    for (int n = 0; n < n_kv; ++n) {
      for (int t = 0; t <= pos; ++t) {
        const std::size_t src_row =
            static_cast<std::size_t>(n) * (pos + 1) + t;
        const std::uint8_t* row_ptr =
            reinterpret_cast<const std::uint8_t*>(cache) +
            kv.row_offset(n, t) * sizeof(__nv_bfloat16);
        const std::string row_name =
            std::string(tname) + " (" + std::to_string(n) + "," +
            std::to_string(t) + ")";
        rc |= compare_bf16(row_name,
                           reinterpret_cast<const __nv_bfloat16*>(row_ptr),
                           gt->host_bytes +
                               src_row * hd * sizeof(__nv_bfloat16),
                           static_cast<std::size_t>(hd), nullptr, stream,
                           bf_atol, bf_rtol);
      }
    }
  }
  return rc;
}

// Load + validate one per-(token,layer) CUDLMG02 golden.
int load_golden(const std::string& path, GoldenFileV2* g) {
  Status s = GoldenFileV2::load(path, g);
  if (!s.ok) {
    std::fprintf(stderr, "  load error %s: %s\n", path.c_str(),
                 s.message.c_str());
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

// Run one scenario: for each token t (in order), feed the golden's micro-stack
// input to the stack (which advances its OWN persistent state), then compare
// every layer's stages + state against the per-(token,layer) golden.
//
// Per-token tolerance (the rationale is the key Phase D correctness argument):
//   t == 0 : the layer-0 input is the seeded micro-stack input (identical in
//            runtime + oracle), so the whole chain is clean — the first token
//            uses the single-layer bar (bf16 1e-2, docs §14; the fp32 state is
//            bit-exact, so the Phase C fp32 bar 1e-5/1e-4 holds).
//   t  >= 1 : each layer's input is the PREVIOUS layer's output, so each layer
//            inherits + amplifies the prior layers' ~1 ulp W4A16 GEMV
//            re-association noise; the persistent state (DeltaNet recurrent,
//            full-attention KV) carries that noise forward. Observed worst
//            over the pinned seed: bf16 stages 3.9e-2, fp32 state 3.1e-3.
//            We gate at bf16 5e-2 and fp32 5e-3/1e-2 (~2x margin) — still
//            orders of magnitude below any O(1) wrong-stage / wrong-index /
//            wrong-weight / state-aliasing bug, which the first-token bar and
//            the per-layer phase B/C gates already pin tightly.
int run_scenario(const char* label, Qwen35HybridMicroStack& stack,
                 const std::string& prefix, const std::vector<int>& tokens,
                 cudaStream_t stream) {
  std::fprintf(stderr, "\n=== %s ===\n", label);
  int rc = 0;
  for (int t : tokens) {
    std::fprintf(stderr, "\n-- token %d --\n", t);
    GoldenFileV2 g[4];
    // Golden-load status is tracked SEPARATELY from the comparison status so
    // a failure on token t still compares every later token t' (the hard gate
    // must cover the full multi-token sequence, not stop at the first miss).
    int load_rc = 0;
    for (int L = 0; L < 4; ++L) {
      const std::string p =
          prefix + "_L" + std::to_string(L) + "_p" + std::to_string(t) +
          ".cudalm";
      load_rc |= load_golden(p, &g[L]);
    }
    if (load_rc != 0) {
      rc |= 1;
      continue;
    }
    const bool first = (t == 0);
    const double bf_atol = first ? 1e-2 : 5e-2;
    const double bf_rtol = first ? 1e-2 : 5e-2;
    const double fp_rtol = first ? 1e-4 : 1e-2;
    const double fp_atol = first ? 1e-5 : 5e-3;
    // Feed the golden's micro-stack input (layer-0 stage.input) to the stack.
    const GoldenV2TensorInfo* xin = g[0].find("stage.input");
    CHECK(xin != nullptr);
    DeviceBuffer xbuf(xin->byte_size);
    xbuf.copy_from_host(xin->host_bytes, xin->byte_size, stream);
    stack.forward(t, xbuf.data<__nv_bfloat16>(), stream);
    rc |= compare_delta_layer(stack, 0, g[0], bf_atol, bf_rtol, fp_rtol,
                              fp_atol, stream);
    rc |= compare_delta_layer(stack, 1, g[1], bf_atol, bf_rtol, fp_rtol,
                              fp_atol, stream);
    rc |= compare_delta_layer(stack, 2, g[2], bf_atol, bf_rtol, fp_rtol,
                              fp_atol, stream);
    rc |= compare_fa_layer(stack, 3, g[3], bf_atol, bf_rtol, stream);
    // Micro-stack final output == layer 3's final output (the chain endpoint).
    const GoldenV2TensorInfo* fin = g[3].find("stage.final_output");
    CHECK(fin != nullptr);
    rc |= compare_bf16("microstack.final_output (== layer3 final)",
                       stack.final_output(), fin->host_bytes,
                       static_cast<std::size_t>(g[3].config().hidden_size),
                       nullptr, stream, bf_atol, bf_rtol);
  }
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr,
                 "usage: %s <out_ms.cudalm> <checkpoint_dir> <prefix_A> "
                 "<prefix_B> <python> <src_dir> [--no-gen]\n",
                 argv[0]);
    return 2;
  }
  const std::string out_ms = argv[1];
  const std::string ckpt = argv[2];
  const std::string prefix_a = argv[3];
  const std::string prefix_b = argv[4];
  const std::string py = argv[5];
  const std::string src = argv[6];
  // Optional 8th arg "--no-gen": skip the offline Python convert + golden
  // generation and assume <out_ms> + the per-(token,layer) goldens already
  // exist. Used to run compute-sanitizer (memcheck) on the CUDA-only path
  // (compute-sanitizer's process reaper deadlocks on std::system children).
  const bool no_gen = (argc >= 8) && (std::string(argv[7]) == "--no-gen");

  if (!no_gen && !file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr,
                 "[SKIP] qwen35 hybrid micro-stack golden: checkpoint not "
                 "present at %s\n",
                 ckpt.c_str());
    return 77;
  }

  if (!no_gen) {
    // 1. Convert the real checkpoint layers 0-3 (Python, offline).
    int rc = run_cmd(py + " " + src + "/tools/convert_qwen35.py" +
                     " --checkpoint-dir " + ckpt + " --out " + out_ms +
                     " --layers 0,1,2,3 --manifest " + out_ms +
                     ".manifest.json");
    CHECK_EQ(rc, 0);
    // 2. Scenario A golden (p=0).
    rc = run_cmd(py + " " + src + "/tools/generate_qwen35_golden.py" +
                 " --cudalm " + out_ms + " --checkpoint-dir " + ckpt +
                 " --microstack-prefix " + prefix_a + " --tokens 0" +
                 " --input-seed 20260209");
    CHECK_EQ(rc, 0);
    // 3. Scenario B golden (sequential p=0,1,2).
    rc = run_cmd(py + " " + src + "/tools/generate_qwen35_golden.py" +
                 " --cudalm " + out_ms + " --checkpoint-dir " + ckpt +
                 " --microstack-prefix " + prefix_b + " --tokens 0,1,2" +
                 " --input-seed 20260209");
    CHECK_EQ(rc, 0);
  }

  // C++ parse + pinned checks.
  WeightFileV2 file;
  Status s = WeightFileV2::load(out_ms, &file);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "load error: %s\n", s.message.c_str());
    return 1;
  }
  CHECK(file.config() == Qwen35Config::qwen35_08b());
  for (int i = 0; i < 4; ++i) {
    Status vl = file.validate_layer(i);
    CHECK(vl.ok);
    if (!vl.ok) {
      std::fprintf(stderr, "validate layer %d: %s\n", i, vl.message.c_str());
      return 1;
    }
  }

  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  CHECK(n >= 1);
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Qwen35HybridMicroStack stack;
  s = Qwen35HybridMicroStack::load(file, stream, &stack);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "micro-stack load error: %s\n", s.message.c_str());
    return 1;
  }
  CHECK(stack.loaded());
  // Pinned hybrid pattern for the first 4 layers.
  CHECK(stack.is_delta(0) && stack.is_delta(1) && stack.is_delta(2));
  CHECK(stack.is_att(3));

  // Scenario A: p=0 (fresh zero state).
  stack.reset_state(stream);
  int rc = run_scenario("scenario A (p=0)", stack, prefix_a, {0}, stream);

  // Scenario B: sequential p=0 -> 1 -> 2 (its own prior-step state).
  stack.reset_state(stream);
  rc |= run_scenario("scenario B (p=0,1,2 sequential)", stack, prefix_b,
                     {0, 1, 2}, stream);

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc == 0) {
    std::fprintf(stderr, "[PASS] test_qwen35_hybrid_microstack_golden\n");
  } else {
    std::fprintf(stderr, "test_qwen35_hybrid_microstack_golden FAILED\n");
  }
  return rc;
}
