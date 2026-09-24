// CUDALM — Qwen3.5 full-attention (layer 3) golden gate (Phase B, HARD
// GATE: native runtime vs quantized PyTorch reference).
//
// Pipeline (all offline; skips with exit 77 when the real checkpoint is
// absent — it lives in /root/models, never in the repo):
//   1. tools/convert_qwen35.py       checkpoint -> .cudalm v2 (layer 3)
//   2. tools/generate_qwen35_golden.py  pinned-transformers oracle with the
//      W4A16-dequantized GEMV weights -> CUDLMG02 goldens at p=0 and p=5
//      (seeded 0.05*randn bf16 hidden states; p=5 carries a non-zero,
//      non-trivial KV history of 5 rows)
//   3. this binary:
//      * C++ parse of the .cudalm v2 (pinned config + layer tensor set)
//      * Qwen35LayerWeights upload (layer 3)
//      * per golden: seed the runtime Qwen35KvCache from the golden KV
//        history rows (BIT-EXACT copy — pins the cache layout), run ONE
//        decode step at the golden position, compare all 21 stages
//        (compare_bf16_stages, 1e-2 — docs/qwen35_architecture.md §14)
//      * p=0 invariants (BIT-EXACT): rope identity (rope_q == q,
//        rope_k == k — cos(0)=1/sin(0)=0 exact in bf16) and
//        attention_raw == v rows (probs=[1] exact)
//      * KV state: history rows BIT-EXACT vs the golden (copy round-trip);
//        the current-position row within tolerance vs the golden AND
//        BIT-EXACT equal to the runtime's own stage.rope_k / stage.v rows
//
// Arguments:
//   <out_l3.cudalm> <checkpoint_dir> <golden_p0> <golden_p5> <python> <src_dir>
//
// The quantization fidelity report (official bf16 vs quantized reference)
// is written by the generator to <golden_p0>.fidelity.json — REPORT ONLY,
// never checked here (docs §11 A/B split).

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cudalm/cuda_check.h"
#include "cudalm/golden_loader_v2.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_full_attention.h"
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

struct StageSpec {
  const char* name;  // golden tensor name
  std::size_t elems; // bf16 element count
  const __nv_bfloat16* (*get)(const Qwen35FullAttentionLayer&);
};

#define ACC(nm)                                                              \
  const __nv_bfloat16* spec_##nm(const Qwen35FullAttentionLayer& b) {        \
    return b.stage_##nm();                                                   \
  }
ACC(input)
ACC(rmsnorm1)
ACC(q_gate)
ACC(q)
ACC(att_gate)
ACC(k)
ACC(v)
ACC(q_norm)
ACC(k_norm)
ACC(rope_q)
ACC(rope_k)
ACC(attention_raw)
ACC(attention_gated)
ACC(o_proj)
ACC(residual1)
ACC(rmsnorm2)
ACC(mlp_gate)
ACC(mlp_up)
ACC(silu_mul)
ACC(mlp_down)
ACC(final_output)

std::vector<StageSpec> stage_specs(const Qwen35Config& c) {
  const std::size_t H = static_cast<std::size_t>(c.hidden_size);
  const std::size_t qo = static_cast<std::size_t>(c.n_heads) * c.head_dim;
  const std::size_t kvo = static_cast<std::size_t>(c.n_kv_heads) * c.head_dim;
  const std::size_t inter = static_cast<std::size_t>(c.intermediate_size);
  return {
      {"stage.input", H, spec_input},
      {"stage.rmsnorm1", H, spec_rmsnorm1},
      {"stage.q_gate", 2 * qo, spec_q_gate},
      {"stage.q", qo, spec_q},
      {"stage.att_gate", qo, spec_att_gate},
      {"stage.k", kvo, spec_k},
      {"stage.v", kvo, spec_v},
      {"stage.q_norm", qo, spec_q_norm},
      {"stage.k_norm", kvo, spec_k_norm},
      {"stage.rope_q", qo, spec_rope_q},
      {"stage.rope_k", kvo, spec_rope_k},
      {"stage.attention_raw", qo, spec_attention_raw},
      {"stage.attention_gated", qo, spec_attention_gated},
      {"stage.o_proj", H, spec_o_proj},
      {"stage.residual1", H, spec_residual1},
      {"stage.rmsnorm2", H, spec_rmsnorm2},
      {"stage.mlp_gate", inter, spec_mlp_gate},
      {"stage.mlp_up", inter, spec_mlp_up},
      {"stage.silu_mul", inter, spec_silu_mul},
      {"stage.mlp_down", H, spec_mlp_down},
      {"stage.final_output", H, spec_final_output},
  };
}

// Compare one device stage tensor against the golden host bytes.
int compare_stage(const char* name, const __nv_bfloat16* dev,
                  const std::uint8_t* ref_bytes, std::size_t elems,
                  double* max_abs_out) {
  std::vector<__nv_bfloat16> act(elems);
  DeviceBuffer tmp(elems * sizeof(__nv_bfloat16));
  CUDA_CHECK(cudaMemcpyAsync(tmp.data(), dev, tmp.bytes(),
                             cudaMemcpyDeviceToDevice, 0));
  CUDA_CHECK(cudaStreamSynchronize(0));
  tmp.copy_to_host(act.data(), act.size() * sizeof(__nv_bfloat16), 0);
  const StageCompareResult r = compare_bf16_stages(
      reinterpret_cast<const std::uint16_t*>(ref_bytes),
      reinterpret_cast<const std::uint16_t*>(act.data()), elems);
  if (max_abs_out) *max_abs_out = r.max_abs_err;
  if (!r.ok) {
    std::fprintf(stderr, "  stage %s: FAIL max_abs_err=%.9g max_rel_err=%.9g"
                         " (idx=%zu, n=%zu)\n", name, r.max_abs_err,
                 r.max_rel_err, r.max_abs_idx, elems);
    return 1;
  }
  return 0;
}

// Bit-exact equality of a device stage row against host bytes.
int cmp_stage_bits(const char* name, const __nv_bfloat16* dev,
                   const std::uint8_t* ref_bytes, std::size_t elems) {
  std::vector<__nv_bfloat16> act(elems);
  DeviceBuffer tmp(elems * sizeof(__nv_bfloat16));
  CUDA_CHECK(cudaMemcpyAsync(tmp.data(), dev, tmp.bytes(),
                             cudaMemcpyDeviceToDevice, 0));
  CUDA_CHECK(cudaStreamSynchronize(0));
  tmp.copy_to_host(act.data(), act.size() * sizeof(__nv_bfloat16), 0);
  if (std::memcmp(act.data(), ref_bytes, elems * sizeof(__nv_bfloat16)) != 0) {
    std::fprintf(stderr, "  stage %s: bit-exact check FAILED\n", name);
    return 1;
  }
  return 0;
}

int run_golden(const std::string& path, const Qwen35LayerWeights& weights,
               cudaStream_t stream) {
  std::fprintf(stderr, "\n[golden] %s\n", path.c_str());
  GoldenFileV2 g;
  Status s = GoldenFileV2::load(path, &g);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "  load error: %s\n", s.message.c_str());
    return 1;
  }
  // Pinned contract.
  CHECK(g.config() == Qwen35Config::qwen35_08b());
  s = g.validate_golden_tensors();
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "  tensor-set error: %s\n", s.message.c_str());
    return 1;
  }
  const int position = g.position();
  const int n_kv = g.config().n_kv_heads;
  const int hd = g.config().head_dim;

  // Seed the runtime KV cache from the golden history rows t < position
  // (BIT-EXACT copy — this pins the cache layout + row mapping).
  Qwen35FullAttentionLayer layer(weights, stream);
  {
    const GoldenV2TensorInfo* ks = g.find("kv.k_state");
    const GoldenV2TensorInfo* vs = g.find("kv.v_state");
    CHECK(ks != nullptr && vs != nullptr);
    for (int n = 0; n < n_kv; ++n) {
      for (int t = 0; t < position; ++t) {
        const std::size_t src_row =
            static_cast<std::size_t>(n) * (position + 1) + t;
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<std::uint8_t*>(layer.kv_cache().k_mut()) +
                layer.kv_cache().row_offset(n, t) * sizeof(__nv_bfloat16),
            ks->host_bytes + src_row * hd * sizeof(__nv_bfloat16),
            hd * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<std::uint8_t*>(layer.kv_cache().v_mut()) +
                layer.kv_cache().row_offset(n, t) * sizeof(__nv_bfloat16),
            vs->host_bytes + src_row * hd * sizeof(__nv_bfloat16),
            hd * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
      }
    }
  }

  // Current-position input (stage.input) -> one decode step.
  const GoldenV2TensorInfo* in = g.find("stage.input");
  CHECK(in != nullptr);
  const int H = g.config().hidden_size;
  DeviceBuffer x_in(static_cast<std::size_t>(H) * sizeof(__nv_bfloat16),
                    stream);
  x_in.copy_from_host(in->host_bytes, in->byte_size, stream);
  layer.forward(position, x_in.data<__nv_bfloat16>(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // All 21 stages vs the golden (tolerance 1e-2).
  double worst = 0.0;
  int rc = 0;
  for (const StageSpec& sp : stage_specs(g.config())) {
    const GoldenV2TensorInfo* gt = g.find(sp.name);
    CHECK(gt != nullptr);
    if (gt == nullptr) {
      std::fprintf(stderr, "  %s missing from golden\n", sp.name);
      rc |= 1;
      continue;
    }
    double max_abs = 0.0;
    rc |= compare_stage(sp.name, sp.get(layer), gt->host_bytes, sp.elems,
                        &max_abs);
    if (max_abs > worst) worst = max_abs;
    std::fprintf(stderr, "  %-26s max_abs=%.9g OK\n", sp.name, max_abs);
  }
  std::fprintf(stderr, "  worst stage max_abs_err = %.9g\n", worst);
  if (rc) return rc;

  // p=0 invariants (BIT-EXACT).
  if (position == 0) {
    // rope's inputs are the per-head-normalized q/k, so the identity
    // compares rope_q/rope_k against q_norm/k_norm (cos(0)=1, sin(0)=0
    // are exact in bf16).
    const GoldenV2TensorInfo* qs = g.find("stage.q_norm");
    const GoldenV2TensorInfo* ks = g.find("stage.k_norm");
    const GoldenV2TensorInfo* vs = g.find("stage.v");
    const __nv_bfloat16* rq = layer.stage_rope_q();
    const __nv_bfloat16* rk = layer.stage_rope_k();
    const __nv_bfloat16* ar = layer.stage_attention_raw();
    const std::size_t qo =
        static_cast<std::size_t>(g.config().n_heads) * hd;
    const std::size_t kvo =
        static_cast<std::size_t>(n_kv) * hd;
    // rope identity (cos(0)=1, sin(0)=0 are exact in bf16)
    rc |= cmp_stage_bits("rope_q==q", rq, qs->host_bytes, qo);
    rc |= cmp_stage_bits("rope_k==k", rk, ks->host_bytes, kvo);
    // attention_raw[h] == v row of kv head h/gqa_group (probs=[1] exact)
    const int n_heads = g.config().n_heads;
    const int group = n_heads / n_kv;
    for (int h = 0; h < n_heads; ++h) {
      const int kh = h / group;
      DeviceBuffer tmp(static_cast<std::size_t>(hd) * sizeof(__nv_bfloat16));
      CUDA_CHECK(cudaMemcpyAsync(
          tmp.data(), ar + h * hd, tmp.bytes(), cudaMemcpyDeviceToDevice, 0));
      CUDA_CHECK(cudaStreamSynchronize(0));
      std::vector<__nv_bfloat16> row(hd);
      tmp.copy_to_host(row.data(), row.size() * sizeof(__nv_bfloat16), 0);
      const std::uint8_t* refrow =
          vs->host_bytes + static_cast<std::size_t>(kh) * hd * 2;
      if (std::memcmp(row.data(), refrow, hd * 2) != 0) {
        std::fprintf(stderr,
                     "  p=0 invariant: attention_raw head %d != v row %d\n",
                     h, kh);
        rc |= 1;
      }
    }
    if (!rc) std::fprintf(stderr, "  p=0 invariants: bit-exact OK\n");
  }

  // KV state: history rows BIT-EXACT (seed round-trip); the position row
  // within tolerance vs the golden AND bit-exact vs the runtime's own
  // stage rows (the KV write is a plain copy). Rows are read straight from
  // the full-size runtime cache (row_offset indexes the [n_kv, max_seq,
  // head_dim] layout).
  {
    const GoldenV2TensorInfo* ks = g.find("kv.k_state");
    const GoldenV2TensorInfo* vs = g.find("kv.v_state");
    auto read_row = [&](std::vector<__nv_bfloat16>* out,
                        const __nv_bfloat16* cache, int n, int t) {
      CUDA_CHECK(cudaMemcpyAsync(
          out->data(),
          reinterpret_cast<const std::uint8_t*>(cache) +
              layer.kv_cache().row_offset(n, t) * sizeof(__nv_bfloat16),
          hd * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, 0));
      CUDA_CHECK(cudaStreamSynchronize(0));
    };
    std::vector<__nv_bfloat16> kh_row(hd), vh_row(hd);
    for (int n = 0; n < n_kv; ++n) {
      for (int t = 0; t < position; ++t) {  // history rows: bit-exact
        const std::size_t src_row =
            static_cast<std::size_t>(n) * (position + 1) + t;
        read_row(&kh_row, layer.kv_cache().k(), n, t);
        read_row(&vh_row, layer.kv_cache().v(), n, t);
        if (std::memcmp(kh_row.data(),
                        ks->host_bytes + src_row * hd * 2, hd * 2) != 0) {
          std::fprintf(stderr, "  kv.k_state history row (%d,%d) mismatch\n",
                       n, t);
          rc |= 1;
        }
        if (std::memcmp(vh_row.data(),
                        vs->host_bytes + src_row * hd * 2, hd * 2) != 0) {
          std::fprintf(stderr, "  kv.v_state history row (%d,%d) mismatch\n",
                       n, t);
          rc |= 1;
        }
      }
      // Current-position row: bit-exact vs the runtime stage rows.
      const std::size_t dst_row =
          static_cast<std::size_t>(n) * (position + 1) + position;
      read_row(&kh_row, layer.kv_cache().k(), n, position);
      std::vector<__nv_bfloat16> rk_stage(hd);
      {
        DeviceBuffer tmp(static_cast<std::size_t>(hd) * sizeof(__nv_bfloat16));
        CUDA_CHECK(cudaMemcpyAsync(
            tmp.data(), layer.stage_rope_k() + n * hd, tmp.bytes(),
            cudaMemcpyDeviceToDevice, 0));
        CUDA_CHECK(cudaStreamSynchronize(0));
        tmp.copy_to_host(rk_stage.data(), rk_stage.size() * 2, 0);
      }
      if (std::memcmp(kh_row.data(), rk_stage.data(), hd * 2) != 0) {
        std::fprintf(stderr, "  kv.k_state pos row %d != stage.rope_k\n", n);
        rc |= 1;
      }
      // Current-position row within tolerance vs the golden state row.
      {
        const StageCompareResult r = compare_bf16_stages(
            reinterpret_cast<const std::uint16_t*>(
                ks->host_bytes + dst_row * hd * 2),
            reinterpret_cast<const std::uint16_t*>(kh_row.data()),
            static_cast<std::size_t>(hd));
        if (!r.ok) {
          std::fprintf(stderr, "  kv.k_state pos row %d vs golden: "
                               "max_abs_err=%.9g\n", n, r.max_abs_err);
          rc |= 1;
        }
      }
      read_row(&vh_row, layer.kv_cache().v(), n, position);
      std::vector<__nv_bfloat16> v_stage(hd);
      {
        DeviceBuffer tmp(static_cast<std::size_t>(hd) * sizeof(__nv_bfloat16));
        CUDA_CHECK(cudaMemcpyAsync(
            tmp.data(), layer.stage_v() + n * hd, tmp.bytes(),
            cudaMemcpyDeviceToDevice, 0));
        CUDA_CHECK(cudaStreamSynchronize(0));
        tmp.copy_to_host(v_stage.data(), v_stage.size() * 2, 0);
      }
      if (std::memcmp(vh_row.data(), v_stage.data(), hd * 2) != 0) {
        std::fprintf(stderr, "  kv.v_state pos row %d != stage.v\n", n);
        rc |= 1;
      }
    }
    if (!rc)
      std::fprintf(stderr, "  kv state: history bit-exact + pos-row "
                           "copy invariants OK\n");
  }
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr,
                 "usage: %s <out_l3.cudalm> <checkpoint_dir> <golden_p0> "
                 "<golden_p5> <python> <src_dir>\n",
                 argv[0]);
    return 2;
  }
  const std::string out_l3 = argv[1];
  const std::string ckpt = argv[2];
  const std::string golden_p0 = argv[3];
  const std::string golden_p5 = argv[4];
  const std::string py = argv[5];
  const std::string src = argv[6];

  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr,
                 "[SKIP] qwen35 full-attention golden: checkpoint not "
                 "present at %s\n",
                 ckpt.c_str());
    return 77;
  }

  // 1. Convert layer 3 (Python toolchain, offline).
  int rc = run_cmd(py + " " + src + "/tools/convert_qwen35.py" +
                   " --checkpoint-dir " + ckpt + " --out " + out_l3 +
                   " --layers 3 --manifest " + out_l3 + ".manifest.json");
  CHECK_EQ(rc, 0);

  // 2. Generate the goldens (pinned-transformers oracle; p=5 carries a
  //    non-zero KV history).
  rc = run_cmd(py + " " + src + "/tools/generate_qwen35_golden.py" +
               " --cudalm " + out_l3 + " --checkpoint-dir " + ckpt +
               " --layer 3 --out " + golden_p0 + " --position 0" +
               " --input-seed 20260209" +
               " --fidelity-report " + golden_p0 + ".fidelity.json");
  CHECK_EQ(rc, 0);
  rc = run_cmd(py + " " + src + "/tools/generate_qwen35_golden.py" +
               " --cudalm " + out_l3 + " --checkpoint-dir " + ckpt +
               " --layer 3 --out " + golden_p5 + " --position 5" +
               " --input-seed 20260209");
  CHECK_EQ(rc, 0);

  // 3. C++ parse + pinned checks.
  WeightFileV2 file;
  Status s = WeightFileV2::load(out_l3, &file);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "load error: %s\n", s.message.c_str());
    return 1;
  }
  CHECK(file.config() == Qwen35Config::qwen35_08b());
  s = file.validate_layer(3);
  CHECK(s.ok);

  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  CHECK(n >= 1);
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Qwen35LayerWeights weights;
  s = Qwen35LayerWeights::load(file, 3, stream, &weights);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "layer weight load error: %s\n", s.message.c_str());
    return 1;
  }
  CHECK(weights.is_full_attention());

  rc |= run_golden(golden_p0, weights, stream);
  rc |= run_golden(golden_p5, weights, stream);

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc == 0) TEST_PASS("test_qwen35_full_attention_golden");
  else std::fprintf(stderr, "test_qwen35_full_attention_golden FAILED\n");
  return rc;
}
