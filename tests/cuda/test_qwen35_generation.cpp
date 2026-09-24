// CUDALM — v0.4 Phase A generation golden hard gate (real checkpoint).
//
// Verifies the Qwen35Generator (single-request SERIAL prefill + greedy decode)
// against the pinned-quantized oracle's generation-level golden (a CUDLMW02
// container, loadable by the C++ WeightFileV2 loader):
//   * scenario A (short prompt [15,16,17], 8 greedy tokens): EVERY generated
//     token id + EVERY generation-step FULL logits [vocab] + the stop reason
//     + the forward count (== the oracle's t_used);
//   * scenario B (longer 16-token prompt, 16-step decode): the generated
//     sequence + each decode-step logits + the FINAL persistent state (18x
//     DeltaNet conv/recurrent + 6x FullAttention K/V used rows) — verifying
//     the DeltaNet state long-chain, the FullAttention KV history, + position
//     continuity across a long serial run;
//   * scenario C (repeat-generate contamination gate): the SAME prompt
//     generate() twice -> identical results (the second call does NOT inherit
//     the first's state — each generate() starts from a fresh reset).
//
// The greedy argmax (the runtime) + the oracle's torch.argmax are the same
// contract (maximize the numeric bf16 logit value; tie -> lowest token id), so
// the generated sequences must match token-for-token.
//
// No tokenizer / sampling / batched prefill — token-ID in, token-ID out,
// greedy only, serial prefill (correctness-first baseline).

#include "../../tests/common/check.h"

#include "cudalm/cuda_check.h"
#include "cudalm/greedy.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_deltanet.h"
#include "cudalm/qwen35_full_attention.h"
#include "cudalm/qwen35_generator.h"
#include "cudalm/qwen35_kv_cache.h"
#include "cudalm/qwen35_model.h"
#include "cudalm/stage_compare.h"
#include "cudalm/weight_loader_v2.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace cudalm;

namespace {

// Deterministic prompts / budgets (MUST match the golden generator args).
// Both prompts are CONFIDENT: the oracle's top-1 vs top-2 logit gap stays well
// above the runtime/oracle bf16-logits rounding (~0.13) at EVERY generation
// step (scenario A min gap 1.0625; scenario B min gap 3.625). A near-tie
// (gap < ~0.25) would let the ~0.13 rounding flip the greedy argmax and make
// the two runtimes pick different tokens — these prompts avoid that, so the
// oracle's generated sequence is a stable golden.
static const int kPromptA[] = {1024, 2048, 3072};
static const int kNA = 3;
static const int kPromptB[] = {1024, 2048, 3072, 1024, 2048, 3072, 1024,
                               2048, 3072, 1024, 2048, 3072, 1024, 2048, 3072,
                               1024};
static const int kNB = 16;
static const int kEos = 248319;
static const int kMaxA = 8;
static const int kMaxB = 16;

// Per-step FULL logits [vocab] + final-state tolerances (the runtime bf16 chain
// vs the oracle fp32-acc matmul; see docs §19). Measured worsts (pinned oracle,
// the confident prompts below, RTX 2080 Ti): per-step logits 0.5625 (scenario A
// t4), DeltaNet conv 0.125 / recurrent 0.033 / FullAttention KV 0.19 (scenario
// B final state). These envelopes carry a small margin over the measured error
// (not a guess) and are re-logged on every run (the compare prints max_abs).
static const double kLogitsAtol = 0.65;
static const double kStateAtol = 0.5;

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

// Parse a comma-separated int list ("15,16,17" -> {15,16,17}).
std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      if (!cur.empty()) { out.push_back(std::atoi(cur.c_str())); cur.clear(); }
    } else {
      cur += c;
    }
  }
  if (!cur.empty()) out.push_back(std::atoi(cur.c_str()));
  return out;
}

// Compare a HOST bf16 buffer against golden host bytes (the per-step logits
// are already on the host — the generator D2H'd them for the argmax).
int compare_bf16_host(const std::string& name, const __nv_bfloat16* act,
                      const std::uint8_t* ref, std::size_t elems, double atol,
                      double rtol) {
  const StageCompareResult r = compare_bf16_stages(
      reinterpret_cast<const std::uint16_t*>(ref),
      reinterpret_cast<const std::uint16_t*>(act), elems, atol, rtol);
  if (!r.ok) {
    std::fprintf(stderr, "  %-42s FAIL max_abs=%.9g max_rel=%.9g (idx=%zu, "
                         "n=%zu, atol=%.4g)\n",
                 name.c_str(), r.max_abs_err, r.max_rel_err, r.max_abs_idx,
                 elems, atol);
    return 1;
  }
  std::fprintf(stderr, "  %-42s max_abs=%.9g OK\n", name.c_str(), r.max_abs_err);
  return 0;
}

template <typename T>
std::vector<T> dev_to_host(const T* dev, std::size_t n, cudaStream_t stream) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * sizeof(T),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h;
}

int compare_bf16_dev(const std::string& name, const __nv_bfloat16* dev,
                     const std::uint8_t* ref, std::size_t elems, double atol,
                     cudaStream_t stream) {
  std::vector<__nv_bfloat16> act = dev_to_host(dev, elems, stream);
  return compare_bf16_host(name, act.data(), ref, elems, atol, 0.0);
}

int compare_fp32_dev(const std::string& name, const float* dev,
                     const std::uint8_t* ref, std::size_t elems, double atol,
                     cudaStream_t stream) {
  std::vector<float> act = dev_to_host(dev, elems, stream);
  const float* refp = reinterpret_cast<const float*>(ref);
  double max_abs = 0.0;
  bool ok = true;
  for (std::size_t i = 0; i < elems; ++i) {
    const double ad = std::fabs(act[i] - refp[i]);
    if (ad > max_abs) max_abs = ad;
    if (!(ad <= atol)) ok = false;
  }
  if (!ok) {
    std::fprintf(stderr, "  %-42s FAIL max_abs=%.9g (n=%zu, atol=%.4g)\n",
                 name.c_str(), max_abs, elems, atol);
    return 1;
  }
  std::fprintf(stderr, "  %-42s max_abs=%.9g OK\n", name.c_str(), max_abs);
  return 0;
}

// Load a generation golden (CUDLMW02) + run the generator + compare the
// generated tokens + stop reason + forward count + every step's FULL logits
// (via the observer). `check_state` adds the final persistent-state compare
// (scenario B).
int run_generation_scenario(Qwen35Model& model, cudaStream_t stream,
                            const std::string& gprefix, const char* sc,
                            const std::vector<int>& prompt, int max_new_tokens,
                            int eos, bool check_state) {
  std::fprintf(stderr, "[scenario %s] prompt=%zu tok, max_new=%d\n", sc,
               prompt.size(), max_new_tokens);
  const std::string path = gprefix + "_gen_" + sc + ".cudalm";
  WeightFileV2 gg;
  Status s = WeightFileV2::load(path, &gg);
  if (!s.ok) {
    std::fprintf(stderr, "  load error %s: %s\n", path.c_str(),
                 s.message.c_str());
    return 1;
  }
  const Qwen35Config& cfg = model.config();
  const int V = cfg.vocab_size;

  const std::string* gstop = gg.meta("gen.stop_reason");
  const std::string* gtok = gg.meta("gen.tokens");
  const std::string* gtused = gg.meta("gen.t_used");
  if (!gstop || !gtok || !gtused) {
    std::fprintf(stderr, "  missing gen metadata in %s\n", path.c_str());
    return 1;
  }
  const std::vector<int> expected_tokens = parse_ints(*gtok);
  const int expected_stop = std::atoi(gstop->c_str());
  const int expected_tused = std::atoi(gtused->c_str());
  const int n_steps = static_cast<int>(expected_tokens.size());

  int rc = 0;
  // The observer compares every step's FULL host logits against the golden.
  std::vector<bool> seen(n_steps, false);
  LogitsObserver obs = [&rc, &seen, &gg, V, n_steps](
                           const __nv_bfloat16* host_logits, int step) {
    if (step < 0 || step >= n_steps) {
      std::fprintf(stderr, "  logits step out of range: %d (n=%d)\n", step,
                   n_steps);
      rc |= 1;
      return;
    }
    seen[step] = true;
    const std::string tn = "gen.logits.t" + std::to_string(step);
    const Qwen35TensorInfo* glog = gg.find(tn);
    if (!glog) {
      std::fprintf(stderr, "  missing golden tensor %s\n", tn.c_str());
      rc |= 1;
      return;
    }
    rc |= compare_bf16_host(tn, host_logits, glog->host_bytes,
                            static_cast<std::size_t>(V), kLogitsAtol, 0.0);
  };

  Qwen35Generator gen(model);
  GenerationResult res =
      gen.generate(prompt, max_new_tokens, eos, stream, &obs);
  if (!res.ok) {
    std::fprintf(stderr, "  generate failed: %s\n", res.error.c_str());
    return 1;
  }

  // EVERY generated token id must match the oracle (token-for-token).
  std::string tname = std::string(sc) + ".generated_tokens";
  if (static_cast<int>(res.generated.size()) != n_steps) {
    std::fprintf(stderr, "  %-42s FAIL got %zu tokens, expected %d\n",
                 tname.c_str(), res.generated.size(), n_steps);
    rc |= 1;
  } else {
    bool tok_ok = true;
    for (int i = 0; i < n_steps; ++i) {
      if (res.generated[i] != expected_tokens[i]) {
        std::fprintf(stderr, "  token[%d]: got %d, expected %d\n", i,
                     res.generated[i], expected_tokens[i]);
        tok_ok = false;
      }
    }
    if (tok_ok)
      std::fprintf(stderr, "  %-42s %d tokens all match OK\n", tname.c_str(),
                   n_steps);
    else
      rc |= 1;
  }
  // The stop reason must match the oracle.
  std::string sname = std::string(sc) + ".stop_reason";
  if (stop_reason_code(res.stop_reason) != expected_stop) {
    std::fprintf(stderr, "  %-42s FAIL got %d (%s), expected %d\n", sname.c_str(),
                 stop_reason_code(res.stop_reason),
                 stop_reason_name(res.stop_reason), expected_stop);
    rc |= 1;
  } else {
    std::fprintf(stderr, "  %-42s %s OK\n", sname.c_str(),
                 stop_reason_name(res.stop_reason));
  }
  // The forward count must equal the oracle's t_used (prefill + decode
  // forwards; the EOS / max_new_tokens last token is never forwarded).
  std::string fname = std::string(sc) + ".forward_count";
  if (res.forward_count != expected_tused) {
    std::fprintf(stderr, "  %-42s FAIL got %d, expected %d (t_used)\n",
                 fname.c_str(), res.forward_count, expected_tused);
    rc |= 1;
  } else {
    std::fprintf(stderr, "  %-42s %d == t_used OK\n", fname.c_str(),
                 res.forward_count);
  }
  // Every step's logits must have been observed + compared.
  for (int i = 0; i < n_steps; ++i) {
    if (!seen[i]) {
      std::fprintf(stderr, "  gen.logits.t%d was not observed\n", i);
      rc |= 1;
    }
  }

  // ---- Final persistent state (scenario B) --------------------------------
  if (check_state) {
    const int n_kv = cfg.n_kv_heads;
    const int hd = cfg.head_dim;
    const std::size_t conv_elems =
        static_cast<std::size_t>(cfg.linear_conv_dim()) *
        cfg.linear_conv_state_len();
    const std::size_t rec_elems =
        static_cast<std::size_t>(cfg.lin_num_v_heads) *
        cfg.lin_key_head_dim * cfg.lin_value_head_dim;
    for (int L = 0; L < cfg.num_hidden_layers; ++L) {
      if (model.is_linear_attention(L)) {
        const Qwen35TensorInfo* gc =
            gg.find("gen.state.conv.L" + std::to_string(L));
        const Qwen35TensorInfo* gr =
            gg.find("gen.state.rec.L" + std::to_string(L));
        if (!gc || !gr) {
          std::fprintf(stderr, "  missing DeltaNet state for layer %d\n", L);
          rc |= 1;
          continue;
        }
        const Qwen35DeltaNetLayer* d = model.delta(L);
        std::string cn = "DeltaNet.conv_after[L" + std::to_string(L) + "]";
        std::string rn = "DeltaNet.recurrent_after[L" + std::to_string(L) + "]";
        rc |= compare_bf16_dev(cn, d->conv_state(), gc->host_bytes, conv_elems,
                               kStateAtol, stream);
        rc |= compare_fp32_dev(rn, d->recurrent_state(), gr->host_bytes,
                               rec_elems, kStateAtol, stream);
      } else {
        const Qwen35TensorInfo* gk =
            gg.find("gen.state.kv.k.L" + std::to_string(L));
        const Qwen35TensorInfo* gv =
            gg.find("gen.state.kv.v.L" + std::to_string(L));
        if (!gk || !gv) {
          std::fprintf(stderr, "  missing FA KV state for layer %d\n", L);
          rc |= 1;
          continue;
        }
        const Qwen35KvCache& kv = model.attention(L)->kv_cache();
        double mk = 0.0, mv = 0.0;
        for (int n = 0; n < n_kv; ++n) {
          // D2H head n's t_used rows (contiguous: row (n,t) is at
          // row_offset(n,0) + t*hd). The golden holds the same rows
          // row-major (n, t) at src_row = n*t_used + t.
          const std::size_t n_elems =
              static_cast<std::size_t>(expected_tused) * hd;
          std::vector<__nv_bfloat16> k_rows(n_elems);
          std::vector<__nv_bfloat16> v_rows(n_elems);
          CUDA_CHECK(cudaMemcpyAsync(
              k_rows.data(), kv.k() + kv.row_offset(n, 0),
              n_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaMemcpyAsync(
              v_rows.data(), kv.v() + kv.row_offset(n, 0),
              n_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          for (int t = 0; t < expected_tused; ++t) {
            const std::size_t src_row =
                static_cast<std::size_t>(n) * expected_tused + t;
            const __nv_bfloat16* kact = k_rows.data() +
                static_cast<std::size_t>(t) * hd;
            const __nv_bfloat16* vact = v_rows.data() +
                static_cast<std::size_t>(t) * hd;
            StageCompareResult rk = compare_bf16_stages(
                reinterpret_cast<const std::uint16_t*>(
                    gk->host_bytes + src_row * hd * sizeof(__nv_bfloat16)),
                reinterpret_cast<const std::uint16_t*>(kact),
                static_cast<std::size_t>(hd), kStateAtol, 0.0);
            StageCompareResult rv = compare_bf16_stages(
                reinterpret_cast<const std::uint16_t*>(
                    gv->host_bytes + src_row * hd * sizeof(__nv_bfloat16)),
                reinterpret_cast<const std::uint16_t*>(vact),
                static_cast<std::size_t>(hd), kStateAtol, 0.0);
            mk = std::fmax(mk, rk.max_abs_err);
            mv = std::fmax(mv, rv.max_abs_err);
            if (!rk.ok || !rv.ok) rc |= 1;
          }
        }
        std::string kn = "FA.KV_used[L" + std::to_string(L) + "]";
        std::fprintf(stderr, "  %-42s K max_abs=%.9g V max_abs=%.9g %s\n",
                     kn.c_str(), mk, mv, (rc ? "FAIL" : "OK"));
      }
    }
  }
  return rc;
}

// Scenario C: the SAME prompt generate() twice must be identical (no state
// leakage between calls — each generate() starts from a fresh reset).
int test_contamination(Qwen35Model& model, cudaStream_t stream,
                       const std::vector<int>& prompt, int max_new_tokens,
                       int eos) {
  std::fprintf(stderr, "[scenario C] repeat-generate contamination gate\n");
  Qwen35Generator gen(model);
  GenerationResult r1 =
      gen.generate(prompt, max_new_tokens, eos, stream, nullptr);
  GenerationResult r2 =
      gen.generate(prompt, max_new_tokens, eos, stream, nullptr);
  if (!r1.ok || !r2.ok) {
    std::fprintf(stderr, "  generate failed: %s / %s\n", r1.error.c_str(),
                 r2.error.c_str());
    return 1;
  }
  int rc = 0;
  if (r1.generated != r2.generated || r1.stop_reason != r2.stop_reason ||
      r1.forward_count != r2.forward_count) {
    std::fprintf(stderr, "  FAIL: the two generate() calls differ (the second "
                         "inherited the first's state)\n");
    rc |= 1;
  }
  std::fprintf(stderr, "  %-42s %d tokens, stop=%s — %s\n",
               "C.repeat_identical", static_cast<int>(r1.generated.size()),
               stop_reason_name(r1.stop_reason),
               rc ? "DIFFER" : "identical OK");
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: %s <out.cudalm> <checkpoint_dir> <python> <src_dir> "
                 "<gen_golden_prefix> [--no-gen]\n",
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
                 "[SKIP] qwen35 generation: checkpoint absent at %s\n",
                 ckpt.c_str());
    return 77;
  }
  if (!no_gen) {
    int rc = run_cmd(py + " " + src + "/tools/generate_qwen35_golden.py" +
                     " --cudalm " + out + " --checkpoint-dir " + ckpt +
                     " --gen-prefix " + gprefix +
                     " --gen-a-tokens 1024,2048,3072 --gen-a-max 8 --gen-a-eos " +
                     std::to_string(kEos) +
                     " --gen-b-tokens 1024,2048,3072,1024,2048,3072,1024,2048,3072,1024,2048,3072,1024,2048,3072,1024" +
                     " --gen-b-max 16 --gen-b-eos " + std::to_string(kEos) +
                     " --gen-b-state 1");
    CHECK_EQ(rc, 0);
  } else {
    CHECK(file_exists(gprefix + "_gen_A.cudalm"));
    CHECK(file_exists(gprefix + "_gen_B.cudalm"));
  }

  // Load the full model.
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

  const std::vector<int> promptA(kPromptA, kPromptA + kNA);
  const std::vector<int> promptB(kPromptB, kPromptB + kNB);

  int rc = 0;
  rc |= run_generation_scenario(model, stream, gprefix, "A", promptA, kMaxA,
                                kEos, /*check_state=*/false);
  rc |= run_generation_scenario(model, stream, gprefix, "B", promptB, kMaxB,
                                kEos, /*check_state=*/true);
  rc |= test_contamination(model, stream, promptA, kMaxA, kEos);

  if (rc != 0) {
    std::fprintf(stderr, "qwen35 generation: FAILURES (rc=%d)\n", rc);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 1;
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  TEST_PASS("qwen35_generation (A short + B longer + C contamination; "
            "per-decode-step FULL logits + final hybrid state)");
  return 0;
}
