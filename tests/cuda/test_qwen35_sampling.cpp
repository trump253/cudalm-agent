// CUDALM — v0.4 Phase C sampling integration gate (real checkpoint).
//
// Drives the new Qwen35Generator::generate(..., SamplingConfig) overload and
// the Qwen35TextGenerator sampling stitch on the full 0.8B model:
//   * greedy EXACT: the old greedy API vs the new API with a greedy config
//     (and with a temperature-0 config carrying k/p fields) -> generated ids,
//     stop reason and forward count all bit-for-bit identical (the frozen
//     Phase A path is untouched);
//   * seed determinism: same prompt + sampling config + seed, twice -> the
//     EXACT same token sequence / stop / forward count;
//   * request-level contamination gate (A(42) -> B(123) -> A(42)): the two
//     A runs are EXACT (ids + text + stop + forward count) — the model state
//     reset AND the per-request sampler RNG reset both leave no cross-request
//     pollution (the Phase A/B discipline applied to sampling);
//   * sampling sanity: every generated id in [0, vocab); the forward-count
//     invariant (prefill + generated - 1) holds for sampling too.
//
// Self-skips (77) when the full model / tokenizer artifacts are absent.

#include "../../tests/common/check.h"

#include "cudalm/cuda_check.h"
#include "cudalm/greedy.h"
#include "cudalm/qwen35_generator.h"
#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_text_generator.h"
#include "cudalm/qwen35_tokenizer.h"
#include "cudalm/weight_loader_v2.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace cudalm;

namespace {

bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return static_cast<bool>(f);
}

// The confident Phase A prompt (top1-top2 gap > 1.0 at every greedy step).
static const int kPrompt[] = {1024, 2048, 3072};
static const int kN = 3;
// A padding-range id that the model effectively never emits -> the runs
// below stop on max_new_tokens (deterministic budget, no EOS flakiness).
static const int kEos = 248319;
static const int kMaxNew = 8;

struct RunStats {
  std::vector<int> ids;
  StopReason stop = StopReason::MaxNewTokens;
  int forwards = 0;
};

bool same_run(const RunStats& a, const RunStats& b, const char* what) {
  bool ok = a.ids == b.ids && a.stop == b.stop && a.forwards == b.forwards;
  if (!ok) {
    std::fprintf(stderr, "sampling run mismatch: %s (ids %zu vs %zu, stop %d "
                         "vs %d, fwd %d vs %d)\n",
                 what, a.ids.size(), b.ids.size(),
                 static_cast<int>(a.stop), static_cast<int>(b.stop),
                 a.forwards, b.forwards);
  }
  return ok;
}

// The forward-count invariant: prefill + (generated - 1); the last token
// (EOS / budget-final) is appended but never forwarded.
int check_forward_invariant(const RunStats& r, const char* what) {
  const int expect = kN + static_cast<int>(r.ids.size()) - 1;
  if (r.forwards != expect) {
    std::fprintf(stderr, "forward-count invariant broken: %s (fwd %d, expect "
                         "%d)\n",
                 what, r.forwards, expect);
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: test_qwen35_sampling <full_model.cudalm> "
                 "<tokenizer.cudaltk>\n");
    return 2;
  }
  const std::string model_path = argv[1];
  const std::string tokenizer_path = argv[2];
  if (!file_exists(model_path) || !file_exists(tokenizer_path)) {
    std::fprintf(stderr,
                 "[SKIP] qwen35 sampling: model/tokenizer artifact absent\n");
    return 77;
  }

  std::printf("test_qwen35_sampling: v0.4 Phase C sampling integration\n");

  WeightFileV2 file;
  Status s = WeightFileV2::load(model_path, &file);
  CHECK(s.ok);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  Qwen35Model model;
  s = Qwen35Model::load(file, stream, &model);
  CHECK(s.ok);
  CHECK(model.loaded());
  const int vocab = model.config().vocab_size;

  Qwen35Generator gen(model);
  const std::vector<int> prompt(kPrompt, kPrompt + kN);
  int rc = 0;

  // ---- 1) greedy EXACT: old API == new API (greedy config) ---------------
  {
    const GenerationResult old_greedy =
        gen.generate(prompt, kMaxNew, kEos, stream);
    CHECK(old_greedy.ok);
    const GenerationResult new_greedy =
        gen.generate(prompt, kMaxNew, kEos, stream,
                     SamplingConfig::greedy());
    CHECK(new_greedy.ok);
    CHECK_EQ(old_greedy.generated, new_greedy.generated);
    CHECK_EQ(old_greedy.stop_reason, new_greedy.stop_reason);
    CHECK_EQ(old_greedy.forward_count, new_greedy.forward_count);
    // A temperature-0 config CARRYING k/p fields is still the frozen path.
    SamplingConfig zero_with_fields = SamplingConfig::greedy();
    zero_with_fields.top_k = 40;
    zero_with_fields.top_p = 0.95f;
    zero_with_fields.seed = 999;
    CHECK(zero_with_fields.is_greedy());
    const GenerationResult zero =
        gen.generate(prompt, kMaxNew, kEos, stream, zero_with_fields);
    CHECK(zero.ok);
    CHECK_EQ(old_greedy.generated, zero.generated);
    CHECK_EQ(old_greedy.stop_reason, zero.stop_reason);
    CHECK_EQ(old_greedy.forward_count, zero.forward_count);
    std::printf("  [ok] greedy EXACT (old API == greedy-config new API)\n");
  }

  // ---- 2) seed determinism (same prompt + config + seed, twice) -----------
  SamplingConfig sampling;
  sampling.temperature = 0.8f;
  sampling.top_k = 40;
  sampling.top_p = 0.95f;
  sampling.seed = 42;
  {
    const GenerationResult r1 = gen.generate(prompt, kMaxNew, kEos, stream,
                                             sampling);
    CHECK(r1.ok);
    const GenerationResult r2 = gen.generate(prompt, kMaxNew, kEos, stream,
                                             sampling);
    CHECK(r2.ok);
    CHECK(r1.generated == r2.generated);
    CHECK(r1.stop_reason == r2.stop_reason);
    CHECK(r1.forward_count == r2.forward_count);
    for (int t : r1.generated) CHECK(t >= 0 && t < vocab);
    rc |= check_forward_invariant(
        RunStats{r1.generated, r1.stop_reason, r1.forward_count},
        "determinism");
    std::printf("  [ok] seed determinism (A(42) == A(42), %zu tokens)\n",
                r1.generated.size());
  }

  // ---- 3) request-level contamination gate: A(42) -> B(123) -> A(42) ------
  {
    auto run = [&](std::uint64_t seed, RunStats* out) -> int {
      SamplingConfig c = sampling;
      c.seed = seed;
      const GenerationResult r = gen.generate(prompt, kMaxNew, kEos, stream,
                                              c);
      if (!r.ok) {
        std::fprintf(stderr, "contamination run failed (seed %llu): %s\n",
                     (unsigned long long)seed, r.error.c_str());
        return 1;
      }
      out->ids = r.generated;
      out->stop = r.stop_reason;
      out->forwards = r.forward_count;
      return 0;
    };
    RunStats a1, b, a2;
    CHECK(run(42, &a1) == 0);
    CHECK(run(123, &b) == 0);
    CHECK(run(42, &a2) == 0);
    CHECK(same_run(a1, a2, "contamination A(42) vs A(42)"));
    // B is allowed to differ (and usually does); the gate is that A is
    // untouched by B.  Also: B's RNG consumption must not shift A(42).
    rc |= check_forward_invariant(a1, "contamination");
    rc |= check_forward_invariant(b, "contamination-B");
    std::printf("  [ok] contamination gate A(42) -> B(123) -> A(42) "
                "(A EXACT; B %zu tokens)\n",
                b.ids.size());
  }

  // ---- 4) text-level stitch: the same gate through Qwen35TextGenerator ----
  std::unique_ptr<Qwen35Tokenizer> tokenizer;
  s = Qwen35Tokenizer::load(tokenizer_path, &tokenizer);
  CHECK(s.ok);
  Qwen35TextGenerator textgen(model, *tokenizer);
  {
    auto run_text = [&](std::uint64_t seed,
                        TextGenerationResult* out) -> int {
      SamplingConfig c = sampling;
      c.seed = seed;
      const TextGenerationResult r =
          textgen.generate_text("The capital of France is", kMaxNew, c,
                                stream);
      if (!r.ok) {
        std::fprintf(stderr, "text run failed (seed %llu): %s\n",
                     (unsigned long long)seed, r.error.c_str());
        return 1;
      }
      *out = r;
      return 0;
    };
    TextGenerationResult t_a1, t_b, t_a2;
    CHECK(run_text(42, &t_a1) == 0);
    CHECK(run_text(123, &t_b) == 0);
    CHECK(run_text(42, &t_a2) == 0);
    CHECK(t_a1.generated_text == t_a2.generated_text);
    CHECK(t_a1.generated_token_ids == t_a2.generated_token_ids);
    CHECK(t_a1.stop_reason == t_a2.stop_reason);
    CHECK(t_a1.forward_count == t_a2.forward_count);
    CHECK(!t_a1.generated_text.empty());
    std::printf("  [ok] text-level A(42) -> B(123) -> A(42): text EXACT "
                "(\"%.24s...\")\n",
                t_a1.generated_text.substr(0, 24).c_str());
    (void)t_b;
  }

  // ---- 5) invalid sampling config fails loud, nothing forwarded -----------
  {
    SamplingConfig bad = sampling;
    bad.top_p = 1.5f;
    const GenerationResult r = gen.generate(prompt, kMaxNew, kEos, stream,
                                            bad);
    CHECK(!r.ok);
    CHECK(!r.error.empty());
    CHECK(r.generated.empty());
    CHECK_EQ(r.forward_count, 0);
    std::printf("  [ok] invalid sampling config (fail loud, no forward)\n");
  }

  cudaStreamDestroy(stream);
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_sampling: FAIL\n");
    return rc;
  }
  std::printf("test_qwen35_sampling: PASS\n");
  return 0;
}
