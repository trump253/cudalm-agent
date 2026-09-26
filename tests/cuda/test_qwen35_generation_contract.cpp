// CUDALM — v0.4 Phase A generator input-contract hardening (real checkpoint).
//
// Verifies Qwen35Generator::generate's input / capacity contract (fail loud).
// The contract is checked on the host BEFORE any CUDA / reset, so:
//   * an INVALID input -> ok == false, generated empty, forward_count == 0
//     (no prefill, no decode, no state mutation);
//   * max_new_tokens == 0 (a VALID input) -> ok == true, generated empty,
//     stop == max_new_tokens, the prefill runs (forward_count == prompt_len)
//     but NO decode (documented).
//
// Cases covered:
//   * the !loaded guard (a default-constructed, not-loaded model fails first —
//     no checkpoint needed);
//   * empty prompt;
//   * invalid prompt token id (< 0 and >= vocab);
//   * invalid eos_token_id (< 0 and >= vocab);
//   * max_new_tokens < 0;
//   * prompt length > max_seq_len (a long valid prompt);
//   * max_new_tokens == 0 (valid; empty generation, no decode).
//
// No API expansion: this only drives the existing generate() with the contract
// inputs and inspects the returned GenerationResult.

#include "../../tests/common/check.h"

#include "cudalm/cuda_check.h"
#include "cudalm/greedy.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_generator.h"
#include "cudalm/qwen35_model.h"
#include "cudalm/weight_loader_v2.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace cudalm;

namespace {

bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return static_cast<bool>(f);
}

// An invalid input must be a clean early failure: ok == false, no tokens, no
// forward (neither prefill nor decode), an error message set.
int check_invalid(const char* name, const GenerationResult& r) {
  int rc = 0;
  if (r.ok) {
    std::fprintf(stderr, "  %-40s FAIL: expected ok == false\n", name);
    rc |= 1;
  }
  if (!r.generated.empty()) {
    std::fprintf(stderr, "  %-40s FAIL: generated not empty (%zu)\n", name,
                 r.generated.size());
    rc |= 1;
  }
  if (r.forward_count != 0) {
    std::fprintf(stderr, "  %-40s FAIL: forward_count == %d (expected 0)\n",
                 name, r.forward_count);
    rc |= 1;
  }
  if (r.error.empty()) {
    std::fprintf(stderr, "  %-40s FAIL: error message empty\n", name);
    rc |= 1;
  }
  if (rc == 0)
    std::fprintf(stderr, "  %-40s ok=false, generated empty, fc=0, err='%s' OK\n",
                 name, r.error.c_str());
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
  const std::string gprefix = argv[5];
  const bool no_gen = (argc >= 7 && std::string(argv[6]) == "--no-gen");

  if (!no_gen && !file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] qwen35 generation contract: checkpoint absent "
                         "at %s\n",
                 ckpt.c_str());
    return 77;
  }
  // The contract cases need no golden; --no-gen only skips the (unnecessary)
  // golden generation.
  (void)gprefix;

  // ---- the !loaded guard (no checkpoint needed) ---------------------------
  {
    Qwen35Model empty;
    Qwen35Generator gen(empty);
    const std::vector<int> prompt = {1024, 2048, 3072};
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const GenerationResult r =
        gen.generate(prompt, 8, 248319, stream, nullptr);
    CUDA_CHECK(cudaStreamDestroy(stream));
    int rc = check_invalid("not_loaded", r);
    if (rc != 0) return 1;
  }

  // ---- load the full model (the other contract cases need a loaded model) --
  if (!file_exists(out)) {
    std::fprintf(stderr, "[SKIP] qwen35 generation contract: full model .cudalm "
                         "absent at %s\n",
                 out.c_str());
    return 77;
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
  const int vocab = cfg.vocab_size;
  const int max_seq = cfg.max_seq_len;
  const int kEos = 248319;
  Qwen35Generator gen(model);
  int rc = 0;

  // empty prompt
  rc |= check_invalid("empty_prompt", gen.generate({}, 8, kEos, stream, nullptr));
  // invalid prompt token id (out of [0, vocab))
  {
    const std::vector<int> neg = {1024, -1};
    rc |= check_invalid("invalid_token_negative",
                        gen.generate(neg, 8, kEos, stream, nullptr));
    const std::vector<int> big = {1024, vocab};
    rc |= check_invalid("invalid_token_oob",
                        gen.generate(big, 8, kEos, stream, nullptr));
  }
  // invalid eos_token_id (out of [0, vocab))
  {
    const std::vector<int> prompt = {1024, 2048, 3072};
    rc |= check_invalid("invalid_eos_negative",
                        gen.generate(prompt, 8, -1, stream, nullptr));
    rc |= check_invalid("invalid_eos_oob",
                        gen.generate(prompt, 8, vocab, stream, nullptr));
  }
  // max_new_tokens < 0
  {
    const std::vector<int> prompt = {1024, 2048, 3072};
    rc |= check_invalid("max_new_tokens_negative",
                        gen.generate(prompt, -1, kEos, stream, nullptr));
  }
  // prompt length > max_seq_len (a long, otherwise-valid prompt)
  {
    std::vector<int> long_prompt(static_cast<std::size_t>(max_seq) + 1, 1024);
    rc |= check_invalid("prompt_len_gt_max_seq",
                        gen.generate(long_prompt, 8, kEos, stream, nullptr));
  }

  // max_new_tokens == 0 (VALID): empty generation, no decode, but the prefill
  // ran (forward_count == prompt_len). ok == true, stop == max_new_tokens.
  {
    const std::vector<int> prompt = {1024, 2048, 3072};
    const GenerationResult r =
        gen.generate(prompt, 0, kEos, stream, nullptr);
    int c = 0;
    if (!r.ok) {
      std::fprintf(stderr, "  %-40s FAIL: expected ok == true (%s)\n",
                   "max_new_tokens_zero", r.error.c_str());
      c |= 1;
    }
    if (!r.generated.empty()) {
      std::fprintf(stderr, "  %-40s FAIL: generated not empty\n",
                   "max_new_tokens_zero");
      c |= 1;
    }
    if (r.stop_reason != StopReason::MaxNewTokens) {
      std::fprintf(stderr, "  %-40s FAIL: stop != max_new_tokens (%s)\n",
                   "max_new_tokens_zero", stop_reason_name(r.stop_reason));
      c |= 1;
    }
    if (r.forward_count != static_cast<int>(prompt.size())) {
      std::fprintf(stderr, "  %-40s FAIL: forward_count == %d (expected %zu = "
                   "prefill, no decode)\n",
                   "max_new_tokens_zero", r.forward_count, prompt.size());
      c |= 1;
    }
    if (c == 0)
      std::fprintf(stderr,
                   "  %-40s ok=true, generated empty, stop=max_new_tokens, "
                   "fc=%d (prefill, no decode) OK\n",
                   "max_new_tokens_zero", r.forward_count);
    rc |= c;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "qwen35 generation contract: FAILURES (rc=%d)\n", rc);
    return 1;
  }
  TEST_PASS("qwen35_generation_contract (empty / invalid token / invalid eos / "
            "max_new_tokens<0 / prompt_len>max_seq / max_new_tokens==0)");
  return 0;
}
