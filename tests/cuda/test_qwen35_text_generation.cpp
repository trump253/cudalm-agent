// CUDALM — v0.4 Phase B text-level generation E2E hard gate (real
// checkpoint).
//
// The full raw-text contract, end to end, on the pinned 0.8B model:
//   raw UTF-8 prompt text
//     -> native encode (Qwen35Tokenizer)
//     -> greedy generation (Qwen35Generator, REAL pinned EOS 248044)
//     -> native decode (Qwen35Tokenizer)
//     -> UTF-8 text
//
// Four CONFIDENT text prompts (2 English, 2 CJK) — each selected with
// tools/diag_gen_gaps.py on the pinned-quantized oracle: the top-1 vs top-2
// LOGIT GAP stays well above the runtime/oracle bf16-logits rounding at
// EVERY generation step (worst observed min-gaps: T1 0.625, T2 2.875,
// T3 1.4375, T4 0.75 — all > 0.4), so the oracle's greedy sequence is a
// stable golden. The prompts deliberately cover: ASCII words, ASCII digits +
// punctuation, CJK ideographs, the full-width comma (U+FF0C), and mixed
// byte widths — exercising the native encode/decode on real multilingual
// text (the unit corpus already gates every stage individually; this test
// gates the STITCH: same prompt ids the HF oracle produced, same generated
// ids the quantized oracle produced, same decoded text HF produces).
//
// Per scenario (golden = generate_qwen35_golden.py --gen-text T1..T4,
// CUDLMW02 container <prefix>_gen_Tk.cudalm):
//   * r.prompt_token_ids   == gen.prompt     (native encode == pinned HF
//                                              encode, EXACT)
//   * r.generated_token_ids== gen.tokens     (greedy decode == pinned-
//                                              quantized oracle, EXACT)
//   * r.stop_reason        == gen.stop_reason
//   * r.forward_count      == gen.t_used
//   * r.generated_text     == gen.hf_decoded (native decode == pinned HF
//                                              decode, skip_special_tokens=
//                                              False, EXACT)
//   * the golden's gen.prompt_text round-trips to the test's prompt
//     constant (drift guard).
//
// Contamination gate (Phase A pattern, text-level): T1, then T3, then T1
// again — the second T1 run must be IDENTICAL (each generate() starts from a
// fresh reset; no state leaks across requests).
//
// Real EOS contract: the generator stops on the tokenizer's pinned real EOS
// (248044), NOT the Phase A sentinel padding id (248319).

#include "../../tests/common/check.h"

#include "cudalm/cuda_check.h"
#include "cudalm/greedy.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_generator.h"
#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_text_generator.h"
#include "cudalm/qwen35_tokenizer.h"
#include "cudalm/weight_loader_v2.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace cudalm;

namespace {

// The confident prompts (MUST match the --gen-text args in main()). Max new
// tokens for every scenario.
struct Scenario {
  const char* name;  // T1..T4 (golden suffix)
  const char* text;  // the raw UTF-8 prompt
};
static const Scenario kScenarios[] = {
    {"T1", "January, February, March,"},
    {"T2", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,"},
    // CJK (UTF-8 in this source file; the full-width comma is U+FF0C).
    {"T3", "\xE4\xB8\x80\xE6\x9C\x88\xEF\xBC\x8C\xE4\xBA\x8C\xE6\x9C\x88"
           "\xEF\xBC\x8C\xE4\xB8\x89\xE6\x9C\x88\xEF\xBC\x8C"},
    {"T4", "\xE4\xB8\x80\xEF\xBC\x8C\xE4\xBA\x8C\xEF\xBC\x8C\xE4\xB8\x89"
           "\xEF\xBC\x8C\xE5\x9B\x9B\xEF\xBC\x8C\xE4\xBA\x94\xEF\xBC\x8C"
           "\xE5\x85\xAD\xEF\xBC\x8C\xE4\xB8\x83\xEF\xBC\x8C\xE5\x85\xAB"
           "\xEF\xBC\x8C\xE4\xB9\x9D\xEF\xBC\x8C\xE5\x8D\x81\xEF\xBC\x8C"},
};
static const int kNumScenarios =
    static_cast<int>(sizeof(kScenarios) / sizeof(kScenarios[0]));
static const int kMaxNew = 8;

int run_cmd(const std::string& cmd) { return std::system(cmd.c_str()); }

bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return f.good();
}

std::vector<std::uint32_t> parse_ids(const std::string& s) {
  std::vector<std::uint32_t> out;
  std::size_t i = 0;
  while (i < s.size()) {
    if (s[i] == ',') {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j < s.size() && s[j] != ',') ++j;
    out.push_back(std::strtoul(s.substr(i, j - i).c_str(), nullptr, 10));
    i = j;
  }
  return out;
}

// Run ONE text scenario end to end + compare every field against the golden.
int run_text_scenario(Qwen35Model& model, const Qwen35Tokenizer& tk,
                      cudaStream_t stream, const std::string& gprefix,
                      const Scenario& sc) {
  std::fprintf(stderr, "[text %s] prompt=%zu bytes, max_new=%d\n", sc.name,
               strlen(sc.text), kMaxNew);
  const std::string path = gprefix + std::string("_gen_") + sc.name +
                           ".cudalm";
  WeightFileV2 gg;
  Status s = WeightFileV2::load(path, &gg);
  if (!s.ok) {
    std::fprintf(stderr, "  load error %s: %s\n", path.c_str(),
                 s.message.c_str());
    return 1;
  }
  const std::string* gprompt = gg.meta("gen.prompt");
  const std::string* gtok = gg.meta("gen.tokens");
  const std::string* gstop = gg.meta("gen.stop_reason");
  const std::string* gtused = gg.meta("gen.t_used");
  const std::string* gtext = gg.meta("gen.prompt_text");
  const std::string* ghfdec = gg.meta("gen.hf_decoded");
  if (!gprompt || !gtok || !gstop || !gtused || !gtext || !ghfdec) {
    std::fprintf(stderr, "  missing gen metadata in %s\n", path.c_str());
    return 1;
  }
  // Drift guard: the golden's stored prompt text must be the test's constant.
  CHECK(*gtext == std::string(sc.text));

  const Qwen35TextGenerator textgen(model, tk);
  const TextGenerationResult r =
      textgen.generate_text(sc.text, kMaxNew, stream);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "  generate_text error: %s\n", r.error.c_str());
    return 1;
  }

  const std::vector<std::uint32_t> want_prompt = parse_ids(*gprompt);
  const std::vector<std::uint32_t> want_gen = parse_ids(*gtok);
  const int want_stop = std::atoi(gstop->c_str());
  const int want_tused = std::atoi(gtused->c_str());

  // 1) native encode == pinned HF encode (EXACT).
  CHECK_EQ(r.prompt_token_ids.size(), want_prompt.size());
  bool prompt_ok =
      r.prompt_token_ids.size() == want_prompt.size();
  for (std::size_t i = 0; prompt_ok && i < want_prompt.size(); ++i) {
    if (r.prompt_token_ids[i] != want_prompt[i]) prompt_ok = false;
  }
  if (!prompt_ok) {
    std::fprintf(stderr, "  prompt ids MISMATCH:\n    want %zu ids\n",
                 want_prompt.size());
    return 1;
  }
  // 2) greedy decode == pinned-quantized oracle (EXACT).
  CHECK_EQ(r.generated_token_ids.size(), want_gen.size());
  bool gen_ok = r.generated_token_ids.size() == want_gen.size();
  for (std::size_t i = 0; gen_ok && i < want_gen.size(); ++i) {
    if (r.generated_token_ids[i] != want_gen[i]) gen_ok = false;
  }
  if (!gen_ok) {
    std::fprintf(stderr, "  generated ids MISMATCH:\n    want %zu ids\n",
                 want_gen.size());
    return 1;
  }
  // 3) stop reason + forward count.
  CHECK_EQ(static_cast<int>(r.stop_reason), want_stop);
  CHECK_EQ(r.forward_count, want_tused);
  // 4) native decode == pinned HF decode (EXACT, skip_special_tokens=False).
  CHECK(r.generated_text == *ghfdec);
  if (r.generated_text != *ghfdec) {
    std::fprintf(stderr, "  decoded text MISMATCH (want %zu bytes)\n",
                 ghfdec->size());
    return 1;
  }
  std::fprintf(stderr,
               "  %-42s OK (prompt %zu tok, generated %zu tok, stop=%d, "
               "text %zu bytes)\n",
               sc.name, r.prompt_token_ids.size(),
               r.generated_token_ids.size(), want_stop,
               r.generated_text.size());
  return 0;
}

// Contamination gate: T1 -> T3 -> T1 again; the second T1 must be identical.
int test_text_contamination(Qwen35Model& model, const Qwen35Tokenizer& tk,
                            cudaStream_t stream) {
  std::fprintf(stderr, "[contamination] T1 -> T3 -> T1 repeat\n");
  const Qwen35TextGenerator textgen(model, tk);
  const TextGenerationResult a1 =
      textgen.generate_text(kScenarios[0].text, kMaxNew, stream);
  CHECK(a1.ok);
  const TextGenerationResult b =
      textgen.generate_text(kScenarios[2].text, kMaxNew, stream);
  CHECK(b.ok);
  const TextGenerationResult a2 =
      textgen.generate_text(kScenarios[0].text, kMaxNew, stream);
  CHECK(a2.ok);
  CHECK(a1.prompt_token_ids == a2.prompt_token_ids);
  CHECK(a1.generated_token_ids == a2.generated_token_ids);
  CHECK(a1.generated_text == a2.generated_text);
  CHECK(a1.stop_reason == a2.stop_reason);
  CHECK(a1.forward_count == a2.forward_count);
  // The intervening T3 run must have used a DIFFERENT prompt (guard against
  // a trivially-equal scenario list).
  CHECK(b.prompt_token_ids != a1.prompt_token_ids);
  std::fprintf(stderr, "  %-42s OK\n", "contamination");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr,
                 "usage: %s <model.cudalm> <tokenizer.cudaltk> "
                 "<checkpoint_dir> <python> <src_dir> <golden_prefix> "
                 "[--no-gen]\n",
                 argv[0]);
    return 2;
  }
  const std::string model_path = argv[1];
  const std::string tk_path = argv[2];
  const std::string ckpt = argv[3];
  const std::string py = argv[4];
  const std::string src = argv[5];
  const std::string gprefix = argv[6];
  const bool no_gen = (argc >= 8 && std::string(argv[7]) == "--no-gen");

  if (!no_gen &&
      (!file_exists(ckpt + "/model.safetensors") ||
       !file_exists(ckpt + "/tokenizer/tokenizer.json"))) {
    std::fprintf(stderr,
                 "[SKIP] qwen35 text generation: checkpoint or tokenizer "
                 "absent at %s\n",
                 ckpt.c_str());
    return 77;
  }
  if (!no_gen) {
    // Self-generate the CUDLMTK1 artifact if absent (the tokenizer unit test
    // also generates it; this keeps the text gate order-independent).
    if (!file_exists(tk_path)) {
      int rc = run_cmd(py + " " + src +
                       "/tools/convert_qwen35_tokenizer.py --out " + tk_path);
      CHECK_EQ(rc, 0);
    }
    std::string cmd = py + " " + src + "/tools/generate_qwen35_golden.py" +
                      " --cudalm " + model_path + " --checkpoint-dir " + ckpt +
                      " --gen-prefix " + gprefix + " --gen-text-max " +
                      std::to_string(kMaxNew);
    for (const Scenario& sc : kScenarios) {
      cmd += " --gen-text '" + std::string(sc.text) + "'";
    }
    int rc = run_cmd(cmd);
    CHECK_EQ(rc, 0);
  } else {
    for (const Scenario& sc : kScenarios) {
      CHECK(file_exists(gprefix + std::string("_gen_") + sc.name +
                        ".cudalm"));
    }
  }

  // Load the native tokenizer (CUDLMTK1) + the full model.
  std::unique_ptr<Qwen35Tokenizer> tk;
  Status s = Qwen35Tokenizer::load(tk_path, &tk);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "Qwen35Tokenizer::load: %s\n", s.message.c_str());
    return 1;
  }
  // The real EOS contract: the generator must stop on the tokenizer's
  // pinned EOS, and that id must be the single pinned 248044.
  CHECK_EQ(tk->eos_token_id(), Qwen35Tokenizer::kEosTokenId);

  WeightFileV2 file;
  s = WeightFileV2::load(model_path, &file);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "WeightFileV2::load: %s\n", s.message.c_str());
    return 1;
  }
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

  int rc = 0;
  for (const Scenario& sc : kScenarios) {
    rc |= run_text_scenario(model, *tk, stream, gprefix, sc);
  }
  rc |= test_text_contamination(model, *tk, stream);
  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc != 0) {
    std::fprintf(stderr, "qwen35 text generation: FAILURES (rc=%d)\n", rc);
    return 1;
  }
  std::fprintf(stderr,
               "[PASS] qwen35 text generation (native encode == HF, "
               "greedy == quantized oracle, native decode == HF, real EOS "
               "248044, contamination clean)\n");
  return 0;
}
