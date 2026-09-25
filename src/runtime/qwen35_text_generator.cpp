// CUDALM — Qwen3.5 text-level generation facade (v0.4 Phase B).
//
// See include/cudalm/qwen35_text_generator.h for the contract. This file
// contains no generation logic: it is the three-line stitch (encode,
// generate, decode) over the two already-gated components.

#include "cudalm/qwen35_text_generator.h"

namespace cudalm {

Status Qwen35TextGenerator::encode_text(const std::string& text,
                                        std::vector<std::uint32_t>* ids) const {
  return tokenizer_.encode(text, ids);
}

Status Qwen35TextGenerator::decode_ids(const std::vector<std::uint32_t>& ids,
                                       bool skip_special_tokens,
                                       std::string* text) const {
  return tokenizer_.decode(ids.data(), ids.size(), skip_special_tokens, text);
}

TextGenerationResult Qwen35TextGenerator::generate_text(
    const std::string& prompt_text, int max_new_tokens,
    cudaStream_t stream) const {
  TextGenerationResult r;

  // 1) Native encode (fail loud on invalid UTF-8 / internal errors).
  Status s = tokenizer_.encode(prompt_text, &r.prompt_token_ids);
  if (!s.ok) {
    r.ok = false;
    r.error = s.message;
    return r;
  }
  if (r.prompt_token_ids.empty()) {
    r.ok = false;
    r.error = "generate_text: empty prompt (the native encode produced no "
              "tokens)";
    return r;
  }

  // 2) Greedy generation with the tokenizer's REAL pinned EOS (Phase A
  //    contract: the generator validates eos / max_new / prompt ids and
  //    never partially forwards on a contract violation).
  const std::vector<int> prompt(r.prompt_token_ids.begin(),
                                r.prompt_token_ids.end());
  Qwen35Generator gen(model_);
  const GenerationResult gr = gen.generate(
      prompt, max_new_tokens, static_cast<int>(tokenizer_.eos_token_id()),
      stream);
  if (!gr.ok) {
    r.ok = false;
    r.error = gr.error;
    return r;
  }
  r.generated_token_ids.assign(gr.generated.begin(), gr.generated.end());
  r.stop_reason = gr.stop_reason;
  r.forward_count = gr.forward_count;

  // 3) Native decode of the generated ids. skip_special_tokens == false: a
  //    generated EOS stays in the decoded text (exactly like the pinned HF
  //    oracle with skip_special_tokens=False, which the golden's
  //    gen.hf_decoded field uses).
  s = tokenizer_.decode(r.generated_token_ids.data(),
                        r.generated_token_ids.size(), false,
                        &r.generated_text);
  if (!s.ok) {
    r.ok = false;
    r.error = s.message;
    return r;
  }

  r.ok = true;
  return r;
}

}  // namespace cudalm
