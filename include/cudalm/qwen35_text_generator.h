// CUDALM — Qwen3.5 text-level generation facade (v0.4 Phase B + C).
//
// A THIN facade over the two already-gated components:
//
//   raw UTF-8 prompt text
//     -> Qwen35Tokenizer::encode      (native, oracle-exact; docs §18)
//     -> Qwen35Generator::generate    (native, Phase A contract; docs §19)
//          eos_token_id = the tokenizer's pinned real EOS (248044 for 0.8B)
//     -> Qwen35Tokenizer::decode      (native, oracle-exact; skip_special
//                                      tokens = false: a generated EOS stays
//                                      in the text, exactly like HF)
//     -> TextGenerationResult
//
// Scope inherits Phase A verbatim: single request, serial prefill, no
// batching, no streaming, no chat template, no KV/Graph optimization.
// Phase A = GREEDY only; Phase C adds a SamplingConfig overload that drives
// the generator's sampling pick (cudalm/sampling.h). This class adds NO
// model state and NO generation/sampling logic — it only stitches the two
// gated contracts together, so every byte of text in/out is covered by an
// existing hard gate (tokenizer corpus vs pinned HF; generation goldens vs
// the pinned-quantized oracle).
//
// No special/control token string literals appear in this file or its
// implementation: the EOS is carried by id (the tokenizer's artifact).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/greedy.h"
#include "cudalm/qwen35_generator.h"
#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_tokenizer.h"
#include "cudalm/weight_format.h"

namespace cudalm {

// The result of one text-level greedy generation request.
struct TextGenerationResult {
  bool ok = false;  // false if any stage failed (error set)
  std::string error;  // the first error message (if !ok)
  std::vector<std::uint32_t> prompt_token_ids;  // native encode of the prompt
  std::vector<std::uint32_t> generated_token_ids;  // greedy decode output
  std::string generated_text;  // native decode (skip_special_tokens == false)
  StopReason stop_reason = StopReason::MaxNewTokens;
  int forward_count = 0;  // prefill + decode forward_token calls
};

// Text in / text out. Owns nothing: the model is driven in place (the same
// non-const contract as Qwen35Generator) and the tokenizer is read-only.
class Qwen35TextGenerator {
 public:
  explicit Qwen35TextGenerator(Qwen35Model& model,
                               const Qwen35Tokenizer& tokenizer)
      : model_(model), tokenizer_(tokenizer) {}

  // Native encode of raw UTF-8 text (fail loud on invalid UTF-8).
  Status encode_text(const std::string& text,
                     std::vector<std::uint32_t>* ids) const;

  // Native decode (fail loud on an id outside [0, model_vocab_size)).
  Status decode_ids(const std::vector<std::uint32_t>& ids,
                    bool skip_special_tokens, std::string* text) const;

  // The pinned real EOS id (248044 for the 0.8B checkpoint) — the id the
  // generator stops on (the Phase A sentinel id is NOT used here).
  std::uint32_t eos_token_id() const { return tokenizer_.eos_token_id(); }

  // encode(prompt_text) -> generate(eos = tokenizer EOS) -> decode.
  // On any stage failure ok == false + error is set and the later fields are
  // left empty/partial (the generator itself never partially forwards on an
  // input-contract violation — Phase A contract).
  TextGenerationResult generate_text(const std::string& prompt_text,
                                     int max_new_tokens,
                                     cudaStream_t stream) const;

  // Phase C: the same three-stage stitch with a sampling config. A greedy
  // config (temperature <= 0) is bit-for-bit the overload above; a sampling
  // config runs the generator's per-request seeded sampler (the facade adds
  // no sampling logic of its own).
  TextGenerationResult generate_text(const std::string& prompt_text,
                                     int max_new_tokens,
                                     const SamplingConfig& sampling,
                                     cudaStream_t stream) const;

 private:
  Qwen35Model& model_;
  const Qwen35Tokenizer& tokenizer_;
};

}  // namespace cudalm
