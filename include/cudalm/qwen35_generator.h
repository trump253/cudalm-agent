// CUDALM — Qwen3.5 generation runtime (v0.4 Phase A).
//
// A thin generation layer ON TOP of Qwen35Model: token-ID-level single-request
// SERIAL prefill + GREEDY decode. It does NOT add generation state to
// Qwen35Model — it drives the model's existing forward_token / logits /
// reset_state accessors.
//
// Scope (v0.4 Phase A + C, docs §19/§21):
//   * single request only (no batching / multi-request / scheduler);
//   * greedy (Phase A — frozen, bit-for-bit) + minimal sampling (Phase C:
//     temperature / top-k / top-p / seed; see cudalm/sampling.h);
//   * SERIAL (token-by-token) prefill — a correctness-first baseline, NOT a
//     batched/chunked prefill and NOT a performance claim;
//   * the runtime threads the model's OWN persistent state (no golden state
//     backfeed); each generate() starts with a fresh reset_state().
//   * no tokenizer / detokenizer (token-ID in, token-ID out);
//   * no Paged KV / CUDA Graph / kernel fusion / performance tuning.
//
// Prefill/decode semantics (docs §19): for prompt [A, B, C]
//   forward(A, 0); forward(B, 1); forward(C, 2)
//   the logits after C (position 2) PREDICT the token at position 3;
// greedy pick D -> append D; forward(D, 3); the logits predict position 4;
//   ... until EOS / max_new_tokens / max_seq_len.
// The prompt's last token is forwarded ONCE (in prefill) — it is never
// re-forwarded as a decode token, there is no position off-by-one, and there
// is NO reset before decode (one reset at the very start).

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cudalm/greedy.h"
#include "cudalm/qwen35_model.h"
#include "cudalm/sampling.h"

namespace cudalm {

// The result of a single greedy generation request.
struct GenerationResult {
  bool ok = false;  // false if an input-contract violation (error set)
  std::string error;  // the error message (if !ok)
  int prompt_count = 0;  // the number of prompt tokens
  std::vector<int> generated;  // the generated token ids (decode output)
  StopReason stop_reason = StopReason::MaxNewTokens;  // why it stopped
  // Number of forward_token calls issued (prefill + decode). The test uses
  // this to prove "no forward after EOS" (the EOS token is never forwarded).
  int forward_count = 0;
};

// An optional observer invoked with the FULL host logits [vocab_size] + the
// step index at each generation step: step 0 = the prefill-last logits
// (which predict the 1st generated token), step k = the logits after decoding
// token (k-1) (which predict token k). The generator does NOT retain the
// logits; the observer copies/compares them if it needs to (the golden test
// D2H's the per-step logits + compares to the generation golden).
using LogitsObserver = std::function<void(const __nv_bfloat16* host_logits,
                                          int step)>;

class Qwen35Generator {
 public:
  // Wraps a loaded Qwen35Model (the generator does NOT own it). A NON-const
  // reference is required: generate() drives the model's stateful accessors
  // (reset_state / forward_token thread the model's persistent state in place).
  explicit Qwen35Generator(Qwen35Model& model) : model_(model) {}

  // Run a single greedy generation request (see the file header for the
  // prefill/decode semantics). `observer` (optional) is invoked with each
  // step's host logits (step 0 = prefill-last). Returns a GenerationResult;
  // on an input-contract violation ok == false + error is set and NOTHING is
  // forwarded (no partial state change beyond the pre-validation).
  //
  // Input / capacity contract (fail loud: ok == false, error set):
  //   * empty prompt;
  //   * an invalid prompt token id (outside [0, vocab_size));
  //   * an invalid eos_token_id (outside [0, vocab_size));
  //   * max_new_tokens < 0;
  //   * prompt length > max_seq_len.
  // Safe handling: once prompt + generated would reach max_seq_len the loop
  // stops (MaxSeqLen) and forward_token is NEVER called with position >=
  // max_seq_len. max_new_tokens == 0 returns an empty `generated` (no decode).
  GenerationResult generate(const std::vector<int>& prompt_tokens,
                            int max_new_tokens, int eos_token_id,
                            cudaStream_t stream,
                            const LogitsObserver* observer = nullptr);

  // Run a single generation request with a sampling config (v0.4 Phase C).
  // Same prefill/decode semantics, input/capacity contract and result shape
  // as the greedy overload above; the ONLY difference is the per-step token
  // pick: a greedy config (temperature <= 0, e.g. SamplingConfig::greedy())
  // runs the FROZEN Phase A argmax path bit-for-bit (zero RNG consumption),
  // while a sampling config uses the fixed temperature -> top-k -> top-p ->
  // normalize -> sample pipeline (cudalm/sampling.h) with a per-request
  // SplitMix64 seeded by `sampling.seed`.
  //
  // Additional contract (fail loud, same style as the input contract):
  // an INVALID sampling config (validate_sampling_config) -> ok == false +
  // error set, NOTHING forwarded. EOS / max_new_tokens / max_seq_len stop
  // semantics and the forward count are UNCHANGED by sampling.
  GenerationResult generate(const std::vector<int>& prompt_tokens,
                            int max_new_tokens, int eos_token_id,
                            cudaStream_t stream,
                            const SamplingConfig& sampling,
                            const LogitsObserver* observer = nullptr);

 private:
  // The shared request core (both public overloads drive it; the greedy
  // overload passes SamplingConfig::greedy(), which is bit-for-bit the
  // Phase A path).
  GenerationResult generate_impl(const std::vector<int>& prompt_tokens,
                                 int max_new_tokens, int eos_token_id,
                                 cudaStream_t stream,
                                 const SamplingConfig& sampling,
                                 const LogitsObserver* observer);

  Qwen35Model& model_;
};

}  // namespace cudalm
