// CUDALM — Qwen3.5 generation runtime implementation (v0.4 Phase A).
//
// Drives the existing Qwen35Model accessors (reset_state / forward_token /
// logits) to run a single-request, serial-prefill, greedy-decode generation.
// No new CUDA kernel, no new model state — the model threads its OWN
// persistent state across the serial steps.

#include "cudalm/qwen35_generator.h"

#include <cuda_bf16.h>

#include "cudalm/cuda_check.h"
#include "cudalm/qwen35_config.h"

namespace cudalm {

GenerationResult Qwen35Generator::generate(
    const std::vector<int>& prompt_tokens, int max_new_tokens, int eos_token_id,
    cudaStream_t stream, const LogitsObserver* observer) {
  GenerationResult result;
  const Qwen35Config& cfg = model_.config();
  const int vocab = cfg.vocab_size;
  const int max_seq = cfg.max_seq_len;

  // ---- Input / capacity contract (fail loud: ok == false, error set) ------
  auto fail = [this, &result](const char* msg) {
    result.ok = false;
    result.error = msg;
  };
  if (!model_.loaded()) { fail("model not loaded"); return result; }
  if (prompt_tokens.empty()) { fail("empty prompt"); return result; }
  if (max_new_tokens < 0) { fail("max_new_tokens < 0"); return result; }
  if (eos_token_id < 0 || eos_token_id >= vocab) {
    fail("invalid eos_token_id"); return result;
  }
  for (int t : prompt_tokens) {
    if (t < 0 || t >= vocab) { fail("invalid prompt token id"); return result; }
  }
  if (static_cast<int>(prompt_tokens.size()) > max_seq) {
    fail("prompt length > max_seq_len"); return result;
  }

  const int N = static_cast<int>(prompt_tokens.size());
  result.prompt_count = N;
  result.ok = true;

  // Host logits scratch (reused each step) for the CPU argmax + the observer.
  std::vector<__nv_bfloat16> h_logits(static_cast<std::size_t>(vocab));
  auto read_logits = [&]() -> int {
    CUDA_CHECK(cudaMemcpyAsync(h_logits.data(), model_.logits(),
                               static_cast<std::size_t>(vocab) *
                                   sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return argmax_bf16(h_logits.data(), vocab);
  };

  // ---- fresh reset (single request; each generate() starts clean) ---------
  model_.reset_state(stream);

  // ---- serial prefill: positions 0..N-1 -----------------------------------
  for (int i = 0; i < N; ++i) {
    model_.forward_token(prompt_tokens[i], i, stream);
    result.forward_count++;
  }

  // ---- max_new_tokens == 0: empty generation, no decode (documented) ------
  if (max_new_tokens == 0) {
    result.stop_reason = StopReason::MaxNewTokens;
    return result;
  }

  // ---- no room for generation (the prompt filled the sequence) ------------
  if (N >= max_seq) {
    result.stop_reason = StopReason::MaxSeqLen;
    return result;
  }

  // ---- prefill-last logits predict the first generated token (position N) --
  int next = read_logits();

  GreedyStopController stop{eos_token_id, max_new_tokens, N, max_seq};
  for (int step = 0; step < max_new_tokens; ++step) {
    // Capacity: the (step+1)-th token would sit at position N+step. If that is
    // out of range the sequence is full; stop (MaxSeqLen) and place nothing.
    if (!stop.placeable(step)) {
      result.stop_reason = StopReason::MaxSeqLen;
      return result;
    }
    // The current host logits predict `next` (the token about to be placed);
    // report them as step `step` (step 0 = the prefill-last logits).
    if (observer) (*observer)(h_logits.data(), step);
    result.generated.push_back(next);
    GreedyStopController::Decision d = stop.stop_after_token(next, step);
    if (d.stop) {
      // EOS / max_new_tokens: the token IS included; do NOT forward it.
      result.stop_reason = d.reason;
      return result;
    }
    // Continue: forward the token + read the next candidate.
    model_.forward_token(next, stop.position_of(step), stream);
    result.forward_count++;
    next = read_logits();
  }
  result.stop_reason = StopReason::MaxNewTokens;
  return result;
}

}  // namespace cudalm
