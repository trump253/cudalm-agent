// CUDALM — greedy generation helpers (v0.4 Phase A). CPU-only + deterministic:
// no CUDA runtime, no model, no stream. The argmax is the greedy token pick
// (maximize the numeric BF16 logit value; ties -> lowest token id) and the
// stop controller is the pure greedy-loop stop decision. The C++ generator
// (qwen35_generator.h) + the standalone argmax / stop unit tests both build on
// this header.
//
// Scope (v0.4 Phase A, docs §19): single-request, greedy ONLY. No tokenizer /
// detokenizer / sampling (temperature, top-k, top-p) / repetition penalty /
// beam search. No batched/chunked prefill / Paged KV / multi-request.

#pragma once

#include <cuda_bf16.h>

namespace cudalm {

// CPU greedy argmax over BF16 logits.
//
// Contract: maximizes the NUMERIC BF16 logit value (each element is compared
// as its float value); on an exact tie the LOWEST token id wins. Precondition:
// n >= 1, logits non-null, and the logits are finite (the model produces
// finite bf16 logits; NaN handling is out of contract).
//
// This is the Phase A correctness-first path: the caller copies the full
// [vocab] bf16 logits D2H and runs this CPU loop (no CUDA argmax/reduction
// kernel — that is explicitly out of scope for Phase A).
inline int argmax_bf16(const __nv_bfloat16* logits, int n) {
  int best = 0;
  float best_val = __bfloat162float(logits[0]);
  for (int i = 1; i < n; ++i) {
    const float v = __bfloat162float(logits[i]);
    if (v > best_val) {  // strict '>' -> an exact tie keeps the lowest index
      best = i;
      best_val = v;
    }
  }
  return best;
}

// Why a greedy generation stopped.
enum class StopReason {
  Eos,          // the EOS token was generated + included; stop right after it.
  MaxNewTokens, // max_new_tokens tokens were generated; stop.
  MaxSeqLen,    // prompt + generated reached max_seq_len; no room for more.
};

inline const char* stop_reason_name(StopReason r) {
  switch (r) {
    case StopReason::Eos: return "eos";
    case StopReason::MaxNewTokens: return "max_new_tokens";
    case StopReason::MaxSeqLen: return "max_seq_len";
  }
  return "unknown";
}

// The generation golden stores the stop reason as this small code.
inline int stop_reason_code(StopReason r) {
  switch (r) {
    case StopReason::Eos: return 0;
    case StopReason::MaxNewTokens: return 1;
    case StopReason::MaxSeqLen: return 2;
  }
  return -1;
}

// Pure greedy stop controller (deterministic; no CUDA / model).
//
// Models the greedy loop, which places the (step+1)-th generated token at
// position `prompt_len + step` (step is 0-based):
//   1. If that position >= max_seq_len -> stop (MaxSeqLen); the token is NOT
//      placed (capacity full).
//   2. Place the token (append to the generated sequence).
//   3. If the token == eos_token_id -> stop (Eos); the token IS included; the
//      caller must NOT forward the EOS token.
//   4. Else if this was the max_new_tokens-th token -> stop (MaxNewTokens).
//   5. Else forward the token + read the next candidate.
// EOS takes priority over MaxNewTokens (a natural stop on the last allowed
// token is still an EOS stop).
struct GreedyStopController {
  int eos_token_id = -1;
  int max_new_tokens = 0;
  int prompt_len = 0;
  int max_seq_len = 0;

  // Position of the (step+1)-th generated token (step is 0-based).
  int position_of(int step) const { return prompt_len + step; }

  // Can the (step+1)-th generated token be placed (its position is within the
  // max_seq_len capacity)?
  bool placeable(int step) const { return position_of(step) < max_seq_len; }

  // After placing `token` as the (step+1)-th generated token, do we stop? If
  // so, why. `stop == false` means: forward the token + read the next.
  struct Decision {
    bool stop = false;
    StopReason reason = StopReason::MaxNewTokens;  // valid iff stop
  };
  Decision stop_after_token(int token, int step) const {
    if (token == eos_token_id) return Decision{true, StopReason::Eos};
    if (step + 1 >= max_new_tokens)
      return Decision{true, StopReason::MaxNewTokens};
    return Decision{false, StopReason::MaxNewTokens};
  }
};

}  // namespace cudalm
