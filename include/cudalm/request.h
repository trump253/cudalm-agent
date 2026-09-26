// CUDALM — v0.6 Phase A: scheduler request runtime (control plane).
//
// A scheduler-level REQUEST (an end-user generation job) is distinct from
// the v0.5 SEQUENCE state owned by Qwen35StateManager:
//
//   Request (this file, owned by the Scheduler)
//     -> SequenceId (a v0.5 live sequence: created at admission, retired
//        EXACTLY ONCE at the request's terminal transition)
//
// RequestId: monotonic, NEVER REUSED (the same discipline as SequenceId).
// A RequestId must NEVER be used as a SequenceId: the two id spaces are
// independent (the Scheduler issues RequestIds from 1 upward; the state
// manager issues SequenceIds from 1 upward on its own — they happen to
// coincide numerically in trivial cases and must not be treated as the
// same thing).
//
// Per-request SAMPLING ISOLATION (v0.4 sampling contract): each request
// owns its SamplingConfig (the seed lives inside) AND its own Sampler
// (the per-request SplitMix64). Two requests never share an RNG
// progression, so interleaving requests can never change any request's
// token stream (pinned by the v0.6 Phase A sampling isolation gate).
//
// Token progression (v0.4 semantics, pinned — "no off-by-one"):
//   * prefill : prompt[0..N-1] is forwarded ONE TOKEN PER scheduler step;
//     NO sampling happens on the early prefill forwards (the next token
//     is already known from the prompt);
//   * the LOGITS of the LAST prompt forward produce the FIRST generated
//     token g0;
//   * decode  : forward g_{k-1} -> logits -> sample g_k; the request
//     completes when g_k == eos_token_id (the EOS token IS included in
//     `generated`, exactly like v0.4) or generated.size() ==
//     max_new_tokens.
// Hence a request that finishes with m generated tokens has forwarded
// exactly N + (m - 1) tokens (recorded in `forward_count`).
//
// This file is header-only data (no model, no CUDA, no policy).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cudalm/qwen35_state_manager.h"  // SequenceId
#include "cudalm/sampling.h"               // SamplingConfig, Sampler

namespace cudalm {

// One end-user generation job, owned by the Scheduler.
using RequestId = std::uint64_t;

enum class RequestStatus {
  Waiting,     // admitted, not yet advanced (no successful forward yet)
  Running,     // at least one successful forward; prefilling or decoding
  Finished,    // terminal: completed (EOS / max_new_tokens)
  Cancelled,   // terminal: cancelled by the caller
  Failed,      // terminal: a fatal Status from the forward path
};

enum class FinishReason {
  None,          // not terminal
  Eos,           // the sampled EOS token completed the request
  MaxNewTokens,  // max_new_tokens generated tokens were sampled
  Cancelled,     // cancelled by the caller
  Failed,        // fatal Status (forward / logits path)
};

inline bool is_terminal(RequestStatus s) {
  return s == RequestStatus::Finished || s == RequestStatus::Cancelled ||
         s == RequestStatus::Failed;
}

struct Request {
  // Constructible only with its full identity + sampling config (the
  // per-request Sampler has NO default constructor — its SplitMix64 must
  // be seeded from the request's config at construction).
  Request() = delete;
  Request(RequestId request_id, SequenceId sequence_id,
          std::vector<int> prompt_tokens, int max_new, int eos,
          const SamplingConfig& cfg)
      : id(request_id),
        sequence_id(sequence_id),
        prompt(std::move(prompt_tokens)),
        max_new_tokens(max_new),
        eos_token_id(eos),
        sampling(cfg),
        sampler(cfg) {}

  RequestId id = 0;
  SequenceId sequence_id = 0;  // 0 = none (manager ids start at 1)
  std::vector<int> prompt;     // prompt token ids (frozen at admission)
  int prefill_pos = 0;         // prompt tokens already forwarded
  std::vector<int> generated;  // generated token ids (sampled, in order)
  int max_new_tokens = 0;
  int eos_token_id = -1;       // -1 = no EOS gate
  SamplingConfig sampling;     // per-request sampling config (seed inside)
  Sampler sampler;             // per-request RNG (constructed from `sampling`)
  RequestStatus status = RequestStatus::Waiting;
  FinishReason finish_reason = FinishReason::None;
  int forward_count = 0;  // successful single-token forwards for this request
};

}  // namespace cudalm
