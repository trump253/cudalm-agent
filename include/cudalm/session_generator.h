// CUDALM — v0.8 Phase B: incremental multi-turn generation over sessions.
//
// A THIN generation engine over the frozen v0.8 Phase A session runtime +
// the frozen v0.5 external-state forward. It adds NO model state and NO
// new kernels: one turn drives Qwen35Model::forward_token_with_state() on
// the session's BOUND sequence, position derived from the bound
// SequenceState::length (the single source of truth, inherited from the
// v0.5 forward gate).
//
// INCREMENTAL MULTI-TURN (the Phase B contract, pinned):
//
//   turn 1:  session A append [a,b,c]  -> generate [d,e]
//   turn 2:  same session A append [f,g] -> generate ...
//
//   * APPEND-ONLY: a turn forwards ONLY `new_input_tokens` (each ONCE,
//     starting at the session's CURRENT logical length) — no re-prefill of
//     history, no session reset, no KV / Delta state copy or rebuild;
//   * the turn then generates up to `max_new_tokens` tokens, continuing
//     from the existing KV + Delta conv/recurrent + position;
//   * CONSEQUENCE (gated bit-identical by
//     tests/cuda/test_qwen35_session_generation.cpp): a session driven in
//     turns produces the SAME per-step full logits, the SAME generated
//     token ids, and the SAME final hybrid state (paged KV + Delta conv +
//     Delta recurrent) as the equivalent ONE-SHOT CONTINUOUS execution of
//     the same token stream — chunking into turns is numerically invisible.
//
// COMMIT CONTRACT (pinned — the anti-"lagging state" rule):
//
//   every token returned in TurnResult::generated has been FORWARDED
//   (committed to the session's KV / Delta conv / Delta recurrent /
//   position state) BEFORE the turn returns — INCLUDING the token that
//   triggered the stop (EOS / max_new_tokens). After a successful turn the
//   session's state is EXACTLY the state of its full committed token
//   history; there is no "generated returned the last token but the
//   session state does not contain it yet" lag. (This deliberately
//   DIFFERS from the frozen v0.4 Qwen35Generator one-shot request, whose
//   contract forwards N + (m-1) tokens and does NOT commit the last
//   generated token — v0.4 is frozen and untouched; the two contracts are
//   each pinned by their own gates.)
//
// max_new_tokens == 0: an INPUT-ONLY append — the input is forwarded
// (committed) and nothing is generated (generated empty,
// stop_reason MaxNewTokens, forward_count == input_count).
//
// EOS: `eos_token_id == -1` DISABLES the EOS gate (the v0.6 Request
// convention); otherwise it must be in [0, vocab_size). When a generated
// token equals it, it is COMMITTED like every other generated token and
// the turn stops (Eos) right after it; EOS takes priority over
// max_new_tokens (the v0.4 convention).
//
// CONTEXT OVERFLOW (the v0.8 policy, inherited from Phase A): there is NO
// eviction of any kind. A turn whose length + input + max_new_tokens
// exceeds max_seq_len is REJECTED at preflight (fail loud) — never a
// silent drop of the earliest tokens/pages. Because of that preflight a
// successful turn can NEVER run out of context mid-execution (the last
// committed token sits at position <= max_seq_len - 1 by construction), so
// a successful turn's stop_reason is Eos or MaxNewTokens only.
//
// PREFLIGHT (fail-loud; ZERO mutation — no forward, no KV allocation, no
// Delta mutation, length unchanged), in check order:
//   1. SessionId is a LIVE session;
//   2. new_input_tokens is non-empty;
//   3. max_new_tokens >= 0;
//   4. the sampling config is valid (validate_sampling_config);
//   5. CONTEXT OVERFLOW: length + input_count + max_new_tokens <=
//      max_seq_len (the exact boundary — equal — is ACCEPTED);
//   6. the stream equals the pools' single stream (the v0.5 single-stream
//      contract, preflighted);
//   7. the model is loaded;
//   8. eos_token_id == -1 or in [0, vocab_size);
//   9. every input token id is in [0, vocab_size);
//  10. the model's config EXACTLY matches the manager's config (the v0.5
//      compatibility gate, preflighted).
// (Checks 1-6 need no model vocab, so the preflight contract is fully
// testable without a checkpoint —
// tests/cpu/test_session_turn_contract.cpp.)
//
// RUNTIME FAILURE (no rollback by design — no snapshot is taken, no
// expensive copy): if a forward fails AFTER some tokens were committed
// (e.g. a KV page-pool OOM — a capacity condition the preflight cannot
// see), every already-committed token is KEPT: the session stays at the
// last successful token boundary (TurnResult::context_length says exactly
// where), the failing token is NOT committed (not in `generated`),
// ok == false + error is set.
//
// SAMPLING: a FRESH per-turn Sampler (its SplitMix64 seeded from
// `sampling.seed`) — per-turn RNG isolation, the v0.4 per-request
// discipline: RNG state never crosses turns. A greedy config
// (temperature <= 0) runs the frozen argmax bit-for-bit (zero RNG).
//
// The caller passes the single CUDA stream; it must equal the pools'
// stream (preflight check 6). The model and the sessions are non-owning
// (the same discipline as Qwen35Generator / ModelForwarder — both
// outlive this generator).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cudalm/greedy.h"           // StopReason
#include "cudalm/qwen35_generator.h" // LogitsObserver
#include "cudalm/qwen35_model.h"
#include "cudalm/sampling.h"         // SamplingConfig, Sampler
#include "cudalm/session.h"          // SessionId, SessionManager

namespace cudalm {

// The result of ONE session turn (see SessionGenerator::generate_turn for
// the full contract).
struct TurnResult {
  bool ok = false;
  std::string error;  // the error message (if !ok)
  // Committed generated token ids, in order. COMMIT CONTRACT: every id in
  // this vector was forwarded (committed to the session state) before the
  // turn returned — including the stop-triggering token.
  std::vector<int> generated;
  // Why generation stopped (valid only when ok == true):
  //   Eos          — the committed EOS token completed the turn;
  //   MaxNewTokens — max_new_tokens generated tokens were committed (also
  //                  the stop reason of an input-only turn, max_new_tokens
  //                  == 0, mirroring the v0.4 documentation).
  StopReason stop_reason = StopReason::MaxNewTokens;
  // Input tokens committed in this turn (0 on a preflight failure).
  int input_count = 0;
  // Committed forwards in this turn (== input_count + generated.size()).
  int forward_count = 0;
  // The session's logical length AFTER this turn (== the length before the
  // turn + forward_count). On a runtime failure this is the exact last
  // successful token boundary — the session is left there (no rollback).
  int context_length = 0;
};

// One incremental multi-turn generation engine over the v0.8 Phase A
// session runtime. Owns nothing: the model is driven in place and the
// sessions are read through the SessionManager (both outlive this).
class SessionGenerator {
 public:
  SessionGenerator(Qwen35Model& model, SessionManager& sessions)
      : model_(model), sessions_(sessions) {}

  SessionGenerator(const SessionGenerator&) = delete;
  SessionGenerator& operator=(const SessionGenerator&) = delete;

  // Run ONE incremental turn on a LIVE session (the full contract is in
  // the file header): append `new_input_tokens` to the session's existing
  // context, then generate up to `max_new_tokens` tokens.
  //
  // `observer` (optional) is invoked after EVERY committed forward of the
  // turn with the step's FULL host logits [vocab_size] and the turn-local
  // step index (0-based: input tokens are steps 0..N-1, generated tokens
  // are steps N..N+M'-1) — the same host buffer the sampler reads, valid
  // until the next forward.
  TurnResult generate_turn(SessionId session_id,
                           const std::vector<int>& new_input_tokens,
                           int max_new_tokens, int eos_token_id,
                           const SamplingConfig& sampling, cudaStream_t stream,
                           const LogitsObserver* observer = nullptr);

 private:
  Qwen35Model& model_;
  SessionManager& sessions_;
  std::vector<__nv_bfloat16> host_logits_;  // D2H scratch (grow-only)
};

}  // namespace cudalm
