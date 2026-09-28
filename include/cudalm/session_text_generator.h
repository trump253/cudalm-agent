// CUDALM — Qwen3.5 text-level MULTI-TURN session facade (v0.8 Phase D).
//
// A THIN facade over the already-gated components (it adds NO model math,
// NO generation loop and NO sampling logic — every byte of behavior below
// is an existing frozen contract):
//
//   raw UTF-8 user text (one turn)
//     -> Qwen35Tokenizer::encode          (native, oracle-exact; fail loud
//                                          on invalid UTF-8)
//     -> Scheduler::admit_session_turn    (Phase C session-bound turn on
//                                          the session's EXISTING bound
//                                          sequence — append-only)
//     -> Scheduler::run                   (the FROZEN v0.6 control plane;
//                                          commit-then-stop semantics)
//     -> Qwen35Tokenizer::decode          (native, oracle-exact)
//
// TEXT-SESSION CONTRACT (pinned — this is a RAW TEXT completion session,
// NOT a chat/instruct serving API):
//   * NO chat template, NO special/control tokens and NO separator of any
//     kind is added to or stripped from the user text: the encoded ids of
//     the user's chunk are appended VERBATIM to the session's token
//     stream. The base model simply continues the stream (a newline or
//     any separator would be a hidden input modification — forbidden).
//   * INCREMENTAL MULTI-TURN: turn 2 encodes ONLY the new text and the
//     scheduler APPENDS it from the session's current logical length —
//     turn 1's history is never re-encoded or re-forwarded (the Phase C
//     append-only contract; gated end-to-end).
//   * COMMIT: on ok == true, EVERY generated id in the result has already
//     been committed (forwarded) into the session's KV / Delta / position
//     state by the Phase B/C commit contract — the session's logical
//     length equals exactly input + committed generated ids.
//   * FAILURES LEAVE NO HALF-TURN: invalid UTF-8, admission failure
//     (unknown / busy session, context overflow, invalid config) and
//     forward failure all return ok == false with the session at its last
//     committed boundary (admission is zero-mutation; a failed forward
//     commits nothing) — the session stays LIVE.
//
// No special/control token string literals appear in this file or its
// implementation: the EOS is carried by id (the tokenizer's artifact).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/qwen35_state_manager.h"
#include "cudalm/qwen35_tokenizer.h"
#include "cudalm/request.h"
#include "cudalm/sampling.h"
#include "cudalm/scheduler.h"
#include "cudalm/session.h"
#include "cudalm/weight_format.h"  // Status

namespace cudalm {

// The result of one text-level session turn.
struct SessionTextTurnResult {
  bool ok = false;  // false if any stage failed (error set)
  std::string error;  // the first error message (if !ok)
  // The native encode of the user text (EMPTY for an encode failure —
  // the raw bytes are not recoverable ids).
  std::vector<int> input_token_ids;
  // Generated token ids. On ok == true: ALL of them are COMMITTED into
  // the session state (the Phase B/C commit contract). On ok == false:
  // only the COMMITTED prefix (a failed turn never reports an
  // uncommitted tail as history).
  std::vector<int> generated_token_ids;
  // Native decode of generated_token_ids (skip_special_tokens == false:
  // a generated EOS stays in the text, exactly like the v0.4 text
  // contract). EMPTY on !ok (decode happens only on a finished turn).
  std::string generated_text;
  FinishReason stop_reason = FinishReason::None;
  // The session's logical length AFTER the turn (== its length before +
  // committed input + committed generated; on a failed turn this is the
  // last committed boundary).
  int context_length = 0;
  // The scheduler request that carried the turn (0 if admission itself
  // failed). Terminal requests stay inspectable via the scheduler.
  RequestId request_id = 0;
  int forward_count = 0;  // committed forwards for this request
};

// Text in / text out, one persistent session. Owns its Scheduler (the
// control plane for the turn requests) and nothing else: the forwarder,
// the state manager, the session manager and the tokenizer are all
// NON-OWNING (they outlive the facade, the v0.5/v0.6 discipline).
class Qwen35SessionTextGenerator {
 public:
  // fwd/mgr/sessions/tokenizer/stream are NON-OWNING. `stream` is the
  // single stream (the v0.5 gate inside forward_token_with_state enforces
  // the pool/stream match fail-loud). The internal scheduler is
  // constructed over (fwd, mgr, stream, &sessions) — the Phase C
  // session-bound mode.
  Qwen35SessionTextGenerator(SequenceForwarder& fwd,
                             const Qwen35Tokenizer& tokenizer,
                             Qwen35StateManager& mgr, SessionManager& sessions,
                             cudaStream_t stream);

  // ---- session lifecycle (thin forwarding to the SessionManager) -------
  Status create_session(SessionId* out_id);
  Status reset_session(SessionId session_id);
  Status destroy_session(SessionId session_id);

  // ---- one text turn -----------------------------------------------------
  // encode(new_text) -> admit_session_turn(eos = the tokenizer's pinned
  // real EOS — the v0.4 text contract) -> run() -> decode. See the file
  // header for the pinned text-session contract.
  SessionTextTurnResult generate_turn(SessionId session_id,
                                      const std::string& new_text,
                                      int max_new_tokens,
                                      const SamplingConfig& sampling);

  // The same turn with an explicit EOS gate (eos_token_id == -1 = no
  // gate, the v0.6 convention).
  SessionTextTurnResult generate_turn(SessionId session_id,
                                      const std::string& new_text,
                                      int max_new_tokens,
                                      const SamplingConfig& sampling,
                                      int eos_token_id);

  // Observability (the scheduler's serving metrics / request records).
  const Scheduler& scheduler() const { return sched_; }

 private:
  SequenceForwarder& fwd_;
  const Qwen35Tokenizer& tokenizer_;
  Qwen35StateManager& mgr_;
  SessionManager& sessions_;
  cudaStream_t stream_;
  Scheduler sched_;  // last: constructed over the references above
};

}  // namespace cudalm
