// CUDALM — v0.9 Phase A: serving admission / backpressure / resource
// guardrails (a THIN policy layer above the frozen v0.8 runtime).
//
// The frozen runtime (SessionManager + Scheduler + pools) is responsible
// for CORRECTNESS; this layer is responsible for POLICY:
//
//   * ADMISSION LIMITS (ServingLimits):
//       - max_sessions:               live sessions quota for
//                                     create_session
//       - max_live_requests:          quota for session-bound turn
//                                     admission (live == non-terminal
//                                     requests admitted through THIS
//                                     controller)
//       - max_context_tokens_per_session: OPTIONAL per-session POLICY
//         cap (0 = disabled). Checked as the projected turn length
//         `length + input + max_new_tokens > cap` — the SAME projection
//         as the scheduler's model overflow check, but against the
//         policy cap. It can only RESTRICT below the model's
//         max_seq_len (it is clamped to max_seq_len in the constructor
//         and the scheduler's own overflow gate still applies below).
//     max_sessions / max_live_requests: -1 = unlimited, 0 = zero
//     capacity (reject EVERY admission), N>0 = capacity N. The context
//     cap uses a different convention: 0 = disabled.
//
//   * REJECT EARLY / FAIL LOUD / ZERO MUTATION: every serving-layer
//     rejection happens BEFORE the frozen runtime is touched:
//       - create_session rejected  -> NO SessionId consumed, no
//                                     sequence created, no slot/page
//                                     mutation;
//       - admit_turn rejected      -> NO RequestId consumed, no
//                                     sequence created, no KV page /
//                                     Delta slot / logical-length
//                                     mutation.
//     The frozen Phase C preflight (instance identity, sampling /
//     vocab / eos / token range, session live, BUSY session, model
//     context overflow) still runs below and is passed through
//     UNCHANGED — this layer does not re-implement it.
//
//   * LIFECYCLE (quota accounting):
//       - request terminal (Finished / Cancelled / Failed) -> live
//         request capacity released EXACTLY ONCE (the live count is
//         DERIVED from the tracked request ids + their scheduler
//         status — there is no counter that could double-decrement);
//       - destroy_session          -> session capacity released by the
//         SessionManager;
//       - reset_session            -> session STAYS live (capacity
//         unchanged).
//
// The controller is the POLICY BOUNDARY: in this phase, sessions,
// turns and driving happen THROUGH the controller. It is
// NON-OWNING (the scheduler and the session manager outlive it, the
// v0.5/v0.6/v0.8 discipline). It adds no model math, no generation
// logic and no sampling logic of its own.

#pragma once

#include <cstdint>
#include <vector>

#include "cudalm/scheduler.h"
#include "cudalm/session.h"
#include "cudalm/weight_format.h"  // Status

namespace cudalm {

// The serving admission limits. For max_sessions / max_live_requests:
// -1 = UNLIMITED, 0 = zero capacity (reject EVERY admission), N>0 = N.
// For max_context_tokens_per_session: 0 = disabled (no policy cap).
struct ServingLimits {
  int max_sessions = -1;  // live session quota (-1 unlimited / 0 none)
  int max_live_requests = -1;  // live request quota (-1 unlimited / 0 none)
  int max_context_tokens_per_session = 0;  // policy cap (0 = disabled);
                                           // clamped to the model's
                                           // max_seq_len (it can only
                                           // restrict, never exceed)
};

// Lightweight serving observability (deliberately small — not a
// telemetry system):
//   live_sessions        — sessions currently live in the manager
//                          (read-through; reset keeps it unchanged,
//                          destroy decrements it);
//   live_requests        — requests admitted through this controller
//                          that are NOT yet terminal (derived on read);
//   total_admitted_*     — cumulative successful admissions (never
//                          decremented);
//   rejected_*_limit     — cumulative SERVING-LAYER limit rejections
//                          (busy / overflow / unknown-session and the
//                          other frozen preflight rejections are NOT
//                          counted here — they are the runtime's own
//                          contract, passed through unchanged).
struct ServingStats {
  int live_sessions = 0;
  int live_requests = 0;
  std::uint64_t total_admitted_sessions = 0;
  std::uint64_t total_admitted_requests = 0;
  std::uint64_t rejected_session_limit = 0;
  std::uint64_t rejected_request_limit = 0;
  std::uint64_t rejected_context_limit = 0;
};

// The serving admission layer (v0.9 Phase A). Thin, non-owning,
// policy-only: see the file header for the pinned semantics.
class ServingController {
 public:
  // sched/sessions are NON-OWNING (they outlive the controller). The
  // context policy cap is clamped to the model's max_seq_len (it can
  // only restrict below the model limit, never exceed it).
  ServingController(Scheduler& scheduler, SessionManager& sessions,
                    ServingLimits limits);

  // ---- session admission (policy) --------------------------------------
  // live sessions < max_sessions -> SessionManager::create_session;
  // limit reached -> REJECT (no SessionId consumed, zero mutation,
  // rejected_session_limit++). The manager's own capacity contract
  // (Delta slots / pool OOM) still applies below.
  Status create_session(SessionId* out_id);

  // ---- turn admission (policy over the frozen Phase C preflight) -------
  // Checks, in order (each rejection is zero mutation):
  //   1. live request quota (max_live_requests) -> rejected_request_limit
  //   2. per-session context policy cap (0 = disabled) ->
  //      rejected_context_limit
  // then the FROZEN Scheduler::admit_session_turn (instance identity /
  // sampling / vocab / non-empty / max_new / eos / token range / session
  // live / BUSY / model overflow) is invoked and its Status is passed
  // through unchanged.
  Status admit_turn(SessionId session_id,
                    const std::vector<int>& new_input_tokens,
                    int max_new_tokens, int eos_token_id,
                    const SamplingConfig& sampling, RequestId* out_request_id);

  // ---- lifecycle (forwarding; no policy of its own) ---------------------
  Status reset_session(SessionId session_id);  // session STAYS live
  Status destroy_session(SessionId session_id);  // releases session quota
  Status cancel(RequestId request_id);  // releases request capacity
                                        // (synced immediately)

  // ---- driving (the frozen scheduler control plane) ---------------------
  // After each drive the controller syncs its quota bookkeeping:
  // terminal requests release their live-request capacity exactly once.
  Status step();
  Status run();

  // Quota sync: reap tracked requests that reached a terminal state
  // (idempotent; safe to call any number of times).
  void sync();

  // Observability: a snapshot (live_* derived on read — read-only).
  ServingStats stats() const;
  const ServingLimits& limits() const { return limits_; }

 private:
  // The live-request count DERIVED from the tracked ids + their
  // current scheduler status (no counter, no double-decrement path).
  int live_request_count() const;

  Scheduler& sched_;
  SessionManager& sessions_;
  ServingLimits limits_;
  std::vector<RequestId> tracked_;  // admitted, not yet reaped
  std::uint64_t total_admitted_sessions_ = 0;
  std::uint64_t total_admitted_requests_ = 0;
  std::uint64_t rejected_session_limit_ = 0;
  std::uint64_t rejected_request_limit_ = 0;
  std::uint64_t rejected_context_limit_ = 0;
};

}  // namespace cudalm
