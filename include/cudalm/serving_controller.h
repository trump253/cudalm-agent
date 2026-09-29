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

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include "cudalm/scheduler.h"
#include "cudalm/session.h"
#include "cudalm/weight_format.h"  // Status

namespace cudalm {

// ---- v0.9 Phase B: committed-token streaming + cancellation/deadline ----

// A MONOTONIC clock (the deadline checks use it exclusively). Inject a
// FAKE clock in CPU tests (deterministic time); nullptr in the
// controller constructor = the real system monotonic clock.
class MonotonicClock {
 public:
  virtual ~MonotonicClock() = default;
  virtual std::chrono::steady_clock::time_point now() const = 0;
};

// The real clock (std::chrono::steady_clock).
class SystemMonotonicClock final : public MonotonicClock {
 public:
  std::chrono::steady_clock::time_point now() const override {
    return std::chrono::steady_clock::now();
  }
};

// One serving event (the output of step_stream / run_stream / poll).
enum class ServingEventKind {
  Token,            // one COMMITTED token id (exactly once, in order)
  RequestTerminal,  // the request reached a terminal state — emitted
                    // AFTER all of its committed tokens
};

struct ServingEvent {
  ServingEventKind kind = ServingEventKind::Token;
  RequestId request_id = 0;
  SessionId session_id = 0;  // the bound session
  int token_id = 0;  // Token events only
  // RequestTerminal events only:
  RequestStatus status = RequestStatus::Waiting;
  FinishReason finish_reason = FinishReason::None;
  // The SERVING-LAYER termination reason: the frozen FinishReason has
  // no deadline reason, so a deadline-cancelled request reports
  // (Cancelled, Cancelled) at the scheduler level AND
  // deadline_exceeded = true here.
  bool deadline_exceeded = false;
};

// ---- v0.9 Phase C: session TTL / LRU eviction -------------------------

// The session eviction policy. The DEFAULT CONSTRUCTION KEEPS THE
// PHASE A/B BEHAVIOR EXACTLY (both features disabled; max_sessions
// reached still REJECTS — no silent behavior change):
//
//   * idle_ttl: std::nullopt = TTL DISABLED (the unambiguous
//     disabled sentinel — there is no 0/-1 ambiguity). TTL == 0 =
//     immediately eligible once TRULY idle; TTL > 0 = the normal
//     idle timeout. No background thread / timer: the sweep happens
//     only at explicit MAINTENANCE POINTS — evict_expired_sessions()
//     and (only when a TTL is enabled) the lightweight sweep before
//     create_session().
//   * lru_on_session_pressure: when TRUE, create_session() at
//     live_sessions == max_sessions (> 0) first tries to evict ONE
//     eligible idle LRU session (oldest last activity, ties broken by
//     the smaller SessionId). max_sessions == 0 can NEVER be bypassed
//     by eviction (the zero-capacity contract). No eligible candidate
//     -> the Phase A rejection (no SessionId consumed, zero mutation).
//
// EVICTION == DESTROY THE WHOLE SESSION (SessionManager::
// destroy_session: the SessionId is invalidated forever — never
// reused; the bound sequence is retired; KV pages + the Delta slot
// are released and zeroed; the logical context is gone). This is NOT
// context truncation: the v0.8 context-overflow contract (REJECT) is
// untouched, and no token / KV / Delta state is ever truncated or
// partially dropped.
struct SessionEvictionPolicy {
  std::optional<std::chrono::steady_clock::duration> idle_ttl;  // nullopt
                                                               // = disabled
  bool lru_on_session_pressure = false;

  SessionEvictionPolicy with_idle_ttl(
      std::chrono::steady_clock::duration ttl) const {
    SessionEvictionPolicy p = *this;
    p.idle_ttl = ttl;
    return p;
  }
  SessionEvictionPolicy with_lru_on_session_pressure(bool on = true) const {
    SessionEvictionPolicy p = *this;
    p.lru_on_session_pressure = on;
    return p;
  }
};

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
  // v0.9 Phase C (AUTOMATIC evictions only — a manual
  // destroy_session() is never counted here):
  std::uint64_t evicted_sessions_ttl = 0;
  std::uint64_t evicted_sessions_lru = 0;
  std::uint64_t eviction_no_candidate = 0;  // a LRU-pressure attempt
                                            // with NO eligible session
};

// The serving layer (v0.9 Phase A admission + Phase B committed-token
// streaming / cancellation / deadline). Thin, non-owning: see the file
// header for the pinned semantics.
class ServingController {
 public:
  // sched/sessions are NON-OWNING (they outlive the controller). The
  // context policy cap is clamped to the model's max_seq_len (it can
  // only restrict below the model limit, never exceed it). `clock` is
  // NON-OWNING and optional (nullptr = the real system monotonic
  // clock); it is used ONLY for the per-request deadline checks.
  // `policy` (v0.9 Phase C) defaults to the DISABLED policy — the
  // Phase A/B behavior is exactly preserved unless eviction is
  // explicitly enabled.
  ServingController(Scheduler& scheduler, SessionManager& sessions,
                    ServingLimits limits,
                    MonotonicClock* clock = nullptr,
                    SessionEvictionPolicy policy = {});

  // ---- session admission (policy) --------------------------------------
  // live sessions < max_sessions -> SessionManager::create_session;
  // limit reached -> REJECT (no SessionId consumed, zero mutation,
  // rejected_session_limit++). The manager's own capacity contract
  // (Delta slots / pool OOM) still applies below.
  //
  // v0.9 Phase C: with a TTL enabled, a lightweight sweep runs FIRST
  // (expired idle sessions are evicted — they no longer count
  // against the limit). With lru_on_session_pressure ENABLED and
  // live_sessions == max_sessions (exactly at the limit, > 0), ONE
  // eligible idle LRU session is evicted before the create is
  // retried; with NO eligible candidate the Phase A rejection
  // stands. When ALREADY OVER the limit (live_sessions >
  // max_sessions) there is NO eviction — straight to the Phase A
  // session-limit rejection (no mutation). The zero-capacity
  // contract (max_sessions == 0) can never be bypassed by
  // eviction. With the DEFAULT policy this method is EXACTLY the
  // Phase A behavior.
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
  // `deadline` (monotonic) is the per-request deadline (v0.9 Phase B):
  // the DEFAULT `time_point::max()` = NO deadline. Checked BEFORE every
  // scheduler step (see step_stream / run_stream / step / run): once
  // expired, the request is CANCELLED before its next forward (no
  // additional token commit; the session stays live). A deadline is a
  // COOPERATIVE BOUNDARY BETWEEN SCHEDULER STEPS — it never preempts
  // in-flight CUDA work.
  Status admit_turn(SessionId session_id,
                    const std::vector<int>& new_input_tokens,
                    int max_new_tokens, int eos_token_id,
                    const SamplingConfig& sampling, RequestId* out_request_id,
                    std::chrono::steady_clock::time_point deadline =
                        std::chrono::steady_clock::time_point::max());

  // ---- lifecycle (forwarding; no policy of its own) ---------------------
  Status reset_session(SessionId session_id);  // session STAYS live
  Status destroy_session(SessionId session_id);  // releases session quota
  // Cancels a live request (Waiting/Running -> Cancelled; the
  // already-terminal idempotent contract is the frozen scheduler's).
  // The request capacity is released IMMEDIATELY (the live count is
  // derived); the session STAYS live at its last committed boundary.
  // Note (v0.9 Phase B): cancel does NOT reap the request's streaming
  // bookkeeping — its COMMITTED-but-not-yet-emitted tokens + the
  // Cancelled terminal event can still be drained (poll / the next
  // step_stream / run_stream — the cancel enters the same terminal-
  // pending lifecycle as a step-terminated request); the PENDING token
  // (sampled, not committed) is never emitted.
  Status cancel(RequestId request_id);

  // ---- driving (the frozen scheduler control plane) ---------------------
  // Every drive runs the per-request DEADLINE CHECK first (cancel the
  // expired requests BEFORE their next forward), then drives the frozen
  // scheduler, then syncs the quota bookkeeping (terminal requests
  // release their live-request capacity exactly once).
  Status step();
  // Drive to quiescence (every tracked request terminal) via steps —
  // the same per-iteration behavior as Scheduler::run, with the
  // deadline check before each step. NON-STREAMING: any pending
  // terminal events encountered along the way are DISCARDED (drained
  // and reaped — not retrievable afterwards; use the *_stream / poll
  // API for events).
  Status run();

  // ---- committed-token streaming (v0.9 Phase B) -------------------------
  // PULL-BASED: the controller never pushes; a token becomes
  // stream-visible only when a drive / poll DRAINS it.
  //
  // COMMIT-BEFORE-VISIBLE (the hard constraint): a token is emitted
  // only after a SUCCESSFUL forward committed it into the session
  // state — i.e. only `generated[0 .. committed_generated)` is ever
  // emitted; the pending tail (sampled, not committed) is NEVER
  // emitted (including on failure and on cancel). Each committed
  // token is emitted EXACTLY ONCE and IN ORDER (a per-request
  // serving-layer emitted cursor); the final EOS / max_new token is
  // committed (and therefore emitted) BEFORE the terminal report.
  //
  // Events per request are contiguous and in order; a request's
  // RequestTerminal event follows ALL of its Token events. One
  // Scheduler::step() may commit tokens for SEVERAL requests; the
  // drain emits each request's own events separately (no cross-
  // request contamination).
  //
  // Drive one scheduler step (deadline check first), then drain:
  // returns the Token + RequestTerminal events of this step (possibly
  // empty). A failed request surfaces as a RequestTerminal event with
  // status Failed.
  std::vector<ServingEvent> step_stream();
  // Drive to quiescence (deadline check before each step), draining
  // after every step; returns ALL events of the whole drive in order.
  // A TERMINAL-but-not-yet-drained request already present in the
  // controller does not lose its pending events (they are drained
  // first / in order) — even if live_requests is already 0.
  std::vector<ServingEvent> run_stream();
  // WITHOUT driving: drain ONE tracked request's not-yet-emitted
  // committed tokens (and its terminal event, if it is terminal).
  // Pinned exactly-once semantics: a request's events are emitted at
  // most ONCE across poll / step_stream / run_stream combined; a poll
  // of an unknown or ALREADY-FULLY-DRAINED request id ERRORS (the
  // fully-drained reap removes it from the tracked set).
  Status poll(RequestId request_id, std::vector<ServingEvent>* out);

  // ---- session eviction (v0.9 Phase C) ---------------------------------
  // EVICTION == DESTROY THE WHOLE SESSION via the frozen
  // SessionManager::destroy_session (SessionId invalidated forever —
  // never reused; sequence retired; KV pages + Delta slot released
  // and zeroed; logical context gone). Never context truncation.
  //
  // A session is ELIGIBLE (idle + safe) only when ALL of: it is
  // MANAGED by this controller (created through it — manager sessions
  // created outside the controller are UNMANAGED: never guessed,
  // never auto-evicted), it is still live, Scheduler::session_busy
  // is FALSE, and it has NO terminal-but-undrained streaming
  // bookkeeping (the Phase B lifecycle: a session whose stream
  // events are not yet consumed is PROTECTED).
  //
  // Explicit TTL sweep (a MAINTENANCE POINT — no background thread
  // or timer exists in this phase): evict every eligible session
  // whose idle age (now - last activity) is >= the configured TTL
  // (TTL == 0: immediately eligible once truly idle).
  // THREE-STATE (pinned): ok + *out_evicted = the evicted SessionIds
  // (deterministic: ascending; empty when nothing qualified);
  // ERROR = a real underlying destroy failure — FAIL LOUD: the
  // original Status is propagated, the failed session is left
  // EXACTLY as it was (the transactional invariant), and the sweep
  // stops. A no-candidate sweep is NOT an error (ok + empty).
  Status evict_expired_sessions(std::vector<SessionId>* out_evicted);
  // Explicit LRU eviction: evict the eligible idle session with the
  // OLDEST last activity (ties: the smaller SessionId first — the
  // pinned deterministic order). THREE-STATE (pinned): ok +
  // *out_evicted == true = one evicted; ok + false = NO eligible
  // candidate (NOT an error — the caller's no-candidate path);
  // ERROR = a real destroy failure (fail loud, the candidate is left
  // untouched, the caller must NOT count it as no-candidate).
  Status evict_one_lru_idle(bool* out_evicted);
  // TEST SEAM (Phase C review fix — not part of the production
  // contract): a swappable destroy hook. When EMPTY (production),
  // the frozen SessionManager::destroy_session is used. Tests install
  // it to inject a controlled destroy failure and verify the fail-
  // loud Status propagation (a real retire_sequence failure cannot be
  // triggered non-invasively — the frozen manager is not modified).
  std::function<Status(SessionId)> destroy_for_test;
  // A session is managed + live + not busy + not
  // terminal-but-undrained.
  bool is_eviction_eligible(SessionId session_id) const;

  // Quota sync (idempotent; safe to call any number of times).
  // REVIEW FIX (streaming lifecycle): a TERMINAL request's stream
  // state is NOT destroyed here — its committed-but-not-yet-emitted
  // tokens + terminal event stay drainable until they are consumed
  // exactly once (the reap happens in the fully-drained path). The
  // live/quota lifecycle is unaffected (the live count is derived).
  void sync();

  // Observability: a snapshot (live_* derived on read — read-only).
  ServingStats stats() const;
  const ServingLimits& limits() const { return limits_; }

 private:
  // The live-request count DERIVED from the tracked ids + their
  // current scheduler status (no counter, no double-decrement path).
  int live_request_count() const;
  // The pre-step deadline check: cancel (and mark deadline_exceeded)
  // every tracked live request whose deadline has expired.
  void check_deadlines_();
  // Drain ONE request: emit its not-yet-emitted COMMITTED tokens
  // (exactly once, in order) and, if it is terminal, its ONE
  // RequestTerminal event (after all its tokens).
  void drain_(const Request* r, std::vector<ServingEvent>* out);
  // One controller step: deadline check -> frozen step -> (optionally)
  // drain ALL tracked requests (admission order) -> sync. Returns the
  // frozen step's Status (the first error; failures also surface as
  // RequestTerminal(Failed) events when draining).
  Status step_drive_(std::vector<ServingEvent>* out);
  // Phase C: refresh the session's last-activity timestamp (only for
  // MANAGED sessions).
  void refresh_activity_(SessionId session_id);
  // Phase C: the LRU order over the MANAGED live sessions (oldest
  // last activity first; ties: smaller SessionId first).
  std::vector<SessionId> lru_order_() const;
  // Phase C: destroy ONE session transactionally (the metadata +
  // counters update only after the frozen destroy SUCCEEDS).
  Status evict_session_(SessionId session_id, bool by_ttl);

  Scheduler& sched_;
  SessionManager& sessions_;
  ServingLimits limits_;
  MonotonicClock* clock_ = nullptr;  // nullptr = use system_clock_
  SystemMonotonicClock system_clock_;
  std::vector<RequestId> tracked_;  // admitted, not yet reaped
  // ---- v0.9 Phase B streaming / deadline bookkeeping --------------------
  std::map<RequestId, int> emitted_;  // per-request emitted cursor
  std::set<RequestId> terminal_reported_;  // terminal event sent once
  std::set<RequestId> terminal_pending_;  // terminal, stream state not yet
                                          // fully drained (kept drainable;
                                          // reaped in drain_ once fully
                                          // drained — the live/quota
                                          // lifecycle is unaffected)
  std::map<RequestId, std::chrono::steady_clock::time_point> deadline_;
  std::set<RequestId> deadline_cancelled_;  // cancelled BY the deadline
  // ---- v0.9 Phase C eviction bookkeeping --------------------------------
  SessionEvictionPolicy policy_;
  // Last-activity timestamp per MANAGED session (created through
  // this controller). Unmanaged manager sessions have NO entry: they
  // are never guessed, never auto-evicted. (A terminal-but-undrained
  // session's protection is DERIVED: a tracked request id that is
  // terminal is by construction not yet fully drained — the reap
  // happens in the fully-drained path — so the eligibility check
  // scans tracked_ directly.)
  std::map<SessionId, std::chrono::steady_clock::time_point> activity_;
  std::uint64_t evicted_sessions_ttl_ = 0;
  std::uint64_t evicted_sessions_lru_ = 0;
  std::uint64_t eviction_no_candidate_ = 0;
  std::uint64_t total_admitted_sessions_ = 0;
  std::uint64_t total_admitted_requests_ = 0;
  std::uint64_t rejected_session_limit_ = 0;
  std::uint64_t rejected_request_limit_ = 0;
  std::uint64_t rejected_context_limit_ = 0;
};

}  // namespace cudalm
