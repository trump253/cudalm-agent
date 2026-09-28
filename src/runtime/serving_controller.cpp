// CUDALM — v0.9 Phase A/B: serving controller implementation. Thin
// policy layer over the frozen v0.8 runtime: admission limits,
// zero-mutation rejection, quota lifecycle, lightweight stats, and
// (Phase B) pull-based committed-token streaming + cancellation +
// cooperative per-request deadlines. No model math, no generation
// logic, no sampling logic (see the header for the pinned semantics).

#include "cudalm/serving_controller.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#include "cudalm/cuda_check.h"  // CUDALM_PRECONDITION

namespace cudalm {

ServingController::ServingController(Scheduler& scheduler,
                                     SessionManager& sessions,
                                     ServingLimits limits,
                                     MonotonicClock* clock)
    : sched_(scheduler),
      sessions_(sessions),
      limits_(limits),
      clock_(clock) {
  // The context policy cap can only RESTRICT below the model limit,
  // never exceed it: clamp to the model's max_seq_len.
  const int max_seq = sessions_.manager().config().max_seq_len;
  if (limits_.max_context_tokens_per_session > 0 &&
      limits_.max_context_tokens_per_session > max_seq) {
    limits_.max_context_tokens_per_session = max_seq;
  }
}

Status ServingController::create_session(SessionId* out_id) {
  // -1 = unlimited, 0 = zero capacity (reject every create), N>0 = N.
  if (limits_.max_sessions >= 0 &&
      sessions_.num_sessions() >= limits_.max_sessions) {
    ++rejected_session_limit_;
    return Status::error("serving: session limit reached (" +
                         std::to_string(limits_.max_sessions) +
                         " live sessions) — rejected before creation "
                         "(zero mutation, no SessionId consumed)");
  }
  Status s = sessions_.create_session(out_id);
  if (!s.ok) return s;
  ++total_admitted_sessions_;
  return s;
}

Status ServingController::admit_turn(
    SessionId session_id, const std::vector<int>& new_input_tokens,
    int max_new_tokens, int eos_token_id, const SamplingConfig& sampling,
    RequestId* out_request_id,
    std::chrono::steady_clock::time_point deadline) {
  // (1) LIVE REQUEST QUOTA — before the frozen runtime is touched.
  // -1 = unlimited, 0 = zero capacity (reject every admission), N>0 = N.
  if (limits_.max_live_requests >= 0 &&
      live_request_count() >= limits_.max_live_requests) {
    ++rejected_request_limit_;
    return Status::error("serving: live request limit reached (" +
                         std::to_string(limits_.max_live_requests) +
                         " live requests) — rejected before admission "
                         "(zero mutation, no RequestId consumed)");
  }
  // (2) PER-SESSION CONTEXT POLICY CAP (0 = disabled). The same
  // projection as the scheduler's model overflow check, against the
  // policy cap. An unknown session skips the cap check and is rejected
  // by the frozen preflight below with its own message.
  if (limits_.max_context_tokens_per_session > 0) {
    int length = 0;
    if (sessions_.context_length_of(session_id, &length).ok) {
      const int n = static_cast<int>(new_input_tokens.size());
      if (length + n + max_new_tokens >
          limits_.max_context_tokens_per_session) {
        ++rejected_context_limit_;
        return Status::error(
            "serving: per-session context limit reached (" +
            std::to_string(length) + " + " + std::to_string(n) + " + " +
            std::to_string(max_new_tokens) + " > " +
            std::to_string(limits_.max_context_tokens_per_session) +
            ") — rejected before admission (zero mutation)");
      }
    }
  }
  // (3) FROZEN Phase C admission (instance identity / sampling / vocab /
  // non-empty / max_new / eos / token range / session live / BUSY /
  // model overflow) — invoked and passed through unchanged.
  Status s = sched_.admit_session_turn(session_id, new_input_tokens,
                                       max_new_tokens, eos_token_id,
                                       sampling, out_request_id);
  if (!s.ok) return s;
  ++total_admitted_requests_;
  tracked_.push_back(*out_request_id);
  // Phase B streaming bookkeeping: a FRESH emitted cursor (0) and the
  // per-request deadline (time_point::max() = no deadline).
  emitted_[*out_request_id] = 0;
  deadline_[*out_request_id] = deadline;
  return s;
}

Status ServingController::reset_session(SessionId session_id) {
  // Forwarding only: the session STAYS live (no quota change).
  return sessions_.reset_session(session_id);
}

Status ServingController::destroy_session(SessionId session_id) {
  // Forwarding: the manager releases the session's state; the session
  // quota is the manager's live count (read through in stats()).
  return sessions_.destroy_session(session_id);
}

Status ServingController::cancel(RequestId request_id) {
  // Forwarding to the frozen scheduler (Waiting/Running -> Cancelled;
  // already-terminal = idempotent no-op; unknown = error). The request
  // capacity is released IMMEDIATELY (the live count is derived from
  // the scheduler status). Deliberately NO sync here: the request's
  // streaming bookkeeping (emitted cursor) survives so its committed-
  // but-not-yet-emitted tokens can still be drained (poll / the next
  // step_stream); the pending token is never emitted.
  return sched_.cancel(request_id);
}

Status ServingController::step_drive_(std::vector<ServingEvent>* out) {
  // (1) DEADLINE CHECK — before the frozen step (the cooperative
  // boundary between scheduler steps): expired requests are cancelled
  // BEFORE their next forward, so no additional token is committed.
  check_deadlines_();
  // (2) THE FROZEN SCHEDULER STEP (batched decode etc. — untouched).
  Status s = sched_.step();
  // (3) DRAIN (optionally): every tracked request's newly committed
  // tokens (exactly once, in order) + terminal events (after all the
  // request's tokens). Failures surface as RequestTerminal(Failed).
  if (out != nullptr) {
    for (RequestId id : tracked_) {
      const Request* r = sched_.get(id);
      if (r != nullptr) drain_(r, out);
    }
  }
  // (4) QUOTA SYNC (reap terminal requests + their bookkeeping).
  sync();
  return s;
}

Status ServingController::step() {
  // One frozen step (deadline check first). Committed tokens stay
  // drainable via poll() / step_stream().
  return step_drive_(nullptr);
}

std::vector<ServingEvent> ServingController::step_stream() {
  std::vector<ServingEvent> out;
  (void)step_drive_(&out);  // failures ride in as RequestTerminal events
  return out;
}

namespace {
// The deterministic per-run bound (the same argument as
// Scheduler::run): a session turn needs at most input + max_new
// forwards; a live request advances at least one committed forward per
// iteration (or becomes terminal). Deadline cancels only shrink the
// live set.
std::uint64_t run_bound(const std::vector<RequestId>& tracked,
                        const Scheduler& sched) {
  std::uint64_t bound = 0;
  for (RequestId id : tracked) {
    const Request* r = sched.get(id);
    if (r == nullptr || is_terminal(r->status)) continue;
    bound += static_cast<std::uint64_t>(r->prompt.size()) +
             static_cast<std::uint64_t>(r->max_new_tokens);
  }
  return bound;
}
}  // namespace

Status ServingController::run() {
  const std::uint64_t bound = run_bound(tracked_, sched_);
  Status first_error;
  for (std::uint64_t i = 0;; ++i) {
    if (live_request_count() == 0) break;
    CUDALM_PRECONDITION(i <= bound,
                        "serving: run did not converge (logic bug)");
    Status s = step_drive_(nullptr);
    if (!s.ok && first_error.ok) first_error = s;  // record + CONTINUE
  }
  return first_error;
}

std::vector<ServingEvent> ServingController::run_stream() {
  const std::uint64_t bound = run_bound(tracked_, sched_);
  std::vector<ServingEvent> all;
  for (std::uint64_t i = 0;; ++i) {
    if (live_request_count() == 0) break;
    CUDALM_PRECONDITION(i <= bound,
                        "serving: run_stream did not converge (logic bug)");
    std::vector<ServingEvent> ev;
    (void)step_drive_(&ev);  // failures ride in as RequestTerminal events
    all.insert(all.end(), ev.begin(), ev.end());
  }
  return all;
}

Status ServingController::poll(RequestId request_id,
                               std::vector<ServingEvent>* out) {
  // WITHOUT driving: drain ONE tracked (not yet reaped) request.
  if (std::find(tracked_.begin(), tracked_.end(), request_id) ==
      tracked_.end()) {
    return Status::error("serving: poll: unknown or already-reaped request "
                         "id " + std::to_string(request_id));
  }
  const Request* r = sched_.get(request_id);
  CUDALM_PRECONDITION(r != nullptr,
                      "serving: poll: tracked id missing from the "
                      "scheduler (logic bug)");
  drain_(r, out);
  return Status::ok_status();
}

void ServingController::check_deadlines_() {
  if (deadline_.empty()) return;
  const std::chrono::steady_clock::time_point now =
      (clock_ != nullptr ? clock_ : &system_clock_)->now();
  for (RequestId id : tracked_) {
    const auto dit = deadline_.find(id);
    if (dit == deadline_.end()) continue;
    if (dit->second == std::chrono::steady_clock::time_point::max()) {
      continue;  // no deadline
    }
    const Request* r = sched_.get(id);
    if (r == nullptr || is_terminal(r->status)) continue;
    if (now >= dit->second) {
      // EXPIRED: cancel BEFORE the next forward — no additional token
      // commit, the session remains live at its last committed
      // boundary. The serving-layer termination reason is recorded
      // (the frozen FinishReason has no deadline reason: at the
      // scheduler level the request reports Cancelled/Cancelled).
      const Status s = sched_.cancel(id);
      if (s.ok) deadline_cancelled_.insert(id);
    }
  }
}

void ServingController::drain_(const Request* r,
                               std::vector<ServingEvent>* out) {
  // COMMIT-BEFORE-VISIBLE: only generated[0 .. committed_generated) is
  // ever emitted (the pending tail — sampled, not committed — is never
  // emitted). Exactly once (the cursor only advances), in order.
  const int committed = r->committed_generated;
  int emitted = 0;
  const auto eit = emitted_.find(r->id);
  if (eit != emitted_.end()) emitted = eit->second;
  for (int i = emitted; i < committed; ++i) {
    ServingEvent e;
    e.kind = ServingEventKind::Token;
    e.request_id = r->id;
    e.session_id = r->session_id;
    e.token_id = r->generated[static_cast<std::size_t>(i)];
    out->push_back(e);
  }
  if (emitted != committed) emitted_[r->id] = committed;
  // TERMINAL (if it is one): ONE RequestTerminal event, and only AFTER
  // all of this request's committed tokens (the loop above just drained
  // them — the final EOS / max_new token is committed before the frozen
  // scheduler reports terminal, so it is always emitted first).
  if (is_terminal(r->status) &&
      terminal_reported_.find(r->id) == terminal_reported_.end()) {
    ServingEvent e;
    e.kind = ServingEventKind::RequestTerminal;
    e.request_id = r->id;
    e.session_id = r->session_id;
    e.status = r->status;
    e.finish_reason = r->finish_reason;
    e.deadline_exceeded = deadline_cancelled_.find(r->id) !=
                          deadline_cancelled_.end();
    out->push_back(e);
    terminal_reported_.insert(r->id);
  }
}

void ServingController::sync() {
  // Reap tracked requests that reached a terminal state. Idempotent:
  // a request is removed once and can never be reaped twice (there is
  // no counter that could double-decrement — the live count is derived
  // from `tracked_` + the scheduler's status). Reaping also removes
  // the request's streaming / deadline bookkeeping.
  std::vector<RequestId> live;
  live.reserve(tracked_.size());
  for (RequestId id : tracked_) {
    const Request* r = sched_.get(id);
    if (r == nullptr || is_terminal(r->status)) {
      // reaped:
      emitted_.erase(id);
      terminal_reported_.erase(id);
      deadline_.erase(id);
      deadline_cancelled_.erase(id);
      continue;
    }
    live.push_back(id);
  }
  tracked_.swap(live);
}

int ServingController::live_request_count() const {
  int n = 0;
  for (RequestId id : tracked_) {
    const Request* r = sched_.get(id);
    if (r != nullptr && !is_terminal(r->status)) ++n;
  }
  return n;
}

ServingStats ServingController::stats() const {
  ServingStats s;
  s.live_sessions = sessions_.num_sessions();
  s.live_requests = live_request_count();
  s.total_admitted_sessions = total_admitted_sessions_;
  s.total_admitted_requests = total_admitted_requests_;
  s.rejected_session_limit = rejected_session_limit_;
  s.rejected_request_limit = rejected_request_limit_;
  s.rejected_context_limit = rejected_context_limit_;
  return s;
}

}  // namespace cudalm
