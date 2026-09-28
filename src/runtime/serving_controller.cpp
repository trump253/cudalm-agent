// CUDALM — v0.9 Phase A: serving controller implementation. Thin
// policy layer over the frozen v0.8 runtime: admission limits,
// zero-mutation rejection, quota lifecycle, lightweight stats. No model
// math, no generation logic, no sampling logic (see the header for the
// pinned semantics).

#include "cudalm/serving_controller.h"

namespace cudalm {

ServingController::ServingController(Scheduler& scheduler,
                                     SessionManager& sessions,
                                     ServingLimits limits)
    : sched_(scheduler), sessions_(sessions), limits_(limits) {
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
    RequestId* out_request_id) {
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
  Status s = sched_.cancel(request_id);
  sync();  // a cancelled request releases its capacity immediately
  return s;
}

Status ServingController::step() {
  Status s = sched_.step();
  sync();
  return s;
}

Status ServingController::run() {
  Status s = sched_.run();
  sync();
  return s;
}

void ServingController::sync() {
  // Reap tracked requests that reached a terminal state. Idempotent:
  // a request is removed once and can never be reaped twice (there is
  // no counter that could double-decrement — the live count is derived
  // from `tracked_` + the scheduler's status).
  std::vector<RequestId> live;
  live.reserve(tracked_.size());
  for (RequestId id : tracked_) {
    const Request* r = sched_.get(id);
    if (r == nullptr || is_terminal(r->status)) continue;  // reaped
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
