// CUDALM — v0.6 Phase A: request scheduler / control plane (impl).
//
// Deterministic FIFO / round-robin over the frozen v0.5 runtime:
// one iteration advances each non-terminal request by at most ONE token
// via ONE single-sequence forward_token_with_state() on the single
// stream (NO batched CUDA compute in Phase A — see scheduler.h for the
// pinned execution model, admission/terminal/cancel contracts, and the
// "no off-by-one" v0.4 token progression).

#include "cudalm/scheduler.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "cudalm/cuda_check.h"

namespace cudalm {

namespace {

// Internal invariant violation (a logic bug, NOT a user error): print a
// diagnostic and abort (the project's fail-loud convention).
[[noreturn]] void scheduler_fatal(const char* what, RequestId request_id,
                                  const char* detail) {
  std::fprintf(stderr,
               "[cudalm][scheduler] INTERNAL ERROR (logic bug, not a user "
               "error):\n"
               "  what   : %s\n"
               "  request: %" PRIu64 "\n"
               "  detail : %s\n",
               what, request_id, detail);
  std::abort();
}

}  // namespace

// ---------------------------------------------------------------------------
// ModelForwarder (the real SequenceForwarder over a loaded Qwen35Model)
// ---------------------------------------------------------------------------

ModelForwarder::ModelForwarder(Qwen35Model& model) : model_(model) {}

Status ModelForwarder::forward_token(int token_id, SequenceId sequence_id,
                                     Qwen35StateManager& mgr,
                                     cudaStream_t stream) {
  // The frozen v0.5 external-state forward (config/stream compatibility
  // gate included — fail loud on a mismatch, zero mutation).
  return model_.forward_token_with_state(token_id, sequence_id, mgr, stream);
}

Status ModelForwarder::logits_to_host(std::vector<__nv_bfloat16>* out,
                                      cudaStream_t stream) const {
  const int vocab = model_.config().vocab_size;
  if (vocab <= 0) {
    return Status::error("scheduler: model not loaded (vocab_size <= 0)");
  }
  out->resize(static_cast<std::size_t>(vocab));
  // The full logits [vocab] bf16, stream-ordered D2H; the scheduler
  // samples host-side, so order the copy (same scratch discipline as the
  // v0.4 generator).
  CUDA_CHECK(cudaMemcpyAsync(
      out->data(), model_.logits(), static_cast<std::size_t>(vocab) *
                                       sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return Status::ok_status();
}

int ModelForwarder::vocab_size() const {
  return model_.config().vocab_size;
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

Scheduler::Scheduler(SequenceForwarder& fwd, Qwen35StateManager& mgr,
                     cudaStream_t stream)
    : fwd_(fwd), mgr_(mgr), stream_(stream) {}

Status Scheduler::admit(const Spec& spec, RequestId* out_request_id) {
  if (out_request_id == nullptr) {
    return Status::error("scheduler: admit: null out_request_id");
  }
  // ---- input validation (fail loud; NOTHING is registered on failure) --
  std::string serr;
  if (!validate_sampling_config(spec.sampling, &serr)) {
    return Status::error("scheduler: admit: invalid sampling config: " + serr);
  }
  const int vocab = fwd_.vocab_size();
  if (vocab <= 0) {
    return Status::error("scheduler: admit: forwarder has no vocab");
  }
  if (spec.prompt.empty()) {
    return Status::error("scheduler: admit: empty prompt");
  }
  if (spec.max_new_tokens < 1) {
    return Status::error("scheduler: admit: max_new_tokens must be >= 1");
  }
  if (spec.eos_token_id < -1 || spec.eos_token_id >= vocab) {
    return Status::error(
        "scheduler: admit: eos_token_id must be -1 or in [0, vocab_size)");
  }
  for (int t : spec.prompt) {
    if (t < 0 || t >= vocab) {
      return Status::error("scheduler: admit: invalid prompt token id " +
                           std::to_string(t));
    }
  }
  // Capacity gate: the sequence length after the last possible forward is
  // prompt.size() + (max_new_tokens - 1); it must fit max_seq_len.
  const Qwen35Config& cfg = mgr_.config();
  if (static_cast<int>(spec.prompt.size()) + spec.max_new_tokens - 1 >
      cfg.max_seq_len) {
    return Status::error(
        "scheduler: admit: prompt + max_new_tokens exceeds max_seq_len");
  }

  // ---- transactional core: create the sequence FIRST -------------------
  // Only on success is the request registered and the id issued — on any
  // failure there is no half-request, no leaked SequenceState, and no
  // consumed RequestId.
  SequenceId sid = 0;
  Status s = mgr_.create_sequence(&sid);
  if (!s.ok) {
    return Status::error("scheduler: admit: sequence creation failed: " +
                         s.message);
  }

  const RequestId id = next_id_++;  // monotonic, never reused (once, here)
  Request r(id, sid, std::move(spec.prompt), spec.max_new_tokens,
            spec.eos_token_id, spec.sampling);  // per-request Sampler seeded
  requests_.emplace(r.id, std::move(r));
  *out_request_id = r.id;
  return Status::ok_status();
}

void Scheduler::finish(Request& r, RequestStatus status, FinishReason reason) {
  r.status = status;
  r.finish_reason = reason;
  // Retire the sequence EXACTLY ONCE (on the terminal transition). The
  // sequence is live (created at admission, never retired before), so a
  // failure here is an internal invariant violation, not a user error.
  Status s = mgr_.retire_sequence(r.sequence_id);
  if (!s.ok) {
    scheduler_fatal("terminal transition: retire_sequence failed", r.id,
                    s.message.c_str());
  }
}

Status Scheduler::cancel(RequestId request_id) {
  auto it = requests_.find(request_id);
  if (it == requests_.end()) {
    // Fail loud: unknown ids are never silently accepted.
    return Status::error("scheduler: cancel: unknown request id " +
                         std::to_string(request_id));
  }
  Request& r = it->second;
  if (is_terminal(r.status)) {
    // Pinned contract: idempotent no-op for already-terminal requests.
    return Status::ok_status();
  }
  // Waiting or Running -> Cancelled + retire (resources reclaimed).
  finish(r, RequestStatus::Cancelled, FinishReason::Cancelled);
  return Status::ok_status();
}

Status Scheduler::advance_one(Request& r) {
  // ---- pick the token: next prompt token (prefill) or the last --------
  // ---- generated token (decode) — exactly one forward per advance -----
  const int prompt_len = static_cast<int>(r.prompt.size());
  const bool prefill = r.prefill_pos < prompt_len;
  int token;
  if (prefill) {
    token = r.prompt[static_cast<std::size_t>(r.prefill_pos)];
    r.prefill_pos++;  // the forward is about to cover this position
  } else {
    // Decode only starts after the first generated token exists (it was
    // sampled from the prefill-last forward); an empty `generated` here
    // is an internal invariant violation.
    if (r.generated.empty()) {
      r.status = RequestStatus::Failed;
      r.finish_reason = FinishReason::Failed;
      scheduler_fatal("decode advance with no generated token", r.id,
                      "state machine desync (logic bug)");
    }
    token = r.generated.back();
  }

  Status s = fwd_.forward_token(token, r.sequence_id, mgr_, stream_);
  if (!s.ok) {
    // Fatal Status: terminal exactly once (Failed) + retire exactly once.
    // The request is never advanced again.
    r.status = RequestStatus::Failed;
    r.finish_reason = FinishReason::Failed;
    finish(r, RequestStatus::Failed, FinishReason::Failed);
    return s;
  }
  r.forward_count++;
  if (r.status == RequestStatus::Waiting) {
    r.status = RequestStatus::Running;
  }

  // ---- sample the next token ONLY when it is not yet known -------------
  // v0.4 semantics (no off-by-one): NO sampling on early prefill
  // forwards; the LAST prompt forward's logits produce g0; each decode
  // forward's logits produce the next generated token.
  if (r.prefill_pos == prompt_len) {
    std::vector<__nv_bfloat16> logits;
    s = fwd_.logits_to_host(&logits, stream_);
    if (!s.ok) {
      r.status = RequestStatus::Failed;
      r.finish_reason = FinishReason::Failed;
      finish(r, RequestStatus::Failed, FinishReason::Failed);
      return s;
    }
    const int next = r.sampler.sample(logits.data(),
                                      static_cast<int>(logits.size()));
    if (next < 0 || next >= fwd_.vocab_size()) {
      // The sampler's contract is [0, vocab) for valid finite logits;
      // anything else is an internal invariant violation.
      r.status = RequestStatus::Failed;
      r.finish_reason = FinishReason::Failed;
      finish(r, RequestStatus::Failed, FinishReason::Failed);
      return Status::error("scheduler: sampler returned token id " +
                           std::to_string(next) +
                           " outside [0, vocab_size)");
    }
    r.generated.push_back(next);
    if (next == r.eos_token_id) {
      finish(r, RequestStatus::Finished, FinishReason::Eos);  // EOS IS kept
    } else if (static_cast<int>(r.generated.size()) >= r.max_new_tokens) {
      finish(r, RequestStatus::Finished, FinishReason::MaxNewTokens);
    }
  }
  return Status::ok_status();
}

Status Scheduler::step() {
  // ---- mutation safety (pinned hard requirement) ------------------------
  // Snapshot the runnable ids FIRST (ascending RequestId == admission
  // order), then re-look-up each id: no container iterator / pointer /
  // reference is held across any advance, and terminal requests are
  // naturally skipped.
  std::vector<RequestId> runnable;
  runnable.reserve(requests_.size());
  for (const auto& kv : requests_) {
    if (!is_terminal(kv.second.status)) {
      runnable.push_back(kv.first);
    }
  }
  std::sort(runnable.begin(), runnable.end());
  Status first_error;
  for (RequestId id : runnable) {
    auto it = requests_.find(id);
    if (it == requests_.end() || is_terminal(it->second.status)) {
      continue;
    }
    Status s = advance_one(it->second);
    if (!s.ok && first_error.ok) {
      first_error = s;  // continue with the remaining snapshot ids
    }
  }
  return first_error;  // ok when no advance failed
}

Status Scheduler::run() {
  // Deterministic bound: request i forwards at most
  // prompt_i.size() + max_new_tokens_i - 1 tokens, one per iteration it
  // is live in, so after the SUM of those bounds every request is
  // terminal. Exceeding it is an internal invariant violation.
  std::uint64_t bound = 0;
  for (const auto& kv : requests_) {
    const Request& r = kv.second;
    if (is_terminal(r.status)) {
      continue;
    }
    bound += static_cast<std::uint64_t>(r.prompt.size()) +
             static_cast<std::uint64_t>(std::max(1, r.max_new_tokens)) - 1;
  }
  for (std::uint64_t i = 0;; ++i) {
    bool any_live = false;
    for (const auto& kv : requests_) {
      if (!is_terminal(kv.second.status)) {
        any_live = true;
        break;
      }
    }
    if (!any_live) {
      return Status::ok_status();
    }
    if (i > bound) {
      scheduler_fatal("run() did not converge", 0,
                      "iteration bound exceeded (logic bug)");
    }
    Status s = step();
    if (!s.ok) {
      return s;
    }
  }
}

const Request* Scheduler::get(RequestId request_id) const {
  auto it = requests_.find(request_id);
  return it == requests_.end() ? nullptr : &it->second;
}

int Scheduler::num_requests() const {
  return static_cast<int>(requests_.size());
}

int Scheduler::num_live() const {
  int n = 0;
  for (const auto& kv : requests_) {
    if (!is_terminal(kv.second.status)) {
      n++;
    }
  }
  return n;
}

RequestId Scheduler::next_request_id() const { return next_id_; }

}  // namespace cudalm
