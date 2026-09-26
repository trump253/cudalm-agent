// CUDALM — v0.6 Phase A/B: request scheduler / control plane (impl).
//
// Deterministic FIFO / round-robin over the frozen v0.5 runtime:
// one iteration advances each non-terminal request by at most ONE token
// on the single stream. Phase A: every advance is ONE single-sequence
// forward_token_with_state(). Phase B (additive): consecutive
// decode-ready requests in the snapshot (a "decode cohort") are advanced
// in ONE TRUE BATCHED forward (forward_batch_with_state — one model
// traversal for the whole cohort) when the forwarder supports it; a
// size-1 cohort or a zero-mutation preflight failure keeps/falls back to
// the frozen serial path. See scheduler.h for the pinned execution model,
// admission/terminal/cancel contracts, the "no off-by-one" v0.4 token
// progression, and the Phase B cohort/fallback rules.

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

Status ModelForwarder::forward_batch(const int* token_ids,
                                     const SequenceId* sids, int B,
                                     Qwen35StateManager& mgr,
                                     cudaStream_t stream) {
  // The v0.6 Phase B TRUE BATCHED external-state forward (zero-mutation
  // preflight; on failure NOTHING changed — the scheduler then falls back
  // to the frozen serial path for the cohort).
  return model_.forward_batch_with_state(token_ids, sids, B, mgr, stream);
}

Status ModelForwarder::logits_batch_to_host(std::vector<__nv_bfloat16>* out,
                                            int B,
                                            cudaStream_t stream) const {
  const int vocab = model_.config().vocab_size;
  if (vocab <= 0) {
    return Status::error("scheduler: model not loaded (vocab_size <= 0)");
  }
  out->resize(static_cast<std::size_t>(B) * static_cast<std::size_t>(vocab));
  // ONE device-to-host copy of the whole [B][vocab] batch (stream-ordered;
  // the scheduler samples each row host-side with its own Sampler).
  CUDA_CHECK(cudaMemcpyAsync(
      out->data(), model_.batch_logits(),
      static_cast<std::size_t>(B) * static_cast<std::size_t>(vocab) *
          sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return Status::ok_status();
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
    // Do NOT advance prefill_pos yet: it counts SUCCESSFULLY forwarded
    // prompt tokens, so the increment happens only after the forward
    // below succeeds (a failed forward leaves prefill_pos unchanged).
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

  // Phase B instrumentation: every call here is ONE single forward.
  ++single_forward_calls_;
  Status s = fwd_.forward_token(token, r.sequence_id, mgr_, stream_);
  if (!s.ok) {
    // Fatal Status: terminal exactly once (Failed) + retire exactly once.
    // The request is never advanced again. The failed forward is NOT
    // committed: prefill_pos (prefill) and forward_count are unchanged
    // (e.g. prompt {50, 99} with 50 ok / 99 failing -> forward_count = 1,
    // prefill_pos = 1, NOT 2).
    r.status = RequestStatus::Failed;
    r.finish_reason = FinishReason::Failed;
    finish(r, RequestStatus::Failed, FinishReason::Failed);
    return s;
  }
  if (prefill) {
    r.prefill_pos++;  // the forward succeeded: commit the progress
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
  std::size_t i = 0;
  while (i < runnable.size()) {
    auto it = requests_.find(runnable[i]);
    if (it == requests_.end() || is_terminal(it->second.status)) {
      ++i;
      continue;
    }
    // Collect the MAXIMAL consecutive run of decode-ready requests
    // starting at i (snapshot order; a prefill request or a terminal id
    // ends the run). The run is advanced as one cohort: batched (size
    // >= 2 + capable forwarder) or serial (size 1 / no batch support —
    // the frozen Phase A path).
    std::size_t j = i;
    while (j < runnable.size()) {
      auto jt = requests_.find(runnable[j]);
      if (jt == requests_.end() || is_terminal(jt->second.status)) {
        break;
      }
      if (!is_decode_ready(jt->second)) {
        break;
      }
      ++j;
    }
    if (j - i < 2 || !fwd_.supports_batch()) {
      // Single path (Phase A semantics unchanged) — including a size-1
      // decode cohort (a batch of 1 keeps the single path).
      Status s = advance_one(it->second);
      if (!s.ok && first_error.ok) {
        first_error = s;  // continue with the remaining snapshot ids
      }
      ++i;
    } else {
      advance_batch_run(runnable, i, j, &first_error);
      i = j;
    }
  }
  return first_error;  // ok when no advance failed
}

bool Scheduler::is_decode_ready(const Request& r) {
  // Prefill complete (all prompt tokens forwarded) AND at least one
  // generated token exists (g0 came from the last-prompt forward): the
  // next forward is a DECODE of the last generated token.
  return r.prefill_pos == static_cast<int>(r.prompt.size()) &&
         !r.generated.empty();
}

void Scheduler::advance_batch_run(const std::vector<RequestId>& runnable,
                                  std::size_t i, std::size_t j,
                                  Status* first_error) {
  const std::size_t B = j - i;
  std::vector<int> tokens(B);
  std::vector<SequenceId> sids(B);
  for (std::size_t k = 0; k < B; ++k) {
    const Request& r = requests_.at(runnable[i + k]);
    CUDALM_PRECONDITION(
        !r.generated.empty(),
        "advance_batch_run: cohort row is not decode-ready (logic bug)");
    tokens[k] = r.generated.back();  // the decode token: last generated
    sids[k] = r.sequence_id;
  }

  Status s =
      fwd_.forward_batch(tokens.data(), sids.data(), static_cast<int>(B),
                         mgr_, stream_);
  if (!s.ok) {
    // The batch preflight is ZERO-MUTATION (no Delta/KV change, no length
    // change), so the fallback is exactly the frozen Phase A serial path:
    // each cohort row advances in snapshot order, a per-row failure marks
    // that row Failed + retires, the others continue, the first error is
    // recorded.
    ++batch_fallback_calls_;
    for (std::size_t k = i; k < j; ++k) {
      auto it = requests_.find(runnable[k]);
      if (it == requests_.end() || is_terminal(it->second.status)) {
        continue;
      }
      Status fs = advance_one(it->second);
      if (!fs.ok && first_error->ok) {
        *first_error = fs;
      }
    }
    return;
  }

  // The batch committed: ONE batch forward for the whole cohort (never B
  // single forwards).
  ++batch_forward_calls_;
  if (static_cast<int>(B) > max_batch_size_) {
    max_batch_size_ = static_cast<int>(B);
  }

  // ONE D2H of the whole [B][vocab] logits, then per-row sampling with
  // each request's OWN Sampler (v0.4 per-request RNG isolation — the
  // sampling call order is the cohort order, as in Phase A).
  std::vector<__nv_bfloat16> logits;
  Status ls = fwd_.logits_batch_to_host(&logits, static_cast<int>(B), stream_);
  if (!ls.ok) {
    // A logits D2H failure is fatal for the whole cohort (Phase A
    // semantics for a logits_to_host failure: mark Failed + retire).
    for (std::size_t k = i; k < j; ++k) {
      auto it = requests_.find(runnable[k]);
      if (it == requests_.end() || is_terminal(it->second.status)) {
        continue;
      }
      Request& r = it->second;
      r.status = RequestStatus::Failed;
      r.finish_reason = FinishReason::Failed;
      finish(r, RequestStatus::Failed, FinishReason::Failed);
    }
    if (first_error->ok) {
      *first_error = ls;
    }
    return;
  }

  const int V = fwd_.vocab_size();
  for (std::size_t k = 0; k < B; ++k) {
    auto it = requests_.find(runnable[i + k]);
    if (it == requests_.end() || is_terminal(it->second.status)) {
      continue;
    }
    Request& r = it->second;
    r.forward_count++;
    if (r.status == RequestStatus::Waiting) {
      r.status = RequestStatus::Running;
    }
    const __nv_bfloat16* row =
        logits.data() +
        static_cast<std::size_t>(k) * static_cast<std::size_t>(V);
    const int next = r.sampler.sample(row, V);
    if (next < 0 || next >= V) {
      // The sampler's contract is [0, vocab) for valid finite logits;
      // anything else is an internal invariant violation.
      r.status = RequestStatus::Failed;
      r.finish_reason = FinishReason::Failed;
      finish(r, RequestStatus::Failed, FinishReason::Failed);
      if (first_error->ok) {
        *first_error = Status::error("scheduler: sampler returned token id " +
                                     std::to_string(next) +
                                     " outside [0, vocab_size)");
      }
      continue;
    }
    r.generated.push_back(next);
    if (next == r.eos_token_id) {
      finish(r, RequestStatus::Finished, FinishReason::Eos);  // EOS IS kept
    } else if (static_cast<int>(r.generated.size()) >= r.max_new_tokens) {
      finish(r, RequestStatus::Finished, FinishReason::MaxNewTokens);
    }
  }
}

Status Scheduler::run() {
  // Failure isolation (pinned contract): run until EVERY request is
  // terminal. A failing step does NOT stop the other live requests: the
  // failed request is terminal (Failed + retired) and is never advanced
  // again, while the remaining requests keep advancing in the following
  // iterations; run() returns the FIRST error encountered (or ok).
  // Termination: each live request either succeeds one forward per
  // iteration (bounded by prompt.size() + max_new_tokens - 1) or becomes
  // terminal on a failed forward — so the loop always ends.
  // Deterministic bound: the SUM of the per-request bounds.
  std::uint64_t bound = 0;
  for (const auto& kv : requests_) {
    const Request& r = kv.second;
    if (is_terminal(r.status)) {
      continue;
    }
    bound += static_cast<std::uint64_t>(r.prompt.size()) +
             static_cast<std::uint64_t>(std::max(1, r.max_new_tokens)) - 1;
  }
  Status first_error;
  for (std::uint64_t i = 0;; ++i) {
    bool any_live = false;
    for (const auto& kv : requests_) {
      if (!is_terminal(kv.second.status)) {
        any_live = true;
        break;
      }
    }
    if (!any_live) {
      return first_error;  // ok when nothing failed, else the first error
    }
    if (i > bound) {
      scheduler_fatal("run() did not converge", 0,
                      "iteration bound exceeded (logic bug)");
    }
    Status s = step();
    if (!s.ok && first_error.ok) {
      first_error = s;  // record + CONTINUE (the other requests proceed)
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
