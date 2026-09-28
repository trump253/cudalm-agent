// CUDALM — Qwen3.5 text-level multi-turn session facade implementation
// (v0.8 Phase D). See the header for the pinned text-session contract:
// this file is a STITCH of four frozen contracts (tokenizer encode /
// Phase C session-bound admission / the frozen scheduler run / tokenizer
// decode) and adds no model math, no generation loop and no sampling
// logic of its own.

#include "cudalm/session_text_generator.h"

namespace cudalm {

namespace {

const char* status_name(RequestStatus s) {
  switch (s) {
    case RequestStatus::Waiting: return "Waiting";
    case RequestStatus::Running: return "Running";
    case RequestStatus::Finished: return "Finished";
    case RequestStatus::Cancelled: return "Cancelled";
    case RequestStatus::Failed: return "Failed";
  }
  return "?";
}

const char* finish_reason_name(FinishReason r) {
  switch (r) {
    case FinishReason::None: return "None";
    case FinishReason::Eos: return "Eos";
    case FinishReason::MaxNewTokens: return "MaxNewTokens";
    case FinishReason::Cancelled: return "Cancelled";
    case FinishReason::Failed: return "Failed";
  }
  return "?";
}

}  // namespace

Qwen35SessionTextGenerator::Qwen35SessionTextGenerator(
    SequenceForwarder& fwd, const Qwen35Tokenizer& tokenizer,
    Qwen35StateManager& mgr, SessionManager& sessions, cudaStream_t stream)
    : fwd_(fwd),
      tokenizer_(tokenizer),
      mgr_(mgr),
      sessions_(sessions),
      stream_(stream),
      sched_(fwd, mgr, stream, &sessions) {}

Status Qwen35SessionTextGenerator::create_session(SessionId* out_id) {
  return sessions_.create_session(out_id);
}

Status Qwen35SessionTextGenerator::reset_session(SessionId session_id) {
  return sessions_.reset_session(session_id);
}

Status Qwen35SessionTextGenerator::destroy_session(SessionId session_id) {
  return sessions_.destroy_session(session_id);
}

SessionTextTurnResult Qwen35SessionTextGenerator::generate_turn(
    SessionId session_id, const std::string& new_text, int max_new_tokens,
    const SamplingConfig& sampling) {
  // The v0.4 text contract: the stop gate is the tokenizer's pinned real
  // EOS (248044 for the 0.8B checkpoint).
  return generate_turn(session_id, new_text, max_new_tokens, sampling,
                        static_cast<int>(tokenizer_.eos_token_id()));
}

SessionTextTurnResult Qwen35SessionTextGenerator::generate_turn(
    SessionId session_id, const std::string& new_text, int max_new_tokens,
    const SamplingConfig& sampling, int eos_token_id) {
  SessionTextTurnResult r;

  // (1) ENCODE — raw UTF-8 -> token ids, verbatim (no template, no
  // separator, no special tokens). Invalid UTF-8 fails loud BEFORE
  // anything is admitted: zero mutation.
  std::vector<std::uint32_t> uids;
  Status s = tokenizer_.encode(new_text, &uids);
  if (!s.ok) {
    r.error = "encode failed: " + s.message;
    return r;
  }
  std::vector<int> ids;
  ids.reserve(uids.size());
  for (std::uint32_t t : uids) ids.push_back(static_cast<int>(t));
  r.input_token_ids = ids;

  // (2) SESSION-BOUND ADMISSION — the Phase C contract: bound to the
  // session's EXISTING bound sequence (no sequence created, no history
  // replay; the input is appended from the current length). ZERO MUTATION
  // on any failure (unknown / busy session, overflow, invalid config).
  Status as = sched_.admit_session_turn(session_id, ids, max_new_tokens,
                                        eos_token_id, sampling,
                                        &r.request_id);
  if (!as.ok) {
    r.request_id = 0;
    r.error = as.message;
    r.context_length = 0;
    const Session* sess = sessions_.lookup(session_id);
    if (sess != nullptr) {
      const SequenceState* seq = mgr_.lookup(sess->sequence_id);
      if (seq != nullptr) r.context_length = seq->length;
    }
    return r;
  }

  // (3) RUN — the frozen scheduler control plane (commit-then-stop for
  // session-bound requests). run() returns the FIRST error it saw (any
  // request); this turn's own verdict is read from the request record.
  sched_.run();
  const Request* req = sched_.get(r.request_id);
  if (req == nullptr) {
    // Cannot happen: the request was just admitted.
    r.ok = false;
    r.error = "internal: admitted request vanished";
    return r;
  }
  r.forward_count = req->forward_count;
  r.stop_reason = req->finish_reason;

  // The session's logical length AFTER the turn (the committed boundary).
  const Session* sess = sessions_.lookup(session_id);
  const SequenceState* seq =
      sess != nullptr ? mgr_.lookup(sess->sequence_id) : nullptr;
  r.context_length = seq != nullptr ? seq->length : 0;

  // On a FAILED / CANCELLED turn: report the COMMITTED prefix only (a
  // pending sampled token is never reported as session history — the
  // Phase C contract) and leave the session LIVE at its boundary.
  if (req->status != RequestStatus::Finished) {
    r.ok = false;
    r.error = std::string("turn request ended ") +
              status_name(req->status) + " (" +
              finish_reason_name(req->finish_reason) + "); the session "
              "stays live at its committed boundary (length " +
              std::to_string(r.context_length) + ")";
    r.generated_token_ids.assign(
        req->generated.begin(),
        req->generated.begin() +
            static_cast<std::ptrdiff_t>(req->committed_generated));
    return r;
  }

  // (4) DECODE — finished turn: every generated id is COMMITTED (the
  // Phase B/C commit contract), so decode them all (skip_special_tokens
  // == false: a generated EOS stays in the text, exactly like v0.4).
  std::vector<std::uint32_t> gids;
  gids.reserve(req->generated.size());
  for (int t : req->generated) gids.push_back(static_cast<std::uint32_t>(t));
  Status ds = tokenizer_.decode(gids.data(), gids.size(), false,
                                &r.generated_text);
  if (!ds.ok) {
    r.ok = false;
    r.error = "decode failed: " + ds.message;
    return r;
  }
  r.generated_token_ids.assign(req->generated.begin(), req->generated.end());
  r.ok = true;
  return r;
}

}  // namespace cudalm
