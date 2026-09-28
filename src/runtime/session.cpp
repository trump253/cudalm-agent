// CUDALM — v0.8 Phase A: session control plane (impl).
//
// A thin binding over the frozen v0.5 Qwen35StateManager: one session ==
// one bound sequence. All DEVICE state (KV pages, the Delta slot) and all
// zero-on-release / zero-on-reset semantics live in the pools and the
// state manager, UNCHANGED — this layer adds only the SessionId ->
// SequenceId binding and the session lifecycle. See session.h for the
// pinned contract.

#include "cudalm/session.h"

namespace cudalm {

Status SessionManager::not_live(SessionId id) const {
  return Status::error("session manager: session " + std::to_string(id) +
                       " is not live (never created or destroyed)");
}

Status SessionManager::create_session(SessionId* out_id) {
  if (out_id == nullptr)
    return Status::error("session manager: create_session: null output");
  // Sequence first: on OOM nothing is registered (no id issued, no
  // half-record, no leaked slot) — the transactional create contract
  // (mirrors Qwen35StateManager::create_sequence, which acquires the
  // Delta slot before registering its record).
  SequenceId sid = 0;
  Status s = mgr_.create_sequence(&sid);
  if (!s.ok) return s;
  Session rec;
  rec.id = next_id_++;
  rec.sequence_id = sid;
  rec.state = SessionState::Active;
  rec.reset_count = 0;
  sessions_.emplace(rec.id, std::move(rec));
  *out_id = rec.id;
  return Status::ok_status();
}

const Session* SessionManager::lookup(SessionId id) const {
  const auto it = sessions_.find(id);
  return it == sessions_.end() ? nullptr : &it->second;
}

Status SessionManager::sequence_id_of(SessionId id, SequenceId* out) const {
  if (out == nullptr)
    return Status::error("session manager: sequence_id_of: null output");
  const Session* rec = lookup(id);
  if (rec == nullptr) return not_live(id);
  *out = rec->sequence_id;
  return Status::ok_status();
}

Status SessionManager::context_length_of(SessionId id, int* out) const {
  if (out == nullptr)
    return Status::error("session manager: context_length_of: null output");
  SequenceId sid = 0;
  Status s = sequence_id_of(id, &sid);
  if (!s.ok) return s;
  const SequenceState* seq = mgr_.lookup(sid);
  if (seq == nullptr)
    return Status::error("session manager: bound sequence " +
                         std::to_string(sid) +
                         " not live (internal invariant violation)");
  *out = seq->length;
  return Status::ok_status();
}

bool SessionManager::fits(SessionId id, int n_tokens) const {
  if (n_tokens < 0) return false;
  const Session* rec = lookup(id);
  if (rec == nullptr) return false;
  const SequenceState* seq = mgr_.lookup(rec->sequence_id);
  if (seq == nullptr) return false;  // internal invariant violation
  const long long new_length =
      static_cast<long long>(seq->length) + n_tokens;
  return new_length <= mgr_.config().max_seq_len;
}

Status SessionManager::reset_session(SessionId id) {
  const Session* rec = lookup(id);
  if (rec == nullptr) return not_live(id);
  // Reset the bound sequence IN PLACE (KV pages released + zeroed, Delta
  // slot zeroed in place, length = 0) — the session stays live under the
  // SAME id. The in-place reset cannot OOM (no allocation).
  Status s = mgr_.reset_sequence(rec->sequence_id);
  if (!s.ok) return s;
  sessions_.find(id)->second.reset_count++;
  return Status::ok_status();
}

Status SessionManager::destroy_session(SessionId id) {
  const Session* rec = lookup(id);
  if (rec == nullptr) return not_live(id);
  // Retire the bound sequence (ALL KV pages released + zeroed, the Delta
  // slot released + zeroed), then drop the record. The id is never
  // reissued (next_id_ only grows).
  Status s = mgr_.retire_sequence(rec->sequence_id);
  if (!s.ok) return s;
  sessions_.erase(id);
  return Status::ok_status();
}

}  // namespace cudalm
