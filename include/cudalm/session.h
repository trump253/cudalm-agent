// CUDALM — v0.8 Phase A: session abstraction / persistent inference state.
//
// A SESSION is the LONG-LIVED OWNER of one Qwen3.5 hybrid inference state:
//
//   Session
//     ├── SessionId                  (monotonic, NEVER REUSED — its own id
//     │                               space, independent of both SequenceId
//     │                               and RequestId)
//     ├── logical position/length    (the bound sequence's `length` — the
//     │                               single source of truth, derived, never
//     │                               duplicated here)
//     ├── Full-Attention paged KV    (the bound sequence's KvBlockTable +
//     │                               the pool's physical pages)
//     ├── DeltaNet conv_state        (the bound sequence's Delta slot,
//     │                               shared by all 18 linear layers)
//     ├── DeltaNet recurrent_state   (same slot)
//     └── lifecycle metadata         (SessionState, reset_count)
//
// A REQUEST is a TRANSIENT operation BOUND TO a session: the new input
// tokens + a generation config + a request execution state. In the session
// runtime a request NEVER owns, releases, or resets the session's state —
// on completion the request is gone and the session continues to exist
// with its state PERSISTED:
//
//   request completion:
//     KV pages        NOT released
//     Delta conv      NOT released, NOT zeroed
//     Delta recurrent NOT released, NOT zeroed
//     position        NOT reset
//
// Only these two operations mutate or release a session's state:
//
//   reset_session(id)   — the session stays LIVE under the SAME SessionId;
//                          the bound sequence is reset in place (all KV
//                          pages released + zeroed, the Delta slot zeroed
//                          in place, length = 0). Afterwards the session
//                          behaves EXACTLY like a fresh one. Cannot OOM
//                          (no allocation happens).
//   destroy_session(id) — the bound sequence is RETIRED (all KV pages
//                          released + zeroed, the Delta slot released +
//                          zeroed) and the session record is removed. The
//                          SessionId is invalidated (never reissued); any
//                          later operation on it is a Status error /
//                          nullptr.
//
// ID SPACES (three independent monotonic, never-reused id spaces — the
// same discipline as SequenceId/RequestId; numerical coincidence is never
// meaningful):
//   * SessionId  : issued by the SessionManager (this file), from 1 upward;
//   * SequenceId : issued by the Qwen35StateManager (one per session — the
//                  session's bound state);
//   * RequestId  : issued by the Scheduler (a request references a session
//                  in the session runtime; the v0.6 request-scoped mode
//                  keeps creating its own short-lived sequence).
//
// IMPLEMENTATION: a thin BINDING layer over the frozen v0.5
// Qwen35StateManager (NON-OWNING — it outlives the session manager, exactly
// like the Scheduler's references). Each session is backed by EXACTLY ONE
// v0.5 sequence; the pools' ownership, zero-on-release/zero-on-reset
// semantics, and accounting are inherited UNCHANGED (no model math is
// touched). The session manager adds only: the SessionId -> SequenceId
// binding, the session lifecycle operations, and the context-capacity
// query `fits()` (the Phase B overflow-policy foundation — see docs
// v08_session_runtime.md).
//
// CONTEXT OVERFLOW POLICY (v0.8 early form, pinned): there is NO eviction
// of any kind (the DeltaNet recurrent state holds compressed history that
// cannot be dropped). `fits(id, n)` reports whether `n` MORE tokens would
// fit: length + n <= max_seq_len. When they do not fit, the caller must
// REJECT loudly — never silently drop the earliest tokens/pages. (The KV
// page POOL OOM is a separate runtime condition — the v0.5 OOM-before-
// mutation contract; `fits()` is purely the context-LENGTH policy.)
//
// PHASE B FUTURE CONTRACT (the API must not preclude it): the bound
// sequence is exactly what forward_token_with_state() /
// forward_batch_with_state() already drive (position derived from the
// sequence length). A future incremental multi-turn request is therefore
// an APPEND-ONLY continuation: it forwards the session's NEW tokens from
// the current length, with no state (re)initialization at the turn
// boundary — the state simply persists across turns (gated by
// tests/cuda/test_qwen35_session_runtime.cpp in Phase A).

#pragma once

#include <cstdint>
#include <map>

#include "cudalm/qwen35_state_manager.h"  // SequenceId, Qwen35StateManager
#include "cudalm/weight_format.h"         // Status

namespace cudalm {

// Monotonically increasing, never-reused session identifier (0 = "no
// session"; the manager issues ids from 1 upward).
using SessionId = std::uint64_t;

// Session lifecycle state. Phase A is SYNCHRONOUS and single-threaded:
// every operation completes before the next one starts, so a session is
// either live (Active) or gone (record removed — lookup returns nullptr,
// like the state manager's retired sequences). The enum exists so future
// phases can extend the lifecycle (e.g. an Evicting state for v0.9-style
// context eviction) without a breaking change.
enum class SessionState {
  Active,  // live: state bound, usable by requests
};

// One session's host-side record (metadata only — the DEVICE state lives
// in the manager's pools, addressed through the bound SequenceId).
struct Session {
  SessionId id = 0;
  SequenceId sequence_id = 0;  // the bound v0.5 sequence (state owner)
  SessionState state = SessionState::Active;
  // Lifecycle metadata (observability only — never gates correctness):
  int reset_count = 0;  // reset_session() calls applied to this session
};

// The v0.8 Phase A session control plane over the frozen v0.5 state
// manager (one SessionManager per Qwen35StateManager; non-owning).
class SessionManager {
 public:
  // mgr: the state manager (NON-OWNING — it must outlive the session
  // manager; its pools' stream is the single-stream contract for everything
  // that later drives the model).
  explicit SessionManager(Qwen35StateManager& mgr) : mgr_(mgr) {}

  SessionManager(const SessionManager&) = delete;
  SessionManager& operator=(const SessionManager&) = delete;

  // ---- lifecycle (see the header contract) ---------------------------------
  // TRANSACTIONAL create: creates a fresh bound sequence (a FRESH ZEROED
  // Delta slot by zero-on-release; an EMPTY KV block table, no pages;
  // length = 0) and binds it to a NEW monotonic never-reused SessionId.
  // On ANY failure (e.g. the Delta pool OOM) NOTHING is registered: no id
  // is issued, no half-record exists, no slot is leaked — the next
  // create_session succeeds on a recovered pool and the id sequence stays
  // monotone.
  Status create_session(SessionId* out_id);

  // nullptr when the id was never created or was destroyed (a query, not
  // an error). A returned pointer is valid until the NEXT mutation of this
  // manager (destroy_session erases records — the same discipline as the
  // state manager: cache the fields you need before a mutation).
  const Session* lookup(SessionId id) const;

  // The session's bound SequenceId (the handle a request uses to drive the
  // model via forward_token_with_state / forward_batch_with_state).
  // Status error when the session is not live.
  Status sequence_id_of(SessionId id, SequenceId* out) const;

  // The session's current logical position (== the bound sequence's
  // `length` — the single source of truth; this is a live read, never a
  // cached copy). Status error when the session is not live.
  Status context_length_of(SessionId id, int* out) const;

  // Context-capacity query (the overflow-policy foundation, pinned):
  // true iff `n_tokens` MORE tokens would fit the bound sequence
  // (length + n_tokens <= max_seq_len) with the session LIVE. false for
  // an unknown/destroyed session or n_tokens < 0 (a query — it never
  // allocates and never mutates). When false, a future appending request
  // must be REJECTED (no silent eviction of the earliest tokens/pages).
  bool fits(SessionId id, int n_tokens) const;

  // Reset: the session stays LIVE under the SAME SessionId; the bound
  // sequence is reset in place (all KV pages released + zeroed, the Delta
  // slot zeroed IN PLACE — same slot id, no release/acquire — length = 0).
  // Afterwards the session behaves EXACTLY like a fresh one. Cannot OOM
  // (no allocation). Status error when the session is not live.
  Status reset_session(SessionId id);

  // Destroy: retires the bound sequence (ALL KV pages released + zeroed,
  // the Delta slot released + zeroed) and removes the session record. The
  // SessionId is invalidated (never reissued). Not idempotent: a second
  // destroy of the same id is a Status error (fail loud, like
  // retire_sequence). Status error when the session is not live.
  Status destroy_session(SessionId id);

  // ---- accounting / introspection --------------------------------------------
  int num_sessions() const {
    return static_cast<int>(sessions_.size());
  }
  // The NEXT SessionId to be issued (1 when nothing has ever been
  // created). Ids are monotone increasing and never reused.
  SessionId next_session_id() const { return next_id_; }

  // The wrapped state manager (non-owning; the pools' streams are the
  // single-stream contract for any model forward that later uses a
  // session's bound sequence).
  Qwen35StateManager& manager() { return mgr_; }
  const Qwen35StateManager& manager() const { return mgr_; }

 private:
  // Status error for an id that is not a live session.
  Status not_live(SessionId id) const;

  Qwen35StateManager& mgr_;
  std::map<SessionId, Session> sessions_;
  SessionId next_id_ = 1;  // ids start at 1; 0 means "no session"
};

}  // namespace cudalm
