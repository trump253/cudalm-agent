// CUDALM — v0.9 Phase D: the CUDALM minimal HTTP API handler.
//
// This is CUDALM's OWN minimal serving API — explicitly NOT
// OpenAI-compatible (no /v1/chat/completions, no chat template).
//
// The single runtime chain (pinned — the v0.8 Qwen35SessionText-
// Generator facade is FORBIDDEN here: it builds its own Scheduler and
// would bypass the v0.9 quota / streaming / deadline / TTL-LRU /
// stats layers):
//
//   Qwen35Model -> ModelForwarder -> Qwen35StateManager ->
//   SessionManager -> Scheduler -> ServingController -> this handler
//
// The RAW-TEXT contract (v0.8, pinned): the request body bytes ->
// tokenizer.encode -> NO trim / NO newline insertion / NO separator /
// NO chat template / NO special-token injection -> admit_turn. The
// response text is tokenizer.decode of the COMMITTED token ids only.
//
// The handler is transport-independent (the CPU contract gate drives
// it WITHOUT sockets): handle() produces a complete response;
// handle_stream() writes NDJSON through a sink whose false return
// signals a client disconnect (the handler then performs the Phase B
// cleanup: cancel + non-streaming drain/reap; the session stays live).
//
// Provenance: CUDALM-native (v0.9 Phase D).

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "cudalm/scheduler.h"
#include "cudalm/session.h"
#include "cudalm/serving_controller.h"
#include "cudalm/qwen35_state_manager.h"
#include "tools/common/http_protocol.h"

namespace cudalm {
namespace http {

// The text codec seam: the server installs the native
// Qwen35Tokenizer; the CPU contract gate installs a deterministic
// fake (the serving runtime is identical either way).
class TextCodec {
 public:
  virtual ~TextCodec() = default;
  virtual Status encode(const std::string& utf8,
                        std::vector<int>* out_ids) const = 0;
  virtual Status decode(const std::vector<int>& ids,
                        std::string* out_utf8) const = 0;
  virtual int eos_token_id() const = 0;
};

// The handler's dependencies (all pointers; the server owns them).
struct ServingHttpDeps {
  ServingController* ctrl = nullptr;
  Scheduler* sched = nullptr;  // read-only: the session-busy check
  SessionManager* sessions = nullptr;
  Qwen35StateManager* mgr = nullptr;  // read-only: the logical length
  const TextCodec* codec = nullptr;  // nullptr = text disabled
};

class ServingHttpApi {
 public:
  // A chunk writer for the streaming endpoint. Returns false on a
  // send failure (client disconnect) — the handler performs the
  // cleanup and stops.
  using WriteFn = std::function<bool(const std::string& bytes)>;

  explicit ServingHttpApi(const ServingHttpDeps& deps);

  // Route ONE request and produce the COMPLETE response (all
  // non-streaming endpoints: /healthz, /v1/stats, session lifecycle,
  // the synchronous /turn). Pinned status contract:
  //   200 ok / 201 session created
  //   400 malformed parameter / invalid UTF-8 / empty body
  //   404 unknown route / unknown session
  //   405 wrong method for a known route
  //   408 a synchronous turn that ended deadline-cancelled
  //   409 admission conflict (the session / request / context limit
  //         or a live request on a destroy)
  //   415 an unsupported turn-body Content-Type (text/plain + the
  //         documented compatibility set only; see the gate in the
  //         .cpp) — the sync and the streaming endpoints agree
  //   500 an internal forward / runtime failure
  HttpResponse handle(const HttpRequest& req);

  // The streaming endpoint (POST /v1/sessions/<id>/turn/stream):
  // validation errors are written through `write` as a complete JSON
  // response (and true is returned); on success the NDJSON stream is
  // written event by event. Returns false when the client DISCONNECTED
  // (the handler already cancelled the request + drained/reaped its
  // stream state; the session stays live and the quota is released).
  bool handle_stream(const HttpRequest& req, const WriteFn& write);

  // The turn-query resolution — PUBLIC so the CPU contract gate can
  // assert the RESOLVED SamplingConfig directly (no sockets, no
  // model). The raw query string -> max_new_tokens + the RESOLVED
  // SamplingConfig + the optional deadline. The sampling resolution
  // mirrors the FROZEN v0.4 `cudalm-generate` contract
  // (include/cudalm/generate_cli.h): any of temperature / top_k /
  // top_p PRESENT enables sampling mode; an omitted temperature then
  // defaults to 1.0; an explicit 0 / -0 selects the frozen greedy
  // path; a seed without a sampling flag is IGNORED (greedy consumes
  // no RNG). The temperature TEXT must round to a representable
  // float — overflow to inf and underflow to zero are rejected
  // (mirroring src/cli/generate_cli.cpp, the same numeric semantics —
  // there is no second, different rule set). Fail-loud: returns false
  // + *error for a malformed / out-of-contract query (a 400 BEFORE
  // any admission / forward).
  bool resolve_turn_query(const std::string& query, int* out_max_new,
                          SamplingConfig* out_sampling,
                          std::chrono::steady_clock::time_point* out_deadline,
                          bool* out_has_deadline, std::string* error) const;

 private:
  ServingHttpDeps deps_;

  // The session id of a "/v1/sessions/<id>/..." path (0 when the path
  // is not a valid session route).
  static bool parse_session_path(const std::string& path, std::string* rest,
                                 SessionId* out_id);
  // The raw-text turn core shared by the sync + stream endpoints.
  // Returns false (with *error) when the turn could not be ADMIITED.
  bool admit_text_turn(SessionId session_id, const std::string& body,
                       const std::vector<int>& tokens, int max_new,
                       const SamplingConfig& sampling,
                       const std::chrono::steady_clock::time_point deadline,
                       bool has_deadline, RequestId* out_request_id,
                       std::string* error) const;
  // The client-disconnect cleanup (pinned): cancel the request (if
  // still live) + the non-streaming drain/reap (the committed state
  // stays, the pending tail is never committed, the quota is released,
  // the stream bookkeeping is drained/reaped, the session STAYS live).
  void cleanup_disconnected(RequestId request_id);
  // The logical context length of a live session (-1 when gone).
  int context_length_of(SessionId session_id) const;
};

}  // namespace http
}  // namespace cudalm
