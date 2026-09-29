// CUDALM — v0.9 Phase D: the CUDALM minimal HTTP API handler (see
// the header for the pinned contracts).

#include "tools/common/http_serving_handler.h"

namespace cudalm {
namespace http {

namespace {

const char* status_name(RequestStatus st) {
  switch (st) {
    case RequestStatus::Waiting: return "Waiting";
    case RequestStatus::Running: return "Running";
    case RequestStatus::Finished: return "Finished";
    case RequestStatus::Cancelled: return "Cancelled";
    case RequestStatus::Failed: return "Failed";
  }
  return "Unknown";
}

const char* finish_name(FinishReason fr) {
  switch (fr) {
    case FinishReason::None: return "None";
    case FinishReason::Eos: return "Eos";
    case FinishReason::MaxNewTokens: return "MaxNewTokens";
    case FinishReason::Cancelled: return "Cancelled";
    case FinishReason::Failed: return "Failed";
  }
  return "Unknown";
}

// A strict non-negative / positive integer parser (fail loud).
bool parse_int(const std::string& v, long* out, bool allow_zero) {
  if (v.empty()) return false;
  long n = 0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    const char c = v[i];
    if (i == 0 && c == '+') continue;
    if (c < '0' || c > '9') return false;
    n = n * 10 + (c - '0');
    if (n > 1000000000L) return false;  // a sane cap (fail loud)
  }
  if (n == 0 && !allow_zero) return false;
  *out = n;
  return true;
}

bool parse_float(const std::string& v, float* out) {
  if (v.empty()) return false;
  char* end = nullptr;
  const double d = std::strtod(v.c_str(), &end);
  if (end == v.c_str() || *end != '\0') return false;
  if (!(d >= 0.0)) return false;  // NaN / negative fail loud
  *out = static_cast<float>(d);
  return true;
}

}  // namespace

ServingHttpApi::ServingHttpApi(const ServingHttpDeps& deps) : deps_(deps) {}

bool ServingHttpApi::parse_session_path(const std::string& path,
                                        std::string* rest,
                                        SessionId* out_id) {
  // "/v1/sessions/<id>" or "/v1/sessions/<id>/<rest...>"
  const std::string prefix = "/v1/sessions/";
  if (path.rfind(prefix, 0) != 0) return false;
  const std::string tail = path.substr(prefix.size());
  std::size_t slash = tail.find('/');
  const std::string id_part = slash == std::string::npos
                                  ? tail
                                  : tail.substr(0, slash);
  long id = 0;
  if (!parse_int(id_part, &id, true) || id <= 0) return false;
  *out_id = static_cast<SessionId>(id);
  if (rest != nullptr) {
    *rest = slash == std::string::npos ? "" : tail.substr(slash + 1);
  }
  return true;
}

int ServingHttpApi::context_length_of(SessionId session_id) const {
  const Session* s = deps_.sessions->lookup(session_id);
  if (s == nullptr) return -1;
  const SequenceState* st = deps_.mgr->lookup(s->sequence_id);
  return st == nullptr ? -1 : st->length;
}

bool ServingHttpApi::parse_turn_query(
    const std::string& query, std::vector<int>* /*out_tokens*/,
    int* out_max_new, SamplingConfig* out_sampling,
    std::chrono::steady_clock::time_point* out_deadline,
    bool* out_has_deadline, std::string* error) const {
  std::map<std::string, std::string> q;
  if (!parse_query(query, &q)) {
    *error = "malformed query string";
    return false;
  }
  // Unknown keys fail loud (no silent typo-swallowing):
  static const char* kKnown[] = {"max_new_tokens", "temperature", "top_k",
                                 "top_p", "seed", "deadline_ms"};
  for (const auto& kv : q) {
    bool known = false;
    for (const char* k : kKnown) {
      if (kv.first == k) {
        known = true;
        break;
      }
    }
    if (!known) {
      *error = "unknown parameter: " + kv.first;
      return false;
    }
  }
  SamplingConfig sampling = SamplingConfig::greedy();
  long max_new = 8;  // the default
  auto it = q.find("max_new_tokens");
  if (it != q.end()) {
    if (!parse_int(it->second, &max_new, false)) {
      *error = "invalid max_new_tokens";
      return false;
    }
  }
  it = q.find("temperature");
  if (it != q.end()) {
    if (!parse_float(it->second, &sampling.temperature)) {
      *error = "invalid temperature";
      return false;
    }
  }
  it = q.find("top_k");
  if (it != q.end()) {
    long v = 0;
    if (!parse_int(it->second, &v, true)) {
      *error = "invalid top_k";
      return false;
    }
    sampling.top_k = static_cast<int>(v);
  }
  it = q.find("top_p");
  if (it != q.end()) {
    if (!parse_float(it->second, &sampling.top_p)) {
      *error = "invalid top_p";
      return false;
    }
  }
  it = q.find("seed");
  if (it != q.end()) {
    long v = 0;
    if (!parse_int(it->second, &v, true)) {
      *error = "invalid seed";
      return false;
    }
    sampling.seed = static_cast<std::uint64_t>(v);
  }
  // The frozen sampling validation gate (same semantics as
  // cudalm-generate):
  std::string verr;
  if (!validate_sampling_config(sampling, &verr)) {
    *error = verr;
    return false;
  }
  *out_has_deadline = false;
  *out_deadline = std::chrono::steady_clock::time_point::max();
  it = q.find("deadline_ms");
  if (it != q.end()) {
    long ms = 0;
    if (!parse_int(it->second, &ms, true)) {
      *error = "invalid deadline_ms";
      return false;
    }
    if (ms > 0) {
      *out_has_deadline = true;
      *out_deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    }
    // deadline_ms == 0 = the deadline is disabled (pinned).
  }
  *out_max_new = static_cast<int>(max_new);
  *out_sampling = sampling;
  return true;
}

bool ServingHttpApi::admit_text_turn(
    SessionId session_id, const std::string& /*body*/,
    const std::vector<int>& tokens, int max_new, const SamplingConfig& sampling,
    const std::chrono::steady_clock::time_point deadline,
    bool /*has_deadline*/, RequestId* out_request_id,
    std::string* error) const {
  if (tokens.empty()) {
    *error = "the body encoded to zero tokens";
    return false;
  }
  // The EOS id comes from the codec (the pinned tokenizer EOS); -1 =
  // no EOS check (a text-disabled codec).
  const int eos = deps_.codec != nullptr
                      ? static_cast<int>(deps_.codec->eos_token_id())
                      : -1;
  const Status s = deps_.ctrl->admit_turn(
      session_id, tokens, max_new, eos, sampling, out_request_id, deadline);
  if (!s.ok) {
    *error = s.message;
    return false;
  }
  return true;
}

void ServingHttpApi::cleanup_disconnected(RequestId request_id) {
  // The pinned client-disconnect cleanup (Phase B lifecycle):
  //   1. cancel the request (if it is still live — a terminal
  //      request is left exactly as it is; the frozen scheduler's
  //      idempotent contract);
  //   2. the NON-STREAMING drain: drive to quiescence while the
  //      terminal events are drained + REAPED (the committed state
  //      stays, the pending tail is never committed, the request
  //      quota is released, the stream bookkeeping is gone).
  // The session STAYS live — a disconnect is never a destroy.
  const Request* r = deps_.sched->get(request_id);
  if (r != nullptr && !is_terminal(r->status)) {
    (void)deps_.ctrl->cancel(request_id);
  }
  (void)deps_.ctrl->run();
}

HttpResponse ServingHttpApi::handle(const HttpRequest& req) {
  // ---- /healthz -----------------------------------------------------
  if (req.path == "/healthz") {
    if (req.method != "GET") {
      return HttpResponse::json_error(405, "", "wrong method");
    }
    return HttpResponse::json(200, "{\"ok\":true}");
  }

  // ---- /v1/stats ------------------------------------------------------
  if (req.path == "/v1/stats") {
    if (req.method != "GET") {
      return HttpResponse::json_error(405, "", "wrong method");
    }
    const ServingStats st = deps_.ctrl->stats();
    std::string b;
    b += "{\"live_sessions\":";
    b += std::to_string(st.live_sessions);
    b += ",\"live_requests\":";
    b += std::to_string(st.live_requests);
    b += ",\"total_admitted_sessions\":";
    b += std::to_string(st.total_admitted_sessions);
    b += ",\"total_admitted_requests\":";
    b += std::to_string(st.total_admitted_requests);
    b += ",\"rejected_session_limit\":";
    b += std::to_string(st.rejected_session_limit);
    b += ",\"rejected_request_limit\":";
    b += std::to_string(st.rejected_request_limit);
    b += ",\"rejected_context_limit\":";
    b += std::to_string(st.rejected_context_limit);
    b += ",\"evicted_sessions_ttl\":";
    b += std::to_string(st.evicted_sessions_ttl);
    b += ",\"evicted_sessions_lru\":";
    b += std::to_string(st.evicted_sessions_lru);
    b += ",\"eviction_no_candidate\":";
    b += std::to_string(st.eviction_no_candidate);
    b += "}";
    return HttpResponse::json(200, b);
  }

  // ---- /v1/sessions (create) ------------------------------------------
  if (req.path == "/v1/sessions") {
    if (req.method != "POST") {
      return HttpResponse::json_error(405, "", "wrong method");
    }
    SessionId sid = 0;
    const Status s = deps_.ctrl->create_session(&sid);
    if (!s.ok) {
      // The admission conflict (the session limit / the zero-
      // capacity contract / an internal policy error):
      return HttpResponse::json_error(409, "", s.message);
    }
    return HttpResponse::json(201,
                              "{\"session_id\":" + std::to_string(sid) +
                                  "}");
  }

  // ---- /v1/sessions/<id>/... ------------------------------------------
  SessionId sid = 0;
  std::string rest;
  if (parse_session_path(req.path, &rest, &sid)) {
    const Session* s = deps_.sessions->lookup(sid);
    if (s == nullptr) {
      return HttpResponse::json_error(404, "", "unknown session");
    }
    if (rest.empty()) {
      // The session object itself:
      if (req.method == "DELETE") {
        // A session with a LIVE request is a conflict (destroying it
        // would strand the request):
        if (deps_.sched->session_busy(sid)) {
          return HttpResponse::json_error(
              409, "", "the session has a live request");
        }
        const Status st = deps_.ctrl->destroy_session(sid);
        if (!st.ok) {
          return HttpResponse::json_error(409, "", st.message);
        }
        return HttpResponse::json(200, "{\"session_id\":" +
                                            std::to_string(sid) +
                                            ",\"destroyed\":true}");
      }
      if (req.method != "GET") {
        return HttpResponse::json_error(405, "", "wrong method");
      }
      const int len = context_length_of(sid);
      return HttpResponse::json(
          200,
          "{\"session_id\":" + std::to_string(sid) +
              ",\"context_length\":" + std::to_string(len) + "}");
    }
    if (rest == "reset") {
      if (req.method != "POST") {
        return HttpResponse::json_error(405, "", "wrong method");
      }
      const Status st = deps_.ctrl->reset_session(sid);
      if (!st.ok) {
        return HttpResponse::json_error(409, "", st.message);
      }
      return HttpResponse::json(200, "{\"session_id\":" +
                                          std::to_string(sid) +
                                          ",\"context_length\":0}");
    }
    if (rest == "turn") {
      if (req.method != "POST") {
        return HttpResponse::json_error(405, "", "wrong method");
      }
      // ---- the raw-text synchronous turn -------------------------
      if (req.body.empty()) {
        return HttpResponse::json_error(400, "", "empty body");
      }
      int max_new = 0;
      SamplingConfig sampling;
      std::chrono::steady_clock::time_point deadline;
      bool has_deadline = false;
      std::string error;
      if (!parse_turn_query(req.query, nullptr, &max_new, &sampling,
                            &deadline, &has_deadline, &error)) {
        return HttpResponse::json_error(400, "", error);
      }
      // RAW-TEXT contract: the body bytes are encoded VERBATIM (NO
      // trim / NO newline insertion / NO separator / NO chat
      // template / NO special-token injection):
      if (deps_.codec == nullptr) {
        return HttpResponse::json_error(400, "",
                                        "the server has no text codec");
      }
      std::vector<int> tokens;
      const Status es = deps_.codec->encode(req.body, &tokens);
      if (!es.ok) {
        return HttpResponse::json_error(400, "",
                                        "invalid UTF-8 body: " + es.message);
      }
      RequestId rid = 0;
      if (!admit_text_turn(sid, req.body, tokens, max_new, sampling,
                           deadline, has_deadline, &rid, &error)) {
        return HttpResponse::json_error(409, "", error);
      }
      // Drive to quiescence (the frozen scheduler through the
      // controller) and collect THIS request's committed events:
      const std::vector<ServingEvent> all = deps_.ctrl->run_stream();
      std::vector<int> generated;
      bool terminal = false;
      RequestStatus tstatus = RequestStatus::Finished;
      FinishReason treason = FinishReason::MaxNewTokens;
      bool deadline_exceeded = false;
      for (const ServingEvent& e : all) {
        if (e.request_id != rid) continue;
        if (e.kind == ServingEventKind::Token) {
          generated.push_back(e.token_id);
        } else {
          terminal = true;
          tstatus = e.status;
          treason = e.finish_reason;
          deadline_exceeded = e.deadline_exceeded;
        }
      }
      if (!terminal) {
        return HttpResponse::json_error(
            500, "", "the turn did not reach a terminal event");
      }
      // A real forward / runtime failure:
      if (tstatus == RequestStatus::Failed) {
        return HttpResponse::json_error(500, "", "runtime forward failure");
      }
      std::string text;
      if (deps_.codec != nullptr) {
        const Status ds = deps_.codec->decode(generated, &text);
        if (!ds.ok) {
          return HttpResponse::json_error(500, "", ds.message);
        }
      }
      const int ctx = context_length_of(sid);
      std::string b;
      b += "{\"request_id\":";
      b += std::to_string(rid);
      b += ",\"session_id\":";
      b += std::to_string(sid);
      b += ",\"generated_token_ids\":[";
      for (std::size_t i = 0; i < generated.size(); ++i) {
        if (i) b += ",";
        b += std::to_string(generated[i]);
      }
      b += "],\"generated_text\":\"";
      b += json_escape(text);  // NUL / quote / control-safe
      b += "\",\"status\":\"";
      b += status_name(tstatus);
      b += "\",\"finish_reason\":\"";
      b += finish_name(treason);
      b += "\",\"deadline_exceeded\":";
      b += deadline_exceeded ? "true" : "false";
      b += ",\"context_length\":";
      b += std::to_string(ctx);
      b += "}";
      // A deadline-cancelled synchronous turn is a 408 (the body is
      // still the normal result object):
      const int code = deadline_exceeded ? 408 : 200;
      return HttpResponse::json(code, b);
    }
    if (rest == "turn/stream") {
      // Routed through handle_stream() — the sync handler must never
      // consume a stream request:
      return HttpResponse::json_error(405, "",
                                      "use the streaming endpoint");
    }
    return HttpResponse::json_error(404, "", "unknown route");
  }

  // ---- unknown route ----------------------------------------------------
  return HttpResponse::json_error(404, "", "unknown route");
}

bool ServingHttpApi::handle_stream(const HttpRequest& req,
                                   const WriteFn& write) {
  // Validation + admission (exactly like the synchronous turn, but
  // the errors are written as a COMPLETE JSON response):
  auto answer_error = [&](int status, const std::string& msg) {
    return write(format_response(HttpResponse::json_error(status, "", msg)));
  };
  if (req.method != "POST") {
    return answer_error(405, "wrong method");
  }
  SessionId sid = 0;
  std::string rest;
  if (!parse_session_path(req.path, &rest, &sid) || rest != "turn/stream") {
    return answer_error(404, "unknown route");
  }
  if (deps_.sessions->lookup(sid) == nullptr) {
    return answer_error(404, "unknown session");
  }
  if (req.body.empty()) {
    return answer_error(400, "empty body");
  }
  int max_new = 0;
  SamplingConfig sampling;
  std::chrono::steady_clock::time_point deadline;
  bool has_deadline = false;
  std::string error;
  if (!parse_turn_query(req.query, nullptr, &max_new, &sampling, &deadline,
                        &has_deadline, &error)) {
    return answer_error(400, error);
  }
  std::vector<int> tokens;
  if (deps_.codec != nullptr) {
    const Status es = deps_.codec->encode(req.body, &tokens);
    if (!es.ok) {
      return answer_error(400, "invalid UTF-8 body: " + es.message);
    }
  } else {
    return answer_error(400, "the server has no text codec");
  }
  RequestId rid = 0;
  if (!admit_text_turn(sid, req.body, tokens, max_new, sampling, deadline,
                       has_deadline, &rid, &error)) {
    return answer_error(409, error);
  }

  // ---- the NDJSON stream -------------------------------------------------
  // The header block (NO Content-Length — the body is terminated by
  // the close):
  {
    const std::string wire =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/x-ndjson\r\n"
        "Connection: close\r\n\r\n";
    if (!write(wire)) goto disconnected;
  }

  // Drive with the FROZEN controller step (commit-before-visible:
  // only the controller's EMITTED committed token events are ever
  // sent — the pending tail is never peeked at):
  for (int i = 0; i < 1000000; ++i) {
    if (deps_.ctrl->stats().live_requests == 0) {
      // The request terminal + drained: the stream is complete
      // (the terminal event was emitted on the last step).
      break;
    }
    const std::vector<ServingEvent> ev = deps_.ctrl->step_stream();
    bool terminal = false;
    for (const ServingEvent& e : ev) {
      if (e.request_id != rid) continue;  // other requests' events
                                          // are NOT this client's
      if (e.kind == ServingEventKind::Token) {
        std::string line = "{\"type\":\"token\",\"request_id\":";
        line += std::to_string(e.request_id);
        line += ",\"token_id\":";
        line += std::to_string(e.token_id);
        line += "}\n";
        if (!write(line)) goto disconnected;
      } else {  // the RequestTerminal (AFTER all the request's tokens)
        terminal = true;
        // The full committed text + the context length (the no-
        // incremental-delta contract: the text arrives ONCE, at the
        // terminal):
        const Request* r = deps_.sched->get(rid);
        const int committed = r != nullptr ? r->committed_generated : 0;
        std::vector<int> committed_ids;
        if (r != nullptr) {
          committed_ids.assign(r->generated.begin(),
                               r->generated.begin() + committed);
        }
        std::string text;
        if (deps_.codec != nullptr) {
          const Status ds = deps_.codec->decode(committed_ids, &text);
          if (!ds.ok) {
            return answer_error(500, ds.message);
          }
        }
        const int ctx = context_length_of(sid);
        std::string line = "{\"type\":\"terminal\",\"request_id\":";
        line += std::to_string(e.request_id);
        line += ",\"status\":\"";
        line += status_name(e.status);
        line += "\",\"finish_reason\":\"";
        line += finish_name(e.finish_reason);
        line += "\",\"deadline_exceeded\":";
        line += e.deadline_exceeded ? "true" : "false";
        line += ",\"generated_token_ids\":[";
        for (std::size_t j = 0; j < committed_ids.size(); ++j) {
          if (j) line += ",";
          line += std::to_string(committed_ids[j]);
        }
        line += "],\"generated_text\":\"";
        line += json_escape(text);
        line += "\",\"context_length\":";
        line += std::to_string(ctx);
        line += "}\n";
        if (!write(line)) goto disconnected;
      }
    }
    if (terminal && deps_.ctrl->stats().live_requests == 0) break;
    // (The request terminal but OTHER requests still live: the loop
    // keeps driving them; their events are not this client's and are
    // discarded — the loop exits when live_requests is 0.)
  }
  return true;

disconnected:
  // The client DISCONNECTED: the pinned cleanup (cancel + the non-
  // streaming drain/reap; the session stays live, the quota is
  // released, the committed state is intact, the pending tail is
  // never committed):
  cleanup_disconnected(rid);
  return false;
}

}  // namespace http
}  // namespace cudalm
