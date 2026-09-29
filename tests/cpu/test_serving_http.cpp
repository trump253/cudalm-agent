// CUDALM — v0.9 Phase D: the Serving HTTP CONTRACT gate (CPU — NO
// sockets, NO checkpoint, NO model: the deterministic fake forwarder +
// real pools + real SessionManager + real Scheduler + the
// ServingController + a FAKE TextCodec, driving the HTTP handler
// DIRECTLY — the handler is transport-independent by design).
//
// Proves the HTTP adapter does NOT bypass the serving runtime:
//   * create / turn / second-turn append-only / stats / reset /
//     destroy / unknown session / wrong method / unknown path;
//   * the HTTP generated ids == the ServingController's COMMITTED
//     event ids (a direct parity check against the scheduler's
//     committed state — the adapter never peeks the pending tail);
//   * the streaming endpoint emits token-id events BEFORE the
//     terminal, and the streamed token ids == the terminal / final
//     generated ids;
//   * the deadline: a synchronous turn that ends deadline-cancelled
//     is a 408 (the fake clock is advanced past the deadline —
//     deterministic, no real-time flakiness);
//   * the raw-text contract: the body bytes are encoded VERBATIM by
//     the codec (the fake codec maps each byte to one token — leading
//     / trailing whitespace and embedded NULs survive to the ids);
//   * a client DISCONNECT mid-stream (a write failure) triggers the
//     pinned cleanup: the request is cancelled + drained/reaped, the
//     committed state stays, the quota is released, and the session
//     stays LIVE for the next turn.
//
// Provenance: CUDALM-native (v0.9 Phase D).

#include "../../tests/common/check.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/scheduler.h"
#include "cudalm/sampling.h"
#include "cudalm/session.h"
#include "cudalm/serving_controller.h"
#include "tools/common/http_protocol.h"
#include "tools/common/http_serving_handler.h"

using namespace cudalm;

namespace {

Qwen35Config small_config() {
  Qwen35Config c;
  c.hidden_size = 256;
  c.num_hidden_layers = 8;
  c.intermediate_size = 512;
  c.vocab_size = 512;
  c.n_heads = 4;
  c.n_kv_heads = 2;
  c.head_dim = 64;
  c.lin_num_k_heads = 4;
  c.lin_num_v_heads = 4;
  c.lin_key_head_dim = 32;
  c.lin_value_head_dim = 32;
  c.lin_conv_kernel_dim = 4;
  c.full_attention_interval = 4;
  c.group_size = 128;
  c.max_seq_len = 64;
  c.eps = 1e-6f;
  c.rope_theta = 1e7f;
  c.partial_rotary_factor = 0.25f;
  c.mrope_section[0] = 4;
  c.mrope_section[1] = 2;
  c.mrope_section[2] = 2;
  return c;
}

// Deterministic CPU forwarder (the Phase B/C pattern).
class FakeForwarder : public SequenceForwarder {
 public:
  int vocab = 512;

  Status forward_token(int /*token_id*/, SequenceId sequence_id,
                       Qwen35StateManager& mgr,
                       cudaStream_t /*stream*/) override {
    last_step_ = count_[sequence_id]++;
    const SequenceState* st = mgr.lookup(sequence_id);
    if (st == nullptr) {
      return Status::error("fake forward: unknown sequence");
    }
    const int pos = st->length;
    Status s = mgr.ensure_kv_capacity(sequence_id, pos);
    if (!s.ok) return s;
    return mgr.set_length(sequence_id, pos + 1);
  }

  Status logits_to_host(std::vector<__nv_bfloat16>* out,
                        cudaStream_t /*stream*/) const override {
    out->assign(static_cast<std::size_t>(vocab),
                __float2bfloat16(-1.0f));
    const int pick = static_cast<int>(
        (static_cast<std::uint64_t>(3) *
             static_cast<std::uint64_t>(last_step_) +
         1) %
        static_cast<std::uint64_t>(vocab));
    (*out)[static_cast<std::size_t>(pick)] = __float2bfloat16(10.0f);
    return Status::ok_status();
  }

  int vocab_size() const override { return vocab; }

 private:
  mutable int last_step_ = -1;
  mutable std::map<SequenceId, int> count_;
};

// A FAKE monotonic clock (manual advance) — the deadline test is
// DETERMINISTIC (advance past the deadline, no real-time race).
class FakeClock : public MonotonicClock {
 public:
  std::chrono::steady_clock::time_point t0 =
      std::chrono::steady_clock::now();
  int ms = 0;

  std::chrono::steady_clock::time_point now() const override {
    return t0 + std::chrono::milliseconds(ms);
  }
  void advance_ms(int n) { ms += n; }
};

// A deterministic FAKE text codec (the TextCodec seam): encode maps
// each body byte to one token id (VERBATIM — no trim / no NUL drop);
// decode maps each id to one byte. The serving runtime is identical to
// the real-tokenizer path (only the codec differs).
class FakeCodec : public http::TextCodec {
 public:
  Status encode(const std::string& utf8,
                std::vector<int>* out) const override {
    out->clear();
    for (const unsigned char b : utf8) {
      out->push_back(static_cast<int>((b % 500u) + 2u));  // ids in [2,501]
    }
    return Status::ok_status();
  }

  Status decode(const std::vector<int>& ids,
                std::string* out) const override {
    out->clear();
    for (int id : ids) {
      out->push_back(static_cast<char>(id % 256));
    }
    return Status::ok_status();
  }

  // id 0 is never produced by the codec (>= 2) nor by the forwarder's
  // pick ((3k+1)%512 == 0 only at k == 341) — EOS never fires in the
  // small generations here:
  int eos_token_id() const override { return 0; }
};

// ---- the runtime chain (the stream is created FIRST) ---------------------

struct Runtime {
  cudaStream_t stream = nullptr;
  Qwen35StateManager mgr;
  SessionManager sm;
  FakeForwarder fwd;
  Scheduler sched;
  FakeClock clk;
  ServingController ctrl;
  FakeCodec codec;
  http::ServingHttpDeps deps;
  http::ServingHttpApi api;

  // The stream is created by the CALLER first (the state manager
  // needs a live stream pointer at construction).
  Runtime(cudaStream_t stream, const ServingLimits& limits,
          const SessionEvictionPolicy& pol = SessionEvictionPolicy())
      : stream(stream),
        mgr(small_config(), 4, 32, 4, stream),
        sm(mgr),
        fwd(),
        sched(fwd, mgr, stream, &sm),
        clk(),
        ctrl(sched, sm, limits, &clk, pol),
        codec(),
        // deps must be COMPLETE before api copies it (the api's
        // deps_ is a COPY — a body assignment would be too late):
        deps(http::ServingHttpDeps{&ctrl, &sched, &sm, &mgr, &codec}),
        api(deps) {}

  ~Runtime() {
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
  }
};


// ---- minimal JSON readers (test-only) ------------------------------------

// A JSON object member "key":<int>
bool json_int(const std::string& j, const std::string& key, long* out) {
  const std::string pat = "\"" + key + "\":";
  const std::size_t p = j.find(pat);
  if (p == std::string::npos) return false;
  const std::size_t v = p + pat.size();
  std::size_t e = v;
  if (e < j.size() && (j[e] == '-' || j[e] == '+')) ++e;
  const std::size_t digits = e;
  while (e < j.size() && j[e] >= '0' && j[e] <= '9') ++e;
  if (e == digits) return false;
  *out = std::stol(j.substr(digits, e - digits));
  return true;
}

// A JSON object member "key":[ints]
bool json_int_array(const std::string& j, const std::string& key,
                    std::vector<int>* out) {
  const std::string pat = "\"" + key + "\":[";
  const std::size_t p = j.find(pat);
  if (p == std::string::npos) return false;
  const std::size_t e = j.find(']', p);
  if (e == std::string::npos) return false;
  const std::string inner = j.substr(p + pat.size(), e - p - pat.size());
  out->clear();
  if (inner.empty()) return true;
  std::size_t pos = 0;
  while (pos <= inner.size()) {
    const std::size_t c = inner.find(',', pos);
    const std::string tok =
        c == std::string::npos ? inner.substr(pos)
                               : inner.substr(pos, c - pos);
    // trim:
    std::size_t b = tok.find_first_not_of(" ");
    if (b == std::string::npos) {
      if (c == std::string::npos) break;
      pos = c + 1;
      continue;
    }
    out->push_back(std::stoi(tok.substr(b)));
    if (c == std::string::npos) break;
    pos = c + 1;
  }
  return true;
}

http::HttpRequest req(const std::string& method, const std::string& path,
                      const std::string& body = "") {
  http::HttpRequest r;
  r.method = method;
  const std::size_t q = path.find('?');
  r.path = q == std::string::npos ? path : path.substr(0, q);
  r.query = q == std::string::npos ? "" : path.substr(q + 1);
  r.body = body;
  return r;
}

// One synchronous turn through the handler; returns the response.
http::HttpResponse turn(http::ServingHttpApi& api, int sid,
                        const std::string& query,
                        const std::string& body) {
  return api.handle(req("POST", "/v1/sessions/" + std::to_string(sid) +
                                    "/turn?" + query, body));
}

}  // namespace

// ---- A. create + turn + second turn (append-only) + parity ---------------

int test_create_turn_parity() {
  cudaStream_t stream0 = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream0));
  Runtime rt(stream0, ServingLimits{-1, -1, 0});

  // ---- create ---------------------------------------------------------
  const http::HttpResponse cr =
      rt.api.handle(req("POST", "/v1/sessions"));
  CHECK_EQ(cr.status, 201);
  long sid = 0;
  CHECK(json_int(cr.body, "session_id", &sid));
  CHECK(sid >= 1);
  const SessionId s = static_cast<SessionId>(sid);
  CHECK(rt.sm.lookup(s) != nullptr);

  // ---- turn 1 (greedy default; NO trim — leading/trailing whitespace +
  //      an embedded NUL are part of the raw-text body) -------------------
  const std::string body1("\n  hello\x00world ", 15);
  const http::HttpResponse t1 =
      turn(rt.api, static_cast<int>(s), "max_new_tokens=4", body1);
  CHECK_EQ(t1.status, 200);
  long rid1 = 0;
  CHECK(json_int(t1.body, "request_id", &rid1));
  std::vector<int> http_ids1;
  CHECK(json_int_array(t1.body, "generated_token_ids", &http_ids1));
  CHECK_EQ(static_cast<int>(http_ids1.size()), 4);
  long ctx1 = -1;
  CHECK(json_int(t1.body, "context_length", &ctx1));

  // The PARITY check: the HTTP ids == the scheduler's COMMITTED ids
  // (generated[0 .. committed_generated) — the adapter never peeks the
  // pending tail):
  const Request* r1 = rt.sched.get(static_cast<RequestId>(rid1));
  CHECK(r1 != nullptr);
  CHECK_EQ(is_terminal(r1->status), true);
  CHECK_EQ(static_cast<int>(r1->committed_generated), 4);
  for (int i = 0; i < 4; ++i) {
    CHECK_EQ(http_ids1[static_cast<std::size_t>(i)],
             r1->generated[static_cast<std::size_t>(i)]);
  }
  // The context grew EXACTLY by the input bytes + the generated ids:
  CHECK_EQ(ctx1, static_cast<long>(body1.size()) + 4);

  // ---- turn 2 (append-only: the context is the previous + new) ---------
  const std::string body2("second turn\n", 12);
  const http::HttpResponse t2 =
      turn(rt.api, static_cast<int>(s), "max_new_tokens=3", body2);
  CHECK_EQ(t2.status, 200);
  std::vector<int> http_ids2;
  CHECK(json_int_array(t2.body, "generated_token_ids", &http_ids2));
  CHECK_EQ(static_cast<int>(http_ids2.size()), 3);
  long ctx2 = -1;
  CHECK(json_int(t2.body, "context_length", &ctx2));
  CHECK_EQ(ctx2, ctx1 + static_cast<long>(body2.size()) + 3);
  // the sequence length agrees with the state manager (the single
  // source of truth):
  const Session* sess = rt.sm.lookup(s);
  CHECK(sess != nullptr);
  const SequenceState* st = rt.mgr.lookup(sess->sequence_id);
  CHECK(st != nullptr);
  CHECK_EQ(st->length, static_cast<int>(ctx2));

  // ---- stats ------------------------------------------------------------
  const http::HttpResponse stt = rt.api.handle(req("GET", "/v1/stats"));
  CHECK_EQ(stt.status, 200);
  long live_sessions = -1, live_requests = -1, admitted = -1;
  CHECK(json_int(stt.body, "live_sessions", &live_sessions));
  CHECK(json_int(stt.body, "live_requests", &live_requests));
  CHECK(json_int(stt.body, "total_admitted_requests", &admitted));
  CHECK_EQ(live_sessions, 1L);
  CHECK_EQ(live_requests, 0L);
  CHECK_EQ(admitted, 2L);

  // ---- unknown session ---------------------------------------------------
  const http::HttpResponse unk =
      turn(rt.api, 999, "max_new_tokens=1", "x");
  CHECK_EQ(unk.status, 404);
  CHECK(unk.body.find("unknown session") != std::string::npos);

  // ---- wrong method / unknown route --------------------------------------
  CHECK_EQ(rt.api.handle(req("GET", "/v1/sessions")).status, 405);
  CHECK_EQ(rt.api.handle(req("GET", "/nope")).status, 404);
  CHECK_EQ(rt.api.handle(req("DELETE", "/v1/stats")).status, 405);

  TEST_PASS("test_serving_http_create_turn_parity");
  return 0;
}

// ---- B. reset + destroy lifecycle -----------------------------------------

int test_reset_destroy() {
  cudaStream_t stream0 = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream0));
  Runtime rt(stream0, ServingLimits{-1, -1, 0});
  const http::HttpResponse cr = rt.api.handle(req("POST", "/v1/sessions"));
  CHECK_EQ(cr.status, 201);
  long sid = 0;
  CHECK(json_int(cr.body, "session_id", &sid));
  const SessionId s = static_cast<SessionId>(sid);

  const http::HttpResponse t1 =
      turn(rt.api, static_cast<int>(s), "max_new_tokens=4", "abcdef");
  CHECK_EQ(t1.status, 200);
  long ctx1 = -1;
  CHECK(json_int(t1.body, "context_length", &ctx1));
  CHECK(ctx1 > 0);

  // ---- reset: same SessionId, the context goes back to 0 ----------------
  const http::HttpResponse rs =
      rt.api.handle(req("POST", "/v1/sessions/" + std::to_string(sid) +
                                    "/reset"));
  CHECK_EQ(rs.status, 200);
  CHECK(json_int(rs.body, "context_length", &ctx1) && ctx1 == 0);
  const Session* sess = rt.sm.lookup(s);
  CHECK(sess != nullptr);  // the SAME session (not a new one)
  const SequenceState* st = rt.mgr.lookup(sess->sequence_id);
  CHECK(st != nullptr);
  CHECK_EQ(st->length, 0);

  // the next turn starts from a FRESH context:
  const http::HttpResponse t2 =
      turn(rt.api, static_cast<int>(s), "max_new_tokens=2", "xy");
  CHECK_EQ(t2.status, 200);
  long ctx2 = -1;
  CHECK(json_int(t2.body, "context_length", &ctx2));
  CHECK_EQ(ctx2, static_cast<long>(2 + 2));  // 2 input + 2 generated

  // ---- destroy -------------------------------------------------------------
  const http::HttpResponse dd = rt.api.handle(
      req("DELETE", "/v1/sessions/" + std::to_string(sid)));
  CHECK_EQ(dd.status, 200);
  CHECK(rt.sm.lookup(s) == nullptr);  // the session is gone
  // a turn on the destroyed session is a 404:
  CHECK_EQ(turn(rt.api, static_cast<int>(s), "max_new_tokens=1", "z").status,
           404);
  // a destroy of the unknown session is a 404:
  CHECK_EQ(rt.api.handle(req("DELETE", "/v1/sessions/" +
                                          std::to_string(sid))).status,
           404);
  // the stats agree:
  const http::HttpResponse stt = rt.api.handle(req("GET", "/v1/stats"));
  long live = -1;
  CHECK(json_int(stt.body, "live_sessions", &live));
  CHECK_EQ(live, 0L);
  long admitted_sessions = -1;
  CHECK(json_int(stt.body, "total_admitted_sessions", &admitted_sessions));
  CHECK_EQ(admitted_sessions, 1L);  // only the FIRST create is managed

  TEST_PASS("test_serving_http_reset_destroy");
  return 0;
}

// ---- C. the streaming endpoint (token events before the terminal; the
//         streamed ids == the terminal ids) ---------------------------------

int test_stream() {
  cudaStream_t stream0 = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream0));
  Runtime rt(stream0, ServingLimits{-1, -1, 0});
  const http::HttpResponse cr = rt.api.handle(req("POST", "/v1/sessions"));
  CHECK_EQ(cr.status, 201);
  long sid = 0;
  CHECK(json_int(cr.body, "session_id", &sid));

  std::string wire;  // everything the sink wrote
  const bool ok = rt.api.handle_stream(
      req("POST", "/v1/sessions/" + std::to_string(sid) +
                     "/turn/stream?max_new_tokens=5", "stream body"),
      [&wire](const std::string& chunk) {
        wire += chunk;
        return true;
      });
  CHECK(ok);  // no disconnect

  // the header block:
  CHECK(wire.rfind("HTTP/1.1 200 OK\r\n", 0) == 0);
  CHECK(wire.find("Content-Type: application/x-ndjson\r\n") !=
        std::string::npos);
  CHECK(wire.find("Content-Length") == std::string::npos);  // close-
                                                            // delimited

  // the NDJSON lines:
  std::vector<std::string> lines;
  {
    std::size_t pos = wire.find("\r\n\r\n");
    CHECK(pos != std::string::npos);
    pos += 4;
    while (pos < wire.size()) {
      const std::size_t e = wire.find('\n', pos);
      const std::size_t end = e == std::string::npos ? wire.size() : e;
      if (end > pos) lines.push_back(wire.substr(pos, end - pos));
      if (e == std::string::npos) break;
      pos = e + 1;
    }
  }
  CHECK_EQ(static_cast<int>(lines.size()), 6);  // 5 tokens + 1 terminal

  std::vector<int> streamed;
  long terminal_rid = -1;
  int terminal_idx = -1;
  for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
    if (lines[static_cast<std::size_t>(i)].rfind(
            "{\"type\":\"token\"", 0) == 0) {
      long tok = -1, trid = -1;
      CHECK(json_int(lines[static_cast<std::size_t>(i)], "token_id", &tok));
      CHECK(json_int(lines[static_cast<std::size_t>(i)], "request_id",
                     &trid));
      if (terminal_rid == -1) terminal_rid = trid;
      CHECK_EQ(trid, terminal_rid);
      streamed.push_back(static_cast<int>(tok));
    } else if (lines[static_cast<std::size_t>(i)].rfind(
                   "{\"type\":\"terminal\"", 0) == 0) {
      terminal_idx = i;
      std::vector<int> final_ids;
      CHECK(json_int_array(lines[static_cast<std::size_t>(i)],
                           "generated_token_ids", &final_ids));
      // the streamed ids == the terminal / final generated ids (the
      // COMMIT-BEFORE-VISIBLE parity):
      CHECK(streamed == final_ids);
      CHECK_EQ(static_cast<int>(final_ids.size()), 5);
      long ctx = -1;
      CHECK(json_int(lines[static_cast<std::size_t>(i)], "context_length",
                     &ctx));
      // 11 input bytes + 5 generated:
      CHECK_EQ(ctx, static_cast<long>(11 + 5));
      CHECK(lines[static_cast<std::size_t>(i)].find(
                "\"deadline_exceeded\":false") != std::string::npos);
    }
  }
  // the terminal is the LAST line (the Phase B order: every token
  // event BEFORE the terminal):
  CHECK_EQ(terminal_idx, 5);
  // and the streamed ids == the scheduler's committed state:
  const Request* r = rt.sched.get(static_cast<RequestId>(terminal_rid));
  CHECK(r != nullptr);
  CHECK_EQ(static_cast<int>(r->committed_generated), 5);
  for (int i = 0; i < 5; ++i) {
    CHECK_EQ(streamed[static_cast<std::size_t>(i)],
             r->generated[static_cast<std::size_t>(i)]);
  }

  TEST_PASS("test_serving_http_stream");
  return 0;
}

// ---- D. the deadline (deterministic via the fake clock) --------------------

int test_deadline() {
  cudaStream_t stream0 = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream0));
  Runtime rt(stream0, ServingLimits{-1, -1, 0});
  const http::HttpResponse cr = rt.api.handle(req("POST", "/v1/sessions"));
  CHECK_EQ(cr.status, 201);
  long sid = 0;
  CHECK(json_int(cr.body, "session_id", &sid));

  // Advance the FAKE clock far past the deadline the handler will
  // compute (real now + 1ms): the controller's deadline check (before
  // the first step) cancels the request deterministically:
  rt.clk.advance_ms(60000);
  const http::HttpResponse d =
      turn(rt.api, static_cast<int>(sid), "deadline_ms=1", "deadline body");
  CHECK_EQ(d.status, 408);
  CHECK(d.body.find("\"deadline_exceeded\":true") != std::string::npos);
  CHECK(d.body.find("\"finish_reason\":\"Cancelled\"") != std::string::npos);
  std::vector<int> ids;
  CHECK(json_int_array(d.body, "generated_token_ids", &ids));
  CHECK_EQ(static_cast<int>(ids.size()), 0);  // no token was committed

  // the session SURVIVES the deadline cancel (it is still live and can
  // take the next turn):
  const http::HttpResponse t2 =
      turn(rt.api, static_cast<int>(sid), "max_new_tokens=2", "ok");
  CHECK_EQ(t2.status, 200);

  TEST_PASS("test_serving_http_deadline");
  return 0;
}

// ---- E. the client-disconnect cleanup ---------------------------------------

int test_disconnect_cleanup() {
  cudaStream_t stream0 = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream0));
  Runtime rt(stream0, ServingLimits{-1, -1, 0});
  const http::HttpResponse cr = rt.api.handle(req("POST", "/v1/sessions"));
  CHECK_EQ(cr.status, 201);
  long sid = 0;
  CHECK(json_int(cr.body, "session_id", &sid));

  // The sink fails on the 2nd token line (a simulated client
  // disconnect mid-stream):
  std::string wire;
  int token_lines = 0;
  const bool ok = rt.api.handle_stream(
      req("POST", "/v1/sessions/" + std::to_string(sid) +
                     "/turn/stream?max_new_tokens=8", "disconnect"),
      [&wire, &token_lines](const std::string& chunk) {
        wire += chunk;
        if (chunk.find("\"type\":\"token\"") != std::string::npos) {
          ++token_lines;
        }
        if (token_lines >= 2) return false;  // the client is GONE
        return true;
      });
  CHECK(!ok);  // a disconnect was detected

  // the PINNED cleanup invariants:
  // (1) the live request quota is released (no leak):
  const ServingStats st = rt.ctrl.stats();
  CHECK_EQ(st.live_requests, 0);
  // (2) the SESSION stays live (a disconnect is NEVER a destroy):
  const SessionId s = static_cast<SessionId>(sid);
  CHECK(rt.sm.lookup(s) != nullptr);
  // (3) the committed state is intact (the sequence kept its
  //     committed length — the pending tail was NOT committed):
  const Session* sess = rt.sm.lookup(s);
  const SequenceState* seq = rt.mgr.lookup(sess->sequence_id);
  CHECK(seq != nullptr);
  CHECK(seq->length >= static_cast<int>(10));  // 10 input bytes committed
  // (4) the session can CONTINUE with the next turn (the pinned
  //     continuation):
  const http::HttpResponse t2 =
      turn(rt.api, static_cast<int>(sid), "max_new_tokens=2", "again");
  CHECK_EQ(t2.status, 200);
  std::vector<int> ids2;
  CHECK(json_int_array(t2.body, "generated_token_ids", &ids2));
  CHECK_EQ(static_cast<int>(ids2.size()), 2);

  TEST_PASS("test_serving_http_disconnect_cleanup");
  return 0;
}


int main() {
  int rc = 0;
  rc |= test_create_turn_parity();
  rc |= test_reset_destroy();
  rc |= test_stream();
  rc |= test_deadline();
  rc |= test_disconnect_cleanup();
  if (rc != 0) {
    std::fprintf(stderr, "test_serving_http: FAILED\n");
    return rc;
  }
  std::fprintf(stderr, "test_serving_http: all [ok]\n");
  return 0;
}
