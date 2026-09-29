// CUDALM — v0.9 Phase D: cudalm-server — the minimal HTTP serving
// runtime.
//
// The SINGLE runtime chain (pinned — the v0.8 text-demo facade is
// FORBIDDEN in this executable):
//
//   Qwen35Model -> ModelForwarder -> Qwen35StateManager ->
//   SessionManager -> Scheduler -> ServingController -> HTTP handler
//
// Pinned limitations (deliberate — documented, not hidden):
//   * Qwen3.5-0.8B-BASE: raw-text completion. There is NO official
//     chat template for the base model — the request body bytes are
//     encoded VERBATIM (no trim / separator / special tokens);
//   * single-threaded: ONE HTTP request is handled at a time, ONE
//     CUDA stream — this is NOT a concurrent web server (the
//     scheduler / controller keep their validated batching under the
//     hood);
//   * minimal HTTP/1.1: Connection: close only; no keep-alive, no
//     HTTP/2, no TLS / auth, no request chunked transfer;
//   * the API is CUDALM's own minimal API — NOT OpenAI-compatible;
//   * no incremental text deltas: the streaming endpoint emits
//     token-id events + the full generated text ONCE at the terminal.
//
// Provenance: CUDALM-native (v0.9 Phase D).

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/qwen35_tokenizer.h"
#include "cudalm/scheduler.h"
#include "cudalm/sampling.h"
#include "cudalm/session.h"
#include "cudalm/serving_controller.h"
#include "cudalm/weight_loader_v2.h"
#include "tools/common/http_protocol.h"
#include "tools/common/http_serving_handler.h"
#include "tools/common/http_transport.h"

using namespace cudalm;

namespace {

struct ServerOptions {
  std::string model;
  std::string tokenizer;
  std::string host = "127.0.0.1";  // NEVER 0.0.0.0 by default
  int port = 8080;
  int page_tokens = 2;
  int pages = 48;
  int slots = 2;
  int max_sessions = -1;  // -1 unlimited, 0 zero capacity, N>0 = N
  int max_live_requests = -1;
  bool ttl_set = false;
  long ttl_ms = 0;
  bool lru_on_pressure = false;
  bool help = false;
};

const char* kUsage =
    "usage: cudalm-server --model <full_model.cudalm> "
    "--tokenizer <tokenizer.cudaltk>\n"
    "                     [--host 127.0.0.1] [--port 8080]\n"
    "                     [--page-tokens 2] [--pages 48] [--slots 2]\n"
    "                     [--max-sessions -1] [--max-live-requests -1]\n"
    "                     [--session-ttl-ms <n>] [--lru-on-pressure]\n"
    "\n"
    "  --session-ttl-ms <n> : enable the idle TTL (n == 0 = immediately\n"
    "                         eligible once idle); ABSENT = disabled\n"
    "  --lru-on-pressure    : enable the LRU-on-session-pressure\n"
    "                         admission recovery\n";

bool take_value(int argc, char** argv, int* i, std::string* out) {
  if (*i + 1 >= argc) return false;
  *out = argv[++(*i)];
  return true;
}

bool parse_args(int argc, char** argv, ServerOptions* o, std::string* err) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    std::string v;
    if (a == "--help" || a == "-h") {
      o->help = true;
    } else if (a == "--model") {
      if (!take_value(argc, argv, &i, &o->model)) return false;
    } else if (a == "--tokenizer") {
      if (!take_value(argc, argv, &i, &o->tokenizer)) return false;
    } else if (a == "--host") {
      if (!take_value(argc, argv, &i, &o->host)) return false;
    } else if (a == "--port") {
      if (!take_value(argc, argv, &i, &v)) return false;
      o->port = std::atoi(v.c_str());
    } else if (a == "--page-tokens") {
      if (!take_value(argc, argv, &i, &v)) return false;
      o->page_tokens = std::atoi(v.c_str());
    } else if (a == "--pages") {
      if (!take_value(argc, argv, &i, &v)) return false;
      o->pages = std::atoi(v.c_str());
    } else if (a == "--slots") {
      if (!take_value(argc, argv, &i, &v)) return false;
      o->slots = std::atoi(v.c_str());
    } else if (a == "--max-sessions") {
      if (!take_value(argc, argv, &i, &v)) return false;
      o->max_sessions = std::atoi(v.c_str());
    } else if (a == "--max-live-requests") {
      if (!take_value(argc, argv, &i, &v)) return false;
      o->max_live_requests = std::atoi(v.c_str());
    } else if (a == "--session-ttl-ms") {
      if (!take_value(argc, argv, &i, &v)) return false;
      o->ttl_set = true;
      o->ttl_ms = std::atol(v.c_str());
    } else if (a == "--lru-on-pressure") {
      o->lru_on_pressure = true;
    } else {
      *err = "unknown option: " + a;
      return false;
    }
  }
  if (!o->help && (o->model.empty() || o->tokenizer.empty())) {
    *err = "--model and --tokenizer are required";
    return false;
  }
  if (o->port <= 0 || o->port > 65535) {
    *err = "invalid --port";
    return false;
  }
  if (o->page_tokens < 1 || o->pages < 0 || o->slots < 0) {
    *err = "invalid pool capacities";
    return false;
  }
  if (o->max_sessions < -1) {
    *err = "invalid --max-sessions";
    return false;
  }
  if (o->max_live_requests < -1) {
    *err = "invalid --max-live-requests";
    return false;
  }
  if (o->ttl_set && o->ttl_ms < 0) {
    *err = "invalid --session-ttl-ms";
    return false;
  }
  return true;
}

// The native tokenizer behind the handler's TextCodec seam.
class TokenizerCodec final : public http::TextCodec {
 public:
  explicit TokenizerCodec(const Qwen35Tokenizer& tok) : tok_(tok) {}

  Status encode(const std::string& utf8, std::vector<int>* out) const override {
    std::vector<std::uint32_t> ids;
    const Status s = tok_.encode(utf8, &ids);
    if (!s.ok) return s;
    out->assign(ids.begin(), ids.end());
    return Status::ok_status();
  }

  Status decode(const std::vector<int>& ids, std::string* out) const override {
    std::vector<std::uint32_t> u32(ids.begin(), ids.end());
    return tok_.decode(u32.data(), u32.size(), false, out);
  }

  int eos_token_id() const override {
    return static_cast<int>(tok_.eos_token_id());
  }

 private:
  const Qwen35Tokenizer& tok_;
};

}  // namespace

int main(int argc, char** argv) {
  ServerOptions opts;
  std::string err;
  if (!parse_args(argc, argv, &opts, &err)) {
    std::fprintf(stderr, "cudalm-server: %s\n%s\n", err.c_str(), kUsage);
    return 2;
  }
  if (opts.help) {
    std::fputs(kUsage, stdout);
    return 0;
  }

  // ---- the CUDA runtime -------------------------------------------------
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "cudalm-server: CUDA unavailable: %s\n",
                 cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    std::fprintf(stderr, "cudalm-server: cudaStreamCreate failed\n");
    return 1;
  }

  WeightFileV2 file;
  Status s = WeightFileV2::load(opts.model, &file);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-server: model load failure: %s\n",
                 s.message.c_str());
    return 1;
  }
  Qwen35Model model;
  s = Qwen35Model::load(file, stream, &model);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-server: model load failure: %s\n",
                 s.message.c_str());
    return 1;
  }
  std::unique_ptr<Qwen35Tokenizer> tokenizer;
  s = Qwen35Tokenizer::load(opts.tokenizer, &tokenizer);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-server: tokenizer load failure: %s\n",
                 s.message.c_str());
    return 1;
  }

  // ---- the SINGLE runtime chain (pinned) --------------------------------
  Qwen35StateManager mgr(model.config(), opts.page_tokens, opts.pages,
                         opts.slots, stream);
  SessionManager sessions(mgr);
  ModelForwarder fwd(model);
  Scheduler sched(fwd, mgr, stream, &sessions);

  SessionEvictionPolicy policy;
  if (opts.ttl_set) {
    policy = policy.with_idle_ttl(std::chrono::milliseconds(opts.ttl_ms));
  }
  if (opts.lru_on_pressure) {
    policy = policy.with_lru_on_session_pressure(true);
  }
  ServingController ctrl(
      sched, sessions,
      ServingLimits{opts.max_sessions, opts.max_live_requests, 0},
      /*clock=*/nullptr, policy);

  TokenizerCodec codec(*tokenizer);
  http::ServingHttpDeps deps;
  deps.ctrl = &ctrl;
  deps.sched = &sched;
  deps.sessions = &sessions;
  deps.mgr = &mgr;
  deps.codec = &codec;
  http::ServingHttpApi api(deps);

  // ---- the listener -------------------------------------------------------
  http::HttpTransport::init();  // suppress SIGPIPE (pinned)
  http::HttpTransport transport;
  std::string lerr;
  if (!transport.listen(opts.host, opts.port, 8, &lerr)) {
    std::fprintf(stderr, "cudalm-server: listen failed: %s\n", lerr.c_str());
    return 1;
  }

  const std::string ttl_str =
      opts.ttl_set
          ? ("idle TTL enabled: " + std::to_string(opts.ttl_ms) + " ms")
          : std::string("idle TTL disabled");
  const char* lru_str = opts.lru_on_pressure ? "enabled" : "disabled";
  std::fprintf(stderr,
               "[cudalm-server] listening on %s:%d (single-threaded; one "
               "HTTP request at a time; one CUDA stream)\n",
               opts.host.c_str(), opts.port);
  std::fprintf(stderr,
               "[cudalm-server] model: %d layers, vocab %d, max_seq_len %d "
               "(Qwen3.5-0.8B-Base)\n",
               model.num_layers(), model.config().vocab_size,
               model.config().max_seq_len);
  std::fprintf(stderr,
               "[cudalm-server] TEXT CONTRACT: raw-text completion on a "
               "BASE model — NO official chat template; request body bytes "
               "are encoded VERBATIM (no trim / separator / special "
               "tokens)\n");
  std::fprintf(stderr,
               "[cudalm-server] API: CUDALM minimal HTTP API — NOT "
               "OpenAI-compatible; no TLS / auth; no HTTP/2; no request "
               "chunked transfer; no incremental text deltas (the stream "
               "emits token-id events + the full text at the terminal)\n");
  std::fprintf(stderr,
               "[cudalm-server] policy: max_sessions=%d, "
               "max_live_requests=%d, %s, LRU-on-pressure: %s\n",
               opts.max_sessions, opts.max_live_requests, ttl_str.c_str(),
               lru_str);

  // ---- the single-threaded accept loop ------------------------------------
  const http::HttpLimits limits;  // the pinned defaults
  for (;;) {
    const int fd = transport.accept_one(std::chrono::milliseconds(200));
    if (fd < 0) continue;
    http::HttpParseResult res;
    std::fprintf(stderr, "[dbg] %f accepted fd=%d\n",
                 (double)std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count() / 1e6,
                 fd);
    if (!transport.read_request(fd, limits, &res)) {
      std::fprintf(stderr, "[dbg] %f TRANSPORT READ FAILURE (close)\n",
                   (double)std::chrono::duration_cast<
                       std::chrono::microseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count() / 1e6);
      http::HttpTransport::close_fd(fd);  // a clean transport failure
      continue;
    }
    std::fprintf(stderr, "[dbg] parsed ok=%d status=%d method=%s path=%s "
                 "body=%zu\n",
                 (int)res.ok, res.status, res.req.method.c_str(),
                 res.req.path.c_str(), res.req.body.size());
    if (!res.ok) {
      (void)http::HttpTransport::send_response(
          fd, http::HttpResponse::json_error(res.status, "", res.message));
      http::HttpTransport::close_fd(fd);
      continue;
    }
    const std::string stream_suffix = "/turn/stream";
    const bool is_stream =
        res.req.method == "POST" &&
        res.req.path.size() >= stream_suffix.size() &&
        res.req.path.compare(res.req.path.size() - stream_suffix.size(),
                             stream_suffix.size(), stream_suffix) == 0;
    std::fprintf(stderr, "[dbg] handler start (stream=%d)\n",
                 (int)is_stream);
    if (is_stream) {
      // The handler writes the header block + the NDJSON events; a
      // false return = the client disconnected (the handler already
      // performed the cancel + drain/reap cleanup; the session stays
      // live):
      (void)api.handle_stream(res.req,
                              [&fd](const std::string& chunk) {
                                return http::HttpTransport::send_all(
                                    fd, chunk.data(), chunk.size());
                              });
    } else {
      const http::HttpResponse resp = api.handle(res.req);
      std::fprintf(stderr, "[dbg] handle done status=%d, sending\n",
                   resp.status);
      (void)http::HttpTransport::send_response(fd, resp);
      std::fprintf(stderr, "[dbg] sent, closing\n");
    }
    http::HttpTransport::close_fd(fd);
  }
  return 0;
}
