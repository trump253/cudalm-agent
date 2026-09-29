// CUDALM — v0.9 Phase D: minimal HTTP/1.1 TRANSPORT (Linux POSIX
// sockets, C++17, single-threaded).
//
// Pinned behavior (the v0.9 HTTP facade is deliberately minimal):
//   * ONE listener, a single-threaded accept loop, ONE connection /
//     request at a time, every response ends with the close;
//   * the server suppresses SIGPIPE and every send uses MSG_NOSIGNAL
//     — a client disconnect can never kill the server process;
//   * socket receive / send timeouts bound a single bad client;
//   * the accepted connection gets a SMALL send buffer (the streaming
//     endpoint flushes per event — a disconnected client is detected
//     promptly instead of being absorbed by a large socket buffer);
//   * NO keep-alive, NO HTTP/2, NO TLS, NO request chunked transfer.
//
// Provenance: CUDALM-native (v0.9 Phase D).

#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "tools/common/http_protocol.h"

#if defined(__linux__)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace cudalm {
namespace http {

class HttpTransport {
 public:
  HttpTransport();
  ~HttpTransport();

  HttpTransport(const HttpTransport&) = delete;
  HttpTransport& operator=(const HttpTransport&) = delete;

  // The one-time process setup: suppress SIGPIPE (a client disconnect
  // must never kill the server). Safe to call more than once.
  static void init();

  // Bind + listen on host:port (default host 127.0.0.1 — NEVER 0.0.0.0
  // by default; an explicit --host 0.0.0.0 is the user's choice).
  bool listen(const std::string& host, int port, int backlog,
              std::string* error);

  // Accept ONE connection (poll-bounded so the loop can check
  // stop conditions); returns -1 on timeout / error.
  int accept_one(std::chrono::milliseconds timeout) const;

  // Read ONE complete request (headers + exactly the Content-Length
  // body bytes) from a connection; returns the PARSE result. The raw
  // capture is bounded by the limits (an over-limit header block or a
  // Content-Length over the body cap is rejected WITHOUT reading the
  // whole payload). `error` distinguishes a clean parse rejection
  // (res.ok == false, answer res.status) from a transport failure
  // (res.ok == false, res.status == 0, answer nothing and close).
  bool read_request(int fd, const HttpLimits& limits,
                    HttpParseResult* res) const;

  // Binary-safe send of the FULL buffer (MSG_NOSIGNAL; a client
  // disconnect returns false instead of killing the process).
  static bool send_all(int fd, const void* data, std::size_t n);

  // Serialize + send a complete response (the body is written in full —
  // never through %s).
  static bool send_response(int fd, const HttpResponse& resp);

  static void close_fd(int fd);

 private:
  int listen_fd_ = -1;
};

}  // namespace http
}  // namespace cudalm
