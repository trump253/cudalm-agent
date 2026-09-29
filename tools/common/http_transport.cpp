// CUDALM — v0.9 Phase D: minimal HTTP/1.1 transport (see the header
// for the pinned behavior).

#include "tools/common/http_transport.h"

#if !defined(__linux__)
#error "the CUDALM HTTP transport is Linux-only (POSIX sockets)"
#endif

#include <cerrno>
#include <cstdio>

namespace cudalm {
namespace http {

namespace {

// The per-connection socket timeouts (a single bad client can only
// stall ONE request for this long — it can never wedge the server).
constexpr std::chrono::milliseconds kRcvTimeout{60000};
constexpr std::chrono::milliseconds kSndTimeout{60000};
// The small send buffer: the streaming endpoint flushes per event, so
// a disconnected client is detected within ~4KB instead of being
// absorbed by the (large, auto-tuned) default socket buffer.
constexpr int kSendBufferSize = 4096;

void set_timeouts(int fd) {
  const timeval rcv{
      static_cast<long>(kRcvTimeout.count() / 1000),
      static_cast<suseconds_t>((kRcvTimeout.count() % 1000) * 1000)};
  const timeval snd{
      static_cast<long>(kSndTimeout.count() / 1000),
      static_cast<suseconds_t>((kSndTimeout.count() % 1000) * 1000)};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rcv, sizeof(rcv));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &snd, sizeof(snd));
}

}  // namespace

HttpTransport::HttpTransport() = default;

HttpTransport::~HttpTransport() { close_fd(listen_fd_); }

void HttpTransport::init() {
  // A client disconnect can never kill the server process:
  signal(SIGPIPE, SIG_IGN);
}

bool HttpTransport::listen(const std::string& host, int port, int backlog,
                           std::string* error) {
  close_fd(listen_fd_);
  listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    if (error != nullptr) *error = "socket() failed";
    return false;
  }
  int one = 1;
  setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  addrinfo hints;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const std::string port_str = std::to_string(port);
  if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0 ||
      res == nullptr) {
    if (error != nullptr) *error = "cannot resolve the host/port";
    close_fd(listen_fd_);
    return false;
  }
  const bool bound =
      ::bind(listen_fd_, res->ai_addr, res->ai_addrlen) == 0 &&
      ::listen(listen_fd_, backlog) == 0;
  freeaddrinfo(res);
  if (!bound) {
    if (error != nullptr) *error = "bind/listen failed";
    close_fd(listen_fd_);
    return false;
  }
  return true;
}

int HttpTransport::accept_one(std::chrono::milliseconds timeout) const {
  if (listen_fd_ < 0) return -1;
  pollfd pfd{listen_fd_, POLLIN, 0};
  const int pr = poll(&pfd, 1, static_cast<int>(timeout.count()));
  if (pr <= 0) return -1;  // timeout / error / reaped signal
  sockaddr_in addr;
  socklen_t alen = sizeof(addr);
  return accept(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &alen);
}

bool HttpTransport::read_request(int fd, const HttpLimits& limits,
                                 HttpParseResult* res) const {
  res->ok = false;
  res->status = 0;  // 0 = a TRANSPORT failure (answer nothing, close)
  res->message = "connection closed";

  set_timeouts(fd);
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &kSendBufferSize,
             sizeof(kSendBufferSize));

  // ---- the header block (bounded) --------------------------------
  std::string head;
  char buf[4096];
  std::size_t header_end = std::string::npos;
  for (;;) {
    const ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) return false;  // EOF / timeout / error mid-headers
    head.append(buf, static_cast<std::size_t>(n));
    if (head.size() > limits.max_header_bytes) {
      res->status = 413;
      res->message = "request headers too large";
      return true;  // a clean parse rejection (answer 413)
    }
    header_end = head.find("\r\n\r\n");
    if (header_end != std::string::npos) break;
  }

  // ---- the body (exactly the Content-Length bytes, bounded) ------
  // A pre-scan for the Content-Length lets us reject an over-limit
  // payload WITHOUT reading it in full:
  {
    const std::size_t he = header_end;
    const std::size_t first_eol = head.find("\r\n");
    std::size_t pos = first_eol == std::string::npos ? he : first_eol + 2;
    std::size_t content_length = 0;
    while (pos < he) {
      const std::size_t eol = head.find("\r\n", pos);
      const std::string line =
          eol == std::string::npos ? head.substr(pos)
                                   : head.substr(pos, eol - pos);
      const bool last_line = eol == std::string::npos;
      if (!last_line) pos = eol + 2;
      const std::size_t colon = line.find(':');
      if (colon != std::string::npos) {
        std::string name = line.substr(0, colon);
        for (char& c : name) {
          if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        }
        if (name == "content-length") {
          std::string v = line.substr(colon + 1);
          if (!v.empty() && v[0] == ' ') v.erase(0, 1);
          if (!v.empty()) {
            std::size_t val = 0;
            bool digits = true;
            for (char c : v) {
              if (c < '0' || c > '9') {
                digits = false;
                break;
              }
              val = val * 10 + static_cast<std::size_t>(c - '0');
            }
            if (digits && val > limits.max_body_bytes) {
              res->status = 413;
              res->message = "request body too large";
              return true;
            }
            if (digits) content_length = val;
          }
        }
      }
      if (last_line) break;
    }
    std::string body;
    body.reserve(std::min(content_length, limits.max_body_bytes));
    // The header recv may have OVER-READ the first body bytes into
    // `head` (TCP delivers what it delivers): they must be RE-USED,
    // not re-requested from the socket:
    const std::size_t body_start = header_end + 4;
    if (head.size() > body_start) {
      body.append(head, body_start, head.size() - body_start);
    }
    std::size_t got = body.size();
    if (got > content_length) {
      // More bytes were delivered than the Content-Length promised
      // (a pipelined next request — unsupported: Connection: close
      // means one request per connection):
      res->status = 400;
      res->message = "unsupported pipelined request";
      return true;
    }
    while (got < content_length) {
      const std::size_t want =
          std::min(sizeof(buf), content_length - got);
      const ssize_t n = recv(fd, buf, want, 0);
      if (n <= 0) return false;  // EOF / timeout mid-body
      body.append(buf, static_cast<std::size_t>(n));
      got += static_cast<std::size_t>(n);
    }
    *res = parse_http_request(head + body, limits);
    return true;
  }
}

bool HttpTransport::send_all(int fd, const void* data, std::size_t n) {
  const char* p = static_cast<const char*>(data);
  std::size_t off = 0;
  while (off < n) {
    const ssize_t w = send(fd, p + off, n - off, MSG_NOSIGNAL);
    if (w <= 0) return false;  // EPIPE / RST / timeout — fail loud
    off += static_cast<std::size_t>(w);
  }
  return true;
}

bool HttpTransport::send_response(int fd, const HttpResponse& resp) {
  const std::string wire = format_response(resp);
  return send_all(fd, wire.data(), wire.size());
}

void HttpTransport::close_fd(int fd) {
  if (fd >= 0) {
    shutdown(fd, SHUT_RDWR);
    ::close(fd);
  }
}

}  // namespace http
}  // namespace cudalm
