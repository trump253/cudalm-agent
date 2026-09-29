// CUDALM — v0.9 Phase D: minimal HTTP/1.1 PROTOCOL layer (header-only,
// CPU-only, no sockets, no third-party dependencies).
//
// The protocol SCOPE (pinned; everything else is rejected fail-loud):
//   * supported: GET / POST / DELETE, a single Content-Length request
//     body, "Connection: close" responses;
//   * NOT supported (rejected): request Transfer-Encoding (chunked or
//     otherwise), keep-alive, HTTP/2, TLS, WebSocket;
//   * the parsed body is a raw byte vector (std::string, NUL-safe) —
//     byte-exact preservation, NO trimming / decoding / re-encoding.
//
// Provenance: CUDALM-native (v0.9 Phase D).

#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace cudalm {
namespace http {

// A parsed HTTP/1.1 request (the body is raw bytes — NUL-safe).
struct HttpRequest {
  std::string method;  // "GET" / "POST" / "DELETE"
  std::string path;    // the target without the query
  std::string query;   // the raw query string (empty when none)
  std::map<std::string, std::string> headers;  // lower-cased names
  std::string body;  // exactly the Content-Length bytes
};

// An HTTP response to be serialized (binary-safe; the body is raw
// bytes and is written in full — never through %s).
struct HttpResponse {
  int status = 500;
  std::string reason = "Internal Server Error";
  std::map<std::string, std::string> headers;
  std::string body;

  HttpResponse with_header(const std::string& name,
                           const std::string& value) {
    headers[name] = value;
    return *this;
  }
  // A stable-JSON error body: {"error":"<escaped message>"}
  static HttpResponse json_error(int status, const std::string& reason,
                                 const std::string& message);
  // A plain JSON body (the caller builds the JSON object).
  static HttpResponse json(int status, const std::string& body);
};

// A request parse result. `ok` == true -> `req` is valid. `ok` == false
// -> `status` is the HTTP status to answer (400 malformed / 413 too
// large / 415 unsupported) and `message` is a stable, client-safe
// description (never echoes raw input).
struct HttpParseResult {
  bool ok = false;
  int status = 400;
  std::string message = "malformed request";
  HttpRequest req;
};

// Limits for a single request.
struct HttpLimits {
  std::size_t max_header_bytes = 64 * 1024;  // request line + headers
  std::size_t max_body_bytes = 64 * 1024 * 1024;  // the Content-Length body
};

// Parse a COMPLETE raw request (headers + the full body) captured by
// the transport. Pure function — no I/O (unit-testable).
//
// Pinned rejections:
//   * a malformed request line (not "METHOD SP TARGET SP HTTP/1.1")
//     or an unsupported version -> 400;
//   * an empty / non-absolute-form target -> 400;
//   * a missing, non-numeric, negative or non-integer Content-Length
//     -> 400 (a POST/DELETE with no Content-Length has an empty body —
//     allowed; a GET with a body reads + ignores it);
//   * any Transfer-Encoding request header -> 415 (request chunked
//     transfer is NOT supported);
//   * the header block over max_header_bytes -> 413;
//   * a Content-Length over max_body_bytes -> 413;
//   * the captured body shorter than the Content-Length -> 400
//     (an incomplete request — the transport hit EOF mid-body).
HttpParseResult parse_http_request(const std::string& raw,
                                   const HttpLimits& limits);

// Serialize a response to the exact wire bytes: the status line, the
// headers (sorted by name — deterministic), an empty line, and the
// body VERBATIM (binary-safe, NUL-safe). A Content-Length header is
// added automatically from the body size UNLESS the caller already
// set one (the streaming endpoint omits it: the body is terminated by
// the close).
std::string format_response(const HttpResponse& resp);

// Escape an arbitrary byte string as a JSON string VALUE (without the
// surrounding quotes): quotes, backslashes, NUL and every other
// control byte are escaped (\u00XX) — the result is always embeddable
// in a JSON document (the generated text is never a C string).
std::string json_escape(const std::string& bytes);

// Parse a raw query string ("a=1&b=2") into a map. Pinned: literal
// values (no percent-decoding in this phase), an empty value is
// allowed ("a=" -> ""), a repeated key keeps the LAST value, and a
// segment without '=' maps an empty value (fail-loud validation of
// the values is the handler's job). Returns false on a malformed
// input (a '%' is not supported — fail loud).
bool parse_query(const std::string& raw,
                 std::map<std::string, std::string>* out);

}  // namespace http
}  // namespace cudalm
