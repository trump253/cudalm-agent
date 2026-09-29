// CUDALM — v0.9 Phase D: minimal HTTP/1.1 protocol layer (see the
// header for the pinned scope).

#include "tools/common/http_protocol.h"

namespace cudalm {
namespace http {

namespace {

std::string lower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

const char* status_reason(int status) {
  switch (status) {
    case 200: return "OK";
    case 201: return "Created";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 409: return "Conflict";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 500: return "Internal Server Error";
    default: return "Error";
  }
}

}  // namespace

HttpResponse HttpResponse::json_error(int status, const std::string& reason,
                                      const std::string& message) {
  HttpResponse r;
  r.status = status;
  r.reason = reason.empty() ? status_reason(status) : reason;
  r.headers["Content-Type"] = "application/json";
  r.body = "{\"error\":\"" + json_escape(message) + "\"}";
  return r;
}

HttpResponse HttpResponse::json(int status, const std::string& body) {
  HttpResponse r;
  r.status = status;
  r.reason = status_reason(status);
  r.headers["Content-Type"] = "application/json";
  r.body = body;
  return r;
}

HttpParseResult parse_http_request(const std::string& raw,
                                   const HttpLimits& limits) {
  HttpParseResult res;

  // ---- the header block ------------------------------------------------
  const std::size_t header_end = raw.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    res.status = 400;
    res.message = "malformed request: the header block is incomplete";
    return res;
  }
  if (header_end > limits.max_header_bytes) {
    res.status = 413;
    res.message = "request headers too large";
    return res;
  }
  const std::string head = raw.substr(0, header_end);

  // ---- the request line ------------------------------------------------
  const std::size_t line_end = head.find("\r\n");
  const std::string request_line =
      line_end == std::string::npos ? head : head.substr(0, line_end);
  std::size_t sp1 = request_line.find(' ');
  if (sp1 == std::string::npos || sp1 == 0) {
    res.status = 400;
    res.message = "malformed request line";
    return res;
  }
  const std::string method = request_line.substr(0, sp1);
  std::size_t sp2 = request_line.find(' ', sp1 + 1);
  if (sp2 == std::string::npos || sp2 == sp1 + 1) {
    res.status = 400;
    res.message = "malformed request line";
    return res;
  }
  const std::string target = request_line.substr(sp1 + 1, sp2 - sp1 - 1);
  const std::string version = request_line.substr(sp2 + 1);
  if (version != "HTTP/1.1") {
    res.status = 400;
    res.message = "unsupported HTTP version";
    return res;
  }
  if (method != "GET" && method != "POST" && method != "DELETE") {
    // The ROUTER answers wrong-but-valid methods with 405 — the parser
    // accepts any token here; an empty / control-character method is
    // malformed:
    bool valid_token = !method.empty();
    for (char c : method) {
      if (static_cast<unsigned char>(c) < 0x21 ||
          static_cast<unsigned char>(c) > 0x7e || c == ' ' || c == '/') {
        valid_token = false;
        break;
      }
    }
    if (!valid_token) {
      res.status = 400;
      res.message = "malformed request line";
      return res;
    }
  }
  if (target.empty() || target[0] != '/') {
    res.status = 400;
    res.message = "malformed request target";
    return res;
  }
  std::string path = target;
  std::string query;
  const std::size_t qmark = target.find('?');
  if (qmark != std::string::npos) {
    path = target.substr(0, qmark);
    query = target.substr(qmark + 1);
  }

  // ---- the headers -------------------------------------------------------
  std::size_t pos = line_end == std::string::npos ? head.size() : line_end + 2;
  bool content_length_set = false;
  std::size_t content_length = 0;
  while (pos < head.size()) {
    const std::size_t eol = head.find("\r\n", pos);
    const std::string line =
        eol == std::string::npos ? head.substr(pos)
                                 : head.substr(pos, eol - pos);
    pos = eol == std::string::npos ? head.size() : eol + 2;
    if (line.empty()) {
      res.status = 400;
      res.message = "malformed header";
      return res;
    }
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos || colon == 0) {
      res.status = 400;
      res.message = "malformed header";
      return res;
    }
    std::string name = lower(line.substr(0, colon));
    std::string value = line.substr(colon + 1);
    if (!value.empty() && value[0] == ' ') value.erase(0, 1);
    if (name == "transfer-encoding") {
      res.status = 415;
      res.message = "request Transfer-Encoding is not supported "
                    "(chunked transfer is out of scope)";
      return res;
    }
    if (name == "content-length") {
      if (content_length_set || value.empty()) {
        res.status = 400;
        res.message = "malformed Content-Length";
        return res;
      }
      for (char c : value) {
        if (c < '0' || c > '9') {
          res.status = 400;
          res.message = "malformed Content-Length";
          return res;
        }
      }
      std::size_t v = 0;
      for (char c : value) {
        v = v * 10 + static_cast<std::size_t>(c - '0');
        if (v > limits.max_body_bytes) {  // overflow-safe + the body cap
          res.status = 413;
          res.message = "request body too large";
          return res;
        }
      }
      content_length = v;
      content_length_set = true;
    }
    res.req.headers[name] = value;  // the LAST value wins (deterministic)
  }
  res.req.method = method;
  res.req.path = path;
  res.req.query = query;

  // ---- the body ----------------------------------------------------------
  const std::size_t body_avail = raw.size() - header_end - 4;
  if (body_avail < content_length) {
    res.status = 400;
    res.message = "incomplete request body";
    return res;
  }
  res.req.body = raw.substr(header_end + 4, content_length);
  res.ok = true;
  res.status = 200;
  res.message.clear();
  return res;
}

std::string format_response(const HttpResponse& resp) {
  std::string out;
  out.reserve(256 + resp.body.size());
  out += "HTTP/1.1 ";
  out += std::to_string(resp.status);
  out += ' ';
  out += resp.reason.empty() ? status_reason(resp.status) : resp.reason;
  out += "\r\n";
  bool content_length_set = false;
  for (const auto& kv : resp.headers) {
    if (kv.first == "Content-Length") content_length_set = true;
    out += kv.first;
    out += ": ";
    out += kv.second;
    out += "\r\n";
  }
  if (!content_length_set) {
    out += "Content-Length: ";
    out += std::to_string(resp.body.size());
    out += "\r\n";
  }
  out += "Connection: close\r\n";
  out += "\r\n";
  out += resp.body;  // VERBATIM (binary-safe, NUL-safe)
  return out;
}

std::string json_escape(const std::string& bytes) {
  std::string out;
  out.reserve(bytes.size() + 8);
  for (const unsigned char c : bytes) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          static const char* hex = "0123456789abcdef";
          out += "\\u00";
          out += hex[(c >> 4) & 0xF];
          out += hex[c & 0xF];
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

bool parse_query(const std::string& raw,
                 std::map<std::string, std::string>* out) {
  if (out == nullptr) return false;
  out->clear();
  if (raw.empty()) return true;
  std::size_t pos = 0;
  while (pos <= raw.size()) {
    const std::size_t amp = raw.find('&', pos);
    const std::string seg = amp == std::string::npos
                                ? raw.substr(pos)
                                : raw.substr(pos, amp - pos);
    if (!seg.empty()) {
      if (seg.find('%') != std::string::npos) return false;  // fail loud
      const std::size_t eq = seg.find('=');
      if (eq == std::string::npos) {
        (*out)[seg] = "";
      } else {
        if (eq == 0) return false;  // an empty key is malformed
        (*out)[seg.substr(0, eq)] = seg.substr(eq + 1);
      }
    }
    if (amp == std::string::npos) break;
    pos = amp + 1;
  }
  return true;
}

}  // namespace http
}  // namespace cudalm
