// CUDALM — v0.9 Phase D: the minimal HTTP/1.1 PROTOCOL CPU gate
// (NO sockets, NO model, NO checkpoint — the pure parser / formatter /
// JSON-escape layer, table-driven).
//
// Covers the pinned protocol contract:
//   * a valid GET (method / path / query split, the header map);
//   * a valid POST + Content-Length (the body is preserved EXACTLY —
//     leading / trailing whitespace and EMBEDDED NUL bytes);
//   * a malformed request line -> 400;
//   * a malformed Content-Length -> 400;
//   * an oversized header block -> 413;
//   * an oversized body (Content-Length over the cap) -> 413;
//   * an unsupported request Transfer-Encoding -> 415;
//   * a wrong-but-valid method -> accepted by the parser (the ROUTER
//     answers 405 — verified in the serving HTTP contract gate);
//   * an unknown path -> accepted by the parser (the ROUTER answers
//     404 — verified in the serving HTTP contract gate);
//   * the JSON escaping (quotes / backslash / NUL / control bytes);
//   * the query parsing (literal values; fail-loud on '%').
//
// Provenance: CUDALM-native (v0.9 Phase D).

#include "../../tests/common/check.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "tools/common/http_protocol.h"

using namespace cudalm;
using namespace cudalm::http;

namespace {

// NOTE: every string with an embedded NUL is built from a char
// initializer list — the C hex-escape pitfall ("a\x00b" is the THREE
// bytes 0x61 0x0B 0x00 because \x consumes ALL following hex digits).
const std::string& nul3() {
  static const std::string v({'a', '\0', 'b'});
  return v;
}

// ---- the table-driven parse cases --------------------------------------

struct ParseCase {
  const char* name;
  std::string raw;
  bool expect_ok;
  int expect_status;  // valid when !expect_ok
};

int test_parse_table() {
  HttpLimits lim;
  lim.max_header_bytes = 512;
  lim.max_body_bytes = 64;

  // "  ab\ncd" is SEVEN bytes (two spaces + a b + \n + c d):
  const ParseCase cases[] = {
      {"valid GET",
       "GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n",
       true, 0},
      {"valid GET with query",
       "GET /v1/sessions/3/turn?max_new_tokens=8 HTTP/1.1\r\n\r\n",
       true, 0},
      {"valid POST + Content-Length (body exact)",
       "POST /v1/sessions/1/turn HTTP/1.1\r\n"
       "Content-Length: 7\r\n"
       "Connection: close\r\n"
       "\r\n"
       "  ab\ncd",
       true, 0},
      {"malformed request line (method contains '/')",
       "GET/healthz HTTP/1.1\r\n\r\n",
       false, 400},
      {"malformed request line (missing target)",
       "GET  HTTP/1.1\r\n\r\n",
       false, 400},
      {"malformed Content-Length (non-numeric)",
       "POST /x HTTP/1.1\r\nContent-Length: abc\r\n\r\n",
       false, 400},
      {"malformed Content-Length (empty)",
       "POST /x HTTP/1.1\r\nContent-Length:\r\n\r\n",
       false, 400},
      {"duplicate Content-Length",
       "POST /x HTTP/1.1\r\nContent-Length: 1\r\n"
       "Content-Length: 2\r\n\r\na",
       false, 400},
      {"incomplete body (shorter than Content-Length)",
       "POST /x HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc",
       false, 400},
      {"oversized header block -> 413",
       std::string("GET / HTTP/1.1\r\nX-Pad: ") +
           std::string(600, 'a') + "\r\n\r\n",
       false, 413},
      {"oversized body (Content-Length over cap) -> 413",
       "POST /x HTTP/1.1\r\nContent-Length: 1000\r\n\r\n",
       false, 413},
      {"unsupported Transfer-Encoding -> 415",
       "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n",
       false, 415},
      {"wrong-but-valid method (router's 405 territory)",
       "PUT /v1/sessions/1/turn HTTP/1.1\r\n\r\n",
       true, 0},
      {"unknown path (router's 404 territory)",
       "GET /nope HTTP/1.1\r\n\r\n",
       true, 0},
      {"non-absolute-form target -> 400",
       "GET example.com/x HTTP/1.1\r\n\r\n",
       false, 400},
      {"unsupported HTTP version -> 400",
       "GET / HTTP/2.0\r\n\r\n",
       false, 400},
  };

  for (const ParseCase& c : cases) {
    const HttpParseResult r = parse_http_request(c.raw, lim);
    if (c.expect_ok) {
      if (!r.ok) {
        std::fprintf(stderr, "  parse case failed: %s (%s)\n", c.name,
                     r.message.c_str());
      }
      CHECK(r.ok);
    } else {
      CHECK(!r.ok);
      CHECK_EQ(r.status, c.expect_status);
    }
  }

  // ---- the body exactness (leading / trailing whitespace + NUL) -------
  {
    // 36 bytes: \t \n space "leading" space NUL space "embedded" space
    // NUL space "trailing" space \r \n space
    const std::string body =
        std::string("\t\n leading ") + std::string(1, '\0') +
        " embedded " + std::string(1, '\0') + " trailing \r\n ";
    CHECK_EQ(body.size(), static_cast<std::size_t>(36));
    const std::string raw =
        "POST /v1/sessions/1/turn HTTP/1.1\r\n"
        "Content-Type: text/plain\r\n"
        "Content-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + body;
    const HttpParseResult r = parse_http_request(raw, lim);
    CHECK(r.ok);
    CHECK_EQ(r.req.body.size(), body.size());
    CHECK(r.req.body == body);
    CHECK_EQ(r.req.method, std::string("POST"));
    CHECK_EQ(r.req.path, std::string("/v1/sessions/1/turn"));
    // the headers are lower-cased + value-trimmed:
    CHECK_EQ(r.req.headers.at("content-type"), std::string("text/plain"));
  }

  // ---- the query parsing ------------------------------------------------
  {
    std::map<std::string, std::string> q;
    CHECK(parse_query("a=1&b=2", &q));
    CHECK_EQ(q.size(), static_cast<std::size_t>(2));
    CHECK_EQ(q["a"], std::string("1"));
    CHECK_EQ(q["b"], std::string("2"));

    std::map<std::string, std::string> q2;
    CHECK(parse_query("max_new_tokens=8", &q2));
    CHECK_EQ(q2["max_new_tokens"], std::string("8"));

    // a repeated key keeps the LAST value:
    std::map<std::string, std::string> q3;
    CHECK(parse_query("a=1&a=2", &q3));
    CHECK_EQ(q3["a"], std::string("2"));

    // an empty value is allowed:
    std::map<std::string, std::string> q4;
    CHECK(parse_query("a=", &q4));
    CHECK_EQ(q4["a"], std::string(""));

    // a trailing '&' is allowed (it names an empty final segment):
    std::map<std::string, std::string> q5;
    CHECK(parse_query("a=1&", &q5));
    CHECK_EQ(q5.size(), static_cast<std::size_t>(1));

    // a '%' is NOT supported (fail loud):
    std::map<std::string, std::string> q6;
    CHECK(!parse_query("a=%20", &q6));
    // an empty key is malformed:
    std::map<std::string, std::string> q7;
    CHECK(!parse_query("=v", &q7));
    // the empty query is valid (no parameters):
    std::map<std::string, std::string> q8;
    CHECK(parse_query("", &q8));
    CHECK_EQ(q8.size(), static_cast<std::size_t>(0));
  }

  TEST_PASS("test_http_protocol_parse");
  return 0;
}

// ---- the JSON escaping ----------------------------------------------------

int test_json_escape() {
  // a plain string passes through:
  CHECK_EQ(json_escape("hello"), std::string("hello"));
  // the quote:
  CHECK_EQ(json_escape("a\"b"), std::string("a\\\"b"));
  // the backslash:
  CHECK_EQ(json_escape("a\\b"), std::string("a\\\\b"));
  // the common controls:
  CHECK_EQ(json_escape("a\nb\rc\td\be\ff"),
           std::string("a\\nb\\rc\\td\\be\\ff"));
  // an EMBEDDED NUL is escaped (the generated text is never a C
  // string — the JSON stays valid + the byte is preserved):
  CHECK_EQ(json_escape(nul3()), std::string("a\\u0000b"));
  // a low control byte (0x01):
  CHECK_EQ(json_escape(std::string(1, 0x01)), std::string("\\u0001"));
  // DEL (0x7f) passes through (it is not a control byte):
  CHECK_EQ(json_escape(std::string(1, 0x7f)), std::string(1, 0x7f));
  // a UTF-8 multi-byte character passes through untouched:
  CHECK_EQ(json_escape("héllo"), std::string("héllo"));

  // the full round-trip shape (a JSON string value with a NUL + quote):
  const std::string value({'a', '\0', '"', 'b', '\\', 'c'});
  const std::string escaped = json_escape(value);
  CHECK(escaped.find('\0') == std::string::npos);  // no raw NUL escapes
  CHECK(escaped.find("\\u0000") != std::string::npos);
  CHECK(escaped.find("\\\"") != std::string::npos);
  TEST_PASS("test_http_protocol_json_escape");
  return 0;
}

// ---- the response formatting ---------------------------------------------

int test_format_response() {
  HttpResponse r;
  r.status = 200;
  r.reason = "OK";
  r.headers["Content-Type"] = "application/json";
  r.body = "{\"ok\":true}";  // 11 bytes
  const std::string wire = format_response(r);
  // the status line:
  CHECK(wire.rfind("HTTP/1.1 200 OK\r\n", 0) == 0);
  // the Content-Length is auto-added from the body size:
  CHECK(wire.find("Content-Length: 11\r\n") != std::string::npos);
  CHECK(wire.find("Connection: close\r\n") != std::string::npos);
  // the body is VERBATIM (binary-safe — an embedded NUL survives):
  HttpResponse r2;
  r2.status = 404;
  r2.reason = "Not Found";
  r2.body = nul3();
  const std::string wire2 = format_response(r2);
  CHECK(wire2.rfind("HTTP/1.1 404 Not Found\r\n", 0) == 0);
  CHECK(wire2.find("Content-Length: 3\r\n") != std::string::npos);
  // the body bytes are present verbatim at the tail:
  CHECK(wire2.compare(wire2.size() - 3, 3, nul3().data(), 3) == 0);

  // a caller-set Content-Length is NOT overwritten (the streaming
  // endpoint's close-delimited body):
  HttpResponse r3;
  r3.status = 200;
  r3.reason = "OK";
  r3.headers["Content-Length"] = "0";
  r3.body = "streamed\n";
  const std::string wire3 = format_response(r3);
  CHECK(wire3.find("Content-Length: 0\r\n") != std::string::npos);
  CHECK(wire3.find("Content-Length: 9\r\n") == std::string::npos);
  TEST_PASS("test_http_protocol_format");
  return 0;
}

}  // namespace

int main() {
  int rc = 0;
  rc |= test_parse_table();
  rc |= test_json_escape();
  rc |= test_format_response();
  if (rc != 0) {
    std::fprintf(stderr, "test_http_protocol: FAILED\n");
    return rc;
  }
  std::fprintf(stderr, "test_http_protocol: all [ok]\n");
  return 0;
}
