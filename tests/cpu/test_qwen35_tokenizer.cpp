// CUDALM — v0.4 Phase B: native tokenizer hard gate (cross-language exactness).
//
// Verifies the PyTorch-free Qwen35Tokenizer (CUDLMTK1 artifact) against the
// pinned Qwen3.5-0.8B-Base oracle:
//
//   * CORPUS EXACTNESS (the gate): a multilingual battery (ASCII / whitespace
//     / punctuation / accent + combining / Greek polytonic / Cyrillic /
//     Arabic / Hebrew / Thai / Devanagari / CJK + Hangul jamo / emoji /
//     added-token literals embedded mid-text / long mixed paragraphs) for
//     which the pinned HF `tokenizers` oracle pre-computed, at test time:
//       E <text_hex> <ids>        encode (add_special_tokens=False)
//       D  <ids> <text_hex>       decode (skip_special_tokens=False)
//       D1 <ids> <text_hex>       decode (skip_special_tokens=True)
//       N  <text_hex> <nfc_hex>   NFC normalization (codepoint level)
//       P  <text_hex> <pre...>    regex pre-tokenization (NFC'd input)
//     The native results must match the oracle EXACTLY (ids equal, bytes
//     equal) — no tolerance of any kind.
//   * LOADER FAILURE CONTRACT: every corruption class (bad magic / version /
//     reserved / crc / truncation) must fail loud with ok == false.
//   * STAGE VECTORS: hand-pinned NFC and pre-tokenization cases, incl. the
//     leftmost-first apostrophe branch ("'mX" -> "'m" + "X"), the
//     DESCENDING-CCC canonical-ordering regressions (UAX #15: canonical
//     ordering by CCC before composition) and the jamo cases the oracles
//     treat as ccc 0 (no jamo reordering).
//   * DECODE CONTRACT: padding ids (248077..248319) decode to "" (the
//     oracle silently drops them); out-of-range ids fail loud;
//     is_special() is exact over the 33 added ids; eos id == 248044.
//
// Special/control token strings are never hardcoded here: the corpus
// generator reads them from the pinned assets, and the C++ side only ever
// sees ids / lengths / sha (docs §18 security note).
//
// Usage: test_qwen35_tokenizer <artifact> <checkpoint_dir> <python>
//        <src_dir> <corpus> [--no-gen]
// Self-skips (77) when the pinned tokenizer assets are absent.

#include "../../tests/common/check.h"

#include "cudalm/qwen35_tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace cudalm;

namespace {

int run_cmd(const std::string& cmd) {
  std::fprintf(stderr, "  $ %s\n", cmd.c_str());
  const int rc = std::system(cmd.c_str());
  if (rc != 0) std::fprintf(stderr, "  command failed (rc=%d)\n", rc);
  return rc;
}

bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return static_cast<bool>(f);
}

std::vector<std::uint8_t> read_file_bytes(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return {};
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::string hex_decode(const std::string& h) {
  std::string out;
  out.reserve(h.size() / 2);
  auto val = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (std::size_t i = 0; i + 1 < h.size(); i += 2) {
    const int hi = val(h[i]);
    const int lo = val(h[i + 1]);
    if (hi < 0 || lo < 0) return {};
    out.push_back(static_cast<char>((hi << 4) | lo));
  }
  return out;
}

std::vector<std::uint32_t> parse_ids(const std::string& csv) {
  std::vector<std::uint32_t> out;
  if (csv.empty()) return out;
  std::istringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    out.push_back(static_cast<std::uint32_t>(std::stoul(tok)));
  }
  return out;
}

// --- loader failure contract -------------------------------------------------
int test_load_failures(const std::vector<std::uint8_t>& good) {
  std::fprintf(stderr, "[load] corruption classes must fail loud\n");
  int rc = 0;
  std::unique_ptr<Qwen35Tokenizer> tk;

  auto must_fail = [&](std::vector<std::uint8_t> blob, const char* what) {
    tk.reset();
    Status s = Qwen35Tokenizer::load_blob(blob.data(), blob.size(), &tk);
    if (s.ok) {
      std::fprintf(stderr, "  FAIL: %s accepted (want rejection)\n", what);
      rc |= 1;
    } else {
      std::fprintf(stderr, "  %-28s rejected: %s\n", what, s.message.c_str());
    }
  };

  // 0-byte and tiny inputs.
  must_fail({}, "empty blob");
  std::vector<std::uint8_t> tiny(10, 0);
  must_fail(tiny, "10-byte blob");

  // Bad magic (flip one header byte).
  {
    auto b = good;
    b[0] ^= 0xFF;
    must_fail(b, "bad magic");
  }
  // Unsupported version (offset 8, u32 LE -> 2).
  {
    auto b = good;
    b[8] = 2;
    must_fail(b, "bad version");
  }
  // Non-zero reserved (offset 12, u32 LE -> 1).
  {
    auto b = good;
    b[12] = 1;
    must_fail(b, "reserved != 0");
  }
  // crc32 mismatch (flip one body byte).
  {
    auto b = good;
    b.back() ^= 0x01;
    must_fail(b, "crc mismatch (flipped byte)");
  }
  // Truncation (cut the body mid-section).
  {
    auto b = good;
    b.resize(20 + b.size() / 2);
    must_fail(b, "truncated body");
  }
  // Truncation inside the header.
  {
    auto b = good;
    b.resize(18);
    must_fail(b, "truncated header");
  }
  // A valid file on disk must still load (positive control).
  tk.reset();
  Status s = Qwen35Tokenizer::load_blob(good.data(), good.size(), &tk);
  CHECK(s.ok);
  return rc;
}

// --- stage vectors ------------------------------------------------------------
int test_nfc_vectors(const Qwen35Tokenizer& tk) {
  std::fprintf(stderr, "[nfc] stage vectors (oracle-verified UAX #15)\n");
  int rc = 0;
  auto run = [&](const char* name, std::vector<std::uint32_t> in,
                 std::vector<std::uint32_t> want) -> int {
    std::vector<std::uint32_t> got;
    Status s = tk.nfc_normalize(in.data(), in.size(), &got);
    CHECK(s.ok);
    if (got != want) {
      std::fprintf(stderr, "  FAIL: %s\n", name);
      auto dump = [](const std::vector<std::uint32_t>& v) {
        std::string o;
        for (std::uint32_t c : v)
          o += " " + std::to_string(c);
        return o;
      };
      std::fprintf(stderr, "    want%s\n    got %s\n", dump(want).c_str(),
                   dump(got).c_str());
      return 1;
    }
    std::fprintf(stderr, "  %-34s OK\n", name);
    return 0;
  };
  // Expectations verified against BOTH oracles (unicodedata.normalize and
  // the pinned tokenizers engine, which applies NFC inside encode): NFC =
  // full canonical decomposition -> CANONICAL ORDERING by CCC (stable within
  // each combining sequence; ccc == 0 is a starter boundary) -> greedy
  // canonical composition.  The 27 composing T-jamo are U+11A8..U+11C2
  // (T1..T27); jamo are ccc 0 in the table (the pinned oracles never
  // reorder jamo); U+11C3+ are unassigned/reserved and never compose (the
  // oracle leaves AC00+U+11F2 and AC00+U+11F6 as-is).
  rc |= run("e+0301 -> U+00E9", {0x65, 0x301}, {0xE9});
  rc |= run("A+0308(diaeresis) -> U+00C4", {0x41, 0x308}, {0xC4});
  rc |= run("A+0308+0301 -> U+00C4+U+0301", {0x41, 0x308, 0x301}, {0xC4, 0x301});
  rc |= run("A+0300+0301 -> U+00C0+U+0301 (greedy L->R)", {0x41, 0x300, 0x301},
      {0xC0, 0x301});
  rc |= run("A+030A(ring)+0301 -> U+01FA", {0x41, 0x30A, 0x301}, {0x1FA});
  rc |= run("A+0301+0300 equal ccc stable -> U+00C1+U+0300",
      {0x41, 0x301, 0x300}, {0xC1, 0x300});
  rc |= run("A+0301+0308 equal ccc stable -> U+00C1+U+0308",
      {0x41, 0x301, 0x308}, {0xC1, 0x308});
  rc |= run("0301+0300 mark-only equal ccc unchanged", {0x301, 0x300},
      {0x301, 0x300});
  rc |= run("0300+A unchanged (no reorder across starter)", {0x300, 0x41},
      {0x300, 0x41});
  // --- DESCENDING ccc: the canonical-ordering regressions.  The old
  // implementation (no reordering) FAILS every case below: without moving
  // the lower-ccc mark before the higher-ccc one, (starter, lower mark)
  // never composes and the sequence stays uncomposed / mis-ordered.
  rc |= run("A+0315+0300 DESC (232,230) -> U+00C0+U+0315", {0x41, 0x315, 0x300},
      {0xC0, 0x315});
  rc |= run("A+0315+0308 DESC -> U+00C4+U+0315", {0x41, 0x315, 0x308},
      {0xC4, 0x315});
  rc |= run("A+0315+0301 DESC -> U+00C1+U+0315", {0x41, 0x315, 0x301},
      {0xC1, 0x315});
  rc |= run("a+0315+0300 DESC -> U+00E0+U+0315", {0x61, 0x315, 0x300},
      {0xE0, 0x315});
  rc |= run("A+0315+0300+0301 3-level -> U+00C0+U+0301+U+0315",
      {0x41, 0x315, 0x300, 0x301}, {0xC0, 0x301, 0x315});
  rc |= run("A+0300+0315 ASC -> U+00C0+U+0315", {0x41, 0x300, 0x315},
      {0xC0, 0x315});
  rc |= run("0315+0300 mark-only DESC -> 0300+0315", {0x315, 0x300},
      {0x300, 0x315});
  rc |= run("0300+0315+0301 mark-only reorder -> 0300+0301+0315",
      {0x300, 0x315, 0x301}, {0x300, 0x301, 0x315});
  // precomposed + marks; recursive decomposition; Greek 1FEE -> 0385.
  rc |= run("U+00C0+0300+0315 precomp+marks", {0xC0, 0x300, 0x315},
      {0xC0, 0x300, 0x315});
  rc |= run("U+0763 recursive decomp -> U+0763", {0x763}, {0x763});
  rc |= run("U+1FEE -> U+0385 (Greek)", {0x1FEE}, {0x385});
  rc |= run("L+V+T jamo -> U+AC01", {0x1100, 0x1161, 0x11A8}, {0xAC01});
  rc |= run("L+V+T2 jamo -> U+AC02", {0x1100, 0x1161, 0x11A9}, {0xAC02});
  rc |= run("L+V+T27 jamo (U+11C2) -> U+AC1B", {0x1100, 0x1161, 0x11C2},
      {0xAC1B});
  rc |= run("L+V+U+11F2 (not T-jamo) unchanged", {0x1100, 0x1161, 0x11F2},
      {0xAC00, 0x11F2});
  rc |= run("L+V+U+11F6 (not T-jamo) unchanged", {0x1100, 0x1161, 0x11F6},
      {0xAC00, 0x11F6});
  rc |= run("L+V jamo -> U+AC00", {0x1100, 0x1161}, {0xAC00});
  rc |= run("TVL jamo: no reorder, L+V tail composes", {0x11A8, 0x1100, 0x1161},
      {0x11A8, 0xAC00});
  rc |= run("U+1FBE 1-part decomp -> U+03B9", {0x1FBE}, {0x3B9});
  rc |= run("CJK unchanged", {0x4E2D, 0x7684}, {0x4E2D, 0x7684});
  rc |= run("emoji astral unchanged", {0x1F389}, {0x1F389});
  return rc;
}

int test_pretokenize_vectors(const Qwen35Tokenizer& tk) {
  std::fprintf(stderr, "[pretok] stage vectors (leftmost-first semantics)\n");
  int rc = 0;
  auto run = [&](const std::string& name, const std::string& in,
                 std::vector<std::string> want) -> int {
    std::vector<std::string> got;
    Status s = tk.pretokenize(in, &got);
    CHECK(s.ok);
    if (got != want) {
      std::fprintf(stderr, "  FAIL: %s\n", name.c_str());
      auto dump = [](const std::vector<std::string>& v) {
        std::string o;
        for (const auto& p : v) {
          o += "[";
          for (unsigned char c : p) {
            if (c >= 0x20 && c < 0x7F) {
              o += static_cast<char>(c);
            } else {
              char buf[8];
              std::snprintf(buf, sizeof(buf), "\\u%04x", c);
              o += buf;
            }
          }
          o += "]";
        }
        return o;
      };
      std::fprintf(stderr, "    want %s\n    got  %s\n", dump(want).c_str(),
                   dump(got).c_str());
      return 1;
    }
    std::fprintf(stderr, "  %-34s OK\n", name.c_str());
    return 0;
  };
  // Ordinary (non-special) probe strings only.
  rc |= run("ascii word", "hello", {"hello"});
  rc |= run("space glue", "a b", {"a", " b"});
  rc |= run("digit then word", "5 apples", {"5", " apples"});
  rc |= run("symbols", "x**2", {"x", "**", "2"});
  rc |= run("contraction", "it's", {"it", "'s"});
  rc |= run("leftmost-first branch1", "'mX", {"'m", "X"});
  rc |= run("tab+newline one pretok", "\t\n", {"\t\n"});
  rc |= run("crlf", "a\r\nb", {"a", "\r\n", "b"});
  // B2 ([^\r\nLN]?(?:LM)+) outranks the whitespace branches: the nbsp is
  // the optional lead char of the word pre-token.
  rc |= run("nbsp glues to following word (B2 first)", "a\u00a0b",
      {"a", "\u00a0b"});
  rc |= run("trailing space", "x ", {"x", " "});
  rc |= run("cjk one pretok", "\xe4\xb8\xad\xe6\x96\x87",
      {"\xe4\xb8\xad\xe6\x96\x87"});
  rc |= run("cjk-letter mix", "a\u4e2db", {"a\u4e2db"});
  rc |= run("accent precomposed", "caf\u00e9", {"caf\u00e9"});
  rc |= run("empty input", "", {});
  return rc;
}

// --- decode contract ------------------------------------------------------------
int test_decode_contract(const Qwen35Tokenizer& tk) {
  std::fprintf(stderr, "[decode] padding / range / special contract\n");
  int rc = 0;
  const std::uint32_t id0 = 0;  // the base-vocab id 0 (GPT-2: '!')
  {
    const std::uint32_t ids[] = {248077u, id0, 248319u};
    std::string got;
    Status s = tk.decode(ids, 3, false, &got);
    CHECK(s.ok);
    const std::uint32_t only0[] = {id0};
    std::string want;
    s = tk.decode(only0, 1, false, &want);
    CHECK(s.ok);
    if (got != want) {
      std::fprintf(stderr, "  FAIL: padding ids must decode to \"\" "
                           "(got %zu bytes, want %zu)\n",
                   got.size(), want.size());
      rc |= 1;
    } else {
      std::fprintf(stderr, "  padding 248077/248319 -> \"\"            OK\n");
    }
  }
  {
    const std::uint32_t bad[] = {0u, 248320u};
    std::string got;
    Status s = tk.decode(bad, 2, false, &got);
    if (s.ok) {
      std::fprintf(stderr, "  FAIL: id 248320 (>= model vocab) accepted\n");
      rc |= 1;
    } else {
      std::fprintf(stderr, "  id out of range rejected: %s\n",
                   s.message.c_str());
    }
  }
  {
    // is_special() must be exact over the 33 added ids (21 special) and
    // false for everything else in the model vocab.
    int special_count = 0;
    for (std::uint32_t id = 0; id < 248320u; ++id) {
      const bool sp = tk.is_special(id);
      if (sp) ++special_count;
      const bool expect_sp = (id >= 248044u && id <= 248076u);
      if (sp && !expect_sp) {
        std::fprintf(stderr, "  FAIL: is_special(%u) but not an added id\n",
                     id);
        rc |= 1;
      }
    }
    if (special_count != 21) {
      std::fprintf(stderr, "  FAIL: %d special ids, want 21\n", special_count);
      rc |= 1;
    } else {
      std::fprintf(stderr, "  is_special: 21 of 33 added, none outside   OK\n");
    }
  }
  CHECK_EQ(tk.eos_token_id(), 248044u);
  // Invalid UTF-8 must fail loud on encode.
  {
    const std::string bad_utf8("\xff\xfe", 2);
    std::vector<std::uint32_t> out;
    Status s = tk.encode(bad_utf8, &out);
    if (s.ok) {
      std::fprintf(stderr, "  FAIL: invalid UTF-8 accepted by encode\n");
      rc |= 1;
    } else {
      std::fprintf(stderr, "  invalid UTF-8 rejected: %s\n", s.message.c_str());
    }
  }
  return rc;
}

// --- corpus ------------------------------------------------------------------
int test_corpus(const Qwen35Tokenizer& tk, const std::string& corpus_path) {
  std::fprintf(stderr, "[corpus] cross-language exactness (E/D/D1/N/P)\n");
  std::ifstream f(corpus_path);
  CHECK(f);
  int nE = 0, nD = 0, nD1 = 0, nN = 0, nP = 0, nX = 0;
  int rc = 0;
  std::string line;
  int lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    if (line.empty()) continue;
    const char kind = line[0];
    std::string rest = line.substr(2);
    if (!rest.empty() && rest[0] == ' ') rest = rest.substr(1);
    auto fail = [&](const std::string& what) {
      std::fprintf(stderr, "  FAIL line %d (%c): %s\n", lineno, kind,
                   what.c_str());
      rc |= 1;
    };
    if (kind == 'E') {
      const auto sp = rest.find(' ');
      const std::string text = hex_decode(rest.substr(0, sp));
      const std::vector<std::uint32_t> want_ids =
          parse_ids(rest.substr(sp + 1));
      std::vector<std::uint32_t> got;
      Status s = tk.encode(text, &got);
      CHECK(s.ok);
      if (got != want_ids) {
        auto dump = [](const std::vector<std::uint32_t>& v,
                       std::size_t cap = 12) {
          std::string o;
          for (std::size_t i = 0; i < v.size() && i < cap; ++i)
            o += (i ? "," : "") + std::to_string(v[i]);
          if (v.size() > cap) o += ",...";
          return o;
        };
        std::fprintf(stderr, "  FAIL line %d encode of %s\n    want %s\n    "
                             "got  %s\n",
                     lineno,
                     [&] {
                       std::string h;
                       for (char c : text) {
                         static const char* d = "0123456789abcdef";
                         h += d[(static_cast<unsigned char>(c) >> 4) & 15];
                         h += d[static_cast<unsigned char>(c) & 15];
                       }
                       return h;
                     }().c_str(),
                     dump(want_ids).c_str(), dump(got).c_str());
        rc |= 1;
      }
      ++nE;
    } else if (kind == 'D' && line[1] == ' ') {
      const auto sp = rest.find(' ');
      const std::vector<std::uint32_t> ids = parse_ids(rest.substr(0, sp));
      const std::string want = hex_decode(rest.substr(sp + 1));
      std::string got;
      Status s = tk.decode(ids.data(), ids.size(), false, &got);
      CHECK(s.ok);
      if (got != want) {
        std::fprintf(stderr, "  FAIL line %d decode (skip=False) of %zu ids\n",
                     lineno, ids.size());
        rc |= 1;
      }
      ++nD;
    } else if (kind == 'D' && line[1] == '1') {
      const auto sp = rest.find(' ');
      const std::vector<std::uint32_t> ids = parse_ids(rest.substr(0, sp));
      const std::string want = hex_decode(rest.substr(sp + 1));
      std::string got;
      Status s = tk.decode(ids.data(), ids.size(), true, &got);
      CHECK(s.ok);
      if (got != want) {
        std::fprintf(stderr,
                     "  FAIL line %d decode (skip=True) of %zu ids\n",
                     lineno, ids.size());
        rc |= 1;
      }
      ++nD1;
    } else if (kind == 'N') {
      const auto sp = rest.find(' ');
      const std::string in_utf8 = hex_decode(rest.substr(0, sp));
      const std::string want = hex_decode(rest.substr(sp + 1));
      std::vector<std::uint32_t> in_cps;
      Status s = tk.utf8_to_codepoints(in_utf8, &in_cps);
      CHECK(s.ok);
      std::vector<std::uint32_t> got_cps;
      s = tk.nfc_normalize(in_cps.data(), in_cps.size(), &got_cps);
      CHECK(s.ok);
      const std::string got =
          Qwen35Tokenizer::codepoints_to_utf8(got_cps.data(), got_cps.size());
      if (got != want) {
        std::fprintf(stderr, "  FAIL line %d nfc of %s\n", lineno,
                     rest.substr(0, sp).c_str());
        rc |= 1;
      }
      ++nN;
    } else if (kind == 'P') {
      const auto sp = rest.find(' ');
      const std::string in = hex_decode(rest.substr(0, sp));
      // The corpus stores each pre-token hex'd, fields joined with 0x1F.
      std::vector<std::string> want;
      {
        std::vector<std::string> hexed;
        std::string cur;
        for (char c : rest.substr(sp + 1)) {
          if (c == 0x1F) {
            hexed.push_back(cur);
            cur.clear();
          } else {
            cur.push_back(c);
          }
        }
        hexed.push_back(cur);
        for (const auto& h : hexed) want.push_back(hex_decode(h));
      }
      std::vector<std::string> got;
      Status s = tk.pretokenize(in, &got);
      CHECK(s.ok);
      if (got != want) {
        std::fprintf(stderr, "  FAIL line %d pretok of %s\n", lineno,
                     rest.substr(0, sp).c_str());
        rc |= 1;
      }
      ++nP;
    } else if (kind == 'X') {
      // Arbitrary token-ID decode (X <ids_csv> <skip 0|1> <expected_hex>):
      // id sequences never produced by encode(text) — raw ByteLevel bytes,
      // padding and added ids in arbitrary mix.  The native decode (byte
      // stream -> lossy UTF-8) must match the pinned engine EXACTLY.
      const auto sp1 = rest.find(' ');
      const std::string ids_csv = rest.substr(0, sp1);
      const std::string tail = rest.substr(sp1 + 1);
      const auto sp2 = tail.find(' ');
      const bool skip = tail.substr(0, sp2) == "1";
      const std::string want = hex_decode(tail.substr(sp2 + 1));
      const std::vector<std::uint32_t> ids = parse_ids(ids_csv);
      std::string got;
      Status s = tk.decode(ids.data(), ids.size(), skip, &got);
      CHECK(s.ok);
      if (got != want) {
        std::fprintf(stderr,
                     "  FAIL line %d decode-X (skip=%d) of %zu ids\n", lineno,
                     static_cast<int>(skip), ids.size());
        rc |= 1;
      }
      ++nX;
    } else {
      fail("unknown line kind");
    }
  }
  std::fprintf(stderr, "  lines: E=%d D=%d D1=%d N=%d P=%d X=%d  %s\n", nE, nD,
               nD1, nN, nP, nX, rc ? "(FAILURES)" : "(all exact)");
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: %s <artifact> <checkpoint_dir> <python> <src_dir> "
                 "<corpus> [--no-gen]\n",
                 argv[0]);
    return 2;
  }
  const std::string artifact = argv[1];
  const std::string ckpt = argv[2];
  const std::string py = argv[3];
  const std::string src = argv[4];
  const std::string corpus = argv[5];
  const bool no_gen = (argc >= 7 && std::string(argv[6]) == "--no-gen");

  const std::string tdir = ckpt + "/tokenizer";
  if (!no_gen && !file_exists(tdir + "/tokenizer.json")) {
    std::fprintf(stderr, "[SKIP] qwen35 tokenizer: pinned assets absent at %s\n",
                 tdir.c_str());
    return 77;
  }
  if (!no_gen) {
    int rc = run_cmd(py + " " + src + "/tools/convert_qwen35_tokenizer.py" +
                     " --tokenizer-dir " + tdir + " --out " + artifact);
    CHECK_EQ(rc, 0);
    rc = run_cmd(py + " " + src + "/tools/gen_tokenizer_refs.py" +
                 " --tokenizer-dir " + tdir + " --out " + corpus);
    CHECK_EQ(rc, 0);
  }

  const std::vector<std::uint8_t> blob = read_file_bytes(artifact);
  CHECK(!blob.empty());

  std::unique_ptr<Qwen35Tokenizer> tk;
  Status s = Qwen35Tokenizer::load_blob(blob.data(), blob.size(), &tk);
  CHECK(s.ok);

  int rc = 0;
  rc |= test_load_failures(blob);
  rc |= test_nfc_vectors(*tk);
  rc |= test_pretokenize_vectors(*tk);
  rc |= test_decode_contract(*tk);
  rc |= test_corpus(*tk, corpus);

  if (rc != 0) {
    std::fprintf(stderr, "qwen35 tokenizer: FAILURES (rc=%d)\n", rc);
    return 1;
  }
  TEST_PASS("qwen35_tokenizer (load contract + NFC/pretok vectors + decode "
            "contract + multilingual corpus exact vs pinned HF oracle)");
  return 0;
}
