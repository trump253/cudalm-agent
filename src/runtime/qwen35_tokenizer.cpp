// CUDALM — Qwen3.5-0.8B-Base native tokenizer (PyTorch-free).
//
// Implementation of include/cudalm/qwen35_tokenizer.h.  The CUDLMTK1
// artifact (built offline by tools/convert_qwen35_tokenizer.py) carries the
// pinned tokenizer pipeline; this file loads it with full bounds-checking
// and implements the exact oracle-verified algorithm:
//   NFC (full decomposition, NO reordering, greedy gated composition)
//   -> added-token split -> leftmost-first regex pre-tokenization -> BPE.
//
// No special/control token string literals exist anywhere in this file; the
// added tokens live in the artifact only (docs §18 security note).

#include "cudalm/qwen35_tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudalm {
namespace {

constexpr std::uint32_t kMaxCp = 0x10FFFFu;
constexpr std::size_t kDenseSize = 0x110000u;  // dense table size
constexpr char kMagic[8] = {'C', 'U', 'D', 'L', 'M', 'T', 'K', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kNumMeta = 15;
constexpr std::size_t kNumSections = 10;

// ---------------------------------------------------------------------------
// crc32 (zlib polynomial, init/xorout 0xFFFFFFFF) — dependency-free.
// ---------------------------------------------------------------------------
std::uint32_t crc32(const std::uint8_t* p, std::size_t n) {
  static std::uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k)
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    init = true;
  }
  std::uint32_t c = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i)
    c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Bounds-checked little-endian reader.
// ---------------------------------------------------------------------------
struct Reader {
  const std::uint8_t* p;
  std::size_t n;
  std::size_t off = 0;
  bool fail = false;
  std::string why;

  void set_fail(std::string w) {
    if (!fail) {
      fail = true;
      why = std::move(w);
    }
  }
  bool u8(std::uint8_t* v) {
    if (fail) return false;
    if (off + 1 > n) return set_fail("truncated u8"), false;
    *v = p[off++];
    return true;
  }
  bool u16(std::uint16_t* v) {
    std::uint8_t b[2];
    if (fail) return false;
    if (off + 2 > n) return set_fail("truncated u16"), false;
    b[0] = p[off];
    b[1] = p[off + 1];
    off += 2;
    *v = static_cast<std::uint16_t>(b[0] | (b[1] << 8));
    return true;
  }
  bool u32(std::uint32_t* v) {
    std::uint8_t b[4];
    if (fail) return false;
    if (off + 4 > n) return set_fail("truncated u32"), false;
    for (int i = 0; i < 4; ++i) b[i] = p[off + i];
    off += 4;
    *v = static_cast<std::uint32_t>(b[0]) |
         (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) |
         (static_cast<std::uint32_t>(b[3]) << 24);
    return true;
  }
  bool bytes(std::size_t len, const std::uint8_t** out) {
    if (fail) return false;
    if (len > n - off) return set_fail("truncated section"), false;
    *out = p + off;
    off += len;
    return true;
  }
  Status finish() {
    if (fail) return Status::error(why);
    if (off != n) return Status::error("trailing garbage in body");
    return Status::ok_status();
  }
};

// ---------------------------------------------------------------------------
// UTF-8 <-> code points.
// ---------------------------------------------------------------------------
int utf8_seq_len(std::uint32_t cp) {
  if (cp < 0x80u) return 1;
  if (cp < 0x800u) return 2;
  if (cp < 0x10000u) return 3;
  return 4;
}

Status utf8_decode(const std::string& s, std::vector<std::uint32_t>* out) {
  out->clear();
  std::size_t i = 0;
  const std::size_t n = s.size();
  while (i < n) {
    const std::uint8_t b0 = static_cast<std::uint8_t>(s[i]);
    std::uint32_t cp = 0;
    int len = 0;
    if (b0 < 0x80u) {
      cp = b0;
      len = 1;
    } else if ((b0 & 0xE0u) == 0xC0u) {
      cp = b0 & 0x1Fu;
      len = 2;
    } else if ((b0 & 0xF0u) == 0xE0u) {
      cp = b0 & 0x0Fu;
      len = 3;
    } else if ((b0 & 0xF8u) == 0xF0u) {
      cp = b0 & 0x07u;
      len = 4;
    } else {
      return Status::error("invalid UTF-8 lead byte");
    }
    if (i + len > n) return Status::error("truncated UTF-8 sequence");
    bool ok = true;
    for (int k = 1; k < len; ++k) {
      const std::uint8_t b = static_cast<std::uint8_t>(s[i + k]);
      if ((b & 0xC0u) != 0x80u) {
        ok = false;
        break;
      }
      cp = (cp << 6) | (b & 0x3Fu);
    }
    if (!ok) return Status::error("invalid UTF-8 continuation byte");
    // Overlong + surrogate + out-of-range checks.
    static const std::uint32_t min_cp[5] = {0, 0, 0x80u, 0x800u, 0x10000u};
    if (cp < min_cp[len] || cp > kMaxCp || (0xD800u <= cp && cp <= 0xDFFFu))
      return Status::error("invalid code point in UTF-8");
    out->push_back(cp);
    i += len;
  }
  return Status::ok_status();
}

std::string utf8_encode(const std::uint32_t* cps, std::size_t n) {
  std::string s;
  s.reserve(n * 2);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t cp = cps[i];
    if (cp < 0x80u) {
      s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
      s.push_back(static_cast<char>(0xC0u | (cp >> 6)));
      s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
      s.push_back(static_cast<char>(0xE0u | (cp >> 12)));
      s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
      s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
      s.push_back(static_cast<char>(0xF0u | (cp >> 18)));
      s.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
      s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
      s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
  }
  return s;
}

// Lossy UTF-8 conversion (the pinned HF/tokenizers ByteLevel decoder
// semantics, Rust `String::from_utf8_lossy`): the input is a COMPLETE byte
// stream (base-token bytes + added-token UTF-8, padding dropped); every
// maximal invalid subpart is replaced by exactly ONE U+FFFD (UTS #35), a
// valid sequence is emitted as-is.  The conversion must be done on the whole
// stream, never per token: a legal multi-byte character may be split across
// several base tokens (E4 + B8 + AD -> U+4E2D).
void utf8_lossy(const std::string& bytes, std::string* out) {
  const std::size_t n = bytes.size();
  std::size_t i = 0;
  auto fffd = [&out]() { out->append("\xEF\xBF\xBD"); };
  auto is_cont = [](std::uint8_t b) { return (b & 0xC0u) == 0x80u; };
  while (i < n) {
    const std::uint8_t b = static_cast<std::uint8_t>(bytes[i]);
    if (b < 0x80u) {
      out->push_back(static_cast<char>(b));
      ++i;
      continue;
    }
    // Multi-byte lead: expected length + per-position continuation ranges.
    int len = 0;
    std::uint8_t lo[4] = {0, 0, 0, 0}, hi[4] = {0, 0, 0, 0};
    if (b >= 0xC2u && b <= 0xDFu) {
      len = 2; lo[0] = 0x80; hi[0] = 0xBF;
    } else if (b == 0xE0u) {
      len = 3; lo[0] = 0xA0; hi[0] = 0xBF; lo[1] = 0x80; hi[1] = 0xBF;
    } else if (b >= 0xE1u && b <= 0xECu) {
      len = 3; lo[0] = 0x80; hi[0] = 0xBF; lo[1] = 0x80; hi[1] = 0xBF;
    } else if (b == 0xEDu) {
      len = 3; lo[0] = 0x80; hi[0] = 0x9F; lo[1] = 0x80; hi[1] = 0xBF;
    } else if (b >= 0xEEu && b <= 0xEFu) {
      len = 3; lo[0] = 0x80; hi[0] = 0xBF; lo[1] = 0x80; hi[1] = 0xBF;
    } else if (b == 0xF0u) {
      len = 4; lo[0] = 0x90; hi[0] = 0xBF; lo[1] = 0x80; hi[1] = 0xBF;
      lo[2] = 0x80; hi[2] = 0xBF;
    } else if (b >= 0xF1u && b <= 0xF3u) {
      len = 4; lo[0] = 0x80; hi[0] = 0xBF; lo[1] = 0x80; hi[1] = 0xBF;
      lo[2] = 0x80; hi[2] = 0xBF;
    } else if (b == 0xF4u) {
      len = 4; lo[0] = 0x80; hi[0] = 0x8F; lo[1] = 0x80; hi[1] = 0xBF;
      lo[2] = 0x80; hi[2] = 0xBF;
    } else {
      // Invalid lead byte (0x80..0xC1, 0xF5..0xFF): one FFFD for this byte.
      fffd();
      ++i;
      continue;
    }
    // Consume the in-range continuation bytes, tracking the maximal
    // subpart.  sub = subpart size including the lead.  A continuation byte
    // IN its position range extends the subpart; the first byte that is not
    // a continuation byte, that is OUT of the position range, or EOF ends
    // the subpart BEFORE it (that byte then starts its own subpart — the
    // pinned engine gives every such byte its own U+FFFD; oracle-verified
    // on ED+A0+80 -> 3 x FFFD, F4+90+80+80 -> 4 x FFFD, E0+80+80 -> 3 x
    // FFFD).  With the per-position ranges above, a fully-consumed sequence
    // is always a valid code point — no surrogates / no > U+10FFFF — so the
    // bytes can be copied verbatim.
    std::size_t sub = 1;
    for (int j = 1; j < len; ++j) {
      if (i + j >= n) break;
      const std::uint8_t cj = static_cast<std::uint8_t>(bytes[i + j]);
      if (!is_cont(cj)) break;
      if (cj < lo[j - 1] || cj > hi[j - 1]) break;
      sub = j + 1;
    }
    if (sub == static_cast<std::size_t>(len)) {
      out->append(bytes, i, len);
      i += len;
    } else {
      fffd();
      i += sub;
    }
  }
}

// ---------------------------------------------------------------------------
// Regex pre-tokenizer (pinned Qwen3.5 pattern, leftmost-first semantics).
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   | \p{N}
//   |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
//
// Implemented as a small backtracking matcher over code points with the
// exact PCRE/Onig leftmost-first + greedy-quantifier semantics.  The pattern
// is a static node pool (integer refs, trivially copyable).
// ---------------------------------------------------------------------------
struct RNode {
  enum Kind : int { kAlt, kSeq, kLit, kClass, kNotClass, kOpt, kPlus, kStar };
  Kind kind = kSeq;
  std::uint32_t cp = 0;  // kLit
  bool casefold = false;
  int class_id = 0;      // kClass / kNotClass
  int items_start = -1;  // kAlt/kSeq: index into the pool
  int n_items = 0;
  int child = -1;        // kOpt/kPlus/kStar
};

struct RPool {
  std::vector<RNode> nodes;
  int add(RNode n) {
    nodes.push_back(n);
    return static_cast<int>(nodes.size()) - 1;
  }
  int add_group(RNode kind_node, std::vector<int> items) {
    kind_node.items_start = static_cast<int>(nodes.size());
    kind_node.n_items = static_cast<int>(items.size());
    for (int i : items) {
      // Copy the value BEFORE push_back: push_back may reallocate the
      // vector, which would dangle a nodes[i] reference (UB, pool corruption).
      const RNode copy = nodes[static_cast<std::size_t>(i)];
      nodes.push_back(copy);
    }
    return add(kind_node);
  }
  int add_quant(RNode::Kind k, int child_idx) {
    RNode n;
    n.kind = k;
    n.child = child_idx;
    return add(n);
  }
};

// Class ids (see Matcher::in_class).
constexpr int kClsL = 0, kClsN = 1, kClsLM = 2, kClsWS = 3, kClsCRNL = 4,
              kClsNotCrlnLN = 5, kClsNotWsLMN = 6, kClsNonWS = 7;

int make_lit(RPool& p, std::uint32_t cp, bool fold = false) {
  RNode n;
  n.kind = RNode::kLit;
  n.cp = cp;
  n.casefold = fold;
  return p.add(n);
}
int make_cls(RPool& p, int id, bool negated = false) {
  RNode n;
  n.kind = negated ? RNode::kNotClass : RNode::kClass;
  n.class_id = id;
  return p.add(n);
}

struct Matcher {
  const std::vector<bool>* l;
  const std::vector<bool>* n;
  const std::vector<bool>* m;
  const std::vector<bool>* ws;
  const RPool* pool;

  bool in_class(int id, std::uint32_t cp) const {
    const bool L = (*l)[cp];
    const bool N = (*n)[cp];
    const bool M = (*m)[cp];
    const bool W = (*ws)[cp];
    switch (id) {
      case kClsL: return L;
      case kClsN: return N;
      case kClsLM: return L || M;
      case kClsWS: return W;
      case kClsCRNL: return cp == 0x0Du || cp == 0x0Au;
      case kClsNotCrlnLN:
        return cp != 0x0Du && cp != 0x0Au && !L && !N;
      case kClsNotWsLMN: return !W && !L && !M && !N;
      case kClsNonWS: return !W;
      default: return false;
    }
  }

  const RNode& node(int i) const { return pool->nodes[static_cast<std::size_t>(i)]; }

  // Single-result node (a quantifier never appears directly here).
  bool match_single(int nd, const std::vector<std::uint32_t>& cps,
                    std::size_t pos, std::size_t* end) const {
    const RNode& n = node(nd);
    switch (n.kind) {
      case RNode::kLit: {
        if (pos >= cps.size()) return false;
        std::uint32_t c = cps[pos];
        std::uint32_t want = n.cp;
        if (n.casefold) {
          if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
          if (want >= 'A' && want <= 'Z') want += 'a' - 'A';
        }
        if (c != want) return false;
        *end = pos + 1;
        return true;
      }
      case RNode::kClass: {
        if (pos >= cps.size()) return false;
        if (!in_class(n.class_id, cps[pos])) return false;
        *end = pos + 1;
        return true;
      }
      case RNode::kNotClass: {
        // (?!) is a zero-width lookahead: it never consumes a character,
        // and it succeeds at end of input.
        bool hit;
        if (pos >= cps.size()) {
          hit = true;
        } else {
          hit = !in_class(n.class_id, cps[pos]);
        }
        if (!hit) return false;
        *end = pos;
        return true;
      }
      case RNode::kOpt: {
        std::size_t p2;
        if (match_single(n.child, cps, pos, &p2)) {
          *end = p2;
          return true;
        }
        *end = pos;
        return true;
      }
      case RNode::kAlt:
        for (int i = 0; i < n.n_items; ++i)
          if (match_single(n.items_start + i, cps, pos, end)) return true;
        return false;
      case RNode::kSeq:
        return match_seq(nd, 0, cps, pos, end);
      default:
        return false;  // quantifiers are handled by match_seq
    }
  }

  // Sequence items [idx..); quantifiers backtrack against the remaining
  // items (greedy: longest first).
  bool match_seq(int nd, int idx, const std::vector<std::uint32_t>& cps,
                 std::size_t pos, std::size_t* end) const {
    const RNode& seq = node(nd);
    if (idx == seq.n_items) {
      *end = pos;
      return true;
    }
    const int cur = seq.items_start + idx;
    const RNode& c = node(cur);
    if (c.kind == RNode::kStar || c.kind == RNode::kPlus) {
      std::vector<std::size_t> ends;
      std::size_t p = pos;
      if (c.kind == RNode::kStar) ends.push_back(pos);
      for (std::size_t guard = 0; guard < 1000000; ++guard) {
        std::size_t p2;
        if (!match_single(c.child, cps, p, &p2)) break;
        ends.push_back(p2);
        p = p2;
      }
      for (std::size_t i = ends.size(); i-- > 0;)
        if (match_seq(nd, idx + 1, cps, ends[i], end)) return true;
      return false;
    }
    if (c.kind == RNode::kOpt) {
      // Greedy optional WITH backtracking (PCRE semantics): try the
      // consumed form first, then the empty form.  Without the fallback
      // the B2 branch `[...]?[\p{L}\p{M}]+` fails on a combining mark
      // that is the first character of a chunk and is followed by a
      // non-L/M character (or end of input): the optional prefix eats the
      // mark and nothing is left for the +.  The pinned engine backtracks
      // (leftmost-first: the optional is skipped and the mark is matched
      // by the +) — oracle-verified on a lone U+0300, "U+0301 space", ...
      std::size_t p2;
      if (match_single(c.child, cps, pos, &p2)) {
        if (match_seq(nd, idx + 1, cps, p2, end)) return true;
      }
      return match_seq(nd, idx + 1, cps, pos, end);
    }
    std::size_t p2;
    if (!match_single(cur, cps, pos, &p2)) return false;
    return match_seq(nd, idx + 1, cps, p2, end);
  }

  bool match_top(int nd, const std::vector<std::uint32_t>& cps, std::size_t pos,
                 std::size_t* end) const {
    return match_single(nd, cps, pos, end);
  }
};

// Build the pinned pre-tokenization pattern (no string literals involved).
int build_pattern(RPool& p) {
  // B1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
  const int b1 = p.add_group(
      RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
      {make_lit(p, '\''),
       p.add_group(
           RNode{RNode::kAlt, 0, false, 0, -1, 0, -1},
           {p.add_group(RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
                        {make_lit(p, 's', true)}),
            p.add_group(RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
                        {make_lit(p, 't', true)}),
            p.add_group(RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
                        {make_lit(p, 'r', true), make_lit(p, 'e', true)}),
            p.add_group(RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
                        {make_lit(p, 'v', true), make_lit(p, 'e', true)}),
            p.add_group(RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
                        {make_lit(p, 'm', true)}),
            p.add_group(RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
                        {make_lit(p, 'l', true), make_lit(p, 'l', true)}),
            p.add_group(RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
                        {make_lit(p, 'd', true)})})});
  // B2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
  const int b2 = p.add_group(
      RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
      {p.add_quant(RNode::kOpt, make_cls(p, kClsNotCrlnLN)),
       p.add_quant(RNode::kPlus, make_cls(p, kClsLM))});
  // B3: \p{N}
  const int b3 = p.add_group(
      RNode{RNode::kSeq, 0, false, 0, -1, 0, -1}, {make_cls(p, kClsN)});
  // B4:  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
  const int b4 = p.add_group(
      RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
      {p.add_quant(RNode::kOpt, make_lit(p, ' ')),
       p.add_quant(RNode::kPlus, make_cls(p, kClsNotWsLMN)),
       p.add_quant(RNode::kStar, make_cls(p, kClsCRNL))});
  // B5: \s*[\r\n]+
  const int b5 = p.add_group(
      RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
      {p.add_quant(RNode::kStar, make_cls(p, kClsWS)),
       p.add_quant(RNode::kPlus, make_cls(p, kClsCRNL))});
  // B6: \s+(?!\S)
  const int b6 = p.add_group(
      RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
      {p.add_quant(RNode::kPlus, make_cls(p, kClsWS)),
       make_cls(p, kClsNonWS, /*negated=*/true)});
  // B7: \s+
  const int b7 = p.add_group(
      RNode{RNode::kSeq, 0, false, 0, -1, 0, -1},
      {p.add_quant(RNode::kPlus, make_cls(p, kClsWS))});
  return p.add_group(RNode{RNode::kAlt, 0, false, 0, -1, 0, -1},
                     {b1, b2, b3, b4, b5, b6, b7});
}

}  // namespace

// ---------------------------------------------------------------------------
// Qwen35Tokenizer
// ---------------------------------------------------------------------------
Status Qwen35Tokenizer::load(const std::string& path,
                             std::unique_ptr<Qwen35Tokenizer>* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return Status::error("cannot open " + path);
  std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
  if (f.bad()) return Status::error("read error on " + path);
  return load_blob(buf.data(), buf.size(), out);
}

Status Qwen35Tokenizer::load_blob(const std::uint8_t* data, std::size_t len,
                                  std::unique_ptr<Qwen35Tokenizer>* out) {
  auto tk = std::unique_ptr<Qwen35Tokenizer>(new Qwen35Tokenizer());
  Status s = tk->init_from_blob(data, len);
  if (!s.ok) return s;
  *out = std::move(tk);
  return Status::ok_status();
}

Status Qwen35Tokenizer::init_from_blob(const std::uint8_t* data,
                                       std::size_t len) {
  if (len < 20) return Status::error("artifact too small");
  if (std::memcmp(data, kMagic, 8) != 0) return Status::error("bad magic");
  std::uint32_t version = 0, reserved = 0, crc = 0;
  {
    std::uint8_t b[4];
    std::memcpy(b, data + 8, 4);
    version = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
    std::memcpy(b, data + 12, 4);
    reserved = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
    std::memcpy(b, data + 16, 4);
    crc = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
  }
  if (version != kVersion) return Status::error("unsupported version");
  if (reserved != 0) return Status::error("reserved != 0");
  const std::size_t body_len = len - 20;
  if (crc32(data + 20, body_len) != crc) return Status::error("crc32 mismatch");

  Reader r{data + 20, body_len, 0, false, {}};
  std::uint32_t meta[kNumMeta] = {0};
  for (std::size_t i = 0; i < kNumMeta; ++i)
    if (!r.u32(&meta[i])) return Status::error(r.why);

  const std::uint32_t base_vocab = meta[0];
  const std::uint32_t model_vocab = meta[1];
  const std::uint32_t num_merges = meta[2];
  const std::uint32_t num_added = meta[3];
  const std::uint32_t first_added = meta[4];
  const std::uint32_t num_special = meta[5];
  const std::uint32_t eos = meta[6];
  const std::uint32_t num_decomp = meta[7];
  const std::uint32_t num_ccc = meta[8];
  const std::uint32_t num_comp = meta[9];
  // meta[10..13] = range counts (ws/L/N/M); meta[14] = reserved.

  // Pinned contract (Qwen3.5-0.8B-Base @ dc7cdfe).
  if (base_vocab != kBaseVocabSize)
    return Status::error("base vocab size != 248044");
  if (model_vocab != kModelVocabSize)
    return Status::error("model vocab size != 248320");
  if (first_added != kBaseVocabSize)
    return Status::error("first added id != 248044");
  if (num_added != kNumAdded) return Status::error("num added != 33");
  if (first_added + num_added > model_vocab)
    return Status::error("added ids exceed model vocab");
  if (num_special > num_added)
    return Status::error("num special > num added");
  if (eos != kEosTokenId)
    return Status::error("eos token id != 248044");
  if (meta[14] != 0) return Status::error("reserved meta != 0");

  base_vocab_size_ = base_vocab;
  model_vocab_size_ = model_vocab;
  first_added_id_ = first_added;
  eos_token_id_ = eos;

  std::vector<std::vector<std::uint8_t>> sections;
  sections.reserve(kNumSections);
  for (std::size_t i = 0; i < kNumSections; ++i) {
    std::uint32_t size = 0;
    if (!r.u32(&size)) return Status::error(r.why);
    if (size > r.n - r.off) return Status::error("section size overflow");
    const std::uint8_t* q = nullptr;
    if (!r.bytes(size, &q)) return Status::error(r.why);
    sections.emplace_back(q, q + size);
  }
  {
    Status s = r.finish();
    if (!s.ok) return s;
  }
  auto sec = [](std::vector<std::vector<std::uint8_t>>& v, std::size_t i) {
    return std::make_pair(v[i].data(), v[i].size());
  };

  // --- S1: base vocab [u16 len][bytes] x base_vocab -----------------------
  {
    auto pr = sec(sections, 0);
    Reader sr{pr.first, pr.second, 0, false, {}};
    base_vocab_.clear();
    base_vocab_.reserve(base_vocab);
    std::vector<std::uint8_t> single_byte(256, 0xFF);
    for (std::uint32_t i = 0; i < base_vocab; ++i) {
      std::uint16_t ln = 0;
      if (!sr.u16(&ln)) return Status::error("S1 truncated");
      if (ln < 1) return Status::error("S1 empty token");
      const std::uint8_t* q = nullptr;
      if (!sr.bytes(ln, &q)) return Status::error("S1 truncated");
      base_vocab_.emplace_back(reinterpret_cast<const char*>(q), ln);
      if (ln == 1) {
        if (single_byte[q[0]] != 0xFF)
          return Status::error("S1 duplicate single-byte token");
        single_byte[q[0]] = 0;
      }
    }
    for (int b = 0; b < 256; ++b)
      if (single_byte[b] == 0xFF)
        return Status::error("S1 missing single-byte token");
    // Duplicate byte-string check (BPE correctness requires injectivity).
    bytes_to_id_.clear();
    bytes_to_id_.reserve(base_vocab * 2);
    for (std::uint32_t i = 0; i < base_vocab; ++i) {
      auto it = bytes_to_id_.emplace(base_vocab_[i], i);
      if (!it.second) return Status::error("S1 duplicate token bytes");
    }
    // Derived: byte -> single-byte id.
    byte_to_base_id_.assign(256, 0);
    for (std::uint32_t i = 0; i < base_vocab; ++i)
      if (base_vocab_[i].size() == 1)
        byte_to_base_id_[static_cast<std::uint8_t>(base_vocab_[i][0])] = i;
  }

  // --- S2: merges [u32 left][u32 right] x num_merges -----------------------
  {
    auto pr = sec(sections, 1);
    if (pr.second != static_cast<std::size_t>(num_merges) * 8)
      return Status::error("S2 size mismatch");
    merges_.clear();
    merges_.reserve(num_merges);
    std::unordered_map<std::uint64_t, std::uint32_t> rank_seen;
    rank_seen.reserve(num_merges * 2);
    for (std::uint32_t i = 0; i < num_merges; ++i) {
      std::uint32_t l = 0, rr = 0;
      std::memcpy(&l, pr.first + i * 8, 4);
      std::memcpy(&rr, pr.first + i * 8 + 4, 4);
      if (l >= base_vocab || rr >= base_vocab)
        return Status::error("S2 id out of range");
      const std::uint64_t key = (static_cast<std::uint64_t>(l) << 32) | rr;
      if (rank_seen.count(key))
        return Status::error("S2 duplicate merge pair");
      rank_seen.emplace(key, i);
      merges_.emplace_back(l, rr);
      const std::string concat = base_vocab_[l] + base_vocab_[rr];
      if (!bytes_to_id_.count(concat))
        return Status::error("S2 merge output not in vocab");
    }
    // (left, right) -> rank, as parallel sorted arrays for binary search.
    std::vector<std::pair<std::uint64_t, std::uint32_t>> v;
    v.reserve(num_merges);
    for (std::uint32_t i = 0; i < num_merges; ++i) {
      const std::uint64_t key =
          (static_cast<std::uint64_t>(merges_[i].first) << 32) |
          merges_[i].second;
      v.emplace_back(key, i);
    }
    std::sort(v.begin(), v.end());
    merge_rank_sorted_.clear();
    merge_rank_values_.clear();
    merge_rank_sorted_.reserve(num_merges);
    merge_rank_values_.reserve(num_merges);
    for (const auto& kv : v) {
      merge_rank_sorted_.push_back(kv.first);
      merge_rank_values_.push_back(kv.second);
    }
  }

  // --- S3: added tokens [u32 id][u8 special][u16 len][utf8] x num_added ----
  {
    auto pr = sec(sections, 2);
    Reader sr{pr.first, pr.second, 0, false, {}};
    added_.clear();
    added_.reserve(num_added);
    int sp = 0;
    for (std::uint32_t i = 0; i < num_added; ++i) {
      std::uint32_t id = 0;
      std::uint8_t spflag = 0;
      std::uint16_t ln = 0;
      if (!sr.u32(&id) || !sr.u8(&spflag) || !sr.u16(&ln))
        return Status::error("S3 truncated");
      const std::uint8_t* q = nullptr;
      if (!sr.bytes(ln, &q)) return Status::error("S3 truncated");
      if (id != first_added + i) return Status::error("S3 ids not contiguous");
      const std::string utf8(reinterpret_cast<const char*>(q), ln);
      {
        std::vector<std::uint32_t> tmp;
        if (!utf8_decode(utf8, &tmp).ok)
          return Status::error("S3 invalid UTF-8");
      }
      if (spflag > 1) return Status::error("S3 special flag > 1");
      if (spflag) ++sp;
      added_.push_back(AddedToken{id, spflag != 0, utf8});
    }
    if (sp != static_cast<int>(num_special))
      return Status::error("S3 special count mismatch");
    for (std::size_t i = 0; i + 1 < added_.size(); ++i)
      if (added_[i].utf8 == added_[i + 1].utf8)
        return Status::error("S3 duplicate added token string");
  }

  // --- S4: ccc [u32 cp][u8 ccc] x num_ccc ----------------------------------
  ccc_.assign(kDenseSize, 0);
  {
    auto pr = sec(sections, 3);
    Reader sr{pr.first, pr.second, 0, false, {}};
    std::uint32_t prev = 0;
    bool first = true;
    for (std::uint32_t i = 0; i < num_ccc; ++i) {
      std::uint32_t cp = 0;
      std::uint8_t cc = 0;
      if (!sr.u32(&cp) || !sr.u8(&cc)) return Status::error("S4 truncated");
      if (cp > kMaxCp || cc < 1 || cc > 254)
        return Status::error("S4 out of range");
      if (!first && cp <= prev) return Status::error("S4 not sorted");
      prev = cp;
      first = false;
      ccc_[cp] = cc;
    }
  }

  // --- S5: decomp [u32 cp][u16 n][u32 cp x n] x num_decomp -----------------
  decomp_.clear();
  decomp_.reserve(num_decomp);
  {
    auto pr = sec(sections, 4);
    Reader sr{pr.first, pr.second, 0, false, {}};
    std::uint32_t prev = 0;
    bool first = true;
    for (std::uint32_t i = 0; i < num_decomp; ++i) {
      std::uint32_t cp = 0;
      std::uint16_t cnt = 0;
      if (!sr.u32(&cp) || !sr.u16(&cnt)) return Status::error("S5 truncated");
      if (cp > kMaxCp || cnt < 1 || cnt > 16)
        return Status::error("S5 out of range");
      if (!first && cp <= prev) return Status::error("S5 not sorted");
      prev = cp;
      first = false;
      DecompEntry e;
      e.cp = cp;
      e.parts.reserve(cnt);
      for (std::uint16_t k = 0; k < cnt; ++k) {
        std::uint32_t p = 0;
        if (!sr.u32(&p)) return Status::error("S5 truncated");
        if (p > kMaxCp) return Status::error("S5 part out of range");
        e.parts.push_back(p);
      }
      decomp_.push_back(std::move(e));
    }
  }

  // --- S6: comp [u32 a][u32 b][u32 c] x num_comp ----------------------------
  comp_.clear();
  comp_.reserve(num_comp);
  {
    auto pr = sec(sections, 5);
    Reader sr{pr.first, pr.second, 0, false, {}};
    std::uint64_t prev = 0;
    bool first = true;
    for (std::uint32_t i = 0; i < num_comp; ++i) {
      std::uint32_t a = 0, b = 0, c = 0;
      if (!sr.u32(&a) || !sr.u32(&b) || !sr.u32(&c))
        return Status::error("S6 truncated");
      if (a > kMaxCp || b > kMaxCp || c > kMaxCp)
        return Status::error("S6 out of range");
      const std::uint64_t key = (static_cast<std::uint64_t>(a) << 21) | b;
      if (!first && key <= prev) return Status::error("S6 not sorted");
      prev = key;
      first = false;
      comp_.emplace_back(key, c);
    }
  }

  // --- S7..S10: range tables -------------------------------------------------
  auto load_ranges = [&](std::size_t sidx,
                         std::vector<std::pair<std::uint32_t, std::uint32_t>>*
                             outv, const char* name) -> Status {
    auto pr = sec(sections, sidx);
    Reader sr{pr.first, pr.second, 0, false, {}};
    const std::size_t count = pr.second / 8;
    if (pr.second % 8 != 0) return Status::error(std::string(name) + " size");
    outv->clear();
    outv->reserve(count);
    std::uint32_t prev_hi = 0;
    bool first = true;
    for (std::size_t i = 0; i < count; ++i) {
      std::uint32_t lo = 0, hi = 0;
      if (!sr.u32(&lo) || !sr.u32(&hi))
        return Status::error(std::string(name) + " truncated");
      if (lo > hi || hi > kMaxCp)
        return Status::error(std::string(name) + " range invalid");
      if (!first && lo <= prev_hi)
        return Status::error(std::string(name) + " ranges overlap");
      prev_hi = hi;
      first = false;
      outv->emplace_back(lo, hi);
    }
    return Status::ok_status();
  };
  {
    Status s = load_ranges(6, &ws_ranges_, "S7");
    if (!s.ok) return s;
    s = load_ranges(7, &letter_ranges_, "S8");
    if (!s.ok) return s;
    s = load_ranges(8, &number_ranges_, "S9");
    if (!s.ok) return s;
    s = load_ranges(9, &mark_ranges_, "S10");
    if (!s.ok) return s;
  }

  // --- Derived dense class tables -------------------------------------------
  is_letter_.assign(kDenseSize, false);
  is_number_.assign(kDenseSize, false);
  is_mark_.assign(kDenseSize, false);
  is_ws_.assign(kDenseSize, false);
  auto fill = [](std::vector<bool>& t,
                 const std::vector<std::pair<std::uint32_t, std::uint32_t>>& r) {
    for (const auto& rg : r)
      for (std::uint32_t c = rg.first; c <= rg.second; ++c) t[c] = true;
  };
  fill(is_letter_, letter_ranges_);
  fill(is_number_, number_ranges_);
  fill(is_mark_, mark_ranges_);
  fill(is_ws_, ws_ranges_);
  return Status::ok_status();
}

bool Qwen35Tokenizer::is_special(std::uint32_t id) const {
  if (id < first_added_id_ || id >= first_added_id_ + added_.size())
    return false;
  return added_[id - first_added_id_].special;
}

Status Qwen35Tokenizer::utf8_to_codepoints(
    const std::string& utf8, std::vector<std::uint32_t>* out) const {
  return utf8_decode(utf8, out);
}

std::string Qwen35Tokenizer::codepoints_to_utf8(const std::uint32_t* cps,
                                                std::size_t n) {
  return utf8_encode(cps, n);
}

Status Qwen35Tokenizer::nfc_normalize(const std::uint32_t* cps, std::size_t n,
                                      std::vector<std::uint32_t>* out) const {
  return nfc_impl(cps, n, out);
}

Status Qwen35Tokenizer::nfc_impl(const std::uint32_t* cps, std::size_t n,
                                 std::vector<std::uint32_t>* out) const {
  out->clear();
  // 1. Full canonical decomposition (recursive; no cycles exist in
  //    canonical decompositions, but guard anyway).
  std::vector<std::uint32_t> stream;
  stream.reserve(n * 2);
  std::vector<std::uint32_t> stack;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t c = cps[i];
    if (c > kMaxCp || (0xD800u <= c && c <= 0xDFFFu))
      return Status::error("nfc: invalid code point input");
    stack.clear();
    stack.push_back(c);
    std::size_t guard = 0;
    while (!stack.empty()) {
      if (++guard > 100000) return Status::error("nfc: decomposition runaway");
      const std::uint32_t x = stack.back();
      stack.pop_back();
      const DecompEntry* d = nullptr;
      {
        auto it = std::lower_bound(
            decomp_.begin(), decomp_.end(), x,
            [](const DecompEntry& e, std::uint32_t v) { return e.cp < v; });
        if (it != decomp_.end() && it->cp == x) d = &*it;
      }
      if (d) {
        for (auto it = d->parts.rbegin(); it != d->parts.rend(); ++it)
          stack.push_back(*it);
      } else {
        stream.push_back(x);
      }
    }
  }
  // 2+3. Canonical ordering + composition, EXACT port of the pinned engine's
  // algorithm (Rust `unicode-normalization` Decompositions + Recompositions,
  // as used by the tokenizers crate's NFC normalizer).  This is NOT
  // "sort everything, then compose": the decomposition stream is emitted in
  // batches — a ccc == 0 code point (or end of input) triggers a STABLE sort
  // of the not-yet-emitted tail by ascending ccc — and the composition runs
  // in the same pass: a non-starter may compose with the current composee
  // over a buffered (delayed) mark only when every buffered mark has a
  // STRICTLY SMALLER ccc; otherwise it is blocked and buffered itself.
  // Observable difference from a pre-sort: "U+0391 U+0301 U+093C" (ccc 0,
  // 230, 7) -> U+0386 U+093C (the 230 mark composes over the delayed 7 mark)
  // — oracle-verified.  Jamo are ccc 0 in the tables (starters; the pinned
  // oracles never reorder jamo sequences).
  struct PCp {
    std::uint8_t ccc;
    std::uint32_t cp;
  };
  const auto ccc_of = [this](std::uint32_t c) -> std::uint8_t { return ccc_[c]; };
  std::vector<PCp> buffer;  // (ccc, cp) pairs in text order
  buffer.reserve(stream.size());
  std::size_t rs = 0, re = 0;  // ready range [rs, re)
  std::size_t ip = 0;          // input position in `stream`
  auto stable_sort_pending = [&]() {
    if (re < buffer.size()) {
      std::stable_sort(buffer.begin() + static_cast<std::ptrdiff_t>(re),
                       buffer.end(),
                       [](const PCp& a, const PCp& b) { return a.ccc < b.ccc; });
    }
    re = buffer.size();
  };
  auto decomp_next = [&](PCp* item) -> bool {
    while (re == 0) {
      if (ip < stream.size()) {
        const std::uint32_t c = stream[ip++];
        if (ccc_of(c) == 0) stable_sort_pending();
        buffer.push_back(PCp{ccc_of(c), c});
      } else {
        if (buffer.empty()) return false;
        stable_sort_pending();
        break;
      }
    }
    *item = buffer[rs];
    ++rs;
    if (rs == re) {
      const std::size_t pending = buffer.size() - re;
      for (std::size_t i = 0; i < pending; ++i) buffer[i] = buffer[re + i];
      buffer.resize(pending);
      rs = re = 0;
    }
    return true;
  };
  const auto composeable = [&](std::uint32_t a, std::uint32_t b,
                               std::uint32_t* out_cp) -> bool {
    const std::uint64_t key =
        (static_cast<std::uint64_t>(a) << 21) | static_cast<std::uint64_t>(b);
    auto it = std::lower_bound(
        comp_.begin(), comp_.end(), key,
        [](const std::pair<std::uint64_t, std::uint32_t>& kv,
           std::uint64_t k) { return kv.first < k; });
    if (it != comp_.end() && it->first == key) {
      *out_cp = it->second;
      return true;
    }
    return false;
  };
  // Recompositions state machine (mirrors the engine iterator).
  std::vector<std::uint32_t> cbuf;  // blocked (delayed) marks
  std::size_t drain_i = 0;
  bool has_composee = false;
  std::uint32_t composee = 0;
  int last_ccc = -1;  // -1 = none
  enum State { kComposing, kPurging, kFinished };
  State state = kComposing;
  while (true) {
    if (state == kComposing) {
      bool early = false;
      std::uint32_t ch = 0;
      PCp item;
      while (!early) {
        if (!decomp_next(&item)) break;
        ch = item.cp;
        const std::uint8_t ch_class = ccc_of(ch);
        if (!has_composee) {
          if (ch_class != 0) {
            out->push_back(ch);
            early = true;
            break;
          }
          composee = ch;
          has_composee = true;
          continue;
        }
        if (last_ccc < 0) {
          std::uint32_t r = 0;
          if (composeable(composee, ch, &r)) {
            composee = r;
            continue;
          }
          if (ch_class == 0) {
            out->push_back(composee);
            composee = ch;
            early = true;
            break;
          }
          cbuf.push_back(ch);
          last_ccc = ch_class;
        } else if (static_cast<std::uint8_t>(last_ccc) >= ch_class) {
          // ch is blocked from the composee.
          if (ch_class == 0) {
            out->push_back(composee);
            composee = ch;
            last_ccc = -1;
            state = kPurging;
            drain_i = 0;
            early = true;
            break;
          }
          cbuf.push_back(ch);
          last_ccc = ch_class;
        } else {
          std::uint32_t r = 0;
          if (composeable(composee, ch, &r)) {
            composee = r;
            continue;
          }
          cbuf.push_back(ch);
          last_ccc = ch_class;
        }
      }
      if (early) continue;
      state = kFinished;
      drain_i = 0;
      if (has_composee) {
        out->push_back(composee);
        has_composee = false;
        continue;
      }
      // fall through to Finished handling
    }
    if (state == kPurging) {
      if (drain_i < cbuf.size()) {
        out->push_back(cbuf[drain_i++]);
        continue;
      }
      cbuf.clear();
      state = kComposing;
      continue;
    }
    // kFinished
    if (drain_i < cbuf.size()) {
      out->push_back(cbuf[drain_i++]);
      continue;
    }
    cbuf.clear();
    if (has_composee) {
      out->push_back(composee);
      has_composee = false;
      continue;
    }
    break;
  }
  return Status::ok_status();
}

Status Qwen35Tokenizer::pretokenize(const std::string& nfc_utf8,
                                    std::vector<std::string>* out) const {
  out->clear();
  std::vector<std::uint32_t> cps;
  Status s = utf8_decode(nfc_utf8, &cps);
  if (!s.ok) return s;
  // Byte offset of each code point (for slice extraction).
  std::vector<std::size_t> byte_off(cps.size());
  {
    std::size_t bo = 0;
    for (std::size_t i = 0; i < cps.size(); ++i) {
      byte_off[i] = bo;
      bo += utf8_seq_len(cps[i]);
    }
  }
  RPool pool;
  const int top = build_pattern(pool);
  Matcher mh{&is_letter_, &is_number_, &is_mark_, &is_ws_, &pool};
  // match_top reports the absolute END of the match (not a length).
  std::size_t pos = 0;
  while (pos < cps.size()) {
    std::size_t m_end = 0;
    if (!mh.match_top(top, cps, pos, &m_end) || m_end == pos)
      return Status::error("pretokenize: no branch matched (internal)");
    if (m_end > cps.size())
      return Status::error("pretokenize: match beyond input (internal)");
    const std::size_t start = byte_off[pos];
    const std::size_t last = m_end - 1;
    const std::size_t end = byte_off[last] + utf8_seq_len(cps[last]);
    out->push_back(nfc_utf8.substr(start, end - start));
    pos = m_end;
  }
  return Status::ok_status();
}

Status Qwen35Tokenizer::bpe_encode(const std::string& pre_token_utf8,
                                   std::vector<std::uint32_t>* out) const {
  out->clear();
  std::vector<std::uint32_t> pieces;
  pieces.reserve(pre_token_utf8.size());
  for (char ch : pre_token_utf8)
    pieces.push_back(byte_to_base_id_[static_cast<std::uint8_t>(ch)]);

  auto merge_rank = [&](std::uint32_t l, std::uint32_t r) -> std::uint32_t {
    const std::uint64_t key = (static_cast<std::uint64_t>(l) << 32) | r;
    auto it = std::lower_bound(merge_rank_sorted_.begin(),
                               merge_rank_sorted_.end(), key);
    if (it != merge_rank_sorted_.end() && *it == key)
      return merge_rank_values_[static_cast<std::size_t>(
          std::distance(merge_rank_sorted_.begin(), it))];
    return std::numeric_limits<std::uint32_t>::max();
  };

  while (pieces.size() >= 2) {
    std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
    long best_pos = -1;
    for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
      const std::uint32_t rk = merge_rank(pieces[i], pieces[i + 1]);
      if (rk < best_rank) {
        best_rank = rk;
        best_pos = static_cast<long>(i);
      }
    }
    if (best_pos < 0) break;
    const std::uint32_t l = pieces[best_pos];
    const std::uint32_t r = pieces[best_pos + 1];
    const std::string concat = base_vocab_[l] + base_vocab_[r];
    auto it = bytes_to_id_.find(concat);
    if (it == bytes_to_id_.end())
      return Status::error("bpe: merge output not in vocab (internal)");
    pieces[best_pos] = it->second;
    pieces.erase(pieces.begin() + best_pos + 1);
  }
  *out = std::move(pieces);
  return Status::ok_status();
}

Status Qwen35Tokenizer::encode(const std::string& utf8_text,
                               std::vector<std::uint32_t>* out) const {
  out->clear();
  // 1. NFC normalize.
  std::vector<std::uint32_t> cps;
  Status s = utf8_decode(utf8_text, &cps);
  if (!s.ok) return s;
  std::vector<std::uint32_t> nfc_cps;
  s = nfc_impl(cps.data(), cps.size(), &nfc_cps);
  if (!s.ok) return s;
  const std::string nfc_utf8 = utf8_encode(nfc_cps.data(), nfc_cps.size());

  // 2. Added-token split + (3) pre-tokenization + (4) BPE, chunk by chunk.
  const std::size_t len = nfc_utf8.size();
  std::size_t pos = 0;
  while (pos < len) {
    // Longest added-token match at pos (no two added tokens overlap by
    // prefix in the pinned set, so at most one matches).
    long best = -1;
    for (std::size_t i = 0; i < added_.size(); ++i) {
      const std::string& t = added_[i].utf8;
      if (t.size() <= len - pos &&
          nfc_utf8.compare(pos, t.size(), t) == 0) {
        if (best < 0 || t.size() > added_[best].utf8.size()) best = static_cast<long>(i);
      }
    }
    if (best >= 0) {
      out->push_back(added_[static_cast<std::size_t>(best)].id);
      pos += added_[static_cast<std::size_t>(best)].utf8.size();
      continue;
    }
    // Next added-token start strictly after pos.
    std::size_t p = len;
    for (std::size_t q = pos + 1; q < len; ++q) {
      bool hit = false;
      for (const auto& a : added_) {
        if (a.utf8.size() <= len - q &&
            nfc_utf8.compare(q, a.utf8.size(), a.utf8) == 0) {
          p = q;
          hit = true;
          break;
        }
      }
      if (hit) break;
    }
    const std::string chunk = nfc_utf8.substr(pos, p - pos);
    std::vector<std::string> pre;
    s = pretokenize(chunk, &pre);
    if (!s.ok) return s;
    for (const auto& pt : pre) {
      std::vector<std::uint32_t> ids;
      s = bpe_encode(pt, &ids);
      if (!s.ok) return s;
      out->insert(out->end(), ids.begin(), ids.end());
    }
    pos = p;
  }
  return Status::ok_status();
}

Status Qwen35Tokenizer::decode(const std::uint32_t* ids, std::size_t n,
                               bool skip_special_tokens,
                               std::string* out_utf8) const {
  out_utf8->clear();
  // 1) token ids -> COMPLETE byte stream: base tokens contribute their raw
  //    ByteLevel bytes (NOT validated here — a legal character may be split
  //    across tokens), added tokens contribute their literal UTF-8 (dropped
  //    when special && skip_special_tokens), padding ids contribute nothing
  //    (the pinned HF/Rust oracle silently drops them).
  std::string bytes;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t id = ids[i];
    if (id >= model_vocab_size_)
      return Status::error("decode: id out of range");
    if (id < base_vocab_size_) {
      bytes.append(base_vocab_[id]);
      continue;
    }
    const std::size_t idx = id - first_added_id_;
    if (idx < added_.size()) {
      if (!(skip_special_tokens && added_[idx].special))
        bytes.append(added_[idx].utf8);
      continue;
    }
  }
  // 2) lossy UTF-8 conversion of the whole stream (maximal invalid subpart
  //    -> one U+FFFD; valid sequences copied verbatim).  The output is
  //    always valid UTF-8.
  utf8_lossy(bytes, out_utf8);
  return Status::ok_status();
}

}  // namespace cudalm
