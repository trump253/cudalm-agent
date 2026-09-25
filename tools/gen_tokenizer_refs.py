#!/usr/bin/env python3
"""CUDALM — reference corpus generator for the native tokenizer tests.

Emits a text-file corpus (hex-encoded fields, line-oriented, no JSON parser
needed on the C++ side) containing the EXACT pinned-HF-oracle results for a
multilingual test battery:

    E <text_hex> <id,id,...>         encode (raw oracle: added-token priority
                                     on; add_special_tokens is a no-op for Qwen)
    D <ids_csv> <text_hex>           decode (skip_special_tokens=False)
    D1 <ids_csv> <text_hex>          decode (skip_special_tokens=True)
    N <text_hex> <nfc_text_hex>      NFC normalization (codepoint level)
    P <text_hex> <pre1>\\x1f<pre2>...  regex pre-tokenization (NFC'd input)
    X <ids_csv> <skip 0|1> <hex>     decode of ARBITRARILY CONSTRUCTED id
                                     sequences (never produced by encode):
                                     raw-byte tokens / invalid UTF-8 / padding
                                     / added tokens in both skip modes

The corpus deliberately covers the whole pinned contract:
  * ASCII text / punctuation / apostrophe contractions (leftmost-first
    branch-1 probes such as "'mX" where branch 1 beats branch 2),
  * whitespace edge cases (spaces, tabs, \\n, \\r\\n, runs, leading/trailing,
    U+00A0 non-breaking space),
  * numbers / code / URLs / mixed punctuation,
  * accented Latin (precomposed AND decomposed forms -> NFC),
  * Greek (including polytonic + ypogegrammeni single-part decomposition),
  * Cyrillic / Arabic / Hebrew / Thai / Devanagari,
  * CJK (Chinese / Japanese / Korean incl. jamo sequences + syllables),
  * emoji / astral code points,
  * combining-mark stress (multi-level, out-of-order ccc, and the
    no-reordering probes the oracles implement),
  * added-token literals (read from the pinned assets, never hardcoded)
    embedded mid-text / at edges / adjacent / near-lookalikes,
  * the empty string and single-character / single-byte inputs,
  * a long mixed-language paragraph.

The P (pre-tokenization) lines use the pinned regex evaluated by an
INDEPENDENT engine (Python re, leftmost-first) with the UAX #44 White_Space
set and unicodedata L/N/M categories — a cross-engine check of the native
leftmost-first matcher.

Special/control token strings are handled only symbolically: they are read
from the pinned tokenizer assets and never printed.
"""

import argparse
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "common"))
import qwen35_tokenizer_ref as ref  # noqa: E402

SEP = "\x1f"  # ASCII unit separator: cannot appear in the corpus data

# UAX #44 White_Space (the set Onig/PCRE2-UCP use for \s in Unicode mode).
WS_RANGES = [(0x0009, 0x000D), (0x0020, 0x0020), (0x0085, 0x0085),
             (0x1680, 0x1680), (0x2000, 0x200A), (0x2028, 0x2029),
             (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]


def hexify(s):
    return s.encode("utf-8").hex()


def _ranges(pred):
    out = []
    start = None
    for c in range(0x110000):
        p = pred(unicodedata.category(chr(c)))
        if p and start is None:
            start = c
        elif not p and start is not None:
            out.append((start, c - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def _esc(c):
    ch = chr(c)
    if ch.isalnum() and ch not in "]\\^":
        return ch
    return "\\u%04x" % c if c < 0x10000 else "\\U%08x" % c


def _class_body(ranges):
    """Class body (no brackets) so classes can be unioned inside other
    classes."""
    return "".join(
        _esc(lo) + (("-" + _esc(hi)) if hi != lo else "")
        for lo, hi in ranges)


def build_pretok_pattern():
    """The pinned Qwen3.5 pre-tokenization regex (leftmost-first), rendered
    for Python re with explicit UAX class definitions."""
    L = _class_body(_ranges(lambda cat: cat.startswith("L")))
    N = _class_body(_ranges(lambda cat: cat.startswith("N")))
    M = _class_body(_ranges(lambda cat: cat.startswith("M")))
    WS = _class_body(WS_RANGES)
    CRNL = r"[\r\n]"
    cls_ws = "[" + WS + "]"
    neg_ws = "[^" + WS + "]"
    cls_lm = "[" + L + M + "]"
    pat = ("(?:"
           + r"'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])"  # B1 (?i:...)
           + r"|[^\r\n" + L + N + r"]?" + cls_lm + r"+"            # B2
           + r"|[" + N + r"]"                                       # B3
           + r"| ?[^" + WS + L + M + N + r"]+" + CRNL + r"*"       # B4
           + r"|" + cls_ws + r"*" + CRNL + r"+"                    # B5
           + r"|" + cls_ws + r"+(?!" + neg_ws + r")"               # B6
           + r"|" + cls_ws + r"+"                                  # B7
           + r")")
    return re.compile(pat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir",
                    default="/root/models/Qwen3.5-0.8B-Base/tokenizer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tdir = a.tokenizer_dir
    try:
        ref.check_assets(tdir)
    except FileNotFoundError as e:
        # ctest self-skip (77): pinned assets are local-only, never committed.
        print("[SKIP] %s" % e, file=sys.stderr)
        return 77
    tok = ref.build_tokenizer(tdir)

    # Added-token strings (symbolic handling: read from the assets).
    added = ref.effective_added_tokens(tdir)
    A = [x["content"] for x in added]
    eos_str = None
    for x in added:
        if x["id"] == ref.EOS_TOKEN_ID:
            eos_str = x["content"]

    texts = []
    add = texts.append

    # --- ASCII basics ------------------------------------------------------
    add("")
    add("Hello, world!")
    add("hello world")
    add("HELLO WORLD")
    add("MixedCase and snake_case and camelCase")
    add("don't stop")
    add("it's a test")
    add("I'll be there")
    add("we're ready")
    add("you've got it")
    add("rock'n'roll")
    add("a'b'c")
    add("'mX")            # leftmost-first: branch 1 ('m) beats branch 2
    add("'m'")
    add("'t up")
    add("S's")            # case-insensitive branch 1
    add("'RE 'LL 'D 'M 'VE 'S 'T")
    add("5 apples")
    add("3.14")
    add("1e10")
    add("123456789012345678901234567890")
    add("0")
    add("a1b2c3")
    add("x**2")
    add("def f(x): return x")
    add("https://example.com/a?b=1&c=2")
    add("{\"k\": [1, 2, 3]}")
    add("punct .,;:!?\"'()[]{}<>|\\/~`@#$%^&*-_=+ ")
    add("!!!")
    add("((((")
    add("--")
    add("a.b.c.d")
    # --- whitespace ---------------------------------------------------------
    add("a b")
    add("a   b")
    add("a\tb")
    add("a\nb")
    add("a\r\nb")
    add(" leading")
    add("trailing ")
    add("  double  ")
    add("tab\t\tnow")
    add("line1\nline2\n")
    add("crlf\r\n\r\n")
    add("\n\n\n")
    add(" \n ")
    add("x\t")
    add("\u00a0")
    add("a\u00a0b")
    add("\u2028\u2029")
    add("nbsp \u2000em\u2003quad")
    # --- accented / script ---------------------------------------------------
    add("héllo wörld café naïve résumé señor")
    add("e\u0301")                 # decomposed e-acute
    add("A\u0308")
    add("A\u0308\u0301")
    add("e\u0301\u0301")
    add("A\u0300\u0301")
    add("A\u0301\u0300")
    add("A\u0344\u0301")
    add("a\u0300b")
    add("\u01fa\u01f9\u01da\u01db")  # A-ring-acute, w-ring, u-ring pairs
    add("\u1e1a\u1e61")              # 3-part precomposed Latin
    add("Γειά σου Κόσμε")
    add("ἄνθρωπος")
    add("\u1fbe\u0385")              # ypogegrammeni + tonos
    add("Привет, мир!")
    add("مرحبا بالعالم")
    add("שלום")
    add("สวัสดี")
    add("नमसते")
    add("Xin chào")
    # --- CJK -----------------------------------------------------------------
    add("中文测试")
    add("日本語のテキスト")
    add("한국어 테스트")
    add("\u1100\u1161\u11a9")        # LVT jamo
    add("\u11a9\u1100\u1161")        # TVL (the oracles do not reorder)
    add("\u1100\u11a9\u1161")        # LTV (no composition)
    add("\uac00\u11a8")              # syllable + T=1 jamo
    add("\u1100\u1161")              # LV only
    add("\ud7a3")
    add("Hello 世界 world")
    add("a中b日c")
    add("中文 and English mixed 混排 text")
    # --- emoji / astral -------------------------------------------------------
    add("a b")
    add("\U0001f389\U0001f38a")
    add("x\U0001f4a9y")
    add("\U0001d165\U0001d16e\U0001d173")
    # --- added tokens (symbolic) ---------------------------------------------
    for s in A:
        add(s)
        add("ab" + s + "cd")
        add(s + " trailing")
        add("leading " + s)
        add(s + s)
        add(s + " " + s)
    add("ab" + A[0] + " " + A[5] + "cd")
    # Lookalike of the special-token FORM but not one of the pinned 33
    # (must BPE like ordinary text).  Built by split-concatenation; the raw
    # form is never emitted (docs §18 security note).
    add("text with lookalike " + "<|" + "not-a-token" + "|>" + " here")
    add("pre" + A[-1])
    if eos_str:
        add("end" + eos_str)
        add(eos_str + eos_str)
    # --- long mixed paragraph --------------------------------------------------
    add(
        "The quick brown fox jumps over the lazy dog. "
        "中文摘要：本测试覆盖多语言、组合记号与特殊标记。"
        "한국어 포함. \u03b3\u03b5\u03b9\u03ac \u03c3\u03bf\u03c5. "
        "e\u0301 is decomposed. " + (A[1] if len(A) > 1 else "") +
        " 12345 https://example.org/path?q=1 "
        "tab\there and line\nbreak.")

    lines = []
    n = len(texts)
    for t in texts:
        ids = list(tok.encode(t).ids)  # raw oracle: added-token priority on
        lines.append("E %s %s" % (hexify(t), ",".join(map(str, ids))))
        dec = tok.decode(list(ids), skip_special_tokens=False)
        lines.append("D %s %s" % (",".join(map(str, ids)), hexify(dec)))
        lines.append("D1 %s %s" % (",".join(map(str, ids)),
                                   hexify(tok.decode(list(ids),
                                                     skip_special_tokens=True))))

    # NFC vectors (codepoint level; the oracle applies UAX #15 canonical
    # ordering by CCC — stable within each combining sequence, ccc == 0 is a
    # starter boundary — before canonical composition).
    nfc_cases = [
        # (1) DESCENDING ccc — the canonical-ordering regressions.  Without
        # reordering, A+U+0315(232)+U+0300(230) would stay uncomposed;
        # standard NFC (and the pinned engine) yields U+00C0+U+0315.
        "A\u0315\u0300", "A\u0315\u0308", "A\u0315\u0301",
        "E\u0315\u0328", "a\u0315\u0300", "A\u0315\u0300\u0301",
        "A\u0315\u0308\u0300",
        # (2) ascending ccc
        "A\u0300\u0315", "A\u0300\u0301\u0315",
        # (3) EQUAL ccc — stable order must be preserved (NOT re-sorted)
        "A\u0301\u0308", "A\u0308\u0301", "A\u0301\u0300",
        "A\u0300\u0301", "e\u0301\u0301",
        # (4) leading non-starter (a mark run at the very start of input)
        "\u0300A", "\u0301\u0300A", "\u0315\u0300",
        "\u0300\u0315\u0301", "\u0344\u0301\u0300",
        # (5) multiple starters
        "A\u0300B\u0300", "a\u0300b\u0301", "r\u0323\u0308",
        # (6) precomposed character + combining marks
        "\u00c0\u0300\u0315", "\u00c5\u0301", "\u0041\u030a\u0301",
        "\u015a\u013b", "\u01fa", "\u1fbe",
        # (7) recursive canonical decomposition (a decomposition part that
        # itself decomposes; and 1FEE -> 0385)
        "\u0763", "\u1fee", "\u1d160\u1d165", "\u03b9\u0385",
        # (8) Hangul decomposition/composition (jamo are ccc 0 in the table:
        # the pinned oracles never reorder jamo)
        "\u1100\u1161\u11a9", "\u11a9\u1100\u1161",
        "\u1100\u11a9\u1161", "\uac00\u11a8", "\u1100\u1161",
        "\ud7a3", "\u1100\u1161\u11f6", "\u11b5\u1100\u1161",
        "\u1161\u1100", "\uac00\u11c2",
        # (9) canonical ordering changes a LATER composition result
        "A\u0315\u0300", "a\u0315\u0300", "A\u0315\u0300\u0301",
        # (10) mark-only sequences (no starter at all)
        "\u0301\u0300", "\u0315\u0300", "\u0300\u0315\u0301",
        # long-standing regressions (kept)
        "e\u0301", "A\u0308", "A\u0344\u0301",
        "h\u0331", "\u0061\u0327\u0042", "\u0430\u0306\u0301",
        "\u0915\u094d\u0915\u094d",
    ]
    for t in nfc_cases:
        nfc_t = unicodedata.normalize("NFC", t)
        # Engine consistency: the pinned engine applies its NFC inside
        # encode, so engine(t) must equal engine(NFC(t)) for every vector;
        # a divergence is a real oracle mismatch — fail loud (do not skip).
        assert list(tok.encode(t).ids) == list(tok.encode(nfc_t).ids), \
            "engine NFC diverges from python NFC for %r" % t
        lines.append("N %s %s" % (hexify(t), hexify(nfc_t)))

    # Pre-tokenization vectors (independent-engine reference; the native
    # code applies the same pattern after NFC on added-token-free chunks).
    pat = build_pretok_pattern()
    pretok_cases = [
        "hello", "a b", "a   b", "a\tb", "a\nb", "a\r\nb",
        " leading", "trailing ", "  x  ", "a\t", "\n\n", " \n ",
        "'mX", "it's", "S's", "a'b", "don't", "ROCK'S",
        "5 apples", "3.14", "x**2", "https://a.b/c?d=e",
        "!!!", "((((", "a.b.c", "punct .,;:!? end",
        "a\u00a0b", "x\U0001f389y", "\u4e2d \u6587", "a\u4e2db",
        "café", "e\u0301", "A\u0308", "a1b2", "UP", "MiXeD",
    ]
    for t in pretok_cases:
        nfc_t = unicodedata.normalize("NFC", t)
        pres = [m.group(0) for m in pat.finditer(nfc_t)]
        assert "".join(pres) == nfc_t, "regex does not tile the input"
        lines.append("P %s %s" % (hexify(nfc_t),
                                  SEP.join(hexify(q) for q in pres)))

    # --- arbitrary token-ID decode oracle (X lines) ----------------------------
    # The D/D1 lines above are text->encode->decode round trips, so their
    # byte streams are structurally valid UTF-8 and cannot exercise the
    # decoder's byte-level boundary.  The X lines construct id sequences
    # DIRECTLY (never produced by any encode): raw ByteLevel byte tokens,
    # padding ids and added tokens in arbitrary mix, with the expected
    # output taken from the pinned engine (decode performs a lossy UTF-8
    # conversion of the COMPLETE byte stream — a valid multi-byte character
    # may be split across several base tokens).
    import json as _json
    import random as _random

    # byte -> base token id, derived from the pinned assets (the GPT-2 /
    # tokenizers ByteLevel bijection: 188 identity bytes map to themselves,
    # the 68 remaining bytes map, in ascending byte order, to U+0100..
    # U+0143).  Never hardcoded per case; verified against the pinned
    # vocab + pinned engine below.
    identity = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    assert len(identity) == 188
    b2c, nn = {}, 0
    for b in range(256):
        b2c[b] = chr(b) if b in identity else chr(0x100 + nn)
        if b not in identity:
            nn += 1
    vocab = _json.load(open(os.path.join(tdir, "vocab.json")))
    assert all(c in vocab for c in b2c.values())
    B2I = {b: vocab[c] for b, c in b2c.items()}
    # Full map verification against the pinned engine:
    for b in range(0x80):
        assert tok.decode([B2I[b]]) == chr(b), b
    for L in range(0xC2, 0xE0):
        for C in range(0x80, 0xC0):
            assert tok.decode([B2I[L], B2I[C]]) == \
                chr(((L & 0x1F) << 6) | (C & 0x3F)), (L, C)
    for L in (0xC0, 0xC1) + tuple(range(0xF5, 0x100)):
        assert tok.decode([B2I[L]]) == "\ufffd", L
    assert tok.decode([B2I[0xE0], B2I[0xA0], B2I[0x80]]) == "\u0800"
    assert tok.decode([B2I[0xED], B2I[0x9F], B2I[0xBF]]) == "\ud7ff"
    assert tok.decode([B2I[0xEF], B2I[0xBF], B2I[0xBF]]) == "\uffff"
    assert tok.decode([B2I[0xF0], B2I[0x90], B2I[0x80],
                       B2I[0x80]]) == "\U00010000"
    assert tok.decode([B2I[0xF4], B2I[0x8F], B2I[0xBF], B2I[0xBF]]) == \
        "\U0010ffff"

    added_ids = [x["id"] for x in added]
    spec_ids = [x["id"] for x in added if x["special"]]
    nonspec_ids = [x["id"] for x in added if not x["special"]]
    PAD_LO, PAD_HI = 248077, 248319  # model-vocab padding ids

    xb = []  # arbitrary id sequences
    # (1) a single invalid continuation byte
    xb.append([B2I[0x80]])
    # (2) a single incomplete UTF-8 prefix
    xb.append([B2I[0xC3]])
    # (3) 2/3/4-byte incomplete sequences
    xb.append([B2I[0xE4]])
    xb.append([B2I[0xE4], B2I[0xB8]])
    xb.append([B2I[0xF0]])
    xb.append([B2I[0xF0], B2I[0x9F]])
    xb.append([B2I[0xF0], B2I[0x9F], B2I[0x98]])
    # (4) invalid lead + continuation combinations
    xb.append([B2I[0xC0], B2I[0x80]])
    xb.append([B2I[0xC1], B2I[0xBF]])
    xb.append([B2I[0x41], B2I[0x80]])
    xb.append([B2I[0xF5]] + [B2I[0x80]] * 3)
    xb.append([B2I[0xFF], B2I[0xFF]])
    xb.append([B2I[0xD8], B2I[0x00], B2I[0x80]])
    xb.append([B2I[0xE0], B2I[0x80], B2I[0x80]])
    xb.append([B2I[0xED], B2I[0xA0], B2I[0x80]])
    xb.append([B2I[0xF4], B2I[0x90], B2I[0x80], B2I[0x80]])
    # (5) multiple consecutive invalid sequences
    xb.append([B2I[0x80], B2I[0x81], B2I[0x82]])
    xb.append([B2I[0x80], B2I[0x41], B2I[0x80]])
    # (6) legal UTF-8 characters split across multiple base tokens
    xb.append([B2I[0xC3], B2I[0x83]])
    xb.append([B2I[0xC2], B2I[0xA0]])
    xb.append([B2I[0xE4], B2I[0xB8], B2I[0xAD]])
    xb.append([B2I[0xE2], B2I[0x80], B2I[0x99]])
    xb.append([B2I[0xF0], B2I[0x9F], B2I[0x98], B2I[0x80]])
    xb.append([B2I[0xF4], B2I[0x8F], B2I[0xBF], B2I[0xBF]])
    xb.append([B2I[0x00], B2I[0x09], B2I[0x7F]])
    # (7) base tokens + padding ids mixed
    xb.append([B2I[0x41], PAD_LO, B2I[0x42]])
    xb.append([PAD_HI, B2I[0x41]])
    xb.append([B2I[0x42], 248100, 248078, B2I[0x43]])
    xb.append([PAD_LO, B2I[0x80], PAD_HI])
    # (8) base tokens + added tokens mixed (ids from the pinned assets)
    xb.append([B2I[0x41], added_ids[0], B2I[0x42]])
    xb.append([added_ids[-1], B2I[0x41]])
    xb.append([B2I[0x41], spec_ids[0], B2I[0x42]])
    xb.append([B2I[0x41], nonspec_ids[0], B2I[0x42]])
    xb.append([B2I[0xE4], added_ids[0], B2I[0xAD]])
    # (9) special added tokens
    xb.append([spec_ids[0]])
    xb.append([spec_ids[0], spec_ids[-1]])
    xb.append([spec_ids[0], B2I[0x80]])
    # (10)+(11) every sequence in BOTH skip modes; plus a seeded random
    # fuzz of arbitrary id sequences (deterministic).
    rng = _random.Random(20250417)
    for _k in range(256):
        ids = []
        for _j in range(rng.randint(1, 12)):
            r = rng.random()
            if r < 0.45:
                ids.append(B2I[rng.randrange(256)])
            elif r < 0.65:
                ids.append(rng.randrange(PAD_LO, PAD_HI + 1))
            elif r < 0.85:
                ids.append(rng.choice(added_ids))
            else:
                ids.append(rng.randrange(ref.BASE_VOCAB_SIZE))
        xb.append(ids)

    nX = 0
    for ids in xb:
        for skip in (0, 1):
            want = tok.decode(list(ids), skip_special_tokens=bool(skip))
            lines.append("X %s %d %s"
                         % (",".join(map(str, ids)), skip, hexify(want)))
            nX += 1

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s: %d lines (%d texts, %d nfc, %d pretok, %d x)"
          % (a.out, len(lines), n, len(nfc_cases), len(pretok_cases), nX))
    return 0


if __name__ == "__main__":
    sys.exit(main())
