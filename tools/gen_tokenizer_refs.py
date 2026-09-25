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

    # NFC vectors (codepoint level; the oracles implement no reordering).
    nfc_cases = [
        "e\u0301", "A\u0308", "A\u0308\u0301", "e\u0301\u0301",
        "A\u0300\u0301", "A\u0301\u0300", "A\u0344\u0301", "\u0300A",
        "\u0301\u0300A", "\u1100\u1161\u11a9", "\u11a9\u1100\u1161",
        "\u1100\u11a9\u1161", "\uac00\u11a8", "\u1100\u1161", "\ud7a3",
        "\u1100\u1161\u11f6", "\u1fbe", "\u1fee", "\u03b9\u0385",
        "\u01fa", "\u0041\u030a\u0301", "\u00c5\u0301", "\u015a\u013b",
        "\u0061\u0327\u0042", "h\u0331", "r\u0323\u0308", "\u1d160\u1d165",
        "\u0430\u0306\u0301", "\u0915\u094d\u0915\u094d",
        "a\u0300b\u0301", "\u11b5\u1100\u1161",
    ]
    for t in nfc_cases:
        lines.append("N %s %s" % (hexify(t),
                                  hexify(unicodedata.normalize("NFC", t))))

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

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s: %d lines (%d texts, %d nfc, %d pretok)"
          % (a.out, len(lines), n, len(nfc_cases), len(pretok_cases)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
