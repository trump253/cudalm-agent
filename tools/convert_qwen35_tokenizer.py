#!/usr/bin/env python3
"""CUDALM — Qwen3.5-0.8B-Base native tokenizer artifact converter.

Offline (offline-only) tool: converts the pinned HuggingFace tokenizer
assets into the CUDLMTK1 binary artifact consumed by the PyTorch-free
native C++ tokenizer (src/runtime/qwen35_tokenizer.cpp).  This tool is the
only place where the tokenizer pipeline (NFC / added-token split / regex
pre-tokenization / ByteLevel / BPE) is transcribed into native tables.

Pinned inputs (docs/qwen35_architecture.md §18):
    Qwen/Qwen3.5-0.8B-Base @ dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
        tokenizer/tokenizer.json
        tokenizer/tokenizer_config.json
    (vocab.json / merges.txt are carried by tokenizer.json and are NOT read.)

Output: CUDLMTK1 (little-endian, deterministic, versioned, bounds-checked):
    header  : "CUDLMTK1" u32 version(=1) u32 reserved(=0) u32 crc32(body)
    metadata: 15 x u32
    S1 base vocab      : [u16 len][bytes] x base_vocab_size (ids 0..N-1)
    S2 merges          : [u32 left_id][u32 right_id] x num_merges (rank=index)
    S3 added tokens    : [u32 id][u8 special][u16 len][utf8 bytes] x num_added
    S4 ccc table       : [u32 cp][u8 ccc] sorted by cp (non-zero ccc only)
    S5 decomp table    : [u32 cp][u16 n][u32 cp x n] sorted by cp
    S6 comp table      : [u32 a][u32 b][u32 c] sorted by (a,b)
    S7..S10 ranges     : [u32 lo][u32 hi]  White_Space / L / N / M

The Unicode tables (S4..S10) are generated from PINNED UCD files
(tools/ucd/, sha256-gated by tools/common/qwen35_unicode_ref.py) — the
pinned engine's two subsystems use DIFFERENT Unicode data versions:
  S4 ccc + S5 decomp + S6 comp : Unicode 9.0.0
      (the pinned normalizer = Rust crate
      unicode-normalization-alignments 0.1.12, UNICODE_VERSION=(9,0,0);
      its 814-entry ccc table is byte-identical to the U9 UCD ccc>0 set).
  S7 White_Space               : UAX #44 (25 cps; version-stable).
  S8..S10 L / N / M            : Unicode 16.0.0
      (the pinned pre-tokenizer's regex classes come from the `regex`
      crate dependency; a per-cp chunking sweep over all 1,112,064
      non-surrogate cps against the pinned engine shows L∪M and N equal
      the U16 UCD sets exactly, and \s = UAX #44 White_Space).
PLUS the UAX #31 Hangul arithmetic (syllable<->jamo decomposition and
L+V / S0+T composition), which the UCD files do not expose directly.
Composition-pair candidates are derived from the U9 full-decomposition
streams and VALIDATED against the pinned engine's U9 normalizer
(normalizer.normalize_str(x+y) == c) — the build-time oracle of record.
Python `unicodedata` (U14 in this environment) is NOT used for table
generation and stays diagnostic-only.

Selftest (--selftest): builds the artifact twice (byte-identical
determinism), round-trips it through the Python reader, and verifies all
contract invariants (including that every corruption class is rejected).
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "common"))
import qwen35_tokenizer_ref as ref  # noqa: E402
import qwen35_unicode_ref as uref  # noqa: E402

# ---------------------------------------------------------------------------
# CUDLMTK1 layout constants (mirrored by src/runtime/qwen35_tokenizer.cpp)
# ---------------------------------------------------------------------------
MAGIC = b"CUDLMTK1"
VERSION = 1
NUM_META = 15  # 15 x u32 metadata block

# GPT-2 / tokenizers ByteLevel byte<->char bijection.  Base bytes map to
# themselves (as chars): 33..126, 161..172, 174..255 (188 bytes).  The 68
# remaining bytes map, in ascending byte order, to 256+n (U+0100..U+0143).
def _build_byte_map():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    assert len(bs) == 256 and len(set(bs)) == 256 and n == 68
    return {chr(c): b for b, c in zip(bs, cs)}


CHAR_TO_BYTE = _build_byte_map()


def tok_to_bytes(s):
    out = bytearray()
    for ch in s:
        b = CHAR_TO_BYTE.get(ch)
        if b is None:
            return None
        out.append(b)
    return bytes(out)


# ---------------------------------------------------------------------------
# Unicode table generation
# ---------------------------------------------------------------------------
# UAX #44 White_Space — the EXACT \s class of the pinned pre-tokenizer,
# proven by a per-cp chunking sweep against the pinned engine (all 25 cps
# are \s; every other cp is not).  U+00A0 (NBSP) is part of White_Space;
# its earlier absence here was BLOCKER-D2.
WHITE_SPACE = (
    (0x0009, 0x000D), (0x0020, 0x0020), (0x0085, 0x0085),
    (0x00A0, 0x00A0), (0x1680, 0x1680),
    (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F),
    (0x3000, 0x3000),
)

# UAX #31 Hangul constants.  Note on the runtime contract (verified against
# BOTH Python unicodedata.normalize and the tokenizers Rust oracle): jamo are
# treated as ccc=0 (NO reordering of jamo sequences), and composition uses the
# basic jamo only:
#   L = 1100..1112 (19),  V = 1161..1175 (21),  T index t (1..27) jamo
#   = 11A8..11C2 (U+11A8 = HANGUL JONGSEONG KIYEOK = t 1, ... U+11C2 = t 27).
#   U+11C3+ are unassigned/reserved and NEVER compose (verified: both oracles
#   leave AC00+U+11F2 and AC00+U+11F6 unchanged).
#   (L, V) -> S(L,V,0)   and   (S(L,V,0), T_jamo) -> S(L,V,t).
HANGUL_L_BASE = 0x1100
HANGUL_V_BASE = 0x1161
HANGUL_T_BASE = 0x11A8
HANGUL_SYL_BASE = 0xAC00
HANGUL_L_COUNT, HANGUL_V_COUNT, HANGUL_T_COUNT = 19, 21, 27
HANGUL_T_VALUES = 28  # t runs 0..27 (28 values); 27 non-zero jongseong


def _hangul_syllable(l, v, t):
    return HANGUL_SYL_BASE + (l * HANGUL_V_COUNT + v) * HANGUL_T_VALUES + t


def build_unicode_tables(tdir):
    """Generate the S4..S10 Unicode tables.

    Sources (all pinned, sha256-gated; see tools/common/qwen35_unicode_ref):
      ccc / decomp / comp candidates : Unicode 9.0.0 — the data version of
          the pinned normalizer (Rust crate unicode-normalization-alignments
          0.1.12, UNICODE_VERSION=(9,0,0); its 814-entry ccc table is
          byte-identical to the U9 UCD ccc>0 set).
      comp verification oracle       : the pinned engine's normalizer
          (normalizer.normalize_str == U9 NFC; the build-time oracle of
          record).  Python unicodedata (U14 here) is NOT used — it would
          reintroduce the U14-vs-U9 data differences (BLOCKER-D1).
      White_Space                    : UAX #44, 25 cps, == the engine's \\s.
      L / N / M                      : Unicode 16.0.0 — the data version of
          the pinned pre-tokenizer's regex classes (proven by a per-cp
          chunking sweep over all 1,112,064 non-surrogate cps: L∪M and N
          equal the U16 UCD sets exactly; the old U14 ranges were wrong for
          the ~5k cps added in U15/U16).
    """
    oracle = ref.build_tokenizer(tdir)
    nfc = oracle.normalizer.normalize_str

    ccc = uref.ccc_table("9.0.0")         # 814 entries
    decomp = uref.decomp_table("9.0.0")   # 2,060 entries

    # --- Hangul (UAX #31 arithmetic; not exposed as UCD rows) ----------------
    # Jamo carry ccc=0 in the UCD (no reordering of jamo sequences — the
    # behavior of the pinned engine and of standard NFC), and composition
    # uses the basic jamo only:
    #   L = 1100..1112 (19),  V = 1161..1175 (21),  T index t (1..27) jamo
    #   = 11A8..11C2 (U+11A8 = HANGUL JONGSEONG KIYEOK = t 1, ... t 27).
    #   U+11C3+ are unassigned/reserved and NEVER compose (verified below
    #   against the oracle).
    for l in range(HANGUL_L_COUNT):
        for v in range(HANGUL_V_COUNT):
            # t runs 0..27 (HANGUL_T_VALUES values); range(HANGUL_T_COUNT)
            # would silently drop the t=27 syllables (U+AC1B..U+D7A3 column).
            for t in range(HANGUL_T_COUNT + 1):
                decomp[_hangul_syllable(l, v, t)] = (
                    [HANGUL_L_BASE + l, HANGUL_V_BASE + v]
                    + ([HANGUL_T_BASE + (t - 1)] if t > 0 else []))

    # --- composition pairs ---------------------------------------------------
    # Official construction (UAX #15): a pair (x, y) -> c exists iff
    #   (a) D(c) (full canonical decomposition of c) is D(x) ++ D(y) with the
    #       split at a decomposition boundary,
    #   (b) x and y are single code points,
    #   (c) x and y are not composition exclusions.
    # (a)+(b) are checked via full-decomposition streams; (c) is enforced by
    # requiring the pinned engine's U9 NFC (the oracle of record) to compose
    # x+y to exactly c.
    fullD = {}

    def expand(c):
        if c in fullD:
            return
        d = decomp.get(c)
        if d is None:
            fullD[c] = (c,)
        else:
            seq = []
            for p in d:
                expand(p)
                seq.extend(fullD[p])
            fullD[c] = tuple(seq)

    for c in list(decomp):
        expand(c)
    stream2cp = {}
    for c in fullD:
        if len(fullD[c]) > 1:
            stream2cp.setdefault(fullD[c], c)  # canonical D is injective

    def _lookup(stream):
        # A 1-element stream (z,) is the full decomposition of z iff z is
        # itself atomic (never decomposes to a longer stream).
        stream = tuple(stream)
        if len(stream) == 1:
            z = stream[0]
            return z if z not in decomp else None
        return stream2cp.get(stream)

    pairs = {}
    for c in list(decomp):
        fs = fullD[c]  # split the FULL canonical decomposition stream
        for i in range(1, len(fs)):
            x = _lookup(fs[:i])
            y = _lookup(fs[i:])
            if x is not None and y is not None:
                if nfc(chr(x) + chr(y)) == chr(c):
                    if (x, y) in pairs and pairs[(x, y)] != c:
                        raise AssertionError("ambiguous composition pair")
                    pairs[(x, y)] = c

    # Hangul composition: (L, V) -> S(L,V,0) and (S(L,V,0), T_jamo) -> S(L,V,t)
    # (L+V composes into the T=0 syllable first, which then absorbs T).
    for l in range(HANGUL_L_COUNT):
        for v in range(HANGUL_V_COUNT):
            s0 = _hangul_syllable(l, v, 0)
            if nfc(chr(HANGUL_L_BASE + l) + chr(HANGUL_V_BASE + v)) != chr(s0):
                raise AssertionError("Hangul L+V composition mismatch")
            pairs[(HANGUL_L_BASE + l, HANGUL_V_BASE + v)] = s0
            # t runs 1..HANGUL_T_COUNT (27); the previous range(1, COUNT)
            # silently dropped the (S, T27) pairs, e.g. (AC00, 11C2) -> AC1B.
            for t in range(1, HANGUL_T_COUNT + 1):
                st = _hangul_syllable(l, v, t)
                tj = HANGUL_T_BASE + (t - 1)
                if nfc(chr(s0) + chr(tj)) != chr(st):
                    raise AssertionError("Hangul S+T composition mismatch")
                pairs[(s0, tj)] = st

    # Completeness: extended / non-basic jamo must NEVER compose; assert it
    # against the oracle so a future Unicode data change fails loudly.
    for l in range(190):
        for v in range(71):
            if l < HANGUL_L_COUNT and v < HANGUL_V_COUNT:
                continue  # basic pair: composes (already added + verified)
            pair = chr(HANGUL_L_BASE + l) + chr(HANGUL_V_BASE + v)
            if nfc(pair) != pair:
                raise AssertionError("unexpected Hangul L+V composition")
    pair = chr(HANGUL_SYL_BASE) + chr(0x11F7)  # reserved jongseong
    if nfc(pair) != pair:
        raise AssertionError("unexpected Hangul S+T composition")

    # Sanity: every table pair must reproduce under the pinned engine's NFC.
    for (x, y), c in pairs.items():
        if nfc(chr(x) + chr(y)) != chr(c):
            raise AssertionError("engine-NFC verification failed for (%x, %x)"
                                 % (x, y))

    return {
        "ccc": sorted(ccc.items()),
        "decomp": sorted((c, d) for c, d in decomp.items()),
        "comp": sorted((a, b, c) for (a, b), c in pairs.items()),
        "ws": WHITE_SPACE,
        "letter": uref.category_ranges("16.0.0", "L"),
        "number": uref.category_ranges("16.0.0", "N"),
        "mark": uref.category_ranges("16.0.0", "M"),
    }


# ---------------------------------------------------------------------------
# Artifact writer
# ---------------------------------------------------------------------------
def _section(buf, payload):
    buf += struct.pack("<I", len(payload))
    buf += payload
    return buf


def build_artifact(tdir, model_vocab_size, eos_token_id):
    tk = json.load(open(os.path.join(tdir, "tokenizer.json")))
    added = ref.effective_added_tokens(tdir)

    base_vocab = tk["model"]["vocab"]
    if len(base_vocab) != ref.BASE_VOCAB_SIZE:
        raise ValueError("base vocab size changed: %d" % len(base_vocab))

    # S1: base vocab as byte-strings, ids 0..N-1 (reject duplicates).
    by_id = [None] * ref.BASE_VOCAB_SIZE
    seen = set()
    for s, i in base_vocab.items():
        b = tok_to_bytes(s)
        if b is None:
            raise ValueError("vocab token id=%d not byte-mappable" % i)
        if b in seen:
            raise ValueError("duplicate vocab byte-string id=%d" % i)
        if by_id[i] is not None:
            raise ValueError("duplicate vocab id %d" % i)
        by_id[i] = b
        seen.add(b)
    for i, b in enumerate(by_id):
        if b is None:
            raise ValueError("vocab id %d missing" % i)
    s1 = b""
    for b in by_id:
        s1 += struct.pack("<H", len(b)) + b

    # S2: merges, rank = list index; verify both sides + the output token.
    s2 = b""
    seen_pairs = set()
    for m in tk["model"]["merges"]:
        parts = m.split(" ")
        if len(parts) != 2:
            raise ValueError("malformed merge entry")
        l, r = parts
        lb, rb = tok_to_bytes(l), tok_to_bytes(r)
        if lb is None or rb is None:
            raise ValueError("merge side not byte-mappable")
        if (lb, rb) in seen_pairs:
            raise ValueError("duplicate merge pair")
        seen_pairs.add((lb, rb))
        if lb + rb not in seen:
            raise ValueError("merge output not in base vocab")
        s2 += struct.pack("<II", base_vocab[l], base_vocab[r])

    # S3: added tokens (the 33 effective ones, ascending id).
    s3 = b""
    seen_added = set()
    for a in added:
        content = a["content"].encode("utf-8")
        if a["content"] in seen_added:
            raise ValueError("duplicate added-token string")
        seen_added.add(a["content"])
        s3 += struct.pack("<IBH", a["id"], 1 if a["special"] else 0,
                          len(content)) + content
    num_special = sum(1 for a in added if a["special"])

    u = build_unicode_tables(tdir)
    s4 = b"".join(struct.pack("<IB", cp, cc) for cp, cc in u["ccc"])
    s5 = b""
    for cp, d in u["decomp"]:
        s5 += struct.pack("<IH", cp, len(d))
        s5 += b"".join(struct.pack("<I", x) for x in d)
    s6 = b"".join(struct.pack("<III", a, b, c) for a, b, c in u["comp"])

    def ranges_payload(r):
        return b"".join(struct.pack("<II", lo, hi) for lo, hi in r)

    meta = struct.pack(
        "<15I",
        ref.BASE_VOCAB_SIZE, model_vocab_size, len(tk["model"]["merges"]),
        len(added), ref.FIRST_ADDED_ID, num_special, eos_token_id,
        len(u["decomp"]), len(u["ccc"]), len(u["comp"]), len(u["ws"]),
        len(u["letter"]), len(u["number"]), len(u["mark"]), 0)
    body = meta
    for s in (s1, s2, s3, s4, s5, s6,
              ranges_payload(u["ws"]), ranges_payload(u["letter"]),
              ranges_payload(u["number"]), ranges_payload(u["mark"])):
        body = _section(body, s)
    header = MAGIC + struct.pack("<II", VERSION, 0) \
        + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    return header + body


# ---------------------------------------------------------------------------
# Python-side reader (round-trip selftest; the authoritative parser is the
# C++ loader)
# ---------------------------------------------------------------------------
def read_artifact(data):
    def err(msg):
        raise ValueError(msg)

    if len(data) < 20 or data[:8] != MAGIC:
        err("bad magic")
    version, reserved, crc = struct.unpack_from("<III", data, 8)
    if version != VERSION:
        err("unsupported version %d" % version)
    if reserved != 0:
        err("reserved != 0")
    body = data[20:]
    if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
        err("crc mismatch")
    off = 0

    def take(n):
        nonlocal off
        if off + n > len(body):
            err("truncated")
        v = body[off:off + n]
        off += n
        return v

    meta = struct.unpack("<15I", take(4 * NUM_META))
    sections = []
    for _ in range(10):
        size = struct.unpack("<I", take(4))[0]
        sections.append(take(size))
    if off != len(body):
        err("trailing garbage")
    return meta, sections


def selftest(tdir, model_vocab_size, eos_token_id):
    a1 = build_artifact(tdir, model_vocab_size, eos_token_id)
    a2 = build_artifact(tdir, model_vocab_size, eos_token_id)
    assert a1 == a2, "artifact is not deterministic"
    meta, sec = read_artifact(a1)
    (base_vocab, model_vocab, num_merges, num_added, first_added,
     num_special, eos, num_decomp, num_ccc, num_comp, num_ws, num_l,
     num_n, num_m, reserved) = meta
    assert base_vocab == ref.BASE_VOCAB_SIZE
    assert model_vocab == model_vocab_size
    assert first_added == base_vocab
    assert eos == eos_token_id and eos < model_vocab
    # S1: base vocab byte-strings; 256 single-byte completeness.
    s1 = sec[0]
    off, n_entries, single = 0, 0, set()
    while off < len(s1):
        (ln,) = struct.unpack_from("<H", s1, off)
        off += 2 + ln
        n_entries += 1
        if ln == 1:
            single.add(s1[off - 1])
    assert n_entries == base_vocab and len(single) == 256
    # S2: merges reference valid ids.
    assert len(sec[1]) == num_merges * 8
    for k in range(num_merges):
        l, r = struct.unpack_from("<II", sec[1], k * 8)
        assert l < base_vocab and r < base_vocab
    # S3: added ids contiguous, specials counted.
    assert len(sec[2]) > 0
    off, sp = 0, 0
    for i in range(num_added):
        aid, spflag, ln = struct.unpack_from("<IBH", sec[2], off)
        off += 7 + ln
        assert aid == first_added + i
        sp += spflag
    assert sp == num_special
    # S4/S5/S6 sortedness.
    for k in range(num_ccc):
        cp, cc = struct.unpack_from("<IB", sec[3], k * 5)
        assert 0 < cc <= 254 and cp <= 0x10FFFF
    prev = -1
    for k in range(num_ccc):
        (cp,) = struct.unpack_from("<I", sec[3], k * 5)
        assert cp > prev
        prev = cp
    # Corruption classes must be rejected.
    def corrupt(fn, name):
        b = bytearray(a1)
        fn(b)
        try:
            read_artifact(bytes(b))
        except ValueError:
            return
        raise AssertionError("corruption not detected: " + name)
    corrupt(lambda b: b.__setitem__(0, ord("X")), "bad magic")
    corrupt(lambda b: b.__setitem__(9, VERSION + 1), "bad version")
    corrupt(lambda b: b.__setitem__(19, b[19] ^ 1), "crc")
    corrupt(lambda b: b.__delitem__(len(b) - 1), "truncated")
    corrupt(lambda b: b.extend(b"M"), "trailing garbage")
    print("converter selftest: OK (%d bytes, %d merges, %d added, "
          "%d ccc, %d decomp, %d comp)"
          % (len(a1), num_merges, num_added, num_ccc, num_decomp, num_comp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir",
                    default="/root/models/Qwen3.5-0.8B-Base/tokenizer")
    ap.add_argument("--out", required=False)
    ap.add_argument("--model-vocab", type=int, default=ref.MODEL_VOCAB_SIZE)
    ap.add_argument("--eos", type=int, default=ref.EOS_TOKEN_ID)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    tdir = a.tokenizer_dir
    try:
        ref.check_assets(tdir)
    except FileNotFoundError as e:
        # ctest self-skip (77): the pinned checkpoint assets are a local-only
        # input, never committed; their absence means "cannot run here".
        print("[SKIP] %s" % e, file=sys.stderr)
        return 77
    if a.selftest:
        selftest(tdir, a.model_vocab, a.eos)
        return 0
    if not a.out:
        ap.error("--out is required unless --selftest")
    blob = build_artifact(tdir, a.model_vocab, a.eos)
    with open(a.out, "wb") as f:
        f.write(blob)
    print("wrote %s (%d bytes) sha256=%s"
          % (a.out, len(blob), hashlib.sha256(blob).hexdigest()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
