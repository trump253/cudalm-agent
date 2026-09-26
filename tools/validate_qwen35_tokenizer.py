#!/usr/bin/env python3
"""CUDALM — reproducible differential validation of the native tokenizer.

Compares the PyTorch-free native tokenizer (CUDLMTK1 artifact, dumped via
tools/dump_qwen35_tokenizer.cpp) against the PINNED tokenizers engine
(Qwen/Qwen3.5-0.8B-Base, sha256-gated by tools/common/qwen35_tokenizer_ref)
stage by stage, EXACTLY.  This is the stop-condition harness of docs
qwen35_architecture.md §20.5: any confirmed mismatch is a BLOCKER — never
exclude code points, never shrink a domain.

Passes (all fixed seeds; `quick` runs reduced sizes, `extended` runs the
full/exhaustive domains):
  regressions   D1 (U9-vs-U14 ccc + Divès Akuru) and D2 (U+00A0) cases at
                the N stage and the final-encode stage.
  nfc_single    NFC of every non-surrogate cp (extended) / a fixed strided
                subset (quick) — native nfc_normalize vs normalize_str.
  enc_single    final encode of "a"+cp — native encode vs engine encode.
  class_probe   pre-tokenizer class membership (L∪M / N / \\s) per cp via
                the chunking probes "a"+cp, "a"+cp+"b", cp+cp+"x".
  ws_battery    all 25 White_Space cps at word/symbol/run boundaries,
                P stage + final encode.
  decomp_cp     every decomposable cp (U9 table + Hangul): NFC + encode.
  comp_pair     every composition pair (U9 table + Hangul): NFC + encode.
  nfc_fuzz      seeded random combining-mark sequences (fixed seed).
  adjacency     structured + seeded two/three-cp sequences around the
                U9/U14 data-difference cps.
  enc_fuzz      seeded multilingual encode fuzz (fixed seed).
  pretok_fuzz   seeded pre-tokenizer stage fuzz (NFC'd input).
  dec_fuzz      seeded arbitrary-id decode fuzz (fixed seed).

Usage:
  validate_qwen35_tokenizer.py --mode quick|extended \
      --dump-binary <build/tool_qwen35_tokenizer_dump> \
      --artifact <build/data/qwen35_tokenizer.cudaltk> \
      --tokenizer-dir <checkpoint>/tokenizer

Prints one summary line per pass (samples / mismatches / seed), the git HEAD
SHA, the pinned-asset sha256 gate status, and a final PASS/FAIL verdict
(exit 0 = every pass has 0 mismatches).
"""

import argparse
import hashlib
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "common"))
import qwen35_tokenizer_ref as ref  # noqa: E402
import qwen35_unicode_ref as uref  # noqa: E402

# Fixed seeds (documented in docs/qwen35_architecture.md §20.5).
SEED_NFC_FUZZ = 777
SEED_ENC_FUZZ = 42
SEED_DEC_FUZZ = 20250417
SEED_ADJ = 20260925
SEED_PRETOK_FUZZ = 999

NFC_FUZZ_QUICK, NFC_FUZZ_EXT = 20000, 200000
ENC_FUZZ_QUICK, ENC_FUZZ_EXT = 10000, 100000
DEC_FUZZ_QUICK, DEC_FUZZ_EXT = 2000, 20000
PRETOK_FUZZ_QUICK, PRETOK_FUZZ_EXT = 10000, 100000
ADJ_QUICK, ADJ_EXT = 5000, 50000
SINGLE_QUICK_STRIDE, SINGLE_EXT_STRIDE = 16, 1

# The 98 cps with a U14-only ccc (ccc > 0 in Python U14, ccc 0 in the pinned
# U9 normalizer) — derived from the pinned U9 UCD vs this interpreter's
# unicodedata (diagnostic); hard-listed here as regression anchors.
import unicodedata as _ud  # noqa: E402  (diagnostic only)
_u9_ccc = uref.ccc_table("9.0.0")
DIFF98 = sorted(cp for cp in range(0x110000)
                if not (0xD800 <= cp <= 0xDFFF)
                and _ud.combining(chr(cp)) != _u9_ccc.get(cp, 0))
assert len(DIFF98) == 98, "U9/U14 ccc diff set changed: %d" % len(DIFF98)
DIVES_AKURU = (0x11935, 0x11930, 0x11938)
WS_CPS = [0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x0020, 0x0085, 0x00A0,
          0x1680] + list(range(0x2000, 0x200B)) + [0x2028, 0x2029, 0x202F,
                                                   0x205F, 0x3000]
assert len(WS_CPS) == 25

# ByteLevel inverse map (engine chunk chars -> raw bytes); identical to the
# map tools/gen_tokenizer_refs.py verifies against the pinned engine.
_IDENTITY = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
_B2C, _NN = {}, 0
for _B in range(256):
    _B2C[_B] = chr(_B) if _B in _IDENTITY else chr(0x100 + _NN)
    if _B not in _IDENTITY:
        _NN += 1
_C2B = {c: b for b, c in _B2C.items()}


def bytelevel_to_raw(s):
    return bytes(_C2B[ch] for ch in s).decode("utf-8")


def git_head():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class Validator:
    def __init__(self, dump_bin, artifact, tdir, extended):
        self.extended = extended
        self.dump_bin = dump_bin
        self.artifact = artifact
        ref.check_assets(tdir)  # sha256 gate on the pinned tokenizer files
        self.tok = ref.build_tokenizer(tdir)
        self.results = []

    def run_dump(self, mode, lines):
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as f:
            f.write("\n".join(lines) + "\n")
            path = f.name
        try:
            r = subprocess.run([self.dump_bin, mode, self.artifact, path],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("dump tool failed (rc=%d): %s"
                                   % (r.returncode, r.stderr[:400]))
            return r.stdout.splitlines()
        finally:
            os.unlink(path)

    def check(self, name, samples, bad, seed, details=()):
        self.results.append((name, samples, bad, seed))
        mark = "OK " if bad == 0 else "FAIL"
        print("[%s] %-14s samples=%-9d mismatches=%-5d seed=%s %s"
              % (mark, name, samples, bad, seed,
                 "first=" + str(details[0]) if details and bad else ""))
        return bad == 0

    # ---- stage helpers ----------------------------------------------------
    def nfc_native(self, cps_batches):
        """cps_batches: list of cp tuples -> list of cps lists (native N)."""
        lines = [",".join("%x" % c for c in b) for b in cps_batches]
        out = self.run_dump("N", lines)
        res = []
        for l in out:
            res.append([int(x, 16) for x in l.split(",") if x] if l else [])
        return res

    def nfc_engine(self, texts):
        return [[ord(c) for c in self.tok.normalizer.normalize_str(t)]
                for t in texts]

    def enc_native(self, texts):
        lines = [t.encode("utf-8").hex() for t in texts]
        out = self.run_dump("E", lines)
        return [[int(x) for x in l.split(",") if x] for l in out]

    def enc_engine(self, texts):
        return [list(self.tok.encode(t).ids) for t in texts]

    def pre_native(self, texts):
        lines = [t.encode("utf-8").hex() for t in texts]
        out = self.run_dump("P", lines)
        # the dump tool emits hex chunks; unhex to raw text (the engine side
        # yields raw text via the ByteLevel inverse map).
        return [[bytes.fromhex(h).decode("utf-8") for h in l.split("\x1f")]
                for l in out]

    def pre_engine(self, texts):
        return [ [bytelevel_to_raw(s) for s, _ in
                  self.tok.pre_tokenizer.pre_tokenize_str(
                      self.tok.normalizer.normalize_str(t))]
                 for t in texts ]

    def dec_native(self, id_seqs, skip):
        lines = ["%d %s" % (skip, ",".join(map(str, s))) for s in id_seqs]
        out = self.run_dump("X", lines)
        # strict: the native decoder is lossy-UTF-8 and must always emit
        # VALID UTF-8 (U+FFFD on bad bytes); a decode error is a bug.
        return [bytes.fromhex(l).decode("utf-8") for l in out]

    def dec_engine(self, id_seqs, skip):
        return [self.tok.decode(list(s), skip_special_tokens=bool(skip))
                for s in id_seqs]

    # ---- passes -----------------------------------------------------------
    def p_regressions(self):
        d1 = ["\u0041\u1715\u0337", "\u0041\u09fe\u0337",
              "\U00011935\U00011930", "\U00011935\U00011930x",
              "\U00011938", "\u0041\u07fd\u0337",
              "\u0041\u10f46\u0337", "\u0041\u1abf\u0337",
              "\u0041\u11839\u0337", "\u0041\u10d24\u0337"]
        d2 = ["a\u00a0\u00a0b", "\u00a0\u00a0word", " \u00a0\u00df",
              "x \u00a0 y", "word\u00a0\u00a0", "\u00a0"]
        bad = 0
        got = self.nfc_native([[ord(c) for c in t] for t in d1])
        want = self.nfc_engine(d1)
        for g, w, t in zip(got, want, d1):
            if g != w:
                bad += 1
                print("  N regression mismatch: %r" % t)
        g = self.enc_native(d1 + d2)
        w = self.enc_engine(d1 + d2)
        for a, b, t in zip(g, w, d1 + d2):
            if a != b:
                bad += 1
                print("  E regression mismatch: %r" % t)
        return self.check("regressions", len(d1) * 2 + len(d2), bad, "-")

    def p_single_cp(self):
        stride = SINGLE_EXT_STRIDE if self.extended else SINGLE_QUICK_STRIDE
        cps = [c for c in range(0x110000)
               if not (0xD800 <= c <= 0xDFFF) and c % stride == 0]
        n = len(cps)
        got = self.nfc_native([[c] for c in cps])
        want = self.nfc_engine([chr(c) for c in cps])
        bad = sum(1 for g, w in zip(got, want) if g != w)
        ok1 = self.check("nfc_single", n, bad, "exhaustive/stride%d" % stride)
        texts = ["a" + chr(c) for c in cps]
        g = self.enc_native(texts)
        w = self.enc_engine(texts)
        bad = sum(1 for a, b in zip(g, w) if a != b)
        ok2 = self.check("enc_single", n, bad, "exhaustive/stride%d" % stride)
        return ok1 and ok2

    def p_class_probe(self):
        stride = SINGLE_EXT_STRIDE if self.extended else SINGLE_QUICK_STRIDE
        cps = [c for c in range(0x110000)
               if not (0xD800 <= c <= 0xDFFF) and c % stride == 0]
        bad = 0
        n = 0  # ACTUAL number of comparisons (probe 3 skips CR/LF)
        BATCH = 20000
        for i in range(0, len(cps), BATCH):
            chunk = cps[i:i + BATCH]
            t1 = ["a" + chr(c) for c in chunk]
            t2 = ["a" + chr(c) + "b" for c in chunk]
            t3 = [chr(c) + chr(c) + "x" for c in chunk if c not in (0x0D, 0x0A)]
            for texts in (t1, t2, t3):
                g = self.pre_native(texts)
                w = self.pre_engine(texts)
                n += len(texts)
                bad += sum(1 for a, b in zip(g, w) if a != b)
        return self.check("class_probe", n, bad,
                          "exhaustive/stride%d" % stride)

    def p_ws_battery(self):
        texts = []
        for w in WS_CPS:
            wc = chr(w)
            texts += [wc, "a" + wc, "a" + wc + "b", "a" + wc + wc + "b",
                      wc + "b", wc + wc + "x", "a" + wc + "-b",
                      "ab" + wc + "cd"]
        g = self.pre_native(texts)
        w = self.pre_engine(texts)
        bad = sum(1 for a, b in zip(g, w) if a != b)
        ge = self.enc_native(texts)
        we = self.enc_engine(texts)
        bad += sum(1 for a, b in zip(ge, we) if a != b)
        return self.check("ws_battery", len(texts) * 2, bad, "-")

    def p_decomp_cp(self):
        decomp = uref.decomp_table("9.0.0")
        lb, vb, tb, sb = 0x1100, 0x1161, 0x11A8, 0xAC00
        for l in range(19):
            for v in range(21):
                for t in range(28):
                    s = sb + (l * 21 + v) * 28 + t
                    decomp[s] = [lb + l, vb + v] + ([tb + t - 1] if t else [])
        cps = sorted(decomp)
        got = self.nfc_native([[c] for c in cps])
        want = self.nfc_engine([chr(c) for c in cps])
        bad = sum(1 for g, w in zip(got, want) if g != w)
        ok1 = self.check("decomp_cp", len(cps), bad, "exhaustive")
        g = self.enc_native([chr(c) for c in cps])
        w = self.enc_engine([chr(c) for c in cps])
        bad = sum(1 for a, b in zip(g, w) if a != b)
        ok2 = self.check("decomp_cp_enc", len(cps), bad, "exhaustive")
        return ok1 and ok2

    def p_comp_pair(self):
        # composition pairs: derive the same candidate set the converter
        # derives (U9 full-decomposition splits with single-cp sides), then
        # compare NFC + encode of x+y against the engine.
        decomp = uref.decomp_table("9.0.0")
        lb, vb, tb, sb = 0x1100, 0x1161, 0x11A8, 0xAC00
        for l in range(19):
            for v in range(21):
                for t in range(28):
                    s = sb + (l * 21 + v) * 28 + t
                    decomp[s] = [lb + l, vb + v] + ([tb + t - 1] if t else [])
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
        s2c = {}
        for c in fullD:
            if len(fullD[c]) > 1:
                s2c.setdefault(fullD[c], c)

        def lookup(stream):
            stream = tuple(stream)
            if len(stream) == 1:
                z = stream[0]
                return z if z not in decomp else None
            return s2c.get(stream)
        pairs = []
        for c in list(decomp):
            fs = fullD[c]
            for i in range(1, len(fs)):
                x = lookup(fs[:i])
                y = lookup(fs[i:])
                if x is not None and y is not None:
                    if (self.tok.normalizer.normalize_str(chr(x) + chr(y))
                            == chr(c)):
                        pairs.append((x, y))
        # keep it tractable: NFC + encode of every pair (extended) or a
        # fixed 4000-sample stride (quick).
        if not self.extended and len(pairs) > 4000:
            step = -(-len(pairs) // 4000)
            pairs = pairs[::step]
        texts = [chr(x) + chr(y) for x, y in pairs]
        got = self.nfc_native([[x, y] for x, y in pairs])
        want = self.nfc_engine(texts)
        bad = sum(1 for g, w in zip(got, want) if g != w)
        ok1 = self.check("comp_pair_nfc", len(pairs), bad, "exhaustive")
        g = self.enc_native(texts)
        w = self.enc_engine(texts)
        bad = sum(1 for a, b in zip(g, w) if a != b)
        ok2 = self.check("comp_pair_enc", len(pairs), bad, "exhaustive")
        return ok1 and ok2

    def p_nfc_fuzz(self):
        n = NFC_FUZZ_EXT if self.extended else NFC_FUZZ_QUICK
        rng = random.Random(SEED_NFC_FUZZ)
        marks = (list(range(0x300, 0x370)) + [0x93C, 0x94D, 0x5B0, 0x64B,
                 0x670, 0x1D165, 0x1D16D, 0x1712, 0x1DC0, 0x102D] + DIFF98 +
                 [DIVES_AKURU[0], DIVES_AKURU[1], DIVES_AKURU[2]])
        starters = [0x41, 0x65, 0x391, 0x430, 0x41F, 0x5D0, 0x627, 0x915,
                    0xE01, 0x4E2D, 0x1100, 0x1161, 0x11A8, 0xAC00, 0x10A,
                    0x1F389, 0x1FEE, 0x1FBE, 0x4, 0x3A9]
        seqs = []
        for _ in range(n):
            k = rng.randint(1, 12)
            cps = [rng.choice(marks) if rng.random() < 0.5
                   else rng.choice(starters) for _ in range(k)]
            seqs.append(cps)
        got = self.nfc_native(seqs)
        want = self.nfc_engine(["".join(chr(c) for c in s) for s in seqs])
        bad = sum(1 for g, w in zip(got, want) if g != w)
        return self.check("nfc_fuzz", n, bad, SEED_NFC_FUZZ)

    def p_adjacency(self):
        n = ADJ_EXT if self.extended else ADJ_QUICK
        rng = random.Random(SEED_ADJ)
        anchor = DIFF98 + [DIVES_AKURU[0], DIVES_AKURU[1], 0x337, 0x301,
                           0x300, 0x93C, 0x315, 0x308, 0x41]
        seqs = []
        # structured: diff-cp against low-ccc and high-ccc neighbors.
        for a in DIFF98[:60 if self.extended else 12]:
            for nb in (0x337, 0x93C, 0x301, 0x300, 0x315, 0x41, 0x627):
                seqs.append([0x41, a, nb])
                seqs.append([0x41, nb, a])
        while len(seqs) < n:
            k = rng.randint(2, 4)
            seqs.append([rng.choice(anchor) for _ in range(k)])
        got = self.nfc_native(seqs[:n])
        want = self.nfc_engine(["".join(chr(c) for c in s)
                               for s in seqs[:n]])
        bad = sum(1 for g, w in zip(got, want) if g != w)
        return self.check("adjacency", n, bad, SEED_ADJ)

    def p_enc_fuzz(self):
        n = ENC_FUZZ_EXT if self.extended else ENC_FUZZ_QUICK
        rng = random.Random(SEED_ENC_FUZZ)
        alpha = (list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)) +
                 list(range(0x30, 0x3A)) + [0x4E2D, 0x6587, 0x3042, 0x4EAC,
                 0x3B1, 0x430, 0xE01, 0x627, 0x915, 0x5D0, 0x1100, 0xAC00,
                 0x1F389, 0xA0, 0x2000, 0x3000, 0x85, 0x9, 0x28, 0x29,
                 0x2C, 0x2E, 0x3A, 0x3F, 0x2019, 0x201C, 0x201D, 0x2013])
        texts = []
        for _ in range(n):
            k = rng.randint(1, 40)
            texts.append("".join(chr(rng.choice(alpha)) for _ in range(k)))
        g = self.enc_native(texts)
        w = self.enc_engine(texts)
        bad = sum(1 for a, b in zip(g, w) if a != b)
        return self.check("enc_fuzz", n, bad, SEED_ENC_FUZZ)

    def p_pretok_fuzz(self):
        n = PRETOK_FUZZ_EXT if self.extended else PRETOK_FUZZ_QUICK
        rng = random.Random(SEED_PRETOK_FUZZ)
        alpha = [0x41, 0x61, 0x30, 0x20, 0x9, 0xA, 0xD, 0x28, 0x2E, 0x2D,
                 0x27, 0xA0, 0x85, 0x1680, 0x2000, 0x200A, 0x2028, 0x2029,
                 0x202F, 0x205F, 0x3000, 0x4E2D, 0x3042, 0x3B1, 0xE01, 0x627,
                 0x5D0, 0x4E, 0xA9, 0xAE, 0x2122, 0x1F389, 0x301, 0x308,
                 0x93C, 0x1715, 0x11935, 0x11930]
        texts = []
        for _ in range(n):
            k = rng.randint(1, 30)
            texts.append("".join(chr(rng.choice(alpha)) for _ in range(k)))
        g = self.pre_native(texts)
        w = self.pre_engine(texts)
        bad = sum(1 for a, b in zip(g, w) if a != b)
        return self.check("pretok_fuzz", n, bad, SEED_PRETOK_FUZZ)

    def p_dec_fuzz(self):
        n = DEC_FUZZ_EXT if self.extended else DEC_FUZZ_QUICK
        rng = random.Random(SEED_DEC_FUZZ)
        # id space: base byte tokens (3..258) + model-vocab padding ids
        # (248077..248319) + random base-vocab ids; the same seeded
        # distribution as the corpus X-line fuzz (tools/gen_tokenizer_refs.py).
        seqs = []
        for _ in range(n):
            k = rng.randint(1, 12)
            s = []
            for _j in range(k):
                r = rng.random()
                if r < 0.6:
                    s.append(rng.randrange(3, 259))
                elif r < 0.8:
                    s.append(rng.randrange(248077, 248320))
                else:
                    s.append(rng.randrange(0, 248044))
            seqs.append(s)
        bad = 0
        for skip in (0, 1):
            g = self.dec_native(seqs, skip)
            w = self.dec_engine(seqs, skip)
            bad += sum(1 for a, b in zip(g, w) if a != b)
        return self.check("dec_fuzz", n * 2, bad, SEED_DEC_FUZZ)

    def run_all(self):
        ok = True
        ok &= self.p_regressions()
        ok &= self.p_ws_battery()
        ok &= self.p_single_cp()
        ok &= self.p_class_probe()
        ok &= self.p_decomp_cp()
        ok &= self.p_comp_pair()
        ok &= self.p_nfc_fuzz()
        ok &= self.p_adjacency()
        ok &= self.p_enc_fuzz()
        ok &= self.p_pretok_fuzz()
        ok &= self.p_dec_fuzz()
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "extended"],
                    default="quick")
    ap.add_argument("--dump-binary", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--tokenizer-dir",
                    default="/root/models/Qwen3.5-0.8B-Base/tokenizer")
    a = ap.parse_args()
    try:
        v = Validator(a.dump_binary, a.artifact, a.tokenizer_dir,
                      a.mode == "extended")
    except FileNotFoundError as e:
        print("[SKIP] %s" % e, file=sys.stderr)
        return 77
    ok = v.run_all()
    total_bad = sum(r[2] for r in v.results)
    print("head=%s mode=%s mismatches_total=%d"
          % (git_head(), a.mode, total_bad))
    print("RESULT: %s" % ("PASS (0 mismatch on every pass)" if ok
                          else "FAIL — report as BLOCKER"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
