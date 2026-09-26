# Pinned Unicode data sources for the Qwen3.5 native tokenizer tables.
#
# The pinned oracle (Qwen/Qwen3.5-0.8B-Base tokenizer, tokenizers 0.22.2)
# uses TWO different Unicode data versions in its two normalization
# subsystems (both proven empirically against the pinned engine, see
# docs/qwen35_architecture.md §20):
#
#   (A) NFC normalizer  = Rust crate `unicode-normalization-alignments`
#       0.1.12, whose tables are generated from Unicode 9.0.0
#       (crate constant UNICODE_VERSION = (9, 0, 0); its 814-entry ccc
#       table is byte-identical to the U9 UCD ccc>0 set).
#       -> tools/ucd/UnicodeData-9.0.0.txt
#
#   (B) pre-tokenizer regex character classes (\pL / \pN / \pM / \s) are
#       supplied by the `regex` crate dependency (tokenizers Cargo.toml:
#       regex = "1.10", resolving to a U16-era build).  A per-cp chunking
#       sweep over all 1,112,064 non-surrogate code points against the
#       pinned engine's pre_tokenizer shows:
#         - L ∪ M == U16.0.0 L ∪ M exactly (143,529 cps)
#         - N     == U16.0.0 N exactly (1,911 cps)
#         - \s    == UAX #44 White_Space exactly (25 cps)
#       -> tools/ucd/UnicodeData-16.0.0.txt
#
# Both files are committed to the repo and sha256-gated below, so the
# table conversion is deterministic and offline.  Provenance:
#   https://www.unicode.org/Public/9.0.0/ucd/UnicodeData.txt
#   https://www.unicode.org/Public/16.0.0/ucd/UnicodeData.txt

import hashlib
import os

_U9_SHA256 = "68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f"
_U16_SHA256 = "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f"

# cp -> (general_category, ccc, canonical_decomp | None)
_UCD = {}


def _ucd_path(version):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "ucd")
    if version == "9.0.0":
        return os.path.normpath(os.path.join(d, "UnicodeData-9.0.0.txt")), \
            _U9_SHA256
    if version == "16.0.0":
        return os.path.normpath(os.path.join(d, "UnicodeData-16.0.0.txt")), \
            _U16_SHA256
    raise ValueError("unknown Unicode version: " + version)


def _parse(path, sha):
    with open(path, "rb") as f:
        raw = f.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        raise ValueError("pinned UCD file %s: sha256 %s != expected %s "
                         "(do not edit the pinned Unicode data)"
                         % (os.path.basename(path), got, sha))
    tbl = {}
    lines = [l.strip() for l in raw.decode("utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        f = lines[i].split(";")
        if len(f) < 6:
            i += 1
            continue
        cp = int(f[0], 16)
        cat = f[2]
        ccc = int(f[3]) if f[3] not in ("<control>", "<none>") else 0
        decomp = None
        d = f[5] if f[5] else None
        if d and "<" not in d:
            decomp = [int(p, 16) for p in d.split()]
        if f[1].endswith(", First>"):
            # UCD range notation: the whole range [cp, end] carries the
            # identical properties.
            end = int(lines[i + 1].split(";")[0], 16)
            for c in range(cp, end + 1):
                tbl[c] = (cat, ccc, None if decomp is None else list(decomp))
            i += 2
        else:
            tbl[cp] = (cat, ccc, decomp)
            i += 1
    return tbl


def load(version):
    """Load the pinned UCD for '9.0.0' or '16.0.0' (sha256-gated)."""
    if version not in _UCD:
        path, sha = _ucd_path(version)
        _UCD[version] = _parse(path, sha)
    return _UCD[version]


def ccc_table(version="9.0.0"):
    """cp -> ccc for all cps with ccc > 0 (U9: 814 entries)."""
    return {cp: v[1] for cp, v in load(version).items() if v[1] > 0}


def decomp_table(version="9.0.0"):
    """cp -> canonical decomposition list (U9: 2,060 entries)."""
    return {cp: v[2] for cp, v in load(version).items() if v[2] is not None}


def category(version, cp):
    v = load(version).get(cp)
    return v[0] if v else "Cn"


def category_ranges(version, prefix):
    """Inclusive (lo, hi) ranges of cps whose UCD category starts with
    prefix (used for the pre-tokenizer L/N/M tables, U16)."""
    out = []
    start = None
    for cp in range(0x110000):
        hit = category(version, cp).startswith(prefix)
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out
