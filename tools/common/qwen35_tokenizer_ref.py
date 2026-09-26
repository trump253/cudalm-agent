#!/usr/bin/env python3
"""Pinned Qwen3.5-0.8B-Base tokenizer reference (offline oracle + tooling).

This module is the SINGLE PYTHON SOURCE OF TRUTH for the Qwen3.5 tokenizer
contract used by the offline converter, the reference-corpus generator, the
generation-gap diagnostic and the cross-language tests.  It is a TOOL/oracle
only — the C++ runtime never imports Python (scripts/check_no_torch.sh).

Pinned sources (see docs/qwen35_architecture.md §18):
  * repo      : Qwen/Qwen3.5-0.8B-Base
  * revision  : dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
  * tokenizer assets (tokenizer/ subdir of the local checkpoint):
        tokenizer.json          (Rust `tokenizers` BPE model, version 1.0)
        tokenizer_config.json   (special-token metadata, chat template)
  * backend   : the `tokenizers` library, PINNED to version 0.22.2
        (EXPECTED_TOKENIZERS_VERSION).  build_tokenizer() enforces the
        pin via check_oracle_version(): a different importable version is
        a provenance violation that fails loud (never skipped, never
        silently substituted — e.g. by a system tokenizers 0.15.1).
        The pinned transformers (4.57.0.dev0 per raw/config.json) uses
        Qwen2TokenizerFast, a pure PreTrainedTokenizerFast subclass —
        no custom encode/decode.

Effective added-token set (transformers `PreTrainedTokenizerFast.__init__`
sync logic, verified against the transformers 4.40.2 source): the 22 added
tokens in tokenizer.json PLUS the 11 config-only entries from
`added_tokens_decoder` (ids 248066..248076, auto-assigned in ascending id
order) = 33 added tokens, ids 248044..248076 (contiguous).  The 22 shared
entries have identical content and special flags in both files; config wins
on any conflict.  Model vocab (248320) pads 248077..248319 with no tokenizer
entries.
"""

import hashlib
import json
import os

# --- pinned contract constants (verified against the pinned assets) --------
EXPECTED_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
BASE_VOCAB_SIZE = 248044          # ids 0..248043 (BPE base vocab)
NUM_ADDED = 33                    # ids 248044..248076
FIRST_ADDED_ID = 248044           # == EOS token id
MODEL_VOCAB_SIZE = 248320         # model LM-head width (padded)
EOS_TOKEN_ID = 248044             # config text_config.eos_token_id, single EOS
# Pinned oracle ENGINE version.  The authoritative tokenizer oracle MUST be
# exactly this `tokenizers` (Rust engine) version: the D1/D2 data-version
# findings (U9 normalizer / U16 regex classes, docs §20) were proven against
# 0.22.2, and the whole correctness evidence is bound to it.
EXPECTED_TOKENIZERS_VERSION = "0.22.2"
# sha256 of the four pinned tokenizer files (downloaded at the pinned
# revision; see the Phase B download record in docs/provenance.md).
EXPECTED_SHA256 = {
    "tokenizer.json":
        "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json":
        "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
    "vocab.json":
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt":
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
}


class OracleVersionError(RuntimeError):
    """The authoritative `tokenizers` oracle is not the pinned version.

    This is a PROVENANCE VIOLATION: it is NOT a "cannot run here" condition
    (missing tokenizer assets are, and callers self-skip 77 for those).
    A wrong oracle version must fail loud — never be skipped, and never be
    silently substituted by another installed tokenizers (e.g. system
    0.15.1).
    """


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_oracle_version(expected=EXPECTED_TOKENIZERS_VERSION):
    """Single version gate for the authoritative tokenizer oracle.

    Every oracle path (converter, corpus generator, differential validator,
    text golden, gap diagnostics) builds its oracle through
    build_tokenizer(), which calls this gate before ANY engine use.  The
    importable `tokenizers` library must be EXACTLY `expected`
    (default: the pinned 0.22.2); anything else raises OracleVersionError.
    """
    import tokenizers
    got = getattr(tokenizers, "__version__", None)
    if got != expected:
        raise OracleVersionError(
            "oracle provenance violation: importable tokenizers version %r "
            "!= pinned %s. The authoritative oracle must be exactly "
            "tokenizers==%s; a different version is a provenance violation "
            "(fail loud), never a silent fallback or a skip."
            % (got, expected, expected))
    return got


def tokenizer_dir(checkpoint_dir):
    return os.path.join(checkpoint_dir, "tokenizer")


def check_assets(tdir):
    """Fail loud if the pinned tokenizer assets (inside tdir) are missing or
    do not match the pinned revision sha256."""
    for name, sha in EXPECTED_SHA256.items():
        p = os.path.join(tdir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError("missing pinned tokenizer asset: " + p)
        got = _sha256(p)
        if got != sha:
            raise ValueError(
                "tokenizer asset %s does not match the pinned revision sha256 "
                "(got %s, want %s)" % (name, got, sha))
    return tdir


def effective_added_tokens(tdir):
    """The 33 effective added tokens (id, content, special) — the union of
    tokenizer.json `added_tokens` and tokenizer_config.json
    `added_tokens_decoder` (config wins), exactly as transformers'
    PreTrainedTokenizerFast materializes them on the Rust object.
    """
    tk = json.load(open(os.path.join(tdir, "tokenizer.json")))
    tc = json.load(open(os.path.join(tdir, "tokenizer_config.json")))
    by_id = {}
    for a in tk["added_tokens"]:
        by_id[a["id"]] = {
            "id": a["id"],
            "content": a["content"],
            "special": bool(a.get("special", False)),
        }
    for k, e in tc["added_tokens_decoder"].items():
        i = int(k)
        by_id[i] = {
            "id": i,
            "content": e["content"],
            "special": bool(e.get("special", False)),
        }
    ids = sorted(by_id)
    if ids != list(range(FIRST_ADDED_ID, FIRST_ADDED_ID + NUM_ADDED)):
        raise ValueError(
            "added-token ids are not the pinned contiguous range %d..%d: %s"
            % (FIRST_ADDED_ID, FIRST_ADDED_ID + NUM_ADDED - 1, ids))
    return [by_id[i] for i in ids]


def build_tokenizer(tdir):
    """Build the effective `tokenizers.Tokenizer` object (tokenizer.json
    pipeline + the 33 effective added tokens).  This is the oracle.

    The engine VERSION gate runs first: if the importable tokenizers is not
    exactly the pinned 0.22.2, OracleVersionError is raised before any
    engine call (see check_oracle_version).
    """
    import tokenizers  # local import: keep the module importable without it
    check_oracle_version()

    tk = json.load(open(os.path.join(tdir, "tokenizer.json")))
    tk["added_tokens"] = [
        {"id": a["id"], "content": a["content"], "single_word": False,
         "lstrip": False, "rstrip": False, "normalized": False,
         "special": a["special"]}
        for a in effective_added_tokens(tdir)
    ]
    # Deterministic JSON (the tokenizers library does not require a file;
    # from_string exists in the modern API — fall back to a temp file).
    blob = json.dumps(tk, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    try:
        return tokenizers.Tokenizer.from_string(blob)
    except AttributeError:  # tokenizers < 0.14 has no from_string
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            f.write(blob)
            tmp = f.name
        try:
            return tokenizers.Tokenizer.from_file(tmp)
        finally:
            os.unlink(tmp)


class PinnedTokenizer(object):
    """Small facade over the oracle with the pinned model constants."""

    def __init__(self, checkpoint_dir):
        tdir = check_assets(tokenizer_dir(checkpoint_dir))
        self._tdir = tdir
        self._tok = build_tokenizer(tdir)
        self.base_vocab_size = BASE_VOCAB_SIZE
        self.model_vocab_size = MODEL_VOCAB_SIZE
        self.eos_token_id = EOS_TOKEN_ID
        self.added = effective_added_tokens(tdir)

    def _check_id(self, i):
        if not isinstance(i, int) or isinstance(i, bool) or i < 0 or \
                i >= self.model_vocab_size:
            raise ValueError("token id out of range: %r" % (i,))

    def encode(self, text, add_special_tokens=False):
        """Exact HF encode.  For this model add_special_tokens is a no-op
        (no BOS, Qwen2 adds nothing on encode) — the flag is accepted for
        API symmetry and BOTH values must produce the same ids (checked by
        the reference corpus generator).
        """
        if not isinstance(text, str):
            raise TypeError("text must be str (UTF-8)")
        enc = self._tok.encode(text)
        return list(enc.ids)

    def decode(self, ids, skip_special_tokens=False):
        """Exact HF decode.  Ids in the model-padding range
        (248077..248319) decode to "" (silently dropped by the Rust engine,
        verified against the oracle); added ids 248044..248076 decode to
        their strings (skipped when special + skip_special_tokens)."""
        for i in ids:
            self._check_id(i)
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def is_special(self, i):
        return any(a["id"] == i and a["special"] for a in self.added)

    def added_string(self, i):
        for a in self.added:
            if a["id"] == i:
                return a["content"]
        return None
