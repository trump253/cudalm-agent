"""Phase D micro-stack ``--tokens`` sequence validation (stdlib-only).

The hybrid micro-stack golden generator (``tools/generate_qwen35_golden.py
--microstack-prefix --tokens``) threads each layer's OWN persistent state
across a token sequence, so the sequence must be a single 0-anchored, strictly
consecutive run of 0-based positions:

    0 | 0,1 | 0,1,2 | 0,1,2,3 | ...

A gap, a duplicate, an out-of-order entry, a non-0 start, or a non-integer
field is a caller error (it would skip or duplicate a decode step and silently
corrupt the state chain), so it is rejected up front rather than producing a
wrong golden. This module is stdlib-only (no torch) so the argument-validation
selftest can run under any python3.
"""
from __future__ import annotations

__all__ = ["parse_token_sequence", "selftest"]


def parse_token_sequence(s: str) -> list:
    """Parse and validate a ``--tokens`` sequence.

    Returns the list of 0-based positions (== ``list(range(len)))``). Raises
    ``ValueError`` on any of: empty/whitespace input, an empty comma field, a
    non-integer (or negative) field, or a sequence that is not strictly
    consecutive starting at 0.
    """
    if s is None or s.strip() == "":
        raise ValueError("--tokens must be non-empty (e.g. '0' or '0,1,2')")
    tokens = []
    for field in (f.strip() for f in s.split(",")):
        if field == "":
            raise ValueError(
                "--tokens has an empty field (e.g. '0,,1' or a trailing ','): "
                f"{s!r}")
        if not field.isdigit():
            raise ValueError(
                f"--tokens field {field!r} is not a non-negative integer")
        tokens.append(int(field))
    expected = 0
    for t in tokens:
        if t != expected:
            raise ValueError(
                "--tokens must be the strictly consecutive 0-anchored sequence "
                f"0,1,2,...; got {t} where {expected} was expected (input: {s!r})")
        expected += 1
    return tokens


def selftest() -> int:
    """Argument-validation battery for ``parse_token_sequence`` (no torch)."""
    valid = {
        "0": [0],
        "0,1": [0, 1],
        "0,1,2": [0, 1, 2],
        "0,1,2,3": [0, 1, 2, 3],
        " 0 , 1 , 2 ": [0, 1, 2],  # surrounding/inner whitespace is stripped
    }
    for s, want in valid.items():
        got = parse_token_sequence(s)
        assert got == want, f"valid {s!r}: got {got}, want {want}"
        assert got == list(range(len(want))), f"{s!r} not 0-anchored run"

    invalid = [
        "",            # empty
        "   ",         # whitespace only
        "2",           # non-0 start
        "1,2",         # non-0 start
        "0,2",         # gap
        "0,1,3",       # gap
        "1,0",         # out of order
        "2,1,0",       # reverse
        "0,0,1",       # duplicate
        "0,1,1",       # duplicate
        "abc",         # non-integer
        "0,abc",       # non-integer field
        "-1",          # negative
        "0,-1",        # negative field
        "0,1.5",       # non-integer
        "0,,1",        # empty middle field
        "0,1,",        # trailing comma -> empty field
        ",0,1",        # leading comma -> empty field
    ]
    for s in invalid:
        try:
            parse_token_sequence(s)
        except ValueError:
            continue
        raise AssertionError(f"invalid {s!r} was accepted")

    print(f"[PASS] token_seq selftest "
          f"({len(valid)} valid, {len(invalid)} invalid)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
