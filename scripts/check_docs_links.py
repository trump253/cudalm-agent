#!/usr/bin/env python3
"""CUDALM docs link guard (stdlib-only, no third-party dependencies).

Checks that repository-relative FILE links in the top-level READMEs and
docs/*.md resolve to files that exist in the repository.

Scope (deliberately narrow — this is a cheap CI guard, not a full
linkchecker):
  * files scanned: README.md, README_EN.md, docs/*.md
  * links checked: markdown links [text](target) and images ![alt](target)
  * skipped: http://, https://, mailto:, absolute paths (/...),
    anchor-only (#...), and protocol-relative (//...) targets
  * fragment (#section) and query parts of a relative target are stripped;
    the file part must exist
  * external URLs are NOT fetched (no network access)

Exit status: 0 = all links resolve, 1 = at least one broken link,
2 = usage / unexpected error.

Usage:
  python3 scripts/check_docs_links.py [REPO_ROOT]
"""

import os
import re
import sys

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

SCANNED = [
    "README.md",
    "README_EN.md",
]


def doc_files(root):
    files = list(SCANNED)
    docs_dir = os.path.join(root, "docs")
    if os.path.isdir(docs_dir):
        for name in sorted(os.listdir(docs_dir)):
            if name.endswith(".md"):
                files.append(os.path.join("docs", name))
    return files


def is_external(target):
    if target.startswith("#"):
        return True
    low = target.lower()
    if low.startswith(("http://", "https://", "mailto:", "//", "data:")):
        return True
    if ":" in target.split("/", 1)[0] and not target.startswith("/"):
        # any other URI scheme (git:, ssh:, ...) — not a repo file link
        return True
    return False


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    broken = []
    checked = 0
    scanned = 0
    for rel in doc_files(root):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        scanned += 1
        text = open(path, encoding="utf-8").read()
        for m in LINK_RE.finditer(text):
            target = m.group(1).strip()
            if is_external(target):
                continue
            file_part = target.split("#", 1)[0].split("?", 1)[0]
            if not file_part:
                continue
            if file_part.startswith("/"):
                continue  # absolute path, not repository-relative
            checked += 1
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), file_part))
            if not os.path.exists(resolved):
                broken.append((rel, target))
    print("docs link check: %d files scanned, %d relative file links checked"
          % (scanned, checked))
    if broken:
        for rel, target in broken:
            print("BROKEN  %s -> %s" % (rel, target))
        print("docs link check: FAIL (%d broken links)" % len(broken))
        return 1
    print("docs link check: OK (no broken repository-relative links)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
