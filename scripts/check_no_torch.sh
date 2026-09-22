#!/usr/bin/env bash
# CUDALM guard: the runtime (include/ + src/) must be PyTorch / pybind /
# Python-free. Any forbidden symbol is a hard failure.
#
# Forbidden:
#   - #include of torch / ATen / c10 / pybind headers
#   - at::Tensor / at:: symbols
#   - TORCH_CHECK / TORCH_ macros
#   - pybind11
#   - py:: / pybind symbols
#
# tools/ (Python) is intentionally NOT scanned: PyTorch is allowed there.

set -euo pipefail
root="${1:-.}"
inc="${root}/include"
src="${root}/src"

status=0
patterns=(
  '#include[[:space:]]*<torch'
  '#include[[:space:]]*"torch'
  '#include[[:space:]]*<ATen'
  '#include[[:space:]]*"ATen'
  '#include[[:space:]]*<c10'
  '#include[[:space:]]*"c10'
  '#include[[:space:]]*<pybind11'
  'at::Tensor'
  'TORCH_CHECK'
  'TORCH_WARN'
  'pybind11'
  '\bpy::'
)

# Provenance/comments may legitimately *mention* these symbols (e.g. "replaces
# CUDALab's TORCH_CHECK layer"). We only flag matches on non-comment code
# lines, so the scan strips comments before matching.
for dir in "$inc" "$src"; do
  [ -d "$dir" ] || continue
  while IFS= read -r -d '' f; do
    # Drop full-line comments (leading // or * or /*) to avoid provenance FPs.
    code_lines=$(sed -E 's:/.*$::' "$f" | grep -vE '^[[:space:]]*(//|\*|/\*)' || true)
    for p in "${patterns[@]}"; do
      if printf '%s\n' "$code_lines" | grep -qE "$p"; then
        echo "[forbidden] ${f}: matches /${p}/"
        printf '%s\n' "$code_lines" | grep -nE "$p" | sed 's/^/    /'
        status=1
      fi
    done
  done < <(find "$dir" -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cc' -o -name '*.cu' -o -name '*.cuh' \) -print0)
done

if [ "$status" -ne 0 ]; then
  echo "forbidden_deps_check FAILED"
  exit 1
fi
echo "forbidden_deps_check OK (no torch/pybind/py symbols in include/ src/)"
