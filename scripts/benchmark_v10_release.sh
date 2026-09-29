#!/usr/bin/env bash
# CUDALM v1.0 Phase B — RELEASE BENCHMARK workflow (reproducible)
#
# Discipline (the exact-SHA benchmark protocol):
#   1. this script is committed FIRST (tooling commit);
#   2. run it from a CHECK-OUT-CLEAN tracked tree (FAIL LOUD otherwise —
#      no "official evidence" is generated from a dirty tree);
#   3. the benchmark executable is BUILT (CMake build / up-to-date
#      validation) at the recorded exact source SHA, and the HEAD SHA
#      is re-verified AFTER the build and AFTER the benchmark — the
#      executed binary is thereby the binary built from the recorded
#      source SHA (the binary sha256 is recorded in environment.txt);
#      a full clean rebuild is NOT required (the build dir may be
#      incremental);
#   4. it runs the canonical workload
#      (benchmarks/bench_qwen35_continuous_batching.cpp) with the
#      release protocol and writes the raw evidence into
#      benchmarks/v10/ (raw files are ALWAYS kept — the summary is a
#      convenience view, never the only evidence);
#   5. the caller then commits the evidence (docs/evidence-only commit)
#      — the large model file (build/data/*.cudalm) is NEVER committed.
#
# Release protocol: warmup-runs = 2, measured-runs = 10.
#   run 1: --mode both     -> benchmarks/v10/release_serial_batched.txt
#   run 2: --mode batched  -> benchmarks/v10/release_batched.txt
#         (the batched-only run is the CONTINUOUS-BATCHED SERVING
#          profile — the independent serving-path evidence)
#
# Usage: scripts/benchmark_v10_release.sh [build_dir]
#   build_dir defaults to ./build

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="${1:-${root}/build}"

model="${build_dir}/data/qwen35_08b_full.cudalm"
ckpt="${CUDALM_CKPT:-/root/models/Qwen3.5-0.8B-Base}"
python3="${CUDALM_PYTHON3:-/root/py311/venv/bin/python3}"
bin="${build_dir}/benchmarks/bench_qwen35_continuous_batching"
out_dir="${root}/benchmarks/v10"

# Deterministic device: always device 0 (visible device 0).
export CUDA_VISIBLE_DEVICES=0

die() { echo "FAIL: $*" >&2; exit 1; }

# ---- 1. clean tracked tree (fail loud — no official evidence) ------------
cd "${root}"
[ -d "${root}/.git" ] || die "not a git repository: ${root}"
if ! git diff --quiet || ! git diff --cached --quiet; then
  die "tracked tree is NOT clean — benchmark evidence must be bound to an exact SHA. Commit or stash the working tree first."
fi
sha="$(git rev-parse HEAD)"
branch="$(git rev-parse --abbrev-ref HEAD)"
echo "git sha:    ${sha}"
echo "git branch: ${branch}"
echo "git tree:   CLEAN (tracked tree)"

# ---- 2. build the benchmark target AT the recorded SHA ----------------------
# CMake build / up-to-date validation of the exact target (a full clean
# rebuild is not required — the build dir may be incremental). FAIL LOUD
# on build failure. The HEAD SHA is re-verified after the build: the
# executed binary must be the binary built from the recorded source SHA.
build_cmd="cmake --build ${build_dir} --target bench_qwen35_continuous_batching -j8"
echo "build: ${build_cmd}"
( cd "${build_dir}" && cmake --build . --target bench_qwen35_continuous_batching -j8 ) \
    || die "benchmark target build failed at ${sha}"
[ "$(git rev-parse HEAD)" = "${sha}" ] || die "git HEAD changed DURING the build"
[ -f "${bin}" ] || die "benchmark binary missing after build: ${bin}"
bin_sha256="$(sha256sum "${bin}" | awk '{print $1}')"

# ---- 2b. prerequisites ------------------------------------------------------
[ -f "${model}" ] || die "converted model missing: ${model} (run tools/convert_qwen35.py --full-model first)"
[ -f "${ckpt}/model.safetensors" ] || die "checkpoint missing: ${ckpt}/model.safetensors"
[ -x "${python3}" ] || die "python missing: ${python3}"
mkdir -p "${out_dir}"

# ---- 3. environment record --------------------------------------------------
gpu_name="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
gpu_mem="$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
driver="$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
cuda_version="$('/usr/local/cuda/bin/nvcc' --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\), V\([0-9.]*\).*/\1 (V\2)/p' | head -1 || echo unknown)"
compiler="$(gcc --version 2>/dev/null | head -1 || echo unknown)"
build_type="$(sed -n 's/^CMAKE_BUILD_TYPE:\(STRING\|UNINITIALIZED\)=//p' "${build_dir}/CMakeCache.txt" | head -1 || echo unknown)"
[ -n "${build_type}" ] || build_type="(default — CMAKE_BUILD_TYPE unset)"

env_file="${out_dir}/environment.txt"
{
  echo "# CUDALM v1.0 release benchmark — environment record"
  echo "date:                 $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "git_sha:              ${sha}"
  echo "git_branch:           ${branch}"
  echo "git_tree_clean:       yes (tracked tree clean at generation — enforced by the script)"
  echo "gpu (device 0):       ${gpu_name} (${gpu_mem})"
  echo "nvidia_driver:        ${driver}"
  echo "cuda_toolkit:         ${cuda_version}"
  echo "cuda_runtime:         (recorded inside each raw report: cuda_runtime / cuda_driver lines)"
  echo "host_compiler:        ${compiler}"
  echo "cmake_build_type:     ${build_type}"
  echo "build_command:        ${build_cmd}  (run at git_sha; HEAD re-verified unchanged after the build and after the benchmark)"
  echo "benchmark_binary:     ${bin}"
  echo "benchmark_binary_sha256: ${bin_sha256}  (computed AFTER the build at git_sha)"
  echo "model_file:           ${model}"
  echo "model_file_sha256:    $(sha256sum "${model}" | awk '{print $1}')"
  echo "checkpoint_dir:       ${ckpt}"
  echo "checkpoint_sha256:    $(sha256sum "${ckpt}/model.safetensors" | awk '{print $1}')  (model.safetensors)"
  echo "python:               ${python3} ($(${python3} --version 2>&1))"
  echo "protocol:             warmup-runs=2 measured-runs=10"
  echo "benchmark_commands:"
  echo "  1: ${bin} ${model} ${ckpt} ${python3} ${root} ${out_dir}/release_serial_batched.txt --no-convert --measured-runs 10 --warmup-runs 2 --mode both"
  echo "  2: ${bin} ${model} ${ckpt} ${python3} ${root} ${out_dir}/release_batched.txt --no-convert --measured-runs 10 --warmup-runs 2 --mode batched"
} > "${env_file}"
echo "wrote ${env_file}"

# ---- 4. run the benchmark (raw evidence) ------------------------------------
run_one() {
  local mode="$1" out="$2"
  echo "=== running --mode ${mode} (warmup=2 measured=10) -> ${out}"
  "${bin}" "${model}" "${ckpt}" "${python3}" "${root}" "${out}" \
      --no-convert --measured-runs 10 --warmup-runs 2 --mode "${mode}" \
      || die "benchmark run --mode ${mode} failed"
}
run_one both     "${out_dir}/release_serial_batched.txt"
run_one batched  "${out_dir}/release_batched.txt"

# ---- 5. summary.json (stdlib-only parsing of the raw reports) ---------------
"${python3}" - "${out_dir}" "${sha}" <<'PYEOF'
import json, re, sys

out_dir, sha = sys.argv[1], sys.argv[2]

METRICS = [
    ("wall_time_s_mean", "wall_time_s mean", float),
    ("wall_time_s_median", "wall_time_s median", float),
    ("wall_time_s_min", "wall_time_s min", float),
    ("wall_time_s_max", "wall_time_s max", float),
    ("logical_tokens", "logical_tokens", int),
    ("logical_tokens_per_s_mean", "logical_tokens_per_s (mean)", float),
    ("single_forward_calls", "single_forward_calls", int),
    ("batch_forward_calls", "batch_forward_calls", int),
    ("model_traversal_calls", "model_traversal_calls", int),
    # avg is a FLOAT in the raw report (e.g. 2.666667); max is an int.
    ("avg_decode_batch_size", "avg_decode_batch_size", float),
    ("max_decode_batch_size", "max_decode_batch_size", int),
]

def parse_report(path):
    """Parse ONE raw report. A `--mode both` report contains BOTH a mode A
    (serial) section and a mode B (batched) section with IDENTICAL metric
    labels — so each section is sliced at its 'mode A /' / 'mode B /'
    header and parsed separately (never by 'last occurrence')."""
    txt = open(path).read()
    end = txt.index("profile_totals") if "profile_totals" in txt else len(txt)
    has_a = "mode A" in txt[:end]
    has_b = "mode B" in txt[:end]
    if has_a and has_b:
        serial_txt = txt[txt.index("mode A"):txt.index("mode B")]
        batched_txt = txt[txt.index("mode B"):end]
    elif has_a:
        serial_txt, batched_txt = txt[txt.index("mode A"):end], None
    else:
        serial_txt, batched_txt = None, txt[txt.index("mode B"):end]

    def section(s):
        out = {}
        for key, label, conv in METRICS:
            m = re.search(r"^\s*" + re.escape(label) + r":\s*([-+0-9.eE]+)\s*$",
                          s, re.M)
            if not m:
                raise SystemExit(f"FAIL: metric '{label}' missing from section "
                                 f"(report {path!r})")
            out[key] = conv(m.group(1))
        return out

    res = {}
    if serial_txt is not None:
        res["serial"] = section(serial_txt)
    if batched_txt is not None:
        res["batched"] = section(batched_txt)
    return res

both = parse_report(f"{out_dir}/release_serial_batched.txt")
batched_only = parse_report(f"{out_dir}/release_batched.txt")
env = {}
for ln in open(f"{out_dir}/environment.txt"):
    if ":" in ln and not ln.startswith("#"):
        k, _, v = ln.partition(":")
        env[k.strip()] = v.strip()

s, b = both.get("serial"), both.get("batched")
summary = {
    "sha": sha,
    "hardware": {
        "gpu": env.get("gpu (device 0)"),
        "nvidia_driver": env.get("nvidia_driver"),
        "cuda_toolkit": env.get("cuda_toolkit"),
        "cmake_build_type": env.get("cmake_build_type"),
    },
    "workload": {
        "requests": 4,
        "prompt_lengths": [2, 5, 3, 4],
        "generated_lengths": [3, 6, 5, 3],
        "logical_token_forwards": 27,
        "arrivals": "A,B first; C after 1 step; D after 2 steps (dynamic)",
        "model": "Qwen3.5-0.8B-Base (raw text, no chat template)",
        "note": "canonical workload — see benchmarks/v10/README.md",
    },
    "warmup_runs": 2,
    "measured_runs": 10,
    "serial": s,
    "batched": b,
    "batched_only_serving_profile": batched_only.get("batched"),
}
if s and b:
    ds = b["wall_time_s_mean"] - s["wall_time_s_mean"]
    dt = b["logical_tokens_per_s_mean"] - s["logical_tokens_per_s_mean"]
    summary["batched_vs_serial_delta"] = {
        "wall_time_s_mean": ds,
        "logical_tokens_per_s_mean": dt,
        "interpretation": "measured on this hardware / checkpoint / canonical "
                          "workload only — not a general speedup claim",
    }

missing = [k for k in ("wall_time_s_mean", "logical_tokens_per_s_mean")
           for m in (s, b, batched_only.get("batched"))
           if m and m.get(k) is None]
if missing:
    raise SystemExit(f"FAIL: unparsed metrics in raw reports: {missing}")

with open(f"{out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")
print("wrote summary.json")
PYEOF

# ---- 6. self-consistency check ----------------------------------------------
# HEAD re-verified AFTER the benchmark: the exact source SHA was unchanged
# across the build AND the run, so the executed binary == the binary built
# from the recorded SHA.
[ "$(git rev-parse HEAD)" = "${sha}" ] || die "git HEAD moved during the benchmark run"
grep -q "git_sha:              ${sha}" "${env_file}" || die "environment.txt SHA mismatch"
grep -q "${sha}" "${out_dir}/summary.json" || die "summary.json SHA mismatch"
# the recorded binary sha256 matches the executed binary (and the disk):
grep -q "benchmark_binary_sha256: ${bin_sha256} " "${env_file}" || die "environment.txt binary sha256 mismatch"
[ "$(sha256sum "${bin}" | awk '{print $1}')" = "${bin_sha256}" ] || die "executed binary hash drifted after the run"
echo "self-consistency: OK (sha ${sha} == HEAD before/after build+run =="
echo "  environment.txt == summary.json; binary sha256 ${bin_sha256} recorded+verified)"
echo "=== done. Evidence in ${out_dir}:"
ls -l "${out_dir}"
