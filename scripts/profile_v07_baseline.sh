#!/usr/bin/env bash
# CUDALM v0.7 Phase A — CANONICAL PROFILING BASELINE (PROFILE FIRST, NO
# OPTIMIZATION).
#
# This script reproduces the v0.7 Phase A profiling evidence on the FROZEN
# v0.6 continuous-batching runtime (no runtime/kernel semantics are changed):
#
#   timing     : canonical benchmark (1 warmup + 5 measured runs, same
#                request workload as the v0.6 benchmark) ->
#                benchmarks/v07_baseline_timing.txt
#   nsys       : Nsight Systems over the canonical workload ->
#                benchmarks/profiling/v07_nsys.nsys-rep +
#                v07_nsys_{kernel,cuda_api,memops}_summary.txt (+CSV)
#   aggregate  : kernel-family aggregation (family table, top-10 by time /
#                by count, short-kernel share, kernels per completed
#                traversal) -> benchmarks/profiling/v07_kernel_families.txt
#   ncu REGEX  : Nsight Compute for ONE kernel-family regex (top hotspots
#                only, reduced workload, launch-count limited) ->
#                benchmarks/profiling/v07_ncu_<name>.txt
#
# Environment (must match the Phase A evidence binding):
#   GPU: RTX 2080 Ti (sm_75), CUDA 11.8 runtime, real checkpoint
#   /root/models/Qwen3.5-0.8B-Base, single stream, serial prefill, true
#   batched decode.
#
# NOTE: one GPU job at a time (do not run two stages in parallel).

set -euo pipefail

SRC="${CUDALM_SRC:-/root/code/cudalm-agent}"
BUILD="$SRC/build"
BENCH="$BUILD/benchmarks/bench_qwen35_continuous_batching"
MODEL="$BUILD/data/qwen35_08b_full.cudalm"
CKPT="${CUDALM_CKPT:-/root/models/Qwen3.5-0.8B-Base}"
PY="${CUDALM_PY:-/root/py311/venv/bin/python3}"
OUT="$SRC/benchmarks/profiling"

# Canonical timing: 1 warmup + 5 measured runs (v0.7 baseline requirement).
MEASURED_RUNS="${CUDALM_MEASURED_RUNS:-5}"
# NCU reduced workload (clearly distinguished from the canonical benchmark;
# real checkpoint + real batched decode + fixed/reproducible): 1 measured
# run, launch-count limited per family.
NCU_MEASURED_RUNS="${CUDALM_NCU_MEASURED_RUNS:-1}"
NCU_LAUNCH_COUNT="${CUDALM_NCU_LAUNCH_COUNT:-12}"

# Kernel-family regexes (NCU -k filters) — the same families used by the
# aggregation:
declare -A FAMILY_REGEX=(
  [w4a16_int4]='(int4gemv)'
  [bf16_lmhead]='(bf16_gemv)'
  [deltanet]='(deltanet)'
  [paged_attention]='(paged)'
  [rmsnorm]='(rmsnorm)'
  [rope]='(rope)'
  [embedding]='(embed)'
  [elementwise]='(qwen35_add|gate_mul|silu_mul|split_q_gate)'
)

SHA="$(git -C "$SRC" rev-parse HEAD)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
CUDA_RT="$(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | head -1 || true)"

header() {
  echo "# ============================================================================="
  echo "# CUDALM v0.7 Phase A profiling — $1"
  echo "# git SHA (frozen v0.6 baseline / v0.7 tooling): $SHA"
  echo "# GPU: $GPU_NAME | ${CUDA_RT}"
  echo "# checkpoint: $CKPT"
  echo "# model: $MODEL"
  echo "# workload: 4 requests (prompts 2/5/3/4 tok; max_new 3/6/5/3;"
  echo "#           1 greedy + 3 seeded); dynamic arrival A,B -> C -> D;"
  echo "#           serial prefill, true batched decode, single stream"
  echo "# date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# ============================================================================="
}

bench_args() {
  # shared benchmark invocation (canonical workload)
  echo "$MODEL $CKPT $PY $SRC --no-convert"
}

cmd_timing() {
  mkdir -p "$OUT"
  local out="$SRC/benchmarks/v07_baseline_timing.txt"
  local tmp="$OUT/.timing_body.txt"
  # The benchmark writes its report to the file itself (truncating), so run
  # it against a temp file first and prepend the header afterwards.
  (cd "$BUILD" && "$BENCH" $(bench_args) "$tmp" --measured-runs "$MEASURED_RUNS" \
     >/dev/null 2>"$OUT/timing_run.log")
  {
    header "canonical baseline timing ($MEASURED_RUNS measured runs)"
    echo "# command:"
    echo "#   cd $BUILD && ./benchmarks/bench_qwen35_continuous_batching \\"
    echo "#     data/qwen35_08b_full.cudalm $CKPT $PY $SRC --no-convert \\"
    echo "#     <tmp> --measured-runs $MEASURED_RUNS   (report prepended with this header)"
    echo "# ------------------------------------------------------------------------"
    cat "$tmp"
  } > "$out"
  rm -f "$tmp"
  echo "wrote $out"
}

cmd_nsys() {
  mkdir -p "$OUT"
  local rep="$OUT/v07_nsys"
  header "nsys" > "$OUT/nsys_header.txt"
  echo "running: nsys profile -> $rep.nsys-rep (canonical, $MEASURED_RUNS measured runs)"
  (cd "$BUILD" && nsys profile -o "$rep" -f true \
     "$BENCH" $(bench_args) --measured-runs "$MEASURED_RUNS" >/dev/null 2>"$OUT/nsys_run.log")
  # Textual evidence (table format, no UI dependency). NOTE: nsys 2022.4
  # report names (newer nsys uses cuda_gpu_kern_sum & co; here: the 2022.x
  # names):
  for r in gpukernsum cudaapisum gpumemtimesum kernexecsum; do
    local base
    case "$r" in
      gpukernsum) base="v07_nsys_kernel_summary" ;;
      cudaapisum) base="v07_nsys_cuda_api_summary" ;;
      gpumemtimesum) base="v07_nsys_memops_summary" ;;
      kernexecsum) base="v07_nsys_kernexec_summary" ;;
    esac
    { cat "$OUT/nsys_header.txt"; nsys stats --report "$r" "$rep.nsys-rep"; } \
      > "$OUT/$base.txt"
    nsys stats --report "$r" --format csv -o "$OUT/$base" "$rep.nsys-rep" >/dev/null 2>&1 || true
  done
  echo "wrote $OUT/v07_nsys_{kernel,cuda_api,memops,kernexec}_summary.txt (+ .csv)"
}

cmd_aggregate() {
  local csv="$OUT/v07_nsys_kernel_summary_gpukernsum.csv"
  local timing="$SRC/benchmarks/v07_baseline_timing.txt"
  # completed traversals over the whole nsys-profiled process (both modes,
  # warmup + measured) — from the benchmark's profile_totals section:
  local trav
  trav="$(grep -oE 'total_completed_traversals: [0-9]+' "$timing" | grep -oE '[0-9]+$' | head -1 || true)"
  [ -n "$trav" ] || trav=0
  mkdir -p "$OUT"
  {
    header "kernel-family aggregation (from nsys kernel summary CSV)"
    echo "# nsys CSV: $csv"
    echo "# completed model traversals over the whole profiled process: $trav"
  } > "$OUT/v07_kernel_families.txt"
  "$PY" "$SRC/scripts/aggregate_kernel_families.py" "$csv" "$trav" >> "$OUT/v07_kernel_families.txt"
  echo "wrote $OUT/v07_kernel_families.txt"
}

cmd_ncu() {
  local name="$1"
  local regex="${FAMILY_REGEX[$name]:-}"
  [ -n "$regex" ] || { echo "unknown family '$name' (have: ${!FAMILY_REGEX[*]})"; exit 2; }
  mkdir -p "$OUT"
  local out="$OUT/v07_ncu_$name.txt"
  {
    header "ncu — kernel family '$name' (regex: $regex)"
    echo "# REDUCED workload (clearly distinguished from the canonical"
    echo "# benchmark): --measured-runs $NCU_MEASURED_RUNS, NCU launch-count"
    echo "# limited to $NCU_LAUNCH_COUNT per matched kernel."
    echo "# command:"
    echo "#   ncu -k regex:$regex --launch-count $NCU_LAUNCH_COUNT \\"
    echo "#       --section SpeedOfLight --section Occupancy \\"
    echo "#       --section MemoryWorkloadAnalysis --section LaunchStats \\"
    echo "#       --section WarpStateStats --section SchedulerStats \\"
    echo "#       $BENCH <canonical workload args> --measured-runs $NCU_MEASURED_RUNS"
  } > "$out"
  (cd "$BUILD" && ncu -k "regex:$regex" --launch-count "$NCU_LAUNCH_COUNT" \
      --section SpeedOfLight --section Occupancy \
      --section MemoryWorkloadAnalysis --section LaunchStats \
      --section WarpStateStats --section SchedulerStats \
      "$BENCH" $(bench_args) --measured-runs "$NCU_MEASURED_RUNS" >> "$out" 2>&1) || true
  echo "wrote $out"
}

stage="${1:-help}"
case "$stage" in
  timing)    cmd_timing ;;
  nsys)      cmd_nsys ;;
  aggregate) cmd_aggregate ;;
  ncu)       cmd_ncu "${2:?family name required}" ;;
  *)
    echo "usage: $0 {timing|nsys|aggregate|ncu <family>}"
    echo "families: ${!FAMILY_REGEX[*]}"
    ;;
esac
