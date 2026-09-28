#!/usr/bin/env bash
# CUDALM v0.7 Phase A — CANONICAL PROFILING BASELINE (PROFILE FIRST, NO
# OPTIMIZATION).
#
# This script reproduces the v0.7 Phase A profiling evidence on the FROZEN
# v0.6 continuous-batching runtime (no runtime/kernel semantics are changed):
#
#   timing        : canonical benchmark, --mode both (1 warmup + 5 measured
#                   runs, same request workload as the v0.6 benchmark) ->
#                   benchmarks/v07_baseline_timing.txt
#   nsys          : Nsight Systems over the canonical workload, --mode both
#                   (COMPARISON-HARNESS / MIXED-MODE profile: mode A serial
#                   + mode B batched in one process) ->
#                   benchmarks/profiling/v07_nsys.nsys-rep +
#                   v07_nsys_{kernel,cuda_api,memops,kernexec}_summary.txt
#   nsys_batched  : CONTINUOUS-BATCHED-ONLY serving profile (--mode batched,
#                   1 warmup + 5 measured runs) — the PRIMARY basis for the
#                   Phase B optimization ranking ->
#                   benchmarks/profiling/v07_batched_nsys.nsys-rep +
#                   v07_batched_nsys_{kernel,cuda_api,memops,kernexec}_summary.txt
#   aggregate     : kernel-family aggregation over the MIXED nsys CSV ->
#                   benchmarks/profiling/v07_kernel_families.txt
#   aggregate_batched : same over the BATCHED-ONLY nsys CSV ->
#                   benchmarks/profiling/v07_batched_kernel_families.txt
#   ncu TARGET    : Nsight Compute for ONE target (top hotspots only, reduced
#                   workload, launch-count limited, batched-only mode) ->
#                   benchmarks/profiling/v07_ncu_<target>.txt
#                   targets: int4_b1 int4_batch lmhead_b1 lmhead_batch
#                            deltanet deltanet_batch rmsnorm
#
# EXACT-SHA EVIDENCE DISCIPLINE: every evidence stage REFUSES to run when
# the TRACKED tree differs from the exact SHA (the artifacts must be
# generated on a clean tree; untracked files are the expected evidence
# outputs of the run itself). The header records that SHA + tree state.
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

# Canonical timing / nsys: 1 warmup + 5 measured runs (v0.7 baseline
# requirement).
MEASURED_RUNS="${CUDALM_MEASURED_RUNS:-5}"
# NCU reduced workload (clearly distinguished from the canonical benchmark;
# real checkpoint + real batched decode + fixed/reproducible): 1 warmup +
# 1 measured run, batched-only mode, launch-count limited.
NCU_MEASURED_RUNS="${CUDALM_NCU_MEASURED_RUNS:-1}"
NCU_LAUNCH_COUNT="${CUDALM_NCU_LAUNCH_COUNT:-12}"

# NCU targets: name -> "regex|mode|launch-skip"
#   * NCU 2022.3 -k regex matches the bare kernel FUNCTION name (no
#     namespace prefix), so '^' anchors the base name: a B=1 target can
#     never match its batch_* sibling (the batch name starts with 'batch_')
#     and vice versa;
#   * all targets run in --mode batched (the serving path): B=1 kernels
#     come from the single-attempt traversals (prefill + size-1 cohorts),
#     batch_* kernels from the committed B>=2 cohorts;
#   * committed-batch cohorts per run are B=2,B=3,B=3 (avg 2.667, max 3);
#     one committed batch traversal issues 186 batch int4 launches
#     (measured in the batched nsys CSV), so int4_batch (skip 0) captures
#     the first cohort (B=2) and int4_batch_late (skip 186) the second
#     cohort (B=3).
declare -A NCU_TARGET=(
  [int4_b1]='^int4gemv_rowtile4_bf16|batched|0'
  [int4_batch]='^batch_int4gemv_rowtile4_bf16|batched|0'
  [int4_batch_late]='^batch_int4gemv_rowtile4_bf16|batched|186'
  [lmhead_b1]='^bf16_gemv_vec4_row|batched|0'
  [lmhead_batch]='^batch_bf16_gemv_vec4_row|batched|0'
  [deltanet]='^deltanet_|batched|0'
  [deltanet_batch]='^batch_deltanet_delta|batched|0'
  [rmsnorm]='rmsnorm|batched|0'
)

SHA="$(git -C "$SRC" rev-parse HEAD)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
CUDA_DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
CUDA_RT_LIB="$(/usr/local/cuda-11.8/bin/nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1 || true)"

# EXACT-SHA EVIDENCE DISCIPLINE: refuse to generate evidence when the
# tracked tree differs from HEAD (modified/deleted files). Untracked files
# are ALLOWED (they are the expected evidence outputs of this very run) but
# are reported for the record — the final docs/evidence commit must contain
# exactly those files, no more.
check_clean_tree() {
  local dirty
  dirty="$(git -C "$SRC" status --porcelain | grep -v '^??' || true)"
  if [ -n "$dirty" ]; then
    {
      echo "ERROR: refusing to generate profiling evidence: tracked tree is"
      echo "NOT clean at $SHA (git status --porcelain shows tracked-file"
      echo "drift):"
      echo "$dirty"
      echo "commit (or stash) first; evidence must be bound to an exact SHA"
      echo "on a clean tree."
    } >&2
    exit 3
  fi
  local untracked
  untracked="$(git -C "$SRC" status --porcelain | grep '^??' || true)"
  if [ -n "$untracked" ]; then
    echo "note: untracked files present (expected evidence outputs of this run):"
    echo "$untracked"
  fi
}

header() {
  echo "# ============================================================================="
  echo "# CUDALM v0.7 Phase A profiling — $1"
  echo "# V07A evidence at exact SHA: $SHA"
  echo "# tree state: tracked tree CLEAN at generation (no modified/deleted"
  echo "#             tracked files; untracked = evidence outputs of this run)"
  echo "# GPU: $GPU_NAME | driver $CUDA_DRV | CUDA ${CUDA_RT_LIB:-11.8}"
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
  check_clean_tree
  mkdir -p "$OUT"
  local out="$SRC/benchmarks/v07_baseline_timing.txt"
  local tmp="$OUT/.timing_body.txt"
  # The benchmark writes its report to the file itself (truncating), so run
  # it against a temp file first and prepend the header afterwards.
  (cd "$BUILD" && "$BENCH" $(bench_args) "$tmp" --measured-runs "$MEASURED_RUNS" \
      >/dev/null 2>"$OUT/timing_run.log")
  {
    header "canonical baseline timing (--mode both, $MEASURED_RUNS measured runs)"
    echo "# command:"
    echo "#   cd $BUILD && ./benchmarks/bench_qwen35_continuous_batching \\"
    echo "#     data/qwen35_08b_full.cudalm $CKPT $PY $SRC --no-convert \\"
    echo "#     <tmp> --measured-runs $MEASURED_RUNS --mode both   (report prepended with this header)"
    echo "# ------------------------------------------------------------------------"
    cat "$tmp"
  } > "$out"
  rm -f "$tmp"
  echo "wrote $out"
}

# Shared nsys-stats extraction: <rep> <name-prefix> <mode-label>
nsys_stats() {
  local rep="$1" prefix="$2" label="$3" hdr="$4"
  local r base
  for r in gpukernsum cudaapisum gpumemtimesum kernexecsum; do
    case "$r" in
      gpukernsum) base="${prefix}_kernel_summary" ;;
      cudaapisum) base="${prefix}_cuda_api_summary" ;;
      gpumemtimesum) base="${prefix}_memops_summary" ;;
      kernexecsum) base="${prefix}_kernexec_summary" ;;
    esac
    { cat "$hdr"; nsys stats --report "$r" "$rep.nsys-rep"; } > "$OUT/$base.txt"
    nsys stats --report "$r" --format csv -o "$OUT/$base" "$rep.nsys-rep" >/dev/null 2>&1 || true
  done
}

cmd_nsys() {
  check_clean_tree
  mkdir -p "$OUT"
  local rep="$OUT/v07_nsys"
  header "nsys — COMPARISON-HARNESS / MIXED-MODE profile (--mode both, $MEASURED_RUNS measured runs)" \
    > "$OUT/nsys_header.txt"
  echo "running: nsys profile -> $rep.nsys-rep (mixed, $MEASURED_RUNS measured runs)"
  (cd "$BUILD" && nsys profile -o "$rep" -f true \
      "$BENCH" $(bench_args) --measured-runs "$MEASURED_RUNS" --mode both \
      >/dev/null 2>"$OUT/nsys_run.log")
  nsys_stats "$rep" "v07_nsys" "both" "$OUT/nsys_header.txt"
  echo "wrote $OUT/v07_nsys_{kernel,cuda_api,memops,kernexec}_summary.txt (+ .csv)"
}

cmd_nsys_batched() {
  check_clean_tree
  mkdir -p "$OUT"
  local rep="$OUT/v07_batched_nsys"
  header "nsys — CONTINUOUS-BATCHED-ONLY serving profile (--mode batched, $MEASURED_RUNS measured runs; PRIMARY Phase-B basis)" \
    > "$OUT/nsys_batched_header.txt"
  echo "running: nsys profile -> $rep.nsys-rep (batched-only, $MEASURED_RUNS measured runs)"
  (cd "$BUILD" && nsys profile -o "$rep" -f true \
      "$BENCH" $(bench_args) --measured-runs "$MEASURED_RUNS" --mode batched \
      >/dev/null 2>"$OUT/nsys_batched_run.log")
  nsys_stats "$rep" "v07_batched_nsys" "batched" "$OUT/nsys_batched_header.txt"
  echo "wrote $OUT/v07_batched_nsys_{kernel,cuda_api,memops,kernexec}_summary.txt (+ .csv)"
}

cmd_aggregate() {
  check_clean_tree
  local kind="${1:-mixed}"
  local csv timing trav key out
  case "$kind" in
    mixed)
      csv="$OUT/v07_nsys_kernel_summary_gpukernsum.csv"
      timing="$SRC/benchmarks/v07_baseline_timing.txt"
      key="total_completed_traversals"
      out="$OUT/v07_kernel_families.txt"
      ;;
    batched)
      csv="$OUT/v07_batched_nsys_kernel_summary_gpukernsum.csv"
      timing="$SRC/benchmarks/v07_baseline_timing.txt"
      key="batched_mode_traversals"
      out="$OUT/v07_batched_kernel_families.txt"
      ;;
    *) echo "unknown aggregate kind '$kind' (want mixed|batched)"; exit 2 ;;
  esac
  # completed traversals over the whole nsys-profiled process — from the
  # canonical timing's profile_totals (the nsys runs use the SAME 1 warmup +
  # 5 measured schedule, so the counts match deterministically):
  trav="$(grep -oE "${key}: [0-9]+" "$timing" | grep -oE '[0-9]+$' | head -1 || true)"
  [ -n "$trav" ] || trav=0
  mkdir -p "$OUT"
  {
    header "kernel-family aggregation ($kind) from nsys kernel summary CSV"
    echo "# nsys CSV: $csv"
    echo "# completed model traversals over the whole profiled process ($key): $trav"
  } > "$out"
  "$PY" "$SRC/scripts/aggregate_kernel_families.py" "$csv" "$trav" >> "$out"
  echo "wrote $out"
}

cmd_ncu() {
  check_clean_tree
  local target="$1"
  local spec="${NCU_TARGET[$target]:-}"
  [ -n "$spec" ] || { echo "unknown ncu target '$target' (have: ${!NCU_TARGET[*]})"; exit 2; }
  local regex="${spec%%|*}"; local rest="${spec#*|}"
  local mode="${rest%%|*}"; local skip="${rest##*|}"
  mkdir -p "$OUT"
  local out="$OUT/v07_ncu_$target.txt"
  {
    header "ncu — target '$target' (regex: $regex, mode: $mode, launch-skip: $skip)"
    echo "# REDUCED workload (clearly distinguished from the canonical"
    echo "# benchmark): --measured-runs $NCU_MEASURED_RUNS --mode $mode, NCU"
    echo "# launch-count limited to $NCU_LAUNCH_COUNT matched launches"
    echo "# (skip $skip)."
    echo "# command:"
    echo "#   ncu -k regex:$regex --launch-skip $skip --launch-count $NCU_LAUNCH_COUNT \\"
    echo "#       --section SpeedOfLight --section Occupancy \\"
    echo "#       --section MemoryWorkloadAnalysis --section LaunchStats \\"
    echo "#       --section WarpStateStats --section SchedulerStats \\"
    echo "#       $BENCH <canonical workload args> --measured-runs $NCU_MEASURED_RUNS --mode $mode"
  } > "$out"
  (cd "$BUILD" && ncu -k "regex:$regex" --launch-skip "$skip" \
      --launch-count "$NCU_LAUNCH_COUNT" \
      --section SpeedOfLight --section Occupancy \
      --section MemoryWorkloadAnalysis --section LaunchStats \
      --section WarpStateStats --section SchedulerStats \
      "$BENCH" $(bench_args) --measured-runs "$NCU_MEASURED_RUNS" --mode "$mode" \
      >> "$out" 2>&1) || true
  echo "wrote $out"
}

stage="${1:-help}"
case "$stage" in
  timing)            cmd_timing ;;
  nsys)              cmd_nsys ;;
  nsys_batched)      cmd_nsys_batched ;;
  aggregate)         cmd_aggregate "${2:-mixed}" ;;
  aggregate_batched) cmd_aggregate "batched" ;;
  ncu)               cmd_ncu "${2:?ncu target required}" ;;
  *)
    echo "usage: $0 {timing|nsys|nsys_batched|aggregate [mixed|batched]|aggregate_batched|ncu <target>}"
    echo "ncu targets: ${!NCU_TARGET[*]}"
    ;;
esac
