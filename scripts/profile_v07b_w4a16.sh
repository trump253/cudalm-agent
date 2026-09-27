#!/usr/bin/env bash
# CUDALM v0.7 Phase B — W4A16 GEMV optimization evidence.
#
# Stages (each REFUSES to run when the tracked tree is not clean at HEAD —
# the EXACT-SHA evidence discipline of Phase A, unchanged):
#
#   census         : authoritative W4A16 production shape census (model
#                    config from the .cudalm file + call-site mirror,
#                    reconciled against the Phase-A batched-only nsys CSV)
#                    -> benchmarks/profiling/v07b_w4a16_shape_census.txt
#   microbench     : dedicated W4A16 GEMV microbenchmark over the census
#                    (frozen R4 vs R1/R2/R8, B=1/2/3, CUDA events,
#                    weighted production score) ->
#                    benchmarks/v07b_w4a16_microbench.txt
#   ncu TARGET     : Nsight Compute over ONE controlled (N,K,B,variant)
#                    GEMV launch (tool_ncu_w4a16_target) ->
#                    benchmarks/profiling/v07b_ncu_<TARGET>.txt
#                    (TARGET = one of the NCU_TARGET keys below)
#   e2e            : canonical continuous-batched end-to-end, --mode
#                    batched, 2 warmup + 10 measured runs (reviewer §15) ->
#                    benchmarks/v07b_continuous_batching.txt
#   nsys_batched   : CONTINUOUS-BATCHED-ONLY nsys profile after the
#                    dispatcher freeze (same canonical workload and 1 warmup
#                    + 5 measured schedule as the Phase-A batched profile,
#                    so the numbers are directly comparable) ->
#                    benchmarks/profiling/v07b_batched_nsys.nsys-rep +
#                    v07b_batched_nsys_{kernel,cuda_api,memops,kernexec}_summary.txt
#   aggregate_batched : kernel-family aggregation over the Phase-B batched
#                    nsys CSV (same tool as Phase A) ->
#                    benchmarks/profiling/v07b_batched_kernel_families.txt
#
# Environment (matches Phase A): RTX 2080 Ti (sm_75), CUDA 11.8 runtime,
# real checkpoint /root/models/Qwen3.5-0.8B-Base, single stream, serial
# prefill, true batched decode. One GPU job at a time.

set -euo pipefail

SRC="${CUDALM_SRC:-/root/code/cudalm-agent}"
BUILD="$SRC/build"
BENCH="$BUILD/benchmarks/bench_qwen35_continuous_batching"
NCUTGT="$BUILD/tool_ncu_w4a16_target"
CENSUS_TOOL="$BUILD/tool_dump_w4a16_shape_census"
MICROBENCH="$BUILD/benchmarks/bench_w4a16_qwen35_shapes"
MODEL="$BUILD/data/qwen35_08b_full.cudalm"
CKPT="${CUDALM_CKPT:-/root/models/Qwen3.5-0.8B-Base}"
PY="${CUDALM_PY:-/root/py311/venv/bin/python3}"
OUT="$SRC/benchmarks/profiling"
PHASEA_CSV="$OUT/v07_batched_nsys_kernel_summary_gpukernsum.csv"

# E2E (reviewer §15): 2 warmup + 10 measured runs, batched-only mode.
E2E_WARMUP_RUNS="${CUDALM_E2E_WARMUP_RUNS:-2}"
E2E_MEASURED_RUNS="${CUDALM_E2E_MEASURED_RUNS:-10}"
# Nsys: SAME schedule as the Phase-A batched profile (1 warmup + 5 measured
# runs) so the family numbers are directly comparable.
NSYS_MEASURED_RUNS="${CUDALM_NSYS_MEASURED_RUNS:-5}"
NSYS_WARMUP_RUNS="${CUDALM_NSYS_WARMUP_RUNS:-1}"
# NCU: 8 warmup launches inside the tool are skipped via --launch-skip;
# 12 measured launches (same launch-count as Phase A).
NCU_ITERS="${CUDALM_NCU_ITERS:-64}"
NCU_LAUNCH_COUNT="${CUDALM_NCU_LAUNCH_COUNT:-12}"
NCU_SKIP="${CUDALM_NCU_SKIP:-8}"

# Evidence tag: which candidate/state this evidence run binds to.
#   CUDALM_EVIDENCE_TAG=candidate  -> v07b_candidate_* artifacts
#   CUDALM_EVIDENCE_TAG=final      -> v07b_final_* artifacts (default)
# CUDALM_EVIDENCE_SHA_TAG: the SHA symbol line in the evidence header
#   (V07B / V07B_CANDIDATE / V07B_FINAL_FUNCTIONAL).
TAG="${CUDALM_EVIDENCE_TAG:-final}"
SHA_TAG="${CUDALM_EVIDENCE_SHA_TAG:-V07B}"

# NCU targets: name -> "N|K|B|variant"
#   * small-N pathological: N=16 (in_proj_b/a) — frozen vs the R1 winner,
#     B=1 and B=3 (batch);
#   * medium: N=512 (k/v_proj) — frozen vs the R2 winner, B=1 and B=2;
#   * large: N=6144 (in_proj_qkv) — frozen (production keeps R4) + R1 at
#     B=3 to record the rejection evidence.
declare -A NCU_TARGET=(
  [n16_b1_frozen]='16|1024|1|frozen'
  [n16_b1_r1]='16|1024|1|r1'
  [n16_b3_frozen]='16|1024|3|frozen'
  [n16_b3_r1]='16|1024|3|r1'
  [n512_b1_frozen]='512|1024|1|frozen'
  [n512_b1_r2]='512|1024|1|r2'
  [n512_b2_frozen]='512|1024|2|frozen'
  [n512_b2_r2]='512|1024|2|r2'
  [n6144_b1_frozen]='6144|1024|1|frozen'
  [n6144_b3_frozen]='6144|1024|3|frozen'
  [n6144_b3_r1]='6144|1024|3|r1'
)
# kernel-name regex per variant (NCU 2022.3 -k regex matches the bare
# demangled FUNCTION name; '^' anchors — see Phase A script comments).
variant_regex() {
  case "$1" in
    frozen)
      if [ "$2" = "1" ]; then echo '^int4gemv_rowtile4_bf16'
      else echo '^batch_int4gemv_rowtile4_bf16'; fi ;;
    r1) echo '^int4gemv_rowtile_bf16_kernel' ;;
    r2) echo '^int4gemv_rowtile_bf16_kernel' ;;
    r8) echo '^int4gemv_rowtile_bf16_kernel' ;;
    *) echo "bad variant '$1'" >&2; return 1 ;;
  esac
}

SHA="$(git -C "$SRC" rev-parse HEAD)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
CUDA_DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
CUDA_RT_LIB="$(/usr/local/cuda-11.8/bin/nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1 || true)"

# EXACT-SHA EVIDENCE DISCIPLINE (same as Phase A): refuse tracked drift;
# untracked files are the expected evidence outputs of this run (reported).
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
  echo "# CUDALM v0.7 Phase B (W4A16 GEMV optimization) — $1"
  echo "# ${SHA_TAG} evidence (tag: $TAG) at exact SHA: $SHA"
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
  echo "$MODEL $CKPT $PY $SRC --no-convert"
}

# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------
cmd_census() {
  check_clean_tree
  mkdir -p "$OUT"
  local out="$OUT/v07b_w4a16_shape_census.txt"
  {
    header "W4A16 production shape census (authoritative)"
    echo "# sources: (1) actual model config from $MODEL (WeightFileV2), (2)"
    echo "# actual call sites (qwen35_full_attention.cpp / qwen35_deltanet.cpp,"
    echo "# shapes pinned by weight_loader_v2.cpp::validate_layer), (3) Phase-A"
    echo "# batched-only nsys CSV reconciliation: $PHASEA_CSV"
    echo "# command:"
    echo "#   $CENSUS_TOOL $MODEL $PHASEA_CSV"
    echo "# ------------------------------------------------------------------------"
    "$CENSUS_TOOL" "$MODEL" "$PHASEA_CSV"
  } > "$out"
  echo "wrote $out"
}

# ---------------------------------------------------------------------------
# microbench
# ---------------------------------------------------------------------------
cmd_microbench() {
  check_clean_tree
  local out="$SRC/benchmarks/v07b_w4a16_microbench.txt"
  local tmp="$OUT/.microbench_body.txt"
  mkdir -p "$OUT"
  (cd "$BUILD" && "$MICROBENCH" "$tmp" >/dev/null)
  {
    header "W4A16 microbenchmark (census shapes x B=1/2/3 x frozen/R1/R2/R8)"
    echo "# command:"
    echo "#   cd $BUILD && ./benchmarks/bench_w4a16_qwen35_shapes <tmp>"
    echo "# ------------------------------------------------------------------------"
    cat "$tmp"
  } > "$out"
  rm -f "$tmp"
  echo "wrote $out"
}

# ---------------------------------------------------------------------------
# ncu TARGET
# ---------------------------------------------------------------------------
cmd_ncu() {
  check_clean_tree
  local target="$1"
  local spec="${NCU_TARGET[$target]:-}"
  [ -n "$spec" ] || { echo "unknown ncu target '$target' (have: ${!NCU_TARGET[*]})"; exit 2; }
  local N K B variant
  IFS='|' read -r N K B variant <<< "$spec"
  local regex
  regex="$(variant_regex "$variant" "$B")"
  mkdir -p "$OUT"
  local out="$OUT/v07b_ncu_$target.txt"
  # For r1/r2/r8 the regex matches the template family; the tool launches
  # only ONE variant, so every matched launch is that variant.
  {
    header "ncu — W4A16 target '$target' (N=$N K=$K B=$B variant=$variant, regex: $regex)"
    echo "# controlled single-shape target (tool_ncu_w4a16_target): $NCU_SKIP"
    echo "# warmup launches skipped, $NCU_LAUNCH_COUNT measured launches of"
    echo "# $NCU_ITERS total (launch-skip $NCU_SKIP, launch-count $NCU_LAUNCH_COUNT)."
    echo "# command:"
    echo "#   cd $BUILD && ncu -k regex:$regex --launch-skip $NCU_SKIP \\"
    echo "#       --launch-count $NCU_LAUNCH_COUNT \\"
    echo "#       --section SpeedOfLight --section Occupancy \\"
    echo "#       --section MemoryWorkloadAnalysis --section LaunchStats \\"
    echo "#       --section WarpStateStats --section SchedulerStats \\"
    echo "#       tool_ncu_w4a16_target $N $K $B $variant $NCU_ITERS"
  } > "$out"
  (cd "$BUILD" && ncu -k "regex:$regex" --launch-skip "$NCU_SKIP" \
      --launch-count "$NCU_LAUNCH_COUNT" \
      --section SpeedOfLight --section Occupancy \
      --section MemoryWorkloadAnalysis --section LaunchStats \
      --section WarpStateStats --section SchedulerStats \
      "$NCUTGT" "$N" "$K" "$B" "$variant" "$NCU_ITERS" >> "$out" 2>&1)
  echo "wrote $out"
}

# ---------------------------------------------------------------------------
# e2e (canonical continuous-batched, batched-only, 2 warmup + 10 measured)
# ---------------------------------------------------------------------------
cmd_e2e() {
  check_clean_tree
  mkdir -p "$OUT"
  local out="$SRC/benchmarks/v07b_${TAG}_e2e.txt"
  local tmp="$OUT/.e2e_body_${TAG}.txt"
  (cd "$BUILD" && "$BENCH" $(bench_args) "$tmp" \
      --measured-runs "$E2E_MEASURED_RUNS" --warmup-runs "$E2E_WARMUP_RUNS" \
      --mode batched >/dev/null 2>"$OUT/e2e_run_${TAG}.log")
  {
    header "canonical end-to-end, --mode batched, $E2E_WARMUP_RUNS warmup + $E2E_MEASURED_RUNS measured runs"
    echo "# command:"
    echo "#   cd $BUILD && ./benchmarks/bench_qwen35_continuous_batching \\"
    echo "#     data/qwen35_08b_full.cudalm $CKPT $PY $SRC --no-convert \\"
    echo "#     <tmp> --measured-runs $E2E_MEASURED_RUNS --warmup-runs $E2E_WARMUP_RUNS --mode batched"
    echo "# ------------------------------------------------------------------------"
    cat "$tmp"
  } > "$out"
  rm -f "$tmp"
  echo "wrote $out"
}

# ---------------------------------------------------------------------------
# nsys_batched (post-dispatcher; Phase-A-comparable schedule 1+5)
# ---------------------------------------------------------------------------
nsys_stats() {
  local rep="$1" prefix="$2" hdr="$3"
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

cmd_nsys_batched() {
  check_clean_tree
  mkdir -p "$OUT"
  local rep="$OUT/v07b_${TAG}_batched_nsys"
  local tmp="$OUT/.nsys_e2e_report_${TAG}.txt"
  header "nsys — CONTINUOUS-BATCHED-ONLY serving profile AFTER the Phase-B dispatcher freeze (--mode batched, $NSYS_WARMUP_RUNS warmup + $NSYS_MEASURED_RUNS measured; Phase-A-comparable schedule)" \
    > "$OUT/v07b_${TAG}_nsys_batched_header.txt"
  echo "running: nsys profile -> $rep.nsys-rep (batched-only)"
  (cd "$BUILD" && nsys profile -o "$rep" -f true \
      "$BENCH" $(bench_args) "$tmp" \
      --measured-runs "$NSYS_MEASURED_RUNS" --warmup-runs "$NSYS_WARMUP_RUNS" \
      --mode batched >/dev/null 2>"$OUT/v07b_${TAG}_nsys_batched_run.log")
  nsys_stats "$rep" "v07b_${TAG}_batched_nsys" "$OUT/v07b_${TAG}_nsys_batched_header.txt"
  echo "wrote $OUT/v07b_${TAG}_batched_nsys_{kernel,cuda_api,memops,kernexec}_summary.txt (+ .csv)"
  # keep the report for the aggregation denominator + e2e cross-check
  mv "$tmp" "$OUT/v07b_${TAG}_batched_nsys_report.txt"
}

# ---------------------------------------------------------------------------
# aggregate_batched (Phase-B batched nsys CSV)
# ---------------------------------------------------------------------------
cmd_aggregate_batched() {
  check_clean_tree
  local csv="$OUT/v07b_${TAG}_batched_nsys_kernel_summary_gpukernsum.csv"
  local rep="$OUT/v07b_${TAG}_batched_nsys_report.txt"
  [ -f "$csv" ] || { echo "missing $csv (run nsys_batched first)"; exit 2; }
  local trav
  trav="$(grep -oE "batched_mode_traversals: [0-9]+" "$rep" | grep -oE '[0-9]+$' | head -1 || true)"
  [ -n "$trav" ] || { echo "cannot read batched_mode_traversals from $rep"; exit 2; }
  local out="$OUT/v07b_${TAG}_kernel_families.txt"
  {
    header "kernel-family aggregation (Phase-B batched) from nsys kernel summary CSV"
    echo "# nsys CSV: $csv"
    echo "# completed model traversals over the whole profiled process (batched_mode_traversals): $trav"
  } > "$out"
  "$PY" "$SRC/scripts/aggregate_kernel_families.py" "$csv" "$trav" >> "$out"
  echo "wrote $out"
}

case "${1:-}" in
  census)           cmd_census ;;
  microbench)       cmd_microbench ;;
  ncu)              cmd_ncu "${2:?ncu target required}" ;;
  e2e)              cmd_e2e ;;
  nsys_batched)     cmd_nsys_batched ;;
  aggregate_batched) cmd_aggregate_batched ;;
  *)
    echo "usage: $0 {census|microbench|ncu <target>|e2e|nsys_batched|aggregate_batched}"
    echo "ncu targets: ${!NCU_TARGET[*]}"
    exit 2
    ;;
esac
