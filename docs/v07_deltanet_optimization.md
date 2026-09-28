# CUDALM v0.7 Phase C — DeltaNet delta-rule optimization (Qwen3.5-0.8B)

**Verdict (final):** the measured DeltaNet delta-rule table is **KEPT for
production**. The final production state (V07C_FINAL_FUNCTIONAL_SHA)
dispatches the Qwen3.5-0.8B delta-rule state update through the measured
static table — **B=1 → `vchunk`** (2-kernel value-chunk split),
**B=2/B=3 → `vvec`** (float4, 4 values/thread) — with the **frozen
baseline as the bit-compatible fallback** for any unrecognized shape. All
four acceptance criteria were met at the exact candidate SHA (§8):
EXACT bit-exact correctness, a clear DeltaNet weighted-microbench
improvement, a batched-only Nsys confirmation of the DeltaNet-family time
reduction, and a reproducible canonical E2E improvement (150 vs 150
independent raw samples, Welch p = 0.012, Mann-Whitney p < 0.0001).
`vreg` was **REJECTED** (superseded: loses to vchunk at B=1, to vvec at
B=2/3). The frozen entry points remain in the tree directly callable
(Phase-B contract), and the parity/microbench/NCU infrastructure remains
for reproduction.

**SHA chain (this document):**

```
Phase B final (frozen production)      a848bbe3d845c8cfdbf3fb4a8329b5c66bf4ff7b
    ↓
DeltaNet candidate infrastructure      V07C_KERNEL_EXPERIMENT_SHA = 387f140be6010b5dc795456ab241831a40073086
    ↓                                 (vreg/vvec/vchunk launchers + bit-exact parity test
                                       + microbench + NCU target; identity table — the
                                       production runtime still calls the frozen entry
                                       points directly)
measured candidate table + wiring      V07C_CANDIDATE_SHA = 4f4395eb8e2c83e7a7e03fb4b6fb8619fa6b8530
    ↓                                 (static measured table; src/runtime/qwen35_deltanet.cpp
                                       calls the measured dispatcher for B=1 and batch)
exact-SHA candidate evidence           (benchmarks/v07c_deltanet_microbench_run1/2.txt,
                                        benchmarks/profiling/v07c_ncu_*.txt,
                                        benchmarks/profiling/v07c_candidate_batched_nsys_*,
                                        benchmarks/v07c_candidate_e2e.txt)
    ↓
all four acceptance criteria met → KEEP
    ↓
final production state (table kept)    V07C_FINAL_FUNCTIONAL_SHA = 6a9d0b5f1d6307618c3c0c1b92f0fc25ec7ccb2b
                                        (comment-only rebinding on top of the candidate;
                                         production behavior byte-identical to it)
    ↓
docs/evidence-only commit              (this document + all v07c_* evidence artifacts;
                                        touches docs/ + benchmarks/ + benchmarks/profiling/
                                        only — no src/include/tests/scripts/CMake drift)
```

Scope: the Phase-A #2 serving GPU hotspot family — the DeltaNet family,
**15.37%** of serving (continuous-batched) GPU kernel time, of which the
delta-rule state-update kernel `deltanet_delta` (+ its batch variant) is
**11.44%** (Phase A §6.3; census: 18 DeltaNet layers × 1 `deltanet_delta`
call per traversal = 2376 calls over the 132-traversal batched census:
2052 B=1 + 108 B=2 + 126 B=3). Only the delta-rule state-update kernel
was optimized (state-parallel decomposition / thread mapping / vectorized
state I/O). Nothing else was touched: the scheduler, state-manager
lifecycle, paged KV, sampling, tokenizer, the W4A16 production path and
the LM-head format are all frozen; the DeltaNet auxiliary kernels
(gbeta/conv/gated_rmsnorm) were NOT fused (fusion was explicitly
secondary; §9). No tolerance-based correctness gate anywhere — the
contract is BIT-IDENTICAL to the frozen kernels.

Frozen contract (enforced, not assumed): every candidate must be
**BIT-IDENTICAL** (all FP32 recurrent-state words, all BF16 `core_out`
words, and all BF16 q/k/v-side outputs) to the frozen v0.6/v0.7A baseline
kernels `deltanet_delta_kernel`
(`src/kernels/qwen35_deltanet_kernels.cu`, B=1) and
`batch_deltanet_delta_kernel` (`src/kernels/batch_decode.cu`, batch).
The per-element math is preserved exactly: the same 128-thread
l2norm/qk reduction tree (4×warp-shfl 5-step + smem[4] +
((s0+s1)+s2)+s3, thread t owns element t), the same per-value sequential
k-ascending accumulation `A_t = Σ_k S[h,k,t]·exp_g·k_s[k]` and
`B_t = Σ_k S[h,k,t]·exp_g·q_s[k]` (one thread per value), the same
`d_t = (v_t − A_t)·beta`, the same single BF16 RNE rounding of
`o_t = B_t + d_t·qk`, the same state update
`S[h,k,t] = S[h,k,t]·exp_g + k_s[k]·d_t`, and the same BF16 rounding
boundaries in the prologue (the prologue is a shared
`delta_prologue` helper compiled from the frozen op sequence).

---

## 1. Baseline characterization (Phase A, frozen SHA a9b5a6e)

From `benchmarks/profiling/v07_ncu_deltanet*.txt` (Phase A) and the
batched census (`benchmarks/profiling/v07_batched_nsys_kernel_summary.txt`):

| shape | kernel | grid | block | duration | regs | waves | occ | SM% | DRAM% | dominant stall |
|---|---|---|---|---|---|---|---|---|---|---|
| B=1 | `deltanet_delta_kernel` | 16 | 128 | 30.1 µs | 64 | 0.03 | 11.6% | 2.4% | 5.4% | const/scoreboard (latency) |
| B=2 | `batch_deltanet_delta_kernel` | 32 | 128 | 31.0–32.1 µs | 62 | 0.06 | 11.7% | — | — | scoreboard ~13 cyc |
| B=3 | (not NCU-sampled in Phase A) | 48 | 128 | (nsys avg 24.5 µs/call) | — | — | — | — | — | — |

Diagnosis (Phase A §P2, confirmed by the Phase-C NCU in §5): the kernel
is **latency/dependency-bound and occupancy-limited**. One 128-thread
block per head does two serial 128-iteration passes (A/B accumulation,
then in-place state update) over a 128-deep column, re-reading the state
from global in the update pass; 16 blocks for B=1 fill 16/68 SMs (waves
0.03), so the ~13–14 cyc L1TEX scoreboard stall per issued instruction
dominates the 30 µs runtime.

## 2. Candidate design (all three bit-exact by construction)

`src/kernels/deltanet_delta_qwen35.cu` (launchers
`deltanet_delta_{vreg,vvec,vchunk}` + `batch_deltanet_delta_{...}` +
experimental dispatchers + `*_regs()` queries). All candidates share the
frozen prologue (l2norm/qk/exp_g/beta) and per-value op sequence; only
the *work mapping* changes:

* **`vreg`** — 1 block/head, 128 threads, one value per thread (t =
  threadIdx.x). The 128-deep column is kept in a per-thread array across
  the A/B pass and the update pass, so the update pass never re-loads the
  state. (nvcc sm_75, -O2: 46 regs/thread; the 128-float column lives in
  L1-backed local memory — the win is the removed global re-read, not
  register residency.)
* **`vvec`** — 1 block/head, 128 threads; threads 0..31 each process 4
  values (t = 4l..4l+3) via `float4` state loads (128-bit vectorized
  column access, 4 independent per-value chains A0..A3/B0..B3), the
  update pass re-reads the L1-hot column. Requires the state base to be
  16B-aligned; otherwise the launcher **falls back to the frozen kernel**
  (bit-identical by construction).
* **`vchunk`** — 2-kernel value-chunk split. Kernel 1 (`red`, grid
  n_heads, 128 threads) runs the shared prologue and publishes
  q_s/k_s/qk to a lazily-allocated device scratch. Kernel 2 (`s`, grid
  (n_heads, 4), 32 threads) does the per-value work with **one value per
  thread** and the column kept across passes (50 regs/thread on the s
  pass; local-memory-backed column as in vreg). The split raises the
  block count (16 → 64 for B=1) for more SM spread, and each thread runs
  a single 128-iteration chain instead of a strided one.

The batch mirrors use the per-row slot state
(`rec_base + d_slots[b]·n_heads·hd·hd + h·hd·hd`). The dispatcher is a
STATIC measured table keyed on (n_heads[, B]) for the pinned 0.8B config
(n_heads = 16) — no runtime autotuning; any unrecognized legal shape
takes the frozen baseline path.

## 3. EXACT correctness evidence (bit-exact parity)

`tests/cuda/test_deltanet_delta_qwen35_optimized` (target
`test_deltanet_delta_qwen35_optimized`), run at V07C_KERNEL_EXPERIMENT_SHA,
V07C_CANDIDATE_SHA and V07C_FINAL_FUNCTIONAL_SHA — **ALL PASS (1047
checks)** at each:

* B=1 single-step bit-exact (frozen vs candidate, independent
  state/output buffers): 4 seeds × 4 input kinds × 5 g-kinds ×
  3 state kinds = **240 combos × 3 variants**.
* B=1 32-step recurrent state chains (state carried across steps):
  3 seeds × 32 steps = **96 steps × 3 variants**.
* Batch B=2/B=3: core_out + full recurrent state + **row-parity**
  (batch row b vs frozen single-step on the same row) — 6 combos ×
  3 variants × 2 B.
* vvec alignment: +32B state base (16B-aligned → float4 path) and +4B
  state base (16B-**un**aligned → frozen fallback), each compared
  against a frozen reference on a separate non-overlapping buffer —
  bit-exact (both cases).
* Dispatcher vs frozen bit-exact, B=1 and batch — under both the
  identity table (experiment SHA) and the measured table (candidate/final
  SHA).

Comparison is exact word equality (`memcmp` on the BF16/FP32 words) — no
tolerance gate. The E2E golden text-generation test
(`test_qwen35_text_generation`, exact token IDs through the full model)
passes at the candidate and final SHAs with the measured table live in
the runtime.

## 4. DeltaNet weighted microbenchmark

`benchmarks/bench_deltanet_delta_qwen35` (per-iteration CUDA-event
timing; 20000 measured iters/cell after warmup; weighted score
W_Bx = 18 × mean_us(B=x), overall = (19·W_B1 + 1·W_B2 + 2·W_B3)/22 per
the Phase-A census B mix 19:1:2 over 22 traversals).

Candidate-SHA exact runs (V07C_CANDIDATE_SHA, 2 independent process
runs) — mean µs (speedup vs frozen in the last column shown per run):

| B | variant | run1 µs (x) | run2 µs (x) | final-SHA µs (x) |
|---|---|---|---|---|
| 1 | frozen | 22.714 (1.000) | 22.583 (1.000) | 23.385 (1.000) |
| 1 | vreg | 14.600 (1.556) | 14.711 (1.535) | 14.599 (1.602) |
| 1 | **vvec** | 15.260 (1.488) | 15.385 (1.468) | 15.251 (1.533) |
| 1 | **vchunk** | **12.671 (1.793)** | **12.733 (1.774)** | **12.587 (1.858)** |
| 2 | frozen | 21.179 (1.000) | 21.188 (1.000) | 21.222 (1.000) |
| 2 | vreg | 18.609 (1.138) | 18.666 (1.135) | 18.527 (1.145) |
| 2 | **vvec** | **15.969 (1.326)** | **16.064 (1.319)** | **15.974 (1.329)** |
| 2 | vchunk | 20.477 (1.034) | 20.548 (1.031) | 20.593 (1.031) |
| 3 | frozen | 21.571 (1.000) | 21.704 (1.000) | 21.559 (1.000) |
| 3 | vreg | 29.394 (0.734) | 29.681 (0.731) | 29.429 (0.733) |
| 3 | **vvec** | **16.355 (1.319)** | **16.347 (1.328)** | **16.395 (1.315)** |
| 3 | vchunk | 32.997 (0.654) | 32.433 (0.669) | (0.66x) |

Per-token µs (B=2: 10.59 → 7.99; B=3: 7.19 → 5.45). Winner per B is
stable across all 3 runs: **B=1 → vchunk, B=2 → vvec, B=3 → vvec**.
`vreg` is REJECTED (superseded: 0.73x at B=3, loses its B=1 margin to
vchunk and its B=2 margin to vvec). Weighted single-variant scores:
vchunk −22.5%/−23.8%, vvec −21.9%/−21.8% vs frozen; the measured mixed
table (vchunk B=1 + vvec B=2/3) scores ≈ **−42%** delta-kernel time per
traversal ((19×18×12.7 + 18×16.0 + 2×18×16.4)/22 ≈ 236 µs vs ≈ 408 µs).

## 5. NCU before/after (candidate SHA, `tools/ncu_deltanet_target`)

12 sampled launches each (8 warmup skipped), deterministic fixture,
`--section SpeedOfLight Occupancy MemoryWorkloadAnalysis LaunchStats
WarpStateStats SchedulerStats`. Median over samples:

| B | kernel | dur (µs) | grid | block | waves | SM% | occ% | regs | WIPC (cyc) | top stall |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | frozen | 30.46 | 16 | 128 | 0.030 | 2.44 | 11.46 | 64 | 15.6 | 11.9 cyc L1TEX scoreboard (76%) |
| 1 | vchunk red | 3.39 | 16 | 128 | 0.030 | 1.97 | 12.48 | 19 | 9.6 | — |
| 1 | vchunk s | 19.01 | 64 | 32 | 0.060 | 5.38 | 3.12 | 50 | 8.1 | 5.4 cyc L1TEX scoreboard (67%) |
| 2 | frozen | 32.32 | 32 | 128 | 0.060 | 4.66 | 11.68 | 62 | 17.0 | 13.2 cyc L1TEX scoreboard (77.6%) |
| 2 | vvec | 27.94 | 32 | 128 | 0.060 | 2.41 | 3.93 | 53 | 8.5 | 4.5 cyc L1TEX scoreboard (53.1%) |
| 3 | frozen | 33.31 | 48 | 128 | 0.090 | 6.75 | 11.80 | 62 | 17.6 | 13.7 cyc L1TEX scoreboard (77.7%) |
| 3 | vvec | 28.61 | 48 | 128 | 0.090 | 3.55 | 3.93 | 53 | 8.6 | 4.5 cyc L1TEX scoreboard (53.0%) |

Reading: the bottleneck is the L1TEX scoreboard stall on the column
loads (13–14 cyc per issued instruction, ~77% of issue cycles); the
candidates halve warp-cycles-per-instruction (15.6→8.1, 17.0→8.5,
17.6→8.6) and the scoreboard stall (11.9→5.4 / 13.2→4.5 / 13.7→4.5 cyc).
vvec's lower *achieved* occupancy (3.9% vs 11.7%) is structural — only 1
of the 4 warps per block stays in the main loop — but the duration
improves because the removed memory latency dominates. NCU kernel-time
deltas: B=1 30.46 → 22.40 µs (vchunk red+s, 1.36x), B=2 32.32 → 27.94
µs (1.16x), B=3 33.31 → 28.61 µs (1.16x). (NCU serializes each launch,
so its absolute durations exceed the pipelined microbench numbers; the
relative behavior agrees with §4.)

## 6. Batched-only Nsys before/after (full model, canonical workload)

`nsys profile --mode batched --measured-runs 5`, `gpukernsum`:
baseline = Phase-B final frozen production (V07B_FINAL_FUNCTIONAL_SHA
46a5998, `benchmarks/profiling/v07b_batched_nsys.nsys-rep` — the
DeltaNet kernels there are byte-identical to the Phase-C experiment-SHA
production); candidate = V07C_CANDIDATE_SHA
(`benchmarks/profiling/v07c_candidate_batched_nsys.nsys-rep`).

| metric (5 measured batched runs, 110 traversals) | baseline (frozen) | candidate | Δ |
|---|---|---|---|
| total GPU kernel time | 512.86 ms | 505.70 ms | −7.16 ms |
| DeltaNet family total | 78.77 ms (15.36%) | 64.58 ms (12.77%) | **−14.19 ms (−18.0%)** |
| delta-rule kernels (per call) | 58.69 ms / 2376 calls = 24.70 µs (11.44%) | 44.43 ms / 2376 calls = 18.70 µs (8.79%) | **−14.26 ms (−24.3%)** |
| B=1 delta (2052 calls) | `deltanet_delta_kernel` 50.25 ms = 24.49 µs/call | vchunk s+red 30.90+6.30 ms = 18.13 µs/call | −26.0% |
| batch delta (324 calls) | `batch_deltanet_delta_kernel` 8.45 ms = 26.08 µs/call | `batch_deltanet_delta_vvec` 7.24 ms = 22.35 µs/call | −14.3% |
| DeltaNet family launch count | 9504 | 11556 (+2052: vchunk adds a red pass per B=1 call) | +21.6% |

Per-kernel diff confirms the family swap is the only family change
(gbeta/conv/gated_rmsnorm unchanged). Note: the *total* GPU-time delta
(−7.16 ms) is smaller than the delta-rule delta (−14.26 ms) because two
large non-family GEMV kernels drift upward between sessions
(`int4gemv_rowtile4_bf16_kernel` +3.97 ms, `bf16_gemv_vec4_row_kernel`
+1.74 ms, +2.3%/+1.7% — session/clock variation on code that is
byte-identical); the in-session E2E A/B below is the drift-free measure.

## 7. Canonical E2E (in-session A/B, raw samples)

15 independent invocations per side (each a fresh process; 2 un-timed
warmup + 10 measured runs each; `--mode batched`; canonical 4-request
workload, 22 traversals/run), raw per-run wall times:
`benchmarks/v07c_baseline_e2e.txt` (frozen production, at
V07C_KERNEL_EXPERIMENT_SHA 387f140 — production byte-identical to the
Phase-B final state) and `benchmarks/v07c_candidate_e2e.txt` (measured
table live, at V07C_CANDIDATE_SHA 4f4395e).

| side | n | mean (ms) | median (ms) | sd (ms) |
|---|---|---|---|---|
| baseline (frozen) | 150 | 111.753 | 110.794 | 3.10 |
| candidate (measured table) | 150 | 110.275 | 108.776 | 6.50 |

* **Δ = −1.479 ms/run (−1.32% of wall)** — Welch two-sided
  **t = −2.515, df = 213.6, p = 0.012**; **Mann-Whitney z = −12.79,
  p < 0.0001**; median Δ = −2.018 ms.
* Both 75-sample halves agree (first 75: −1.688 ms, t = −1.66; last 75:
  −1.269 ms, t = −2.15) — the improvement is stable across the session,
  not a single-noisy-half artifact.
* The candidate side carries 2–3 genuinely high-noise runs (max
  171.9 ms vs baseline max 126.4 ms) that inflate its sd and weaken the
  t-statistic; the rank-based test (which is insensitive to that) is the
  stronger signal and is far below any conventional threshold.
* Cross-check: the Phase-B frozen-baseline raw samples
  (`benchmarks/v07b_final_e2e.txt`, 50 samples, mean 111.2 ms) sit
  between the two in-session side means — consistent with the candidate
  being faster than the frozen baseline by ~1.5–2 ms/run.

## 8. KEEP decision (all four acceptance criteria)

1. **EXACT correctness PASS** — §3: 1047 bit-exact checks (all variants,
   B=1/2/3, 32-step chains, alignment float4/fallback, dispatcher vs
   frozen) + E2E golden text generation exact + compute-sanitizer 0
   errors at the experiment, candidate and final SHAs.
2. **DeltaNet weighted microbench clearly improves** — §4: measured
   table winners B=1 vchunk 1.77–1.86x, B=2 vvec 1.32–1.33x, B=3 vvec
   1.31–1.33x, stable across 3 independent runs; mixed-table weighted
   delta-kernel time ≈ −42%.
3. **Batched-only Nsys confirms the DeltaNet family reduction** — §6:
   family 15.36% → 12.77% of GPU kernel time; delta-rule kernels
   58.70 → 44.44 ms (−24.3%); B=1 per-call 24.5 → 18.1 µs; launch count
   reported (9504 → 11556).
4. **Canonical E2E improvement reproducible** — §7: −1.479 ms/run,
   Welch p = 0.012, Mann-Whitney p < 0.0001, both halves negative.

**Decision: KEEP.** The final production state (V07C_FINAL_FUNCTIONAL_SHA)
keeps the measured table; `vreg` is REJECTED (superseded, §4).

## 9. Final-state validation (V07C_FINAL_FUNCTIONAL_SHA 6a9d0b5)

* Full `ctest`: **62/62 passed, 0 failed, 0 skipped** (includes the
  parity test and the exact E2E golden text generation).
* No-torch dependency scan (src/include/tests/benchmarks/tools/CMake):
  **CLEAN** (0 hits).
* `compute-sanitizer --tool memcheck` on the parity test:
  **0 errors** (also covers the vchunk lazy-scratch allocation path).
* Parity test re-run: ALL PASS (1047 checks).
* Final-SHA microbench: `benchmarks/v07c_deltanet_microbench_final.txt`
  (B=1 vchunk 12.587 µs 1.858x; B=2 vvec 15.974 µs 1.329x; B=3 vvec
  16.395 µs 1.315x — consistent with the candidate-SHA runs).
* `git diff 4f4395e..6a9d0b5` is comment-only (no executable-code
  change); the candidate-SHA performance evidence therefore applies to
  the final SHA exactly.
* W4A16 production path, LM head, scheduler, paged KV, sampling,
  tokenizer: untouched (the final functional commit touches
  `src/kernels/deltanet_delta_qwen35.cu` comments only; the candidate
  commit touches the DeltaNet dispatcher .cu/.h and the two
  `src/runtime/qwen35_deltanet.cpp` call sites).

## 10. Secondary (not done — by scope)

gbeta/conv/gated_rmsnorm fusion into the delta preamble was explicitly
secondary and gated on EXACT parity holding first. It was **not**
attempted: the delta-kernel KEEP already delivers the family win
(family 15.36% → 12.77%), the remaining ~3 auxiliary launches per
DeltaNet layer (~5.6 ms over the 5-run nsys window) are a smaller,
higher-risk target, and Phase C's stop criteria are met.

## 11. Evidence inventory

| artifact | location | SHA of generation |
|---|---|---|
| parity test binary/source | `tests/cuda/test_deltanet_delta_qwen35_optimized.cpp` | 387f140 (re-verified at 4f4395e, 6a9d0b5) |
| microbench (candidate, x2) | `benchmarks/v07c_deltanet_microbench_run1/2.txt` | 4f4395e |
| microbench (final) | `benchmarks/v07c_deltanet_microbench_final.txt` | 6a9d0b5 |
| NCU before/after (7 files) | `benchmarks/profiling/v07c_ncu_*.txt` | 4f4395e |
| NCU target tool | `tools/ncu_deltanet_target.cpp` (target `tool_ncu_deltanet_target`) | 387f140 |
| batched-only nsys (candidate) | `benchmarks/profiling/v07c_candidate_batched_nsys.{nsys-rep,sqlite,*_summary*}` | 4f4395e |
| batched-only nsys (baseline) | `benchmarks/profiling/v07b_batched_nsys.*` | 46a5998 (Phase B final) |
| canonical E2E raw (candidate) | `benchmarks/v07c_candidate_e2e.txt` (150 samples) | 4f4395e |
| canonical E2E raw (baseline) | `benchmarks/v07c_baseline_e2e.txt` (150 samples) | 387f140 |

Reproduction: `cmake -B build -DCUDALM_CUDA_HOME=/usr/local/cuda-11.8 &&
cmake --build build -j8`; then `./build/tests/
test_deltanet_delta_qwen35_optimized`,
`./build/benchmarks/bench_deltanet_delta_qwen35 <out.txt>`,
`ncu -k regex:deltanet_delta --launch-skip 8 --launch-count 12 --section
SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis
--section LaunchStats --section WarpStateStats --section SchedulerStats
./build/tool_ncu_deltanet_target <B> <variant> 24`
(B=1/2/3, variant frozen|vreg|vvec|vchunk), and the E2E command in the
raw-sample headers. GPU one job at a time (RTX 2080 Ti, sm_75, CUDA
11.8, driver 570.172.08).
