# CUDALM v0.7 Phase C — DeltaNet delta-rule optimization (Qwen3.5-0.8B)

**Verdict (final):** all candidates **REJECTED for production**. The
frozen delta-rule baseline is **retained**, and the final production state
(V07C_FINAL_FUNCTIONAL_SHA, REJECT) calls the frozen entry points
**directly** — `src/runtime/qwen35_deltanet.cpp` is **byte-identical to
V07C_KERNEL_EXPERIMENT_SHA 387f140** (`git diff 387f140 <final> --
src/runtime/` is empty). The vreg/vvec/vchunk launchers, the
**experimental measured dispatcher**, the bit-exact parity test, the
microbenchmark and the NCU target remain in the tree so the candidate can
be re-benchmarked/reproduced.

The rejection is driven by the **wall-clock** evidence, not the kernel
evidence. The candidates are genuinely faster in isolation (EXACT
bit-exact parity, §4 microbenchmark −22…−24%, §5 NCU −14…−26% kernel
duration, §6 corrected Nsys delta-rule −24% of GPU kernel time), but the
canonical continuous-batched E2E does **not** clearly favor them at the
proper statistical unit. The **pre-specified paired A/B** (20 fixed pairs,
fresh-process invocations, alternating order — §7.3) yields a **95% CI of
the paired mean that INCLUDES 0** ([-2.57, +0.58] ms), so the binding KEEP
criterion is not met. This is the same "REJECTED, baseline retained"
outcome the Phase-B criteria anticipate.

**Two evidence defects in the earlier (superseded) KEEP call**, both fixed
here:

1. **Nsys baseline mis-attribution.** The earlier doc compared the
   candidate against `benchmarks/profiling/v07b_batched_nsys.*` (a
   Phase-B candidate-era profile) and labelled it `46a59984`. The true
   exact-SHA Phase-B final baseline is
   `benchmarks/profiling/v07b_final_batched_nsys.*` /
   `v07b_final_kernel_families.txt` at `46a59984`. §6 recomputes the
   before/after from the true baseline (no Nsys re-run).
2. **Pseudo-replicated E2E.** The earlier doc pooled 15 fresh processes ×
   10 measured runs = n=150 per side and treated all 150 as independent
   (Welch p=0.012, Mann-Whitney p<0.0001). The 10 runs inside one process
   are **not** independent; the independent unit is the **fresh-process
   invocation**. At the correct unit (15 vs 15 invocation means) the
   unpaired Welch p ≈ 0.105 and the 95% CI includes 0 (§7.2). The
   pre-specified paired design (§7.3) is the binding test.

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
SUPERSEDED provisional KEEP            6a9d0b5… (final functional, KEEP) + a244601… (evidence)
                                        — based on the two defective analyses above;
                                        superseded by the paired A/B below, retained in
                                        history only
    ↓
paired A/B E2E -> 95% CI includes 0 -> REJECT
    ↓
production frozen-direct restored      V07C_FINAL_FUNCTIONAL_SHA = 97c0c32e5323a133676fee2bf2740d71ded2b114
                                        (runtime reverted to the frozen direct calls —
                                         byte-identical to 387f140; stale KEEP comments
                                         corrected to "NOT a production entry point")
    ↓
docs/evidence-only commit              (this document + paired A/B raw + corrected-Nsys
                                        artifact; touches docs/ + benchmarks/ +
                                        benchmarks/profiling/ only)
```

Scope: the Phase-A #2 serving GPU hotspot family — the DeltaNet family,
**15.4%** of serving (continuous-batched) GPU kernel time, of which the
delta-rule state-update kernel `deltanet_delta` (+ its batch variant) is
**11.6%** (Phase A §6.3; census: 18 DeltaNet layers × 1 `deltanet_delta`
call per traversal = 2376 calls over the 132-traversal batched census:
2052 B=1 + 324 batch). Only the delta-rule state-update kernel was
optimized (state-parallel decomposition / thread mapping / vectorized
state I/O). Nothing else was touched: the scheduler, state-manager
lifecycle, paged KV, sampling, tokenizer, the W4A16 production path and
the LM-head format are all frozen; the DeltaNet auxiliary kernels
(gbeta/conv/gated_rmsnorm) were NOT fused (fusion was explicitly
secondary; §10). No tolerance-based correctness gate anywhere — the
contract is BIT-IDENTICAL to the frozen kernels.

Frozen contract (enforced, not assumed): every candidate must be
**BIT-IDENTICAL** (all FP32 recurrent-state words, all BF16 `core_out`
words, and all BF16 q/k/v-side outputs) to the frozen v0.6/v0.7A baseline
kernels `deltanet_delta_kernel`
(`src/kernels/qwen35_deltanet_kernels.cu`, B=1) and
`batch_deltanet_delta_kernel` (`src/kernels/batch_decode.cu`, batch).
The per-element math is preserved exactly: the same 128-thread
l2norm/qk reduction tree, the same per-value sequential k-ascending
accumulation `A_t = Σ_k S[h,k,t]·exp_g·k_s[k]` and
`B_t = Σ_k S[h,k,t]·exp_g·q_s[k]` (one thread per value), the same
`d_t = (v_t − A_t)·beta`, the same single BF16 RNE rounding of
`o_t = B_t + d_t·qk`, the same state update
`S[h,k,t] = S[h,k,t]·exp_g + k_s[k]·d_t`, and the same BF16 rounding
boundaries in the prologue (a shared `delta_prologue` helper compiled from
the frozen op sequence).

---

## 1. Baseline characterization (Phase A, frozen SHA a9b5a6e)

From `benchmarks/profiling/v07_ncu_deltanet*.txt` (Phase A) and the
batched census:

| shape | kernel | grid | block | duration | regs | waves | occ | SM% | DRAM% | dominant stall |
|---|---|---|---|---|---|---|---|---|---|---|
| B=1 | `deltanet_delta_kernel` | 16 | 128 | 30.1 µs | 64 | 0.03 | 11.6% | 2.4% | 5.4% | const/scoreboard (latency) |
| B=2 | `batch_deltanet_delta_kernel` | 32 | 128 | 31–32 µs | 62 | 0.06 | 11.7% | — | — | scoreboard ~13 cyc |

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
frozen prologue and per-value op sequence; only the *work mapping*
changes:

* **`vreg`** — 1 block/head, 128 threads, one value per thread; the
  128-deep column is kept across the A/B pass and the update pass so the
  update pass never re-loads the state (46 regs/thread; the column is
  L1-backed local memory — the win is the removed global re-read).
* **`vvec`** — 1 block/head, 128 threads; threads 0..31 each process 4
  values via `float4` state loads (4 independent chains), update pass
  re-reads the L1-hot column. Requires a 16B-aligned state base;
  otherwise the launcher **falls back to the frozen kernel**.
* **`vchunk`** — 2-kernel value-chunk split. Kernel 1 (`red`, grid
  n_heads, 128 threads) runs the prologue and publishes q_s/k_s/qk to
  scratch. Kernel 2 (`s`, grid (n_heads,4), 32 threads) does one value
  per thread with the column kept across passes (50 regs/thread on `s`).
  The split raises the block count (16 → 64 for B=1).

The batch mirrors use the per-row slot state. The dispatcher is a STATIC
measured table keyed on (n_heads[, B]) for the pinned 0.8B config
(n_heads = 16) — no runtime autotuning; any unrecognized legal shape takes
the frozen baseline path.

## 3. EXACT correctness evidence (bit-exact parity)

`tests/cuda/test_deltanet_delta_qwen35_optimized`, run at
V07C_KERNEL_EXPERIMENT_SHA, V07C_CANDIDATE_SHA and the REJECT final SHA —
**ALL PASS (1047 checks)** at each:

* B=1 single-step bit-exact (frozen vs candidate, independent
  state/output buffers): **240 combos × 3 variants**.
* B=1 32-step recurrent state chains: **96 steps × 3 variants**.
* Batch B=2/B=3: core_out + full recurrent state + **row-parity** —
  6 combos × 3 variants × 2 B.
* vvec alignment: +32B state base (16B-aligned → float4) and +4B (16B-
  **un**aligned → frozen fallback), each vs a frozen reference on a
  separate non-overlapping buffer — bit-exact.
* Dispatcher vs frozen bit-exact, B=1 and batch, under both the identity
  and the measured tables.

Comparison is exact word equality (`memcmp`) — no tolerance gate. The E2E
golden text-generation test (exact token IDs) passes at the candidate and
final SHAs.

## 4. DeltaNet weighted microbenchmark (kernel-level, candidate SHA)

`benchmarks/bench_deltanet_delta_qwen35` (per-iteration CUDA-event
timing; 20000 measured iters/cell; weighted score W_Bx = 18 × mean_us(B=x),
overall = (19·W_B1 + 1·W_B2 + 2·W_B3)/22 per the Phase-A census B mix).

Candidate-SHA runs (2 independent runs) — mean µs (speedup vs frozen):

| B | variant | run1 µs (x) | run2 µs (x) |
|---|---|---|---|
| 1 | frozen | 22.714 (1.000) | 22.583 (1.000) |
| 1 | vreg | 14.600 (1.556) | 14.711 (1.535) |
| 1 | vvec | 15.260 (1.488) | 15.385 (1.468) |
| 1 | **vchunk** | **12.671 (1.793)** | **12.733 (1.774)** |
| 2 | frozen | 21.179 (1.000) | 21.188 (1.000) |
| 2 | vreg | 18.609 (1.138) | 18.666 (1.135) |
| 2 | **vvec** | **15.969 (1.326)** | **16.064 (1.319)** |
| 2 | vchunk | 20.477 (1.034) | 20.548 (1.031) |
| 3 | frozen | 21.571 (1.000) | 21.704 (1.000) |
| 3 | vreg | 29.394 (0.734) | 29.681 (0.731) |
| 3 | **vvec** | **16.355 (1.319)** | **16.347 (1.328)** |
| 3 | vchunk | 32.997 (0.654) | 32.433 (0.669) |

Winner per B stable across runs: **B=1 → vchunk, B=2 → vvec, B=3 →
vvec**. `vreg` is REJECTED (superseded: 0.73x at B=3). Weighted
single-variant scores: vchunk −22.5%/−23.8%, vvec −21.9%/−21.8% vs
frozen. This is a real *kernel-duration* improvement — it just does not
translate into a clear wall-clock gain (§7), because the delta kernel is
only ~11.6% of GPU kernel time and the E2E wall is not purely
GPU-busy-bound at this size.

## 5. NCU before/after (candidate SHA, `tools/ncu_deltanet_target`)

12 sampled launches each (8 warmup skipped), deterministic fixture. Median
over samples:

| B | kernel | dur (µs) | grid | block | waves | SM% | occ% | regs | WIPC | top stall |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | frozen | 30.46 | 16 | 128 | 0.030 | 2.44 | 11.46 | 64 | 15.6 | 11.9 cyc L1TEX scoreboard (76%) |
| 1 | vchunk red | 3.39 | 16 | 128 | 0.030 | 1.97 | 12.48 | 19 | 9.6 | — |
| 1 | vchunk s | 19.01 | 64 | 32 | 0.060 | 5.38 | 3.12 | 50 | 8.1 | 5.4 cyc L1TEX scoreboard (67%) |
| 2 | frozen | 32.32 | 32 | 128 | 0.060 | 4.66 | 11.68 | 62 | 17.0 | 13.2 cyc L1TEX scoreboard (77.6%) |
| 2 | vvec | 27.94 | 32 | 128 | 0.060 | 2.41 | 3.93 | 53 | 8.5 | 4.5 cyc L1TEX scoreboard (53.1%) |
| 3 | frozen | 33.31 | 48 | 128 | 0.090 | 6.75 | 11.80 | 62 | 17.6 | 13.7 cyc L1TEX scoreboard (77.7%) |
| 3 | vvec | 28.61 | 48 | 128 | 0.090 | 3.55 | 3.93 | 53 | 8.6 | 4.5 cyc L1TEX scoreboard (53.0%) |

Warp-cycles-per-instruction halved (15.6→8.1, 17.0→8.5, 17.6→8.6); the
L1TEX scoreboard stall dropped (11.9→5.4 / 13.2→4.5 / 13.7→4.5 cyc).
vvec's lower *achieved* occupancy is structural (one of four warps per
block active in the main loop) but the duration still improves. NCU
kernel-time deltas: B=1 30.46 → 22.40 µs (red+s), B=2 32.32 → 27.94,
B=3 33.31 → 28.61. (NCU serializes launches, so its absolute durations
exceed the pipelined microbench numbers; the relative behavior agrees.)

## 6. Batched-only Nsys before/after — CORRECTED baseline

The baseline is the **true exact-SHA Phase-B final** profile
(`benchmarks/profiling/v07b_final_batched_nsys.nsys-rep`, SHA
`46a59984`), not the earlier-misattributed `v07b_batched_nsys.*`. Both
reps cover **132 traversals** (2376 logical delta-rule calls = 6 batched
runs × 22 traversals), so all figures are directly comparable. Full
artifact: `benchmarks/profiling/v07c_nsys_corrected_before_after.txt`.

| metric (batched, 132 traversals) | baseline (46a5998) | candidate (4f4395e) | Δ |
|---|---|---|---|
| total GPU kernel time | 525.986 ms | 505.701 ms | −20.284 ms |
| DeltaNet family total | 81.211 ms (15.44%) | 64.579 ms (12.77%) | **−16.631 ms (−20.5%)** |
| delta-rule kernels | 60.801 ms (11.56%) | 44.432 ms (8.79%) | **−16.369 ms (−26.9%)** |
| delta-rule per logical call | 25.59 µs ×2376 | 18.70 µs ×2376 | −6.89 µs |
| B=1 delta (2052 calls) | 52.299 ms = 25.49 µs/call | vchunk s+red 37.197 ms = 18.13 µs/call | −28.9% |
| batch delta (324 calls) | 8.503 ms = 26.24 µs/call | vvec 7.236 ms = 22.33 µs/call | −14.9% |
| DeltaNet family launch count | 9828 | 11880 | +2052 (vchunk red pass) |
| delta-rule launch count | 2376 | 4428 | +2052 (B=1 split into 2) |

This is a real **kernel-duration** reduction in the DeltaNet family. It is
**not** a wall-clock guarantee: the family is ~15% of GPU kernel time,
the extra vchunk red launches add host-side cost, and the E2E wall is not
purely GPU-busy-bound. The wall-clock question is decided in §7.

## 7. Canonical E2E — corrected analysis

### 7.1 The earlier 150-run analysis was invalid (pseudo-replication)

The superseded doc pooled 15 fresh processes × 10 measured runs = n=150
per side and reported Welch p=0.012 / Mann-Whitney p<0.0001. The 10 runs
inside one process share the same warm GPU, clock state and host context —
they are **correlated**, not independent. Treating them as independent
inflated the effective n by ~10× and overstated significance. This
analysis is **withdrawn**.

### 7.2 Invocation-level (correct unit): 15 vs 15, unpaired

Taking the **fresh-process invocation** as the independent unit (one value
per invocation = mean of its 10 measured runs), from
`benchmarks/v07c_baseline_e2e.txt` and `benchmarks/v07c_candidate_e2e.txt`
(15 invocations each, `--mode batched`, canonical workload):

| side | n (invocations) | mean (ms) | sd (ms) |
|---|---|---|---|
| baseline (frozen) | 15 | 111.753 | 1.482 |
| candidate (measured table) | 15 | 110.275 | 3.036 |

Δ = **−1.479 ms/run**; Welch two-sided **t = −1.695, df = 20.31,
p ≈ 0.105**; 95% CI of the difference ≈ [−3.20, +0.24] ms — **includes
0**. The candidate side carries a few genuinely high-noise invocations
(max 116.6 ms) that inflate its sd. Not significant.

### 7.3 Pre-specified PAIRED A/B (the binding test)

Two clean worktrees/builds — baseline at `387f140` (frozen direct calls)
and candidate at `4f4395e` (measured dispatcher); the only functional
difference is the DeltaNet delta-rule path (`git diff 387f140 4f4395e`
touches 3 DeltaNet files). **20 fixed pairs** (no early stop),
alternating order to counterbalance session drift (odd pair: baseline →
candidate; even pair: candidate → baseline). Each invocation: fresh
process, 2 warmup + 10 measured, `--mode batched`, canonical workload.
One statistical value per invocation = the mean of its 10 measured runs;
paired difference = candidate_mean − baseline_mean. Raw data + both
binary SHAs/hashes: `benchmarks/v07c_paired_ab_raw.txt`.

| pair | order | base (ms) | cand (ms) | Δ (ms) |
|---|---|---|---|---|
| 1 | base→cand | 110.213 | 107.913 | −2.299 |
| 2 | cand→base | 109.666 | 109.086 | −0.581 |
| 3 | base→cand | 109.704 | 110.739 | +1.035 |
| 4 | cand→base | 109.653 | 108.101 | −1.552 |
| 5 | base→cand | 111.328 | 107.814 | −3.514 |
| 6 | cand→base | 113.528 | 107.621 | −5.907 |
| 7 | base→cand | 116.092 | 111.493 | −4.599 |
| 8 | cand→base | 109.751 | 107.730 | −2.021 |
| 9 | base→cand | 109.985 | 111.684 | +1.699 |
| 10 | cand→base | 110.585 | 108.023 | −2.561 |
| 11 | base→cand | 110.494 | 108.054 | −2.440 |
| 12 | cand→base | 110.620 | 110.550 | −0.070 |
| 13 | base→cand | 110.792 | 108.810 | −1.982 |
| 14 | cand→base | 110.283 | 112.128 | +1.844 |
| 15 | base→cand | 110.464 | 113.013 | +2.549 |
| 16 | cand→base | 110.946 | 108.344 | −2.602 |
| 17 | base→cand | 110.088 | 119.550 | +9.462 |
| 18 | cand→base | 109.692 | 110.304 | +0.612 |
| 19 | base→cand | 112.459 | 107.531 | −4.928 |
| 20 | cand→base | 111.377 | 109.363 | −2.014 |

**Primary inference (n=20 pairs):**

* paired mean delta = **−0.994 ms/run** (SE 0.754)
* paired median delta = **−1.998 ms/run**
* paired t = **−1.318, df = 19, two-sided p = 0.203**
* **95% CI of the paired mean = [−2.572, +0.585] ms — includes 0**

**Supplemental:** Wilcoxon signed-rank W+ = 76, W− = 134, z = −1.505,
two-sided **p = 0.132**. 14/20 pairs have Δ < 0.

The candidate is faster in the majority of pairs and in the median, but
the improvement is not statistically distinguishable from zero: the 95% CI
of the paired mean spans 0, the paired t is far from significant, and the
Wilcoxon is not significant. A few high-noise candidate invocations
(pairs 15, 17, 14, 9) are the main reason the mean is pulled toward zero,
but the pre-specified decision rule does not allow cherry-picking them.

## 8. REJECT decision

Binding KEEP criterion: **mean delta < 0 AND median delta < 0 AND the 95%
CI of the paired mean excludes 0.**

| condition | value | met? |
|---|---|---|
| paired mean delta < 0 | −0.994 ms | yes |
| paired median delta < 0 | −1.998 ms | yes |
| 95% CI of paired mean excludes 0 | [−2.57, +0.58] | **NO (includes 0)** |

**Decision: REJECT.** The kernel-level evidence (EXACT parity, §4
microbenchmark, §5 NCU, §6 corrected Nsys) is real but insufficient: the
wall-clock E2E — the metric that actually matters for serving — does not
clearly favor the candidate. The production runtime is restored to the
frozen delta-rule direct path. No correctness gate was changed to buy
performance; the frozen path is the verified baseline.

## 9. Final-state validation (V07C_FINAL_FUNCTIONAL_SHA 97c0c32, REJECT)

* `src/runtime/qwen35_deltanet.cpp` is **byte-identical to 387f140**
  (frozen direct calls); the stale KEEP comments in the dispatcher .h/.cu
  are corrected to state the dispatcher is **NOT a production entry
  point**.
* Full `ctest`: **62/62 passed, 0 failed, 0 skipped** (includes the parity
  test and the exact E2E golden text generation on the frozen path).
* No-torch dependency scan (src/include/tests/benchmarks/tools/CMake):
  **CLEAN** (0 hits).
* `compute-sanitizer --tool memcheck` on the parity test: **0 errors**.
* Parity test re-run: ALL PASS (1047 checks).
* W4A16 production path, LM head, scheduler, paged KV, sampling,
  tokenizer: untouched. The REJECT final commit touches only
  `src/runtime/qwen35_deltanet.cpp` (revert) + the two DeltaNet
  comment blocks.

## 10. Secondary (not done — by scope)

gbeta/conv/gated_rmsnorm fusion into the delta preamble was explicitly
secondary and was **not** attempted: the delta-rule candidate was
REJECTED, and fusion on top of a rejected delta path is out of scope.

## 11. Evidence inventory

| artifact | location | SHA of generation |
|---|---|---|
| parity test | `tests/cuda/test_deltanet_delta_qwen35_optimized.cpp` | 387f140 (re-verified at 4f4395e, 97c0c32) |
| microbench (candidate ×2) | `benchmarks/v07c_deltanet_microbench_run1/2.txt` | 4f4395e |
| microbench (final) | `benchmarks/v07c_deltanet_microbench_final.txt` | 6a9d0b5 (superseded) |
| NCU before/after (7) | `benchmarks/profiling/v07c_ncu_*.txt` | 4f4395e |
| NCU target tool | `tools/ncu_deltanet_target.cpp` | 387f140 |
| batched-only nsys (candidate) | `benchmarks/profiling/v07c_candidate_batched_nsys.*` | 4f4395e |
| batched-only nsys (true baseline) | `benchmarks/profiling/v07b_final_batched_nsys.*`, `v07b_final_kernel_families.txt` | 46a5998 (Phase B final) |
| **corrected Nsys before/after** | `benchmarks/profiling/v07c_nsys_corrected_before_after.txt` | this commit |
| E2E raw (candidate, 15 invocations) | `benchmarks/v07c_candidate_e2e.txt` | 4f4395e |
| E2E raw (baseline, 15 invocations) | `benchmarks/v07c_baseline_e2e.txt` | 387f140 |
| **paired A/B raw (20 pairs)** | `benchmarks/v07c_paired_ab_raw.txt` | baseline 387f140 / candidate 4f4395e (binary hashes recorded) |

Reproduction: see the raw-sample headers for the exact E2E command and the
paired runner for the worktree/SHA/binary-hash record. GPU one job at a
time (RTX 2080 Ti, sm_75, CUDA 11.8, driver 570.172.08).
