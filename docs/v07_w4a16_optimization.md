# CUDALM v0.7 Phase B — W4A16 GEMV optimization (Qwen3.5-0.8B)

**Verdict (final):** all candidates **REJECTED for production**. The
frozen R4 row-tile baseline is **retained**, and the final production
state calls it **directly** — the Qwen3.5 runtime sources are
byte-identical to Phase A (`git diff a9b5a6e -- src/runtime/` is empty).
The R1/R2/R8 variants, the explicit variant launchers and the
**experimental / measured dispatcher** (renamed from the earlier
"production dispatcher"; benchmark/test/profiling infrastructure only,
NOT a production entry point) remain in the tree so the candidate can be
re-benchmarked/reproduced. The rejection reason is a failed E2E
acceptance criterion: at the exact candidate SHA the W4A16 kernel family
drops ≈ −4.9% drift-corrected, but the canonical continuous-batched E2E
improvement (50 vs 50 independent measured runs, Welch two-sided
p = 0.284) is **indistinguishable from noise** (§8, §9). This is the
"REJECTED, baseline retained" outcome the Phase-B criteria anticipate.

**SHA chain (this document):**

```
Phase A baseline (frozen)            a9b5a6e…
    ↓
R1/R2/R8 kernel experiment           V07B_KERNEL_EXPERIMENT_SHA = d9c2f85e7e000c1fb60b7c91e650d78c72bef285
    ↓                                (variants + launchers + parity test + microbench + NCU + census)
measured candidate table             V07B_CANDIDATE_SHA = 427861033481b121419ca368913395b40d609457
    ↓                                (runtime enters the dispatcher with the measured table)
exact-SHA candidate Nsys/E2E         (benchmarks/profiling/v07b_candidate_*, benchmarks/v07b_candidate_e2e.txt)
    ↓
E2E within noise → REJECT
    ↓
production direct-R4 restored        V07B_FINAL_FUNCTIONAL_SHA = 46a59984a70a42b8baedf017d4bdcaa14edc91e1
                                     (runtime reverted to the frozen direct calls; dispatcher
                                      renamed experimental; alignment-fallback test fixed)
    ↓
final baseline-retained evidence     (benchmarks/profiling/v07b_final_*, benchmarks/v07b_final_e2e.txt,
                                      microbench run1/run2, sanitizer — this commit)
```

Scope: the Phase-A #1 serving GPU hotspot — W4A16 / INT4 GEMV, **42.72%**
of serving (continuous-batched) GPU kernel time (B=1 33.82% + B>1 8.90%,
Phase A §6.1). Nothing else was touched (no DeltaNet, no norm/elementwise
fusion, no CUDA Graph, no LM-head changes, no paged-attention/scheduler/
state-manager changes, no new quant format, no prefill, no multi-stream,
no Tensor Core, no tolerance-based correctness). No new CUDA optimization
was added in the evidence/production-freeze follow-up round — only the
candidate SHA, the exact-SHA evidence, the production revert and the
test fix below.

Frozen contract (enforced, not assumed): the optimized path must be
**BIT-IDENTICAL** (BF16 output words) to the frozen v0.6/v0.7A baseline
kernels `int4gemv_rowtile4_bf16_kernel` (`src/kernels/int4_gemv_bf16.cu`)
and `batch_int4gemv_rowtile4_bf16_kernel` (`src/kernels/batch_decode.cu`)
— no tolerance gate anywhere in the Phase-B verification.

---

## 1. Authoritative shape census

Source: `tools/dump_w4a16_shape_census.cpp` (run at the exact SHA; output
`benchmarks/profiling/v07b_w4a16_shape_census.txt`). Built from the actual
model config (the `.cudalm` WeightFileV2 file) + the actual call sites
(`src/runtime/qwen35_full_attention.cpp`, `src/runtime/qwen35_deltanet.cpp`,
shapes pinned by `weight_loader_v2.cpp::validate_layer`), and reconciled
against the Phase-A batched-only nsys CSV (fail-loud on mismatch).

| N | K | grid (R4) | calls/trav | roles |
|---|---|---|---|---|
| 16 | 1024 | 4 | 36 | linear_attn.in_proj_b + in_proj_a |
| 512 | 1024 | 128 | 12 | self_attn.k_proj + v_proj |
| 1024 | 2048 | 256 | 24 | self_attn.o_proj + linear_attn.out_proj (grid-256 group, 48 calls/trav total with ↓) |
| 1024 | 3584 | 256 | 24 | mlp.down_proj |
| 2048 | 1024 | 512 | 18 | linear_attn.in_proj_z |
| 3584 | 1024 | 896 | 48 | mlp.gate_proj + up_proj |
| 4096 | 1024 | 1024 | 6 | self_attn.q_proj (fused [q;gate]) |
| 6144 | 1024 | 1536 | 18 | linear_attn.in_proj_qkv |

**186 W4A16 calls per model traversal**; 8 unique (N,K). Reconciliation
with the Phase-A CSV is exact: implied traversals B1=114, B2=6, B3=12
(sum 132 = 6 runs × 22 traversals; canonical B distribution 19:1:2 per
run). No NCU-grid guessing was used.

Two structural facts that drive the candidate design:

* **K=1024 for 7 of 8 shapes** → `nvec = K/32 = 32` uint4 fragments, so
  in the frozen 128-thread block **only threads 0..31 (one warp) ever
  execute the K-loop**; the other 3 warps are idle until the cross-warp
  reduction (NCU confirms: "Avg. Active Threads Per Warp ≈ 30.7" on every
  K=1024 shape). Every block does the work of 1 warp + a barrier/reduction
  chain.
* **N=16 → grid 4** (degenerate: 4 blocks on 68 SMs, occupancy 11.3%,
  waves 0.01) and **N=512 → grid 128** (occupancy 21.2%, waves 0.24) are
  the two parallelism-starved shapes; N≥1024 shapes already run ≥2.8
  waves at ~83% occupancy.

## 2. Candidate design: adaptive N-dimension row tiling

The ONLY degree of freedom changed vs the frozen baseline is **rows per
block R** (128-thread block, `ceil(N/R)` grid; batch: `(ceil(N/R), B)`):

* `threadIdx.x → v` mapping: unchanged (`v = tid; v < nvec; v += 128`);
* per-row K-loop order: unchanged (v ascending, in-vector k ascending,
  lo nibble before hi);
* warp-reduction order: unchanged (per-acc `shfl_down` 16→8→4→2→1,
  `__shared__ warp_sums[R][4]`, warp0 2-step);
* final reduction + single `__float2bfloat16_rn` store: unchanged.

Therefore the per-row term sequence and both reduction trees are
**identical to the frozen R=4 kernel for every R ∈ {1,2,8}** → the output
is bit-identical **by construction** (independently verified by the
parity test, §4). This is the CUDALab frozen-reference (`cb6a6a9`,
`int4gemv_rowtile8_hx.cu`, experiment INT4GEMV-0003) row-tile technique —
borrowed, with the dispatcher and the batch path redesigned for CUDALM
(CUDALab tuned 4096²-style shapes; its winners do NOT transfer
automatically and are re-measured here).

Explicitly NOT done (per reviewer): no split-K, no K-partition change,
no reduction-tree change, no atomics, no blockDim change, no weight
cooperative scheme.

## 3. Implementation (kept in tree)

* `src/kernels/int4_gemv_qwen35.cu` +
  `include/cudalm/kernels/int4_gemv_qwen35.h`: one template
  `int4gemv_rowtile_bf16_kernel<R>` serving B=1 (1-D grid) and batch
  (2-D grid, offsets from `blockIdx.y`) — math term-for-term the frozen
  kernel; `int4_gemv_bf16_rowtile{1,2,8}` /
  `batch_int4_gemv_bf16_rowtile{1,2,8}` explicit launchers (with the
  vectorized-contract fallback to the frozen entry); and the
  **experimental / measured dispatcher**
  `int4_gemv_bf16_qwen35_experimental` /
  `batch_int4_gemv_bf16_qwen35_experimental` — static, keyed
  (N,K[,B]), no runtime autotuning, frozen-R4 bit-compatible fallback
  for unknown legal shapes (incl. the scalar fallback for
  non-16B-aligned inputs). **The production runtime does not call it.**
* The frozen baselines are **untouched and directly callable** for paired
  A/B (only behavior-neutral `*_rowtile4_regs()` query helpers were added
  to the two frozen TUs).
* **Production call sites (30):** at `V07B_FINAL_FUNCTIONAL_SHA` they are
  the Phase-A frozen direct calls `int4_gemv_bf16(...)` /
  `kernels::batch_int4_gemv_bf16(...)` again; `src/runtime/` is
  byte-identical to Phase A HEAD `a9b5a6e` (verified). At
  `V07B_CANDIDATE_SHA` the same 30 call sites entered the experimental
  dispatcher with the measured table, solely for the full-model
  candidate benchmark.
* `tools/ncu_w4a16_target.cpp`: controlled single-(N,K,B,variant) NCU
  harness (no model, no interleaved kernels).
* `benchmarks/bench_w4a16_qwen35_shapes.cpp`: dedicated microbench —
  census shapes × B=1/2/3 × {frozen R4, R1, R2, R8} (explicit variant
  pointers, NOT via the dispatcher); CUDA events, warmup ≥30, measured
  iters scaled to kernel duration (200–2000; reported per cell);
  weighted production score = census call rates × canonical B
  distribution (19:1:2).
* `benchmarks/bench_qwen35_continuous_batching.cpp`: tooling-only
  `--warmup-runs N` (default 1 = byte-identical Phase-A behavior; the
  workload is unchanged).
* `scripts/profile_v07b_w4a16.sh`: exact-SHA evidence pipeline (census /
  microbench / NCU / E2E / batched-only nsys / aggregation); refuses
  tracked drift at HEAD; evidence tagging
  (`CUDALM_EVIDENCE_TAG`/`CUDALM_EVIDENCE_SHA_TAG`) produces
  independent `v07b_candidate_*` / `v07b_final_*` artifacts with the
  binding SHA in each header.

## 4. EXACT parity gate (bit-exactness)

`tests/cuda/test_w4a16_qwen35_optimized` — **9808 bit-exact checks
(memcmp on BF16 words), 0 mismatches** at `V07B_FINAL_FUNCTIONAL_SHA`
(9804 at `V07B_CANDIDATE_SHA`; +4 from the alignment-fallback fix below):

* shapes: all 8 census (N,K) + 6 edges (1×128, 3×128, 5×256, 9×1024,
  15×1024, 1023×1024);
* variants R1/R2/R8 (B=1 and batch) vs the frozen baseline; experimental
  dispatcher vs frozen (B=1 and batch);
* seeds: 4 × input classes: 5 (deterministic random, all-zero, small
  magnitude, mixed sign, boundary INT4 nibbles {0x7,0x9,0x8,0x0,0xF} +
  zero-group scales 1/5);
* frozen contracts re-gated: batch row-parity (frozen batch row b ==
  frozen single on row b), K%128/contract fallback, legal-shape guard,
  and the alignment fallback — **fixed in the final functional commit**:
  `check_alignment_fallback` previously wrote the R1 variant and the
  dispatcher into the SAME output buffer and made one comparison, so the
  R1 misaligned fallback was not actually verified. It now uses
  independent output buffers and two independent comparisons (frozen vs
  `int4_gemv_bf16_rowtile1`; frozen vs
  `int4_gemv_bf16_qwen35_experimental`) on a 16B-misaligned activation
  (+8-byte offset). No batch-misalignment claim is made (the contract is
  on the activation base; batch rows are 16B-aligned by construction).

Full ctest at both exact SHAs: **61/61 passed, 0 failed, 0 skipped**
(candidate SHA: 548.27 s; final SHA: 537.79 s; including all existing
single/batched/continuous/state-isolation/scheduler tests). The
real-checkpoint FULL-logits / token-ID / hybrid-state EXACT gates are the
existing suite's (unchanged oracle; the production path IS the frozen
baseline, so they hold by construction and are re-run at both SHAs).

## 5. Microbenchmark (kernel level)

Two independent exact-SHA runs at `V07B_FINAL_FUNCTIONAL_SHA`
(the variant code is identical to `V07B_KERNEL_EXPERIMENT_SHA`):
`benchmarks/v07b_w4a16_microbench_run1.txt` / `_run2.txt` (per-shape
mean/median/min/p90 µs, µs/token, speedup, regs). A third raw run at
`V07B_KERNEL_EXPERIMENT_SHA` is `benchmarks/v07b_w4a16_microbench.txt`.
Per-cell values vary between sessions with GPU clock/thermal state (the
cross-session-stable numbers are the weighted totals below and the NCU
durations); rankings do not flip except at N=512 B=1, where R2 stays the
winner in every committed run (1.15×–1.47× across runs).

Per-shape geo-mean speedup over B=1/2/3 (from the two committed runs):

| shape (N,K) | R1 run1 / run2 | R2 run1 / run2 | R8 run1 / run2 |
|---|---|---|---|
| 16,1024 | **1.40 / 1.33** | 1.33 / 1.24 | 0.76 / 0.72 |
| 512,1024 | 1.02 / 1.03 | **1.04 / 1.06** | 0.75 / 0.76 |
| 1024,2048 | 0.86 / 0.86 | 0.97 / 0.98 | 0.88 / 0.87 |
| 1024,3584 | 0.84 / 0.84 | 0.99 / 0.99 | 0.89 / 0.90 |
| 2048,1024 | 0.84 / 0.84 | 0.94 / 0.94 | 0.90 / 0.90 |
| 3584,1024 | 0.84 / 0.83 | 0.95 / 0.95 | 0.97 / 0.97 |
| 4096,1024 | 0.80 / 0.80 | 0.93 / 0.93 | 0.99 / 0.99 |
| 6144,1024 | 0.78 / 0.78 | 0.91 / 0.91 | 0.92 / 0.92 |

(regs: frozen 56/58 (B=1/batch), R1 47, R2 53, R8 63.)

Table-selected cells (the shapes the measured table changes):

| cell | frozen run1/run2 (µs) | selected | run1 speedup | run2 speedup |
|---|---|---|---|---|
| N=16 B=1 | 7.27 / 5.68 | R1 | 1.42× | 1.32× |
| N=16 B=2 | 7.14 / 5.84 | R1 | 1.40× | 1.32× |
| N=16 B=3 | 7.13 / 5.60 | R1 | 1.38× | 1.33× |
| N=512 B=1 | 6.90 / 6.35 | R2 | 1.15× | 1.17× |
| N=512 B=2 | 7.09 / 6.39 | R2 | 1.09× | 1.08× |
| N=512 B=3 | 7.06 / 6.80 | frozen (R2 = 0.90× / 0.95×) | — | — |

**Weighted production score** (census call rates × B distribution
0.864:0.045:0.091; `W_Bx = Σ calls/trav × mean(shape,B)`, overall =
(19·W_B1 + W_B2 + 2·W_B3)/22):

| variant | run1 overall (µs/trav) | vs frozen run1 | run2 overall | vs frozen run2 |
|---|---|---|---|---|
| frozen R4 (all shapes) | 1756.088 | — | 1662.740 | — |
| R1 everywhere | 1909.388 | +8.73% (worse) | 1851.218 | +11.34% (worse) |
| R2 everywhere | 1750.549 | −0.32% | 1679.515 | +1.01% |
| R8 everywhere | 2017.406 | +14.88% (worse) | 1903.335 | +14.47% (worse) |
| **measured table** (N=16→R1; N=512 B=1/2→R2; else frozen) | **1669.857** | **−4.91%** | **1602.672** | **−3.61%** |

measured-table by batch class (run1 / run2): B=1 −5.43% / −4.01%,
B=2 −3.65% / −2.63%, B=3 −2.54% / −1.87%. The measured-table row is
recomputed from the raw per-cell data of each committed run (table
selection applied per shape/B). The reviewer's hypothesis is confirmed
shape-by-shape: R=1 wins only at the degenerate small-N shape, R=2 only
at N=512 (B=1, B=2), and the frozen R4 remains best at every N≥1024 for
all B. "R=1 fastest" / "R=2 best" were **not** assumed — measured.

## 6. NCU before/after (why it is faster — and why it is not at large N)

`benchmarks/profiling/v07b_ncu_*.txt` (NCU 2022.3, controlled single-
shape launches; 12 measured launches each, 8 skipped warmups; generated
at `V07B_KERNEL_EXPERIMENT_SHA`, where the variant code is identical):

| target | grid | regs | waves | occ% | SM% | DRAM% | GB/s | L2 hit% | duration |
|---|---|---|---|---|---|---|---|---|---|
| N=16 B=1 frozen | 4 | 56 | 0.01 | 11.27 | 0.28 | 0.30 | 1.87 | 72.2 | 5.70 µs |
| N=16 B=1 R1 | 16 | 47 | 0.03 | 11.58 | 0.69 | 0.57 | 3.49 | 87.5 | **3.04 µs (−47%)** |
| N=16 B=3 frozen | 12 | 58 | 0.02 | 11.25 | 0.85 | 0.42 | 2.58 | 80.4 | 5.70 µs |
| N=16 B=3 R1 | 48 | 47 | 0.09 | 11.66 | 2.01 | 0.77 | 4.65 | 87.6 | **3.17 µs (−44%)** |
| N=512 B=1 frozen | 128 | 56 | 0.24 | 21.22 | 7.73 | 6.68 | 41.1 | 42.7 | 6.62 µs |
| N=512 B=1 R2 | 256 | 53 | 0.47 | 42.47 | 11.30 | 8.73 | 53.6 | 42.8 | **5.09 µs (−23%)** |
| N=512 B=2 frozen | 256 | 58 | 0.47 | 41.85 | 15.25 | 6.57 | 41.1 | 70.0 | 6.69 µs |
| N=512 B=2 R2 | 512 | 53 | 0.94 | 80.20 | 20.07 | 7.85 | 48.2 | 70.1 | **5.70 µs (−15%)** |
| N=6144 B=1 frozen | 1536 | 56 | 2.82 | 82.61 | 34.62 | 29.68 | 184.8 | 12.3 | 17.57 µs |
| N=6144 B=3 frozen | 4608 | 58 | 8.47 | 83.89 | 44.30 | 12.66 | 78.7 | 70.4 | 41.28 µs |
| N=6144 B=3 R1 | 18432 | 47 | 33.88 | 78.83 | 41.14 | 9.46 | 59.8 | 70.9 | 54.40 µs (**+32%**) |

**Why the small/medium-N winners are faster** — parallelism/occupancy,
exactly the Phase-A hypothesis: with K=1024 only 1 of 4 warps per block
is active (30.7/32 active threads), so wall time ≈ fixed latency chain ×
(1 / active-warp parallelism). N=16: 4 blocks → 16 blocks
(waves 0.01→0.03, SM 0.28→0.69%, 5.70→3.04 µs); N=512: 128→256 blocks
(waves 0.24→0.47, occ 21→42%, 6.62→5.09 µs). Registers also drop
(56→47/53) because acc[R] shrinks. The dominant stall is the CTA barrier
in all variants (58.9% of cycles at N=16 frozen; 41.9% at R1) — the
bottleneck *structure* is unchanged (consistent with Phase A: it does
not change with B either); more blocks simply give the GPU more
independent latency chains to overlap.

**Why large-N stays frozen** — N≥1024 shapes already run ≥2.8 waves at
~83% occupancy (SM- and DRAM-leaning, 30% DRAM at N=6144 B=1); there is
no parallelism headroom. R1 multiplies blocks 4× (18432 at N=6144 B=3)
with duplicated x-fragment reloads and per-block barrier overhead:
occupancy *drops* (78.8%), DRAM% drops (12.7→9.5%), duration **worsens
+32%**. R8 (63 regs, acc[8]) loses on every batch shape and on 6144 —
register pressure, not tile size, is the limit there.

**B>1 cache behavior (reviewer §6 question):** the row-tile change
leaves the per-block weight access pattern untouched (each (row-block,
b) block still issues its own logical weight loads; cross-batch-row
sharing is L2-only, as Phase A established). Measured: batch N=16
L2-hit 80.4→87.6% (R1 *strengthens* it — more blocks share the same
8 KB weight); N=512 B=2 70.04→70.06%; N=6144 B=3 70.37→70.86% —
**never destroyed**. The cross-b DRAM-per-token gain (DRAM% 29.7→12.7
for B=1→B=3 at N=6144) is preserved.

## 7. Batched-only nsys (real serving path) — candidate vs final

All three nsys runs use the same schedule (1 warmup + 5 measured, 132
traversals, `--mode batched`); every one is a committed exact-SHA
artifact:

| state | SHA | W4A16 ms | W4A16 % | total ms | launches |
|---|---|---|---|---|---|
| Phase A (frozen baseline) | `a9b5a6e` | 220.581 | 42.72% | 516.364 | 58422 |
| **candidate** (measured table in runtime) | `V07B_CANDIDATE_SHA` | **213.796** | 41.53% | 514.781 | 58422 |
| **final** (production reverted, direct frozen R4) | `V07B_FINAL_FUNCTIONAL_SHA` | **225.292** | 42.83% | 525.986 | 58422 |

Candidate per-kernel rows (the measured kernels really execute — the
frozen N=16/N=512-B1/B2 rows disappear and reappear as R1/R2 grids):

| kernel | grid | Phase A avg (ns) | candidate avg (ns) | Δ |
|---|---|---|---|---|
| rowtile4 | 4,1,1 | 4859.4 | — (moved) | |
| rowtile_bf16 (R1) | 16,1,1 | — | 2892.0 (4104×) | **−40.5%** (19.94→11.87 ms) |
| rowtile4 | 4,2,1 | 4826.4 | — (moved) | |
| rowtile_bf16 (R1) | 16,2,1 | — | 2795.4 (216×) | **−42.1%** (1.04→0.60 ms) |
| rowtile4 | 4,3,1 | 4791.6 | — (moved) | |
| rowtile_bf16 (R1) | 16,3,1 | — | 2879.5 (432×) | **−39.9%** (2.07→1.24 ms) |
| rowtile4 | 128,1,1 | 5482.3 | — (moved) | |
| rowtile_bf16 (R2) | 256,1,1 | — | 4433.6 (1368×) | **−19.1%** (7.50→6.07 ms) |
| rowtile4 (batch) | 128,2,1 | 5566.3 | — (moved) | |
| rowtile_bf16 (R2) | 256,2,1 | — | 4670.6 (72×) | **−16.1%** (0.40→0.34 ms) |

Drift control (this candidate run): the UNCHANGED shapes (same frozen
kernel, same grid) run +2.14% (+4.05 ms on 189.6 ms) at B=1 and ≈0% at
B=2/B=3 — the GPU was running slightly slower in this session. The
observed family delta (−6.786 ms, −3.08%) minus the B=1 drift on the
unchanged cells gives a drift-corrected W4A16 reduction of
**≈ −10.85 ms (−4.9%)** — consistent with the microbench weighted
(−4.91%/−3.61%) and the NCU per-target numbers (−47%/−44%/−23%/−15% on
the changed cells).

**Final state proof:** the final nsys contains **only** frozen
`int4gemv_rowtile4`/`batch_int4gemv_rowtile4` W4A16 kernels (verified:
zero non-rowtile4 int4gemv kernel rows); W4A16 225.292 ms / 42.83% vs
Phase A 220.581 / 42.72% (+2.1% = session drift, i.e. this baseline run is
slower than the Phase-A session — no improvement claimed), launches 58422
identical.
Chain proven: candidate measured → candidate rejected → production
reverted.

## 8. End-to-end (canonical continuous-batched) — exact-SHA raw samples

Both sides are committed raw-sample artifacts, same canonical workload
(4 requests; 27 logical tokens/run; 22 traversals/run), same schedule:
**5 independent invocations × (2 warmup + 10 measured), `--mode
batched`** → n = 50 measured per-run samples per side (each sample =
steady_clock wall of one measured run, bracketed by
`cudaStreamSynchronize`):

* candidate: `benchmarks/v07b_candidate_e2e.txt`
  (built at `V07B_CANDIDATE_SHA`)
* baseline (final production, direct frozen R4):
  `benchmarks/v07b_final_e2e.txt` (built at
  `V07B_FINAL_FUNCTIONAL_SHA`)

Pooled per-run analysis (from the committed raw samples):

| side | n | mean (ms) | median (ms) | sd (ms) | min (ms) | max (ms) |
|---|---|---|---|---|---|---|
| baseline (final, direct R4) | 50 | 110.654 | 109.870 | 2.278 | 108.191 | 118.953 |
| candidate (measured table) | 50 | 109.853 | 108.127 | 4.734 | 106.283 | 136.363 |
| Δ (candidate − baseline) | | **−0.802 (−0.72%)** | **−1.743** | | | |

Welch two-sample t (pooled sample variances, Welch–Satterthwaite df,
two-sided p from the t distribution):

* **t = −1.079, df = 70.5, p = 0.284 → indistinguishable from zero.**

Interpretation: the GPU-side savings are real and accounted for (W4A16
family −4.9% drift-corrected ≈ ≈1.9 ms per 22-traversal run, §7; Phase
A: GPU kernels are ~79% of the 109 ms wall, the rest is CPU/launch/sync
gap), but the canonical workload is short (22 traversals/run) and its
per-run wall noise (sd 2.3–4.7 ms, with within-invocation thermal drift
of a few ms; the candidate side shows one 136 ms outlier run) exceeds the
~0.7%-of-wall effect. **Conclusion: the E2E improvement is
indistinguishable from noise** → the E2E acceptance criterion fails →
REJECT (§9). (Earlier dev-round A/B tables are deliberately NOT
reported here: they came from uncommitted builds and their pooled t
value had been miscomputed; only the committed raw samples above are
used.)

## 9. KEEP / REJECT table and production decision

| candidate | shapes | microbench (weighted, run1/run2) | nsys (candidate, committed) | E2E | decision |
|---|---|---|---|---|---|
| R1 (all shapes) | — | +8.73% / +11.34% (worse) | — | — | **REJECT** (loses on 6/8 shapes) |
| R2 (all shapes) | — | −0.32% / +1.01% | — | — | **REJECT** (≈ neutral weighted; loses on 4/8, flat on 2) |
| R8 (any) | any | +14.88% / +14.47% (worse) | — | — | **REJECT** (register pressure; loses on every shape) |
| measured table (R1 @ N=16; R2 @ N=512 B=1/B=2) | 60+24 calls/trav | **−4.91% / −3.61%** | family −3.08% raw / **−4.9% drift-corrected** | pooled n=50/50, **p = 0.284** | **REJECTED for production** (E2E within noise) |
| B-tiled weight-reuse (same row, multiple b, W fragment shared in registers) | N≥1024 batch | not implemented | — | — | **REJECT without implementation** (reviewer: complexity high / gain unclear, don't dig): NCU shows the batch kernels at large N are SM-leaning (44% SM / 12.7% DRAM, not DRAM-bound) and already enjoy cross-b L2 weight reuse (70% L2 hit); expected gain bounded, register cost high (acc[R][B]) |

**Decision:** the candidate is **REJECTED for production**.
Acceptance-criterion mapping:

1. all EXACT gates PASS — yes (9808 checks, 0 mismatches; ctest 61/61 at
   both SHAs);
2. weighted W4A16 clearly improves — yes (−4.91%/−3.61% microbench;
   −4.9% nsys drift-corrected);
3. **canonical continuous-batched E2E median improvement reproducible —
   NO** (pooled n=50/50, Δmedian −1.743 ms, Welch p = 0.284);
4. no major regression on high-frequency shapes — yes (none).

Criterion 3 fails → "do not claim success" → per the criteria the code
may remain in the tree only as bench/test infrastructure with the
dispatcher NOT in the production path. That is exactly the final state
(§11): the measured table stays in the experimental dispatcher for
future re-benchmarking, the production runtime calls the frozen baseline
directly, and no speedup is claimed.

## 10. Failed experiments (this is part of the record)

* **R8 everywhere:** +15.6%/+16.3% weighted — acc[8] register pressure
  (63 regs) at 128 threads; loses on every shape. Dead end documented,
  kernel kept for the record.
* **R1 at large N:** N=6144 B=3 R1 measured **54.40 µs vs 41.28 µs
  frozen (+32%)** — 18432 blocks, 33.9 waves, occupancy drops to 78.8%,
  duplicated x-fragment traffic. R1 is good ONLY at the degenerate
  small-N shape.
* **R2 at N=512 B=3:** 0.94× in run1 (loses) — with 3 batches the grid
  (256,3,1)=768 blocks is no longer parallelism-starved and R2's longer
  per-block chain costs more than it saves. Table keeps frozen R4 there.
* **Single-variant dispatch (R1-everywhere, R2-everywhere):** both lose
  the weighted score (R2-everywhere is only ≈ neutral because it wins
  exactly the cells it should) — per-shape adaptivity is what makes the
  kernel-level win; a one-size row tile does not.
* **B-tiled weight reuse:** rejected without implementation (§9) — NCU
  says the batch path at large N is not DRAM-bound and L2 already
  reuses the weights across batch rows; register cost (acc[R][B]) high.
* **E2E variance reduction:** 5×10 independent invocations per side
  (n=50 each) was still not enough to resolve a ~0.5%-of-wall effect
  (sd 2.2–4.7 ms/run). A workload with more logical tokens per run
  (or a longer decode) would be needed to make this kernel-level win
  E2E-visible; that is a future-benchmark design point, not a Phase-B
  scope change.
* **Misalignment-fallback test gap:** the original
  `check_alignment_fallback` compared only the last writer to the frozen
  output (dispatcher overwrote the R1 buffer), so the R1 fallback was
  silently unverified. Fixed in `V07B_FINAL_FUNCTIONAL_SHA` with
  independent outputs + independent comparisons (§4).

## 11. Production state (final)

* `src/runtime/qwen35_full_attention.cpp` and
  `src/runtime/qwen35_deltanet.cpp` are **byte-identical to Phase A**
  (`git diff a9b5a6e -- src/runtime/` is empty at
  `V07B_FINAL_FUNCTIONAL_SHA`): all 30 W4A16 call sites are the frozen
  direct calls `int4_gemv_bf16(...)` /
  `kernels::batch_int4_gemv_bf16(...)`; the runtime no longer includes
  `int4_gemv_qwen35.h` and **never enters the experimental dispatcher**.
  The full-model EXACT gates (full logits / token IDs / hybrid state)
  hold by construction and were re-run in ctest at both SHAs.
* `src/kernels/int4_gemv_qwen35.cu` / `int4_gemv_qwen35.h` remain as
  **benchmark/test/profiling infrastructure**: R1/R2/R8 variant
  kernels, explicit variant launchers (frozen-fallback contract intact),
  and the **experimental / measured dispatcher**
  (`int4_gemv_bf16_qwen35_experimental` /
  `batch_int4_gemv_bf16_qwen35_experimental`) with the measured table
  retained — it is exactly what `V07B_CANDIDATE_SHA` ran, so the
  candidate benchmark is reproducible. Unknown legal shapes fall back to
  the frozen R4 baseline. It is NOT a production entry point and is not
  called from `src/runtime/`.
* The frozen baselines are untouched and directly callable (paired A/B +
  oracle); only behavior-neutral `*_rowtile4_regs()` query helpers were
  added to the frozen TUs.

## 12. Evidence index

All evidence generated at the exact SHAs below, on the RTX 2080 Ti /
driver 570.172.08 / CUDA 11.8 / real-checkpoint binding. Reproduce:
`bash scripts/profile_v07b_w4a16.sh {census|microbench|ncu <t>|e2e|nsys_batched|aggregate_batched}`
with `CUDALM_EVIDENCE_TAG`/`CUDALM_EVIDENCE_SHA_TAG` (refuses tracked
drift at HEAD — same EXACT-SHA discipline as Phase A).

| evidence | SHA | artifact |
|---|---|---|
| authoritative shape census (186 calls/trav, reconciles Phase-A CSV) | `V07B_KERNEL_EXPERIMENT_SHA` | `benchmarks/profiling/v07b_w4a16_shape_census.txt` |
| microbench raw (3rd run) | `V07B_KERNEL_EXPERIMENT_SHA` | `benchmarks/v07b_w4a16_microbench.txt` |
| microbench raw run 1 / run 2 (weighted −4.91% / −3.61%) | `V07B_FINAL_FUNCTIONAL_SHA` | `benchmarks/v07b_w4a16_microbench_run1.txt` / `_run2.txt` |
| NCU before/after (11 targets) | `V07B_KERNEL_EXPERIMENT_SHA` | `benchmarks/profiling/v07b_ncu_*.txt` |
| candidate batched-only nsys (R1 N=16 + R2 N=512 kernels present) | `V07B_CANDIDATE_SHA` | `benchmarks/profiling/v07b_candidate_batched_nsys.*` |
| candidate kernel families (W4A16 213.796 ms / 41.53%) | `V07B_CANDIDATE_SHA` | `benchmarks/profiling/v07b_candidate_kernel_families.txt` |
| candidate E2E raw (5 invocations × 10 measured, n=50) | `V07B_CANDIDATE_SHA` | `benchmarks/v07b_candidate_e2e.txt` |
| final production nsys (R4-only kernels; W4A16 225.292 ms / 42.83%) | `V07B_FINAL_FUNCTIONAL_SHA` | `benchmarks/profiling/v07b_final_batched_nsys.*` |
| final kernel families | `V07B_FINAL_FUNCTIONAL_SHA` | `benchmarks/profiling/v07b_final_kernel_families.txt` |
| final production E2E raw (5 invocations × 10 measured, n=50) | `V07B_FINAL_FUNCTIONAL_SHA` | `benchmarks/v07b_final_e2e.txt` |
| sanitizer memcheck (13 targets, 0 errors) | `V07B_FINAL_FUNCTIONAL_SHA` | `benchmarks/profiling/sanitizer_v07b_w4a16.txt` |
| Phase-A baseline evidence (frozen, untouched) | `a9b5a6e` | `benchmarks/profiling/v07_*` (W4A16 220.581 ms / 42.72%, total 516.364 ms, 132 traversals) |

Verification at `V07B_CANDIDATE_SHA`: build clean; ctest 61/61 (0 failed
/ 0 skipped, 548.27 s); `check_no_torch` CLEAN.
Verification at `V07B_FINAL_FUNCTIONAL_SHA`: build clean; ctest 61/61
(0 failed / 0 skipped, 537.79 s); `check_no_torch` CLEAN; sanitizer
13/13 `ERROR SUMMARY: 0 errors` (B=1 optimized shapes N=16/N=512 ×
R1/R2/R8, B=2/3 batch shapes, small-N edges N=1/3, full parity test
driving the variants + experimental dispatcher on all shapes × B=1/2/3).
