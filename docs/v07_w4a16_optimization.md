# CUDALM v0.7 Phase B — W4A16 GEMV optimization (Qwen3.5-0.8B)

**Verdict (final):** all candidates **REJECTED for production** — the
frozen R4 row-tile baseline is **retained** for every production shape
(the production path is bit-identical to Phase A). The adaptive row-tile
variants, the dispatcher, the parity test and the microbenchmark remain
in the tree as bench/test infrastructure and a measured extension point.
The rejection reason is a failed E2E acceptance criterion (kernel-level
W4A16 time drops −5.99%, but the canonical continuous-batched wall
improvement is within measurement noise; see §9). This is the
"REJECTED, baseline retained" outcome the Phase-B criteria anticipate.

Scope: the Phase-A #1 serving GPU hotspot — W4A16 / INT4 GEMV, **42.72%**
of serving (continuous-batched) GPU kernel time (B=1 33.82% + B>1 8.90%,
Phase A §6.1). Nothing else was touched (no DeltaNet, no norm/elementwise
fusion, no CUDA Graph, no LM-head changes, no paged-attention/scheduler/
state-manager changes, no new quant format, no prefill, no multi-stream,
no Tensor Core, no tolerance-based correctness).

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
9804-check parity test, §4). This is the CUDALab frozen-reference
(`cb6a6a9`, `int4gemv_rowtile8_hx.cu`, experiment INT4GEMV-0003)
row-tile technique — borrowed, with the dispatcher and the batch path
redesigned for CUDALM (CUDALab tuned 4096²-style shapes; its winners do
NOT transfer automatically and are re-measured here).

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
  vectorized-contract fallback to the frozen entry); production
  dispatcher `int4_gemv_bf16_qwen35` / `batch_int4_gemv_bf16_qwen35` —
  static, keyed (N,K[,B]), no runtime autotuning, frozen-R4
  bit-compatible fallback for unknown legal shapes (incl. the scalar
  fallback for non-16B-aligned inputs).
* The frozen baselines are **untouched and directly callable** for paired
  A/B (only behavior-neutral `*_rowtile4_regs()` query helpers were added
  to the two frozen TUs).
* Runtime call sites (30) now enter through the dispatcher; with the
  retained-baseline table (§9) the production path IS the frozen R4
  baseline, bit-for-bit.
* `tools/ncu_w4a16_target.cpp`: controlled single-(N,K,B,variant) NCU
  harness (no model, no interleaved kernels).
* `benchmarks/bench_w4a16_qwen35_shapes.cpp`: dedicated microbench —
  census shapes × B=1/2/3 × {frozen R4, R1, R2, R8}; CUDA events,
  warmup ≥30, measured iters scaled to kernel duration (200–2000;
  reported per cell); weighted production score = census call rates ×
  canonical B distribution (19:1:2).
* `benchmarks/bench_qwen35_continuous_batching.cpp`: tooling-only
  `--warmup-runs N` (default 1 = byte-identical Phase-A behavior; the
  workload is unchanged).

## 4. EXACT parity gate (bit-exactness)

`tests/cuda/test_w4a16_qwen35_optimized` — **9804 bit-exact checks (memcmp
on BF16 words), 0 mismatches**:

* shapes: all 8 census (N,K) + 6 edges (1×128, 3×128, 5×256, 9×1024,
  15×1024, 1023×1024);
* variants R1/R2/R8 (B=1 and batch) vs the frozen baseline, B=1 and
  batch; dispatcher vs frozen;
* seeds: 4 × input classes: 5 (deterministic random, all-zero, small
  magnitude, mixed sign, boundary INT4 nibbles {0x7,0x9,0x8,0x0,0xF} +
  zero-group scales 1/5) → 8×6 × 3 × 4 × 5 coverage;
* frozen contracts re-gated: batch row-parity (frozen batch row b ==
  frozen single on row b), alignment fallback (misaligned x → variants
  and dispatcher must equal the frozen result on the same input),
  K%128/contract fallback, legal-shape guard.

Full ctest at the final SHA: **61/61 passed, 0 failed, 0 skipped**
(including all existing single/batched/continuous/state-isolation/
scheduler tests). The real-checkpoint FULL-logits / token-ID / hybrid-
state EXACT gates are the existing suite's (unchanged oracle; the
production path is the frozen baseline, so they hold by construction and
are re-run at the final SHA).

## 5. Microbenchmark (kernel level)

`benchmarks/v07b_w4a16_microbench.txt` (2 independent runs; per-shape
mean/median/min/p90 µs, µs/token, speedup, regs). Robust cells
(|run-to-run delta| < 3%):

| shape (N,K) | B | frozen R4 (µs) | R1 (×) | R2 (×) | R8 (×) | winner |
|---|---|---|---|---|---|---|
| 16,1024 | 1 | 7.24 | **5.07 (1.43×)** | 5.69 (1.27×) | 10.29 (0.70×) | **R1** |
| 16,1024 | 2 | 7.19 | **5.09 (1.41×)** | 5.67 (1.27×) | 10.30 (0.70×) | **R1** |
| 16,1024 | 3 | 7.19 | **5.15 (1.39×)** | 5.66 (1.27×) | 10.32 (0.70×) | **R1** |
| 512,1024 | 1 | 7.96 | 6.29 (1.27×) | **5.49 (1.45×)** | 8.55 (0.93×) | **R2** |
| 512,1024 | 2 | 6.51 | 6.40 (1.02×) | **5.94 (1.10×)** | 8.64 (0.75×) | **R2** |
| 512,1024 | 3 | 6.83 | 7.53 (0.91×) | 7.29 (0.94×) | 8.56 (0.80×) | **R4** |
| 1024,2048 | 1 | 7.16 | 7.71 (0.93×) | 7.01 (1.02×) | 9.05 (0.79×) | R4 (R2 ≈ flat) |
| 1024,2048 | 2 | 8.60 | 10.66 (0.81×) | 9.42 (0.91×) | 9.46 (0.91×) | **R4** |
| 1024,2048 | 3 | 11.45 | 13.64 (0.84×) | 11.56 (0.99×) | 12.03 (0.95×) | **R4** |
| 1024,3584 | 1 | 8.45 | 9.43 (0.90×) | 8.44 (1.00×) | 9.33 (0.91×) | **R4** (R2 ≈ flat) |
| 1024,3584 | 2 | 11.40 | 13.84 (0.82×) | 11.80 (0.97×) | 12.44 (0.92×) | **R4** |
| 1024,3584 | 3 | 14.40 | 18.29 (0.79×) | 14.53 (0.99×) | 16.94 (0.85×) | **R4** |
| 2048,1024 | 1 | 7.11 | 8.39 (0.85×) | 7.80 (0.91×) | 8.99 (0.79×) | **R4** |
| 2048,1024 | 2 | 10.28 | 12.23 (0.84×) | 10.87 (0.95×) | 10.09 (1.02×) | **R4** |
| 2048,1024 | 3 | 13.07 | 15.99 (0.82×) | 13.83 (0.94×) | 14.57 (0.90×) | **R4** |
| 3584,1024 | 1 | 9.92 | 11.10 (0.89×) | 10.29 (0.96×) | 9.81 (1.01×) | **R4** |
| 3584,1024 | 2 | 15.01 | 17.78 (0.84×) | 15.34 (0.98×) | 15.28 (0.98×) | **R4** |
| 3584,1024 | 3 | 18.93 | 24.37 (0.78×) | 20.66 (0.92×) | 20.42 (0.93×) | **R4** |
| 4096,1024 | 1 | 10.32 | 12.28 (0.84×) | 10.90 (0.95×) | 10.19 (1.01×) | **R4** |
| 4096,1024 | 2 | 15.83 | 19.79 (0.80×) | 17.04 (0.93×) | 16.07 (0.99×) | **R4** |
| 4096,1024 | 3 | 21.22 | 27.39 (0.77×) | 23.04 (0.92×) | 21.64 (0.98×) | **R4** |
| 6144,1024 | 1 | 12.78 | 15.82 (0.81×) | 14.17 (0.90×) | 16.29 (0.78×) | **R4** |
| 6144,1024 | 2 | 21.14 | 27.35 (0.77×) | 23.04 (0.92×) | 21.38 (0.99×) | **R4** |
| 6144,1024 | 3 | 29.11 | 39.03 (0.75×) | 32.50 (0.90×) | 32.13 (0.91×) | **R4** |

(2-run means; per-run values and mean/median/min/p90/µs-token/regs per
cell in `benchmarks/v07b_w4a16_microbench.txt`; run-to-run deltas <3% on
every cell except the rejected R8@6144-B1 outlier, 14.7–17.9 µs.)

(regs: frozen 56/58 (B=1/batch), R1 47, R2 53, R8 63.)

**Weighted production score** (census call rates × B distribution
0.864:0.045:0.091; `W_Bx = Σ calls/trav × mean(shape,B)`):

| variant | W_B1 (µs/trav) | W_B2 | W_B3 | overall | vs frozen |
|---|---|---|---|---|---|
| frozen R4 (all shapes) | 1609.8 | 2180.0 | 2733.7 | 1737.9 | — |
| R1 everywhere | 1688.4 | 2512.7 | 3350.6 | 1876.9 | **+8.00%** (worse) |
| R2 everywhere | 1573.6 | 2206.7 | 2861.9 | 1719.5 | −1.06% |
| R8 everywhere | 1902.4 | 2374.1 | 3096.0 | 2032.4 | **+16.94%** (worse) |
| **measured table** (N=16→R1; N=512→R2 B=1/2; else R4) | 1502.4 | 2096.3 | 2663.2 | **1635.0** | **−5.92%** (run2: −5.84%) |

The reviewer's hypothesis is confirmed shape-by-shape: R=1 wins only at
the degenerate small-N shape, R=2 only at N=512 (B=1, B=2), and the
frozen R4 remains best at every N≥1024 for all B. "R=1 fastest" /
"R=2 best" were **not** assumed — measured. (single-variant rows above:
run1 weighted block; the measured-table row is recomputed from the raw
per-cell data of both runs: run1 −5.92%, run2 −5.84%.)

## 6. NCU before/after (why it is faster — and why it is not at large N)

`benchmarks/profiling/v07b_ncu_*.txt` (NCU 2022.3, controlled single-
shape launches; 12 measured launches each, 8 skipped warmups).

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

## 7. Batched-only nsys before/after (real serving path)

Phase A (frozen, 1 warmup + 5 measured, 132 traversals) vs Phase B with
the measured dispatcher (same schedule, 132 traversals):

NOTE on evidence provenance: the "B" column below is the measured-
dispatcher dev run (recorded in this doc; that build is not the final
production state). The COMMITTED nsys evidence
(`benchmarks/profiling/v07b_batched_nsys.*`, generated at
V07B_EVIDENCE_SHA) is the FINAL retained-baseline state and shows
W4A16 **219.436 ms / 42.79%** (avg 8.938 µs/call) vs Phase A
220.581 ms / 42.72% — −0.52%, i.e. run-to-run drift, production
unchanged; total 512.864 vs 516.364 ms (−0.68%); launches 58422 =
identical. That is the expected "baseline retained" result.

| family | A: ms | A: % | B: ms | B: % | Δ |
|---|---|---|---|---|---|
| **w4a16/int4 projection** | **220.581** | 42.72% | **207.370** | 41.50% | **−13.211 ms (−5.99%)** |
| total GPU kernel time | 516.364 | 100% | 499.697 | 100% | −16.667 ms (−3.23%) |
| launches | 58422 | | 58422 | | 0 (as expected) |

Per-kernel (W4A16 rows; grid = (X,Y,Z)):

| kernel | grid | A avg (ns) | B avg (ns) | Δ |
|---|---|---|---|---|
| rowtile4 | 4,1,1 | 4859.4 | — | moved → |
| rowtile_bf16 (R1) | 16,1,1 | — | 2784.9 | **−42.7%** (19.94→11.43 ms) |
| rowtile4 | 4,2,1 | 4826.4 | — | moved → |
| rowtile_bf16 (R1) | 16,2,1 | — | 2808.0 | **−41.8%** (1.04→0.61 ms) |
| rowtile4 | 4,3,1 | 4791.6 | — | moved → |
| rowtile_bf16 (R1) | 16,3,1 | — | 2814.9 | **−41.2%** (2.07→1.22 ms) |
| rowtile4 | 128,1,1 | 5482.3 | — | moved → |
| rowtile_bf16 (R2) | 256,1,1 | — | 4285.3 | **−21.8%** (7.50→5.86 ms) |
| rowtile4 (batch) | 128,2,1 | 5566.3 | — | moved → |
| rowtile_bf16 (R2) | 256,2,1 | — | 4684.9 | **−15.8%** (0.40→0.34 ms) |
| rowtile4 (batch) | 128,3,1 | 5691.3 | 5552.5 | −2.4% (kept frozen by design; R2 measured 0.94× there) |
| rowtile4 (B=1, unchanged code) | 256/512/896/1024/1536,1,1 | 6517–14069 | 6455–13970 | −0.5…−1.0% (run drift) |
| rowtile4 (B=2, unchanged code) | 256..1536,2,1 | 9401–23122 | 9358–23134 | −0.5…+0.8% (run drift) |
| rowtile4 (B=3, unchanged code) | 256..1536,3,1 | 12742–32036 | 12499–31284 | −1.7…−2.4% (run drift) |

**Run-to-run drift control:** all shapes whose code is *unchanged* show
a small systematic shift between the two nsys runs (B=1 avg −0.97%,
B=2 +0.10%, B=3 −2.19%, ≈ −1.7 ms total on 149.8 ms). Subtracting it,
the dispatcher-attributable W4A16 reduction is **≈ −11.5 ms/132 trav
(−5.2%)** — matching the weighted microbench (−5.9%) and the NCU
per-target numbers. The family headline (−5.99%) is stated as measured;
the drift-corrected number (−5.2%) is the conservative one.

## 8. End-to-end (canonical continuous-batched)

`--mode batched`, same canonical workload (4 requests, 27 logical
tokens/run, 22 traversals/run), before = Phase-A-SHA binary (with only
the tooling-only `--warmup-runs` param applied), after = Phase-B binary
with the measured dispatcher.

Per-invocation (wall per run, seconds):

| run | schedule | before mean / median | after mean / median |
|---|---|---|---|
| 1 | 1 warmup + 10 | 0.110378 / 0.108560 | 0.108059 / 0.107348 |
| 2 | 2 warmup + 10 | — | 0.108280 / 0.107535 |
| 3 | 1+10 (before) / 2+10 (after) | 0.110930 / 0.109378 | 0.112118 / 0.110085 |
| 4 (long) | 2 warmup + 30 | 0.109488 / 0.109286 | 0.109601 / 0.109778 |

**Pooled per-run analysis (n = 50 before / 60 after):**

* before: mean 109.954 ms, median 108.992 ms, sd 2.584 ms (107.8–120.2)
* after: mean 109.543 ms, median 108.249 ms, sd 3.581 ms (106.2–129.1)
* Δmean = **−0.411 ms (−0.37%)**, Δmedian = **−0.743 ms (−0.68%)**
* Welch t = **−0.020** (df ≈ 104, p ≈ 0.98) — **not distinguishable
  from zero**; per-invocation medians alternate direction
  (−1.21, +0.71, +0.49 ms).

Interpretation: the GPU-side savings are real and accounted for
(≈ 87 µs/trav ≈ 1.9 ms/run from §7; Phase A: GPU kernels are ~79% of
the 109 ms wall, the rest is CPU/launch/sync gap), but the canonical
workload is short (22 traversals/run) and its per-run wall noise
(2.6–3.6 ms, with within-file thermal drift of a few ms) exceeds the
~1.7%-of-wall effect. The measured E2E improvement is therefore
**within noise** — not "no improvement", not "regression" (pooled
direction is favorable in all 4 invocations' means), simply not
statistically claimable on this workload.

**8.2 Final retained-baseline state (committed evidence at
V07B_EVIDENCE_SHA):** `--mode batched`, 2 warmup + 10 measured —
mean 0.111764 s, median 0.110048 s, min 0.109217 s, max 0.116900 s
(241.6 tok/s) vs the Phase-A-SHA before runs above (mean
0.109488–0.110930, median 0.108560–0.109378): the difference is within
per-run noise — the production path is the frozen baseline, as intended
(baseline retained; no e2e claim in either direction).

## 9. KEEP / REJECT table and production decision

| candidate | shapes | microbench | nsys | E2E | decision |
|---|---|---|---|---|---|
| R1 (all shapes) | — | +8.00% (worse) | — | — | **REJECT** (loses on 7/8 shapes) |
| R2 (all shapes) | — | −1.06% | — | — | **REJECT** (loses on 6/8 shapes) |
| R8 (any) | any | +16.94% (worse) | — | — | **REJECT** (register pressure; loses on every batch shape) |
| R1 @ N=16 (B=1/2/3) | 36 calls/trav | 1.38–1.43× | −41.2…−42.7% | pooled | **REJECTED for production** (E2E within noise), kept as bench/test infra |
| R2 @ N=512 B=1 | 12 calls/trav | 1.43–1.47× | −21.8% | pooled | **REJECTED for production** (same reason), kept as bench/test infra |
| R2 @ N=512 B=2 | 12 calls/trav | 1.09–1.11× | −15.8% | pooled | **REJECTED for production** (same reason), kept as bench/test infra |
| R2 @ N=512 B=3 | 12 calls/trav | 0.94× (loses) | −2.4% (≈ drift) | — | **REJECT** (kernel-level loss) |
| B-tiled weight-reuse (same row, multiple b, W fragment shared in registers) | N≥1024 batch | not implemented | — | — | **REJECT without implementation** (reviewer §7: complexity high / gain unclear, don't dig): NCU shows the batch kernels at large N are SM-leaning (44% SM / 12.7% DRAM, not DRAM-bound) and already enjoy cross-b L2 weight reuse (70% L2 hit); expected gain bounded, register cost high (acc[R][B]) |

**Decision:** all candidates **REJECTED for production**. The
dispatcher table returns the **frozen R4 baseline for every production
shape** (`src/kernels/int4_gemv_qwen35.cu` — production path is
bit-identical to Phase A). Acceptance-criterion mapping:

1. all EXACT gates PASS — **yes** (9804 bit-exact checks, ctest 61/61,
   no-torch CLEAN, sanitizer 0 errors, full-logits/token-IDs/state
   EXACT via the existing suite);
2. weighted W4A16 clearly improves — **yes** (−5.9% weighted, −5.99%
   nsys family, −5.2% drift-corrected);
3. canonical continuous-batched E2E median improvement reproducible —
   **NO** (Welch t = −0.02; per-invocation medians alternate);
4. no major regression on high-frequency shapes — yes (dispatcher would
   only touch N=16/N=512, both improving).

Criterion 3 fails ⇒ per the criteria, "do not claim success"; the code
is kept in bench/test without replacing the production dispatcher. No
speedup is manufactured: the final production path is the frozen
baseline and the final evidence shows it.

## 10. Failed experiments (kept in the record — 项目含金量)

* **R8 row tiling** — CUDALab's rowtile8 line (INT4GEMV-0003) was
  re-measured on the Qwen3.5-0.8B shapes: 63 regs, loses on every batch
  shape and on N=6144 (−10…−14%), flat at best on 3584/4096 B=1. The
  CUDALab winner does not transfer (their shapes were large-N 4096²-
  style; here 6 of 8 shapes are K=1024 one-active-warp GEMVs).
* **R1 for large N** — the intuitive "more blocks = faster" breaks at
  N≥2048: 4× more blocks, 4× x-fragment reloads, per-block barrier
  overhead, occupancy *down* at N=6144 B=3 (78.8%), duration +32%.
  "R=1 fastest" was explicitly tested and rejected.
* **R2 at N=512 B=3** — loses 0.94× (reproducible across 2 runs): at
  B=3 the batch grid is already 384 blocks (R4) / 768 (R2), so the
  extra R2 blocks buy nothing, while R4's 4-rows-per-block keeps the
  x-fragment load amortized across 4 rows per block.
* **Whole-family single-variant dispatch** (R1-everywhere +8.0%,
  R2-everywhere −1.06%, R8-everywhere +16.94%) — a static table is the
  only shape-aware form that works; even then E2E is within noise.
* **B-tiled weight reuse** — evaluated on NCU evidence, rejected
  without implementation (see §9).
* **E2E variance reduction** — 3 independent after-invocations +
  30-run long runs + pooled per-run Welch test were all run; the
  canonical workload's 2.6–3.6 ms per-run sd (with within-file drift)
  simply exceeds the ~1.7% effect. A longer-decode workload would
  resolve it, but the canonical workload is fixed by the criteria.

## 11. Production state and next-phase recommendation

* Production: **frozen R4 baseline retained** (bit-identical to Phase
  A); the dispatcher is a static, (N,K[,B])-keyed extension point that
  currently selects the frozen baseline for every production shape, with
  the measured table documented in
  `src/kernels/int4_gemv_qwen35.cu` for the next decision.
* The N=16 (grid-4) and N=512 (grid-128) W4A16 GEMVs remain the single
  biggest *known* kernel-level lever (−42% and −22% per call,
  ≈ 87 µs/trav of W4A16 time). They do not move the canonical short-
  workload E2E beyond noise, but a serving workload with longer decodes
  (or a model with a larger small-N call share) would — the measured
  table is ready to be re-validated there.
* The remaining serving time is elsewhere (LM head 29.2%, DeltaNet
  15.7%) — out of scope for Phase B by the frozen scope list.

## 12. Evidence index (all at V07B_EVIDENCE_SHA, clean tree)

| artifact | path |
|---|---|
| shape census (reconciled) | `benchmarks/profiling/v07b_w4a16_shape_census.txt` |
| microbench (weighted score) | `benchmarks/v07b_w4a16_microbench.txt` |
| NCU before/after (11 targets) | `benchmarks/profiling/v07b_ncu_*.txt` |
| batched-only nsys (FINAL retained-baseline state; W4A16 219.436 ms ≈ Phase A) | `benchmarks/profiling/v07b_batched_nsys.*` |
| batched kernel families (FINAL state) | `benchmarks/profiling/v07b_batched_kernel_families.txt` |
| E2E (FINAL state, 2 warmup + 10) | `benchmarks/v07b_continuous_batching.txt` |
| sanitizer memcheck | `benchmarks/profiling/sanitizer_v07b_w4a16.txt` |
| before references (Phase-A SHA a9b5a6e binary + tooling-only `--warmup-runs`) | this doc §8 (per-run tables are the committed record) |
| Phase-A baseline evidence (frozen, untouched) | `benchmarks/profiling/v07_*` (W4A16 220.581 ms / 42.72%, total 516.364 ms, 132 traversals) |

Reproduce: `bash scripts/profile_v07b_w4a16.sh {census|microbench|ncu <t>|e2e|nsys_batched|aggregate_batched}`
(refuses tracked drift at HEAD — same EXACT-SHA discipline as Phase A).
