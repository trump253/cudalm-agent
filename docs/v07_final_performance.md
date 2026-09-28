# CUDALM v0.7 — Final Performance Sign-off (Phase D)

Phase D goal: **reduce small-kernel / launch fragmentation on the Qwen3.5
serving path and complete the v0.7 final performance sign-off.** This document
is the v0.7 final sign-off: the fusion KEEP/REJECT decisions, the corrected
launch / GPU / host accounting, the Phase-A → v0.7 comparison, and the hard-gate
results, all bound to exact check-out-clean SHAs.

**Decision summary: candidate 1 (fused residual-add + zero-centered RMSNorm)
is KEPT into production. Candidate 2 (Full-Attention q/k norm+RoPE fusion) was
evaluated and is NOT pursued in this phase (documented rationale, §5).**

---

## 1. SHA discipline

Every performance evidence artifact in this phase is bound to an exact
check-out-clean SHA (tracked tree clean at generation).

| SHA | meaning |
|---|---|
| `97c0c32e5323a133676fee2bf2740d71ded2b114` | Phase C final functional (production before D; B/C rejected → runtime == Phase-A) |
| `9db196e0dfc95c8fbf618f50c7149662bd6a0e53` | **Phase D EXPERIMENT** — fused kernel + EXACT parity gate + microbench; **runtime unchanged** (== production) |
| `9edd9ef6aec84b8dcf66a262652c2e285ff73793` | **Phase D CANDIDATE = FINAL FUNCTIONAL (KEEP)** — runtime wired to the fused kernel at all 4 post-attention sites |
| (docs/evidence-only HEAD) | this commit: docs + all Phase-D evidence |

`git diff 9db196e 9edd9ef` touches **only** the two runtime files
(`src/runtime/qwen35_deltanet.cpp`, `src/runtime/qwen35_full_attention.cpp`) —
a clean A/B: the candidate differs from the baseline by exactly the 4
post-attention add→fused rewrites.

---

## 2. Candidate 1 — fused residual-add + zero-centered RMSNorm (KEEP)

### 2.1 What it does

Replaces the 2-launch post-attention residual sequence

```
qwen35_add_bf16(a, b, res)            // res = bf16(f32(a)+f32(b))
qwen35_rmsnorm_zc_bf16(res, w, norm)  // norm = zc-rmsnorm(res)
```

with ONE kernel `qwen35_fused_add_rmsnorm_zc_bf16` at all 4 post-attention
sites (DeltaNet B=1 + batch, Full-Attention B=1 + batch). **BIT-EXACT by
construction**: the residual is a single bf16 RNE of the fp32 sum (frozen add)
and is stored as the MLP residual; the RMSNorm then consumes the **BF16-ROUNDED
residual** and reuses the frozen reduction tree (per-thread partial order, warp
shuffle, smem `warp_sums`, `rsqrtf`) and per-element op order verbatim, with one
bf16 RNE per norm output. No numeric reordering. Saves **1 launch/layer ×
24 layers = −24 launches/traversal**.

The 4 MLP final-adds and the 8 input/q/k zero-centered norms remain on the
frozen kernels (out of scope; no reordering).

### 2.2 EXACT correctness

- `test_qwen35_fused_add_rmsnorm` — BIT-EXACT hard gate (BF16 residual **and**
  BF16 norm, `memcmp`) vs the frozen 2-launch sequence: M=1/2/3, H=256/1024
  (PER=1/4), production eps=1e-6 + alt eps, multi-seed random +
  zero/small/large edge classes. **ALL PASS.**
- Model-level EXACT (full logits / token / state) at the CANDIDATE/FINAL SHA:
  `test_qwen35_state_parity`, `test_qwen35_full_attention_golden`,
  `test_qwen35_deltanet_golden`, `test_qwen35_full_model`,
  `test_qwen35_full_forward` all pass.

### 2.3 Microbenchmark (kernel-level, EXPERIMENT SHA)

Stream-elapsed GPU time/call, H=1024, production shape:

| M | frozen 2-launch (µs) | fused 1-launch (µs) | Δ (µs) |
|---|---|---|---|
| 1 | 4.733 | 2.875 | −1.858 |
| 2 | 4.908 | 2.921 | −1.986 |
| 3 | 4.904 | 2.931 | −1.973 |

Raw artifact (run at exact SHA `9db196e`, check-out-clean tree, with GPU /
CUDA / binary-sha256 / command in the header):
`benchmarks/v07d_fused_add_rmsnorm_microbench.txt`. (The fused family is also
faster on pure GPU kernel time per Nsys: 3.10 µs/1-launch vs ~5.15 µs for the
replaced 2-launch pair, §2.4.)

### 2.4 Nsys before/after (batched-only, 1 warmup + 5 measured, 132 traversals each)

Directly comparable: delta-rule = 2052 + 324 = **2376 logical calls = 132
traversals** in both. (`benchmarks/profiling/v07d_nsys_before_after.txt`,
`v07d_{base,cand}_batched_nsys.*`.)

| metric | BASE (`9db196e`) | CAND (`9edd9ef`) | Δ |
|---|---|---|---|
| total kernel launches | 58422 (**442.6/trav**) | 55254 (**418.6/trav**) | **−3168 = −24.0/trav** |
| host `cudaLaunchKernel` | 296.17 ms (**2243.7 µs/trav**) | 279.68 ms (**2118.7 µs/trav**) | **−125.0 µs/trav (−5.6%)** |
| total GPU kernel time | 518.078 ms | 518.819 ms | +0.741 ms (+0.14%, run noise) |
| add+rmsnorm family GPU | 37.365 ms | 31.176 ms | **−6.19 ms** |
| fused kernel per-call | — | 9.829 ms / 3168 = 3.10 µs | vs replaced pair ~5.15 µs |

The **−24 launches/traversal is exactly the predicted −1 launch/layer × 24
layers.** Host launch time drops 125 µs/traversal. Total GPU time is flat within
noise (the fused family is faster, but the dominant GEMV families — ~80% of GPU
time — vary run-to-run and offset it). **No important kernel regression.**

### 2.5 Paired A/B E2E (the binding test) — KEEP

Two clean worktrees (baseline `9db196e` == production; candidate `9edd9ef`),
**20 fixed pairs, no early stop**, alternating order (odd: base→cand; even:
cand→base). Each invocation = fresh process, 2 warmup + 10 measured,
`--mode batched`, canonical workload. **One statistical value per invocation =
mean of its 10 runs; in-process runs are NOT independent samples.** Raw data +
both binary SHAs + sha256 + model hash: `benchmarks/v07d_paired_ab_raw.txt`.

| statistic | value |
|---|---|
| **paired mean delta** | **−1.238 ms/run** (SD 2.562, SE 0.573) |
| **paired median delta** | **−1.067 ms/run** |
| **paired t** | **−2.162, df = 19, two-sided p = 0.0436** |
| **95% CI of paired mean** | **[−2.437, −0.040] ms — EXCLUDES 0** |
| pairs with Δ < 0 | 15 / 20 |
| Wilcoxon signed-rank (supplemental) | W+ = 54, W− = 156, **exact two-sided p = 0.0583** |

**Pre-specified KEEP criterion** (mean Δ < 0 AND median Δ < 0 AND 95% CI of the
paired mean excludes 0) is **MET** → **KEEP.** The Nsys independently confirms
the mechanism (−24 launches/trav, −125 µs/trav host, no kernel regression).

---

## 3. v0.7 final production sign-off (SHA `9edd9ef`)

### 3.1 Canonical timing (`benchmarks/v07d_final_timing.txt`)

| mode | wall time (mean) | logical tok/s (mean) |
|---|---|---|
| A — independent / serial | 0.130968 s | 206.16 |
| B — scheduler continuous batched (serving) | **0.117271 s** | **230.24** |

(4 requests, prompts 2/5/3/4 tok, max_new 3/6/5/3; 1 warmup + 5 measured.)

### 3.2 Batched-only Nsys + kernel-family table (`v07d_cand_batched_nsys.*`)

Total GPU kernel time **518.819 ms**; total launches **55254 (418.6/trav)**;
host `cudaLaunchKernel` **279.68 ms (2118.7 µs/trav)**. Top kernel families
(132 traversals):

| kernel family | GPU time (ms) | launches |
|---|---|---|
| int4gemv_rowtile4_bf16 | 178.570 | 21204 |
| bf16_gemv_vec4_row (LM head) | 106.226 | 114 |
| deltanet_delta | 52.058 | 2052 |
| batch_int4gemv_rowtile4_bf16 | 46.273 | 3348 |
| batch_bf16_gemv_vec4_row | 43.911 | 18 |
| qwen35_rmsnorm_zc (input/q/k) | 13.637 | 4884 |
| **qwen35_fused_add_rmsnorm_zc (new)** | **9.829** | **3168** |
| qwen35_silu_mul | 9.190 | 3168 |
| batch_deltanet_delta | 8.557 | 324 |
| qwen35_add (MLP final only) | 7.710 | 3168 |
| deltanet_gated_rmsnorm | 6.754 | 2376 |
| qwen35_paged_attention_scores | 6.318 | 684 |
| deltanet_conv | 5.632 | 2052 |
| deltanet_gbeta | 5.431 | 2052 |
| qwen35_partial_rope | 3.297 | 1368 |
| (full per-family + memops + CUDA API in the CSVs) | | |

**CUDA API summary** (final): `cudaMemcpyAsync` 11102 calls / 450.1 ms;
`cudaLaunchKernel` 55254 calls / 279.7 ms; `cudaStreamCreate` 1 / 260.5 ms;
`cudaFree` 2612 / 100.9 ms; `cudaMalloc` 2612 / 47.7 ms; `cudaMemsetAsync`
2148 / 9.7 ms.

**MemOps summary** (final): memcpy HtoD 1592 / 119.7 ms; memcpy DtoD 9438 /
20.4 ms; memset 2148 / 10.4 ms; memcpy DtoH 72 / 4.0 ms.

### 3.3 Phase-A baseline vs v0.7 final

| metric | Phase-A baseline (`v07_batched_nsys`) | v0.7 final (`9edd9ef`) | Δ |
|---|---|---|---|
| wall time (batched, mean) | 116.368 ms | 117.271 ms | +0.90 ms (within run noise) |
| tok/s (batched, mean) | 232.02 | 230.24 | −1.78 (within run noise) |
| total GPU kernel time | 516.364 ms | 518.819 ms | +2.45 ms (run noise) |
| **kernels / traversal** | **442.6** | **418.6** | **−24.0 (exact, attributable)** |
| host `cudaLaunchKernel` / traversal | 2542.4 µs | 2118.7 µs | −423.7 µs |
| add+rmsnorm family (post-attn) | 2 launches/layer | 1 fused launch/layer | −24 launches/trav |

**Honesty note on wall time / GPU time:** the single-sample canonical wall and
GPU times differ by <1–3 ms, well within the run-to-run noise measured by the
paired E2E (SE ≈ 2.5 ms/invocation). The **binding, attributable** evidence for
the fusion's effect is the **paired A/B E2E (−1.238 ms, 95% CI excludes 0)** and
the **exact launch accounting (−24 launches/traversal)**. The host
`cudaLaunchKernel` per-traversal figure also reflects run-to-run host-timing
variance between the Phase-A capture and the final capture; the candidate-1
incremental host effect measured under identical conditions is **−125 µs/trav**
(BASE→CAND, §2.4).

> The Phase B (W4A16) and Phase C (DeltaNet) candidates were both **REJECTED**
> and are **NOT** part of the v0.7 production runtime; no rejected-candidate
> speedup is claimed here. The only production change from the Phase-A runtime
> is the Phase D candidate-1 fusion.

---

## 4. Hard gates (final SHA `9edd9ef`)

| gate | result |
|---|---|
| full ctest | **63/63 passed, 0 failed, 0 skipped** (794.55 s) |
| check_no_torch | **CLEAN** (no torch/pybind/py symbols in include/ src/) |
| compute-sanitizer (memcheck) | **0 errors** — `test_qwen35_fused_add_rmsnorm` (new kernel) + `test_qwen35_full_model` (in-context); raw log `benchmarks/profiling/v07d_sanitizer_final.log` |
| EXACT correctness | fused parity (kernel, both outputs), state parity, full-attention golden, deltanet golden, full model, full forward — **ALL PASS** |

---

## 5. Candidate 2 — Full-Attention q/k RMSNorm + RoPE fusion (evaluated, NOT pursued)

**Scope considered:** fuse the per-head zero-centered q/k norm (H=256, PER=1)
with the following partial rotate-half RoPE into one `norm+rope` kernel, for
q (n_heads=8) and k (n_kv=2), B=1 and batch (per-row positions).

**Feasibility (verified, bit-exact-achievable):** one block per row (256
threads) computes the frozen norm (same reduction tree) → one bf16 RNE → smem,
`__syncthreads`, then each thread applies the **frozen 3-step bf16 RoPE chain**
reading its own + paired bf16 value from smem; a single `unit_rows` parameter
unifies B=1 (b=0, flat cos/sin) and batch (b=m/unit_rows, per-row cos/sin). The
bf16 rounding boundary and RoPE op order are preserved verbatim.

**Value:** −12 launches/traversal (6 Full-Attention layers × q+k), the q/k
norm+rope kernels being among the smallest in the census (µs-scale).

**Decision: NOT implemented in this phase.** Rationale, tied to the task's
explicit guardrail ("don't chase a launch-count number; only keep fusions with
**real E2E value and reasonable maintenance cost**"):
1. Candidate 1 already delivers the primary launch-fragmentation reduction
   (−24 launches/traversal) and is KEEP with the 95% CI excluding 0 — the Phase
   D goal is met.
2. Candidate 2's **incremental** value is modest (−12/trav, µs-scale kernels,
   ~−60 µs/trav host) relative to the **higher** implementation/maintenance cost
   (a fused block-reduction norm + cross-thread paired RoPE + cos/sin tables +
   per-row batch positions, plus a full parity/microbench/Nsys/20-pair-E2E loop).
3. It is framed conditionally ("only if first succeeds", "only implement if
   bit-exact preservable"); the bar it clears is feasibility, not cost/benefit.

Candidate 2 remains a **valid future optimization** and is documented here (with
the full technical assessment) for the external reviewer to weigh. It is **not**
in the v0.7 production runtime.

---

## 6. Evidence inventory (all bound to exact SHAs)

- `benchmarks/v07d_paired_ab_raw.txt` — 20-pair paired E2E raw (both binary SHAs + sha256 + model hash).
- `benchmarks/v07d_paired_ab_analysis.txt` — paired statistics + KEEP decision.
- `benchmarks/v07d_fused_add_rmsnorm_microbench.txt` — fused add+rmsnorm microbench raw (exact SHA `9db196e`, clean tree; GPU/CUDA/command in header).
- `benchmarks/v07d_final_timing.txt` — canonical timing at the final SHA.
- `benchmarks/profiling/v07d_nsys_before_after.txt` — Nsys before/after (base vs cand).
- `benchmarks/profiling/v07d_{base,cand}_batched_nsys.{nsys-rep,sqlite,*_summary.txt,*_summary_*.csv,header.txt}` — full batched Nsys artifacts.
- `benchmarks/profiling/v07d_sanitizer_final.log` — compute-sanitizer raw log (0 errors).
- Phase C (superseded context): `benchmarks/v07c_paired_ab_raw.txt`,
  `benchmarks/profiling/v07c_nsys_corrected_before_after.txt`,
  `docs/v07_deltanet_optimization.md`.
