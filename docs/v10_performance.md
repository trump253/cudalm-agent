# CUDALM v1.0 — Performance & Reproducibility

**Audience:** GitHub readers and interviewers. This document is the
portfolio-level performance record: the current release benchmark, a
reproducible benchmark workflow, and the historical v0.7 optimization
case study (profile-first, KEEP/REJECT discipline). It is deliberately
short — the deep dives live in the referenced documents.

**Evidence base:** everything in §3 is bound to the exact commit
`eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a` (branch
`v1.0-portfolio-release`, check-out-clean tracked tree; the
benchmark executable was CMake-built at this SHA and its sha256 is
recorded in `benchmarks/v10/environment.txt`) and the raw
reports in `benchmarks/v10/`. All numbers are **measured on this
hardware / this checkpoint / this canonical workload** — nothing here
is a general performance claim.

---

## 1. Executive Summary

CUDALM is a PyTorch-free, native C++17/CUDA inference runtime for
Qwen3.5-0.8B-Base on a single RTX 2080 Ti (CUDA 11.8). At the v1.0
release point:

- The runtime executes a **true batched decode**: the continuous
  batching scheduler commits decode cohorts to a real batched GPU
  path (`forward_batch_with_state`), not per-request simulation. On
  the canonical 4-request workload (27 logical token-forwards), the
  scheduler completes the same work in **22 model traversals instead
  of 27** (3 committed batches covering 8 batched tokens; avg decode
  batch size 2.67, max 3).
- Measured on this hardware / checkpoint / workload:
  independent/serial **0.1327 s mean** (203.5 logical tok/s) vs
  continuous-batched **0.1161 s mean** (232.6 logical tok/s) — a
  **−16.6 ms (−12.5%) wall-time delta** on the 27-token workload.
  The batched-only serving profile (no serial interleaving in the
  process) measures **0.1100 s mean (245.4 logical tok/s)**.
- The project's optimization history is **profile-driven and
  evidence-gated**: v0.7 profiled the frozen runtime first, then
  evaluated candidates against pre-specified KEEP/REJECT criteria.
  One candidate was kept (fused residual-add + RMSNorm:
  **−24 kernel launches per traversal**, paired E2E **−1.238 ms/run**,
  95% CI [−2.437, −0.040] excluding 0); two were rejected (W4A16 GEMV
  and DeltaNet delta-rule rewrites) — including one whose kernels
  were measurably faster in isolation but which failed the E2E
  wall-clock gate.
- The benchmark workflow itself is reproducible:
  `scripts/benchmark_v10_release.sh` refuses to run on a dirty
  tracked tree, records SHA / GPU / driver / CUDA / compiler / build
  type / binary hash / model + checkpoint identity, and writes raw
  reports + a machine-readable summary under `benchmarks/v10/`.

What this document does **not** claim: comparison against vLLM or
llama.cpp, production concurrent QPS, Tensor Core usage, or
"fully optimized" kernels. See §7.

## 2. Benchmark Environment

| item | value |
|---|---|
| GPU (device 0) | NVIDIA GeForce RTX 2080 Ti (11264 MiB, sm_75) |
| NVIDIA driver | 570.172.08 (CUDA 12.8 driver) |
| CUDA toolkit / runtime | 11.8 (V11.8.89) |
| Host compiler | gcc 9.4.0 |
| CMake build type | Release |
| Model | Qwen/Qwen3.5-0.8B-Base (`/root/models/Qwen3.5-0.8B-Base`), raw text, **no chat template** |
| Converted weights | `build/data/qwen35_08b_full.cudalm` (gitignored build artifact; sha256 recorded in `benchmarks/v10/environment.txt`) |
| Evidence SHA | `eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a` (check-out-clean tracked tree; benchmark binary CMake-built at this SHA, sha256 in `environment.txt`) |

The full environment record (date, exact benchmark commands, binary
sha256, model/checkpoint sha256) is in
`benchmarks/v10/environment.txt`.

## 3. Current v1.0 Release Benchmark

### 3.1 Workload definition (canonical, unchanged since v0.6)

Four requests — A greedy, B/C/D seeded — with prompt lengths
**2 / 5 / 3 / 4** tokens and generated lengths **3 / 6 / 5 / 3**,
admitted with **dynamic arrivals** (A,B first; C after 1 step; D
after 2 steps). Total: 14 prompt + 17 generated = **27 logical
sequence-token forwards** (Σ(N+m−1) = 4+10+7+6). Both modes do the
**same total work** (identical prompts / max_new / seeds; all four
requests complete; no cancellation in the timed run), so the
comparison is apples-to-apples.

Two execution modes, one single CUDA stream, prefill stays serial:

- **mode A — independent / serial:** each request runs alone (fresh
  state, direct single-token forward), one at a time, summed.
- **mode B — scheduler continuous batched:** the requests are
  admitted dynamically into the scheduler; its decode cohorts take
  the **true batched GPU path** (`forward_batch_with_state`).

Timing discipline: 2 un-timed warmup runs, then **10 measured runs**
per mode, each bracketed by `cudaStreamSynchronize` (wall time
includes all enqueued GPU work; no bleed between runs), host
`steady_clock`. Protocol: `--warmup-runs 2 --measured-runs 10`.

### 3.2 Results (measured on this hardware / checkpoint / workload)

From `benchmarks/v10/release_serial_batched.txt` (`--mode both`):

| metric | mode A — serial | mode B — continuous batched |
|---|---|---|
| wall time mean | **0.132707 s** | **0.116064 s** |
| wall time median | 0.131392 s | 0.115457 s |
| wall time min / max | 0.129797 / 0.138380 s | 0.114167 / 0.120369 s |
| logical tok/s (mean) | 203.46 | 232.63 |
| single forward calls | 27 | 19 |
| batch forward calls | 0 | 3 |
| completed model traversals | 27 | 22 |
| avg / max decode batch size | 0 / 0 | 2.67 / 3 |

From `benchmarks/v10/release_batched.txt` (`--mode batched` only —
the independent serving-path profile, same protocol):

| metric | mode B — continuous batched (batched-only process) |
|---|---|
| wall time mean / median | **0.110017 s / 0.109119 s** |
| wall time min / max | 0.107920 / 0.113852 s |
| logical tok/s (mean) | 245.42 |
| single / batch forward calls | 19 / 3 |
| completed model traversals | 22 |
| avg / max decode batch size | 2.67 / 3 |

**Observed delta (batched vs serial, `--mode both` run):**
wall time **−16.64 ms (−12.5%)** on the 27-token workload; logical
throughput **+29.2 tok/s**. The per-run wall times are in the raw
reports (run-to-run min–max spread ≈5–7% — normal on this desktop machine).
This delta is a property of **this hardware / checkpoint / canonical
workload**, not a general "CUDALM is X% faster than serial" claim.

**Historical comparison with the frozen v0.7 sign-off
(descriptive only).** The historical v0.7 measurements
(`docs/v07_final_performance.md` §3.1, same machine / checkpoint,
v0.7 protocol 1 warmup + 5 measured): serial 0.130968 s, batched
0.117271 s. The v1.0 point estimates are approximately **+1.3%
serial** (0.132707 s) and **−1.0% batched** (0.116064 s) relative
to those historical values. The measurements were collected in
separate runs with different warmup/sample counts and are **not a
paired A/B experiment** (different dates, no alternation, no
pairing). Therefore this comparison is descriptive only; **no
regression or improvement is attributed** from these cross-version
point estimates.

The full parse is in `benchmarks/v10/summary.json` (a convenience
view — the raw reports are the evidence).

## 4. What Continuous Batching Demonstrates

On the canonical workload, batching saves **5 model traversals
(27 → 22)**: after the dynamic arrivals, the scheduler commits
decode cohorts of 2–3 sequences per forward (3 committed batches
covering 8 batched tokens; the remaining 19 forwards are single
sequences — the requests have staggered lengths, so cohorts shrink
as requests finish). Each committed batch of B is **B logical tokens
in ONE traversal** — that arithmetic, visible directly in the
report (`single_forward_calls 19 + batch_forward_calls 3 = 22`
traversals for 27 logical tokens), is the mechanism behind the wall
delta in §3.2.

What this demonstrates, precisely:

- The serving path is a **real batched decode** — the batched GEMV /
  DeltaNet / attention kernels execute multi-sequence cohorts on the
  GPU (the kernels are correctness-first, not tuned — see §7).
- The scheduler's **continuous** behavior (mid-stream arrivals,
  per-step cohort formation, commit-before-visible) is what groups
  the work; the workload's dynamic arrivals are what exercise it.
- It does **not** demonstrate concurrency of the HTTP frontend (that
  endpoint is single-threaded, one request at a time — §7), and it
  does not extrapolate to large-batch or long-sequence regimes.

## 5. v0.7 Optimization Case Study (frozen history)

The optimization history is kept as a **case study of method**, not
re-run in v1.0: profile first, pre-specify the gate, measure the
candidate, then KEEP or REJECT on evidence. Full provenance:
`docs/v07_profiling.md`, `docs/v07_w4a16_optimization.md`,
`docs/v07_deltanet_optimization.md`, `docs/v07_final_performance.md`,
with raw evidence under `benchmarks/` and `benchmarks/profiling/`.

### 5.1 Profile-first methodology

v0.7 Phase A changed **nothing** in the runtime. It built the
Nsight Systems / Nsight Compute baseline on the frozen v0.6
continuous-batching runtime (exact-SHA discipline: every evidence
artifact bound to a check-out-clean SHA; the profiling script
**fails loud** on a dirty tracked tree — an early dirty-tree batch of
evidence was declared invalid and deleted). The profile identified
the dominant cost structure (GEMV families ≈ 80% of GPU time;
**442.6 kernel launches per traversal** — heavy small-kernel
fragmentation) and set the optimization targets. No optimization was
started before the baseline existed.

### 5.2 W4A16 GEMV rewrite — REJECTED

Candidate: replace the frozen bf16 GEMV with a W4A16 (4-bit weight,
16-bit activation) GEMV across row-tile variants (R1/R2/R8).
Result: **all candidates REJECTED for production** — R1 lost on 6/8
production shapes (+8.73% / +11.34% worse), R8 lost on every shape
(+14.88%, register pressure), R2 ≈ neutral. A second measured table
showed a **−4.91% / −3.61% microbench edge** on two shapes, but the
pooled A/B was **p = 0.284 — the E2E improvement was within noise**.
The baseline bf16 GEMV was retained. This is the discipline in
action: a kernel-isolated microbench advantage that does not
survive the E2E gate is a REJECT.

### 5.3 DeltaNet delta-rule rewrite — REJECTED

Candidate: rewrite the DeltaNet delta-rule update (vreg/vvec/vchunk
launchers) with **EXACT bit-exact parity** (hard gate vs the frozen
sequence). The candidates were genuinely faster in isolation —
microbenchmark **−22…−24%**, NCU kernel duration **−14…−26%** — but
the pre-specified paired A/B E2E gate (alternating order, fresh
process per invocation) had a **95% CI that included 0**. Verdict:
**REJECTED for production; the frozen baseline retained** (the final
`src/runtime/qwen35_deltanet.cpp` is byte-identical to the
pre-candidate file). The rejection was driven by **wall-clock
evidence, not kernel evidence** — the documented reason a faster
kernel is not automatically a faster program (launch structure,
memory traffic, and the rest of the 442-launch traversal dominate).

### 5.4 Fused residual-add + zero-centered RMSNorm — KEPT

Candidate: replace the 2-launch post-attention residual sequence
(`add` → `rmsnorm_zc`) with ONE kernel `qwen35_fused_add_rmsnorm_zc_bf16`
at all 4 post-attention sites (DeltaNet B=1 + batch, Full-Attention
B=1 + batch). **Bit-exact by construction** (bf16-rounded residual
reused verbatim; frozen reduction tree and op order) with a
`memcmp` hard gate.

Evidence (`docs/v07_final_performance.md` §2, bound to exact SHAs):

- Nsight Systems A/B: total launches 58422 → 55254 = **−24.0
  launches/traversal — exactly the predicted −1 launch/layer × 24
  layers**; host launch time −125 µs/traversal; total GPU kernel
  time flat within run noise; the add+rmsnorm family itself
  −6.19 ms. No important kernel regression.
- Paired A/B E2E (20 fixed pairs, alternating, fresh process,
  2 warmup + 10 measured, canonical workload, batched mode):
  **paired mean Δ = −1.238 ms/run** (SD 2.562, SE 0.573); median
  −1.067; t = −2.162, df = 19, p = 0.0436; **95% CI [−2.437, −0.040]
  excludes 0**; 15/20 pairs faster. Pre-specified KEEP criterion
  (mean < 0 AND median < 0 AND CI excludes 0) **MET → KEPT into
  production**.

### 5.5 The lesson (stated honestly)

- **Profile before optimizing.** The launch-fragmentation target
  came from measurement, not intuition.
- **Gate on E2E wall time, not kernel time.** Two of three
  candidates had real kernel-isolated wins and both were rejected
  because the E2E gate did not clear.
- **Bit-exactness is a hard gate, not a nice-to-have.** Every
  candidate (including the kept one) had an EXACT parity test before
  any timing.
- A REJECTED candidate's microbench speedup is **not** a production
  speedup — it is never reported as one anywhere in this repo.

## 6. Reproduction

```bash
# 1. at the evidence SHA (any check-out-clean tree works):
git checkout v1.0-portfolio-release
git checkout eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a   # the evidence SHA

# 2. configure + build (inside an existing build dir or a fresh one)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 --target bench_qwen35_continuous_batching

# 3. the release benchmark (FAILS LOUD if the tracked tree is dirty;
#    records SHA/environment, runs warmup=2 measured=10 in both
#    passes, writes benchmarks/v10/*, self-checks SHA consistency)
./scripts/benchmark_v10_release.sh

# manual equivalent of pass 1 (the script runs both passes):
build/benchmarks/bench_qwen35_continuous_batching \
    build/data/qwen35_08b_full.cudalm /root/models/Qwen3.5-0.8B-Base \
    <python3> $(pwd) benchmarks/v10/release_serial_batched.txt \
    --no-convert --measured-runs 10 --warmup-runs 2 --mode both
```

Prerequisites: the checkpoint at `/root/models/Qwen3.5-0.8B-Base`
(or set `CUDALM_CKPT`), a converted `build/data/qwen35_08b_full.
cudalm` (`tools/convert_qwen35.py --full-model`, or let the binary
convert it), and an RTX 2080 Ti-class GPU. Re-running produces
**new** numbers for **your** environment — compare structure
(traversals, batch counts) rather than expecting identical wall
times. The v0.7 evidence reproduction commands are in the v0.7 docs.

## 7. Scope / Limitations

- **Numbers are local.** Every figure in this document is measured on
  one RTX 2080 Ti, one checkpoint, one 27-token canonical workload,
  CUDA 11.8. They are not portable, and no general "faster than X"
  claim is made. There is no vLLM / llama.cpp / any-external-system
  comparison — none exists in this repo, and a fair one would need a
  dedicated same-machine, same-model campaign.
- **The batch kernels are correctness-first, not tuned.** v0.7
  established that and explicitly deferred tuning; v1.0 makes no
  performance changes at all. "Fully optimized kernels" / "Tensor
  Core optimized" would both be false statements.
- **No HTTP performance numbers.** The HTTP frontend
  (`cudalm-server`) is single-threaded, one request at a time, one
  CUDA stream. Concurrent QPS / multi-client throughput / HTTP
  continuous-batching throughput do not hold for it, and this phase
  deliberately reports none. (Its correctness contract is covered by
  the v0.9/v1.0 CPU + e2e gates, not by performance claims.)
- **Small-workload statistics.** 27 tokens is a micro-workload; the
  per-run wall spread (≈5–7% min–max) is comparable to the batched-vs-serial
  delta. The v0.7 paired A/B (20 fresh-process pairs) is the
  statistically binding evidence for the only optimization this
  project kept — this release benchmark is a reproducibility
  baseline, not a statistical experiment.
- **What v1.0 Phase B did NOT do:** no new NCU / Nsys campaign, no
  kernel tuning, no candidate evaluation, no re-run of the v0.7 A/B
  (frozen history), no benchmark of the HTTP path, no changes to
  `src/`, `include/`, or any CUDA code.
