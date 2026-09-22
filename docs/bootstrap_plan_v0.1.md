# CUDALM v0.1 Bootstrap Plan

CUDALM = **Native C++/CUDA Quantized LLM Inference Engine** (W4A16).
This plan covers v0.1 only: a small, clear runtime plus **one Llama-style
decoder block**, correctness-first. No full model, no framework.

Scope is frozen per the project brief. Sections: Environment / Repository
state / Proposed architecture / Weight format / Tensor & KV layout / Kernel
port map / Golden-reference strategy / Milestones / Risks / First task.

---

## 1. Environment

| Item | Value |
|------|-------|
| Host GPU | NVIDIA RTX 2080 Ti (×2), **Turing sm_75**, 11 GB |
| Driver | 570.172.08 (CUDA runtime driver 12.8) |
| CUDA toolkit | **11.8.89** at `/usr/local/cuda-11.8` (`nvcc`, `compute-sanitizer` both present) |
| Compiler host | gcc (C++17), CMake **3.16.3** (see §8 risk R3) |
| Python (offline tools only) | 3.8.10 + **torch 2.3.1+cpu** (installed for `tools/`; never linked into runtime) |
| Target arch | `sm_75` (Turing). Kernels may use fp16 intrinsics (`__half`, `__half2`, `__half22float2`). |

Constraints honored:
- Runtime (compiled `libcudalm` + test/bench binaries) has **no PyTorch, no
  pybind, no `at::Tensor`, no `TORCH_CHECK`, no Python**. Kernels take raw
  pointers + `cudaStream_t`.
- Python exists **only** under `tools/` (weight conversion, golden
  reference, offline test data). It is not a runtime dependency.

---

## 2. Repository state

- Working dir: `/root/code/cudalm-agent` — was empty. Fresh `git init`,
  branch `main`. No existing history, no remote.
- Upstream reference (read-only, **not** merged into this repo): CUDALab at
  `/root/code/cuda`, HEAD `cb6a6a9` (tag `v0.7.1`). Only the incumbent
  kernels + INT4 quantization contract were read (see §6 port map).
- CUDALab stays the upstream kernel-research repo; CUDALM is the downstream
  inference-system repo. Ports record provenance (see §6 + `docs/provenance.md`).

Planned GitHub remote: `trump253/CUDALM`. If no `origin` is configured, work
locally and, at the end, tell the user:

```
git remote add origin <repo-url>
git push -u origin main
```

---

## 3. Proposed architecture

Small, explicit runtime. No framework, no registry, no auto-diff.

```
CUDALM/
├── CMakeLists.txt
├── README.md
├── docs/            # this plan, weight-format spec, provenance, notes
├── include/cudalm/
│   ├── cuda_check.h       # CUDA_CHECK(...) + launch-error helper
│   ├── tensor.h           # Dtype, Shape, TensorView (non-owning, contiguous-only)
│   ├── device_buffer.h    # RAII cudaMalloc/cudaFree, no implicit copy
│   ├── model_config.h     # BlockConfig (shapes, heads, GQA, head_dim, group)
│   ├── weight_format.h    # binary header/record structs + constants
│   ├── weight_loader.h    # parse + bounds-check + upload to DeviceBuffers
│   ├── kv_cache.h         # KV layout + write/read (RAII)
│   ├── ops.h              # operator entry points (native C++ API)
│   └── kernels/           # kernel decls: rmsnorm, int4gemv, rope, attention, kv, elementwise
├── src/
│   ├── runtime/           # runtime.cpp, weight_loader.cpp, kv_cache.cpp
│   ├── ops/               # op host-side launchers
│   └── kernels/           # .cu kernel implementations
├── tests/
│   ├── cpu/               # weight-file parser, tensor meta, int4 pack/offsets
│   ├── cuda/              # per-op unit tests vs CPU golden
│   ├── common/            # shared compare/alloc helpers
│   └── golden/            # stage-by-stage golden comparison driver
├── tools/
│   ├── convert_weights.py # build .cudalm weight file (quantize + pack)
│   ├── generate_golden.py # fixed-seed decoder-block case → golden file
│   └── common/            # shared python IO (cudalm binary reader)
├── benchmarks/
│   └── bench_block.cu     # CUDA-event latency breakdown per stage
└── scripts/               # build + test + sanitize helpers
```

Design rules:
- `DeviceBuffer`: RAII `cudaMalloc`/`cudaFree`, explicit `size`, no implicit
  host↔device copy. Move-only.
- `TensorView`: non-owning `{ptr, dtype, shape}`. v0.1 is **contiguous
  only**; stride is not modeled (asserted contiguous at construction).
- `CUDA_CHECK(...)`: single error layer. Every `cuda*` call and every kernel
  launch goes through it; a failed launch is checked via
  `cudaGetLastError()` immediately after.
- One explicit `cudaStream_t` owned by the runtime; **no** default-stream
  assumptions. All ops take the stream as a parameter.

### Decoder block (v0.1 fixed topology)

```
x ─▶ RMSNorm ─▶ [Q,K,V W4A16 Linear] ─▶ RoPE(Q,K) ─▶ KV write/read
    ─▶ causal Attention ─▶ W4A16 O-proj ─▶ (+x) residual
    ─▶ RMSNorm ─▶ [gate,up W4A16] ─▶ SiLU(gate)*up ─▶ W4A16 down ─▶ (+residual)
```

v0.1 config (single GPU, fp16 activations, W4A16 linears, one block). Concrete
numbers are chosen small for fast, deterministic tests:

| param | value |
|-------|-------|
| hidden `H` | 1024 |
| `n_heads` | 8 |
| `n_kv_heads` | 4 (GQA ratio 2; interface reserves GQA) |
| `head_dim` | 128 |
| `intermediate` (SwiGLU) | 2816 (= 1024*2.75, rounded to group mult) |
| `group_size` | 128 (fixed, W4A16 contract) |
| `max_seq_len` | 512 |
| `eps` (RMSNorm) | 1e-5 |
| `rope_theta` | 10000.0 |

`intermediate` must be a multiple of `group_size` (128) and of 2 (INT4
packing). 2816 = 22*128 ✓. All projection `K = H = 1024` (multiple of 128 ✓).

GQA mapping: query head `h` reads KV head `h / (n_heads/n_kv_heads)`.

---

## 4. Weight format

CUDALM owns a simple, versioned, **deterministic** binary container
(`.cudalm`). The loader is pure C++ (no PyTorch). Python `convert_weights.py`
**writes** it; `weight_loader.cpp` **reads** it. Both share the spec in
`docs/weight_format.md` (single source of truth).

Header (little-endian):

```
magic     : 8s  b"CUDLMW01"
version   : u32 = 1
flags     : u32 (reserved, 0)
n_tensors : u32
header bytes reserved for config:
  ModelConfig serialized (see §3) — fixed-width fields, LE
tensor table: n_tensors × TensorRecord
payload    : tensor byte blobs, each 16-byte aligned
```

`TensorRecord`:

```
name_len  : u32
name      : name_len bytes (ASCII, no NUL)
dtype     : u8   (FP16=1, INT4_PACKED=2, FP16_SCALE=3)
ndim      : u8
dims      : u32[ndim]
offset    : u64  (byte offset from payload start)
byte_size : u64  (payload size in bytes)
align     : u8   (required alignment of payload, power of 2, <=16)
pad       : u8[7]
```

Supports the three dtypes v0.1 needs:
- **FP16** tensor (RMSNorm weights, RoPE cos/sin, activations not stored)
- **INT4_PACKED** tensor (`W_packed`, uint8, two signed INT4 / byte)
- **FP16_SCALE** tensor (group scales `[N, K/128]`)

W4A16 pairing is by convention + validation: every `INT4_PACKED` weight
`X.weight` has a matching `X.scale` (`FP16_SCALE`) of shape `[N, K/128]`.
The loader validates names/shapes and rejects mismatches.

Loader guarantees:
- **Deterministic** parse (no randomness, stable ordering).
- **Versioned** (`version` field; unknown version → hard error).
- **Bounds checked** (every `offset+byte_size` within payload; table sizes
  consistent with declared counts; no overlap; alignment honored).
- **Explicit alignment** (payload 16B aligned; per-record `align` checked).
- **No PyTorch** in the loader path.

RoPE `cos/sin` tables are produced by the Python converter (fixed seed) and
stored as FP16 tensors so the runtime needs no transcendental at load time
(bandwidth/correctness deterministic).

---

## 5. Tensor / KV layout

All tensors contiguous row-major. Pointers passed to kernels are the raw
`DeviceBuffer` base (or validated offset).

- `TensorView{ void* data; Dtype dtype; std::vector<int64_t> shape; }`
  `.bytes()`, `.numel()`, `.contiguous()` (always true in v0.1).

**KV cache** (decode, single request). Simple, auditable layout:

```
K, V each: [n_kv_heads][max_seq_len][head_dim]  fp16
```

- `K_base + ((n * max_seq_len) + t) * head_dim + d`
- `t` = position (0-based), `n` = kv head, `d` = head dim.
- Write path: at decode position `p`, write the current `k[n, p, :]` and
  `v[n, p, :]` for all `n_kv_heads`. Read path: attention reads
  `[0..p]` inclusive for each kv head used by the query head.
- `KvCache` is RAII (two `DeviceBuffer`s: K and V), pre-allocated to
  `max_seq_len`. `write(position, k, v)` and read-only accessors.

This is intentionally flat (no paging, no ring). `position` bounds are checked
host-side (`0 <= position < max_seq_len`).

---

## 6. Kernel port map: CUDALab → CUDALM

Upstream reference: CUDALab `cb6a6a9` (tag `v0.7.1`). Only incumbents + the
INT4 contract were read. Provenance is recorded per kernel in
`docs/provenance.md` and as file headers (upstream tag/commit, original
kernel, CUDALM port commit).

| # | CUDALab upstream | Original kernel | CUDALM port | Changes |
|---|------------------|-----------------|-------------|---------|
| 1 | `kernels/rmsnorm/rmsnorm_v4.cu` | `rmsnorm_v4_half_kernel<PER>` (incumbent `v4_vec_reg`) | `src/kernels/rmsnorm.cu` → `rmsnorm_fp16` | Drop fp32 path (v0.1 is fp16-only). Replace `at::Tensor`/`TORCH_CHECK`/`getCurrentCUDAStream` with raw pointers + `cudaStream_t` + `CUDALM_CHECK`. Keep 256-thread block, `PER∈{4,8,16,32}`, `float4`/`half2` vector loads, single-pass register-resident x, warp+block reduce, `rsqrtf`. Keep 16B alignment contract (v0.1 H=1024 → PER=4 → half2 path, 4B align). |
| 2 | `kernels/rope/rope_v3_half2.cu` | `rope_v3_half2_kernel<__half>` + scalar fallback (incumbent `rope_v3_half2`) | `src/kernels/rope.cu` → `rope_fp16` | Same interleaved-pair contract, `__half2` pack/unpack, FP32 rotation, 4B-align contract + scalar fallback. Raw pointers + stream. |
| 3 | `kernels/int4gemv/int4gemv_rowtile4_hx.cu` + `int4gemv_common.h` | `int4gemv_rowtile4_hx_kernel` (incumbent) + shared `int4gemv_scalar_kernel` + `int4gemv_unpack_byte` + `U32I4` | `src/kernels/int4_gemv.cu` → `int4_gemv` | Port the R=4 half-resident x vectorized kernel **and** the shared scalar kernel (alignment-contract fallback, bit-identical per upstream). Replace at::Tensor/stream with raw pointers + `cudaStream_t`. Keep nibble contract (low=k=2b, high=k=2b+1, two's complement), `scale fp16 [N,K/128]`, G=128, FP32 accumulate, fp16 out. Host-side 16B align check → scalar fallback. |
| 4 | `cudalab/int4gemv_quantize.py` | `quantize_w` / `pack_q` / `unpack_w` (symmetric G=128, q∈[-7,7], scale=amax/7, fp16 store) | `tools/convert_weights.py` (Python, offline) | Port the quantization contract exactly (round-half-to-even, zero-group safe, nibble packing). Offline only — never in runtime. |

New CUDALM-native kernels (no upstream, written here, correctness-first):

| # | Kernel | Purpose |
|---|--------|---------|
| A | `add_fp16` (residual) | `y = a + b`, fp16 in/out, fp32 math |
| B | `silu_mul_fp16` | `y = silu(g)*u`, fp32 math, fp16 out |
| C | `kv_write_fp16` | scatter current K/V row into cache at `position` |
| D | `attention_decode_fp16` | correctness-first causal decode attention (multi-kernel: scores → softmax → PV), reads KV cache |
| E | `softmax_fp32` (internal to D) | row softmax over `[0..p]` |

### Attention v0.1 (correctness-first, decode)

Given query `q` for the current token at `position p` (one query vector per
query head), and KV cache filled at `[0..p]`:

```
scores[h, t] = dot(q_h, K[h_kv][t]) / sqrt(head_dim)      t in [0..p]
probs[h, t]  = softmax(scores[h])                          (causal: t<=p)
out_h        = sum_t probs[h,t] * V[h_kv][t]
```

Implementation: a small multi-kernel pipeline (no FlashAttention, no
persistent, no online-softmax tricks — clarity first):
1. **scores**: one thread per (query-head, t), or block-per-head; compute
   qk over `head_dim` with fp32 accum.
2. **softmax**: block-per-query-head over `[0..p]`, fp32, numerically stable
   (subtract max).
3. **PV**: one thread per (query-head, head_dim), reduce over `t∈[0..p]` with
   fp32 accum, fp16 store.

Must handle: `n_heads`, `n_kv_heads` (GQA mapping), `head_dim`, `position`,
and the flat KV layout above. Performance is explicitly **not** the stop
condition.

---

## 7. Golden-reference strategy

`tools/generate_golden.py` (PyTorch, fixed seed) generates **one** decoder-block
case and exports every intermediate stage. The C++ golden test compares
**stage-by-stage** (not just the final output).

Stages exported (all fp16 unless noted), with the same tensor names the
runtime uses:

```
input            x (H,)                       [seeded fp16]
rmsnorm1         (H,)
q,k,v            (n_heads*hd, / n_kv*hd, ...) pre-RoPE
rope_q, rope_k   post-RoPE
kv_state         pre-existing KV at positions < p  (if p>0; else empty)
attention_output (n_heads*hd,)  = o_proj input
output_projection( H,)  = o_proj(x)
residual1        input + output_projection
rmsnorm2         (H,)
gate, up         (inter,)
silu_gate_mul_up (inter,)
down             (H,)
final_output     residual1 + down
```

Plus metadata:
```
config (ModelConfig), weights / quantized weights (the .cudalm payload, or
referenced file), position p, rope_theta, eps.
```

For `p > 0` the generator also emits the **existing KV state** (positions
`[0..p-1]`) with a **non-all-zero** history so the KV-cache read path is
exercised. Two golden cases are produced: `p=0` and `p>0` (e.g. `p=7`), both
with non-trivial KV for the latter.

Comparison: C++ golden test loads the golden file, replays the block, and
compares each stage with a tolerance appropriate to fp16 (report max-abs and
relative error per stage; gate on a fixed tolerance, e.g. max-abs ≤ 1e-2 and
relative ≤ 1e-2 for fp16 stages, tuned and pinned in the test). The golden
file format is a small, deterministic container (reuses the CUDALM binary
format with a "golden" flag or a sidecar JSON manifest + `.cudalm` payload).

The generator is **offline** (PyTorch allowed here) and is never imported by
the C++ runtime.

---

## 8. Risks

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | CUDALab kernels are entangled with PyTorch (`at::Tensor`, `TORCH_CHECK`, `getCurrentCUDAStream`, `CUDAGuard`). Naive copy would drag the forbidden deps in. | Port kernel **bodies** only; rewrite the host launch wrappers to raw pointers + `cudaStream_t` + `CUDALM_CHECK`. Add a build/grep guard that fails if `torch/`, `pybind`, `at::`, `TORCH_` appear in `src/` or `include/`. |
| R2 | Golden numeric drift between PyTorch (CPU fp32/fp64) and CUDA fp16 kernels (different reduction order, fp16 rounding). | Compare per-stage with a pinned, documented fp16 tolerance (not bit-exact). Accumulate in fp32 in kernels, cast to fp16 only at store — mirrors PyTorch. Keep reduction shapes identical where feasible. |
| R3 | System CMake is **3.16.3** — older than many CUDA CMake modules expect. | Use minimal, well-supported CMake (plain `add_library`/`add_executable`, `enable_language(CUDA)` if available else a manual `nvcc` compile rule). Pin arch to `sm_75`. Avoid CMake ≥3.18-only features. Verify the build actually runs on 3.16.3. |
| R4 | `compute-sanitizer` may be absent from `PATH` (toolkit at `/usr/local/cuda-11.8`). | Reference it by absolute path in `scripts/`; if truly unavailable, note it in the final report rather than fake it. |
| R5 | INT4 nibble packing / sign-extension is a classic source of silent bugs. | Port CUDALab's `int4gemv_unpack_byte` **unchanged** (it is pinned by upstream's 3-layer correctness). Add a dedicated CPU test for pack/unpack round-trip over the full `[-8,7]` domain (mirrors upstream `tests/test_int4gemv_cpu.py`). |
| R6 | KV/attention indexing errors (GQA mapping, position bounds, off-by-one). | Flat, explicitly-documented KV layout (§5); host-side bounds checks; golden test covers `p=0` and `p>0` with non-zero history; unit test for the GQA head mapping. |
| R7 | Scope creep toward a full model / premature perf tuning. | Stop conditions (§brief §13) are the gate. Attention stays correctness-first; benchmarks measure only, they do not change the algorithm. |

---

## 9. Milestones

Each milestone = **correctness → unit test → commit**. Small, ordered commits;
no force-push; no rewriting published evidence.

| MS | Deliverable | Gate |
|----|-------------|------|
| M0 | Repo bootstrap: `git init`+`main`, CMake skeleton, `cuda_check.h`, `device_buffer.h`, `tensor.h`, `model_config.h`; first commit `chore: bootstrap CUDALM native runtime` | CMake configures; empty lib compiles; CPU smoke test passes |
| M1 | Weight format spec + `weight_format.h` + `weight_loader.cpp` + CPU parser tests; `tools/convert_weights.py` writes a valid file | CPU tests: parse, bounds, alignment, dtype, int4 offsets all green |
| M2 | `tools/generate_golden.py` + golden file IO + stage-compare harness | Golden file produced for p=0 and p>0; harness loads & lists stages |
| M3 | Elementwise ops (`add_fp16`, `silu_mul_fp16`) + **RMSNorm port** + unit tests | RMSNorm unit test vs CPU golden green; op tests green |
| M4 | **W4A16 int4 GEMV port** (`int4_gemv`, hx + scalar fallback) + unit tests | GEMV unit test vs CPU dequant reference green (incl. pack/unpack CPU test) |
| M5 | **RoPE port** + unit tests | RoPE unit test vs CPU golden green |
| M6 | **KV cache** + `kv_write_fp16` + unit tests (p=0 and p>0, non-zero history) | KV write/read round-trip green; bounds checks green |
| M7 | **Causal decode Attention** (scores→softmax→PV) + unit tests | Attention unit test vs CPU golden green (p=0, p>0) |
| M8 | **Full DecoderBlock** wiring (runtime) + stage-by-stage golden PASS | Golden comparison PASS for p=0 and p>0 |
| M9 | `compute-sanitizer` clean on DecoderBlock + CUDA-event **block latency breakdown** | Sanitizer summary saved; benchmark JSON with per-stage + total times |
| M10 | README (architecture + build + reproduce), final stop-condition checklist, `git clean` | All §13 stop conditions checked; tree clean |

Integration order inside M3–M8 follows the brief's kernel/operator order:
elementwise → RMSNorm → W4A16 → RoPE → KV → Attention → SiLU → Block.

---

## 10. First implementation task

**M0 — Bootstrap:**
1. `git init` + `git branch -M main` in `/root/code/cudalm-agent`.
2. Write `CMakeLists.txt` (minimal, sm_75, builds `libcudalm` from `src/`).
3. Add `include/cudalm/cuda_check.h` (`CUDA_CHECK` + launch check),
   `device_buffer.h` (RAII), `tensor.h` (Dtype/Shape/TensorView, contiguous),
   `model_config.h` (BlockConfig struct + validation).
4. Add a trivial CPU smoke test (`tests/cpu/test_bootstrap.cpp`) and a
   `tests/CMakeLists`-free standalone compile via CMake `enable_testing()`.
5. Commit `chore: bootstrap CUDALM native runtime`.

Then proceed straight into M1–M10 without per-step approval; stop only if a
decision would change a core contract/architecture (e.g. weight-format
layout, KV layout, or the W4A16 numeric contract) — those are already fixed
here and in the brief.
