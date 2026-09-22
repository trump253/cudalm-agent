# CUDALM

Native C++/CUDA inference engine for quantized LLMs. **v0.1** implements one
complete Llama-style decoder block — W4A16-quantized linears, fp16
activations, single-GPU autoregressive decode — with **no PyTorch (or any
Python) dependency in the runtime**. C++17, CUDA 11.8, `sm_75`
(RTX 2080 Ti). Kernels take raw pointers + `cudaStream_t`; Python lives only
in `tools/` (offline fixture generation) and never links into the runtime.

## v0.1 scope — and what is deliberately NOT here

Implemented, end to end:

```
x ─► RMSNorm ─► Q/K/V (W4A16 GEMV) ─► RoPE (interleaved pairs)
  ─► KV-cache write @ position ─► causal decode attention (fp32 math)
  ─► O-proj ─► + residual ─► RMSNorm ─► gate/up (W4A16 GEMV)
  ─► SiLU(gate)·up ─► down (W4A16 GEMV) ─► + residual ─► y
```

Deliberately out of scope for v0.1 (hard stop after v0.1 — see checklist
below): FlashAttention, batching, tensor parallelism, multi-GPU, CUDA Graphs,
tokenizer, full multi-layer model.

## Pinned contracts

| Item | Contract |
|------|----------|
| Config | `H=1024, n_heads=8, n_kv_heads=4, head_dim=128, intermediate=2816, group=128, max_seq=512, eps=1e-5, rope_theta=10000` |
| Weights | W4A16: symmetric group-wise INT4, G=128, q∈[−7,7], zero_point=0, `scale=amax/7` fp32→stored fp16 (**the stored fp16 scale is the contract**), nibble pack low=k=2b / high=k=2b+1 (4-bit two's complement). Format spec: [`docs/weight_format.md`](docs/weight_format.md) |
| KV cache | K, V each fp16 `[n_kv_heads][max_seq_len][head_dim]`; row (n,t) offset `((n*max_seq_len)+t)*head_dim`; zero-initialized; bounds-checked (fatal preconditions) |
| RoPE | Interleaved pairs: `y[2i]=a·c−b·s`, `y[2i+1]=a·s+b·c`; cos/sin tables fp16 `[max_seq, hd/2]`; per-row `positions[m]` lookup; packed 4B path when base 4B-aligned, scalar fallback otherwise (never rejects) |
| Attention | GQA integer mapping `kh = h * n_kv // n_heads`; `scale = 1/√head_dim`; max-subtracted softmax with elementwise divide; **all math fp32, single fp16 RNE at the store** |
| Stage tolerance | `|a−r| ≤ 1e-2 + 1e-2·|r|` vs golden (observed max ≤ 2.4e-4: kernel FMA contraction vs golden's separate mul+add). p=0 rope identity, all 18 weight tensors, and seeded KV-history rows are pinned **bit-exact**; the current-position KV row is tolerance-vs-golden + bit-exact vs the runtime's own rope_k/v stage rows |

## Repository layout

```
include/cudalm/           runtime headers (decoder_block.h, kv_cache.h,
                          weight_format.h, kernels/{rmsnorm,int4_gemv,rope,
                          kv,attention,elementwise}.h, device_buffer.h, ...)
src/kernels/              .cu kernels (raw pointer + cudaStream_t)
src/runtime/              decoder_block.cpp, kv_cache.cpp, weight_loader.cpp,
                          golden_loader.cpp, device_buffer.cpp
tests/cpu/                no-GPU tests: file formats, crosslang weight check
tests/cuda/               kernel + end-to-end golden tests (16 ctest tests)
tools/                    OFFLINE python (torch 2.3.1+cpu):
                          convert_weights.py (quantizer), generate_golden.py,
                          common/binfmt.py — never linked into the runtime
benchmarks/               bench_decoder_block.cpp + committed results +
                          sanitizer evidence
scripts/check_no_torch.sh guard: hard-fails on any torch/pybind/py symbol
                          in include/ + src/
docs/                     bootstrap_plan_v0.1.md, provenance.md,
                          weight_format.md
```

**Provenance.** Three kernels are line-for-line ports from the upstream
CUDALab research repo (frozen at `cb6a6a9`, tag `v0.7.1`, read-only):
`rmsnorm` (v4 fp16), `int4_gemv` (rowtile4_hx + scalar fallback), `rope`
(v3 half2, interleaved-pair). The host layer (PyTorch extension API) is
replaced by raw pointers + `cudaStream_t` + `CUDA_CHECK`; kernel math and
control flow are preserved 1:1. Everything else — KV cache, attention
pipeline, decoder-block wiring, weight format, tests, benchmarks — is
CUDALM-native. Full per-file table with deviations:
[`docs/provenance.md`](docs/provenance.md).

## Build

Requirements: CMake ≥ 3.16, CUDA 11.8 toolkit (`nvcc` for `sm_75`), a C++17
compiler, and (only to *generate test fixtures*, not to build) python3 with
`torch` (CPU build is fine). The build itself links no Python.

```sh
cmake -S . -B build
cmake --build build -j
```

## Reproduce the evidence

All fixtures are generated automatically by `ctest` from `tools/` into
`build/data/` (seeded, deterministic: shared seed `20250922`, history seed
`20250923`).

**1. Full test suite (16/16):**

```sh
cd build && ctest
```

Covers: file-format round-trips, crosslang weight byte-equality (Python
quantizer → C++ loader), per-kernel tests (rmsnorm, int4_gemv, rope,
kv_cache, attention, elementwise), and the end-to-end golden test
(`test_decoder_block`, positions 0 and 7: 18 weight tensors + 16 stage
tensors + KV state, per the pinned contracts above).

**2. Per-stage latency breakdown** (CUDA events, 17 stages + total):

```sh
./build/benchmarks/bench_decoder_block build/data/block_v01.cudalm <position> [iters=100] [out.json]
```

Committed results (`benchmarks/results/`, RTX 2080 Ti, CUDA 11.8, driver
570.172.08, iters=100):

| position | total mean | tokens/s |
|----------|-----------|----------|
| 0   | 126.6 µs | 7 897 |
| 511 | 152.4 µs | 6 562 |

Attention is the only position-sensitive stage (13.0 µs @ p=0 → 41.5 µs @
p=511); the block is GEMV-dominated (7 quantized projections ≈ 100 µs).

**3. Memory sanitizer** (evidence committed at
`benchmarks/sanitizer_decoder_block.txt`; rerun):

```sh
cd build
/usr/local/cuda-11.8/bin/compute-sanitizer --tool memcheck \
  tests/test_decoder_block data/block_v01.cudalm \
  data/block_v01_golden_p0.cudalm data/block_v01_golden_p7.cudalm
# → ERROR SUMMARY: 0 errors
```

**4. No-torch runtime guard:**

```sh
bash scripts/check_no_torch.sh
# → forbidden_deps_check OK
```

## v0.1 stop-condition checklist

- [x] One Llama-style decoder block runs end-to-end on real (seeded) weights
- [x] End-to-end golden PASS at p=0 and p=7 (1e-2 tolerance; bit-exact pins
      per contract: rope identity, 18 weight tensors, seeded KV history)
- [x] 16/16 ctest green (CPU + CUDA)
- [x] `compute-sanitizer --tool memcheck`: 0 errors
- [x] CUDA-event per-stage latency breakdown; p=0 / p=511 JSONs committed
- [x] Runtime PyTorch/Python-free (guard script, `tools/` only)
- [x] Provenance documented per file (upstream ports + native list)
- [x] Working tree clean; all evidence artifacts committed; no remote yet
      (attach instructions in the delivery report); no force push

**STOP.** Per the brief, development stops after v0.1. Nothing above was
scoped beyond the single decoder block.
