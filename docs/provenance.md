# CUDALM kernel provenance

CUDALM ports kernels from the upstream CUDALab research repository
(`/root/code/cuda`). This file records the exact upstream provenance for every
ported kernel, per the project brief (upstream tag/commit, original kernel,
CUDALM port commit).

**Upstream reference (frozen for v0.1):**

| Item | Value |
|------|-------|
| Repo | CUDALab (`/root/code/cuda`) |
| Commit | `cb6a6a9ef76394cc66d272c99aa8697db0a34f1e` |
| Tag | `v0.7.1` |
| Role | Upstream kernel research repo (read-only reference; NOT merged into CUDALM) |

Ports replace the PyTorch extension host layer (`at::Tensor`, `TORCH_CHECK`,
`at::cuda::getCurrentCUDAStream`) with CUDALM's native layer (raw pointers,
`cudaStream_t`, `CUDA_CHECK`). Kernel **math and control flow** are preserved;
any deviation from the upstream body is called out in the table.

| CUDALM file | Upstream file (commit `cb6a6a9`) | Original kernel | CUDALM port commit | Deviations |
|-------------|----------------------------------|-----------------|--------------------|------------|
| _(filled in as ports land)_ | | | | |

## Carried contracts (offline tooling, not kernel ports)

Data contracts implemented in offline Python tooling (tools/ never links into
the runtime):

| CUDALM file | Upstream file (commit `cb6a6a9`) | Contract carried | Notes |
|-------------|----------------------------------|------------------|-------|
| `tools/convert_weights.py` (`quantize_w`, `unpack_w`) | `cudalab/int4gemv_quantize.py` (`quantize_w`, `pack_q`, `unpack_w`) | symmetric G=128 group-wise INT4; q ∈ [-7,7]; zero_point=0; `scale=amax/7` computed fp32, stored fp16; round-half-to-even; zero-group safe (scale=0, q≡0); K%128==0; nibble pack low=k=2b / high=k=2b+1 (4-bit two's complement) | Line-for-line port of the offline quantizer; pinned by `convert_weights.py --selftest` (Python) and `test_weights_crosslang` (C++ loads the Python-generated file) |

## CUDALM-native (no upstream)

These have no CUDALab provenance; they are written from the v0.1 spec:
`cuda_check.h`, `device_buffer.h`, `tensor.h`, `model_config.h`, weight
format/loader, KV cache, attention pipeline, elementwise ops (add, silu_mul).
