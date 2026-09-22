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
| `src/kernels/rmsnorm.cu`, `include/cudalm/kernels/rmsnorm.h` (`rmsnorm_fp16`) | `kernels/rmsnorm/rmsnorm_v4.cu` (`rmsnorm_v4_half_kernel` + `launch_half` + `v4_precheck` fp16 branch) | `rmsnorm_v4` (v4_vec_reg) fp16 specialization | `80e4a7b` | Mechanical only: PyTorch host layer (`at::Tensor`, `TORCH_CHECK`, `getCurrentCUDAStream`, `C10_CUDA_KERNEL_LAUNCH_CHECK`) replaced by raw pointers + `cudaStream_t` + `CUDALM_PRECONDITION` + `CUDA_CHECK_LAUNCH`; the fp32 specialization is not ported (v0.1 is fp16-only). Kernel math/control flow (256-thread block, register-resident x, `PER = H/256` ∈ {4,8,16,32}, float4/half2 vector loads, warp shfl + `rsqrtf(v/H + eps)`, fp16 RNE store) is preserved 1:1, including the strict pre-check with no scalar fallback |
| `src/kernels/int4_gemv.cu`, `include/cudalm/kernels/int4_gemv.h` (`int4_gemv`) | `kernels/int4gemv/int4gemv_rowtile4_hx.cu` (`int4gemv_rowtile4_hx_kernel` + `int4gemv_rowtile4_hx_fwd`) + `kernels/int4gemv/int4gemv_common.h` (`int4gemv_unpack_byte`, `U32I4`, `int4gemv_vec_acc_unpack`, `int4gemv_scalar_kernel`, `launch_int4gemv_scalar`, `int4gemv_vec_contract_ok`) | `int4gemv_rowtile4_hx` (W4A16 GEMV) + its scalar fallback (`int4gemv_baseline` core) | `ca1e8e4` | Mechanical only: PyTorch host layer replaced by raw pointers + `cudaStream_t` + `CUDALM_PRECONDITION` + `CUDA_CHECK_LAUNCH`; `int64_t` dims → `int` (loop bounds only, no math effect); `int4gemv_vec_contract_ok` inlined into the host entry. Kernel math/control flow preserved 1:1 — R=4 row tile, 128-thread block, strided v loop, 4× LDG.128 x fragments held as `__half2[16]`, single-group lemma `g = v>>2`, 32 nibble unpack + 32 (MUL+FFMA) per row, shfl 5-step + `shared[4][4]` + shfl 2-step reduction; scalar fallback (256-thread, one block/row) and the 16B-alignment contract (fall back, never reject) preserved |

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
