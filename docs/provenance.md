# CUDALM kernel 溯源

CUDALM 从上游 CUDALab 研究仓库（`/root/code/cuda`）移植 kernel。本文件按
项目简报要求，为每个移植 kernel 记录确切的 upstream 溯源（上游
tag/commit、原始 kernel、CUDALM 移植 commit）。

**上游参考（v0.1 冻结）：**

| 项 | 值 |
|------|-------|
| 仓库 | CUDALab（`/root/code/cuda`） |
| Commit | `cb6a6a9ef76394cc66d272c99aa8697db0a34f1e` |
| Tag | `v0.7.1` |
| 角色 | 上游 kernel 研究仓库（只读参考；**不**合并进 CUDALM） |

移植内容是把 PyTorch extension host 层（`at::Tensor`、`TORCH_CHECK`、
`at::cuda::getCurrentCUDAStream`）替换为 CUDALM 原生层（裸指针、
`cudaStream_t`、`CUDA_CHECK`）。kernel 的**数学与控制流**保持不变；凡与
upstream 函数体的任何偏差，均在表中标注。

| CUDALM 文件 | 上游文件（commit `cb6a6a9`） | 原始 kernel | CUDALM 移植 commit | 偏差 |
|-------------|----------------------------------|-----------------|--------------------|------------|
| `src/kernels/rmsnorm.cu`、`include/cudalm/kernels/rmsnorm.h`（`rmsnorm_fp16`） | `kernels/rmsnorm/rmsnorm_v4.cu`（`rmsnorm_v4_half_kernel` + `launch_half` + `v4_precheck` fp16 分支） | `rmsnorm_v4`（v4_vec_reg）fp16 特化 | `80e4a7b` | 仅机械性替换：PyTorch host 层（`at::Tensor`、`TORCH_CHECK`、`getCurrentCUDAStream`、`C10_CUDA_KERNEL_LAUNCH_CHECK`）换为裸指针 + `cudaStream_t` + `CUDALM_PRECONDITION` + `CUDA_CHECK_LAUNCH`；fp32 特化不移植（v0.1 仅 fp16）。kernel 数学/控制流 1:1 保留（256 线程块、寄存器驻留 x、`PER = H/256` ∈ {4,8,16,32}、float4/half2 向量加载、warp shfl + `rsqrtf(v/H + eps)`、fp16 RNE 存储），包括"严格前置检查、无标量回退"的契约 |
| `src/kernels/int4_gemv.cu`、`include/cudalm/kernels/int4_gemv.h`（`int4_gemv`） | `kernels/int4gemv/int4gemv_rowtile4_hx.cu`（`int4gemv_rowtile4_hx_kernel` + `int4gemv_rowtile4_hx_fwd`）+ `kernels/int4gemv/int4gemv_common.h`（`int4gemv_unpack_byte`、`U32I4`、`int4gemv_vec_acc_unpack`、`int4gemv_scalar_kernel`、`launch_int4gemv_scalar`、`int4gemv_vec_contract_ok`） | `int4gemv_rowtile4_hx`（W4A16 GEMV）+ 其标量回退（`int4gemv_baseline` 核心） | `ca1e8e4` | 仅机械性替换：PyTorch host 层换为裸指针 + `cudaStream_t` + `CUDALM_PRECONDITION` + `CUDA_CHECK_LAUNCH`；`int64_t` 维度 → `int`（仅循环边界，无数学影响）；`int4gemv_vec_contract_ok` 内联进 host 入口。kernel 数学/控制流 1:1 保留 —— R=4 行分块、128 线程块、strided v 循环、4× LDG.128 x 片段存于 `__half2[16]`、单组引理 `g = v>>2`、每行 32 次 nibble 解包 + 32 次 (MUL+FFMA)、shfl 5 步 + `shared[4][4]` + shfl 2 步归约；标量回退（256 线程、一行一块）与 16B 对齐契约（回退、从不拒绝）均保留 |
| `src/kernels/rope.cu`、`include/cudalm/kernels/rope.h`（`rope_fp16`） | `kernels/rope/rope_v3_half2.cu`（`rope_v3_half2_kernel<__half>` + `rope_v3_half2_scalar_kernel` + `rope_v3_half2_fwd` fp16 分支）+ `kernels/rope/rope_common.h`（交错对算子契约、`el_to_float`/`el_from_float<__half>`） | `rope_v3_half2` fp16 特化（交错对 RoPE） | `d46142f` | 仅机械性替换：PyTorch host 层换为裸指针 + `cudaStream_t` + `CUDALM_PRECONDITION` + `CUDA_CHECK_LAUNCH`；fp32 模板分支（`T = float`）不移植（v0.1 仅 fp16），故 kernel 去模板化为其 `__half` 分支原文；`el_to_float`/`el_from_float` 辅助函数内联进本文件。kernel 数学/控制流 1:1 保留 —— 每对 1 线程、按行查 `positions[m]` 表、交错对 `a*c - b*s` / `a*s + b*c` 以 fp32 计算、打包路径（4B `__half2` 加载 + `__floats2half2_rn` + 4B 存储）、标量回退（全 2B 访问、逐值 RNE）、v0.4.1 基址对齐契约（回退、从不拒绝）均保留 |

## 承接的契约（离线工具，非 kernel 移植）

在离线 Python 工具中实现的数据契约（tools/ 绝不链接进运行时）：

| CUDALM 文件 | 上游文件（commit `cb6a6a9`） | 承接的契约 | 说明 |
|-------------|----------------------------------|------------------|-------|
| `tools/convert_weights.py`（`quantize_w`、`unpack_w`） | `cudalab/int4gemv_quantize.py`（`quantize_w`、`pack_q`、`unpack_w`） | 对称 G=128 分组 INT4；q ∈ [-7,7]；zero_point=0；`scale=amax/7` 以 fp32 计算、fp16 存储；四舍六入五成双（round-half-to-even）；零组安全（scale=0，q≡0）；K%128==0；nibble 打包 low=k=2b / high=k=2b+1（4 位补码） | 离线量化器的逐行移植；由 `convert_weights.py --selftest`（Python）与 `test_weights_crosslang`（C++ 加载 Python 生成的文件）钉死 |

## CUDALM 原生（无上游）

以下内容没有 CUDALab 溯源，按 v0.1 规范新写：`cuda_check.h`、
`device_buffer.h`、`tensor.h`、`model_config.h`、权重格式/加载器、
KV 缓存、注意力流水线、逐元素算子（add、silu_mul）、decoder block 接线
（`decoder_block.h/.cpp`，代码 commit `beeebb6`）及其端到端黄金测试
（`tests/cuda/test_decoder_block.cpp`）、逐 stage CUDA 事件计时 API
（`DecoderBlock::forwardTimed` / `stage_names`，代码 commit `bc7e430`）、
时延 benchmark（`benchmarks/bench_decoder_block.cpp`）及其 memcheck 证据
（`benchmarks/sanitizer_decoder_block.txt`）与结果 JSON
（`benchmarks/results/`）。

## CUDALM v0.2 Phase B（Qwen3.5 Full Attention，BF16 运行时路径）

Phase B 在 v0.1.1 之上为 Qwen3.5 全注意力层新增 BF16 运行时路径
（W4A16 权重契约不变：G=128、q∈[-7,7]、scale FP16、FP32 累加；
激活/输出走 BF16，**绝不静默转 FP16**）。事实源 = 官方 pinned
transformers（`fc9137225880`，`modeling_qwen3_5.py` sha256
`b6f02dcd1b66610df293084e00bf9bea4fc6a7e5336ffc6ff446edc7ddcd8601`，
生成器运行时断言）+ 真实 checkpoint（`Qwen/Qwen3.5-0.8B-Base`，
revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`，
sha256 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`）。

### 上游移植（CUDALab `cb6a6a9`，只读参考）

| CUDALM 文件 | 上游文件 | 原始 kernel | 偏差 |
|-------------|----------|-------------|------|
| `src/kernels/int4_gemv_bf16.cu`、`include/cudalm/kernels/int4_gemv_bf16.h` | `kernels/int4gemv/int4gemv_rowtile4_hx.cu` + `int4gemv_common.h`（与 v0.1 `int4_gemv` 同源，同 `ca1e8e4` 移植谱系） | `int4gemv_rowtile4_hx`（W4A16 GEMV）+ 标量回退 | 数学/控制流 1:1 保留（R=4 行分块、128 线程、单组引理 `g=v>>2`、32 次 nibble 解包 + FFMA、shfl 归约、16B 对齐回退契约）；唯一差异 = 激活/输出 dtype 从 `__half` 换为 `__nv_bfloat16`（每行 fp32 累加后单次 bf16 RNE 存储，对应官方 `F.linear` 的 bf16 输出舍入）。CUDA 11.8 无 `__bfloat1622float2` 内建，新增 `bf162_to_float2(__nv_bfloat162)` 辅助（纯位运算，无数学偏差） |

### CUDALM 原生（无上游，依据 pinned modeling 新写）

- `src/kernels/qwen35_kernels.cu` / `include/cudalm/kernels/qwen35_kernels.h`：
  零中心 RMSNorm（`x·(1+w)`，fp32 链 → 单次 bf16）、q/gate 拆分、
  partial RoPE（§5 复制频率布局 `freqs[d%32]`，cos/sin fp32 表 → 乘前 bf16）、
  KV 写入、decode 注意力（bf16 QK/PV matmul、fp32 累加、fp32 max-subtract
  softmax、精确 `1/16` 缩放）、BF16 add/silu_mul/gate_mul。
- `src/runtime/qwen35_kv_cache.{h,cpp}`、`src/runtime/qwen35_full_attention.{h,cpp}`
  （21-stage 层 + `forwardTimed` 44 事件 + 宿主 libm RoPE 表，与 CPU golden
  位级一致）、`src/runtime/golden_loader_v2.{h,cpp}`。
- `tools/common/golden_v2.py`（CUDLMG02 读写，与 C++ loader 字节级互认）、
  `tools/generate_qwen35_golden.py`（oracle：pinned transformers 官方层 +
  同一 W4A16 反量化权重；`--selftest` 无需 checkpoint；
  `--fidelity-report` 只报告量化保真度，不进硬门）。
- 测试：`tests/cuda/test_int4_gemv_bf16.cpp`、
  `tests/cuda/test_qwen35_kernels.cpp`、
  `tests/cuda/test_qwen35_full_attention_golden.cpp`
  （真实 checkpoint 硬门：p=0 + p=5 带 KV 历史，21 stage + KV 状态）；
  基准 `benchmarks/bench_qwen35_full_attention.cpp`
  （结果 `benchmarks/results/bench_qwen35_full_attention_p{0,5}.json`）。

验收证据（2026-09-23，RTX 2080 Ti）：硬门 p=0 最差 6.1e-05、p=5 最差
4.9e-04（tolerance 1e-2，`include/cudalm/stage_compare.h` 记录容差理由）；
p=0 不变式位级成立；27/27 ctest；compute-sanitizer memcheck 0 错误；
no-torch 守卫 CLEAN。详见 `docs/qwen35_architecture.md` §14。
