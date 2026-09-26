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

> 注：上述两份已提交结果 JSON 中 `whole_layer_gpu_us` 的语义描述属于
> **历史描述**（为保持历史 benchmark artifact 不可变而原样保留，未改动
> 数值或文本）；当前权威 timing 语义以基准生成器
> `benchmarks/bench_qwen35_full_attention.cpp` 与
> `docs/qwen35_architecture.md` §14.3 为准。

验收证据（2026-09-23，RTX 2080 Ti）：硬门 p=0 最差 6.1e-05、p=5 最差
4.9e-04（tolerance 1e-2，`include/cudalm/stage_compare.h` 记录容差理由）；
p=0 不变式位级成立；27/27 ctest；compute-sanitizer memcheck 0 错误；
no-torch 守卫 CLEAN。详见 `docs/qwen35_architecture.md` §14。

## CUDALM v0.2 Phase C（Qwen3.5 Gated DeltaNet，BF16 运行时路径）

Phase C 在 Phase B 之上为 Qwen3.5 Gated DeltaNet 层新增 BF16 解码运行时
（W4A16 权重契约不变：G=128、q∈[-7,7]、scale FP16、FP32 累加；激活/输出
走 BF16，recurrent 状态保持 FP32，**绝不** bf16/量化状态混用）。事实源 =
官方 pinned transformers（`fc9137225880`，`modeling_qwen3_5.py` sha256
`b6f02dcd1b66610df293084e00bf9bea4fc6a7e5336ffc6ff446edc7ddcd8601`，
生成器运行时断言）+ 真实 checkpoint（`Qwen/Qwen3.5-0.8B-Base`，同 Phase
B revision / sha256）。

### 上游核查（CUDALab `cb6a6a9`，只读参考）

已核查 frozen `CUDALab@cb6a6a9`：**无** proven DeltaNet / causal-conv1d /
recurrent kernel。`kernels/` 仅含 `gemv`、`int4gemv`、`qgemv`、`rmsnorm`、
`rope`、`softmax`；仓库内 `delta` / `recurrent` / `conv` 字符串命中均为误报
（`cudalab/evaluator` 的 `FILTER_LOG_DELTA` 阈值常量、注释中的 "convert"）。
**因此 Phase C 的 DeltaNet 解码 kernel 为 CUDALM 原生（无上游移植），数学
语义来自 pinned Qwen3.5 实现**的纯 torch 回退路径
（`torch_recurrent_gated_delta_rule` / `torch_causal_conv1d_update` /
FLA 对齐 `l2norm`，架构文档 §1.2 钉死）。

### CUDALM 原生（无上游，依据 pinned modeling 新写）

- `src/kernels/qwen35_deltanet_kernels.cu` /
  `include/cudalm/kernels/qwen35_deltanet_kernels.h`：causal conv decode
  （fp32 累加 → 一次 bf16 RNE → 对 bf16 结果 SiLU；conv_state 位级
  shift+insert）、g/beta 预计算（g fp32、beta bf16）、FLA 对齐 **bf16
  l2norm** + fp32 recurrent 状态就地递推（输出取自更新后状态；bf16 rsqrt =
  `bf16(1.0 / bf16(sqrt(f32(t))))`，见架构文档 §8 修正）、gated RMSNorm
  （两处 bf16 舍入）。
- `src/runtime/qwen35_deltanet.{h,cpp}`：`Qwen35DeltaNetLayer`（复用
  Phase A/B W4A16 GEMV + 零中心 RMSNorm + add + silu-mul；持有 conv_state
  bf16 `[6144,3]` + recurrent_state fp32 `[16,128,128]`，就地更新，
  `reset_state` / `seed_state`）+ `qwen35_deltanet_require_supported_config`
  （supported-config 硬契约：kernel-4 / state-3 / head-dim-128 / k==v，
  非法配置 abort）。
- `tools/generate_qwen35_golden.py`（DeltaNet 分支 + `--state-seed` 确定性
  非零初始状态）、`tools/common/golden_v2.py` +
  `src/runtime/golden_loader_v2.cpp`（CUDLMG02 DeltaNet 张量集：23 stage +
  4 state）。
- 测试：`tests/cpu/test_qwen35_deltanet_config.cpp`（supported-config
  契约，子进程 abort-check）、`tests/cuda/test_qwen35_deltanet_golden.cpp`
  （真实 checkpoint 硬门：首 token / 连续 / 非零初始状态，均比对输出 +
  conv_state + recurrent_state；`--no-gen` 供 memcheck）。

> 注：Phase C **复用** Phase B 已移植的 `int4_gemv_bf16`（W4A16 GEMV，
> `ca1e8e4` 谱系）作为 4 个 in_proj + out_proj + MLP 的 GEMV，**不是**新的
> 上游移植；新增的仅是 DeltaNet 专属 kernel 与层接线。

验收证据（RTX 2080 Ti）：三场景硬门全 PASS（A bit-exact、B/C 1 ulp；
recurrent_state ≤ 5.96e-08）；29/29 ctest；compute-sanitizer memcheck 0
错误（`benchmarks/sanitizer_qwen35_deltanet.txt`）；no-torch 守卫 CLEAN。
详见 `docs/qwen35_architecture.md` §15。

## CUDALM v0.2 Phase D（Qwen3.5 4 层混合 micro-stack）

Phase D 在 Phase C 之上把**真实 checkpoint 前 4 层**（layer 0/1/2 = Gated
DeltaNet，layer 3 = 全注意力）组成 **v0.2 最终混合 decoder micro-stack**。
**无新 kernel、无新上游移植**：micro-stack 是纯接线（逐层 dispatch + 独立
持久状态 + 整 stack 时序钩子），数学语义完全来自**复用的冻结 Phase B/C 单层
运行时**（其数学 = pinned Qwen3.5 实现，见 Phase B/C 溯源）。事实源同 Phase
B/C：官方 pinned transformers（`fc9137225880`）+ 真实 checkpoint
（`Qwen/Qwen3.5-0.8B-Base`）。

### CUDALM 原生（无上游）

- `src/runtime/qwen35_hybrid_microstack.cpp` /
  `include/cudalm/qwen35_hybrid_microstack.h`：`Qwen35HybridMicroStack`——
  `load`（逐层 `validate_layer` + 按 config hybrid 排表 dispatch 到
  `Qwen35DeltaNetLayer` / `Qwen35FullAttentionLayer`）、`forward`（0→1→2→3
  链，每层 final output 喂下一层；输出 = layer 3 final）、`reset_state`
  （逐层独立重置，**无跨层状态 alias/reuse**）、`forwardTimed`（4 对 per-layer
  + 1 对整 stack 事件）。**仅编排，不复制任何单层 kernel**；每层持久状态由
  该层自己持有（DeltaNet conv+recurrent、全注意力 KV），micro-stack 只做接线。
- `tools/generate_qwen35_golden.py`（`--microstack-prefix` / `--tokens` →
  `generate_microstack`）：oracle 顺序跑 layers 0→1→2→3（**同一组** W4A16
  量化权重，无 bf16/量化混用），逐 (token, layer) 写一个 CUDLMG02（**复用
  历史单容器**，不新建格式，不破坏既有单容器测试）。
- 测试：`tests/cuda/test_qwen35_hybrid_microstack_golden.cpp`（真实
  checkpoint 硬门：A 首 token p=0 + B 顺序 p=0→1→2，逐层比对 stage + 状态 +
  micro-stack final；`--no-gen` 供 memcheck）。
- benchmark：`benchmarks/bench_qwen35_hybrid_microstack.cpp`（未优化
  baseline，三视图 + p=0/p=512；evidence
  `benchmarks/results/bench_qwen35_hybrid_microstack.json`）。

> 注：Phase D **复用** Phase B 的 `Qwen35FullAttentionLayer` 与 Phase C 的
> `Qwen35DeltaNetLayer`（及其全部 kernel），**不是**新的上游移植、**不新增**
> kernel；新增的仅是 4 层链的编排（dispatch / 状态 / 时序）与 micro-stack
> golden 的逐层生成。

验收证据（RTX 2080 Ti）：A 全链 bit-exact、B 顺序（worst bf16 3.9e-2 /
fp32 3.1e-3，在跨层复合容差内）；30/30 ctest；compute-sanitizer memcheck 0
错误（覆盖连续多 token micro-stack 运行，
`benchmarks/sanitizer_qwen35_hybrid_microstack.txt`）；no-torch 守卫 CLEAN。
详见 `docs/qwen35_architecture.md` §16。

## CUDALM v0.3 Phase A（Qwen3.5 全模型结构 + 权重摄入口径）

Phase A 在 v0.2 之上完成**完整 24 层模型**的结构 + 权重摄入口径 + 模型级
runtime 骨架（embedding / 24 层 / 最终 norm / tied LM head 的**所有权**与
**状态生命周期**）。**无新 kernel、无新上游移植**：`Qwen35Model` 是纯接线
（模型级张量 + 24 层逐层 dispatch + 统一 reset_state），数学语义完全来自
**复用的冻结 v0.2 单层运行时**（`Qwen35FullAttentionLayer` /
`Qwen35DeltaNetLayer`）。本 Phase **不跑**全模型 forward / logits（Phase B）。
事实源：官方 pinned transformers（`fc9137225880`）+ 真实 checkpoint
（`Qwen/Qwen3.5-0.8B-Base`）——全模型契约见 `docs/qwen35_architecture.md` §17。

### 上游核查（CUDALab `cb6a6a9`，只读参考）

已核查 frozen `CUDALab@cb6a6a9`：本 Phase **不新增任何 kernel**（只复用 Phase
B/C 单层运行时 + 其已移植 kernel）。模型级 wiring（embedding / 24 层 dispatch /
tied LM head / 统一 reset_state）为 CUDALM 原生，无可移植的上游 kernel。

### CUDALM 原生（无上游）

- `include/cudalm/qwen35_model.h` / `src/runtime/qwen35_model.cpp`：
  `Qwen35Model`——`load`（`validate_full_model`：embedding + 最终 norm + 24/24 层
  + tie metadata；逐层 `Qwen35LayerWeights::load` + 按 config 排表构造
  `Qwen35DeltaNetLayer`(18) / `Qwen35FullAttentionLayer`(6)；embedding 上传
  `[248320,1024]` bf16、最终 norm `[1024]` bf16、LM head **alias 词嵌入**）、
  `reset_state`（遍历**全部 24 层**逐层独立重置，无跨层 alias）、访问器
  （embedding/final_norm/lm_head/layer_weights/delta/attention/排表）。**仅
  编排 + 所有权，不复制任何 kernel**。
- `tools/convert_qwen35.py`（`--full-model`）：一次转出全 24 层 +
  `embed_tokens.weight` + `norm.weight` + metadata `tie_word_embeddings`
  （读 config）。**不加 `--full-model` 时逐字节不变**（不破坏 Phase A-D）。
  `tools/common/cudalm_v2.py` 仅新增 metadata 常量 `META_TIE_WORD_EMBEDDINGS`。
- `include/cudalm/weight_loader_v2.h` / `src/runtime/weight_loader_v2.cpp`：
  新增 `validate_model_embedding()` + `validate_full_model()`（缺张量/shape/
  dtype/tie 不符都 loudly fail）。
- 测试：`tests/cuda/test_qwen35_full_model.cpp`（真实 checkpoint 硬门：
  CPU 全模型结构契约 + GPU 所有权 + `reset_state` 覆盖全部 24 层；
  `--no-gen` 供 memcheck）。

> 注：Phase A **复用** Phase B 的 `Qwen35FullAttentionLayer` 与 Phase C 的
> `Qwen35DeltaNetLayer`（及其全部 kernel），**不是**新的上游移植、**不新增**
> kernel；新增的仅是全模型接线（embedding/norm/LM-head 所有权 + 24 层 dispatch
> + 统一 reset_state）与全模型摄入。

验收证据（RTX 2080 Ti）：真实 checkpoint 全量转换 PASS（506 张量 / 766 MB /
layers 0..23）；24/24 层 validate PASS + 层排表 exact（全注意力 3,7,11,15,19,23）
+ embedding/final-norm/LM-head 张量契约 PASS；full model load/unload PASS；
`reset_state()` 覆盖全部 24 层；完整 ctest 34/34 PASS（旧 31 全回归 + 1 Phase A
 full-model + 1 Phase B full-forward + 1 standalone bf16_gemv）；
 compute-sanitizer
memcheck 0 错误（`--no-gen` CUDA-only 路径，覆盖全模型 load + 24 层 seed +
reset + unload 的新 CUDA 分配生命周期）；no-torch 守卫 CLEAN。详见
`docs/qwen35_architecture.md` §17。

## v0.3 Phase B —— 完整单-token forward（LM-head BF16 GEMV + golden oracle）

范围 = 完整单-token forward（`token_id → embedding → 24 层 → final RMSNorm →
tied LM head → logits [248320]`），correctness bring-up（不 generation /
tokenizer / sampling / benchmark 优化）。全模型契约 + 逐段舍入 + 实测 envelope
见 `docs/qwen35_architecture.md` §18（§18.1-18.3）。

### 上游核查（CUDALab `cb6a6a9`，只读参考）

LM-head 需要一个**新** BF16 GEMV（`[V=248320, K=1024]`，tied：W = embedding）。
按规则**先核查** frozen `CUDALab@cb6a6a9` 的已优化 GEMV 结构再移植：

- 采用 proven 的 **16B 向量化 load** 变体 `kernels/gemv/gemv_vec4_row.cu`
  （GEMV-0001）：每行一个 block、256 线程、fp32 累加、warp-shfl + shared +
  warp0 归约、末尾**单次** RNE store；16B 向量 load（`epv = 16/sizeof(T)`）。
- **标量回退**复用其共享的 `kernels/gemv/gemv_common.h` 的
  `gemv_scalar_kernel<T>`（每行一 block、256 线程、fp32 累加、同一归约）。
- 未采用 `gemv_baseline.cu` / `gemv_splitk4.cu` / `gemv_warp_vec4.cu`（本
  `[V=248320,K=1024]` 形状下 vec4_row 是 proven 主路径；splitk 是为大 K 的
  占用率优化，此处 K=1024 不需要）。

### CUDALM 移植（`kernels/bf16_gemv.h/.cu`）

- 从 `gemv_vec4_row.cu`（主路径）+ `gemv_common.h`（标量回退）**机械适配**：
  `__half → __nv_bfloat16`（`el_to_float→__bfloat162float`、
  `el_from_float→__float2bfloat16_rn`，与 Qwen3.5 bf16 kernel 族同一
  "边界单次 RNE" 舍入口径）；PyTorch 绑定（at::Tensor / getCurrentCUDAStream /
  C10_CUDA_KERNEL_LAUNCH_CHECK）改为 **raw pointer + cudaStream_t +
  CUDA_CHECK_LAUNCH**；数学 / 控制流不变。
- 向量化契约与 W4A16 GEMV 一致：W 基 16B ∧ x 基 16B ∧ K%8==0（epv=8）走
  vec4_row，否则标量回退；合法输入永不拒（DeviceBuffer 256B 对齐、K=1024 是
  8 的倍数 → runtime 全部走 vec4_row）。
- **tied**：`Qwen35Model::forward_token` 的 LM-head GEMV 直接以 `embedding`
  的 device buffer 为 `W`（`logits = normed @ embedding^T`），**不复制权重**。
- `forward_token`（`src/runtime/qwen35_model.cpp`）：embedding 行拷贝（D2D，
  bit-exact）→ 复用冻结 v0.2 `Qwen35DeltaNetLayer`(18)/`Qwen35FullAttentionLayer`
  (6) 逐层链（`stage_final_output` 递推）→ 复用 `qwen35_rmsnorm_zc_bf16`
  （final norm，M=1 H=1024 eps=1e-6）→ 新 `bf16_gemv`（LM head）。token_id /
  position OOB **loudly fail**（host precondition）。持久状态用 runtime 自身
  历史（逐层 in-place 更新，顺序 forward 自线程）。

### golden oracle（`tools/generate_qwen35_golden.py --fullmodel-prefix`）

- **pinned quantized oracle**：用**与 runtime 同一份** `.cudalm v2` W4A16
  权重 + tied BF16 embedding（`tensor_from_v2` 直接读 v2 文件的
  `embed_tokens.weight`/`norm.weight`），复用 `build_quantized_*_layer`
  构建全部 24 层；final norm 用 pinned `Qwen3_5RMSNorm`（weight = checkpoint
  norm，零中心 `1+w`）；LM head = `normed @ embedding^T`（fp32 累加、bf16 出）。
- 场景 **A**（p0 fresh）+ **B**（p0→p1→p2 顺序，oracle 自线程状态，**不回填
  golden 状态**）。每 token 输出：per-(token,layer) CUDLMG02（24 层，含全部
  stage + 持久状态 before/after，同 micro-stack 张量集）+ per-token model 级
  CUDLMG02（`model.embedding_output`/`model.final_norm_output`/`model.logits`
  `[248320]` 全向量）。确定性 token：A=[15]，B=[15,16,17]。

### 测试

**独立 `bf16_gemv` numeric 硬门（`tests/cuda/test_bf16_gemv.cpp`，无
checkpoint）**：kernel 正确性**不靠** full-model logits 证明。deterministic
BF16 输入 vs **CPU FP32-accumulate → BF16 RNE** 参考（`compare_bf16_stages`
1e-2，抓 O(magnitude) 的错索引/错累加/错舍入/错 dtype），覆盖 **vec4 路径**
（K%8==0 ∧ 16B 对齐，含真实 LM-head `[N=248320,K=1024]` + 多 N/K）、**scalar
回退**（K%8!=0 及 K%8==0 但 2 字节错位基址）、scalar-vs-vec4 同数据交叉、
全零 weight 行 → bit-exact 0。

**full-forward golden 硬门（`tests/cuda/test_qwen35_full_forward.cpp`）**：
真实 checkpoint，A（p0 fresh）+ B（p0→p1→p2，runtime 自线程状态）逐 token
比较 embedding（bit-exact）/ 24× 层 final / final norm / **FULL logits
[248320]** / 18× DeltaNet conv+recurrent / 6× FA K/V rows 0..p，tolerance =
**per-layer / depth-aware 最小必要 envelope**（每层 = 该层实测 worst × 1.3 +
1e-3，L0 紧、L23 松、L23 不放宽 L0；final norm / logits 保留模型级，§18.2）。
**正确性只由 per-layer envelope 硬门决定**（`actual_error[L] <= k*Atol[L]`）；
深度趋势由 `report_smooth_growth` **diagnostic 报告**（总体随 depth 增大、非严格
单调，**非 PASS/FAIL 门**，误差不因变小而 fail）。`--no-gen` 供 memcheck。

`tests/CMakeLists.txt` 注册：`test_bf16_gemv`（纯 kernel，always）+
`test_qwen35_full_forward`（real-checkpoint 门，self-skip 77，TIMEOUT 1800）。

验收证据（RTX 2080 Ti / CUDA 11.8）：完整 forward A+B 全类别 PASS（§18.2
per-layer envelope 硬门；深度趋势为 diagnostic）；`test_bf16_gemv` PASS；完整
ctest
**34/34 PASS**；`check_no_torch.sh` **CLEAN**；`compute-sanitizer --tool
memcheck` **0 错误**，两份 evidence：`benchmarks/sanitizer_qwen35_full_forward.txt`
（完整 24 层 forward + 新 `bf16_gemv` LM-head GEMV + A/B 顺序路径）+
`benchmarks/sanitizer_bf16_gemv.txt`（vec4 + scalar 回退）。详见
`docs/qwen35_architecture.md` §18.

## CUDALM v0.4 Phase A（token-ID 级单请求 serial prefill + 贪心 decode 生成核）

范围 = **token-ID 级单请求生成**（`prompt_token_ids + max_new_tokens +
eos_token_id → generated_token_ids + stop_reason`），correctness bring-up：
serial prefill（正确性优先，**不** batched/chunked）+ 贪心 decode（greedy
only）。**不做** tokenizer / prompt 字符串 / detokenizer / sampling /
temperature / top-k / top-p / repetition penalty / beam / batched-chunked
prefill / Paged KV / multi-request / scheduler / continuous batching / NCU /
kernel fusion / CUDA Graph / 性能优化。serial prefill 速度**不是**最终性能
数字。契约 + golden oracle + 测试详见 `docs/qwen35_architecture.md` §19。

### 上游核查（CUDALab `cb6a6a9`，只读参考）

生成核是**编排层**（orchestration），**无**可移植的上游 kernel：它复用冻结
v0.3 全模型 runtime（`Qwen35Model::reset_state` / `forward_token`，其内核族
溯源见 §v0.2/v0.3）+ 新增两个 CPU-only 组件（贪心 argmax + stop 控制器，无
CUDA kernel）。核查结论：frozen `CUDALab@cb6a6a9` **无** generation /
decoding 编排 kernel 可移植（其 generation 是 Python 采样循环，Phase A 明确
不采样）；贪心 argmax 按规则**不用** CUDA argmax/reduction kernel（Phase A
正确性优先，CPU bf16 argmax）。

### CUDALM 原生（无上游）

- **贪心 argmax + stop 控制器**（`include/cudalm/greedy.h`，CPU-only）：
  `argmax_bf16` **最大化 bf16 数值 logit**、**tie → 最小 token id**（D2H 全
  [248320] bf16 logits → CPU argmax，**不**用 CUDA argmax/reduction kernel）；
  `GreedyStopController` 确定性 stop（**EOS 选中立即 stop**、token 在序列里、
  不再 forward；否则 max_new_tokens；否则 max_seq_len；同一步 EOS **优先于**
  max_new_tokens）。
- **生成核**（`include/cudalm/qwen35_generator.h` +
  `src/runtime/qwen35_generator.cpp`）：持**非 const** `Qwen35Model&`（模型
  自身持久状态 in-place 线程化整条序列）；每次 `generate()` 开头 **reset
  一次** → **serial prefill** `forward_token(t_i, i)` i=0..N-1（**故意**
  serial，正确性优先）→ **greedy decode** 直到 eos / max_new_tokens /
  max_seq_len；每个 decode forward 前 position 守卫（`position_of(step) =
  prompt_len + step < max_seq_len`，**绝不**越界）；**禁止**重 forward prompt
  末 token / off-by-one / decode 前再 reset / golden 状态回填；`max_new_tokens
  == 0` → 空生成不 decode；容量契约 loudly fail（空 prompt / 非法 token id /
  非法 eos / max_new_tokens < 0 / prompt_len > max_seq_len）。`forward_count`
  = prefill N + decode forward（EOS/max_new_tokens 末 token 不 forward），
  等于 oracle `t_used`。

### golden oracle（`tools/generate_qwen35_golden.py --gen-prefix`）

- **pinned quantized oracle**：复用 §v0.3 Phase B 全模型 oracle（同一份
  `.cudalm v2` W4A16 权重 + tied BF16 embedding + 18× DeltaNet / 6×
  FullAttention + pinned `Qwen3_5RMSNorm` + tied LM head）。
- **serial prefill + greedy decode 镜像 C++ 循环**（同一 reset / position
  守卫 / argmax 契约 / stop 顺序 / stop token 不 forward / **不回填** golden
  状态）。
- 输出 **CUDLMW02 容器**（C++ `WeightFileV2::load` 可载）：生成 token 序列 +
  stop_reason + `t_used`（metadata）+ **每个生成步的全 [248320] bf16 logits**
  +（场景 B）**最终持久状态**（18× DeltaNet conv bf16 [6144,3] + recurrent
  fp32 [16,128,128]；6× FA K/V used rows bf16 [kv, t_used, 256]）。
- **confident prompt（钉死）**：oracle top-1 vs top-2 logit gap 每步**显著
  高于** runtime/oracle bf16-logits 舍入（~0.13），贪心序列才是稳定 golden
  （near-tie 会让舍入翻转 argmax → 序列分叉）。`tools/diag_gen_gaps.py` 测
  逐步 gap；选定 prompt min gap **1.0625**（A）/ **3.625**（B）。

### 测试

- **`test_greedy_argmax`**（CPU，always，无 checkpoint）：argmax（normal /
  negative / exact tie→最小 id / single / bf16-rounding tie）。
- **`test_greedy_stop`**（CPU，always，无 checkpoint）：确定性控制器（eos /
  max_new_tokens / max_seq_len / 末 token eos 优先）。
- **`test_qwen35_generation`**（real-checkpoint 门，self-skip 77，TIMEOUT
  1800，`--no-gen` 供 memcheck）：场景 A（短 confident prompt，8 token）+ B
  （长 confident prompt，16 token，最终状态）+ C（repeat-generate 污染门，
  **强化**：两次 generate 用 `LogitsObserver` 抓 **EVERY 生成步全 [248320]
  logits** 逐 step **原始字节（memcmp）bit-exact** 比较（**非**仅数值相等，`+0/-0`
   等 bit 不同也 FAIL），证明第二次 reset 后数值轨迹与第一次相
  同，而非仅 greedy token 恰好没变）。比较 **EVERY 生成 token id** + **EVERY
  生成步全 [248320] logits** + stop_reason + forward_count（== t_used）+（B）
  最终混合持久状态（18× DeltaNet conv/recurrent + 6× FA K/V used rows），
  **收紧的 per-step / per-layer envelope**（每步 / 每层独立 = 实测 worst × 1.3
  + floor，**非**共享 atol；**BF16** logits/conv/KV 用 `+ 0.01`、**FP32**
  recurrent 用 `+ 0.001`；recurrent 实测 ≤ 0.0326、**非** 0.5）。
- **`test_qwen35_generation_contract`**（real-checkpoint 门，self-skip 77）：
  生成核**输入契约** hardening —— `not_loaded` / 空 prompt / 非法 token id
  （<0 与 ≥vocab）/ 非法 eos（<0 与 ≥vocab）/ `max_new_tokens < 0` /
  `prompt_len > max_seq_len` 全部 → `ok == false` + generated 空 +
  `forward_count == 0`（无 prefill/decode、无状态变更）；`max_new_tokens == 0`
  → `ok == true` + 空生成 + stop=max_new_tokens + prefill 已跑但**不** decode。
  不扩大 API（只驱动现有 `generate()`）。

### 验收证据（RTX 2080 Ti / CUDA 11.8）

- 完整 ctest **38/38 PASS**（旧 34 全回归 + **4 新增**：`test_greedy_argmax` +
  `test_greedy_stop` + `test_qwen35_generation` + `test_qwen35_generation_contract`）。
- `scripts/check_no_torch.sh` **CLEAN**（include/、src/ 无 torch/pybind）。
- `compute-sanitizer --tool memcheck` **0 错误**
  （`benchmarks/sanitizer_qwen35_generation.txt`，`--no-gen` CUDA-only，覆盖
  **multi-token prompt + multi-step greedy decode**：真实 prefill/decode 状态
  转换 + 重复 tied-LM-head GEMV + KV history 增长 + DeltaNet recurrent 更新 +
  最终状态读回）。

## CUDALM v0.4 Phase B（原生 tokenizer + prompt→text，PyTorch-free 文本级全链路）

范围 = **raw UTF-8 prompt → 原生 encode → token id → 冻结生成核（真实 EOS
248044）→ 原生 decode → raw UTF-8 text**。`Qwen35TextGenerator` 是纯拼接层
（encode → generate → decode），不新增生成逻辑 / sampling / chat-template /
streaming / batching。契约 + 硬门详见 `docs/qwen35_architecture.md` §20。

### 上游核查（pinned 官方源，只取 tokenizer 文件）

- 源 = `Qwen/Qwen3.5-0.8B-Base` @ revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`（与 §1 / v0.3 同一 pinned
  源）。**只下载 tokenizer 文件**（不下载 9B 模型、不下载本模型权重 ——
  权重已由 v0.3 摄入并 sha256 钉死）：本地
  `/root/models/Qwen3.5-0.8B-Base/tokenizer/`，4 文件 sha256 全部钉死于
  `tools/common/qwen35_tokenizer_ref.py`（读取前强制校验，不符即 fail
  loud）：

  | 文件 | sha256 |
  |---|---|
  | `tokenizer.json` | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |
  | `tokenizer_config.json` | `e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de` |
  | `vocab.json` | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
  | `merges.txt` | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |

 - **oracle（pinned + 版本门）**：主 = 本地 venv（py3.11.16，
   tokenizers 0.22.2，Rust 引擎，sha-gated pinned tokenizer 资产）。
   oracle 版本钉死 `tokenizers == 0.22.2`：`qwen35_tokenizer_ref.
   check_oracle_version()` 在 `build_tokenizer()` 内强制执行 —— 所有
   authoritative 路径（converter / 语料生成 / differential validator /
   text golden）都经过同一门；实际版本 ≠ 0.22.2 = provenance
   violation，**fail loud，不可 skip**（资产缺失仍按既有规则 self-skip
   77 —— 二者不同）；converter self-test 含版本门回归（0.22.2 接受；
   0.15.1 / 0.22.3 / None 拒绝）。系统 `tokenizers 0.15.1` **不是**
   可接受 oracle（无 silent fallback）。Python `unicodedata`（本机 U14
   数据）**仅作诊断**，不作等价 oracle（其数据版本与引擎不同）。
   **无** transformers 参与 encode/decode。
- 转录是**纯离线查表**：C++ 运行时**不**链接/调用任何 tokenizer 库、不
  读 JSON、不做正则库调用 —— 全部结构（词汇字节、merge rank、added token、
  NFC 分解/合成/ccc 表、`\s/\pL/\pN/\pM` 精确区间）由 converter 预先算进
  CUDLMTK1 artifact，C++ 只做界检 + 查表 + 贪心 BPE。

### CUDALM 原生（无上游）

- **NFC**（**pinned 引擎算法的精确移植**，本分支修复：旧实现先是
  “无重排”，后“整体稳定排序 + 纯表合成”，均与引擎不符）：引擎
  NFC = Rust `unicode-normalization`（tokenizers 依赖
  `unicode-normalization-alignments` 0.1.12）的 Decompositions +
  Recompositions **流式**算法，C++ 逐语句移植（`nfc_impl`）：完全
  分解（含单部件）→ **流式规范重排**（读入 `ccc==0` cp 或输入结束
  时，对**尚未发射的尾部**按 CCC 升序、稳定排序后放行；jamo ccc 0，
  序列永不重排）→ **合成与重排同趟进行**（composee 仅当**所有已缓冲
  （延迟）mark 的 ccc 都严格更小**时才与入流 mark 合成，否则该 mark
  缓冲延后发射；纯合成表查表，exclusions 含于表内）。该算法即标准
  UAX #15：差分验证中 pinned 引擎与 Python `unicodedata` 算法层面
  100% 一致（0 未解释发散）；行为差异全部来自数据版本（引擎 U9 vs
  本机 U14），已用 pinned U9/U16 UCD 表消除（见上）。oracle =
  pinned 引擎（构建期 + 验证期）。
  关键回归：descending CCC `U+0041 U+0315 U+0300`
  （232,230）→ `U+00C0 U+0315`（旧“无重排”实现在此 FAIL，本分支
  修复）。
- **左优先正则预分词**（B1..B7，`(?i:)` 仅分支 1；`\s` 等类用 artifact
  精确区间表）：左优先语义 + B6 零宽 lookahead（**不消费**字符）+ B2 可选
  前导字符黏连（`a   b`→`a`,`  `,` b`）均逐条 oracle 对齐；B6 `\s+`
   **贪心回退**实证：k≥2 个 `\s` run 后跟 `\S` 时匹配前 k−1 个
   （`"  x"`→`" "`|`" x"`，引擎/C++ 逐条一致）。**BLOCKER-D2 已修复**：
   引擎 \s 含 U+00A0，旧 artifact \s 表（converter `WHITE_SPACE`）漏之
   （pre-existing Phase B 缺陷）—— 已加 `(0x00A0,0x00A0)` 并重生成，
   回归见架构文档 §20.5（E 151/154/160 + P 1097-1103 + 25-cp battery）。
  实现为自研节点池 matcher（RNode/RPool/Matcher，整型引用、无堆分配递归
  状态），量词回溯 = 贪心（长→短，**`?` 含回退**：先试消费形、失败再试
  空形 —— 无回退时 B2 在“chunk 起始的 combining mark 后接非 L/M 字符”
  上 fail loud，而 pinned 引擎按 PCRE 语义回退成功；differential fuzz
  发现，已修并加阶段向量）；所有分支不匹配 → **fail loud**（内部错误，
  无静默 fallback）。
- **ByteLevel BPE**（byte_fallback=false；256 单字节 + 全 merge 产物在词汇
  内，artifact 构建时逐条校验）：每预分词独立，贪心最低 rank（tie→最左），
  输出 = 拼接字节查表。
- **decode**（pinned 引擎语义）：token ids → **完整字节流**（base token 的
  原始 ByteLevel 字节 + added token 字面量 UTF-8，`skip_special_tokens`
  丢弃 21 special added，padding id 248077..248319 贡献空）→ 对**整个**
  字节流做一次有损 UTF-8 转换（Rust `from_utf8_lossy` / UTS #35 语义：
  每个最大非法子段 → 一个 U+FFFD；越界 continuation 字节**不**并入子段、
  各自单独 FFFD —— oracle 验证 `ED A0 80`→3×FFFD、`F4 90 80 80`→4×FFFD）；
  合法的跨 token 拆分多字节字符（`E4`+`B8`+`AD` 三个 base token）按一个
  字符解出（U+4E2D）。**逐 token 校验是错的**（会产出 `FFFD FFFD AD→?`
  而非 U+4E2D）；输出恒为合法 UTF-8。
- **CUDLMTK1 artifact**（~4.7MB，确定性，gitignore，**不**入库）：header
  （magic/version/reserved/crc32）+ 15×u32 钉死 meta + 10 个 size-prefixed
  section；C++ loader 对全部损坏类 **fail loud**（测试逐项覆盖）。
- **安全契约**：全仓库零 raw 特殊/控制 token 字符串字面量 —— added token
  只由 artifact 承载，语料只以 UTF-8 hex 出现，文档只引 id/名称/长度/sha。

### 硬门（ctest 新增 3 项，旧门全回归）

- `test_qwen35_tokenizer_python_selftest`：converter 自测（确定性 /
  round-trip / 损坏类；资产缺失 → 77）。
- `test_qwen35_tokenizer`：loader 失败契约 + NFC/预分词阶段向量（含
  descending-CCC 规范重排回归、孤立 combining mark 的 `?` 回退回归）+
  decode 契约（padding→""、越界、is_special 21/33、EOS==248044、非法
   UTF-8）+ **跨语言精确语料 1883 行**（E=305 / D=305 / D1=305 / N=65
   （11 类 NFC 向量，期望 = pinned 引擎 `normalize_str`（U9），含
   BLOCKER-D1 回归类）/ P=317（期望 = pinned 引擎 `pre_tokenize_str`，
   含 25-cp White_Space 边界 battery（BLOCKER-D2 回归）+ U15/U16 类
   探测）/ X=586
   （293 任意 id 序列 ×2 skip 模式：非法字节/不完整序列/非法组合/跨 token
   拆分字符/base+padding/base+added/special，byte→id 映射从 pinned vocab
   动态推导，期望 = pinned 引擎 decode，hex 编码））与 pinned oracle
  **EXACT**。
 - **regression-first**：reviewed HEAD（63b884f）converter 重建的旧
   U14 artifact 跑**新**语料 → **21 行 FAIL**（恰为 D1/D2/U16 回归行）；
   新 artifact → 1883/1883 PASS。
 - **differential validation 固化为仓库工具**：`tools/
   dump_qwen35_tokenizer.cpp`（CMake `tool_qwen35_tokenizer_dump`）+
   `tools/validate_qwen35_tokenizer.py`（sha-gated pinned oracle，固定
   seed，打印 samples/mismatches/seed/HEAD）。**Phase B final evidence
   executed on**: `PHASE_B_EVIDENCE_SHA = 
   0dc3b576d8171bedebe96d487cf83a59d5f98bef`（此后任何 src/include/
   tools/tests/functional CMake 配置修改 → evidence 失效必须重跑；仅
   docs 修改不失效，此时注明最终 HEAD ≠ evidence SHA）。于该 SHA 的
   extended（穷举）结果：nfc_single 1,112,064/0、enc_single 1,112,064/0、
   class_probe **3,336,190/0**（实际比较数，第三探针跳过 CR/LF）、
   decomp_cp 13,232/0、comp_pair 12,118/0（各含 encode 对照）、
   nfc_fuzz 200k/0（seed 777）、adjacency 50k/0（seed 20260925）、
   enc_fuzz 100k/0（seed 42）、pretok_fuzz 100k/0（seed 999）、
   dec_fuzz 40k/0（seed 20250417）、regressions 26/0、ws_battery 400/0
   —— **全 pass 0 mismatch（停止条件达成）**。命令见架构文档 §20.5。
- `test_qwen35_text_generation`（real checkpoint + tokenizer，self-skip 77，
  TIMEOUT 1800）：4 个 confident 文本 prompt（2 EN + 2 CJK，min top1-top2
  gap 0.625 / 2.875 / 1.4375 / 0.75，`tools/diag_gen_gaps.py --text` 选定）：
  native encode == HF encode、greedy 生成 == pinned-quantized oracle、native
  decode == HF decode（skip_special_tokens=False）、stop_reason +
  forward_count 全 **EXACT**；EOS = **真实 248044**（非 Phase A 哨兵
  248319）；污染门 T1→T3→T1 逐字段相同。
- `tools/generate_qwen35_golden.py` 扩展 `--gen-text`（Phase A 参数不动）：
  文本经 pinned HF tokenizer 编码走同一 pinned-quantized oracle，golden 附
  `gen.prompt_text` + `gen.hf_decoded` metadata。
- `tools/diag_gen_gaps.py` 扩展 `--text`（原始文本 prompt 的逐步 gap 分析 +
  HF 解码回显，供 Phase B prompt 选型；Phase A `--prompt` 不动）。

### 验收证据（RTX 2080 Ti / CUDA 11.8）

- 完整 ctest 全 PASS（旧 38 全回归 + **3 新增**：
  `test_qwen35_tokenizer_python_selftest` + `test_qwen35_tokenizer` +
  `test_qwen35_text_generation`）。
- `scripts/check_no_torch.sh` **CLEAN**（include/、src/ 无 torch/pybind ——
  tokenizer 与文本层均为纯 C++，运行时零 Python 依赖）。
- v0.3 模型数学 / Qwen35Model/Generator 语义 / CUDA kernel / Phase A golden
  **零改动**（Phase A 场景 A/B + 污染门原样通过）。

 ---

## CUDALM v0.4 Phase C（基础 sampling + `cudalm-generate` CLI）

 - **承接**：Phase B **DONE/FROZEN**（functional/evidence SHA
   `0dc3b576d8171bedebe96d487cf83a59d5f98bef`）。tokenizer / NFC / BPE /
   decode 语义**未动**（Phase C 只新增 sampling + CLI；`src/
   runtime/qwen35_tokenizer*`、`tools/*tokenizer*` 零修改）。
 - **新增 runtime 代码**：`include/cudalm/sampling.h`（SamplingConfig /
   SplitMix64 / 纯流水线 / Sampler，CPU-only）、
   `include/cudalm/generate_cli.h` + `src/cli/generate_cli.cpp`（CLI
   参数解析，CPU-only）、`tools/cudalm_generate.cpp`（CLI 入口）。
   既有文件的改动仅为**接线**：`qwen35_generator.{h,cpp}`（新
   sampling overload；旧 greedy overload 委托共享 core +
   `SamplingConfig::greedy()` —— 冻结路径本体）、`qwen35_text_
   generator.{h,cpp}`（新 sampling overload，thin facade 不变）。
 - **新增 CUDA runtime path = 0**：无新 kernel、无新 CUDA memory
   分配、无新 stream 语义 —— sampler 作用在 generator 原本就 D2H 的
   host logits 上。Phase A CUDA sanitizer 证据因此**可复用**；另在
   Phase C 于本 SHA 对 `cudalm-generate` greedy 路径重跑
   `compute-sanitizer --tool memcheck`（`--launch-timeout 900`，
   RTX 2080 Ti / CUDA 11.8）：**0 错误**（记录：
   `benchmarks/sanitizer_cudalm_generate.txt`）。
 - **RNG**：SplitMix64（常量在 `sampling.h` 内完全指定 → 跨平台确定性）；
   每请求一个实例（`Sampler` 随 generate() 新建），无全局随机状态；
   greedy 零消耗。
 - **测试（新增 4 个，全部 PASS）**：`test_sampling`（CPU synthetic
   logits：greedy 兼容 / temperature / top-k / top-p / k+p 组合顺序 /
   seed 复现 / 不同 seed / 非法 config / 极值与全负 logits / 精确 tie /
   单一幸存者）；`test_qwen35_sampling`（real checkpoint：greedy EXACT、
   seed 确定性、A(42)→B(123)→A(42) 污染门 id 级+text 级、forward-count
   不变式、非法 config fail loud）；`test_generate_cli_args`（CPU 参数
   解析全矩阵）；`test_cudalm_generate_cli`（真二进制：--help / 用法
   错误 / 坏路径 / 非法 config / greedy / sampling 同 seed 逐字节相同 /
   CJK prompt）。
 - **Phase A/B evidence 不变**（完整 regression 于本 SHA 全 PASS：
   **ctest 45/45 PASS、0 skipped**，见架构文档 §21.6）：Phase A
   greedy generation golden（EXACT）、
   repeat-generate 污染门、Phase B tokenizer exactness（1883/1883
   corpus）、text-generation E2E（EXACT）、`tokenizers == 0.22.2`
   provenance 门（converter selftest 内版本门回归）。tokenizer
   **quick** differential validation 于本 SHA 0 mismatch（extended
   百万级穷举未重做 —— Phase C 未修改 tokenizer 实现/工具，Phase B
   extended evidence 仍绑定 `0dc3b576`）。
 - **`scripts/check_no_torch.sh`**：CLEAN（新代码全部 PyTorch/
   pybind-free）。
 - **Review 修复轮（functional，commit `cb3cb66`）**：两处
   correctness blocker + 两处 cleanup，不扩 scope ——
   (1) 极小正 temperature 数值溢出：原 float 流水线
   `logit / denorm_min -> inf`，随后 `inf - max -> NaN`，
   `sample_token()` 返回 -1（已复现）；修复为 scaled/max/exp 全程
   **double**（有限 bf16 logit / 最小合法正 float ≈ 2.4e83，在
   double 范围内）→ 对**每个**合法有限正 float temperature，有限
   logits 都产生合法分布（p 全 finite、sum ≈ 1）；`sample_token()`
   契约文档化为 vocab ≥ 1 时恒返回 `[0, vocab)`，generator 另加
   防御性 range 检查（-1 永不进 `forward_token`）。回归：
   `test_sampling` 第 12 组（FLT_MIN / denorm_min / FLT_MAX ×
   正/负极值与平手 logits，16 seeds 下 sampled id ∈ [0, vocab)）——
   旧实现必然击穿。(2) CLI stdout 由 `fputs(c_str())` 改为
   **binary-safe length-aware** `write_generated_text()`（精确字节数
   fwrite + 结尾换行，短写 exit 1）：原生 decode 合法产生的 embedded
   NUL 不再被截断。回归：`test_generate_cli_args` 用真实 FILE
   roundtrip `ab\0cd` / 单 NUL / 空串逐字节比对。另外：CLI
   `--temperature` 文本 parse 增加 float range 检查（overflow 到 inf /
   underflow 到 0 → usage error，不静默变 inf 或 0/greedy；denormal
   到 denorm_min 合法）；`top_k` 措辞统一为 `== 0` 禁用 / `< 0` 非法 /
   `> vocab` clamp（代码 + 文档）。
 - **第二轮 review 修复（functional，commit `5aba21fe`）**：
   `--temperature` 的 **strtod 级 underflow**——上一轮只拦截了
   「非零 double 转 float 变 0.0f」，但 `1e-5000` / `-1e-5000` 这类
   文本会让 `strtod()` 本身就 range-underflow 到 ±0.0（errno
   ERANGE、结果为零），旧检查 `f == 0.0f && v != 0.0` 识别不到，
   静默变成 greedy（已复现）。修复：`parse_temperature()` 在
   `strtod` 前 reset `errno`，检测 `errno == ERANGE && v == 0.0`
   并拒绝——非零 temperature 文本绝不能静默变 0/-0；显式
   `0`/`-0` 仍合法，float subnormal（到 denorm_min）仍合法，
   overflow/inf/nan 仍拒绝。回归：`test_generate_cli_args` 新增
   `1e-5000`/`-1e-5000` → usage error，保留 `0`/`-0`/`1e-45`/
   `1.4e-45`/FLT_MIN/near-FLT_MAX 合法用例。sampler/generator/
   tokenizer 语义零修改。
 - **evidence 绑定**：`V04_EVIDENCE_SHA = 5aba21fe0b351079850600f3f8fe7f55a77c8745`（clean
   tree、HEAD == SHA；完整 ctest 45/45 PASS 0 skipped +
   check_no_torch CLEAN + quick tokenizer validation 0 mismatch +
   CLI compute-sanitizer 0 错误，均于该 SHA）。更早的 evidence SHA
   `a008b437` 与 `cb3cb668` 分别因两轮 functional 修复按规则**失效**
   （未删除历史，仅声明失效）。失效规则：此后任何 `src/` /
   `include/` / `tools/` / `tests/` / functional CMake 修改 →
   evidence 失效必须重跑；仅 **docs/evidence** 修改（文档 +
   benchmark 证据记录；例如 `b7fb4b1`、`bfd01fa` 的性质就是
   docs/evidence-only，不是严格 docs-only）不失效（此时最终 HEAD ≠
   evidence SHA）。
 - **sign-off**：external reviewer 判 **PASS** —— v0.4 正式
   **DONE / FROZEN**；`v0.4-generation` 已 merge 进 main（延续
   v0.2 / v0.3 的里程碑模式）。v0.4 冻结范围：Qwen3.5-0.8B
   single-request serial 生成核（greedy 为 Phase A 冻结路径）+
   基础 sampling（temperature / top-k / top-p / seed）+ 原生
   tokenizer prompt→text + `cudalm-generate` CLI；evidence 绑定
   `V04_EVIDENCE_SHA = 5aba21fe0b351079850600f3f8fe7f55a77c8745`
   （失效规则见上）。

## CUDALM v0.5 Phase A（Hybrid State Manager Foundations：state 控制面 + 设备 state 池）

 - **承接**：v0.4 **DONE/FROZEN**（`V04_EVIDENCE_SHA = 5aba21fe…`，已
   merge 进 main）。Phase A 分支 `v0.5-state-manager` 从 main（`48ede94`）
   开出；**只做 state ownership**：多序列的 state 控制面 + 设备 state
   池，standalone，**不**接入 `Qwen35Model` forward（Phase B）。
   v0.4 冻结面（model forward / attention / DeltaNet kernel /
   generation / sampling / tokenizer / CLI）**零修改**；旧
   `Qwen35KvCache` 与 `Qwen35DeltaNetLayer` 自持 state 原样保留
   （Phase B 才被池取代）。
 - **新增 runtime 代码（全部 CUDALM-native，无移植）**：
   `include/cudalm/fixed_id_pool.h`（`FixedIdPool`，CPU 分配器核心，
   header-only）、`include/cudalm/qwen35_state_layout.h`（config
   推导的池尺寸公式，header-only）、
   `include/cudalm/kv_block_table.h`（`KvBlockTable` + `KvPageSource`，
   header-only）、`include/cudalm/qwen35_kv_page_pool.h` +
   `src/runtime/qwen35_kv_page_pool.cpp`（`Qwen35KvPagePool`，RAII
   设备池）、`include/cudalm/qwen35_delta_state_pool.h` +
   `src/runtime/qwen35_delta_state_pool.cpp`（`Qwen35DeltaStatePool`，
   RAII 设备池）、`include/cudalm/qwen35_state_manager.h` +
   `src/runtime/qwen35_state_manager.cpp`（`SequenceState` +
   `Qwen35StateManager`，host 控制面）。**既有 src/include 文件零
   修改**（唯一改动的既有文件是 `tests/CMakeLists.txt` 的测试注册）。
 - **无新 kernel / 无新 CUDA API 类别**：新 CUDA path 仅为池构造的
   `cudaMalloc` + 整池/整页/整 slot 的 `cudaMemsetAsync`（zero-on-
   release / zero-on-reset，池 stream 上 ordered）+ D2H 验证读取
   （测试内）。`page_tokens` 是显式构造参数（非硬编码；测试用 4/8/
   16）；所有记账公式从 `Qwen35Config` 混合 schedule 推导（6 full /
   18 linear 于 0.8B，无 magic constant）。
 - **语义钉死（见架构文档 §22）**：KV 页布局
   `K/V 各 bf16 [n_full][num_pages][n_kv][page_tokens][head_dim]`，
   一个物理 page id 跨所有 full 层同指一个逻辑 token block（每序列
   一张 `KvBlockTable`，不建 per-layer 表）；Delta slot 布局
   每层 `conv bf16 [capacity, 6144, 3]` + `recurrent fp32
   [capacity, 16, 128, 128]`（0.8B），一个 slot 寻址全部 18 个
   DeltaNet 层，state 永不落 host。两个池均为 **zero-on-release**
   （构造整池清零 + 释放/重置时逐层清零 → 任何 acquire/reset 后
   资源保证全零），single-stream 资源，Status OOM（不 abort、不部分
   分配），double-free/double-release/越界 fail loud 且状态不变。
   `KvBlockTable`：位置分解精确、前缀 append-only、同一逻辑 block
   仅一次物理 page、**事务性 OOM**（中途失败全回滚，表与池
   accounting 完全不变、无泄漏 page）。`Qwen35StateManager`：
   create（唯一全零 slot + 空表 + length 0；slot OOM 什么都不登记）/
   ensure_kv_capacity（事务性）/ set_length + advance（纯 metadata，
   不分配 page）/ reset（同 id 同 slot，page 释放、slot 就地清零、
   length 0，不可能 OOM）/ retire（全资源释放，id 永久失效）。
   **`SequenceId`（uint64）单调递增、永不复用**（物理资源可复用，
   外部 id 不可 → 杜绝 stale handle）。
 - **新增测试（4 个，全部 PASS；3 CPU + 1 CUDA，均无 checkpoint
   依赖）**：`test_fixed_id_pool`（全唯一 / capacity+1 OOM / LIFO
   复用 / double-free 与越界拒绝 / reset / 零容量 / **fixed-seed
   200,000-op 混合 allocate/free stress**（均匀 op 硬币 + live 偏置
   release → 占用率全区间扫过 [0,64]；独立 host live-set 模型每步
   对账：live 唯一、**完整 live-set 相等**（非仅 size）、accounting
   每步精确、满池 OOM 零状态变化、合法 release 精确、非法/重复
   release 拒绝、released id 实际复用；结束覆盖证明：五类分支计数
   均 > 0 且 min_occ*2 < CAP 且 max_occ == CAP）；`test_kv_block_table`（位置分解边界矩阵 0 / pt-1 / pt
   / pt+1 / 最后有效 / max（含不整除 ceil）/ 最小前缀增长与幂等 /
   同 block 单 page / **中途 OOM 全回滚无泄漏** / 越界 fail loud /
   clear 精确归还 / LIFO 复用身份，counting fake source）；
   `test_state_pool_formulas`（0.8B 6/18 层数、page_tokens 4/8/16 的
   每页字节（16 → 196,608 B）、每 slot 19,537,920 B、小型合成
   config 钉死）；`test_qwen35_state_manager`（CUDA，小型 valid()
   config：8 层 = 2 full + 6 linear，max_seq_len 64，page_tokens 4——
   池分配器全矩阵 + **fixed-seed 100,000-op 设备侧混合 stress**
   （同 CPU 版 workload 语义，占用率扫过 [0,32]）+ **真
   设备复用污染门**：对 slot 写非零 pattern（conv + recurrent，全部
   6 层）/ 对 page 写非零 pattern（K + V，全部 2 层）→ 释放 → 同一
   物理资源被复用 → 新 owner D2H 读回**逐字节全零**（不只 metadata）
   + block table 真池边界/OOM 无泄漏/单表共享 + 生命周期
   （A/B 不别名、A reset 后 B 的 pattern 不受影响且 A 归零、A retire
   回收资源且 id 永久失效、C 复用物理资源但 `SequenceId != A`——
   retire 前缓存 A 的物理 slot（retire 后 map 节点已 erase，记录
   指针悬垂，测试不再解引用；C 的 slot 与缓存值比对，不依赖
   可能被 node-reuse 制造 false PASS 的悬垂指针）；已 retire id
   一切操作 fail loud）+ OOM 事务性（跨页边界失败 ensure
   表与记账完全不变；create OOM 零登记、id 保持单调））。
 - **完整 regression（于本 evidence SHA）**：完整 ctest **49/49
   PASS、0 skipped**（v0.4 全部门面回归：generation / sampling /
   tokenizer / CLI / 各 golden 全 PASS）；`scripts/check_no_torch.sh`
   **CLEAN**。
 - **Sanitizer**：v0.5 Phase A **引入新 CUDA 分配/清零/生命周期
   path**（不同于 v0.4 Phase C 的零新 path）：于本 SHA 对
   `test_qwen35_state_manager` 跑 `compute-sanitizer --tool memcheck`
   （`--launch-timeout 1200`，RTX 2080 Ti / CUDA 11.8）：**ERROR
   SUMMARY: 0 errors**（记录：
   `benchmarks/sanitizer_qwen35_state_manager.txt`）。
 - **Review 修复轮（tests/header-doc only，commit `2319a26`）**：
   (1) `test_sequence_lifecycle` host **use-after-free**：retire 后
   `ra`（指向已被 `std::map::erase` 的 `SequenceState`）仍被解引用
   （`ra->delta_slot`）——改为 retire 前缓存 `a_slot`，retire 后不
   再解引用 `ra`；C create 后以缓存值验证「`C id != A` 且
   `C delta_slot == a_slot`」（物理 slot 复用证明不再依赖悬垂
   指针，杜绝 node-reuse false PASS）。(2) 两个 allocator stress
   的 workload 由 acquire 偏置（未满必然 acquire → release 只发生
   在满池）改为真正的 fixed-seed **混合 allocate/free** workload
   （均匀 op 硬币 + live 偏置 release；占用率全区间扫过；完整
   live-set 相等 + 每步精确 accounting + 五类分支覆盖证明，见
   上）。(3) `Qwen35StateManager::next_sequence_id()` 注释修正
   （返回**下一个将发放的** id，非「历史最大 id」；API 行为不
   变）。(4) KV 池新增 `live_pages()` test/diag accessor。**runtime
   语义零修改**（KV/Delta 池行为、manager 行为、所有冻结面均未
   动）。
 - **evidence 绑定**：`V05A_EVIDENCE_SHA = 2319a261543e1af75f544a6a57592b0193074312`
   （clean tree、HEAD == SHA；完整 ctest 49/49 PASS 0 skipped +
   check_no_torch CLEAN + state-manager compute-sanitizer 0 错误，
   均于该 SHA）。更早的 evidence SHA
   `866a2e46142f2a3a77deddf72809081eb58b47a1` 因上述 tests/header
   修改按规则**失效**（未删除历史，仅声明失效）。Phase A 未修改
   tokenizer 实现/工具 → tokenizer quick differential 不需要重做
   （v0.4 evidence 的 tokenizer 门仍绑定 `5aba21fe`）。失效规则同
   v0.4：此后任何 `src/` / `include/` / `tools/` / `tests/` /
   functional CMake 修改 → 本 evidence 失效必须重跑；仅
   docs/evidence 修改不失效。
 - **边界（明说）**：Phase A **不** merge 进 main、**不**自启
   Phase B；无 paged-attention kernel / 无 external-state model
   forward / 无多序列执行 / 无 scheduler / 无 batching 类特性（见
   架构文档 §22.5）。待 external reviewer 签核。

## CUDALM v0.5 Phase B（External Hybrid State 接入 + Paged KV Kernel）

 - **承接**：v0.4 **DONE/FROZEN**（`V04_EVIDENCE_SHA = 5aba21fe…`，已
   merge 进 main）+ v0.5 Phase A（state 控制面 + 设备 state 池，
   分支 `v0.5-state-manager`）。Phase B 把 Phase A 的 external
   hybrid state **接入 model** 并落地**真 paged KV kernel**（写 +
   因果 decode attention，块表寻址在 kernel 内，**禁止先 gather 成
   contiguous KV 再调旧 kernel**，无 KV 值 host roundtrip）。冻结的
   v0.4 数学**零语义修改**：legacy `forward()` /
   `forward_token()` / `Qwen35Generator` / CLI 行为不变，legacy
   contiguous `Qwen35KvCache` 原样保留（`forwardImpl` 尾部仅多一个
   `PagedStateRef* paged = nullptr` 分支参数；DeltaNet `forward()`
   改为以层自持 state 委托同一 `forward_impl`，数学零修改）。
 - **新增 runtime 代码（全部 CUDALM-native，无移植）**：
   `include/cudalm/kernels/paged_kv.h` + `src/kernels/paged_kv.cu`
   —— `qwen35_paged_kv_write_bf16`（把一个 token 的 `n_kv*head_dim`
   K/V 元素写入块表指向 page 的 `(n*pt+off)*hd` 行；部分块 guard）
   + `qwen35_paged_attention_decode_bf16`（对 `[0, position]` 全 KV
   行做因果 decode attention，块表寻址；scores 每 (h,t) 内 d 升序
   fp32 累加 / GQA `kh = h*n_kv/n_heads`、softmax 为冻结 kernel 的
   1:1 镜像（3-pass、O(num_warps) smem、同一 reduction 树）、PV 每
   (h,d) 内 t 升序 —— **op 序与冻结 contiguous kernel 逐一对齐 →
   parity 要求 bit-identical 而非容差**；scratch =
   `[scores2 | probs]` 各 `n_heads*(position+1)` bf16）。
   **修改**：`qwen35_deltanet.{h,cpp}`（新增
   `forward_with_state(position, x, ext_conv, ext_rec, stream)` +
   `forward_impl`，state 指针外提，数学零修改）、
   `qwen35_full_attention.{h,cpp}`（新增 `PagedStateRef` +
   `forward_with_paged_state`，只替换 KV 写与 attention 读两个
   stage）、`qwen35_model.{h,cpp}`（新增
   `forward_token_with_state(token, seq_id, mgr, stream)` +
   grow-only `block_table_scratch_`）、
   `qwen35_kv_page_pool.h`（`page_stride_elems() = page_elems()`：
   池布局 row-major `[n_full][num_pages][…]`，**同一 ordinal 相邻
   page 物理相邻**，ordinal 间距 = capacity*page_elems —— Phase B
   开发中曾因把 page_stride 误设为 ordinal 间距导致 "page p" 落到
   ordinal (ord+p) 的 page 0，parity 门抓出后修正并固化，见架构
   §23.2）、`qwen35_state_manager.h`（`kv_pool_mut()` /
   `delta_pool_mut()` / `page_tokens()`）、`kv_block_table.h`
   （`page_ids()` host 视图）。
 - **设备块表策略**：每序列一张 `KvBlockTable`（Phase A，CPU
   前缀表）；每 token forward 做一次**小 H2D `cudaMemcpyAsync`**
   把当前前缀 `num_blocks` 个 page id 拷进 grow-only 设备 buffer
   （`max_blocks()` int，永不重分配）——唯一跨 H/D 的每 token
   元数据拷贝，stream-ordered；kernel 只读
   `[0, position/page_tokens]` 前缀（page id 精确、无 stale 读；
   旧尾部残留的大 id 永不被触及，独立门用 9999 sentinel 双证）。
 - **Compatibility gate + OOM-before-mutation 契约（钉死）**：
   `forward_token_with_state` 顺序 = 加载/token 检查 →
   **compatibility gate（任何 state mutation 前的最先一致性检查）**：
   `mgr.config() == config()` + Phase A/B **单 stream 契约**（`stream
   == mgr.kv_pool().stream() && stream == mgr.delta_pool().stream()`；
   v0.5 correctness-first，不引入 cross-stream event machinery）→
   lookup →
   `position = rec->length` → `position >= max_seq_len` 检查 →
   **`ensure_kv_capacity`（最先的 state 触点）** → 块表 H2D →
   embedding → 24 层 external forward → final norm + LM head →
   **`advance(+1)` 最后**。config/stream mismatch（compatibility
   gate）/ KV OOM / 未知-retire 序列 / 越界长度 /
   非法 token 全部 Status fail-loud 且**零 mutation**（无 delta 变化、
   无 KV 变化、length 不变、不进入任何 layer forward、无部分
   forward；硬门以 48 项 state 前后 bit-identical + 4 项
   accounting 不变钉死）。
 - **新增测试（2 个，均 PASS；合成门无 checkpoint 依赖，parity 门
   真实检查点）**：`test_paged_kv`（合成：写门 非平凡映射
   {3,0,5,1,4,2}/{5,2,0,3,1,4} × position {0, pt-1, pt, pt+1}
   物理行 EXACT + 未写行保持零 + stale sentinel；attention 门 16
   组 (position × mapping) 对冻结 contiguous kernel BIT-IDENTICAL；
   真实形状门：真实 Qwen3.5-0.8B attention 维度直接从冻结
   `Qwen35Config::qwen35_08b()` 读取并 CHECK 钉死（n_heads 8 /
   n_kv_heads 2 / **head_dim 256** —— H1024/8，无手写常量；原
   16/8/64 为错误常量，reviewer 修复轮更正）+ 2-ordinal miniature
   池布局 fixture × pt=2 ×（相邻 page + ordinal 间距 =
   capacity*page_elems）×
   映射 {2,0,1} × T {2,3,4,5,6}（逐 token 跨页边界）
   BIT-IDENTICAL）；`test_qwen35_state_parity`（真实
   Qwen3.5-0.8B checkpoint，CLI 参数
   `<full_model.cudalm> <checkpoint_dir> <python> <src_dir>`，
   缺失时 exit 77 → ctest SKIP_RETURN_CODE 77 / TIMEOUT 1800：
   A = legacy `reset_state` + 6 × `forward_token` vs B = 全新
   manager（page_tokens 2、池 3 page 恰 6 token 容量、4 delta
   slot）+ `create_sequence` + 6 × `forward_token_with_state`，
   token 流 {1024, 2048, 3072, 15, 16, 17}（跨 pt=2 页边界、恰
   填满池）；每 step embedding + 24 层 final + final norm +
   全量 logits[248320] **bit-identical**（runtime-vs-runtime →
   atol=0 / memcmp，无容差）；终态 18 × DeltaNet conv/rec +
   6 × 全注意力**逻辑行 0..5 K/V**（external 侧经 host 块表从
   物理 page device→host 逐行读；legacy 侧按
   `[n_kv][max_seq][hd]` 的 `(n*max_seq+t)*hd` 逐行取）
   **bit-identical**；第 7 token OOM → `!s.ok` + 48 项 state 前后
   bit-identical + length/num_blocks/used_state_bytes/live_pages
   不变；compatibility gate（reviewer 修复轮新增）：config 不一致的
   manager（n_kv_heads 4 vs 2）→ `!s.ok` 零 mutation（无 KV page、
   无 Delta 变化、length 不变、不进任何 layer forward）；stream
   不一致（单 stream 契约）→ `!s.ok` 且 48 项 state 前后
   bit-identical + length/num_blocks/used_state_bytes/live_pages 不
   变；未知 id 999 与非法 token（vocab+7）fail-loud 且 length
   不变；`reset_sequence` 后重放同 6 token → 每 step logits +
   终态 48 项与首次 B 运行 bit-identical、length == 6）。
 - **完整 regression（于本 evidence SHA，clean tree）**：完整
   ctest **51/51 PASS、0 skipped**（含 v0.4 全部门面回归：
   generation / sampling / tokenizer / CLI / 各 golden /
   full-model forward 全绿；parity 门在 evidence 环境**真实运行**，
   非 77 skip）；`scripts/check_no_torch.sh` **CLEAN**。
 - **Sanitizer（于本 evidence SHA）**：v0.5 Phase B **引入新
   kernel + 新 H2D 元数据 path**：`compute-sanitizer --tool
   memcheck`（RTX 2080 Ti / CUDA 11.8）对 `test_paged_kv`：
   **ERROR SUMMARY: 0 errors**（原始日志记录：
   `benchmarks/sanitizer_paged_kv.txt`）；对
   `test_qwen35_state_parity`（真实 checkpoint 全流程，含块表
   H2D + paged kernel + compatibility gate + state capture 的
   device→host 拷贝）：**ERROR SUMMARY: 0 errors**（原始日志记录：
   `benchmarks/sanitizer_qwen35_state_parity.txt`）。
 - **Review 修复轮（external reviewer Phase B 修复轮，commit
   `a29b596`；不重设计 Phase B、不启动 Phase C）**：(1)
   `test_paged_kv` 真实形状门更正：原手写常量 n_heads 16 / n_kv 8
   / head_dim 64 **错误**；真实 Qwen3.5-0.8B 全注意力维度为
   `Qwen35Config::qwen35_08b()` 的 **n_heads 8 / n_kv_heads 2 /
   head_dim 256** —— 改为直接从冻结 config 读维度并 CHECK 钉死
   （无手写常量）；测试文档明确「真实 0.8B attention 维度 +
   2-ordinal miniature 池布局 fixture」（真正 6 个 full-attention
   layer 的 real-checkpoint path 由 `test_qwen35_state_parity`
   覆盖）。(2) page-stride 错误注释修正（`paged_kv.h` /
   `paged_kv.cu`）：kernel `page_stride` = page_elems = n_kv*pt*hd
   （同一 ordinal 相邻 page），ordinal 间距 =
   capacity_pages*page_elems **不是** kernel page_stride；kernel
   base = k_page(ord, 0) —— 与 `Qwen35KvPagePool::
   page_stride_elems() == page_elems()`（正确实现，行为不变，仅
   注释）对齐；同步清理 Phase B 新代码/测试中的错误 real-shape
   注释（16/8/64、16 query heads、runtime 16*256；DeltaNet 合法
   16-head 描述未动）。(3) `Qwen35Model::forward_token_with_state`
   新增 **compatibility gate**（任何 state mutation 前验证）：
   `mgr.config() == config()` + Phase A/B 单 stream 契约
   （`stream == mgr.kv_pool().stream() ==
   mgr.delta_pool().stream()`；不引入 cross-stream event
   machinery）；不一致 → Status fail-loud 零 mutation（无 KV
   allocation、无 Delta mutation、length 不变、不进入任何 layer
   forward）。(4) `test_qwen35_state_parity` 新增两个 contract
   测试（config 不一致 / stream 不一致 → mutation 前拒绝，零
   mutation 钉死）。**kernel 行为与 bit-exact 语义零修改**（paged
   kernel 本体、冻结 pipeline、legacy 路径均未动）。
 - **evidence 绑定**：`V05B_EVIDENCE_SHA =
   a29b59610b39a0ad24fc6d79ce2c61088897330e`（clean tree、
   HEAD == SHA；完整 ctest 51/51 PASS 0 skipped +
   check_no_torch CLEAN + `test_paged_kv` 与
   `test_qwen35_state_parity`（真实 checkpoint 真实运行，非 77
   skip）的 compute-sanitizer memcheck 各 0 错误，均于该 SHA；
   原始日志：`benchmarks/sanitizer_paged_kv.txt` /
   `benchmarks/sanitizer_qwen35_state_parity.txt`）。**失效声明
   （未删除历史）**：reviewer 修复轮（commit `a29b596`）修改了
   `include/`、`src/`（runtime + kernels 注释）、`tests/`，按失效
   规则，原 Phase B evidence 绑定 **`V05B_EVIDENCE_SHA =
   79d1ac523e4d96a91e75c33b389487ceb37eb2fa` 失效**（未删除历史，
   仅声明失效）；更早的 **`V05A_EVIDENCE_SHA =
   2319a261543e1af75f544a6a57592b0193074312`（及其更早绑定
   `866a2e46142f2a3a77deddf72809081eb58b47a1`）** 与 **`V04_
   EVIDENCE_SHA = 5aba21fe…`** 维持同规则失效 —— 其全部门面回归
   已在本 SHA 的 51/51 内重跑全绿。失效规则延续：此后任何
   `src/` / `include/` / `tools/` / `tests/` / functional CMake
   修改 → 本 evidence 失效必须重跑；仅 docs/evidence 修改不
   失效。
 - **边界（明说）**：Phase B **不** merge 进 main、**不**自启
   Phase C；无 双序列交错硬门 / scheduler / admission / continuous
   batching / batched decode / chunked prefill / streaming /
   PagedAttention perf / CUDA Graph / NCU / fusion / HTTP-OpenAI
   server（见架构文档 §23.4）。待 external reviewer 签核。

## CUDALM v0.5 Phase C（多序列交错硬门 + v0.5 最终签核）

 - **承接**：v0.5 Phase A（state 控制面 + 设备 state 池，FROZEN）+
   Phase B（external-state forward + paged KV kernel，FROZEN，
   `V05B_EVIDENCE_SHA = a29b59610b39a0ad24fc6d79ce2c61088897330e`，
   分支 `v0.5-state-manager`）。Phase C 证明 external-state runtime
   在**多个 sequence 交错执行**时仍严格正确，并完成 v0.5 最终
   sign-off。范围钉死：**单 model、单 CUDA stream、每 step 一次
   `forward_token_with_state()`**，只有调用顺序在多个 SequenceId
   之间交错（显式测试脚本，**非** scheduler）；**无** scheduler /
   request queue / admission policy、**无** continuous batching /
   batched decode / batched GEMV、**无** chunked prefill / streaming /
   multi-CUDA-stream / CUDA Graph / NCU / kernel fusion /
   HTTP-OpenAI API（均属 v0.6+）；**零** src/include 修改 —— 冻结
   的 Phase B runtime 原样复用（本阶段仅新增测试 + 测试 CMake）。
 - **新增测试（1 个，PASS；真实 Qwen3.5-0.8B-Base checkpoint；
   ctest SKIP_RETURN_CODE 77 / TIMEOUT 1800；evidence 环境必须真实
   跑，77 skip 不是签核）**：`test_qwen35_state_interleave`：
   - token 流（全部跨 page_tokens=2 页边界）：A =
     {1024, 2048, 3072, 4096, 5000, 6000}（6 tok → 3 page）、B =
     {15, 16, 17, 18, 19}（5 tok → 3 page）、C = {7, 8, 9, 10, 11,
     12}（6 tok → 3 page）；池 6 page（A+B 同时 live）、4 delta
     slot；
   - **独立 reference**（fresh manager、单序列独占）：每 step
     24 层 final + final norm + **FULL logits[248320]**；终态 18 ×
     DeltaNet conv（bf16）+ rec（fp32）+ 6 × 全注意力**逻辑行
     K/V**（经块表从物理 page 读，无 host gather）+ length + 块表
     形状；
   - **交错运行**（一个 manager，A+B live，非平凡 schedule
     `A0 B0 A1 A2 B1 A3 B2 A4 B3 A5 B4`）：**A 每步 == refA、B
     每步 == refB**（runtime-vs-runtime，**BIT-IDENTICAL**，
     atol=0 / memcmp）；终态 A/B hybrid state 分别 bit-identical；
     A/B 物理 page 各自唯一且互斥；
   - **跨序列隔离门**（关键 step 做**完整 hybrid state** pre/post，
     不只 length/accounting）：A4 前捕获 B（len 3）→ forward A4 →
     B bit-identical；B2 前捕获 A（len 4）→ forward B2 → A
     bit-identical；
   - **retire / 复用污染门**：`retire_sequence(A)` → A 的
     SequenceId **永久 invalid**（lookup nullptr；advance /
     ensure_kv_capacity 拒绝）、A 的 3 KV page + Delta slot 回收
     （精确字节 accounting：used_state_bytes 恰降
     3*bytes_per_page + bytes_per_slot）、B 的 state
     bit-identical；`create_sequence(C)` → **C id != A id**（id 永
     不发放两次），且 C **实际复用** A 释放的 Delta slot（同 slot
     id）与 A 释放的 KV 物理 page（page id 集合相等 —— 池里只有
     A 的 3 page 可分配）；C 从 fresh-zero state 跑完整 token 流：
     每 step 24 层 final + norm + FULL logits + 终态 Delta state +
     逻辑 KV 与独立 fresh-C reference **bit-identical**；C 的执行
     不改变仍 live 的 B（完整 state pre/post，含关键 step C0）；
   - **reset 单序列隔离**：`reset_sequence(B)` **只清 B**（length
     = 0、全部 page 释放、Delta slot 原地清零 —— 18 ordinal 的
     conv/rec 逐字节验证为零）、live C 的 state bit-identical。
 - **完整 regression（于本 evidence SHA，clean tree）**：完整
   ctest **52/52 PASS、0 failed、0 skipped**（含 v0.4 全部门面回归
   + Phase A/B 全部硬门；interleave 门在 evidence 环境**真实运
   行**，非 77 skip）；`scripts/check_no_torch.sh` **CLEAN**。
 - **Sanitizer（于本 evidence SHA）**：Phase C **引入多序列交错
   执行路径**（池分配/回收/复用在多序列间交错发生）：
   `compute-sanitizer --tool memcheck`（RTX 2080 Ti / CUDA 11.8）
   对 `test_qwen35_state_interleave`（真实 checkpoint 全流程，34
   次真实 forward，含 retire/reuse + 全部 state capture 的
   device→host 拷贝）：**PASS + ERROR SUMMARY: 0 errors**（原始
   日志：`benchmarks/sanitizer_qwen35_state_interleave.txt`）。
 - **evidence 绑定**：`V05C_EVIDENCE_SHA =
   1054b69c4f72f3f0238361d5b5e5f5ab8463489c`（clean tree、
   HEAD == SHA；完整 ctest 52/52 PASS 0 skipped + check_no_torch
   CLEAN + interleave 门 compute-sanitizer memcheck 0 错误，均于该
   SHA）。**失效声明（未删除历史）**：Phase C functional commit
   修改了 `tests/` 与测试 CMake，按失效规则：**`V05B_EVIDENCE_SHA
   = a29b59610b39a0ad24fc6d79ce2c61088897330e`（及其更早绑定
   `79d1ac523e4d96a91e75c33b389487ceb37eb2fa`）失效**；更早的
   **`V05A_EVIDENCE_SHA = 2319a261543e1af75f544a6a57592b0193074312`
   （及其更早绑定 `866a2e46142f2a3a77deddf72809081eb58b47a1`）**
   与 **`V04_EVIDENCE_SHA = 5aba21fe…`** 维持同规则失效 —— 其全部
   门面回归已在本 SHA 的 52/52 内重跑全绿。失效规则延续：此后
   任何 `src/` / `include/` / `tools/` / `tests/` / functional
   CMake 修改 → 本 evidence 失效必须重跑；仅 docs/evidence 修改
   不失效。
 - **v0.5 终态（明说）**：**v0.5 Phase A / Phase B / Phase C 全部
   DONE**，**v0.5 Hybrid State Manager DONE**（单序列 parity +
   多序列交错 + 生命周期/复用污染全部 bit-exact 签核）。本阶段**不**
   merge 进 main、**不**自启 v0.6（scheduler / admission /
   continuous batching / batched decode / chunked prefill /
   streaming / multi-stream / PagedAttention perf 均属 v0.6+，见
   架构文档 §24）。待 external reviewer 签核。

## CUDALM v0.6 Phase A（Request Scheduler / Control Plane — semantics only）

 - **承接**：v0.5 Hybrid State Manager（Phase A/B/C 全部 DONE）已由
   external reviewer 确认并 **merge 进 main**（no-ff merge commit
   `88ce595`，merge tree 与 v0.5-state-manager 的
   `c21f8cb071dc860b9602be0b5d1bcc0619365c99` tree **完全一致**）；
   本阶段在分支 `v0.6-scheduler`（从新 main 切出）上工作。v0.6 目标：
   把**冻结的 v0.5 multi-sequence runtime** 上层接入一个**正确、确定、
   可测试**的 request scheduler / control plane。本阶段**只**解决：
   request lifecycle、admission、waiting/running/finished、iteration
   scheduling、prefill/decode progression、EOS / max-new-tokens
   completion、state create/retire ownership、deterministic
   multi-request execution。**不实现真正的 batched CUDA compute**。
 - **Phase A 执行模型（钉死）**：**scheduler semantics only —— GPU
   执行仍然是每次一个 sequence forward**：一个 iteration
   （`Scheduler::step()`）让每个 eligible request 最多前进一步，每步 =
   一次单序列 `forward_token_with_state()`（单 model、单 CUDA stream）；
   **无** batched CUDA kernels / batched forward / batched GEMV / true
   GPU continuous batch / PagedAttention 优化 / chunked prefill /
   multi-stream / CUDA Graph / NCU / fusion / HTTP-OpenAI / async /
   priority scheduler / beam search / speculative decoding（均属
   Phase B+ / v0.6+）。Phase A 目的：证明 **scheduler/control-plane
   语义 == 独立执行语义**。**无吞吐改进声明、无 true batched GPU
   execution 声明。**
 - **冻结 v0.5 runtime（零修改）**：`Qwen35Model` /
   `Qwen35StateManager` / paged KV / Delta state pool /
   `forward_token_with_state()` / tokenizer / sampling / legacy
   generator / CLI 全部原样复用；v0.5 已证明的 multi-SequenceId /
   interleaved forward / retire-reuse / reset isolation 语义继续成立。
 - **新增代码（3 个文件）**：
   - `include/cudalm/request.h`：`RequestId`（**monotonic、永不复
     用**；与 `SequenceId` 是两个独立 id 空间，绝不混用）、
     `RequestStatus`（Waiting/Running/Finished/Cancelled/Failed）、
     `FinishReason`、`Request`（prompt、prefill_pos、generated、
     max_new_tokens、eos_token_id、**per-request** SamplingConfig +
     **per-request** Sampler（per-request SplitMix64 —— 两个 request
     绝不共享 RNG progression）、status/finish_reason/forward_count）；
     头部钉死 **v0.4 token progression**（early prefill 不采样；最后
     一个 prompt token 的 logits 产生 g0；之后 forward g_{k-1} →
     采样 g_k；以 m 个 generated 结束的 request 恰好 forward
     N + m - 1 次）；
   - `include/cudalm/scheduler.h` + `src/runtime/scheduler.cpp`：
     `SequenceForwarder`（抽象 forward/logits 源，CPU 可 fake）+
     `ModelForwarder`（真实包装 loaded Qwen35Model，委托冻结的 v0.5
     gate-including forward）+ `Scheduler`：
     * **admission（transactional）**：validate → create_sequence →
       register → issue RequestId；任何失败：**无 half-request、无泄
       漏 SequenceState、无被消耗 RequestId**（id 只在 create 成功之
       后发放）；
     * **iteration（deterministic FIFO / round-robin）**：每轮开始先
       **snapshot** non-terminal RequestId（升序 == admission 顺序），
       然后逐 id **重新 lookup** 各前进一步（mutation safety 硬要求：
       不跨 advance 持有 iterator/pointer/reference；terminal 自然跳
       过）；iteration 之间 admit 的 request 从**下一轮**开始执行（绝
       不插进当前 snapshot）；无 priority/fairness heuristic；
     * **TERMINAL EXACTLY ONCE**：恰好一次 terminal 转移 + 恰好一次
       `retire_sequence`；stale request 永不再 advance；terminal
       record 保留可 inspect；
     * **cancel（钉死契约，测试 + 文档化）**：Waiting/Running →
       Cancelled + retire；**已 terminal → 幂等 ok**（无状态变化）；
       未知 id → fail-loud Status error；
     * **fatal Status**：forward 失败 → 该 request Failed + retire
       （恰一次）；同一 snapshot 其余 id 继续推进；step() 返回首个错
       误。Scheduler 不拥有 model / state manager / stream（non-owning
       引用；v0.5 config/stream compatibility gate 在
       forward_token_with_state 内部 fail-loud 强制执行）；policy 不
       塞进 StateManager。
 - **新增测试（2 个，PASS）**：
   - `test_qwen35_scheduler`（**CPU control-plane gate**；无
     checkpoint / 无 model —— 确定性 fake forwarder 跑在**真实**
     Qwen35StateManager 池上，create/retire/capacity 都是真的）：
     request lifecycle、monotonic never-reused RequestId、
     transactional admission（delta-slot 耗尽 + 输入校验：无泄漏
     sequence / 无 half-request / 无消耗 id；finish/cancel 后可再
     admit）、FIFO/round-robin snapshot 语义（forward 调用顺序日志；
     晚 admit 进下一轮）、prefill/decode 共存、EOS + max_new_tokens
     （精确 token 流 + forward count）、cancel + retire（幂等
     re-cancel / 未知 id fail-loud / stale 永不 advance）、per-request
     采样 RNG 隔离（A(seed42) 交错 == A 单独；B(seed123) 交错 == B
     单独；同 prompt+同 seed ⇒ 同流，与 interleave/admission 顺序无
     关）、fatal Status（Failed + retire 恰一次；snapshot 其余 id 继
     续推进；**失败的 forward 不提交 progress**：`forward_count` /
     `prefill_pos` 均排除之 —— `{50,99}` 门钉死 50 ok/99 failing →
     `forward_count = 1`、`prefill_pos = 1`（**不是 2**）；**`run()`
     failure isolation** 硬门：A 在 `run()` 中 fatal、B 正常 →
     `!s.ok`、A Failed、B Finished、num_live == 0、manager 0，且 A 的
     progress 钉死 + sequence 只 retire 一次）；
   - `test_qwen35_scheduler_integration`（**real-checkpoint
     integration gate**；真实 Qwen3.5-0.8B-Base；无 checkpoint 自
     skip 77，evidence 环境**必须真实运行**，非签核）：A（3-tok
     prompt，max_new 3，seed 42）+ B（7-tok prompt，max_new 2，seed
     123）先 admit，2 轮后**动态 admit** C（2-tok prompt，max_new 3，
     seed 7），跑到底 —— 每 request 的 **generated token IDs、每个
     generated step 的 FULL logits[248320]、forward count、finish
     reason** 与独立 fresh-manager reference（直接 forward，不走
     scheduler）全部 **EXACT（memcmp）**；B 的完整 hybrid state 在
     length 4（A finish 之前）与 length 5（A finish + retire 之后）
     与 B-alone reference 同 length 状态 bit-identical（另一 request
     的 finish/retire 不触碰仍-live request 的 state）；real-logits
     采样隔离（同 prompt + 同 seed ⇒ X 单独 == Y 单独 == X+Y 交错
     相同流）。
 - **完整 regression（于本 evidence SHA，clean tree）**：完整 ctest
   **54/54 PASS、0 failed、0 skipped**（含 v0.4/v0.5 全部门面回归 +
   两个新 gate；integration gate 真实运行，非 77 skip）；
   `scripts/check_no_torch.sh` **CLEAN**。
 - **Sanitizer（于本 evidence SHA）**：Phase A **引入 scheduler 控制
   面**（动态 admission / 中途 finish+retire / 每 forward 全量 logits
   D2H / per-request sampler 在真实 forward 之上交错）：
   `compute-sanitizer --tool memcheck`（RTX 2080 Ti / CUDA 11.8）对
   `test_qwen35_scheduler_integration`（真实 checkpoint 全流程，55 次
   真实 forward + 全部 state capture）：**PASS + ERROR SUMMARY: 0
   errors**（原始日志：`benchmarks/sanitizer_qwen35_scheduler.txt`）。
 - **reviewer fix round（未 amend）**：reviewer 在原 functional SHA
   `459ff12f1a4d1365112d65f8863a4e5010710c11` 上发现两个 scheduler
   state-machine blocker，均已在后续 commit 修复（不 amend、不
   rebase）：
   - **failed-prefill progress commit**：`advance_one()` 原先在
     `forward_token()` **之前**就 `prefill_pos++`，违反
     `prefill_pos = 已成功 forward 的 prompt token 数` 的 contract。
     改为：选 `prompt[prefill_pos]` → `forward_token()` → **成功才**
     `prefill_pos++`。失败的 forward 既不提交 `prefill_pos` 也不计入
     `forward_count`（钉死：prompt `{50,99}`、50 ok / 99 failing →
     status Failed、forward_count = 1、**prefill_pos = 1 非 2**）。
     正常 prefill / last-prompt 采样 / decode 的 off-by-one 语义不
     变（real-checkpoint integration gate 全量 EXACT 重跑确认）。
   - **`run()` failure isolation**：`run()` 的公开 contract 是
     「run until every request is terminal」，原先一个 request 的
     fatal Status 会让 `run()` 立即返回、其他 live request 永久停
     住。改为：失败 request → Failed + retire（不再被 advance），其
     余 request 在后续 iteration 继续推进，全部 terminal 后返回遇到
     的**首个错误**；`step()` 行为不变（同一 snapshot 剩余 id 继续、
     返回首个错误）。新增 hard gate：A 在 `run()` 中 fatal、B 正常
     → `!s.ok`、A Failed、B Finished、num_live == 0、manager 0（+
     A 的 progress 钉死、sequence 只 retire 一次）。
 - **evidence 绑定**：`V06A_EVIDENCE_SHA =
   928d0a772f698bcc22e55a6d2ff1a19f48e0f037`（clean tree、HEAD ==
   SHA；完整 ctest 54/54 PASS 0 skipped + check_no_torch CLEAN +
   scheduler integration 门 compute-sanitizer memcheck 0 错误，均于
   该 SHA）。**失效声明（未删除历史）**：本 reviewer fix commit 修改
   了 `src/` 与 `tests/`，按失效规则：**原
   `V06A_EVIDENCE_SHA = 459ff12f1a4d1365112d65f8863a4e5010710c11`
   失效**（历史 commit 与 docs 不变，仅 evidence 绑定推进到
   `928d0a7…`）。更早地，Phase A 首版修改了 `src/`、`include/`、
   `tests/` 与测试 CMake，按失效规则：**`V05C_EVIDENCE_SHA =
   1054b69c4f72f3f0238361d5b5e5f5ab8463489c`**（及其更早的 V05B
   `a29b59610b39a0ad24fc6d79ce2c61088897330e` / V05A
   `2319a261543e1af75f544a6a57592b0193074312` / V04 绑定）对本 tree
   **失效** —— 其全部门面回归已在本 SHA 的 54/54 内重跑全绿（v0.5
   的 merge 状态与历史 SHA 不变，仅 evidence 绑定按规则推进）。失效
   规则延续：此后任何 `src/` / `include/` / `tools/` / `tests/` /
   functional CMake 修改 → 本 evidence 失效必须重跑；仅 docs/evidence
   修改不失效。
 - **v0.6 Phase A 终态（明说）**：**Phase A = scheduler semantics
   only；GPU 执行仍是每次一个 sequence forward；无吞吐改进声明、无
   true batched GPU execution 声明。** 本阶段**不** merge 进 main、
   **不**自启 Phase B（batched GPU execution / true continuous
   batching）—— 待 external reviewer 签核。
