# CUDALM

原生 C++/CUDA 量化 LLM 推理引擎。**v0.1** 实现一个完整的 Llama 风格
decoder block —— W4A16 量化线性层、fp16 激活、单 GPU 自回归解码 ——
**运行时不依赖 PyTorch（也不依赖任何 Python）**。C++17、CUDA 11.8、
`sm_75`（RTX 2080 Ti）。kernel 只接收裸指针 + `cudaStream_t`；Python
只存在于 `tools/`（离线生成测试数据），绝不链接进运行时。

**v0.1.1（架构清理）**：泛化 decoder 投影形状契约
（Q 宽度不再等于 hidden_size）、benchmark 的"stage 之和"与"整块 GPU
时间"分离、速率指标更名 `block_steps_per_second`、权重来源显式化
（固定种子合成权重）。不新增模型架构、不改 W4A16 kernel 算法、
不重新优化 Attention、不扩大 scope。

## v0.1 范围 —— 以及刻意不做的部分

已实现（端到端）：

```
x ─► RMSNorm ─► Q/K/V（W4A16 GEMV）──► RoPE（交错对）
  ─► KV 缓存写入 @ position ─► 因果解码注意力（fp32 计算）
  ─► O 投影 ─► + 残差 ─► RMSNorm ─► gate/up（W4A16 GEMV）
  ─► SiLU(gate)·up ─► down（W4A16 GEMV）──► + 残差 ─► y
```

v0.1 刻意不在范围内（v0.1.1 之后硬停止 —— 见文末清单）：FlashAttention、
批处理（batching）、张量并行、多 GPU、CUDA Graphs、tokenizer、完整
多层模型。

## 钉死的契约

| 项 | 契约 |
|------|----------|
| 配置（v0.1 默认） | `H=1024, n_heads=8, n_kv_heads=4, head_dim=128, intermediate=2816, group=128, max_seq=512, eps=1e-5, rope_theta=10000` |
| 投影形状（v0.1.1 泛化） | q_proj `(q_proj_out, H)`、k/v_proj `(kv_proj_out, H)`、o_proj `(H, q_proj_out)`、gate/up `(inter, H)`、down `(H, inter)`，其中 `q_proj_out = n_heads·head_dim`、`kv_proj_out = n_kv_heads·head_dim`。**q_proj_out 独立于 hidden_size**（默认配置二者恰好相等，8×128=1024）；`ModelConfig::valid()` 要求 q_proj_out 与 inter 均为 group_size 的倍数，不再要求 `hidden_size == n_heads·head_dim`。完整形状契约：[`docs/weight_format.md`](docs/weight_format.md) |
| 权重 | W4A16：对称分组 INT4，G=128，q∈[−7,7]，zero_point=0，`scale=amax/7` 以 fp32 计算、以 fp16 存储（**存储的 fp16 scale 即契约**），nibble 打包 low=k=2b / high=k=2b+1（4 位补码）。格式规范：[`docs/weight_format.md`](docs/weight_format.md) |
| KV 缓存 | K、V 各为 fp16 `[n_kv_heads][max_seq_len][head_dim]`；行 (n,t) 偏移 `((n*max_seq_len)+t)*head_dim`；零初始化；越界即致命前置检查 |
| RoPE | 交错对：`y[2i]=a·c−b·s`，`y[2i+1]=a·s+b·c`；cos/sin 表 fp16 `[max_seq, hd/2]`；按行查 `positions[m]`；基址 4B 对齐走 4B 打包路径，否则走标量回退（从不拒绝） |
| 注意力 | GQA 整数映射 `kh = h * n_kv // n_heads`；`scale = 1/√head_dim`；减最大值的 softmax，逐元素除法；**全部 fp32 计算，仅在存储时做一次 fp16 RNE** |
| stage 容差 | 相对黄金判据 `\|a−r\| ≤ 1e-2 + 1e-2·\|r\|`（实测最大偏差 ≤ 2.4e-4，来自 kernel FMA 收缩 vs 黄金的独立 mul+add）。p=0 RoPE 恒等、全部 18 个权重张量、带种子的 KV 历史行均钉死为**逐位精确**；当前位置的 KV 行按"对黄金容差 + 对运行时自身 rope_k/v stage 行逐位一致"校验 |
| Benchmark 计时（v0.1.1 语义） | 三个互不混淆的视图：`stage_sum_us` = 17 个 stage CUDA 事件时长之和（**不含**事件未覆盖的位置 H2D 等工作，是分解小计，不是端到端时延）；`whole_block_gpu_us` = 包围整个 forward GPU 工作的一对 CUDA 事件（真实整块 GPU 时间）；`host_api_wall_us` = forward 调用 + stream 同步的 CPU 墙钟。速率指标 `block_steps_per_second` = 1e6 / whole_block 均值 —— **一次 decoder block 解码步 ≠ 一个模型 token**，故不再叫 tokens/s |

## 仓库结构

```
include/cudalm/           运行时头文件（decoder_block.h、kv_cache.h、
                          weight_format.h、kernels/{rmsnorm,int4_gemv,rope,
                          kv,attention,elementwise}.h、device_buffer.h …）
src/kernels/              .cu kernel（裸指针 + cudaStream_t）
src/runtime/              decoder_block.cpp、kv_cache.cpp、weight_loader.cpp、
                          golden_loader.cpp、device_buffer.cpp
tests/cpu/                无 GPU 测试：文件格式、跨语言权重一致性
tests/cuda/               kernel 测试 + 端到端黄金测试（共 18 个 ctest
                          用例；v0.1.1 新增泛化投影形状测试）
tools/                    离线 Python（torch 2.3.1+cpu）：
                          convert_weights.py（量化器）、generate_golden.py、
                          common/binfmt.py —— 绝不链接进运行时
benchmarks/               bench_decoder_block.cpp + 已提交的结果
                          （v0.1.1 计时 schema）+ sanitizer 证据
                          （v0.1 与 v0.1.1 各一份）
scripts/check_no_torch.sh 守卫脚本：include/ + src/ 出现任何 torch/pybind/py
                          符号即硬失败
docs/                     bootstrap_plan_v0.1.md、provenance.md、
                          weight_format.md
```

**溯源（Provenance）。** 三个 kernel 是从上游 CUDALab 研究仓库
（冻结于 `cb6a6a9`，tag `v0.7.1`，只读）逐行移植的：`rmsnorm`（v4 fp16）、
`int4_gemv`（rowtile4_hx + 标量回退）、`rope`（v3 half2，交错对）。
host 层（PyTorch extension API）被替换为裸指针 + `cudaStream_t` +
`CUDA_CHECK`；kernel 的数学与控制流 1:1 保留。其余部分 —— KV 缓存、
注意力流水线、decoder block 接线、权重格式、测试、benchmark —— 均为
CUDALM 原生。逐文件明细表（含偏差说明）见
[`docs/provenance.md`](docs/provenance.md)。

## 权重来源（固定种子合成权重）

本仓库的权重与 golden 参考数据全部由 `tools/` 脚本（固定种子的离线
PyTorch，`convert_weights.py` / `generate_golden.py`，种子 `20250922`）
生成的**确定性合成权重（fixed-seed synthetic weights）**，**不是**
HuggingFace checkpoint 转换器的输出。v0.1 / v0.1.1 的目标是验证：
原生运行时架构（权重容器、加载器、DecoderBlock 接线）、移植 kernel
的集成、以及逐 stage 的 golden 正确性 —— 与具体模型权重无关。

真实 HuggingFace safetensors checkpoint 的权重摄入（ingestion）**不
在 v0.1.1 范围内**，随 v0.2 Qwen3.5 bring-up 一并完成。

## 构建

要求：CMake ≥ 3.16、CUDA 11.8 工具链（`nvcc` 面向 `sm_75`）、C++17
编译器，以及（**仅用于生成测试数据**，构建本身不需要）python3 +
`torch`（CPU 版即可）。构建过程不链接任何 Python。

```sh
cmake -S . -B build
cmake --build build -j
```

## 复现证据

所有测试数据由 `ctest` 从 `tools/` 自动生成到 `build/data/`
（带种子、确定性：共享种子 `20250922`，历史种子 `20250923`）。

**1. 完整测试套件（18/18）：**

```sh
cd build && ctest
```

覆盖：文件格式往返、跨语言权重字节一致性（Python 量化器 → C++ 加载器）、
逐 kernel 测试（rmsnorm、int4_gemv、rope、kv_cache、attention、
elementwise），以及端到端黄金测试（`test_decoder_block`，位置 0 与 7：
18 个权重张量 + 16 个 stage 张量 + KV 状态，按上表钉死的契约校验）。
v0.1.1 另含泛化投影形状用例（`gen_general_weights` +
`test_decoder_block_general`：H=1024、n_heads=16、head_dim=128 →
q_proj_out=2048 ≠ H，非对称形状下走完整量化/黄金流水线）。

**2. 逐 stage 时延分解 + 整块计时**（CUDA 事件，17 个 stage +
整块事件对；三个视图的语义见上表"benchmark 计时"行）：

```sh
./build/benchmarks/bench_decoder_block build/data/block_v01.cudalm <position> [iters=100] [out.json]
```

已提交结果（`benchmarks/results/`，RTX 2080 Ti，CUDA 11.8，驱动
570.172.08，iters=100；v0.1.1 计时 schema，JSON 内 `timing_semantics`
字段记录各指标含义）：

| 位置 | stage_sum_us（均值） | whole_block_gpu_us（均值） | host_api_wall_us（均值） | block steps/s |
|------|-----------|------------------|------------------|----------|
| 0   | 121.6 µs | 160.5 µs | 178.0 µs | 6 230 |
| 511 | 153.5 µs | 193.8 µs | 212.4 µs | 5 161 |

`block steps/s = 1e6 / whole_block 均值`；一个 decoder block 解码步不是
一个模型 token，故不再报告 tokens/s。stage_sum 是分解小计（不含事件
未覆盖的位置 H2D 等工作），不能当作端到端时延；整块 GPU 时间与
stage_sum 的差即未覆盖部分（约 39–40 µs）。注意力是唯一随位置变化的
stage（12.9 µs @ p=0 → 42.1 µs @ p=511）；整个 block 由 GEMV 主导
（7 个量化投影 ≈ 100 µs）。

**3. 内存 sanitizer**（v0.1 证据
`benchmarks/sanitizer_decoder_block.txt`；v0.1.1 证据
`benchmarks/sanitizer_decoder_block_v011.txt`，含默认配置与泛化配置两次
运行；重跑方式）：

```sh
cd build
/usr/local/cuda-11.8/bin/compute-sanitizer --tool memcheck \
  tests/test_decoder_block data/block_v01.cudalm \
  data/block_v01_golden_p0.cudalm data/block_v01_golden_p7.cudalm
# → ERROR SUMMARY: 0 errors
# v0.1.1 泛化配置：data/block_general.cudalm + 对应 golden（同文件）
```

**4. 运行时无 PyTorch 守卫：**

```sh
bash scripts/check_no_torch.sh
# → forbidden_deps_check OK
```

## v0.1 停止条件清单

- [x] 一个 Llama 风格 decoder block 在固定种子合成权重上端到端运行
- [x] p=0 与 p=7 的端到端黄金 PASS（1e-2 容差；按契约逐位钉死：RoPE
      恒等、18 个权重张量、带种子 KV 历史）
- [x] 16/16 ctest 全绿（CPU + CUDA）
- [x] `compute-sanitizer --tool memcheck`：0 错误
- [x] CUDA 事件逐 stage 时延分解；p=0 / p=511 JSON 已提交
- [x] 运行时不依赖 PyTorch/Python（守卫脚本，Python 仅在 `tools/`）
- [x] 逐文件溯源已记录（上游移植 + 原生清单）
- [x] 工作树干净；全部证据产物已提交；remote `origin`
      （`github.com/trump253/cudalm-agent`，SSH 传输）已配置，`main`
      已推送；无 force push

## v0.1.1 完成清单（架构清理）

- [x] 投影形状契约泛化：Q/K/V/O 形状不再假设 Q 宽度 = hidden_size；
      新增 ctest `test_decoder_block_general`（H=1024、n_heads=16、
      head_dim=128 → q_proj_out=2048 ≠ H）PASS
- [x] benchmark 计时语义拆分为 `stage_sum_us` / `whole_block_gpu_us` /
      `host_api_wall_us` 三视图；stage 之和不再称为"总端到端时延"；
      JSON 内 `timing_semantics` 显式记录
- [x] 速率指标更名 `block_steps_per_second_mean`（基于整块 GPU 时间
      均值；一个 decoder block 步 ≠ 一个模型 token）
- [x] p=0 / p=511 benchmark 按新 schema 重新生成并提交
- [x] 文档明确权重为固定种子合成权重；HF safetensors 摄入归属
      Qwen bring-up
- [x] 18/18 ctest 全绿；`compute-sanitizer` 0 错误（默认 + 泛化配置，
      证据 `benchmarks/sanitizer_decoder_block_v011.txt`）；no-torch
      守卫 PASS
- [x] 标签 `v0.1.1` 已打；`main` 已推送
- [x] 未新增模型架构、未改 W4A16 kernel 算法、未重新优化 Attention、
      未扩大 scope

**停止（STOP）。** 按任务简报要求，开发在 v0.1.1（架构清理）之后停止。
以上内容均不超出单个 decoder block 的范围。下一阶段（未开始）：
**CUDALM v0.2 — Qwen3.5 混合架构 bring-up**（含真实 HuggingFace
checkpoint 权重摄入）。

## v0.2 Phase B 完成清单（Qwen3.5 Full Attention，BF16）

分支 `v0.2-qwen35`。范围 = Qwen3.5 全注意力层（layers 3/7/11/15/19/23）
的 BF16 运行时路径：BF16 W4A16 GEMV、零中心 BF16 RMSNorm、partial RoPE
（复制频率布局）、Q/K norm、GQA 注意力、attention gate、BF16 逐元素、
Qwen35 KV cache、`Qwen35FullAttentionLayer`。W4A16 权重契约不变
（G=128、q∈[-7,7]、scale FP16、FP32 累加），激活/输出走 BF16，
**绝不静默转 FP16**。oracle = 官方 pinned transformers（`fc9137225880`）
+ 真实 checkpoint + 同一 W4A16 反量化权重。

- [x] BF16 W4A16 GEMV（`int4_gemv_bf16`）单元位级/0 错误
- [x] 零中心 RMSNorm / split / partial RoPE / KV 写 / 注意力 / 逐元素
      kernel 单元位级/0 错误（RoPE 对精确舍入 CPU 参考位级一致）
- [x] 真实 checkpoint 硬门 `test_qwen35_full_attention_golden`（layer 3，
      seed 20260209）：p=0 最差 6.1e-05、p=5（5 行非零 KV 历史）最差
      4.9e-04，21 stage 全 PASS（tolerance 1e-2，理由见
      `include/cudalm/stage_compare.h`）
- [x] p=0 不变式位级成立（rope==q/k_norm、attention_raw 每头 == 对应 v 行、
      KV 行 == stage 拷贝）；p=5 KV 历史位级 round-trip
- [x] 旧回归全绿 + Phase A ingestion：**27/27 ctest**
- [x] `compute-sanitizer --tool memcheck` 0 错误（kernel/GEMV 单元 +
      bench 全层 p=0/p=5；golden 测试的 sanitizer 启动器挂起说明见
      `benchmarks/sanitizer_qwen35_full_attention.txt`）
- [x] no-torch 守卫 PASS（`scripts/check_no_torch.sh`）
- [x] 时延基准 p=0/p=5 JSON 已提交（`benchmarks/results/`，报告性，非调优）
- [x] 量化保真度 REPORT ONLY（`*.fidelity.json`，权重 cosine≈0.992、
      层输出 cosine≈0.982），不进硬门
- [x] 文档 + 溯源更新（`docs/qwen35_architecture.md` §5/§11.1/§14、
      `docs/provenance.md`、本清单）
- [x] 工作树干净；commit 已推送 `v0.2-qwen35`

**停止（STOP）。** 按交接简报要求，Phase B（Full Attention）完成后停止，
**不自动进入 Phase C（DeltaNet）**。下一阶段（未开始）：Phase C
`Qwen35DeltaNetLayer` + state 转移硬门；Phase D 4 层混合 micro-stack。

 ---

## v0.4：文本生成（`cudalm-generate` CLI）

v0.4（Phase A/B/C，分支 `v0.4-generation`）在 Qwen3.5-0.8B 上提供
**single-request、serial prefill** 的 token 级生成：原生 tokenizer
（prompt → ids → 文本，oracle-exact）+ 生成核（greedy 为 Phase A/B
冻结路径；Phase C 增加基础 sampling）+ 一个真实可用的 native CLI。
**不是** production serving engine（无 streaming / chat template /
batching / scheduler / Paged KV —— 见下"当前限制"）。

### 用法

```bash
cmake -S . -B build && cmake --build build -j

./build/cudalm-generate \
  --model build/data/qwen35_08b_full.cudalm \
  --tokenizer build/data/qwen35_tokenizer.cudaltk \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  --seed 42
```

- 成功：stdout **只有生成的文本**（binary-safe / length-aware 写入：
  原生 decode 合法产生的 embedded NUL 字节不会被截断）；错误：
  stderr + 非零退出（1 = 运行时失败：model/tokenizer 加载、生成
  契约；2 = 用法错误：缺参 / 坏参数 / 非法 sampling config）。
- 模式：默认 **greedy**（冻结 Phase A 路径，bit-for-bit）；
  `--temperature` / `--top-k` / `--top-p` 任一出现 → sampling
  （未显式给 `--temperature` 时默认 1.0）；`--greedy` 显式关闭
  sampling（与 sampling 参数互斥）；`--seed` 只在 sampling 模式生效
  （相同 prompt + config + seed → 完全相同输出）。
- `--help` 打印完整用法。

### Sampling 语义（摘要）

固定流水线 `temperature → top-k → top-p → normalize → sample`
（scaled/max/exp 用 double，对任何合法有限正 temperature 都保持
数值合法）：`logits/T`；top-k 留最高的 k 个（`top_k == 0` 禁用、
`top_k < 0` 非法、`>vocab` 时 clamp 到 vocab，tie → 最小 id）；
top-p 在 k 幸存者上保留 cumulative 概率达到 p 的最小前缀（≥1 个）；
softmax 先减 max（数值稳定）。CLI 的 `--temperature` 文本若 overflow 到 inf 或
underflow 到 0（含 `strtod` 级 underflow，如 `1e-5000`）→ usage error
（不静默变 inf / 0/greedy）。
详见 `docs/qwen35_architecture.md` §21。

### 当前限制（v0.4 边界）

single request / serial prefill（correctness-first，非性能声明）；
无 streaming、无 chat template / 会话历史、无 batching / chunked
prefill、无 multi-request / scheduler / continuous batching、无
Paged KV / state pool、无 beam search / repetition / frequency /
presence penalty / typical / min-p / speculative decoding、无 HTTP
server / OpenAI API、无 NCU / CUDA Graph / kernel fusion / 性能调优。
后续阶段（v0.5+）再进入。

### v0.4 最终 evidence

`V04_EVIDENCE_SHA = 5aba21fe0b351079850600f3f8fe7f55a77c8745` —— 完整 ctest + `scripts/check_no_torch.sh`
+ tokenizer quick differential validation 于该 SHA（clean tree、
HEAD == SHA）执行；失效规则：此后任何 `src/`/`include/`/`tools/`/
`tests/`/functional CMake 修改 → evidence 失效必须重跑（仅
 docs/evidence 修改——文档 + benchmark 证据记录——不失效）。细节见
 `docs/qwen35_architecture.md` §21.6 与
`docs/provenance.md`（v0.4 Phase C）。

**sign-off**：external reviewer 判 PASS —— v0.4 正式 **DONE /
FROZEN**，`v0.4-generation` 已 merge 进 main。

## v0.7：Profile-Guided 性能优化（continuous batching serving 路径）

**状态：DONE / FROZEN**（`v0.7-profile-opt` 已 merge 进 main）。

v0.7 在 v0.4 的 Qwen3.5-0.8B continuous batching serving 路径上做
**profile-driven 性能优化**（PROFILE FIRST：先量化 baseline，再按证据优化）。
四个阶段都走完整证据链（baseline → BIT-EXACT 门 → microbench → Nsys → paired
E2E → KEEP/REJECT），并**保持逐位精确**（full logits/token/state EXACT；不改
数值语义、不引入 CUDA Graph、不重调 W4A16/DeltaNet）。

### 阶段结论

| 阶段 | 内容 | 结论 |
|------|------|------|
| Phase A | profiling baseline（PROFILE FIRST，无优化） | **DONE** |
| Phase B | W4A16 GEMV row-tile 变体 | **REJECTED**（提升 selected kernels/shapes，但无稳健 E2E 收益，未进 production） |
| Phase C | DeltaNet delta-rule 变体 | **REJECTED**（delta-rule GPU 时间大幅下降，但 paired E2E 门未过，未进 production） |
| Phase D | **BIT-EXACT fused residual-add + RMSNorm**（post-attention 残差路径，4 处） | **KEPT**（进 production） |

> Phase B / C 候选被 **REJECTED**，**不是** production 提速。v0.7 production
> runtime 相对 Phase-A baseline 的唯一功能性改动是 Phase D 的 fused
> add+rmsnorm（README 不堆 profiler 细节，完整数据见下链接）。

### production 收益（仅列有证据支持者）

- **kernels/traversal：442.6 → 418.6（−24 / traversal）**（exact，= −1 launch/层 × 24 层）
- **host `cudaLaunchKernel`：约 −125 µs / traversal**（paired Nsys capture 下）
- **canonical paired E2E：mean delta ≈ −1.238 ms，95% CI [−2.437, −0.040] ms**（20 对 fresh-process 配对，CI 不含 0 → KEEP）
- 全量 ctest **63/63**、`check_no_torch` **CLEAN**、compute-sanitizer **0 errors**、full logits/token/state **EXACT**。

Phase D fused kernel 为 **BIT-EXACT**（residual 与 norm 双双 `memcmp` 一致），
冻结 RMSNorm 归约树 / 算子序不变，RMSNorm 基于**已 BF16 舍入的 residual** 计算。

### 详细证据（链接）

- [`docs/v07_profiling.md`](docs/v07_profiling.md) — Phase A profiling baseline
- [`docs/v07_w4a16_optimization.md`](docs/v07_w4a16_optimization.md) — Phase B W4A16 实验（REJECTED）
- [`docs/v07_deltanet_optimization.md`](docs/v07_deltanet_optimization.md) — Phase C DeltaNet 实验（REJECTED）
- [`docs/v07_final_performance.md`](docs/v07_final_performance.md) — Phase D final sign-off（KEEP + 硬门 + Phase-A vs v0.7 对比）

### v0.7 最终 evidence

`V07D_FINAL_FUNCTIONAL_SHA = 9edd9ef6aec84b8dcf66a262652c2e285ff73793`
（production runtime 最终态 = fused add+rmsnorm）；`V07_FINAL_HEAD =
c5d20efe3424a82ac0c6e4b01849645c74a1aa40`（其后 commits 均为 docs/evidence-only，
无 runtime functional drift）。完整 ctest（63/63）+ `scripts/check_no_torch.sh`
+ compute-sanitizer 于该 SHA（clean tree、HEAD == SHA）执行；失效规则：此后任何
`src/`/`include/`/`tools/`/`tests/`/functional CMake 修改 → evidence 失效必须重跑
（仅 docs/evidence 修改不失效）。

**sign-off**：v0.7 正式 **DONE / FROZEN**，`v0.7-profile-opt` 已 merge 进 main。

## v0.8：Multi-turn / Session Runtime

**状态：v0.8 DONE / FROZEN / MERGED**（Phase A + B + C + D complete，
external review **PASS**；开发分支 `v0.8-session-runtime`（从
`cf28abd`（v0.7 merge）创建）已 `--no-ff` merge 进 main，tag
**`v0.8`**）。

**最终冻结点**：

```text
V08D_FINAL_FUNCTIONAL_SHA: 482f4ae7f7a6a0f87a42b64fb7ee7b9cd7682ab1
V08D_FINAL_TESTED_SHA:     bd8329a62096b5f00ec3b4e483df07d3d0e6f8be
V08_FINAL_BRANCH_HEAD:     3d212d2a8d547d5d75f37cb8b40b67989d215a46
```

v0.8 把"一次性 Request 生命周期"升级为 **persistent Session + multiple
Requests/turns**：**SessionId 与 RequestId 分离**；一个 Session 跨 turn
持有模型推理状态（Qwen3.5 hybrid：paged KV + DeltaNet conv/recurrent +
logical position + slot/page 所有权 + lifecycle metadata）；一个 Request
是绑定到 Session 的**临时操作** —— request 完成后 KV pages / Delta
conv / Delta recurrent / position **全部保留**（只有 `reset_session` /
`destroy_session` 才执行明确的 reset/release）。

### Phase A：Session abstraction + persistent-state lifecycle

- **新控制面**：`SessionId` / `SessionState` / `Session` /
  `SessionManager`（`include/cudalm/session.h` + `src/runtime/session.cpp`）
  —— 冻结 v0.5 `Qwen35StateManager` 之上的**薄非拥有绑定层**（一个
  session == 一个 bound sequence）；池的 zero-on-release / zero-on-reset /
  精确 byte accounting 语义**全部继承**，**未改动任何模型数学语义**；
- **所有权迁移**：state lifetime 从 request 迁到 session —— request 完成
  = **无 state 操作**（v0.6 request-scoped 模式冻结并存；scheduler 接入
  属 Phase C）；
- **overflow policy**：`fits()` context 容量纯查询；
  `context_length + new_tokens > max_context` → **明确 reject**（无
  eviction、不静默丢最早 token/page）；
- **硬门**：`test_session_manager`（CPU，真实设备池：create/destroy、
  事务式 OOM、A/B 隔离、reset 不影响他者、destroy + slot/page 复用无
  残留、fits 精确边界、unknown-id fail loud）+
  `test_qwen35_session_runtime`（真实 Qwen3.5-0.8B-Base checkpoint：
  request 边界持久化 == 一次性连续 reference **bit-identical**、真实使用
  下隔离、reset parity、destroy + 复用，全 memcmp）。

### Phase B：incremental multi-turn execution（单 Session 真多轮）

- **新 API**：`SessionGenerator::generate_turn(session_id,
  new_input_tokens, max_new_tokens, eos_token_id, sampling, stream,
  observer)`（`include/cudalm/session_generator.h`）—— 薄引擎，只驱动
  冻结的 `forward_token_with_state`，**append-only**：turn 只 forward
  新 input（从 session 当前逻辑长度起），不 re-prefill 历史、不 reset
  session、不拷贝/重建 KV 或 Delta state；随后从既有 KV + Delta
  conv/recurrent + position 继续生成；per-turn 全新 `Sampler`
  （RNG 不跨 turn）；`eos_token_id == -1` = 无 EOS gate（v0.6 约定）；
- **commit 契约（pinned）**：turn 返回的每个 generated token（**包括**
  触发停止的 EOS / max_new 最后一个）**都已提交**进 session state ——
  turn 后 session 的 KV / Delta / logical length 与已提交 token history
  完全一致（无"滞后最后一 token"语义；与冻结的 v0.4 one-shot 请求
  `N + m - 1` 契约不同，v0.4 未改）；
- **语义**：`max_new_tokens == 0` = input-only append；EOS token 提交后
  停止；context overflow（`length + input + max_new_tokens >
  max_seq_len`，精确边界相等 = 接受）= **明确 reject，无
  eviction/truncation**；preflight 失败 = **zero mutation**；执行中
  失败（如 KV OOM）= 已提交 token 保留、失败 token 不提交、停在最后
  成功 token 边界（无 snapshot/rollback）；
- **硬门**：`test_session_turn_contract`（CPU，无 checkpoint：preflight
  全失败路径逐一 zero-mutation，含 overflow reject 与精确边界接受）+
  `test_qwen35_session_generation`（真实 Qwen3.5-0.8B-Base checkpoint：
  **turn 1 + turn 2 == 等价 one-shot continuous execution，逐步全量
  logits / generated ID / 长度 / paged KV / Delta conv+rec 全部
  bit-identical**；input-only、EOS commit、overflow、reset 后重执行 ==
  fresh、KV OOM partial-commit，全 memcmp）。

### Phase C：scheduler + Session 集成 / 多 session interleaving

- **核心关系**：**Session = persistent model state owner；Request = one
  scheduled turn/job**；`RequestId != SessionId != SequenceId`；**request
  终态 ≠ session 销毁/重置**（只有 `reset_session` / `destroy_session`
  能动 session state）；
- **additive API**（legacy `admit()` 行为 byte-identical，未动）：
  `Scheduler(fwd, mgr, stream, SessionManager* sessions = nullptr)` +
  `admit_session_turn(session_id, new_input_tokens, max_new_tokens,
  eos_token_id, sampling, &request_id)` + `session_busy(session_id)`。
  session-bound request **绑定 session 的已有 bound sequence**（不创建
  新 sequence、不 replay 历史、从当前 length 追加）；preflight 失败
  （unknown session / **busy session（每 session 至多一个 live
  request）** / 非法 token・eos・sampling / `max_new < 0` / context
  overflow（`length + input + max_new > max_seq_len`，精确边界接受，
  无 eviction）/ **instance identity（SessionManager 必须绑定本
  scheduler 的 state manager，跨 manager 的 SequenceId 数值巧合不
  会静默驱动错误 sequence）**）= **zero mutation**；
  `max_new_tokens == 0` = input-only turn；
- **commit 语义（hard gate，sampled != committed）**：session turn 的
  generated token **先 forward 成功（commit）才可能触发终态** ——
  EOS / max_new 的 stop 判定在 commit **之后**（legacy 的"采样时判定、
  末 token 不 forward、N+m-1"冻结不变）；一个 commit m 个 generated
  的 turn forward 共 `input + m` 次；cancel/failure 时 pending
  （已采样未 commit）token **不进入 session 历史**，committed 状态
  保留、session live、无 rollback；
- **多 session batching**：session-bound request 与 legacy 共用现有
  FIFO snapshot / one logical token per request per iteration /
  decode cohort / **true batched forward** / batch fallback / per-
  request sampler 隔离；每个 batch 行访问自己的 SequenceId / KV /
  Delta slot；turn 终态后 session + sequence + KV/Delta 保留，下一
  turn 从上一 turn 的最终 committed state 继续；
- **硬门**：`test_qwen35_scheduler_session`（CPU：准入零 mutation 全
  路径、不创建 sequence、terminal != retired、commit 计数、cancel /
  failure 保留 committed、next turn 不 replay、input-only turn、legacy
  不变）+ `test_qwen35_scheduler_session_integration`（真实
  Qwen3.5-0.8B-Base checkpoint：session A/B 各两 turn 交错执行 vs
  独立连续参考 —— **每步全量 logits（含 batched 行）/ generated ID /
  turn 边界与最终 length / paged KV / Delta conv+rec 全部
  bit-identical**；batch 证据 `batch_forward_calls == 4`、
  `max_batch_size == 2`、`decode_cohort_trace == [1,2,2,1,2,2]`；
  busy / overflow 零 mutation；拆除后 accounting 归零）。

### Phase D：text-level multi-turn + demo + final sign-off

- **text facade**（`Qwen35SessionTextGenerator`，
  `include/cudalm/session_text_generator.h`）：薄 facade，不写
  generation loop —— 只把冻结 contract 串起来：
  `Qwen35Tokenizer::encode`（native，oracle-exact）→
  `Scheduler::admit_session_turn`（Phase C session-bound，绑定已有
  bound sequence，**append-only**）→ `Scheduler::run`（frozen
  control plane，commit-then-stop）→ `Qwen35Tokenizer::decode`。
  **turn N 只 encode / append 新文本**，turn 1..N-1 不 re-encode /
  不 replay；`ok == true` 时 result 的**每个** generated id 都已
  commit 进 session state；错误（invalid UTF-8 / unknown / busy /
  overflow / forward failure）= fail loud，session 停在 last
  committed boundary 且保持 LIVE（无半个 turn）；
- **`cudalm-chat` CLI**（`tools/cudalm_chat.cpp`）：persistent
  **raw text** 多轮 demo REPL（`--model` / `--tokenizer` /
  `--max-new-tokens` / `--temperature` / `--top-k` / `--top-p` /
  `--seed` / `--greedy` / `--page-tokens` / `--pages` / `--slots`）。
  用户输入**原样（逐字节、不 trim）** encode 追加到同一 persistent
  session（**不加** chat template / special token / separator）；
  response fully committed 后才显示。**这是 persistent text-session
  demo，不是完整 instruct/chat-template serving API**（Qwen3.5-0.8B-
  Base 是 base 模型，仓库没有冻结的 official chat-template
  contract）。**REPL 命令语义（pinned，exact whole-line
  matching）**：命令识别**只认整行精确匹配**——`reset` / `quit` /
  `exit` 是命令；` reset `（带空格）是 **raw text** 不是命令；
  **只有真空行 `""` 被忽略（"empty line is ignored"）**；
  whitespace-only 非空行（如 `"   "`）是 raw-text turn（不被 trim
  掉）。非命令行逐字节传给 `generate_turn`（不增删任何空格 / 换行 /
  separator / special token）。generated text 用 **length-aware
  （binary-safe）** 写输出（embedded NUL 逐字节保留，不用 `%s`）；
  response 末尾若无 `\n`，CLI 补一个**仅显示用**的换行（UX，**不
  进入** session / token history）；
- **真实 CLI 示例**（真实 checkpoint 输出）：

  ```text
  $ ./build/cudalm-chat --model build/data/qwen35_08b_full.cudalm \
      --tokenizer build/data/qwen35_tokenizer.cudaltk --max-new-tokens 24
  cudalm-chat: v0.8 persistent text session (session 1)
    model: build/data/qwen35_08b_full.cudalm | sampling: greedy | max_new_tokens: 24
    RAW TEXT contract: your input is appended VERBATIM — no
    chat template, no special tokens, no separator. ...
  user> The capital of France is
  model>  located in the northern part of the country.
  A. True
  B. False
    [turn 1: +5 input, +24 generated, stop max_new_tokens, context 29]
  user> And its most famous monument is the
  model>  Eiffel Tower.
  A. True
  B. False
    [turn 2: +7 input, +24 generated, stop max_new_tokens, context 60]
  user> quit
  session destroyed (state released)
  ```

  （turn 2 从 turn 1 的 committed state 继续：context 29 → 60 =
  29 + 7 + 24 精确累加；turn 2 没有 re-encode / re-forward turn 1。）

- **硬门**：`test_qwen35_session_text`（CPU contract gate：真实
  tokenizer artifact + deterministic fake forwarder + 真实池 /
  SessionManager / Scheduler 经 facade 驱动 —— encode →
  session-bound request → decode；turn 2 只 encode 新文本且**不
  replay** turn 1（per-sequence step 计数证明续接）；reset 同
  SessionId 重新开始；错误全路径零 mutation + session LIVE）+
  `test_qwen35_session_text_e2e`（真实 checkpoint + 真实 tokenizer：
  两轮 text turn 与**直接 token-level Phase C 路径**逐轮比较 ——
  encoded ids / generated ids / decoded text / final context length
  全部相同，final context 恰为 `n1 + m1 + n2 + m2`；reset /
  overflow 零 mutation / clean teardown）。

### v0.8 能力总览（portfolio 视角）

| 能力 | 说明 |
|---|---|
| **SessionId vs RequestId** | 两个独立、单调、不复用的 id 空间（外加 SequenceId 共三个）：**Session = persistent model state owner；Request = 一次被调度的 turn/job**；request 终态 ≠ session 销毁/重置 |
| **persistent paged KV + Delta recurrent state** | Qwen3.5-0.8B hybrid（6 full-attention 层 paged KV + 18 DeltaNet 层 conv/recurrent）的 state 跨 request 持久化；zero-on-release / zero-on-reset / 精确 byte accounting（v0.5 池冻结继承） |
| **incremental multi-turn** | turn N 只 forward 新 input，从既有 KV + Delta + position 继续；已证明 incremental == 等价 one-shot continuous（bit-identical） |
| **scheduler + true batched multi-session decode** | session-bound turn 经 v0.6 scheduler；多 session 交错、true batched decode cohort、per-request sampler 隔离；commit-then-stop 语义 |
| **native tokenizer** | PyTorch-free 原生 encode/decode（CUDLMTK1 artifact，对 pinned HF oracle 全量 EXACT） |
| **text-level multi-turn demo** | `cudalm-chat`：raw text persistent session（无 chat template），response fully committed 后显示 |
| **sampling** | greedy / seeded temperature + top-k + top-p（per-request 全新 sampler，RNG 不跨 request/turn） |
| **reset / destroy lifecycle** | `reset_session`（state 归零、同 SessionId、序列复用）/ `destroy_session`（释放 slot/pages；池 accounting 归零有硬门） |

### v0.8 当前限制（诚实清单）

- 模型为 **Qwen3.5-0.8B-Base**（base 模型；无 official chat template ——
  仓库没有可证明的 pinned template contract，Phase D 的 text demo 是
  raw text completion，不是 instruct serving）；
- **无 HTTP / OpenAI-compatible API**（只有进程内 C++ API + 两个 CLI）；
- **无 streaming**（turn 的 generated token 在 request 终态后整体可见）；
- **无 eviction / sliding window / TTL / LRU**（context overflow =
  明确 reject，精确边界接受）；
- **单 CUDA stream**（池与 model 同 stream；无 multi-stream / CUDA
  Graph）；
- **同 Session 同时至多一个 live turn**（busy session 的第二个 live
  request 直接 reject，fail loud）；
- 生成循环 correctness-first：每步 host 全量 logits D2H + host 采样；
  池容量（pages/slots）构造期固定。

详见 [`docs/v08_session_runtime.md`](docs/v08_session_runtime.md)（§1–
§12：四个 phase 的 contract、测试与验收证据）与
[`docs/provenance.md`](docs/provenance.md)（逐 phase 的 SHA 绑定
证据）。

## v0.9：Serving Hardening

### Phase A：Serving Admission / Backpressure / Resource Guardrails

**状态：Phase A 完成（external review PASS / FROZEN）**（分支
`v0.9-serving-hardening`，从 `1d6c83a`（v0.8 merge）创建；**不**在
main 开发，**不** merge）。

在冻结的 v0.8 persistent Session + Scheduler 之上增加一个**独立薄
控制层** `ServingController`（policy / quota / admission /
observability；frozen runtime 仍负责 correctness，零 frozen code
改动）：

- **Serving limits**（`ServingLimits`）：`max_sessions`（live session
  配额）、`max_live_requests`（live request 配额）——**quota 三态**：
  `-1 = unlimited`、`0 = zero capacity`（拒绝一切 admission）、
  `N>0 = capacity N`；`max_context_tokens_per_session`（可选
  per-session policy cap，`0 = 禁用`——它不是 quota；构造时钳制到
  模型 `max_seq_len`——只能收紧、不能突破）；
- **reject early / fail loud / zero mutation**：limit rejection 发生
  在触碰 frozen runtime **之前**——no SessionId / RequestId
  consumed、no sequence created、no KV page / Delta slot / logical-
  length mutation；frozen Phase C preflight（identity / busy / model
  overflow 等）仍由 scheduler 执行并原样透传，不重复实现；
- **quota 生命周期**：request 终态（Finished / Cancelled / Failed）
  恰好一次释放 request 配额（live 计数从 tracked id + scheduler
  status 推导，结构上无 double-decrement 路径）；`destroy_session`
  释放 session 配额；`reset_session` 不释放；
- **serving stats**（刻意轻量）：`live_sessions` / `live_requests` /
  `total_admitted_sessions` / `total_admitted_requests` /
  `rejected_session_limit` / `rejected_request_limit` /
  `rejected_context_limit`。

Phase A **不做**：streaming、deadline / timeout、TTL / LRU、eviction、
HTTP server、OpenAI API、multi-stream、CUDA Graph、kernel 优化、chat
template、动态 quota 系统（streaming / deadline 已在 Phase B 做入，
其余属后续 Phase）。

### Phase B：Committed-token Streaming + Cancellation / Deadline

**状态：Phase B 完成（等待 external review）**。

在 `ServingController` 上增加（单线程、单 CUDA stream，无 HTTP / 无
异步线程；不修改模型数学 / CUDA kernel / v0.8 commit-then-stop
语义）：

- **pull-based committed-token streaming**（`ServingEvent` +
  `step_stream()` / `run_stream()` / `poll(request_id)`）：**commit-
  before-visible**——只有 `generated[0 .. committed_generated)`
  （forward 成功进入 Session KV / Delta / position 的 token）可
  emit；**pending token 永不 emit**；每个 committed token
  **exactly once、按序**；**EOS / max-new 最后 token 先 emit 再
  报告 terminal**；继续驱动 frozen `Scheduler::step()`（不复制
  generation loop）；
- **explicit cancellation**：cancel 后不再 forward；committed 且未
  emit 的 prefix 可正常 drain；pending 不 emit；**Session 保持
  live**（context / KV / Delta 停在最后 committed boundary）；quota
  正常释放；next turn 从 boundary 继续；
- **per-request deadline**（monotonic clock，可注入 fake clock）：
  **每次 scheduler step 之前检查**——过期 → cancel before its next
  forward → no additional token commit → session 保持 live；
  **deadline = cooperative boundary between scheduler steps**（不
  preemption）；serving-layer 独立 termination reason
  （`deadline_exceeded`，不改 frozen `FinishReason`）；
- **多 request streaming**：A/B 同时 live 仍走 batched decode——一次
  step 多 request 各自新 committed token，serving 层分别 drain，无
  跨 request contamination。

Phase B streaming 目前是 **token-level（committed token IDs）**：
text-byte streaming / incremental UTF-8 decoder / HTTP / OpenAI API /
threads 属后续 Phase。

详见 [`docs/v09_serving_hardening.md`](docs/v09_serving_hardening.md)
与 [`docs/provenance.md`](docs/provenance.md)（SHA 绑定证据）。
