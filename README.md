# CUDALM

原生 C++/CUDA 量化 LLM 推理引擎。**v0.1** 实现一个完整的 Llama 风格
decoder block —— W4A16 量化线性层、fp16 激活、单 GPU 自回归解码 ——
**运行时不依赖 PyTorch（也不依赖任何 Python）**。C++17、CUDA 11.8、
`sm_75`（RTX 2080 Ti）。kernel 只接收裸指针 + `cudaStream_t`；Python
只存在于 `tools/`（离线生成测试数据），绝不链接进运行时。

**v0.1.1（架构清理，当前版本）**：泛化 decoder 投影形状契约
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
