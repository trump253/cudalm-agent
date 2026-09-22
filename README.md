# CUDALM

原生 C++/CUDA 量化 LLM 推理引擎。**v0.1** 实现一个完整的 Llama 风格
decoder block —— W4A16 量化线性层、fp16 激活、单 GPU 自回归解码 ——
**运行时不依赖 PyTorch（也不依赖任何 Python）**。C++17、CUDA 11.8、
`sm_75`（RTX 2080 Ti）。kernel 只接收裸指针 + `cudaStream_t`；Python
只存在于 `tools/`（离线生成测试数据），绝不链接进运行时。

## v0.1 范围 —— 以及刻意不做的部分

已实现（端到端）：

```
x ─► RMSNorm ─► Q/K/V（W4A16 GEMV）──► RoPE（交错对）
  ─► KV 缓存写入 @ position ─► 因果解码注意力（fp32 计算）
  ─► O 投影 ─► + 残差 ─► RMSNorm ─► gate/up（W4A16 GEMV）
  ─► SiLU(gate)·up ─► down（W4A16 GEMV）──► + 残差 ─► y
```

v0.1 刻意不在范围内（v0.1 之后硬停止 —— 见文末清单）：FlashAttention、
批处理（batching）、张量并行、多 GPU、CUDA Graphs、tokenizer、完整
多层模型。

## 钉死的契约

| 项 | 契约 |
|------|----------|
| 配置 | `H=1024, n_heads=8, n_kv_heads=4, head_dim=128, intermediate=2816, group=128, max_seq=512, eps=1e-5, rope_theta=10000` |
| 权重 | W4A16：对称分组 INT4，G=128，q∈[−7,7]，zero_point=0，`scale=amax/7` 以 fp32 计算、以 fp16 存储（**存储的 fp16 scale 即契约**），nibble 打包 low=k=2b / high=k=2b+1（4 位补码）。格式规范：[`docs/weight_format.md`](docs/weight_format.md) |
| KV 缓存 | K、V 各为 fp16 `[n_kv_heads][max_seq_len][head_dim]`；行 (n,t) 偏移 `((n*max_seq_len)+t)*head_dim`；零初始化；越界即致命前置检查 |
| RoPE | 交错对：`y[2i]=a·c−b·s`，`y[2i+1]=a·s+b·c`；cos/sin 表 fp16 `[max_seq, hd/2]`；按行查 `positions[m]`；基址 4B 对齐走 4B 打包路径，否则走标量回退（从不拒绝） |
| 注意力 | GQA 整数映射 `kh = h * n_kv // n_heads`；`scale = 1/√head_dim`；减最大值的 softmax，逐元素除法；**全部 fp32 计算，仅在存储时做一次 fp16 RNE** |
| stage 容差 | 相对黄金判据 `\|a−r\| ≤ 1e-2 + 1e-2·\|r\|`（实测最大偏差 ≤ 2.4e-4，来自 kernel FMA 收缩 vs 黄金的独立 mul+add）。p=0 RoPE 恒等、全部 18 个权重张量、带种子的 KV 历史行均钉死为**逐位精确**；当前位置的 KV 行按"对黄金容差 + 对运行时自身 rope_k/v stage 行逐位一致"校验 |

## 仓库结构

```
include/cudalm/           运行时头文件（decoder_block.h、kv_cache.h、
                          weight_format.h、kernels/{rmsnorm,int4_gemv,rope,
                          kv,attention,elementwise}.h、device_buffer.h …）
src/kernels/              .cu kernel（裸指针 + cudaStream_t）
src/runtime/              decoder_block.cpp、kv_cache.cpp、weight_loader.cpp、
                          golden_loader.cpp、device_buffer.cpp
tests/cpu/                无 GPU 测试：文件格式、跨语言权重一致性
tests/cuda/               kernel 测试 + 端到端黄金测试（共 16 个 ctest 用例）
tools/                    离线 Python（torch 2.3.1+cpu）：
                          convert_weights.py（量化器）、generate_golden.py、
                          common/binfmt.py —— 绝不链接进运行时
benchmarks/               bench_decoder_block.cpp + 已提交的结果 +
                          sanitizer 证据
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

**1. 完整测试套件（16/16）：**

```sh
cd build && ctest
```

覆盖：文件格式往返、跨语言权重字节一致性（Python 量化器 → C++ 加载器）、
逐 kernel 测试（rmsnorm、int4_gemv、rope、kv_cache、attention、
elementwise），以及端到端黄金测试（`test_decoder_block`，位置 0 与 7：
18 个权重张量 + 16 个 stage 张量 + KV 状态，按上表钉死的契约校验）。

**2. 逐 stage 时延分解**（CUDA 事件，17 个 stage + 总计）：

```sh
./build/benchmarks/bench_decoder_block build/data/block_v01.cudalm <position> [iters=100] [out.json]
```

已提交结果（`benchmarks/results/`，RTX 2080 Ti，CUDA 11.8，驱动
570.172.08，iters=100）：

| 位置 | 总时延均值 | tokens/s |
|------|-----------|----------|
| 0   | 126.6 µs | 7 897 |
| 511 | 152.4 µs | 6 562 |

注意力是唯一随位置变化的 stage（13.0 µs @ p=0 → 41.5 µs @ p=511）；
整个 block 由 GEMV 主导（7 个量化投影 ≈ 100 µs）。

**3. 内存 sanitizer**（证据已提交于
`benchmarks/sanitizer_decoder_block.txt`；重跑方式）：

```sh
cd build
/usr/local/cuda-11.8/bin/compute-sanitizer --tool memcheck \
  tests/test_decoder_block data/block_v01.cudalm \
  data/block_v01_golden_p0.cudalm data/block_v01_golden_p7.cudalm
# → ERROR SUMMARY: 0 errors
```

**4. 运行时无 PyTorch 守卫：**

```sh
bash scripts/check_no_torch.sh
# → forbidden_deps_check OK
```

## v0.1 停止条件清单

- [x] 一个 Llama 风格 decoder block 在真实（带种子）权重上端到端运行
- [x] p=0 与 p=7 的端到端黄金 PASS（1e-2 容差；按契约逐位钉死：RoPE
      恒等、18 个权重张量、带种子 KV 历史）
- [x] 16/16 ctest 全绿（CPU + CUDA）
- [x] `compute-sanitizer --tool memcheck`：0 错误
- [x] CUDA 事件逐 stage 时延分解；p=0 / p=511 JSON 已提交
- [x] 运行时不依赖 PyTorch/Python（守卫脚本，Python 仅在 `tools/`）
- [x] 逐文件溯源已记录（上游移植 + 原生清单）
- [x] 工作树干净；全部证据产物已提交；remote 尚未配置（挂载说明见交付
      报告）；无 force push

**停止（STOP）。** 按任务简报要求，开发在 v0.1 之后停止。以上内容均不
超出单个 decoder block 的范围。
