# CUDALM v0.1 引导计划

CUDALM = **原生 C++/CUDA 量化 LLM 推理引擎**（W4A16）。本计划只覆盖
v0.1：一个小型、清晰的运行时 + **一个 Llama 风格 decoder block**，正确
性优先。不做完整模型，不引入框架。

范围按项目简报冻结。章节：环境 / 仓库状态 / 拟定架构 / 权重格式 / 张量
与 KV 布局 / kernel 移植映射 / 黄金参考策略 / 里程碑 / 风险 / 首个任
务。

---

## 1. 环境

| 项 | 值 |
|------|-------|
| 主机 GPU | NVIDIA RTX 2080 Ti（×2），**Turing sm_75**，11 GB |
| 驱动 | 570.172.08（CUDA 运行时驱动 12.8） |
| CUDA 工具链 | **11.8.89**，位于 `/usr/local/cuda-11.8`（`nvcc`、`compute-sanitizer` 均在） |
| 主机编译器 | gcc（C++17），CMake **3.16.3**（见 §8 风险 R3） |
| Python（仅离线工具） | 3.8.10 + **torch 2.3.1+cpu**（为 `tools/` 安装；绝不链接进运行时） |
| 目标架构 | `sm_75`（Turing）。kernel 可使用 fp16 内建（`__half`、`__half2`、`__half22float2`）。 |

遵守的约束：
- 运行时（编译出的 `libcudalm` + 测试/benchmark 二进制）**无 PyTorch、
  无 pybind、无 `at::Tensor`、无 `TORCH_CHECK`、无 Python**。kernel 只接
  收裸指针 + `cudaStream_t`。
- Python **只**存在于 `tools/`（权重转换、黄金参考、离线测试数据）。它
  不是运行时依赖。

---

## 2. 仓库状态

- 工作目录：`/root/code/cudalm-agent` —— 原为空目录。全新 `git init`，
  分支 `main`。无既有历史，无 remote。
- 上游参考（只读，**不**合并进本仓库）：`/root/code/cuda` 的 CUDALab，
  HEAD `cb6a6a9`（tag `v0.7.1`）。只读取了存量 kernel + INT4 量化契约
  （见 §6 移植映射）。
- CUDALab 保持上游 kernel 研究仓库的定位；CUDALM 是下游推理系统仓库。
  移植记录溯源（见 §6 + `docs/provenance.md`）。

计划中的 GitHub remote：`trump253/CUDALM`。若未配置 `origin`，则本地
工作，并在最后告知用户：

```
git remote add origin <repo-url>
git push -u origin main
```

---

## 3. 拟定架构

小型、显式的运行时。无框架、无注册表、无自动微分。

```
CUDALM/
├── CMakeLists.txt
├── README.md
├── docs/            # 本计划、权重格式规范、溯源、笔记
├── include/cudalm/
│   ├── cuda_check.h       # CUDA_CHECK(...) + launch 错误辅助
│   ├── tensor.h           # Dtype, Shape, TensorView（非拥有，仅连续）
│   ├── device_buffer.h    # RAII cudaMalloc/cudaFree，无隐式拷贝
│   ├── model_config.h     # BlockConfig（形状、头数、GQA、head_dim、分组）
│   ├── weight_format.h    # 二进制表头/记录结构 + 常量
│   ├── weight_loader.h    # 解析 + 边界检查 + 上传到 DeviceBuffer
│   ├── kv_cache.h         # KV 布局 + 写/读（RAII）
│   ├── ops.h              # 算子入口（原生 C++ API）
│   └── kernels/           # kernel 声明：rmsnorm, int4gemv, rope, attention, kv, elementwise
├── src/
│   ├── runtime/           # runtime.cpp, weight_loader.cpp, kv_cache.cpp
│   ├── ops/               # 算子 host 端 launcher
│   └── kernels/           # .cu kernel 实现
├── tests/
│   ├── cpu/               # 权重文件解析、张量元信息、int4 打包/偏移
│   ├── cuda/              # 逐算子单元测试（对 CPU 黄金）
│   ├── common/            # 共享比较/分配辅助
│   └── golden/            # 逐 stage 黄金比较驱动
├── tools/
│   ├── convert_weights.py # 构建 .cudalm 权重文件（量化 + 打包）
│   ├── generate_golden.py # 固定种子的 decoder block 用例 → 黄金文件
│   └── common/            # 共享 python IO（cudalm 二进制读取器）
├── benchmarks/
│   └── bench_block.cu     # 逐 stage CUDA 事件时延分解
└── scripts/               # 构建 + 测试 + 清理辅助脚本
```

设计规则：
- `DeviceBuffer`：RAII `cudaMalloc`/`cudaFree`，显式 `size`，无隐式
  host↔device 拷贝。仅可移动（move-only）。
- `TensorView`：非拥有 `{ptr, dtype, shape}`。v0.1 **仅连续布局**；不模
  型 stride（构造时断言连续）。
- `CUDA_CHECK(...)`：单一错误层。每个 `cuda*` 调用与每次 kernel launch
  都经过它；launch 失败后随即用 `cudaGetLastError()` 检查。
- 运行时持有唯一显式 `cudaStream_t`；**不**依赖默认流假设。所有算子以
  参数接收 stream。

### Decoder block（v0.1 固定拓扑）

```
x ─▶ RMSNorm ─▶ [Q,K,V W4A16 Linear] ─▶ RoPE(Q,K) ─▶ KV write/read
    ─▶ causal Attention ─▶ W4A16 O-proj ─▶ (+x) residual
    ─▶ RMSNorm ─▶ [gate,up W4A16] ─▶ SiLU(gate)*up ─▶ W4A16 down ─▶ (+residual)
```

v0.1 配置（单 GPU、fp16 激活、W4A16 线性层、一个 block）。具体数值取
小，以便快速、确定性测试：

| 参数 | 值 |
|-------|-------|
| hidden `H` | 1024 |
| `n_heads` | 8 |
| `n_kv_heads` | 4（GQA 比例 2；接口预留 GQA） |
| `head_dim` | 128 |
| `intermediate`（SwiGLU） | 2816（= 1024*2.75，取分组倍数） |
| `group_size` | 128（固定，W4A16 契约） |
| `max_seq_len` | 512 |
| `eps`（RMSNorm） | 1e-5 |
| `rope_theta` | 10000.0 |

`intermediate` 必须是 `group_size`（128）的倍数且为 2 的倍数（INT4 打
包）。2816 = 22*128 ✓。所有投影 `K = H = 1024`（128 的倍数 ✓）。

GQA 映射：query head `h` 读取 KV head `h / (n_heads/n_kv_heads)`。

---

## 4. 权重格式

CUDALM 自有一个简单、带版本、**确定性**的二进制容器（`.cudalm`）。加载
器为纯 C++（无 PyTorch）。Python `convert_weights.py` **写入**它；
`weight_loader.cpp` **读取**它。两端共享 `docs/weight_format.md` 中的
规范（单一事实来源）。

表头（小端）：

```
magic     : 8s  b"CUDLMW01"
version   : u32 = 1
flags     : u32（保留，0）
n_tensors : u32
表头中为 config 预留的字节：
  ModelConfig 序列化（见 §3）— 定宽字段，小端
张量记录表：n_tensors × TensorRecord
payload    : 张量字节块，每个 16 字节对齐
```

`TensorRecord`：

```
name_len  : u32
name      : name_len 字节（ASCII，无 NUL）
dtype     : u8   (FP16=1, INT4_PACKED=2, FP16_SCALE=3)
ndim      : u8
dims      : u32[ndim]
offset    : u64  （相对 payload 起始的字节偏移）
byte_size : u64  （payload 字节大小）
align     : u8   （payload 要求的对齐，2 的幂，<=16）
pad       : u8[7]
```

支持 v0.1 所需的三种 dtype：
- **FP16** 张量（RMSNorm 权重、RoPE cos/sin；激活不存储）
- **INT4_PACKED** 张量（`W_packed`，uint8，每字节两个有符号 INT4）
- **FP16_SCALE** 张量（分组 scale `[N, K/128]`）

W4A16 配对靠约定 + 校验：每个 `INT4_PACKED` 权重 `X.weight` 都有配套的
`X.scale`（`FP16_SCALE`），形状 `[N, K/128]`。加载器校验名称/形状，不
匹配即拒绝。

加载器保证：
- **确定性**解析（无随机性、顺序稳定）。
- **带版本**（`version` 字段；未知版本 → 硬错误）。
- **边界检查**（每个 `offset+byte_size` 都在 payload 内；表大小与声明数
  量一致；无重叠；遵守对齐）。
- **显式对齐**（payload 16B 对齐；逐记录检查 `align`）。
- 加载路径**无 PyTorch**。

RoPE `cos/sin` 表由 Python 转换器（固定种子）生成，存为 FP16 张量，使
运行时加载时无需超越函数（带宽/正确性确定性）。

---

## 5. 张量 / KV 布局

所有张量连续行优先。传给 kernel 的指针是 `DeviceBuffer` 基址（或经过校
验的偏移）。

- `TensorView{ void* data; Dtype dtype; std::vector<int64_t> shape; }`
  `.bytes()`、`.numel()`、`.contiguous()`（v0.1 中恒为 true）。

**KV 缓存**（解码、单请求）。简单、可审计的布局：

```
K, V each: [n_kv_heads][max_seq_len][head_dim]  fp16
```

- `K_base + ((n * max_seq_len) + t) * head_dim + d`
- `t` = 位置（0 基），`n` = kv head，`d` = head 维度。
- 写路径：在解码位置 `p`，对所有 `n_kv_heads` 写入当前 `k[n, p, :]` 与
  `v[n, p, :]`。读路径：注意力对每个 query head 所使用的 kv head 读取
  `[0..p]`（含两端）。
- `KvCache` 为 RAII（两个 `DeviceBuffer`：K 与 V），预分配到
  `max_seq_len`。提供 `write(position, k, v)` 与只读访问器。

刻意采用扁平布局（无分页、无环形）。`position` 边界在 host 端检查
（`0 <= position < max_seq_len`）。

---

## 6. Kernel 移植映射：CUDALab → CUDALM

上游参考：CUDALab `cb6a6a9`（tag `v0.7.1`）。只读取了存量 kernel +
INT4 契约。每个 kernel 的溯源记录在 `docs/provenance.md` 及文件头（上
游 tag/commit、原始 kernel、CUDALM 移植 commit）中。

| # | CUDALab 上游 | 原始 kernel | CUDALM 移植 | 改动 |
|---|------------------|-----------------|-------------|---------|
| 1 | `kernels/rmsnorm/rmsnorm_v4.cu` | `rmsnorm_v4_half_kernel<PER>`（存量 `v4_vec_reg`） | `src/kernels/rmsnorm.cu` → `rmsnorm_fp16` | 弃 fp32 路径（v0.1 仅 fp16）。`at::Tensor`/`TORCH_CHECK`/`getCurrentCUDAStream` 换为裸指针 + `cudaStream_t` + `CUDALM_CHECK`。保留 256 线程块、`PER∈{4,8,16,32}`、`float4`/`half2` 向量加载、单遍寄存器驻留 x、warp+block 归约、`rsqrtf`。保留 16B 对齐契约（v0.1 H=1024 → PER=4 → half2 路径，4B 对齐）。 |
| 2 | `kernels/rope/rope_v3_half2.cu` | `rope_v3_half2_kernel<__half>` + 标量回退（存量 `rope_v3_half2`） | `src/kernels/rope.cu` → `rope_fp16` | 同为交错对契约，`__half2` 打包/解包，FP32 旋转，4B 对齐契约 + 标量回退。裸指针 + stream。 |
| 3 | `kernels/int4gemv/int4gemv_rowtile4_hx.cu` + `int4gemv_common.h` | `int4gemv_rowtile4_hx_kernel`（存量）+ 共享 `int4gemv_scalar_kernel` + `int4gemv_unpack_byte` + `U32I4` | `src/kernels/int4_gemv.cu` → `int4_gemv` | 移植 R=4 x 半驻存向量化 kernel，**并且**移植共享标量 kernel（对齐契约回退，按上游逐位一致）。at::Tensor/stream 换为裸指针 + `cudaStream_t`。保留 nibble 契约（low=k=2b，high=k=2b+1，补码）、`scale fp16 [N,K/128]`、G=128、FP32 累加、fp16 输出。host 端 16B 对齐检查 → 标量回退。 |
| 4 | `cudalab/int4gemv_quantize.py` | `quantize_w` / `pack_q` / `unpack_w`（对称 G=128，q∈[-7,7]，scale=amax/7，fp16 存储） | `tools/convert_weights.py`（Python，离线） | 精确移植量化契约（四舍六入五成双、零组安全、nibble 打包）。仅离线 — 绝不在运行时。 |

CUDALM 原生新 kernel（无上游，在此编写，正确性优先）：

| # | Kernel | 用途 |
|---|--------|---------|
| A | `add_fp16`（残差） | `y = a + b`，fp16 输入/输出，fp32 计算 |
| B | `silu_mul_fp16` | `y = silu(g)*u`，fp32 计算，fp16 输出 |
| C | `kv_write_fp16` | 将当前 K/V 行散射写入 `position` 处的缓存 |
| D | `attention_decode_fp16` | 正确性优先的因果解码注意力（多 kernel：scores → softmax → PV），读取 KV 缓存 |
| E | `softmax_fp32`（D 的内部件） | 对 `[0..p]` 的行 softmax |

### Attention v0.1（正确性优先，解码）

给定位置 `p` 处当前 token 的 query `q`（每个 query head 一个 query 向
量），且 KV 缓存已填充 `[0..p]`：

```
scores[h, t] = dot(q_h, K[h_kv][t]) / sqrt(head_dim)      t in [0..p]
probs[h, t]  = softmax(scores[h])                          (causal: t<=p)
out_h        = sum_t probs[h,t] * V[h_kv][t]
```

实现：小型多 kernel 流水线（无 FlashAttention、无 persistent、无
online-softmax 技巧 — 清晰优先）：
1. **scores**：每 (query-head, t) 一个线程，或每 head 一个块；对
   `head_dim` 求 qk，fp32 累加。
2. **softmax**：每 query head 一个块作用于 `[0..p]`，fp32，数值稳定
   （减最大值）。
3. **PV**：每 (query-head, head_dim) 一个线程，对 `t∈[0..p]` 归约，
   fp32 累加，fp16 存储。

必须处理：`n_heads`、`n_kv_heads`（GQA 映射）、`head_dim`、`position`
以及上述扁平 KV 布局。性能明确**不是**停止条件。

---

## 7. 黄金参考策略

`tools/generate_golden.py`（PyTorch，固定种子）生成**一个** decoder
block 用例并导出所有中间 stage。C++ 黄金测试**逐 stage** 比较（不只比
最终输出）。

导出的 stage（除注明外全为 fp16），张量名与运行时相同：

```
input            x (H,)                       [带种子的 fp16]
rmsnorm1         (H,)
q,k,v            (n_heads*hd, / n_kv*hd, ...) RoPE 前
rope_q, rope_k   RoPE 后
kv_state         位置 < p 处已有的 KV（若 p>0；否则为空）
attention_output (n_heads*hd,)  = o_proj 输入
output_projection( H,)  = o_proj(x)
residual1        input + output_projection
rmsnorm2         (H,)
gate, up         (inter,)
silu_gate_mul_up (inter,)
down             (H,)
final_output     residual1 + down
```

另有元数据：
```
config（ModelConfig）、权重/量化权重（.cudalm payload 或被引用文件）、
位置 p、rope_theta、eps。
```

对 `p > 0`，生成器还导出**已有 KV 状态**（位置 `[0..p-1]`），历史**非
全零**，以覆盖 KV 缓存读路径。共生成两个黄金用例：`p=0` 与 `p>0`（如
`p=7`），后者含非平凡 KV。

比较：C++ 黄金测试加载黄金文件、回放 block，并以适合 fp16 的容差逐
stage 比较（每 stage 报告最大绝对误差与相对误差；以固定容差把关，例如
fp16 stage 最大绝对 ≤ 1e-2 且相对 ≤ 1e-2，在测试中调定并钉死）。黄金
文件格式是一个小型确定性容器（复用 CUDALM 二进制格式加 "golden" 标志，
或 sidecar JSON manifest + `.cudalm` payload）。

生成器是**离线**的（此处允许 PyTorch），绝不被 C++ 运行时引用。

---

## 8. 风险

| ID | 风险 | 缓解 |
|----|------|------------|
| R1 | CUDALab kernel 与 PyTorch 纠缠（`at::Tensor`、`TORCH_CHECK`、`getCurrentCUDAStream`、`CUDAGuard`）。照抄会带入被禁止的依赖。 | 只移植 kernel **函数体**；host 启动包装改写为裸指针 + `cudaStream_t` + `CUDALM_CHECK`。加构建/grep 守卫：`src/` 或 `include/` 出现 `torch/`、`pybind`、`at::`、`TORCH_` 即失败。 |
| R2 | 黄金数值漂移：PyTorch（CPU fp32/fp64）与 CUDA fp16 kernel 之间（归约顺序不同、fp16 舍入）。 | 逐 stage 以钉死并有文档的 fp16 容差比较（不要求逐位一致）。kernel 内 fp32 累加，仅在存储时转 fp16 — 与 PyTorch 镜像。可行处保持归约形状一致。 |
| R3 | 系统 CMake 为 **3.16.3** — 比许多 CUDA CMake 模块所预期的旧。 | 用最小、支持良好的 CMake（普通 `add_library`/`add_executable`，可用 `enable_language(CUDA)` 则用之，否则手写 `nvcc` 编译规则）。架构钉死 `sm_75`。避免仅 CMake ≥3.18 的特性。验证构建确实在 3.16.3 上运行。 |
| R4 | `compute-sanitizer` 可能不在 `PATH` 中（工具链位于 `/usr/local/cuda-11.8`）。 | 在 `scripts/` 中用绝对路径引用；若确实不可用，在最终报告中注明，而非伪造。 |
| R5 | INT4 nibble 打包 / 符号扩展是经典静默 bug 来源。 | **原样**移植 CUDALab 的 `int4gemv_unpack_byte`（它被上游三层正确性钉死）。对完整 `[-8,7]` 域加专门的 pack/unpack 往返 CPU 测试（镜像上游 `tests/test_int4gemv_cpu.py`）。 |
| R6 | KV/attention 索引错误（GQA 映射、位置边界、off-by-one）。 | 扁平、有文档的 KV 布局（§5）；host 端边界检查；黄金测试覆盖带非零历史的 `p=0` 与 `p>0`；GQA head 映射单元测试。 |
| R7 | 范围蔓延向完整模型 / 过早性能调优。 | 停止条件（brief §13）是闸门。注意力保持正确性优先；benchmark 只测量，不改变算法。 |

---

## 9. 里程碑

每个里程碑 = **正确性 → 单元测试 → 提交**。小而有序的提交；禁止
force-push；不改写已发布的证据。

| MS | 交付物 | 闸门 |
|----|-------------|------|
| M0 | 仓库引导：`git init`+`main`、CMake 骨架、`cuda_check.h`、`device_buffer.h`、`tensor.h`、`model_config.h`；首个提交 `chore: bootstrap CUDALM native runtime` | CMake 可配置；空库可编译；CPU 冒烟测试通过 |
| M1 | 权重格式规范 + `weight_format.h` + `weight_loader.cpp` + CPU 解析测试；`tools/convert_weights.py` 写出合法文件 | CPU 测试：解析、边界、对齐、dtype、int4 偏移全绿 |
| M2 | `tools/generate_golden.py` + 黄金文件 IO + stage 比较框架 | p=0 与 p>0 均产生黄金文件；框架可加载并列出 stage |
| M3 | 逐元素算子（`add_fp16`、`silu_mul_fp16`）+ **RMSNorm 移植** + 单元测试 | RMSNorm 对 CPU 黄金的单元测试绿；算子测试绿 |
| M4 | **W4A16 int4 GEMV 移植**（`int4_gemv`，hx + 标量回退）+ 单元测试 | GEMV 对 CPU 反量化参考的单元测试绿（含 pack/unpack CPU 测试） |
| M5 | **RoPE 移植** + 单元测试 | RoPE 对 CPU 黄金的单元测试绿 |
| M6 | **KV 缓存** + `kv_write_fp16` + 单元测试（p=0 与 p>0，非零历史） | KV 写/读往返绿；边界检查绿 |
| M7 | **因果解码 Attention**（scores→softmax→PV）+ 单元测试 | Attention 对 CPU 黄金的单元测试绿（p=0、p>0） |
| M8 | **完整 DecoderBlock** 接线（运行时）+ 逐 stage 黄金 PASS | p=0 与 p>0 的黄金比较 PASS |
| M9 | DecoderBlock 上 `compute-sanitizer` 干净 + CUDA 事件 **block 时延分解** | Sanitizer 摘要已存档；benchmark JSON 含逐 stage + 总时延 |
| M10 | README（架构 + 构建 + 复现）、最终停止条件清单、`git clean` | §13 停止条件全部勾选；树干净 |

M3–M8 内部的集成顺序遵循 brief 的 kernel/算子顺序：逐元素 → RMSNorm →
W4A16 → RoPE → KV → Attention → SiLU → Block。

---

## 10. 首个实现任务

**M0 — 引导：**
1. 在 `/root/code/cudalm-agent` 中 `git init` + `git branch -M main`。
2. 编写 `CMakeLists.txt`（最小、sm_75、从 `src/` 构建 `libcudalm`）。
3. 添加 `include/cudalm/cuda_check.h`（`CUDA_CHECK` + launch 检查）、
   `device_buffer.h`（RAII）、`tensor.h`（Dtype/Shape/TensorView，连
   续）、`model_config.h`（BlockConfig 结构 + 校验）。
4. 添加一个琐碎的 CPU 冒烟测试（`tests/cpu/test_bootstrap.cpp`），并经由
   CMake `enable_testing()` 做不依赖 `tests/CMakeLists` 的独立编译。
5. 提交 `chore: bootstrap CUDALM native runtime`。

随后直接进入 M1–M10，无需逐步批准；仅当某决策会改变核心契约/架构（例
如权重格式布局、KV 布局、或 W4A16 数值契约）时才停下 — 这些在本计划与
brief 中均已固定。
