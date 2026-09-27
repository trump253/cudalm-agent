# CUDALM v0.7 Phase A — Profiling Baseline（PROFILE FIRST，NO OPTIMIZATION）

> 本阶段**唯一目标**：用 Nsight Systems + Nsight Compute 对**冻结的 v0.6
> continuous-batching runtime** 建立可信 baseline、定位真实瓶颈、决定后续优化
> 对象。**未做任何 kernel/runtime 优化**（无 CUDA Graph / fusion / layout
> rewrite / 多 stream / batched prefill / Tensor Core rewrite / CUDALab port）。
> 所有结论仅基于本机（RTX 2080 Ti, CUDA 11.8）在本 checkpoint + 本 workload
> 上的实测数据；证据不足处明确标注 INCONCLUSIVE。

---

## 1. Environment / exact SHA

| 项 | 值 |
|---|---|
| repo | `trump253/cudalm-agent` |
| v0.6 frozen HEAD（merge 前） | `5e88c2d9a9563e719694a7c4d33cd3466e333a05` |
| **V06_MERGE_SHA**（main） | `1972326ec5aabf0caeb7be13f154e4d263c2b9db`（parents: `88ce5954…` + `5e88c2d9…`；merge tree == frozen v0.6 tree `8f07211a…`） |
| branch | `v0.7-profile-opt`（从 merged main 创建） |
| **profiling 时 tree SHA** | `1972326ec5aabf0caeb7be13f154e4d263c2b9db`（= merged main；见 §14 工具性修改说明） |
| GPU | NVIDIA GeForce RTX 2080 Ti（sm_75, 68 SM, GDDR6 614GB/s 标称） |
| CUDA runtime / driver | 11.8 / driver CUDA 12.8 |
| 模型 / checkpoint | Qwen/Qwen3.5-0.8B-Base（`/root/models/Qwen3.5-0.8B-Base`；preconverted `build/data/qwen35_08b_full.cudalm`） |
| 工具 | Nsight Systems 2022.4.2.50、Nsight Compute 2022.3.0.0（均为 2022 代版本，与 CUDA 11.8 匹配） |
| 流 | 单 CUDA stream；prefill 串行；decode cohort 真 batch |

## 2. Workload / methodology

**Canonical workload（与 v0.6 benchmark 完全相同，保证 before/after 可比较）**：
4 个 request（A greedy、B/C/D seeded），prompts 2/5/3/4 tokens、
max_new 3/6/5/3；动态 arrival：A,B → C（1 步后）→ D（2 步后）；共
14 prompt + 17 generated = **27 logical sequence-token forwards**。
mode A = serial（每 request 单独跑）、mode B = scheduler continuous
batched（动态 arrival，真 batched decode）。

**工具性修改（不改变 execution behavior，详见 §14）**：
benchmark 增加 `--measured-runs N`（默认仍 3）与报告字段
（mean/median/min/max、`profile_totals`）；`src/` + `include/` 未动。

**阶段划分**：
- **Baseline timing**：1 warmup + **5** measured runs，
  `benchmarks/v07_baseline_timing.txt`。
- **Nsys**：同一 canonical workload（1 warmup + 5 measured runs），
  全程 trace → `benchmarks/profiling/v07_nsys.nsys-rep`（+ 3 份 summary + CSV）。
- **Nsys 聚合**：kernel family 表 / top-10 / launch-overhead →
  `benchmarks/profiling/v07_kernel_families.txt`。
- **Ncu**：**只**对 nsys 排名出的 top 4 kernel family
  （`int4gemv` / `bf16_gemv` / `deltanet` / `rmsnorm`），每 family
  `--launch-count 12`，**缩短 workload（1 warmup + 1 measured run，明确区别于
  canonical benchmark：真实 checkpoint + 真实 batched decode + 固定可复现）**，
  sections = SpeedOfLight + Occupancy + MemoryWorkloadAnalysis + LaunchStats +
  WarpStateStats + SchedulerStats → `benchmarks/profiling/v07_ncu_*.txt`。
- **CUDALab 映射**：仅在 nsys+ncu 排名完成后检查（frozen SHA
  `cb6a6a9ef76394cc66d272c99aa8697db0a34f1e`，clone 于 `/root/code/CUDALab`，
  只读）。

复现：`bash scripts/profile_v07_baseline.sh {timing|nsys|aggregate|ncu <family>}`
（GPU 串行，一次一个任务）。

## 3. Baseline timing（canonical，1 warmup + 5 measured runs）

`benchmarks/v07_baseline_timing.txt`（SHA `1972326…`，2026-09-27）：

| mode | wall mean (s) | median | min | max | logical tok/s (mean) |
|---|---|---|---|---|---|
| A — serial | **0.133847** | 0.130278 | 0.129846 | 0.141171 | 201.72 |
| B — continuous batched | **0.118034** | 0.118703 | 0.114611 | 0.121702 | 228.75 |

固定 workload 下 per-run 计数恒定（5 runs 完全一致）：

| metric（每次 measured run） | serial | batched |
|---|---|---|
| single_forward_calls（issued） | 27 | 19 |
| successful_single_forward_calls | 27 | 19 |
| batch_forward_calls（committed） | 0 | 3 |
| model_traversal_calls（completed） | 27 | 22 |
| logical_token_forwards / batched_sequence_tokens | 27 / 0 | 27 / 8 |
| avg / max decode batch size | 0 / 0 | 2.667 / 3 |

**profile_totals（整个进程：1 warmup + 5 measured，双 mode）**：
serial 162 traversals + batched 132 traversals = **294 completed
traversals**（= nsys launch 统计的分母）。

> 无性能通过阈值、无提速声明（v0.6 correctness-first batch kernels 未调优，
> tuning 正是 v0.7）。batched 在本环境比 serial 快 ~11.8%，仅为该环境的
> 观察值。

## 4. Nsight Systems — 系统级归因

证据文件：`benchmarks/profiling/v07_nsys_{kernel,cuda_api,memops,kernexec}_summary.txt`
（+ 同名 `.csv`）与 `v07_nsys.nsys-rep`。全程（294 traversals）：
**GPU kernel 总时长 1049.7 ms**，**memops 总时长 168.7 ms**
（HtoD 96.2 ms 含一次性 60.9 ms 模型加载拷贝；DtoD 45.8 ms；memset 18.6 ms；
DtoH 8.0 ms）。

### 4.1 GPU 是否充分利用？

- 整个 profiled 进程 GPU busy（kernel+memops，单 stream 近似不重叠）≈
  **1218 ms**；两个 mode 的 6 次 measured run 的 host wall 合计
  6×(0.133847+0.118034) ≈ **1511 ms** → 稳态 GPU 利用率 ≈ **80.6%**
  （未计模型加载时段；因冻结 runtime 无 NVTX 区间，更细的 prefill/decode
  分段 gap 归属为 **INCONCLUSIVE**，见 §13）。
- 每次 completed traversal：GPU kernel 3.57 ms vs batched wall 5.37 ms /
  serial wall 4.96 ms → 每 traversal 约 1.4~1.8 ms 非纯 GPU kernel 时间
  （host 入队 + 同步 + 小 memops；精确拆分 INCONCLUSIVE）。

### 4.2 CPU/GPU gaps 与 API overhead

CUDA API Summary（host 侧 busy 时间，含阻塞等待）：

| API | calls | host 总时间 | avg/call |
|---|---|---|---|
| `cudaLaunchKernel` | **129,702** | 626.0 ms | **4.83 µs** |
| `cudaMemcpyAsync` | 25,136 | 614.0 ms | 24.4 µs（median 6.4 µs；max 61 ms = 模型加载） |
| `cudaStreamCreate` | 1 | 270.2 ms | 一次性（含 context 初始化） |
| `cudaFree` | 3,524 | 197.8 ms | 56.1 µs |
| `cudaMalloc` | 3,524 | 110.9 ms | 31.5 µs |
| `cudaMemsetAsync` | 4,932 | 26.6 ms | 5.4 µs |

- kernel launch host 成本 4.83 µs/次 × 129,702 = 626 ms（全程）≈
  **2.13 ms/traversal** —— 约为 GPU kernel 时间/traversal（3.57 ms）的 60%：
  host 平均跟得上（异步入队），但存在突发（prefill 步集中入队）。
- `cudaMalloc/cudaFree` 各 3,524 次（≈12/traversal，pool 的
  admit/finish/cancel 分配活动）+ `cudaMemsetAsync` 4,932 次：稳态 memops
  合计 ≈ **101.7 次/traversal**，GPU 时间 ≈ 92.7 ms/294 ≈ 0.31 ms/traversal。

### 4.3 H2D/D2H / sync

- HtoD：2,726 次、96.2 ms —— 其中**一次性 60.9 ms 为模型加载**；稳态 2,725 次
  共 35.3 ms（avg 12.9 µs，median 0.99 µs）。
- DtoH：仅 174 次、8.0 ms（avg 46 µs = 每次 run 的 logits 回读，设计使然）。
- DtoD：22,236 次、45.8 ms（avg 2.06 µs）+ memset 4,932 次、18.6 ms：
  稳态小拷贝/清零开销（§12 候选 P5）。
- `cudaStreamSynchronize` host 时间仅 0.29 ms（93 次调用本身很轻；真正的
  等待发生在 `cudaMemcpyAsync`/`cudaFree` 的阻塞路径上）。

### 4.4 kernel launch 是否过碎？（launch-overhead 定量，§7）

- **总 kernel launches = 129,702**；**completed traversals = 294** →
  **441.16 kernels/traversal**。
- **99.77% 的 launch（129,408/129,702）来自 avg < 100 µs 的 kernel，占
  kernel 时间的 72.6%**；只有 2 个 kernel（`bf16_gemv_vec4_row` 245.1 ms +
  `batch_bf16_gemv_vec4_row` 31.6 ms，共 288 次 launch = 0.22% of launches）
  占 kernel 时间 **26.4%**。
- 结论（基于数据，非猜测）：**mixed** —— GPU 时间被少数 heavy kernel
  （LM head GEMV）显著主导（26.4% @ 0.22% launches），但其余 ~73.6% 分散在
  12.9 万次短 launch 上（avg 7 µs，大量 waves<1）；host launch 成本
  2.13 ms/traversal 与 GPU 3.57 ms/traversal 同量级 → **既不是纯
  heavy-kernel dominated，也不是纯 launch-fragmentation dominated**。

## 5. Kernel-family 排名

`benchmarks/profiling/v07_kernel_families.txt`（由
`scripts/aggregate_kernel_families.py` 从 nsys CSV 聚合；单位：GPU kernel 时间
占比以 1049.7 ms 为 100%）：

| family | total GPU time | % kernel time | calls | avg latency |
|---|---|---|---|---|
| **w4a16/int4 projection**（`int4gemv_rowtile4_bf16` 等） | 449.1 ms | **42.79%** | 54,684 | 8.21 µs |
| **bf16 lm head**（`bf16_gemv_vec4_row` / `batch_bf16_gemv_vec4_row`） | 287.2 ms | **27.36%** | 294 | 977 µs |
| **deltanet**（conv/gbeta/delta/gated_rmsnorm） | 167.2 ms | **15.92%** | 21,492 | 7.78 µs |
| elementwise/residual/swiglu（add/silu_mul/gate_mul/split_q_gate） | 60.4 ms | 5.75% | 24,696 | 2.45 µs |
| rmsnorm | 47.6 ms | 4.54% | 17,934 | 2.65 µs |
| paged attention（scores/softmax/pv/kv_write） | 30.1 ms | 2.87% | 7,056 | 4.26 µs |
| rope | 8.0 ms | 0.76% | 3,528 | 2.28 µs |
| embedding | 0.046 ms | ~0% | 18 | 2.58 µs |

**Top 10 kernels by total GPU time**：

| kernel | family | total | % | calls | avg |
|---|---|---|---|---|---|
| `bf16_gemv_vec4_row_kernel` | lm head | 245.1 ms | 23.35% | 276 | 888 µs |
| `int4gemv_rowtile4_bf16_kernel`（grid 896） | int4 | 128.9 ms | 12.28% | 13,248 | 9.7 µs |
| `deltanet_delta_kernel` | deltanet | 116.8 ms | **11.13%** | 4,968 | 23.5 µs |
| `int4gemv_rowtile4_bf16_kernel`（grid 256） | int4 | 98.2 ms | 9.35% | 13,248 | 7.4 µs |
| `int4gemv_rowtile4_bf16_kernel`（grid 1536） | int4 | 67.0 ms | 6.39% | 4,968 | 13.5 µs |
| `int4gemv_rowtile4_bf16_kernel`（grid 4） | int4 | 46.3 ms | 4.41% | 9,936 | 4.7 µs |
| `qwen35_rmsnorm_zc_kernel<4>` | rmsnorm | 36.1 ms | 3.44% | 13,524 | 2.67 µs |
| `batch_bf16_gemv_vec4_row_kernel` | lm head | 31.6 ms | 3.01% | 12 | 2632 µs |
| `int4gemv_rowtile4_bf16_kernel`（grid 512） | int4 | 31.1 ms | 2.96% | 4,968 | 6.3 µs |
| `qwen35_add_kernel` | elementwise | 30.5 ms | 2.91% | 13,248 | 2.3 µs |

**Top 10 by invocation count**：`qwen35_rmsnorm_zc<4>` 13,524；
`int4gemv_rowtile4_bf16`（896/256 grid）各 13,248；`qwen35_add` 13,248；
`int4gemv`（4 grid）9,936；`qwen35_silu_mul` 6,624；`deltanet_delta` 4,968；
`int4gemv`（1536/512 grid）4,968；`deltanet_gated_rmsnorm` 4,968。

排序以 **end-to-end total impact**（total GPU time）为准，不看单次 latency。

## 6. Nsight Compute — Top 4 hotspot（每 family 12 个 representative launch）

> NCU 为**缩短 workload**（1 warmup + 1 measured run，真实 checkpoint + 真实
> batched decode，固定可复现，明确区别于 canonical benchmark）。12 个 launch
> 落在 warmup/首批 measured 的 **serial-mode（B=1）** 变体上（serial 先跑）；
> batch-mode 变体（`batch_*`）未单独 NCU（其 nsys 级数据见 §4/§5）。
> 数据：`benchmarks/profiling/v07_ncu_{w4a16_int4,bf16_lmhead,deltanet,rmsnorm}.txt`。

### 6.1 `bf16_gemv_vec4_row`（LM head，27.4%）→ **memory/DRAM-bound（roofline）**

12/12 launch 形态一致：grid=248,320（vocab/row，1 block/行）、23 regs、
Duration **979 µs**、Memory Throughput **520 GB/s（≈ 2080 Ti 峰值的 85~94%）**、
**DRAM 80%**、SM 42.3%、L2 hit 1.6%、occupancy 72.6%、waves 913。
= 单次全量流式 503 MB BF16 权重矩阵（1024×248,320），**已贴在 DRAM 带宽
roofline 上** → kernel 微优化无空间（结论有数据：带宽已近峰值）。
（batch 变体 `batch_bf16_gemv`：2.63 ms/call，B≤3，nsys 级 3.0%。）

### 6.2 `int4gemv_rowtile4_bf16`（W4A16 投影，42.8%）→ **mixed**

同一 kernel、不同 grid（对应不同投影形状，hidden=1024 系小 K）的 12 个
representative launch（56 regs）：

| grid | dur | SM% | Mem GB/s | DRAM% | occ% | waves |
|---|---|---|---|---|---|---|
| 1536 | 17.4 µs | 35.0 | 187 | 30.0 | 82.5 | 2.82 |
| 896 | 12.3 µs | 28.4 | 153 | 24.4 | 76.2 | 1.65 |
| 512 | 7.7 µs | 26.6 | 140 | 22.7 | 79.3 | 0.94 |
| 256 | 10.1 µs | 33.4 | 187 | 31.3 | **41.4** | **0.47** |
| 256（另一形状） | 8.0 µs | 22.7 | 136 | 22.6 | **42.4** | **0.47** |
| 4 | 5.8 µs | **0.28** | 1.8 | **0.30** | **11.3** | **0.01** |

- 中大 grid（512–1536）：DRAM 22~31%、SM 26~35% → 偏 **memory-bound** 但
  远未贴 roofline（136~189 GB/s ≈ 峰值 22~31%）；occupancy 76~82%。
- grid=256：occupancy 41~42%、waves 0.47 → **occupancy-limited / latency**。
- grid=4：68 SM 只用 4 block、SM 0.28%、5.8 µs 近乎纯 latency →
  **launch/latency-dominated 的 degenerate shape**（该投影 N 极小，
  N 维并行度只有 4）。
- 判定：**mixed**（family 内部按形状分裂；无单一瓶颈标签）。

### 6.3 `deltanet_*`（15.9%；其中 `deltanet_delta` 单 kernel 11.1%）→
**latency/dependency-bound + occupancy-limited**

12 launch = 4 个 serial-mode kernel × 3：

| kernel | grid | dur | SM% | DRAM% | occ% | regs | waves |
|---|---|---|---|---|---|---|---|
| `deltanet_delta`（状态更新，recurrent） | **16** | **30.5 µs** | **2.3** | 5.3 | **11.9** | 64 | **0.03** |
| `deltanet_gbeta` | **1** | 2.6 µs | 0.01 | 0.02 | 3.1 | 16 | ~0 |
| `deltanet_conv` | 24 | 2.7 µs | 2.3 | 5.9 | 24.7 | 20 | 0.09 |
| `deltanet_gated_rmsnorm` | 16 | 3.0 µs | 0.8 | 0.5 | 12.4 | 18 | 0.03 |

- `deltanet_delta`：30.5 µs × 4,968 次 = 11.1% 总 kernel 时间；grid=16 block
  对 68 SM、waves 0.03、occupancy 11.9%、64 regs → recurrent state update
  的**串行依赖 + 极小并行度**（SM 2.3%、DRAM 5.3% 均无压力，纯
  latency/dependency-bound）。
- `deltanet_gbeta` grid=1：单 block 算 g/beta，≈ launch 延迟。
- 判定：**latency/dependency-bound（叠加 occupancy-limited）**。

### 6.4 `qwen35_rmsnorm_zc` / `deltanet_gated_rmsnorm`（rmsnorm 4.5%）→
**launch-overhead dominated**

12 launch：`qwen35_rmsnorm_zc<4>`（grid=1，22 regs，2.9 µs，SM 0.09%，
DRAM 0.23%）×7、`qwen35_rmsnorm_zc<1>`（grid=8/2，~2.85 µs，SM 0.09~0.6%）
×2、`deltanet_gated_rmsnorm`（grid=16，2.94 µs，SM 0.83%）×3。
B=1 时每行 RMSNorm（hidden=1024）只投 1~8 个 block → waves≈0、SM<1%：
每个 ~2.9 µs ≈ launch + 微小计算。17,934 次 × 2.65 µs = 47.6 ms（4.5%）。
判定：**launch-overhead dominated**（kernexecsum 佐证：host API 时间
~4.5~5 µs/kernel 与 GPU 执行 2.9 µs 同量级）。

## 7. Launch-overhead 分析（nsys 定量结论）

| 量 | 值 |
|---|---|
| total kernel launches（全程 294 traversals） | **129,702** |
| completed model traversals | 294 |
| **kernels / traversal** | **441.16** |
| total GPU kernel time | 1049.7 ms |
| `cudaLaunchKernel` host 总时间 / avg | 626.0 ms / 4.83 µs |
| host launch 时间 / traversal vs GPU kernel / traversal | 2.13 ms vs 3.57 ms（≈60%） |
| short-duration kernels（avg < 100 µs） | 129,408/129,702 launches（**99.77%**）；76/79 kernel 种；占 kernel 时间 **72.6%** |
| heavy kernels（avg ≥ 100 µs） | 2 种（LM head ×2 变体）、288 launches（0.22%）、占 kernel 时间 **26.4%** |

**最终判定：mixed。** GPU 时间一半以上由 W4A16 GEMV（42.8%，54,684 次
短 launch）+ LM head（27.4%，288 次 heavy launch）两个 family 主导；
launch fragmentation 是**真实的次级因素**（441 launch/traversal、host
launch 与 GPU 同量级、99.77% 短 kernel、大量 waves<1），但不是唯一因素。

## 8. Bottleneck classification（每个 NCU hotspot，附数据依据）

| hotspot | 分类 | 数据依据 |
|---|---|---|
| `bf16_gemv_vec4_row`（LM head） | **memory-bound（DRAM roofline）** | 520 GB/s ≈ 峰值 85~94%；DRAM 80%；SM 42.3%；L2 hit 1.6%（全流式）；979 µs ≈ 503 MB/520 GB/s |
| `int4gemv_rowtile4_bf16`（W4A16） | **mixed**（按形状：大 grid 偏 memory-bound、中 grid occupancy-limited、grid=4 latency-dominated） | DRAM 22~31% + SM 26~35%（512~1536）；occ 41% / waves 0.47（256）；SM 0.28% / occ 11.3% / waves 0.01（4） |
| `deltanet_delta` + `deltanet_gbeta`/`conv`/`gated_rmsnorm` | **latency/dependency-bound + occupancy-limited**（`gbeta` 单项可标 launch-overhead dominated） | delta：30.5 µs @ SM 2.3%、DRAM 5.3%、occ 11.9%、grid 16、waves 0.03、64 regs；gbeta：grid=1、2.6 µs |
| `qwen35_rmsnorm_zc` / `deltanet_gated_rmsnorm` | **launch-overhead dominated** | grid 1~16、waves≈0、SM 0.09~0.83%、每个 ~2.9 µs ≈ host API 时间（~4.5 µs）同量级 |
| elementwise（add/silu_mul/…，nsys 级） | **launch-overhead dominated（倾向）** | 24,696 calls、avg 2.45 µs、SM 个位数（kernexec 佐证）；未 NCU → 严格标注为倾向 |
| paged attention / rope / embedding（nsys 级） | **INCONCLUSIVE（未 NCU）** | 占比 ≤ 2.9%，未进 top-4；仅有 nsys 级 total 数据 |

## 9. CUDALab 映射（**仅在 profiling 完成后**检查；frozen SHA
`cb6a6a9ef76394cc66d272c99aa8697db0a34f1e`，只读 clone）

> CUDALab = 自主 CUDA kernel 优化实验室（operator 集：RMSNorm / Softmax /
> RoPE / GEMV(BF16) / QGEMV(INT8) / INT4GEMV(W4A16 G=128, fp16 scale)）。
> **本 Phase 未 port 任何优化**（仅建立映射）。

| measured CUDALM hotspot | CUDALab 对应 | 可借鉴 technique / 状态 |
|---|---|---|
| **W4A16/INT4 GEMV（42.8%）**：`int4gemv_rowtile4_bf16_kernel` | `kernels/int4gemv/`（v0.7 线）：baseline → vec16_row(KEEP 2.0×) → rowtile4(KEEP 1.19×) → rowtile8(NEUTRAL) → **rowtile4_hx（incumbent，2.59× vs baseline）** | **核心 technique 已 port**：CUDALM 的 rowtile4_bf16 = R=4 行块 + x 寄存器驻留（bf162）+ 16B 向量化 W load + single-group scale lemma（g=v>>2），与 CUDALab hx incumbent 同源。**未覆盖**：(a) CUDALM 小形状（hidden=1024，K≤3072，N 可到 3072 但含 N=极小的投影 → grid=4/256 degenerate case）的**形状特定调优**（CUDALab 主形状 4096²、K≥1024）；(b) **batch 变体**（`batch_int4gemv_rowtile4_bf16`）——CUDALab 全部 B=1，B>1 需**重新设计 batch dimension**；(c) NCU 结论「occupancy 是比 MLP 更强的 DRAM 杠杆」可用于 CUDALM 小 shape 的调优方向 |
| **BF16 LM head（27.4%）**：`bf16_gemv_vec4_row_kernel` / `batch_bf16_gemv` | `kernels/gemv/`（v0.5 线）：**gemv_vec4_row（incumbent 1.54×）**；warp_vec4_b256/b512（KEEP 但更慢）；splitk4（REJECT） | **已是 CUDALab 最优变体**；CUDALab 无更快的 BF16 GEMV → 无直接可 port。实测已贴 DRAM roofline（520 GB/s）→ 唯一杠杆是**减少权重字节数（LM head 量化，语义变更，需 reviewer 决策）** |
| **DeltaNet（15.9%）**：`deltanet_delta/gbeta/conv/gated_rmsnorm` | **no direct CUDALab analogue**（operator 集无线性注意力/recurrent state update） | 无对应；需原创设计（状态更新的并行化/分块/融合） |
| **Paged attention（2.9%）**：`qwen35_paged_*` / `batch_paged_*` | **no direct CUDALab analogue**（CUDALab softmax 为非 paged 全行；paged-KV 语义/间接寻址无对应） | 无对应 |
| **RMSNorm（4.5%）**：`qwen35_rmsnorm_zc<1/4>` / `rmsnorm_v4_half` | `kernels/rmsnorm/`（v4：向量化 + 寄存器驻留单遍，已验证） | **v4 已 port**（`rmsnorm_v4_half_kernel`）；CUDALM 的 zero-centered 变体（Qwen3.5 专用）与 grid=1（B=1、hidden=1024）的**发射几何**是 CUDALM 特有的，CUDALab 无对应小 hidden 形状 |
| **RoPE（0.76%）**：`rope_v3_half2` / `qwen35_partial_rope` | `kernels/rope/`（rope_v3_half2 已验证） | **已 port** |
| **elementwise（5.75%）**：add/silu_mul/gate_mul/split_q_gate | **no direct CUDALab analogue**（CUDALM 特有 elementwise 链） | 无对应；融合（fusion）亦无 CUDALab 对应物 |
| **scheduler / continuous batching / launch fragmentation** | **no direct CUDALab analogue**（CUDALab 是 B=1 单 op 优化闭环，无调度器/多请求语义） | 无对应 |

## 10. Ranked Phase B optimization backlog

排序依据 = measured end-to-end impact × 证据置信度 ÷ 实现风险（全部来自
§4~§8 实测，不凭直觉）。

**P1 — W4A16 INT4 GEMV 形状特定调优（occupancy/向量化/N 并行度）**
- Hotspot：`int4gemv_rowtile4_bf16_kernel`（42.8% 总 kernel 时间；54,684 launches）
- Profiler evidence：grid=4 → SM 0.28%/occ 11.3%/waves 0.01（5.8 µs 近乎纯
  latency）；grid=256 → occ 41%/waves 0.47；grid=512~1536 → DRAM 22~31%、
  136~189 GB/s（峰值 22~31%，远未贴 roofline）
- Bottleneck：mixed（degenerate 小 N 形状 latency/occupancy-limited；中形状
  memory-bound 有带宽余量）
- Optimization hypothesis：(a) 极小 N 投影换映射（K 维并行/多块 + 确定性
  combine，或合并相邻小投影）；(b) 中形状提高 DRAM 利用率（CUDALab 结论：
  occupancy 杠杆 > MLP；更多 N 并行 + 保持 k-order 的向量化 ILP）
- Expected e2e impact：family 42.8% 中若 30~50% 可压缩 → 13~21% kernel
  时间 ≈ **6~10% wall**（按 GPU 3.57 ms/traversal、wall 5.37 ms/traversal）
- Risk/complexity：中（kernel 局部、无调度/状态语义变更；**首要风险 = 精确
  parity 合同**：任何改变 fp32 累加顺序的改动会破坏 EXACT logits 门 →
  优先选**保持 k-order 累加顺序**的 occupancy/向量化子集；顺序保持子集之外
  需 reviewer 裁决容忍门）
- CUDALab reference：`kernels/int4gemv/int4gemv_rowtile4_hx.cu`（incumbent）、
  INT4GEMV-0001..0004 实验链 + paired benchmark harness（可作验证工具）

**P2 — DeltaNet delta-rule 状态更新 kernel（+ gbeta/conv/gated_rmsnorm 融合）**
- Hotspot：`deltanet_delta_kernel` 单 kernel 11.1% + deltanet family 15.9%
- Profiler evidence：30.5 µs @ SM 2.3%、DRAM 5.3%、occ 11.9%、grid=16（68 SM
  用 16）、waves 0.03、64 regs；`gbeta` grid=1
- Bottleneck：latency/dependency-bound + occupancy-limited
- Optimization hypothesis：状态更新（recurrent over state dim）重块化/分维
  并行（确定性 combine）+ 状态行向量化；gbeta/conv/gated_rmsnorm 融入
  delta 前序（减 3 次 launch/层）
- Expected e2e impact：delta 3× → 省 0.265 ms/traversal ≈ **~5% wall**；
  融合再减 ~3.8% kernel 时间的一部分
- Risk/complexity：中高（recurrent 语义 + 精确 parity 风险同上；64 regs 需
  控制）
- CUDALab reference：**no direct analogue**（原创设计）

**P3 — elementwise/norm/rope 融合（降 launch 数，441 → 目标 ~300/traversal）**
- Hotspot：rmsnorm 4.5% + elementwise 5.75% + rope 0.76%（共 ~11% kernel
  时间，全部 2.3~2.7 µs 的 launch 延迟主导 kernel）
- Profiler evidence：grid≤8、waves≈0、SM<1%；441 kernels/traversal；host
  launch 2.13 ms/traversal（GPU 的 60%）
- Bottleneck：launch-overhead dominated（小 kernel 簇）
- Optimization hypothesis：层内 elementwise 链融合（add+silu_mul+
  split_q_gate；rmsnorm 并入 producer），不动 CUDA Graph
- Expected e2e impact：8~12% kernel 时间 + host API 时间下降 ≈ **3~6% wall**
- Risk/complexity：中（融合代码面大；parity 风险同上；无 Graph 则收益上限
  有限）
- CUDALab reference：rmsnorm v4 / gemv 已 port（无增量）；fusion 本身
  **no direct CUDALab analogue**

**P4 — LM head 量化（W4A16，语义变更 → reviewer 决策项，非默认执行）**
- Hotspot：`bf16_*_gemv_vec4_row` 27.4%（288 launches，26.4% kernel 时间）
- Profiler evidence：520 GB/s ≈ DRAM roofline（80% DRAM、L2 hit 1.6%）
  → kernel 微优化无空间（memory-bound 已触顶）
- Bottleneck：memory-bound（roofline）
- Optimization hypothesis：LM head 权重 BF16→INT4（0.5B 字节）→ IO 减半
  以上 → 979 µs → ~400 µs 量级
- Expected e2e impact：最高 ~**15~20% wall**（若量化可接受）
- Risk/complexity：**高**（模型格式/语义变更；输出 logits 变化 → 精确 parity
  合同必须重构为量化感知；CUDALM 尚无 LM head 量化路径）
- CUDALab reference：INT4GEMV incumbent（rowtile4_hx）可作 head 的 kernel
  底座（若 reviewer 批准量化方向）

**P5 — 稳态 memops 批处理（DtoD 45.8 ms + HtoD 稳态 35.3 ms + memset 18.6 ms，
≈101.7 memops/traversal）**
- Bottleneck：小拷贝/清零碎片（nsys memops summary）
- Optimization hypothesis：per-step 元数据单次 HtoD（device 侧 seed/常量）、
  免 DtoD 中转
- Expected e2e impact：~**1~2% wall**
- Risk/complexity：低
- CUDALab reference：no direct analogue

**推荐 Phase B 首先优化：P1（W4A16 INT4 GEMV 形状特定调优）**。理由：
(1) 最大 measured family（42.8% GPU kernel 时间；第二大的 LM head 27.4%
已贴 DRAM roofline，kernel 层面无可挤空间）；(2) 缺口被 profiler 量化到
每个 grid 形状（waves/occupancy/DRAM 均有数据）；(3) 有直接 CUDALab
reference（incumbent 已验证 + paired harness 可复用为验证工具）；(4)
改动面最小（单 kernel family，无调度/状态/格式语义）；(5) 存在**保持
fp32 累加顺序**的安全子集（N 维并行几何、k-order 向量化、block/寄存器
调优），可在不破坏 v0.6 EXACT parity 门的前提下先落地。

## 11. Limitations

- **workload 小**（27 logical tokens，prompts 2~5 tokens）：prefill 形状
  （N>1 的 token 维并行）在 nsys 全程中占比极低；**prefill-heavy profile
  未做**（Phase B 应补）。kernel family 的**相对**占比在本 model/regime
  内可信，绝对 wall 收益随 token 数缩放。
- **NCU 为缩短 workload**（1 warmup + 1 measured、每 family 12 launches，
  落在 serial-mode B=1 变体）；`batch_*`（B=2/3）变体未 NCU —— batch 形状
  的热点行为推断自 nsys 级数据（`batch_deltanet_delta` 30.6→2.6 µs 等
  kernexec 行），**标注为推断**。
- 冻结 runtime **无 NVTX 区间** → prefill/decode 分段的 GPU gap 归属只能用
  比例估计（§4.1），更细粒度 INCONCLUSIVE；加 NVTX 属 tooling 修改，
  留待 Phase B 决策。
- nsys 2022.4 / ncu 2022.3 为 CUDA 11.8 匹配版本，无 2023+ 的新 metrics
  （如 `gpu__time_active` 直接指标）；GPU 利用率由 kernel+memops busy 时间
  与 wall 的比值推算（单 stream 近似）。
- 单 GPU、单环境（RTX 2080 Ti sm_75）；数字不是跨环境结论。
- CUDALab 映射基于其 docs/kernels 静态阅读（frozen SHA 只读 clone），未运行
  CUDALab 任何 benchmark。

## 12. Reproduction commands

```bash
# （一次性）build
cmake --build build -j8
cd build && ctest --output-on-failure        # 60/60
bash scripts/check_no_torch.sh               # CLEAN

# baseline timing（1 warmup + 5 measured）→ benchmarks/v07_baseline_timing.txt
bash scripts/profile_v07_baseline.sh timing

# nsys（canonical workload）→ benchmarks/profiling/v07_nsys.nsys-rep + summaries
bash scripts/profile_v07_baseline.sh nsys
bash scripts/profile_v07_baseline.sh aggregate   # → v07_kernel_families.txt

# ncu（top 4 families，缩短 workload，每 family 12 launches）
bash scripts/profile_v07_baseline.sh ncu w4a16_int4
bash scripts/profile_v07_baseline.sh ncu bf16_lmhead
bash scripts/profile_v07_baseline.sh ncu deltanet
bash scripts/profile_v07_baseline.sh ncu rmsnorm

# CUDALab（只读，profiling 完成后）
git clone git@github.com:trump253/CUDALab.git /root/code/CUDALab
git -C /root/code/CUDALab checkout cb6a6a9ef76394cc66d272c99aa8697db0a34f1e
```

所有 evidence 文件（`benchmarks/v07_baseline_timing.txt`、
`benchmarks/profiling/*`）头部均记录：exact git SHA、GPU、CUDA、checkpoint、
workload、warmup/measured runs、date；绑定 `V07A_EVIDENCE_SHA`（commit 后
记录）。

## 13. 未决问题（供 Phase B / reviewer）

1. **精确 parity 合同 vs 优化**：v0.6 的门是 EXACT logits parity（vs 独立
   参考）。P1~P3 中任何改变浮点累加顺序的 kernel 改动都会破坏该门。
   需要 reviewer 明确：(a) 强制「保持累加顺序」的优化子集；或 (b) 批准
   在受控范围内切换为 tolerance 门（并重新生成参考/证据基线）。
2. **LM head 量化（P4）**是否进入 v0.7 范围（语义/格式变更）。
3. **prefill-heavy profile** 是否作为 Phase B 前置证据（当前 baseline 的
   decode 偏重）。
4. **NVTX 插桩**（tooling 修改，不改变 execution behavior）是否允许进
   Phase B，以支持分段 gap 归因。

## 14. 工具性修改声明（本 Phase 唯一的 tree 变更，非优化）

runtime/kernel functional tree（`src/` + `include/`）**未做任何修改**，与
merged v0.6 baseline（tree `8f07211a…`）完全一致。唯一 tree 变更为
**profiling 工具**：

- `benchmarks/bench_qwen35_continuous_batching.cpp`：
  (a) 新增 `--measured-runs N` CLI（默认 3，v0.6 行为不变）；(b) 报告新增
  mean/median/min/max、`successful_single_forward_calls`、
  `model_traversal_calls`（serial mode）与 `profile_totals` 段（全程
  completed traversals，作为 nsys launch 统计分母）。**每个 timed run 的
  execution（workload、kernel、同步顺序、计时方法）与 v0.6 逐字节相同**，
  仅重复次数与报告字段变化。
- 新增：`scripts/profile_v07_baseline.sh`、`scripts/aggregate_kernel_families.py`
  （profiling 编排 + nsys CSV 聚合；纯 host 工具，不触 GPU 执行路径）。
- 新增 evidence：`benchmarks/v07_baseline_timing.txt`、
  `benchmarks/profiling/*`（nsys rep/summaries/csv、ncu 输出、family 表）。

## 15. STOP CONDITION 自检

- v0.6 已 merge main（merge tree == frozen v0.6 tree）✓
- v0.7-profile-opt 从 merged main 创建并 push ✓
- baseline correctness：60/60、0 failed、0 skipped；no-torch CLEAN ✓
- canonical timing baseline 记录（5 measured runs，含 SHA/GPU/CUDA/
  checkpoint/workload/warmup）✓
- Nsys：kernel / CUDA API / memops summary + kernel-family ranking +
  launch-overhead 结论（mixed，附数据）✓
- NCU：top 4 hotspot 测量 + evidence-backed 分类（memory-bound roofline /
  mixed / latency-dependency+occupancy / launch-overhead）✓
- CUDALab：仅在 profiling 后检查（frozen SHA 只读）；映射记录；**未 port
  任何优化** ✓
- ranked Phase B backlog（P1~P5）+ 明确推荐首目标（P1 + 理由）✓
- `V07A_EVIDENCE_SHA` 记录于 commit（见 git log）；branch 已 push ✓
- **未开始 Phase B** ✓
