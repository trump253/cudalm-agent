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
| **V07A_FUNCTIONAL_SHA**（profiling 工具 commit，tree 干净） | `0a952879da6e6cbb26a4aca181e184f26d2a0eef` |
| **V07A_EVIDENCE_SHA** | **`0a952879da6e6cbb26a4aca181e184f26d2a0eef`**（= V07A_FUNCTIONAL_SHA；**全部**最终证据在该 exact SHA 的干净 tracked tree 上生成，每份证据头部记录该 SHA + tree 状态） |
| 最终 HEAD | V07A_EVIDENCE_SHA 之上的 1 个 docs/evidence-only commit（diff 仅限 `docs/` + `benchmarks/profiling/` + benchmark evidence 文本） |
| GPU | NVIDIA GeForce RTX 2080 Ti（sm_75, 68 SM, GDDR6 614GB/s 标称） |
| CUDA runtime / driver | 11.8 / driver CUDA 12.8（driver_version 570.172.08） |
| 模型 / checkpoint | Qwen/Qwen3.5-0.8B-Base（`/root/models/Qwen3.5-0.8B-Base`；preconverted `build/data/qwen35_08b_full.cudalm`） |
| 工具 | Nsight Systems 2022.4.2.50、Nsight Compute 2022.3.0.0（均为 2022 代版本，与 CUDA 11.8 匹配） |
| 流 | 单 CUDA stream；prefill 串行；decode cohort 真 batch |

**exact-SHA 证据纪律（reviewer 要求）**：第一轮证据（header 记录
`1972326…`，profiling 工具当时仅存在于 dirty working tree / `8d002e6…`）
**判定无效并已全部从 tree 删除**（dirty-tree 生成，非 exact-SHA-bound）。
本轮流程：工具 commit → tracked tree 干净（`git status --porcelain` 无
tracked 漂移）→ 在该 exact SHA 上 `cmake --build build -j8`、
`ctest --output-on-failure`（**60/60，0 failed，0 skipped**）、
`bash scripts/check_no_torch.sh`（**CLEAN**）→ 在该 exact SHA 上重跑全部
timing/nsys/NCU/聚合 → docs/evidence-only commit。profiling 脚本对
**tracked tree 漂移 fail-loud**（非干净即拒绝生成证据，exit 3）。

## 2. Workload / methodology

**Canonical workload（与 v0.6 benchmark 完全相同，保证 before/after 可比较）**：
4 个 request（A greedy、B/C/D seeded），prompts 2/5/3/4 tokens、
max_new 3/6/5/3；动态 arrival：A,B → C（1 步后）→ D（2 步后）；共
14 prompt + 17 generated = **27 logical sequence-token forwards**。
mode A = serial（每 request 单独跑）、mode B = scheduler continuous
batched（动态 arrival，真 batched decode）。

**工具性修改（不改变 execution behavior，详见 §14）**：
benchmark 增加 `--measured-runs N`（默认仍 3）与报告字段
（mean/median/min/max、`profile_totals`）；**新增 `--mode both|serial|batched`
选择器（默认 both = v0.6 行为逐字节不变；`--mode batched` 只跑 scheduler
continuous-batched workload —— 同 checkpoint / 同 requests / 同 dynamic
arrivals / 同 warmup+measured 方法学 / 单 stream / 串行 prefill / 真
batched decode；只改「哪些 mode 执行」，不改任何 kernel/runtime 行为）**；
`src/` + `include/` 未动。

**阶段划分**：
- **Baseline timing**（`--mode both`）：1 warmup + **5** measured runs，
  `benchmarks/v07_baseline_timing.txt`。
- **Nsys（mixed = comparison-harness profile）**：同一 canonical workload
  （1 warmup + 5 measured，`--mode both`）全程 trace →
  `benchmarks/profiling/v07_nsys.nsys-rep`（+ 4 份 summary + CSV）。
  **这是双 mode 混合进程的对照 profile，不是 serving 路径归因**（保留作
  comparison；Phase B 排名以 batched-only profile 为准，§4.5/§5.2）。
- **Nsys（batched-only = 连续批处理 serving profile，Phase B 排名主依据）**：
  `--mode batched`，1 warmup + 5 measured → `v07_batched_nsys.nsys-rep`
  （+ 4 份 summary + CSV）。
- **Nsys 聚合**：kernel family 表 / top-10 / launch-overhead →
  `v07_kernel_families.txt`（mixed）与 `v07_batched_kernel_families.txt`
  （batched-only；分母 = 各自进程 completed traversals：294 / 132）。
- **Ncu（batch-aware，8 targets，全部 `--mode batched`）**：对
  `int4gemv` / `bf16_gemv` / `deltanet` / `rmsnorm` 的 **B=1 与 batch_* 变体**
  分别 profile，每 target `--launch-count 12`（`--launch-skip` 区分 cohort），
  **缩短 workload（1 warmup + 1 measured run，明确区别于 canonical
  benchmark：真实 checkpoint + 真实 batched decode + 固定可复现）**，
  sections = SpeedOfLight + Occupancy + MemoryWorkloadAnalysis + LaunchStats +
  WarpStateStats + SchedulerStats → `benchmarks/profiling/v07_ncu_<target>.txt`。
  targets：`int4_b1` / `int4_batch`（B=2 cohort）/ `int4_batch_late`（B=3
  cohort）/ `lmhead_b1` / `lmhead_batch` / `deltanet`（B=1）/
  `deltanet_batch`（B=2）/ `rmsnorm`。NCU 2022.3 `-k regex` 匹配**裸 kernel
  函数名**（无命名空间），故用 `^` 锚定区分 B=1 与 `batch_*`（batch 名以
  `batch_` 开头，永不交叉匹配）。
- **CUDALab 映射**：仅在 nsys+ncu 排名完成后检查（frozen SHA
  `cb6a6a9ef76394cc66d272c99aa8697db0a34f1e`，clone 于 `/root/code/CUDALab`，
  只读）。

复现：`bash scripts/profile_v07_baseline.sh
{timing|nsys|nsys_batched|aggregate [mixed|batched]|aggregate_batched|ncu <target>}`
（GPU 串行，一次一个任务；tracked tree 非干净则 fail-loud）。

## 3. Baseline timing（canonical，1 warmup + 5 measured runs）

`benchmarks/v07_baseline_timing.txt`（**V07A_EVIDENCE_SHA `0a952879…`，干净
tracked tree**，2026-09-27）：

| mode | wall mean (s) | median | min | max | logical tok/s (mean) |
|---|---|---|---|---|---|
| A — serial | **0.133140** | 0.132032 | 0.131350 | 0.136306 | 202.80 |
| B — continuous batched | **0.116368** | 0.115541 | 0.115458 | 0.117940 | 231.99 |

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
> tuning 正是 v0.7）。batched 在本环境比 serial 快 ~12.6%（wall mean），
> 仅为该环境的观察值。

## 4. Nsight Systems — 系统级归因

> **profile 定位（reviewer 要求）**：`v07_nsys.*` 是 **comparison-harness /
> mixed-mode profile**（`--mode both`：mode A serial + mode B batched 混在
> 同一进程）。它用于对照与整体 launch 统计，**不是** serving 路径归因。
> **Phase B 优化排名的主依据是 batched-only serving profile（§4.5/§5.2，
> `v07_batched_nsys.*`）**。两份 profile 均在 V07A_EVIDENCE_SHA 上、同一
> canonical workload（1 warmup + 5 measured）下生成。

### 4.0 Mixed profile（comparison-harness，294 traversals）

证据文件：`benchmarks/profiling/v07_nsys_{kernel,cuda_api,memops,kernexec}_summary.txt`
（+ 同名 `.csv`）与 `v07_nsys.nsys-rep`。全程（294 traversals）：
**GPU kernel 总时长 1075.0 ms**，**memops 总时长 202.0 ms**
（HtoD 128.6 ms 含一次性 63.9 ms 模型加载拷贝；DtoD 46.8 ms；memset 18.6 ms；
DtoH 8.0 ms）。

### 4.1 GPU 是否充分利用？（mixed）

- 整个 profiled 进程 GPU busy（kernel+memops，单 stream 近似不重叠）≈
  **1277 ms**（6 个 workload 执行 = 1 warmup + 5 measured，294 traversals）；
  measured 区 5 runs 的 host wall 合计 5×(0.133140+0.116368) ≈ **1247.5 ms**，
  同区 GPU busy ≈ 1277×245/294 ≈ 1064.5 ms → 稳态 GPU 利用率 ≈ **85.3%**
  （未计模型加载时段；因冻结 runtime 无 NVTX 区间，更细的 prefill/decode
  分段 gap 归属为 **INCONCLUSIVE**，见 §13）。
- 每次 completed traversal：GPU kernel 3.66 ms vs batched wall 5.29 ms /
  serial wall 4.93 ms → 每 traversal 约 1.3~1.6 ms 非纯 GPU kernel 时间
  （host 入队 + 同步 + 小 memops；精确拆分 INCONCLUSIVE）。

### 4.2 CPU/GPU gaps 与 API overhead（mixed）

CUDA API Summary（host 侧 busy 时间，含阻塞等待）：

| API | calls | host 总时间 | avg/call |
|---|---|---|---|
| `cudaLaunchKernel` | **129,702** | 636.2 ms | **4.91 µs** |
| `cudaMemcpyAsync` | 25,136 | 624.5 ms | 24.8 µs（median ~6 µs；max 64 ms = 模型加载） |
| `cudaStreamCreate` | 1 | ~272 ms | 一次性（含 context 初始化） |
| `cudaFree` | 3,524 | 197.8 ms | 56.1 µs |
| `cudaMalloc` | 3,524 | 110.9 ms | 31.5 µs |
| `cudaMemsetAsync` | 4,932 | 26.6 ms | 5.4 µs |

- kernel launch host 成本 4.91 µs/次 × 129,702 = 636.2 ms（全程）≈
  **2.16 ms/traversal** —— 约为 GPU kernel 时间/traversal（3.66 ms）的 59%：
  host 平均跟得上（异步入队），但存在突发（prefill 步集中入队）。
- `cudaMalloc/cudaFree` 各 3,524 次（≈12/traversal，pool 的
  admit/finish/cancel 分配活动）+ `cudaMemsetAsync` 4,932 次：稳态 memops
  合计 ≈ **101.7 次/traversal**，GPU 时间 ≈ 93.9 ms/294 ≈ 0.32 ms/traversal。

### 4.3 H2D/D2H / sync（mixed）

- HtoD：2,726 次、128.6 ms —— 其中**一次性 63.9 ms 为模型加载**；稳态 2,725 次
  共 64.7 ms（avg 23.8 µs，median 0.99 µs）。
- DtoH：仅 174 次、8.0 ms（avg 46 µs = 每次 run 的 logits 回读，设计使然）。
- DtoD：22,236 次、46.8 ms（avg 2.10 µs）+ memset 4,932 次、18.6 ms：
  稳态小拷贝/清零开销（§12 候选 P5）。
- `cudaStreamSynchronize` host 时间仅 ~0.3 ms（百次级调用本身很轻；真正的
  等待发生在 `cudaMemcpyAsync`/`cudaFree` 的阻塞路径上）。

### 4.4 kernel launch 是否过碎？（launch-overhead 定量，§7；mixed）

- **总 kernel launches = 129,702**；**completed traversals = 294** →
  **441.16 kernels/traversal**。
- **99.77% 的 launch（129,408/129,702）来自 avg < 100 µs 的 kernel，占
  kernel 时间的 72.8%**；只有 2 个 kernel（`bf16_gemv_vec4_row` 250.0 ms +
  `batch_bf16_gemv_vec4_row` 31.6 ms，共 288 次 launch = 0.22% of launches）
  占 kernel 时间 **26.3%**。
- 结论（基于数据，非猜测）：**mixed** —— GPU 时间被少数 heavy kernel
  （LM head GEMV）显著主导（26.3% @ 0.22% launches），但其余 ~73.7% 分散在
  12.9 万次短 launch 上（avg 8 µs，大量 waves<1）；host launch 成本
  2.16 ms/traversal 与 GPU 3.66 ms/traversal 同量级 → **既不是纯
  heavy-kernel dominated，也不是纯 launch-fragmentation dominated**。
  （batched-only 的同项统计见 §4.5/§7。）

### 4.5 Batched-only serving profile（**Phase B 排名主依据**，132 traversals）

证据文件：`benchmarks/profiling/v07_batched_nsys_{kernel,cuda_api,memops,kernexec}_summary.txt`
（+ 同名 `.csv`）、`v07_batched_nsys.nsys-rep`、`v07_batched_kernel_families.txt`。
`--mode batched`（1 warmup + 5 measured，纯 continuous-batched serving 路径）：
**GPU kernel 总时长 516.4 ms**，**58,422 kernel launches**，
**132 completed traversals**（= 6 runs × 22）。

- **GPU 利用**：kernel 516.4 ms + 稳态 memops ~102.3 ms（HtoD 稳态 67.7 ms
  剔除一次性 67.7 ms 加载 + DtoD 20.2 + memset 10.4 + DtoH 4.0）≈ 618.7 ms
  （6 runs：1 warmup + 5 measured）→ 折算到 measured 区 ≈ 618.7×5/6 ≈
  515.6 ms vs 5 次 measured wall 5×0.116368 ≈ 581.8 ms → 稳态 GPU busy ≈
  **88.6%**（比例估计，单 stream 近似；未计加载时段）。
- **API（host）**：`cudaLaunchKernel` **58,422 次 / 335.6 ms / 5.74 µs**
  ≈ **2.54 ms/traversal**（vs GPU kernel 3.91 ms/traversal ≈ 65%）；
  `cudaMemcpyAsync` 11,102 次 / 447.4 ms（含 67.8 ms 一次性加载）；
  `cudaMalloc/cudaFree` 各 2,612 次（≈19.8/traversal）；
  `cudaMemsetAsync` 2,148 次（≈16.3/traversal）。
- **memops（GPU）**：HtoD 1,592 次 / 135.4 ms（一次性 67.7 ms = 加载；稳态
  1,591 次 67.7 ms、median 1.09 µs）；DtoD 9,438 次 / 20.2 ms；
  memset 2,148 次 / 10.4 ms；DtoH 72 次 / 4.0 ms（12/run logits 回读）。
- **launch 碎片化**：**442.59 kernels/traversal**（与 mixed 的 441.16 几乎
  相同 —— serial mode 并不额外增加 launch 密度）；**99.77% 的 launch
  （58,290/58,422）来自 avg < 100 µs 的 kernel，占 kernel 时间 71.4%**；
  2 个 heavy kernel（LM head 双变体，288 launches = 0.49% of launches）占
  kernel 时间 **26.5%**（`bf16_gemv` 20.19% + `batch_bf16_gemv` 6.34%）。
- **serving-path 归因结论**：batched-only 数据下瓶颈结构与 mixed 一致
  （§5.2 家族表），且 **LM head 占比不降反升（28.64% vs mixed 27.17%）**
  —— 混合 profile 并未误导排名；batched-only 是 Phase B 的正式依据。

## 5. Kernel-family 排名

由 `scripts/aggregate_kernel_families.py` 从 nsys CSV 聚合（单位：GPU kernel
时间占比）。**排序以 end-to-end total impact（total GPU time）为准，不看单次
latency。**

### 5.1 Mixed（comparison-harness，294 traversals，1075.0 ms = 100%）

`benchmarks/profiling/v07_kernel_families.txt`：

| family | total GPU time | % kernel time | calls | avg latency |
|---|---|---|---|---|
| **w4a16/int4 projection**（`int4gemv_rowtile4_bf16` 等） | 460.8 ms | **42.86%** | 54,684 | 8.43 µs |
| **bf16 lm head**（`bf16_gemv_vec4_row` / `batch_bf16_gemv_vec4_row`） | 292.1 ms | **27.17%** | 294 | 993.7 µs |
| **deltanet**（conv/gbeta/delta/gated_rmsnorm） | 172.4 ms | **16.03%** | 21,492 | 8.02 µs |
| elementwise/residual/swiglu（add/silu_mul/gate_mul/split_q_gate） | 61.8 ms | 5.75% | 24,696 | 2.50 µs |
| rmsnorm | 48.8 ms | 4.54% | 17,934 | 2.72 µs |
| paged attention（scores/softmax/pv/kv_write） | 30.9 ms | 2.87% | 7,056 | 4.38 µs |
| rope | 8.3 ms | 0.77% | 3,528 | 2.35 µs |
| embedding | 0.047 ms | ~0% | 18 | 2.60 µs |

Top 10 by total GPU time：`bf16_gemv_vec4_row` 250.0 ms / 23.25%（276 calls，
905.8 µs）；`int4gemv`（896 grid）132.8 ms / 12.36%；`deltanet_delta`
121.2 ms / 11.27%（4,968，24.4 µs）；`int4gemv`（256 grid）100.6 ms / 9.35%；
`int4gemv`（1536 grid）69.0 ms / 6.41%；`int4gemv`（4 grid）47.6 ms / 4.43%；
`qwen35_rmsnorm_zc<4>` 37.1 ms / 3.45%；`int4gemv`（512 grid）32.0 ms / 2.98%；
`batch_bf16_gemv_vec4_row` 31.6 ms / 2.94%（12，2633 µs）；`qwen35_add`
31.2 ms / 2.90%。

### 5.2 Batched-only serving profile（**Phase B 排名主依据**，132 traversals，
516.4 ms = 100%）

`benchmarks/profiling/v07_batched_kernel_families.txt`：

| family | total GPU time | % kernel time | calls | avg latency |
|---|---|---|---|---|
| **w4a16/int4 projection**（`int4gemv_rowtile4_bf16` 21,204 + `batch_int4gemv_rowtile4_bf16` 3,348） | 220.6 ms | **42.72%** | 24,552 | 8.98 µs |
| **bf16 lm head**（`bf16_gemv_vec4_row` 114 + `batch_bf16_gemv_vec4_row` 18） | 147.9 ms | **28.64%** | 132 | 1120.3 µs |
| **deltanet**（`deltanet_*` 9,504 + `batch_deltanet_*` 324×4） | 79.4 ms | **15.37%** | 9,828 | 8.08 µs |
| elementwise/residual/swiglu | 28.2 ms | 5.47% | 11,088 | 2.55 µs |
| rmsnorm | 22.2 ms | 4.30% | 8,052 | 2.76 µs |
| paged attention（`qwen35_paged_*` 684×4 + `batch_paged_*` 108×4） | 14.2 ms | 2.75% | 3,168 | 4.49 µs |
| rope | 3.8 ms | 0.74% | 1,584 | 2.40 µs |
| embedding | 0.049 ms | ~0% | 18 | 2.70 µs |

Top 10 by total GPU time：`bf16_gemv_vec4_row`（B=1）**104.2 ms / 20.19%**
（114 calls，914.3 µs）；`int4gemv`（896 grid）55.6 ms / 10.77%（5,472，
10.2 µs）；`deltanet_delta` 50.7 ms / 9.81%（2,052，24.7 µs）；
`int4gemv`（256 grid）42.1 ms / 8.14%；`batch_bf16_gemv_vec4_row` **32.7 ms /
6.34%**（12 calls，2727.8 µs）；`int4gemv`（1536 grid）28.9 ms / 5.59%；
`int4gemv`（512 grid）19.9 ms / 3.86%；`qwen35_rmsnorm_zc<4>` 15.5 ms / 3.01%
（5,586）；`int4gemv`（4 grid）13.4 ms / 2.59%；`qwen35_add` 13.1 ms / 2.54%。

Top 10 by invocation count：`qwen35_rmsnorm_zc<4>` 5,586；
`int4gemv`（896/256 grid）各 5,472；`qwen35_add` 5,472；`int4gemv`（512
grid）4,104；`qwen35_silu_mul` 2,736；`deltanet_delta` 2,052；
`int4gemv`（1536/4 grid）各 2,052；`deltanet_gated_rmsnorm` 2,052。

**batched-only 逐 kernel 实例数**（6 runs × 22 traversals；用于 NCU
cohort/skip 计算与交叉核对）：`int4gemv` B=1 21,204（= 19 B=1 traversals/run
× 186）；`batch_int4gemv` 3,348（= 3 batch traversals/run × **186/batch
traversal**）；`bf16_gemv` B=1 114（= 19×6）；`batch_bf16_gemv` 18（= 3×6）；
`deltanet_delta` 2,052 / `batch_deltanet_delta` 324（= 3×6×18 / 3×6）；
`batch_paged_*` 各 108。

**serving 路径关键观察（batched-only）**：

1. **家族排名与 mixed 完全一致**（W4A16 42.7% > LM head 28.6% > DeltaNet
   15.4%）→ 混合 profile 未误导；LM head 在纯 serving 路径占比**更高**
   （28.64% vs 27.17%）。
2. **B=1 vs batch（B>1）变体精确拆分**（从 batched nsys CSV 逐 kernel 行
   聚合；`batch_*` kernel 仅出现在 committed B≥2 cohort）：

   | family | B=1 变体 | batch 变体 | 合计 |
   |---|---|---|---|
   | w4a16/int4 | 174.61 ms（33.82%，21,204 calls） | 45.97 ms（**8.90%**，3,348） | 220.58 ms / 42.72% |
   | bf16 lm head | 104.23 ms（20.19%，114） | 43.65 ms（**8.45%**，18） | 147.88 ms / 28.64% |
   | deltanet | 68.31 ms（13.23%，8,532） | 11.04 ms（2.14%，1,296） | 79.36 ms / 15.37% |
   | paged attention | 11.78 ms（2.28%，2,736） | 2.45 ms（0.47%，432） | 14.22 ms / 2.75% |
   | elementwise/rmsnorm/rope/embedding | 53.75 ms（10.41%） | 0.57 ms（0.11%） | 54.32 ms / 10.52% |

   → **B>1 batch 变体合计 103.69 ms = 20.08% 的 serving kernel 时间**
   （18 个 batch traversals × 5.76 ms/traversal）；B=1 变体 79.4%。
   **Phase B 的 kernel 优化必须同时覆盖 B=1 与 batch 变体**（batch 变体在
   W4A16 与 LM head 两个最大 family 里各占 ~9%）。
3. 每 batch traversal 的 launch 数与 B=1 traversal 相同（186 int4 + 其余
   形状不变，batch 维度在 kernel 内部处理）→ **continuous batching 不减少
   launch 数**，442.59 kernels/traversal 与 mixed 441.16 一致。

## 6. Nsight Compute — batch-aware 测量（B=1 vs B=2/3，serving 路径）

> NCU 为**缩短 workload**（1 warmup + 1 measured run，**`--mode batched`
> serving 路径**，真实 checkpoint + 真实 committed batch，固定可复现，明确
> 区别于 canonical benchmark）。8 个 batch-aware target（`^` 锚定裸 kernel
> 名区分 B=1 / `batch_*`；committed cohort 为 B=2,B=3,B=3，每个 batch
> traversal 186 个 batch-int4 launches，`--launch-skip 186` 从 B=2 cohort
> 移到 B=3 cohort）：`int4_b1` / `int4_batch`(B=2) / `int4_batch_late`(B=3) /
> `lmhead_b1` / `lmhead_batch` / `deltanet`(B=1) / `deltanet_batch`(B=2) /
> `rmsnorm`。每 target 12 launches（`lmhead_batch` 进程内只有 6 次 → 全采）。
> 数据：`benchmarks/profiling/v07_ncu_<target>.txt`（均含 exact SHA header）。

### 6.1 W4A16 `int4gemv_rowtile4_bf16` / `batch_int4gemv_rowtile4_bf16`
（family 42.7% serving）→ **mixed（按形状分裂），batch 维度有显著 per-token 收益**

B=1（12 launches，56 regs，grid 与投影形状一一对应，block=128）：

| grid | dur | SM% | Mem GB/s | DRAM% | occ% | waves | 主 stall（Warp State） |
|---|---|---|---|---|---|---|---|
| 1536 | 17.3 µs | 35.0 | 187 | 30.0 | 82.5 | 2.82 | barrier 9~10 cyc（WCI 17~18） |
| 896 | 12.3 µs | 28.8 | 154 | 24.5 | 75.7 | 1.65 | barrier ~9 cyc |
| 512 | 7.9 µs | 26.4 | 137 | 22.5 | 79.6 | 0.94 | barrier 8.6~9.1 cyc |
| 256（两形状） | 8.1 / 9.8 µs | 22.5 / 33.6 | 135 / 193 | 22.4 / 31.7 | **42.3 / 41.5** | **0.47** | barrier ~9 cyc |
| 4 | 5.9 µs | **0.28** | 1.8 | **0.30** | **11.3** | **0.01** | barrier 10 cyc（WCI 18） |

B=2（`int4_batch`，12 launches，58 regs，grid = 2×B=1）：

| grid | dur | SM% | Mem GB/s | DRAM% | occ% | waves |
|---|---|---|---|---|---|---|
| 3072 | 29.0 µs | 40.9~41.5 | 112 | 17.3~17.6 | 83.5 | 5.65 |
| 1792 | 19.7 µs | 36.1 | 96 | 15.3 | 81.0 | 3.29 |
| 1024 | 12.6 µs | 33.4 | 86 | 14.1 | 80.9 | 1.88 |
| 512（两形状） | 9.7 / 14.8 µs | 36.5 / 43.6 | 112 / 129 | 17.9 / 20.3 | 79.6 / 75.8 | 0.94 |
| 8 | 5.8 µs | **0.56** | 2.2 | **0.36** | **11.3** | **0.01** |

B=3（`int4_batch_late`，12 launches，58 regs，grid = 3×B=1）：

| grid | dur | SM% | Mem GB/s | DRAM% | occ% | waves |
|---|---|---|---|---|---|---|
| 4608 | 41.3 µs | 44.4 | 79 | 12.5 | 84.1 | 8.47 |
| 2688 | 25.5 µs | 41.5 | 75 | 11.6 | 83.5 | 4.94 |
| 1536 | 16.7 µs | 37.1~37.7 | 65 | 10.5 | 82.0 | 2.82 |
| 768（两形状） | 14.9 / 18.8 µs | 35.4 / 51.7 | 74 / 102 | 11.6 / 16.0 | 78.4 | 1.41 |
| 12 | 5.8 µs | **0.85** | 2.5 | **0.42** | **11.3** | **0.02** |

- 中大 grid（≥512）：DRAM 22~32% + SM 26~52% → 偏 **memory-bound** 但远未
  贴 roofline（65~194 GB/s ≈ 峰值 10~32%）；occupancy 76~84%。
- grid=256（B=1）：occ 41~42%、waves 0.47 → **occupancy-limited**。
- grid=4/8/12（degenerate 小 N 形状）：SM 0.3~0.9%、occ 11.3%、waves 0.01 →
  **launch/latency-dominated**；**batch 维度对该形状无改善**（5.8→5.8 µs，
  瓶颈是 N 维并行度而非 batch）。
- **batch-size 效应（显式回答）**：**显著**。同一投影形状下 per-token 时间
  随 B 下降（大 grid 1536/3072/4608：17.3 → 14.5（B=2）→ 13.8（B=3）µs/
  token，**−16%/−21%**；896 系：12.3 → 9.8 → 8.5；512 系：7.9 → 6.3 →
  5.5 µs/token）。机理（NCU 观察）：随着 B 增大，跨 batch-row 的 weight
  cache reuse 增强——L2 hit 提高、effective DRAM traffic/token 下降（每个
  batch-row block 仍独立发起逻辑 weight load，`blockIdx.y = b` 各读各的、
  相邻 block 的 weight 请求命中同一 cache line；当前 kernel **没有**显式
  single-load-and-share 的 weight cooperative scheme）→ per-token DRAM
  流量下降（DRAM% 30→17.4→12.5），kernel 从 DRAM-leaning 移向
  SM-leaning（SM% 35→41→44.4），occupancy 微升（82.5→84.1），regs 56→58。主 warp stall 三个 B 档一致（CTA barrier ~8~10 cyc，占 issue
  间隔 ~54%）→ 瓶颈结构不随 B 改变，只有**带宽利用/效率**随 B 改善。

### 6.2 `bf16_gemv_vec4_row` / `batch_bf16_gemv_vec4_row`（LM head，28.6%
serving）→ **memory/DRAM-bound（roofline）；batch-size 无 per-token 效应**

B=1（12/12 launch 形态一致）：grid=248,320（vocab/row，1 block/行）、
block=256、23 regs、Duration **979 µs**（NCU replay；nsys 914 µs）、
Memory Throughput **520 GB/s（≈ 2080 Ti 峰值的 85~94%）**、**DRAM 80%**、
SM 42.3%、L2 hit 1.6%、occupancy 72.6%、waves 913、主 stall barrier 6.5 cyc
（WCI 18.5）。= 单次全量流式 ~503 MB BF16 权重（1024×248,320），**已贴在
DRAM 带宽 roofline 上** → kernel 微优化无空间。

batch 变体（6 launches：B=2 grid=496,640 ×2、B=3 grid=744,960 ×4；28 regs）：
Duration NCU replay **1.98~2.00 ms（B=2）/ 2.96 ms（B=3）**（nsys
1819.9 / 2727.8 µs）、Memory **510~517 GB/s**、**DRAM 79.1~79.3%**、SM
37.5%、occupancy 73.4~73.5%、waves 1826/2739、主 stall barrier 5.7 cyc。

- **batch-size 效应（显式回答）**：**无（per-token 不变）**。per-token 时间
  B=1 914.3 → B=2 909.9 → B=3 909.3 µs/token（nsys），Duration 对 B
  **严格线性**（B=2 = 1.99×、B=3 = 2.98× B=1）：batch kernel 每个
  (row, b) block 独立读取整行权重（`blockIdx.y = b`，无跨 batch-row 的
  weight 共享），总 DRAM 流量随 B 线性增长，而 kernel 已贴 DRAM roofline
  （510~517 GB/s、带宽饱和）→ Duration 随 B 严格线性扩展、per-token 不变，
  纯线性扩展。→ LM head 上**批处理不产生摊销收益**；唯一杠杆是减少权重
  字节数（P4 量化，需 reviewer 决策）。

### 6.3 `deltanet_*` / `batch_deltanet_delta`（15.4% serving）→
**latency/dependency-bound + occupancy-limited；B=2 有 per-token 收益**

B=1（12 launch = 4 kernel × 3；serving 路径 B=1 = prefill + size-1 cohort）：

| kernel | grid | dur | SM% | DRAM% | occ% | regs | waves | 主 stall |
|---|---|---|---|---|---|---|---|---|
| `deltanet_delta`（状态更新，recurrent） | **16** | **30.1 µs** | **2.4** | 5.4 | **11.6** | 64 | **0.03** | const-cache 5.4~6.6 + scoreboard 6.0~7.1 cyc |
| `deltanet_gbeta` | **1** | 2.6 µs | 0.01 | 0.02 | 3.1 | 16 | ~0 | — |
| `deltanet_conv` | 24 | 2.8 µs | 2.2 | 5.7 | 24.6 | 20 | 0.09 | — |
| `deltanet_gated_rmsnorm` | 16 | 2.9 µs | 0.8 | 0.5 | 12.4 | 18 | 0.03 | — |

`batch_deltanet_delta`（B=2 cohort，12 launches 全部来自首个 batch
traversal，grid=32 = 2×B=1；62 regs）：Duration **31.0~32.1 µs**、SM
4.4~4.7%、DRAM 9.9~10.5%、occupancy 11.6~11.9%、waves 0.06、主 stall
**scoreboard dependency 12.9~13.2 cyc**（WCI 16.9~17.0，≈ 依赖等待主导）。

- `deltanet_delta`：serving 中 50.7 ms / 9.81%（B=1 2,052 次）+ batch 变体
  （B=2 15.8 µs/token vs B=1 30.1 µs/token → **per-token −48%**；B=3 cohort
  grid=48 未被 NCU 采样 —— 12 launches 全落在首个 B=2 traversal，**标注为
  未测**）。
- 判定：**latency/dependency-bound（叠加 occupancy-limited）**；batch 维度
  带来 per-token 收益但**不改变瓶颈分类**（SM 2.4~4.7%、occ ~11.7%、
  waves 0.03~0.06 双档均无 SM/DRAM 压力）。

### 6.4 `qwen35_rmsnorm_zc` / `deltanet_gated_rmsnorm`（rmsnorm 4.3%
serving）→ **launch-overhead dominated**

12 launch（`--mode batched` 下首批 B=1）：`qwen35_rmsnorm_zc`（grid=1，
22 regs，2.88~2.94 µs，SM 0.09~0.10%，occupancy 22.6~22.9%）×9、
`deltanet_gated_rmsnorm`（grid=16，2.94~2.98 µs，SM 0.83%）×3。
B=1 时每行 RMSNorm（hidden=1024）只投 1~16 个 block → waves≈0、SM<1%：
每个 ~2.9 µs ≈ launch + 微小计算。serving 全程 8,052 次 × 2.76 µs =
22.2 ms（4.3%）。判定：**launch-overhead dominated**（kernexecsum 佐证：
host API ~4.8~5.7 µs/kernel 与 GPU 执行 2.9 µs 同量级）。

### 6.5 Batch-size 效应 —— 汇总显式回答（reviewer blocker 4）

| hotspot | batch-size 是否显著改变 latency/occupancy/bandwidth 行为 | 数据 |
|---|---|---|
| W4A16 INT4 GEMV（42.7%） | **显著**：per-token 时间随 B 下降（−16%~−30%）；DRAM% 30→17→12.5（权重摊销）；SM% 35→41→44；occ 82.5→84.1；regs 56→58；degenerate 小 N 形状（grid≤12）三档均 ~5.8 µs、occ 11.3%，**不受 B 影响** | §6.1 三张表 |
| BF16 LM head（28.6%） | **无**：per-token 914.3/909.9/909.3 µs（B=1/2/3）；Duration 严格线性（1.99×/2.98×）；双档均 DRAM ~79-80%、510-520 GB/s roofline、occ 72.6→73.4；regs 23→28 | §6.2 |
| DeltaNet delta（family 15.4%） | **per-token 显著（B=2 −48%）、分类不变**：B=1 30.1 µs/token vs B=2 15.8 µs/token；双档均 SM 2.4~4.7%、occ ~11.7%、waves 0.03~0.06 的 latency/dependency-bound；主 stall 从 const-cache/scoreboard 混合（B=1）变为 scoreboard 主导（B=2，13 cyc） | §6.3 |
| rmsnorm / elementwise | 未测 batch 变体（B=1 已 launch-overhead dominated；batch 变体 serving 占比 0.11%） | §6.4 |

**结论**：batch-size 对最大 family（W4A16）的 per-token 效率有明确收益
（权重摊销 → DRAM 下降、SM 上升），对 LM head 无 per-token 效应（已贴
roofline，线性扩展），对 DeltaNet 有 per-token 收益但不改变瓶颈分类。
→ **Phase B 优化必须同时覆盖 B=1 与 batch 变体，且 W4A16 的 batch 变体
（`batch_int4gemv_rowtile4_bf16`，8.9% serving）与 B=1 变体（33.8%）是
同一 family 的两个优化对象。**

## 7. Launch-overhead 分析（nsys 定量结论）

**Batched-only serving profile（主依据，132 traversals）**：

| 量 | 值 |
|---|---|
| total kernel launches（`--mode batched` 全程） | **58,422** |
| completed model traversals | 132 |
| **kernels / traversal** | **442.59** |
| total GPU kernel time | 516.4 ms（3.91 ms/traversal） |
| `cudaLaunchKernel` host 总时间 / avg | 335.6 ms / 5.74 µs |
| host launch 时间 / traversal vs GPU kernel / traversal | 2.54 ms vs 3.91 ms（≈65%） |
| short-duration kernels（avg < 100 µs） | 58,290/58,422 launches（**99.77%**）；76/79 kernel 种；占 kernel 时间 **71.4%** |
| heavy kernels（avg ≥ 100 µs） | 2 种（LM head ×2 变体）、288 launches（0.49%）、占 kernel 时间 **26.5%** |

Mixed（对照，294 traversals）：129,702 launches、**441.16 kernels/
traversal**、1075.0 ms（3.66 ms/traversal）、host launch 636.2 ms / 4.91 µs
（2.16 ms/traversal ≈ GPU 的 59%）、short 99.77%（72.8% of time）、heavy
288 launches（0.22%）占 26.3%。

**最终判定：mixed。** GPU 时间一半以上由 W4A16 GEMV（serving 42.7%，
24,552 次短 launch 含 3,348 batch）+ LM head（28.6%，132 次 heavy launch）
两个 family 主导；launch fragmentation 是**真实的次级因素**（442.59
launch/traversal —— continuous batching **不减少** launch 数、host launch
与 GPU 同量级（65%）、99.77% 短 kernel、大量 waves<1），但不是唯一因素。

## 8. Bottleneck classification（每个 NCU hotspot，附数据依据）

| hotspot | 分类 | 数据依据 |
|---|---|---|
| `bf16_gemv_vec4_row`（LM head，B=1）+ `batch_bf16_gemv`（B=2/3） | **memory-bound（DRAM roofline），双档一致** | B=1：520 GB/s ≈ 峰值 85~94%、DRAM 80%、SM 42.3%、L2 hit 1.6%、979 µs ≈ 503 MB/520 GB/s；batch：510~517 GB/s、DRAM 79.1~79.3%、SM 37.5%、occ 73.4%；per-token 不变（§6.2） |
| `int4gemv_rowtile4_bf16`（W4A16，B=1）+ `batch_int4gemv_rowtile4_bf16`（B=2/3） | **mixed**（按形状：大 grid 偏 memory-bound、中 grid occupancy-limited、grid≤12 latency-dominated；三档 batch 形状分裂一致） | B=1：DRAM 22~32% + SM 26~35%（512~1536）；occ 41~42% / waves 0.47（256）；SM 0.28% / occ 11.3% / waves 0.01（4）；B=2/3 同形状同分类（DRAM 10~20%、SM 33~52%，§6.1） |
| `deltanet_delta` + `batch_deltanet_delta` + `deltanet_gbeta`/`conv`/`gated_rmsnorm` | **latency/dependency-bound + occupancy-limited**（`gbeta` 单项可标 launch-overhead dominated；batch 变体同分类） | B=1 delta：30.1 µs @ SM 2.4%、DRAM 5.4%、occ 11.6%、grid 16、waves 0.03、64 regs；B=2 delta：31.6 µs @ SM 4.6%、occ 11.7%、waves 0.06、62 regs、scoreboard stall 13 cyc；gbeta：grid=1、2.6 µs |
| `qwen35_rmsnorm_zc` / `deltanet_gated_rmsnorm` | **launch-overhead dominated** | grid 1~16、waves≈0、SM 0.09~0.83%、每个 ~2.9 µs ≈ host API 时间（~5.7 µs）同量级 |
| elementwise（add/silu_mul/…，nsys 级） | **launch-overhead dominated（倾向）** | serving 11,088 calls、avg 2.55 µs、SM 个位数（kernexec 佐证）；未 NCU → 严格标注为倾向 |
| paged attention / rope / embedding（nsys 级） | **INCONCLUSIVE（未 NCU）** | serving 占比 ≤ 2.75%，未进 top-4；仅有 nsys 级 total 数据 |

## 9. CUDALab 映射（**仅在 profiling 完成后**检查；frozen SHA
`cb6a6a9ef76394cc66d272c99aa8697db0a34f1e`，只读 clone）

> CUDALab = 自主 CUDA kernel 优化实验室（operator 集：RMSNorm / Softmax /
> RoPE / GEMV(BF16) / QGEMV(INT8) / INT4GEMV(W4A16 G=128, fp16 scale)）。
> **本 Phase 未 port 任何优化**（仅建立映射）。

| measured CUDALM hotspot | CUDALab 对应 | 可借鉴 technique / 状态 |
|---|---|---|
| **W4A16/INT4 GEMV（serving 42.7%，含 batch 8.9%）**：`int4gemv_rowtile4_bf16_kernel` + `batch_int4gemv_rowtile4_bf16_kernel` | `kernels/int4gemv/`（v0.7 线）：baseline → vec16_row(KEEP 2.0×) → rowtile4(KEEP 1.19×) → rowtile8(NEUTRAL) → **rowtile4_hx（incumbent，2.59× vs baseline）** | **核心 technique 已 port**：CUDALM 的 rowtile4_bf16 = R=4 行块 + x 寄存器驻留（bf162）+ 16B 向量化 W load + single-group scale lemma（g=v>>2），与 CUDALab hx incumbent 同源。**未覆盖**：(a) CUDALM 小形状（hidden=1024，K≤3072，N 可到 3072 但含 N=极小的投影 → grid=4/256 degenerate case）的**形状特定调优**（CUDALab 主形状 4096²、K≥1024）；(b) **batch 变体**（`batch_int4gemv_rowtile4_bf16`）——CUDALab 全部 B=1，B>1 需**重新设计 batch dimension**；batch-aware NCU（§6.1）给出 B=2/3 的 per-grid 基线（occ 76~84%、DRAM 10~20%、SM 33~52%）可作为 batch 变体调优的验收参照；(c) NCU 结论「occupancy 是比 MLP 更强的 DRAM 杠杆」可用于 CUDALM 小 shape 的调优方向 |
| **BF16 LM head（serving 28.6%，含 batch 8.5%）**：`bf16_gemv_vec4_row_kernel` / `batch_bf16_gemv_vec4_row` | `kernels/gemv/`（v0.5 线）：**gemv_vec4_row（incumbent 1.54×）**；warp_vec4_b256/b512（KEEP 但更慢）；splitk4（REJECT） | **已是 CUDALab 最优变体**；CUDALab 无更快的 BF16 GEMV → 无直接可 port。B=1 与 batch 双档均贴 DRAM roofline（510~520 GB/s、DRAM ~79-80%）、**per-token 时间不随 B 变化**（§6.2）→ 唯一杠杆是**减少权重字节数（LM head 量化，语义变更，需 reviewer 决策）**；batch 化本身不产生收益 |
| **DeltaNet（15.9%）**：`deltanet_delta/gbeta/conv/gated_rmsnorm` | **no direct CUDALab analogue**（operator 集无线性注意力/recurrent state update） | 无对应；需原创设计（状态更新的并行化/分块/融合） |
| **Paged attention（2.9%）**：`qwen35_paged_*` / `batch_paged_*` | **no direct CUDALab analogue**（CUDALab softmax 为非 paged 全行；paged-KV 语义/间接寻址无对应） | 无对应 |
| **RMSNorm（4.5%）**：`qwen35_rmsnorm_zc<1/4>` / `rmsnorm_v4_half` | `kernels/rmsnorm/`（v4：向量化 + 寄存器驻留单遍，已验证） | **v4 已 port**（`rmsnorm_v4_half_kernel`）；CUDALM 的 zero-centered 变体（Qwen3.5 专用）与 grid=1（B=1、hidden=1024）的**发射几何**是 CUDALM 特有的，CUDALab 无对应小 hidden 形状 |
| **RoPE（0.76%）**：`rope_v3_half2` / `qwen35_partial_rope` | `kernels/rope/`（rope_v3_half2 已验证） | **已 port** |
| **elementwise（5.75%）**：add/silu_mul/gate_mul/split_q_gate | **no direct CUDALab analogue**（CUDALM 特有 elementwise 链） | 无对应；融合（fusion）亦无 CUDALab 对应物 |
| **scheduler / continuous batching / launch fragmentation** | **no direct CUDALab analogue**（CUDALab 是 B=1 单 op 优化闭环，无调度器/多请求语义） | 无对应 |

## 10. Ranked Phase B optimization backlog

排序依据 = measured end-to-end impact × 证据置信度 ÷ 实现风险（全部来自
§4~§8 实测，不凭直觉）。

> **本 round 以 batched-only serving profile（§4.5/§5.2，Phase B 的正式
> 依据）重新验证排名**：家族排名与 mixed 完全一致（W4A16 42.72% > LM head
> 28.64% > DeltaNet 15.37%），且 batch-size 效应数据（§6.5）明确 W4A16 的
> B=1 与 batch 变体是同一 family 的两个优化对象（serving 合计 42.72%，其中
> batch 8.90%）。**P1 维持为 Phase B 首目标**（理由见本节末），以下数字
> 除注明外均为 serving（batched-only）口径。

**P1 — W4A16 INT4 GEMV 形状特定调优（occupancy/向量化/N 并行度；**同时
覆盖 B=1 与 batch 变体**）**
- Hotspot：`int4gemv_rowtile4_bf16_kernel`（serving 33.82%）+
  `batch_int4gemv_rowtile4_bf16_kernel`（serving 8.90%）= family 42.72%
  （24,552 launches）
- Profiler evidence：grid=4/8/12（B=1/2/3 各档）→ SM 0.28~0.85%/occ 11.3%/
  waves 0.01（~5.8 µs 近乎纯 latency，**batch 不改善该形状**）；grid=256
  （B=1）→ occ 41~42%/waves 0.47；grid=512~1536（B=1）→ DRAM 22~31%、
  137~194 GB/s（峰值 22~31%，远未贴 roofline）；batch 变体同形状 B=2/3
  已测（DRAM 10~20%、SM 33~52%、occ 76~84%，§6.1）
- Bottleneck：mixed（degenerate 小 N 形状 latency/occupancy-limited；中形状
  memory-bound 有带宽余量）
- Optimization hypothesis：(a) 极小 N 投影换映射（K 维并行/多块 + 确定性
  combine，或合并相邻小投影）；(b) 中形状提高 DRAM 利用率（CUDALab 结论：
  occupancy 杠杆 > MLP；更多 N 并行 + 保持 k-order 的向量化 ILP）；
  (c) **batch 变体（B>1）以 §6.1 的 B=2/3 per-grid 基线为验收参照**
  （per-token 时间目标 ≤ B=1 per-token 的 80%）
- Expected e2e impact：family 42.72% 中若 30~50% 可压缩 → 12.8~21.4%
  serving kernel 时间 ≈ **6~11% wall**（按 serving GPU 3.91 ms/traversal、
  wall 5.29 ms/traversal）
- Risk/complexity：中（kernel 局部、无调度/状态语义变更；**首要风险 = 精确
  parity 合同**：任何改变 fp32 累加顺序的改动会破坏 EXACT logits 门 →
  优先选**保持 k-order 累加顺序**的 occupancy/向量化子集；顺序保持子集之外
  需 reviewer 裁决容忍门）
- CUDALab reference：`kernels/int4gemv/int4gemv_rowtile4_hx.cu`（incumbent）、
  INT4GEMV-0001..0004 实验链 + paired benchmark harness（可作验证工具）

**P2 — DeltaNet delta-rule 状态更新 kernel（+ gbeta/conv/gated_rmsnorm 融合）**
- Hotspot：`deltanet_delta_kernel`（serving 9.81%，2,052 次）+
  `batch_deltanet_delta`（B=2 cohort）+ deltanet family 15.37%（9,828 calls）
- Profiler evidence：B=1 30.1 µs @ SM 2.4%、DRAM 5.4%、occ 11.6%、grid=16、
  waves 0.03、64 regs；B=2 31.6 µs @ SM 4.6%、occ 11.7%、waves 0.06、
  62 regs（per-token −48% 但分类不变，§6.3）；`gbeta` grid=1
- Bottleneck：latency/dependency-bound + occupancy-limited
- Optimization hypothesis：状态更新（recurrent over state dim）重块化/分维
  并行（确定性 combine）+ 状态行向量化；gbeta/conv/gated_rmsnorm 融入
  delta 前序（减 3 次 launch/层）；batch 变体同步覆盖（其 nsys 时间 2.14%
  serving）
- Expected e2e impact：delta 3× → 省 ~0.36 ms/traversal ≈ **~7% wall**
  （按 serving 3.91 ms/traversal、5.29 ms/traversal）；融合再减 kernel
  时间的一部分
- Risk/complexity：中高（recurrent 语义 + 精确 parity 风险同上；64 regs 需
  控制）
- CUDALab reference：**no direct analogue**（原创设计）

**P3 — elementwise/norm/rope 融合（降 launch 数，442.59 → 目标 ~300/traversal）**
- Hotspot：rmsnorm 4.30% + elementwise 5.47% + rope 0.74%（共 ~10.5%
  serving kernel 时间，全部 2.3~2.8 µs 的 launch 延迟主导 kernel）
- Profiler evidence：grid≤16、waves≈0、SM<1%（rmsnorm NCU 佐证）；442.59
  kernels/traversal；host launch 2.54 ms/traversal（GPU 的 65%）
- Bottleneck：launch-overhead dominated（小 kernel 簇）
- Optimization hypothesis：层内 elementwise 链融合（add+silu_mul+
  split_q_gate；rmsnorm 并入 producer），不动 CUDA Graph
- Expected e2e impact：8~12% serving kernel 时间 + host API 时间下降 ≈
  **3~6% wall**
- Risk/complexity：中（融合代码面大；parity 风险同上；无 Graph 则收益上限
  有限）
- CUDALab reference：rmsnorm v4 / gemv 已 port（无增量）；fusion 本身
  **no direct CUDALab analogue**

**P4 — LM head 量化（W4A16，语义变更 → reviewer 决策项，非默认执行）**
- Hotspot：`bf16_*_gemv_vec4_row` serving 28.64%（132 launches，26.5%
  serving kernel 时间）
- Profiler evidence：B=1 与 batch 双档均 510~520 GB/s ≈ DRAM roofline
  （DRAM ~79-80%、L2 hit 1.6%）、**per-token 不随 B 变化**（§6.2）→
  kernel 微优化与批处理摊销均无空间（memory-bound 已触顶）
- Bottleneck：memory-bound（roofline）
- Optimization hypothesis：LM head 权重 BF16→INT4（~0.5 GB）→ IO 减半
  以上 → 914 µs/token → ~400 µs 量级
- Expected e2e impact：最高 ~**15~20% wall**（若量化可接受）
- Risk/complexity：**高**（模型格式/语义变更；输出 logits 变化 → 精确 parity
  合同必须重构为量化感知；CUDALM 尚无 LM head 量化路径）
- CUDALab reference：INT4GEMV incumbent（rowtile4_hx）可作 head 的 kernel
  底座（若 reviewer 批准量化方向）

**P5 — 稳态 memops 批处理（serving：DtoD 20.2 ms + HtoD 稳态 67.7 ms +
memset 10.4 ms + DtoH 4.0 ms ≈ 100.4 memops/traversal）**
- Bottleneck：小拷贝/清零碎片（nsys memops summary）
- Optimization hypothesis：per-step 元数据单次 HtoD（device 侧 seed/常量）、
  免 DtoD 中转
- Expected e2e impact：~**1~2% wall**
- Risk/complexity：低
- CUDALab reference：no direct analogue

**推荐 Phase B 首先优化：P1（W4A16 INT4 GEMV 形状特定调优，B=1 + batch
变体）**。基于 serving-path 数据的理由：
(1) serving 路径最大 family（42.72% GPU kernel 时间；第二大的 LM head
28.64% 双档均已贴 DRAM roofline 且 per-token 不随 B 变化，kernel 层面无可
挤空间）；(2) 缺口被 profiler 量化到每个 grid 形状 × 每个 B 档（waves/
occupancy/DRAM/regs/主 stall 均有数据，§6.1）；(3) 有直接 CUDALab
reference（incumbent 已验证 + paired harness 可复用为验证工具）；(4) 改动面
最小（单 kernel family，无调度/状态/格式语义）；(5) 存在**保持 fp32 累加
顺序**的安全子集（N 维并行几何、k-order 向量化、block/寄存器调优），可在
不破坏 v0.6 EXACT parity 门的前提下先落地；(6) batch-size 数据（§6.5）
表明 serving 的真实形态是 B=1 与 B=2/3 混合，P1 范围显式包含 batch 变体
（8.90% serving）。

## 11. Limitations

- **workload 小**（27 logical tokens，prompts 2~5 tokens；committed batch
  仅 B=2/3）：prefill 形状（N>1 的 token 维并行）在 nsys 全程中占比极低；
  **prefill-heavy profile 未做**（Phase B 应补）。kernel family 的**相对**
  占比在本 model/regime 内可信，绝对 wall 收益随 token 数缩放。
- **NCU 为缩短 workload**（1 warmup + 1 measured、`--mode batched`、每
  target 12 launches）；B=1 与 B=2/3 batch 变体**均已 NCU**（§6），例外：
  `batch_deltanet_delta` 的 **B=3 cohort（grid=48）未采样**（12 launches
  全落在首个 B=2 traversal），该档 per-token 行为标注为**未测**；
  elementwise / paged / rope 未 NCU（占比 ≤ 2.75%）。
- 冻结 runtime **无 NVTX 区间** → prefill/decode 分段的 GPU gap 归属只能用
  比例估计（§4.1/§4.5），更细粒度 INCONCLUSIVE；加 NVTX 属 tooling 修改，
  留待 Phase B 决策。
- nsys 2022.4 / ncu 2022.3 为 CUDA 11.8 匹配版本，无 2023+ 的新 metrics
  （如 `gpu__time_active` 直接指标）；GPU 利用率由 kernel+memops busy 时间
  与 wall 的比值推算（单 stream 近似）。
- 单 GPU、单环境（RTX 2080 Ti sm_75）；数字不是跨环境结论。
- CUDALab 映射基于其 docs/kernels 静态阅读（frozen SHA 只读 clone），未运行
  CUDALab 任何 benchmark。

## 12. Reproduction commands

**exact-SHA 纪律（reviewer 要求）**：全部证据必须在
**V07A_EVIDENCE_SHA = V07A_FUNCTIONAL_SHA = `0a952879da6e6cbb26a4aca181e184f26d2a0eef`**
的**干净 tracked tree**（`git status --porcelain` 无 tracked 漂移）上生成；
profiling 脚本对该纪律 **fail-loud**（tracked tree 漂移即 exit 3 拒绝生成）。
复现顺序：

```bash
# （0）绑定 exact SHA：checkout 工具 commit，确认 tracked tree 干净
git checkout 0a952879da6e6cbb26a4aca181e184f26d2a0eef
git status --porcelain    # 必须无 tracked 漂移（untracked 的旧证据输出可忽略）

# （1）build + 正确性
cmake --build build -j8
cd build && ctest --output-on-failure        # 60/60
bash scripts/check_no_torch.sh               # CLEAN

# （2）baseline timing（--mode both，1 warmup + 5 measured）
bash scripts/profile_v07_baseline.sh timing          # → benchmarks/v07_baseline_timing.txt

# （3）nsys：mixed（comparison-harness）+ batched-only（serving 主依据）
bash scripts/profile_v07_baseline.sh nsys            # → v07_nsys.nsys-rep + summaries
bash scripts/profile_v07_baseline.sh nsys_batched    # → v07_batched_nsys.nsys-rep + summaries
bash scripts/profile_v07_baseline.sh aggregate       # → v07_kernel_families.txt（mixed）
bash scripts/profile_v07_baseline.sh aggregate_batched  # → v07_batched_kernel_families.txt

# （4）ncu（8 个 batch-aware targets，--mode batched，缩短 workload，每 target 12 launches）
for t in int4_b1 int4_batch int4_batch_late lmhead_b1 lmhead_batch deltanet deltanet_batch rmsnorm; do
  bash scripts/profile_v07_baseline.sh ncu $t
done

# （5）CUDALab（只读，profiling 完成后）
git clone git@github.com:trump253/CUDALab.git /root/code/CUDALab
git -C /root/code/CUDALab checkout cb6a6a9ef76394cc66d272c99aa8697db0a34f1e
```

所有 evidence 文件（`benchmarks/v07_baseline_timing.txt`、
`benchmarks/profiling/*`）头部均记录：**exact git SHA（= V07A_EVIDENCE_SHA）**、
tracked-tree 状态、GPU、driver、CUDA、checkpoint、workload、warmup/measured
runs、date。

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
**profiling 工具**（V07A_FUNCTIONAL_SHA `0a952879…`）：

- `benchmarks/bench_qwen35_continuous_batching.cpp`：
  (a) 新增 `--measured-runs N` CLI（默认 3，v0.6 行为不变）；(b) 报告新增
  mean/median/min/max、`successful_single_forward_calls`、
  `model_traversal_calls`（serial mode）与 `profile_totals` 段（全程
  completed traversals，作为 nsys launch 统计分母）；
  (c) **新增 `--mode both|serial|batched` 选择器（默认 both = v0.6 行为
  逐字节不变）**：`--mode batched` 只执行 scheduler continuous-batched
  workload（同 checkpoint / requests / dynamic arrivals / warmup+measured
  方法学 / 单 stream / 串行 prefill / 真 batched decode），用于隔离
  serving-path profile。**只改「哪些 mode 执行」，不改任何 kernel/runtime
  行为**（每个 timed run 的 execution 与 v0.6 逐字节相同；`--mode both`
  下 per-run 计数与 v0.6 benchmark 完全一致：serial 27/0、batched 19/3）。
- 新增：`scripts/profile_v07_baseline.sh`、`scripts/aggregate_kernel_families.py`
  （profiling 编排 + nsys CSV 聚合；纯 host 工具，不触 GPU 执行路径）。
  profiling 脚本含 **exact-SHA fail-loud 检查**（tracked tree 漂移 → exit 3）
  与 8 个 batch-aware NCU target（`^` 锚定区分 B=1 / `batch_*`）。
- 新增 evidence：`benchmarks/v07_baseline_timing.txt`、
  `benchmarks/profiling/*`（mixed + batched-only nsys rep/summaries/csv、
  8 份 batch-aware ncu 输出、mixed + batched family 表）—— 全部在
  V07A_EVIDENCE_SHA 的干净 tree 上生成，由最终 docs/evidence-only commit
  纳入（该 commit 的 diff 仅限 `docs/` + `benchmarks/profiling/` +
  benchmark evidence 文本）。

## 15. STOP CONDITION 自检

- v0.6 已 merge main（merge tree == frozen v0.6 tree）✓
- v0.7-profile-opt 从 merged main 创建并 push ✓
- **exact-SHA 证据绑定**：V07A_EVIDENCE_SHA = V07A_FUNCTIONAL_SHA =
  `0a952879…`；全部最终证据在该 exact SHA 的干净 tracked tree 上重新生成，
  每份证据头部记录 SHA + tree 状态 + GPU + driver + CUDA + checkpoint +
  workload + mode + date；第一轮 dirty-tree 证据已判定无效并删除；
  profiling 脚本对 tracked tree 漂移 fail-loud ✓
- baseline correctness（at exact SHA）：`cmake --build build -j8` 成功、
  `ctest --output-on-failure` **60/60、0 failed、0 skipped**、
  `bash scripts/check_no_torch.sh` **CLEAN** ✓
- canonical timing baseline 记录（`--mode both`，1 warmup + 5 measured runs，
  含 exact SHA/GPU/CUDA/checkpoint/workload/warmup）✓
- Nsys **batched-only serving profile**（Phase B 主依据）：kernel / CUDA API /
  memops summary + kernel-family ranking + launch-overhead 结论（mixed，附
  数据）；mixed profile 保留并明确标注为 comparison-harness ✓
- NCU **batch-aware**（8 targets，B=1 vs B=2/3）：evidence-backed 分类
  （memory-bound roofline / mixed / latency-dependency+occupancy /
  launch-overhead）+ **batch-size 效应显式回答**（W4A16 显著 per-token 收益；
  LM head 无 per-token 效应；DeltaNet per-token 收益但分类不变）✓
- CUDALab：仅在 profiling 后检查（frozen SHA 只读）；映射记录并更新
  （P1 显式覆盖 batch 变体；LM head 批处理无收益）；**未 port 任何优化** ✓
- ranked Phase B backlog（P1~P5，serving 口径重新验证）+ 明确推荐首目标
  （P1 W4A16 B=1+batch + 理由）✓
- `V07A_EVIDENCE_SHA` 记录于本文件与 commit（见 git log）；branch 已 push ✓
- **未开始 Phase B** ✓
