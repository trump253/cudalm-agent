# CUDALM v1.0 — 性能与可复现性

**面向**：GitHub 读者与面试官。本文是 [Performance (English)](v10_performance.md)
的中文 companion：当前 release benchmark、可复现的 benchmark 工作流、
以及 v0.7 优化案例（profile-first、KEEP/REJECT 纪律）。本文刻意保持
简短——深度细节在引用的文档里。

**证据基础**：§3 的全部内容绑定 exact commit
`eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a`（分支
`v1.0-portfolio-release`，tracked tree 干净；benchmark executable
在该 SHA 上经 CMake build 验证，binary sha256 记录在
`benchmarks/v10/environment.txt`），raw 报告在 `benchmarks/v10/`。
本文所有数字均为 **在当前硬件 / 当前 checkpoint / 当前 canonical
workload 上的测量**，不是任何通用性能声明。

---

## 1. Executive Summary

CUDALM 是一个 PyTorch-free 的 native C++17/CUDA 推理 runtime，在本机
（单张 RTX 2080 Ti，CUDA 11.8）上推理 Qwen3.5-0.8B-Base。v1.0 release
状态：

- runtime 执行 **真 batched decode**：continuous batching 调度器把
  decode cohort 提交到真 batched GPU path（`forward_batch_with_state`），
  不是 per-request 模拟。canonical 4-request workload（27 logical
  token-forwards）上，调度器用 **22 次 model traversal 完成 serial
  模式 27 次的工作**（3 个 committed batch 覆盖 8 个 batched token；
  avg decode batch size 2.67，max 3）。
- 在当前硬件 / checkpoint / workload 上测得：
  independent/serial **0.1327 s mean**（203.5 logical tok/s）vs
  continuous-batched **0.1161 s mean**（232.6 logical tok/s）——
  27-token workload 上 **−16.6 ms（−12.5%）wall-time delta**。
  batched-only serving profile（进程内不与 serial 交替）测得
  **0.1100 s mean（245.4 logical tok/s）**。
- 项目的优化史是 **profile-driven、证据 gated** 的：v0.7 先 profile
  frozen runtime，再按 pre-specified KEEP/REJECT 判据评估候选。
  一个候选被 KEEP（fused residual-add + RMSNorm：**−24 kernel
  launches/traversal**，paired E2E **−1.238 ms/run**，95% CI
  [−2.437, −0.040] 排除 0）；两个被 REJECT（W4A16 GEMV 与 DeltaNet
  delta-rule 重写）——包括一个 kernel 隔离明显更快但 E2E wall-clock
  gate 未过的候选。
- benchmark 工作流本身可复现：`scripts/benchmark_v10_release.sh`
  在 dirty tracked tree 上拒绝运行，记录 SHA / GPU / driver / CUDA /
  compiler / build type / binary hash / model + checkpoint 身份，把
  raw 报告 + 机器可读 summary 写入 `benchmarks/v10/`。

本文**不**声明：与 vLLM 或 llama.cpp 的比较、production 并发 QPS、
Tensor Core 使用、"fully optimized kernels"。见 §7。

## 2. Benchmark Environment

| 项 | 值 |
|---|---|
| GPU (device 0) | NVIDIA GeForce RTX 2080 Ti (11264 MiB, sm_75) |
| NVIDIA driver | 570.172.08 (CUDA 12.8 driver) |
| CUDA toolkit / runtime | 11.8 (V11.8.89) |
| 主机编译器 | gcc 9.4.0 |
| CMake build type | Release |
| 模型 | Qwen/Qwen3.5-0.8B-Base（`/root/models/Qwen3.5-0.8B-Base`），raw text，**无 chat template** |
| 转换后权重 | `build/data/qwen35_08b_full.cudalm`（gitignored 构建产物；sha256 记录在 `benchmarks/v10/environment.txt`） |
| Evidence SHA | `eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a`（tracked tree 干净；benchmark binary 在该 SHA 上 CMake build，sha256 记录在 `environment.txt`） |

完整环境记录（日期、benchmark 命令、build command、binary sha256、
model/checkpoint sha256）在 `benchmarks/v10/environment.txt`。

## 3. 当前 v1.0 Release Benchmark

### 3.1 Workload 定义（canonical，自 v0.6 未变）

四个 request——A greedy，B/C/D seeded——prompt 长度
**2 / 5 / 3 / 4** tokens，生成长度 **3 / 6 / 5 / 3**，以
**动态到达**方式 admit（A,B 先；C 在 1 步后；D 在 2 步后）。
合计：14 prompt + 17 generated = **27 logical
sequence-token forwards**（Σ(N+m−1) = 4+10+7+6）。两种模式做
**完全相同的总工作量**（相同 prompts / max_new / seeds；四个 request
全部完成；计时 run 内无取消），比较是同质的（apples-to-apples）。

两种执行模式，单 CUDA stream，prefill 串行：

- **mode A — independent / serial**：每个 request 单独跑（全新 state、
  直接单 token forward），逐个相加；
- **mode B — scheduler continuous batched**：request 动态 admit 进
  调度器；其 decode cohort 走**真 batched GPU path**
  （`forward_batch_with_state`）。

计时纪律：2 次不计时 warmup，随后每模式 **10 次 measured run**，每次
用 `cudaStreamSynchronize` 包围（wall time 包含全部已入队 GPU 工作；
run 之间无残留），host `steady_clock`。协议：
`--warmup-runs 2 --measured-runs 10`。

### 3.2 结果（measured on this hardware / checkpoint / workload）

来自 `benchmarks/v10/release_serial_batched.txt`（`--mode both`）：

| 指标 | mode A — serial | mode B — continuous batched |
|---|---|---|
| wall time mean | **0.132707 s** | **0.116064 s** |
| wall time median | 0.131392 s | 0.115457 s |
| wall time min / max | 0.129797 / 0.138380 s | 0.114167 / 0.120369 s |
| logical tok/s (mean) | 203.46 | 232.63 |
| single forward calls | 27 | 19 |
| batch forward calls | 0 | 3 |
| completed model traversals | 27 | 22 |
| avg / max decode batch size | 0 / 0 | 2.67 / 3 |

来自 `benchmarks/v10/release_batched.txt`（仅 `--mode batched`——独立
serving-path profile，同协议）：

| 指标 | mode B — continuous batched（batched-only 进程） |
|---|---|
| wall time mean / median | **0.110017 s / 0.109119 s** |
| wall time min / max | 0.107920 / 0.113852 s |
| logical tok/s (mean) | 245.42 |
| single / batch forward calls | 19 / 3 |
| completed model traversals | 22 |
| avg / max decode batch size | 2.67 / 3 |

**观测 delta（batched vs serial，`--mode both` 同次实验）：**
27-token workload 上 wall time **−16.64 ms（−12.5%）**；logical
吞吐 **+29.2 tok/s**。per-run wall time 在 raw 报告中
（run-to-run min–max spread ≈5–7%——本机桌面机正常水平）。
这个 delta 是 **本硬件 / 本 checkpoint / 本 canonical workload**
的性质，不是"CUDALM 比 serial 快 X%"的泛化声明。

**与 frozen v0.7 sign-off 的历史对比（仅描述性）**。历史 v0.7
测量（`docs/v07_final_performance.md` §3.1，同机同 checkpoint，
v0.7 协议 1 warmup + 5 measured）：serial 0.130968 s，batched
0.117271 s。v1.0 点估计相对这些历史值约 **+1.3% serial**
（0.132707 s）与 **−1.0% batched**（0.116064 s）。两次测量在
不同 run、不同 warmup/样本数、不同时间收集，**不是 paired A/B
实验**（无配对、无交替）。因此该对比仅描述性；**不从中得出任何
regression 或 improvement 结论**。

完整解析在 `benchmarks/v10/summary.json`（convenience view——raw
报告才是证据）。

## 4. Continuous Batching 说明了什么

canonical workload 上，batching 省了 **5 次 model traversal
（27 → 22）**：动态到达之后，调度器每次 forward 提交 2–3 条
sequence 的 decode cohort（3 个 committed batch 覆盖 8 个 batched
token；其余 19 次 forward 是单 sequence——request 长度错开，cohort
随 request 完成而缩小）。每个 committed batch of B 是 **B 个
logical token、1 次 traversal**——这个算术直接可见于报告
（`single_forward_calls 19 + batch_forward_calls 3 = 22` 次
traversal 对应 27 个 logical token），就是 §3.2 wall delta 的机制。

它精确地说明了：

- serving path 是**真 batched decode**——batched GEMV / DeltaNet /
  attention kernel 在 GPU 上执行多 sequence cohort（这些 kernel 是
  correctness-first、未调优的——见 §7）；
- 调度器的 **continuous** 行为（流中到达、逐步组 cohort、
  commit-before-visible）正是把工作量分组的东西；workload 的动态
  到达正是对它的有效验证；
- 它**不**说明 HTTP frontend 的并发（该端点 single-threaded、一次
  一个 request——§7），也**不**外推到大批次或长序列形态。

## 5. v0.7 优化案例（frozen 历史）

优化历史作为**方法论案例**保留，v1.0 不重跑：先 profile，
pre-specify gate，测量候选，然后按证据 KEEP 或 REJECT。完整
provenance：`docs/v07_profiling.md`、`docs/v07_w4a16_optimization.md`、
`docs/v07_deltanet_optimization.md`、`docs/v07_final_performance.md`，
raw 证据在 `benchmarks/` 与 `benchmarks/profiling/`。

### 5.1 Profile-first 方法论

v0.7 Phase A **没有改** runtime 任何东西。它在 frozen v0.6
continuous-batching runtime 上建立 Nsight Systems / Nsight Compute
baseline（exact-SHA 纪律：每份证据绑定 check-out-clean SHA；profiling
脚本对 dirty tracked tree **fail loud**——早期 dirty-tree 生成的证据
被判无效并全部删除）。profile 定位了主要成本结构（GEMV 家族 ≈ 80%
GPU 时间；**442.6 kernel launches/traversal**——重度小 kernel
碎片化）并确定优化对象。baseline 不存在之前不开始任何优化。

### 5.2 W4A16 GEMV 重写 — REJECTED

候选：用 W4A16（4-bit 权重、16-bit 激活）GEMV 替换冻结 bf16 GEMV，
跨 row-tile 变体（R1/R2/R8）。结果：**全部候选 REJECTED for
production**——R1 在 6/8 生产 shape 上更差（+8.73% / +11.34%），
R8 在每个 shape 上更差（+14.88%，register pressure），R2 ≈ 中性。
第二组实测表在两个 shape 上有 **−4.91% / −3.61% 的 microbench
边缘**，但 pooled A/B 是 **p = 0.284——E2E 改进在 noise 内**。
baseline bf16 GEMV 保留。这就是纪律：kernel 隔离的 microbench
优势如果过不了 E2E gate，就是 REJECT。

### 5.3 DeltaNet delta-rule 重写 — REJECTED

候选：重写 DeltaNet delta-rule 更新（vreg/vvec/vchunk launcher），
带 **EXACT bit-exact parity**（对冻结序列的硬门禁）。候选在隔离
状态下确实更快——microbenchmark **−22…−24%**、NCU kernel 时长
**−14…−26%**——但 pre-specified 的 paired A/B E2E gate（交替顺序、
每次 invocation 全新进程）的 **95% CI 包含 0**。结论：**REJECTED
for production；frozen baseline 保留**（最终
`src/runtime/qwen35_deltanet.cpp` 与候选前文件 byte-identical）。
拒绝由 **wall-clock 证据而非 kernel 证据**驱动——一个更快的 kernel
不是自动地一个更快的程序（launch 结构、内存流量、以及 442-launch
traversal 的其余部分占主导）。

### 5.4 Fused residual-add + zero-centered RMSNorm — KEPT

候选：把 2-launch 的 post-attention 残差序列（`add` → `rmsnorm_zc`）
替换为 1 个 kernel `qwen35_fused_add_rmsnorm_zc_bf16`，覆盖全部 4 个
post-attention 位点（DeltaNet B=1 + batch、Full-Attention B=1 +
batch）。**构造即 bit-exact**（bf16-rounded residual 原样复用；冻结
reduction tree 与 op 顺序），并带 `memcmp` 硬门禁。

证据（`docs/v07_final_performance.md` §2，绑定 exact SHAs）：

- Nsight Systems A/B：总 launches 58422 → 55254 = **−24.0
  launches/traversal——恰好是预测的 −1 launch/layer × 24 层**；
  host launch 时间 −125 µs/traversal；总 GPU kernel 时间在 run
  noise 内持平；add+rmsnorm 家族自身 −6.19 ms。无重要 kernel
  regression。
- Paired A/B E2E（20 个 fixed pairs、交替、全新进程、2 warmup +
  10 measured、canonical workload、batched mode）：**paired mean
  Δ = −1.238 ms/run**（SD 2.562，SE 0.573）；median −1.067；
  t = −2.162，df = 19，p = 0.0436；**95% CI [−2.437, −0.040] 排除
  0**；15/20 pair 更快。Pre-specified KEEP 判据（mean < 0 且
  median < 0 且 CI 排除 0）**MET → KEEP 进 production**。

### 5.5 经验（诚实表述）

- **先 profile 再优化。** launch 碎片化这个优化目标来自测量，不是
  直觉。
- **以 E2E wall time 为 gate，不以 kernel time。** 三个候选中两个
  有真实的 kernel 隔离收益，都因 E2E gate 未过被 REJECT。
- **Bit-exact 是硬门禁，不是加分项。** 每个候选（包括被 KEEP 的）
  在任何计时之前都有 EXACT parity 测试。
- REJECTED 候选的 microbench speedup **不是** production speedup——
  本仓库任何地方都不把它当作 production speedup 报告。

## 6. 复现（Reproduction）

```bash
# 1. 在 evidence SHA（任何 check-out-clean tree 都可以）：
git checkout v1.0-portfolio-release
git checkout eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a   # the evidence SHA

# 2. configure + build（现有或全新 build dir）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 --target bench_qwen35_continuous_batching

# 3. release benchmark（tracked tree dirty 时 FAIL LOUD；
#    记录 SHA/环境，warmup=2 measured=10 两遍，
#    写 benchmarks/v10/*，自检 SHA 一致性）
./scripts/benchmark_v10_release.sh

# 手动等价于 pass 1（脚本会跑两遍）：
build/benchmarks/bench_qwen35_continuous_batching \
    build/data/qwen35_08b_full.cudalm /root/models/Qwen3.5-0.8B-Base \
    <python3> $(pwd) benchmarks/v10/release_serial_batched.txt \
    --no-convert --measured-runs 10 --warmup-runs 2 --mode both
```

前提：checkpoint 在 `/root/models/Qwen3.5-0.8B-Base`（或设
`CUDALM_CKPT`）、转换后的 `build/data/qwen35_08b_full.cudalm`
（`tools/convert_qwen35.py --full-model`，或让 binary 自行转换）、
以及一张 RTX 2080 Ti 级 GPU。重跑产生的是**你的**环境的**新**数字
——比较结构（traversals、batch 数），不要期待 wall time 逐位相同。
v0.7 证据的复现命令在各 v0.7 文档中。

## 7. Scope / Limitations

- **数字是本地性质。** 本文每个数字都测自一张 RTX 2080 Ti、一个
  checkpoint、一个 27-token canonical workload、CUDA 11.8。它们不可
  移植，也没有任何"比 X 快"的泛化声明。本仓库不存在 vLLM /
  llama.cpp / 任何外部系统的比较；公平比较需要专门的同机同模型
  campaign。
- **batch kernel 是 correctness-first，未调优。** v0.7 明确了这点并
  把调优推迟；v1.0 不做任何性能改动。"fully optimized kernels" /
  "Tensor Core optimized" 都会是错误表述。
- **没有 HTTP 性能数字。** HTTP frontend（`cudalm-server`）是
  single-threaded、一次一个 request、单 CUDA stream。并发 QPS /
  多客户端吞吐 / HTTP continuous-batching 吞吐对它都不成立，本
  phase 刻意不报告任何一项。（其正确性合同由 v0.9/v1.0 的 CPU +
  e2e gate 覆盖，不由性能声明覆盖。）
- **小 workload 统计。** 27 token 是 micro-workload；per-run wall
  spread（≈5–7%）与 batched-vs-serial delta 同量级。v0.7 的
  paired A/B（20 个 fresh-process pairs）是本项目唯一被 KEEP 的
  优化的统计绑定证据——本 release benchmark 是复现性 baseline，
  不是统计实验。
- **v1.0 Phase B 没做的事**：无新 NCU / Nsys campaign、无 kernel
  调优、无候选评估、不重跑 v0.7 A/B（frozen history）、无 HTTP path
  benchmark、无 `src/`、`include/` 或任何 CUDA 代码改动。
