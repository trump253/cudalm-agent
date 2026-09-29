# CUDALM v1.0 Portfolio & Release — Release Notes / 工作日志

**基线**：`main` @ `b429c8f51eaf3f6e58166f08de630b17a88ba524`
（= v0.9 merge commit，tag `v0.9`）。分支
`v1.0-portfolio-release`。

**v1.0 原则**：不新增 inference feature，不做 CUDA/kernel 优化，
不重构 frozen runtime——只做 release-quality cleanup，把 codebase
从“开发完成”整理成“可以公开展示和发布”的状态。

**阶段规划**：Phase A: Release Hygiene（本文件当前范围）；
Phase B: benchmark 整理；Phase C: README 重写；Phase D: release
gate（full ctest / sanitizer / CI）。各阶段的范围以 external
review 任务为准。

---

## v1.0 Release Summary

v1.0 是 CUDALM 的 portfolio / release 状态（分支
`v1.0-portfolio-release`，详见下方各 Phase 记录与
`docs/v10_release_validation.md`）：

- Native C++17/CUDA Qwen3.5-0.8B runtime（PyTorch-free production
  runtime；W4A16 projection + BF16 激活）
- 24 层混合模型：Full Attention（partial RoPE / GQA / zero-centered
  RMSNorm）+ Gated DeltaNet（conv + delta-rule 递推）
- 原生 tokenizer（CUDLMTK1）+ greedy / temperature / top-k / top-p
  采样（v0.4 冻结合同）
- Paged KV（Full Attention）+ Delta conv/recurrent state（DeltaNet），
  session-bound 生命周期
- True batched decode（`forward_batch_with_state`）+ continuous
  batching（动态到达、逐步组 cohort）
- 持久化多轮 Session（persistent multi-turn raw-text completion）
- Committed-token streaming（commit-before-visible）+ cancel /
  deadline
- Admission / backpressure（zero-mutation 拒绝）+ TTL / LRU 驱逐
- 最小原生 HTTP serving（`cudalm-server`，thin JSON/NDJSON 合同）
- Profile-driven CUDA 优化（v0.7 KEEP/REJECT 证据纪律：fused
  add+rmsnorm KEEP，W4A16 / DeltaNet 候选 REJECT）
- 可复现性能证据（Phase B，exact-SHA 绑定：
  `benchmarks/v10/` + `docs/v10_performance.md`）

Limitations（如实记录，不改变）：仅 Qwen3.5-0.8B-Base；单 GPU；单
CUDA stream；prefill 串行；HTTP frontend single-threaded（one request
at a time）；raw-text completion（无官方 Qwen chat template、非
OpenAI-compatible）；无 TP / multi-GPU / speculative decoding / CUDA
Graph / distributed serving。

Release 工件：MIT `LICENSE`；lightweight hosted CI
（`.github/workflows/ci.yml` → repository-checks，只做
repository/static guards，不做 hosted CUDA build）；release
validation evidence（`docs/v10_release_validation.md`）。

---

## Phase A：Release Hygiene

### A.1 清理 cudalm-server 临时 debug 输出

`tools/cudalm_server.cpp` 的 accept loop 中残留的 6 处开发期
`[dbg] ...` 逐请求 stderr 日志（accepted / transport-read-failure /
parsed / handler-start / handle-done / sent）**全部删除**。保留：
启动配置信息（model 维度、raw-text 合同声明、API 声明、policy
摘要）、fatal error（CUDA / model / tokenizer / listen 失败）、
usage 输出。正常运行不再逐请求打日志。

> 注：这 6 行是 v0.9 Phase D 开发期加入、当时清除未完全落到
> functional commit 的遗漏——v0.9 已 FROZEN（不改历史），在 v1.0
> Phase A 按 release hygiene 清理。

### A.2 收敛 `destroy_for_test` test seam

Phase C 的 `ServingController::destroy_for_test`
（`std::function<Status(SessionId)>`）原本是 **public** 成员——
production caller 可以随意替换 Session destroy 语义。现收敛为：

- 成员移入 **private** 区；
- 通过 **test-only friend**（`cudalm::ServingControllerTestSeams`，
  **只在** `tests/cpu/test_serving_eviction.cpp` 中定义）可达——
  没有任何 production 翻译单元定义该 struct，因此没有任何
  production 代码能安装 hook；
- 现有 eviction fault-injection test 通过
  `ServingControllerTestSeams::install/clear` 正常工作
  （`test_serving_eviction` PASS，含全部 fail-loud case）；
- 未重构 SessionManager；hook 为空时行为与 v0.9 完全一致（frozen
  `SessionManager::destroy_session`）。

**public API 变化**：`ServingController` 移除一个 public 成员
（`destroy_for_test`）；新增一个 friend 声明（production 可见但
不可用——名字是唯一的 production 可见痕迹）。这是 Phase A 唯一的
public API 面变化（移除泄漏，非新增 API）。

### A.3 HTTP Content-Type contract（turn / turn/stream）

明确 turn body 的媒体类型合同（sync + stream 两个端点一致；
415 发生在 admission / forward **之前**）：

**接受**（body 一律是 opaque raw UTF-8 bytes，媒体类型不改变
处理）：

```text
Content-Type: text/plain                 （含任意参数，如
                                          ; charset=utf-8；
                                          大小写不敏感）
缺失 Content-Type                        （文档化的兼容策略：
                                          按 text/plain 处理——
                                          快速客户端 /
                                          不带 CT 的 python
                                          http.client）
Content-Type: application/x-www-form-urlencoded
                                     （curl --data 默认——与
                                          现有文档 curl 示例
                                          兼容；body 仍是原始
                                          bytes）
Content-Type: application/octet-stream
```

**拒绝（415，稳定 JSON error body）**：其余全部——尤其
`application/json`（JSON prompt body 不是合同的一部分——body
是 raw text）与 `multipart/*`（不做 multipart 解析）。不做
content negotiation。

真实 server 验证：`text/plain` / `text/plain; charset=utf-8` /
`application/x-www-form-urlencoded` / 缺失 → 200；
`application/json`（sync + stream）→ 415。

### A.4 Release-quality source audit（include/ src/ tools/）

- `TODO` / `FIXME` / `XXX` / `HACK` / `[dbg]`：**无残留**（清理
  后全库扫描为空）；
- 临时 `printf`：无（tools/ 中的 printf 均为 intentional 工具
  输出——selftest / 转换报告 / 诊断工具 / CLI banner）；
- 开发期 Python 字节码缓存（`__pycache__` / `*.pyc`）：未被 git
  跟踪（`.gitignore` 已覆盖 `__pycache__/` / `*.pyc`）——无需
  清理；
- 编译 warning：全量构建 `-Wall -Wextra -Werror` **零 warning**；
- stale 注释：修正 `http_serving_handler.h` 的 status contract
  注释（补 415 Content-Type 行）；`parse_turn_query` 改名后无
  代码残留引用（provenance 中的提及是历史记录，准确）；
- `tools/` 中的诊断/性能工具（`dump_*` / `ncu_*` /
  `microbench_*`）为 intentional 工具集（v0.7 profiling 文档
  引用），**保留**，不属临时痕迹。

### A.5 Public API / naming audit（只记录，不扩大）

- `ModelConfig::v011_general_test()`（`include/cudalm/model_config.
  h`）：名字含 `_test` 的 public static fixture（v0.1.1 广义
  shape 测试配置），被 `tests/cpu/test_weight_file.cpp` 与
  `tools/convert_weights.py`（Python 镜像 twin + selftest 模式）
  跨语言成对使用。**评审结论：保留**——改名是 cosmetic、跨 C++/
  Python/测试的多文件 diff，且工具侧有真实的（selftest 模式）
  用途，不是纯 test-only 泄漏；记录于此供后续决定。
- 其余 public header 无 test-only API 泄漏（`destroy_for_test`
  已在 A.2 收敛）、无错误 contract 注释。

### Phase A 明确未改变的 runtime semantics

- model math / CUDA kernel / quantization / weight format：**零
  改动**；
- v0.8 session runtime、v0.9 Phase A quota / Phase B
  commit-before-visible / Phase C eviction 语义：**零改动**
  （`destroy_for_test` 收敛只改可见性，hook 为空时行为逐字节
  一致）；
- sampling math / v0.4 sampling 合同、deadline / TTL / LRU
  语义：**零改动**；
- HTTP transport（socket / 单线程 / `Connection: close` /
  Content-Length only / Transfer-Encoding → 415）、streaming
  NDJSON 合同、disconnect 合同：**零改动**（A.3 只在 handler
  层新增 turn-body Content-Type 门）。

### 测试

- `test_serving_http` 新增 `test_content_type_contract`（接受集
  全收 / 拒绝集全 415 / stream 端点同合同 / 415 前 nothing
  admitted）——全部 7 个 [PASS]；
- `test_serving_eviction`（fault-injection 经新 friend seam）→
  **PASS**（含全部 fail-loud case）；
- targeted `ctest -R "http|server|evict|serving|stream|
  scheduler|session" --output-on-failure` → **24/24 PASS**
  （30.49 s；含 3 个真实 checkpoint gate——缺失 CT 兼容策略在
  真实 python http.client 下验证通过）；
- 真实 server CT 边界 spot-check：见 A.3。
- 按 Phase A 任务边界：不跑 profiling / benchmark / full ctest
  （full release gate 留到 Phase D）。

## Phase B：Benchmark & Reproducibility

### B.1 Release benchmark（current v1.0 tree，重新测量）

- 工具：复用 `benchmarks/bench_qwen35_continuous_batching.cpp`
  （canonical workload 不变：4 requests，prompts 2/5/3/4，generated
  3/6/5/3，27 logical token-forwards，动态 arrival，单 CUDA
  stream，prefill serial，decode 真 batch）；
- release protocol：`--warmup-runs 2 --measured-runs 10`，两遍：
  `--mode both` → `benchmarks/v10/release_serial_batched.txt`；
  `--mode batched`（独立 serving profile）→
  `benchmarks/v10/release_batched.txt`；
- 结果（measured on this hardware / checkpoint / canonical
  workload；RTX 2080 Ti，CUDA 11.8，driver 570.172.08，gcc 9.4，
  Release build；evidence SHA `eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a`）：

  | metric | serial | continuous batched |
  |---|---|---|
  | wall mean / median | 0.132707 / 0.131392 s | 0.116064 / 0.115457 s |
  | wall min / max | 0.129797 / 0.138380 s | 0.114167 / 0.120369 s |
  | logical tok/s (mean) | 203.46 | 232.63 |
  | single / batch forward calls | 27 / 0 | 19 / 3 |
  | model traversals | 27 | 22 |
  | avg / max decode batch | 0 / 0 | 2.67 / 3 |

  batched-only serving profile：mean 0.110017 s（245.42 tok/s）。
  observed delta（both 遍，同次实验比较）：wall **−16.64 ms
  （−12.5%）**，+29.2 tok/s——仅本硬件/本 checkpoint/本 workload
  的测量，不是泛化 speedup 声明。
- 与历史 v0.7 sign-off 的跨版本对比（**descriptive only**，下同）：
  v0.7（1 warmup + 5 measured，`docs/v07_final_performance.md`
  §3.1）serial 0.130968 s、batched 0.117271 s；v1.0 点估计约
  **+1.3% serial**（0.132707 s）、**−1.0% batched**（0.116064 s）。
  两次测量是不同 run、不同 warmup/样本数、不同时间、非 paired
  A/B、非交替顺序——因此只做描述性对比，**不从中得出任何
  regression / improvement 结论**（review fix：删除了原先的
  "no regression" 因果结论）。

### B.2 exact-SHA benchmark workflow

`scripts/benchmark_v10_release.sh`（最终 tooling commit
`eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a`，含 review fix 1 的
binary provenance 修复）：

- tracked tree dirty → **fail loud**（不生成正式 evidence）；
- **binary provenance 链**（review fix 1：executed binary 必须 ==
  在记录的 exact source SHA 上 CMake build 的 binary）：
  clean tree → 记录 exact HEAD SHA → `cmake --build build
  --target bench_qwen35_continuous_batching -j8`（build /
  up-to-date validation，build 失败 fail loud；不要求 clean
  rebuild）→ **build 后再验 HEAD 未变** → 计算 binary sha256 →
  运行 benchmark → **benchmark 后再验 HEAD 未变**；
- 记录：git SHA / tree cleanliness / date / GPU / nvidia driver /
  CUDA toolkit / host compiler / CMake build type / **build
  command** / benchmark command / model + checkpoint identity（含
  sha256）/ **benchmark_binary_sha256**（build 后计算）→
  `benchmarks/v10/environment.txt`；
- 运行两遍 benchmark → raw reports（**永远保留**）；
- Python stdlib 解析 raw reports → `benchmarks/v10/summary.json`
  （sha / hardware / workload / serial / batched metrics /
  delta）——summary 只是 convenience view，不是唯一证据；
- 自检：HEAD（build 前后 + benchmark 后）== environment.txt SHA ==
  summary.json SHA；记录的 binary sha256 == 实际执行 binary 的
  sha256（run 后复验未漂移）；
- 大模型文件留在 `build/data`（gitignored，**不 commit**）。

v1.0 evidence 归入 `benchmarks/v10/`（不再平铺到 benchmarks/
根目录）；v0.6 / v0.7 的 evidence 文件原位不动（frozen history）。

### B.3 性能文档 + v0.7 case study

新增 `docs/v10_performance.md`（≈2300 词，面向 GitHub 用户 /
面试官）：Executive Summary / Environment / 当前 v1.0 release
benchmark（serial + continuous batched + workload 定义）/ What
Continuous Batching Demonstrates / **v0.7 优化 case study**
（profile-first 方法论；W4A16 **REJECT**——microbench 有 −4.91%
边缘但 pooled E2E p=0.284 在 noise 内；DeltaNet **REJECT**——
kernel 隔离 −22…−24% 但 paired E2E 95% CI 含 0，以 wall-clock
为准拒绝；fused add+rmsnorm **KEEP**——−24 launches/traversal、
paired E2E −1.238 ms、95% CI [−2.437, −0.040] 排除 0）/
Reproduction / Scope & Limitations。

诚实边界（文档中明确）：无 vLLM/llama.cpp 对比、无并发 QPS、
无 "Tensor Core optimized" / "fully optimized kernels" 声明；
HTTP frontend 是 single-threaded one-request-at-a-time——本 Phase
**不做 HTTP 性能 benchmark**（无法稳定定义 methodology，宁可不加）；
rejected candidate 的 microbench speedup 绝不当 production
speedup 写。

### Phase B 明确未做的事

- NO new NCU / NO new Nsys campaign / NO kernel tuning / NO
  candidate evaluation / 不重跑 v0.7 A/B（frozen history）；
- `src/`、`include/`、任何 CUDA 代码：**零改动**（Phase B 只加
  scripts/ + benchmarks/v10/ + docs/）；
- README 只字未改（完整重写属 Phase C）。

### Phase B 测试 / validation

- benchmark executable build PASS（`cmake --build build --target
  bench_qwen35_continuous_batching`，在记录的 exact SHA 上执行
  build / up-to-date validation；build 前后 HEAD 复验未变）；
- release benchmark 完整运行 PASS（两遍 + self-consistency OK，
  evidence 绑定 tooling SHA `eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a`；
  benchmark 后 HEAD 复验未变）；
- binary provenance：executed binary 的 sha256
  `c908f5a79438f2521ffa2f58968358939cc72c0f8333443cee2ac3e56f56f54a`
  记录于 environment.txt 并在 run 后复验未漂移；
- `bash scripts/check_no_torch.sh` → **CLEAN**（include/ + src/ 零
  改动，guard 复验）；
- evidence 自洽：SHA（HEAD build 前后/benchmark 后 == environment.
  txt == summary.json）/ binary sha256 / model + checkpoint sha256
  同一记录。
- 按任务边界：无 C++ runtime 改动 → 不需要 full ctest /
  compute-sanitizer；不跑 profiling。

---

## Phase C：Bilingual Portfolio README & Architecture Documentation

### C.1 语言结构

中文是主要 portfolio 展示语言（用户求职以国内 AI Infra / CUDA /
LLM inference 公司为主），英文保留完整 companion version；代码 /
API / 文件名 / CUDA/C++ identifier / benchmark terminology 保持
原始英文。文档布局：

```text
README.md                       中文主 README（彻底重写，非 append）
README_EN.md                    完整英文 companion（同章节结构）
docs/v10_architecture_overview.md        中文系统架构总览
docs/v10_architecture_overview_en.md     英文系统架构总览
docs/v10_performance.md         Phase B 冻结英文性能文档（核心内容未改）
docs/v10_performance_zh.md      中文 companion（数字与 frozen 英文逐一对齐）
```

README 顶部互相提供语言入口；不把完整中英文塞进同一文件。

### C.2 README 重写

旧 README（813 行 / 4148 词）是自 v0.1 起的 append-only 开发日志，
首页从 single decoder block 开始，已不能代表 v1.0——**整体重写**为
portfolio README（13 节：项目简介 / 核心能力 / 系统架构 / 推理引擎
实现 / 快速开始 / 生成·Multi-turn·HTTP 示例 / 性能 / 正确性与工程
验证 / 核心工程设计 / 当前限制 / 仓库结构 / 技术文档 / Roadmap），
历史 changelog 不再由 README 承担（真实来源留在 docs/provenance.md、
docs/v07_*.md、docs/v08_session_runtime.md、docs/v09_serving_hardening.
md、docs/v10_performance.md 等）。中英文两版章节结构相同、技术事实
完全一致（feature support / limitations / 性能数字 / HTTP 行为 /
multi-turn 语义 / scheduler 语义 / future work 逐项对齐）。

### C.3 Quick Start 与 HTTP 示例的核实

README 中所有命令都从实际代码核实（不凭空创造 flag）：

- `cudalm-generate` / `cudalm-chat` / `cudalm-server` 的 flag 逐一
  对 `--help` 输出与 source 核对（含 --slots / --pages / --page-
  tokens / --max-sessions / --max-live-requests / --session-ttl-ms /
  --lru-on-pressure 等）；
- HTTP 路由与 handler source 核对：`GET /healthz`、`GET /v1/stats`、
  `POST /v1/sessions`、`POST /v1/sessions/<id>/turn`、
  `POST /v1/sessions/<id>/turn/stream`、`POST /v1/sessions/<id>/reset`、
  `DELETE /v1/sessions/<id>`（错误码 / NDJSON 事件格式链接到
  docs/v09_serving_hardening.md）；
- 所有 turn POST 示例显式 `-H 'Content-Type: text/plain'`（Phase A
  合同：不依赖 curl 默认 Content-Type）；
- 转换命令对 `tools/convert_qwen35.py --help` /
  `tools/convert_qwen35_tokenizer.py --help` 核实。

### C.4 架构图与语义文档

- README（中英）新增 Mermaid 系统架构图（Frontend → Tokenizer →
  ServingController → Scheduler → SessionManager → StateManager →
  Paged KV / Delta state → Qwen35Model → CUDA Kernels；Full
  Attention→Paged KV、Gated DeltaNet→conv+recurrent state 正确
  体现；不含 multi-GPU / tensor parallel / CUDA Graph / async pool /
  speculative decoding 等不存在的组件）；
- 架构总览（中英）新增 request lifecycle / streaming 语义
  （admit → prefill → decode → sample → commit → stream-visible →
  terminal → reap 的 Mermaid 图 + 硬不变量 `sampled token !=
  stream-visible token`、`generated[0..committed)` 才可暴露、
  terminal final token 先 emit 后 terminal）与 state ownership 图
  （SessionId → SequenceId → Paged KV + Delta slot）；
- scheduler 诚实边界（底层 continuous batching 已就绪，但 HTTP
  frontend 仍 single-threaded one-request-at-a-time，不能描述为
  concurrent HTTP serving）在 README 与架构总览（中英）中均明确
  写出；
- multi-turn 节明确 **persistent multi-turn raw-text completion**
  语义 + "无官方 Qwen chat template / 非 OpenAI-compatible / 不是
  ChatGPT-style conversation API"。

### C.5 性能叙事与限制

- README 性能节数字全部来自 frozen Phase B evidence（SHA
  `eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a`）：serial
  132.707 ms / 203.46 tok/s / 27 traversals；continuous batched
  116.064 ms / 232.63 tok/s / 22 traversals / avg-max batch 2.67/3；
  batched-only 110.017 ms / 245.42 tok/s；同实验 delta
  −16.64 ms（−12.5%）；附"仅对应当前硬件 / checkpoint / canonical
  workload"声明；
- profile-driven optimization 短叙事（v0.7：profile first →
  candidate → correctness → microbench → profiler → paired E2E →
  KEEP/REJECT；W4A16 REJECT / DeltaNet REJECT / fused KEEP；
  −24 launches/traversal、paired E2E −1.238 ms、95% CI
  [−2.437, −0.040]；"更快的 kernel 不代表更快的程序"）；
- Limitations 中英一致（Qwen3.5-0.8B-Base only / single GPU /
  single stream / serial prefill / HTTP single-threaded /
  raw-text no chat template / not OpenAI-compatible / 无 TP /
  multi-GPU / speculative / CUDA Graph / distributed / 跨重启
  session 存储）；
- 无虚假 badge（Phase D 才有真实 CI）。

### Phase C 明确未做的事

- 不改任何代码（src/ include/ tools/ tests/ 零改动；本 Phase
  docs-only）；不重跑 benchmark；不做新性能分析；不重写
  qwen35_architecture.md / provenance.md / v07_*.md /
  v08_session_runtime.md / v09_serving_hardening.md /
  v10_performance.md 核心内容；不 merge main、不 tag v1.0。

### Phase C validation（docs-only）

- 全部新增 README 相对链接解析通过、引用文件存在、executable 名
  （cudalm-generate / cudalm-chat / cudalm-server）、CLI flag（对
  --help）、HTTP 路由（对 handler source）逐一核实；
- 中英性能数字与 frozen Phase B evidence（summary.json / raw
  报告）一致；中英 limitation / feature / HTTP / multi-turn /
  scheduler / future work 逐项对齐（无"中文支持 X 英文不支持 X"）;
- 无 unsupported claim 扫描（无 production-grade HTTP / 高并发 /
  OpenAI 兼容 / Tensor Core / fully optimized 等表述）；
- 未把历史限制（no tokenizer / no batching / no scheduler / no
  Paged KV / no session / no HTTP）误写为当前限制——这些现在都已
  实现；
- 按 Phase 边界：docs-only → 无 full ctest / compute-sanitizer /
  benchmark rerun。

---

## Phase C External Review Fixes（documentation correctness）

只修文档正确性（docs-only；src/ include/ tests/ benchmarks/
scripts/ 零改动；无 ctest / sanitizer / benchmark rerun）：

- **Corrected frontend/runtime topology**：README（中英）与架构总览
  （中英）的架构图改为三条 user-facing 执行 path——
  (A) `cudalm-generate` → Qwen35Tokenizer → Qwen35TextGenerator →
  Qwen35Generator → Qwen35Model（legacy model-owned state，
  `model.reset_state` / `model.forward_token`；无
  ServingController/Scheduler/SessionManager/StateManager）；
  (B) `cudalm-chat` → Qwen35SessionTextGenerator（owns 自己的
  Scheduler）→ Scheduler → SessionManager → Qwen35StateManager →
  Qwen35Model（无 ServingController）；(C) `cudalm-server` → HTTP
  handler（ServingHttpApi）→ ServingController → Scheduler →
  SessionManager → Qwen35StateManager → Qwen35Model（v0.9 pinned
  serving chain）。三分支对实际代码逐一核实（tools/*.cpp 的
  include 与对象图、session_text_generator.h 的 "Owns its
  Scheduler" 成员）。明确写入：**ServingController 是 HTTP serving
  path 的策略边界，不是所有 CUDALM 执行模式的通用 frontend 层**。
- **Scoped Session ownership to session-bound paths**：删除
  "session 是资源与策略的唯一单位 / no stray state" 的无限定表述；
  改为"在 session-bound multi-turn / serving path 中绑定到
  `SessionId → SequenceId` 生命周期"，并明确 legacy one-shot /
  request-scoped execution path（含 `Scheduler::admit()`）仍存在，
  Session 不是整个引擎所有状态的唯一生命周期单位。
- **Distinguished tolerance-based golden validation from
  bit-exact parity**：README/总览（中英）的 correctness 表述改为
  两项分工——Golden numerical correctness（pinned official /
  quantized oracle，显式容差：bf16 stage tolerance、depth-aware
  full-model envelope，非 bit-for-bit）与 Semantic parity
  （external-state vs frozen legacy、paged-state parity、fused
  add+rmsnorm vs frozen 2-launch、reset/interleave parity 用
  bit-exact / `memcmp` 硬门）。
- **Corrected sampling ownership**：Model runtime + kernels 只负责
  forward → logits → KV/Delta state update（无采样）；greedy /
  temperature / top-k / top-p / seed 是 host-side generation/
  control logic（Qwen35Generator，Path A；Scheduler 的 per-request
  Sampler，Path B/C）；明确 CUDA kernel 不实现 top-k/top-p。
- 架构总览新增 §3.1 "Legacy vs external state" 短图（legacy
  model-owned state vs SessionId → SequenceId external state）+
  "external-state path was validated against the frozen legacy
  path with bit-exact parity where required"。
- 保持不动：Quick Start / HTTP 示例 / 性能数字 / Phase B evidence
  SHA（eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a）/ limitations /
  continuous batching 证据 / v0.7 KEEP/REJECT 叙事；
  docs/v10_performance.md 与 docs/v10_performance_zh.md、
  benchmarks/v10/* 未改。

Validation：三条 path 拓扑对实际代码逐项审计；claim scan（无
"Session is the only lifecycle unit / all state belongs to Session /
official oracle is bit-exact / Model-Kernels own sampling / all
frontends go through ServingController" 类未限定表述）；四份文档
（README 中英 + 架构总览中英）相对链接全部可解析；中英六项事实
（three execution paths / ServingController scope / legacy state
path / session-bound state ownership / sampling ownership /
golden tolerance vs bit-exact parity）逐项一致。

---

## Phase C Final Review Fix

- corrected HTTP handler/tokenizer ordering in the portfolio
  architecture diagram (README zh/en: three per-path tokenizer
  nodes — HTTP request handling precedes raw-text tokenization,
  which happens inside `ServingHttpApi::handle` before
  `admit_turn`; the shared-tokenizer `A/B/C → TOK` edges that
  implied the reverse order are gone);
- separated HttpTransport / server-loop ownership (bind / listen,
  single-threaded accept loop, connection I/O, one request at a
  time) from ServingHttpApi responsibilities (routing, request
  validation, tokenizer codec invocation, controller adaptation,
  JSON/NDJSON response semantics) in the architecture overviews
  (zh/en) and the README responsibility tables.

Docs-only; no other Phase C content touched.

---

## Deferred / Future Work

- **README curl 示例**：现有 `curl --data` 示例在新 CT 合同下
  仍可用（form-urlencoded 被接受），但应在 Phase C README 重写
  时显式加 `-H 'Content-Type: text/plain'` 作为规范写法。
- **`ModelConfig::v011_general_test` 命名**：见 A.5——保留，供
  v1.0 后续或 v1.1 决定是否重命名（需跨 C++/Python 同步）。
- **LICENSE**：Phase A 明确不决定——保持现状，后续单独确定。
- **GitHub Actions / CI**：Phase D 范围。
- **benchmark 整理**：Phase B 范围。
- **README 重写**：Phase C 范围。
