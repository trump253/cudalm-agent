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
