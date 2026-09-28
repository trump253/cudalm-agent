# CUDALM v0.9 Serving Hardening — Phase A（serving admission / backpressure / resource guardrails）

> **状态：Phase A 完成（等待 external review）**。
>
> **Phase A 目标**：在冻结的 v0.8 persistent Session + Scheduler 之上
> 增加一个明确的 **serving admission layer**——服务在资源不足时
> **reject early / fail loud / zero mutation**，而不是先接收 request、
> 最后在 GPU state allocation / forward 时才失败。
>
> 基线：`main` @ `1d6c83a8c5b64c92a8f4a8df4d750411dab3f36f`（v0.8
> merge，tag `v0.8`）。开发分支：`v0.9-serving-hardening`。
>
> v0.8 已 DONE / FROZEN：**不修改** frozen Session / Scheduler /
> CUDA model semantics（Phase A 全部为 additive 新文件 + 测试 +
> 文档，零 frozen code 改动）。

---

## 1. 架构：独立薄控制层

```text
ServingController          （v0.9 Phase A 新增——POLICY 层）
  │  limits + admission + quota 生命周期 + stats
  │
  ├── Scheduler            （frozen v0.6 control plane：admit_session_turn
  │      │                  preflight / commit-then-stop / batching）
  │      └── SequenceForwarder（ModelForwarder / fake）
  └── SessionManager       （frozen v0.8：session<->sequence 绑定、
                             create/reset/destroy、池 accounting）
```

- 底层 frozen runtime 负责 **correctness**；serving layer 负责
  **policy / quota / admission / observability**；
- `ServingController` **非拥有**（scheduler 与 session manager 均
  outlive controller，v0.5/v0.6/v0.8 纪律）；
- controller 不写 model math、不写 generation loop、不写 sampling
  逻辑；
- **frozen Phase C preflight 不重复实现**：instance identity /
  sampling / vocab / non-empty / max_new / eos / token range /
  session live / **busy session** / **model context overflow** 仍由
  `Scheduler::admit_session_turn` 在下方执行，controller 只在其
  之上**追加** serving limits 检查并原样透传其 Status。

文件：`include/cudalm/serving_controller.h` +
`src/runtime/serving_controller.cpp`。

## 2. Serving limits（`ServingLimits`）

```cpp
struct ServingLimits {
  int max_sessions = -1;            // live session 配额（-1 = unlimited）
  int max_live_requests = -1;       // live request 配额（-1 = unlimited）
  int max_context_tokens_per_session = 0;  // 可选 policy cap（0 = 禁用）
};
```

- **max_sessions**：`create_session` 时 `live sessions < max_sessions`
  → allow；`==` → **reject**；
- **max_live_requests**：turn admission 时 `live requests <
  max_live_requests` → allow；达到 → **reject**。live == 经**本
  controller** 准入且尚未终态的 request；
- **max_context_tokens_per_session**（可选）：per-session **policy
  cap**（0 = 禁用）。用与 scheduler model overflow **相同的投影**
  `length + input + max_new_tokens > cap` 判定。构造时**钳制到模型
  `max_seq_len`**——它只能**收紧**、不能突破模型上限；scheduler 的
  模型 overflow gate 仍在下方生效。
- 不引入复杂动态 quota 系统。

## 3. Backpressure / reject 语义（zero mutation）

serving-layer 的每个 limit rejection 都发生在**触碰 frozen runtime
之前**：

```text
create_session 被拒（session limit）
  → NO SessionId consumed（create_session 根本不被调用）
  → no sequence created
  → no Delta slot / KV page mutation

admit_turn 被拒（request limit / context limit）
  → NO RequestId consumed（admit_session_turn 根本不被调用）
  → no sequence created
  → no KV page / Delta slot / logical-length mutation
```

admission 检查顺序（每步失败 = fail loud + zero mutation）：

1. **live request quota**（`max_live_requests`）→
   `rejected_request_limit++`；
2. **per-session context policy cap**（`max_context_tokens_per_session`
   > 0 时；unknown session 跳过本检查，由下方 frozen preflight 以
   自己的消息 reject）→ `rejected_context_limit++`；
3. **frozen Phase C admission**（透传，语义不变）。

## 4. Quota 生命周期

```text
request terminal（Finished / Cancelled / Failed）→ live_requests 释放（恰好一次）
destroy_session                                    → live_sessions 减少
reset_session                                      → live_sessions 不变
```

- **不 double-decrement**：live request 计数**不是计数器**——它是
  从 `tracked_`（经本 controller 准入、尚未 reaped 的 request id）+
  每个 id 当前的 scheduler status **推导**出来的；`sync()`（每次
  drive / cancel 后，或显式调用）把已终态的 id 从 `tracked_` 移除，
  幂等、可任意次调用，结构上不存在 double-decrement 路径；
- Cancelled / Failed request 与 Finished 一样释放 admission
  capacity（`cancel()` 内部立即 sync）；
- session 配额是 manager 的 live 计数（`stats()` 读穿），reset 不
  减少、destroy 减少。

## 5. Observability（`ServingStats`——刻意轻量，非 telemetry 系统）

```cpp
struct ServingStats {
  int live_sessions;             // 当前 live session 数（读穿 manager）
  int live_requests;             // 经本 controller 准入且未终态的 request 数
                                 //（读取时推导）
  std::uint64_t total_admitted_sessions;   // 累计成功 session 准入（不减）
  std::uint64_t total_admitted_requests;   // 累计成功 turn 准入（不减）
  std::uint64_t rejected_session_limit;    // serving 层 session limit 拒绝
  std::uint64_t rejected_request_limit;    // serving 层 request limit 拒绝
  std::uint64_t rejected_context_limit;    // serving 层 context cap 拒绝
};
```

计数语义：**只有 serving-layer 的 limit 拒绝**计入 `rejected_*`；
frozen preflight 的拒绝（busy / model overflow / unknown session /
identity 等）是 runtime 自己的 contract，原样透传、**不**计入这些
计数器（可用 `Status::message` 区分）。

## 6. API 一览

```cpp
ServingController(Scheduler&, SessionManager&, ServingLimits);

Status create_session(SessionId*);                    // session 准入（policy）
Status admit_turn(SessionId, const std::vector<int>&, // turn 准入（policy +
                  int, int, const SamplingConfig&,    //   frozen preflight）
                  RequestId*);
Status reset_session(SessionId);      // 透传；session 保持 live
Status destroy_session(SessionId);    // 透传；释放 session 配额
Status cancel(RequestId);             // 透传 + 立即 sync（释放 request 配额）
Status step();  Status run();         // 驱动 frozen scheduler；drive 后 sync
void sync();                    // 幂等 quota 同步
ServingStats stats() const;      // 快照（live_* 读取时推导）
const ServingLimits& limits() const;
```

## 7. 测试

- **`test_serving_admission`**（CPU contract gate：deterministic fake
  forwarder + 真实池 + 真实 SessionManager + Scheduler + controller）：
  max_sessions（limit 处 reject、zero mutation、无 SessionId 消耗；
  **reset 不释放**；**destroy 释放 + reuse**）；max_live_requests
  （limit 处 reject、zero mutation、无 RequestId 消耗；**terminal
  释放 + reuse**——limit reached → reject → existing finishes → new
  admission succeeds，证明计数不永久卡死；重复 `sync()` 幂等）；
  context policy cap（低于模型上限处 reject、精确边界 `==` 接受、
  构造钳制 999999 → 64）；**cancel 与 forward failure 释放配额**
  （failure 后 session 从 committed boundary 重试）；
- **`test_serving_integration`**（真实 Qwen3.5-0.8B-Base checkpoint
  integration regression）：controller 驱动 A（greedy）/ B（seeded）
  各两 turn（两个 live 同时——batched 路径）与**直接 raw
  Scheduler 参考**（独立 manager）逐 turn 比较——generated ids +
  turn 边界 / 最终 length 全部**相同**（serving 层不改变
  token/state correctness）；真实 runtime 上的 session/request
  limit 强制 + reuse；context cap（8 << 262144）边界接受 / 超限
  reject（zero mutation）；clean teardown（所有 manager 池
  accounting 归零）。

验收证据见 `docs/provenance.md` v0.9 Phase A 章节（绑定 exact SHA）。

## 8. 当前 non-goals（Phase A 不做）

streaming、deadline / timeout、TTL / LRU、eviction、HTTP server、
OpenAI API、multi-stream、CUDA Graph、kernel 优化、chat template、
动态 quota 系统——全部属于 v0.9 后续 Phase 或之后版本。
