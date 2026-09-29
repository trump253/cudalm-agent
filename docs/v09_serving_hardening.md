# CUDALM v0.9 Serving Hardening — Phase A + Phase B（serving admission / backpressure / resource guardrails + committed-token streaming / cancellation / deadline）

> **状态：Phase A 完成（external review PASS）；Phase B 完成（等待
> external review）**。
>
> **Phase A 目标**：在冻结的 v0.8 persistent Session + Scheduler 之上
> 增加一个明确的 **serving admission layer**——服务在资源不足时
> **reject early / fail loud / zero mutation**，而不是先接收 request、
> 最后在 GPU state allocation / forward 时才失败。
>
> **Phase B 目标**：在 serving layer 增加 **pull-based committed-token
> streaming**、**explicit cancellation**、**per-request deadline**
> （cooperative，scheduler step 边界）——保持单线程、单 CUDA stream，
> 不引入 HTTP 或异步线程。
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
  int max_sessions = -1;            // live session 配额（三态，见下）
  int max_live_requests = -1;       // live request 配额（三态，见下）
  int max_context_tokens_per_session = 0;  // 可选 policy cap（0 = 禁用）
};
```

**quota 三态语义**（`max_sessions` / `max_live_requests`）：

```text
-1  = unlimited        （不设限）
0   = zero capacity    （拒绝一切 admission）
N>0 = capacity N
```

> **Review fix（2025，SHA 见 §7 / provenance）**：最初实现用
> `limit > 0` 判断，导致 `0` 被错误解释为 unlimited。已固定为上面的
> 三态：判断改为 `limit >= 0`。`max_context_tokens_per_session` **不**
> 是 quota（是可选 policy cap），保持 `0 = 禁用` 不变。

- **max_sessions**：`create_session` 时 `live sessions < max_sessions`
  → allow；`==` → **reject**（`0` → 第一次 create 即 reject）；
- **max_live_requests**：turn admission 时 `live requests <
  max_live_requests` → allow；达到 → **reject**（`0` → 第一次
  admit 即 reject，session 仍可创建）。live == 经**本
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

## 7. Phase B：Committed-token Streaming（commit-before-visible）

**硬约束：`sampled token != stream-visible token`**。SessionBound
request 中 `generated` 可能含 sampled-but-pending token（0 或 1 个，
frozen Phase C commit-then-stop contract）；**只有
`generated[0 .. committed_generated)` 允许对外 streaming**：

- token 只有在 **forward 成功、已经进入 Session KV / Delta /
  position** 之后才能 emit；
- **pending token 永远不 emit**（包括 forward failure 与 cancel 时）；
- 每个 committed token **exactly once、按顺序** emit（serving-layer
  的 per-request `emitted_count` cursor）；
- **EOS / max-new 的最后 token 必须先 emit，再报告 terminal**
  （frozen commit-then-stop 保证最后 token 先 committed 才 terminal）；
- **pull-based**：controller 从不 push——token 只在 drive / poll
  **drain** 时变得可见；
- **不复制 generation loop**：继续驱动 frozen `Scheduler::step()`
  （含 batched decode）；serving 层只做 drain 与事件包装。

**API**（`ServingEvent` + 三个入口）：

```cpp
struct ServingEvent {
  ServingEventKind kind;        // Token / RequestTerminal
  RequestId request_id; SessionId session_id;
  int token_id;                 // Token 事件
  RequestStatus status; FinishReason finish_reason;  // Terminal 事件
  bool deadline_exceeded;       // serving-layer termination reason
};
std::vector<ServingEvent> step_stream();   // 一次 frozen step + drain
std::vector<ServingEvent> run_stream();    // 驱动到 quiescence（逐步 drain）
Status poll(RequestId, std::vector<ServingEvent>*);  // 不 drive，单请求 drain
```

**多 request streaming**：A/B 多 Session 同时 live 时仍用现有
scheduler / batched decode——一次 `Scheduler::step()` 可以让多个
request 各自产生新的 committed token，serving 层**分别 drain**；每个
request 的事件连续、有序，`RequestTerminal` 事件在该 request 的**所有
Token 事件之后**；**无重复 / 无跨 request contamination**。

**Streaming 生命周期（review fix：解耦 live/quota 与 stream drain）**：

- **`terminal != streaming state destroyed`**：request terminal 时
  **live request quota 立即释放**（derived live count，terminal-but-
  undrained 永不计 live），**但**未 drain 的 committed tokens /
  terminal event **继续保留**，直到被 drain exactly once——reap 只
  发生在 fully-drained 之后（`sync()` 不再无条件 reap terminal）；
- **`step()` + `poll()`**：用普通 `step()`（不 drain）驱动到
  terminal 后，`poll(rid)` 必须返回**所有未 emit 的 committed Token
  events + RequestTerminal**（顺序正确、exactly once）；
- **poll 的 pinned 语义**：一个 request 的事件在 poll / step_stream /
  run_stream 之间**至多 emit 一次**；poll unknown 或 **already
  fully-drained** 的 id → **error**（明确且一致）；
- **`run_stream()`**：controller 中已存在 terminal-but-undrained
  request 时，**不会**仅因 `live_requests == 0` 直接返回而遗失
  pending events——先/按序 drain 再驱动 live request；
- **`run()`（non-streaming）**：途中遇到的 pending terminal events
  被 **discard**（drain 推进 cursor 并 reap fully-drained request）
  ——**之后不可再 poll**；需要事件请用 `*_stream` / `poll` 在
  `run()` 之前取走；
- 不修改 frozen `Scheduler` 的 Request 生命周期（controller 只管
  自己的 bookkeeping）。

## 8. Phase B：Cancellation（cancel preserves Session）

`ServingController::cancel(request_id)` 透传 frozen scheduler
（Waiting/Running → Cancelled；already-terminal 保持现有 idempotent
contract）：

- 后续**不得再 forward**；
- **已 committed 且尚未 emit 的 token 可正常 drain**（cancel 不 reap
  streaming bookkeeping——`poll` / 下一次 `step_stream` 仍可取走）；
- **pending token 不得 emit**；
- **Session 保持 live**，context / KV / Delta 保持**最后 committed
  boundary**；
- **request quota 正常释放**（derived live count，立即生效）；
- **next turn 能从该 boundary 继续**；
- **统一 streaming 生命周期（review fix）**：`cancel()` 成功且
  request 变为 terminal 时，**纳入与 step-terminated 相同的
  terminal-pending lifecycle**——committed 未 emit 的 token +
  Cancelled terminal event 保持可 drain（`poll` / `step_stream` /
  `run_stream` 均不遗失；仅因 `live_requests == 0` 不会丢事件）；
  fully-drained 的 request 已被 reap（不在 tracked 集），**不会**
  重新进入、不会重复 terminal event；already-terminal cancel 保持
  frozen scheduler 的 idempotent 语义。

## 9. Phase B：Deadline（cooperative boundary between scheduler steps）

- **monotonic clock**：`MonotonicClock`（抽象）+
  `SystemMonotonicClock`；CPU test 注入 **fake clock**（构造参数，
  `nullptr` = 系统时钟）；
- **per-request deadline**：`admit_turn(..., deadline)`（默认
  `time_point::max()` = 无 deadline）；
- **语义固定**：**每次 scheduler step 之前检查 deadline**（`step` /
  `run` / `step_stream` / `run_stream` 全部在 step 前检查）；已过期
  → **cancel request before its next forward** → **no additional
  token commit** → **session remains live**；
- **deadline = cooperative boundary between scheduler steps**：不要求
  中断已发出的 CUDA work，不做 kernel preemption；
- **独立 termination reason**：`ServingEvent::deadline_exceeded`
  （serving-layer 标志）——**不修改 frozen `FinishReason`**（scheduler
  层面该 request 报告 Cancelled/Cancelled）。

## 10. Phase B 当前限制

streaming 目前是 **token-level（committed token IDs）**；**不做**
text-byte streaming / incremental UTF-8 decoder / HTTP / OpenAI API /
threads / async runtime（留给后续 Phase）。

## 11. 测试

- **`test_serving_admission`**（CPU contract gate：deterministic fake
  forwarder + 真实池 + 真实 SessionManager + Scheduler + controller）：
  max_sessions（limit 处 reject、zero mutation、无 SessionId 消耗；
  **reset 不释放**；**destroy 释放 + reuse**）；max_live_requests
  （limit 处 reject、zero mutation、无 RequestId 消耗；**terminal
  释放 + reuse**——limit reached → reject → existing finishes → new
  admission succeeds，证明计数不永久卡死；重复 `sync()` 幂等）；
  context policy cap（低于模型上限处 reject、精确边界 `==` 接受、
  构造钳制 999999 → 64）；**cancel 与 forward failure 释放配额**
  （failure 后 session 从 committed boundary 重试）；**quota 三态
  边界（review fix）**：`max_sessions = 0` → 第一次 create 即
  reject（zero mutation、无 SessionId 消耗、`rejected_session_limit`
  +1）；`max_live_requests = 0` → session 可创建、第一次 admit 即
  reject（zero mutation、无 RequestId 消耗、`rejected_request_limit`
  +1）；`-1` 仍 = unlimited（多 session + 并发 live request 自由
  准入、零 limit 拒绝）；
- **`test_serving_integration`**（真实 Qwen3.5-0.8B-Base checkpoint
  integration regression）：controller 驱动 A（greedy）/ B（seeded）
  各两 turn（两个 live 同时——batched 路径）与**直接 raw
  Scheduler 参考**（独立 manager）逐 turn 比较——generated ids +
  turn 边界 / 最终 length 全部**相同**（serving 层不改变
  token/state correctness）；真实 runtime 上的 session/request
  limit 强制 + reuse；context cap（8 << 262144）边界接受 / 超限
  reject（zero mutation）；clean teardown（所有 manager 池
  accounting 归零）。

- **`test_serving_streaming`**（Phase B CPU contract gate：deterministic
  fake forwarder + 真实池 + 真实 SessionManager / Scheduler +
  controller streaming API）：pending g0 不可见 → forward 成功后才
  emit；exactly-once（后续 poll 不再 emit）；max-new / EOS 最后
  token 先 emit 再 terminal；**cancel**（committed prefix 可 drain、
  pending 不 emit、cancel 后无 forward、session 保持 live、next turn
  从 boundary 继续、quota 释放）；**deadline**（fake monotonic
  clock：deadline 前正常推进、过期后 next step 前 cancel、不多
  commit 一个 token、terminal 事件 `deadline_exceeded = true`）；
  **A/B interleaving**（per-request 事件流 == 各自 final committed
  ids、无跨 request contamination）；**review fix 生命周期
  边界**：terminal `step()` → `poll()`（最后 committed token +
  terminal event 可 poll、exactly once、re-poll error、quota 已释放
  且立即可复用）；terminal-but-undrained + `run_stream()`（pending
  events 不因 `live_requests == 0` 丢失）；
- **`test_serving_stream_integration`**（Phase B 真实 Qwen3.5-0.8B-
  Base checkpoint integration gate）：A（greedy）/ B（seeded）两
  Session **一起**经 `step_stream` 驱动（batched decode 路径——一次
  step 为两个 request 提交、分别 drain）：每 turn **concatenated
  streamed ids == request 最终 committed generated ids**（exactly
  once、按序）；final context length 精确（6/5 → 10/9）；session
  可继续下一 turn（turn 3，final A 13）；clean teardown（池
  accounting 归零）。

验收证据见 `docs/provenance.md` v0.9 Phase A / Phase B 章节（绑定
exact SHA）。

## 12. 当前 non-goals（v0.9 Phase A/B 不做）

streaming 目前是 **token-level**——**text-byte streaming /
incremental UTF-8 decoder**、HTTP server、OpenAI API、threads / async
runtime、multi-stream、chat template、CUDA Graph、kernel 优化、
动态 quota 系统——全部属于 v0.9 后续 Phase 或之后版本。（TTL / LRU /
eviction 已由 **Phase C** 完成，见 §13。）

## 13. Phase C：Session TTL / LRU Eviction

**核心原则：`eviction == destroy whole Session`**——policy 完全在
`ServingController`（零 frozen Scheduler / SessionManager / CUDA /
model 改动），复用 frozen `SessionManager::destroy_session`：

```text
SessionId invalidated forever（monotonic、never reused）
bound Sequence retired
KV pages released + zeroed
Delta slot released + zeroed
logical context gone
```

**绝对不是** context truncation / 只丢 KV / 截短 context——v0.8 的
context overflow contract（REJECT）原样保持；Session eviction 与
context truncation 是两回事。

**默认 policy = Phase A/B 行为完全不变**（`SessionEvictionPolicy{}`
= TTL disabled + LRU pressure disabled）：`max_sessions` reached →
仍按 Phase A reject，**不**偷偷改冻结的 admission contract。

### 13.1 Policy API（additive）

```cpp
struct SessionEvictionPolicy {
  // nullopt = DISABLED（明确的 sentinel，无 0/-1 歧义）；
  // 0 = 一旦 truly idle 立即 eligible；> 0 = 正常 idle timeout
  std::optional<std::chrono::steady_clock::duration> idle_ttl;
  bool lru_on_session_pressure = false;
  // with_idle_ttl(...) / with_lru_on_session_pressure(bool) builders
};
// 构造：ServingController(..., MonotonicClock* clock = nullptr,
//                              SessionEvictionPolicy policy = {})
// 显式 maintenance API：
std::vector<SessionId> evict_expired_sessions();  // TTL sweep
bool evict_one_lru_idle();                        // 显式 LRU
bool is_eviction_eligible(SessionId);             // idle + safe
```

**无 background thread / timer**——TTL sweep 只在明确 maintenance
point 发生：显式 `evict_expired_sessions()` 调用 +（仅当 TTL 启用
时）`create_session()` 前的 lightweight sweep。

### 13.2 Activity / idle 定义（Phase B `MonotonicClock`，无 wall
clock）

每 MANAGED session（经本 controller 创建）记录 last-activity
时间戳。**Activity 更新点**：successful `create_session` /
`reset_session` / `admit_turn` / `cancel` / deadline terminal，以及
**每个驱动该 session 非 terminal request 的 step**（长 request 的
TTL 锚定在近期执行，而非 admit 时间）。`poll()` / event drain **不**
刷新——消费输出不是新的推理活动。

**deadline terminal 的锚定（review fix）**：`check_deadlines_`
的 deadline cancel **也**刷新该 session 的 activity——TTL 从
deadline terminal 重新锚定，而不是停留在旧的 admission / drive
时间（与显式 `cancel()` 同一 contract；最小实现在
`check_deadlines_` 内，不经 public `cancel()`，deadline check 仍
在 scheduler step 前、无 additional forward / commit、session
保持 live）。

**unmanaged session**（直接经 SessionManager 创建、不经
controller）：不猜 activity、**不**自动 eviction——仍占
`live_sessions` / `max_sessions`；找不到 eligible managed session
时 pressure admission 正常 reject。

### 13.3 TTL 语义

`idle_age = now - last_activity`；`idle_age >= TTL` → eligible。
TTL == 0 = 一旦 truly idle 立即 eligible。sweep 结果确定性
（ascending SessionId）。

### 13.4 LRU 顺序 + tie-break

只在 eligible idle session 中选：**oldest last activity first**；
时间完全相同 → **SessionId smaller first**（pinned deterministic，
便于测试与日志复现）。

### 13.5 保护（eligibility = 全部满足）

```text
MANAGED（经 controller 创建）
仍 live
Scheduler::session_busy == false
无 terminal-but-undrained streaming bookkeeping
```

最后一条是 Phase C 的 hard contract：Phase B 已区分
`request terminal != stream events fully drained`——stream 事件尚未
消费完的 session **protected**，不会在它的 Token / RequestTerminal
events 尚未消费时被自动回收。Waiting / Running request 始终
protected（busy）。

### 13.6 max_sessions pressure 语义

- **默认 policy**：`max_sessions` reached → reject（完全不变）；
- **启用 LRU pressure 后**：`create_session()` 且
  `live_sessions == max_sessions`（**严格等于**）→ 先尝试 evict
  ONE eligible LRU idle session；成功 → 建新 session；无 eligible
  candidate → 保持 Phase A reject（**no SessionId consumed、no
  partial mutation**、`eviction_no_candidate++`）；
- **over-limit fail-safe（review fix）**：`live_sessions >
  max_sessions`（例如 unmanaged session 占掉了容量）→ **不**
  eviction、直接走 Phase A session-limit reject（no SessionId
  consumed、no session destroyed、no pool mutation；本 phase 不做
  multi-eviction recovery）；
- `max_sessions == 0` **永远**不能靠 eviction 绕过 zero-capacity
  contract。

### 13.7 Transactional 语义

只有 frozen `destroy_session` **成功**后才：删除 activity
metadata + eviction counter++ + 记录 evicted id。destroy 失败 →
fail loud、metadata 保留、不假装成功。SessionId never reused。

### 13.8 Stats（additive）

`evicted_sessions_ttl` / `evicted_sessions_lru` /
`eviction_no_candidate`——只计 **automatic** eviction；手工
`destroy_session()` 不计入。

### 13.9 Phase C 测试

- **`test_serving_eviction`**（CPU contract gate——fake forwarder +
  fake monotonic clock + 真实 pools / SessionManager / Scheduler）：
  A. TTL（更老的 B 过期被 evict、最近 activity 的 A 存活、B 的
  SessionId 永久失效、KV/Delta/sequence accounting 正确释放、复用的
  Delta slot 验证 **zeroed**——evict 前 slot 被 pattern，无 stale
  contamination）；B. **busy 保护**（TTL 过期但有 live request →
  不 evict、不 cancel、不改状态）；C. **terminal-but-undrained
  保护**（request terminal 但未 drain → sweep 不得 evict；drain
  完成 + TTL 后**才**可 evict——Phase C hard contract）；D. LRU
  pressure（max_sessions=2 时 evict 最老的 eligible、新 session
  拿到新 monotonic id、num_sessions 保持 2）；E. **无 candidate**
  （A busy + B terminal-undrained → create C reject、no SessionId
  consumed、no pool mutation、A/B untouched——pressure safety
  gate）；F. 默认 policy = 严格 Phase A（不自动 eviction）；
  G. unmanaged session 永不 auto-evict（仍占 limit）；
  H. **review fix**：deadline terminal 刷新 activity（TTL 从
  deadline terminal 重新锚定——deadline 后第一次 `step_stream`
  cancel + drain，TTL 前 sweep 不 evict、`idle_age == TTL` 才
  evict）；
  I. **review fix（over-limit fail-safe）**：`live_sessions >
  max_sessions`（unmanaged 占容量）→ create reject 且**不**
  eviction（no id consumed、no pool mutation、eligible managed
  session 存活）；
- **`test_serving_eviction_integration`**（真实 Qwen3.5-0.8B-Base
  checkpoint，小型 gate——不做深度 KV/Delta memcmp）：B（2+2）
  先跑、A（3+2）后跑（A 更 recent）→ fake-clock TTL sweep 只 evict
  B（B 无法继续——admit error；Delta accounting 释放）→ A 正常
  continuation（streamed ids == final generated ids、length 精确、
  无 evicted 邻居污染）→ 新建 C fresh 生成（新 monotonic id）→
  clean teardown（pool accounting 归零）。

### 13.10 Phase C non-goals

HTTP server、OpenAI-compatible API、text-byte streaming、async
threads、**background timer thread**（当前 sweep 只在显式
maintenance point）、distributed session store、session persistence
to disk、KV swap-to-CPU、partial context eviction、sliding-window
truncation、multi-stream、CUDA Graph、kernel 优化、chat template。
**尤其禁止**：context overflow → evict oldest tokens。
