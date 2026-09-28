# CUDALM v0.8 Phase A — Session Runtime（Session abstraction + persistent-state lifecycle）

> 本阶段**唯一目标**：建立正确、清晰、可测试的 **Session abstraction**，并把
> 现有 KV / DeltaNet state 的生命周期从 **Request ownership 迁移到
> Session ownership**（state 跨 request 持久化）。
>
> 本阶段**不做**：chat template、HTTP server、OpenAI API、多轮文本对话
> 历史管理、context eviction / sliding window / TTL / LRU、multi-stream、
> CUDA Graph、batched prefill、新 kernel 优化（见 §9 当前限制）。
>
> 基线：`main` @ `cf28abd60b0d9f493149db68b0958ebb9c5214b5`
> （v0.7 merge，`V07_FINAL_FUNCTIONAL_SHA = 9edd9ef6aec84b8dcf66a262652c2e285ff73793`）。
> 开发分支：`v0.8-session-runtime`（不直接在 main 开发；本阶段不 merge）。

---

## 1. 现状

v0.7 结束时，runtime 的推理状态生命周期是 **request-scoped** 的：

| 组件 | 拥有的资源 | 生命周期 |
|---|---|---|
| `Qwen35StateManager`（v0.5） | `SequenceState`：一个 DeltaNet state slot（18 个 linear 层共享）+ 一个 `KvBlockTable`（6 个 full-attention 层共享）+ 逻辑 `length` | `create_sequence` → `retire_sequence`；池本身由 manager 构造时分配 |
| `Qwen35KvPagePool` / `Qwen35DeltaStatePool`（v0.5） | 设备存储（paged K/V、conv/recurrent） | 进程内长生命周期；`zero-on-release` / `zero-on-reset` 语义 |
| `Request`（v0.6） | `RequestId` + **在 admission 时新建的 `SequenceId`** + prompt + sampling 配置 + 执行状态 | admission → 终态（Finished/Cancelled/Failed） |
| `Scheduler`（v0.6） | 所有 `Request` 记录 | `finish()`（**任何**终态转移）立即调用 `retire_sequence()` —— **请求结束即释放全部推理状态** |

即：**当前 state 的 lifetime == request 的 lifetime**。一次 generate 结束，
KV pages、DeltaNet conv/recurrent、position 全部释放/清零。这正是
"一次性 Request 生命周期"，无法表达"同一会话的下一个 turn 继续用上一轮
推理状态"。

模型数学（kernel、forward、采样）全部冻结，本阶段**不改动任何模型数学语义**；
迁移只发生在**状态的所有权/生命周期控制面**。

## 2. 目标

把"一次性 Request 生命周期"升级为：

```text
persistent Session  +  multiple Requests / turns
```

本轮（Phase A）交付：

1. **Session abstraction**：`SessionId` / `SessionState` / `Session` /
   `SessionManager`（`include/cudalm/session.h` +
   `src/runtime/session.cpp`）；
2. **持久化语义**：request 完成后 session 的 KV pages、DeltaNet
   conv/recurrent state、position **全部保留**；只有
   `reset_session` / `destroy_session` 才执行明确的 reset/release；
3. **可测试的 lifecycle 门**：create/destroy、A/B 隔离、reset 不影响他
   者、destroy 资源回收、slot/page 复用无残留、context 容量查询
   （见 §7）；
4. **不破坏现有路径**：v0.5 state manager、v0.6 scheduler（request-scoped
   模式）全部冻结不动，既有测试 0 failed / 0 skipped。

## 3. Session / Request ownership

### 3.1 三个互相独立的 id 空间

```text
SessionId   —— SessionManager 签发，从 1 单调递增，永不复用
SequenceId  —— Qwen35StateManager 签发（每个 session 绑定一个）
RequestId   —— Scheduler 签发（v0.8 中 request 引用某个 session；
               v0.6 request-scoped 模式仍自建短命 sequence）
```

三个 id 空间**数值上可能巧合**（如第一个 session 和第一个 sequence 都是
1），但语义上**永不可互换**；沿用 v0.5/v0.6 的"monotone + never reused"
纪律（陈旧句柄不可能静默指向新对象）。

### 3.2 Session 与 Request

```text
Session（长生命周期，v0.8 新增）
├── SessionId
├── logical token position / context length   （== 绑定 sequence 的 length，
│                                              单一事实来源，不复制）
├── Full-Attention paged KV state            （绑定 sequence 的 KvBlockTable
│                                              + 池内物理 pages）
├── DeltaNet conv_state                      （绑定 sequence 的 Delta slot，
├── DeltaNet recurrent_state                 18 个 linear 层共享一个 slot）
├── state-slot / page ownership              （池的 zero-on-release 语义）
└── lifecycle metadata                       （SessionState、reset_count）

Request（临时操作，v0.6 概念，v0.8 语义变化）
├── RequestId
├── SessionId          ← 绑定到某个 session（Phase C 由 scheduler 完成）
├── new input tokens
├── generation config
└── request execution state
```

**所有权规则（v0.8 核心语义）**：request **从不**拥有、释放、重置
session 的 state。request 完成时：

```text
Request 记录释放；
KV pages        不释放
Delta conv      不释放、不清零
Delta recurrent 不释放、不清零
position        不清零
Session 继续存在
```

只有 session 自己的 `reset_session` / `destroy_session` 能改变/释放 state。

### 3.3 与 v0.6 request-scoped 模式的关系

v0.6 `Scheduler`（`admit` 时 `create_sequence`、终态时 `retire_sequence`）
是**冻结的 legacy 执行模式**，本阶段一行未改——它仍然表达"state lifetime
== request lifetime"。v0.8 session runtime 是**新的规范路径**；
Phase C 会把 scheduler 接入 session（request 在 admission 时绑定到已存在
的 session 的 bound sequence，request 终态时**不再** retire）。两种模式
在 Phase A 并存、互不干扰。

## 4. state lifecycle

```text
create_session()
   │  事务式：先在 manager 建 sequence（Delta slot 保证全零 by
   │  zero-on-release；KV block table 空、无 page；length = 0），
   │  成功后才登记 Session 记录并签发新 SessionId。
   │  OOM 时：什么都不登记（无 id、无半记录、无泄漏 slot）。
   ▼
Active（state 绑定；可被 request 驱动）
   │  request 驱动：forward_token_with_state / forward_batch_with_state
   │  （position 从 bound sequence 的 length 派生，单一事实来源）
   │  request 完成：【无任何 state 操作】← v0.8 语义变化点
   ▼
reset_session()（可选，任意多次）
   │  同一 SessionId 继续存活；bound sequence 就地重置：
   │  全部 KV pages 释放（zero-on-release）、Delta slot 就地清零
   │  （同一 slot id，不 release/re-acquire —— reset 不会 OOM）、
   │  length = 0。之后 session 与全新 session 行为完全一致。
   ▼
destroy_session()
   │  bound sequence 被 retire（全部 KV pages + Delta slot 释放并清零），
   │  Session 记录删除，SessionId 失效（永不复用）；
   │  之后对该 id 的任何操作都是 Status error / nullptr（fail loud，
   │  与 retire_sequence 的纪律一致；double-destroy 也报错）。
   ▼
（id 空间只增不减；物理 slot/pages 由池回收、复用、清零后发还）
```

与 v0.5 的对应关系（一一映射，无新状态语义）：

| Session 操作 | 底层 manager 操作 | 语义 |
|---|---|---|
| `create_session` | `create_sequence` | 新鲜零状态 + 空 KV table |
| `reset_session` | `reset_sequence` | 就地清零，id 不变 |
| `destroy_session` | `retire_sequence` + 删记录 | 释放全部资源，id 失效 |
| request 完成 | （无操作） | **v0.6 是 `retire_sequence`，v0.8 是空** |

## 5. API 设计

`include/cudalm/session.h`（实现 `src/runtime/session.cpp`）：

```cpp
using SessionId = std::uint64_t;   // 单调递增，永不复用；0 = 无 session

enum class SessionState {
  Active,  // Phase A 唯一存活态（同步单线程生命周期：session 要么 live
           // 要么已删除；枚举为未来扩展预留，如 v0.9 eviction 态）
};

struct Session {
  SessionId  id = 0;
  SequenceId sequence_id = 0;      // 绑定的 v0.5 sequence（state 的句柄）
  SessionState state = SessionState::Active;
  int reset_count = 0;             // lifecycle metadata（仅观测）
};

class SessionManager {
 public:
  explicit SessionManager(Qwen35StateManager& mgr);  // 非拥有引用

  Status  create_session(SessionId* out_id);   // 事务式（§4）
  const Session* lookup(SessionId id) const;   // nullptr = 未知/已销毁（查询）
  Status  sequence_id_of(SessionId id, SequenceId* out) const;
  Status  context_length_of(SessionId id, int* out) const;  // 活读 length
  bool    fits(SessionId id, int n_tokens) const;           // 纯查询
  Status  reset_session(SessionId id);     // 就地重置，同 id 存活
  Status  destroy_session(SessionId id);   // retire + 失效 + 删记录

  int       num_sessions() const;
  SessionId next_session_id() const;
  Qwen35StateManager& manager();           // 池 stream = 单 stream 契约
};
```

设计决策：

- **薄绑定层**：`SessionManager` 只持有 `SessionId -> Sequence` 绑定 +
  lifecycle metadata；**所有设备 state、池的 zero-on-release 语义、byte
  accounting 全部继承自冻结的 v0.5 组件**，不复制、不缓存（`length` 永远
  活读 bound sequence，避免双事实来源漂移）；
- **非拥有引用**：与 `Scheduler` 对 `Qwen35StateManager` 的引用纪律一致
  （manager 必须 outlive session manager）；
- **`lookup` 指针纪律**：`destroy_session` 会删除 map 记录，返回的指针
  在下一次 mutation 前有效（与 v0.5 state manager 的 `lookup` 纪律相同；
  测试中"destroy 前缓存字段"的写法即为此）；
- **fail loud**：未知 id 的任何操作返回 Status error（查询类返回
  nullptr/false）；double-destroy 报错（与 `retire_sequence` 一致，不做
  幂等吞错）；
- **`fits()` 纯查询**：不分配、不 mutation；负数/未知/已销毁一律 false。

### Context overflow policy（v0.8 初版，pinned）

DeltaNet recurrent state 含历史压缩信息，**不能丢任意历史**。初版策略：

```text
context_length + new_tokens > max_context  →  明确 reject（fail loud）
```

`fits(id, n)` 报告 `length + n <= max_seq_len` 是否成立（精确边界，
`n = max_seq_len - length` 为最后一个可容纳值）；不成立时未来的
appending request 必须被拒绝，**绝不静默丢弃最早 token/page**。
（KV 池 OOM 是另一个独立的运行时条件——沿用 v0.5
OOM-before-mutation 契约，`fits()` 只管 context length 策略。）

## 6. 资源模型

```text
进程
└── Qwen35StateManager（v0.5，冻结）
    ├── Qwen35KvPagePool        [n_full][num_pages][n_kv][pt][head_dim] × {K,V}
    │      物理 page id 空间；zero-on-release；LIFO 复用
    ├── Qwen35DeltaStatePool    per linear-ordinal: conv bf16 + recurrent fp32
    │      物理 slot 空间（一个 slot 覆盖全部 18 个 linear 层）；zero-on-release
    └── SequenceState × N      （每 session 恰好一个）
           ├── delta_slot   ──→ DeltaStatePool 物理 slot（1:1）
           ├── KvBlockTable ──→ KvPagePool 物理 pages（逻辑块:物理页 1:1）
           └── length        （position 单一事实来源）

└── SessionManager（v0.8 新增）
    └── Session × M ⊆ N       （每个 live session 恰好绑定一个 live sequence）
           └── sequence_id ──→ SequenceState
```

不变量（每次 mutation 后由测试断言）：

- `num_sessions == num_live_sequences`（1:1 绑定，无泄漏/悬挂 sequence）；
- 池 accounting 精确：`capacity == used + free`（继承 v0.5）；
- live 资源永不 alias（不同 session 的 slot/page 必不相交）；
- 复用必清零：释放→复用后新 owner 读回**全零**（zero-on-release 继承）；
- id 空间单调：SessionId / SequenceId / RequestId 各自只增不减。

## 7. Phase A correctness tests

### 7.1 `tests/cpu/test_session_manager.cpp`（无 checkpoint；真实设备池，
小合成 config：8 层 / 2 full + 6 linear / max_seq_len 64）

| 门 | 覆盖的 §6 语义 |
|---|---|
| create → valid | lookup 非空、Active、绑定 sequence 鲜活、length 0、空 KV table、live Delta slot、metadata 新鲜 |
| destroy → invalid | lookup nullptr；**所有**操作对该 id 均 Status error（含 double-destroy）；id 永不复用；bound sequence 同步 retire |
| 1:1 绑定不变量 | 每次 mutation 后 `num_sessions == num_live_sequences` |
| create 事务性 | Delta 池耗尽：失败 create 不登记任何东西（无 id/半记录/泄漏 slot），下次 create 单调成功 |
| **A/B 隔离** | slot 必不相交；A 弄脏 delta conv+recurrent（所有层）+ 两块 KV page（K+V）后，B 读回**全零**、metadata 全新鲜 |
| **reset** | A 同 id 存活：length 0、pages 全释放、slot **就地**清零（同一 slot id）、`reset_count` +1；B **逐字节**不受影响；reset 可重复、不会 OOM；未知 id fail loud |
| **destroy + 复用** | A 的全部物理资源回收后，C 以**新 SessionId** 复用 A 的物理 slot/pages（LIFO 顺序被精确断言），C 读回**全零**（无 A 残留）；B 完好 |
| `fits()` 边界 | `length+n <= max_seq_len` 精确边界（0 / max / max+1 / 负数 / 未知 / 已销毁）；纯查询（无分配无 mutation） |
| error cases | 未知 id 的所有操作 fail loud，池 accounting 与 session 集合**完全不变** |

### 7.2 `tests/cuda/test_qwen35_session_runtime.cpp`（真实
Qwen3.5-0.8B-Base checkpoint；self-skip 77 当 checkpoint 缺失）

用真实模型（24 层、18 DeltaNet + 6 full attention）通过冻结的
`forward_token_with_state` 驱动 session 的 bound sequence（"request" =
一个 token 块驱动，即未来 session-aware scheduler 每次 advance 会发的
调用），硬门全部 **BIT-IDENTICAL**（memcmp）：

| 门 | 证明 |
|---|---|
| create + binding | A/B 鲜活、fresh、slot 不相交、1:1 绑定 |
| **request 边界持久化（v0.8 核心语义）** | A 的 request 1（6 tokens）完成后**释放任何**：3 个 KV page 保留、Delta slot live 且**脏**（真实执行状态）、position = 6；A 的 request 2（2 tokens）是**append-only 续写**（从 position 6 继续，turn 边界无 state 重初始化）；8 步 FULL logits[248320] + 最终 hybrid state（18×conv/rec + 6×逻辑 K/V 行经 block table 读回）与**独立一次性连续 reference run** 逐位一致——若存在"request 结束清状态"类 bug，第 6、7 步必然破裂 |
| **真实使用下的隔离** | B 的 4-token request（在 A 的 request 运行期间驱动）与独立 fresh B run 逐位一致（logits + state） |
| **reset parity（真实模型）** | `reset_session(A)` 后 length 0、pages 释放、Delta slot 设备级**全零**、同 SessionId、`reset_count` 1；重放同样 6 tokens 与首轮**逐位一致**（无 reset 污染） |
| **destroy + 复用（真实模型）** | A 销毁后 SessionId 失效（所有操作报错）、对已 retire sequence 的 forward 被拒（绑定已拆除，无僵尸句柄）、池 accounting 回到**恰好** B 的资源；C 以新 id 复用 A 的物理资源，3-token run 与 fresh reference 逐位一致（无 A 残留）；B 最终状态仍等于其 fresh reference |

### 7.3 完整验证

验证环境：RTX 2080 Ti（sm_75）/ CUDA 11.8 / Qwen3.5-0.8B-Base checkpoint
（`/root/models/Qwen3.5-0.8B-Base`，preconverted
`build/data/qwen35_08b_full.cudalm`）。所有 self-skip（77）门在本环境
**真实运行**（checkpoint 在场）。

**基线（变更之前，`main` @ `cf28abd`）**：

```text
ctest --output-on-failure
→ 63/63 passed, 0 failed, 0 skipped（Total 843.67s）
```

**最终（functional + tests tree @ `9b69c37`，clean tree）**：

```text
cmake --build build -j8        # 全量构建：clean（-Wall -Wextra -Werror）
ctest --output-on-failure
→ 65/65 passed, 0 failed, 0 skipped（Total 784.65s）
  （= 基线 63 门全回归 + 新增 test_session_manager、
    test_qwen35_session_runtime 2 门真实运行）
bash scripts/check_no_torch.sh
→ forbidden_deps_check OK（no torch/pybind/py symbols in include/ src/）
  → no-torch CLEAN
```

**新增 device allocation/reset/reuse 路径（compute-sanitizer）**：

```text
compute-sanitizer --tool memcheck ./build/tests/test_session_manager
→ PASS + ERROR SUMMARY: 0 errors

compute-sanitizer --tool memcheck ./build/tests/test_qwen35_session_runtime \
    build/data/qwen35_08b_full.cudalm /root/models/Qwen3.5-0.8B-Base \
    /root/py311/venv/bin/python3 /root/code/cudalm-agent
→ PASS + ERROR SUMMARY: 0 errors
```

（`compute-sanitizer` = `/usr/local/cuda-11.8/bin/compute-sanitizer`。）

## 8. Phase B future contract

Phase B 要证明：

```text
turn 1: session A append [a,b,c]  → generate [d,e]
turn 2: same session A append [f,g] → generate ...
```

与等价的一次性连续执行在定义的 numerical contract 下保持一致
（append-only incremental execution 的 bit-exact 数值等价）。

Phase A 为此留下的清晰路径（均已实现并被门住）：

1. **position 派生自 bound sequence 的 `length`**（单一事实来源）——
   incremental append 就是"从当前 length 继续 forward 新 token"，
   `forward_token_with_state` 今天已这样工作，session 层不引入第二个
   position 计数器；
2. **turn 边界零操作**——request 完成不触碰 state，因此 turn 2 天然
   续写 turn 1 的 KV pages 与 Delta conv/recurrent（无重初始化、无拷贝）；
3. **`sequence_id_of()` / `context_length_of()`** 暴露了 Phase B/C 驱动
   模型所需的全部句柄；`fits()` 是 overflow reject 的策略入口；
4. 本阶段 7.2 的"chunked-on-session == one-shot"门就是 Phase B 数值
   等价门的**状态层前身**——Phase B 在其上加采样/生成循环与 EOS/
   max_new_tokens 语义，等价性要求不变（同一 token 流、同一 kernel、
   同一 op 顺序 → bit-identical）。

Phase A 的 API **不预设**任何阻碍 append-only incremental execution 的
约束：没有 turn 计数、没有 prompt 快照、没有 state 拷贝/重放机制。

## 9. 当前限制

- **尚无完整 multi-turn generation**：Phase A 的"request"是测试内直接
  的 token 驱动；带 generation config 的 session-bound 生成循环属于
  Phase B；
- **scheduler 尚未接入 session**：v0.6 `Scheduler` 仍是 request-scoped
  （终态 retire sequence）；session-aware admission / 多 session
  interleaving 属于 Phase C；
- **无 eviction**：context overflow = 明确 reject（§5），无 sliding
  window / TTL / LRU（v0.9 以后）；DeltaNet recurrent state 含压缩历史，
  任何"丢最早 token"策略都未实现也暂不设计；
- **单 stream**：继承 v0.5 单 stream 契约（池与 model 同 stream）；
  无 multi-stream / CUDA Graph；
- **无 HTTP/OpenAI API、无 chat template、无 conversation history 管理**；
- `SessionState` 目前只有 `Active`（同步单线程语义下 session 要么 live
  要么已删除；枚举为未来扩展预留）；
- 池容量（pages/slots）仍是构造期固定配置：session 数受
  `delta_capacity_slots` 上限约束，KV pages 受 `kv_capacity_pages`
  约束（超限 = 明确 OOM Status，不静默）。

## 10. 已知风险（诚实清单）

- **生命周期风险**：`lookup()` 返回的 `Session*` 在下一次
  `destroy_session` 后悬垂（与 v0.5 `SequenceState*` 同纪律，已在头文件
  注释与测试中显式处理：mutation 前缓存字段）；
- **资源泄漏风险**：session 不 destroy 就一直持有 slot/pages（by design，
  持久化语义；进程内上限 = 池容量，超限 create 明确 OOM）；
- **slot/page 复用风险**：继承 v0.5 zero-on-release；本阶段新增设备级
  复用门（destroy → recreate → 读回全零 → 真实模型 run 逐位一致），
  未新增风险面；
- **未来 incremental append 的 API 限制**：无。唯一约束是 §9 的
  overflow-reject 策略（这是 pinned 语义，不是 API 缺陷）。
