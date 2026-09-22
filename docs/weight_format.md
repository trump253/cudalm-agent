# CUDALM 权重文件格式（`.cudalm`）— v1

单个 decoder block 权重（CUDALMW01）的小型、**确定性、带版本、边界
检查**的二进制容器，外加黄金参考容器（CUDLMG01，见本文档第二部分）。
权重文件由 `tools/convert_weights.py`（Python，离线）写入，由
`src/runtime/weight_loader.cpp`（纯 C++，无 PyTorch）读取。本文档是
单一事实来源（single source of truth）：C++ 加载器与 Python 写入器都
实现它，并由测试交叉校验。

所有整数均为**小端**。无 NUL 结尾。除 fp16/fp32 张量数据外不含任何
浮点载荷。文件布局：

```
+---------+--------------------------------------------------------------+
| 表头    |  固定，72 字节                                                |
+---------+--------------------------------------------------------------+
| 记录表  |  n_tensors × TensorRecord（变长，紧密排列）                    |
+---------+--------------------------------------------------------------+
| 数据区  |  张量字节块，每个 16 字节对齐                                  |
+---------+--------------------------------------------------------------+
```

**权重来源（语义）。** 本格式承载的权重是 `tools/` 脚本生成的
**确定性合成权重（固定种子合成权重，fixed-seed synthetic weights）**：
v0.1 / v0.1.1 用它验证原生运行时架构（权重容器、加载器、DecoderBlock
接线）、移植 kernel 集成与逐 stage golden 正确性，与任何真实模型无关。
真实 HuggingFace safetensors checkpoint 的权重摄入不在 v0.1.1 范围，
随 v0.2 Qwen3.5 bring-up 完成。

## 表头（72 字节）

| 偏移 | 大小 | 类型  | 字段             | 取值 / 含义 |
|-------:|-----:|-------|----------------|-----------------|
| 0      | 8    | 8s    | `magic`        | 字节 `"CUDALMW01"` |
| 8      | 4    | u32   | `version`      | `1`（未知 → 硬错误） |
| 12     | 4    | u32   | `flags`        | 保留，必须为 `0` |
| 16     | 4    | u32   | `n_tensors`    | TensorRecord 数量 |
| 20     | 36   | —     | `config`       | ModelConfig 数据块（见下） |
| 56     | 8    | u64   | `table_offset` | 记录表相对文件起始的偏移（== 72） |
| 64     | 8    | u64   | `payload_offset` | 数据区相对文件起始的偏移 |

### `config` 数据块（36 字节，固定顺序）

| 偏移 | 大小 | 类型 | 字段                |
|-------:|-----:|------|-------------------|
| 0      | 4    | i32  | `hidden_size`     |
| 4      | 4    | i32  | `n_heads`         |
| 8      | 4    | i32  | `n_kv_heads`      |
| 12     | 4    | i32  | `head_dim`        |
| 16     | 4    | i32  | `intermediate_size` |
| 20     | 4    | i32  | `group_size`      |
| 24     | 4    | i32  | `max_seq_len`     |
| 28     | 4    | f32  | `eps`             |
| 32     | 4    | f32  | `rope_theta`      |

## TensorRecord（变长，紧密排列）

| 字段        | 大小       | 类型 | 含义 |
|-----------|------:|------|---------|
| `name_len`| 4     | u32  | `name` 的长度（≤ 255） |
| `name`    | `name_len` | bytes | ASCII 张量名，无 NUL |
| `dtype`   | 1     | u8   | `1`=FP16，`2`=INT4_PACKED，`3`=FP16_SCALE |
| `ndim`    | 1     | u8   | `1` 或 `2` |
| `dims`    | `4*ndim` | u32[] | 形状，行优先 |
| `offset`  | 8     | u64  | 相对**数据区起始**的字节偏移 |
| `byte_size`| 8    | u64  | 数据区字节大小 |
| `align`   | 1     | u8   | 要求的对齐（2 的幂，≤ 16） |
| `pad`     | 7     | u8   | 保留，必须为 `0` |

## 数据区（Payload）

张量块按**记录表顺序**排列。张量 *i* 的区域从 `payload_offset + offset_i`
开始，长度 `byte_size_i` 字节。

- `payload_offset` 为 `72 + table_size` **向上**取 16 的倍数（间隙零填
  充）。这保证数据区基址 16B 对齐。
- 每个区域起点 `payload_offset + offset_i` 都是 16 的倍数（写入端对每个
  区域做 16B 对齐；`offset` 相对数据区起始计量）。`byte_size` 必须等于
  `numel(dims) * element_bytes(dtype)`：

| dtype        | 元素字节数 | 形状约定 |
|--------------|--------------:|------------------|
| FP16         | 2             | 逻辑元素数 |
| FP16_SCALE   | 2             | `[N, K/128]` |
| INT4_PACKED  | 1             | `[N, K/2]`（打包字节数） |

## W4A16 配对与校验

每个名为 `X.weight` 的 `INT4_PACKED` 权重必须有配套的名为 `X.scale` 的
`FP16_SCALE`，形状为 `[N, K/128]`，其中 `K = 2 * weight.shape[1]`。
加载器校验：

- magic、version、flags
- 表头自洽（`table_offset`、`payload_offset` 在范围内且有序）
- 每条记录：字段边界、`offset+byte_size ≤ payload_size`、无重叠、区域
  16 字节对齐、遵守 `align`、`byte_size` 与 dtype/形状一致
- 张量名唯一
- config 通过 `ModelConfig::valid()`
- （按需）完整 decoder block 张量集齐且形状正确（见下 `BlockWeights`）

## decoder block 的张量集（v0.1.1 泛化形状契约）

形状一律以 config 字段表达，**不假设 Q 宽度等于 hidden_size**（v0.1.1）。
记 `Q = q_proj_out = n_heads·head_dim`（注意 Q 与 H 是**独立**的标量，
默认 v0.1 配置恰好 Q=H=1024，但契约不要求相等）、
`kv = kv_proj_out = n_kv·hd`：

| 名称                    | dtype        | 形状 |
|-----------------------|--------------|-------|
| `attn_norm.weight`    | FP16         | `[H]` |
| `ffn_norm.weight`     | FP16         | `[H]` |
| `attn.rope_cos`       | FP16         | `[max_seq_len, head_dim/2]` |
| `attn.rope_sin`       | FP16         | `[max_seq_len, head_dim/2]` |
| `attn.q_proj.weight`  | INT4_PACKED  | `[Q, H/2]` |
| `attn.q_proj.scale`   | FP16_SCALE   | `[Q, H/128]` |
| `attn.k_proj.weight`  | INT4_PACKED  | `[kv, H/2]` |
| `attn.k_proj.scale`   | FP16_SCALE   | `[kv, H/128]` |
| `attn.v_proj.weight`  | INT4_PACKED  | `[kv, H/2]` |
| `attn.v_proj.scale`   | FP16_SCALE   | `[kv, H/128]` |
| `attn.o_proj.weight`  | INT4_PACKED  | `[H, Q/2]` |
| `attn.o_proj.scale`   | FP16_SCALE   | `[H, Q/128]` |
| `mlp.gate_proj.weight`| INT4_PACKED  | `[inter, H/2]` |
| `mlp.gate_proj.scale` | FP16_SCALE   | `[inter, H/128]` |
| `mlp.up_proj.weight`  | INT4_PACKED  | `[inter, H/2]` |
| `mlp.up_proj.scale`   | FP16_SCALE   | `[inter, H/128]` |
| `mlp.down_proj.weight`| INT4_PACKED  | `[H, inter/2]` |
| `mlp.down_proj.scale` | FP16_SCALE   | `[H, inter/128]` |

（`H = hidden_size`，`hd = head_dim`，`n_kv = n_kv_heads`，
`Q = q_proj_out = n_heads·head_dim`，`kv = kv_proj_out = n_kv·hd`，
`inter = intermediate_size`。`ModelConfig::valid()` 要求
`Q % group_size == 0` 且 `inter % group_size == 0`，不再要求
`H == n_heads·head_dim`。）

## INT4 量化契约（承自 CUDALab，仅离线）

对称分组量化，`group_size G = 128`，`zero_point = 0`：

```
q[n,k]    ∈ [-7, 7]
scale[n,g] = max_{k∈group g} |W[n,k]| / 7      (fp32，再转 fp16)
q[n,k]    = clamp(round(W[n,k] / scale[n,g]), -7, 7)   (四舍六入五成双)
```

- 零组 → `scale = 0`、`q ≡ 0`（无除零，贡献为 0）。
- 要求 `K % 128 == 0`（写入端与加载器否则均拒绝）。
- **nibble 打包**（每字节两个有符号 INT4，4 位补码）：
  - `W_packed[n, b]` 的低 nibble = 元素 `k = 2b`
  - `W_packed[n, b]` 的高 nibble = 元素 `k = 2b + 1`
- `scale` 以 **fp16** 存储（scale 的 fp16 舍入是数据契约的一部分；
  kernel 与参考实现都读取存储的 fp16 scale）。

量化 + 打包**只**发生在离线 Python 转换器中，绝不在运行时进行。

# CUDALM 黄金参考文件格式（`.cudalm`，CUDLMG01）— v1

黄金容器在一个文件中内嵌运行时复现"某一个解码位置上的一个 decoder
block"所需的一切：18 个 block 权重张量（与同种子下 CUDALMW01 权重文件
字节相同）、解码位置、参考模拟的全部 16 个 stage 张量、以及两个 KV 状态
张量。由 `tools/generate_golden.py`（Python，离线）写入，由
`src/runtime/golden_loader.cpp`（纯 C++，无 PyTorch）读取。

它原样复用 CUDALMW01 的记录格式（TensorRecord、数据区对齐、校验规则），
仅表头不同：**80 字节**而非 72，magic 为 `"CUDLMG01"`，config 数据块后
多两个 32 位字段。

```
+---------+--------------------------------------------------------------+
| 表头    |  固定，80 字节                                                |
+---------+--------------------------------------------------------------+
| 记录表  |  n_tensors × TensorRecord（变长，紧密排列）                    |
+---------+--------------------------------------------------------------+
| 数据区  |  张量字节块，每个 16 字节对齐                                  |
+---------+--------------------------------------------------------------+
```

## 表头（80 字节）

| 偏移 | 大小 | 类型  | 字段             | 取值 / 含义 |
|-------:|-----:|-------|----------------|-----------------|
| 0      | 8    | 8s    | `magic`        | 字节 `"CUDLMG01"` |
| 8      | 4    | u32   | `version`      | `1`（未知 → 硬错误） |
| 12     | 4    | u32   | `flags`        | 保留，必须为 `0` |
| 16     | 4    | u32   | `n_tensors`    | TensorRecord 数量（v0.1 / v0.1.1 为 36） |
| 20     | 36   | —     | `config`       | ModelConfig 数据块（与 CUDALMW01 相同） |
| 56     | 4    | i32   | `position`     | 解码位置 `p`，`0 ≤ p < max_seq_len` |
| 60     | 4    | i32   | `reserved`     | 必须为 `0` |
| 64     | 8    | u64   | `table_offset` | 记录表相对文件起始的偏移（== 80） |
| 72     | 8    | u64   | `payload_offset` | 数据区相对文件起始的偏移 |

`payload_offset` 为 `80 + table_size` **向上**取 16 的倍数（零填充间
隙）；区域对齐与无重叠规则与 CUDALMW01 相同。

## 张量集（36 个，按表顺序）

1. 18 个 block 权重张量，与上面 CUDALMW01 集完全一致（同名、同 dtype、
   同形状、同字节块）。
2. 16 个 stage 张量，全部 **FP16**，全部形状 `[1, X]`：

| 名称 | X |
|------|---|
| `stage.input` | `H` |
| `stage.rmsnorm1` | `H` |
| `stage.q` | `Q` |
| `stage.k` | `kv` |
| `stage.v` | `kv` |
| `stage.rope_q` | `Q` |
| `stage.rope_k` | `kv` |
| `stage.attention_output` | `Q` |
| `stage.output_projection` | `H` |
| `stage.residual1` | `H` |
| `stage.rmsnorm2` | `H` |
| `stage.gate` | `inter` |
| `stage.up` | `inter` |
| `stage.silu_gate_mul_up` | `inter` |
| `stage.down` | `H` |
| `stage.final_output` | `H` |

（`H`、`Q`、`kv`、`inter` 定义同上；`Q = q_proj_out` 独立于
`H = hidden_size`。）

3. 两个 KV 状态张量，FP16，形状 `[n_kv·(position+1), head_dim]`，扁平行
   顺序为 **(kv_head, position)** 行优先 — 即 head `h` 在位置 `t` 的行是
   扁平行 `h·(position+1) + t`：

| 名称 |
|------|
| `kv.k_state`（每位置行为 RoPE 后的 k） |
| `kv.v_state`（每位置行为投影后的 v） |

## 参考数学契约（C++ block 必须镜像的内容）

- 全部算术以 **fp32** 进行；**每个 stage 边界做一次 fp16 转换**（存储的
  stage 张量*本身*就是契约 — kernel 必须产生相同的 stage 粒度与 dtype）。
- W4A16 线性层：`y = (q × scale_fp16_as_fp32) @ x_fp32 → fp16`。**存储的
  fp16 scale** 是契约（不是重新计算的 fp32 `amax/7`）。
- RMSNorm：`x · rsqrt(mean(x²) + eps) · w → fp16`（fp32 累加）。
- RoPE：交错对（CUDALab `rope_v3_half2` 约定）：`y[2i] = a·c − b·s`、
  `y[2i+1] = a·s + b·c`，cos/sin 从 fp16 表按行 `position` 读取。
- 解码注意力作用于缓存行 `0..position`，fp32 数值稳定 softmax，
  `scale = 1/√head_dim`；GQA：query head `h` 读取 kv head
  `h · n_kv_heads / n_heads`。
- `silu_gate_mul_up = silu(gate)·up` 以 fp32 计算 → fp16。
- 残差：两个 fp16 行做 fp32 相加 → fp16。

**KV 历史（p > 0）：** 位置 `0..p−1` 通过让带种子的随机 hidden state
（`0.05·randn(1,H)`，fp16，`history_seed`）流经 attn_norm → k/v 投影 →
RoPE 填充；文件内嵌 `p` 处写入完成后的完整缓存。因此，对**任意**位置，
`position` 处的缓存行与 `stage.rope_k` / `stage.v` 逐位相等（由测试校
验）。

**钉死的 p=0 不变量**（逐位精确，由 `tests/cpu/test_golden_file.cpp`
校验）：cos 第 0 行 == 1 且 sin 第 0 行 == 0 精确成立，因此
`stage.rope_q == stage.q` 与 `stage.rope_k == stage.k` 逐位相等；且单元
素 softmax 恰为 1，因此每个 query head 的 `stage.attention_output` 行与
其 GQA kv head 的 `stage.v` 行逐位相等。

## 比较容差（测试契约）

kernel 与黄金 fp16 行逐 stage 比较，判据为 `|a − r| ≤ atol + rtol·|r|`，
`atol = rtol = 1e-2`（`include/cudalm/stage_compare.h`）— 比 fp16 ulp
（~9.8e-4）松一个数量级以吸收结合顺序差异，又足够严以捕获任何错 stage /
错索引 bug。
