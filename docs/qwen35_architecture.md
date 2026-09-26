# Qwen3.5 架构契约（v0.2）

Qwen3.5-0.8B 混合解码器 bring-up 所用的官方钉死（pinned）源 + 精确数学契约。
本文档是 v0.2 的架构 oracle（事实基准）：本文档与钉死源之间的任何不一致，
意味着**本文档错了**；任务简报（prompt）描述与钉死源之间的不一致，一律以
钉死源为准（已发现的不一致见 §2「与简报的差异」）。

范围：仅**文本**解码器（`Qwen3_5TextModel`，config 键 `text_config`）。
checkpoint 中存在的视觉塔（`model.visual.*`）与 MTP 模块（`mtp.*`）
**不在** v0.2 范围内（已记入 backlog）。

---

## 1. 官方源钉死（pins）

### 1.1 模型 / checkpoint（形状 + 数值的事实源）

| 项 | 值 |
|---|---|
| 模型仓库 | `Qwen/Qwen3.5-0.8B-Base` |
| revision | 分支 `main`，commit `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| config.json sha256 | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| model.safetensors | `model.safetensors-00001-of-00001.safetensors`，1,746,942,600 字节，sha256 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`（同时写入每个转换后的 `.cudalm` v2 文件的 `checkpoint_sha256` 元数据） |
| 权重索引 | `model.safetensors.index.json`，488 个张量，`total_size = 1746882752` |
| 传输通道 | `https://hf-mirror.com`（本环境访问不到 huggingface.co；镜像只作传输通道——事实源仍是官方仓库） |
| 本地路径 | `/root/models/Qwen3.5-0.8B-Base/`（永不提交进 Git） |
| 原生 dtype | `bfloat16`（text_config.dtype）；`mamba_ssm_dtype: float32` |

### 1.2 modeling 源码（数学的事实源）

| 项 | 值 |
|---|---|
| 仓库 | `huggingface/transformers` |
| commit | `fc9137225880a9d03f130634c20f9dbe36a7b8bf` —— "Adding Support for Qwen3.5 (#43830)"，2026-02-09 |
| 钉死文件 | `src/transformers/models/qwen3_5/modeling_qwen3_5.py`（sha256 `b6f02dcd1b66610df293084e00bf9bea4fc6a7e5336ffc6ff446edc7ddcd8601`）、`configuration_qwen3_5.py`（sha256 `2280c6e6bd9d66d7281155243f67cf5fed2756a828af41316566afe611ff16c0`），副本存放于 `/root/models/Qwen3.5-0.8B-Base/provenance/transformers_fc91372/` |

**溯源冲突（已记录，已裁决）：** checkpoint 的 `config.json` 声明
`transformers_version: 4.57.0.dev0`，但 v4.57.0（2025-10-03 发布）时
qwen3_5 modeling 尚不存在——它于 2026-02-09 加入 main（即上面的 commit，
Qwen 贡献，PR #43830）。该版本字符串是导出环境的陈旧残留。钉死的 commit
才是官方实现；golden 生成器**精确安装该 commit** 的 transformers，使
oracle 代码与本文档逐字节一致。

钉死文件自包含（下文所有类都定义在 `modeling_qwen3_5.py` 中；不 import
qwen3_next）。golden 环境跑纯 torch 回退路径（无 `flash-linear-attention`、
无 `causal-conv1d`），即 `torch_chunk_gated_delta_rule` /
`torch_recurrent_gated_delta_rule` / `torch_causal_conv1d_update`——
下文参考方程以此为准。

### 1.3 模型卡（仅作佐证）

仓库 `README.md`：层布局 `6 × (3 × (Gated DeltaNet → FFN) → 1 ×
(Gated Attention → FFN))`，DeltaNet 16 QK 头 / 16 V 头 @ 128，
Gated Attention 8 Q / 2 KV @ 256、RoPE 维 64，FFN 中间维 3584——
与 config.json 全部一致。

---

## 2. 模型配置（来自官方 config.json 的 text_config）

| 键 | 值 |
|---|---|
| model_type | `qwen3_5_text` |
| architectures | `Qwen3_5ForConditionalGeneration`（多模态 wrapper；文本部分 = `model.language_model`） |
| hidden_size | **1024** |
| intermediate_size | **3584** |
| num_hidden_layers | **24** |
| hidden_act | `silu` |
| vocab_size | 248320 |
| tie_word_embeddings | true（不在 v0.2 范围——无 LM head） |
| num_attention_heads | **8**（全注意力） |
| num_key_value_heads | **2**（GQA，group = 4） |
| head_dim | **256**（注意：q_proj 输出 = 8·256·2 = 4096 ≠ hidden——v0.1.1 的「Q 宽度 ≠ H」泛化在这里是真实需求） |
| attention_bias | false（所有投影无 bias） |
| attention_dropout | 0.0 |
| attn_output_gate | **true** |
| full_attention_interval | 4 → 见下文章排表 |
| linear_conv_kernel_dim | **4** |
| linear_num_key_heads | **16** |
| linear_num_value_heads | **16** |
| linear_key_head_dim | **128** |
| linear_value_head_dim | **128** |
| mamba_ssm_dtype | float32（递归状态 dtype） |
| rms_norm_eps | **1e-6** |
| max_position_embeddings | 262144 |
| rope_parameters | `{rope_type: "default", rope_theta: 1e7, partial_rotary_factor: 0.25, mrope_section: [11,11,10], mrope_interleaved: true}` |
| mtp_num_hidden_layers | 1（超范围） |

### 精确层排表（24 层）

```
i:    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
type: DN DN DN FA DN DN DN FA DN DN DN FA DN DN DN FA DN DN DN FA DN DN DN FA
```

`i % 4 != 3` 时 `layer_types[i] = "linear_attention"`（Gated DeltaNet），
`i % 4 == 3` 时为 `"full_attention"`。全注意力层索引：
**{3, 7, 11, 15, 19, 23}**（18 个 DeltaNet 层 + 6 个全注意力层）。

**与 v0.2 简报的差异（官方源优先）：**
- 简报假设模式可能是「DeltaNet×3 → Full Attention」——已由官方 config
  确认（`full_attention_interval=4`，每块最后一层）。
- 简报未提该模型是**多模态** checkpoint
  （`Qwen3_5ForConditionalGeneration`，视觉塔 + mRoPE）。纯文本位置下文本
  解码数学不受影响（mRoPE 退化为普通 RoPE——见 §5），但 checkpoint 摄入
  必须选 `model.language_model.*` 前缀、跳过 `model.visual.*` / `mtp.*`。

---

## 3. v0.1.1 → Qwen3.5 差异表

| # | v0.1.1 现状假设 | Qwen3.5 要求 | 行动 |
|---|---|---|---|
| 1 | RMSNorm `x * rsqrt(mean(x²)+eps) * w`，普通权重，H ∈ {1024,2048,4096,8192} | **零中心** `x.float() * rsqrt(mean(x.float()²)+eps) * (1 + w.float())` → 转回输入 dtype；还用于 head_dim=256（逐头）与最终 norm | **新增** `qwen35_rmsnorm` kernel + 模式；v0.1.1 kernel 保持不动 |
| 2 | Gated RMSNorm 不存在 | DeltaNet 输出 norm：`(w * (x·rsqrt(mean(x²)+eps)).to(bf16)) * silu(gate)`，两个内部边界处为精确 bf16 舍入，w 普通（初值 ones），H=128 | **新增** `qwen35_rmsnorm_gated` kernel |
| 3 | RoPE 交错对（interleaved-pair）、整 head_dim、权重文件内 fp16 cos/sin **表** | **partial rotary**（旋转维 = 64 = 0.25·256，作用于每头的**前** 64 维）、**rotate-half** 布局、theta=1e7、cos/sin 在当前位置由 inv_freq 以 fp32 计算再转 bf16；mRoPE sections [11,11,10]（纯文本位置为 no-op：3 个位置维全相等） | **新增** `qwen35_rope` kernel（按位置现算，无表）；保留 v0.1.1 kernel |
| 4 | 全注意力：普通 q/k/v + RoPE + causal GQA + o_proj | + RoPE **之前**对 q/k 逐头 RMSNorm（零中心，跨 head_dim=256）；q_proj 输出融合 `[q; gate]`（out = 2·n_heads·head_dim）；注意力输出在 o_proj **之前** × sigmoid(gate)；scaling = head_dim^-0.5 = 1/16；softmax 走 fp32 | **新增** `Qwen35FullAttentionLayer`；KV cache 布局 [n_kv_heads, seq, head_dim]（与 v0.1.1 KvCache 同一思路） |
| 5 | 无线性注意力原语 | Gated DeltaNet：in_proj_qkv/z/b/a、depthwise causal conv1d（k=4、depthwise、无 bias）带持久 conv 状态、delta-rule 递推带持久 [16,128,128] fp32 状态、gated RMSNorm、out_proj | **新增** `Qwen35DeltaNetLayer` + `Qwen35DeltaState` |
| 6 | 单一注意力类型、一个 `DecoderBlock` | 混合排布 18×DeltaNet + 6×FullAttention；逐层按类型分发 | **新增** `Qwen35Config` + 混合 micro-stack（v0.2：最少 4 层） |
| 7 | `.cudalm` v1：单 block、fp16 + int4、rope 表 | v2：架构 id、Qwen35Config blob、层排布、任意命名张量表（bf16/fp32 dtype）、边界/唯一名校验；**v1 保持不可变** | **新增** v2 格式；v1 loader 不动 |
| 8 | W4A16 G=128 源权重为 **fp16** | 同一打包契约（q∈[-7,7]、fp16 scale = amax/7、低 nibble = k=2b），但源 dtype 为 **bf16**；scale 仍存 fp16（GEMV 契约不变） | **扩展** 量化器（接受 bf16）；保留打包布局 + GEMV |
| 9 | 每 block 7 个 GEMV（q,k,v,o,gate,up,down） | 全注意力层：同样 7 个（q_proj N=4096！）；DeltaNet 层：in_proj_qkv（N=6144）、in_proj_z（N=2048）、in_proj_b/in_proj_a（N=16）、out_proj（N=1024）。所有 K 维均为 128 的倍数 ✓ | **扩展** GEMV 使用（N=16 行合法：K=1024） |
| 10 | 仅 KV cache | + 每个 linear 层增加 DeltaNet conv 状态 [conv_dim=6144, 3] + 递归状态 [16, 128, 128] fp32 | **新增** 状态归属（ownership） |
| 11 | 默认 eps=1e-5 | eps=**1e-6** | **扩展** 经 config 传入（v0.1.1 常量不动） |
| 12 | SwiGLU silu、无 bias | 相同（silu、无 bias） | **保留** |

---

## 4. Checkpoint 张量映射

语言模型张量剥离前缀：`model.language_model.`（checkpoint 是多模态
`Qwen3_5ForConditionalGeneration`）。共 488 个张量；v0.2 摄入 **187** 个
（见逐层清单），跳过 299 个视觉 + 8 个 MTP 张量（backlog：完整模型）。

### 逐层张量

全注意力层 `i ∈ {3,7,11,15,19,23}`（`self_attn.*`，全 bf16、无 bias）：

| 张量 | 形状 | 备注 |
|---|---|---|
| `self_attn.q_proj.weight` | [4096, 1024] | 输出行 `h*512+j`：j<256 → 查询头 h，j≥256 → **gate** 头 h |
| `self_attn.k_proj.weight` | [512, 1024] | 2 KV 头 × 256 |
| `self_attn.v_proj.weight` | [512, 1024] | |
| `self_attn.q_norm.weight` | [256] | 跨 head_dim 的零中心 RMSNorm |
| `self_attn.k_norm.weight` | [256] | |
| `self_attn.o_proj.weight` | [1024, 2048] | |

DeltaNet 层 `i ∉ {3,7,11,15,19,23}`（`linear_attn.*`）：

| 张量 | 形状 | 备注 |
|---|---|---|
| `linear_attn.in_proj_qkv.weight` | [6144, 1024] | conv 输入 = 2·key_dim + value_dim = 3·2048 |
| `linear_attn.in_proj_z.weight` | [2048, 1024] | 输出 norm 的 gate |
| `linear_attn.in_proj_a.weight` | [16, 1024] | 衰减率输入 |
| `linear_attn.in_proj_b.weight` | [16, 1024] | beta（更新门）输入 |
| `linear_attn.conv1d.weight` | [6144, 1, 4] | depthwise causal conv、无 bias；保留 **bf16**（非 GEMV） |
| `linear_attn.dt_bias` | [16] | 保留 bf16 |
| `linear_attn.A_log` | [16] | 官方 checkpoint 中为 **fp32**（已在 safetensors header 验证，全部 18 个 DeltaNet 层一致）；v2 直通 fp32；官方代码计算 `A_log.float().exp()`，故衰减数学无论如何都是 fp32 |
| `linear_attn.norm.weight` | [128] | gated RMSNorm 权重（普通、初值 ones）；官方 checkpoint 中为 **fp32**；v2 直通 fp32 |
| `linear_attn.out_proj.weight` | [1024, 2048] | |

每一层（两种类型都有）：

| 张量 | 形状 |
|---|---|
| `input_layernorm.weight` | [1024]（零中心） |
| `post_attention_layernorm.weight` | [1024]（零中心） |
| `mlp.gate_proj.weight` | [3584, 1024] |
| `mlp.up_proj.weight` | [3584, 1024] |
| `mlp.down_proj.weight` | [1024, 3584] |

模型级（为文本模型的完整性而摄入；v0.2 的 4 层 micro-stack 不用，但摄入时
校验）：

| 张量 | 形状 |
|---|---|
| `embed_tokens.weight` | [248320, 1024]（bf16；不在 GEMV 范围） |
| `norm.weight` | [1024]（最终零中心 RMSNorm） |

**跳过的（v0.2 backlog）：** 全部 `model.visual.*`（299 个张量）与 `mtp.*`
（8 个张量：`mtp.layers.0.*`、`mtp.fc`、`mtp.norm`、`mtp.pre_fc_norm_*`）。

### W4A16 量化目标（v0.2）

每层全部 12 个投影 GEMV（全注意力层 7 个、DeltaNet 层 5 个）→ 打包 int4 +
fp16 scale，G=128 对称，源 dtype bf16：

| GEMV | N | K | K%128 |
|---|---|---|---|
| q_proj | 4096 | 1024 | ✓ |
| k_proj / v_proj | 512 | 1024 | ✓ |
| o_proj | 1024 | 2048 | ✓ |
| gate_proj / up_proj | 3584 | 1024 | ✓ |
| down_proj | 1024 | 3584 | ✓ |
| in_proj_qkv | 6144 | 1024 | ✓ |
| in_proj_z | 2048 | 1024 | ✓ |
| in_proj_b / in_proj_a | 16 | 1024 | ✓ |
| out_proj (deltanet) | 1024 | 2048 | ✓ |

非 GEMV 的 bf16 张量（layernorms、q/k_norm、conv1d、dt_bias、
embed_tokens）：在 v2 文件中以 bf16 原样保存。有两个张量在官方 checkpoint
中以 fp32 存储——`linear_attn.A_log` [16] 与 `linear_attn.norm.weight`
[128]（逐头 gated-norm 权重）——在 v2 文件中同样以 fp32 原样保存。（dtype
事实源读自 checkpoint 的 safetensors header，24 层全部一致。）

量化（扩展 v1 契约，bf16 源）：`scale32 = amax(group)/7`（fp32）；
`q = clamp(round-half-to-even(W/scale32), -7, 7)`；零组 → scale=0、q=0；
**scale 以 fp16 存储**（与 v1 相同，存储的 fp16 scale 即契约）。
反量化参考 = `q · scale_fp16.as_f32`。

---

## 5. RoPE 契约（Qwen3.5，仅全注意力层）

出自钉死的 `Qwen3_5TextRotaryEmbedding` + `apply_rotary_pos_emb`：

- `rope_type = "default"`，`rope_theta = 1e7`，`partial_rotary_factor = 0.25`
- `head_dim = 256` → **rotary_dim = 64**；j = 0..31 时
  `inv_freq[j] = 1e7^(-2j/64)`（fp32）
- 解码位置 p（纯文本）：`freqs[j] = p · inv_freq[j]`（fp32）
- **布局 = rotate-half（不是 v0.1.1 的交错对）：**
  `emb = [freqs(32), freqs(32)]`（64 = 2×32）；
  `x_rot = x[0:64]`；`q' = q_rot·cos + rotate_half(q_rot)·sin`，其中
  `rotate_half(z) = [-z[32:64], z[0:32]]`；64..255 维原样直通
- **逐元素频率映射（已锁定）：** 旋转部分的元素 `d`（0..63）使用
  `freqs[d % 32]`，并与跨半边的元素 `d ± 32` 配对——这正是
  `emb = cat(freqs, freqs)` 的复制式布局。它**不是** `(2j, 2j+1)` 共享
  `freqs[j]` 的交错约定。p=0 恒等式在两种约定下都成立；只有 p>0 的
  golden 门能区分二者（交错索引正是在那里被抓住，并在 Phase B 签核前
  修掉于 `qwen35_partial_rope_bf16`）
- cos/sin 以 **fp32** 计算，**乘法之前转成 bf16**（官方：
  `cos.to(dtype=x.dtype)`）；q·cos + rot·sin 的加法在 bf16 中进行
- mRoPE：纯文本解码时 `position_ids` 为 `[3, bs, seq]` 且三个维都等于 p，
  故 `apply_interleaved_mrope` 在数学上是 no-op（T/H/W 三向 freqs 相同）。
  v0.2 运行时只实现文本特化；视觉网格在 backlog。
- 注意力缩放：`head_dim ** -0.5 = 1/16`（与 RoPE 无关）。

---

## 6. RMSNorm 契约

### 6.1 零中心 RMSNorm（`Qwen3_5RMSNorm`）——layernorm + q/k norm + 最终 norm

```
y = (x_f32 · rsqrt(mean(x_f32²) + eps) · (1 + w_f32)).to(input_dtype)
```
- `w` 是**零中心**权重（checkpoint 值；初值 zeros）
- `eps = 1e-6`
- 使用位置：`input_layernorm` [1024]、`post_attention_layernorm` [1024]、
  `q_norm`/`k_norm` [256]（逐头、最后一维）、最终 `norm` [1024]
- **v0.1.1 RMSNorm 语义（`x·rsqrt(·)·w`、普通权重）不变**——只新增
  kernel/模式。

### 6.2 Gated RMSNorm（`Qwen3_5RMSNormGated`）——仅 DeltaNet 输出

精确官方 dtype 流，x 与 gate 的最后一维为 128：

```
1. n = (x.to(f32) · rsqrt(mean(x.to(f32)²) + eps)).to(bf16)
2. a = w · n                                # w = norm.weight（存 fp32）
3. y = (a · silu(gate.to(f32))).to(bf16)
```
`w` 在官方 checkpoint 中以 **fp32** 存储，因此官方 torch 代码的第 2 步会
提升为 fp32（唯一内部舍入是第 1 步的 bf16 cast）；若 loader 把 `w` cast
成 bf16，第 2 步就会变成 bf16 乘法。Phase B 的 golden 参考钉死了运行时
必须匹配哪一种情况（v2 文件无论如何都把 `w` 以 fp32 逐字节保留）。
CUDA kernel 必须精确镜像钉死流的各舍入点，而不只是 fp32 理想值。

---

## 7. 全注意力层契约（decode 步，位置 p）

所有投影无 bias；模型 dtype bf16。输入 `x ∈ R^{1024}`（bf16、batch-1、
单 token）。

```
h0  = zero_rmsnorm(x, input_layernorm.w)                    # bf16
q_g = W_q @ h0                                              # [4096] = 逐头交错
      (rows h*512..h*512+255 = q_h, h*512+256..h*512+511 = gate_h, h=0..7)
q_h = zero_rmsnorm_over_head(q_h, q_norm.w)   per head h     # 跨 256
k_h = zero_rmsnorm_over_head(k_h, k_norm.w)   per head (2 kv heads)
v_h = W_v @ h0                                          # [2·256]
q'_h, k'_h = partial_rotate_half_rope(q_h, k_h, p)      # §5，前 64 维
KV cache: K[p] ← k', V[p] ← v                            # 每个 kv 头 [2, p+1, 256]
attn_h: s = (q'_h · K_kv[h//4][0..p]) / 16 ; softmax_fp32 ; out = w · V_kv[...]   # [256]
out = Σ heads concat → [2048]
out = out ⊙ sigmoid(gate)                                 # gate 来自 q_g（bf16 sigmoid）
attn_out = W_o @ out                                      # [1024]
res1 = x + attn_out                                       # bf16 加法
h1  = zero_rmsnorm(res1, post_attention_layernorm.w)
mlp = W_down @ (silu(W_gate @ h1) ⊙ (W_up @ h1))          # SwiGLU、无 bias
y   = res1 + mlp
```

状态：仅 KV cache，K 与 V 布局 `[n_kv_heads=2, seq, 256]`（bf16）。

## 8. Gated DeltaNet 层契约（decode 步，位置 p）

维度：`key_dim = 16·128 = 2048`、`value_dim = 16·128 = 2048`、
`conv_dim = 6144`。所有投影无 bias。模型 dtype bf16。

```
mixed = W_qkv @ x            # [6144]
z     = W_z   @ x            # [2048] = 16 头 × 128（norm 的 gate）
b     = W_b   @ x            # [16]
a     = W_a   @ x            # [16]

# depthwise causal conv1d、kernel 4、无 bias —— DECODE 时带 conv 状态 cs[6144][3]：
buf   = [cs(3), mixed(1)]    # [6144][4]，bf16
cs    = buf[:, -3:]          # 状态更新（就地）
c     = Σ_{j=0..3} W_conv[·,j] · buf[:,j]   (per channel; bf16)
mixed = silu(c)

q,k,v = split(mixed, [2048,2048,2048]) → reshape [16,128] each
beta  = sigmoid(b)           # bf16 [16]
g     = -exp(A_log_f32) · softplus(a_f32 + dt_bias_f32)   # fp32 [16]

# delta-rule 递推，状态 S ∈ R^{16×128×128} 保持 FP32：
for each head h (independent):
  q_h = l2norm(q_h, eps=1e-6); k_h = l2norm(k_h, eps=1e-6)   # bf16（见下）
  q_h = q_h / sqrt(128)
  S_h = S_h · exp(g_h)
  m   = S_h · k_h                       # [128]（对 key 维收缩）
  d   = (v_h - m) · beta_h
  S_h = S_h + k_h ⊗ d                   # 外积
  o_h = S_h · q_h                       # [128]   # 注意：输出用的是更新后的 S

core = concat o_h → [2048] → [16,128]
out  = gated_rmsnorm(core, gate=z, w=norm.w)    # §6.2
y    = W_out @ out                              # [1024]
```

`l2norm(x) = x * inv_norm`，其中 `inv_norm = torch.rsqrt((x*x).sum(dim,
keepdim=True) + eps)`——这是 pinned torch 回退路径的 FLA 对齐 `l2norm`
（§1.2），**在 bf16 上进行，不是纯 fp32**（早期本文的 `# fp32` 表述有误，
以钉死源为准）。本实现逐位精确镜像的舍入序列为：`prod = bf16(x·x)`；
`ss = fp32(Σ prod)`（bf16 乘积的 fp32 求和）；`t = bf16(ss + eps)`；
`inv = bf16(rsqrt_bf16(t))`——注意 torch 对 **bf16 张量**的 `rsqrt` 是
`bf16(1.0 / bf16(sqrt(f32(t))))`，即 fp32 开方舍入到 bf16、再做 fp32 倒数
舍入到 bf16（**两次** bf16 舍入，而非单次 fp32 rsqrt）；最后
`out = bf16(x · inv)`。

**状态（每个 DeltaNet 层，显式归属）：**
- `conv_state`：bf16 `[6144, 3]`（qkv 流最后 kernel-1=3 个 token）
- `recurrent_state`：**fp32** `[16, 128, 128]`（head、key_dim、value_dim）
- 初始状态 = 全零（首 token）；两者在每个 decode 步就地更新

**运算顺序不变式（golden 验证）：** 衰减 → delta 更新 → 用更新后的状态算
输出；conv 状态在读取 conv 输出**之前**更新；g 以 fp32 计算；A_log 在
checkpoint 中存 fp32（直接使用，官方代码里的 `A_log.float()` 是 no-op）；
dt_bias 存 bf16 → cast 成 fp32。

---

## 9. Decoder 层 / micro-stack 接线

标准 pre-norm（两种层类型）：

```
res1 = x + mixer(zero_rmsnorm(x, input_layernorm))
y    = res1 + mlp(zero_rmsnorm(res1, post_attention_layernorm))
```

**混合 micro-stack（v0.2 顶层目标）：** 真实 checkpoint 的 0、1、2 层
（DeltaNet）+ 第 3 层（第一个 Full Attention），decode 步 p = 0,1,2,…，
batch-1 token。输入：带种子的随机 bf16 hidden states [1, 1, 1024]
（embedding 超范围——已记 backlog）；第 p 步用位置 p。逐步骤验证：每层
输出、全注意力 KV 状态、DeltaNet conv + 递归状态、micro-stack 最终输出——
对照钉死 commit 的官方 transformers forward（换入量化权重，见 §11）。

---

## 10. .cudalm v2 规划

v1（`CUDLMW01`）不可变。v2（`CUDLMW02`）新增：

- magic `CUDLMW02`，version u32 = 2
- `Qwen35Config` 的固定 config blob（上述维度 + eps + rope_theta +
  partial_rotary_factor + mrope_section[3] + full_attention_interval +
  group_size + max_seq_len）
- 架构 id 字符串（`qwen35-text`）；模型 repo/revision 与 transformers
  commit + 文件 hash 放在 metadata TLV 段（溯源随权重走）
- 任意命名张量表：name（≤255B、唯一）、dtype ∈ {fp16, bf16, fp32,
  int4_packed, fp16_scale, int8}、ndim ≤ 8、dims i64、offset/byte_size/
  align；payload 16B 对齐；完整边界校验
- **无** RoPE 表（cos/sin 按位置现算——§5）
- 张量命名：官方 checkpoint 名加 `model.language_model.` 前缀（如
  `layers.3.self_attn.q_proj.weight`、`layers.0.linear_attn.conv1d.weight`、
  `layers.0.input_layernorm.weight`，以及量化对
  `*.weight`→int4 + `*.scale`→fp16）——不搞巨型 C++ switch；loader 按
  命名表解析。

转换器：`tools/convert_qwen35.py`（离线；可用 Python/safetensors/torch）
读官方 config.json + safetensors，套用共享的 bf16→W4A16 量化器，写出 v2。
运行时（`include/`+`src/`）保持无 torch。

## 11. Golden 策略

`tools/generate_qwen35_golden.py`（离线）：

1. 安装/使用钉死在 `fc9137225880` 的 transformers（oracle 代码）
2. 加载真实 checkpoint（bf16），取 `model.language_model`
   （`Qwen3_5TextModel`）
3. 把每层 12 个 GEMV 权重换成**反量化 W4A16** 值（与转换器同一共享量化器）
   →「量化参考」
4. 用历史 token（带种子的随机 bf16 hidden states）跑官方 forward 建立真实
   KV + conv + 递归状态，再跑位置 p 的 decode 步
5. hook 捕获：层输入/输出、norm 输出、投影输出、norm 后的 q/k、RoPE 后、
   注意力 gate/输出、DeltaNet conv 状态 / g / beta / 递归状态（前后）、
   FFN 输出、最终输出、KV 状态
6. 写出 `CUDLMG02` golden 容器（见下）

**A/B 切分（硬门 vs 报告）：**
- A. 运行时正确性：CUDALM 运行时（量化权重 + golden 播种的状态）vs 本
  golden —— **硬门**
- B. 量化保真度：官方 bf16 权重 vs 量化参考（每权重张量 + 层输出的
  max_abs、RMSE、cosine）——**仅报告**，写入同目录 `.fidelity.json`，
  绝不混进运行时误差数字

golden 内记录溯源：repo、revision、transformers commit、transformers 版本、
输入种子、位置、dtype、fast-path=off。

### 11.1 CUDLMG02 容器（已锁定）

`magic "CUDLMG02"` | u32 version=1 | u32 flags=0 | u32 n_tensors |
u32 _pad | 88 B Qwen35Config blob | i32 position | i32 layer_idx |
i32 input_seed | i32 _reserved | u64 table_offset (=144) |
u64 payload_offset（16 B 对齐）。表**区域**为 `[table_offset,
payload_offset)`：n 条记录占据其前部，其余字节为零填充（与 v2 权重容器
不同，表头**没有** `table_size` 字段；读者恰好解析 n 条记录，并要求其余
字节为零）。每条记录：`name_len u8 | name | dtype u8 | ndim u8 |
pad u16 | dims i64[8] | offset u64 | byte_size u64 | align u8 | pad u8`。
payload 张量 blob 按表序排列，偏移自 `payload_offset` 起，每个 16 B 对齐。

张量（23 个 stage 张量 + KV 状态），除注明外全为 bf16：
`stage.{input,rmsnorm1,q_gate,q,att_gate,k,v,q_norm,k_norm,rope_q,rope_k,
attention_raw,attention_gated,o_proj,residual1,rmsnorm2,mlp_gate,mlp_up,
silu_mul,mlp_down,final_output}`，另加 `kv.k_state` / `kv.v_state`，形状
`[n_kv*(position+1), head_dim]`、行序 `(kv_head, position)`——历史行正是
运行时播种其 cache 的来源，因此 KV round-trip 按位精确（bit-exact）校验。

**Oracle 舍入契约**（运行时 1:1 镜像，这正是硬门在 p=0 位级精确、p>0 在
1 个 bf16 ulp 内的原因）：GEMV = fp32 累加 → 一次 bf16 RNE；零中心
RMSNorm = 全 fp32 链 → 一次 bf16 cast；RoPE = cos/sin 先 fp32→bf16 再
参与乘法，之后逐元素三次 bf16 舍入 `bf16(x·cos_b)`、`bf16(rot·sin_b)`、
`bf16(sum)`；注意力 = bf16 QK matmul（fp32 acc → bf16）× `1/16`（bf16）
→ fp32 减最大值 softmax → bf16 → bf16 PV（fp32 acc → bf16）；silu = fp32
`x/(1+exp(-x))` **先**舍入到 bf16、再做 bf16 乘法；sigmoid 同理。oracle
里的 QuantLinear 是 fp32 反量化 matmul → 一次 bf16 cast，因此量化保真度
被排除在硬门之外。

**p=0 不变式（位级精确）：** rope_q == q_norm 且 rope_k == k_norm
（位置 0 的 RoPE 相对其输入是恒等，而非相对原始 q/k）；attention_raw
每个头 h 的行 == KV 头 `h // num_key_value_groups` 的 v 行（probs 恰为
[1]）；KV 行 == stage 拷贝。

**Stage 容差：** C++ `compare_bf16_stages` 用 atol=rtol=1e-2。oracle 与
运行时在每次舍入后相差 ~1 个 bf16 ulp；该容差吸收 GEMV/注意力 matmul
内部的 fp32 结合序差异与 libm 对 torch `cosf/sinf` 的 ≤1-ulp 差异，
同时仍能抓住任何 O(1) 的接线或 dtype 错误。见
`include/cudalm/stage_compare.h`。

## 12. 里程碑

- **A** —— 本文档 + `Qwen35Config` + `.cudalm` v2 + `convert_qwen35.py` +
  摄入测试（名/形状/dtype/数值/排布/round-trip）【已完成】
- **B** —— `Qwen35FullAttentionLayer`（+ 共享的 rmsnorm / rope /
  gated-norm kernel）+ 真实 checkpoint golden PASS（p=0、p>0、非零 KV
  历史）【已完成，见 §14】
- **C** —— `Qwen35DeltaNetLayer` + 状态转移 golden PASS（首 token、连续
  token、非零前一状态；输出**和** `conv_state` **和** `recurrent_state`
  都比对）【已完成，见 §15】
- **D** —— 4 层混合 micro-stack golden PASS（逐层 + 全部状态 + 最终输出），
  顺序 p=0,1,2,…；benchmark 三视图；compute-sanitizer 干净；文档/证据
  更新；推送 `v0.2-qwen35`【已完成，见 §16】

## 13. 架构风险

| 风险 | 缓解 |
|---|---|
| golden 环境中 sm_75（Turing）上的 bf16 conv1d | 尽早测试；若 GPU 上 cuDNN bf16 conv 失败，改在 CPU bf16 上跑 oracle，或把 conv 隔离在 fp32 并按官方舍入点处理（仅在不可避免时记录为偏差） |
| `@use_kernelized_func` 装饰器把 RoPE 路由到别处 | golden 环境无 kernel 库 → 走纯函数路径；golden 生成时断言回退路径已激活 |
| gated RMSNorm / g / beta 的 bf16 舍入细节 | golden 以模型 dtype 捕获逐级值；kernel 精确镜像舍入序列（逐 stage 验证，而非只看最终输出） |
| 版本字符串冲突（4.57.0.dev0 vs modeling 2026-02 才加入） | 已记录于 §1.2；pin = 官方 PR commit，而非版本字符串 |
| conv 状态形状（kernel-1 = 3）vs docstring 的 `d_conv` | 钉死源里的 docstring 不精确；以 `torch_causal_conv1d_update` 的经验布局为准：[conv_dim, 3] |
| q_proj 融合 [q;gate] 布局误读 | 钉死源：`view(-1, head_dim*2)` 后 `chunk(2, dim=-1)` → 每头先 q 后 gate；已对照 checkpoint 形状 4096 验证 |
| 量化 vs bf16 参考之间的状态播种 | golden 用与运行时**相同**的量化权重建立状态（无混合状态污染） |

---

## 14. Phase B 完成记录（Qwen3.5 全注意力）

Phase B **已完成**并通过签核验证。范围 = 一个全注意力层
（layers 3/7/11/15/19/23）的 Qwen3.5 专属 BF16 运行时路径、真实
checkpoint 权重、真实 checkpoint golden 在 p=0 及带非零 KV 历史的 p>0
均 PASS。DeltaNet（Phase C）现已**完成**（见 §15）；混合 micro-stack
（Phase D）现已**完成**（见 §16）。

### 14.1 落地内容

- `int4_gemv_bf16`（`.h`/`.cu`）——v0.1 `int4_gemv` rowtile4 kernel 的
  移植，带 BF16 激活/输出路径（W4A16 G=128、q∈[-7,7]、FP16 scale、FP32
  累加、一次 bf16 RNE 存储）。CUDA 11.8 没有 `__bfloat1622float2`，故使用
  `bf162_to_float2` 辅助函数。
- `qwen35_kernels`（`.h`/`.cu`）——原生 BF16 kernel：`qwen35_rmsnorm_zc_bf16`
  （零中心 `x·(1+w)`）、`qwen35_split_q_gate_bf16`、`qwen35_partial_rope_bf16`
  （复制式频率 §5，fp32 cos/sin 表 → 乘法处转 bf16）、`qwen35_kv_write_bf16`、
  `qwen35_attention_decode_bf16`（scores/softmax/PV，bf16 matmul 配 fp32
  累加、精确 `1/16` 缩放），以及 `qwen35_add_bf16` /
  `qwen35_silu_mul_bf16` / `qwen35_gate_mul_bf16`。
- `Qwen35KvCache`（`.h`/`.cpp`）——BF16 `[n_kv, max_seq, head_dim]`
  cache，K = RoPE 后行、V = 原始行；`k_mut()/v_mut()` 供测试播种。
- `Qwen35FullAttentionLayer`（`.h`/`.cpp`）——21 个 stage 的 decode 层，
  带 `forward` / `forwardTimed`（21 对 stage 事件 + 整层一对），持有
  KV cache + 各 buffer + fp32 rope 表（宿主 libm，与 CPU golden 位级一致）。
- `golden_loader_v2`（`.h`/`.cpp`）+ `tools/common/golden_v2.py` ——
  CUDLMG02 的 C++/Python 一对（§11.1），逐字节 round-trip。
- `tools/generate_qwen35_golden.py` —— oracle（钉死 transformers、真实
  checkpoint、共享量化器）+ `--selftest`（合成权重、无需 checkpoint）+
  `--fidelity-report`。

### 14.2 签核证据（本次运行）

- `test_qwen35_full_attention_golden`（layer 3，种子 20260209）：
  - p=0：21 个 stage 全 OK，**最差 max_abs_err = 6.1e-05**，p=0 不变式
    位级精确，KV pos 行拷贝不变式 OK。
  - p=5（5 行非零 KV 历史）：21 个 stage 全 OK，
    **最差 max_abs_err = 4.9e-04**，KV 历史位级精确，pos 行不变式 OK。
- kernel 单元测试：`test_int4_gemv_bf16` 与 `test_qwen35_kernels` 全部
  位级精确 / 0 错误（RoPE 对精确舍入 CPU 参考、注意力 p=0 == v 行、
  p=4 0 错误）。
- 完整 `ctest`：**27/27 PASS**（旧 v0.1/v0.1.1 回归 + Phase A 摄入 +
  Phase B 单元 + Phase B golden）。
- `compute-sanitizer --tool memcheck`：**0 错误**——kernel + GEMV 单元
  测试，以及经 bench 在 p=0 与 p=5 跑的全层（**bench 的 KV 历史为
  zero-initialized**，与 golden 测试 C++ 阶段同一内存面；证据 + golden 测试
  启动器挂起说明见 `benchmarks/sanitizer_qwen35_full_attention.txt`）。
- `scripts/check_no_torch.sh`：**CLEAN**（include/、src/ 无 torch/pybind）。
- 量化保真度（仅报告，非硬门）：逐权重 cosine ≈ 0.991–0.993，层输出
  cosine ≈ 0.982（`build/data/qwen35_golden_l3_p0.cudalm.fidelity.json`）。

### 14.3 时延（RTX 2080 Ti，sm_75，100 次迭代，仅报告）

存档于 `benchmarks/results/`：

| 位置 | whole_layer_gpu_us（均值） | layer_steps_per_second_mean |
|---|---|---|
| p=0  | ≈ 205.6 | ≈ 4865 |
| p=5  | ≈ 208.2 | ≈ 4803 |

> **时延语义：** `whole_layer_gpu_us` 是一个 **CUDA-event device-timeline
> 区间**（整次 forward 的一对 event），包含该区间内 enqueue 到 GPU 的工作
> （含 stage 之间的 RoPE 表 H2D 拷贝），并可能包含 host enqueue gap 导致的
> device idle；它**不是**对 host 侧 CPU 时间的直接测量（host CPU wall 时间
> 归入 `host_api_wall_us`）。bench 的 p=5 行使用 **zero-initialized** KV
> 历史（只覆盖 p>0 的寻址/访存面，不播种非零历史）。

Phase B 未做任何性能调优（设计上超范围）；这些数字是 Phase D benchmark
视图的已验证正确性基线。

---

## 15. Phase C 完成记录（Qwen3.5 Gated DeltaNet）

Phase C **已完成**并通过硬门验证。范围 = 一个 Gated DeltaNet 层
（非全注意力层，本验证取 layer 0）的 CUDALM 原生解码运行时路径、真实
checkpoint 权重、真实 checkpoint golden 在三种状态转移场景下均 PASS。

### 15.1 落地内容

- `qwen35_deltanet_kernels`（`.h`/`.cu`）——DeltaNet 解码 kernel：
  `deltanet_conv_kernel`（depthwise causal conv decode，**位级精确**：
  fp32 累加 → 一次 bf16 RNE，再对 bf16 结果做 SiLU；conv_state 为 bf16
  位级 shift+insert）、`deltanet_gbeta_kernel`（g 以 fp32、beta 以 bf16）、
  `deltanet_delta_kernel`（FLA 对齐的 **bf16 l2norm**（§8 修正）+ fp32
  recurrent 状态就地递推，输出取自更新后的状态）、
  `deltanet_gated_rmsnorm_kernel`（gated RMSNorm，两处 bf16 舍入）。
- `Qwen35DeltaNetLayer`（`.h`/`.cpp`）——layer 0 解码层，复用冻结的
  Phase A/B W4A16 GEMV / 零中心 RMSNorm / add / silu-mul；持有持久状态
  （`conv_state` bf16 `[6144,3]`、`recurrent_state` fp32 `[16,128,128]`），
  每步就地更新，提供 `reset_state` / `seed_state`。
- **supported-config contract（本阶段硬化）**：`Qwen35DeltaNetLayer` 构造
  前经 `qwen35_deltanet_require_supported_config` 校验；非法配置
  （`lin_conv_kernel_dim != 4`、`linear_conv_state_len() != 3`、
  `lin_key_head_dim != lin_value_head_dim`、`lin_key_head_dim != 128`、
  `lin_num_k_heads != lin_num_v_heads`、非线性注意力层）一律 abort——
  因 kernel 固定 kernel-4 / state-3 / head-dim-128 / k==v，避免对其它合法
  `Qwen35Config` 静默 OOB / 语义错误。契约测试 `test_qwen35_deltanet_config`
  （CPU，子进程 abort-check）。
- `tools/generate_qwen35_golden.py` ——DeltaNet 层 golden 生成（**同一组**
  W4A16 量化权重，无 bf16/量化状态混用）+ `--state-seed`（确定性非零
  初始状态）。
- `test_qwen35_deltanet_golden` ——三场景硬门（§15.2）+ `--no-gen`
  （memcheck 用的 CUDA-only 路径）。

### 15.2 状态转移硬门（三场景，均比对 输出 + conv_state + recurrent_state）

oracle 与运行时使用**同一组**量化权重（`build_quantized_deltanet_layer`
把 8 个 GEMV 换成与运行时相同的 W4A16 反量化）。三种场景：

- **A 首 token**：conv_state 与 recurrent_state 全零，跑 p=0；
- **B 连续**：运行时从**自身的** p=0 状态链到 p=1（p0→p1），不重置；
- **C 非零初始状态**：以确定性非零 conv/recurrent 状态播种，跑 p=0。

每种场景比对最终输出**以及** `conv_state` **以及** `recurrent_state`
（连同 22 个 bf16 stage）。**容差：** 22 个 bf16 stage 用
`compare_bf16_stages`（atol=rtol=1e-2）；`stage.g`（fp32）与
recurrent_state 用紧 fp32 容差（atol=1e-5、rtol=1e-4）。l2norm 与 causal
conv 已位级精确复刻（对 pinned torch op 逐元素验证 mismatch=0），故残余仅
为 fp32 递推收缩的结合序（本实现按线程循环 vs torch `.sum`），实测 ≤ 6e-8。

### 15.3 签核证据（本次运行）

- `test_qwen35_deltanet_golden`（layer 0）三场景全 PASS：
  - **A**：输出 bit-exact（worst bf16 max_abs_err = 0），`recurrent_state`
    max_abs = 0；
  - **B**：worst bf16 = 0.000488（1 ulp），`recurrent_state` max_abs =
    5.96e-08；
  - **C**：worst bf16 = 0.000244（1 ulp），`recurrent_state` max_abs =
    7.45e-09。
- 完整 `ctest`：**29/29 PASS**（旧 v0.1/v0.1.1 回归 + Phase A 摄入 +
  Phase B 单元 + Phase B golden + **Phase C 契约 + Phase C golden**）。
- `compute-sanitizer --tool memcheck`：**0 错误**（DeltaNet golden，
  `--no-gen` CUDA-only 路径；证据 `benchmarks/sanitizer_qwen35_deltanet.txt`）。
- `scripts/check_no_torch.sh`：**CLEAN**（include/、src/ 无 torch/pybind）。

Phase C 未做性能调优（设计上超范围）；不跑 benchmark / NCU（Phase D 范围）。

**Phase C 后 STOP** —— Phase C（Gated DeltaNet）已完成（见 §15）并推送
`v0.2-qwen35`；等待外部 review，**不要自动开始 Phase D**（4 层混合
micro-stack / benchmark / NCU）。

---

## 16. Phase D 完成记录（Qwen3.5 4 层混合 micro-stack）

Phase D **已完成**并通过硬门验证。范围 = 真实 checkpoint **前 4 层**
（layer 0/1/2 = Gated DeltaNet，layer 3 = 全注意力）构成的 **v0.2 最终
混合 decoder micro-stack** 的 CUDALM 原生解码运行时、逐层 dispatch、独立
持久状态、真实 checkpoint golden（首 token + 顺序多 token）+ baseline
benchmark。**复用**冻结的 Phase B/C 单层运行时（`Qwen35FullAttentionLayer` /
`Qwen35DeltaNetLayer`）——不复制任何单层 kernel，不改冻结数学语义。

### 16.1 落地内容

- `Qwen35HybridMicroStack`
  （`include/cudalm/qwen35_hybrid_microstack.h` +
  `src/runtime/qwen35_hybrid_microstack.cpp`）——4 层混合 micro-stack：
  - `load(file, stream, out)`：从解析后的 v2 文件加载真实 checkpoint
    layers 0-3（逐层 `validate_layer`），按 config 的 hybrid 排表 dispatch
    到 `Qwen35DeltaNetLayer`（0/1/2）/ `Qwen35FullAttentionLayer`（3）。
  - `forward(position, x_in, stream)`：链式跑 layer 0→1→2→3，把每层
    final output 喂给下一层；micro-stack 输出 = layer 3 final output
    （device bf16 `[1024]`）。一个 decode step = 一个 bf16 hidden `[1024]`。
  - `reset_state(stream)`：逐层独立重置（DeltaNet 零 conv+recurrent；
    全注意力零 KV cache）。**无跨层状态 alias/reuse/contamination**——每层
    持有自己的持久状态（DeltaNet `conv_state` bf16 `[6144,3]` +
    `recurrent_state` fp32 `[16,128,128]`；全注意力 K/V cache）。
  - `forwardTimed(...)`：4 对 per-layer 事件 + 1 对整 stack 事件（benchmark）。
  - 逐层访问器 `delta(i)` / `attention(i)`（golden 测试比对逐层 stage/状态）。

- `tools/generate_qwen35_golden.py` ——新增 `--microstack-prefix` /
  `--tokens` 模式（`generate_microstack`）：把真实 checkpoint layers 0-3
  按 0→1→2→3 顺序、用**同一组** W4A16 量化权重（无 bf16/量化状态混用）
  顺序跑每个 token，逐层链（layer L 输入 = 上一层 final output）+ 逐层
  持久状态跨 token 线程；每个 (token, layer) 写一个 CUDLMG02 文件
  （`<prefix>_L{L}_p{t}.cudalm`）。**复用历史单容器**（CUDLMG02）——不新建
  格式，不破坏既有 `CUDLMG02` 单容器测试。
- `test_qwen35_hybrid_microstack_golden` ——硬门（§16.2）+ `--no-gen`
  （memcheck 用的 CUDA-only 路径，避免 compute-sanitizer 进程回收器与
  std::system Python 子进程的死锁）。
- `bench_qwen35_hybrid_microstack` ——baseline benchmark（§16.4）。

### 16.2 micro-stack golden 硬门

oracle（`generate_microstack`）与运行时用**同一组**量化权重，把
layers 0→1→2→3 顺序链，每层持久状态跨 token 线程。逐 (token, layer) 一个
CUDLMG02（`<prefix>_L{L}_p{t}.cudalm`）。两场景（每个都从 `reset_state` 起）：

- **A 首 token**：p=0（4 层链，全零状态）；
- **B 顺序**：p=0→1→2，运行时从**自身的**上一步持久状态链（中间不 reset）。

每层比对：所有 bf16 pipeline stage（DeltaNet 22 + fp32 `stage.g`；
全注意力 21）+ 持久状态（DeltaNet `conv_state` bf16 `[6144,3]` +
`recurrent_state` fp32 `[16,128,128]`；全注意力 K/V cache 行 0..p）+
micro-stack 最终输出（layer 3 final）。

**容差（关键正确性论证）：** 链式多层的 W4A16 GEMV 结合序噪声会跨层复合
（每层把前几层 ~1 ulp 的 bf16 输出噪声继承并放大；持久状态把该噪声向前携带）。
- **t=0**：layer 0 输入是种子输入（运行时/oracle 相同），全链干净——首 token
  用单层标准（bf16 `compare_bf16_stages` atol=rtol=1e-2，§14）；fp32 状态用
  Phase C 紧标准（atol=1e-5、rtol=1e-4）——实测**位级精确**（recurrent
  max_abs = 0）。
- **t≥1**：每层输入 = 上一层输出，继承并放大前几层噪声。钉死种子（20260209）
  实测最差：bf16 stage **3.9e-2**、fp32 状态 **3.1e-3**。门设在 bf16
  5e-2（atol=rtol）、fp32 5e-3/1e-2（~2× 余量）——仍远低于任何 O(1) 的错
  stage / 错索引 / 错权重 / **状态 alias / contamination** bug（这类 bug 已被
  t=0 标准 + 逐层 Phase B/C 门钉死）。

### 16.3 签核证据（本次运行）

- `test_qwen35_hybrid_microstack_golden`：
  - **A（p=0）**：4 层全链干净；逐层 `recurrent_state` **max_abs = 0**
    （位级精确）；`conv_state` / KV 行 / 全部 stage / micro-stack final OK。
  - **B（p=0→1→2 顺序）**：t=0 位级精确；t=1/t=2 逐层 stage + 状态在复合
    容差内（worst bf16 3.9e-2、fp32 3.1e-3）；逐层链（layer L 输入 ==
    layer L-1 输出）+ micro-stack final 全 OK。
- 完整 `ctest`：**34/34 PASS**（旧 v0.1/v0.1.1 回归 + Phase A 摄入 +
  Phase B 单元 + Phase B golden + Phase C 契约 + Phase C golden +
  **Phase D micro-stack golden** + Phase D `--tokens` 校验 + v0.3 Phase A
  full-model + v0.3 Phase B full-forward + **standalone bf16_gemv**）。
- `compute-sanitizer --tool memcheck`：**0 错误**（micro-stack golden
  `--no-gen` CUDA-only 路径，**覆盖连续多 token micro-stack 运行**——场景 B
  的 p=0→1→2 全链；证据 `benchmarks/sanitizer_qwen35_hybrid_microstack.txt`）。
- `scripts/check_no_torch.sh`：**CLEAN**（include/、src/ 无 torch/pybind）。

Phase D 只做**未优化 baseline**（无 NCU / 融合 / CUDA Graph）；**不进**
24 层全模型 / embedding / LM head / tokenizer / 生成 / Paged KV /
batching / scheduler / kernel 优化 / NCU。

### 16.4 baseline benchmark（未优化，仅报告）

`bench_qwen35_hybrid_microstack`（RTX 2080 Ti，sm_75，100 迭代，5 步
warmup，全局时钟预热 1500 步；evidence
`benchmarks/results/bench_qwen35_hybrid_microstack.json`）：

**状态语义（关键）**——每个 warmup/sample 都**从相同 pre-state 重建后**再计时
（不实现 snapshot/restore，直接 replay 前缀）：

```text
p0:    reset_state() -> timed forward(0)
p512:  reset_state() -> untimed forward(0..511) -> timed forward(512)
```

pre-state 重建不计入计时（计时窗口前的 `cudaStreamSynchronize` 保证 wall-clock
与 CUDA event 只覆盖 timed `forward(position)`）。这样每个 sample 都测**同一条
decode step 从同一 pre-state** 的代价——否则 p0 会因 recurrent 状态在 sample
间累积而偏离零状态、p512 会因每步推进而偏离 0..511 的 pre-state。

| 情况 | stage/layer sum | whole_microstack_gpu_us | host_api_wall_us | steps/s |
|---|---|---|---|---|
| **p=0**（零状态，FA 上下文深度 1）| 404.4 us | 414.7 us | 425.3 us | 2412 |
| **p=512**（顺序状态，FA 上下文深度 513）| 436.7 us | 446.5 us | 460.2 us | 2240 |

- DeltaNet 层与 position 无关（recurrent 状态是固定 `[16,128,128]` 矩阵，
  每步就地更新）：p=0/p=512 逐层一致（~102.6–113.3 us/层，layer0 略高，
  两情况完全吻合，印证 position 无关性）。
- 全注意力层随上下文深度上升：p=0（1 行 KV）85.0 us → p=512（513 行 KV）
  115.9 us。p=512 比 p=0 慢的 ~32 us 全部来自该层更深的注意力。
- **micro-stack step ≠ model token**（model 有 24 层，此处只跑前 4 层）；
  速率以 `microstack_steps_per_second_mean`（= 1e6 / whole_microstack_gpu_us
  mean）报告，**不**称 tokens/s。

**Phase D 后 STOP** —— Phase D（4 层混合 micro-stack）已完成（见 §16）并
推送 `v0.2-qwen35`；等待外部 review。**v0.2 最终 hybrid decoder micro-stack**
（3×Gated DeltaNet + 1×FullAttention，真实 checkpoint layers 0-3）已就绪。

## 17. v0.3 Full Model（Phase A：完整 24 层结构 + 权重摄入口径）

Phase A（v0.3）范围 = **完整 24 层模型结构 + 权重摄入口径 + 模型级 runtime
骨架的所有权/状态生命周期**。**不做**：tokenizer / 生成 / 采样 / scheduler /
Paged KV / 性能优化；**不做**全模型 forward / logits（那是 v0.3 Phase B）。

### 17.1 钉死的全模型契约（来自 pinned 官方源 + 真实 checkpoint，无猜测）

pinned oracle = `transformers@fc9137225880`（`modeling_qwen3_5.py`），真实
checkpoint = `Qwen3.5-0.8B-Base`（`raw/config.json` + `model.safetensors`，
语言张量在 `model.language_model.*` 前缀下）。

| 组件 | checkpoint 张量名 | shape / dtype | 官方源依据 |
|---|---|---|---|
| 词嵌入 | `model.language_model.embed_tokens.weight` | **BF16 `[248320, 1024]`** | `nn.Embedding(vocab_size=248320, hidden_size=1024, pad_token_id)`（modeling L1298）|
| 24 decoder 层 | `model.language_model.layers.{0..23}.*` | 见 §7/§8 | `num_hidden_layers=24`（config）|
| 最终 norm | `model.language_model.norm.weight` | **BF16 `[1024]`** | `Qwen3_5RMSNorm(hidden_size=1024, eps=1e-6)`（modeling L1302）|
| LM head | **无独立张量（tied）** | 复用 embed_tokens `[248320,1024]` | `_tied_weights_keys={"lm_head.weight":"model.language_model.embed_tokens.weight"}`（L1825）；`lm_head=nn.Linear(1024,248320,bias=False)`（L1833）；`logits = hidden @ embed_weight.T`（L1957）|

- **权重绑定（weight tying）**：config `tie_word_embeddings: true`（顶层 +
  `text_config` 均 true）→ checkpoint **没有** `lm_head` 张量；LM head 与词嵌入
  **共享同一份权重**（logits 用其转置）。本契约把 LM head 记录为「alias 词嵌入」，
  绑定事实写进 `.cudalm` v2 **metadata**（`tie_word_embeddings=true`），不新建张量。
- **层排表（exact hybrid schedule）**：config `layer_types` = 3×`linear_attention`
  + 1×`full_attention` 重复 → **全注意力在 layer 3,7,11,15,19,23**（共 6 层），
  其余 18 层 = **Gated DeltaNet**（`linear_attention`）。与 config 派生的
  `is_full_attention(i)=(i+1)%full_attention_interval==0`（interval=4）完全一致。
- **dtype**：权重绝大多数 BF16；DeltaNet 的 `linear_attn.A_log` `[16]` 与
  `linear_attn.norm.weight` `[128]` 为 **FP32**（`mamba_ssm_dtype=float32`）；
  DeltaNet recurrent 状态 **FP32** `[16,128,128]`。
- **关键 config 值**：`hidden_size=1024`、`num_hidden_layers=24`、
  `intermediate_size=3584`、`vocab_size=248320`、`num_attention_heads=8`、
  `num_key_value_heads=2`、`head_dim=256`、`linear_num_key_heads=16`、
  `linear_num_value_heads=16`、`linear_key_head_dim=128`、`linear_value_head_dim=128`、
  `linear_conv_kernel_dim=4`、`full_attention_interval=4`、
  `max_position_embeddings=262144`、`rms_norm_eps=1e-6`。
- **v0.3 文本范围之外**：`model.visual.*`（视觉塔）与 `mtp.*`（多 token 预测）
  张量**不摄**入（converter manifest 记为 `skipped`）。
- 单层张量集见 §7（全注意力 11 张量）/ §8（DeltaNet 14 张量），此处不重复。

### 17.2 .cudalm v2 全模型摄入扩展（向后兼容）

`.cudalm` v2 是「命名张量容器」（name→blob）。全模型 = 在**同一容器**里新增
模型级命名张量 + metadata，**不改** 88 字节 config blob、**不改** 既有张量集、
**不破坏** Phase A-D 任何 fixture：

- `tools/convert_qwen35.py` 新增 `--full-model`：强制全 24 层 + 写
  `embed_tokens.weight`（BF16 `[vocab,hidden]`）+ `norm.weight`（BF16 `[hidden]`，
  原本就写）+ metadata `tie_word_embeddings`（读 config，`true`/`false`）。
  `manifest` 记 `full_model` + `skipped=["visual","mtp"]`（`embed_tokens` 不再 skipped）。
- **不加 `--full-model`** 时行为**完全不变**（per-layer `--layers 0,1,2,3` 仍只写
  该层张量 + `norm.weight`，无 embedding、无 tie metadata）——Phase A-D
  golden/ingestion 输出逐字节保持。
- C++ `WeightFileV2` 新增 `validate_model_embedding()`（`embed_tokens.weight`
  `[vocab,hidden]` bf16）+ `validate_full_model()`（embedding + norm + 24/24 层
  + tie metadata 必须为 `"true"`）；缺张量 / shape / dtype / tie 不符都 **loudly fail**。
  `Qwen35Config` 本就含 `vocab_size`（v0.2 未用），此处直接复用。

### 17.3 `Qwen35Model`（模型级 runtime 骨架，CUDALM 原生，复用 v0.2 单层）

`include/cudalm/qwen35_model.h` + `src/runtime/qwen35_model.cpp`：

- **embedding 所有权**：`embedding()` → device BF16 `[248320,1024]`。
- **24 层所有权 + dispatch**：`load()` 逐层 `Qwen35LayerWeights::load` + 按
  config 排表构造 runtime（`Qwen35DeltaNetLayer` 18 层 / `Qwen35FullAttentionLayer`
  6 层）——**复用**冻结的 v0.2 单层运行时，**不复制任何 kernel**，**不改**冻结数学。
- **最终 norm 所有权**：`final_norm()` → device BF16 `[1024]`。
- **LM head 所有权（tied）**：`lm_head()` **alias 词嵌入**（同一 device buffer，
  无独立分配；logits = hidden @ lm_head^T）；`tie_word_embeddings()=true`。
- **统一 `reset_state(stream)`**：遍历**全部 24 层**逐层独立重置（DeltaNet 零
  conv+recurrent；全注意力零 KV）——无跨层 alias。embedding/norm/lm_head 是权重
  非状态，不受影响。
- **Phase A 边界**：本类**不跑**全模型 forward / 不计算 logits（Phase B）；
  只完成 load（上传权重）/ 所有权 / 逐层 dispatch / 状态生命周期 / unload（析构）。
- 析构顺序：层 runtime 先于其权重集销毁（`weights_` 声明在 `layers_` 之前）；
  embedding/norm buffer 与 alias 它们的 `TensorView`（view 平凡销毁）。

### 17.4 硬门与签核证据（本次运行）

- **真实 checkpoint 全量转换 PASS**：`convert_qwen35.py --full-model` 一次转出
  **506 张量 / 766 MB / layers 0..23**（embedding `[248320,1024]` bf16 +
  `norm.weight` `[1024]` bf16 + 24 层 + tie metadata）。
- **24/24 层 validate PASS** + **层排表 exact**（全注意力恰在 3,7,11,15,19,23；
  DeltaNet 18 层）+ **embedding/final-norm/LM-head 张量契约 PASS**（shape/dtype
  精确、无独立 `lm_head` 张量、tie metadata=`"true"`）。
- **full model load/unload PASS**：`Qwen35Model::load` 上传全模型（device 约
  ~3.9 GB：embedding 509 MB + 6×FA KV ~3.2 GB + 24 层权重 ~250 MB），析构即 unload。
- **`reset_state()` 覆盖全部 24 层**：把每层持久状态 seed 成非零 sentinel
  （DeltaNet 经 `seed_state` H2D；FA 经 `cudaMemsetAsync`），reset 后逐层回读
  全部归零。
- 新增 `test_qwen35_full_model`（CPU 结构门 + GPU 所有权/状态门；无 checkpoint
  时 self-skip 77）。完整 ctest **34/34 PASS**（旧 31 全回归 + 1 Phase A
  full-model + 1 Phase B full-forward + 1 standalone bf16_gemv）；
  `check_no_torch.sh` **CLEAN**；
  `compute-sanitizer --tool memcheck` **0 错误**（`--no-gen` CUDA-only 路径，
  覆盖全模型 load + 24 层 seed + reset + unload 的**新 CUDA 分配生命周期**）。

**Phase A 后 STOP** —— v0.3 Phase A（完整 24 层结构 + 权重摄入口径 + 模型骨架
所有权/状态生命周期）已完成并推送 `v0.3-full-model`。**不自动开始**全模型
forward / logits（v0.3 Phase B）。

## 18. v0.3 Phase B —— 完整单-token forward 契约（钉死，来自 pinned 官方源）

范围 = 真实 Qwen3.5-0.8B 的**完整单-token forward**（correctness bring-up）：
`token_id → embedding → layers 0..23 → final RMSNorm → tied LM head →
logits [248320]`。**不做** generation / tokenizer / sampling / benchmark 优化。

全部来自 pinned 官方源 `transformers@fc9137225880`（`modeling_qwen3_5.py`，
下称 `modeling`），无通用 Qwen 经验推断：

- **embedding lookup（dtype/rounding）**：`Qwen3_5TextModel.forward` 里
  `inputs_embeds = self.embed_tokens(input_ids)`（`modeling` L1325）；
  `embed_tokens = nn.Embedding(vocab_size=248320, hidden_size=1024)`（L1298）。
  `nn.Embedding` 是**纯索引拷贝**（无算术、无舍入）：`out[batch,t,:] =
  weight[input_ids[batch,t], :]`。权重 BF16 → 输出 **BF16 `[1024]`**，与
  `embed_tokens.weight[token_id]` **逐位一致**。
- **24 层执行顺序**：`for layer_idx, decoder_layer in
  enumerate(self.layers[:num_hidden_layers]): hidden_states =
  decoder_layer(hidden_states, ...)`（L1370-1379）。**顺序** 0→23；第 L 层输入 =
  第 L-1 层输出（第 0 层输入 = embedding）。每层数学 = 冻结 v0.2 Phase B/C
  （§7/§8），此处不重复。
- **final RMSNorm 精确语义**：`hidden_states = self.norm(hidden_states)`
  （L1381）；`norm = Qwen3_5RMSNorm(hidden_size=1024, eps=1e-6)`（L1302）。
  `Qwen3_5RMSNorm.forward`（L807-819）精确为：
  `out = ( x.float() * rsqrt(mean(x.float()^2) + eps) * (1.0 + w.float()) )`
  全链 **fp32**，末尾 **一次** `.type_as(x)` 回 BF16（RNE）。weight 初始化为
  0（`nn.Parameter(torch.zeros(dim))`）→ **零中心**（有效缩放 = `1 + w`），
  与 §6.1 / 现有 `qwen35_rmsnorm_zc_bf16` kernel **同一公式**（复用，不新写）。
- **tied LM head 精确数学**：`logits = self.lm_head(hidden_states[:,
  slice_indices, :])`（L1957）；`lm_head = nn.Linear(1024, 248320, bias=False)`
  （L1833）；`_tied_weights_keys = {"lm_head.weight":
  "model.language_model.embed_tokens.weight"}`（L1825）+ `post_init()` →
  `lm_head.weight` **就是** `embed_tokens.weight`（**同一份 BF16 `[248320,1024]`**，
  无独立张量）。单 token：`logits[v] = Σ_h normed_hidden[h] ·
  embed_tokens.weight[v, h]`，即 `logits = normed_hidden @ embedding^T`。
- **logits 输出 dtype**：源注释「do not upcast them to float if we are not
  computing the loss」（L1955）；`nn.Linear(BF16 输入, BF16 权重) → BF16`
  输出。故 `logits` = **BF16 `[248320]`**。
- **accumulation / rounding boundaries（逐段舍入契约）**：
  1. embedding：**无舍入**（BF16 行拷贝）；
  2. 每层：冻结 v0.2 逐 stage 契约（BF16 stage 边界单次 RNE；DeltaNet
     recurrent 状态 FP32）；
  3. final norm：**fp32 全链 + 末一次 BF16 RNE**；
  4. tied LM head：BF16 GEMV，**fp32 累加 + 每个输出元素一次 BF16 RNE**
     （对齐 PyTorch bf16 `nn.Linear` 的 cublas fp32-acc 语义；累加**顺序**
     与 cublas 可不同，但都是合法 fp32 顺序 → 差在 fp32 舍入界内，golden 容差
     覆盖，见 §18.1）。

### 18.1 LM head GEMV（新增 BF16 GEMV kernel）与 golden 容差

- 新增 `bf16_gemv`（`kernels/bf16_gemv.h/.cu`）：`y[n] = Σ_k W[n,k]·x[k]`，
  `W` BF16 `[N,K]`、`x` BF16 `[K]`、`y` BF16 `[N]`；**fp32 累加 + 单次 BF16
  RNE store**。参考 frozen `CUDALab@cb6a6a9` 的 FP16 GEMV proven 结构
  （`kernels/gemv/gemv_vec4_row.cu`：16B 向量 load、每行一 block、fp32 累加、
  warp+shared 归约）适配到 raw pointer + cudaStream_t + BF16 + `[V=248320,
  K=1024]`；标量回退路径同 `gemv_scalar_kernel`。详见 `docs/provenance.md`。
- **独立 numeric 硬门 `test_bf16_gemv`（不靠 full-model logits 证明 kernel
  正确性）**：deterministic BF16 输入 vs **CPU FP32-accumulate → BF16 RNE
  参考**（`compare_bf16_stages` 1e-2，抓 O(magnitude) 的错索引/错累加/错舍入/
  错 dtype），覆盖：**vec4 路径**（K%8==0 ∧ 16B 对齐，含真实 LM-head 形状
  `[N=248320,K=1024]` + N=1/7/100/1024/4096、K=8/16/64/512）；**scalar 回退**
  （K%8!=0：K=1/3/7/9/1000/1025 **及** K%8==0 但 2 字节错位基址）；scalar-vs-
  vec4 同数据交叉校验；全零 weight 行 → bit-exact 0。
- **golden 容差**：24 层 BF16 链 + fp32 recurrent 状态跨层复合误差，从 v0.2
  标准出发（bf16 stage atol=rtol=1e-2；fp32 状态 atol=1e-5/rtol=1e-4 紧标准）。
  最终 **FULL logits `[248320]`** 做**全向量** golden 比较（非 top-k）；若实测
  24 层链需要放宽，只按实测设 **per-layer / depth-aware 最小必要 envelope** 并
  记录 rationale（§18.2）。

### 18.2 实测最小必要 envelope（**per-layer / depth-aware**）与 rationale

`test_qwen35_full_forward`（真实 checkpoint，A：p0 fresh；B：p0→p1→p2 顺序、
runtime 自线程状态）对**每 token**比较：embedding / 24× 层 final / final norm /
**FULL logits `[248320]`** / 18× DeltaNet conv+recurrent / 6× FA K/V rows 0..p。
embedding 为纯行拷贝，**bit-exact（max=0）**。

**per-layer / depth-aware envelope（取代旧的"单类共享 tolerance"）**：每个**层**
的 threshold = 该层在确定性 A+B 运行的**实测 worst × 1.3 + 1e-3**（小余量），
**不是**所有层共享一个值（旧的 layer=0.09 / conv=0.31 / recurrent=0.055 /
KV=0.12 已废弃）。这样：
- **L0/早期层保持接近 v0.2 紧标准**：L0 layer-final 实测 4.88e-4 →
  threshold ~1.6e-3（比 v0.2 的 1e-2 **更紧**）；L0 conv 实测 A+B worst 0.0156
  （场景 B p2，状态逐 token 线程后；场景 A/B-p0 为 0）；
- **不允许 L23 的误差上限放宽 L0**：每层 threshold 由**该层自身**实测 worst
  决定（L23 layer-final 7.8e-2 → threshold ~1.0e-1），一个 L23 量级的 bug 落在
  L0 会被 L0 的紧 threshold（~1.6e-3）抓住（共享 0.09 会漏掉）；
- **final norm / FULL logits 保留模型级 envelope**（下表末两行）。

代表性层实测 worst → 采用 atol（完整 24 层表在测试
`kLayerFinalAtol[24]` / `kConvAtol[24]` / `kRecurAtol[24]` / `kKvAtol[24]`；
错误类别的层不比较、threshold=0）：

| 类别（层） | dtype | 实测 worst → 采用 atol |
| --- | --- | --- |
| layer-final L0 | bf16 | 4.88e-4 → 1.6e-3 |
| layer-final L6 | bf16 | 1.17e-2 → 1.6e-2 |
| layer-final L12 | bf16 | 1.17e-2 → 1.6e-2 |
| layer-final L18 | bf16 | 4.69e-2 → 6.2e-2 |
| layer-final L23 | bf16 | 7.81e-2 → 1.0e-1 |
| DeltaNet conv L0 | bf16 | 0.0156 → 2.1e-2 |
| DeltaNet conv L20（peak） | bf16 | 0.281 → 0.367 |
| DeltaNet recurrent L0 | fp32 | 7.94e-4 → 2.0e-3 |
| DeltaNet recurrent L18（peak） | fp32 | 4.72e-2 → 6.2e-2 |
| FA KV L3 | bf16 | 0.0625 → 8.2e-2 |
| FA KV L15（peak） | bf16 | 0.109 → 0.143 |
| model.final_norm_output | bf16 | 0.4453 → 0.580（模型级） |
| model.logits `[248320]` | bf16 | 0.3125 → 0.407（模型级） |

**深度趋势（diagnostic / report-only，非 PASS/FAIL 门）**：误差**总体呈随 depth
增大的趋势，但不要求严格单调**（实测有局部噪声，例如 conv L6 < L5、L12 < L11，
某层可能比前一层略小）。`report_smooth_growth` 只**报告**每个类别的
**后段**（L16..23）实测 worst vs **前段**（L0..7）实测 worst
（layer-final 0.078 vs 0.012；conv 0.281 vs 0.125；recurrent 0.047 vs 0.013；
KV 0.094 vs 0.063），**不影响 PASS/FAIL**。**正确性只由 per-layer envelope
硬门决定**（`actual_error[L] <= k*Atol[L]`）——因此若某次构建误差**变小**（更紧），
测试**不会失败**（不因误差变小而 fail）。

**为什么是这些量级（而非 v0.2 的 1e-2）**：v0.2 标准是**单层**（或 4 层
micro-stack，worst 3.9e-2）的界。完整 24 层 BF16 链把每层 ~1-ulp 的 GEMV
**fp32 累加顺序**差异（runtime `int4_gemv_bf16`/`bf16_gemv` vs golden 的
fp32 matmul）**逐层复合**：layer final 误差**总体随 depth 增大**（非严格单调，
有局部噪声；L0 4.9e-4 → L9 1.6e-2 → L23 7.8e-2，**无 O(1) 突变**）；final norm
的 `(1+w)`
零中心缩放把 L23 误差放大（worst 0.4453）；`[vocab]` logits GEMV 再复合
（worst 0.3125）。

**为何确定是舍入复合、而非错 stage/错权重/错 dtype（那会是 O(1)）**：
(a) embedding **bit-exact**（lookup 正确）；(b) layer final **总体随 depth 增大、
无 O(1) 单层突变**（若是某层错权重/错索引，会在该层出现 O(1) 跳变）；(c) **全部
持久状态**（18× DeltaNet conv/recurrent + 6× FA K/V rows 0..p）在**小量级**
内匹配且随深度增长（层内数学正确）；(d) 冻结 v0.2 单层在 micro-stack golden
已通过 1e-2（单层正确）。

### 18.3 签核证据（本次运行，RTX 2080 Ti / CUDA 11.8）

- 真实 checkpoint 完整 forward PASS：A（p0 fresh）+ B（p0→p1→p2 顺序，runtime
  自线程状态）全部类别在 §18.2 **per-layer / depth-aware envelope** 内（正确性
  唯一判据；深度趋势为 diagnostic report，非门）（`test_qwen35_full_forward`）。
- 独立 `test_bf16_gemv` PASS（vec4 含 `[248320,1024]` + scalar K%8!=0/错位 +
  scalar-vs-vec4 + 全零行，vs CPU FP32→BF16 RNE 参考）。
- 完整 ctest **34/34 PASS**（旧 31 全回归 + 1 v0.3 Phase A full-model +
  **1 v0.3 Phase B full-forward + 1 standalone bf16_gemv**）。
- `scripts/check_no_torch.sh` **CLEAN**（include/、src/ 无 torch/pybind）。
- `compute-sanitizer --tool memcheck` **0 错误**，两份 evidence：
  - `benchmarks/sanitizer_qwen35_full_forward.txt`（`--no-gen` CUDA-only，覆盖
    **完整 24 层 forward + 新增 `bf16_gemv` LM-head GEMV + A/B 顺序路径** 的新
    CUDA 分配生命周期）；
  - `benchmarks/sanitizer_bf16_gemv.txt`（覆盖 **bf16_gemv vec4 路径 + scalar
    回退（K%8!=0 及错位）**）。

 ## 19. v0.4 Phase A —— token-ID 级单请求 serial prefill + 贪心 decode 生成核

 范围 = **token-ID 级单请求生成**（`prompt_token_ids + max_new_tokens +
 eos_token_id → generated_token_ids + stop_reason`），correctness bring-up：
 serial prefill（正确性优先，**不** batched/chunked）+ 贪心 decode（greedy
 only）。**不做** tokenizer / prompt 字符串 / detokenizer / sampling /
 temperature / top-k / top-p / repetition penalty / beam / batched-chunked
 prefill / Paged KV / multi-request / scheduler / continuous batching / NCU /
 kernel fusion / CUDA Graph / 性能优化。serial prefill 的速度**不是**最终性能
 数字（本 phase 不 benchmark）。

 ### 19.1 生成核契约（`include/cudalm/qwen35_generator.h` +
 `src/runtime/qwen35_generator.cpp`）

 - **接口**：`Qwen35Generator::generate(const std::vector<int>& prompt_tokens,
   int max_new_tokens, int eos_token_id, cudaStream_t stream,
   const LogitsObserver* = nullptr) → GenerationResult`。`GenerationResult`
   = `{ok, error, prompt_count, generated_token_ids, stop_reason,
   forward_count}`。generator 持 **非 const** `Qwen35Model&`（模型自身持久
   状态 in-place 线程化整条序列）。
 - **生命周期（钉死）**：每次 `generate()` 开头 **`reset_state()` 一次**；
   **serial prefill** `forward_token(t_i, i)` for i=0..N-1（**故意** serial，
   正确性优先，不 batched/chunked）；prefill 最后一个 token 的 logits 预测
   position N；**greedy decode**：`next = argmax(prefill-last logits)` →
   position N，放置 `next`；若不 stop 则 `forward_token(next, N+step)` →
   position N+step+1，再 `argmax`。**禁止**：重 forward prompt 末 token、
   position off-by-one、decode 前再 reset、golden 状态回填 runtime。
 - **position 守卫**：`position_of(step) = prompt_len + step`；每个 decode
   forward 前守卫 `position_of(step) < max_seq_len`（**绝不**以
   `position >= max_seq_len` 调 `forward_token`）。
 - **贪心 argmax（CPU，`include/cudalm/greedy.h`）**：`argmax_bf16` **最大化
   bf16 数值 logit**，**tie → 最小 token id**；**不**用 CUDA argmax/reduction
   kernel（Phase A 正确性优先：D2H 全 [248320] bf16 logits → CPU argmax）。
 - **EOS/stop 控制器（确定性，`greedy.h::GreedyStopController`）**：
   **EOS 选中 → 立即 stop**（该 token **在** `generated_token_ids` 里、**不**
   再 forward）；否则 `step+1 >= max_new_tokens` → stop；否则可继续；
   `max_seq_len` 由 position 守卫触发（`position_of(step) >= max_seq_len`）。
   同一步 EOS **优先于** max_new_tokens。
 - **容量契约（loudly fail）**：空 prompt / 非法 token id / 非法 eos（不在
   `[0, vocab)`）/ `max_new_tokens < 0` / `prompt_len > max_seq_len`。
   `max_new_tokens == 0` → 空生成、不 decode（文档化）。
 - **forward_count 语义**：= prefill N + decode forward 次数。EOS 在 step k →
   decode forward = k（EOS 不 forward）；max_new_tokens（G token）→ decode
   forward = G-1（末 token 不 forward）；max_seq_len（G_avail token）→ decode
   forward = G_avail。等于 oracle 的 `t_used`。

 ### 19.2 生成级 golden oracle（pinned quantized，
 `tools/generate_qwen35_golden.py --gen-prefix`）

 - **pinned quantized oracle**：复用 §18 的全模型 oracle（同一份 `.cudalm v2`
   W4A16 权重 + tied BF16 embedding + 18× DeltaNet / 6× FullAttention 层 +
   pinned `Qwen3_5RMSNorm` + tied LM head = `normed @ embedding^T`）。
 - **serial prefill + greedy decode 镜像 C++ 循环**：同一 reset、同一 position
   守卫、同一 argmax 契约（bf16 数值 max、tie→最小 id，`torch.argmax` on bf16）、
   同一 stop 顺序、stop token 不 forward、**不回填 golden 状态**。
 - **输出 CUDLMW02 容器**（C++ `WeightFileV2::load` 可载）：metadata = 生成
   token 序列 + stop_reason + `t_used`；张量 = **每个生成步的全 [248320] bf16
   logits**（`gen.logits.t{k}`，预测 `generated[k]`）+（场景 B）**最终持久
   状态**（18× DeltaNet conv bf16 [6144,3] + recurrent fp32 [16,128,128]；
   6× FullAttention K/V used rows bf16 [kv, t_used, 256]）。
 - **confident prompt（钉死）**：oracle 的 top-1 vs top-2 logit gap 在**每个**
   生成步都需**显著高于** runtime/oracle bf16-logits 舍入（~0.13），贪心序列
   才是**稳定 golden**（near-tie 会让 ~0.13 舍入翻转 argmax → 两 runtime 选
   不同 token → 序列分叉）。`tools/diag_gen_gaps.py` 测逐步 gap；选定 prompt
   min gap **1.0625**（A=[1024,2048,3072]）/ **3.625**（B=[1024,2048,3072]×5+
   [1024]）。
 - **收紧的 per-step / per-layer envelope**：比较阈值**不**用共享 atol，而是
   每步 / 每层独立 envelope = `实测 worst × 1.3 + floor`（**不让**某个 late
   layer / worst step 放宽其他 step/layer）。**BF16 与 FP32 状态用不同
   floor**：per-step 全 logits + DeltaNet conv + FullAttention KV（均 BF16）
   用 `+ 0.01`；DeltaNet **recurrent（FP32）** 用 `+ 0.001`（其 chain 更紧，
   实测 worst ~0.033，远低于 0.01 floor）。envelope：per-step 全 logits（A 8
   步 / B 16 步各自）+ per-layer DeltaNet conv / recurrent / FullAttention KV
   （B 最终状态：18× conv/recurrent + 6× KV，非适用层为 0）。实测 worst
   （pinned oracle，RTX 2080 Ti）：per-step logits ≤ 0.5625（A t4）、
   conv ≤ 0.15625（L17）、recurrent ≤ 0.0326（L20，**非** 0.5）、KV ≤ 0.1875
   （L11 K）。每次运行 compare 打印 max_abs，可重新收紧到实测误差（非猜值）。

 ### 19.3 测试与签核证据

 - **`test_greedy_argmax`**（CPU，无 checkpoint）：argmax（normal / negative /
   exact tie→最小 id / single / bf16-rounding tie）。
 - **`test_greedy_stop`**（CPU，无 checkpoint）：确定性控制器（eos /
   max_new_tokens / max_seq_len / 末 token eos 优先）。
 - **`test_qwen35_generation`**（real checkpoint，self-skip 77）：场景 A（短
   confident prompt，8 token）+ B（长 confident prompt，16 token，最终状态）+
   C（repeat-generate 污染门，**强化**：两次 `generate()` 用 `LogitsObserver`
   抓 **EVERY 生成步全 [248320] logits** 逐 step **原始字节（memcmp）bit-exact** 比较（**非**仅数值相等，`+0/-0`
   等 bit 不同也 FAIL），证明第二次
   reset 后数值轨迹与第一次相同，而非仅 greedy token 恰好没变）。比较 **EVERY
   生成 token id** + **EVERY 生成步全 [248320] logits**（per-step envelope）+
   stop_reason + forward_count（== t_used）+（B）最终混合持久状态（18× DeltaNet
   conv/recurrent + 6× FA K/V used rows，per-layer envelope）。
 - **`test_qwen35_generation_contract`**（real checkpoint，self-skip 77）：生成核
   **输入契约** hardening —— `not_loaded` / 空 prompt / 非法 token id（<0 与
   ≥vocab）/ 非法 eos（<0 与 ≥vocab）/ `max_new_tokens < 0` /
   `prompt_len > max_seq_len` 全部 → `ok == false` + generated 空 +
   `forward_count == 0`（无 prefill/decode、无状态变更）；`max_new_tokens == 0`
   → `ok == true` + 空生成 + stop=max_new_tokens + prefill 已跑（forward_count
   == prompt_len）但**不** decode。不扩大 API（只驱动现有 `generate()`）。
 - 签核（RTX 2080 Ti / CUDA 11.8）：完整 ctest **38/38 PASS**；
   `check_no_torch.sh` **CLEAN**；`compute-sanitizer --tool memcheck` **0 错误**
   （`benchmarks/sanitizer_qwen35_generation.txt`，`--no-gen` CUDA-only，覆盖
   multi-token prompt + multi-step greedy decode：真实 prefill/decode 状态转换 +
   重复 tied-LM-head GEMV + KV history 增长 + DeltaNet recurrent 更新 + 最终
   状态读回）。

 ## 20. v0.4 Phase B —— 原生 tokenizer + prompt→text（钉死，来自 pinned 官方源）

 范围 = **raw UTF-8 prompt → token id → 生成 → token id → raw UTF-8 text** 的
 全链路原生（PyTorch-free）实现：

 ```
 raw UTF-8 prompt
   -> Qwen35Tokenizer::encode      （NFC → added-token 切分 → 左优先正则
                                    预分词 → ByteLevel BPE）
   -> Qwen35Generator::generate    （§19 冻结生成核；EOS = 真实 tokenizer
                                    EOS 248044，**非** Phase A 哨兵 248319）
   -> Qwen35Tokenizer::decode      （ids → 完整字节流：base 原始字节 /
                                    added 字面量 / padding 丢弃；再对**整个**
                                    字节流做一次有损 UTF-8 转换 —— 每个最大
                                    非法子段 → 一个 U+FFFD；合法的跨 token
                                    多字节字符仍按一个字符解出）
   -> raw UTF-8 生成文本（恒为合法 UTF-8）
 ```

 `Qwen35TextGenerator`（`include/cudalm/qwen35_text_generator.h`）是**纯
 拼接层**（thin facade）：encode → generate → decode 三行缝合，**不**新增任何
 模型状态 / 生成逻辑 / sampling / chat-template / streaming / batching ——
 每一字节输入输出都由既有硬门覆盖（tokenizer 语料门 + 生成 golden 门）。

 ### 20.1 官方源钉死（pinned，只取 tokenizer 文件，**不**下载权重）

 - 仓库 `Qwen/Qwen3.5-0.8B-Base` @ revision
   `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`（与 §1 同一 pinned 源）。
 - 本地已下载于 `/root/models/Qwen3.5-0.8B-Base/tokenizer/`，4 个文件 sha256
   钉死（`tools/common/qwen35_tokenizer_ref.py` 的 `EXPECTED_SHA256`，读取
   前强制校验）：

   | 文件 | sha256 |
   |---|---|
   | `tokenizer.json` | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |
   | `tokenizer_config.json` | `e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de` |
   | `vocab.json` | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
   | `merges.txt` | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |

 - **oracle 版本（pinned + 版本门）**：主 oracle = 本地 venv
   （py3.11.16）`tokenizers 0.22.2`（Rust 引擎，pinned 语义）。版本在
   `tools/common/qwen35_tokenizer_ref.py` 中钉死：
   `EXPECTED_TOKENIZERS_VERSION = "0.22.2"` + `check_oracle_version()`；
   `build_tokenizer()` 在任何引擎调用**之前**跑该门，因此**所有
   authoritative oracle 路径**（converter、语料生成、differential
   validator、text golden）都经过同一个检查。实际版本 ≠ 0.22.2 =
   **provenance violation，fail loud，不可 skip**（资产缺失仍按既有
   规则 self-skip 77 —— 二者不同）；converter `--selftest` 含版本门
   回归（0.22.2 接受；0.15.1 / 0.22.3 / None 拒绝）。系统
   `tokenizers 0.15.1` **不是**可接受 oracle（无 silent fallback）。
   **pinned `tokenizer.json` 含 `"normalizer": {"type": "NFC"}`** ——
   引擎在预分词前对输入做 NFC，encode 的可观测行为包含 NFC。
 - **Unicode 数据口径（BLOCKER-D1/D2 —— 已解决）**：converter 的
   Unicode 表**不再**来自构建机 Python `unicodedata`（本机 U14，与引擎
   数据版本不一致 —— 上一轮上报 BLOCKER-D1 的根因）。现口径（全部
   sha256 钉死、离线可复现、无网络）：
   - **normalizer 子系统（ccc/分解/合成）= Unicode 9.0.0** —— pinned
     引擎的真实数据版本（`unicode-normalization-alignments` 0.1.12
     crate 常量 `UNICODE_VERSION=(9,0,0)`，全表行为实证一致）。数据源
     = `tools/ucd/UnicodeData-9.0.0.txt`（sha256
     `68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f`，
     unicode.org 官方 U9 UCD；`tools/common/qwen35_unicode_ref.py`
     读取前强制校验 sha，含 UCD range 记法展开）。合成对由 U9 完全
     分解流推导，且**每个 (x,y)→z 构建时逐一过 pinned 引擎
     `normalizer.normalize_str` 验证**（引擎 = 构建期 oracle；Python
     U14 在此不可用 —— 它对 98 个 ccc 差异 cp 与 Divès Akuru 对的
     行为就是 U14 行为）。
   - **regex 类子系统（\pL/\pN/\pM）= Unicode 16.0.0** —— pinned
     regex 引擎（tokenizers `regex = "1.10"`）的真实数据版本，由
     **全 1,112,064 非代理 cp 逐 cp chunking 探测**实证（`"a"+cp` /
     `"a"+cp+"b"` / `cp+cp+"x"` 三探针 vs 引擎
     `pre_tokenize_str`）：L∪M == U16 UCD 恰（143,529 cp）、N == U16
     UCD 恰（1,911 cp）。数据源 = `tools/ucd/UnicodeData-16.0.0.txt`
     （sha256 `ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f`）。
     旧 U14 L/N/M 区间对 ~5k 个 U15/U16 新增 cp 分类错误（预分词阶段
     可观测；id 级恰因稀有文字无跨边界 BPE 合并而侥幸一致 —— 语料
     P 行 + 逐 cp 探测是真正闸门）。
   - **\s = UAX #44 White_Space，恰 25 个 cp**（09-0D/20/85/**A0**/
     1680/2000-200A/2028/2029/202F/205F/3000），逐 cp 实证。**U+00A0
     已含**（BLOCKER-D2 修复）。
   - 引擎 vs Python `unicodedata`（U14）的**全部**差异 = 上述数据版本
     差：恰 98 个 ccc 差异 cp（U14 ccc>0、U9 ccc=0，如 U+1715）+
     Divès Akuru 对 (U+11935,U+11930)↔U+11938（U13 新增，U9 中原子）。
     **算法本身无差异**：200k 样本（seed 20260925）NFC 差分 +
     针对性 ccc/合成 battery，引擎与 Python 算法层面 100% 一致（0
     未解释发散）。上一轮文档中“Python 给 `U+0391 U+093C U+0301`”
     的示例系**测量错误**（Python 实测 `U+0386 U+093C`，与引擎相同），
     已删除；不声称任何算法级语义差异（未观察到，亦无证据）。
 - 词汇结构（钉死）：base BPE 词汇 **248044**（id 0..248043，GPT-2 byte
   map：33..126/161..172/174..255 自映射，其余 68 字节 → U+0100..U+0143）；
   added token **33** 个（id 248044..248076 = `tokenizer.json` added_tokens
   与 `tokenizer_config.json` 的并集、config 优先；其中 **21 special +
   12 non-special**）；**padding id 248077..248319**（model vocab 248320）
   decode → `""`（HF/Rust 均静默丢弃，逐一对齐）；**单 EOS = 248044**
   （= `raw/config.json` `text_config.eos_token_id`）。

 ### 20.2 管线契约（逐段 oracle 验证，C++ 只做离线转录的查表）

 1. **NFC**（**pinned 引擎算法的精确移植**，非教科书简化版）：
    引擎的 NFC = Rust `unicode-normalization`（tokenizers crate 依赖
    `unicode-normalization-alignments` 0.1.12）的 Decompositions +
    Recompositions 双迭代器算法，C++ 逐语句移植（`nfc_impl`）：
    (a) 完全规范分解（含单部件分解，如 U+1FBE→U+03B9、Hangul 音节→jamo）；
    (b) **流式规范重排**：分解流分批发射 —— 读入一个 `ccc == 0` 的
    codepoint（或输入结束）时，对**尚未发射的尾部**按 ccc **升序、稳定**
    排序后放行（`ccc == 0` = starter / 序列边界；jamo 在表中 ccc 0，
    序列永不重排 —— oracle 验证 `V+L`(U+1161,U+1100) 原样不动，而
    `L+V`→U+AC00）；
    (c) **合成与重排同趟进行**（关键差异）：composee 只能与**入流**
    非 starter 合成，且仅当**所有已缓冲（延迟）mark 的 ccc 都严格小于**
    该 mark；否则该 mark 被缓冲、延后发射。纯合成表查表（表 = 恰好
    `NFC(x y) == 单 codepoint` 的 (x,y) 集合，exclusions 已含）。
   该算法即标准 UAX #15 NFC：差分验证（200k seed-20260925 扫描 +
   ccc/合成 battery）中 pinned 引擎与 Python `unicodedata` 在**算法**
   层面 100% 一致（0 未解释发散）；两者行为差异**全部**来自数据版本
   （引擎 U9 vs 本机 U14，见 §20.1），已用 pinned U9/U16 UCD 表消除。
   oracle 排序：**pinned 引擎**（构建期 + 验证期）> Python
   `unicodedata`（仅诊断；其 U14 数据与引擎差 98+1 cp，见 §20.1）。
    关键回归（descending CCC，旧“不重排”实现全部 FAIL、修复后 PASS）：
    `U+0041 U+0315 U+0300`（ccc 232,230）→ **`U+00C0 U+0315`**。
    Hangul：27 个可合成 T-jamo = **U+11A8..U+11C2**（U+11C3+ 为保留区，
    **永不**合成 —— oracle 验证 `AC00+U+11F2/U+11F6` 原样不动）；
    (L,V)→S(L,V,0)，(S(L,V,0),T_t)→S(L,V,t)，音节 = 0xAC00+(L×21+V)×28+T，
    jongseong(T=t) = 0x11A8+(t−1)。
 2. **added-token 切分**：先 NFC；每位置**最长** added-token 匹配 → 单 id；
    否则取到下一个 added-token 起点之前的最大 chunk，chunk 走正则+BPE。
 3. **左优先正则预分词**（leftmost-first，PCRE/Onig 语义；`(?i:)` 仅作用于
    分支 1 的 ASCII tolower casefold）：

    | # | 分支 |
    |---|---|
    | B1 | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)` |
    | B2 | `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` |
    | B3 | `\p{N}` |
    | B4 | ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` |
    | B5 | `\s*[\r\n]+` |
    | B6 | `\s+(?!\S)` |
    | B7 | `\s+` |

    `\s`/`\pL`/`\pN`/`\pM` 用 artifact 的**精确区间表**（Unicode
    White_Space = 09-0D/20/85/**A0**/1680/2000-200A/2028/2029/202F/205F/
    3000，恰 25 cp，逐 cp 实证；**U+00A0 已含** —— 旧表漏 U+00A0，
    已修复并重生成 artifact/语料，回归见 §20.5）。L/N/M 区间来自
    pinned U16 UCD（§20.1）。
    语义要点（oracle 对齐）：B1 优先于 B2（`'mX`→`'m`+`X`）；B2 的可选前导
    字符可吃空格/非字母（`a   b`→`a`,`  `,` b`）；B6 的 `\s+` 是**贪心
    带回退**：一个 k 个 \s 的 run 后跟 \S 时匹配前 k−1 个（lookahead 在
    run 内回溯成功），末尾或单 \s 除外 —— 零宽 lookahead 本身**不消费**
    字符；BPE **只在预分词内部**合并。
 4. **ByteLevel BPE**：`byte_fallback=false`（256 单字节 token + 全部 merge
    产物均在词汇表内，artifact 构建时逐条校验）；每预分词独立；贪心取
    全位置**最低 rank** merge（tie → 最左）；输出 = 拼接字节查词汇表。

 ### 20.3 CUDLMTK1 artifact（确定性、全界检、fail-loud）

 离线转录：`tools/convert_qwen35_tokenizer.py`（tokenizers lib，无 checkpoint
 权重）→ `build/data/qwen35_tokenizer.cudaltk`（~4.7MB，gitignore）。
 格式（LE）：`"CUDLMTK1"` + u32 version(1) + u32 reserved(0) + u32
 crc32（zlib，poly 0xEDB88320，覆盖 body）+ body = 15×u32 meta（全钉死
 常量，逐条校验）+ 10 个 size-prefixed section：S1 base 词汇（[u16 长][字节]
 ×248044）；S2 merges（[u32 l][u32 r] ×247587）；S3 added（[u32 id][u8
 special][u16 len][utf8] ×33）；S4 ccc 表（**814** = pinned U9 UCD 的
 ccc>0 全集）；S5 分解表（**13232** = U9 2,060 + Hangul 音节 11,172）；
 S6 合成表（**12118** = U9 946 + Hangul 11,172，key=(a<<21)|b，**每对
 构建时过 pinned 引擎验证**）；S7–S10 WS/L/N/M 区间表（WS **10** 区间/
 25 cp；L/N/M 来自 pinned U16 UCD，**677/144/321** 区间）。数据源与
 sha 见 §20.1。C++ loader
 （`src/runtime/qwen35_tokenizer.cpp`）对 magic/version/reserved/crc/截断/
 重复/越界/区间重叠/非 UTF-8/词汇连续性**全部 fail loud**（
 `test_qwen35_tokenizer` 的 load-failure 契约逐项覆盖）。

 ### 20.4 安全契约（无裸特殊 token 字面量）

 全仓库（C++/Python 测试/文档）**不出现**任何 raw 特殊/控制 token 字符串
 （EOS、im-start/im-end 等 `<|...|>` 形态）：added token 只由 artifact
 承载（id + utf8 字节）；测试语料中它们只以 **UTF-8 hex** 出现；文档只引用
 id/名称/长度/sha。C++ 侧所需的一切字符串都来自 artifact 查表或拆分拼接，
 源码零字面量。

 ### 20.5 硬门（ctest）

 - **`test_qwen35_tokenizer_python_selftest`**（无 checkpoint）：converter
   自测（确定性双跑 bit-exact、round-trip、损坏类；pinned 资产缺失 → 77）。
 - **`test_qwen35_tokenizer`**（pinned 资产，self-skip 77）：
   (a) loader 失败契约（空/过小/坏 magic/坏 version/坏 reserved/crc 翻转/
   body 截断/header 截断）；(b) NFC 阶段向量（**descending-CCC 规范重排
   回归**（`A+U+0315+U+0300`→`U+00C0+U+0315` 等 7 条，旧“不重排”实现
   全部 FAIL）、等 ccc 稳定不重排、jamo T27、U+11F6 不合成、递归分解、
   Greek 1FEE→0385 等 oracle 验证期望）；(c) 预分词阶段向量（leftmost-first、
   nbsp 黏词、trailing space、CRLF、CJK、**孤立 combining mark**
   （B2 的 `?` 需回退，kOpt 回退回归））；(d) decode 契约（padding→""、
   越界 fail loud、is_special 21/33、EOS==248044、非法 UTF-8 fail loud）；
   (e) **跨语言精确语料**：`tools/gen_tokenizer_refs.py` 生成 **1883 行**
   = E（encode，**305** 文本，含 D1/D2/U16 类回归文本）/ D（decode
   skip_special=false，305）/ D1（skip_special=true，305）/ N（NFC，
   **65** 向量，11 类 —— 期望 = pinned 引擎 `normalizer.normalize_str`
   （U9）逐 cp 精确，含 (11) **BLOCKER-D1 回归类**：98 个 ccc 差异 cp 的
   代表性 `A+<cp>+U+0337` 序列（ccc 9/220/230 三档）+ Divès Akuru
   `U+11935 U+11930` 对及其 +x/单独 U+11938 变体）/ P（预分词，**317**
   文本 —— 期望 = pinned 引擎 `pre_tokenize_str(normalize_str(t))`（经
   ByteLevel 逆映射回 raw text），含 **25 个 White_Space cp × 11 上下文
   的边界 battery**（BLOCKER-D2 回归：U+00A0 在词/符/run 边界）+ U15/U16
   新增 L/N/M cp 的类探测）/ **X（任意 id 序列 decode，586 行 = 293 序列
   × 2 skip 模式：单非法字节、2/3/4 字节不完整、非法 lead+cont 组合、
   连续非法、合法字符跨 token 拆分、base+padding、base+added、special；
   byte→id 映射从 pinned vocab 动态推导，期望 = pinned 引擎 decode，hex
   编码）** —— C++ 逐行与 pinned oracle **EXACT** 相等。Python
   `unicodedata`/re 镜像降为**二级诊断**（U14 数据 ≠ 引擎 U9，不可作
   等价断言）。
 - **regression-first 证据（旧实现必须 FAIL 新语料）**：用 reviewed HEAD
   （63b884f，U14 表 + 9 区间 WS）的 converter 重建 artifact（sha256
   `729da2cf274cfb23819ebcfedff1c5365dea013a73786a3464696aff08b980ac`），
   跑**新**语料 → **21 行 FAIL**（恰为 D1/D2/U16 回归行：E 151/154/160/
   169/172/175/178、N 971-979、P 1097/1100/1102/1103/1292/1293/1294/1296），
   新 artifact → **1883/1883 PASS**。即新语料对旧实现是**穿透性回归**，
   非自我印证。
 - **differential validation（已固化为仓库工具，固定 seed，非 ctest）**：
   `tools/dump_qwen35_tokenizer.cpp`（CMake target
   `tool_qwen35_tokenizer_dump`，native 各阶段 dump）+ `tools/
   validate_qwen35_tokenizer.py`（pinned 引擎 sha-gated oracle + 全阶段
   EXACT 比对，打印每 pass 的 samples/mismatches/seed 与 git HEAD）。
   命令：
   ```
   cmake --build build --target tool_qwen35_tokenizer_dump
   /root/py311/venv/bin/python3 tools/validate_qwen35_tokenizer.py \
     --mode extended \
     --dump-binary build/tool_qwen35_tokenizer_dump \
     --artifact build/data/qwen35_tokenizer.cudaltk \
     --tokenizer-dir /root/models/Qwen3.5-0.8B-Base/tokenizer
   ```
   **Phase B final evidence executed on:**
   `PHASE_B_EVIDENCE_SHA = 0dc3b576d8171bedebe96d487cf83a59d5f98bef`
   （工作树干净、HEAD == 该 SHA；artifact 由该 SHA 的 converter 重新
   生成，未复用旧件）。**失效规则**：此后任何对 `src/`、`include/`、
   `tools/`、`tests/` 或 functional CMake 配置的修改都使该 evidence
   **失效**，必须整套重跑；仅 docs 修改不使其失效（此时注明最终 HEAD
   与 evidence SHA 不同）。
   extended（穷举）真实结果（于上述 SHA）：regressions 26/0；
   ws_battery 400/0；nfc_single **1,112,064/0（全 cp 穷举）**；
   enc_single **1,112,064/0**（`"a"+cp` 全 cp）；class_probe
   **3,336,190/0**（**实际比较数**：3 探针 × 全 cp，第三探针跳过
   CR/LF → 3×1,112,064−2）；decomp_cp(+enc) **13,232/0（全可分解 cp
   穷举）**；comp_pair nfc(+enc) **12,118/0（全合成对穷举）**；
   nfc_fuzz 200,000/0（seed 777）；adjacency 50,000/0（seed
   20260925）；enc_fuzz 100,000/0（seed 42）；pretok_fuzz 100,000/0
   （seed 999）；dec_fuzz 40,000/0（seed 20250417，双 skip 模式）。
   quick 模式 = 同 seed 缩减规模（stride 16 单 cp、20k/10k fuzz），
   供常规 CI。
 - **BLOCKER-D1/D2 —— 本轮已解决**（上一轮按纪律上报，本轮批准修复）：
   - **D1（Unicode 数据版本）**：修复 = converter 表改从 pinned U9 UCD
     （sha-gated）+ 引擎构建期合成对验证 + L/N/M 改 pinned U16 UCD（见
     §20.1）。**算法未动**（`nfc_impl` 冻结）—— 全部差异是数据。回归：
     N 行 971-979 + E 行 169/172/175/178（旧 U14 artifact 全部 FAIL、新
     PASS）+ nfc_single/enc_single/comp_pair/decomp_cp 全穷举 0。
   - **D2（\s 缺 U+00A0）**：修复 = `WHITE_SPACE`/`WS_RANGES` 各加
     `(0x00A0,0x00A0)` + 重生成（一行数据修复）。回归：E 行 151/154/160
     （**final-encode 级**：`"a\u00a0\u00a0b"` 旧=64,8965,65 新=64,3966,3966,65
     == 引擎）+ P 行 1097-1103 + ws_battery 25 cp 全边界 0。
   - **停止条件（本轮达成）**：上表所有 pass **0 mismatch** + 语料 1883/1883
     EXACT + regression-first 21 行对旧实现 FAIL。任何新发现的 mismatch
     一律报 BLOCKER，禁止排除 cp / 缩小域。
 - **`test_qwen35_text_generation`**（real checkpoint + tokenizer，self-skip
   77）：E2E 文本级门。4 个 **confident prompt**（2 EN + 2 CJK；
   `tools/diag_gen_gaps.py --text` 在 pinned-quantized oracle 上选得，
   每步 top1-top2 logit gap **> 0.4**：T1 `January, February, March,`
   min 0.625；T2 `1, 2, 3, ..., 10,` min 2.875；T3 `一月，二月，三月，`
   min 1.4375；T4 `一，二，三，…，十，` min 0.75）。每场景比较：
   **native encode == pinned HF encode（EXACT）** + **greedy 生成 ==
   pinned-quantized oracle（EXACT）** + **native decode == pinned HF decode
   （EXACT，skip_special_tokens=False，生成 EOS 留在文本里）** + stop_reason
   + forward_count（== t_used）+ golden `gen.prompt_text` 回环 drift 守卫；
   污染门 T1→T3→T1 二次运行**逐字段相同**。EOS 用**真实 248044**。
 - **oracle 扩展**：`tools/generate_qwen35_golden.py --gen-text`（新参数，
   Phase A `--gen-a/b-tokens` 不动）：文本经 pinned HF tokenizer 编码后走
   同一 pinned-quantized 生成 oracle，golden 额外携带 `gen.prompt_text` +
   `gen.hf_decoded` 两个 metadata 字段。

 ---

 ## 21. v0.4 Phase C —— 基础 sampling + `cudalm-generate` CLI

 Phase B（tokenizer/prompt→text）**冻结**之后（functional/evidence SHA
 `0dc3b576d8171bedebe96d487cf83a59d5f98bef`，DONE/FROZEN），Phase C 在
 其上增加：**基础 sampling**（greedy / temperature / top-k / top-p /
 seed）+ **用户可用 CLI**（raw prompt → 生成 UTF-8 文本）。保持
 single request + serial prefill/decode；不进入 v0.5 state manager /
 v0.6 scheduler。tokenizer/NFC/BPE/decode 语义未动。

 ### 21.1 Sampling API（`include/cudalm/sampling.h`，CPU-only）

 ```cpp
 struct SamplingConfig {
   float temperature = 0.0f;   // <=0 -> 冻结 greedy 路径; >0 -> sampling
   int   top_k = 0;            // 0 -> 不启用; <0 非法; >vocab_size -> clamp 到 vocab
   float top_p = 1.0f;         // (0,1]; 1.0 -> 不启用
   std::uint64_t seed = 0;     // 每请求 RNG 种子（greedy 不消耗 RNG）
 };
 ```

 - **greedy**：`temperature <= 0`（或 `SamplingConfig::greedy()`）走**冻结
   的 Phase A `argmax_bf16`**（数值最大 BF16 logit；tie → 最小 id），
   bit-for-bit，**零 RNG 消耗**。greedy config 即使携带 top-k/top-p 字段
   也是 greedy 路径（`is_greedy()` 只看 temperature）。
 - **固定流水线顺序**（与调用顺序无关，sampling 模式）：
   `temperature → top-k → top-p → normalize → sample`。
   1. **temperature**：`f[i] = logits[i] / temperature`（T > 0；非法值
      fail loud，见下）。**scaled logits / max / exp 全程用 double**：
      有限 BF16 logit 除以最小合法正 float temperature（denorm_min）
      约 2.4e83 —— 远超 float 范围但在 double 之内，因此**对每个合法
      有限正 float temperature**，有限 logits 都产生合法分布（全部
      p finite、至少一个严格为正、sum ≈ 1）；`sample_token()` 对
      vocab ≥ 1 恒返回 `[0, vocab)`（generator 另有防御性 range
      检查，`-1` 永不进 `forward_token`）。
   2. **top-k**：只留概率最高的 k 个（`top_k == 0` 不启用；`top_k < 0` 非法 fail loud；
      **`top_k > vocab_size` clamp 到 vocab_size**（已测试并文档化）；
      值相同（tie）时**最小 token id** 占名额）。
   3. **top-p**：在 **top-k 幸存者**上按概率从高到低（tie → 最小 id），
      保留 cumulative probability 达到 top_p 的**最小前缀**（`top_p == 1`
      不裁剪；**至少保留 1 个** token）。
   4. **normalize**：`p[i] = exp(f[i] - max_f) / sum_survivors`（double）
      —— **先减 max 再 exp**（数值稳定，无 overflow）；被排除 token
      p == 0。
   5. **sample**：单次 RNG 抽取 u ∈ [0,1)，按 token id 升序走 cumulative，
      首个 `u < cum` 者胜；u 落在尾部舍入缝隙时取**最后一个**幸存者。
 - **fail loud（单一校验门 `validate_sampling_config`）**：temperature
   NaN/inf、`top_k < 0`、top_p NaN/≤0/>1 → 拒绝（generator 层面：
   ok == false + error，**不 forward 任何 token**）。temperature ≤ 0
   合法（= greedy）。
 - **组合顺序证明**：CPU 门里有一个 k-then-p 与 p-alone 存活集合**不同**
   的用例（5 个 3.0 平手 + 1 个 0.0：top-k=2 + top-p=0.8 留 {0,1}；
   单独 top-p=0.8 留 {0..4}）——顺序是钉死的，不是隐式的。

 ### 21.2 RNG / 确定性（每请求，无全局随机状态）

 - 生成器：**SplitMix64**（常量与运算在 `sampling.h` 中完全指定，跨平台/
   跨编译器确定性）；`next_double()` = 高 53 位 × 2^-53 ∈ [0,1)。
 - `Sampler` 对象**每 generate() 新建**，seed 来自请求 config —— 上一
   请求消耗的 RNG 状态**不可能**污染下一次请求（Phase A/B state
   contamination 纪律在 sampler 上的延伸）。
 - 语义：**相同 (prompt, sampling config, seed, model) → 完全相同 token
   序列**；不同 seed 产生不同 sampling stream（synthetic-logits 单测
   证明：16 个 seed 的首 token 覆盖全部 4 个候选；8192 次抽取均匀在
   [20%,30%] 内）。
 - greedy 路径**不消耗 RNG**（状态不变，单测断言）。

 ### 21.3 Generator / TextGenerator 集成

 - `Qwen35Generator::generate(prompt, max_new, eos, stream,
   SamplingConfig, observer)`（新重载）；**旧 greedy API 原样保留**。
   两个 overload 共享一个 request core；greedy overload 传
   `SamplingConfig::greedy()` —— 即冻结路径本体（**generated ids /
   stop reason / forward count 完全一致**，有集成门）。EOS /
   max_new_tokens / max_seq_len 语义与 position 语义不变；仍然禁止
   re-forward last prompt token；forward-count 不变式
   `= prompt_len + generated - 1` 对 sampling 同样成立。
 - `Qwen35TextGenerator::generate_text(prompt, max_new, SamplingConfig,
   stream)`（新重载）：**仍为 thin facade**（encode → generate →
   decode 三行缝合），不复制任何 sampler/generation 逻辑；旧 greedy
   overload 委托新 overload + greedy config。
 - 新增 **CUDA runtime path = 0**：sampler 是纯 CPU（作用在 generator
   原本就 D2H 的 host logits 上）；没有新 kernel / 新 CUDA memory 分配 /
   新 stream 语义 —— Phase A CUDA sanitizer 证据可复用（见 §21.6）。

 ### 21.4 `cudalm-generate` CLI（`tools/cudalm_generate.cpp`）

 ```bash
 ./build/cudalm-generate \
   --model build/data/qwen35_08b_full.cudalm \
   --tokenizer build/data/qwen35_tokenizer.cudaltk \
   --prompt "The capital of France is" \
   --max-new-tokens 32 --temperature 0.8 --top-k 40 --top-p 0.95 --seed 42
 ```

 - 参数：`--model` / `--tokenizer` / `--prompt`（必填）；
   `--max-new-tokens`（默认 64）；`--temperature` / `--top-k` /
   `--top-p` / `--seed`（可选）；`--greedy`（显式关闭 sampling，与
   sampling 参数互斥）；`--help`/`-h`。
 - **模式解析**：默认 greedy；出现任一 `--temperature`/`--top-k`/
   `--top-p` → sampling（未显式给 `--temperature` 时默认 1.0；
   `--temperature 0` → 冻结 greedy 路径）；`--seed` 单独出现**不**
   启用 sampling（greedy 不消耗 RNG，seed 被忽略）。
 - **`--temperature` 数值 range**：文本必须能舍入到可表示的 float。
   overflow 到 inf（如 `1e40`、`inf`、`nan`）或 underflow 到 0 →
   **usage error（exit 2）**——不允许静默变 inf 或静默变 0/greedy。
   underflow 在**两层**检测：double 级（非零 double 转 float 变
   0.0f，如 `1e-50`、`7e-46`）和 **strtod 级**（文本小到 `strtod`
   本身就 range-underflow 到 ±0.0，errno ERANGE，如 `1e-5000`、
   `-1e-5000`——非零文本绝不能静默变成 0/-0）。denormal 到
   denorm_min、最大到 FLT_MAX 都是合法值（有测试钉死）；显式
   `0`/`-0` 仍是文档化的 greedy 写法。
 - 输出：成功时 stdout **只有生成的文本**（正常模式不打印 logits /
   debug tensor）；错误走 stderr。stdout 写入是 **binary-safe /
   length-aware**（`write_generated_text`：按精确字节数 fwrite +
   约定的结尾换行，短写 → exit 1）——原生 decode 合法产生的
   **embedded NUL 字节不会被截断**。
 - **exit code**：0 成功；1 运行时失败（model load failure /
   tokenizer load failure / 生成契约违规，如 empty prompt、prompt 超
   max_seq_len）；2 用法错误（缺参 / 坏参数 / 非法 sampling config，
   在**任何 CUDA 工作之前**失败）。
 - 参数解析是独立可测逻辑（`include/cudalm/generate_cli.h` +
   `src/cli/generate_cli.cpp`，CPU-only），CLI smoke 测试在其上跑真
   二进制。

 ### 21.5 测试门

 - `test_sampling`（CPU，synthetic logits，数学钉死）：greedy 兼容
   （== 冻结 argmax、零 RNG）；temperature scaling（== softmax(logits/T)
   fp32/double 对照）；top-k（含 tie → 最小 id、>vocab clamp）；top-p
   （最小前缀、≥1 保留、1.0 禁用）；k+p 组合（顺序证明）；seed 复现；
   不同 seed 不同 stream + 均匀性；非法 config（单一门 fail loud）；
   极值 logits（1000/1000 量级，max 相减无 overflow）/ 全负 logits；
   精确 tie（greedy 最小 id、sampling 均匀、k=1 钉死）；单一幸存者
   （top_k=1 任意 seed 都 = argmax）；**极值 temperature（FLT_MIN /
   denorm_min / FLT_MAX × 正/负极值与平手 logits：所有 p finite、
   sum ≈ 1、16 个 seed 下 sampled id ∈ [0, vocab)）——该组 regression
   在旧的 float 流水线下必然击穿（NaN + sample == -1）**。
 - `test_qwen35_sampling`（real checkpoint，self-skip 77）：greedy
   EXACT（旧 API == greedy-config 新 API == temperature-0 带字段 config，
   ids/stop/forward 全同）；seed 确定性（A(42) == A(42)）；
   **request 级污染门 A(42) → B(123) → A(42)**（id 级 + text 级，
   generated ids/text/stop/forward 全 EXACT —— model state reset +
   sampler RNG reset 都无跨请求污染）；sampling 健全性（id ∈ vocab、
   forward-count 不变式）；非法 config fail loud 且不 forward。
 - `test_generate_cli_args`（CPU）：--help / 缺必填 / 缺值 / 未知参数 /
   坏数值（含溢出、部分消费）/ **`--temperature` float range
   （`1e40`/`1e308`/`inf`/`nan`/`1e-50`/`7e-46`/`1e-5000`/
   `-1e-5000`（strtod 级 underflow）→ usage error；
   `0`/`-0`/`1.4e-45`/`1e-45`/`1.17549435e-38`/`3.4e38` → 合法并
   钉死精确 float 值）**/ 模式解析 / `--greedy` 互斥 / 解析后 config
   过单一校验门 / **binary-safe stdout（`write_generated_text`
   roundtrip：`ab\0cd`、单个 NUL、空串 → 逐字节全保留 + 结尾换行）**。
 - `test_cudalm_generate_cli`（真二进制 fork/exec，无 shell，CJK 原样
   传递）：--help；用法错误 exit 2 + 信息；坏 model/tokenizer 路径
   exit 1 + 信息；非法 sampling config exit 2；greedy 路径 exit 0 +
   非空文本；sampling 路径 exit 0 + **同 seed 两次 stdout 逐字节相同**；
   UTF-8（CJK）prompt exit 0 + 非空。

 ### 21.6 v0.4 最终 evidence（Phase C 完成）

 **Phase C / v0.4 functional evidence executed on:**
 `V04_EVIDENCE_SHA = 5aba21fe0b351079850600f3f8fe7f55a77c8745`（工作树
 干净、HEAD == 该 SHA）。

 **历史**：第一版 functional evidence 曾绑定
 `a008b4373a94427695ebe0dafb7086468afd9c90`；第一轮 review 之后做了
 两处 correctness 修复（极小正 temperature 的数值溢出 → double
 流水线 + sampler/generator 防御检查；CLI stdout 改 binary-safe）+
 CLI --temperature 数值 range 检查 + top_k 措辞统一 → evidence 重绑
 `cb3cb668`；第二轮 review 又发现 `--temperature` 的 **strtod 级
 underflow**（`1e-5000` 这类文本让 `strtod` 本身就 range-underflow
 到 ±0.0，静默变 greedy）→ functional 修复（commit `5aba21fe`），
 按规则 `cb3cb668` 的 evidence 再次**失效并整套重跑**（下文即新
 SHA 的结果）。中间提交 `b7fb4b1`（README/docs + `benchmarks/
 sanitizer_cudalm_generate.txt`）与 `bfd01fa` 的性质都是
 **docs/evidence-only**（含 benchmark 证据记录，不是严格
 docs-only）。

 **失效规则**：此后任何对 `src/`、`include/`、`tools/`、`tests/` 或
 functional CMake 配置的修改都使该 evidence **失效**，必须整套重跑；
 仅 docs/evidence 修改（文档 + benchmark 证据记录）不使其失效（此时
 注明最终 HEAD 与 evidence SHA 不同）。若 Phase C 之后任何人修改了
 **冻结的 tokenizer 语义**（§18/§20），必须停止并报告，而不是继续。

 **sign-off**：external reviewer 判 **PASS** —— v0.4 正式
 **DONE / FROZEN**；`v0.4-generation` 已 merge 进 main（v0.5+ 再
 开新分支）。

 证据内容（于该 SHA，全部真实执行）：
 - **完整 ctest：45/45 PASS，0 failed，0 skipped**（含 Phase A/B 全部
   既有门：`test_qwen35_generation` greedy golden EXACT、
   `test_qwen35_generation_contract`、`test_qwen35_text_generation`
   E2E EXACT、`test_qwen35_tokenizer` 1883/1883 corpus、
   `test_qwen35_tokenizer_python_selftest`（含 `tokenizers == 0.22.2`
   版本门回归）；新增 `test_sampling` / `test_qwen35_sampling` /
   `test_generate_cli_args` / `test_cudalm_generate_cli` 全 PASS）。
 - **`scripts/check_no_torch.sh`：CLEAN**（`forbidden_deps_check`
   亦在 ctest 内 PASS）。
 - **tokenizer quick differential validation：14 pass 全 0 mismatch**
   （regressions 26、ws_battery 400、nfc_single/enc_single 各
   69,504、class_probe 208,512、decomp_cp(+enc) 各 13,232、
   comp_pair 各 3,030、nfc_fuzz 20,000/seed 777、adjacency 5,000/
   seed 20260925、enc_fuzz 10,000/seed 42、pretok_fuzz 10,000/seed
   999、dec_fuzz 4,000/seed 20250417；脚本自报
   `head=5aba21fe…`）。Phase C 未修改 tokenizer 实现/工具，extended
   百万级穷举不需要重做，Phase B extended evidence 仍绑定
   `0dc3b576`。
 - **新 CUDA runtime path = 0**（sampler 纯 CPU，无新 kernel/
   分配/stream 语义）：Phase A CUDA sanitizer 证据**复用**，并于
   该 SHA 对 `cudalm-generate` greedy 路径（真实 model 加载 +
   serial prefill + 16 decode + 每步全量 logits D2H + decode）
   **重跑 compute-sanitizer：0 错误**（记录：
   `benchmarks/sanitizer_cudalm_generate.txt`）。

 ### 21.7 当前限制（v0.4 边界，明说）

 single request / serial prefill（token-by-token，correctness-first
 基线，**非**性能声明）；**无** streaming、**无** chat template /
 会话历史、**无** batching / chunked prefill、**无** multi-request /
 scheduler / continuous batching、**无** Paged KV / state pool、**无**
 beam search / repetition & frequency & presence penalty / typical /
 min-p / speculative decoding、**无** HTTP server / OpenAI API、**无**
 NCU / CUDA Graph / kernel fusion / 性能调优。**不是** production
 serving engine —— 是 v0.4 的 correctness-first 单请求生成核 +
 最小采样 + 一个真实可用的 CLI。

 ## 22. v0.5 Phase A —— Hybrid State Manager Foundations（多序列 state 控制面 + 设备 state 池）

 v0.4 **冻结**（`V04_EVIDENCE_SHA = 5aba21fe…`）之后的第一阶段。
 Phase A **只做 state ownership**：为多序列执行建立 state 控制面与
 设备 state 池。**不**接入 `Qwen35Model` forward（Phase B）、**不**
 实现 paged-attention kernel、**不**做 scheduler / admission policy /
 continuous batching / chunked prefill / streaming（后续阶段）。
 冻结的 v0.4 runtime 数学（model forward / attention kernel / DeltaNet
 kernel / generation / sampling / tokenizer / CLI）**零修改**。

 ### 22.1 落地文件（CUDALM-native，standalone，未接线）

 - `include/cudalm/fixed_id_pool.h` + （header-only）：`FixedIdPool`
   —— 两个设备池共享的 CPU 分配器核心。固定 capacity；live id 绝不
   重复发放（无 live 别名）；LIFO 复用（确定性，测试钉死）；
   double-free / 越界 id → Status error（fail loud，不 abort）；
   耗尽 → OOM Status；accounting 每步精确（capacity == used + free）。
 - `include/cudalm/qwen35_state_layout.h`（header-only，CPU）：**所有
   池尺寸只从 `Qwen35Config` 混合 schedule 推导，无 magic constant**：
   `qwen35_num_full_layers` / `qwen35_num_linear_layers`（0.8B：
   24 层、interval 4 → 6 full（3,7,11,15,19,23）+ 18 linear）；
   `qwen35_kv_page_bytes`、`qwen35_delta_slot_bytes`（公式见 22.2/22.3）。
 - `include/cudalm/kv_block_table.h`（header-only，CPU）：`KvBlockTable`
   + `KvPageSource` 接口（池实现之，CPU 测试用 counting fake）。
 - `include/cudalm/qwen35_kv_page_pool.h` + `src/runtime/…`：
   `Qwen35KvPagePool`（RAII，move-only）。
 - `include/cudalm/qwen35_delta_state_pool.h` + `src/runtime/…`：
   `Qwen35DeltaStatePool`（RAII，move-only）。
 - `include/cudalm/qwen35_state_manager.h` + `src/runtime/…`：
   `SequenceState` + `Qwen35StateManager`（统一 host 控制面）。
 - 测试：`tests/cpu/test_fixed_id_pool.cpp`、
  `tests/cpu/test_kv_block_table.cpp`、
  `tests/cpu/test_state_pool_formulas.cpp`、
  `tests/cuda/test_qwen35_state_manager.cpp`（无 checkpoint 依赖）。

 ### 22.2 KV Page Pool 契约（`Qwen35KvPagePool` + `KvBlockTable`）

 **`page_tokens` 是显式配置**（构造参数，**不**是硬编码 16；测试用
 4/8/16 三种取值）。**一个物理 page id = 所有 full-attention 层里
 同一个逻辑 token block**；一个序列只有 **一张** block table
 （`KvBlockTable`），不建 per-layer 表。

 设备布局（一个池内，K/V 各一份，full 层共享 id 空间）：

 ```
 K: bf16 [n_full_layers][num_pages][n_kv_heads][page_tokens][head_dim]
 V: bf16 [n_full_layers][num_pages][n_kv_heads][page_tokens][head_dim]
   bytes_per_page = 2 (K+V) * n_full * n_kv_heads * page_tokens * head_dim * 2
   （0.8B、page_tokens=16 时 = 2*6*2*16*256*2 = 196,608 B；测试钉死）
 ```

 每层页 slice 连续（`n_kv_heads * page_tokens * head_dim` 个 bf16）；
 层 ordinal 0..n_full-1 按层索引升序（0.8B：ordinal 0..5 = 层
 3,7,11,15,19,23），`full_layer_ordinal()/layer_of_full_ordinal()`
 双向映射。

 分配器语义（`FixedIdPool` 委托；CUDA 测试钉死）：live page 绝不
 重复；`free_page` 后可再分配（LIFO）；double-free / 越界 → Status
 error 且状态不变；耗尽 → **Status OOM（不 abort、不部分分配）**；
 accounting 精确（`capacity_pages == used + free`、
 `used_bytes == used_pages * bytes_per_page`）。

 **清零语义（显式，测试钉死）：zero-on-release** —— 构造时整池清零；
 `free_page`/`reset` 释放前把该 page 在**每一层**（K 和 V）清零
 （在池的 stream 上 ordered）。因此任何一次 acquire 拿到的 page
 保证全零，与前任写过什么无关。correctness-first：本阶段不优化
 memset 成本。池是 **single-stream 资源**（构造时传入 stream；
 内部清零都在该 stream 上）。

 `KvBlockTable`（每序列一张，共享于所有 full 层）：
 - 位置分解精确：`block = p / page_tokens`，`offset = p % page_tokens`
   （边界矩阵 0 / pt-1 / pt / pt+1 / 最后有效位 / max 全部钉死，含
   max_seq_len 不整除 pt 的 ceil 情形：`max_blocks = ceil(max_seq_len / pt)`）。
 - 表永远是**前缀**（block `[0, num_blocks)` 已分配，append-only）；
   `ensure_capacity(position)` 只补新 tail block，幂等。
 - **同一逻辑 block 只分配一次物理 page**（block 进入前缀的那次）。
 - **事务性 OOM**：tail 分配中途失败 → 本次 call 已拿到的 page 全部
   归还，表与池 accounting **完全不变**（不泄漏 page）；
   position 越界 `[0, max_seq_len)` → Status error。
 - `clear` 归还全部 page；`lookup(b)` 是纯查询（未分配 → -1，不分配）。

 ### 22.3 DeltaNet State Pool 契约（`Qwen35DeltaStatePool`）

 **一个序列一个 slot；同一 slot id 寻址所有 18（0.8B）个 DeltaNet
 层的 conv + recurrent state**（不建 per-layer slot 表）。state
 **永远在设备上**，从不拷贝到 host。

 设备布局（每 pool；按 linear 层 ordinal `l`（0..n_linear-1，升序）
 每层两个张量；slot 的每层 state 是连续 slice，整个 slot 跨层不连续
 —— 正是未来 kernel 需要的「每层一个 base + slot 偏移」）：

 ```
 conv:      bf16 [capacity, linear_conv_dim, 3]      （= [capacity, 6144, 3] @0.8B）
 recurrent: fp32 [capacity, lin_num_k_heads, lin_key_head_dim, lin_value_head_dim]
            （= [capacity, 16, 128, 128] @0.8B）
   bytes_per_slot = n_linear * (conv_dim*3*2 + nk*kd*vd*4)
   （0.8B：18 * (36,864 + 1,048,576) = 19,537,920 B；测试钉死）
 ```

 语义：`acquire_slot` / `release_slot` / `reset_slot`（live slot 就地
 清零，不变 owner）；live slot 不别名；**acquired / reused / reset
 后的 slot 保证全零**（zero-on-release + zero-on-reset，构造时整池
 清零；`release_slot`/`reset_slot` 把该 slot 在**每一层**的 conv 与
 recurrent 清零，池 stream 上 ordered）；double-release / 越界 /
 对 dead slot reset → Status error 且状态不变；耗尽 → Status OOM；
 accounting 精确。同样 single-stream 资源。

 ### 22.4 统一 `SequenceState` + `Qwen35StateManager`（生命周期契约）

 host 记录（`SequenceState`）：`id`（`SequenceId = uint64`）、
 `delta_slot`（int，-1 无）、`block_table`（`KvBlockTable`）、
 `length`（已放置 token 数）。**`SequenceId` 单调递增、永不复用**
 （从 1 起；retire 后该 id 永久失效 —— 物理 page/slot 可复用，
 外部 id 不可，杜绝 stale handle 静默指向新请求）。

 - **create**：新唯一 id + **全新全零 Delta slot**（zero-on-release
   保证）+ **空 block table**（尚无 page）+ length 0。delta slot
   OOM 时**什么都不登记**（不发 id、无半记录；恢复后下次 create
   成功，id 序列保持单调）。
 - **ensure_kv_capacity(id, position)**：使 `position` 所在 block
   （及更早 block）有物理 page（= 表对池的 `ensure_capacity`）。
   **事务性**：OOM 时表、池 accounting、used/free **完全不变**
   （无泄漏 page）。position 越界 → Status error。
 - **set_length / advance**：**纯 metadata** 长度更新
   （`0 <= new_length <= max_seq_len`；advance: `n >= 0`，
   新长度 = length + n）。**不分配 page** —— page 覆盖由调用方
   `ensure_kv_capacity` 负责（Phase B 写 token p 前先 ensure）。
   收缩允许（不归还 page；只有 reset/retire 归还）。
 - **reset**：序列**保持 live、同一 id、同一 delta slot**：length 0、
   全部 KV page 释放（释放时清零）、Delta slot **就地清零**
   （不 release/acquire → reset 不可能 OOM）。之后等价于全新序列。
 - **retire**：释放全部 KV page + delta slot（均清零）；记录删除；
   id 永不复用；之后任何操作（含 lookup）→ Status error / nullptr。
 - **lookup(id)**：live 记录指针，未知/已 retire → nullptr（查询，非错误）。

 **记账**：全部从 config 推导（22.2/22.3 公式）；manager 级
 `total_state_bytes = KV total + Delta total`、
 `used_state_bytes = KV used + Delta used`（池记账的精确和）。

 ### 22.5 本阶段明确不做（v0.5 边界，明说）

 **无** paged-attention kernel、**无** external-state model forward、
 **无** 多序列 model 执行、**无** scheduler / admission policy /
 continuous batching / batched decode / chunked prefill / streaming、
 **无** CUDA Graph / NCU / kernel fusion、**无** HTTP/OpenAI server。
 旧 `Qwen35KvCache`（per-layer `[n_kv][max_seq][head_dim]`）与
 `Qwen35DeltaNetLayer` 自持 state **原样保留**（Phase B 才会被池
 取代）；v0.4 冻结面零修改。Phase B = external hybrid state 接入
 layers/model + paged KV 读写/attention；Phase C = 多序列交错
 正确性 + 复用污染 + v0.5 最终 evidence。

## 23. v0.5 Phase B —— External Hybrid State 接入 + Paged KV Kernel（真实检查点 parity 门）

Phase A（§22，state 控制面 + 设备 state 池，standalone）之后的接线阶段。
Phase B 把 Phase A 的 external hybrid state **接入 model**，并落地
**真 paged KV kernel**（写 + 因果 decode attention）。冻结的 v0.4 数学
（legacy `forward()` / attention kernel / DeltaNet kernel / generation /
sampling / tokenizer / CLI）**零语义修改**；legacy contiguous KV cache
**原样保留**（`Qwen35KvCache` 与 `forward_token()` 不变）。

### 23.1 External-state forward API（三层接线）

- **`Qwen35DeltaNetLayer::forward_with_state(position, x_in, ext_conv,
  ext_rec, stream)`**：与 legacy `forward()` 同一条 pipeline（RMSNorm /
  in_proj / conv decode / delta-rule / out_proj / residual），只是
  conv state 与 recurrent state 指向 **caller 提供的设备指针**（池
  slot）。`forward()` 改为以层自持 `conv_state_` / `rec_state_` 委托
  同一 `forward_impl` —— **数学零修改**（同一 kernel、同一 op 序）。
- **`Qwen35FullAttentionLayer::PagedStateRef`** +
  **`forward_with_paged_state(position, x_in, ps, stream)`**：
  `PagedStateRef{ k_pages, v_pages, block_table, page_tokens,
  page_stride }`（k/v base = 该层 ordinal 的 page 0 基址，block_table
  = 设备端每序列页表，page_stride = 同一 ordinal 相邻 page 的元素
  间距，见 23.2）。**只有 KV 写（stage 6）与 attention 读（stage 7）
  被 paged kernel 替换**；RMSNorm / QKV 投影 / RoPE / gate / MLP /
  residual 全部复用冻结 stage（`forwardImpl` 尾部多一个
  `const PagedStateRef* paged = nullptr` 参数，legacy 路径为
  nullptr 时走原 `kv_->write` + 冻结 contiguous kernel，逐字节不变）。
- **`Qwen35Model::forward_token_with_state(token_id, seq_id, mgr,
  stream)`**（执行顺序，全部 stream-ordered）：
  1. model 未加载 → Status；`token_id ∉ [0, vocab)` → Status；
  2. **compatibility gate（任何 state mutation 前的最先一致性检查）**：
     `mgr.config() == config()` —— config 不一致的 manager 会以错误布局
     寻址池，直接 Status 返回；且 enforce Phase A 单 stream 契约（v0.5
     correctness-first，**不引入** cross-stream event machinery）：
     `stream == mgr.kv_pool().stream() && stream ==
     mgr.delta_pool().stream()`，否则 Status —— 两者失败均**零
     mutation**（无 KV allocation、无 Delta 变化、length 不变、不进入
     任何 layer forward，硬门钉死）；
  3. `mgr.lookup(seq_id)` → nullptr（未知 / 已 retire）→ Status；
  4. **`position = rec->length`**（位置永远从 SequenceState.length 推导，
     不接受外部 position 参数）；`position >= max_seq_len` → Status；
  5. **`mgr.ensure_kv_capacity(seq_id, position)` 最先的 state 触点**
     （在任何模型 state 变更前）→ OOM 时 Status 返回，**零 mutation**
     （无 delta 变化、无 KV 变化、length 不变、无部分 forward —— 硬门
     钉死）；
  6. block table 上设备：grow-only `block_table_scratch_`
     （`max_blocks()` 个 int 的单一设备 buffer，永不重分配）+ 把当前
     前缀 `num_blocks` 个 page id 做**每 token 一次的小 H2D
     `cudaMemcpyAsync`**（唯一跨 H/D 的每 token 元数据拷贝；page id
     精确 —— kernel 只读 `block_table[t / page_tokens]`，`t ≤
     position`，全部在当前前缀内，无 stale 读；旧 buffer 尾部残留的
     大 id 永不被触及）；
  7. embedding 行 D2D 拷贝 → 24 层循环：linear 层走
     `delta(i)->forward_with_state(position, x,
     dpool.conv_mut(ord, slot), dpool.recurrent_mut(ord, slot),
     stream)`（每层一个 base + slot 偏移，Phase A 布局天然支持）；
     full 层组 `PagedStateRef{ kpool.k_page_mut(ord, 0),
     kpool.v_page_mut(ord, 0), d_block_table, page_tokens,
     kpool.page_stride_elems() }` 走 `forward_with_paged_state`；
  8. final RMSNorm → LM head `bf16_gemv` → 全量 logits；
  9. **`mgr.advance(seq_id, 1)` 最后**（一次成功 = position=len、
     KV page 已 ensure、24 层 external forward 完成、len += 1）。

  fail-loud：config/stream mismatch（compatibility gate）、
  未知/retire 序列、`len >= max_seq_len`、KV OOM、
  非法 token —— 全部 Status 返回（不 abort），且失败时 state 零
  变化（硬门钉死）。

### 23.2 Paged KV kernel（`include/cudalm/kernels/paged_kv.h` +
`src/kernels/paged_kv.cu`）+ 池布局契约

**禁止先 gather 成 contiguous KV 再调旧 kernel** —— 本阶段 kernel 直接
在池布局上读写，块表寻址在 kernel 内完成（`paged_row(t, pt,
block_table)` → `page = block_table[t/pt]`, `off = t%pt`；
`paged_row_offset(page, off, n, pt, hd, stride) = page*stride +
(n*pt+off)*hd`）。

- **池布局契约（Phase A 布局的精确读法，钉死）**：
  `K/V 各 bf16 [n_full][num_pages][n_kv][page_tokens][head_dim]`，
  row-major —— **同一 ordinal 的相邻 page 物理相邻**（间距 =
  `page_elems = n_kv*page_tokens*head_dim` 个 bf16），**ordinal 之间
  相隔 `capacity_pages * page_elems`**。`page_ptr` =
  `(ordinal * capacity_pages + page_id) * page_elems`。因此 paged
  kernel 的 `page_stride` 参数 = **`page_elems()`**
  （`Qwen35KvPagePool::page_stride_elems()` 即此；传错为 ordinal
  间距会让 "page p" 落到 ordinal (ord+p) 的 page 0 —— Phase B 开发中
  的真实 bug，parity 门抓出后修正并固化为本契约）。kernel 的 page
  数组基址传 `k_page(ord, 0)`（选中 ordinal），precondition
  `page_stride >= n_kv*page_tokens*head_dim`。
- **`qwen35_paged_kv_write_bf16(k_src, v_src, k_pages, v_pages,
  block_table, position, page_tokens, n_kv_heads, head_dim,
  page_stride, stream)`**：把 position 的一个 token 的全部
  `n_kv*head_dim` 个 K/V 元素（rope 之后）写入
  `block_table[position/page_tokens]` 指向 page 的
  `(n*pt + off)*hd` 行。grid 向上取整 + 部分块 guard（冻结的
  contiguous 写 kernel 因 runtime 形状整除 128 从未需要 guard）。
- **`qwen35_paged_attention_decode_bf16(q, k_pages, v_pages,
  block_table, position, page_tokens, out, n_heads, n_kv_heads,
  head_dim, page_stride, scratch, stream)`**：因果 decode attention
  对 `[0, position]` 全部 KV 行，块表寻址。scratch =
  `[scores2 | probs]`，各 `n_heads*(position+1)` 个 bf16（packed，
  行距 T）。**op 序与冻结 contiguous kernel 逐一对齐**：scores 每
  (h,t) 内 d 升序累加（fp32 acc，GQA `kh = h*n_kv/n_heads`）；
  softmax 为冻结 kernel 的 1:1 镜像（3-pass、O(num_warps) 共享内存、
  同一 reduction 树）；PV 每 (h,d) 内 t 升序累加。scale =
  `1/sqrt(head_dim)`。因此 parity 门要求 **bit-identical**（atol=0 /
  memcmp），而不是 BF16 容差。
- 块表陈旧槽（`num_blocks` 之后）可能残留任意大 id —— kernel 永远
  只读 `[0, position/page_tokens]` 前缀；独立门把陈旧槽填 9999
  （越界值），sanitizer + bit-exact 双证「从未被读」。

### 23.3 Parity 纪律（Phase B 硬门）

parity 是 **runtime-vs-runtime**（legacy 冻结路径 vs 新 external
路径，同一模型、同一权重、同一 token 流），paged kernel 保持冻结 op
序 → **一切比较 bit-exact（atol=0 / memcmp）**，无容差。

- **`test_paged_kv`**（合成数据、无 checkpoint、独立硬门）：
  - 写门：非平凡物理映射（`{3,0,5,1,4,2}` / `{5,2,0,3,1,4}`），
    position ∈ {0, pt-1, pt, pt+1}，物理行 **EXACT**、未写行精确
    保持零、块表含 stale sentinel（9999）；
  - attention 门：16 组 (position × mapping)（T ∈ {1,2,4,5,6,8,9,13}，
    pt=4，4 head/2 kv/8 hd）对冻结 `qwen35_attention_decode_bf16`
    **BIT-IDENTICAL**；
  - 真实形状门：**真实 Qwen3.5-0.8B attention 维度**（直接从冻结
    `Qwen35Config::qwen35_08b()` 读取并 CHECK 钉死：**n_heads 8、
    n_kv_heads 2、head_dim 256** —— H1024/8，无手写常量）+ **2-ordinal
    miniature 池布局 fixture**（page_tokens 2 = parity 门同款小页；相邻
    page + ordinal 间距 = capacity*page_elems），映射 `{2,0,1}`，
    T ∈ {2,3,4,5,6}（从 position 2 起逐 token 跨页边界）
    **BIT-IDENTICAL**（真正 6 个 full-attention layer 的 real-checkpoint
    path 由 `test_qwen35_state_parity` 覆盖）。
- **`test_qwen35_state_parity`**（真实 Qwen3.5-0.8B checkpoint，CLI
  参数 `<full_model.cudalm> <checkpoint_dir> <python> <src_dir>`，
  SKIP 77 / TIMEOUT 1800；**evidence 环境必须真实跑，77 skip 不是
  Phase B 签核**）：
  - A = `reset_state` + 6 × `forward_token`（legacy）；
    B = 全新 `Qwen35StateManager`（page_tokens 2、池 3 page = 恰 6
    token 容量、4 delta slot）+ `create_sequence` + 6 ×
    `forward_token_with_state`（external）；token 流
    {1024, 2048, 3072, 15, 16, 17}（position 0..5，跨 pt=2 页边界，
    恰填满池）；
  - **每 step**：embedding、24 层 final、final norm、**全量
    logits[248320]** 全部 bit-identical（6 × 27 项）；
  - **终态**：18 × DeltaNet conv（bf16）+ rec（fp32）+ 6 × 全注意力
    **逻辑行 0..5 K/V**（external 侧经 host 块表从物理 page
    device→host 逐行读，legacy 侧按 `[n_kv][max_seq][hd]` 的
    `(n*max_seq + t)*hd` 逐行取）bit-identical（48 项）；
  - **OOM-before-mutation**：第 7 个 token（position 6 → block 3 >
    容量 3）必须 `!s.ok`，且前后 **48 项 state 全部 bit-identical**
    + length / num_blocks / used_state_bytes / live_pages 不变；
  - **compatibility gate**：config 不一致的 manager（n_kv_heads 4 vs
    2）→ `!s.ok` 且**零 mutation**（无 KV page allocation、无 Delta
    变化、length 不变、不进任何 layer forward）；config 一致但 stream
    不一致（单 stream 契约）→ `!s.ok` 且 **48 项 state 全部
    bit-identical** + length / num_blocks / used_state_bytes /
    live_pages 不变；
  - **fail-loud**：未知 SequenceId 999 → `!s.ok`；非法 token
    （vocab+7）→ `!s.ok` 且 length 不变；
  - **reset parity**：`reset_sequence` 后重放同一 6 token →
    每 step logits + 终态 48 项与 B 首次运行 bit-identical、
    length == 6。
- **完整 regression（于 evidence SHA）**：ctest **51/51 PASS、0
  skipped**（v0.4 全部门面回归全绿）；`scripts/check_no_torch.sh`
  **CLEAN**；`compute-sanitizer --tool memcheck` 对
  `test_paged_kv` 与 `test_qwen35_state_parity` 均
  **ERROR SUMMARY: 0 errors**。

### 23.4 本阶段明确不做（v0.5 边界，明说）

**无** 双序列交错硬门、**无** scheduler / admission policy、**无**
continuous batching / batched decode、**无** chunked prefill、**无**
streaming、**无** PagedAttention perf 工作、**无** CUDA Graph / NCU /
kernel fusion、**无** HTTP/OpenAI server。legacy 路径
（`forward()` / `forward_token()` / `Qwen35Generator` /
`cudalm-generate` CLI）行为不变、legacy contiguous cache 保留。
Phase C = 多序列交错正确性硬门 + 复用污染 + v0.5 最终 evidence。

## 24. v0.5 Phase C —— 多序列交错硬门 + v0.5 最终签核（DONE）

v0.5 Hybrid State Manager 的最后一块：证明 external-state runtime 在
**多个 sequence 交错执行**时仍严格正确。范围钉死（明说）：

- **单 model、单 CUDA stream、每 step 一次 `forward_token_with_state()`**
  —— 只是**调用顺序**在多个 SequenceId 之间交错；
- **无** scheduler / request queue / admission policy、**无**
  continuous batching / batched decode / batched GEMV、**无** chunked
  prefill / streaming、**无** multi-CUDA-stream、**无** CUDA Graph / NCU /
  kernel fusion、**无** HTTP/OpenAI API（均属 v0.6+）；
- 不改 paged-attention 数学、不重构已冻结的 Phase B runtime（本阶段
  **零** src/include 修改，纯新增测试）。

### 24.1 Interleave 硬门（`tests/cuda/test_qwen35_state_interleave.cpp`，
真实 Qwen3.5-0.8B-Base checkpoint）

- **token 流**（全部跨 `page_tokens = 2` 页边界）：
  A = {1024, 2048, 3072, 4096, 5000, 6000}（6 tok → 3 page）、
  B = {15, 16, 17, 18, 19}（5 tok → 3 page）、
  C = {7, 8, 9, 10, 11, 12}（6 tok → 3 page）；池 6 page（A+B 同时
  live）、4 delta slot。
- **独立 reference**（fresh manager、单序列独占）：每 step 记录
  24 层 final + final norm + **FULL logits[248320]**；终态记录
  18 × DeltaNet conv（bf16）+ rec（fp32）+ 6 × 全注意力**逻辑行
  K/V**（经块表从物理 page 读，无 host gather）+ length + 块表形状。
- **交错运行**（一个 manager，A+B live，非平凡 schedule
  `A0 B0 A1 A2 B1 A3 B2 A4 B3 A5 B4`）：**A 每步 == refA、B 每步
  == refB**（runtime-vs-runtime，**BIT-IDENTICAL**，atol=0）；终态
  A/B hybrid state 分别 bit-identical；A/B 物理 page 各自唯一且
  互斥。
- **跨序列隔离门**（关键 step 做**完整 hybrid state** pre/post，不
  只 length/accounting）：A4 前捕获 B（len 3）→ forward A4 → B
  bit-identical；B2 前捕获 A（len 4）→ forward B2 → A
  bit-identical。
- **retire / 复用污染门**：`retire_sequence(A)` → A 的 SequenceId
  **永久 invalid**（lookup nullptr；advance / ensure_kv_capacity 拒绝）、
  A 的 3 KV page + Delta slot 回收（**精确字节 accounting**）、B 的
  state bit-identical；随后 `create_sequence(C)` → **C id != A id**
  （id 永不复用），但 C **实际复用** A 释放的 Delta slot 与 A 释放的
  KV 物理 page（集合相等）；C 从 fresh-zero state 跑完整 token 流：
  每 step 输出 + 终态 Delta state + 逻辑 KV 与独立 fresh-C reference
  **bit-identical**；C 的执行不改变仍 live 的 B（完整 state pre/post，
  含关键 step C0）。
- **reset 单序列隔离**：`reset_sequence(B)` **只清 B**（length = 0、
  全部 page 释放、Delta slot 原地清零——逐字节验证）、live C 的 state
  bit-identical。

parity 纪律同 §23.3：runtime-vs-runtime，**一切比较 bit-exact**。

### 24.2 Evidence（于 `V05C_EVIDENCE_SHA`，clean tree）

- 完整 ctest **52/52 PASS、0 failed、0 skipped**（Phase C 门在
  evidence 环境**真实运行**，非 77 skip）；
- `scripts/check_no_torch.sh` **CLEAN**；
- `compute-sanitizer --tool memcheck` 对 `test_qwen35_state_interleave`
  （真实 checkpoint 全流程，34 次真实 forward）：**PASS + ERROR
  SUMMARY: 0 errors**（原始日志：
  `benchmarks/sanitizer_qwen35_state_interleave.txt`）。

### 24.3 v0.5 终态（明说）

**v0.5 Phase A / Phase B / Phase C 全部 DONE**，v0.5 Hybrid State
Manager **DONE**。本阶段不 merge 进 main、不自启 v0.6 —— 待 external
reviewer 签核。

## 25. v0.6 Phase A —— Request Scheduler / Control Plane（**semantics only**，DONE）

v0.5 已签核并 merge 进 main（merge commit `88ce595`，tree == `c21f8cb`）。
v0.6 在**冻结的 v0.5 multi-sequence runtime 之上**接入一个**正确、确定、
可测试**的 request scheduler / control plane。本阶段只解决：request
lifecycle、admission、waiting/running/finished、iteration scheduling、
prefill/decode progression、EOS / max-new-tokens completion、state
create/retire ownership、deterministic multi-request execution。

### 25.1 Phase A 执行模型（钉死，明说）

**Phase A 只做 scheduler semantics；GPU 执行仍然是每次一个 sequence
forward。** 一个 iteration（`Scheduler::step()`）让每个 eligible
request **最多前进一步**，每一步 = 一次单序列
`forward_token_with_state()`（单 model、单 CUDA stream）——**不是**
batched CUDA compute / true continuous batch（那是 Phase B），**无**
吞吐声明。Phase A 的目的：证明
**scheduler/control-plane 语义 == 独立执行语义**。

明确不做（v0.6 边界）：batched CUDA kernels / batched forward /
batched GEMV / true GPU continuous batch、PagedAttention 优化、chunked
prefill、multi-stream、CUDA Graph / NCU / fusion、HTTP / OpenAI API /
async、priority scheduler、beam search、speculative decoding。

### 25.2 代码结构（v0.5 runtime 零修改）

```text
include/cudalm/request.h    // RequestId / RequestStatus / FinishReason /
                            // Request（per-request SamplingConfig + Sampler）
include/cudalm/scheduler.h  // SequenceForwarder（抽象）+ ModelForwarder
                            //（真实包装 Qwen35Model）+ Scheduler
src/runtime/scheduler.cpp   // Scheduler 实现
```

- **Request**：`RequestId`（**monotonic、永不复用**；与 `SequenceId`
  是**两个独立 id 空间**——绝不把 RequestId 当 SequenceId 用）；prompt、
  `prefill_pos`、generated、`max_new_tokens`、`eos_token_id`（-1 = 无
  EOS 门）、**per-request** `SamplingConfig`（seed 在内）+ **per-request**
  `Sampler`（per-request SplitMix64 —— 两个 request 绝不共享 RNG
  progression）、status / finish_reason / `forward_count`。
- **Token progression（v0.4 语义，钉死，防 off-by-one）**：prefill
  p0..pN-1 每次一个 token，**early prefill 不采样**（next token 已知）；
  **最后一个 prompt token 的 logits 产生第一个 generated token g0**；
  之后 forward g_{k-1} → logits → 采样 g_k；EOS 命中（EOS token
  **包含**在 generated 里，同 v0.4）或 generated.size() == max_new_tokens
  结束。一个以 m 个 generated token 结束的 request 恰好 forward
  **N + m - 1** 次（`forward_count` 精确记录）。
- **Scheduler ownership**：`Request -> SequenceId lifecycle ->
  Qwen35StateManager`。Scheduler **不拥有** model / state manager /
  stream（non-owning 引用；v0.5 的 config/stream compatibility gate 在
  `forward_token_with_state` 内部 fail-loud 强制执行）。StateManager
  继续只管 sequence state；policy 不塞进 StateManager。

### 25.3 钉死的 control-plane 契约

- **admission（transactional）**：validate input → create_sequence →
  register → issue RequestId。**任何一步失败：无 half-request、无泄漏
  SequenceState、无被消耗的 RequestId**（id 只在 create_sequence 成功
  之后发放）。
- **iteration（deterministic FIFO / round-robin）**：每轮开始先
  **snapshot** 全部 non-terminal RequestId（升序 == admission 顺序），
  然后**逐 id 重新 lookup** 并各前进一步（mutation safety 硬要求：不跨
  advance 持有 iterator/pointer/reference；terminal request 自然跳过）；
  **iteration 之间** admit 的 request 从**下一轮**才开始执行（绝不插进
  当前 snapshot）。**无** priority / fairness heuristic。
- **TERMINAL EXACTLY ONCE**：request 恰好一次转入 terminal
  （EOS / max_new_tokens / cancel / fatal Status），sequence 恰好
  **retire 一次**；stale request **永不**再被 advance；terminal
  record 保留可 inspect。
- **cancel（钉死契约）**：Waiting/Running → Cancelled + retire（资源
  回收，以后不再执行）；**已 terminal → 幂等 ok**（无状态变化）；
  **未知 id → fail-loud Status error**。
- **fatal Status**：forward 失败 → 该 request Failed + retire（恰一次）；
  同一 snapshot 的其余 id **继续**推进；`step()` 返回首个错误。
  **失败的 forward 不提交 progress**（reviewer fix，钉死）：
  `prefill_pos`（prefill 时）与 `forward_count` 都不计入失败的 forward
  —— prompt `{50, 99}`，50 成功 / 99 失败 → `forward_count = 1`、
  `prefill_pos = 1`（**不是 2**）：`prefill_pos` 的 contract 是「已
  成功 forward 的 prompt token 数」。
- **`run()` failure isolation**（reviewer fix，钉死）：`run()` 的公开
  contract 是「run until every request is terminal」，因此**一个
  request 的 fatal Status 不能让其他 live request 永久停住**：失败
  request → Failed + retire（永不再被 advance），其余 request 在后续
  iteration 继续推进；全部 terminal 后 `run()` 返回遇到的**首个错误**
  （无错误则 ok）。`step()` 行为不变（同一 snapshot 中剩余 request
  继续执行、返回首个错误）。
- **采样隔离（v0.4 contract）**：每个 request 自己的 SamplingConfig +
  Sampler；interleave / admission 顺序**绝不**改变任何 request 的 token
  流（A(seed42) 交错 == A(seed42) 单独；同 prompt + 同 seed ⇒ 同流）。

### 25.4 Gates

- **CPU control-plane gate**（`tests/cpu/test_qwen35_scheduler.cpp`；无
  checkpoint / 无 model —— 确定性 fake forwarder 跑在**真实**
  Qwen35StateManager 池上，create/retire/capacity 都是真的）：lifecycle
  （Waiting/Running/terminal records 可 inspect）、monotonic never-reused
  RequestId、transactional admission（delta-slot 耗尽 + 输入校验：无泄漏
  sequence / 无 half-request / 无消耗 id；finish/cancel 后可再 admit）、
  FIFO/round-robin snapshot 语义（forward 调用顺序日志；晚 admit 进**下一
  轮**）、prefill/decode 共存、EOS + max_new_tokens（精确 token 流 +
  forward count）、cancel + retire（幂等 re-cancel / 未知 id fail-loud /
  stale 永不 advance）、per-request 采样 RNG 隔离、fatal Status（Failed
  + retire 恰一次；snapshot 其余 id 继续推进；**失败 forward 不提交
  progress**：`forward_count`/`prefill_pos` 均排除之，`{50,99}` 门钉死
  `prefill_pos = 1` 非 2；**`run()` failure isolation** 硬门：A 在
  `run()` 中 fatal、B 正常 → `!s.ok`、A Failed、B Finished、
  num_live == 0、manager 0，且 A 的 progress 钉死 + sequence 只
  retire 一次）。
- **real-checkpoint integration gate**（`tests/cuda/test_qwen35_scheduler_
  integration.cpp`；真实 Qwen3.5-0.8B-Base；无 checkpoint 时自 skip 77，
  evidence 环境**必须真实运行**）：A（3-tok prompt，max_new 3，seed 42）+
  B（7-tok prompt，max_new 2，seed 123）先 admit，2 轮后**动态 admit**
  C（2-tok prompt，max_new 3，seed 7），跑到底 —— 每个 request 的
  **generated token IDs、每个 generated step 的 FULL logits[248320]、
  forward count、finish reason** 与**独立 fresh-manager reference**
  （直接 forward，不走 scheduler）全部 **EXACT（memcmp）**；B 的完整
  hybrid state 在 length 4（A finish **之前**）与 length 5（A finish +
  retire **之后**）与 B-alone reference 同 length 状态 bit-identical
  （另一 request 的 finish/retire 不触碰仍-live request 的 state）；
  real-logits 采样隔离（同 prompt + 同 seed ⇒ X 单独 == Y 单独 ==
  X+Y 交错 相同流）。

### 25.5 Evidence（于 `V06A_EVIDENCE_SHA =
928d0a772f698bcc22e55a6d2ff1a19f48e0f037`，clean tree）

（reviewer fix round：failed-prefill progress commit + `run()` failure
isolation；旧绑定 `459ff12f1a4d1365112d65f8863a4e5010710c11` 按失效
规则声明失效、历史保留。正常路径的 off-by-one 语义不变 —— integration
gate 全量 EXACT 重跑确认。）

完整 ctest **54/54 PASS、0 failed、0 skipped**（integration gate 在
evidence 环境真实运行，非 77 skip）；`scripts/check_no_torch.sh`
**CLEAN**；`compute-sanitizer --tool memcheck` 对
`test_qwen35_scheduler_integration`（真实 checkpoint 全流程）**PASS +
ERROR SUMMARY: 0 errors**（原始日志：
`benchmarks/sanitizer_qwen35_scheduler.txt`）。

### 25.6 v0.6 边界（明说）

**v0.6 Phase A = scheduler semantics only。GPU 执行仍是每次一个
sequence forward；无吞吐改进声明、无 true batched GPU execution 声明。**
Phase A 不 merge 进 main、不自启 Phase B（batched GPU execution /
true continuous batching）—— 待 external reviewer 签核。

## 26. v0.6 Phase B —— True Batched GPU Decode 执行（**correctness-first**，DONE）

在**冻结的 v0.6 Phase A scheduler / control plane** 之上，把 GPU 执行从
「每 sequence 一次单序列 forward」升级为**真 batched GPU decode**：一个
`forward_batch_with_state` 对**整个 decode cohort 一次性遍历全部 24 层**
（真 batched GEMV / DeltaNet / paged attention），**不是** host 循环 N 次
单 `forward_token_with_state()`，**不是** N 次全层 forward（`for layer in
24` 允许）。Phase B = **serial prefill + true batched decode only**；最后一个
prompt token 保持 serial（产生 g0），g0 的 forward 在**下一个** iteration
进入 decode batch。

### 26.1 Phase B 执行模型（钉死，明说）

- **correctness-first，无吞吐声明**：目标是证明 **batched decode 的每一行
  与冻结单序列路径 bit-identical**；**不**做任何数值优化 / 性能调优 /
  Tensor Core 重写 / kernel fusion / NCU。
- **单 model + 单 CUDA stream**（沿用 v0.5/v0.6 Phase A 约束）。
- **保持**：每 scheduler iteration 每 request 恰好一个 token；Phase A 的
  FIFO / snapshot 语义；per-request sampler / RNG（per-request SplitMix64）。
- **明确不做（Phase B 边界）**：batched / chunked prefill、multi-stream、
  CUDA Graph、FlashAttention、Tensor Core 重写、kernel fusion、NCU、
  HTTP / OpenAI、priority scheduler、speculative decoding。无吞吐目标。

### 26.2 Bit-exact 契约（钉死）

每个 batched row 必须与**冻结单序列路径 `forward_token_with_state()`**
**BIT-IDENTICAL**（runtime-vs-runtime memcmp，**无**数值优化）。测试读取
layer-final `[layer, B, H]`、final-norm `[B, H]`、logits `[B, vocab]`。
batch kernel 是单 kernel 数学的 **batch 维扩展**（frozen 上游
`trump253/CUDALab @ cb6a6a9` **无** batched kernel），逐行必须与冻结单
kernel bit-identical。

### 26.3 新 API + 真 batch kernels

- **新 API**：`Qwen35Model::forward_batch_with_state(token_ids[B],
  sequence_ids[B], B, mgr, stream)` —— 对 cohort 一次性遍历 24 层；输出
  accessor：`batch_embedding_output()[B][H]`、`batch_layer_final_output(i)
  [B][H]`、`batch_final_norm_output()[B][H]`、`batch_logits()[B][vocab]`。
- **新增真 batch kernels**（`src/kernels/batch_decode.cu` +
  `include/cudalm/kernels/batch_decode.h`），每个逐行 bit-identical 到冻结
  单 kernel：
  * **W4A16 GEMV**（`batch_int4gemv_rowtile4_bf16_kernel`）：N×K int4 权重
    × bf16 激活。**GEMV stride 律（本次钉死的 bug）**：`nvec = K/32` 是
    **int4 权重行**步距（K int4 = K/32 × 16B uint4），但 bf16 **激活行** =
    K/8 uint4 = **4·nvec**；行偏移必须 `x + b*(4*nvec)`（**不是** `b*nvec`）
    —— 用 `nvec` 是 row-1（b≥1）diverge 的根因，已修。
  * **BF16 LM-head GEMV**（`batch_bf16_gemv`）：激活行步距 K/8。
  * **paged KV write**（`qwen35_paged_kv_write_bf16` batch）：行 b 只写
    block_table[b] 的 logical [0..position[b]]。
  * **paged attention decode**（`qwen35_paged_attention_decode_bf16`
    batch）。
  * **DeltaNet stateful**（conv / gbeta / delta 三条）batch over per-row
    delta slot。
  * **embedding gather / RoPE（heterogeneous positions）/ RMSNorm**：
    stateless elementwise / RMSNorm 通过 M-flatten 复用冻结 kernel。
- **heterogeneous state**：行 b 只触碰 `delta_slot[b]`、`block_table[b]`、
  logical KV `[0..position[b]]`；各行的 decode position / page 数 / block
  table 可以不同。

### 26.4 零突变 preflight + 回退（Phase A 语义保留）

`forward_batch_with_state` 的 **batch preflight 是 ZERO-MUTATION**：任何
失败 → **无突变、无部分分配**；scheduler 回退到冻结 serial 路径
（`batch_fallback_calls_++`，逐行 `advance_one`）—— Phase A 语义完整保留。

### 26.5 Scheduler 集成（不重排 cohort）

- 保持 Phase A snapshot 顺序（升序 == admission 顺序）；**不重排**地形成
  decode cohort（snapshot 中**连续** decode-ready 的 request 组成一个
  run）；**batch size 1 保持 single**。
- **decode cohort B≥2 → 恰好一次 batch forward**（绝不 B 次 single
  forward）。
- **Instrumentation**：`batch_forward_calls`、`single_forward_calls`、
  `max_batch_size`、`batch_fallback_calls`。
- **采样**：一次 D2H 拿 logits `[B, vocab]`，然后**逐行**
  `request[b].sampler.sample(logits[b])`（per-request RNG 隔离）。

### 26.6 Gates（全部 PASS，真实 Qwen3.5-0.8B-Base checkpoint）

- **kernel-level 逐行 parity**（`tests/cuda/test_qwen35_batch_kernels.cpp`；
  CUDA、**无** checkpoint、合成 LCG 输入）：batched W4A16 GEMV / BF16 GEMV /
  paged KV write / paged attention / DeltaNet（conv·gbeta·delta）逐行
  **BIT-IDENTICAL** 到冻结单 kernel。
- **full-model 真检查点硬门**（`tests/cuda/test_qwen35_batched_decode.cpp`；
  无 checkpoint 自 skip 77，evidence 环境**必须真实运行**）：**B=3
  heterogeneous**（3/5/2-token prompt、`page_tokens=2` 强制 page 边界、
  非连续 slot / page、不同 position）+ **B=1**，**两个 decode step**：每行
  embedding / **24× layer-final** / final-norm / **FULL logits[248320]** /
  最终 hybrid state（18× Delta conv·rec + 6× FA logical K/V）/ length 全部
  **BIT-IDENTICAL** 到冻结单序列路径（equivalent state 起）。
- **scheduler 真检查点 batch 门**（`tests/cuda/test_qwen35_scheduler_batch.
  cpp`）：三个**不同 prompt 长度**（3/5/2）的 request，max_new 钉死成 cohort
  先 **B=3** 后 A finish 收缩到 **B=2**：`batch_forward_calls=2`、
  `max_batch_size=3`、`batch_fallback_calls=0`、forwarder 观测的 cohort size
  序列 **恰好 {3,2}**（B≥2 cohort 恰好一次 batch forward，绝不 B 次 single
  forward）；每 request 的 generated IDs / 每生成步 FULL logits / forward
  count / finish reason / **最终 hybrid state** 与独立 fresh-manager
  reference **全部 EXACT**。
- Phase A 门 `test_qwen35_scheduler_integration` **保持全绿**。

### 26.7 Evidence（于 `V06B_EVIDENCE_SHA =
31b3ad2c3122465c3a43eee8c2b49f68f029a7a7`，clean tree）

完整 ctest **57/57 PASS、0 failed、0 skipped**（三个新 gate 在 evidence
环境真实运行，非 77 skip）；`scripts/check_no_torch.sh` **CLEAN**；
`compute-sanitizer --tool memcheck` 对 `test_qwen35_batched_decode`（真实
checkpoint 全流程，B=3×2-step + B=1）**PASS + ERROR SUMMARY: 0
errors**（原始日志：`benchmarks/sanitizer_qwen35_batched_decode.txt`）。

### 26.8 v0.6 边界 / 停（明说）

**Phase B = true batched GPU decode（correctness-first）；prefill 仍
serial；单 stream；无吞吐声明。** 本阶段**不** merge 进 main、**不**自启
Phase C —— 待 external reviewer 签核。
