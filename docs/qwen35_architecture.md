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
  更新；推送 `v0.2-qwen35`【未开始】

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
（Phase D）**未开始**。

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
