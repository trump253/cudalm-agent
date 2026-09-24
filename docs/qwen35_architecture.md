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
   每步 / 每层独立 envelope = `实测 worst × 1.3 + 0.01`（**不让**某个 late
   layer / worst step 放宽其他 step/layer）：per-step 全 logits envelope（A 8
   步 / B 16 步各自）+ per-layer DeltaNet conv / recurrent / FullAttention KV
   envelope（B 最终状态：18× conv/recurrent + 6× KV，非适用层为 0）。实测
   worst（pinned oracle，RTX 2080 Ti）：per-step logits ≤ 0.5625（A t4）、
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
   抓 **EVERY 生成步全 [248320] logits** 逐 step **bit-exact** 比较，证明第二次
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
