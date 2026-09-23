# Qwen3.5 Architecture Contract (v0.2)

Pinned official sources + exact math contract for the Qwen3.5-0.8B hybrid
decoder bring-up. This document is the architecture oracle for v0.2: any
discrepancy between this document and the pinned sources means this document
is wrong; any discrepancy between the prompt description and the pinned
sources was resolved in favor of the pinned sources (see §2 "Deltas vs
prompt" for the ones found).

Scope: the **text** decoder (`Qwen3_5TextModel`, config key `text_config`)
only. The vision tower (`model.visual.*`) and the MTP module (`mtp.*`)
present in the checkpoint are **out of scope** for v0.2 (recorded in
backlog).

---

## 1. Official source pins

### 1.1 Model / checkpoint (fact source for shapes + values)

| item | value |
|---|---|
| model repo | `Qwen/Qwen3.5-0.8B-Base` |
| revision | branch `main`, commit `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| config.json sha256 | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| model.safetensors | `model.safetensors-00001-of-00001.safetensors`, 1,746,942,600 bytes, sha256 `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c` (also written into every converted `.cudalm` v2 file's `checkpoint_sha256` metadata) |
| weight index | `model.safetensors.index.json`, 488 tensors, `total_size = 1746882752` |
| transfer channel | `https://hf-mirror.com` (huggingface.co unreachable from this environment; mirror is transport only — fact source is the official repo) |
| local path | `/root/models/Qwen3.5-0.8B-Base/` (never committed to Git) |
| native dtype | `bfloat16` (text_config.dtype); `mamba_ssm_dtype: float32` |

### 1.2 Modeling source (fact source for math)

| item | value |
|---|---|
| repo | `huggingface/transformers` |
| commit | `fc9137225880a9d03f130634c20f9dbe36a7b8bf` — "Adding Support for Qwen3.5 (#43830)", 2026-02-09 |
| files pinned | `src/transformers/models/qwen3_5/modeling_qwen3_5.py` (sha256 `b6f02dcd1b66610df293084e00bf9bea4fc6a7e5336ffc6ff446edc7ddcd8601`), `configuration_qwen3_5.py` (sha256 `2280c6e6bd9d66d7281155243f67cf5fed2756a828af41316566afe611ff16c0`), copied under `/root/models/Qwen3.5-0.8B-Base/provenance/transformers_fc91372/` |

**Provenance conflict (documented, resolved):** the checkpoint `config.json`
declares `transformers_version: 4.57.0.dev0`, but the qwen3_5 modeling did not
exist at v4.57.0 (released 2025-10-03) — it was added to main on 2026-02-09
(commit above, contributed by Qwen, PR #43830). The version string is a stale
artifact of the export environment. The pinned commit is the official
implementation; the golden generator installs transformers **at exactly this
commit** so the oracle code byte-matches this document.

The pinned file is self-contained (all classes below are defined in
`modeling_qwen3_5.py`; no qwen3_next import). The golden environment runs the
pure-torch fallback path (no `flash-linear-attention`, no `causal-conv1d`),
i.e. `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule` /
`torch_causal_conv1d_update` — these are the reference equations below.

### 1.3 Model card (corroboration only)

`README.md` of the repo: layout `6 × (3 × (Gated DeltaNet → FFN) → 1 ×
(Gated Attention → FFN))`, DeltaNet 16 QK heads / 16 V heads @ 128, Gated
Attention 8 Q / 2 KV @ 256 with RoPE dim 64, FFN intermediate 3584 — all
consistent with config.json.

---

## 2. Model config (from official config.json, text_config)

| key | value |
|---|---|
| model_type | `qwen3_5_text` |
| architectures | `Qwen3_5ForConditionalGeneration` (multimodal wrapper; text part = `model.language_model`) |
| hidden_size | **1024** |
| intermediate_size | **3584** |
| num_hidden_layers | **24** |
| hidden_act | `silu` |
| vocab_size | 248320 |
| tie_word_embeddings | true (out of v0.2 scope — no LM head) |
| num_attention_heads | **8** (full attention) |
| num_key_value_heads | **2** (GQA, group = 4) |
| head_dim | **256** (note: q_proj out = 8·256·2 = 4096 ≠ hidden — the "Q width ≠ H" generalization of v0.1.1 is a real requirement here) |
| attention_bias | false (all projections bias-free) |
| attention_dropout | 0.0 |
| attn_output_gate | **true** |
| full_attention_interval | 4 → layer schedule below |
| linear_conv_kernel_dim | **4** |
| linear_num_key_heads | **16** |
| linear_num_value_heads | **16** |
| linear_key_head_dim | **128** |
| linear_value_head_dim | **128** |
| mamba_ssm_dtype | float32 (recurrent state dtype) |
| rms_norm_eps | **1e-6** |
| max_position_embeddings | 262144 |
| rope_parameters | `{rope_type: "default", rope_theta: 1e7, partial_rotary_factor: 0.25, mrope_section: [11,11,10], mrope_interleaved: true}` |
| mtp_num_hidden_layers | 1 (out of scope) |

### Exact layer schedule (24 layers)

```
i:    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
type: DN DN DN FA DN DN DN FA DN DN DN FA DN DN DN FA DN DN DN FA DN DN DN FA
```

`layer_types[i] = "linear_attention"` (Gated DeltaNet) for `i % 4 != 3`,
`"full_attention"` for `i % 4 == 3`. Full-attention layer indices:
**{3, 7, 11, 15, 19, 23}** (18 DeltaNet layers + 6 full-attention layers).

**Deltas vs the v0.2 prompt (official source wins):**
- Prompt assumed the pattern might be "DeltaNet×3 → Full Attention" — CONFIRMED
  by official config (`full_attention_interval=4`, last of each block).
- Prompt did not mention the model is a **multimodal** checkpoint
  (`Qwen3_5ForConditionalGeneration`, vision tower + mRoPE). Text-decode math
  is unaffected for pure-text positions (mRoPE degenerates to plain RoPE —
  see §5), but the checkpoint ingestion must select the
  `model.language_model.*` prefix and skip `model.visual.*` / `mtp.*`.

---

## 3. v0.1.1 → Qwen3.5 delta table

| # | current v0.1.1 assumption | Qwen3.5 requirement | action |
|---|---|---|---|
| 1 | RMSNorm `x * rsqrt(mean(x²)+eps) * w`, plain weight, H ∈ {1024,2048,4096,8192} | **Zero-centered** `x.float() * rsqrt(mean(x.float()²)+eps) * (1 + w.float())` → cast to input dtype; also used at head_dim=256 (per-head) and final norm | **NEW** `qwen35_rmsnorm` kernel + mode; KEEP v0.1.1 kernel unchanged |
| 2 | Gated RMSNorm does not exist | DeltaNet output norm: `(w * (x·rsqrt(mean(x²)+eps)).to(bf16)) * silu(gate)` with exact bf16 roundings at two internal boundaries, w plain (init ones), H=128 | **NEW** `qwen35_rmsnorm_gated` kernel |
| 3 | RoPE interleaved-pair, full head_dim, fp16 cos/sin **table** in weight file | **Partial rotary** (rotary dim = 64 = 0.25·256, applied to the FIRST 64 dims of each head), **rotate-half** layout, theta=1e7, cos/sin computed in fp32 from inv_freq at the position, cast to bf16; mRoPE sections [11,11,10] (no-op for pure-text positions: all 3 position dims equal) | **NEW** `qwen35_rope` kernel (compute-on-the-fly, no table); KEEP v0.1.1 kernel |
| 4 | Full attention: plain q/k/v + RoPE + causal GQA + o_proj | + q/k per-head RMSNorm (zero-centered, over head_dim=256) **before** RoPE; q_proj outputs fused `[q; gate]` (out = 2·n_heads·head_dim); attention output **× sigmoid(gate)** before o_proj; scaling = head_dim^-0.5 = 1/16; softmax in fp32 | **NEW** `Qwen35FullAttentionLayer`; KV-cache layout [n_kv_heads, seq, head_dim] (same spirit as v0.1.1 KvCache) |
| 5 | No linear-attention primitive | Gated DeltaNet: in_proj_qkv/z/b/a, depthwise causal conv1d (k=4, depthwise, no bias) with persistent conv state, delta-rule recurrence with persistent [16,128,128] fp32 state, gated RMSNorm, out_proj | **NEW** `Qwen35DeltaNetLayer` + `Qwen35DeltaState` |
| 6 | Single attention type, one `DecoderBlock` | Hybrid schedule 18×DeltaNet + 6×FullAttention; per-layer type dispatch | **NEW** `Qwen35Config` + hybrid micro-stack (v0.2: 4-layer minimum) |
| 7 | `.cudalm` v1: one block, fp16 + int4, rope tables | v2: architecture id, Qwen35Config blob, layer schedule, arbitrary named tensor table with bf16/fp32 dtypes, bounds/unique-name validation; **v1 stays immutable** | **NEW** v2 format; v1 loader untouched |
| 8 | W4A16 G=128 from **fp16** source weights | Same packed contract (q∈[-7,7], fp16 scale = amax/7, low nibble = k=2b) but source dtype is **bf16**; scale still stored fp16 (GEMV contract unchanged) | **EXTEND** quantizer (accept bf16); KEEP packed layout + GEMV |
| 9 | 7 GEMVs per block (q,k,v,o,gate,up,down) | Full-attn layers: same 7 (q_proj N=4096!); DeltaNet layers: in_proj_qkv (N=6144), in_proj_z (N=2048), in_proj_b/in_proj_a (N=16), out_proj (N=1024). All K dims are multiples of 128 ✓ | **EXTEND** GEMV usage (N=16 rows is legal: K=1024) |
| 10 | KV cache only | + DeltaNet conv state [conv_dim=6144, 3] + recurrent state [16, 128, 128] fp32 per linear layer | **NEW** state ownership |
| 11 | eps=1e-5 default | eps=**1e-6** | **EXTEND** via config (v0.1.1 constant untouched) |
| 12 | SwiGLU silu, no bias | identical (silu, bias-free) | **KEEP** |

---

## 4. Checkpoint tensor mapping

Prefix stripped for language-model tensors: `model.language_model.` (the
checkpoint is the multimodal `Qwen3_5ForConditionalGeneration`). 488 tensors
total; v0.2 ingests **187** (see per-layer lists), skips 299 vision + 8 MTP
tensors (backlog: full model).

### Per-layer tensors

Full-attention layer `i ∈ {3,7,11,15,19,23}` (`self_attn.*`, all bf16, no bias):

| tensor | shape | notes |
|---|---|---|
| `self_attn.q_proj.weight` | [4096, 1024] | out row `h*512+j`: j<256 → query head h, j≥256 → **gate** head h |
| `self_attn.k_proj.weight` | [512, 1024] | 2 KV heads × 256 |
| `self_attn.v_proj.weight` | [512, 1024] | |
| `self_attn.q_norm.weight` | [256] | zero-centered RMSNorm over head_dim |
| `self_attn.k_norm.weight` | [256] | |
| `self_attn.o_proj.weight` | [1024, 2048] | |

DeltaNet layer `i ∉ {3,7,11,15,19,23}` (`linear_attn.*`):

| tensor | shape | notes |
|---|---|---|
| `linear_attn.in_proj_qkv.weight` | [6144, 1024] | conv input = 2·key_dim + value_dim = 3·2048 |
| `linear_attn.in_proj_z.weight` | [2048, 1024] | gate for output norm |
| `linear_attn.in_proj_a.weight` | [16, 1024] | decay-rate input |
| `linear_attn.in_proj_b.weight` | [16, 1024] | beta (update gate) input |
| `linear_attn.conv1d.weight` | [6144, 1, 4] | depthwise causal conv, no bias; kept **bf16** (not a GEMV) |
| `linear_attn.dt_bias` | [16] | kept bf16 |
| `linear_attn.A_log` | [16] | **fp32** in the official checkpoint (verified in safetensors header, all 18 DeltaNet layers); v2 pass-through fp32; official code computes `A_log.float().exp()` so the decay math is fp32 regardless |
| `linear_attn.norm.weight` | [128] | gated RMSNorm weight (plain, init ones); **fp32** in the official checkpoint; v2 pass-through fp32 |
| `linear_attn.out_proj.weight` | [1024, 2048] | |

Every layer (both types):

| tensor | shape |
|---|---|
| `input_layernorm.weight` | [1024] (zero-centered) |
| `post_attention_layernorm.weight` | [1024] (zero-centered) |
| `mlp.gate_proj.weight` | [3584, 1024] |
| `mlp.up_proj.weight` | [3584, 1024] |
| `mlp.down_proj.weight` | [1024, 3584] |

Model-level (ingested for completeness of the text model; not used by the
4-layer micro-stack in v0.2 but validated at ingestion):

| tensor | shape |
|---|---|
| `embed_tokens.weight` | [248320, 1024] (bf16; out of GEMV scope) |
| `norm.weight` | [1024] (final zero-centered RMSNorm) |

**Skipped (v0.2 backlog):** all `model.visual.*` (299 tensors) and `mtp.*`
(8 tensors: `mtp.layers.0.*`, `mtp.fc`, `mtp.norm`, `mtp.pre_fc_norm_*`).

### W4A16 quantization targets (v0.2)

All 12 projection GEMVs of each layer (7 per full-attn layer, 5 per DeltaNet
layer) → packed int4 + fp16 scale, G=128 symmetric, source dtype bf16:

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

Non-GEMV bf16 tensors (layernorms, q/k_norm, conv1d, dt_bias,
embed_tokens): stored bf16 in the v2 file, unmodified. Two tensors are
stored fp32 in the official checkpoint — `linear_attn.A_log` [16] and
`linear_attn.norm.weight` [128] (per-head gated-norm weight) — and are
stored fp32 in the v2 file, unmodified. (dtype ground truth read from the
checkpoint's safetensors header, consistent across all 24 layers.)

Quantization (extends v1 contract, bf16 source):
`scale32 = amax(group)/7` (fp32); `q = clamp(round-half-to-even(W/scale32),
-7, 7)`; zero group → scale=0, q=0; **scale stored as fp16** (the stored fp16
scale is the contract, as in v1). Dequantized reference =
`q · scale_fp16.as_f32`.

---

## 5. RoPE contract (Qwen3.5, full-attention layers only)

From pinned `Qwen3_5TextRotaryEmbedding` + `apply_rotary_pos_emb`:

- `rope_type = "default"`, `rope_theta = 1e7`, `partial_rotary_factor = 0.25`
- `head_dim = 256` → **rotary_dim = 64**; `inv_freq[j] = 1e7^(-2j/64)` for
  j = 0..31 (fp32)
- For decode position p (pure text): `freqs[j] = p · inv_freq[j]` (fp32)
- **Layout = rotate-half (NOT v0.1.1 interleaved pairs):**
  `emb = [freqs(32), freqs(32)]` (64 = 2×32);
  `x_rot = x[0:64]`; `q' = q_rot·cos + rotate_half(q_rot)·sin` where
  `rotate_half(z) = [-z[32:64], z[0:32]]`; dims 64..255 pass through unchanged
- cos/sin computed in **fp32**, then **cast to bf16** before the multiply
  (official: `cos.to(dtype=x.dtype)`); the q·cos + rot·sin addition happens in
  bf16
- mRoPE: for pure-text decoding `position_ids` is `[3, bs, seq]` with all
  three dims equal to p, so `apply_interleaved_mrope` is a mathematical no-op
  (freqs identical across T/H/W). The v0.2 runtime implements the text
  specialization only; vision grids are backlog.
- Attention scaling: `head_dim ** -0.5 = 1/16` (separate from RoPE).

---

## 6. RMSNorm contracts

### 6.1 Zero-centered RMSNorm (`Qwen3_5RMSNorm`) — layernorms + q/k norm + final norm

```
y = (x_f32 · rsqrt(mean(x_f32²) + eps) · (1 + w_f32)).to(input_dtype)
```
- `w` is the **zero-centered** weight (checkpoint values; init zeros)
- `eps = 1e-6`
- used at: `input_layernorm` [1024], `post_attention_layernorm` [1024],
  `q_norm`/`k_norm` [256] (per head, last dim), final `norm` [1024]
- **v0.1.1 RMSNorm semantics (`x·rsqrt(·)·w`, plain weight) are NOT changed**
  — new kernel/mode only.

### 6.2 Gated RMSNorm (`Qwen3_5RMSNormGated`) — DeltaNet output only

Exact official dtype flow, x and gate have last dim 128:

```
1. n = (x.to(f32) · rsqrt(mean(x.to(f32)²) + eps)).to(bf16)
2. a = w · n                                # w = norm.weight (stored fp32)
3. y = (a · silu(gate.to(f32))).to(bf16)
```
`w` is stored **fp32** in the official checkpoint, so step 2 in the official
torch code promotes to fp32 (the only internal rounding is the bf16 cast in
step 1); had the loader cast `w` to bf16, step 2 would be a bf16 multiply.
The Phase B golden reference pins which case the runtime must match (the v2
file keeps `w` fp32 byte-exact either way). The CUDA kernel must mirror the
pinned flow's rounding points exactly, not just the fp32 ideal.

---

## 7. Full-attention layer contract (decode step, position p)

All projections bias-free; model dtype bf16. Input `x ∈ R^{1024}` (bf16,
batch-1, one token).

```
h0  = zero_rmsnorm(x, input_layernorm.w)                    # bf16
q_g = W_q @ h0                                              # [4096] = interleave per head
      (rows h*512..h*512+255 = q_h, h*512+256..h*512+511 = gate_h, h=0..7)
q_h = zero_rmsnorm_over_head(q_h, q_norm.w)   per head h     # over 256
k_h = zero_rmsnorm_over_head(k_h, k_norm.w)   per head (2 kv heads)
v_h = W_v @ h0                                          # [2·256]
q'_h, k'_h = partial_rotate_half_rope(q_h, k_h, p)      # §5, first 64 dims
KV cache: K[p] ← k', V[p] ← v                            # [2, p+1, 256] per kv head
attn_h: s = (q'_h · K_kv[h//4][0..p]) / 16 ; softmax_fp32 ; out = w · V_kv[...]   # [256]
out = Σ heads concat → [2048]
out = out ⊙ sigmoid(gate)                                 # gate from q_g (bf16 sigmoid)
attn_out = W_o @ out                                      # [1024]
res1 = x + attn_out                                       # bf16 add
h1  = zero_rmsnorm(res1, post_attention_layernorm.w)
mlp = W_down @ (silu(W_gate @ h1) ⊙ (W_up @ h1))          # SwiGLU, no bias
y   = res1 + mlp
```

State: KV cache only, layout `[n_kv_heads=2, seq, 256]` for K and V (bf16).

## 8. Gated DeltaNet layer contract (decode step, position p)

Dimensions: `key_dim = 16·128 = 2048`, `value_dim = 16·128 = 2048`,
`conv_dim = 6144`. All projections bias-free. Model dtype bf16.

```
mixed = W_qkv @ x            # [6144]
z     = W_z   @ x            # [2048] = 16 heads × 128 (gate for norm)
b     = W_b   @ x            # [16]
a     = W_a   @ x            # [16]

# depthwise causal conv1d, kernel 4, no bias — DECODE with conv state cs[6144][3]:
buf   = [cs(3), mixed(1)]    # [6144][4], bf16
cs    = buf[:, -3:]          # state update (in place)
c     = Σ_{j=0..3} W_conv[·,j] · buf[:,j]   (per channel; bf16)
mixed = silu(c)

q,k,v = split(mixed, [2048,2048,2048]) → reshape [16,128] each
beta  = sigmoid(b)           # bf16 [16]
g     = -exp(A_log_f32) · softplus(a_f32 + dt_bias_f32)   # fp32 [16]

# delta-rule recurrence, state S ∈ R^{16×128×128} kept in FP32:
for each head h (independent):
  q_h = l2norm(q_h, eps=1e-6); k_h = l2norm(k_h, eps=1e-6)   # fp32
  q_h = q_h / sqrt(128)
  S_h = S_h · exp(g_h)
  m   = S_h · k_h                       # [128] (contract over key dim)
  d   = (v_h - m) · beta_h
  S_h = S_h + k_h ⊗ d                   # outer product
  o_h = S_h · q_h                       # [128]   # NOTE: out uses UPDATED S

core = concat o_h → [2048] → [16,128]
out  = gated_rmsnorm(core, gate=z, w=norm.w)    # §6.2
y    = W_out @ out                              # [1024]
```

`l2norm(x) = x · rsqrt(Σ x² + 1e-6)` (FLA-conformant, per §1.2).

**State (per DeltaNet layer, explicit ownership):**
- `conv_state`: bf16 `[6144, 3]` (last kernel-1=3 tokens of the qkv stream)
- `recurrent_state`: **fp32** `[16, 128, 128]` (head, key_dim, value_dim)
- initial state = zeros (first token); both updated in place every decode step

**Order-of-operations invariants (golden-verified):** decay → delta update →
output from updated state; conv state updated BEFORE conv output read; g
computed in fp32; A_log is stored fp32 in the checkpoint (used directly,
`A_log.float()` in official code is a no-op), dt_bias stored bf16 → cast
fp32.

---

## 9. Decoder layer / micro-stack wiring

Standard pre-norm (both layer types):

```
res1 = x + mixer(zero_rmsnorm(x, input_layernorm))
y    = res1 + mlp(zero_rmsnorm(res1, post_attention_layernorm))
```

**Hybrid micro-stack (v0.2 top target):** layers 0,1,2 (DeltaNet) + layer 3
(first Full Attention) of the real checkpoint, decode steps p = 0,1,2,… with
batch-1 tokens. Inputs: seeded random bf16 hidden states [1, 1, 1024]
(embedding is out of scope — recorded in backlog); position p at step p.
Verified per step: every layer output, full-attn KV state, DeltaNet conv +
recurrent states, final micro-stack output — against the official
transformers forward at the pinned commit with the quantized weights swapped
in (see §11).

---

## 10. .cudalm v2 plan

v1 (`CUDLMW01`) is immutable. v2 (`CUDLMW02`) additions:

- magic `CUDLMW02`, version u32 = 2
- fixed config blob for `Qwen35Config` (dims above + eps + rope_theta +
  partial_rotary_factor + mrope_section[3] + full_attention_interval +
  group_size + max_seq_len)
- architecture id string (`qwen35-text`), model repo/revision and
  transformers commit + file hashes in a metadata TLV section (provenance
  travels with the weights)
- arbitrary named tensor table: name (≤255B, unique), dtype ∈ {fp16, bf16,
  fp32, int4_packed, fp16_scale, int8}, ndim ≤ 8, dims i64, offset/byte_size/
  align; 16B payload alignment; full bounds validation
- NO RoPE tables (cos/sin computed on the fly per position — §5)
- tensor names: official checkpoint names with the `model.language_model.`
  prefix (e.g. `layers.3.self_attn.q_proj.weight`, `layers.0.linear_attn.conv1d.weight`,
  `layers.0.input_layernorm.weight`, plus quantized pairs
  `*.weight`→int4 + `*.scale`→fp16) — no giant C++ switch; the loader resolves
  by name table.

Converter: `tools/convert_qwen35.py` (offline; may use Python/safetensors/
torch) reads official config.json + safetensors, applies the shared
bf16→W4A16 quantizer, writes v2. Runtime (`include/`+`src/`) stays
no-torch.

## 11. Golden strategy

`tools/generate_qwen35_golden.py` (offline):

1. installs/uses transformers pinned at `fc9137225880` (oracle code)
2. loads the real checkpoint (bf16), takes `model.language_model`
   (`Qwen3_5TextModel`)
3. swaps the 12 per-layer GEMV weights with the **dequantized W4A16** values
   (same shared quantizer as the converter) → "quantized reference"
4. runs history tokens (seeded random bf16 hidden states) through the
   official forward to build real KV + conv + recurrent state, then the
   decode step at position p
5. hooks capture: layer inputs/outputs, norm outputs, projection outputs,
   q/k after norm, after RoPE, attention gate/output, DeltaNet conv state /
   g / beta / recurrent state (before+after), FFN outputs, final output, KV
   state
6. writes a `CUDLMG02`-style golden container (same spirit as CUDLMG01)

**A/B split (hard gate vs report):**
- A. runtime correctness: CUDALM runtime (quantized weights + golden-seeded
  states) vs this golden — HARD GATE
- B. quantization fidelity: official bf16 weights vs quantized reference
  (max_abs, RMSE, cosine per weight tensor + layer outputs) — REPORT ONLY

Provenance recorded in the golden: repo, revision, transformers commit,
transformers version, input seed, positions, dtype, fast-path=off.

## 12. Milestones

- **A** — this doc + `Qwen35Config` + `.cudalm` v2 + `convert_qwen35.py` +
  ingestion test (names/shapes/dtype/values/schedule/roundtrip)
- **B** — `Qwen35FullAttentionLayer` (+ rmsnorm / rope / gated-norm kernels
  shared) + real-checkpoint golden PASS (p=0, p>0, non-zero KV history)
- **C** — `Qwen35DeltaNetLayer` + state transition golden PASS (first token,
  sequential tokens, non-zero previous state; output AND state compared)
- **D** — 4-layer hybrid micro-stack golden PASS (per-layer + all states +
  final), sequential p=0,1,2,…; benchmark three views; compute-sanitizer
  clean; docs/evidence update; push `v0.2-qwen35`

## 13. Architecture risks

| risk | mitigation |
|---|---|
| bf16 conv1d on sm_75 (Turing) in the golden env | test early; if cuDNN bf16 conv fails on GPU, run oracle on CPU bf16 or isolate conv in fp32 with the official rounding points (documented deviation only if unavoidable) |
| `@use_kernelized_func` decorator routing RoPE elsewhere | golden env has no kernel libraries → pure function path; assert at golden-gen time that the fallback path is active |
| bf16 rounding subtleties in gated RMSNorm / g / beta | golden captures staged values in model dtype; kernel mirrors the exact rounding sequence (verified per-stage, not just final output) |
| version-string conflict (4.57.0.dev0 vs modeling added 2026-02) | documented §1.2; pin = the official PR commit, not the version string |
| conv state shape (kernel-1 = 3) vs docstring `d_conv` | docstring in pinned source is imprecise; empirical layout from `torch_causal_conv1d_update` is authoritative: [conv_dim, 3] |
| q_proj fused [q;gate] layout misread | pinned source: `view(-1, head_dim*2)` then `chunk(2, dim=-1)` → per-head q then gate; verified against checkpoint shape 4096 |
| state seeding across quantized vs bf16 reference | golden builds states from the SAME quantized weights the runtime uses (no mixed-state contamination) |
