# CUDALM weight file format (`.cudalm`) — v1

A small, **deterministic, versioned, bounds-checked** binary container for a
single decoder block's weights (CUDALMW01), plus the golden-reference
container (CUDLMG01, see the second section of this document). The weight
file is written by `tools/convert_weights.py`
(Python, offline), read by `src/runtime/weight_loader.cpp` (pure C++, no
PyTorch). This document is the single source of truth; the C++ loader and the
Python writer both implement it and are cross-checked by tests.

All integers are **little-endian**. No NUL terminators. No floating-point
payload except fp16/fp32 tensor data. File layout:

```
+---------+--------------------------------------------------------------+
| Header  |  fixed, 72 bytes                                            |
+---------+--------------------------------------------------------------+
| Table   |  n_tensors × TensorRecord (variable length, packed)          |
+---------+--------------------------------------------------------------+
| Payload |  tensor byte blobs, each 16-byte aligned                     |
+---------+--------------------------------------------------------------+
```

## Header (72 bytes)

| Offset | Size | Type  | Field          | Value / meaning |
|-------:|-----:|-------|----------------|-----------------|
| 0      | 8    | 8s    | `magic`        | bytes `"CUDLMW01"` |
| 8      | 4    | u32   | `version`      | `1` (unknown → hard error) |
| 12     | 4    | u32   | `flags`        | reserved, must be `0` |
| 16     | 4    | u32   | `n_tensors`    | number of TensorRecords |
| 20     | 36   | —     | `config`       | ModelConfig blob (below) |
| 56     | 8    | u64   | `table_offset` | offset of table from file start (== 72) |
| 64     | 8    | u64   | `payload_offset` | offset of payload from file start |

### `config` blob (36 bytes, fixed order)

| Offset | Size | Type | Field             |
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

## TensorRecord (variable, packed)

| Field     | Size  | Type | Meaning |
|-----------|------:|------|---------|
| `name_len`| 4     | u32  | length of `name` (≤ 255) |
| `name`    | `name_len` | bytes | ASCII tensor name, no NUL |
| `dtype`   | 1     | u8   | `1`=FP16, `2`=INT4_PACKED, `3`=FP16_SCALE |
| `ndim`    | 1     | u8   | `1` or `2` |
| `dims`    | `4*ndim` | u32[] | shape, row-major |
| `offset`  | 8     | u64  | byte offset **from payload start** |
| `byte_size`| 8    | u64  | payload byte size |
| `align`   | 1     | u8   | required alignment (power of 2, ≤ 16) |
| `pad`     | 7     | u8   | reserved, must be `0` |

## Payload

Tensor blobs in **table order**. Region for tensor *i* starts at
`payload_offset + offset_i` and spans `byte_size_i` bytes.

- `payload_offset` is `72 + table_size` rounded **up** to a multiple of 16
  (the gap is zero-filled). This makes the payload base 16B-aligned.
- Every region start `payload_offset + offset_i` is a multiple of 16
  (the writer 16B-aligns each region; `offset` is measured from the payload
  start). `byte_size` must equal `numel(dims) * element_bytes(dtype)`:

| dtype        | element bytes | shape convention |
|--------------|--------------:|------------------|
| FP16         | 2             | logical elements |
| FP16_SCALE   | 2             | `[N, K/128]` |
| INT4_PACKED  | 1             | `[N, K/2]` (packed bytes) |

## W4A16 pairing & validation

Every `INT4_PACKED` weight named `X.weight` must have a matching
`FP16_SCALE` named `X.scale` with shape `[N, K/128]` where `K = 2 * weight.shape[1]`.
The loader validates:

- magic, version, flags
- header self-consistency (`table_offset`, `payload_offset` in range, ordered)
- every record: field bounds, `offset+byte_size ≤ payload_size`, no overlap,
  16-byte region alignment, `align` honored, `byte_size` matches dtype/shape
- unique tensor names
- config `ModelConfig::valid()`
- (when requested) the full decoder-block tensor set is present with correct
  shapes (see `BlockWeights` below)

## Tensor set for a v0.1 decoder block

| Name                  | dtype        | shape |
|-----------------------|--------------|-------|
| `attn_norm.weight`    | FP16         | `[H]` |
| `ffn_norm.weight`     | FP16         | `[H]` |
| `attn.rope_cos`       | FP16         | `[max_seq_len, head_dim/2]` |
| `attn.rope_sin`       | FP16         | `[max_seq_len, head_dim/2]` |
| `attn.q_proj.weight`  | INT4_PACKED  | `[H, H/2]` |
| `attn.q_proj.scale`   | FP16_SCALE   | `[H, H/128]` |
| `attn.k_proj.weight`  | INT4_PACKED  | `[n_kv*hd, H/2]` |
| `attn.k_proj.scale`   | FP16_SCALE   | `[n_kv*hd, H/128]` |
| `attn.v_proj.weight`  | INT4_PACKED  | `[n_kv*hd, H/2]` |
| `attn.v_proj.scale`   | FP16_SCALE   | `[n_kv*hd, H/128]` |
| `attn.o_proj.weight`  | INT4_PACKED  | `[H, H/2]` |
| `attn.o_proj.scale`   | FP16_SCALE   | `[H, H/128]` |
| `mlp.gate_proj.weight`| INT4_PACKED  | `[inter, H/2]` |
| `mlp.gate_proj.scale` | FP16_SCALE   | `[inter, H/128]` |
| `mlp.up_proj.weight`  | INT4_PACKED  | `[inter, H/2]` |
| `mlp.up_proj.scale`   | FP16_SCALE   | `[inter, H/128]` |
| `mlp.down_proj.weight`| INT4_PACKED  | `[H, inter/2]` |
| `mlp.down_proj.scale` | FP16_SCALE   | `[H, inter/128]` |

(`H = hidden_size`, `hd = head_dim`, `n_kv = n_kv_heads`,
`inter = intermediate_size`.)

## INT4 quantization contract (carried from CUDALab, offline only)

Symmetric group-wise, `group_size G = 128`, `zero_point = 0`:

```
q[n,k]    ∈ [-7, 7]
scale[n,g] = max_{k∈group g} |W[n,k]| / 7      (fp32, then cast to fp16)
q[n,k]    = clamp(round(W[n,k] / scale[n,g]), -7, 7)   (round-half-to-even)
```

- A zero group → `scale = 0`, `q ≡ 0` (no divide-by-zero, contributes 0).
- `K % 128 == 0` is required (writer and loader both reject otherwise).
- **Nibble packing** (two signed INT4 per byte, 4-bit two's complement):
  - low  nibble of `W_packed[n, b]` = element `k = 2b`
  - high nibble of `W_packed[n, b]` = element `k = 2b + 1`
- `scale` is stored as **fp16** (the fp16 rounding of the scale is part of the
  data contract; the kernel and reference both read the stored fp16 scale).

Quantization + packing happen **only** in the offline Python converter, never
in the runtime.

# CUDALM golden reference file format (`.cudalm`, CUDLMG01) — v1

The golden container embeds, in one file, everything the runtime needs to
reproduce one decoder block at one decode position: the 18 block weight
tensors (identical bytes to the CUDALMW01 weight file for the same seed),
the decode position, all 16 stage tensors of the reference simulation, and
the two KV-state tensors. Written by `tools/generate_golden.py` (Python,
offline), read by `src/runtime/golden_loader.cpp` (pure C++, no PyTorch).

It reuses the CUDALMW01 record format verbatim (TensorRecord, payload
alignment, validation rules) and differs only in the header: **80 bytes**
instead of 72, magic `"CUDLMG01"`, with two extra 32-bit fields after the
config blob.

```
+---------+--------------------------------------------------------------+
| Header  |  fixed, 80 bytes                                            |
+---------+--------------------------------------------------------------+
| Table   |  n_tensors × TensorRecord (variable length, packed)          |
+---------+--------------------------------------------------------------+
| Payload |  tensor byte blobs, each 16-byte aligned                     |
+---------+--------------------------------------------------------------+
```

## Header (80 bytes)

| Offset | Size | Type  | Field          | Value / meaning |
|-------:|-----:|-------|----------------|-----------------|
| 0      | 8    | 8s    | `magic`        | bytes `"CUDLMG01"` |
| 8      | 4    | u32   | `version`      | `1` (unknown → hard error) |
| 12     | 4    | u32   | `flags`        | reserved, must be `0` |
| 16     | 4    | u32   | `n_tensors`    | number of TensorRecords (36 for v0.1) |
| 20     | 36   | —     | `config`       | ModelConfig blob (identical to CUDALMW01) |
| 56     | 4    | i32   | `position`     | decode position `p`, `0 ≤ p < max_seq_len` |
| 60     | 4    | i32   | `reserved`     | must be `0` |
| 64     | 8    | u64   | `table_offset` | offset of table from file start (== 80) |
| 72     | 8    | u64   | `payload_offset` | offset of payload from file start |

`payload_offset` is `80 + table_size` rounded **up** to a multiple of 16
(zero-filled gap); the same region-alignment and overlap rules as CUDALMW01
apply.

## Tensor set (36 tensors, table order)

1. The 18 block weight tensors, exactly the CUDALMW01 set above (same
   names, dtypes, shapes, byte blobs).
2. The 16 stage tensors, all **FP16**, all shape `[1, X]`:

| Name | X |
|------|---|
| `stage.input` | `H` |
| `stage.rmsnorm1` | `H` |
| `stage.q` | `H` |
| `stage.k` | `n_kv·hd` |
| `stage.v` | `n_kv·hd` |
| `stage.rope_q` | `H` |
| `stage.rope_k` | `n_kv·hd` |
| `stage.attention_output` | `H` |
| `stage.output_projection` | `H` |
| `stage.residual1` | `H` |
| `stage.rmsnorm2` | `H` |
| `stage.gate` | `inter` |
| `stage.up` | `inter` |
| `stage.silu_gate_mul_up` | `inter` |
| `stage.down` | `H` |
| `stage.final_output` | `H` |

3. The two KV-state tensors, FP16, shape `[n_kv·(position+1), head_dim]`,
   flat row order **(kv_head, position)** row-major — i.e. head `h`'s row at
   position `t` is flat row `h·(position+1) + t`:

| Name |
|------|
| `kv.k_state` (row at each position is the RoPE'd k) |
| `kv.v_state` (row at each position is the projected v) |

## Reference math contract (what the C++ block must mirror)

- All arithmetic in **fp32**; an **fp16 cast at every stage boundary** (the
  stored stage tensors *are* the contract — kernels must produce the same
  stage granularity and dtypes).
- W4A16 linear: `y = (q × scale_fp16_as_fp32) @ x_fp32 → fp16`. The **stored
  fp16 scale** is the contract (not a recomputed fp32 `amax/7`).
- RMSNorm: `x · rsqrt(mean(x²) + eps) · w → fp16` (fp32 accum).
- RoPE: interleaved pairs (CUDALab `rope_v3_half2` convention):
  `y[2i] = a·c − b·s`, `y[2i+1] = a·s + b·c`, cos/sin read from the fp16
  tables at row `position`.
- Decode attention over cache rows `0..position`, stable softmax in fp32,
  `scale = 1/√head_dim`; GQA: query head `h` reads kv head
  `h · n_kv_heads / n_heads`.
- `silu_gate_mul_up = silu(gate)·up` in fp32 → fp16.
- Residuals: fp32 add of two fp16 rows → fp16.

**KV history (p > 0):** positions `0..p−1` are filled by running seeded
random hidden states (`0.05·randn(1,H)`, fp16, `history_seed`) through
attn_norm → k/v projections → RoPE; the file embeds the full cache after the
write at `p`. Therefore, for **any** position, the cache row at `position`
equals `stage.rope_k` / `stage.v` bit-for-bit (validated by the tests).

**Pinned p=0 invariants** (bit-exact, checked in
`tests/cpu/test_golden_file.cpp`): cos row 0 == 1 and sin row 0 == 0 exactly,
so `stage.rope_q == stage.q` and `stage.rope_k == stage.k` bit-for-bit; and
softmax over one element is exactly 1, so each query head's
`stage.attention_output` row equals its GQA kv head's `stage.v` row
bit-for-bit.

## Comparison tolerance (test contract)

Kernels are compared stage-by-stage against the golden fp16 rows with
`|a − r| ≤ atol + rtol·|r|`, `atol = rtol = 1e-2`
(`include/cudalm/stage_compare.h`) — one order of magnitude looser than fp16
ulp (~9.8e-4) to absorb reassociation differences, tight enough to catch any
wrong-stage / wrong-index bug.
