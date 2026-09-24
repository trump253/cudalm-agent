"""CUDALM CUDLMG02 golden-reference container — Python writer/reader.

Mirror of include/cudalm/golden_loader_v2.h (normative layout) and
docs/qwen35_architecture.md §14. Cross-checked by
tests/cuda/test_qwen35_full_attention_golden.cpp (Python writes, C++ reads).

Stdlib only (no torch, no numpy); reuses tools/common/cudalm_v2.py for the
Qwen35Config blob, the dtype table, and the V2Tensor record (bit-identical
tensor-record format to the .cudalm v2 weight container).

Layout (little-endian), fixed 144-byte header:

  magic(8)='CUDLMG02' | version u32 (=1) | flags u32 (=0)
  | n_tensors u32 | pad u32 (=0)
  | Qwen35Config blob (88 B, cudalm_v2 field order)
  | position i32 (0 <= position < max_seq_len)
  | layer_idx i32 (full-attention layer index per the config)
  | input_seed i32 (seeded hidden-state stream seed)
  | reserved i32 (=0)
  | table_offset u64 (=144) | payload_offset u64 (16B-aligned)

  Table:   n x [name_len u8 | name | dtype u8 | ndim u8 | pad u16
                 | dims i64[8] | offset u64 | byte_size u64 | align u8 | pad u8]
          The table REGION is [table_offset, payload_offset): the n records
          occupy its front, the remaining bytes are zero padding.
  Payload: tensor blobs, offsets from payload start, each 16B-aligned
"""
from __future__ import annotations

import struct
from typing import Dict, List, Optional

from cudalm_v2 import (
    CONFIG_BYTES,
    DT_BF16,
    DT_FP32,
    DTYPE_NAMES,
    MAX_DIM,
    MAX_NAME_LEN,
    MAX_TENSORS,
    PAYLOAD_ALIGN,
    Qwen35Config,
    V2FormatError,
    V2Tensor,
    VALID_DTYPES,
    _ELEMENT_BYTES,
)

MAGIC = b"CUDLMG02"
VERSION = 1
HEADER_BYTES = 144  # 24 fixed + 88 config + 16 decode state + 16 section words
OFFSET_CONFIG = 24                            # 24 (Qwen35Config blob)
OFFSET_POSITION = OFFSET_CONFIG + CONFIG_BYTES  # 112
OFFSET_LAYER_IDX = OFFSET_POSITION + 4       # 116
OFFSET_INPUT_SEED = OFFSET_LAYER_IDX + 4     # 120
OFFSET_RESERVED = OFFSET_INPUT_SEED + 4      # 124
OFFSET_TABLE_FIELD = OFFSET_RESERVED + 4     # 128
OFFSET_PAYLOAD_FIELD = OFFSET_TABLE_FIELD + 8  # 136


class GoldenV2File:
    """In-memory CUDLMG02 golden container."""

    def __init__(self, config: Qwen35Config, position: int, layer_idx: int,
                 input_seed: int,
                 tensors: Optional[List[V2Tensor]] = None) -> None:
        if not config.valid():
            raise V2FormatError("Qwen35Config failed validation")
        if not (0 <= position < config.max_seq_len):
            raise V2FormatError("position out of bounds")
        if not (0 <= layer_idx < config.num_hidden_layers
                and (config.is_full_attention(layer_idx)
                     or config.is_linear_attention(layer_idx))):
            raise V2FormatError("layer_idx is not a valid layer type")
        self.config = config
        self.position = int(position)
        self.layer_idx = int(layer_idx)
        self.input_seed = int(input_seed)
        self.tensors: List[V2Tensor] = list(tensors or [])
        names = [t.name for t in self.tensors]
        for n in names:
            if names.count(n) != 1:
                raise V2FormatError(f"duplicate tensor name {n!r}")

    # ------------------------------------------------------------------ writer
    def to_bytes(self) -> bytes:
        if len(self.tensors) > MAX_TENSORS:
            raise V2FormatError("too many tensors")

        table_parts = []
        payload_buf = bytearray()
        for t in self.tensors:
            dims8 = list(t.dims) + [0] * (8 - len(t.dims))
            off = len(payload_buf)
            entry = (struct.pack("<B", len(t.name)) + t.name.encode("utf-8")
                     + struct.pack("<BBH", t.dtype, len(t.dims), 0)
                     + struct.pack("<8q", *dims8)
                     + struct.pack("<QQ", off, t.byte_size)
                     + struct.pack("<BB", t.align, 0))
            table_parts.append(entry)
            payload_buf += t.data
            payload_buf += b"\x00" * ((-len(payload_buf)) % PAYLOAD_ALIGN)
        table_bytes = b"".join(table_parts)

        body = struct.pack(
            "<8sIIII", MAGIC, VERSION, 0, len(self.tensors), 0)
        body += self.config.to_blob()
        body += struct.pack("<iiii", self.position, self.layer_idx,
                            self.input_seed, 0)
        # The 16-byte (table_offset, payload_offset) field that follows
        # this point is part of the 144-byte header, so the payload starts
        # at len(body) + 16 + len(table_bytes).
        payload_off = len(body) + 16 + len(table_bytes)
        payload_off += (-payload_off) % PAYLOAD_ALIGN
        body += struct.pack("<QQ", HEADER_BYTES, payload_off)
        body += table_bytes
        body += b"\x00" * (payload_off - len(body))
        body += bytes(payload_buf)
        if len(body) < HEADER_BYTES:
            raise V2FormatError("internal: body shorter than header")
        return body

    # ----------------------------------------------------------------- reader
    @classmethod
    def from_bytes(cls, raw: bytes) -> "GoldenV2File":
        n = len(raw)
        if n < HEADER_BYTES:
            raise V2FormatError("file too small for CUDLMG02 header")
        (magic, version, flags, num_tensors, pad) = struct.unpack_from(
            "<8sIIII", raw, 0)
        if magic != MAGIC:
            raise V2FormatError("bad magic (not CUDLMG02)")
        if version != VERSION:
            raise V2FormatError(f"unsupported version {version}")
        if flags != 0 or pad != 0:
            raise V2FormatError("reserved header words must be 0")
        config = Qwen35Config.from_blob(raw[OFFSET_CONFIG: OFFSET_POSITION])
        position, layer_idx, input_seed, reserved = struct.unpack_from(
            "<iiii", raw, OFFSET_POSITION)
        if reserved != 0:
            raise V2FormatError("reserved header word must be 0")
        if not (0 <= position < config.max_seq_len):
            raise V2FormatError("position out of bounds")
        if not (0 <= layer_idx < config.num_hidden_layers
                and (config.is_full_attention(layer_idx)
                     or config.is_linear_attention(layer_idx))):
            raise V2FormatError("layer_idx is not a valid layer type")
        (table_off, payload_off) = struct.unpack_from("<QQ", raw,
                                                      OFFSET_TABLE_FIELD)
        if table_off != HEADER_BYTES:
            raise V2FormatError("table_offset must be 144")
        if payload_off > n or payload_off % PAYLOAD_ALIGN != 0:
            raise V2FormatError("bad payload_offset")
        payload_size = n - payload_off
        if num_tensors > MAX_TENSORS:
            raise V2FormatError("num_tensors exceeds limit")

        tensors: List[V2Tensor] = []
        seen = set()
        pos, end = table_off, payload_off
        for _ in range(num_tensors):
            if pos + 1 > end:
                raise V2FormatError("table name_len out of bounds")
            nlen = raw[pos]
            pos += 1
            if not (1 <= nlen <= MAX_NAME_LEN):
                raise V2FormatError("bad name_len")
            if pos + nlen + 1 + 1 + 2 + 64 + 8 + 8 + 1 + 1 > end:
                raise V2FormatError("table entry out of bounds")
            name = raw[pos:pos + nlen].decode("utf-8")
            pos += nlen
            if name in seen:
                raise V2FormatError(f"duplicate tensor name {name!r}")
            seen.add(name)
            dtype = raw[pos]
            pos += 1
            ndim = raw[pos]
            pos += 1
            (resv,) = struct.unpack_from("<H", raw, pos)
            pos += 2
            if resv != 0:
                raise V2FormatError(f"{name!r}: reserved pad must be 0")
            dims = list(struct.unpack_from("<8q", raw, pos))[:ndim]
            pos += 64
            if any(d != 0 for d in struct.unpack_from("<8q", raw, pos - 64)[ndim:]):
                raise V2FormatError(f"{name!r}: unused dim must be 0")
            if dtype not in VALID_DTYPES:
                raise V2FormatError(f"bad dtype in {name!r}")
            if not (1 <= ndim <= 8):
                raise V2FormatError(f"bad ndim in {name!r}")
            if not all(0 < d <= MAX_DIM for d in dims):
                raise V2FormatError(f"bad dim in {name!r}")
            (toff, byte_size) = struct.unpack_from("<QQ", raw, pos)
            pos += 16
            align = raw[pos]
            pos += 1
            if raw[pos] != 0:
                raise V2FormatError(f"{name!r}: reserved pad must be 0")
            pos += 1
            if toff > payload_size or byte_size > payload_size - toff:
                raise V2FormatError(f"{name!r} out of payload bounds")
            if (payload_off + toff) % align != 0:
                raise V2FormatError(f"{name!r} not {align}B-aligned")
            numel = 1
            for d in dims:
                numel *= d
            if byte_size != numel * _ELEMENT_BYTES[dtype]:
                raise V2FormatError(f"{name!r}: byte_size/shape mismatch")
            tensors.append(V2Tensor(name, dtype, tuple(dims),
                                    bytes(raw[payload_off + toff:
                                              payload_off + toff + byte_size]),
                                    align))
        # Table region is [table_offset, payload_offset); records occupy
        # the front, the rest is zero padding to the 16B-aligned
        # payload_offset (no table_size field in the CUDLMG02 header).
        if any(b != 0 for b in raw[pos:end]):
            raise V2FormatError("table padding must be zero")
        return cls(config, position, layer_idx, input_seed, tensors)

    # ------------------------------------------------------------- conveniences
    def find(self, name: str) -> Optional[V2Tensor]:
        for t in self.tensors:
            if t.name == name:
                return t
        return None


def read_golden_v2(path: str) -> GoldenV2File:
    with open(path, "rb") as f:
        return GoldenV2File.from_bytes(f.read())


def write_golden_v2(path: str, obj: GoldenV2File) -> None:
    with open(path, "wb") as f:
        f.write(obj.to_bytes())


# ---------------------------------------------------------------------------
# Expected tensor set for a full-attention layer at a given position.
# Names/shapes are part of the format (docs §14); the C++ loader enforces
# the same set in GoldenFileV2::validate_golden_tensors.
# ---------------------------------------------------------------------------
def golden_tensor_expectations(config: Qwen35Config, position: int,
                               layer_idx: int) -> List[tuple]:
    H = config.hidden_size
    inter = config.intermediate_size
    B = DT_BF16
    F32 = DT_FP32
    if config.is_linear_attention(layer_idx):
        # Phase C: Gated DeltaNet decode stages (bf16 except stage.g fp32) +
        # the persistent-state hard gate (conv bf16 [conv_dim,3], recurrent
        # fp32 [n_heads, key_dim, value_dim], each before/after the step).
        conv_dim = config.linear_conv_dim
        key_dim = config.linear_key_dim
        value_dim = config.linear_value_dim
        n_heads = config.lin_num_v_heads
        hd_k = config.lin_key_head_dim
        hd_v = config.lin_value_head_dim
        return [
            ("stage.input", (1, H), B),
            ("stage.rmsnorm1", (1, H), B),
            ("stage.in_proj_qkv", (1, conv_dim), B),
            ("stage.in_proj_z", (1, value_dim), B),
            ("stage.in_proj_b", (1, n_heads), B),
            ("stage.in_proj_a", (1, n_heads), B),
            ("stage.conv_out", (1, conv_dim), B),
            ("stage.conv_silu", (1, conv_dim), B),
            ("stage.q", (1, key_dim), B),
            ("stage.k", (1, key_dim), B),
            ("stage.v", (1, value_dim), B),
            ("stage.beta", (1, n_heads), B),
            ("stage.g", (1, n_heads), F32),
            ("stage.core_out", (1, value_dim), B),
            ("stage.gated_norm", (1, value_dim), B),
            ("stage.out_proj", (1, H), B),
            ("stage.residual1", (1, H), B),
            ("stage.rmsnorm2", (1, H), B),
            ("stage.mlp_gate", (1, inter), B),
            ("stage.mlp_up", (1, inter), B),
            ("stage.silu_mul", (1, inter), B),
            ("stage.mlp_down", (1, H), B),
            ("stage.final_output", (1, H), B),
            ("state.conv_before", (conv_dim, 3), B),
            ("state.conv_after", (conv_dim, 3), B),
            ("state.recurrent_before", (n_heads, hd_k, hd_v), F32),
            ("state.recurrent_after", (n_heads, hd_k, hd_v), F32),
        ]
    # Full-attention (Phase B) tensor set.
    qo = config.n_heads * config.head_dim
    kvo = config.n_kv_heads * config.head_dim
    rows = config.n_kv_heads * (position + 1)
    return [
        ("stage.input", (1, H), B),
        ("stage.rmsnorm1", (1, H), B),
        ("stage.q_gate", (1, 2 * qo), B),
        ("stage.q", (1, qo), B),
        ("stage.att_gate", (1, qo), B),
        ("stage.k", (1, kvo), B),
        ("stage.v", (1, kvo), B),
        ("stage.q_norm", (1, qo), B),
        ("stage.k_norm", (1, kvo), B),
        ("stage.rope_q", (1, qo), B),
        ("stage.rope_k", (1, kvo), B),
        ("stage.attention_raw", (1, qo), B),
        ("stage.attention_gated", (1, qo), B),
        ("stage.o_proj", (1, H), B),
        ("stage.residual1", (1, H), B),
        ("stage.rmsnorm2", (1, H), B),
        ("stage.mlp_gate", (1, inter), B),
        ("stage.mlp_up", (1, inter), B),
        ("stage.silu_mul", (1, inter), B),
        ("stage.mlp_down", (1, H), B),
        ("stage.final_output", (1, H), B),
        ("kv.k_state", (rows, config.head_dim), B),
        ("kv.v_state", (rows, config.head_dim), B),
    ]
