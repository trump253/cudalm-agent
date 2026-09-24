"""CUDALM `.cudalm` v2 container — Python writer/reader.

Single source of truth: include/cudalm/weight_format_v2.h (normative layout)
+ docs/qwen35_architecture.md §10. Cross-checked by
tests/cpu/test_cudalm_v2_format.cpp (Python writes, C++ reads) and
tools/verify_qwen35_ingestion.py (Python reads what this module wrote).

Stdlib only (no torch, no numpy) so it can be imported from any tool.

Layout (little-endian), file sections are non-overlapping:

  Header (88 B):
    magic(8)='CUDLMW02' | version u32 (=2) | flags u32 (=0)
    | num_tensors u32 | num_metadata u16 | pad u16 (=0)
    | config_offset u64 | config_size u64
    | meta_offset u64 | meta_size u64
    | table_offset u64 | table_size u64
    | payload_offset u64 (16B-aligned) | payload_size u64

  Config: Qwen35Config blob (88 B, field order in weight_format_v2.h)
  Metadata: n x [key_len u8 | key | val_len u16 | value]
  Table:    n x [name_len u8 | name | dtype u8 | ndim u8 | pad u16
                 | dims i64[8] | offset u64 | byte_size u64 | align u8 | pad u8]
  Payload:  tensor blobs, offsets from payload start, each 16B-aligned
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants (mirror include/cudalm/weight_format_v2.h)
# ---------------------------------------------------------------------------
MAGIC = b"CUDLMW02"
VERSION = 2
CONFIG_BYTES = 88
HEADER_BYTES = 88  # 24 fixed + 8 x u64 section fields (mirror kHeaderBytes)
PAYLOAD_ALIGN = 16
MAX_NAME_LEN = 255
MAX_DIM = 100000000
MAX_TENSORS = 100000
MAX_METADATA_ENTRIES = 256
MAX_META_VALUE_LEN = 4096

# dtype ids (mirror cudalm::Dtype)
DT_FP16 = 1
DT_INT4_PACKED = 2
DT_FP16_SCALE = 3
DT_FP32 = 4
DT_BF16 = 5
_ELEMENT_BYTES = {
    DT_FP16: 2,
    DT_INT4_PACKED: 1,   # shape expressed in packed bytes
    DT_FP16_SCALE: 2,
    DT_FP32: 4,
    DT_BF16: 2,
}
DTYPE_NAMES = {
    DT_FP16: "fp16",
    DT_INT4_PACKED: "int4_packed",
    DT_FP16_SCALE: "fp16_scale",
    DT_FP32: "fp32",
    DT_BF16: "bf16",
}
VALID_DTYPES = frozenset(_ELEMENT_BYTES)

# Canonical metadata key order (writers emit in this order; readers may not
# rely on order).
META_ARCH = "arch"
META_MODEL_REPO = "model_repo"
META_MODEL_REVISION = "model_revision"
META_CONFIG_SHA256 = "config_sha256"
META_CHECKPOINT_SHA256 = "checkpoint_sha256"
META_TRANSFORMERS_COMMIT = "transformers_commit"
META_TRANSFORMERS_VERSION = "transformers_version"
META_SOURCE_DTYPE = "source_dtype"
META_GENERATOR = "generator"
# v0.3 full model: LM-head weight tying (pinned Qwen3.5-0.8B is tied, so the
# checkpoint has no separate lm_head tensor and the LM head aliases the
# embedding). Emitted as "true"/"false"; the loader requires "true" for the
# pinned 0.8B full-model contract.
META_TIE_WORD_EMBEDDINGS = "tie_word_embeddings"


class V2FormatError(ValueError):
    """Raised on any format violation (reader side) or writer misuse."""


# ---------------------------------------------------------------------------
# Qwen35Config
# ---------------------------------------------------------------------------
class Qwen35Config:
    """Fixed-width 88-byte config blob (field order is part of the format)."""

    __slots__ = (
        "hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size",
        "n_heads", "n_kv_heads", "head_dim",
        "lin_num_k_heads", "lin_num_v_heads", "lin_key_head_dim",
        "lin_value_head_dim", "lin_conv_kernel_dim",
        "full_attention_interval", "group_size", "max_seq_len",
        "eps", "rope_theta", "partial_rotary_factor",
        "mrope_section",
    )

    def __init__(self, hidden_size, num_hidden_layers, intermediate_size,
                 vocab_size, n_heads, n_kv_heads, head_dim,
                 lin_num_k_heads, lin_num_v_heads, lin_key_head_dim,
                 lin_value_head_dim, lin_conv_kernel_dim,
                 full_attention_interval, group_size, max_seq_len,
                 eps, rope_theta, partial_rotary_factor,
                 mrope_section=(11, 11, 10)) -> None:
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.intermediate_size = int(intermediate_size)
        self.vocab_size = int(vocab_size)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.lin_num_k_heads = int(lin_num_k_heads)
        self.lin_num_v_heads = int(lin_num_v_heads)
        self.lin_key_head_dim = int(lin_key_head_dim)
        self.lin_value_head_dim = int(lin_value_head_dim)
        self.lin_conv_kernel_dim = int(lin_conv_kernel_dim)
        self.full_attention_interval = int(full_attention_interval)
        self.group_size = int(group_size)
        self.max_seq_len = int(max_seq_len)
        self.eps = float(eps)
        self.rope_theta = float(rope_theta)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.mrope_section = tuple(int(x) for x in mrope_section)

    # -- derived (mirror qwen35_config.h) -----------------------------------
    @property
    def q_proj_out(self) -> int:
        return self.n_heads * self.head_dim * 2

    @property
    def kv_proj_out(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def o_proj_in(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor + 0.5)

    @property
    def linear_key_dim(self) -> int:
        return self.lin_num_k_heads * self.lin_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.lin_num_v_heads * self.lin_value_head_dim

    @property
    def linear_conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    def is_full_attention(self, i: int) -> bool:
        return (0 <= i < self.num_hidden_layers
                and (i + 1) % self.full_attention_interval == 0)

    def is_linear_attention(self, i: int) -> bool:
        # Gated DeltaNet layer (mirror qwen35_config.h is_linear_attention).
        return (0 <= i < self.num_hidden_layers
                and not self.is_full_attention(i))

    def valid(self) -> bool:
        c = self
        if (c.hidden_size <= 0 or c.num_hidden_layers <= 0
                or c.intermediate_size <= 0):
            return False
        if c.n_heads <= 0 or c.n_kv_heads <= 0 or c.head_dim <= 0:
            return False
        if c.n_heads % c.n_kv_heads != 0:
            return False
        if c.lin_num_k_heads <= 0 or c.lin_num_v_heads <= 0:
            return False
        if c.lin_num_v_heads % c.lin_num_k_heads != 0:
            return False
        if c.lin_key_head_dim <= 0 or c.lin_value_head_dim <= 0:
            return False
        if c.lin_conv_kernel_dim < 2:
            return False
        if c.full_attention_interval < 2:
            return False
        if c.group_size <= 0 or c.max_seq_len <= 0:
            return False
        if c.eps <= 0.0 or c.rope_theta <= 0.0:
            return False
        rd = c.rotary_dim
        if rd <= 0 or rd > c.head_dim or rd % 2 != 0:
            return False
        if sum(c.mrope_section) != rd // 2:
            return False
        if c.hidden_size % c.group_size != 0:
            return False
        if c.o_proj_in % c.group_size != 0:
            return False
        if c.intermediate_size % c.group_size != 0:
            return False
        if c.linear_value_dim % c.group_size != 0:
            return False
        return True

    # -- blob (field order is part of the format) ----------------------------
    def to_blob(self) -> bytes:
        return struct.pack(
            "<15i3f4i",
            self.hidden_size, self.num_hidden_layers, self.intermediate_size,
            self.vocab_size, self.n_heads, self.n_kv_heads, self.head_dim,
            self.lin_num_k_heads, self.lin_num_v_heads, self.lin_key_head_dim,
            self.lin_value_head_dim, self.lin_conv_kernel_dim,
            self.full_attention_interval, self.group_size, self.max_seq_len,
            self.eps, self.rope_theta, self.partial_rotary_factor,
            self.mrope_section[0], self.mrope_section[1],
            self.mrope_section[2], 0,
        )

    @classmethod
    def from_blob(cls, blob: bytes) -> "Qwen35Config":
        if len(blob) != CONFIG_BYTES:
            raise V2FormatError(f"config blob must be {CONFIG_BYTES} bytes")
        (hidden_size, num_hidden_layers, intermediate_size, vocab_size,
         n_heads, n_kv_heads, head_dim, lin_num_k_heads, lin_num_v_heads,
         lin_key_head_dim, lin_value_head_dim, lin_conv_kernel_dim,
         full_attention_interval, group_size, max_seq_len,
         eps, rope_theta, partial_rotary_factor,
         m0, m1, m2, reserved) = struct.unpack("<15i3f4i", blob)
        if reserved != 0:
            raise V2FormatError("config blob reserved word must be 0")
        return cls(hidden_size, num_hidden_layers, intermediate_size,
                   vocab_size, n_heads, n_kv_heads, head_dim,
                   lin_num_k_heads, lin_num_v_heads, lin_key_head_dim,
                   lin_value_head_dim, lin_conv_kernel_dim,
                   full_attention_interval, group_size, max_seq_len,
                   eps, rope_theta, partial_rotary_factor,
                   (m0, m1, m2))

    def __eq__(self, other) -> bool:
        return (isinstance(other, Qwen35Config)
                and self.to_blob() == other.to_blob())

    # Pinned Qwen3.5-0.8B text config (docs/qwen35_architecture.md §2).
    @classmethod
    def qwen35_08b(cls) -> "Qwen35Config":
        return cls(
            hidden_size=1024, num_hidden_layers=24, intermediate_size=3584,
            vocab_size=248320, n_heads=8, n_kv_heads=2, head_dim=256,
            lin_num_k_heads=16, lin_num_v_heads=16, lin_key_head_dim=128,
            lin_value_head_dim=128, lin_conv_kernel_dim=4,
            full_attention_interval=4, group_size=128, max_seq_len=262144,
            eps=1e-6, rope_theta=1e7, partial_rotary_factor=0.25,
            mrope_section=(11, 11, 10),
        )


# ---------------------------------------------------------------------------
# Tensors / container
# ---------------------------------------------------------------------------
@dataclass
class V2Tensor:
    """One tensor-table record. `dims` for INT4_PACKED are in PACKED bytes
    (last dim = K/2), exactly as in v1."""
    name: str
    dtype: int
    dims: Tuple[int, ...]
    data: bytes
    align: int = PAYLOAD_ALIGN

    def __post_init__(self) -> None:
        if self.dtype not in VALID_DTYPES:
            raise V2FormatError(f"bad dtype id {self.dtype}")
        if not (1 <= len(self.name) <= MAX_NAME_LEN):
            raise V2FormatError(f"bad name length: {self.name!r}")
        if not all(0 < d <= MAX_DIM for d in self.dims) or not 1 <= len(self.dims) <= 8:
            raise V2FormatError(f"bad dims {self.dims} for {self.name!r}")
        expected = _ELEMENT_BYTES[self.dtype]
        numel = 1
        for d in self.dims:
            numel *= d
        if len(self.data) != numel * expected:
            raise V2FormatError(
                f"{self.name!r}: data length {len(self.data)} != "
                f"expected {numel * expected} for dtype {DTYPE_NAMES[self.dtype]} {self.dims}")
        if self.align not in (1, 2, 4, 8, 16):
            raise V2FormatError(f"bad align {self.align} for {self.name!r}")

    @property
    def byte_size(self) -> int:
        return len(self.data)


class V2File:
    """In-memory .cudalm v2 container."""

    def __init__(self, config: Qwen35Config,
                 metadata: Optional[Dict[str, str]] = None,
                 tensors: Optional[List[V2Tensor]] = None) -> None:
        self.config = config
        self.metadata = dict(metadata or {})
        self.tensors: List[V2Tensor] = list(tensors or [])
        for t in self.tensors:
            if t.name in {x.name for x in self.tensors if x is not t}:
                raise V2FormatError(f"duplicate tensor name {t.name!r}")

    # ------------------------------------------------------------------ writer
    def to_bytes(self) -> bytes:
        if len(self.tensors) > MAX_TENSORS:
            raise V2FormatError("too many tensors")
        if len(self.metadata) > MAX_METADATA_ENTRIES:
            raise V2FormatError("too many metadata entries")

        config_blob = self.config.to_blob()
        assert len(config_blob) == CONFIG_BYTES

        # Layout: header | config | metadata | table | pad | payload
        meta_parts = []
        for k, v in self.metadata.items():
            kb, vb = k.encode("utf-8"), v.encode("utf-8")
            if len(kb) > 255:
                raise V2FormatError(f"metadata key too long: {k!r}")
            if len(vb) > MAX_META_VALUE_LEN:
                raise V2FormatError(f"metadata value too long: {k!r}")
            meta_parts.append(struct.pack("<B", len(kb)) + kb
                              + struct.pack("<H", len(vb)) + vb)
        meta_bytes = b"".join(meta_parts)

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
            # Padding bytes between tensors are real payload bytes: the C++
            # loader requires the declared payload section to exist in-file.
            payload_buf += b"\x00" * ((-len(payload_buf)) % PAYLOAD_ALIGN)
        table_bytes = b"".join(table_parts)
        payload_size = len(payload_buf)

        header_size = HEADER_BYTES
        config_off = header_size
        meta_off = config_off + len(config_blob)
        table_off = meta_off + len(meta_bytes)
        payload_off = table_off + len(table_bytes)
        pad = (-payload_off) % PAYLOAD_ALIGN
        payload_off += pad

        header = struct.pack(
            "<8sI I I H H Q Q Q Q Q Q Q Q",
            MAGIC, VERSION, 0, len(self.tensors), len(self.metadata), 0,
            config_off, len(config_blob),
            meta_off, len(meta_bytes),
            table_off, len(table_bytes),
            payload_off, payload_size,
        )
        assert len(header) == HEADER_BYTES
        body = header + config_blob + meta_bytes + table_bytes
        body += b"\x00" * (payload_off - len(body))
        body += bytes(payload_buf)
        return body

    # ----------------------------------------------------------------- reader
    @classmethod
    def from_bytes(cls, raw: bytes) -> "V2File":
        n = len(raw)
        if n < HEADER_BYTES:
            raise V2FormatError("file too small for v2 header")
        (magic, version, flags, num_tensors, num_metadata, pad,
         config_off, config_size, meta_off, meta_size, table_off, table_size,
         payload_off, payload_size) = struct.unpack_from(
            "<8sII I H H QQQQQQQQ", raw, 0)
        if magic != MAGIC:
            raise V2FormatError("bad magic (not CUDLMW02)")
        if version != VERSION:
            raise V2FormatError(f"unsupported version {version}")
        if flags != 0 or pad != 0:
            raise V2FormatError("reserved header words must be 0")
        if num_tensors > MAX_TENSORS or num_metadata > MAX_METADATA_ENTRIES:
            raise V2FormatError("count exceeds limit")
        if config_size != CONFIG_BYTES:
            raise V2FormatError("config_size must be 88")

        def within(off: int, size: int) -> bool:
            return off <= n and size <= n - off
        for off, size in ((config_off, config_size), (meta_off, meta_size),
                          (table_off, table_size), (payload_off, payload_size)):
            if not within(off, size):
                raise V2FormatError("section out of bounds")
        sections = sorted([(config_off, config_size), (meta_off, meta_size),
                           (table_off, table_size), (payload_off, payload_size)])
        for a, b in zip(sections, sections[1:]):
            if a[0] < b[0] + b[1] and b[0] < a[0] + a[1]:
                raise V2FormatError("sections overlap")
        if payload_off % PAYLOAD_ALIGN != 0:
            raise V2FormatError("payload_offset must be 16B-aligned")

        config = Qwen35Config.from_blob(raw[config_off:config_off + config_size])
        if not config.valid():
            raise V2FormatError("Qwen35Config blob failed validation")

        metadata: Dict[str, str] = {}
        pos, end = meta_off, meta_off + meta_size
        for _ in range(num_metadata):
            if pos + 1 > end:
                raise V2FormatError("metadata key_len out of bounds")
            klen = raw[pos]
            pos += 1
            if pos + klen + 2 > end:
                raise V2FormatError("metadata key/value_len out of bounds")
            key = raw[pos:pos + klen].decode("utf-8")
            pos += klen
            (vlen,) = struct.unpack_from("<H", raw, pos)
            pos += 2
            if vlen > MAX_META_VALUE_LEN:
                raise V2FormatError("metadata value too long")
            if pos + vlen > end:
                raise V2FormatError("metadata value out of bounds")
            metadata[key] = raw[pos:pos + vlen].decode("utf-8")
            pos += vlen
        if pos != end:
            raise V2FormatError("metadata trailing bytes")

        tensors: List[V2Tensor] = []
        seen = set()
        pos, end = table_off, table_off + table_size
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
            (toff, byte_size) = struct.unpack_from("<QQ", raw, pos)
            pos += 16
            align = raw[pos]
            pos += 1
            if raw[pos] != 0:
                raise V2FormatError(f"{name!r}: reserved pad must be 0")
            pos += 1
            if dtype not in VALID_DTYPES:
                raise V2FormatError(f"bad dtype in {name!r}")
            if not (1 <= ndim <= 8):
                raise V2FormatError(f"bad ndim in {name!r}")
            if not all(0 < d <= MAX_DIM for d in dims):
                raise V2FormatError(f"bad dim in {name!r}")
            if toff > payload_size or byte_size > payload_size - toff:
                raise V2FormatError(f"{name!r} out of payload bounds")
            if (payload_off + toff) % align != 0:
                raise V2FormatError(f"{name!r} not {align}B-aligned")
            expected = _ELEMENT_BYTES[dtype]
            numel = 1
            for d in dims:
                numel *= d
            if byte_size != numel * expected:
                raise V2FormatError(f"{name!r}: byte_size/shape mismatch")
            tensors.append(V2Tensor(name, dtype, tuple(dims),
                                    bytes(raw[payload_off + toff:
                                              payload_off + toff + byte_size]),
                                    align))
        if pos != end:
            raise V2FormatError("table trailing bytes")

        obj = cls(config, metadata, tensors)
        return obj

    # ------------------------------------------------------------- conveniences
    def find(self, name: str) -> Optional[V2Tensor]:
        for t in self.tensors:
            if t.name == name:
                return t
        return None

    def meta(self, key: str) -> Optional[str]:
        return self.metadata.get(key)


def read_v2(path: str) -> V2File:
    with open(path, "rb") as f:
        return V2File.from_bytes(f.read())


def write_v2(path: str, obj: V2File) -> None:
    with open(path, "wb") as f:
        f.write(obj.to_bytes())
