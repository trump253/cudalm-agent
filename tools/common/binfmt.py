"""CUDALM `.cudalm` binary container (v1) — Python reader/writer.

Single source of truth: docs/weight_format.md. This module and
`include/cudalm/weight_format.h` + `src/runtime/weight_loader.cpp` implement
the same layout and are cross-checked by tests (Python selftest + the C++
`test_weights_crosslang`).

Stdlib only (no torch, no numpy) so it can be imported from any tool.

Layout (all integers little-endian):
    Header 72 B = magic(8) | version u32 | flags u32 | n_tensors u32
                    | config blob (36 B) | table_offset u64 (=72)
                    | payload_offset u64 (16B-aligned)
    Table   n_tensors x TensorRecord (packed)
    Payload tensor blobs in table order, each region 16B-aligned
            (offsets measured from payload start)
"""
from __future__ import annotations

import struct
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants (mirror include/cudalm/weight_format.h)
# ---------------------------------------------------------------------------
MAGIC = b"CUDLMW01"
VERSION = 1
FLAGS_RESERVED = 0
CONFIG_BYTES = 36
HEADER_BYTES = 72
PAYLOAD_ALIGN = 16
MAX_NAME_LEN = 255
MAX_DIM = 10000000
MAX_TENSORS = 10000

DT_FP16 = 1
DT_INT4_PACKED = 2
DT_FP16_SCALE = 3

_ELEMENT_BYTES = {DT_FP16: 2, DT_INT4_PACKED: 1, DT_FP16_SCALE: 2}
DTYPE_NAMES = {
    DT_FP16: "FP16",
    DT_INT4_PACKED: "INT4_PACKED",
    DT_FP16_SCALE: "FP16_SCALE",
}


class ConfigError(ValueError):
    """Raised on any format violation (reader side) or writer misuse."""


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------
class ModelConfig:
    """Fixed-width 36-byte config blob (field order is part of the format)."""

    __slots__ = (
        "hidden_size", "n_heads", "n_kv_heads", "head_dim",
        "intermediate_size", "group_size", "max_seq_len", "eps", "rope_theta",
    )

    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, intermediate_size: int, group_size: int,
                 max_seq_len: int, eps: float, rope_theta: float) -> None:
        self.hidden_size = int(hidden_size)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.intermediate_size = int(intermediate_size)
        self.group_size = int(group_size)
        self.max_seq_len = int(max_seq_len)
        self.eps = float(eps)
        self.rope_theta = float(rope_theta)

    # -- blob ---------------------------------------------------------------
    def to_blob(self) -> bytes:
        return struct.pack(
            "<7i2f",
            self.hidden_size, self.n_heads, self.n_kv_heads, self.head_dim,
            self.intermediate_size, self.group_size, self.max_seq_len,
            self.eps, self.rope_theta,
        )

    @classmethod
    def from_blob(cls, b: bytes) -> "ModelConfig":
        if len(b) != CONFIG_BYTES:
            raise ConfigError(f"config blob must be {CONFIG_BYTES} bytes")
        (h, nh, nkv, hd, inter, g, msl, eps, theta) = struct.unpack("<7i2f", b)
        return cls(h, nh, nkv, hd, inter, g, msl, eps, theta)

    # -- semantics (mirror ModelConfig::valid in model_config.h) ------------
    def valid(self) -> bool:
        if self.hidden_size <= 0 or self.n_heads <= 0 or self.n_kv_heads <= 0:
            return False
        if self.head_dim <= 0 or self.intermediate_size <= 0:
            return False
        if self.max_seq_len <= 0 or self.group_size <= 0:
            return False
        if self.n_heads % self.n_kv_heads != 0:
            return False
        if self.hidden_size % self.head_dim != 0:
            return False
        if self.hidden_size % self.group_size != 0:
            return False
        if self.intermediate_size % self.group_size != 0:
            return False
        if self.head_dim % 2 != 0:
            return False
        return True

    @staticmethod
    def _f32(x: float) -> float:
        """Round through IEEE float32 (config floats are f32 in the blob,
        so defaults must hold the f32-rounded value to round-trip exactly)."""
        return struct.unpack("<f", struct.pack("<f", x))[0]

    @classmethod
    def v01_default(cls) -> "ModelConfig":
        return ModelConfig(
            hidden_size=1024, n_heads=8, n_kv_heads=4, head_dim=128,
            intermediate_size=2816, group_size=128, max_seq_len=512,
            eps=cls._f32(1e-5), rope_theta=cls._f32(10000.0),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return all(getattr(self, s) == getattr(other, s) for s in self.__slots__)

    def __repr__(self) -> str:
        fields = ", ".join(f"{s}={getattr(self, s)!r}" for s in self.__slots__)
        return f"ModelConfig({fields})"


# ---------------------------------------------------------------------------
# TensorRecord
# ---------------------------------------------------------------------------
class TensorRecord:
    """One table entry. `data` (bytes) is set by the reader, or must match
    `blob` supplied to the writer; `offset`/`byte_size` are assigned by
    `WeightFileWriter.build` (writer side) or parsed (reader side)."""

    def __init__(self, name: str, dtype: int, dims: Sequence[int],
                 offset: int = 0, byte_size: int = 0, align: int = 16) -> None:
        self.name = str(name)
        self.dtype = int(dtype)
        self.dims = [int(d) for d in dims]
        self.offset = int(offset)
        self.byte_size = int(byte_size)
        self.align = int(align)
        self.data: Optional[bytes] = None  # payload bytes (reader / writer)

    @property
    def expected_bytes(self) -> int:
        numel = 1
        for d in self.dims:
            numel *= d
        if numel <= 0:
            return 0
        return numel * _ELEMENT_BYTES.get(self.dtype, 0)

    def encode(self) -> bytes:
        """Serialize the record (name first, then fixed fields)."""
        name_b = self.name.encode("ascii")
        return (
            struct.pack("<I", len(name_b))
            + name_b
            + struct.pack("<BB", self.dtype, len(self.dims))
            + struct.pack(f"<{len(self.dims)}I", *self.dims)
            + struct.pack("<QQB", self.offset, self.byte_size, self.align)
            + b"\x00" * 7  # reserved pad (must be zero)
        )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class WeightFileWriter:
    """Deterministic writer. Tensors are placed in table order; each region
    start is 16B-aligned relative to the payload start; the payload base is
    16B-aligned (gap zero-filled)."""

    def __init__(self, cfg: ModelConfig) -> None:
        if not cfg.valid():
            raise ConfigError("invalid ModelConfig")
        self.cfg = cfg
        self.records: List[TensorRecord] = []
        self._seen = set()

    def add(self, name: str, dtype: int, dims: Sequence[int], blob: bytes) -> None:
        rec = TensorRecord(name, dtype, dims)
        exp = rec.expected_bytes
        if exp == 0:
            raise ConfigError(f"{name}: bad dtype/dims")
        if len(blob) != exp:
            raise ConfigError(
                f"{name}: blob {len(blob)} bytes != expected {exp}")
        if dtype not in _ELEMENT_BYTES:
            raise ConfigError(f"{name}: unknown dtype {dtype}")
        if not (1 <= len(name) <= MAX_NAME_LEN):
            raise ConfigError(f"bad name length: {name!r}")
        name.encode("ascii")  # raises on non-ASCII
        if name in self._seen:
            raise ConfigError(f"duplicate tensor name: {name}")
        self._seen.add(name)
        rec.data = bytes(blob)
        self.records.append(rec)

    def build(self) -> bytes:
        if not (1 <= len(self.records) <= MAX_TENSORS):
            raise ConfigError("bad n_tensors")
        # Pass 1: per-record payload offsets (16B aligned, table order).
        off = 0
        for r in self.records:
            while off % PAYLOAD_ALIGN != 0:
                off += 1
            r.offset = off
            r.byte_size = len(r.data)
            off += r.byte_size
        payload_size = off

        # Pass 2: table bytes.
        table = b"".join(r.encode() for r in self.records)
        payload_off = HEADER_BYTES + len(table)
        while payload_off % PAYLOAD_ALIGN != 0:
            payload_off += 1

        out = bytearray()
        out += MAGIC
        out += struct.pack("<III", VERSION, FLAGS_RESERVED, len(self.records))
        out += self.cfg.to_blob()
        out += struct.pack("<QQ", HEADER_BYTES, payload_off)
        out += table
        out += b"\x00" * (payload_off - len(out))  # zero gap
        cur = 0
        for r in self.records:
            while cur % PAYLOAD_ALIGN != 0:
                out += b"\x00"
                cur += 1
            out += r.data
            cur += len(r.data)
        if len(out) != payload_off + payload_size:
            raise ConfigError("internal: payload size mismatch")
        return bytes(out)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class ParsedWeightFile:
    """Validated in-memory view of a `.cudalm` file."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.cfg: Optional[ModelConfig] = None
        self.records: List[TensorRecord] = []
        self.payload_offset: int = 0
        self._parse()

    # -- parsing + full validation (mirror WeightFile::load) ----------------
    def _parse(self) -> None:
        raw = self.raw
        n = len(raw)
        if n < HEADER_BYTES:
            raise ConfigError("file smaller than header (72 bytes)")
        if raw[0:8] != MAGIC:
            raise ConfigError("bad magic")
        version, flags, n_tensors = struct.unpack_from("<III", raw, 8)
        if version != VERSION:
            raise ConfigError(f"unsupported version {version}")
        if flags != 0:
            raise ConfigError("flags must be 0")
        if not (1 <= n_tensors <= MAX_TENSORS):
            raise ConfigError("n_tensors out of range")
        self.cfg = ModelConfig.from_blob(raw[20:20 + CONFIG_BYTES])
        if not self.cfg.valid():
            raise ConfigError("config blob is invalid")
        table_off, payload_off = struct.unpack_from("<QQ", raw, 56)
        if table_off != HEADER_BYTES:
            raise ConfigError("table_offset must be 72")
        if not (table_off <= payload_off <= n):
            raise ConfigError("payload_offset out of range")
        if payload_off % PAYLOAD_ALIGN != 0:
            raise ConfigError("payload_offset not 16B aligned")
        self.payload_offset = payload_off

        seen = set()
        regions: List[Tuple[int, int]] = []
        pos = table_off
        for i in range(n_tensors):
            if pos + 4 > payload_off:
                raise ConfigError("table runs past payload")
            (name_len,) = struct.unpack_from("<I", raw, pos)
            pos += 4
            if not (1 <= name_len <= MAX_NAME_LEN):
                raise ConfigError("tensor name_len out of range")
            if pos + name_len > payload_off:
                raise ConfigError("table runs past payload")
            name = raw[pos:pos + name_len]
            if not all(0x21 <= b <= 0x7E for b in name):
                raise ConfigError("tensor name not printable ASCII")
            name = name.decode("ascii")
            pos += name_len
            if name in seen:
                raise ConfigError(f"duplicate tensor name: {name}")
            seen.add(name)
            if pos + 18 > payload_off:
                raise ConfigError("table runs past payload")
            dtype, ndim = struct.unpack_from("<BB", raw, pos)
            pos += 2
            if dtype not in _ELEMENT_BYTES:
                raise ConfigError(f"bad dtype in record for {name}")
            if ndim not in (1, 2):
                raise ConfigError(f"bad ndim in record for {name}")
            dims = list(struct.unpack_from(f"<{ndim}I", raw, pos))
            pos += 4 * ndim
            for d in dims:
                if not (1 <= d <= MAX_DIM):
                    raise ConfigError(f"bad dim value in record for {name}")
            if pos + 17 > payload_off:
                raise ConfigError("table runs past payload")
            offset, byte_size, align = struct.unpack_from("<QQB", raw, pos)
            pos += 17
            if any(raw[pos + r] != 0 for r in range(7)):
                raise ConfigError(f"reserved bytes nonzero in {name}")
            pos += 7

            rec = TensorRecord(name, dtype, dims, offset, byte_size, align)
            if byte_size != rec.expected_bytes:
                raise ConfigError(f"byte_size/shape mismatch for {name}")
            if offset + byte_size > n - payload_off:
                raise ConfigError(f"payload region out of bounds for {name}")
            region_addr = payload_off + offset
            if region_addr % PAYLOAD_ALIGN != 0:
                raise ConfigError(f"payload region not 16B aligned for {name}")
            if not (1 <= align <= 16) or (align & (align - 1)) != 0:
                raise ConfigError(f"bad align field for {name}")
            if region_addr % align != 0:
                raise ConfigError(f"align field violated for {name}")
            rec.data = bytes(raw[region_addr:region_addr + byte_size])
            regions.append((offset, byte_size))
            self.records.append(rec)

        if pos > payload_off:
            raise ConfigError("table runs past payload")
        regions.sort()
        for i in range(1, len(regions)):
            prev_end = regions[i - 1][0] + regions[i - 1][1]
            if regions[i][0] < prev_end:
                raise ConfigError("payload regions overlap")

    # -- convenience ---------------------------------------------------------
    def find(self, name: str) -> Optional[TensorRecord]:
        for r in self.records:
            if r.name == name:
                return r
        return None

    # -- decoder-block tensor set (mirror validate_block_tensors) -----------
    def validate_block_tensors(self) -> None:
        c = self.cfg
        if c is None:
            raise ConfigError("not parsed")
        if c.group_size != 128:
            raise ConfigError("block requires group_size == 128")
        H, hd, nkv, inter = c.hidden_size, c.head_dim, c.n_kv_heads, c.intermediate_size
        expects = [
            ("attn_norm.weight", DT_FP16, (H,)),
            ("ffn_norm.weight", DT_FP16, (H,)),
            ("attn.rope_cos", DT_FP16, (c.max_seq_len, hd // 2)),
            ("attn.rope_sin", DT_FP16, (c.max_seq_len, hd // 2)),
            ("attn.q_proj.weight", DT_INT4_PACKED, (H, H // 2)),
            ("attn.q_proj.scale", DT_FP16_SCALE, (H, H // 128)),
            ("attn.k_proj.weight", DT_INT4_PACKED, (nkv * hd, H // 2)),
            ("attn.k_proj.scale", DT_FP16_SCALE, (nkv * hd, H // 128)),
            ("attn.v_proj.weight", DT_INT4_PACKED, (nkv * hd, H // 2)),
            ("attn.v_proj.scale", DT_FP16_SCALE, (nkv * hd, H // 128)),
            ("attn.o_proj.weight", DT_INT4_PACKED, (H, H // 2)),
            ("attn.o_proj.scale", DT_FP16_SCALE, (H, H // 128)),
            ("mlp.gate_proj.weight", DT_INT4_PACKED, (inter, H // 2)),
            ("mlp.gate_proj.scale", DT_FP16_SCALE, (inter, H // 128)),
            ("mlp.up_proj.weight", DT_INT4_PACKED, (inter, H // 2)),
            ("mlp.up_proj.scale", DT_FP16_SCALE, (inter, H // 128)),
            ("mlp.down_proj.weight", DT_INT4_PACKED, (H, inter // 2)),
            ("mlp.down_proj.scale", DT_FP16_SCALE, (H, inter // 128)),
        ]
        for name, dtype, dims in expects:
            r = self.find(name)
            if r is None:
                raise ConfigError(f"missing tensor: {name}")
            if r.dtype != dtype:
                raise ConfigError(f"dtype mismatch for {name}")
            if tuple(r.dims) != tuple(dims):
                raise ConfigError(f"shape mismatch for {name}: {r.dims} != {list(dims)}")


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------
def read_file(path: str) -> ParsedWeightFile:
    with open(path, "rb") as f:
        return ParsedWeightFile(f.read())


def write_file(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def int4_pack_byte(lo: int, hi: int) -> int:
    """Two signed INT4 (two's complement, [-8, 7]) -> one byte.
    Low nibble = lo (k=2b), high nibble = hi (k=2b+1)."""
    return ((hi & 0xF) << 4) | (lo & 0xF)


def int4_unpack(b: int) -> Tuple[int, int]:
    """Inverse of int4_pack_byte (sign-extended)."""
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    if lo > 7:
        lo -= 16
    if hi > 7:
        hi -= 16
    return lo, hi


# Type alias kept for readability in tool code.
WeightMap = Dict[str, TensorRecord]
