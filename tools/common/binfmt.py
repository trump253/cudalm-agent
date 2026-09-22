"""CUDALM binary containers (v1) — Python reader/writer.

Single source of truth: docs/weight_format.md. This module and
`include/cudalm/weight_format.h` + `src/runtime/weight_loader.cpp` (weights)
and `include/cudalm/golden_loader.h` + `src/runtime/golden_loader.cpp`
(golden) implement the same layout and are cross-checked by tests (Python
selftests + the C++ `test_weights_crosslang` / `test_golden_file`).

Stdlib only (no torch, no numpy) so it can be imported from any tool.

Two containers share the record/payload layout; only the header differs:

  Weight  `.cudalm`  magic CUDLMW01, header 72 B
    magic(8) | version u32 | flags u32 | n_tensors u32
    | config blob (36 B) | table_offset u64 (=72) | payload_offset u64

  Golden  `.cudalm`  magic CUDLMG01, header 80 B (adds decode state)
    magic(8) | version u32 | flags u32 | n_tensors u32
    | config blob (36 B) | position i32 | reserved i32 (=0)
    | table_offset u64 (=80) | payload_offset u64

  Table   n_tensors x TensorRecord (packed, identical in both)
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
GOLDEN_MAGIC = b"CUDLMG01"
VERSION = 1
FLAGS_RESERVED = 0
CONFIG_BYTES = 36
# Weight header: magic(8)|version(4)|flags(4)|n_tensors(4)|config(36)
#                |table_offset(8)|payload_offset(8) = 72 B
HEADER_BYTES = 72
# Golden header adds |position i32|reserved i32| after the config blob,
# pushing the offsets to 64/72 and the header size to 80 B.
GOLDEN_EXTRA_BYTES = 8
GOLDEN_HEADER_BYTES = HEADER_BYTES + GOLDEN_EXTRA_BYTES  # 80
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
    def q_proj_out(self) -> int:
        return self.n_heads * self.head_dim

    def kv_proj_out(self) -> int:
        return self.n_kv_heads * self.head_dim

    def valid(self) -> bool:
        if self.hidden_size <= 0 or self.n_heads <= 0 or self.n_kv_heads <= 0:
            return False
        if self.head_dim <= 0 or self.intermediate_size <= 0:
            return False
        if self.max_seq_len <= 0 or self.group_size <= 0:
            return False
        if self.n_heads % self.n_kv_heads != 0:
            return False
        if self.head_dim % 2 != 0:
            return False
        # W4A16 GEMV: every GEMV K must be a multiple of group_size (128).
        # K values: H (q/k/v/gate/up), q_proj_out (o_proj), inter (down).
        # NOTE: hidden_size == n_heads * head_dim is NOT required (v0.1.1).
        if self.hidden_size % self.group_size != 0:
            return False
        if self.q_proj_out() % self.group_size != 0:
            return False
        if self.intermediate_size % self.group_size != 0:
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

    @classmethod
    def v011_general_test(cls) -> "ModelConfig":
        # v0.1.1 generalized shape test: q_proj_out = 16*128 = 2048 !=
        # H = 1024 — proves the plumbing no longer assumes Q width == H.
        return ModelConfig(
            hidden_size=1024, n_heads=16, n_kv_heads=8, head_dim=128,
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
class _ContainerWriter:
    """Shared deterministic container writer (weight + golden containers).

    Tensors are placed in table order; each region start is 16B-aligned
    relative to the payload start; the payload base is 16B-aligned (gap
    zero-filled). The header is `magic | version u32 | flags u32 |
    n_tensors u32 | config blob (36 B) | middle | table_offset u64 |
    payload_offset u64`; `middle` is empty for weight files and holds the
    golden `position i32 | reserved i32` pair for golden files."""

    def __init__(self, cfg: ModelConfig, magic: bytes, middle: bytes) -> None:
        if not cfg.valid():
            raise ConfigError("invalid ModelConfig")
        self.cfg = cfg
        self.magic = bytes(magic)
        if len(self.magic) != 8:
            raise ConfigError("magic must be exactly 8 bytes")
        self.middle = bytes(middle)
        self.header_bytes = 56 + len(self.middle) + 16
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
        payload_off = self.header_bytes + len(table)
        while payload_off % PAYLOAD_ALIGN != 0:
            payload_off += 1

        out = bytearray()
        out += self.magic
        out += struct.pack("<III", VERSION, FLAGS_RESERVED, len(self.records))
        out += self.cfg.to_blob()
        out += self.middle
        out += struct.pack("<QQ", self.header_bytes, payload_off)
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


class WeightFileWriter(_ContainerWriter):
    """Writer for `.cudalm` weight files (magic CUDLMW01, 72 B header)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, MAGIC, b"")


class GoldenFileWriter(_ContainerWriter):
    """Writer for golden-reference files (magic CUDLMG01, 80 B header).

    The header carries the decode `position` (i32, >= 0) plus one reserved
    i32 that must stay zero."""

    def __init__(self, cfg: ModelConfig, position: int) -> None:
        position = int(position)
        if position < 0:
            raise ConfigError("position must be >= 0")
        super().__init__(cfg, GOLDEN_MAGIC,
                         struct.pack("<ii", position, 0))
        self.position = position


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def _parse_container(raw: bytes, magic: bytes, middle_size: int, label: str):
    """Parse + fully validate a container (weight or golden).

    Mirrors WeightFile::load / GoldenFile::load in the C++ runtime.
    Returns (cfg, middle_bytes, records, payload_offset)."""
    n = len(raw)
    header_bytes = 56 + middle_size + 16
    if n < header_bytes:
        raise ConfigError(f"{label}: file smaller than header ({header_bytes} bytes)")
    if raw[0:8] != magic:
        raise ConfigError(f"{label}: bad magic")
    version, flags, n_tensors = struct.unpack_from("<III", raw, 8)
    if version != VERSION:
        raise ConfigError(f"{label}: unsupported version {version}")
    if flags != 0:
        raise ConfigError(f"{label}: flags must be 0")
    if not (1 <= n_tensors <= MAX_TENSORS):
        raise ConfigError(f"{label}: n_tensors out of range")
    cfg = ModelConfig.from_blob(raw[20:20 + CONFIG_BYTES])
    if not cfg.valid():
        raise ConfigError(f"{label}: config blob is invalid")
    middle = raw[56:56 + middle_size]
    table_off, payload_off = struct.unpack_from("<QQ", raw, 56 + middle_size)
    if table_off != header_bytes:
        raise ConfigError(f"{label}: table_offset must be {header_bytes}")
    if not (table_off <= payload_off <= n):
        raise ConfigError(f"{label}: payload_offset out of range")
    if payload_off % PAYLOAD_ALIGN != 0:
        raise ConfigError(f"{label}: payload_offset not 16B aligned")

    seen = set()
    regions: List[Tuple[int, int]] = []
    records: List[TensorRecord] = []
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
        records.append(rec)

    if pos > payload_off:
        raise ConfigError("table runs past payload")
    regions.sort()
    for i in range(1, len(regions)):
        prev_end = regions[i - 1][0] + regions[i - 1][1]
        if regions[i][0] < prev_end:
            raise ConfigError("payload regions overlap")

    return cfg, bytes(middle), records, payload_off


# ---------------------------------------------------------------------------
# Expected tensor sets (single source of truth; mirrored in the C++ runtime)
# ---------------------------------------------------------------------------
def block_tensor_expectations(cfg: ModelConfig):
    """The 18 decoder-block weight tensors: (name, dtype, dims) tuples.

    Shape contract (docs/weight_format.md): q_proj (q_proj_out, H),
    k/v_proj (kv_proj_out, H), o_proj (H, q_proj_out) — Q width is
    independent of hidden_size (v0.1.1)."""
    H, hd, inter = cfg.hidden_size, cfg.head_dim, cfg.intermediate_size
    qo, kv = cfg.q_proj_out(), cfg.kv_proj_out()
    return [
        ("attn_norm.weight", DT_FP16, (H,)),
        ("ffn_norm.weight", DT_FP16, (H,)),
        ("attn.rope_cos", DT_FP16, (cfg.max_seq_len, hd // 2)),
        ("attn.rope_sin", DT_FP16, (cfg.max_seq_len, hd // 2)),
        ("attn.q_proj.weight", DT_INT4_PACKED, (qo, H // 2)),
        ("attn.q_proj.scale", DT_FP16_SCALE, (qo, H // 128)),
        ("attn.k_proj.weight", DT_INT4_PACKED, (kv, H // 2)),
        ("attn.k_proj.scale", DT_FP16_SCALE, (kv, H // 128)),
        ("attn.v_proj.weight", DT_INT4_PACKED, (kv, H // 2)),
        ("attn.v_proj.scale", DT_FP16_SCALE, (kv, H // 128)),
        ("attn.o_proj.weight", DT_INT4_PACKED, (H, qo // 2)),
        ("attn.o_proj.scale", DT_FP16_SCALE, (H, qo // 128)),
        ("mlp.gate_proj.weight", DT_INT4_PACKED, (inter, H // 2)),
        ("mlp.gate_proj.scale", DT_FP16_SCALE, (inter, H // 128)),
        ("mlp.up_proj.weight", DT_INT4_PACKED, (inter, H // 2)),
        ("mlp.up_proj.scale", DT_FP16_SCALE, (inter, H // 128)),
        ("mlp.down_proj.weight", DT_INT4_PACKED, (H, inter // 2)),
        ("mlp.down_proj.scale", DT_FP16_SCALE, (H, inter // 128)),
    ]


def golden_tensor_expectations(cfg: ModelConfig, position: int):
    """The 18 golden tensors: 16 stage outputs + the full KV state after the
    write at `position`. Stages are flat batch-1 rows [1, X] in the exact
    layout the runtime produces (see docs/weight_format.md)."""
    H, hd, inter = cfg.hidden_size, cfg.head_dim, cfg.intermediate_size
    qo, kv = cfg.q_proj_out(), cfg.kv_proj_out()
    kv_rows = cfg.n_kv_heads * (position + 1)
    return [
        ("stage.input", DT_FP16, (1, H)),
        ("stage.rmsnorm1", DT_FP16, (1, H)),
        ("stage.q", DT_FP16, (1, qo)),
        ("stage.k", DT_FP16, (1, kv)),
        ("stage.v", DT_FP16, (1, kv)),
        ("stage.rope_q", DT_FP16, (1, qo)),
        ("stage.rope_k", DT_FP16, (1, kv)),
        ("stage.attention_output", DT_FP16, (1, qo)),
        ("stage.output_projection", DT_FP16, (1, H)),
        ("stage.residual1", DT_FP16, (1, H)),
        ("stage.rmsnorm2", DT_FP16, (1, H)),
        ("stage.gate", DT_FP16, (1, inter)),
        ("stage.up", DT_FP16, (1, inter)),
        ("stage.silu_gate_mul_up", DT_FP16, (1, inter)),
        ("stage.down", DT_FP16, (1, H)),
        ("stage.final_output", DT_FP16, (1, H)),
        ("kv.k_state", DT_FP16, (kv_rows, hd)),
        ("kv.v_state", DT_FP16, (kv_rows, hd)),
    ]


def _check_expectations(find, expectations, label: str) -> None:
    for name, dtype, dims in expectations:
        r = find(name)
        if r is None:
            raise ConfigError(f"{label}: missing tensor: {name}")
        if r.dtype != dtype:
            raise ConfigError(f"{label}: dtype mismatch for {name}")
        if tuple(r.dims) != tuple(dims):
            raise ConfigError(
                f"{label}: shape mismatch for {name}: {r.dims} != {list(dims)}")


def check_block_tensors(cfg: ModelConfig, find) -> None:
    """Validate the 18 block weight tensors. `find`: name -> record|None."""
    if cfg is None:
        raise ConfigError("not parsed")
    if cfg.group_size != 128:
        raise ConfigError("block requires group_size == 128")
    _check_expectations(find, block_tensor_expectations(cfg), "block")


def check_golden_tensors(cfg: ModelConfig, find, position: int) -> None:
    """Validate weights + all golden tensors for the given position."""
    check_block_tensors(cfg, find)
    _check_expectations(find, golden_tensor_expectations(cfg, position),
                        "golden")


class ParsedWeightFile:
    """Validated in-memory view of a `.cudalm` weight file."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.cfg: Optional[ModelConfig] = None
        self.records: List[TensorRecord] = []
        self.payload_offset: int = 0
        cfg, _middle, records, payload_off = _parse_container(
            raw, MAGIC, 0, "weight")
        self.cfg = cfg
        self.records = records
        self.payload_offset = payload_off

    def find(self, name: str) -> Optional[TensorRecord]:
        for r in self.records:
            if r.name == name:
                return r
        return None

    def validate_block_tensors(self) -> None:
        check_block_tensors(self.cfg, self.find)


class ParsedGoldenFile:
    """Validated in-memory view of a CUDLMG01 golden-reference file.

    Holds config, decode `position`, and every embedded tensor (weights,
    stage outputs, KV state)."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.cfg: Optional[ModelConfig] = None
        self.records: List[TensorRecord] = []
        self.payload_offset: int = 0
        self.position = 0
        cfg, middle, records, payload_off = _parse_container(
            raw, GOLDEN_MAGIC, GOLDEN_EXTRA_BYTES, "golden")
        position, reserved = struct.unpack("<ii", middle)
        if reserved != 0:
            raise ConfigError("golden: reserved header field must be 0")
        if position < 0:
            raise ConfigError("golden: position must be >= 0")
        if position >= cfg.max_seq_len:
            raise ConfigError("golden: position must be < max_seq_len")
        self.cfg = cfg
        self.records = records
        self.payload_offset = payload_off
        self.position = position

    def find(self, name: str) -> Optional[TensorRecord]:
        for r in self.records:
            if r.name == name:
                return r
        return None

    def validate_block_tensors(self) -> None:
        check_block_tensors(self.cfg, self.find)

    def validate_golden_tensors(self) -> None:
        check_golden_tensors(self.cfg, self.find, self.position)


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------
def read_file(path: str) -> ParsedWeightFile:
    with open(path, "rb") as f:
        return ParsedWeightFile(f.read())


def read_golden_file(path: str) -> ParsedGoldenFile:
    with open(path, "rb") as f:
        return ParsedGoldenFile(f.read())


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
