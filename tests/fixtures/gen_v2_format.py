#!/usr/bin/env python3
"""CUDALM — fixture generator for test_cudalm_v2_format (stdlib only).

Writes into an output directory:
  valid.cudalm      a fully valid synthetic v2 file (small config, layer 0
                    DeltaNet + layer 3 Full-Attention tensor sets + norm)
  corrupt_*.cudalm  one file per corruption case (C++ reader must reject)
  sums.txt          per-tensor `name \\t byte_size \\t byte_sum` lines for the
                    valid file (C++ recomputes the sums over host payloads)

The synthetic config is a deliberately small 4-layer hybrid model (interval 4
-> layer 3 full attention) that satisfies every Qwen35Config validity rule,
so the container can carry a *valid* config while keeping fixtures tiny.
"""
from __future__ import annotations

import os
import struct
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tools", "common"))

import cudalm_v2 as v2  # noqa: E402


def prb(i: int) -> bytes:
    """Deterministic pseudo-random byte stream (mirrors the C++ tests' prb)."""
    x = (i + 1) * 0x9E3779B97F4A7C15
    x &= (1 << 64) - 1
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & ((1 << 64) - 1)
    x ^= x >> 33
    return bytes([x & 0xFF])


def prb_block(n: int, seed: int) -> bytes:
    out = bytearray()
    for i in range(n):
        out += prb(seed * 1000003 + i)
    return bytes(out)


def synth_config() -> v2.Qwen35Config:
    return v2.Qwen35Config(
        hidden_size=256, num_hidden_layers=4, intermediate_size=768,
        vocab_size=1000, n_heads=2, n_kv_heads=1, head_dim=64,
        lin_num_k_heads=2, lin_num_v_heads=2, lin_key_head_dim=32,
        lin_value_head_dim=64, lin_conv_kernel_dim=4,
        full_attention_interval=4, group_size=128, max_seq_len=512,
        eps=1e-6, rope_theta=1e7, partial_rotary_factor=0.25,
        mrope_section=(3, 3, 2),
    )


def build_valid_tensors(conf):
    H, G = conf.hidden_size, conf.group_size
    ts = []

    def add(name, dtype, dims, seed):
        numel = 1
        for d in dims:
            numel *= d
        ts.append(v2.V2Tensor(name, dtype, dims,
                              prb_block(numel * v2._ELEMENT_BYTES[dtype], seed)))

    # Layer 3: full attention.
    qo, kvo, oin = conf.q_proj_out, conf.kv_proj_out, conf.o_proj_in
    add("layers.3.input_layernorm.weight", v2.DT_BF16, (H,), 1)
    add("layers.3.self_attn.q_proj.weight", v2.DT_INT4_PACKED, (qo, H // 2), 2)
    add("layers.3.self_attn.q_proj.scale", v2.DT_FP16_SCALE, (qo, H // G), 3)
    add("layers.3.self_attn.k_proj.weight", v2.DT_INT4_PACKED, (kvo, H // 2), 4)
    add("layers.3.self_attn.k_proj.scale", v2.DT_FP16_SCALE, (kvo, H // G), 5)
    add("layers.3.self_attn.v_proj.weight", v2.DT_INT4_PACKED, (kvo, H // 2), 6)
    add("layers.3.self_attn.v_proj.scale", v2.DT_FP16_SCALE, (kvo, H // G), 7)
    add("layers.3.self_attn.o_proj.weight", v2.DT_INT4_PACKED, (H, oin // 2), 8)
    add("layers.3.self_attn.o_proj.scale", v2.DT_FP16_SCALE, (H, oin // G), 9)
    add("layers.3.self_attn.q_norm.weight", v2.DT_BF16, (conf.head_dim,), 10)
    add("layers.3.self_attn.k_norm.weight", v2.DT_BF16, (conf.head_dim,), 11)
    add("layers.3.post_attention_layernorm.weight", v2.DT_BF16, (H,), 12)
    add("layers.3.mlp.gate_proj.weight", v2.DT_INT4_PACKED, (conf.intermediate_size, H // 2), 13)
    add("layers.3.mlp.gate_proj.scale", v2.DT_FP16_SCALE, (conf.intermediate_size, H // G), 14)
    add("layers.3.mlp.up_proj.weight", v2.DT_INT4_PACKED, (conf.intermediate_size, H // 2), 15)
    add("layers.3.mlp.up_proj.scale", v2.DT_FP16_SCALE, (conf.intermediate_size, H // G), 16)
    add("layers.3.mlp.down_proj.weight", v2.DT_INT4_PACKED, (H, conf.intermediate_size // 2), 17)
    add("layers.3.mlp.down_proj.scale", v2.DT_FP16_SCALE, (H, conf.intermediate_size // G), 18)

    # Layer 0: Gated DeltaNet.
    ck, cv, cd = conf.linear_key_dim, conf.linear_value_dim, conf.linear_conv_dim
    add("layers.0.input_layernorm.weight", v2.DT_BF16, (H,), 19)
    add("layers.0.linear_attn.in_proj_qkv.weight", v2.DT_INT4_PACKED, (cd, H // 2), 20)
    add("layers.0.linear_attn.in_proj_qkv.scale", v2.DT_FP16_SCALE, (cd, H // G), 21)
    add("layers.0.linear_attn.in_proj_z.weight", v2.DT_INT4_PACKED, (cv, H // 2), 22)
    add("layers.0.linear_attn.in_proj_z.scale", v2.DT_FP16_SCALE, (cv, H // G), 23)
    add("layers.0.linear_attn.in_proj_b.weight", v2.DT_INT4_PACKED, (conf.lin_num_v_heads, H // 2), 24)
    add("layers.0.linear_attn.in_proj_b.scale", v2.DT_FP16_SCALE, (conf.lin_num_v_heads, H // G), 25)
    add("layers.0.linear_attn.in_proj_a.weight", v2.DT_INT4_PACKED, (conf.lin_num_v_heads, H // 2), 26)
    add("layers.0.linear_attn.in_proj_a.scale", v2.DT_FP16_SCALE, (conf.lin_num_v_heads, H // G), 27)
    add("layers.0.linear_attn.out_proj.weight", v2.DT_INT4_PACKED, (H, cv // 2), 28)
    add("layers.0.linear_attn.out_proj.scale", v2.DT_FP16_SCALE, (H, cv // G), 29)
    add("layers.0.linear_attn.conv1d.weight", v2.DT_BF16, (cd, 1, conf.lin_conv_kernel_dim), 30)
    add("layers.0.linear_attn.dt_bias", v2.DT_BF16, (conf.lin_num_v_heads,), 31)
    add("layers.0.linear_attn.A_log", v2.DT_FP32, (conf.lin_num_v_heads,), 32)
    add("layers.0.linear_attn.norm.weight", v2.DT_FP32, (conf.lin_value_head_dim,), 33)
    add("layers.0.post_attention_layernorm.weight", v2.DT_BF16, (H,), 34)
    add("layers.0.mlp.gate_proj.weight", v2.DT_INT4_PACKED, (conf.intermediate_size, H // 2), 35)
    add("layers.0.mlp.gate_proj.scale", v2.DT_FP16_SCALE, (conf.intermediate_size, H // G), 36)
    add("layers.0.mlp.up_proj.weight", v2.DT_INT4_PACKED, (conf.intermediate_size, H // 2), 37)
    add("layers.0.mlp.up_proj.scale", v2.DT_FP16_SCALE, (conf.intermediate_size, H // G), 38)
    add("layers.0.mlp.down_proj.weight", v2.DT_INT4_PACKED, (H, conf.intermediate_size // 2), 39)
    add("layers.0.mlp.down_proj.scale", v2.DT_FP16_SCALE, (H, conf.intermediate_size // G), 40)

    add("norm.weight", v2.DT_BF16, (H,), 41)
    return ts


def build_file(conf, tensors, metadata):
    obj = v2.V2File(conf, metadata, tensors)
    return obj.to_bytes()


def patch_u64(raw: bytearray, off: int, val: int) -> None:
    raw[off:off + 8] = struct.pack("<Q", val)


def patch_u32(raw: bytearray, off: int, val: int) -> None:
    raw[off:off + 4] = struct.pack("<I", val)


def entry_positions(raw: bytes):
    """Walk the table of a v2 file; yield (name, after_name, entry_off,
    dtype_off, pad_off, offset_field, size_field)."""
    (tbl_off, tbl_size) = struct.unpack_from("<QQ", raw, 56)
    pos = tbl_off
    end = tbl_off + tbl_size
    while pos < end:
        nlen = raw[pos]
        name = raw[pos + 1:pos + 1 + nlen].decode("utf-8")
        f = pos + 1 + nlen
        # after name: dtype u8 @f, ndim u8 @f+1, pad u16 @f+2,
        # dims i64[8] @f+4..f+68, offset u64 @f+68, byte_size u64 @f+76,
        # align u8 @f+84, pad u8 @f+85; entry ends at f+86.
        yield name, f, pos, f, f + 2, f + 68, f + 76
        pos = f + 86


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    conf = synth_config()
    assert conf.valid(), "synthetic config must satisfy Qwen35Config.valid()"
    tensors = build_valid_tensors(conf)
    meta = {
        "arch": "synthetic-qwen3.5-format-test",
        "model_repo": "cudalm/synthetic",
        "model_revision": "0000",
    }
    valid = build_file(conf, tensors, meta)
    with open(os.path.join(out_dir, "valid.cudalm"), "wb") as f:
        f.write(valid)

    # sums.txt for the C++ side (u64 wrapped byte sums).
    with open(os.path.join(out_dir, "sums.txt"), "w") as f:
        for t in tensors:
            f.write(f"{t.name}\t{len(t.data)}\t{sum(t.data) & 0xFFFFFFFFFFFFFFFF}\n")

    def out(case, raw):
        with open(os.path.join(out_dir, f"corrupt_{case}.cudalm"), "wb") as f:
            f.write(raw)

    r = bytearray(valid)
    r[0] = ord("X")
    out("bad_magic", bytes(r))

    r = bytearray(valid)
    patch_u32(r, 8, 3)
    out("bad_version", bytes(r))

    r = bytearray(valid)
    patch_u32(r, 12, 1)
    out("bad_flags", bytes(r))

    r = bytearray(valid)
    patch_u64(r, 32, 87)  # config_size
    out("bad_config_size", bytes(r))

    r = bytearray(valid)
    patch_u64(r, 40, 88)  # meta_offset := config_offset -> overlap
    out("section_overlap", bytes(r))

    r = bytearray(valid)
    poff = struct.unpack_from("<Q", valid, 72)[0]
    patch_u64(r, 72, poff - 4)  # payload_offset not 16B-aligned
    out("payload_unaligned", bytes(r))

    r = bytearray(valid)
    (psz,) = struct.unpack_from("<Q", valid, 80)
    # first entry's offset -> payload_size - 4 (byte_size unchanged -> OOB)
    name, f, e, dtype_off, pad_off, off_off, size_off = next(entry_positions(valid))
    patch_u64(r, off_off, psz - 4)
    out("payload_oob", bytes(r))

    r = bytearray(valid)
    r[dtype_off] = 9
    out("bad_dtype", bytes(r))

    r = bytearray(valid)
    struct.pack_into("<H", r, pad_off, 1)  # entry pad u16 != 0 (at f+2, not f+1)
    out("entry_pad_nonzero", bytes(r))

    r = bytearray(valid)
    (tbl_off, tbl_size) = struct.unpack_from("<QQ", valid, 56)
    patch_u64(r, 64, tbl_size + 4)  # table_size extends past the table
    out("table_trailing", bytes(r))

    # Reconstructed files (writer-level corruptions the byte-patcher can't do).
    # V2File.__init__ rejects duplicate names, so bypass it for the fixture.
    dup = v2.V2File.__new__(v2.V2File)
    dup.config = conf
    dup.metadata = dict(meta)
    dup.tensors = [tensors[0], tensors[0]]
    out("dup_name", dup.to_bytes())

    badname = v2.V2Tensor("bad name with spaces", v2.DT_BF16, (4,), b"\x00" * 8)
    out("bad_name_charset",
        build_file(conf, [tensors[0], badname], meta))

    badconf = v2.Qwen35Config(
        hidden_size=256, num_hidden_layers=4, intermediate_size=768,
        vocab_size=1000, n_heads=2, n_kv_heads=3, head_dim=64,
        lin_num_k_heads=2, lin_num_v_heads=2, lin_key_head_dim=32,
        lin_value_head_dim=64, lin_conv_kernel_dim=4,
        full_attention_interval=4, group_size=128, max_seq_len=512,
        eps=1e-6, rope_theta=1e7, partial_rotary_factor=0.25,
        mrope_section=(3, 3, 2))
    assert not badconf.valid()
    out("invalid_config", build_file(badconf, tensors, meta))

    print(f"gen_v2_format: wrote valid + 13 corruption cases to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
