#!/usr/bin/env python3
"""CUDALM v0.1 — offline weight converter (decoder block).

Generates a deterministic, seeded decoder-block weight file (`.cudalm`) for
the CUDALM runtime. Python/torch is OFFLINE-ONLY tooling; the runtime never
sees PyTorch.

What it produces (18 tensors, see docs/weight_format.md):
  * fp16:  attn_norm.weight [H], ffn_norm.weight [H],
           attn.rope_cos / attn.rope_sin [max_seq_len, head_dim/2]
           (interleaved-pair RoPE tables, fp32-computed, fp16-stored)
  * W4A16: q/k/v/o/gate/up/down projections, each as
           INT4_PACKED weight [N, K/2] + FP16_SCALE [N, K/128]

INT4 quantization contract — carried verbatim from CUDALab cb6a6a9
(cudalab/int4gemv_quantize.py; see docs/provenance.md):
  * symmetric group-wise, G = 128, zero_point = 0, q in [-7, 7]
  * scale[n,g] = amax(group)/7 computed in FP32, STORED AS FP16
  * q = clamp(round(W/scale), -7, 7), torch.round = round-half-to-even
  * zero group -> scale = 0, q == 0 (no division by zero)
  * K % 128 == 0 required
  * nibble pack: low nibble of byte b = k=2b, high nibble = k=2b+1

Determinism: torch.manual_seed(seed) with a fixed draw order; the same
(seed, config) always yields the same file bytes (verified by --selftest).

Usage:
  python3 tools/convert_weights.py --out data/block_v01.cudalm [--seed 20250922]
  python3 tools/convert_weights.py --selftest        # no torch GPU needed
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))

import binfmt  # noqa: E402  (local module, after sys.path fix)
import torch  # noqa: E402  (offline tooling only; never linked into the runtime)
import warnings

# numpy is not a dependency of this tool; torch warns if it can't find one.
warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")

INT4_MAX = 7
GROUP_SIZE = 128

# Names in the fixed block-tensor order (matches BlockWeights::load).
NORM_NAMES = ("attn_norm.weight", "ffn_norm.weight")
ROPE_NAMES = ("attn.rope_cos", "attn.rope_sin")
PROJ_NAMES = (
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def _tensor_to_bytes(t) -> bytes:
    """Contiguous torch tensor -> raw little-endian bytes (no numpy)."""
    u8 = t.contiguous().view(torch.uint8)
    return bytes(u8.flatten().tolist())


def _bytes_to_tensor(b: bytes, dtype, shape) -> "torch.Tensor":
    """Raw bytes -> torch tensor (no numpy). `b` is copied, not owned."""
    t = torch.frombuffer(bytearray(b), dtype=torch.uint8)
    return t.view(dtype).view(*shape)


# ---------------------------------------------------------------------------
# INT4 quantization (CUDALab contract; see module docstring)
# ---------------------------------------------------------------------------
def quantize_w(W):
    """W: fp16 (N, K) contiguous, K % 128 == 0, finite.
    Returns (W_packed uint8 (N, K/2), scale fp16 (N, K/128))."""
    if W.dim() != 2:
        raise ValueError(f"expected 2-D W, got {W.dim()}D")
    if W.dtype != torch.float16:
        raise ValueError(f"quantizer accepts fp16 only, got {W.dtype}")
    if not W.is_contiguous():
        raise ValueError("quantizer requires contiguous W")
    N, K = W.shape
    if K % GROUP_SIZE != 0:
        raise ValueError(f"K must be a multiple of {GROUP_SIZE}, got {K}")
    if not torch.isfinite(W.float()).all():
        raise ValueError("quantizer requires finite W")

    G = K // GROUP_SIZE
    Wg = W.float().view(N, G, GROUP_SIZE)
    amax = Wg.abs().amax(dim=2)                 # (N, G) fp32
    scale32 = amax / INT4_MAX
    zero = scale32 == 0
    safe = torch.where(zero, torch.ones_like(scale32), scale32)
    q = torch.round(Wg / safe.unsqueeze(2))     # round-half-to-even
    q = q.clamp_(-INT4_MAX, INT4_MAX)
    if bool(zero.any()):
        q = torch.where(zero.unsqueeze(2), torch.zeros_like(q), q)
    q = q.view(N, K).to(torch.int32)

    q2 = q.view(N, K // 2, 2)
    lo = q2[..., 0] & 15
    hi = q2[..., 1] & 15
    W_packed = (((hi << 4) | lo).to(torch.uint8).view(N, K // 2).contiguous())
    scale16 = scale32.to(torch.float16).view(N, G).contiguous()
    return W_packed, scale16, q, scale32


def unpack_w(W_packed):
    """uint8 (N, K/2) -> int32 (N, K), sign-extended (CUDALab reference)."""
    N, H = W_packed.shape
    lo = (W_packed & 15).to(torch.int32)
    hi = ((W_packed >> 4) & 15).to(torch.int32)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack((lo, hi), dim=-1).view(N, 2 * H).contiguous()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def projection_shapes(cfg):
    """name -> (N, K) for the seven W4A16 projections (in block order)."""
    H, hd, nkv, inter = cfg.hidden_size, cfg.head_dim, cfg.n_kv_heads, cfg.intermediate_size
    return {
        "attn.q_proj": (H, H),
        "attn.k_proj": (nkv * hd, H),
        "attn.v_proj": (nkv * hd, H),
        "attn.o_proj": (H, H),
        "mlp.gate_proj": (inter, H),
        "mlp.up_proj": (inter, H),
        "mlp.down_proj": (H, inter),
    }


def build_block_tensors(cfg, seed):
    """Draw all block tensors in a fixed order. Returns (records, stats).
    `records` is a list of (name, dtype, dims, bytes)."""
    torch.manual_seed(seed)
    shapes = projection_shapes(cfg)
    records = []
    stats = {}

    # 1) RMSNorm weights: values near 1.0 (deterministic draw order).
    for name in NORM_NAMES:
        w = (1.0 + 0.02 * torch.randn(cfg.hidden_size)).to(torch.float16)
        records.append((name, binfmt.DT_FP16, (cfg.hidden_size,),
                        _tensor_to_bytes(w)))

    # 2) RoPE tables (interleaved-pair convention): freq_i = theta^(-2i/hd),
    #    angle(p, i) = p * freq_i; tables [max_seq_len, hd/2] fp16.
    #    Computed in fp64 -> fp32 -> fp16 to pin the values exactly.
    hd, msl, theta = cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    i = torch.arange(hd // 2, dtype=torch.float64)
    freq = torch.exp(-2.0 * i / hd * torch.tensor(float(theta)).log())
    pos = torch.arange(msl, dtype=torch.float64)
    ang = pos.unsqueeze(1) * freq.unsqueeze(0)  # (msl, hd/2)
    for name, fn in ((ROPE_NAMES[0], torch.cos), (ROPE_NAMES[1], torch.sin)):
        t = fn(ang).to(torch.float32).to(torch.float16)
        records.append((name, binfmt.DT_FP16, (msl, hd // 2),
                        _tensor_to_bytes(t)))

    # 3) W4A16 projections. Fixed draw order = PROJ_NAMES order.
    #    One group of q_proj row 0 is intentionally zeroed to pin the
    #    zero-group contract (scale=0, q==0) end-to-end.
    for name in PROJ_NAMES:
        N, K = shapes[name]
        W = (0.02 * torch.randn(N, K)).to(torch.float16)
        if name == "attn.q_proj":
            W[0, 0:GROUP_SIZE] = 0
        packed, scale16, q, scale32 = quantize_w(W)
        records.append((name + ".weight", binfmt.DT_INT4_PACKED, (N, K // 2),
                        _tensor_to_bytes(packed)))
        records.append((name + ".scale", binfmt.DT_FP16_SCALE, (N, K // 128),
                        _tensor_to_bytes(scale16)))
        s = scale32
        qf = q.float()
        srep = s.repeat_interleave(GROUP_SIZE, dim=1)
        step_err = ((W.float() - qf * srep).abs() / srep.clamp_min(1e-30)).max()
        stats[name] = {
            "N": N, "K": K,
            "max_abs_q": int(q.abs().max().item()),
            "n_zero_groups": int((scale16 == 0).sum().item()),
            "max_dequant_step_err": float(step_err.item()),
            "max_scale": float(s.max().item()),
        }
    return records, stats


def build_file(cfg, seed) -> bytes:
    records, stats = build_block_tensors(cfg, seed)
    w = binfmt.WeightFileWriter(cfg)
    for name, dtype, dims, blob in records:
        w.add(name, dtype, dims, blob)
    return w.build(), stats


# ---------------------------------------------------------------------------
# Selftest (determinism + contract invariants + file round-trip)
# ---------------------------------------------------------------------------
def _check(cond, msg):
    if not cond:
        print(f"[SELFTEST FAIL] {msg}")
        sys.exit(1)


def selftest() -> int:
    import tempfile

    cfg = binfmt.ModelConfig.v01_default()

    # -- quantization contract on a small tensor ----------------------------
    torch.manual_seed(1234)
    W = (0.5 * torch.randn(4, 256)).to(torch.float16)
    W[0, 0:128] = 0  # zero group
    packed, scale16, q, scale32 = quantize_w(W)
    qback = unpack_w(packed)
    _check(bool((qback == q).all()), "pack/unpack round-trip")
    _check(int(q.abs().max().item()) <= INT4_MAX, "q in [-7,7]")
    # zero group: scale == 0 and q == 0
    _check(float(scale16[0, 0].item()) == 0.0, "zero group scale == 0")
    _check(bool((q[0, 0:128] == 0).all()), "zero group q == 0")
    # nonzero group: the max-|.| element must quantize to exactly +-7
    qg = q.view(4, 2, 128)
    sg = scale32
    for n in range(4):
        for g in range(2):
            if sg[n, g] > 0:
                _check(int(qg[n, g].abs().max().item()) == INT4_MAX,
                       f"group ({n},{g}): max|q| must be 7")
    # dequant fidelity bound: |W - q*s| / s <= 0.5 + 7*2^-11 (+margin)
    srep = sg.repeat_interleave(GROUP_SIZE, dim=1)
    step = ((W.float() - q.float() * srep).abs() / srep.clamp_min(1e-30)).max()
    _check(float(step.item()) <= 0.5 + 7 * 2 ** -11 + 1e-3,
           f"dequant step err {float(step.item()):.5f} above bound")

    # -- int4 pack/unpack full domain (mirrors the C++ CPU test) ------------
    for lo in range(-8, 8):
        for hi in range(-8, 8):
            b = binfmt.int4_pack_byte(lo, hi)
            l, h = binfmt.int4_unpack(b)
            _check(l == lo and h == hi, f"pack/unpack domain ({lo},{hi})")

    # -- full file: build twice, byte-identical, round-trip through reader --
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "selftest.cudalm")
        blob1, stats = build_file(cfg, 20250922)
        blob2, _ = build_file(cfg, 20250922)
        _check(blob1 == blob2, "determinism: same seed -> same bytes")
        binfmt.write_file(path, blob1)
        pf = binfmt.read_file(path)
        _check(pf.cfg == cfg, "config round-trip")
        pf.validate_block_tensors()
        # zero-group invariant survived the file round-trip (q_proj row 0,
        # group 0 was deliberately zeroed by the generator)
        qw = pf.find("attn.q_proj.weight")
        sc = pf.find("attn.q_proj.scale")
        qb = unpack_w(_bytes_to_tensor(qw.data, torch.uint8, qw.dims))
        sc16 = _bytes_to_tensor(sc.data, torch.float16, sc.dims)
        _check(float(sc16[0, 0].item()) == 0.0, "zero group scale==0 in file")
        _check(bool((qb[0, 0:128] == 0).all()), "zero group q==0 in file")
        # rope invariants: row 0 exact; cos^2+sin^2 ~ 1
        cos = _bytes_to_tensor(pf.find("attn.rope_cos").data,
                               torch.float16, pf.find("attn.rope_cos").dims)
        sin = _bytes_to_tensor(pf.find("attn.rope_sin").data,
                               torch.float16, pf.find("attn.rope_sin").dims)
        _check(bool((cos[0] == 1.0).all()), "rope cos row0 == 1")
        _check(bool((sin[0] == 0.0).all()), "rope sin row0 == 0")
        resid = (cos.float() ** 2 + sin.float() ** 2 - 1.0).abs().max()
        _check(float(resid.item()) <= 1e-2, f"rope cos^2+sin^2 resid {resid}")
        # every group in the file: scale>0 -> max|q| == 7; scale==0 -> q == 0
        for pname in PROJ_NAMES:
            wrec = pf.find(pname + ".weight")
            srec = pf.find(pname + ".scale")
            N, Hh = wrec.dims
            K = 2 * Hh
            qv = unpack_w(_bytes_to_tensor(wrec.data, torch.uint8, wrec.dims))
            sv = _bytes_to_tensor(srec.data, torch.float16, srec.dims).float()
            # Implications over all groups: scale>0 -> max|q|==7;
            # scale==0 -> q==0.  (Written as implications, not conjunctions!)
            qmax = qv.view(N, K // 128, 128).abs().amax(dim=2)
            nz = sv > 0
            _check(bool((~nz | (qmax == 7)).all()),
                   f"file {pname}: nonzero-scale group without max|q|==7")
            _check(bool((nz | (qmax == 0)).all()),
                   f"file {pname}: zero-scale group with nonzero q")

    for name, st in stats.items():
        print(f"  {name:<16s} N={st['N']:>5d} K={st['K']:>5d} "
              f"max|q|={st['max_abs_q']} zero_groups={st['n_zero_groups']} "
              f"max_step_err={st['max_dequant_step_err']:.4f} "
              f"max_scale={st['max_scale']:.5f}")
    print(f"[SELFTEST PASS] file {len(blob1)} bytes, {len(cfg.__slots__)} cfg fields")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/block_v01.cudalm")
    ap.add_argument("--seed", type=int, default=20250922)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    cfg = binfmt.ModelConfig.v01_default()
    blob, stats = build_file(cfg, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    binfmt.write_file(a.out, blob)
    print(f"wrote {a.out} ({len(blob)} bytes, seed={a.seed})")
    for name, st in stats.items():
        print(f"  {name:<16s} N={st['N']:>5d} K={st['K']:>5d} "
              f"max|q|={st['max_abs_q']} zero_groups={st['n_zero_groups']} "
              f"max_step_err={st['max_dequant_step_err']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
