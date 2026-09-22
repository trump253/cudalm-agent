#!/usr/bin/env python3
"""CUDALM v0.1 — golden reference generator (one decoder block, one decode
position). PyTorch on CPU; offline tooling only, never linked into the
runtime.

Math contract (what the C++ kernels must mirror):
  * all arithmetic in FP32
  * FP16 cast at every stage boundary (the stored stage tensors)
  * W4A16 linear: y = (q * scale_fp16_as_fp32) @ x_fp32  -> fp16
    (the STORED fp16 scale is the contract, not the fp32 amax/7)
  * RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w -> fp16
  * RoPE: interleaved pairs (CUDALab rope_v3_half2 convention):
      y[2i]   = a*cos_i - b*sin_i
      y[2i+1] = a*sin_i + b*cos_i
    cos/sin read from the fp16 tables at row `position`
  * attention (decode, causal history 0..p):
      s_h   = (rope_q_h . K_hkv[0..p]) / sqrt(head_dim)
      w_h   = exp(s_h - max) / sum          (fp32, stable)
      out_h = w_h @ V_hkv[0..p]
    GQA: query head h reads kv head h // (n_heads / n_kv_heads)
  * silu_mul: silu(gate) * up in fp32 -> fp16
  * residual: fp32 add of two fp16 rows -> fp16

KV history for position p > 0: positions 0..p-1 are filled by running
seeded random hidden states (0.05*randn, fp16) through the same
attn_norm -> k/v projections -> RoPE chain, so the stored KV state is
non-trivial. The file embeds the FULL cache after the write at p.

Output container: CUDLMG01 v1 (see docs/weight_format.md) — the 18 block
weight tensors (same generator and seed as convert_weights.py), config,
decode position, all 16 stage tensors, and the two KV-state tensors
[kv_rows, head_dim] with kv_rows = n_kv_heads * (position + 1), row order
(kv_head, position) row-major.

Stages (flat batch-1 rows, exact runtime layout):
  stage.input, stage.rmsnorm1, stage.q, stage.k, stage.v, stage.rope_q,
  stage.rope_k, stage.attention_output, stage.output_projection,
  stage.residual1, stage.rmsnorm2, stage.gate, stage.up,
  stage.silu_gate_mul_up, stage.down, stage.final_output,
  kv.k_state, kv.v_state

Usage:
  python3 tools/generate_golden.py --out data/golden_p0.cudalm --position 0
  python3 tools/generate_golden.py --out data/golden_p7.cudalm --position 7
  python3 tools/generate_golden.py --selftest
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import binfmt  # noqa: E402
import convert_weights as cw  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*")


def _f16(x) -> "torch.Tensor":
    return x.to(torch.float16)


# ---------------------------------------------------------------------------
# Stage references (fp32 math, fp16 at the boundary)
# ---------------------------------------------------------------------------
def rmsnorm_ref(x16, w16, eps: float) -> "torch.Tensor":
    """x16 [1, H] fp16, w16 [H] fp16 -> [1, H] fp16."""
    x = x16.float()
    ms = (x * x).mean(dim=1, keepdim=True)
    y = x * torch.rsqrt(ms + eps) * w16.float()
    return y.to(torch.float16)


def w4a16_ref(x16, W_deq) -> "torch.Tensor":
    """x16 [1, K] fp16, W_deq [N, K] fp32 (dequantized) -> [1, N] fp16."""
    y = W_deq @ x16.float().reshape(-1)
    return y.reshape(1, -1).to(torch.float16)


def rope_ref(x16, cos_row16, sin_row16, n_rows: int, head_dim: int) -> "torch.Tensor":
    """x16 [1, n_rows*head_dim] fp16; cos/sin rows [head_dim/2] fp16.
    Interleaved-pair rotation -> [1, n_rows*head_dim] fp16."""
    x = x16.float().reshape(n_rows, head_dim)
    c = cos_row16.float()
    s = sin_row16.float()
    a = x[:, 0::2]
    b = x[:, 1::2]
    ya = a * c - b * s
    yb = a * s + b * c
    y = torch.stack((ya, yb), dim=-1).reshape(1, -1)
    return y.to(torch.float16)


def attention_ref(rope_q16, K_state, V_state, n_heads: int, n_kv: int,
                  head_dim: int, position: int) -> "torch.Tensor":
    """rope_q16 [1, n_heads*head_dim] fp16; K/V state [n_kv, position+1,
    head_dim] fp16 (as stored) -> [1, n_heads*head_dim] fp16."""
    q = rope_q16.float().reshape(n_heads, head_dim)
    K = K_state.float()
    V = V_state.float()
    scale = 1.0 / math.sqrt(head_dim)
    rows = []
    for h in range(n_heads):
        kh = h * n_kv // n_heads  # GQA: h // (n_heads / n_kv)
        s = (K[kh] @ q[h]) * scale          # [position+1] fp32
        s = s - s.max()
        w = torch.exp(s)
        w = w / w.sum()
        rows.append(w @ V[kh])
    return torch.stack(rows).reshape(1, -1).to(torch.float16)


def dequant_proj(packed_u8, scale16) -> "torch.Tensor":
    """Packed INT4 + fp16 scales -> fp32 dequantized weight [N, K]."""
    q = cw.unpack_w(packed_u8)
    s = scale16.float().repeat_interleave(cw.GROUP_SIZE, dim=1)
    return q.float() * s


# ---------------------------------------------------------------------------
# Block simulation
# ---------------------------------------------------------------------------
def generate(cfg: binfmt.ModelConfig, position: int, seed: int,
             history_seed: int):
    """Simulate one block at `position`. Returns (records, summary) where
    records is a list of (name, dtype, dims, bytes) in file order."""
    assert 0 <= position < cfg.max_seq_len
    H, hd, n_heads, nkv, inter = (cfg.hidden_size, cfg.head_dim,
                                  cfg.n_heads, cfg.n_kv_heads,
                                  cfg.intermediate_size)
    eps = cfg.eps

    # -- weights: identical to convert_weights for the same seed ------------
    weight_records, stats = cw.build_block_tensors(cfg, seed)
    W = {}
    for name, dtype, dims, blob in weight_records:
        W[name] = (dims, blob)

    def w16(name):
        dims, blob = W[name]
        return cw._bytes_to_tensor(blob, torch.float16, dims)

    cos = w16("attn.rope_cos")          # [max_seq, hd/2] fp16
    sin = w16("attn.rope_sin")

    def deq(proj):
        dims_w, blob_w = W[proj + ".weight"]
        dims_s, blob_s = W[proj + ".scale"]
        packed = cw._bytes_to_tensor(blob_w, torch.uint8, dims_w)
        scale = cw._bytes_to_tensor(blob_s, torch.float16, dims_s)
        return dequant_proj(packed, scale)

    Wdeq = {p: deq(p) for p in cw.PROJ_NAMES}

    # -- KV history + current step -------------------------------------------
    torch.manual_seed(history_seed)
    K_state = torch.zeros(nkv, position + 1, hd, dtype=torch.float16)
    V_state = torch.zeros(nkv, position + 1, hd, dtype=torch.float16)

    def step_kv(h_t: "torch.Tensor", t: int):
        """h_t [1, H] fp16 -> (rms1, k, v, rope_k), all fp16 rows."""
        rms = rmsnorm_ref(h_t, w16("attn_norm.weight"), eps)
        k = w4a16_ref(rms, Wdeq["attn.k_proj"])
        v = w4a16_ref(rms, Wdeq["attn.v_proj"])
        rk = rope_ref(k, cos[t], sin[t], nkv, hd)
        return rms, k, v, rk

    for t in range(position + 1):
        h_t = _f16(0.05 * torch.randn(1, H))
        _rms, k, v, rk = step_kv(h_t, t)
        K_state[:, t] = rk.reshape(nkv, hd)
        V_state[:, t] = v.reshape(nkv, hd)
        if t == position:
            x_in, rms1, k_cur, v_cur, rk_cur = h_t, _rms, k, v, rk

    # -- current-position stages ---------------------------------------------
    q = w4a16_ref(rms1, Wdeq["attn.q_proj"])
    rq = rope_ref(q, cos[position], sin[position], n_heads, hd)
    attn = attention_ref(rq, K_state, V_state, n_heads, nkv, hd, position)
    oproj = w4a16_ref(attn, Wdeq["attn.o_proj"])
    res1 = (x_in.float() + oproj.float()).to(torch.float16)
    rms2 = rmsnorm_ref(res1, w16("ffn_norm.weight"), eps)
    gate = w4a16_ref(rms2, Wdeq["mlp.gate_proj"])
    up = w4a16_ref(rms2, Wdeq["mlp.up_proj"])
    sgmu = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.float16)
    down = w4a16_ref(sgmu, Wdeq["mlp.down_proj"])
    final = (res1.float() + down.float()).to(torch.float16)

    stages = {
        "stage.input": x_in,
        "stage.rmsnorm1": rms1,
        "stage.q": q,
        "stage.k": k_cur,
        "stage.v": v_cur,
        "stage.rope_q": rq,
        "stage.rope_k": rk_cur,
        "stage.attention_output": attn,
        "stage.output_projection": oproj,
        "stage.residual1": res1,
        "stage.rmsnorm2": rms2,
        "stage.gate": gate,
        "stage.up": up,
        "stage.silu_gate_mul_up": sgmu,
        "stage.down": down,
        "stage.final_output": final,
    }

    # -- invariants (cheap, exact) --------------------------------------------
    for name, t in stages.items():
        if not torch.isfinite(t.float()).all():
            raise AssertionError(f"{name}: non-finite stage values")
    # cache rows at `position` must be exactly the current step's k/v
    if not torch.equal(K_state[:, position].reshape(1, -1), rk_cur):
        raise AssertionError("kv.k_state row mismatch at position")
    if not torch.equal(V_state[:, position].reshape(1, -1), v_cur):
        raise AssertionError("kv.v_state row mismatch at position")
    if position == 0:
        # cos row0 == 1, sin row0 == 0 exactly -> RoPE is the identity
        if not torch.equal(rq, q):
            raise AssertionError("p=0: rope_q != q (identity broken)")
        if not torch.equal(rk_cur, k_cur):
            raise AssertionError("p=0: rope_k != k (identity broken)")
        # softmax over one element is 1 -> each query head h's output is
        # exactly its GQA kv head's v row
        for h in range(n_heads):
            kh = h * nkv // n_heads
            if not torch.equal(attn[:, h * hd:(h + 1) * hd],
                               v_cur[:, kh * hd:(kh + 1) * hd]):
                raise AssertionError("p=0: attention head != v row")

    # -- assemble records in file order ---------------------------------------
    records = list(weight_records)
    order = binfmt.golden_tensor_expectations(cfg, position)
    extra = {
        "kv.k_state": K_state.reshape(nkv * (position + 1), hd),
        "kv.v_state": V_state.reshape(nkv * (position + 1), hd),
    }
    for name, dtype, dims in order:
        t = stages.get(name)
        if t is None:
            t = extra[name]
        if t.dtype != torch.float16:
            raise AssertionError(f"{name}: expected fp16 tensor")
        records.append((name, binfmt.DT_FP16, tuple(dims),
                        cw._tensor_to_bytes(t)))

    summary = {
        "position": position,
        "seed": seed,
        "history_seed": history_seed,
        "n_tensors": len(records),
        "stage_max_abs": {n: float(t.float().abs().max())
                          for n, t in stages.items()},
        "weight_stats": stats,
    }
    return records, summary


def build_golden_file(cfg: binfmt.ModelConfig, position: int, seed: int,
                      history_seed: int) -> bytes:
    records, summary = generate(cfg, position, seed, history_seed)
    w = binfmt.GoldenFileWriter(cfg, position)
    for name, dtype, dims, blob in records:
        w.add(name, dtype, dims, blob)
    blob = w.build()
    # round-trip through the reader before handing the bytes out
    g = binfmt.ParsedGoldenFile(blob)
    assert g.position == position
    g.validate_golden_tensors()
    return blob


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------
def _check(cond, msg):
    if not cond:
        print(f"[SELFTEST FAIL] {msg}")
        sys.exit(1)


def selftest() -> int:
    cfg = binfmt.ModelConfig.v01_default()
    with tempfile.TemporaryDirectory() as td:
        paths = {}
        for p in (0, 7):
            blob = build_golden_file(cfg, p, 20250922, 20250923)
            # determinism
            blob2 = build_golden_file(cfg, p, 20250922, 20250923)
            _check(blob == blob2, f"p={p}: determinism (same bytes)")
            path = os.path.join(td, f"golden_p{p}.cudalm")
            binfmt.write_file(path, blob)
            paths[p] = path

        for p in (0, 7):
            g = binfmt.read_golden_file(paths[p])
            _check(g.position == p, f"p={p}: position round-trip")
            _check(g.cfg == cfg, f"p={p}: config round-trip")
            g.validate_golden_tensors()
            # weights embedded in the golden must match the plain weight file
            blobw, _ = cw.build_file(cfg, 20250922)
            wf = binfmt.ParsedWeightFile(blobw)
            for name, dtype, dims in binfmt.block_tensor_expectations(cfg):
                a, b = g.find(name), wf.find(name)
                _check(a.data == b.data, f"p={p}: weight {name} differs from weight file")
            # stage values sane: input magnitude ~0.05*N(0,1), all finite,
            # and nonzero
            for name, _dt, _dims in binfmt.golden_tensor_expectations(cfg, p):
                t = cw._bytes_to_tensor(g.find(name).data, torch.float16,
                                        g.find(name).dims)
                tf = t.float()
                _check(bool(torch.isfinite(tf).all()),
                       f"p={p}: {name} non-finite")
                if name != "stage.silu_gate_mul_up":
                    _check(bool((tf != 0).any()), f"p={p}: {name} all zero")
        # p=0 invariants visible from the file: each attention head == its
        # GQA kv head's v row (softmax over one element is exactly 1)
        g0 = binfmt.read_golden_file(paths[0])
        a0 = cw._bytes_to_tensor(g0.find("stage.attention_output").data,
                                 torch.float16, (1, cfg.hidden_size))
        v0 = cw._bytes_to_tensor(g0.find("stage.v").data, torch.float16,
                                 (1, cfg.n_kv_heads * cfg.head_dim))
        hd = cfg.head_dim
        for h in range(cfg.n_heads):
            kh = h * cfg.n_kv_heads // cfg.n_heads
            _check(torch.equal(a0[:, h * hd:(h + 1) * hd],
                               v0[:, kh * hd:(kh + 1) * hd]),
                   f"p=0: attention head {h} != v kv-head {kh} in file")
        # p=7: attention must differ from the current v (history used);
        # compare query head 0 against its kv head's v row
        g7 = binfmt.read_golden_file(paths[7])
        a7 = cw._bytes_to_tensor(g7.find("stage.attention_output").data,
                                 torch.float16, (1, cfg.hidden_size))
        v7 = cw._bytes_to_tensor(g7.find("stage.v").data, torch.float16,
                                 (1, cfg.n_kv_heads * cfg.head_dim))
        _check(not torch.equal(a7[:, 0:hd], v7[:, 0:hd]),
               "p=7: attention head0 == v kv-head0 (history unused?)")
        # KV state rows: position rows differ from each other
        ks = cw._bytes_to_tensor(g7.find("kv.k_state").data, torch.float16,
                                 (cfg.n_kv_heads * 8, cfg.head_dim))
        for h in range(cfg.n_kv_heads):
            row_a = ks[h * 8]
            row_b = ks[h * 8 + 1]
            _check(not torch.equal(row_a, row_b),
                   f"p=7: kv.k_state head {h} pos0==pos1 (degenerate)")

    print("[SELFTEST PASS] golden p0+p7: container, weights, invariants")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--position", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20250922)
    ap.add_argument("--history-seed", type=int, default=None,
                    help="defaults to seed + 1")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    hseed = a.history_seed if a.history_seed is not None else a.seed + 1
    if a.position < 0 or a.position >= binfmt.ModelConfig.v01_default().max_seq_len:
        print(f"position {a.position} out of range", file=sys.stderr)
        return 2

    cfg = binfmt.ModelConfig.v01_default()
    recs, summary = generate(cfg, a.position, a.seed, hseed)
    w = binfmt.GoldenFileWriter(cfg, a.position)
    for name, dtype, dims, blob_r in recs:
        w.add(name, dtype, dims, blob_r)
    blob = w.build()
    # round-trip through the reader before writing
    g = binfmt.ParsedGoldenFile(blob)
    g.validate_golden_tensors()
    out = a.out or f"data/block_v01_golden_p{a.position}.cudalm"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    binfmt.write_file(out, blob)
    print(f"wrote {out} ({len(blob)} bytes, position={a.position}, "
          f"seed={a.seed}, history_seed={hseed})")
    for name, _dt, _dims, _blob in recs:
        if name.startswith("stage."):
            print(f"  {name:<26s} max_abs={summary['stage_max_abs'][name]:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
