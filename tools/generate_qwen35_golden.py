#!/usr/bin/env python3
"""CUDALM v0.2 — Qwen3.5 full-attention-layer golden generator (Phase B).

Oracle (HARD GATE reference for the C++ runtime):
  * the OFFICIAL pinned transformers (transformers fc91372 — the generator
    asserts at run time that the installed
    transformers/models/qwen3_5/modeling_qwen3_5.py is byte-identical to
    the provenance copy under the checkpoint), driven with the REAL
    checkpoint weights, EXCEPT that the 7 per-layer GEMV Linear modules
    (q/k/v/o_proj, mlp gate/up/down) are swapped for QuantLinear:
        y = (W_deq_fp32 @ x.float()).to(bf16)
    i.e. the exact CUDALM W4A16 GEMV contract (fp32 accumulation of
    (q * scale_fp16_as_f32) * x, one bf16 RNE at the boundary). The
    dequantized weights come from the SAME .cudalm v2 file the runtime
    loads (same shared quantizer, docs/qwen35_architecture.md §4/§11).
  * every non-GEMV op runs the official pinned module/function in bf16
    (zero-centered RMSNorm, partial rotate-half RoPE via the official
    Qwen3_5TextRotaryEmbedding + apply_rotary_pos_emb, the official eager
    attention math via repeat_kv/matmul/softmax, sigmoid gating, residual
    adds, SwiGLU) — so the bf16 rounding boundaries are the official
    ones, and the C++ kernels mirror them (docs §5-§7, §14).

A/B split (docs §11):
  A (runtime correctness, HARD GATE): CUDALM runtime (quantized weights +
    seeded state) vs this golden.
  B (quantization fidelity, REPORT ONLY): official bf16 weights vs the
    quantized reference (--fidelity-report; per-weight + layer-output
    max_abs/RMSE/cosine). It is written to a side report and never gates.

KV history for position p > 0: positions 0..p-1 are filled by running the
SAME quantized reference layer on seeded random hidden states
(0.05*randn, fp32 -> bf16, one torch.manual_seed(input_seed) stream for
ALL positions including p — the v0.1.1 convention), so the stored KV state
is non-trivial and built from the same quantized weights the runtime uses
(docs §13 "state seeding"). The file embeds the FULL cache after the write
at p (rows (kv_head, position) row-major).

Output container: CUDLMG02 (tools/common/golden_v2.py; the C++ mirror is
include/cudalm/golden_loader_v2.h). 23 bf16 tensors: 21 stages + 2 KV
states. The decode state (position, layer_idx, input_seed) is in the
header.

Usage:
  python3 tools/generate_qwen35_golden.py \
      --cudalm build/data/qwen35_08b_l3.cudalm \
      --checkpoint-dir /root/models/Qwen3.5-0.8B-Base \
      --layer 3 --out data/qwen35_golden_l3_p0.cudalm \
      --position 0 [--input-seed 20260209] [--fidelity-report r.json]
  python3 tools/generate_qwen35_golden.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import cudalm_v2 as v2  # noqa: E402
import golden_v2 as gv2  # noqa: E402
from w4a16_quant import dequant_reference, fidelity_stats, quantize_w4a16  # noqa: E402
import token_seq  # noqa: E402

# --- Pinned provenance (docs/qwen35_architecture.md §1) ----------------------
MODEL_REPO = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
CONFIG_SHA256 = "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
CHECKPOINT_SHA256 = (
    "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c")
TRANSFORMERS_COMMIT = "fc9137225880a9d03f130634c20f9dbe36a7b8bf"
# modeling_qwen3_5.py sha256 of the pinned transformers commit (also the
# provenance copy under the checkpoint).
MODELING_SHA256 = (
    "b6f02dcd1b66610df293084e00bf9bea4fc6a7e5336ffc6ff446edc7ddcd8601")
ARCH_ID = "qwen3.5-text"

BF16 = torch.bfloat16

GEMV_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
NORM_NAMES = [
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
]

STAGE_NAMES = [
    "stage.input", "stage.rmsnorm1", "stage.q_gate", "stage.q",
    "stage.att_gate", "stage.k", "stage.v", "stage.q_norm", "stage.k_norm",
    "stage.rope_q", "stage.rope_k", "stage.attention_raw",
    "stage.attention_gated", "stage.o_proj", "stage.residual1",
    "stage.rmsnorm2", "stage.mlp_gate", "stage.mlp_up", "stage.silu_mul",
    "stage.mlp_down", "stage.final_output",
]

# --- Phase C (Gated DeltaNet) tensor names (docs §4/§8) ----------------------
# GEMV Linears swapped for QuantLinear on a DeltaNet layer (5 linear_attn +
# 3 mlp).
DELTANET_GEMV_NAMES = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
# Canonical DeltaNet decode stage order (all bf16 except stage.g, which is
# fp32 — see stage_tensors_deltanet).
STAGE_NAMES_DN = [
    "stage.input", "stage.rmsnorm1", "stage.in_proj_qkv", "stage.in_proj_z",
    "stage.in_proj_b", "stage.in_proj_a", "stage.conv_out", "stage.conv_silu",
    "stage.q", "stage.k", "stage.v", "stage.beta", "stage.g", "stage.core_out",
    "stage.gated_norm", "stage.out_proj", "stage.residual1", "stage.rmsnorm2",
    "stage.mlp_gate", "stage.mlp_up", "stage.silu_mul", "stage.mlp_down",
    "stage.final_output",
]
# Persistent-state tensor names (the Phase C hard gate): conv_state bf16
# [conv_dim, 3], recurrent_state fp32 [n_heads, key_dim, value_dim], each
# captured before AND after the decode step.
STATE_NAMES_DN = [
    "state.conv_before", "state.conv_after",
    "state.recurrent_before", "state.recurrent_after",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_of(t: torch.Tensor) -> bytes:
    """Contiguous raw bytes of a torch tensor (any element width)."""
    t = t.contiguous()
    return t.view(torch.uint8).reshape(-1).numpy().tobytes()


# ---------------------------------------------------------------------------
# Pinned-oracle imports (asserted byte-identical to the provenance copy)
# ---------------------------------------------------------------------------
def import_pinned_transformers(ckpt_dir: str):
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    mod_path = os.path.abspath(mq.__file__)
    sha = sha256_file(mod_path)
    if sha != MODELING_SHA256:
        raise SystemExit(
            f"installed transformers modeling_qwen3_5.py sha256 {sha} != "
            f"pinned {MODELING_SHA256} (docs/qwen35_architecture.md §1.2); "
            "refusing to generate goldens with a non-pinned oracle")
    prov = os.path.join(
        ckpt_dir, "provenance", "transformers_fc91372",
        "modeling_qwen3_5.py")
    if os.path.isfile(prov) and sha256_file(prov) != MODELING_SHA256:
        raise SystemExit(
            "provenance modeling copy under the checkpoint differs from "
            "the pinned sha256 (corrupted checkout?)")
    # The oracle uses the pure-torch path: no flash-linear-attention, no
    # causal-conv1d (docs §1.2). For Phase B (full attention only) the
    # relevant kernel is the plain eager attention, which is what the
    # forward code below calls explicitly.
    return transformers


class QuantLinear(torch.nn.Module):
    """Mirror of the CUDALM W4A16 GEMV contract (docs §4 row 8):
    y = bf16( fp32-matmul(W_deq, x_fp32) ) — fp32 accumulation, one bf16
    RNE at the boundary. W_deq = q * scale_fp16_as_f32 (exact dequant of
    the stored .cudalm v2 tensors)."""

    def __init__(self, W_deq: torch.Tensor) -> None:
        super().__init__()
        self.W_deq = W_deq.contiguous()  # fp32 [N, K], non-parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear semantics: (x_2d @ W^T) -> (*batch, N); batch dims
        # flattened and restored; fp32 matmul, one bf16 RNE at the boundary.
        batch = x.shape[:-1]
        x2 = x.float().reshape(-1, x.shape[-1])
        y = x2 @ self.W_deq.t()
        return y.reshape(batch + (self.W_deq.shape[0],)).to(x.dtype)


def tensor_from_v2(file, name: str, dtype: torch.dtype) -> torch.Tensor:
    t = file.find(name)
    if t is None:
        raise SystemExit(f"tensor {name!r} missing from the v2 file")
    dims = t.dims
    arr = torch.frombuffer(bytearray(t.data), dtype=torch.uint8)
    # Reshape to the container's recorded shape (packed int4 is [N, K/2]).
    return arr.view(dtype).view(dims)


def dequant_v2(file, prefix: str, proj: str) -> torch.Tensor:
    packed = tensor_from_v2(file, f"{prefix}{proj}.weight", torch.uint8)
    scale16 = tensor_from_v2(file, f"{prefix}{proj}.scale", torch.float16)
    return dequant_reference(packed, scale16)  # fp32 [N, K]


def build_quantized_layer(v2file, text_cfg, layer_idx: int):
    """Official Qwen3_5DecoderLayer with the 7 GEMV Linears swapped for
    QuantLinear (dequantized W4A16) and the bf16 norm weights loaded
    byte-exact from the same .cudalm v2 file the runtime loads."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    layer = Qwen3_5DecoderLayer(text_cfg, layer_idx)
    prefix = f"layers.{layer_idx}."
    for proj in GEMV_NAMES:
        mod, leaf = proj.rsplit(".", 1)
        parent = layer.self_attn if mod == "self_attn" else layer.mlp
        setattr(parent, leaf, QuantLinear(dequant_v2(v2file, prefix, proj)))
    for mod_path in NORM_NAMES:
        t = tensor_from_v2(v2file, prefix + mod_path + ".weight", BF16)
        obj = layer
        for p in mod_path.split("."):
            obj = getattr(obj, p)
        setattr(obj, "weight", torch.nn.Parameter(t))
    return layer


def build_quantized_deltanet_layer(v2file, text_cfg, layer_idx: int):
    """Official Qwen3_5DecoderLayer (DeltaNet) with the 8 GEMV Linears swapped
    for QuantLinear (dequantized W4A16), the bf16 layernorm weights loaded
    byte-exact, and the fp32 linear_attn.norm.weight + A_log loaded byte-exact
    from the SAME .cudalm v2 file the runtime loads (docs §4/§11)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    layer = Qwen3_5DecoderLayer(text_cfg, layer_idx)
    prefix = f"layers.{layer_idx}."
    for proj in DELTANET_GEMV_NAMES:
        mod, leaf = proj.rsplit(".", 1)
        parent = layer.linear_attn if mod == "linear_attn" else layer.mlp
        setattr(parent, leaf, QuantLinear(dequant_v2(v2file, prefix, proj)))
    # bf16 zero-centered layernorms (replace the module's .weight in place).
    for mod_path in ("input_layernorm", "post_attention_layernorm"):
        t = tensor_from_v2(v2file, prefix + mod_path + ".weight", BF16)
        getattr(layer, mod_path).weight = torch.nn.Parameter(t)
    # fp32 gated-norm weight (docs §6.2: norm.weight is fp32) + A_log fp32.
    norm_w = tensor_from_v2(v2file, prefix + "linear_attn.norm.weight",
                            torch.float32)
    setattr(layer.linear_attn.norm, "weight",
            torch.nn.Parameter(norm_w))
    a_log = tensor_from_v2(v2file, prefix + "linear_attn.A_log",
                           torch.float32)
    setattr(layer.linear_attn, "A_log", torch.nn.Parameter(a_log))
    # bf16 conv1d depthwise weights + dt_bias (docs §4).
    conv_w = tensor_from_v2(v2file, prefix + "linear_attn.conv1d.weight", BF16)
    layer.linear_attn.conv1d.weight = torch.nn.Parameter(conv_w)
    dt_bias = tensor_from_v2(v2file, prefix + "linear_attn.dt_bias", BF16)
    setattr(layer.linear_attn, "dt_bias", torch.nn.Parameter(dt_bias))
    return layer


def build_official_layer(safetensors, text_cfg, layer_idx: int):
    """Official layer with the REAL bf16 checkpoint weights (oracle side
    of the REPORT-ONLY fidelity split, docs §11-B)."""
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    layer = Qwen3_5DecoderLayer(text_cfg, layer_idx)
    prefix = f"model.language_model.layers.{layer_idx}."
    with safe_open(safetensors, framework="pt") as f:
        keys = [k for k in f.keys() if k.startswith(prefix)]
        for k in keys:
            leaf = k[len(prefix):]
            t = f.get_tensor(k).to(BF16)
            parts = leaf.split(".")
            obj = layer
            for p in parts[:-1]:
                obj = getattr(obj, p)
            if leaf.endswith(".weight"):
                setattr(obj, parts[-1], torch.nn.Parameter(t))
    return layer


# ---------------------------------------------------------------------------
# Reference forward (official op order; GEMVs are the swapped QuantLinears)
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_step(layer, rope, x: torch.Tensor, position: int,
                 cache):
    """One decode step at `position`. x: [1,1,H] bf16. Returns
    (stages dict, new cache (Kc, Vc)).

    Op order mirrors the pinned official forward verbatim
    (Qwen3_5DecoderLayer.forward + Qwen3_5Attention.forward, the pure-torch
    eager path with attention_mask=None): RMSNorm -> q_proj [q;gate] ->
    chunk/split -> q_norm/k_norm -> partial RoPE -> cache update ->
    repeat_kv -> matmul*scaling -> softmax(fp32).to(bf16) -> matmul ->
    sigmoid gate -> o_proj -> residual -> RMSNorm -> SwiGLU -> residual.
    """
    attn = layer.self_attn
    B, S = x.shape[0], x.shape[1]
    input_shape = (B, S)
    q_heads = attn.config.num_attention_heads
    kv_heads = attn.config.num_key_value_heads
    hd = attn.head_dim

    stages = {"input": x}

    h0 = layer.input_layernorm(x)  # bf16 (official zero-centered norm)
    stages["rmsnorm1"] = h0

    q_g = attn.q_proj(h0)  # [1,1,2*heads*hd] (QuantLinear)
    stages["q_gate"] = q_g
    q, gate = torch.chunk(
        q_g.view(*input_shape, q_heads, 2 * hd), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    stages["q"] = q
    stages["att_gate"] = gate

    q_n = attn.q_norm(
        q.view(B, S, q_heads, hd)).transpose(1, 2)  # [1,heads,1,hd]
    k = attn.k_proj(h0)  # [1,1,kv*hd]
    k_n = attn.k_norm(k.view(B, S, kv_heads, hd)).transpose(1, 2)
    v = attn.v_proj(h0).view(B, S, kv_heads, hd).transpose(1, 2)
    stages["k"] = k
    stages["v"] = v
    stages["q_norm"] = q_n
    stages["k_norm"] = k_n

    pos_ids = torch.tensor([[position]], dtype=torch.long, device=x.device)
    cos, sin = rope(x, pos_ids)  # bf16 [1,1,rotary_dim]
    q_r, k_r = apply_rotary_pos_emb(q_n, k_n, cos, sin)
    stages["rope_q"] = q_r
    stages["rope_k"] = k_r

    if cache is None:
        Kc, Vc = k_r, v
    else:
        Kc = torch.cat([cache[0], k_r], dim=2)
        Vc = torch.cat([cache[1], v], dim=2)

    n_rep = attn.num_key_value_groups
    k_rep = repeat_kv(Kc, n_rep)
    v_rep = repeat_kv(Vc, n_rep)
    attn_w = torch.matmul(q_r, k_rep.transpose(2, 3)) * attn.scaling
    attn_w = torch.nn.functional.softmax(
        attn_w, dim=-1, dtype=torch.float32).to(x.dtype)
    attn_out = torch.matmul(attn_w, v_rep)
    attn_out = attn_out.transpose(1, 2).contiguous()
    attn_out = attn_out.reshape(*input_shape, -1).contiguous()
    stages["attention_raw"] = attn_out

    gated = attn_out * torch.sigmoid(gate)
    stages["attention_gated"] = gated

    o = attn.o_proj(gated)  # QuantLinear
    stages["o_proj"] = o

    res1 = x + o
    stages["residual1"] = res1

    h1 = layer.post_attention_layernorm(res1)
    stages["rmsnorm2"] = h1

    g = layer.mlp.gate_proj(h1)
    u = layer.mlp.up_proj(h1)
    sg = torch.nn.functional.silu(g) * u
    stages["mlp_gate"] = g
    stages["mlp_up"] = u
    stages["silu_mul"] = sg

    d = layer.mlp.down_proj(sg)
    stages["mlp_down"] = d

    y = res1 + d
    stages["final_output"] = y

    return stages, (Kc, Vc)


def apply_rotary_pos_emb(q, k, cos, sin):
    # Module-global import point (resolved from the pinned module below).
    return _P.apply_rotary_pos_emb(q, k, cos, sin)


def repeat_kv(hidden_states, n_rep: int):
    return _P.repeat_kv(hidden_states, n_rep)


_P = None  # pinned modeling module (set by import_pinned_transformers)


def run_reference(layer, rope, position: int, input_seed: int, H: int):
    """Drive positions 0..position on the seeded input stream; return the
    stages captured at `position` and the KV cache after the write there.
    ONE torch.manual_seed(input_seed) stream for all positions (v0.1.1
    convention: history and current input from the same draw order)."""
    torch.manual_seed(input_seed)
    cache = None
    stages = None
    for t in range(position + 1):
        x = (0.05 * torch.randn(1, 1, H, dtype=torch.float32)).to(BF16)
        stages, cache = forward_step(layer, rope, x, t, cache)
    Kc, Vc = cache
    return stages, Kc, Vc


# ---------------------------------------------------------------------------
# Phase C — Gated DeltaNet reference forward (decode path, docs §8)
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_step_deltanet(layer, x: torch.Tensor, position: int,
                          conv_state: torch.Tensor,
                          recurrent_state):
    """One DeltaNet DECODE step at `position` (the pinned decode path:
    torch_causal_conv1d_update + torch_recurrent_gated_delta_rule). x: [1,1,H]
    bf16. conv_state: [1, conv_dim, 3] bf16 (updated IN PLACE). recurrent_state:
    [1, n_heads, 128, 128] fp32 or None. Returns
    (stages, conv_state, rec_after, conv_before, rec_before).

    The conv mirrors the pinned torch_causal_conv1d_update op-for-op (cat the
    OLD state + new token, copy_ the state in place, F.conv1d over the OLD
    window, F.silu); the recurrent rule uses the pinned
    torch_recurrent_gated_delta_rule (l2norm applied to bf16 q/k inside the
    kernel, then fp32; decay -> delta -> output from the UPDATED state); the
    gated norm uses the pinned Qwen3_5RMSNormGated module (two bf16 roundings,
    docs §6.2).
    """
    dn = layer.linear_attn
    B, S = x.shape[0], x.shape[1]
    key_dim = dn.key_dim
    value_dim = dn.value_dim
    conv_dim = dn.conv_dim
    n_heads = dn.num_v_heads
    hd_k = dn.head_k_dim
    hd_v = dn.head_v_dim

    stages = {"input": x}
    h0 = layer.input_layernorm(x)
    stages["rmsnorm1"] = h0

    mixed = dn.in_proj_qkv(h0)            # [1,1,conv_dim] bf16 (QuantLinear)
    stages["in_proj_qkv"] = mixed
    z = dn.in_proj_z(h0).reshape(B, S, -1, hd_v)  # [1,1,n_heads,hd_v]
    stages["in_proj_z"] = z
    b = dn.in_proj_b(h0)                   # [1,1,n_heads] bf16
    a = dn.in_proj_a(h0)                   # [1,1,n_heads] bf16
    stages["in_proj_b"] = b
    stages["in_proj_a"] = a

    # depthwise causal conv1d decode update (mirror torch_causal_conv1d_update)
    mixed_t = mixed.transpose(1, 2)        # [1,conv_dim,S]
    conv_before = conv_state.clone()
    w = dn.conv1d.weight                   # [conv_dim,1,kernel] bf16
    hidden_new = torch.cat([conv_state, mixed_t], dim=-1).to(w.dtype)
    conv_state.copy_(hidden_new[:, :, -3:])  # IN-PLACE state update
    conv_out = F.conv1d(hidden_new, w, None, padding=0,
                        groups=conv_dim)[:, :, -S:]
    stages["conv_out"] = conv_out          # [1,conv_dim,S] bf16 (pre-SiLU)
    mixed_conv = F.silu(conv_out).to(mixed.dtype)  # [1,conv_dim,S] bf16
    stages["conv_silu"] = mixed_conv
    conv_after = conv_state.clone()

    mixed_conv = mixed_conv.transpose(1, 2)  # [1,1,conv_dim]
    query, key, value = torch.split(mixed_conv,
                                    [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(B, S, -1, hd_k)    # [1,1,n_heads,hd_k]
    key = key.reshape(B, S, -1, hd_k)
    value = value.reshape(B, S, -1, hd_v)
    stages["q"] = query
    stages["k"] = key
    stages["v"] = value

    beta = b.sigmoid()                     # bf16 [1,1,n_heads]
    stages["beta"] = beta
    g = -dn.A_log.float().exp() * F.softplus(a.float() + dn.dt_bias)  # fp32
    stages["g"] = g

    # gated delta-rule recurrent (pinned function; l2norm in-kernel)
    rec_before = (recurrent_state.clone() if recurrent_state is not None
                  else torch.zeros(B, n_heads, hd_k, hd_v,
                                   dtype=torch.float32))
    core, rec_after = _P.torch_recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=recurrent_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    stages["core_out"] = core              # bf16 [1,1,n_heads,hd_v]

    # gated RMSNorm (pinned Qwen3_5RMSNormGated module, docs §6.2)
    core_2d = core.reshape(-1, hd_v)       # [n_heads,hd_v]
    z_2d = z.reshape(-1, hd_v)
    gated = dn.norm(core_2d, z_2d)         # bf16 [n_heads,hd_v]
    stages["gated_norm"] = gated
    out = dn.out_proj(gated.reshape(B, S, -1))  # bf16 [1,1,H]
    stages["out_proj"] = out

    # decoder residual wiring (docs §9)
    res1 = x + out
    stages["residual1"] = res1
    h1 = layer.post_attention_layernorm(res1)
    stages["rmsnorm2"] = h1
    g2 = layer.mlp.gate_proj(h1)
    u2 = layer.mlp.up_proj(h1)
    sg = F.silu(g2) * u2
    stages["mlp_gate"] = g2
    stages["mlp_up"] = u2
    stages["silu_mul"] = sg
    d = layer.mlp.down_proj(sg)
    stages["mlp_down"] = d
    y = res1 + d
    stages["final_output"] = y

    return stages, conv_state, rec_after, conv_before, rec_before


def make_state_seed(state_seed: int, conv_dim: int, n_heads: int,
                    hd_k: int, hd_v: int):
    """Deterministic non-zero (conv_state, recurrent_state) pair for scenario C
    (seeded non-zero previous state), independent of the input stream."""
    gen = torch.Generator().manual_seed(state_seed)
    conv = (0.05 * torch.randn(1, conv_dim, 3, generator=gen,
                               dtype=torch.float32)).to(BF16)
    rec = 0.01 * torch.randn(1, n_heads, hd_k, hd_v, generator=gen,
                             dtype=torch.float32)
    return conv, rec


def run_reference_deltanet(layer, position: int, input_seed: int, H: int,
                           state_seed=None):
    """Drive positions 0..position on the seeded input stream (ONE
    torch.manual_seed(input_seed) stream for all positions — v0.1.1
    convention), threading the persistent conv/recurrent state through. If
    `state_seed` is given, build a deterministic non-zero state and run exactly
    ONE step at `position` (scenario C). Returns
    (stages, conv_state, rec_after, conv_before, rec_before)."""
    torch.manual_seed(input_seed)
    dn = layer.linear_attn
    if state_seed is not None:
        conv_state, recurrent_state = make_state_seed(
            state_seed, dn.conv_dim, dn.num_v_heads, dn.head_k_dim,
            dn.head_v_dim)
        x = (0.05 * torch.randn(1, 1, H, dtype=torch.float32)).to(BF16)
        return forward_step_deltanet(layer, x, position, conv_state,
                                     recurrent_state)
    conv_state = torch.zeros(1, dn.conv_dim, 3, dtype=BF16)
    recurrent_state = None
    stages = None
    for t in range(position + 1):
        x = (0.05 * torch.randn(1, 1, H, dtype=torch.float32)).to(BF16)
        stages, conv_state, recurrent_state, conv_before, rec_before = (
            forward_step_deltanet(layer, x, t, conv_state, recurrent_state))
    return stages, conv_state, recurrent_state, conv_before, rec_before


def stage_tensors_deltanet(stages, conv_before, conv_after, rec_before,
                           rec_after) -> list:
    """Flatten to (name, dims, dtype, blob) in canonical golden order. All
    stages are bf16 EXCEPT stage.g (fp32); the persistent state is conv bf16
    [conv_dim, 3] and recurrent fp32 [n_heads, 128, 128]."""
    out = []
    for name in STAGE_NAMES_DN:
        t = stages[name[len("stage."):]]
        is_g = (name == "stage.g")
        dt = torch.float32 if is_g else BF16
        t = t.contiguous().view(dt)
        out.append((name, (1, int(t.numel())),
                    v2.DT_FP32 if is_g else v2.DT_BF16, bytes_of(t)))
    cb = conv_before[0].contiguous()
    ca = conv_after[0].contiguous()
    rb = rec_before[0].contiguous()
    ra = rec_after[0].contiguous()
    out.append(("state.conv_before", (int(cb.shape[0]), int(cb.shape[1])),
                v2.DT_BF16, bytes_of(cb)))
    out.append(("state.conv_after", (int(ca.shape[0]), int(ca.shape[1])),
                v2.DT_BF16, bytes_of(ca)))
    out.append(("state.recurrent_before",
                (int(rb.shape[0]), int(rb.shape[1]), int(rb.shape[2])),
                v2.DT_FP32, bytes_of(rb)))
    out.append(("state.recurrent_after",
                (int(ra.shape[0]), int(ra.shape[1]), int(ra.shape[2])),
                v2.DT_FP32, bytes_of(ra)))
    return out


def stage_tensors(stages: dict, Kc: torch.Tensor, Vc: torch.Tensor,
                  position: int) -> list:
    """Flatten to (name, dims, bytes) in canonical golden order (bf16)."""
    out = []
    for name in STAGE_NAMES:
        t = stages[name[len("stage."):]]
        t = t.contiguous().view(BF16)
        dims = (1, int(t.numel()))
        out.append((name, dims, bytes_of(t)))
    # KV state rows in (kv_head, position) row-major: Kc [1, kv, p+1, hd]
    k_state = Kc[0].contiguous()  # [kv, p+1, hd] -> flat rows (kv, t)
    v_state = Vc[0].contiguous()
    rows = k_state.shape[0] * (position + 1)
    out.append(("kv.k_state", (rows, int(k_state.shape[-1])),
                bytes_of(k_state)))
    out.append(("kv.v_state", (rows, int(v_state.shape[-1])),
                bytes_of(v_state)))
    return out


def check_invariants(stages: dict, Kc: torch.Tensor, Vc: torch.Tensor,
                     position: int) -> None:
    for name in STAGE_NAMES:
        t = stages[name[len("stage."):]].float()
        if not torch.isfinite(t).all():
            raise SystemExit(f"invariant: non-finite values in {name}")
    # The cache row at `position` equals the stage rows (bit-exact copy).
    k_row = Kc[0, :, position, :].reshape(-1)
    if not torch.equal(k_row.view(torch.uint8),
                       stages["rope_k"].reshape(-1).view(torch.uint8)):
        raise SystemExit("invariant: KV k-state row at p != stage.rope_k")
    v_row = Vc[0, :, position, :].reshape(-1)
    if not torch.equal(v_row.view(torch.uint8),
                       stages["v"].reshape(-1).view(torch.uint8)):
        raise SystemExit("invariant: KV v-state row at p != stage.v")
    if position == 0:
        # cos(0)=1 / sin(0)=0 are exact in bf16 -> rope is the identity on
        # its inputs (the per-head-normalized q/k).
        if not torch.equal(
                stages["rope_q"].reshape(-1).view(torch.uint8),
                stages["q_norm"].reshape(-1).view(torch.uint8)):
            raise SystemExit("invariant p=0: rope_q != q_norm (rope identity)")
        if not torch.equal(
                stages["rope_k"].reshape(-1).view(torch.uint8),
                stages["k_norm"].reshape(-1).view(torch.uint8)):
            raise SystemExit("invariant p=0: rope_k != k_norm (rope identity)")
        # Single-key causal attention: probs = [1] exactly, PV = V row.
        # attention_raw is [1, 1, n_heads*hd] (heads-major rows); the row
        # of Q head h equals the V row of KV head h // group.
        ar = stages["attention_raw"]
        v = stages["v"]
        n_kv = v.shape[1]
        n_heads = stages["q"].shape[2]
        hd = v.shape[-1]
        group = n_heads // n_kv
        for h in range(n_heads):
            kh = h // group
            if not torch.equal(
                    ar[0, 0, h * hd:(h + 1) * hd].view(torch.uint8),
                    v[0, kh, 0, :].view(torch.uint8)):
                raise SystemExit(
                    f"invariant p=0: attention_raw head {h} != v row "
                    f"{kh} (probs=[1] exact)")


# ---------------------------------------------------------------------------
# Fidelity report (REPORT ONLY — docs §11-B; never a gate)
# ---------------------------------------------------------------------------
def fidelity_report(v2file, safetensors, text_cfg, layer_idx: int,
                    position: int, input_seed: int, H: int, transformers):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding)
    rope = Qwen3_5TextRotaryEmbedding(text_cfg)

    q_layer = build_quantized_layer(v2file, text_cfg, layer_idx)
    o_layer = build_official_layer(safetensors, text_cfg, layer_idx)

    q_stages, _, _ = run_reference(q_layer, rope, position, input_seed, H)
    o_stages, _, _ = run_reference(o_layer, rope, position, input_seed, H)

    out = {
        "note": ("REPORT ONLY: quantization fidelity (official bf16 vs "
                 "quantized reference). NOT the runtime correctness gate "
                 "(that is the C++ runtime vs this tool's golden file)."),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "transformers_version": transformers.__version__,
        "layer_idx": layer_idx,
        "position": position,
        "input_seed": input_seed,
        "group_size": 128,
        "weights": {},
        "layer_output": {},
    }
    v2prefix = f"layers.{layer_idx}."
    for proj in GEMV_NAMES:
        packed = tensor_from_v2(v2file, v2prefix + proj + ".weight",
                                torch.uint8)
        scale16 = tensor_from_v2(v2file, v2prefix + proj + ".scale",
                                 torch.float16)
        # original bf16 weight from the safetensors checkpoint
        from safetensors import safe_open
        stkey = f"model.language_model.layers.{layer_idx}.{proj}.weight"
        with safe_open(safetensors, framework="pt") as f:
            W_orig = f.get_tensor(stkey).to(BF16)
        out["weights"][proj] = fidelity_stats(W_orig, packed, scale16)
    qy = q_stages["final_output"].float().reshape(-1)
    oy = o_stages["final_output"].float().reshape(-1)
    err = (qy - oy).abs()
    out["layer_output"] = {
        "max_abs": float(err.max().item()),
        "rmse": float(err.pow(2).mean().sqrt().item()),
        "cosine": float(torch.nn.functional.cosine_similarity(
            qy.unsqueeze(0), oy.unsqueeze(0)).item()),
    }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate(cudalm_path: str, ckpt_dir: str, layer_idx: int, out_path: str,
             position: int, input_seed: int, fidelity_path=None,
              state_seed=None) -> int:
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig)

    st_path = os.path.join(ckpt_dir, "model.safetensors")
    cfg_path = os.path.join(ckpt_dir, "raw", "config.json")
    if not (os.path.isfile(st_path) and os.path.isfile(cfg_path)):
        print(f"error: checkpoint dir {ckpt_dir!r} incomplete "
              f"(need model.safetensors + raw/config.json)", file=sys.stderr)
        return 2
    if sha256_file(cfg_path) != CONFIG_SHA256:
        print("error: config.json sha256 mismatch (not the pinned "
              "revision)", file=sys.stderr)
        return 1
    if sha256_file(st_path) != CHECKPOINT_SHA256:
        print("error: model.safetensors sha256 mismatch (not the pinned "
              "checkpoint)", file=sys.stderr)
        return 1

    transformers = import_pinned_transformers(ckpt_dir)
    global _P
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    _P = mq

    # v2 weight file (the SAME file the runtime loads).
    v2file = v2.read_v2(cudalm_path)
    if v2file.metadata.get("arch") != ARCH_ID:
        print("error: v2 file arch mismatch", file=sys.stderr)
        return 1
    pinned = v2.Qwen35Config.qwen35_08b()
    if v2file.config != pinned:
        print("error: v2 file config deviates from the pinned 0.8B "
              "contract", file=sys.stderr)
        return 1
    is_dn = v2file.config.is_linear_attention(layer_idx)
    is_fa = v2file.config.is_full_attention(layer_idx)
    if not (is_dn or is_fa):
        print(f"error: layer {layer_idx} is neither a full-attention nor a "
              f"linear-attention (Gated DeltaNet) layer", file=sys.stderr)
        return 2

    # Official text config from the checkpoint (drives the oracle modules).
    raw_cfg = json.load(open(cfg_path))["text_config"]
    text_cfg = Qwen3_5TextConfig(**raw_cfg)
    H = text_cfg.hidden_size

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding)
    rope = Qwen3_5TextRotaryEmbedding(text_cfg)

    if is_dn:
        layer = build_quantized_deltanet_layer(v2file, text_cfg, layer_idx)
        stages, conv_state, rec_after, conv_before, rec_before = (
            run_reference_deltanet(layer, position, input_seed, H, state_seed))
        specs = stage_tensors_deltanet(stages, conv_before, conv_state,
                                       rec_before, rec_after)
        tensors = [v2.V2Tensor(name, dt, dims, blob)
                   for name, dims, dt, blob in specs]
    else:
        layer = build_quantized_layer(v2file, text_cfg, layer_idx)
        stages, Kc, Vc = run_reference(layer, rope, position, input_seed, H)
        check_invariants(stages, Kc, Vc, position)
        specs = [(name, dims, v2.DT_BF16, blob)
                 for name, dims, blob
                 in stage_tensors(stages, Kc, Vc, position)]
        tensors = [v2.V2Tensor(name, dt, dims, blob)
                   for name, dims, dt, blob in specs]

    g = gv2.GoldenV2File(v2file.config, position, layer_idx, input_seed,
                         tensors)
    blob = g.to_bytes()

    # Round-trip + expected-set validation before writing.
    rt = gv2.GoldenV2File.from_bytes(blob)
    expected = gv2.golden_tensor_expectations(v2file.config, position,
                                              layer_idx)
    if len(rt.tensors) != len(expected):
        raise SystemExit("round-trip: tensor count mismatch")
    for (ename, edims, edtype), t in zip(expected, rt.tensors):
        if (t.name != ename or tuple(t.dims) != edims
                or t.dtype != edtype):
            raise SystemExit(
                f"round-trip: tensor {t.name} {tuple(t.dims)} "
                f"dtype={t.dtype} != expected {ename} {edims} dtype={edtype}")
    if bytes(blob) != rt.to_bytes():
        raise SystemExit("round-trip: bytes are not idempotent")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    gv2.write_golden_v2(out_path, g)
    print(f"wrote {out_path} ({len(blob)} bytes, layer={layer_idx}, "
          f"position={position}, input_seed={input_seed}"
          + (f", state_seed={state_seed}" if state_seed is not None else ""))
    for t in g.tensors:
        if t.name.startswith("stage."):
            torch_dt = (torch.float32 if t.dtype == v2.DT_FP32
                        else torch.bfloat16)
            arr = torch.frombuffer(bytearray(t.data), dtype=torch.uint8)
            max_abs = float(arr.view(torch_dt).float().abs().max().item())
            print(f"  {t.name:<26s} max_abs={max_abs:.6f}")

    if fidelity_path and is_dn:
        print("note: --fidelity-report is full-attention only; skipped for "
              "the DeltaNet layer (the Phase C gate is the state-validated "
              "golden)", file=sys.stderr)
        return 0
    if fidelity_path:
        rep = fidelity_report(v2file, st_path, text_cfg, layer_idx, position,
                              input_seed, H, transformers)
        os.makedirs(os.path.dirname(os.path.abspath(fidelity_path)),
                    exist_ok=True)
        with open(fidelity_path, "w") as f:
            json.dump(rep, f, indent=2, sort_keys=True)
        print(f"wrote fidelity report (report only) {fidelity_path}")
    return 0


# ---------------------------------------------------------------------------
# Phase D — 4-layer hybrid micro-stack (layers 0-3: 3x DeltaNet + 1x full
# attention). Runs the real checkpoint layers IN ORDER 0->1->2->3 for each
# token in `tokens`, threading each layer's OWN persistent state across the
# token sequence (DeltaNet conv/recurrent, full-attention KV). Reuses the SAME
# W4A16 quantized weights the runtime loads (no bf16/quantized mixing). Emits
# one CUDLMG02 file per (token, layer) — the historical per-layer container is
# reused unchanged; the micro-stack golden is the SET of these files, whose
# layer L input == layer L-1 output is what the runtime chain must reproduce.
# ---------------------------------------------------------------------------
def generate_microstack(cudalm_path: str, ckpt_dir: str, out_prefix: str,
                        tokens, input_seed: int) -> int:
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig)

    st_path = os.path.join(ckpt_dir, "model.safetensors")
    cfg_path = os.path.join(ckpt_dir, "raw", "config.json")
    if not (os.path.isfile(st_path) and os.path.isfile(cfg_path)):
        print(f"error: checkpoint dir {ckpt_dir!r} incomplete "
              f"(need model.safetensors + raw/config.json)", file=sys.stderr)
        return 2
    if sha256_file(cfg_path) != CONFIG_SHA256:
        print("error: config.json sha256 mismatch (not the pinned "
              "revision)", file=sys.stderr)
        return 1
    if sha256_file(st_path) != CHECKPOINT_SHA256:
        print("error: model.safetensors sha256 mismatch (not the pinned "
              "checkpoint)", file=sys.stderr)
        return 1

    import_pinned_transformers(ckpt_dir)
    global _P
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    _P = mq

    v2file = v2.read_v2(cudalm_path)
    if v2file.metadata.get("arch") != ARCH_ID:
        print("error: v2 file arch mismatch", file=sys.stderr)
        return 1
    pinned = v2.Qwen35Config.qwen35_08b()
    if v2file.config != pinned:
        print("error: v2 file config deviates from the pinned 0.8B "
              "contract", file=sys.stderr)
        return 1
    for L in range(4):
        if not (v2file.config.is_linear_attention(L)
                or v2file.config.is_full_attention(L)):
            print(f"error: layer {L} has an unexpected type", file=sys.stderr)
            return 2

    raw_cfg = json.load(open(cfg_path))["text_config"]
    text_cfg = Qwen3_5TextConfig(**raw_cfg)
    H = text_cfg.hidden_size
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding)
    rope = Qwen3_5TextRotaryEmbedding(text_cfg)

    # Build the 4 quantized layers (the same W4A16 weights the runtime loads).
    layers = []
    for L in range(4):
        if v2file.config.is_linear_attention(L):
            layers.append(build_quantized_deltanet_layer(v2file, text_cfg, L))
        else:
            layers.append(build_quantized_layer(v2file, text_cfg, L))

    # Persistent state, threaded across the token sequence and kept
    # INDEPENDENT per layer (the micro-stack state-ownership hard gate).
    dn = layers[0].linear_attn
    conv_dim = dn.conv_dim
    conv_states = [torch.zeros(1, conv_dim, 3, dtype=BF16) for _ in range(3)]
    rec_states = [None, None, None]  # None = zero start
    kv_cache = None  # (Kc, Vc) for the full-attention layer

    torch.manual_seed(input_seed)
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    os.makedirs(out_dir, exist_ok=True)
    for t in tokens:
        # ONE seeded draw per token = the micro-stack input fed to layer 0.
        x = (0.05 * torch.randn(1, 1, H, dtype=torch.float32)).to(BF16)
        for L in range(4):
            layer = layers[L]
            if v2file.config.is_linear_attention(L):
                stages, conv_states[L], rec_after, conv_before, rec_before = (
                    forward_step_deltanet(layer, x, t, conv_states[L],
                                          rec_states[L]))
                rec_states[L] = rec_after
                specs = stage_tensors_deltanet(
                    stages, conv_before, conv_states[L], rec_before, rec_after)
            else:
                stages, kv_cache = forward_step(layer, rope, x, t, kv_cache)
                specs = [(name, dims, v2.DT_BF16, blob)
                         for name, dims, blob
                         in stage_tensors(stages, kv_cache[0], kv_cache[1], t)]
            tensors = [v2.V2Tensor(name, dt, dims, blob)
                       for name, dims, dt, blob in specs]
            g = gv2.GoldenV2File(v2file.config, t, L, input_seed, tensors)
            blob = g.to_bytes()
            # Round-trip + expected-set validation (same as generate()).
            rt = gv2.GoldenV2File.from_bytes(blob)
            expected = gv2.golden_tensor_expectations(v2file.config, t, L)
            if len(rt.tensors) != len(expected):
                raise SystemExit(
                    f"microstack: tensor count mismatch (t={t}, L={L})")
            for (ename, edims, edtype), tt in zip(expected, rt.tensors):
                if (tt.name != ename or tuple(tt.dims) != edims
                        or tt.dtype != edtype):
                    raise SystemExit(
                        f"microstack: tensor {tt.name!r} mismatch "
                        f"(t={t}, L={L})")
            if bytes(blob) != rt.to_bytes():
                raise SystemExit(f"microstack: bytes not idempotent (t={t})")
            outp = f"{out_prefix}_L{L}_p{t}.cudalm"
            gv2.write_golden_v2(outp, g)
            x = stages["final_output"]  # chain layer L's output to layer L+1
    print(f"wrote micro-stack goldens: {len(tokens)} token(s) x 4 layers "
          f"under {out_prefix!r} (per-(token,layer) CUDLMG02)")
    return 0


# ---------------------------------------------------------------------------
# Selftest: no checkpoint. Synthetic weights through the shared quantizer;
# exercises the reference pipeline, invariants, container round-trip, and
# determinism.
# ---------------------------------------------------------------------------
def selftest_impl() -> int:
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig)

    torch.manual_seed(20260209)
    cfg = v2.Qwen35Config.qwen35_08b()
    H = cfg.hidden_size
    qo = cfg.n_heads * cfg.head_dim
    kvo = cfg.n_kv_heads * cfg.head_dim
    inter = cfg.intermediate_size

    tensors = []

    def synth_w4a16(name: str, N: int, K: int):
        W = (0.05 * torch.randn(N, K, dtype=torch.float32)).to(BF16)
        packed, scale16, _q, _s = quantize_w4a16(W)
        tensors.append(v2.V2Tensor(f"{name}.weight", v2.DT_INT4_PACKED,
                                   (N, K // 2), bytes_of(packed)))
        tensors.append(v2.V2Tensor(f"{name}.scale", v2.DT_FP16_SCALE,
                                   (N, K // 128), bytes_of(scale16)))

    def synth_bf16(name: str, shape):
        t = (0.05 * torch.randn(*shape, dtype=torch.float32)).to(BF16)
        tensors.append(v2.V2Tensor(name, v2.DT_BF16, tuple(shape), bytes_of(t)))

    # The converter emits layer tensors under a "layers.{i}." prefix; the
    # oracle loader expects the same, so the synthetic file mirrors it.
    LP = "layers.3."
    synth_w4a16(LP + "self_attn.q_proj", 2 * qo, H)
    synth_w4a16(LP + "self_attn.k_proj", kvo, H)
    synth_w4a16(LP + "self_attn.v_proj", kvo, H)
    synth_w4a16(LP + "self_attn.o_proj", H, qo)
    synth_w4a16(LP + "mlp.gate_proj", inter, H)
    synth_w4a16(LP + "mlp.up_proj", inter, H)
    synth_w4a16(LP + "mlp.down_proj", H, inter)
    synth_bf16(LP + "input_layernorm.weight", (H,))
    synth_bf16(LP + "post_attention_layernorm.weight", (H,))
    synth_bf16(LP + "self_attn.q_norm.weight", (cfg.head_dim,))
    synth_bf16(LP + "self_attn.k_norm.weight", (cfg.head_dim,))

    tmpdir = tempfile.mkdtemp(prefix="cudalm_qwen35_golden_selftest_")
    cudalm_path = os.path.join(tmpdir, "synthetic_l3.cudalm")
    v2f = v2.V2File(cfg, {"arch": ARCH_ID}, tensors)
    v2.write_v2(cudalm_path, v2f)
    v2file = v2.read_v2(cudalm_path)

    # No checkpoint needed for the selftest: build the oracle from the
    # synthetic v2 file + a text config mirroring the pinned 0.8B fields.
    text_cfg = Qwen3_5TextConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        rms_norm_eps=cfg.eps,
        max_position_embeddings=cfg.max_seq_len,
        full_attention_interval=cfg.full_attention_interval,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": cfg.rope_theta,
            "partial_rotary_factor": cfg.partial_rotary_factor,
            "mrope_section": list(cfg.mrope_section),
            "mrope_interleaved": True,
        },
        linear_num_key_heads=cfg.lin_num_k_heads,
        linear_num_value_heads=cfg.lin_num_v_heads,
        linear_key_head_dim=cfg.lin_key_head_dim,
        linear_value_head_dim=cfg.lin_value_head_dim,
        linear_conv_kernel_dim=cfg.lin_conv_kernel_dim,
    )

    # _P must be the pinned module (asserted separately against a real
    # checkpoint in the full test; here we only need the op functions).
    global _P
    if _P is None:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        _P = mq
        if sha256_file(os.path.abspath(mq.__file__)) != MODELING_SHA256:
            print("selftest: SKIP (installed transformers is not the "
                  "pinned commit; the real-checkpoint test covers this)",
                  file=sys.stderr)
            return 77

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding)
    rope = Qwen3_5TextRotaryEmbedding(text_cfg)
    layer = build_quantized_layer(v2file, text_cfg, 3)

    for position, seed in ((0, 20260209), (3, 20260209)):
        stages, Kc, Vc = run_reference(layer, rope, position, seed, H)
        check_invariants(stages, Kc, Vc, position)
        tensors_g = []
        for name, dims, blob in stage_tensors(stages, Kc, Vc, position):
            tensors_g.append(v2.V2Tensor(name, v2.DT_BF16, dims, blob))
        g = gv2.GoldenV2File(cfg, position, 3, seed, tensors_g)
        b1 = g.to_bytes()
        rt = gv2.GoldenV2File.from_bytes(b1)
        if rt.to_bytes() != b1:
            raise SystemExit("selftest: golden round-trip not idempotent")
        expected = gv2.golden_tensor_expectations(cfg, position, 3)
        if [(t.name, tuple(t.dims), t.dtype) for t in rt.tensors] != expected:
            raise SystemExit("selftest: golden tensor set mismatch")
        # Determinism: regenerate, bytes must be identical.
        stages2, Kc2, Vc2 = run_reference(layer, rope, position, seed, H)
        tensors_g2 = []
        for name, dims, blob in stage_tensors(stages2, Kc2, Vc2, position):
            tensors_g2.append(v2.V2Tensor(name, v2.DT_BF16, dims, blob))
        if gv2.GoldenV2File(cfg, position, 3, seed, tensors_g2).to_bytes() \
                != b1:
            raise SystemExit("selftest: golden generation not deterministic")
        outp = os.path.join(tmpdir, f"golden_p{position}.cudalm")
        gv2.write_golden_v2(outp, g)
        if gv2.read_golden_v2(outp).to_bytes() != b1:
            raise SystemExit("selftest: file round-trip mismatch")
    print("[selftest] qwen35 golden generator OK (p0+p3, round-trip, "
          "determinism, invariants)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cudalm")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--position", type=int, default=0)
    ap.add_argument("--input-seed", type=int, default=20260209)
    ap.add_argument("--fidelity-report")
    ap.add_argument("--state-seed", type=int, default=None,
                    help="Phase C scenario C: seed a deterministic non-zero "
                         "conv+recurrent previous state (DeltaNet only)")
    ap.add_argument("--microstack-prefix", default=None,
                    help="Phase D: emit per-(token,layer) CUDLMG02 goldens "
                         "under <prefix>_L{layer}_p{pos}.cudalm for the "
                         "4-layer hybrid micro-stack (layers 0-3)")
    ap.add_argument("--tokens", default="0",
                    help="Phase D micro-stack: strictly consecutive 0-anchored "
                         "0-based token sequence: 0 / 0,1 / 0,1,2 / ... "
                         "(gaps, duplicates, non-0 start and out-of-order are "
                         "rejected). e.g. 0 (scenario A) or 0,1,2 (scenario B)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest_impl())
    if a.microstack_prefix:
        if not (a.cudalm and a.checkpoint_dir):
            ap.error("--cudalm and --checkpoint-dir are required for "
                     "--microstack-prefix")
        try:
            tokens = token_seq.parse_token_sequence(a.tokens)
        except ValueError as e:
            ap.error(str(e))
        sys.exit(generate_microstack(a.cudalm, a.checkpoint_dir,
                                     a.microstack_prefix, tokens,
                                     a.input_seed))
    if not (a.cudalm and a.checkpoint_dir and a.out):
        ap.error("--cudalm, --checkpoint-dir and --out are required "
                 "(or use --selftest)")
    sys.exit(generate(a.cudalm, a.checkpoint_dir, a.layer, a.out,
                      a.position, a.input_seed, a.fidelity_report,
                      a.state_seed))
