#!/usr/bin/env python3
"""CUDALM v0.2 — .cudalm v2 ingestion verifier (Python side).

Checks a converted .cudalm v2 file against the source safetensors checkpoint:

  1. Provenance: config blob == text_config of the pinned config.json;
     metadata sha256s match recomputed sha256s of config.json and
     model.safetensors; arch/repo/revision match the pinned constants.
  2. bf16 pass-through tensors: payload bytes bit-exact vs the checkpoint.
  3. W4A16 tensors: unpacked q in [-7, 7]; per-group dequantization error
     |deq - w| <= 0.51 * scale_stored (round-half-to-even + fp16-scale
     storage bound; see w4a16_quant docstring); zero-scale groups have
     zero q and a zero original group.
  3b. FP16 companion scales: BIT-EXACT binary contract
     stored_scale == fp16(amax(original_group) / 7), recomputed from the
     checkpoint's .weight twin (verifies the W4A16 scale contract directly,
     not just the dequant result).
  4. Completeness: for every requested layer the expected tensor-name set
     (per layer type) is exactly present; model norm present.

Fidelity numbers (max_abs/rmse/cosine, original vs dequantized) are REPORTED
only — the runtime-correctness gate (docs §11) compares the C++ runtime
against a quantized-weights reference, not against the original checkpoint.

Usage:
  python3 tools/verify_qwen35_ingestion.py --cudalm FILE \
      --checkpoint-dir DIR [--layers 0,1,2,3] [--report out.json]
Exit status 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import cudalm_v2 as v2  # noqa: E402
from w4a16_quant import dequant_reference, fidelity_stats, unpack_int4  # noqa: E402
from convert_qwen35 import (  # noqa: E402
    ARCH_ID, CHECKPOINT_SHA256, CONFIG_SHA256, LANG_PREFIX,
    MODEL_REPO, MODEL_REVISION,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fp32_sum_f32(data: bytes) -> float:
    f = np.frombuffer(data, dtype=np.float32)
    return float(f.sum(dtype=np.float32))


def bf16_sum_f32(data: bytes) -> float:
    """Sum of the bf16 values as fp32 (bit trick: bf16 << 16 == fp32 bits)."""
    u = np.frombuffer(data, dtype="uint16").astype(np.uint32) << 16
    f = np.ascontiguousarray(u).view(np.float32)
    return float(f.sum(dtype=np.float32))


def expected_names(conf: v2.Qwen35Config, i: int):
    L = f"layers.{i}."
    G = conf.group_size
    H = conf.hidden_size
    names = {
        L + "input_layernorm.weight", L + "post_attention_layernorm.weight",
        "norm.weight",
    }
    if conf.is_full_attention(i):
        qo, kvo, oin = conf.q_proj_out, conf.kv_proj_out, conf.o_proj_in
        names.update([
            L + "self_attn.q_proj.weight", L + "self_attn.q_proj.scale",
            L + "self_attn.k_proj.weight", L + "self_attn.k_proj.scale",
            L + "self_attn.v_proj.weight", L + "self_attn.v_proj.scale",
            L + "self_attn.o_proj.weight", L + "self_attn.o_proj.scale",
            L + "self_attn.q_norm.weight", L + "self_attn.k_norm.weight",
        ])
    else:
        ck, cv, cd = conf.linear_key_dim, conf.linear_value_dim, conf.linear_conv_dim
        names.update([
            L + "linear_attn.in_proj_qkv.weight",
            L + "linear_attn.in_proj_qkv.scale",
            L + "linear_attn.in_proj_z.weight", L + "linear_attn.in_proj_z.scale",
            L + "linear_attn.in_proj_b.weight", L + "linear_attn.in_proj_b.scale",
            L + "linear_attn.in_proj_a.weight", L + "linear_attn.in_proj_a.scale",
            L + "linear_attn.out_proj.weight", L + "linear_attn.out_proj.scale",
            L + "linear_attn.conv1d.weight", L + "linear_attn.dt_bias",
            L + "linear_attn.A_log", L + "linear_attn.norm.weight",
        ])
    names.update([
        L + "mlp.gate_proj.weight", L + "mlp.gate_proj.scale",
        L + "mlp.up_proj.weight", L + "mlp.up_proj.scale",
        L + "mlp.down_proj.weight", L + "mlp.down_proj.scale",
    ])
    return names


def verify(cudalm_path: str, checkpoint_dir: str, layers, report_path=None,
           sidecar_path=None):
    errors: list[str] = []
    obj = v2.read_v2(cudalm_path)
    cfg_path = os.path.join(checkpoint_dir, "raw", "config.json")
    st_path = os.path.join(checkpoint_dir, "model.safetensors")

    conf = obj.config
    pinned = v2.Qwen35Config.qwen35_08b()
    if conf != pinned:
        errors.append("config blob differs from the pinned 0.8B contract")

    # ---- provenance metadata ------------------------------------------------
    cfg = json.load(open(cfg_path))
    checks = {
        v2.META_ARCH: ARCH_ID,
        v2.META_MODEL_REPO: MODEL_REPO,
        v2.META_MODEL_REVISION: MODEL_REVISION,
        v2.META_CONFIG_SHA256: CONFIG_SHA256,
        v2.META_CHECKPOINT_SHA256: CHECKPOINT_SHA256,
    }
    for k, want in checks.items():
        got = obj.meta(k)
        if got != want:
            errors.append(f"metadata {k!r}: got {got!r}, want {want!r}")
    if sha256_file(cfg_path) != CONFIG_SHA256:
        errors.append("source config.json sha256 no longer matches the pin")
    if sha256_file(st_path) != CHECKPOINT_SHA256:
        errors.append("source model.safetensors sha256 no longer matches the pin")

    # ---- tensor set ----------------------------------------------------------
    present = {t.name for t in obj.tensors}
    want = {"norm.weight"}
    for i in layers:
        want |= expected_names(conf, i)
    missing = want - present
    extra = present - want
    if missing:
        errors.append(f"missing tensors: {sorted(missing)[:8]}...")
    if extra:
        errors.append(f"unexpected tensors: {sorted(extra)[:8]}...")

    # ---- value checks ---------------------------------------------------------
    stats = {
        "bf16_exact": 0, "fp32_exact": 0, "w4a16_checked": 0,
        "fp16_scale_exact": 0,
    }
    fidelity = []
    with safe_open(st_path, framework="pt") as f:
        for t in obj.tensors:
            if t.name not in want:
                continue
            if t.dtype == v2.DT_FP16_SCALE:
                # Exact W4A16 scale binary contract:
                #     stored_scale == fp16(amax(original_group) / 7)
                # The scale tensor is converter-synthetic (absent from the
                # checkpoint), so recompute it from the checkpoint's .weight
                # twin and require a BIT-EXACT match. This pins the stored
                # scale bits directly; the INT4 dequant bound below only
                # bounds the error and does not verify the scale contract.
                N, Gc = t.dims
                G = conf.group_size
                weight_name = t.name[: -len(".scale")] + ".weight"
                try:
                    wsrc = f.get_tensor(LANG_PREFIX + weight_name)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{t.name}: missing checkpoint weight twin "
                                  f"{weight_name} ({e})")
                    continue
                w = wsrc.float().view(N, Gc, G)
                amax = w.abs().amax(dim=2)
                scale_want = (amax / 7.0).to(torch.float16)
                scale_got = torch.from_numpy(
                    np.frombuffer(t.data, dtype="uint8").copy()
                ).view(torch.float16).view(N, Gc)
                if not bool(torch.equal(scale_got, scale_want)):
                    d = (scale_got.float() - scale_want.float()).abs()
                    idx = d.argmax()
                    errors.append(
                        f"{t.name}: stored scale != "
                        f"fp16(amax(original_group)/7) "
                        f"(max diff {d.flatten()[idx].item():.6g})")
                stats["fp16_scale_exact"] += 1
                continue
            src_name = LANG_PREFIX + t.name
            try:
                src = f.get_tensor(src_name)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{t.name}: not in checkpoint ({e})")
                continue

            if t.dtype == v2.DT_BF16:
                want_bytes = src.contiguous().view(torch.uint8).numpy().tobytes()
                if t.data != want_bytes:
                    errors.append(f"{t.name}: bf16 payload not bit-exact")
                else:
                    stats["bf16_exact"] += 1
            elif t.dtype == v2.DT_FP32:
                want_bytes = src.contiguous().view(torch.uint8).numpy().tobytes()
                if t.data != want_bytes:
                    errors.append(f"{t.name}: fp32 payload not bit-exact")
                else:
                    stats["fp32_exact"] += 1
            elif t.dtype == v2.DT_INT4_PACKED:
                N, K2 = t.dims
                K = 2 * K2
                packed = torch.from_numpy(
                    np.frombuffer(t.data, dtype="uint8").copy().reshape(t.dims))
                scale_t = obj.find(t.name.replace(".weight", ".scale"))
                if scale_t is None:
                    errors.append(f"{t.name}: missing companion scale")
                    continue
                scale = torch.from_numpy(
                    np.frombuffer(scale_t.data, dtype="uint8").copy()
                ).view(torch.float16).view(*scale_t.dims)
                q = unpack_int4(packed)
                if not (q.min().item() >= -7 and q.max().item() <= 7):
                    errors.append(f"{t.name}: quant codes out of [-7,7]")
                G = conf.group_size
                deq = dequant_reference(packed, scale)
                w = src.float()
                err = (deq - w).abs().view(N, K // G, G)
                per_group = err.amax(dim=2)
                s = scale.float()  # (N, K/G): one scale per group
                # Zero stored-scale groups are exempt here (bound degenerates
                # to 0); they get the dedicated underflow/zero check below.
                bad = (per_group > 0.51 * s) & (s > 0)
                if bool(bad.any()):
                    idx = bad.nonzero()[0]
                    errors.append(
                        f"{t.name}: dequant bound violated at row {int(idx[0])} "
                        f"group {int(idx[1])}: err={per_group[idx[0], idx[1]]:.6g} "
                        f"bound={0.51 * s[idx[0], idx[1]]:.6g}")
                    continue
                # zero stored-scale groups: either an all-zero original group
                # (dequant exactly 0, error 0) or an fp16-underflow group:
                # stored scale = fp16(amax/7) == 0  <=>  amax < 7*2^-25
                # (round-half-even), so the whole group dequantizes to exactly
                # 0 with absolute error < 7*2^-25, far below bf16 resolution.
                if bool((s == 0).any()):
                    underflow_limit = 7.0 * (2.0 ** -25)
                    zero_mask = (s == 0).nonzero()
                    for r_, g_ in zero_mask.tolist():
                        amax_g = float(
                            w[r_, g_ * G:(g_ + 1) * G].abs().max())
                        if amax_g > underflow_limit:
                            errors.append(
                                f"{t.name}: zero-scale group with amax "
                                f"{amax_g:.6g} above fp16 underflow limit "
                                f"{underflow_limit:.6g} (row {r_}, group {g_})")
                stats["w4a16_checked"] += 1
                fidelity.append({
                    "name": t.name, **fidelity_stats(src, packed, scale)})
            else:
                # FP16_SCALE is handled above; any other dtype here is a bug.
                errors.append(
                    f"{t.name}: unexpected dtype {t.dtype} in value checks")

    # Sidecar for the C++ side (tests/cpu/test_qwen35_ingestion_real.cpp):
    # per tensor (file order): name, byte size, fp32 sum of the bf16/fp32
    # values ("-1" for int4/scale tensors). The C++ test recomputes these
    # over the payloads it parsed (dtype-aware), cross-checking the container
    # end to end.
    if sidecar_path:
        with open(sidecar_path, "w") as fp:
            for t in obj.tensors:
                if t.dtype == v2.DT_BF16:
                    s = f"{bf16_sum_f32(t.data):.9g}"
                elif t.dtype == v2.DT_FP32:
                    s = f"{fp32_sum_f32(t.data):.9g}"
                else:
                    s = "-1"
                fp.write(f"{t.name}\t{len(t.data)}\t{s}\n")

    report = {
        "cudalm": os.path.abspath(cudalm_path),
        "layers": layers,
        "stats": stats,
        "errors": errors,
        "fidelity_report_only": fidelity,
    }
    if report_path:
        with open(report_path, "w") as fp:
            json.dump(report, fp, indent=1)
    if errors:
        print(f"VERIFY FAILED: {len(errors)} error(s)", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"VERIFY OK: {stats['bf16_exact']} bf16 tensors bit-exact, "
          f"{stats['w4a16_checked']} W4A16 tensors within dequant bound, "
          f"{stats['fp16_scale_exact']} FP16 scales bit-exact "
          f"(== fp16(amax(original_group)/7))")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cudalm", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--layers")
    ap.add_argument("--report")
    ap.add_argument("--sidecar")
    a = ap.parse_args()
    if a.layers:
        layers = [int(x) for x in a.layers.split(",")]
    else:
        # Infer the layer set from the file itself.
        obj = v2.read_v2(a.cudalm)
        layers = sorted({int(t.name.split(".")[1]) for t in obj.tensors
                         if t.name.startswith("layers.")})
    sys.exit(verify(a.cudalm, a.checkpoint_dir, layers, a.report, a.sidecar))
