#!/usr/bin/env python3
"""CUDALM v0.2 — Qwen3.5-0.8B-Base checkpoint -> .cudalm v2 converter.

Offline tool (torch + safetensors; the C++ runtime never imports this).
Reads the pinned official checkpoint (bf16 source) and emits a CUDLMW02
container whose tensor table follows docs/qwen35_architecture.md §4:

  * W4A16 G=128 (int4 packed + fp16 scale) for every linear projection:
      full-attention: q/k/v/o_proj
      gated DeltaNet: in_proj_qkv/z/b/a, out_proj
      both:           mlp gate/up/down_proj
  * bf16 pass-through (byte-exact from the checkpoint):
      layernorms, q/k_norm, conv1d.weight, dt_bias, model norm.
  * fp32 pass-through (byte-exact; stored fp32 in the official checkpoint):
      A_log, linear_attn.norm.weight (per-head gated norm).
  * Skipped (out of v0.2 scope): vision tower, MTP module, embed_tokens
    (tie_word_embeddings; no embedding/LM head in v0.2).

All provenance (repo, revision, sha256 of config + checkpoint, transformers
pin) is written into the container metadata; the config blob is built from
the checkpoint's text_config and asserted equal to the pinned 0.8B contract.

Usage:
  python3 tools/convert_qwen35.py \
      --checkpoint-dir /root/models/Qwen3.5-0.8B-Base \
      --out data/qwen35_08b.cudalm [--layers 0,1,2,3] [--manifest m.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import cudalm_v2 as v2  # noqa: E402
from w4a16_quant import fidelity_stats, quantize_w4a16  # noqa: E402

# --- Pinned provenance (docs/qwen35_architecture.md §1) ----------------------
MODEL_REPO = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
CONFIG_SHA256 = "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
CHECKPOINT_SHA256 = (
    "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c")
TRANSFORMERS_COMMIT = "fc9137225880a9d03f130634c20f9dbe36a7b8bf"
ARCH_ID = "qwen3.5-text"

LANG_PREFIX = "model.language_model."


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


def config_from_checkpoint(cfg: dict) -> v2.Qwen35Config:
    tc = cfg["text_config"]
    rp = tc["rope_parameters"]
    return v2.Qwen35Config(
        hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        intermediate_size=tc["intermediate_size"],
        vocab_size=tc["vocab_size"],
        n_heads=tc["num_attention_heads"],
        n_kv_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        lin_num_k_heads=tc["linear_num_key_heads"],
        lin_num_v_heads=tc["linear_num_value_heads"],
        lin_key_head_dim=tc["linear_key_head_dim"],
        lin_value_head_dim=tc["linear_value_head_dim"],
        lin_conv_kernel_dim=tc["linear_conv_kernel_dim"],
        full_attention_interval=tc["full_attention_interval"],
        group_size=128,  # W4A16 G=128: runtime contract, not a checkpoint field
        max_seq_len=tc["max_position_embeddings"],
        eps=tc["rms_norm_eps"],
        rope_theta=rp["rope_theta"],
        partial_rotary_factor=rp["partial_rotary_factor"],
        mrope_section=tuple(rp["mrope_section"]),
    )


def convert(checkpoint_dir: str, out_path: str, layers, manifest_path=None,
            full_model=False):
    cfg_path = os.path.join(checkpoint_dir, "raw", "config.json")
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not (os.path.isfile(cfg_path) and os.path.isfile(st_path)):
        raise SystemExit(f"checkpoint dir {checkpoint_dir!r} incomplete "
                         f"(need raw/config.json + model.safetensors)")

    cfg = json.load(open(cfg_path))
    if sha256_file(cfg_path) != CONFIG_SHA256:
        raise SystemExit("config.json sha256 mismatch: not the pinned revision")
    if sha256_file(st_path) != CHECKPOINT_SHA256:
        raise SystemExit("model.safetensors sha256 mismatch: not the pinned "
                         "checkpoint")

    conf = config_from_checkpoint(cfg)
    pinned = v2.Qwen35Config.qwen35_08b()
    if conf != pinned:
        raise SystemExit("checkpoint text_config deviates from the pinned "
                         "Qwen3.5-0.8B contract (docs/qwen35_architecture.md "
                         "§2); refusing to convert")
    if not conf.valid():
        raise SystemExit("Qwen35Config failed validation")

    layer_types = cfg["text_config"]["layer_types"]
    n_layers = conf.num_hidden_layers
    if len(layer_types) != n_layers:
        raise SystemExit("layer_types length mismatch")
    for i in range(n_layers):
        expected = "full_attention" if conf.is_full_attention(i) else "linear_attention"
        if layer_types[i] != expected:
            raise SystemExit(f"layer {i}: config says {layer_types[i]!r}, "
                             f"schedule says {expected!r}")

    if full_model:
        # Full model: convert every decoder layer (0..n_layers-1) so the file
        # can be loaded whole by the v0.3 Qwen35Model runtime.
        layers = list(range(n_layers))
    if layers is None:
        layers = list(range(n_layers))
    layers = sorted(set(layers))
    for i in layers:
        if not 0 <= i < n_layers:
            raise SystemExit(f"layer {i} out of range")

    tensors = []
    manifest = {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "config_sha256": CONFIG_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "layers": layers,
        "full_model": full_model,
        # embed_tokens is converted (and the LM head is tied to it) in the full
        # model; visual + mtp are always out of v0.2/v0.3 text scope.
        "skipped": (["visual", "mtp"] if full_model
                    else ["visual", "mtp", "embed_tokens"]),
        "tensors": [],
        "fidelity_report_only": [],
    }
    c = conf
    G = c.group_size

    def add(name, dtype, dims, data, fidelity=None):
        tensors.append(v2.V2Tensor(name, dtype, dims, data))
        rec = {"name": name, "dtype": v2.DTYPE_NAMES[dtype], "dims": list(dims),
               "bytes": len(data)}
        if fidelity:
            rec.update(fidelity)
        manifest["tensors"].append(rec)
        if fidelity:
            manifest["fidelity_report_only"].append(rec)

    def bf16(name, t):
        add(name, v2.DT_BF16, tuple(t.shape), bytes_of(t))

    def fp32(name, t):
        if t.dtype != torch.float32:
            raise SystemExit(f"{name}: expected fp32 source, got {t.dtype}")
        add(name, v2.DT_FP32, tuple(t.shape), bytes_of(t))

    def w4(name, t):
        if t.dtype != torch.bfloat16:
            raise SystemExit(f"{name}: expected bf16 source, got {t.dtype}")
        N, K = t.shape
        packed, scale16, _q, _s32 = quantize_w4a16(t)
        packed_b, scale_b = bytes_of(packed), bytes_of(scale16)
        add(name + ".weight", v2.DT_INT4_PACKED, (N, K // 2), packed_b)
        add(name + ".scale", v2.DT_FP16_SCALE, (N, K // G), scale_b,
            fidelity=fidelity_stats(t, packed, scale16))

    with safe_open(st_path, framework="pt") as f:
        for i in layers:
            L = LANG_PREFIX + f"layers.{i}."
            kind = "fa" if conf.is_full_attention(i) else "dn"
            bf16(f"layers.{i}.input_layernorm.weight", f.get_tensor(L + "input_layernorm.weight"))
            if kind == "fa":
                w4(f"layers.{i}.self_attn.q_proj", f.get_tensor(L + "self_attn.q_proj.weight"))
                w4(f"layers.{i}.self_attn.k_proj", f.get_tensor(L + "self_attn.k_proj.weight"))
                w4(f"layers.{i}.self_attn.v_proj", f.get_tensor(L + "self_attn.v_proj.weight"))
                w4(f"layers.{i}.self_attn.o_proj", f.get_tensor(L + "self_attn.o_proj.weight"))
                bf16(f"layers.{i}.self_attn.q_norm.weight", f.get_tensor(L + "self_attn.q_norm.weight"))
                bf16(f"layers.{i}.self_attn.k_norm.weight", f.get_tensor(L + "self_attn.k_norm.weight"))
            else:
                w4(f"layers.{i}.linear_attn.in_proj_qkv", f.get_tensor(L + "linear_attn.in_proj_qkv.weight"))
                w4(f"layers.{i}.linear_attn.in_proj_z", f.get_tensor(L + "linear_attn.in_proj_z.weight"))
                w4(f"layers.{i}.linear_attn.in_proj_b", f.get_tensor(L + "linear_attn.in_proj_b.weight"))
                w4(f"layers.{i}.linear_attn.in_proj_a", f.get_tensor(L + "linear_attn.in_proj_a.weight"))
                w4(f"layers.{i}.linear_attn.out_proj", f.get_tensor(L + "linear_attn.out_proj.weight"))
                bf16(f"layers.{i}.linear_attn.conv1d.weight", f.get_tensor(L + "linear_attn.conv1d.weight"))
                bf16(f"layers.{i}.linear_attn.dt_bias", f.get_tensor(L + "linear_attn.dt_bias"))
                fp32(f"layers.{i}.linear_attn.A_log", f.get_tensor(L + "linear_attn.A_log"))
                fp32(f"layers.{i}.linear_attn.norm.weight", f.get_tensor(L + "linear_attn.norm.weight"))
            bf16(f"layers.{i}.post_attention_layernorm.weight", f.get_tensor(L + "post_attention_layernorm.weight"))
            w4(f"layers.{i}.mlp.gate_proj", f.get_tensor(L + "mlp.gate_proj.weight"))
            w4(f"layers.{i}.mlp.up_proj", f.get_tensor(L + "mlp.up_proj.weight"))
            w4(f"layers.{i}.mlp.down_proj", f.get_tensor(L + "mlp.down_proj.weight"))
        bf16("norm.weight", f.get_tensor(LANG_PREFIX + "norm.weight"))
        if full_model:
            # Embedding (v0.3 full model). The LM head is TIED to this tensor
            # (pinned 0.8B: tie_word_embeddings=true), so there is NO separate
            # lm_head tensor in the checkpoint; the tie is recorded in metadata
            # below and the loader aliases the embedding as the LM head.
            bf16("embed_tokens.weight",
                 f.get_tensor(LANG_PREFIX + "embed_tokens.weight"))

    metadata = {
        v2.META_ARCH: ARCH_ID,
        v2.META_MODEL_REPO: MODEL_REPO,
        v2.META_MODEL_REVISION: MODEL_REVISION,
        v2.META_CONFIG_SHA256: CONFIG_SHA256,
        v2.META_CHECKPOINT_SHA256: CHECKPOINT_SHA256,
        v2.META_TRANSFORMERS_COMMIT: TRANSFORMERS_COMMIT,
        v2.META_TRANSFORMERS_VERSION: f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_COMMIT}",
        v2.META_SOURCE_DTYPE: "bf16",
        v2.META_GENERATOR: "tools/convert_qwen35.py (CUDALM v0.2)",
    }
    if full_model:
        # Record the LM-head weight tying from the pinned config. The pinned
        # Qwen3.5-0.8B ties lm_head to the embedding (tie_word_embeddings=true,
        # source _tied_weights_keys), so the checkpoint has no separate lm_head
        # tensor; the loader aliases the embedding as the LM head.
        tie = bool(cfg.get("tie_word_embeddings",
                           cfg["text_config"].get("tie_word_embeddings", False)))
        metadata[v2.META_TIE_WORD_EMBEDDINGS] = "true" if tie else "false"
    obj = v2.V2File(conf, metadata, tensors)
    raw = obj.to_bytes()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(raw)

    manifest["out_file"] = os.path.abspath(out_path)
    manifest["file_bytes"] = len(raw)
    manifest["num_tensors"] = len(tensors)
    if manifest_path:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"wrote {out_path}: {len(raw)} bytes, {len(tensors)} tensors, "
          f"layers={layers}")
    return manifest


def selftest():
    """Python-side contract selftest (no checkpoint needed): container
    round-trip with a tiny synthetic layer + quantizer invariants."""
    import numpy as np

    from cudalm_v2 import V2File, V2Tensor, Qwen35Config, write_v2, read_v2

    conf = Qwen35Config.qwen35_08b()
    rng = np.random.default_rng(1234)
    tensors = []
    # One FA layer (i=3) tensor set, tiny random payloads shaped per contract.
    def t(name, dtype, dims, data):
        tensors.append(V2Tensor(name, dtype, dims, data))

    for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        t(f"layers.3.{n}", v2.DT_BF16, (conf.hidden_size,),
          _bf16_bytes(rng, conf.hidden_size))
    t("norm.weight", v2.DT_BF16, (conf.hidden_size,), _bf16_bytes(rng, conf.hidden_size))
    for name, N in (("self_attn.q_proj", 4096), ("self_attn.k_proj", 512),
                    ("self_attn.v_proj", 512), ("self_attn.o_proj", 1024)):
        K = conf.hidden_size if "o_proj" not in name else conf.o_proj_in
        packed = rng.integers(0, 256, size=(N, K // 2), dtype="uint8").tobytes()
        scale = (rng.standard_normal((N, K // 128)) * 0.01).astype("float16").tobytes()
        t(f"layers.3.{name}.weight", v2.DT_INT4_PACKED, (N, K // 2), packed)
        t(f"layers.3.{name}.scale", v2.DT_FP16_SCALE, (N, K // 128), scale)
    for n in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
        t(f"layers.3.{n}", v2.DT_BF16, (conf.head_dim,), _bf16_bytes(rng, conf.head_dim))
    for name, N, K in (("mlp.gate_proj", 3584, 1024), ("mlp.up_proj", 3584, 1024),
                       ("mlp.down_proj", 1024, 3584)):
        packed = rng.integers(0, 256, size=(N, K // 2), dtype="uint8").tobytes()
        scale = (rng.standard_normal((N, K // 128)) * 0.01).astype("float16").tobytes()
        t(f"layers.3.{name}.weight", v2.DT_INT4_PACKED, (N, K // 2), packed)
        t(f"layers.3.{name}.scale", v2.DT_FP16_SCALE, (N, K // 128), scale)

    obj = V2File(conf, {v2.META_ARCH: ARCH_ID}, tensors)
    raw = obj.to_bytes()
    back = V2File.from_bytes(raw)
    assert back.config == conf
    assert len(back.tensors) == len(tensors)
    for a, b in zip(tensors, back.tensors):
        assert a.name == b.name and a.dtype == b.dtype
        assert a.dims == b.dims and a.data == b.data
    assert back.meta(v2.META_ARCH) == ARCH_ID

    # Quantizer invariant on a bf16 tensor: |deq - w| <= 0.51 * scale_stored.
    w = torch.tensor(rng.standard_normal((64, 256)).astype("float32"),
                     dtype=torch.float32).bfloat16()
    packed, scale16, q, _ = quantize_w4a16(w)
    from w4a16_quant import dequant_reference
    deq = dequant_reference(packed, scale16)
    err = (deq - w.float()).abs().view(64, 2, 128).amax(dim=2)
    bound = 0.51 * scale16.float()
    assert bool((err <= bound).all()), "quantization bound violated"
    assert int(q.max()) <= 7 and int(q.min()) >= -7
    print("convert_qwen35 selftest OK")


def _bf16_bytes(rng, n) -> bytes:
    import numpy as np
    # fp32 -> bf16 via round-to-nearest of the 16-bit mantissa (numpy has no
    # bf16); only used for synthetic selftest payloads.
    x = rng.standard_normal(n).astype("float32") * 0.02
    u = x.view(np.uint32)
    r = (u >> 16) & 1
    u_bf = (u + 0x7FFF + r) & 0xFFFF0000
    return (u_bf >> 16).astype("uint16").tobytes()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--out")
    ap.add_argument("--layers", help="comma-separated layer indices "
                                     "(default: all 24)")
    ap.add_argument("--manifest", help="write a JSON manifest here")
    ap.add_argument("--full-model", action="store_true",
                    help="v0.3: emit the FULL model — all 24 decoder layers + "
                         "embed_tokens.weight + final norm.weight + the "
                         "tie_word_embeddings metadata (LM head is tied to the "
                         "embedding). Implies all layers; for the Qwen35Model "
                         "runtime.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        layers = None
        if a.layers:
            layers = [int(x) for x in a.layers.split(",")]
        if a.full_model and a.layers:
            ap.error("--full-model converts all 24 layers; do not also pass "
                     "--layers")
        convert(a.checkpoint_dir, a.out, layers, a.manifest,
                full_model=a.full_model)
