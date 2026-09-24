#!/usr/bin/env python3
"""Diagnostic: for a set of candidate prompts, run the pinned-quantized
oracle's serial-prefill + greedy-decode generation and report the
top-1 vs top-2 LOGIT GAP at every generation step. A prompt is "confident"
when the gap stays well above the runtime/oracle bf16-logits rounding (~0.13)
at every step — then the runtime's greedy argmax is guaranteed to match the
oracle's (no near-tie can flip it), so the golden's generated token sequence
is stable.

Usage:
    python diag_gen_gaps.py --cudalm <path> --checkpoint-dir <dir> \
        --prompt "15,16,17" --max 8 [--prompt "..." --max 8 ...]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cudalm_v2 as v2  # noqa: E402
import generate_qwen35_golden as g  # noqa: E402

BF16 = torch.bfloat16


def build(cudalm_path, ckpt_dir):
    g.import_pinned_transformers(ckpt_dir)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    g._P = mq
    v2file = v2.read_v2(cudalm_path)
    pinned = v2.Qwen35Config.qwen35_08b()
    raw_cfg = json.load(open(os.path.join(ckpt_dir, "raw", "config.json")))
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig)
    text_cfg = Qwen3_5TextConfig(**raw_cfg["text_config"])
    H = text_cfg.hidden_size
    V = text_cfg.vocab_size
    eps = text_cfg.rms_norm_eps
    rope = mq.Qwen3_5TextRotaryEmbedding(text_cfg)
    n_layers = pinned.num_hidden_layers
    layers = []
    for L in range(n_layers):
        if pinned.is_linear_attention(L):
            layers.append(g.build_quantized_deltanet_layer(v2file, text_cfg, L))
        else:
            layers.append(g.build_quantized_layer(v2file, text_cfg, L))
    embed = g.tensor_from_v2(v2file, "embed_tokens.weight", BF16)
    norm_w = g.tensor_from_v2(v2file, "norm.weight", BF16)
    norm_mod = mq.Qwen3_5RMSNorm(H, eps=eps)
    norm_mod.weight = torch.nn.Parameter(norm_w)
    conv_dim = layers[0].linear_attn.conv_dim \
        if pinned.is_linear_attention(0) else None
    return (pinned, n_layers, V, H, rope, layers, embed, norm_mod, conv_dim)


def fresh(pinned, n_layers, conv_dim):
    conv = [None] * n_layers
    rec = [None] * n_layers
    kv = [None] * n_layers
    for L in range(n_layers):
        if pinned.is_linear_attention(L):
            conv[L] = torch.zeros(1, conv_dim, 3, dtype=BF16)
    return conv, rec, kv


def gen_gap(model, prompt, max_new, eos):
    pinned, n_layers, V, H, rope, layers, embed, norm_mod, conv_dim = model
    conv, rec, kv = fresh(pinned, n_layers, conv_dim)

    def fwd(x, pos):
        for L in range(n_layers):
            layer = layers[L]
            if pinned.is_linear_attention(L):
                st, conv[L], ra, _, _ = g.forward_step_deltanet(
                    layer, x, pos, conv[L], rec[L])
                rec[L] = ra
            else:
                st, kv[L] = g.forward_step(layer, rope, x, pos, kv[L])
            x = st["final_output"]
        no = norm_mod(x).contiguous()
        return (no.reshape(1, H).float() @ embed.float().T).reshape(V).to(BF16)

    N = len(prompt)
    cur = None
    for pos in range(N):
        x = embed[prompt[pos]].contiguous().unsqueeze(0).unsqueeze(0)
        cur = fwd(x, pos)
    gaps = []
    toks = []
    nxt = int(torch.argmax(cur).item())
    for step in range(max_new):
        vals, idx = torch.topk(cur.float(), 2)
        gap = float(vals[0] - vals[1])
        top1, top2 = int(idx[0]), int(idx[1])
        gaps.append((step, gap, top1, top2))
        toks.append(nxt)
        if nxt == eos or step + 1 >= max_new:
            break
        x = embed[nxt].contiguous().unsqueeze(0).unsqueeze(0)
        cur = fwd(x, N + step)
        nxt = int(torch.argmax(cur).item())
    return gaps, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cudalm", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--max", action="append", type=int, default=[])
    ap.add_argument("--eos", type=int, default=248319)
    a = ap.parse_args()
    model = build(a.cudalm, a.checkpoint_dir)
    # pair --prompt with --max (max defaults to 8)
    prompts = []
    for i, p in enumerate(a.prompt):
        toks = [int(x) for x in p.split(",") if x.strip()]
        m = a.max[i] if i < len(a.max) else 8
        prompts.append((toks, m))
    for toks, m in prompts:
        gaps, gen = gen_gap(model, toks, m, a.eos)
        mingap = min(g[1] for g in gaps)
        worst = min(gaps, key=lambda g: g[1])
        status = "CONFIDENT" if mingap > 0.4 else "near-tie!"
        print(f"prompt={toks} max={m}")
        print(f"  generated={gen}")
        print(f"  min_gap={mingap:.6f}  ({status})  worst step "
              f"{worst[0]}: gap={worst[1]:.6f} top1={worst[2]} top2={worst[3]}")
        for step, gap, top1, top2 in gaps:
            flag = "  <-- NEAR-TIE" if gap < 0.4 else ""
            print(f"    step {step}: gap={gap:.6f} top1={top1} top2={top2}{flag}")


if __name__ == "__main__":
    main()
