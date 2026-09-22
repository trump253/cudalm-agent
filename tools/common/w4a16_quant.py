"""CUDALM W4A16 G=128 quantizer — shared Python contract.

Single source of truth: docs/weight_format.md §4 (v1 contract, fp16 source)
extended for Qwen3.5 (v0.2, bf16 source). The quantization math is identical
to tools/convert_weights.py::quantize_w except the input dtype is bf16; all
math is fp32, the scale is stored as fp16 = amax/7, and round is
round-half-to-even with clamp to [-7, 7] and zero-group -> scale 0 / q 0.

Runtime reference (C++): src/runtime/kernels/int4_gemv.cu unpacks the low
nibble at k = 2b, high nibble at k = 2b+1, sign-extends, multiplies by the
per-group fp16 scale (fp32 accumulation). `dequant_reference` here is the
exact reference used by the runtime-correctness gate (A) in
docs/qwen35_architecture.md §11.

Requires torch (tools only — the C++ runtime never imports this).
"""
from __future__ import annotations

import torch

GROUP_SIZE = 128
INT4_MAX = 7


def quantize_w4a16(W: torch.Tensor):
    """W: (N, K) bf16 or fp16, contiguous, K % 128 == 0, finite.

    Returns (W_packed uint8 (N, K/2), scale fp16 (N, K/128),
             q int32 (N, K), scale32 fp32 (N, K/128)).
    """
    if W.dim() != 2:
        raise ValueError(f"expected 2-D W, got {W.dim()}D")
    if W.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"quantizer accepts fp16/bf16 only, got {W.dtype}")
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


def unpack_int4(W_packed: torch.Tensor) -> torch.Tensor:
    """uint8 (N, K/2) -> int32 (N, K), sign-extended (CUDALab reference)."""
    N, H = W_packed.shape
    lo = (W_packed & 15).to(torch.int32)
    hi = ((W_packed >> 4) & 15).to(torch.int32)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    q = torch.stack((lo, hi), dim=-1).view(N, K := H * 2)
    return q


def dequant_reference(W_packed: torch.Tensor, scale16: torch.Tensor) -> torch.Tensor:
    """Exact dequantization used by the runtime: q * scale_fp16.as_f32 (fp32).

    W_packed: uint8 (N, K/2); scale16: fp16 (N, K/128). Returns fp32 (N, K).
    """
    q = unpack_int4(W_packed)
    N, K = q.shape
    s = scale16.float().repeat_interleave(GROUP_SIZE, dim=1)
    return q * s


def fidelity_stats(W_orig: torch.Tensor, W_packed: torch.Tensor,
                   scale16: torch.Tensor) -> dict:
    """Quantization fidelity (report only, NOT the correctness gate):
    max_abs error, RMSE, cosine similarity between the original (fp32 view)
    and the dequantized weights."""
    orig = W_orig.float()
    deq = dequant_reference(W_packed, scale16)
    err = (deq - orig).abs()
    cos = (torch.nn.functional.cosine_similarity(
        orig.flatten().unsqueeze(0), deq.flatten().unsqueeze(0))
        .item())
    return {
        "max_abs": float(err.max().item()),
        "rmse": float(err.pow(2).mean().sqrt().item()),
        "cosine": cos,
    }
