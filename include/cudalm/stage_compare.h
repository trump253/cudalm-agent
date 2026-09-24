// CUDALM — stage-wise golden comparison (host side, no CUDA, no PyTorch).
//
// Compares an fp16 stage tensor produced by the runtime (copied back to the
// host) against the fp16 values stored in a CUDLMG01 golden file.
//
// Pinned v0.1 correctness bar: the golden and the kernels both compute in
// FP32 and round to FP16 at the same stage boundaries, so per-element
// differences are a few fp16 ulps (fp16 ulp near 1.0 is ~9.8e-4). The
// tolerance below (|a-r| <= atol + rtol*|r| with atol = rtol = 1e-2) is
// deliberately an order of magnitude looser than the fp16 noise floor: it
// must pass for any faithful re-association of fp32 adds, and it still
// catches wrong-stage / wrong-index / wrong-weight bugs, which are O(1).
//
// Provenance: CUDALM-native.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cudalm {

// Bit-exact IEEE-754 binary16 -> binary32 decode (host, no libm beyond cmath).
// Handles ±0, subnormals, normals, ±inf, NaN (payload preserved in the low
// 13 mantissa bits of the float NaN).
inline float fp16_bits_to_f32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t expf = (h >> 10) & 0x1Fu;
  std::uint32_t mant = h & 0x3FFu;
  std::uint32_t bits;
  if (expf == 0) {
    if (mant == 0) {
      bits = sign;  // ±0
    } else {
      // Subnormal: value = mant * 2^-24. Normalize the mantissa, then map
      // the 10-bit half mantissa into the float's 23-bit field (<< 13).
      std::uint32_t e = 127u - 15u + 1u;  // unbiased exponent accumulator
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --e;
      }
      mant &= 0x3FFu;
      bits = sign | (e << 23) | (mant << 13);
    }
  } else if (expf == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);  // ±inf / NaN
  } else {
    bits = sign | ((expf - 15u + 127u) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Result of one stage comparison.
struct StageCompareResult {
  bool ok = true;
  std::size_t n = 0;
  double max_abs_err = 0.0;
  std::size_t max_abs_idx = 0;
  // max of |a-r| / max(|r|, 1.0) — relative error with an absolute floor so
  // near-zero references do not blow up the ratio.
  double max_rel_err = 0.0;
};

// Bit-exact IEEE-754 bfloat16 -> binary32 decode (host, no libm beyond
// cmath). bf16 IS the high 16 bits of the fp32 representation, so the
// decode is a pure left shift; NaN payloads are preserved in the low 16
// mantissa bits of the float NaN, infinities/±0 map exactly.
inline float bf16_bits_to_f32(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Elementwise comparison of two fp16 arrays (raw bit patterns).
// Pass criterion per element: |a - r| <= atol + rtol * |r|.
// NaN/inf in `act` always fails (the comparison with NaN is false).
inline StageCompareResult compare_fp16_stages(const std::uint16_t* ref,
                                              const std::uint16_t* act,
                                              std::size_t n,
                                              double atol = 1e-2,
                                              double rtol = 1e-2) {
  StageCompareResult res;
  res.n = n;
  for (std::size_t i = 0; i < n; ++i) {
    const double r = static_cast<double>(fp16_bits_to_f32(ref[i]));
    const double a = static_cast<double>(fp16_bits_to_f32(act[i]));
    const double d = std::fabs(a - r);
    const double rel = d / std::fmax(std::fabs(r), 1.0);
    if (d > res.max_abs_err) {
      res.max_abs_err = d;
      res.max_abs_idx = i;
    }
    if (rel > res.max_rel_err) res.max_rel_err = rel;
    if (!(d <= atol + rtol * std::fabs(r))) res.ok = false;
  }
  return res;
}

// Elementwise comparison of two BF16 arrays (raw bit patterns; the v0.2
// Qwen3.5 stage path). Same pass criterion as the fp16 comparator.
//
// v0.2 tolerance rationale (docs/qwen35_architecture.md §14): the golden
// and the kernels round to bf16 at the SAME stage boundaries (the official
// bf16 dtype flow), so per-element differences are at most a few bf16 ulps
// from (a) fp32 re-association of the GEMV/attention sums and (b) ~1-ulp
// host/device libm differences in expf/cosf/sinf before the bf16 rounding.
// A bf16 ulp near 1.0 is ~3.9e-3 (8-bit mantissa); the default
// atol = rtol = 1e-2 is ~2.5 ulps — loose enough for any faithful
// re-association, tight enough to catch wrong-stage / wrong-index /
// wrong-weight / wrong-dtype bugs, which are O(1).
// NaN/inf in `act` always fails (the comparison with NaN is false).
inline StageCompareResult compare_bf16_stages(const std::uint16_t* ref,
                                              const std::uint16_t* act,
                                              std::size_t n,
                                              double atol = 1e-2,
                                              double rtol = 1e-2) {
  StageCompareResult res;
  res.n = n;
  for (std::size_t i = 0; i < n; ++i) {
    const double r = static_cast<double>(bf16_bits_to_f32(ref[i]));
    const double a = static_cast<double>(bf16_bits_to_f32(act[i]));
    const double d = std::fabs(a - r);
    const double rel = d / std::fmax(std::fabs(r), 1.0);
    if (d > res.max_abs_err) {
      res.max_abs_err = d;
      res.max_abs_idx = i;
    }
    if (rel > res.max_rel_err) res.max_rel_err = rel;
    if (!(d <= atol + rtol * std::fabs(r))) res.ok = false;
  }
  return res;
}

}  // namespace cudalm
