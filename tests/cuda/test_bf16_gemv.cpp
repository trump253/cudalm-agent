// CUDALM — standalone CUDA hard gate for the plain BF16 GEMV (the tied
// LM-head matmul, v0.3 Phase B).
//
// This proves the kernel's numeric correctness on its OWN (not via the
// full-model logits): deterministic BF16 inputs, compared against a CPU
// FP32-accumulate -> BF16 RNE reference (the same intended math as the
// kernel: fp32 accumulation, one bf16 RNE per output). Kernel and reference
// use different fp32 association orders, so the comparison is
// compare_bf16_stages (default 1e-2), which catches wrong-index /
// wrong-accumulation / wrong-rounding / wrong-dtype bugs (O(magnitude)
// errors) while absorbing a few-ulp fp32 re-association noise.
//
// Coverage (per docs/qwen35_architecture.md §18.1):
//   * vec4 path: K % 8 == 0 AND 16B-aligned W/x bases (the 16B-vectorized
//     proven kernel) — including the actual LM-head shape [N=248320, K=1024];
//   * scalar fallback: K % 8 != 0 (the scalar kernel);
//   * scalar fallback: K % 8 == 0 but misaligned W/x bases (2-byte offset —
//     2-byte-aligned but not 16B-aligned, a legal __nv_bfloat16 access);
//   * a zero weight row -> bit-exact zero output;
//   * multiple N/K shapes across both paths.
//
// No checkpoint / no Python (pure kernel + CPU reference).

#include "../../tests/common/check.h"

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/bf16_gemv.h"
#include "cudalm/stage_compare.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cudalm;

namespace {

// Deterministic BF16 GEMV inputs (a simple LCG; W in [-1,1], x in [-1,1]).
struct Bf16Fixture {
  std::vector<__nv_bfloat16> W;  // [N, K] row-major
  std::vector<__nv_bfloat16> x;  // [K]
};

void make_fixture(Bf16Fixture* fx, int N, int K, std::uint32_t seed) {
  std::uint32_t s = seed;
  auto next = [&s]() -> float {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(s >> 8) * (2.0f / static_cast<float>(1u << 24)) -
           1.0f;  // [-1, 1)
  };
  fx->W.resize(static_cast<std::size_t>(N) * static_cast<std::size_t>(K));
  for (auto& w : fx->W) w = __float2bfloat16_rn(next());
  fx->x.resize(static_cast<std::size_t>(K));
  for (auto& xi : fx->x) xi = __float2bfloat16_rn(next());
}

// FP32-accumulate GEMV reference, BF16 RNE output: y[n] = bf16(sum_k f32(W[n,k])
// * f32(x[k])). Same intended math as the kernel (different fp32 order).
void ref_gemv_bf16(const __nv_bfloat16* W, const __nv_bfloat16* x, int N, int K,
                   std::vector<__nv_bfloat16>* y) {
  y->resize(static_cast<std::size_t>(N));
  for (int n = 0; n < N; ++n) {
    const __nv_bfloat16* wrow = W + static_cast<std::size_t>(n) * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
      acc += __bfloat162float(wrow[k]) * __bfloat162float(x[k]);
    (*y)[static_cast<std::size_t>(n)] = __float2bfloat16_rn(acc);
  }
}

// Run bf16_gemv on (possibly misaligned) device buffers + compare to the
// reference. `w_off` / `x_off` are byte offsets into the W/x bases (a
// non-16B-aligned offset, e.g. 2, forces the scalar fallback for a K%8==0
// shape). Returns 0 on success.
int run_vs_ref(cudaStream_t stream, const Bf16Fixture& fx, int N, int K,
               int w_off, int x_off, const char* label) {
  const std::size_t w_bytes =
      static_cast<std::size_t>(N) * static_cast<std::size_t>(K) *
      sizeof(__nv_bfloat16);
  const std::size_t x_bytes = static_cast<std::size_t>(K) * sizeof(__nv_bfloat16);
  DeviceBuffer dW(w_bytes + w_off, stream);
  DeviceBuffer dx(x_bytes + x_off, stream);
  DeviceBuffer dy(static_cast<std::size_t>(N) * sizeof(__nv_bfloat16), stream);

  std::vector<std::uint8_t> wpad(w_bytes + w_off, 0);
  std::memcpy(wpad.data() + w_off, fx.W.data(), w_bytes);
  std::vector<std::uint8_t> xpad(x_bytes + x_off, 0);
  std::memcpy(xpad.data() + x_off, fx.x.data(), x_bytes);
  dW.copy_from_host(wpad.data(), dW.bytes(), stream);
  dx.copy_from_host(xpad.data(), dx.bytes(), stream);

  const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<const std::uint8_t*>(dW.data()) + w_off);
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<const std::uint8_t*>(dx.data()) + x_off);
  bf16_gemv(wp, xp, dy.data<__nv_bfloat16>(), N, K, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> ref;
  ref_gemv_bf16(fx.W.data(), fx.x.data(), N, K, &ref);
  std::vector<__nv_bfloat16> act(N);
  dy.copy_to_host(act.data(), act.size() * sizeof(__nv_bfloat16), stream);
  const StageCompareResult r = compare_bf16_stages(
      reinterpret_cast<const std::uint16_t*>(ref.data()),
      reinterpret_cast<const std::uint16_t*>(act.data()),
      static_cast<std::size_t>(N));
  if (!r.ok) {
    std::fprintf(stderr, "  %s: FAIL max_abs_err=%.9g max_rel_err=%.9g "
                         "(idx=%zu, N=%d K=%d)\n",
                 label, r.max_abs_err, r.max_rel_err, r.max_abs_idx, N, K);
    return 1;
  }
  std::fprintf(stderr, "  %-28s N=%-6d K=%-5d max_abs_err=%.9g OK\n", label, N, K,
               r.max_abs_err);
  return 0;
}

// vec4 path: K % 8 == 0 AND 16B-aligned bases (w_off = x_off = 0).
int test_vectorized_paths(cudaStream_t stream) {
  const int shapes[][2] = {
      {248320, 1024},  // the actual tied-LM-head shape
      {1, 1024},       // single row
      {7, 1024},       // small N (block tail)
      {100, 1024},
      {1024, 1024},
      {4096, 1024},    // large N
      {100, 8},        // smallest K (one 16B vector per row)
      {100, 16},
      {100, 64},
      {100, 512},
      {3, 8},          // N < block size, minimal K
  };
  for (const auto& sh : shapes) {
    Bf16Fixture fx;
    make_fixture(&fx, sh[0], sh[1], 0x51ed51u + sh[0] + sh[1]);
    if (int rc = run_vs_ref(stream, fx, sh[0], sh[1], 0, 0, "vec4")) return rc;
  }
  return 0;
}

// scalar fallback: K % 8 != 0 (the vectorized path cannot run; the scalar
// kernel handles it). Same intended math -> same reference.
int test_scalar_k_mod(cudaStream_t stream) {
  const int shapes[][2] = {
      {5, 1000},   // K % 8 == 4
      {5, 1025},   // K % 8 == 1
      {1, 1},      // minimal
      {3, 3},
      {100, 7},    // K % 8 == 7
      {100, 9},    // K % 8 == 1
  };
  for (const auto& sh : shapes) {
    Bf16Fixture fx;
    make_fixture(&fx, sh[0], sh[1], 0x77aa77u + sh[0] * 131 + sh[1]);
    if (int rc = run_vs_ref(stream, fx, sh[0], sh[1], 0, 0, "scalar (K%8!=0)"))
      return rc;
  }
  return 0;
}

// scalar fallback: K % 8 == 0 but a 2-byte-offset W/x base (the vec4 path
// requires 16B alignment, so a 2-byte offset — 2-byte-aligned but not
// 16B-aligned, a legal __nv_bfloat16 access — forces the scalar kernel on a
// K%8==0 shape). Same data + same math must give the SAME result as the vec4
// path.
int test_scalar_misaligned(cudaStream_t stream) {
  const int N = 37, K = 1024;  // a K%8==0 shape (the LM-head K)
  Bf16Fixture fx;
  make_fixture(&fx, N, K, 0xbeefu);
  // 2-byte offset on both bases -> 2-byte-aligned, not 16B -> scalar kernel.
  if (int rc = run_vs_ref(stream, fx, N, K, 2, 2, "scalar (misaligned)"))
    return rc;
  // Cross-check: scalar (2-byte offset) vs vec4 (aligned) on the SAME data
  // must agree (same math, different order) -> catches a wrong scalar kernel.
  const std::size_t wb = N * static_cast<std::size_t>(K) * 2;
  const std::size_t xb_ = static_cast<std::size_t>(K) * 2;
  DeviceBuffer da(wb + 2, stream);
  DeviceBuffer db(wb, stream);
  DeviceBuffer xa(xb_ + 2, stream), xb(xb_, stream);
  std::vector<std::uint8_t> wpad1(wb + 2, 0);
  std::memcpy(wpad1.data() + 2, fx.W.data(), wb);
  std::vector<std::uint8_t> xpad1(xb_ + 2, 0);
  std::memcpy(xpad1.data() + 2, fx.x.data(), xb_);
  da.copy_from_host(wpad1.data(), da.bytes(), stream);
  xa.copy_from_host(xpad1.data(), xa.bytes(), stream);
  db.copy_from_host(fx.W.data(), db.bytes(), stream);
  xb.copy_from_host(fx.x.data(), xb.bytes(), stream);
  DeviceBuffer ya(N * 2, stream), yb(N * 2, stream);
  bf16_gemv(reinterpret_cast<const __nv_bfloat16*>(
                static_cast<const std::uint8_t*>(da.data()) + 2),
            reinterpret_cast<const __nv_bfloat16*>(
                static_cast<const std::uint8_t*>(xa.data()) + 2),
            ya.data<__nv_bfloat16>(), N, K, stream);
  bf16_gemv(db.data<__nv_bfloat16>(), xb.data<__nv_bfloat16>(),
            yb.data<__nv_bfloat16>(), N, K, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> ra(N), rb(N);
  ya.copy_to_host(ra.data(), N * 2, stream);
  yb.copy_to_host(rb.data(), N * 2, stream);
  const StageCompareResult r = compare_bf16_stages(
      reinterpret_cast<const std::uint16_t*>(ra.data()),
      reinterpret_cast<const std::uint16_t*>(rb.data()),
      static_cast<std::size_t>(N));
  if (!r.ok) {
    std::fprintf(stderr, "  scalar vs vec4 (same data): max_abs_err=%.9g\n",
                 r.max_abs_err);
    return 1;
  }
  std::fprintf(stderr, "  scalar vs vec4 (same data) max_abs_err=%.9g OK\n",
               r.max_abs_err);
  return 0;
}

// A fully-zero weight row must give a BIT-EXACT zero output (0 * x = 0 in fp32,
// and bf16(0) is exactly 0). Catches a spurious-bias / wrong-zero-path bug.
int test_zero_row(cudaStream_t stream) {
  const int N = 4, K = 1024;
  Bf16Fixture fx;
  make_fixture(&fx, N, K, 0x0bad5e5du);
  // Zero out row 1's weights (keep x).
  std::vector<__nv_bfloat16> W(fx.W);
  for (int k = 0; k < K; ++k) W[static_cast<std::size_t>(1) * K + k] =
      __float2bfloat16_rn(0.0f);
  DeviceBuffer dW(W.size() * 2, stream), dx(fx.x.size() * 2, stream),
      dy(N * 2, stream);
  dW.copy_from_host(W.data(), dW.bytes(), stream);
  dx.copy_from_host(fx.x.data(), dx.bytes(), stream);
  bf16_gemv(dW.data<__nv_bfloat16>(), dx.data<__nv_bfloat16>(),
            dy.data<__nv_bfloat16>(), N, K, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> act(N);
  dy.copy_to_host(act.data(), N * 2, stream);
  if (act[1] != __float2bfloat16_rn(0.0f)) {
    std::fprintf(stderr, "  zero row: output[1] != 0 (got %.9g)\n",
                 __bfloat162float(act[1]));
    return 1;
  }
  std::fprintf(stderr, "  %-28s zero row -> bit-exact 0 OK\n", "zero_row");
  return 0;
}

}  // namespace

int main() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  CHECK(n >= 1);
  CUDA_CHECK(cudaSetDevice(0));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  if (int rc = test_vectorized_paths(stream)) return rc;
  if (int rc = test_scalar_k_mod(stream)) return rc;
  if (int rc = test_scalar_misaligned(stream)) return rc;
  if (int rc = test_zero_row(stream)) return rc;

  CUDA_CHECK(cudaStreamDestroy(stream));
  TEST_PASS("test_bf16_gemv (vec4 K%8==0 + scalar K%8!=0/misaligned + "
            "zero-row, FP32-ref -> BF16 RNE)");
  return 0;
}
