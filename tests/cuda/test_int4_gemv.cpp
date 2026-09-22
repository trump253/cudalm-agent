// CUDALM — CUDA tests for the W4A16 GEMV port (CUDALab
// int4gemv_rowtile4_hx + scalar fallback).
//
// CPU reference follows the golden math contract (docs/weight_format.md):
//   y[n] = Σ_k unpack(W)[n,k] * scale_fp16_as_fp32[n, k/128] * x_fp32[k]
// with FP32 accumulation and one FP16 RNE store. Kernel and reference use
// different fp32 association orders (per-thread partials + warp reduction
// vs sequential), so the comparison is compare_fp16_stages (1e-2), which
// catches wrong-index / wrong-nibble / wrong-group bugs (O(1) errors) while
// absorbing ~1 ulp fp32 noise.

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/int4_gemv.h"
#include "cudalm/stage_compare.h"

#include "../../tests/common/check.h"

using namespace cudalm;

namespace {

struct Rng {
  std::uint32_t s;
  explicit Rng(std::uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
  std::uint32_t u32() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float n01() { return static_cast<float>(u32()) * (2.0f / 4294967296.0f) - 1.0f; }
};

// Nibble contract mirror (same code as the kernel / converter):
// low nibble = k = 2b, high nibble = k = 2b+1, 4-bit two's complement.
inline float unpack_lo(std::uint8_t b) {
  return static_cast<float>(static_cast<std::int8_t>((b & 0x0Fu) << 4) >> 4);
}
inline float unpack_hi(std::uint8_t b) {
  return static_cast<float>(static_cast<std::int8_t>(b) >> 4);
}

// FP32-accumulate GEMV reference, fp16 RNE output.
void ref_gemv(const std::uint8_t* W, const __half* S, const __half* x,
              int N, int K, std::vector<__half>* y) {
  y->resize(static_cast<std::size_t>(N));
  for (int n = 0; n < N; ++n) {
    const std::uint8_t* wrow = W + static_cast<std::size_t>(n) * (K / 2);
    const __half* srow = S + static_cast<std::size_t>(n) * (K / 128);
    float acc = 0.f;
    for (int b = 0; b < K / 2; ++b) {
      const float s = __half2float(srow[b >> 6]);
      acc += unpack_lo(wrow[b]) * s * __half2float(x[2 * b]);
      acc += unpack_hi(wrow[b]) * s * __half2float(x[2 * b + 1]);
    }
    (*y)[static_cast<std::size_t>(n)] = __float2half_rn(acc);
  }
}

struct GemvFixture {
  std::vector<std::uint8_t> W;  // N*K/2 random packed bytes (full nibble domain)
  std::vector<__half> S;        // N*K/128 random scales in [0.001, 0.021)
  std::vector<__half> x;        // K random values in [-1, 1)
};

void make_fixture(GemvFixture* f, int N, int K, std::uint32_t seed) {
  Rng rng(seed);
  f->W.resize(static_cast<std::size_t>(N) * (K / 2));
  for (auto& b : f->W) b = static_cast<std::uint8_t>(rng.u32());
  f->S.resize(static_cast<std::size_t>(N) * (K / 128));
  for (auto& s : f->S) {
    s = __float2half_rn(0.001f + 0.02f * static_cast<float>(rng.u32()) *
                                     (1.0f / 4294967296.0f));
  }
  f->x.resize(static_cast<std::size_t>(K));
  for (auto& v : f->x) v = __float2half_rn(rng.n01());
}

// One vectorized-path run against the CPU reference.
int run_vs_ref(cudaStream_t stream, const GemvFixture& fx, int N, int K,
               const char* label) {
  DeviceBuffer dW(fx.W.size(), stream);
  DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
  DeviceBuffer dx(fx.x.size() * sizeof(__half), stream);
  DeviceBuffer dy(static_cast<std::size_t>(N) * sizeof(__half), stream);
  dW.copy_from_host(fx.W.data(), dW.bytes(), stream);
  dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
  dx.copy_from_host(fx.x.data(), dx.bytes(), stream);

  int4_gemv(dW.data<std::uint8_t>(), dS.data<__half>(), dx.data<__half>(),
            dy.data<__half>(), N, K, stream);

  std::vector<__half> y(static_cast<std::size_t>(N));
  dy.copy_to_host(y.data(), dy.bytes(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__half> ref;
  ref_gemv(fx.W.data(), fx.S.data(), fx.x.data(), N, K, &ref);
  StageCompareResult r = compare_fp16_stages(
      reinterpret_cast<const std::uint16_t*>(ref.data()),
      reinterpret_cast<const std::uint16_t*>(y.data()),
      static_cast<std::size_t>(N));
  if (!r.ok) {
    std::fprintf(stderr, "  %s (N=%d K=%d): max_abs_err=%.9g idx=%llu\n", label,
                 N, K, r.max_abs_err,
                 static_cast<unsigned long long>(r.max_abs_idx));
    return 1;
  }
  return 0;
}

int test_vectorized_paths(cudaStream_t stream) {
  // The four v0.1 projection shapes + N%4 != 0 guard + minimal K=128
  // (nvec = 4 < 128 threads: only 4 threads hold x fragments).
  const int shapes[][2] = {{1024, 1024}, {512, 1024}, {2816, 1024},
                           {1024, 2816}, {1005, 1024}, {1, 128}};
  const int nshapes = static_cast<int>(sizeof(shapes) / sizeof(shapes[0]));
  for (int i = 0; i < nshapes; ++i) {
    GemvFixture fx;
    make_fixture(&fx, shapes[i][0], shapes[i][1], 0x1000 + 7 * i + 1);
    if (int rc = run_vs_ref(stream, fx, shapes[i][0], shapes[i][1],
                            "vectorized")) {
      return rc;
    }
  }
  return 0;
}

// Zero-group safety: scale == 0 → the whole group contributes exactly 0
// (no divide-by-zero, no NaN); outputs must be exactly ±0.
int test_zero_group(cudaStream_t stream) {
  const int N = 8, K = 128;
  GemvFixture fx;
  make_fixture(&fx, N, K, 0x700);
  for (auto& s : fx.S) s = __float2half_rn(0.0f);

  DeviceBuffer dW(fx.W.size(), stream);
  DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
  DeviceBuffer dx(fx.x.size() * sizeof(__half), stream);
  DeviceBuffer dy(static_cast<std::size_t>(N) * sizeof(__half), stream);
  dW.copy_from_host(fx.W.data(), dW.bytes(), stream);
  dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
  dx.copy_from_host(fx.x.data(), dx.bytes(), stream);
  int4_gemv(dW.data<std::uint8_t>(), dS.data<__half>(), dx.data<__half>(),
            dy.data<__half>(), N, K, stream);
  std::vector<__half> y(N);
  dy.copy_to_host(y.data(), dy.bytes(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const std::uint16_t* bits = reinterpret_cast<const std::uint16_t*>(y.data());
  for (int n = 0; n < N; ++n) {
    // exactly ±0 (bits 0x0000 or 0x8000): finite and zero magnitude
    CHECK_EQ(bits[n] & 0x7FFFu, 0u);
  }
  return 0;
}

// Scalar fallback: misaligned base pointers must route to the scalar kernel
// and still match the CPU reference (legal inputs are never rejected).
int test_scalar_fallback(cudaStream_t stream) {
  const int N = 512, K = 1024;
  GemvFixture fx;
  make_fixture(&fx, N, K, 0xabc);

  // (a) x misaligned by 2 halves (4B): scalar path on x[2..K+1].
  {
    DeviceBuffer dW(fx.W.size(), stream);
    DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
    DeviceBuffer dx((K + 2) * sizeof(__half), stream);
    DeviceBuffer dy(static_cast<std::size_t>(N) * sizeof(__half), stream);
    dW.copy_from_host(fx.W.data(), dW.bytes(), stream);
    dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
    std::vector<__half> xbuf(K + 2);
    Rng rng(0xdef);
    for (auto& v : xbuf) v = __float2half_rn(rng.n01());
    dx.copy_from_host(xbuf.data(), dx.bytes(), stream);

    const __half* x2 = dx.data<__half>() + 2;
    int4_gemv(dW.data<std::uint8_t>(), dS.data<__half>(), x2,
              dy.data<__half>(), N, K, stream);
    std::vector<__half> y(N);
    dy.copy_to_host(y.data(), dy.bytes(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__half> ref;
    ref_gemv(fx.W.data(), fx.S.data(), xbuf.data() + 2, N, K, &ref);
    StageCompareResult r = compare_fp16_stages(
        reinterpret_cast<const std::uint16_t*>(ref.data()),
        reinterpret_cast<const std::uint16_t*>(y.data()),
        static_cast<std::size_t>(N));
    CHECK(r.ok);
    if (!r.ok) {
      std::fprintf(stderr, "  scalar fallback (x off 2): max_abs_err=%.9g\n",
                   r.max_abs_err);
      return 1;
    }
  }
  // (b) weight misaligned by 8 bytes: scalar path on W[8..].
  {
    const std::size_t Wbytes = static_cast<std::size_t>(N) * (K / 2) + 8;
    std::vector<std::uint8_t> wbuf(Wbytes);
    Rng rng(0x987);
    for (auto& b : wbuf) b = static_cast<std::uint8_t>(rng.u32());
    DeviceBuffer dW(Wbytes, stream);
    DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
    DeviceBuffer dx(fx.x.size() * sizeof(__half), stream);
    DeviceBuffer dy(static_cast<std::size_t>(N) * sizeof(__half), stream);
    dW.copy_from_host(wbuf.data(), Wbytes, stream);
    dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
    dx.copy_from_host(fx.x.data(), dx.bytes(), stream);

    const std::uint8_t* W2 = dW.data<std::uint8_t>() + 8;
    int4_gemv(W2, dS.data<__half>(), dx.data<__half>(), dy.data<__half>(), N, K,
              stream);
    std::vector<__half> y(N);
    dy.copy_to_host(y.data(), dy.bytes(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__half> ref;
    ref_gemv(wbuf.data() + 8, fx.S.data(), fx.x.data(), N, K, &ref);
    StageCompareResult r = compare_fp16_stages(
        reinterpret_cast<const std::uint16_t*>(ref.data()),
        reinterpret_cast<const std::uint16_t*>(y.data()),
        static_cast<std::size_t>(N));
    CHECK(r.ok);
    if (!r.ok) {
      std::fprintf(stderr, "  scalar fallback (W off 8B): max_abs_err=%.9g\n",
                   r.max_abs_err);
      return 1;
    }
  }
  // (c) same data through both paths (vectorized on an aligned copy vs
  //     scalar on a +2-half view) must agree within the stage tolerance.
  {
    std::vector<__half> xbuf(K + 2);
    Rng rng(0x246);
    for (auto& v : xbuf) v = __float2half_rn(rng.n01());

    DeviceBuffer dW(fx.W.size(), stream);
    DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
    DeviceBuffer dxa((K + 2) * sizeof(__half), stream);  // misaligned view
    DeviceBuffer dxb(K * sizeof(__half), stream);        // aligned copy of x[2..]
    DeviceBuffer dya(static_cast<std::size_t>(N) * sizeof(__half), stream);
    DeviceBuffer dyb(static_cast<std::size_t>(N) * sizeof(__half), stream);
    dW.copy_from_host(fx.W.data(), dW.bytes(), stream);
    dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
    dxa.copy_from_host(xbuf.data(), dxa.bytes(), stream);
    dxb.copy_from_host(xbuf.data() + 2, dxb.bytes(), stream);

    int4_gemv(dW.data<std::uint8_t>(), dS.data<__half>(),
              dxa.data<__half>() + 2, dya.data<__half>(), N, K, stream);  // scalar
    int4_gemv(dW.data<std::uint8_t>(), dS.data<__half>(), dxb.data<__half>(),
              dyb.data<__half>(), N, K, stream);  // vectorized
    std::vector<__half> ya(N), yb(N);
    dya.copy_to_host(ya.data(), dya.bytes(), stream);
    dyb.copy_to_host(yb.data(), dyb.bytes(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    StageCompareResult r = compare_fp16_stages(
        reinterpret_cast<const std::uint16_t*>(ya.data()),
        reinterpret_cast<const std::uint16_t*>(yb.data()),
        static_cast<std::size_t>(N));
    CHECK(r.ok);
    if (!r.ok) {
      std::fprintf(stderr, "  scalar vs vectorized (same data): max_abs_err=%.9g\n",
                   r.max_abs_err);
      return 1;
    }
  }
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
  if (int rc = test_zero_group(stream)) return rc;
  if (int rc = test_scalar_fallback(stream)) return rc;

  CUDA_CHECK(cudaStreamDestroy(stream));
  TEST_PASS("test_int4_gemv");
  return 0;
}
