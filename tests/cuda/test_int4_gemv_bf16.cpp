// CUDALM — CUDA tests for the W4A16 GEMV bf16 specialization (port of
// CUDALab int4gemv_rowtile4_hx, Qwen3.5 v0.2 path).
//
// CPU reference follows the W4A16 contract with the bf16 boundary
// (docs/weight_format.md §4 + docs/qwen35_architecture.md §4 row 8):
//   y[n] = bf16( Σ_k unpack(W)[n,k] * scale_fp16_as_fp32[n, k/128]
//                * f32(x_bf16)[k] )
// with FP32 accumulation and ONE bf16 RNE store. Kernel and reference use
// different fp32 association orders, so the comparison is
// compare_bf16_stages (default 1e-2), which catches wrong-index /
// wrong-nibble / wrong-group bugs (O(1) errors) while absorbing a few ulp
// fp32 noise.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/int4_gemv_bf16.h"
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

// FP32-accumulate GEMV reference, bf16 RNE output.
void ref_gemv_bf16(const std::uint8_t* W, const __half* S,
                   const __nv_bfloat16* x, int N, int K,
                   std::vector<__nv_bfloat16>* y) {
  y->resize(static_cast<std::size_t>(N));
  for (int n = 0; n < N; ++n) {
    const std::uint8_t* wrow = W + static_cast<std::size_t>(n) * (K / 2);
    const __half* srow = S + static_cast<std::size_t>(n) * (K / 128);
    float acc = 0.f;
    for (int b = 0; b < K / 2; ++b) {
      const float s = __half2float(srow[b >> 6]);
      acc += unpack_lo(wrow[b]) * s * __bfloat162float(x[2 * b]);
      acc += unpack_hi(wrow[b]) * s * __bfloat162float(x[2 * b + 1]);
    }
    (*y)[static_cast<std::size_t>(n)] = __float2bfloat16_rn(acc);
  }
}

struct GemvFixture {
  std::vector<std::uint8_t> W;  // N*K/2 random packed bytes (full nibble domain)
  std::vector<__half> S;        // N*K/128 random scales in [0.001, 0.021)
  std::vector<__nv_bfloat16> x; // K random values in [-1, 1)
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
  for (auto& v : f->x) v = __float2bfloat16_rn(rng.n01());
}

// One vectorized-path run against the CPU reference.
int run_vs_ref(cudaStream_t stream, const GemvFixture& fx, int N, int K,
               const char* label) {
  DeviceBuffer dW(fx.W.size(), stream);
  DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
  DeviceBuffer dx(fx.x.size() * sizeof(__nv_bfloat16), stream);
  DeviceBuffer dy(static_cast<std::size_t>(N) * sizeof(__nv_bfloat16), stream);
  dW.copy_from_host(fx.W.data(), dW.bytes(), stream);
  dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
  dx.copy_from_host(fx.x.data(), dx.bytes(), stream);

  int4_gemv_bf16(dW.data<std::uint8_t>(), dS.data<__half>(),
                 dx.data<__nv_bfloat16>(), dy.data<__nv_bfloat16>(), N, K,
                 stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> ref;
  ref_gemv_bf16(fx.W.data(), fx.S.data(), fx.x.data(), N, K, &ref);
  std::vector<__nv_bfloat16> act(N);
  dy.copy_to_host(act.data(), act.size() * sizeof(__nv_bfloat16), stream);
  const StageCompareResult r =
      compare_bf16_stages(reinterpret_cast<const std::uint16_t*>(ref.data()),
                          reinterpret_cast<const std::uint16_t*>(act.data()),
                          static_cast<std::size_t>(N));
  if (!r.ok) {
    std::fprintf(stderr, "  %s: FAIL max_abs_err=%.9g max_rel_err=%.9g "
                         "(idx=%zu)\n", label, r.max_abs_err, r.max_rel_err,
                 r.max_abs_idx);
    return 1;
  }
  std::fprintf(stderr, "  %s: N=%d K=%d max_abs_err=%.9g OK\n", label, N, K,
               r.max_abs_err);
  return 0;
}

int test_vectorized_paths(cudaStream_t stream) {
  // The real Qwen3.5 layer-3 GEMV shapes + a few structural cases.
  const int shapes[][2] = {
      {4096, 1024},  // q_proj (fused [q;gate])
      {512, 1024},   // k_proj / v_proj
      {1024, 2048},  // o_proj
      {3584, 1024},  // gate/up_proj
      {1024, 3584},  // down_proj
      {16, 1024},    // small N (row-tile tail)
      {3, 128},      // N < 4 (last-block guard) x smallest K
  };
  for (const auto& sh : shapes) {
    GemvFixture fx;
    make_fixture(&fx, sh[0], sh[1], 0x51ed51u + sh[0] + sh[1]);
    if (int rc = run_vs_ref(stream, fx, sh[0], sh[1], "vectorized")) return rc;
  }
  return 0;
}

// A fully-zero weight row must give a BIT-EXACT zero output (scale 0 or q 0
// both neutralize the row; the zero-group path stores scale 0).
int test_zero_group(cudaStream_t stream) {
  const int N = 8, K = 256;
  std::vector<std::uint8_t> W(static_cast<std::size_t>(N) * (K / 2), 0);
  std::vector<__half> S(static_cast<std::size_t>(N) * (K / 128),
                        __float2half_rn(0.0f));
  std::vector<__nv_bfloat16> x(K);
  Rng rng(123u);
  for (auto& v : x) v = __float2bfloat16_rn(rng.n01());

  DeviceBuffer dW(W.size(), stream);
  DeviceBuffer dS(S.size() * sizeof(__half), stream);
  DeviceBuffer dx(x.size() * sizeof(__nv_bfloat16), stream);
  DeviceBuffer dy(static_cast<std::size_t>(N) * sizeof(__nv_bfloat16), stream);
  dW.copy_from_host(W.data(), dW.bytes(), stream);
  dS.copy_from_host(S.data(), dS.bytes(), stream);
  dx.copy_from_host(x.data(), dx.bytes(), stream);

  int4_gemv_bf16(dW.data<std::uint8_t>(), dS.data<__half>(),
                 dx.data<__nv_bfloat16>(), dy.data<__nv_bfloat16>(), N, K,
                 stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> act(N);
  dy.copy_to_host(act.data(), act.size() * sizeof(__nv_bfloat16), stream);
  for (int n = 0; n < N; ++n) {
    if (act[n] != __nv_bfloat16(0.0f)) {
      std::fprintf(stderr, "  zero group: row %d not exactly zero\n", n);
      return 1;
    }
  }
  return 0;
}

// Misaligned base pointer -> scalar fallback kernel. Same data must give
// the SAME result as the vectorized path (within the fp32-association
// tolerance — same math, different order).
int test_scalar_fallback(cudaStream_t stream) {
  const int N = 37, K = 256;
  GemvFixture fx;
  make_fixture(&fx, N, K, 777u);

  DeviceBuffer dW(fx.W.size() + 2, stream);  // +2B: room for a 1B offset
  DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
  DeviceBuffer dx(fx.x.size() * sizeof(__nv_bfloat16) + 2, stream);
  DeviceBuffer dy(static_cast<std::size_t>(N) * sizeof(__nv_bfloat16), stream);
  // Offset W and x by 1 byte: legal input, misaligned bases.
  std::vector<std::uint8_t> Wpad(fx.W.size() + 2, 0);
  std::memcpy(Wpad.data() + 1, fx.W.data(), fx.W.size());
  std::vector<__nv_bfloat16> xpad(fx.x.size() + 1, __nv_bfloat16(0.0f));
  std::memcpy(xpad.data() + 1, fx.x.data(), fx.x.size() * sizeof(__nv_bfloat16));
  dW.copy_from_host(Wpad.data(), dW.bytes(), stream);
  dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
  dx.copy_from_host(xpad.data(), xpad.size() * sizeof(__nv_bfloat16), stream);

  int4_gemv_bf16(reinterpret_cast<std::uint8_t*>(dW.data()) + 1,
                 dS.data<__half>(),
                 reinterpret_cast<__nv_bfloat16*>(dx.data()) + 1,
                 dy.data<__nv_bfloat16>(), N, K, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> act(N);
  dy.copy_to_host(act.data(), act.size() * sizeof(__nv_bfloat16), stream);

  std::vector<__nv_bfloat16> ref;
  ref_gemv_bf16(fx.W.data(), fx.S.data(), fx.x.data(), N, K, &ref);
  const StageCompareResult r =
      compare_bf16_stages(reinterpret_cast<const std::uint16_t*>(ref.data()),
                          reinterpret_cast<const std::uint16_t*>(act.data()),
                          static_cast<std::size_t>(N));
  if (!r.ok) {
    std::fprintf(stderr, "  scalar fallback: FAIL max_abs_err=%.9g\n",
                 r.max_abs_err);
    return 1;
  }

  // Cross-check scalar vs vectorized on the same fixture.
  DeviceBuffer dyb(static_cast<std::size_t>(N) * sizeof(__nv_bfloat16), stream);
  DeviceBuffer dW2(fx.W.size(), stream);
  DeviceBuffer dx2(fx.x.size() * sizeof(__nv_bfloat16), stream);
  dW2.copy_from_host(fx.W.data(), dW2.bytes(), stream);
  dx2.copy_from_host(fx.x.data(), dx2.bytes(), stream);
  int4_gemv_bf16(dW2.data<std::uint8_t>(), dS.data<__half>(),
                 dx2.data<__nv_bfloat16>(), dyb.data<__nv_bfloat16>(), N, K,
                 stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> actb(N);
  dyb.copy_to_host(actb.data(), actb.size() * sizeof(__nv_bfloat16), stream);
  const StageCompareResult r2 =
      compare_bf16_stages(
          reinterpret_cast<const std::uint16_t*>(act.data()),
          reinterpret_cast<const std::uint16_t*>(actb.data()),
          static_cast<std::size_t>(N));
  if (!r2.ok) {
    std::fprintf(stderr, "  scalar vs vectorized (same data): max_abs_err=%.9g\n",
                 r2.max_abs_err);
    return 1;
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
  TEST_PASS("test_int4_gemv_bf16");
  return 0;
}
