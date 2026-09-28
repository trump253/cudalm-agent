// CUDALM — v0.7 Phase B: NCU target harness for the W4A16 GEMV variants.
//
// Runs ONE (N, K, B, variant) GEMV repeatedly so NCU can profile a
// controlled single shape (no model, no interleaved kernels):
//
//   ncu_w4a16_target <N> <K> <B> <variant> [iters]
//     variant: frozen | r1 | r2 | r8
//     B = 1        -> B=1 launcher (frozen int4_gemv_bf16 / rowtileR)
//     B >= 2       -> batch launcher (batch_* family)
//     iters        -> number of launches (default 64; use with NCU
//                      --launch-skip/--launch-count)
//
// Deterministic synthetic fixture (same Rng as the microbench), 16B-aligned
// buffers (vectorized path). Host-only interface, no checkpoint.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cudalm/cuda_check.h"
#include "cudalm/device_buffer.h"
#include "cudalm/kernels/batch_decode.h"
#include "cudalm/kernels/int4_gemv_bf16.h"
#include "cudalm/kernels/int4_gemv_qwen35.h"

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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <N> <K> <B> <variant> [iters]\n", argv[0]);
    return 2;
  }
  const int N = std::atoi(argv[1]);
  const int K = std::atoi(argv[2]);
  const int B = std::atoi(argv[3]);
  const std::string variant = argv[4];
  const int iters = argc >= 6 ? std::atoi(argv[5]) : 64;
  if (N < 1 || B < 1 || K % 128 != 0 || iters < 1) {
    std::fprintf(stderr, "invalid args (need N>=1, B>=1, K%%128==0, iters>=1)\n");
    return 2;
  }

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 1) return 1;
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Rng rng(0x51ed);
  std::vector<std::uint8_t> hW(static_cast<std::size_t>(N) * (K / 2));
  std::vector<__half> hS(static_cast<std::size_t>(N) * (K / 128));
  std::vector<__nv_bfloat16> hxB(B * K);
  for (auto& x : hW) x = static_cast<std::uint8_t>(rng.u32());
  for (auto& x : hS)
    x = __float2half_rn(0.001f + 0.02f * static_cast<float>(rng.u32()) *
                                       (1.0f / 4294967296.0f));
  for (auto& x : hxB) x = __float2bfloat16_rn(rng.n01());

  cudalm::DeviceBuffer dW(hW.size(), stream);
  cudalm::DeviceBuffer dS(hS.size() * sizeof(__half), stream);
  cudalm::DeviceBuffer dx(hxB.size() * sizeof(__nv_bfloat16), stream);
  cudalm::DeviceBuffer dy(static_cast<std::size_t>(B) * N *
                              sizeof(__nv_bfloat16),
                          stream);
  dW.copy_from_host(hW.data(), hW.size(), stream);
  dS.copy_from_host(hS.data(), hS.size() * 2, stream);
  dx.copy_from_host(hxB.data(), hxB.size() * 2, stream);

  const std::uint8_t* dWp = dW.data<std::uint8_t>();
  const __half* dSp = dS.data<__half>();
  const __nv_bfloat16* dxp = dx.data<__nv_bfloat16>();
  __nv_bfloat16* dyp = dy.data<__nv_bfloat16>();

  auto run = [&]() {
    if (B == 1) {
      if (variant == "frozen")
        cudalm::int4_gemv_bf16(dWp, dSp, dxp, dyp, N, K, stream);
      else if (variant == "r1")
        cudalm::int4_gemv_bf16_rowtile1(dWp, dSp, dxp, dyp, N, K, stream);
      else if (variant == "r2")
        cudalm::int4_gemv_bf16_rowtile2(dWp, dSp, dxp, dyp, N, K, stream);
      else if (variant == "r8")
        cudalm::int4_gemv_bf16_rowtile8(dWp, dSp, dxp, dyp, N, K, stream);
      else {
        std::fprintf(stderr, "bad variant %s\n", variant.c_str());
        std::exit(2);
      }
    } else {
      if (variant == "frozen")
        cudalm::kernels::batch_int4_gemv_bf16(dWp, dSp, dxp, dyp, N, K, B,
                                              stream);
      else if (variant == "r1")
        cudalm::batch_int4_gemv_bf16_rowtile1(dWp, dSp, dxp, dyp, N, K, B,
                                              stream);
      else if (variant == "r2")
        cudalm::batch_int4_gemv_bf16_rowtile2(dWp, dSp, dxp, dyp, N, K, B,
                                              stream);
      else if (variant == "r8")
        cudalm::batch_int4_gemv_bf16_rowtile8(dWp, dSp, dxp, dyp, N, K, B,
                                              stream);
      else {
        std::fprintf(stderr, "bad variant %s\n", variant.c_str());
        std::exit(2);
      }
    }
  };

  // Warmup (kept out of the measured launches via NCU --launch-skip).
  for (int i = 0; i < 8; ++i) run();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < iters; ++i) run();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
