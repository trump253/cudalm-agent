// CUDALM — v0.7 Phase C: NCU target harness for the DeltaNet delta-rule
// variants.
//
// Runs ONE (B, variant) delta update repeatedly so NCU can profile a
// controlled single shape (no model, no interleaved kernels):
//
//   ncu_deltanet_target <B> <variant> [iters]
//     B = 1        -> B=1 launcher (frozen / vreg / vvec / vchunk)
//     B >= 2       -> batch launcher (batch_* family)
//     variant: frozen | vreg | vvec | vchunk
//     iters        -> number of launches (default 64; use with NCU
//                      --launch-skip/--launch-count). vchunk launches two
//                      kernels per call (red + s pass); NCU -k regex on
//                      "deltanet_delta" captures both.
//
// Deterministic synthetic fixture (same Rng/fixture as the microbench),
// 16B-aligned buffers (cudaMalloc). Host-only interface, no checkpoint.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cudalm/cuda_check.h"
#include "cudalm/device_buffer.h"
#include "cudalm/kernels/batch_decode.h"
#include "cudalm/kernels/deltanet_delta_qwen35.h"
#include "cudalm/kernels/qwen35_deltanet_kernels.h"

namespace {

constexpr int kNHeads = 16;
constexpr int kHd = 128;
constexpr float kEps = 1e-6f;

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
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <B> <variant> [iters]\n"
                 "  variant: frozen | vreg | vvec | vchunk\n",
                 argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[1]);
  const std::string variant = argv[2];
  const int iters = argc >= 4 ? std::atoi(argv[3]) : 64;
  if (B < 1 || iters < 1) {
    std::fprintf(stderr, "invalid args (need B>=1, iters>=1)\n");
    return 2;
  }

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 1) return 1;
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const std::size_t n = static_cast<std::size_t>(B) * kNHeads * kHd;
  Rng rng(0xd31c);
  std::vector<__nv_bfloat16> hq(n), hk(n), hv(n);
  std::vector<float> hg(static_cast<std::size_t>(B) * kNHeads);
  std::vector<__nv_bfloat16> hb(static_cast<std::size_t>(B) * kNHeads);
  std::vector<float> hs(B * kNHeads * kHd * kHd);
  for (std::size_t i = 0; i < n; ++i) {
    hq[i] = __float2bfloat16_rn(rng.n01() * 2.0f);
    hk[i] = __float2bfloat16_rn(rng.n01() * 2.0f);
    hv[i] = __float2bfloat16_rn(rng.n01() * 2.0f);
  }
  for (std::size_t i = 0; i < hg.size(); ++i)
    hg[i] = -expf(1.0f) * (0.01f + 0.5f * (0.5f + 0.5f * rng.n01()));
  for (std::size_t i = 0; i < hb.size(); ++i)
    hb[i] = __float2bfloat16_rn(1.0f / (1.0f + expf(-rng.n01() * 6.0f)));
  for (auto& x : hs) x = rng.n01() * 4.0f;

  cudalm::DeviceBuffer dq(n * 2, stream);
  cudalm::DeviceBuffer dk(n * 2, stream);
  cudalm::DeviceBuffer dv(n * 2, stream);
  cudalm::DeviceBuffer dg(hg.size() * sizeof(float), stream);
  cudalm::DeviceBuffer db(hb.size() * 2, stream);
  cudalm::DeviceBuffer dstate(hs.size() * sizeof(float), stream);
  cudalm::DeviceBuffer dout(n * 2, stream);
  const int slots[3] = {0, 1, 2};
  cudalm::DeviceBuffer dslots(sizeof(int) * B, stream);
  dq.copy_from_host(hq.data(), n * 2, stream);
  dk.copy_from_host(hk.data(), n * 2, stream);
  dv.copy_from_host(hv.data(), n * 2, stream);
  dg.copy_from_host(hg.data(), hg.size() * sizeof(float), stream);
  db.copy_from_host(hb.data(), hb.size() * 2, stream);
  dstate.copy_from_host(hs.data(), hs.size() * sizeof(float), stream);
  dslots.copy_from_host(slots, sizeof(int) * B, stream);

  const __nv_bfloat16* q = static_cast<const __nv_bfloat16*>(dq.data());
  const __nv_bfloat16* k = static_cast<const __nv_bfloat16*>(dk.data());
  const __nv_bfloat16* v = static_cast<const __nv_bfloat16*>(dv.data());
  const float* g = static_cast<const float*>(dg.data());
  const __nv_bfloat16* beta = static_cast<const __nv_bfloat16*>(db.data());
  float* S = static_cast<float*>(dstate.data());
  __nv_bfloat16* out = static_cast<__nv_bfloat16*>(dout.data());
  const int* slots_p = static_cast<const int*>(dslots.data());

  auto run = [&]() {
    if (B == 1) {
      if (variant == "frozen")
        cudalm::kernels::qwen35_deltanet_delta_rule_fp32(q, k, v, g, beta, S,
                                                         out, kNHeads, kHd,
                                                         kEps, stream);
      else if (variant == "vreg")
        cudalm::kernels::deltanet_delta_vreg(q, k, v, g, beta, S, out, kNHeads,
                                             kHd, kEps, stream);
      else if (variant == "vvec")
        cudalm::kernels::deltanet_delta_vvec(q, k, v, g, beta, S, out, kNHeads,
                                             kHd, kEps, stream);
      else if (variant == "vchunk")
        cudalm::kernels::deltanet_delta_vchunk(q, k, v, g, beta, S, out,
                                               kNHeads, kHd, kEps, stream);
      else {
        std::fprintf(stderr, "bad variant %s\n", variant.c_str());
        std::exit(2);
      }
    } else {
      if (variant == "frozen")
        cudalm::kernels::batch_deltanet_delta_rule_fp32(q, k, v, g, beta, S,
                                                        slots_p, out, kNHeads,
                                                        kHd, kEps, B, stream);
      else if (variant == "vreg")
        cudalm::kernels::batch_deltanet_delta_vreg(q, k, v, g, beta, S,
                                                   slots_p, out, kNHeads, kHd,
                                                   kEps, B, stream);
      else if (variant == "vvec")
        cudalm::kernels::batch_deltanet_delta_vvec(q, k, v, g, beta, S,
                                                   slots_p, out, kNHeads, kHd,
                                                   kEps, B, stream);
      else if (variant == "vchunk")
        cudalm::kernels::batch_deltanet_delta_vchunk(q, k, v, g, beta, S,
                                                     slots_p, out, kNHeads,
                                                     kHd, kEps, B, stream);
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
