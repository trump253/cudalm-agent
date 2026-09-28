// CUDALM — v0.7 Phase D candidate 1: fused residual-add + zero-centered
// RMSNorm kernel-level EXACT parity gate.
//
// The contract: the fused kernel qwen35_fused_add_rmsnorm_zc_bf16 must be BF16
// BIT-IDENTICAL — for BOTH outputs (the residual `res` and the norm `norm`) —
// to the frozen 2-launch sequence
//     qwen35_add_bf16(a, b, res, ...)
//     qwen35_rmsnorm_zc_bf16(res, w, norm, ...)
// on the same inputs, for M in {1,2,3}, H in {256, 1024} (PER=1 and PER=4
// paths), production eps and a couple of alternate eps, and the edge input
// classes (deterministic random multi-seed, all-zero, small, large, mixed
// sign). Bit-identity holds by construction: the fused kernel computes the
// residual as one bf16 RNE of the fp32 sum (frozen add), then runs the frozen
// rmsnorm reduction tree on the BF16-ROUNDED residual, with one bf16 RNE per
// norm output element. This is a bit-exact gate (memcmp on the BF16 words),
// NOT a tolerance comparison.

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/qwen35_kernels.h"

#include "../../tests/common/check.h"

using namespace cudalm;

namespace {
int g_failures = 0;
}  // namespace

#define CHECK_BITS(cond)                                               \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::fprintf(stderr, "CHECK failed: %s  (%s:%d)\n", #cond,      \
                   __FILE__, __LINE__);                                \
      ++g_failures;                                                    \
      return -1;                                                       \
    }                                                                  \
  } while (0)

namespace {

// Deterministic xorshift32 in [-1, 1).
struct Rng {
  std::uint32_t s;
  explicit Rng(std::uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
  float n01() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s * (2.0f / 4294967296.0f) - 1.0f;
  }
};

void fill_bf16(std::vector<__nv_bfloat16>* out, std::size_t n, float scale,
               std::uint32_t seed) {
  Rng rng(seed);
  out->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    (*out)[i] = __float2bfloat16_rn(rng.n01() * scale);
  }
}

void fill_zero(std::vector<__nv_bfloat16>* out, std::size_t n) {
  out->assign(n, __float2bfloat16_rn(0.0f));
}

// One parity case: frozen (add -> rmsnorm) vs fused, both outputs bit-exact.
int run_case(int M, int H, float eps, const std::vector<__nv_bfloat16>& a,
             const std::vector<__nv_bfloat16>& b,
             const std::vector<__nv_bfloat16>& w, cudaStream_t stream) {
  const std::size_t rows = static_cast<std::size_t>(M) * H;
  DeviceBuffer da(rows * sizeof(__nv_bfloat16), stream);
  DeviceBuffer db(rows * sizeof(__nv_bfloat16), stream);
  DeviceBuffer dw(static_cast<std::size_t>(H) * sizeof(__nv_bfloat16), stream);
  da.copy_from_host(a.data(), da.bytes(), stream);
  db.copy_from_host(b.data(), db.bytes(), stream);
  dw.copy_from_host(w.data(), dw.bytes(), stream);

  // Frozen 2-launch sequence (independent buffers).
  DeviceBuffer res_f(rows * sizeof(__nv_bfloat16), stream);
  DeviceBuffer norm_f(rows * sizeof(__nv_bfloat16), stream);
  cudalm::kernels::qwen35_add_bf16(da.data<__nv_bfloat16>(),
                                   db.data<__nv_bfloat16>(),
                                   res_f.data<__nv_bfloat16>(), rows, stream);
  cudalm::kernels::qwen35_rmsnorm_zc_bf16(
      res_f.data<__nv_bfloat16>(), dw.data<__nv_bfloat16>(),
      norm_f.data<__nv_bfloat16>(), M, H, eps, stream);

  // Fused single kernel (independent buffers).
  DeviceBuffer res_c(rows * sizeof(__nv_bfloat16), stream);
  DeviceBuffer norm_c(rows * sizeof(__nv_bfloat16), stream);
  cudalm::kernels::qwen35_fused_add_rmsnorm_zc_bf16(
      da.data<__nv_bfloat16>(), db.data<__nv_bfloat16>(),
      dw.data<__nv_bfloat16>(), res_c.data<__nv_bfloat16>(),
      norm_c.data<__nv_bfloat16>(), M, H, eps, stream);

  std::vector<__nv_bfloat16> rf(rows), nf(rows), rc(rows), nc(rows);
  res_f.copy_to_host(rf.data(), res_f.bytes(), stream);
  norm_f.copy_to_host(nf.data(), norm_f.bytes(), stream);
  res_c.copy_to_host(rc.data(), res_c.bytes(), stream);
  norm_c.copy_to_host(nc.data(), norm_c.bytes(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CHECK_BITS(std::memcmp(rf.data(), rc.data(),
                         rows * sizeof(__nv_bfloat16)) == 0);
  CHECK_BITS(std::memcmp(nf.data(), nc.data(),
                         rows * sizeof(__nv_bfloat16)) == 0);
  return 0;
}

// A family of input classes over (M, H): multi-seed random at several scales,
// all-zero a/b (degenerate residual), and a mixed large/small case.
int run_gate(int M, int H, cudaStream_t stream) {
  const std::size_t rows = static_cast<std::size_t>(M) * H;
  const float eps_prod = 1e-6f;
  const float eps_alt[] = {1e-5f, 1e-8f};

  // multi-seed random at scale 1 (production-like magnitudes)
  for (std::uint32_t seed : {0xD4A1u, 0xBEEFu, 0x1234u, 0x9E37u}) {
    std::vector<__nv_bfloat16> a, b, w;
    fill_bf16(&a, rows, 1.0f, seed);
    fill_bf16(&b, rows, 1.0f, seed ^ 0x5A5Au);
    fill_bf16(&w, static_cast<std::size_t>(H), 1.0f, seed ^ 0xC0FFEEu);
    if (run_case(M, H, eps_prod, a, b, w, stream) != 0) return -1;
  }
  // small + large magnitude (normalization range)
  {
    std::vector<__nv_bfloat16> a, b, w;
    fill_bf16(&a, rows, 0.01f, 0x00AAu);
    fill_bf16(&b, rows, 0.01f, 0x00BBu);
    fill_bf16(&w, static_cast<std::size_t>(H), 0.1f, 0x00CCu);
    if (run_case(M, H, eps_prod, a, b, w, stream) != 0) return -1;
    fill_bf16(&a, rows, 100.0f, 0x01AAu);
    fill_bf16(&b, rows, 100.0f, 0x01BBu);
    fill_bf16(&w, static_cast<std::size_t>(H), 2.0f, 0x01CCu);
    if (run_case(M, H, eps_prod, a, b, w, stream) != 0) return -1;
  }
  // all-zero a and b (residual = 0; rms of 0 row)
  {
    std::vector<__nv_bfloat16> a, b, w;
    fill_zero(&a, rows);
    fill_zero(&b, rows);
    fill_bf16(&w, static_cast<std::size_t>(H), 1.0f, 0x00D0u);
    if (run_case(M, H, eps_prod, a, b, w, stream) != 0) return -1;
  }
  // alternate eps
  {
    std::vector<__nv_bfloat16> a, b, w;
    fill_bf16(&a, rows, 1.0f, 0xE001u);
    fill_bf16(&b, rows, 1.0f, 0xE002u);
    fill_bf16(&w, static_cast<std::size_t>(H), 1.0f, 0xE003u);
    for (float eps : eps_alt)
      if (run_case(M, H, eps, a, b, w, stream) != 0) return -1;
  }
  return 0;
}

}  // namespace

int main() {
  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    std::fprintf(stderr, "failed to create stream\n");
    return 1;
  }
  int r = 0;
  // H=1024 (PER=4, production) and H=256 (PER=1); M=1 (B=1) and M=2,3 (batch).
  for (int H : {1024, 256}) {
    for (int M : {1, 2, 3}) {
      r |= (run_gate(M, H, stream) != 0);
    }
  }
  cudaStreamDestroy(stream);
  if (r != 0) {
    std::fprintf(stderr,
                 "fused add+rmsnorm qwen35: FAIL (%d gate failure(s))\n",
                 g_failures);
    return 1;
  }
  std::printf("fused add+rmsnorm qwen35: ALL PASS\n");
  return 0;
}
