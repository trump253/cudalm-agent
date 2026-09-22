// CUDALM — CUDA tests for the elementwise ops (add_fp16, silu_mul_fp16) and
// the rmsnorm_v4 fp16 port.
//
// CPU references follow the golden math contract (fp32 math, fp16 RNE at the
// boundary — see docs/weight_format.md):
//   * add_fp16 is pinned BIT-EXACT (one fp32 add + one RNE, same on both
//     sides) via memcmp;
//   * silu_mul_fp16 and rmsnorm_fp16 are compared with
//     compare_fp16_stages (atol = rtol = 1e-2): host expf/rsqrtf and fp32
//     summation order may differ from the device by ~1 ulp.

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/elementwise.h"
#include "cudalm/kernels/rmsnorm.h"
#include "cudalm/stage_compare.h"

#include "../../tests/common/check.h"

using namespace cudalm;

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

void fill_half(std::vector<__half>* out, std::size_t n, float scale,
               std::uint32_t seed) {
  Rng rng(seed);
  out->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    (*out)[i] = __float2half_rn(rng.n01() * scale);
  }
}

// ---------------------------------------------------------------------------
// add_fp16 (bit-exact pin)
// ---------------------------------------------------------------------------
int test_add_fp16(cudaStream_t stream) {
  const std::size_t n = 1024;
  std::vector<__half> ha, hb;
  fill_half(&ha, n, 2.0f, 0xa1);
  fill_half(&hb, n, 2.0f, 0xb2);

  DeviceBuffer da(n * sizeof(__half), stream);
  DeviceBuffer db(n * sizeof(__half), stream);
  DeviceBuffer dy(n * sizeof(__half), stream);
  da.copy_from_host(ha.data(), da.bytes(), stream);
  db.copy_from_host(hb.data(), db.bytes(), stream);
  add_fp16(da.data<__half>(), db.data<__half>(), dy.data<__half>(), n, stream);

  std::vector<__half> y(n);
  dy.copy_to_host(y.data(), dy.bytes(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // CPU reference: fp32 add, fp16 RNE — must match the kernel bit-for-bit.
  std::vector<__half> ref(n);
  for (std::size_t i = 0; i < n; ++i) {
    ref[i] = __float2half_rn(__half2float(ha[i]) + __half2float(hb[i]));
  }
  CHECK(std::memcmp(y.data(), ref.data(), n * sizeof(__half)) == 0);
  return 0;
}

// ---------------------------------------------------------------------------
// silu_mul_fp16
// ---------------------------------------------------------------------------
int test_silu_mul_fp16(cudaStream_t stream) {
  const std::size_t sizes[] = {1024, 2816};  // H and intermediate of v0.1
  for (const std::size_t n : sizes) {
    std::vector<__half> g, u;
    fill_half(&g, n, 4.0f, 0xc3 + static_cast<std::uint32_t>(n));
    fill_half(&u, n, 4.0f, 0xd4 + static_cast<std::uint32_t>(n));

    DeviceBuffer dg(n * sizeof(__half), stream);
    DeviceBuffer du(n * sizeof(__half), stream);
    DeviceBuffer dy(n * sizeof(__half), stream);
    dg.copy_from_host(g.data(), dg.bytes(), stream);
    du.copy_from_host(u.data(), du.bytes(), stream);
    silu_mul_fp16(dg.data<__half>(), du.data<__half>(), dy.data<__half>(), n,
                  stream);

    std::vector<__half> y(n);
    dy.copy_to_host(y.data(), dy.bytes(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // CPU reference (fp32 silu·mul, fp16 RNE).
    std::vector<__half> ref(n);
    for (std::size_t i = 0; i < n; ++i) {
      const float gv = __half2float(g[i]);
      const float sv = gv / (1.0f + expf(-gv));
      ref[i] = __float2half_rn(sv * __half2float(u[i]));
    }
    StageCompareResult r = compare_fp16_stages(
        reinterpret_cast<const std::uint16_t*>(ref.data()),
        reinterpret_cast<const std::uint16_t*>(y.data()), n);
    if (!r.ok) {
      std::fprintf(stderr, "  silu_mul n=%zu: max_abs_err=%.9g idx=%llu\n", n,
                   r.max_abs_err, static_cast<unsigned long long>(r.max_abs_idx));
      return 1;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// rmsnorm_fp16 (port of CUDALab rmsnorm_v4)
// ---------------------------------------------------------------------------
int test_rmsnorm_fp16(cudaStream_t stream) {
  const int H = 1024;  // v0.1: PER = 4 half2 path
  const int Ms[] = {1, 5};
  for (const int M : Ms) {
    std::vector<__half> x, w;
    fill_half(&x, static_cast<std::size_t>(M) * H, 1.0f, 0xe5 + M);
    fill_half(&w, static_cast<std::size_t>(H), 1.0f, 0xf6 + M);

    DeviceBuffer dx(static_cast<std::size_t>(M) * H * sizeof(__half), stream);
    DeviceBuffer dw(static_cast<std::size_t>(H) * sizeof(__half), stream);
    DeviceBuffer dy(static_cast<std::size_t>(M) * H * sizeof(__half), stream);
    dx.copy_from_host(x.data(), dx.bytes(), stream);
    dw.copy_from_host(w.data(), dw.bytes(), stream);
    const float eps = 1e-5f;
    rmsnorm_fp16(dx.data<__half>(), dw.data<__half>(), dy.data<__half>(), M, H,
                 eps, stream);

    std::vector<__half> y(static_cast<std::size_t>(M) * H);
    dy.copy_to_host(y.data(), dy.bytes(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // CPU reference: fp32 mean-of-squares (sequential), rsqrtf, fp16 RNE.
    // Sum order differs from the kernel's warp reduction → ~1 ulp fp32
    // difference, covered by the 1e-2 stage tolerance.
    std::vector<__half> ref(static_cast<std::size_t>(M) * H);
    for (int row = 0; row < M; ++row) {
      const __half* xr = x.data() + static_cast<std::size_t>(row) * H;
      float ss = 0.f;
      for (int k = 0; k < H; ++k) {
        const float f = __half2float(xr[k]);
        ss += f * f;
      }
      // (Host C++ has no rsqrtf; 1/sqrt differs from rsqrtf by at most an
      // fp32 ulp — covered by the 1e-2 stage tolerance.)
      const float inv = 1.0f / std::sqrt(ss / static_cast<float>(H) + eps);
      __half* refrow = ref.data() + static_cast<std::size_t>(row) * H;
      for (int k = 0; k < H; ++k) {
        refrow[k] = __float2half_rn(__half2float(xr[k]) * inv *
                                    __half2float(w[static_cast<std::size_t>(k)]));
      }
      StageCompareResult cmp = compare_fp16_stages(
          reinterpret_cast<const std::uint16_t*>(refrow),
          reinterpret_cast<const std::uint16_t*>(
              y.data() + static_cast<std::size_t>(row) * H),
          static_cast<std::size_t>(H));
      if (!cmp.ok) {
        std::fprintf(stderr, "  rmsnorm M=%d row=%d: max_abs_err=%.9g idx=%llu\n",
                     M, row, cmp.max_abs_err,
                     static_cast<unsigned long long>(cmp.max_abs_idx));
        return 1;
      }
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

  if (int rc = test_add_fp16(stream)) return rc;
  if (int rc = test_silu_mul_fp16(stream)) return rc;
  if (int rc = test_rmsnorm_fp16(stream)) return rc;

  CUDA_CHECK(cudaStreamDestroy(stream));
  TEST_PASS("test_elementwise_rmsnorm");
  return 0;
}
