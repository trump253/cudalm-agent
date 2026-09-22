// CUDALM — CUDA tests for the rope_v3_half2 fp16 port (interleaved pair).
//
// CPU references follow the golden math contract (fp32 math, fp16 RNE at the
// boundary — see docs/weight_format.md):
//   * the p = 0 case (cos row 0 == 1, sin row 0 == 0, exactly) is the rope
//     identity and is pinned BIT-EXACT via memcmp;
//   * p > 0 cases are compared with compare_fp16_stages (atol = rtol = 1e-2)
//     because fp32 FMA contraction between host and device may differ by
//     ~1 ulp;
//   * the scalar fallback path (2B-misaligned base pointers) is compared
//     against the CPU reference and the packed path with the same tolerance.

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/rope.h"
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

// CPU reference of the interleaved-pair RoPE contract (fp32 math, one fp16
// RNE per output element — the golden math contract).
void ref_rope(const __half* x, const std::int64_t* positions,
              const __half* cos_t, const __half* sin_t, __half* y, int M,
              int D) {
  const int d2 = D / 2;
  for (int m = 0; m < M; ++m) {
    const int pos = static_cast<int>(positions[m]);
    for (int i = 0; i < d2; ++i) {
      const float a = __half2float(x[m * D + 2 * i]);
      const float b = __half2float(x[m * D + 2 * i + 1]);
      const float c =
          __half2float(cos_t[static_cast<std::size_t>(pos) * d2 + i]);
      const float s =
          __half2float(sin_t[static_cast<std::size_t>(pos) * d2 + i]);
      y[m * D + 2 * i] = __float2half_rn(a * c - b * s);
      y[m * D + 2 * i + 1] = __float2half_rn(a * s + b * c);
    }
  }
}

struct RopeFixture {
  int M = 0, D = 0, L = 0;
  std::vector<__half> x, cos_t, sin_t;
  std::vector<std::int64_t> positions;
};

// Builds random data and, when identity_row0 is true, overwrites table row 0
// with cos == 1 / sin == 0 (the golden p = 0 pin). Distinct positions per row
// (except the pinned row 0) so a wrong position lookup is caught.
RopeFixture make_fixture(int M, int D, int L, std::uint32_t seed,
                         bool identity_row0) {
  RopeFixture f;
  f.M = M;
  f.D = D;
  f.L = L;
  const int d2 = D / 2;
  fill_half(&f.x, static_cast<std::size_t>(M) * D, 2.0f, seed);
  fill_half(&f.cos_t, static_cast<std::size_t>(L) * d2, 1.0f, seed + 1);
  fill_half(&f.sin_t, static_cast<std::size_t>(L) * d2, 1.0f, seed + 2);
  if (identity_row0) {
    for (int i = 0; i < d2; ++i) {
      f.cos_t[i] = __float2half_rn(1.0f);
      f.sin_t[i] = __float2half_rn(0.0f);
    }
  }
  f.positions.resize(M);
  for (int m = 0; m < M; ++m) {
    f.positions[m] =
        (identity_row0 && m == 0) ? 0 : (m * 17 + 5) % L;
  }
  return f;
}

// Runs rope_fp16 with the given half offsets into the x / y buffers (0 or 1;
// a 1-half offset is 2B-misaligned and routes to the scalar fallback path).
// x_data must have M*D + x_off elements; the kernel sees x_data[x_off ..].
// Returns the output view (M*D elements, from the offset y slot).
std::vector<__half> run_rope(const RopeFixture& f, const __half* x_data,
                             int x_off, int y_off, cudaStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(f.M) * f.D;
  DeviceBuffer dx((n + static_cast<std::size_t>(x_off)) * sizeof(__half),
                  stream);
  DeviceBuffer dy((n + static_cast<std::size_t>(y_off)) * sizeof(__half),
                  stream);
  DeviceBuffer dc(static_cast<std::size_t>(f.L) * (f.D / 2) * sizeof(__half),
                  stream);
  DeviceBuffer ds(static_cast<std::size_t>(f.L) * (f.D / 2) * sizeof(__half),
                  stream);
  DeviceBuffer dpos(f.positions.size() * sizeof(std::int64_t), stream);
  dx.copy_from_host(x_data, dx.bytes(), stream);
  dc.copy_from_host(f.cos_t.data(), dc.bytes(), stream);
  ds.copy_from_host(f.sin_t.data(), ds.bytes(), stream);
  dpos.copy_from_host(f.positions.data(), dpos.bytes(), stream);

  kernels::rope_fp16(dx.data<__half>() + x_off, dpos.data<std::int64_t>(),
                     dc.data<__half>(), ds.data<__half>(),
                     dy.data<__half>() + y_off, f.M, f.D, stream);

  std::vector<__half> ybuf(dy.bytes() / sizeof(__half));
  dy.copy_to_host(ybuf.data(), dy.bytes(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return std::vector<__half>(ybuf.begin() + y_off, ybuf.begin() + y_off + n);
}

// ---------------------------------------------------------------------------
// p = 0 identity pin (bit-exact): cos row 0 == 1, sin row 0 == 0.
// ---------------------------------------------------------------------------
int test_rope_identity_p0(cudaStream_t stream) {
  // q layout (n_heads = 8 rows) and k layout (n_kv_heads = 4 rows).
  const int layouts[][2] = {{8, 128}, {4, 128}};
  for (const auto& L : layouts) {
    RopeFixture f = make_fixture(L[0], L[1], 64, 0x1234, /*identity_row0=*/true);
    f.positions.assign(f.M, 0);  // every row at position 0
    std::vector<__half> y = run_rope(f, f.x.data(), 0, 0, stream);
    // Identity: y must equal x bit-for-bit (a*1 - b*0 == a exactly, fp32).
    CHECK(std::memcmp(y.data(), f.x.data(), y.size() * sizeof(__half)) == 0);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// p > 0 vs CPU reference (tolerance 1e-2), per-row position lookup.
// ---------------------------------------------------------------------------
int test_rope_random(cudaStream_t stream) {
  struct Case {
    int M, D, L, seed;
  };
  const Case cases[] = {
      {8, 128, 64, 0xa1},   // q layout, distinct positions per row
      {4, 128, 64, 0xb2},   // k layout
      {3, 128, 64, 0xc3},   // odd M
      {1, 2, 64, 0xd4},     // minimum shape (one pair)
  };
  for (const Case& c : cases) {
    RopeFixture f = make_fixture(c.M, c.D, c.L, c.seed, false);
    std::vector<__half> y = run_rope(f, f.x.data(), 0, 0, stream);

    std::vector<__half> ref(static_cast<std::size_t>(c.M) * c.D);
    ref_rope(f.x.data(), f.positions.data(), f.cos_t.data(), f.sin_t.data(),
             ref.data(), c.M, c.D);
    StageCompareResult r = compare_fp16_stages(
        reinterpret_cast<const std::uint16_t*>(ref.data()),
        reinterpret_cast<const std::uint16_t*>(y.data()), ref.size());
    CHECK(r.ok);
    if (!r.ok) {
      std::fprintf(stderr,
                   "rope random M=%d D=%d L=%d: max_abs_err=%.3e idx=%zu\n",
                   c.M, c.D, c.L, r.max_abs_err, r.max_abs_idx);
      return 1;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Scalar fallback: 2B-misaligned x and/or y base pointers must route to the
// scalar kernel and still match the CPU reference (legal inputs are never
// rejected). Same-data scalar-vs-packed agreement is checked under the stage
// tolerance.
// ---------------------------------------------------------------------------
int test_rope_scalar_fallback(cudaStream_t stream) {
  const int M = 8, D = 128, L = 64;
  const std::size_t n = static_cast<std::size_t>(M) * D;
  RopeFixture f = make_fixture(M, D, L, 0xe5, false);

  // Same shifted x data for every misaligned run below.
  std::vector<__half> xbuf(n + 1);
  std::memcpy(xbuf.data(), f.x.data(), n * sizeof(__half));
  Rng rng(0x5a1);
  xbuf[n] = __float2half_rn(rng.n01());  // slack element (read by off-1 view)

  // Packed path on the aligned data (reference for the cross-checks).
  std::vector<__half> y_packed = run_rope(f, f.x.data(), 0, 0, stream);

  // (a) x misaligned by 1 half (2B): scalar path on xbuf[1 .. M*D].
  {
    std::vector<__half> y = run_rope(f, xbuf.data(), 1, 0, stream);
    std::vector<__half> ref(n);
    ref_rope(xbuf.data() + 1, f.positions.data(), f.cos_t.data(),
             f.sin_t.data(), ref.data(), M, D);
    StageCompareResult r = compare_fp16_stages(
        reinterpret_cast<const std::uint16_t*>(ref.data()),
        reinterpret_cast<const std::uint16_t*>(y.data()), n);
    CHECK(r.ok);
    if (!r.ok) {
      std::fprintf(stderr, "  rope scalar (x off 1) vs ref: max_abs_err=%.3e\n",
                   r.max_abs_err);
      return 1;
    }
  }
  // (b) y misaligned by 1 half (2B): scalar path, output at y[1 .. M*D].
  {
    std::vector<__half> y = run_rope(f, f.x.data(), 0, 1, stream);
    std::vector<__half> ref(n);
    ref_rope(f.x.data(), f.positions.data(), f.cos_t.data(), f.sin_t.data(),
             ref.data(), M, D);
    StageCompareResult r = compare_fp16_stages(
        reinterpret_cast<const std::uint16_t*>(ref.data()),
        reinterpret_cast<const std::uint16_t*>(y.data()), n);
    CHECK(r.ok);
    if (!r.ok) {
      std::fprintf(stderr, "  rope scalar (y off 1) vs ref: max_abs_err=%.3e\n",
                   r.max_abs_err);
      return 1;
    }
  }
  // (c) same data through both paths: scalar on the +1-half x view vs packed
  //     on an aligned copy of that same data — must agree within tolerance.
  {
    std::vector<__half> xaligned(n);
    std::memcpy(xaligned.data(), xbuf.data() + 1, n * sizeof(__half));
    std::vector<__half> y_scalar = run_rope(f, xbuf.data(), 1, 0, stream);
    std::vector<__half> y_packed2 = run_rope(f, xaligned.data(), 0, 0, stream);

    StageCompareResult r = compare_fp16_stages(
        reinterpret_cast<const std::uint16_t*>(y_scalar.data()),
        reinterpret_cast<const std::uint16_t*>(y_packed2.data()), n);
    CHECK(r.ok);
    if (!r.ok) {
      std::fprintf(stderr,
                   "  rope scalar vs packed (same data): max_abs_err=%.3e\n",
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

  if (int rc = test_rope_identity_p0(stream)) return rc;
  if (int rc = test_rope_random(stream)) return rc;
  if (int rc = test_rope_scalar_fallback(stream)) return rc;

  CUDA_CHECK(cudaStreamDestroy(stream));
  TEST_PASS("test_rope");
  return 0;
}
