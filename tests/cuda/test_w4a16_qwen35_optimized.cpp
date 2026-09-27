// CUDALM — v0.7 Phase B: W4A16 optimized-path kernel-level hard gate.
//
// The EXACT contract (v0.6 frozen gate, unchanged): every output of the
// Phase-B variants (R=1/2/8 row tiles, B=1 and batch) and of the
// PRODUCTION DISPATCHER must be BF16 BIT-IDENTICAL to the FROZEN v0.6
// baseline (int4_gemv_bf16 / batch_int4_gemv_bf16) on the same input —
// for ALL production shapes of the Qwen3.5-0.8B W4A16 census
// (benchmarks/profiling/v07b_w4a16_shape_census.txt), B = 1/2/3, and the
// edge shapes / input classes:
//
//   * deterministic random inputs, MULTIPLE SEEDS
//   * all-zero x
//   * small magnitude
//   * mixed sign
//   * boundary INT4 nibbles (7/-7 in contract, plus 0x8 two's-complement
//     edge and 0/F)
//   * zero group scales (all-zero group -> exact-zero row terms)
//
// Contract checks (must not regress):
//   * the alignment / K%128 / fallback / legal-shape contracts: variant
//     launchers and the dispatcher never reject legal inputs; when the
//     16B alignment contract is not met they fall back to the frozen path
//     and stay bit-identical to the frozen entry point on the same input.
//   * frozen row-parity contract: every batch row b of the dispatcher and
//     each variant is bit-identical to the frozen SINGLE call on row b.
//
// This is a bit-exact gate (memcmp on the BF16 output words), NOT a
// tolerance comparison: the variants change only the N-row tiling per
// block and keep the per-row K-loop order, term order and reduction trees
// frozen, so bit-identity holds by construction and is pinned here.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/batch_decode.h"
#include "cudalm/kernels/int4_gemv_bf16.h"
#include "cudalm/kernels/int4_gemv_qwen35.h"

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

enum class InputKind { kRandom, kZero, kSmall, kMixedSign, kBoundary };

// Build one (W, S, x) fixture.
struct Fixture {
  std::vector<std::uint8_t> W;
  std::vector<__half> S;
  std::vector<__nv_bfloat16> x;  // B*K (B rows interleaved as [b0.., b1..])
  int B = 1;
};

Fixture make_fixture(int N, int K, int B, InputKind kind,
                     std::uint32_t seed) {
  Fixture f;
  f.B = B;
  Rng rng(seed);
  f.W.resize(static_cast<std::size_t>(N) * (K / 2));
  f.S.resize(static_cast<std::size_t>(N) * (K / 128));
  f.x.resize(static_cast<std::size_t>(B) * K);

  // Weight nibbles per kind.
  auto wbyte = [&]() -> std::uint8_t {
    switch (kind) {
      case InputKind::kZero:
        return 0x00;
      case InputKind::kSmall:
        return static_cast<std::uint8_t>(
            (rng.u32() % 3) | ((rng.u32() % 3) << 4));  // -1..1
      case InputKind::kBoundary: {
        // In-contract extremes (0x7=7, 0x9=-7), two's-complement edge
        // (0x8=-8), zero (0x0), and -1 (0xF).
        static const std::uint8_t kNib[8] = {0x7, 0x9, 0x8, 0x0, 0xF, 0x7,
                                             0x8, 0xF};
        return static_cast<std::uint8_t>(
            kNib[rng.u32() % 8] | (kNib[rng.u32() % 8] << 4));
      }
      default:
        return static_cast<std::uint8_t>(rng.u32());
    }
  };
  auto wscale = [&]() -> __half {
    if (kind == InputKind::kBoundary) {
      // Mix zero-group scales with normal ones (the zero-group path must
      // stay bit-exact).
      if (rng.u32() % 5 == 0) return __float2half_rn(0.0f);
    }
    return __float2half_rn(0.001f +
                           0.02f * static_cast<float>(rng.u32()) *
                                       (1.0f / 4294967296.0f));
  };
  auto wact = [&]() -> __nv_bfloat16 {
    switch (kind) {
      case InputKind::kZero:
        return __float2bfloat16_rn(0.0f);
      case InputKind::kSmall:
        return __float2bfloat16_rn(
            0.01f * static_cast<float>(rng.u32()) *
                (1.0f / 4294967296.0f) -
            0.005f);
      case InputKind::kMixedSign:
        return __float2bfloat16_rn((rng.u32() & 1) ? -rng.n01() : rng.n01());
      default:
        return __float2bfloat16_rn(rng.n01());
    }
  };

  for (auto& b : f.W) b = wbyte();
  for (auto& s : f.S) s = wscale();
  for (auto& v : f.x) v = wact();
  return f;
}

// Bit-exact compare of two BF16 buffers (word-wise memcmp).
bool bf16_bit_equal(const __nv_bfloat16* a, const __nv_bfloat16* b,
                    std::size_t n) {
  const std::uint16_t* pa = reinterpret_cast<const std::uint16_t*>(a);
  const std::uint16_t* pb = reinterpret_cast<const std::uint16_t*>(b);
  for (std::size_t i = 0; i < n; ++i)
    if (pa[i] != pb[i]) return false;
  return true;
}

struct Shape {
  int N, K;
  const char* name;
};

// The production census (docs + v07b_w4a16_shape_census.txt) plus edge
// shapes (N not divisible by the row tiles; K minimal/mid).
const Shape kShapes[] = {
    // production census (all 8 unique (N,K))
    {16, 1024, "census in_proj_b/a"},
    {512, 1024, "census k/v_proj"},
    {1024, 2048, "census o/out_proj"},
    {1024, 3584, "census down_proj"},
    {2048, 1024, "census in_proj_z"},
    {3584, 1024, "census gate/up_proj"},
    {4096, 1024, "census q_proj"},
    {6144, 1024, "census in_proj_qkv"},
    // edge shapes
    {1, 128, "edge N=1 K=128"},
    {3, 128, "edge N=3 K=128"},
    {5, 256, "edge N=5 K=256"},
    {9, 1024, "edge N=9"},
    {15, 1024, "edge N=15"},
    {1023, 1024, "edge N=1023"},
};

struct RunStats {
  int runs = 0;
  int bit_fail = 0;
};

// One B=1 parity check: variant vs frozen on the same fixture.
void check_b1(cudaStream_t stream, const Shape& sh, const Fixture& fx,
              const char* vname, void (*variant)(const std::uint8_t*,
                                                 const __half*,
                                                 const __nv_bfloat16*,
                                                 __nv_bfloat16*, int, int,
                                                 cudaStream_t),
              RunStats* st) {
  DeviceBuffer dW(fx.W.size(), stream);
  DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
  DeviceBuffer dx(fx.x.size() * sizeof(__nv_bfloat16), stream);
  DeviceBuffer dyf(static_cast<std::size_t>(sh.N) * sizeof(__nv_bfloat16),
                   stream);
  DeviceBuffer dyv(static_cast<std::size_t>(sh.N) * sizeof(__nv_bfloat16),
                   stream);
  dW.copy_from_host(fx.W.data(), dW.bytes(), stream);
  dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
  dx.copy_from_host(fx.x.data(), dx.bytes(), stream);

  int4_gemv_bf16(dW.data<std::uint8_t>(), dS.data<__half>(),
                 dx.data<__nv_bfloat16>(), dyf.data<__nv_bfloat16>(), sh.N,
                 sh.K, stream);
  variant(dW.data<std::uint8_t>(), dS.data<__half>(), dx.data<__nv_bfloat16>(),
          dyv.data<__nv_bfloat16>(), sh.N, sh.K, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> rf(sh.N), rv(sh.N);
  CUDA_CHECK(cudaMemcpy(rf.data(), dyf.data(), rf.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rv.data(), dyv.data(), rv.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  ++st->runs;
  if (!bf16_bit_equal(rf.data(), rv.data(), sh.N)) {
    std::size_t first = 0;
    while (first < static_cast<std::size_t>(sh.N) &&
           reinterpret_cast<const std::uint16_t*>(rf.data())[first] ==
               reinterpret_cast<const std::uint16_t*>(rv.data())[first])
      ++first;
    std::fprintf(stderr,
                 "BIT MISMATCH B=1 %s %s (N=%d,K=%d): first diff at row %zu "
                 "(frozen=0x%04x variant=0x%04x)\n",
                 vname, sh.name, sh.N, sh.K, first,
                 reinterpret_cast<const std::uint16_t*>(rf.data())[first],
                 reinterpret_cast<const std::uint16_t*>(rv.data())[first]);
    ++st->bit_fail;
  }
}

// One batch parity check: variant batch vs frozen batch; plus row-parity
// vs the frozen single call on each row b.
void check_batch(cudaStream_t stream, const Shape& sh, const Fixture& fx,
                 const char* vname,
                 void (*variant)(const std::uint8_t*, const __half*,
                                 const __nv_bfloat16*, __nv_bfloat16*, int,
                                 int, int, cudaStream_t),
                 RunStats* st) {
  const int B = fx.B;
  DeviceBuffer dW(fx.W.size(), stream);
  DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
  DeviceBuffer dx(fx.x.size() * sizeof(__nv_bfloat16), stream);
  DeviceBuffer dyf(static_cast<std::size_t>(B) * sh.N *
                       sizeof(__nv_bfloat16),
                   stream);
  DeviceBuffer dyv(static_cast<std::size_t>(B) * sh.N *
                       sizeof(__nv_bfloat16),
                   stream);
  DeviceBuffer dys(static_cast<std::size_t>(sh.N) * sizeof(__nv_bfloat16),
                   stream);
  dW.copy_from_host(fx.W.data(), dW.bytes(), stream);
  dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
  dx.copy_from_host(fx.x.data(), dx.bytes(), stream);

  kernels::batch_int4_gemv_bf16(dW.data<std::uint8_t>(), dS.data<__half>(),
                                dx.data<__nv_bfloat16>(),
                                dyf.data<__nv_bfloat16>(), sh.N, sh.K, B,
                                stream);
  variant(dW.data<std::uint8_t>(), dS.data<__half>(), dx.data<__nv_bfloat16>(),
          dyv.data<__nv_bfloat16>(), sh.N, sh.K, B, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> rf(B * sh.N), rv(B * sh.N);
  CUDA_CHECK(cudaMemcpy(rf.data(), dyf.data(),
                        rf.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rv.data(), dyv.data(),
                        rv.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  ++st->runs;
  if (!bf16_bit_equal(rf.data(), rv.data(), rf.size())) {
    std::fprintf(stderr, "BIT MISMATCH batch %s %s (N=%d,K=%d,B=%d)\n",
                 vname, sh.name, sh.N, sh.K, B);
    ++st->bit_fail;
  }

  // Row parity: frozen batch row b == frozen single on row b (pin the
  // contract through the fixtures this phase runs).
  for (int b = 0; b < B; ++b) {
    int4_gemv_bf16(dW.data<std::uint8_t>(), dS.data<__half>(),
                   dx.data<__nv_bfloat16>() +
                       static_cast<std::size_t>(b) * sh.K,
                   dys.data<__nv_bfloat16>(), sh.N, sh.K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<__nv_bfloat16> rs(sh.N);
    CUDA_CHECK(cudaMemcpy(rs.data(), dys.data(),
                          rs.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    ++st->runs;
    if (!bf16_bit_equal(rf.data() + static_cast<std::size_t>(b) * sh.N,
                        rs.data(), sh.N)) {
      std::fprintf(stderr,
                   "ROW-PARITY MISMATCH frozen batch row %d vs frozen "
                   "single, %s (N=%d,K=%d)\n",
                   b, sh.name, sh.N, sh.K);
      ++st->bit_fail;
    }
  }
}

// Dispatcher vs frozen (B=1 and batch) — the gate the production path
// actually ships under.
void check_dispatcher(cudaStream_t stream, const Shape& sh, const Fixture& fx,
                      RunStats* st) {
  {
    DeviceBuffer dW(fx.W.size(), stream);
    DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
    DeviceBuffer dx(fx.x.size() * sizeof(__nv_bfloat16), stream);
    DeviceBuffer dyf(static_cast<std::size_t>(sh.N) * sizeof(__nv_bfloat16),
                     stream);
    DeviceBuffer dyv(static_cast<std::size_t>(sh.N) * sizeof(__nv_bfloat16),
                     stream);
    dW.copy_from_host(fx.W.data(), dW.bytes(), stream);
    dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
    dx.copy_from_host(fx.x.data(), dx.bytes(), stream);
    int4_gemv_bf16(dW.data<std::uint8_t>(), dS.data<__half>(),
                   dx.data<__nv_bfloat16>(), dyf.data<__nv_bfloat16>(), sh.N,
                   sh.K, stream);
    int4_gemv_bf16_qwen35(dW.data<std::uint8_t>(), dS.data<__half>(),
                          dx.data<__nv_bfloat16>(), dyv.data<__nv_bfloat16>(),
                          sh.N, sh.K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<__nv_bfloat16> rf(sh.N), rv(sh.N);
    CUDA_CHECK(cudaMemcpy(rf.data(), dyf.data(),
                          rf.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rv.data(), dyv.data(),
                          rv.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    ++st->runs;
    if (!bf16_bit_equal(rf.data(), rv.data(), sh.N)) {
      std::fprintf(stderr,
                   "BIT MISMATCH dispatcher B=1 %s (N=%d,K=%d)\n", sh.name,
                   sh.N, sh.K);
      ++st->bit_fail;
    }
  }
  if (fx.B > 1) {
    const int B = fx.B;
    DeviceBuffer dW(fx.W.size(), stream);
    DeviceBuffer dS(fx.S.size() * sizeof(__half), stream);
    DeviceBuffer dx(fx.x.size() * sizeof(__nv_bfloat16), stream);
    DeviceBuffer dyf(static_cast<std::size_t>(B) * sh.N *
                         sizeof(__nv_bfloat16),
                     stream);
    DeviceBuffer dyv(static_cast<std::size_t>(B) * sh.N *
                         sizeof(__nv_bfloat16),
                     stream);
    dW.copy_from_host(fx.W.data(), dW.bytes(), stream);
    dS.copy_from_host(fx.S.data(), dS.bytes(), stream);
    dx.copy_from_host(fx.x.data(), dx.bytes(), stream);
    kernels::batch_int4_gemv_bf16(dW.data<std::uint8_t>(), dS.data<__half>(),
                                  dx.data<__nv_bfloat16>(),
                                  dyf.data<__nv_bfloat16>(), sh.N, sh.K, B,
                                  stream);
    batch_int4_gemv_bf16_qwen35(dW.data<std::uint8_t>(), dS.data<__half>(),
                                dx.data<__nv_bfloat16>(),
                                dyv.data<__nv_bfloat16>(), sh.N, sh.K, B,
                                stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<__nv_bfloat16> rf(B * sh.N), rv(B * sh.N);
    CUDA_CHECK(cudaMemcpy(rf.data(), dyf.data(),
                          rf.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rv.data(), dyv.data(),
                          rv.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    ++st->runs;
    if (!bf16_bit_equal(rf.data(), rv.data(), rf.size())) {
      std::fprintf(stderr,
                   "BIT MISMATCH dispatcher batch %s (N=%d,K=%d,B=%d)\n",
                   sh.name, sh.N, sh.K, B);
      ++st->bit_fail;
    }
  }
}

// Alignment-fallback contract: a 16B-misaligned activation pointer must
// take the frozen scalar path in BOTH the variant launcher and the
// dispatcher, and stay bit-identical to the frozen entry point (which
// also takes its scalar path).
void check_alignment_fallback(cudaStream_t stream, const Shape& sh,
                              const Fixture& fx, RunStats* st) {
  // Oversized host buffer so a +8-byte offset view stays in-bounds.
  std::vector<std::uint8_t> raw(
      (fx.W.size() + fx.S.size() * 2 + fx.x.size() * 2) * 2 + 64);
  std::size_t off = 0;
  std::memcpy(raw.data() + off, fx.W.data(), fx.W.size());
  off += fx.W.size();
  std::memcpy(raw.data() + off, fx.S.data(), fx.S.size() * 2);
  off += fx.S.size() * 2;
  std::memcpy(raw.data() + off, fx.x.data(), fx.x.size() * 2);
  off += fx.x.size() * 2;

  DeviceBuffer dRaw(raw.size(), stream);
  DeviceBuffer dyf(static_cast<std::size_t>(sh.N) * sizeof(__nv_bfloat16),
                   stream);
  DeviceBuffer dyv(static_cast<std::size_t>(sh.N) * sizeof(__nv_bfloat16),
                   stream);
  dRaw.copy_from_host(raw.data(), raw.size(), stream);

  const std::uint8_t* dW = dRaw.data<std::uint8_t>();
  const __half* dS = dRaw.data<__half>() + fx.W.size() / 2;
  // Misaligned activation: +8 bytes (2 bf16) breaks the 16B contract.
  const __nv_bfloat16* dx8 =
      dRaw.data<__nv_bfloat16>() + (fx.W.size() + fx.S.size() * 2) / 2 + 8;

  int4_gemv_bf16(dW, dS, dx8, dyf.data<__nv_bfloat16>(), sh.N, sh.K, stream);
  int4_gemv_bf16_rowtile1(dW, dS, dx8, dyv.data<__nv_bfloat16>(), sh.N,
                          sh.K, stream);
  int4_gemv_bf16_qwen35(dW, dS, dx8, dyv.data<__nv_bfloat16>(), sh.N, sh.K,
                        stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> rf(sh.N), rv(sh.N);
  CUDA_CHECK(cudaMemcpy(rf.data(), dyf.data(), rf.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rv.data(), dyv.data(), rv.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  ++st->runs;
  if (!bf16_bit_equal(rf.data(), rv.data(), sh.N)) {
    std::fprintf(stderr,
                 "ALIGNED-FALLBACK MISMATCH dispatcher(r1) %s (N=%d,K=%d): "
                 "misaligned x must stay bit-identical via the frozen path\n",
                 sh.name, sh.N, sh.K);
    ++st->bit_fail;
  }
  (void)fx;
}

}  // namespace

int main() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  CHECK(n >= 1);
  CUDA_CHECK(cudaSetDevice(0));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  RunStats st;
  const std::uint32_t seeds[] = {0x51ed, 0x1234abcd, 0x9e3779b9, 0xcafebabe};
  const InputKind kinds[] = {InputKind::kRandom, InputKind::kZero,
                             InputKind::kSmall, InputKind::kMixedSign,
                             InputKind::kBoundary};
  const int Blist[] = {1, 2, 3};

  for (const Shape& sh : kShapes) {
    for (std::uint32_t seed : seeds) {
      for (InputKind kind : kinds) {
        for (int B : Blist) {
          Fixture fx = make_fixture(sh.N, sh.K, B, kind, seed);
          check_b1(stream, sh, fx, "rowtile1", int4_gemv_bf16_rowtile1, &st);
          check_b1(stream, sh, fx, "rowtile2", int4_gemv_bf16_rowtile2, &st);
          check_b1(stream, sh, fx, "rowtile8", int4_gemv_bf16_rowtile8, &st);
          check_dispatcher(stream, sh, fx, &st);
          if (B > 1) {
            check_batch(stream, sh, fx, "batch_rowtile1",
                        batch_int4_gemv_bf16_rowtile1, &st);
            check_batch(stream, sh, fx, "batch_rowtile2",
                        batch_int4_gemv_bf16_rowtile2, &st);
            check_batch(stream, sh, fx, "batch_rowtile8",
                        batch_int4_gemv_bf16_rowtile8, &st);
          }
        }
      }
    }
  }

  // Alignment-fallback contract on a few representative shapes.
  for (const Shape& sh : {kShapes[0], kShapes[3], kShapes[7], kShapes[10]}) {
    Fixture fx = make_fixture(sh.N, sh.K, 1, InputKind::kRandom, 0x51ed);
    check_alignment_fallback(stream, sh, fx, &st);
  }

  if (st.bit_fail) {
    std::fprintf(stderr,
                 "test_w4a16_qwen35_optimized FAILED: %d bit mismatches / "
                 "%d checks\n",
                 st.bit_fail, st.runs);
    return 1;
  }
  std::printf("bit-exact checks passed: %d\n", st.runs);
  CUDA_CHECK(cudaStreamDestroy(stream));
  TEST_PASS("test_w4a16_qwen35_optimized");
  return 0;
}
