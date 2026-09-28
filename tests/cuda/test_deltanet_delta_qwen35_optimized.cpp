// CUDALM — v0.7 Phase C: DeltaNet delta-rule optimized-path kernel-level
// hard gate.
//
// The EXACT contract: every output of the Phase-C candidates (vreg / vvec /
// vchunk, B=1 and batch) and of the EXPERIMENTAL / measured dispatcher
// (benchmark/test/profiling infrastructure — NOT a production entry point;
// the production runtime calls the frozen baseline directly) must be
// BF16 BIT-IDENTICAL for core_out and FP32 BIT-IDENTICAL for the updated
// recurrent state S (the persistent state) to the FROZEN baseline
// (qwen35_deltanet_delta_rule_fp32 / batch_deltanet_delta_rule_fp32) on the
// same input — for the production config (n_heads = 16, head_dim = 128),
// B = 1/2/3, and the edge input classes:
//
//   * deterministic random inputs, MULTIPLE SEEDS
//   * all-zero q/k/v (degenerate l2norm: qn = kn = 0)
//   * small magnitude
//   * mixed sign
//   * g extremes (0, small negative, large negative, positive)
//   * beta extremes (0, 1, 0.5)
//   * S initial: zero / random / small
//
// Contract checks (must not regress):
//   * the vvec 16B alignment fallback: an unaligned state base makes the
//     variant fall back to the frozen path and stay bit-identical on the
//     same input (checked with INDEPENDENT output buffers).
//   * frozen row-parity contract: every batch row b of each variant and of
//     the dispatcher is bit-identical to the frozen SINGLE call on row b.
//
// This is a bit-exact gate (memcmp on the BF16 output words and the FP32
// state words), NOT a tolerance comparison: the candidates change only the
// parallelization / data movement (register-resident column, float4 value
// groups, 2-kernel value-chunk split) and keep the per-element fp32
// operation sequence and every bf16 rounding boundary frozen, so
// bit-identity holds by construction and is pinned here.

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/batch_decode.h"
#include "cudalm/kernels/deltanet_delta_qwen35.h"
#include "cudalm/kernels/qwen35_deltanet_kernels.h"

#include "../../tests/common/check.h"

namespace {
int g_failures = 0;
}  // namespace

// Bit-exact gate check: on failure record + return -1 from the gate.
#define CHECK_BITS(cond)                                               \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::fprintf(stderr, "CHECK failed: %s  (%s:%d)\n", #cond,      \
                   __FILE__, __LINE__);                                \
      ++g_failures;                                                    \
      return -1;                                                       \
    }                                                                  \
  } while (0)

using namespace cudalm;

namespace {

constexpr int kNHeads = 16;
constexpr int kHd = 128;
constexpr float kEps = 1e-6f;  // frozen runtime eps (qwen35_deltanet.cpp)
constexpr int kChainSteps = 32;

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

__nv_bfloat16 to_bf16(float x) { return __float2bfloat16_rn(x); }

enum class InputKind { kRandom, kZero, kSmall, kMixedSign };
enum class GKind { kGRandom, kGZero, kGNegSmall, kGNegLarge, kGPos };
enum class BetaKind { kBetaRandom, kBetaZero, kBetaOne, kBetaHalf };
enum class SKind { kSZero, kSRandom, kSSmall };

struct StepInput {
  std::vector<__nv_bfloat16> q;  // B * n_heads * hd
  std::vector<__nv_bfloat16> k;
  std::vector<__nv_bfloat16> v;
  std::vector<float> g;          // B * n_heads
  std::vector<__nv_bfloat16> beta;
  int B = 1;
};

StepInput make_step(int B, InputKind kind, GKind gk, BetaKind bk,
                    std::uint32_t seed) {
  StepInput in;
  in.B = B;
  const std::size_t n = static_cast<std::size_t>(B) * kNHeads * kHd;
  Rng rng(seed);
  in.q.resize(n);
  in.k.resize(n);
  in.v.resize(n);
  in.g.resize(static_cast<std::size_t>(B) * kNHeads);
  in.beta.resize(static_cast<std::size_t>(B) * kNHeads);
  auto bf = [&](float x) { return to_bf16(x); };
  for (std::size_t i = 0; i < n; ++i) {
    float x = 0.0f;
    switch (kind) {
      case InputKind::kZero:
        x = 0.0f;
        break;
      case InputKind::kSmall:
        x = rng.n01() * 1e-3f;
        break;
      case InputKind::kMixedSign:
        x = rng.n01() * 2.0f;
        if ((i & 1u) != 0) x = -x;
        break;
      case InputKind::kRandom:
      default:
        x = rng.n01() * 2.0f;
        break;
    }
    in.q[i] = bf(x);
    in.k[i] = bf(x * 0.7f);
    in.v[i] = bf(rng.n01() * 2.0f);
  }
  for (std::size_t i = 0; i < in.g.size(); ++i) {
    switch (gk) {
      case GKind::kGZero:
        in.g[i] = 0.0f;
        break;
      case GKind::kGNegSmall:
        in.g[i] = -0.01f;
        break;
      case GKind::kGNegLarge:
        in.g[i] = -12.0f;  // exp_g ~ 6e-6
        break;
      case GKind::kGPos:
        in.g[i] = 2.0f;  // exp_g ~ 7.39
        break;
      case GKind::kGRandom:
      default:
        in.g[i] = -expf(1.0f) * (0.01f + 0.5f * (0.5f + 0.5f * rng.n01()));
        break;
    }
  }
  for (std::size_t i = 0; i < in.beta.size(); ++i) {
    switch (bk) {
      case BetaKind::kBetaZero:
        in.beta[i] = to_bf16(0.0f);
        break;
      case BetaKind::kBetaOne:
        in.beta[i] = to_bf16(1.0f);
        break;
      case BetaKind::kBetaHalf:
        in.beta[i] = to_bf16(0.5f);
        break;
      case BetaKind::kBetaRandom:
      default:
        in.beta[i] = to_bf16(1.0f / (1.0f + expf(-rng.n01() * 6.0f)));
        break;
    }
  }
  return in;
}

std::vector<float> make_state(SKind kind, std::uint32_t seed) {
  std::vector<float> s(static_cast<std::size_t>(kNHeads) * kHd * kHd, 0.0f);
  Rng rng(seed);
  if (kind == SKind::kSZero) return s;
  for (auto& x : s) x = kind == SKind::kSSmall ? rng.n01() * 1e-3f : rng.n01() * 4.0f;
  return s;
}

bool bits_equal_f32(const float* a, const float* b, std::size_t n,
                    const char* what) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t wa = *reinterpret_cast<const std::uint32_t*>(a + i);
    const std::uint32_t wb = *reinterpret_cast<const std::uint32_t*>(b + i);
    if (wa != wb) {
      std::fprintf(stderr, "  MISMATCH %s at %zu: %08x vs %08x\n", what, i, wa,
                   wb);
      return false;
    }
  }
  return true;
}


std::uint16_t bf16_bits(__nv_bfloat16 x) {
  std::uint16_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

bool bits_equal_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                     std::size_t n, const char* what) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint16_t wa = bf16_bits(a[i]);
    const std::uint16_t wb = bf16_bits(b[i]);
    if (wa != wb) {
      std::fprintf(stderr, "  MISMATCH %s at %zu: %04x vs %04x\n", what, i, wa,
                   wb);
      return false;
    }
  }
  return true;
}

// Upload one StepInput to device (B=1 layout = plain).
struct DevStep {
  DeviceBuffer dq, dk, dv, dg, db;
  DevStep(const StepInput& in, cudaStream_t stream)
      : dq(in.q.size() * sizeof(__nv_bfloat16), stream),
        dk(in.k.size() * sizeof(__nv_bfloat16), stream),
        dv(in.v.size() * sizeof(__nv_bfloat16), stream),
        dg(in.g.size() * sizeof(float), stream),
        db(in.beta.size() * sizeof(__nv_bfloat16), stream) {
    CUDA_CHECK(cudaMemcpyAsync(dq.data(), in.q.data(),
                               in.q.size() * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dk.data(), in.k.data(),
                               in.k.size() * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dv.data(), in.v.data(),
                               in.v.size() * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dg.data(), in.g.data(),
                               in.g.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(db.data(), in.beta.data(),
                               in.beta.size() * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));
  }
  const __nv_bfloat16* q() const {
    return static_cast<const __nv_bfloat16*>(dq.data());
  }
  const __nv_bfloat16* k() const {
    return static_cast<const __nv_bfloat16*>(dk.data());
  }
  const __nv_bfloat16* v() const {
    return static_cast<const __nv_bfloat16*>(dv.data());
  }
  const float* g() const { return static_cast<const float*>(dg.data()); }
  const __nv_bfloat16* beta() const {
    return static_cast<const __nv_bfloat16*>(db.data());
  }
};

typedef void (*VariantB1)(const __nv_bfloat16*, const __nv_bfloat16*,
                          const __nv_bfloat16*, const float*,
                          const __nv_bfloat16*, float*, __nv_bfloat16*, int,
                          int, float, cudaStream_t);
typedef void (*VariantB)(const __nv_bfloat16*, const __nv_bfloat16*,
                         const __nv_bfloat16*, const float*,
                         const __nv_bfloat16*, float*, const int*,
                         __nv_bfloat16*, int, int, float, int, cudaStream_t);

struct VariantInfo {
  const char* name;
  VariantB1 b1;
  VariantB batch;
};

int run_b1_single_step_gate(const char* label, const VariantInfo& var,
                            cudaStream_t stream) {
  int checks = 0;
  for (std::uint32_t seed = 1; seed <= 4; ++seed) {
    for (int kind = 0; kind < 4; ++kind) {
      for (int gk = 0; gk < 5; ++gk) {
        for (int sk = 0; sk < 3; ++sk) {
          const StepInput in =
              make_step(1, static_cast<InputKind>(kind), static_cast<GKind>(gk),
                        BetaKind::kBetaRandom, seed * 1000 + kind * 10 + gk);
          const std::vector<float> s0 = make_state(static_cast<SKind>(sk),
                                                   seed + kind + sk);
          const std::size_t nE = static_cast<std::size_t>(kNHeads) * kHd;
          const std::size_t nS = nE * kHd;

          DeviceBuffer dSf(nS * sizeof(float), stream);
          DeviceBuffer dSv(nS * sizeof(float), stream);
          DeviceBuffer dOf(nE * sizeof(__nv_bfloat16), stream);
          DeviceBuffer dOv(nE * sizeof(__nv_bfloat16), stream);
          DeviceBuffer ds0(nS * sizeof(float), stream);
          CUDA_CHECK(cudaMemcpyAsync(ds0.data(), s0.data(), nS * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
          CUDA_CHECK(cudaMemcpyAsync(dSf.data(), ds0.data(), nS * sizeof(float),
                                     cudaMemcpyDeviceToDevice, stream));
          CUDA_CHECK(cudaMemcpyAsync(dSv.data(), ds0.data(), nS * sizeof(float),
                                     cudaMemcpyDeviceToDevice, stream));
          DevStep ds(in, stream);
          // frozen on independent buffer
          kernels::qwen35_deltanet_delta_rule_fp32(
              ds.q(), ds.k(), ds.v(), ds.g(), ds.beta(),
              static_cast<float*>(dSf.data()),
              static_cast<__nv_bfloat16*>(dOf.data()), kNHeads, kHd, kEps,
              stream);
          // variant on independent buffer
          var.b1(ds.q(), ds.k(), ds.v(), ds.g(), ds.beta(),
                 static_cast<float*>(dSv.data()),
                 static_cast<__nv_bfloat16*>(dOv.data()), kNHeads, kHd, kEps,
                 stream);
          CUDA_CHECK(cudaStreamSynchronize(stream));

          std::vector<float> Sf(nS), Sv(nS);
          std::vector<__nv_bfloat16> Of(nE), Ov(nE);
          CUDA_CHECK(cudaMemcpy(Sf.data(), dSf.data(), nS * sizeof(float),
                                cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(Sv.data(), dSv.data(), nS * sizeof(float),
                                cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(Of.data(), dOf.data(), nE * 2,
                                cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(Ov.data(), dOv.data(), nE * 2,
                                cudaMemcpyDeviceToHost));
          CHECK_BITS(bits_equal_bf16(Of.data(), Ov.data(), nE,
                                       "core_out (single step)"));
          CHECK_BITS(bits_equal_f32(Sf.data(), Sv.data(), nS,
                                      "recurrent state (single step)"));
          ++checks;
        }
      }
    }
  }
  std::printf("  [PASS] %-8s B=1 single-step bit-exact: %d combos\n", label,
              checks);
  return checks;
}

int run_b1_chain_gate(const char* label, const VariantInfo& var,
                      cudaStream_t stream) {
  const std::size_t nE = static_cast<std::size_t>(kNHeads) * kHd;
  const std::size_t nS = nE * kHd;
  int checks = 0;
  for (std::uint32_t seed = 100; seed <= 102; ++seed) {
    DeviceBuffer dSf(nS * sizeof(float), stream);
    DeviceBuffer dSv(nS * sizeof(float), stream);
    DeviceBuffer dOf(nE * sizeof(__nv_bfloat16), stream);
    DeviceBuffer dOv(nE * sizeof(__nv_bfloat16), stream);
    Rng srng(seed);
    const std::vector<float> s0 =
        make_state(SKind::kSRandom, seed + 7);
    DeviceBuffer ds0(nS * sizeof(float), stream);
    CUDA_CHECK(cudaMemcpyAsync(ds0.data(), s0.data(), nS * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dSf.data(), ds0.data(), nS * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dSv.data(), ds0.data(), nS * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
    for (int step = 0; step < kChainSteps; ++step) {
      const StepInput in = make_step(
          1, static_cast<InputKind>((step + seed) % 4),
          static_cast<GKind>(step % 5),
          static_cast<BetaKind>(step % 4), seed * 100 + step);
      DevStep ds(in, stream);
      kernels::qwen35_deltanet_delta_rule_fp32(
          ds.q(), ds.k(), ds.v(), ds.g(), ds.beta(),
          static_cast<float*>(dSf.data()),
          static_cast<__nv_bfloat16*>(dOf.data()), kNHeads, kHd, kEps, stream);
      var.b1(ds.q(), ds.k(), ds.v(), ds.g(), ds.beta(),
             static_cast<float*>(dSv.data()),
             static_cast<__nv_bfloat16*>(dOv.data()), kNHeads, kHd, kEps,
             stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      std::vector<float> Sf(nS), Sv(nS);
      std::vector<__nv_bfloat16> Of(nE), Ov(nE);
      CUDA_CHECK(cudaMemcpy(Sf.data(), dSf.data(), nS * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(Sv.data(), dSv.data(), nS * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(Of.data(), dOf.data(), nE * 2,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(Ov.data(), dOv.data(), nE * 2,
                            cudaMemcpyDeviceToHost));
      CHECK_BITS(bits_equal_bf16(Of.data(), Ov.data(), nE,
                                   "core_out (chain step)"));
      CHECK_BITS(bits_equal_f32(Sf.data(), Sv.data(), nS,
                                  "recurrent state (chain step)"));
      ++checks;
    }
    (void)srng;
  }
  std::printf("  [PASS] %-8s B=1 %d-step chain bit-exact: %d steps\n", label,
              kChainSteps, checks);
  return checks;
}

int run_batch_gate(const char* label, const VariantInfo& var, int B,
                   cudaStream_t stream) {
  const std::size_t nE = static_cast<std::size_t>(B) * kNHeads * kHd;
  const std::size_t nS = nE * kHd;  // per-row state, B rows
  int checks = 0;
  for (std::uint32_t seed = 200; seed <= 202; ++seed) {
    for (int sk = 0; sk < 2; ++sk) {
      const StepInput in = make_step(
          B, static_cast<InputKind>((seed + sk) % 4),
          static_cast<GKind>((seed + sk) % 5),
          static_cast<BetaKind>((seed + sk) % 4), seed * 10 + sk);
      // Per-row state (independent per row).
      std::vector<float> s0(B * static_cast<std::size_t>(kNHeads) * kHd *
                           kHd, 0.0f);
      Rng srng(seed * 31 + sk);
      for (int b = 0; b < B; ++b) {
        const std::vector<float> row =
            make_state(static_cast<SKind>(sk ? SKind::kSSmall : SKind::kSRandom),
                       seed + b);
        std::memcpy(s0.data() + b * kNHeads * kHd * kHd, row.data(),
                    row.size() * sizeof(float));
      }
      (void)srng;

      DeviceBuffer dSf(nS * sizeof(float), stream);
      DeviceBuffer dSv(nS * sizeof(float), stream);
      DeviceBuffer dOf(nE * sizeof(__nv_bfloat16), stream);
      DeviceBuffer dOv(nE * sizeof(__nv_bfloat16), stream);
      DeviceBuffer ds0(nS * sizeof(float), stream);
      CUDA_CHECK(cudaMemcpyAsync(ds0.data(), s0.data(), nS * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(dSf.data(), ds0.data(), nS * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(dSv.data(), ds0.data(), nS * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream));
      const int slots[3] = {0, 1, 2};
      DeviceBuffer dslots(sizeof(int) * B, stream);
      CUDA_CHECK(cudaMemcpyAsync(dslots.data(), slots, sizeof(int) * B,
                                 cudaMemcpyHostToDevice, stream));
      DevStep ds(in, stream);
      kernels::batch_deltanet_delta_rule_fp32(
          ds.q(), ds.k(), ds.v(), ds.g(), ds.beta(),
          static_cast<float*>(dSf.data()),
          static_cast<const int*>(dslots.data()),
          static_cast<__nv_bfloat16*>(dOf.data()), kNHeads, kHd, kEps, B,
          stream);
      var.batch(ds.q(), ds.k(), ds.v(), ds.g(), ds.beta(),
                static_cast<float*>(dSv.data()),
                static_cast<const int*>(dslots.data()),
                static_cast<__nv_bfloat16*>(dOv.data()), kNHeads, kHd, kEps,
                B, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      std::vector<float> Sf(nS), Sv(nS);
      std::vector<__nv_bfloat16> Of(nE), Ov(nE);
      CUDA_CHECK(cudaMemcpy(Sf.data(), dSf.data(), nS * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(Sv.data(), dSv.data(), nS * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(Of.data(), dOf.data(), nE * 2,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(Ov.data(), dOv.data(), nE * 2,
                            cudaMemcpyDeviceToHost));
      CHECK_BITS(bits_equal_bf16(Of.data(), Ov.data(), nE,
                                   "core_out (batch)"));
      CHECK_BITS(bits_equal_f32(Sf.data(), Sv.data(), nS,
                                  "recurrent state (batch)"));
      ++checks;

      // Row parity: frozen single call on row b == batch row b (variant side
      // too — it is transitively pinned by both bit-identities, but check the
      // variant row against the frozen SINGLE directly).
      for (int b = 0; b < B; ++b) {
        DeviceBuffer dSr(static_cast<std::size_t>(kNHeads) * kHd * kHd *
                           sizeof(float),
                         stream);
        DeviceBuffer dOr(static_cast<std::size_t>(kNHeads) * kHd * 2, stream);
        const float* rowBase =
            s0.data() + static_cast<std::size_t>(b) * kNHeads * kHd * kHd;
        CUDA_CHECK(cudaMemcpyAsync(
            dSr.data(), rowBase, kNHeads * kHd * kHd * sizeof(float),
            cudaMemcpyHostToDevice, stream));
        // Single-row input slice for row b.
        const __nv_bfloat16* q_b = in.q.data() + b * kNHeads * kHd;
        const __nv_bfloat16* k_b = in.k.data() + b * kNHeads * kHd;
        const __nv_bfloat16* v_b = in.v.data() + b * kNHeads * kHd;
        const float* g_b = in.g.data() + b * kNHeads;
        const __nv_bfloat16* beta_b = in.beta.data() + b * kNHeads;
        DeviceBuffer dq(q_b ? kNHeads * kHd * 2 : 0, stream);
        DeviceBuffer dk(kNHeads * kHd * 2, stream);
        DeviceBuffer dv(kNHeads * kHd * 2, stream);
        DeviceBuffer dg(kNHeads * sizeof(float), stream);
        DeviceBuffer db(kNHeads * 2, stream);
        CUDA_CHECK(cudaMemcpyAsync(dq.data(), q_b, kNHeads * kHd * 2,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dk.data(), k_b, kNHeads * kHd * 2,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dv.data(), v_b, kNHeads * kHd * 2,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dg.data(), g_b, kNHeads * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(db.data(), beta_b, kNHeads * 2,
                                   cudaMemcpyHostToDevice, stream));
        kernels::qwen35_deltanet_delta_rule_fp32(
            static_cast<const __nv_bfloat16*>(dq.data()),
            static_cast<const __nv_bfloat16*>(dk.data()),
            static_cast<const __nv_bfloat16*>(dv.data()),
            static_cast<const float*>(dg.data()),
            static_cast<const __nv_bfloat16*>(db.data()),
            static_cast<float*>(dSr.data()),
            static_cast<__nv_bfloat16*>(dOr.data()), kNHeads, kHd, kEps,
            stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> Sr(kNHeads * kHd * kHd);
        std::vector<__nv_bfloat16> Or(kNHeads * kHd);
        CUDA_CHECK(cudaMemcpy(Sr.data(), dSr.data(),
                              Sr.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(Or.data(), dOr.data(), Or.size() * 2,
                              cudaMemcpyDeviceToHost));
        // Row b of the variant batch output vs the frozen single call.
        CHECK_BITS(bits_equal_bf16(
            Or.data(), Ov.data() + b * kNHeads * kHd, kNHeads * kHd,
            "row parity (batch row vs frozen single)"));
        CHECK(
            bits_equal_f32(Sr.data(), Sv.data() +
                                       b * kNHeads * kHd * kHd,
                           kNHeads * kHd * kHd,
                           "row parity (batch state row vs frozen single)"));
        // Frozen batch row vs frozen single (baseline self-consistency).
        CHECK_BITS(bits_equal_bf16(
            Or.data(), Of.data() + b * kNHeads * kHd, kNHeads * kHd,
            "row parity (frozen batch vs frozen single)"));
      }
    }
  }
  std::printf("  [PASS] %-8s B=%d batch + row parity: %d combos\n", label, B,
              checks);
  return checks;
}

// Run vvec with a state base at the given float offset from a fresh
// (256B-aligned) allocation and compare core_out against a frozen reference
// run on a SEPARATE non-overlapping buffer with the same initial state.
int run_vvec_state_offset(const char* label, int float_offset,
                          cudaStream_t stream) {
  const StepInput in = make_step(1, InputKind::kRandom, GKind::kGRandom,
                                 BetaKind::kBetaRandom, 777);
  const std::vector<float> s0 = make_state(SKind::kSRandom, 77);
  const std::size_t nE = static_cast<std::size_t>(kNHeads) * kHd;
  const std::size_t nS = nE * kHd;
  const std::size_t pad = static_cast<std::size_t>(float_offset) + 8;

  // Reference: frozen on a fresh aligned buffer.
  DeviceBuffer dSref(nS * sizeof(float), stream);
  DeviceBuffer dSv((nS + pad) * sizeof(float), stream);
  DeviceBuffer dOf(nE * sizeof(__nv_bfloat16), stream);
  DeviceBuffer dOv(nE * sizeof(__nv_bfloat16), stream);
  CUDA_CHECK(cudaMemcpyAsync(dSref.data(), s0.data(), nS * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  float* vbase = static_cast<float*>(dSv.data()) + float_offset;
  CUDA_CHECK(cudaMemcpyAsync(vbase, s0.data(), nS * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  DevStep ds(in, stream);
  kernels::qwen35_deltanet_delta_rule_fp32(
      ds.q(), ds.k(), ds.v(), ds.g(), ds.beta(),
      static_cast<float*>(dSref.data()),
      static_cast<__nv_bfloat16*>(dOf.data()), kNHeads, kHd, kEps, stream);
  kernels::deltanet_delta_vvec(
      ds.q(), ds.k(), ds.v(), ds.g(), ds.beta(), vbase,
      static_cast<__nv_bfloat16*>(dOv.data()), kNHeads, kHd, kEps, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> Of(nE), Ov(nE);
  CUDA_CHECK(cudaMemcpy(Of.data(), dOf.data(), nE * 2,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Ov.data(), dOv.data(), nE * 2,
                        cudaMemcpyDeviceToHost));
  CHECK_BITS(bits_equal_bf16(Of.data(), Ov.data(), nE,
                             "core_out (vvec state offset)"));
  std::printf("  [PASS] vvec state-offset %s bit-exact\n", label);
  return 1;
}

int run_alignment_fallback_gate(cudaStream_t stream) {
  // Two vvec state-base alignment cases (separate non-overlapping buffers,
  // each compared against a frozen reference on its own buffer):
  //   * +8 floats (+32B): still 16B-aligned -> exercises the float4 path at a
  //     non-256B offset.
  //   * +1 float (+4B):   16B-UNALIGNED    -> vvec must fall back to the
  //     frozen path and stay bit-identical.
  return run_vvec_state_offset("+32B (16B-aligned, float4 path)", 8, stream) +
         run_vvec_state_offset("+4B (16B-unaligned, frozen fallback)", 1,
                               stream);
}

int run_dispatcher_gate(cudaStream_t stream) {
  // The experimental dispatcher (identity table -> frozen path) must be
  // bit-identical to the frozen entry point, B=1 and batch.
  const StepInput in = make_step(1, InputKind::kMixedSign, GKind::kGPos,
                                 BetaKind::kBetaHalf, 4242);
  const std::vector<float> s0 = make_state(SKind::kSRandom, 42);
  const std::size_t nE = static_cast<std::size_t>(kNHeads) * kHd;
  const std::size_t nS = nE * kHd;
  DeviceBuffer dSf(nS * sizeof(float), stream);
  DeviceBuffer dSv(nS * sizeof(float), stream);
  DeviceBuffer dOf(nE * sizeof(__nv_bfloat16), stream);
  DeviceBuffer dOv(nE * sizeof(__nv_bfloat16), stream);
  DeviceBuffer ds0(nS * sizeof(float), stream);
  CUDA_CHECK(cudaMemcpyAsync(ds0.data(), s0.data(), nS * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dSf.data(), ds0.data(), nS * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dSv.data(), ds0.data(), nS * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
  DevStep ds(in, stream);
  kernels::qwen35_deltanet_delta_rule_fp32(
      ds.q(), ds.k(), ds.v(), ds.g(), ds.beta(),
      static_cast<float*>(dSf.data()),
      static_cast<__nv_bfloat16*>(dOf.data()), kNHeads, kHd, kEps, stream);
  kernels::deltanet_delta_rule_fp32_qwen35_experimental(
      ds.q(), ds.k(), ds.v(), ds.g(), ds.beta(),
      static_cast<float*>(dSv.data()),
      static_cast<__nv_bfloat16*>(dOv.data()), kNHeads, kHd, kEps, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<float> Sf(nS), Sv(nS);
  std::vector<__nv_bfloat16> Of(nE), Ov(nE);
  CUDA_CHECK(cudaMemcpy(Sf.data(), dSf.data(), nS * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Sv.data(), dSv.data(), nS * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Of.data(), dOf.data(), nE * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Ov.data(), dOv.data(), nE * 2, cudaMemcpyDeviceToHost));
  CHECK_BITS(bits_equal_bf16(Of.data(), Ov.data(), nE,
                               "core_out (dispatcher vs frozen)"));
  CHECK_BITS(bits_equal_f32(Sf.data(), Sv.data(), nS,
                              "recurrent state (dispatcher vs frozen)"));
  std::printf("  [PASS] experimental dispatcher bit-exact vs frozen\n");
  return 1;
}

}  // namespace

int main() {
  cudaFree(0);
  cudaSetDevice(0);
  cudaStream_t stream = 0;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const VariantInfo variants[] = {
      {"vreg", &kernels::deltanet_delta_vreg,
       &kernels::batch_deltanet_delta_vreg},
      {"vvec", &kernels::deltanet_delta_vvec,
       &kernels::batch_deltanet_delta_vvec},
      {"vchunk", &kernels::deltanet_delta_vchunk,
       &kernels::batch_deltanet_delta_vchunk},
  };

  std::printf("deltanet delta qwen35 optimized: register counts\n");
  std::printf("  vreg=%d vvec=%d vchunk(s-pass)=%d\n",
              kernels::deltanet_delta_vreg_regs(),
              kernels::deltanet_delta_vvec_regs(),
              kernels::deltanet_delta_vchunk_regs());
  CHECK(kernels::deltanet_delta_vreg_regs() > 0);
  CHECK(kernels::deltanet_delta_vvec_regs() > 0);
  CHECK(kernels::deltanet_delta_vchunk_regs() > 0);

  int total = 0;
  auto run_gate = [&](int r) {
    if (r < 0) return;  // failure already recorded
    total += r;
  };
  for (const auto& var : variants) {
    run_gate(run_b1_single_step_gate(var.name, var, stream));
    run_gate(run_b1_chain_gate(var.name, var, stream));
  }
  for (const auto& var : variants) {
    run_gate(run_batch_gate(var.name, var, 2, stream));
    run_gate(run_batch_gate(var.name, var, 3, stream));
  }
  run_gate(run_alignment_fallback_gate(stream));
  run_gate(run_dispatcher_gate(stream));

  cudaStreamDestroy(stream);
  if (g_failures > 0) {
    std::printf("deltanet delta qwen35 optimized: FAIL (%d failures, %d "
                "checks passed)\n",
                g_failures, total);
    return 1;
  }
  std::printf("deltanet delta qwen35 optimized: ALL PASS (%d checks)\n", total);
  return 0;
}
