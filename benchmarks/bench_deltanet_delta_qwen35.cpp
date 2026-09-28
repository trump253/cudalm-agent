// CUDALM — v0.7 Phase C: dedicated DeltaNet delta-rule microbenchmark over
// the Qwen3.5-0.8B production config.
//
// Compares, in ONE process (no cross-process environment drift):
//   * frozen baseline (deltanet_delta_kernel / batch_deltanet_delta_kernel)
//   * vreg / vvec / vchunk candidates (B=1 and batch)
// for n_heads = 16, head_dim = 128 and B = 1 / 2 / 3, with CUDA event
// timing (warmup >= 30, measured iterations scaled to ~0.5-1.5 s per cell).
//
// Reports per (B, variant): mean / median / min / p90 (us/call), us/token,
// speedup vs the frozen baseline, and registers/thread. Then the WEIGHTED
// PRODUCTION SCORE (Phase-A census, docs/v07_profiling.md §6.3/§6.5):
// 18 DeltaNet layers per traversal -> 18 deltanet_delta calls per traversal
// (2052 B=1 + 324 batch launches over 132 traversals = 114 B=1 : 6 B=2 :
// 12 B=3 traversals; 18 x (114+6+12) = 2376 = 2052+324). The canonical
// workload's per-run B distribution (19 B=1 : 1 B=2 : 2 B=3 of 22
// traversals) gives:
//   W_B1   = 18 x t_B1          [us per traversal, all-B=1]
//   W_B2   = 18 x t_B2
//   W_B3   = 18 x t_B3
//   overall = (19*W_B1 + 1*W_B2 + 2*W_B3) / 22
// reported per variant with the delta vs the frozen baseline.
//
// Usage:
//   bench_deltanet_delta_qwen35 [output.txt]
// No checkpoint required (synthetic deterministic fixture).

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
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

struct Fixture {
  std::vector<__nv_bfloat16> q, k, v;  // B * n_heads * hd
  std::vector<float> g;                // B * n_heads
  std::vector<__nv_bfloat16> beta;     // B * n_heads
  std::vector<float> state;            // B * n_heads * hd * hd (steady-state
                                       // magnitude ~N(0,4))
  int B;
};

Fixture make_fixture(int B) {
  Fixture f;
  f.B = B;
  const std::size_t n = static_cast<std::size_t>(B) * kNHeads * kHd;
  Rng rng(0xd31c);
  f.q.resize(n);
  f.k.resize(n);
  f.v.resize(n);
  f.g.resize(static_cast<std::size_t>(B) * kNHeads);
  f.beta.resize(static_cast<std::size_t>(B) * kNHeads);
  for (std::size_t i = 0; i < n; ++i) {
    f.q[i] = __float2bfloat16_rn(rng.n01() * 2.0f);
    f.k[i] = __float2bfloat16_rn(rng.n01() * 2.0f);
    f.v[i] = __float2bfloat16_rn(rng.n01() * 2.0f);
  }
  for (std::size_t i = 0; i < f.g.size(); ++i)
    f.g[i] = -expf(1.0f) * (0.01f + 0.5f * (0.5f + 0.5f * rng.n01()));
  for (std::size_t i = 0; i < f.beta.size(); ++i)
    f.beta[i] = __float2bfloat16_rn(1.0f / (1.0f + expf(-rng.n01() * 6.0f)));
  f.state.resize(B * static_cast<std::size_t>(kNHeads) * kHd * kHd);
  for (auto& x : f.state) x = rng.n01() * 4.0f;
  return f;
}

struct DevFixture {
  cudalm::DeviceBuffer dq, dk, dv, dg, db, dstate, dout, dslots;
  int B;
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
  float* state() const { return static_cast<float*>(dstate.data()); }
  __nv_bfloat16* out() const {
    return static_cast<__nv_bfloat16*>(dout.data());
  }
  const int* slots() const { return static_cast<const int*>(dslots.data()); }
};

void upload(const Fixture& f, DevFixture& d, cudaStream_t stream) {
  const int B = f.B;
  const std::size_t n = static_cast<std::size_t>(B) * kNHeads * kHd;
  d.dq.allocate(n * 2, stream);
  d.dk.allocate(n * 2, stream);
  d.dv.allocate(n * 2, stream);
  d.dg.allocate(static_cast<std::size_t>(B) * kNHeads * sizeof(float), stream);
  d.db.allocate(static_cast<std::size_t>(B) * kNHeads * 2, stream);
  d.dstate.allocate(f.state.size() * sizeof(float), stream);
  d.dout.allocate(n * 2, stream);
  const int slots[3] = {0, 1, 2};
  d.dslots.allocate(sizeof(int) * B, stream);
  d.dq.copy_from_host(f.q.data(), n * 2, stream);
  d.dk.copy_from_host(f.k.data(), n * 2, stream);
  d.dv.copy_from_host(f.v.data(), n * 2, stream);
  d.dg.copy_from_host(f.g.data(), f.g.size() * sizeof(float), stream);
  d.db.copy_from_host(f.beta.data(), f.beta.size() * 2, stream);
  d.dstate.copy_from_host(f.state.data(), f.state.size() * sizeof(float),
                          stream);
  d.dslots.copy_from_host(slots, sizeof(int) * B, stream);
  d.B = B;
}

typedef void (*FnB1)(const __nv_bfloat16*, const __nv_bfloat16*,
                     const __nv_bfloat16*, const float*, const __nv_bfloat16*,
                     float*, __nv_bfloat16*, int, int, float, cudaStream_t);
typedef void (*FnB)(const __nv_bfloat16*, const __nv_bfloat16*,
                    const __nv_bfloat16*, const float*, const __nv_bfloat16*,
                    float*, const int*, __nv_bfloat16*, int, int, float, int,
                    cudaStream_t);

struct Variant {
  const char* name;
  FnB1 b1;
  FnB batch;
  int (*regs)();
};

int frozen_regs() { return -1; }

double time_us(FnB1 fn, const DevFixture& d, int iters, cudaStream_t stream) {
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  const __nv_bfloat16* q = d.q();
  const __nv_bfloat16* k = d.k();
  const __nv_bfloat16* v = d.v();
  const float* g = d.g();
  const __nv_bfloat16* beta = d.beta();
  float* S = d.state();
  __nv_bfloat16* out = d.out();
  for (int i = 0; i < 30; ++i)
    fn(q, k, v, g, beta, S, out, kNHeads, kHd, kEps, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventRecord(e0, stream));
  for (int i = 0; i < iters; ++i)
    fn(q, k, v, g, beta, S, out, kNHeads, kHd, kEps, stream);
  CUDA_CHECK(cudaEventRecord(e1, stream));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms * 1000.0 / iters;
}

struct Stats {
  double mean = 0, median = 0, min = 0, p90 = 0;
};

// Per-iteration event timing for the distribution (iters2 << iters).
Stats time_dist(FnB1 fn, const DevFixture& d, int iters2,
                cudaStream_t stream) {
  const __nv_bfloat16* q = d.q();
  const __nv_bfloat16* k = d.k();
  const __nv_bfloat16* v = d.v();
  const float* g = d.g();
  const __nv_bfloat16* beta = d.beta();
  float* S = d.state();
  __nv_bfloat16* out = d.out();
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<double> us;
  us.reserve(iters2);
  for (int i = 0; i < iters2; ++i) {
    CUDA_CHECK(cudaEventRecord(e0, stream));
    fn(q, k, v, g, beta, S, out, kNHeads, kHd, kEps, stream);
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    us.push_back(ms * 1000.0);
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  std::sort(us.begin(), us.end());
  Stats st;
  double sum = 0.0;
  for (double x : us) sum += x;
  st.mean = sum / us.size();
  st.median = us[us.size() / 2];
  st.min = us.front();
  st.p90 = us[static_cast<std::size_t>(0.9 * (us.size() - 1))];
  return st;
}

double time_us_batch(FnB fn, const DevFixture& d, int iters,
                     cudaStream_t stream) {
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  const int B = d.B;
  for (int i = 0; i < 30; ++i)
    fn(d.q(), d.k(), d.v(), d.g(), d.beta(), d.state(), d.slots(), d.out(),
       kNHeads, kHd, kEps, B, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventRecord(e0, stream));
  for (int i = 0; i < iters; ++i)
    fn(d.q(), d.k(), d.v(), d.g(), d.beta(), d.state(), d.slots(), d.out(),
       kNHeads, kHd, kEps, B, stream);
  CUDA_CHECK(cudaEventRecord(e1, stream));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms * 1000.0 / iters;
}

Stats time_dist_batch(FnB fn, const DevFixture& d, int iters2,
                      cudaStream_t stream) {
  const int B = d.B;
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<double> us;
  us.reserve(iters2);
  for (int i = 0; i < iters2; ++i) {
    CUDA_CHECK(cudaEventRecord(e0, stream));
    fn(d.q(), d.k(), d.v(), d.g(), d.beta(), d.state(), d.slots(), d.out(),
       kNHeads, kHd, kEps, B, stream);
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    us.push_back(ms * 1000.0);
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  std::sort(us.begin(), us.end());
  Stats st;
  double sum = 0.0;
  for (double x : us) sum += x;
  st.mean = sum / us.size();
  st.median = us[us.size() / 2];
  st.min = us.front();
  st.p90 = us[static_cast<std::size_t>(0.9 * (us.size() - 1))];
  return st;
}

FILE* g_out = nullptr;
void P(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(g_out, fmt, ap);
  va_end(ap);
  std::fflush(g_out);
}

}  // namespace

int main(int argc, char** argv) {
  const char* out_path = argc >= 2 ? argv[1]
                                   : "bench_deltanet_delta_qwen35.txt";
  g_out = std::fopen(out_path, "w");
  if (!g_out) {
    std::fprintf(stderr, "cannot open %s\n", out_path);
    return 1;
  }

  cudaFree(0);
  cudaSetDevice(0);
  cudaStream_t stream = 0;
  CUDA_CHECK(cudaStreamCreate(&stream));

  P("== CUDALM v0.7 Phase C DeltaNet delta-rule microbench ==\n");
  P("config: n_heads=%d head_dim=%d (pinned Qwen3.5-0.8B); B=1/2/3\n", kNHeads,
    kHd);
  P("variants: frozen (baseline), vreg (register-resident column), vvec "
    "(float4 x4 values), vchunk (2-kernel value-chunk)\n");
  P("census (Phase A): 18 deltanet_delta calls/traversal; canonical B mix "
    "19:1:2 (B1:B2:B3 of 22 traversals)\n\n");

  const Variant variants[] = {
      {"frozen", &cudalm::kernels::qwen35_deltanet_delta_rule_fp32,
       &cudalm::kernels::batch_deltanet_delta_rule_fp32, &frozen_regs},
      {"vreg", &cudalm::kernels::deltanet_delta_vreg,
       &cudalm::kernels::batch_deltanet_delta_vreg,
       &cudalm::kernels::deltanet_delta_vreg_regs},
      {"vvec", &cudalm::kernels::deltanet_delta_vvec,
       &cudalm::kernels::batch_deltanet_delta_vvec,
       &cudalm::kernels::deltanet_delta_vvec_regs},
      {"vchunk", &cudalm::kernels::deltanet_delta_vchunk,
       &cudalm::kernels::batch_deltanet_delta_vchunk,
       &cudalm::kernels::deltanet_delta_vchunk_regs},
  };

  // Weighted production score accumulators: overall_us per variant.
  const int nvar = 4;
  double w_b1[nvar] = {0, 0, 0, 0};
  double w_b2[nvar] = {0, 0, 0, 0};
  double w_b3[nvar] = {0, 0, 0, 0};

  for (int B : {1, 2, 3}) {
    const Fixture fx = make_fixture(B);
    DevFixture d;
    upload(fx, d, stream);

    P("== B=%d (n_heads=%d, hd=%d) ==\n", B, kNHeads, kHd);
    P("%-8s %8s %10s %10s %10s %10s %12s %10s %8s\n", "variant", "iters",
      "mean_us", "median_us", "min_us", "p90_us", "us_per_token", "speedup",
      "regs");
    double frozen_us = 0.0;
    for (int vi = 0; vi < nvar; ++vi) {
      const Variant& var = variants[vi];
      // Estimate one call, choose iters for ~0.6-1.2 s of measured time.
      double one = (B == 1) ? time_us(var.b1, d, 1, stream)
                            : time_us_batch(var.batch, d, 1, stream);
      int iters = static_cast<int>(800000.0 / std::max(one, 1.0));
      iters = std::max(iters, 500);
      iters = std::min(iters, 20000);
      double mean_us = (B == 1) ? time_us(var.b1, d, iters, stream)
                                : time_us_batch(var.batch, d, iters, stream);
      const int iters2 = std::min(200, std::max(40, iters / 50));
      Stats st = (B == 1) ? time_dist(var.b1, d, iters2, stream)
                          : time_dist_batch(var.batch, d, iters2, stream);
      const double token = mean_us / B;
      const int regs = var.regs();
      if (vi == 0) {
        frozen_us = mean_us;
        P("%-8s %8d %10.3f %10.3f %10.3f %10.3f %12.3f %10s %8d\n", var.name,
          iters, mean_us, st.median, st.min, st.p90, token, "1.000x", regs);
      } else {
        P("%-8s %8d %10.3f %10.3f %10.3f %10.3f %12.3f %10.3fx %8d\n",
          var.name, iters, mean_us, st.median, st.min, st.p90, token,
          frozen_us / mean_us, regs);
      }
      // W_Bx = 18 calls/traversal x mean_us.
      if (B == 1) w_b1[vi] = 18.0 * st.mean;
      if (B == 2) w_b2[vi] = 18.0 * st.mean;
      if (B == 3) w_b3[vi] = 18.0 * st.mean;
    }
    P("\n");
  }

  P("== weighted production score (18 calls/traversal x canonical B "
    "mix 19:1:2) ==\n");
  P("W_Bx = 18 x mean_us(B=x);  overall = (19*W_B1 + 1*W_B2 + 2*W_B3) / 22 "
    "[us per traversal]\n\n");
  P("%-8s %14s %14s %14s %14s %12s\n", "variant", "W_B1_us", "W_B2_us",
    "W_B3_us", "overall_us", "vs_frozen");
  const double base_overall =
      (19.0 * w_b1[0] + 1.0 * w_b2[0] + 2.0 * w_b3[0]) / 22.0;
  for (int vi = 0; vi < nvar; ++vi) {
    const double overall =
        (19.0 * w_b1[vi] + 1.0 * w_b2[vi] + 2.0 * w_b3[vi]) / 22.0;
    P("%-8s %14.3f %14.3f %14.3f %14.3f %12.2f%%\n", variants[vi].name,
      w_b1[vi], w_b2[vi], w_b3[vi], overall,
      100.0 * (overall - base_overall) / base_overall);
  }

  std::fclose(g_out);
  std::fprintf(stderr, "microbench written to %s\n", out_path);
  return 0;
}
