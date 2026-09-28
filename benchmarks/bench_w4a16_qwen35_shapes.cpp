// CUDALM — v0.7 Phase B: dedicated W4A16 GEMV microbenchmark over the
// Qwen3.5-0.8B production shape census.
//
// Compares, in ONE process (no cross-process environment drift):
//   * frozen R=4 baseline (int4_gemv_bf16 / batch_int4_gemv_bf16)
//   * R=1 / R=2 / R=8 row-tile variants (B=1 and batch)
// for EVERY production census shape (N, K) and B = 1 / 2 / 3, with CUDA
// event timing (warmup >= 30, measured iterations >= 200, scaled up for
// short kernels).
//
// Reports per (shape, B, variant): mean / median / min / p90 (us),
// us/token, speedup vs the frozen baseline, and registers/thread. Then the
// WEIGHTED PRODUCTION SCORE (reviewer §9): the Phase-A serving call-rate
// weights from the census (calls per model traversal) and the canonical
// workload's per-traversal B distribution (19 B=1 : 1 B=2 : 2 B=3 per run —
// Phase-A canonical workload, docs/v07_profiling.md §3):
//   W_B1   = sum_sh  calls/trav(sh) x t_B1(sh)          [us / traversal]
//   W_B2   = sum_sh  calls/trav(sh) x t_B2(sh)
//   W_B3   = sum_sh  calls/trav(sh) x t_B3(sh)
//   overall = (19*W_B1 + 1*W_B2 + 2*W_B3) / 22
// reported per variant with the delta vs the frozen baseline.
//
// Usage:
//   bench_w4a16_qwen35_shapes [output.txt]
// No checkpoint required (synthetic deterministic weights/activations).

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
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

struct Shape {
  int N, K;
  int calls_per_trav;
  const char* role;
};

// The production census (v07b_w4a16_shape_census.txt): 186 calls/trav.
const Shape kCensus[] = {
    {16, 1024, 36, "in_proj_b/a"},
    {512, 1024, 12, "k/v_proj"},
    {1024, 2048, 24, "o/out_proj"},
    {1024, 3584, 24, "down_proj"},
    {2048, 1024, 18, "in_proj_z"},
    {3584, 1024, 48, "gate/up_proj"},
    {4096, 1024, 6, "q_proj"},
    {6144, 1024, 18, "in_proj_qkv"},
};
constexpr int kNumShapes = 8;

// Canonical-workload B distribution per run (Phase A): 19 B=1 + 1 B=2 +
// 2 B=3 traversals per run.
constexpr double kB1W = 19.0 / 22.0;
constexpr double kB2W = 1.0 / 22.0;
constexpr double kB3W = 2.0 / 22.0;

struct Stats {
  double mean = 0.0, median = 0.0, minv = 0.0, p90 = 0.0;
  int iters = 0;
};

Stats summarize(std::vector<float>* us) {
  std::sort(us->begin(), us->end());
  Stats st;
  const int m = static_cast<int>(us->size());
  st.iters = m;
  double sum = 0.0;
  for (float v : *us) sum += v;
  st.mean = sum / m;
  st.median = (*us)[m / 2];
  st.minv = (*us)[0];
  st.p90 = (*us)[static_cast<int>(0.9 * (m - 1))];
  return st;
}

// One variant under test.
struct Variant {
  const char* name;
  int regs;
  void (*b1)(const std::uint8_t*, const __half*, const __nv_bfloat16*,
             __nv_bfloat16*, int, int, cudaStream_t);
  void (*batch)(const std::uint8_t*, const __half*, const __nv_bfloat16*,
                __nv_bfloat16*, int, int, int, cudaStream_t);
};

void bench_variant(const char* tag, const Variant& va, const Shape& sh, int B,
                   const std::uint8_t* dW, const __half* dS,
                   const __nv_bfloat16* dx, __nv_bfloat16* dy,
                   cudaStream_t stream, cudaEvent_t ev0, cudaEvent_t ev1,
                   std::vector<float>* out) {
  // Warmup (>= 30).
  for (int i = 0; i < 30; ++i) {
    if (B == 1)
      va.b1(dW, dS, dx, dy, sh.N, sh.K, stream);
    else
      va.batch(dW, dS, dx, dy, sh.N, sh.K, B, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Probe ~20 iters to pick the measured iteration count (>= 200; scaled up
  // for short kernels to keep the timer well above noise).
  std::vector<float> probe;
  probe.reserve(20);
  for (int i = 0; i < 20; ++i) {
    CUDA_CHECK(cudaEventRecord(ev0, stream));
    if (B == 1)
      va.b1(dW, dS, dx, dy, sh.N, sh.K, stream);
    else
      va.batch(dW, dS, dx, dy, sh.N, sh.K, B, stream);
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    probe.push_back(ms * 1000.f);
  }
  std::sort(probe.begin(), probe.end());
  const float est_us = probe[10];
  int iters;
  if (est_us < 8.0f)
    iters = 2000;
  else if (est_us < 40.0f)
    iters = 800;
  else
    iters = 300;
  if (iters < 200) iters = 200;

  out->reserve(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(ev0, stream));
    if (B == 1)
      va.b1(dW, dS, dx, dy, sh.N, sh.K, stream);
    else
      va.batch(dW, dS, dx, dy, sh.N, sh.K, B, stream);
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    out->push_back(ms * 1000.f);
  }
  (void)tag;
}

}  // namespace

int main(int argc, char** argv) {
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 1) return 1;
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  const char* out_path = (argc >= 2) ? argv[1] : "/dev/stdout";
  FILE* out = std::fopen(out_path, "w");
  if (!out) {
    std::fprintf(stderr, "cannot open %s\n", out_path);
    return 1;
  }
  auto P = [&](const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(out, fmt, ap);
    va_end(ap);
    std::fflush(out);
  };

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));

  const Variant kVariants[] = {
      {"frozen_R4", cudalm::int4_gemv_bf16_rowtile4_regs(),
       cudalm::int4_gemv_bf16, cudalm::kernels::batch_int4_gemv_bf16},
      {"R1", cudalm::int4_gemv_bf16_rowtile1_regs(),
       cudalm::int4_gemv_bf16_rowtile1, cudalm::batch_int4_gemv_bf16_rowtile1},
      {"R2", cudalm::int4_gemv_bf16_rowtile2_regs(),
       cudalm::int4_gemv_bf16_rowtile2, cudalm::batch_int4_gemv_bf16_rowtile2},
      {"R8", cudalm::int4_gemv_bf16_rowtile8_regs(),
       cudalm::int4_gemv_bf16_rowtile8, cudalm::batch_int4_gemv_bf16_rowtile8},
  };
  constexpr int kNumVariants = 4;

  P("# CUDALM v0.7 Phase B — W4A16 GEMV microbenchmark (Qwen3.5-0.8B census)\n");
  P("# device: %s (SM %d)\n", prop.name, prop.multiProcessorCount);
  P("# warmup>=30, measured iters per (shape,B,variant) as reported (>=200;\n");
  P("# short kernels scaled up to 2000). CUDA-event timing, single process.\n");
  P("# regs: cudaFuncGetAttributes numRegs.\n");
  P("\n");

  // results[shape][B-1][variant] = Stats
  struct Cell {
    Stats st;
    bool have = false;
  };
  Cell cells[kNumShapes][3][kNumVariants];

  int b1 = 0;
  for (const Shape& sh : kCensus) {
    // Deterministic fixture for this shape (shared by all B / variants).
    Rng rng(0x51ed);
    std::vector<std::uint8_t> hW(static_cast<std::size_t>(sh.N) * (sh.K / 2));
    std::vector<__half> hS(static_cast<std::size_t>(sh.N) * (sh.K / 128));
    std::vector<__nv_bfloat16> hxB(3 * sh.K);  // max B = 3
    for (auto& x : hW) x = static_cast<std::uint8_t>(rng.u32());
    for (auto& x : hS)
      x = __float2half_rn(0.001f + 0.02f * static_cast<float>(rng.u32()) *
                                         (1.0f / 4294967296.0f));
    for (auto& x : hxB) x = __float2bfloat16_rn(rng.n01());

    cudalm::DeviceBuffer dW(hW.size(), stream);
    cudalm::DeviceBuffer dS(hS.size() * sizeof(__half), stream);
    cudalm::DeviceBuffer dx(3 * sh.K * sizeof(__nv_bfloat16), stream);
    cudalm::DeviceBuffer dy(3 * sh.N * sizeof(__nv_bfloat16), stream);
    dW.copy_from_host(hW.data(), hW.size(), stream);
    dS.copy_from_host(hS.data(), hS.size() * 2, stream);
    dx.copy_from_host(hxB.data(), 3 * sh.K * 2, stream);

    P("== shape (N=%d, K=%d) %s  calls/trav=%d  gridR4=%d gridR1=%d ==\n",
      sh.N, sh.K, sh.role, sh.calls_per_trav, (sh.N + 3) / 4, sh.N);
    P("%-9s %-3s %6s %10s %10s %10s %10s %10s %8s %8s\n", "variant", "B",
      "iters", "mean_us", "median_us", "min_us", "p90_us", "us/token",
      "speedup", "regs");
    for (int B : {1, 2, 3}) {
      for (int v = 0; v < kNumVariants; ++v) {
        std::vector<float> us;
        bench_variant(kVariants[v].name, kVariants[v], sh, B,
                      dW.data<std::uint8_t>(), dS.data<__half>(),
                      dx.data<__nv_bfloat16>(), dy.data<__nv_bfloat16>(),
                      stream, ev0, ev1, &us);
        Stats st = summarize(&us);
        cells[b1][B - 1][v].st = st;
        cells[b1][B - 1][v].have = true;
        const Stats& base = cells[b1][B - 1][0].st;
        P("%-9s %-3d %6d %10.3f %10.3f %10.3f %10.3f %10.3f %8.3fx %8d\n",
          kVariants[v].name, B, st.iters, st.mean, st.median, st.minv, st.p90,
          st.mean / B, v > 0 ? base.mean / st.mean : 1.0, kVariants[v].regs);
      }
    }
    P("\n");
    ++b1;
  }

  // ---- weighted production score ------------------------------------------
  P("== weighted production score (census call rates x canonical B "
    "distribution %.3f:%.3f:%.3f) ==\n",
    kB1W, kB2W, kB3W);
  P("W_Bx = sum over census shapes of calls/trav x mean_us(shape, B)  [us per "
    "model traversal]\n");
  P("overall = (19*W_B1 + 1*W_B2 + 2*W_B3) / 22\n\n");
  P("%-9s %14s %14s %14s %14s %12s\n", "variant", "W_B1_us", "W_B2_us",
    "W_B3_us", "overall_us", "vs_frozen");
  double w_base[3] = {0, 0, 0};
  for (int B = 0; B < 3; ++B)
    for (int s = 0; s < kNumShapes; ++s)
      w_base[B] += kCensus[s].calls_per_trav * cells[s][B][0].st.mean;
  const double base_overall =
      (19.0 * w_base[0] + 1.0 * w_base[1] + 2.0 * w_base[2]) / 22.0;
  for (int v = 0; v < kNumVariants; ++v) {
    double w[3] = {0, 0, 0};
    for (int B = 0; B < 3; ++B)
      for (int s = 0; s < kNumShapes; ++s)
        w[B] += kCensus[s].calls_per_trav * cells[s][B][v].st.mean;
    const double overall =
        (19.0 * w[0] + 1.0 * w[1] + 2.0 * w[2]) / 22.0;
    P("%-9s %14.3f %14.3f %14.3f %14.3f %11.2f%%\n", kVariants[v].name, w[0],
      w[1], w[2], overall, 100.0 * (overall - base_overall) / base_overall);
  }

  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::fclose(out);
  std::fprintf(stderr, "microbench written to %s\n", out_path);
  return 0;
}
