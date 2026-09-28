// CUDALM — v0.7 Phase D (candidate 1) microbenchmark: fused residual-add +
// zero-centered RMSNorm vs the frozen 2-launch sequence
//     qwen35_add_bf16(a, b, res) -> qwen35_rmsnorm_zc_bf16(res, w, norm)
//
// Measures STREAM-ELAPSED GPU time per call (CUDA events) for the production
// post-attention shape (H = 1024, M = 1/2/3) so the launch-fragmentation
// benefit (2 launches + inter-kernel gap -> 1 launch) is visible at the micro
// level. The fused kernel is BIT-EXACT to the 2-launch sequence (see
// tests/cuda/test_qwen35_fused_add_rmsnorm.cpp), so this is a pure
// performance comparison on identical inputs. Kernel-DURATION sums (pure GPU
// busy time, no launch gaps) are measured separately via Nsys.
//
// No model, no checkpoint. Deterministic random bf16 inputs.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <functional>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/qwen35_kernels.h"

using namespace cudalm;

namespace {

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

float time_loop(cudaStream_t stream, cudaEvent_t t0, cudaEvent_t t1, int iters,
                std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 50; ++i) fn();
  cudaStreamSynchronize(stream);
  cudaEventRecord(t0, stream);
  for (int i = 0; i < iters; ++i) fn();
  cudaEventRecord(t1, stream);
  cudaEventSynchronize(t1);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, t0, t1);
  return ms * 1000.f / static_cast<float>(iters);  // us per call
}

}  // namespace

int main() {
  const int H = 1024;
  const float eps = 1e-6f;
  const int iters = 3000;
  cudaStream_t stream;
  cudaEvent_t t0, t1;
  cudaStreamCreate(&stream);
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  std::printf(
      "fused add+rmsnorm microbench (H=%d, eps=1e-6, %d iters, stream-elapsed "
      "us/call)\n",
      H, iters);
  std::printf("%-6s %-22s %-22s %-14s\n", "M", "frozen 2-launch (us)",
              "fused 1-launch (us)", "delta (us)");
  for (int M : {1, 2, 3}) {
    const std::size_t rows = static_cast<std::size_t>(M) * H;
    std::vector<__nv_bfloat16> ha(rows), hb(rows), hw(H);
    Rng ra(0xFA01u + M), rb(0xFA02u + M), rw(0xFA03u + M);
    for (std::size_t i = 0; i < rows; ++i)
      ha[i] = __float2bfloat16_rn(ra.n01());
    for (std::size_t i = 0; i < rows; ++i)
      hb[i] = __float2bfloat16_rn(rb.n01());
    for (std::size_t i = 0; i < H; ++i)
      hw[i] = __float2bfloat16_rn(rw.n01());

    DeviceBuffer da(rows * 2, stream), db(rows * 2, stream), dw(H * 2, stream);
    DeviceBuffer res_f(rows * 2, stream), norm_f(rows * 2, stream);
    DeviceBuffer res_c(rows * 2, stream), norm_c(rows * 2, stream);
    da.copy_from_host(ha.data(), da.bytes(), stream);
    db.copy_from_host(hb.data(), db.bytes(), stream);
    dw.copy_from_host(hw.data(), dw.bytes(), stream);

    const __nv_bfloat16* a = da.data<__nv_bfloat16>();
    const __nv_bfloat16* b = db.data<__nv_bfloat16>();
    const __nv_bfloat16* w = dw.data<__nv_bfloat16>();

    float frozen_us = time_loop(
        stream, t0, t1, iters, [&] {
          kernels::qwen35_add_bf16(a, b, res_f.data<__nv_bfloat16>(), rows,
                                   stream);
          kernels::qwen35_rmsnorm_zc_bf16(res_f.data<__nv_bfloat16>(), w,
                                          norm_f.data<__nv_bfloat16>(), M, H,
                                          eps, stream);
        });
    float fused_us = time_loop(stream, t0, t1, iters, [&] {
      kernels::qwen35_fused_add_rmsnorm_zc_bf16(
          a, b, w, res_c.data<__nv_bfloat16>(), norm_c.data<__nv_bfloat16>(),
          M, H, eps, stream);
    });
    std::printf("%-6d %-22.3f %-22.3f %-14.3f\n", M, frozen_us, fused_us,
                fused_us - frozen_us);
  }
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaStreamDestroy(stream);
  return 0;
}
