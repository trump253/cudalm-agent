// CUDALM — CUDA smoke test: device present, DeviceBuffer alloc/copy round-trip
// works on an explicit stream, CUDA_CHECK compiles.

#include "../../tests/common/check.h"

#include "cudalm/device_buffer.h"
#include "cudalm/cuda_check.h"
#include <cuda_fp16.h>

#include <utility>
#include <vector>

using namespace cudalm;

int main() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  CHECK(n >= 1);

  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));

  // Explicit non-default stream (CUDALM never assumes the default stream).
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const std::size_t N = 1024;
  DeviceBuffer buf(N * sizeof(float), stream);
  CHECK(buf.valid());
  CHECK_EQ(buf.bytes(), N * sizeof(float));

  std::vector<float> h(N);
  for (std::size_t i = 0; i < N; ++i) h[i] = static_cast<float>(i) * 0.5f;
  buf.copy_from_host(h.data(), N * sizeof(float), stream);

  // Read the same buffer back and verify the copy path.
  std::vector<float> back(N, -1.0f);
  buf.copy_to_host(back.data(), N * sizeof(float), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (std::size_t i = 0; i < N; ++i) {
    if (back[i] != static_cast<float>(i) * 0.5f) {
      TEST_FAIL("cuda_smoke", "round-trip mismatch");
    }
  }

  // fp16 path: DeviceBuffer<__half> typing
  DeviceBuffer hbuf(N * sizeof(__half), stream);
  std::vector<__half> hh(N);
  for (std::size_t i = 0; i < N; ++i) hh[i] = __float2half_rn(static_cast<float>(i));
  hbuf.copy_from_host(hh.data(), N * sizeof(__half), stream);
  std::vector<__half> back_half(N);
  hbuf.copy_to_host(back_half.data(), N * sizeof(__half), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (std::size_t i = 0; i < N; ++i) {
    if (__half2float(back_half[i]) != static_cast<float>(i)) {
      TEST_FAIL("cuda_smoke", "fp16 round-trip mismatch");
    }
  }

  // Move semantics
  DeviceBuffer moved = std::move(buf);
  CHECK(moved.valid());
  CHECK(!buf.valid());

  CUDA_CHECK(cudaStreamDestroy(stream));
  TEST_PASS("cuda_smoke");
  return 0;
}
