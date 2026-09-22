// CUDALM — unified CUDA error handling.
//
// Every cuda* API call and every kernel launch in CUDALM goes through
// CUDA_CHECK (or CUDA_CHECK_LAUNCH for launches). There is no error-code
// returning path in v0.1: a CUDA failure is fatal and aborts with a
// diagnostic (expression, file, line, driver error string).
//
// Provenance: CUDALM-native (replaces CUDALab's C10_CUDA_KERNEL_LAUNCH_CHECK /
// TORCH_CHECK host-side validation layer; see docs/provenance.md).

#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace cudalm {

// Prints a fatal diagnostic and aborts. `expr` is the source text of the
// failed call for traceability.
[[noreturn]] inline void cuda_fatal(const char* expr, const char* file,
                                    int line, cudaError_t err) {
  std::fprintf(stderr,
               "[cudalm] CUDA failure: %s\n"
               "  expression : %s\n"
               "  file       : %s:%d\n"
               "  error      : %s (%d)\n",
               (err == cudaSuccess ? "kernel launch error (checked post-launch)"
                                   : "cuda API call"),
               expr, file, line, cudaGetErrorString(err), static_cast<int>(err));
  std::abort();
}

}  // namespace cudalm

// Check a cuda* API call's return code.
#define CUDA_CHECK(call)                                              \
  do {                                                                \
    const cudaError_t cudalm_err__ = (call);                          \
    if (cudalm_err__ != cudaSuccess) {                                \
      ::cudalm::cuda_fatal(#call, __FILE__, __LINE__, cudalm_err__);  \
    }                                                                 \
  } while (0)

// Check for launch errors after a kernel launch (`<<<>>>` itself cannot be
// checked directly). Must be the first CUDA check after a launch.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
