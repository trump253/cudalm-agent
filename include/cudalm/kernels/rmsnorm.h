// CUDALM — RMSNorm fp16 kernel (port of CUDALab rmsnorm_v4, fp16
// specialization). Raw pointers + cudaStream_t; no PyTorch.
//
// y[r, :] = x[r, :] * rsqrt(mean_k(x[r, k]^2) + eps) * w
// with fp32 accumulation (per-thread partial sums + warp/shfl reduction,
// exactly as upstream) and an fp16 RNE store.
//
// Provenance: CUDALab commit cb6a6a9, kernels/rmsnorm/rmsnorm_v4.cu
// (rmsnorm_v4_half_kernel). See docs/provenance.md for the port record and
// deviations.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudalm {

// RMSNorm over M rows of H elements.
//
// Precondition (host-checked, abort on violation — same strict contract as
// upstream v4_precheck, no scalar fallback):
//   * H % 256 == 0 and PER = H/256 ∈ {4, 8, 16, 32}
//     (i.e. H ∈ {1024, 2048, 4096, 8192})
//   * fp16 pointer alignment: PER % 8 == 0 (float4 loads) requires x, w, y
//     16B aligned; PER == 4 (half2 loads) requires 4B aligned. Row stride
//     H*2B is a multiple of the required alignment for all supported PER, so
//     base-pointer alignment implies row alignment. (DeviceBuffer
//     allocations are 256B aligned, so this holds for all runtime buffers;
//     e.g. v0.1's H = 1024 → PER = 4 → 4B.)
void rmsnorm_fp16(const __half* x, const __half* w, __half* y, int M, int H,
                  float eps, cudaStream_t stream);

}  // namespace cudalm
