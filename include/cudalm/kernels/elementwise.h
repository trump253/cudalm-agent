// CUDALM — fp16 elementwise kernels (residual add, SiLU·mul for the MLP
// gate). Raw pointers + cudaStream_t; no PyTorch.
//
// Both ops implement the golden math contract (docs/weight_format.md):
// fp32 math inside, a single fp16 round-to-nearest-even (RNE) store at the
// stage boundary.
//
// Provenance: CUDALM-native (no upstream kernel; see docs/provenance.md).

#pragma once

#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cudalm {

// y[i] = a[i] + b[i]   (fp32 add of the two fp16 values, fp16 RNE store).
// This is the residual contract: stage.residual1 = stage.input +
// stage.output_projection, stage.final_output = stage.residual1 + stage.down.
//
// Precondition (host-checked, abort on violation):
//   * n % 8 == 0
//   * a, b, y are 16-byte aligned (DeviceBuffer allocations are 256B
//     aligned, so this holds for all runtime buffers).
void add_fp16(const __half* a, const __half* b, __half* y, std::size_t n,
              cudaStream_t stream);

// y[i] = silu(gate[i]) * up[i],  silu(x) = x / (1 + exp(-x))
// (fp32 math, fp16 RNE store) — the stage.silu_gate_mul_up contract.
//
// Precondition (host-checked, abort on violation):
//   * n % 8 == 0
//   * gate, up, y are 16-byte aligned.
void silu_mul_fp16(const __half* gate, const __half* up, __half* y,
                   std::size_t n, cudaStream_t stream);

}  // namespace cudalm
