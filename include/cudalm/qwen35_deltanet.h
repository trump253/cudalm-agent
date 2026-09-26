// CUDALM — Qwen3.5 Gated DeltaNet decoder layer (v0.2, Phase C), single-GPU
// autoregressive decode step (batch-1, one token). CUDALM-native wiring of
// the bf16 Qwen3.5 kernel family + the W4A16 GEMV (bf16 specialization) +
// the new DeltaNet decode kernels (qwen35_deltanet_kernels.h).
//
// Pipeline (docs/qwen35_architecture.md §8 + §9, one decode step, batch-1):
//
//   x -> zero-centered RMSNorm(input_layernorm)                -> rmsnorm1
//       -> mixed = in_proj_qkv(rms1)    (W4A16, [6144])
//       -> z     = in_proj_z(rms1)      (W4A16, [2048])
//       -> b     = in_proj_b(rms1)      (W4A16, [16])
//       -> a     = in_proj_a(rms1)      (W4A16, [16])
//       -> depthwise causal conv1d(kernel 4) with persistent conv state
//            [cs0,cs1,cs2,new] -> state := [cs1,cs2,new]; c = conv (bf16);
//            mixed2 = SiLU(c)                                    -> conv_silu
//       -> split: q [2048] / k [2048] / v [2048]
//       -> beta = sigmoid(b) (bf16); g = -exp(A_log)*softplus(a+dt_bias) (fp32)
//       -> delta-rule recurrent update (state S fp32 [16,128,128], in place):
//            FLA-aligned l2norm on bf16 q/k (bf16 rounding semantics),
//            q/=sqrt(128); the normalized bf16 q/k then feed the fp32
//            recurrent delta-rule / state math: decay -> delta -> output
//            from UPDATED S                                       -> core (bf16)
//       -> gated RMSNorm(core, gate=z, w=linear_norm fp32)     -> gated (bf16)
//       -> o = out_proj(gated)        (W4A16, [1024])
//       -> res1 = x + o               (bf16 add)
//       -> rms2 = zero-centered RMSNorm(res1, post_attention_layernorm)
//       -> g2 = gate_proj(rms2), u2 = up_proj(rms2)  (W4A16)
//       -> sgmu = silu(g2) * u2        (SwiGLU, bf16)
//       -> d = down_proj(sgmu)         (W4A16)
//       -> y = res1 + d
//
// Persistent state (owned, updated IN PLACE each step; docs §3-row-10):
//   conv_state      bf16 [6144, 3]
//   recurrent_state fp32 [16, 128, 128]
// Reset to zero for the first token; the state IS the Phase C hard gate.
//
// All stage buffers are owned device memory, valid from forward() until the
// next forward() or destruction — the golden test compares them stage-by-stage
// against a CUDLMG02 golden file (runtime-vs-quantized-reference HARD GATE;
// tolerance in stage_compare.h). DeltaNet has NO position dependence (the
// `position` argument is accepted for API symmetry / metadata only).
//
// The layer is driven by a Phase-A Qwen35LayerWeights (v2 .cudalm layer
// tensor set) loaded for a linear-attention layer index.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cudalm/device_buffer.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/weight_loader_v2.h"

namespace cudalm {

// Validates that `cfg` is a Gated DeltaNet configuration the Phase C kernels
// actually support (the pinned Qwen3.5-0.8B) and that `layer_idx` is a
// linear-attention layer. The DeltaNet decode kernels hardcode a depthwise
// causal conv of kernel 4 (conv_state len 3), an fp32 recurrent state of head
// dim 128, and no repeat_interleave (num_k_heads == num_v_heads), so a
// config outside this contract would silently corrupt or go out of bounds
// rather than fail loud. This is the single gate the constructor and the CPU
// contract test both check. Aborts on any violation (CUDALM has no
// error-returning path in v0.2).
void qwen35_deltanet_require_supported_config(const Qwen35Config& cfg,
                                              int layer_idx);

class Qwen35DeltaNetLayer {
 public:
  // `weights` must outlive the layer (non-owning) and must have been loaded
  // for a linear-attention (Gated DeltaNet) layer index.
  explicit Qwen35DeltaNetLayer(const Qwen35LayerWeights& weights,
                               cudaStream_t stream = 0);

  // One decode step. `x_in` is a device bf16 [hidden_size] buffer. `position`
  // is unused by the DeltaNet math (no RoPE / position dependence) but is
  // validated against max_seq_len and kept for API symmetry with the
  // full-attention layer. The persistent state is updated in place.
  void forward(int position, const __nv_bfloat16* x_in,
               cudaStream_t stream);

  // v0.5 Phase B: the SAME pipeline as forward(), but the persistent state
  // lives OUTSIDE the layer: `ext_conv` is a device bf16 buffer
  // [linear_conv_dim(), 3] and `ext_rec` a device fp32 buffer
  // [lin_num_v_heads, lin_value_head_dim, lin_value_head_dim] (the exact
  // layout of one slot of a Phase-A Qwen35DeltaStatePool,
  // conv(ordinal, slot) / recurrent(ordinal, slot)). The state is read +
  // updated IN PLACE at those external addresses (same kernels, same math,
  // no host roundtrip); the layer's OWN conv_state_/rec_state_ are NOT
  // touched. Precondition (host-checked, abort on violation):
  // 0 <= position < max_seq_len.
  void forward_with_state(int position, const __nv_bfloat16* x_in,
                          __nv_bfloat16* ext_conv, float* ext_rec,
                          cudaStream_t stream);

  // ---- persistent state (Phase C hard gate) --------------------------------
  // Zero both conv_state and recurrent_state (first-token / reset).
  void reset_state(cudaStream_t stream);
  // Seed the state from HOST buffers: `conv` bf16 [conv_dim, 3],
  // `rec` fp32 [n_heads, head_dim, head_dim] (test seeding, scenario C).
  void seed_state(const __nv_bfloat16* conv, const float* rec,
                  cudaStream_t stream);
  // Device pointers to the persistent state (for the test to read back).
  const __nv_bfloat16* conv_state() const { return conv_state_.data<__nv_bfloat16>(); }
  const float* recurrent_state() const { return rec_state_.data<float>(); }

  // ---- stage buffers (device bf16, stable until next forward()/destruction) -
  const __nv_bfloat16* stage_input() const { return input_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_rmsnorm1() const { return rms1_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_in_proj_qkv() const { return mixed_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_in_proj_z() const { return z_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_in_proj_b() const { return b_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_in_proj_a() const { return a_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_conv_out() const { return conv_out_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_conv_silu() const { return conv_silu_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_q() const { return q_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_k() const { return k_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_v() const { return v_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_beta() const { return beta_.data<__nv_bfloat16>(); }
  const float* stage_g() const { return g_.data<float>(); }
  const __nv_bfloat16* stage_core_out() const { return core_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_gated_norm() const { return gated_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_out_proj() const { return out_proj_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_residual1() const { return res1_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_rmsnorm2() const { return rms2_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_mlp_gate() const { return mlp_gate_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_mlp_up() const { return mlp_up_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_silu_mul() const { return silu_mul_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_mlp_down() const { return mlp_down_.data<__nv_bfloat16>(); }
  const __nv_bfloat16* stage_final_output() const { return final_.data<__nv_bfloat16>(); }

  // Shared pipeline (frozen v0.2 math, unchanged): the persistent conv/rec
  // state is addressed through `conv_state` / `rec_state` — the layer's own
  // buffers for forward(), an external slot (Phase B) for
  // forward_with_state().
  void forward_impl(int position, const __nv_bfloat16* x_in,
                    __nv_bfloat16* conv_state, float* rec_state,
                    cudaStream_t stream);

 private:
  const Qwen35LayerWeights* w_;
  Qwen35Config cfg_{};

  // Persistent state.
  DeviceBuffer conv_state_;   // bf16 [conv_dim, 3]
  DeviceBuffer rec_state_;    // fp32 [n_heads, head_dim, head_dim]

  // Stage buffers (bf16 unless noted).
  DeviceBuffer input_;        // [H]
  DeviceBuffer rms1_;         // [H]
  DeviceBuffer mixed_;        // [conv_dim]  in_proj_qkv
  DeviceBuffer z_;            // [value_dim] in_proj_z
  DeviceBuffer b_;            // [n_heads]   in_proj_b
  DeviceBuffer a_;            // [n_heads]   in_proj_a
  DeviceBuffer conv_out_;     // [conv_dim]  conv (pre-SiLU)
  DeviceBuffer conv_silu_;    // [conv_dim]  SiLU(conv)
  DeviceBuffer q_;            // [key_dim]
  DeviceBuffer k_;            // [key_dim]
  DeviceBuffer v_;            // [value_dim]
  DeviceBuffer beta_;         // [n_heads] bf16
  DeviceBuffer g_;            // [n_heads] fp32
  DeviceBuffer core_;         // [value_dim]
  DeviceBuffer gated_;        // [value_dim]
  DeviceBuffer out_proj_;     // [H]
  DeviceBuffer res1_;         // [H]
  DeviceBuffer rms2_;         // [H]
  DeviceBuffer mlp_gate_;     // [inter]
  DeviceBuffer mlp_up_;       // [inter]
  DeviceBuffer silu_mul_;     // [inter]
  DeviceBuffer mlp_down_;     // [H]
  DeviceBuffer final_;        // [H]
};

}  // namespace cudalm
