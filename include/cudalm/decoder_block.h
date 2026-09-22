// CUDALM — one Llama-style decoder block (v0.1), single-GPU autoregressive
// decode step. CUDALM-native wiring of the ported/native kernels.
//
//   x -> RMSNorm(attn_norm)
//       -> q = q_proj(rms1), k = k_proj(rms1), v = v_proj(rms1)   (W4A16)
//       -> rope_q = RoPE(q, position), rope_k = RoPE(k, position)
//       -> KV write at `position` (K cache <- rope_k, V cache <- v)
//       -> attention_output = causal attention(rope_q, KV[0..position])
//       -> o_proj -> residual1 = x + o_proj
//       -> RMSNorm(ffn_norm)
//       -> gate = gate_proj(rms2), up = up_proj(rms2)             (W4A16)
//       -> silu_gate_mul_up = SiLU(gate) * up
//       -> down = down_proj(silu_gate_mul_up)                     (W4A16)
//       -> final_output = residual1 + down
//
// All stage buffers are owned device memory (fp16), valid from forward()
// until the next forward() or destruction — the golden test compares them
// stage-by-stage against a CUDLMG01 golden file.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cudalm/device_buffer.h"
#include "cudalm/kv_cache.h"
#include "cudalm/model_config.h"
#include "cudalm/weight_loader.h"

namespace cudalm {

class DecoderBlock {
 public:
  // `weights` must outlive the block (non-owning).
  explicit DecoderBlock(const BlockWeights& weights, cudaStream_t stream = 0);

  // One decode step at 0-based `position`. `x_in` is a device fp16
  // [hidden_size] buffer (the current token's hidden state).
  //
  // Precondition (host-checked, abort on violation):
  // 0 <= position < max_seq_len.
  void forward(int position, const __half* x_in, cudaStream_t stream);

  // ---- Stage buffers (device fp16; valid after forward) ----------------
  const __half* stage_input() const { return input_.data<__half>(); }
  const __half* stage_rmsnorm1() const { return rms1_.data<__half>(); }
  const __half* stage_q() const { return q_.data<__half>(); }
  const __half* stage_k() const { return k_.data<__half>(); }
  const __half* stage_v() const { return v_.data<__half>(); }
  const __half* stage_rope_q() const { return rope_q_.data<__half>(); }
  const __half* stage_rope_k() const { return rope_k_.data<__half>(); }
  const __half* stage_attention_output() const {
    return attn_.data<__half>();
  }
  const __half* stage_output_projection() const {
    return oproj_.data<__half>();
  }
  const __half* stage_residual1() const { return res1_.data<__half>(); }
  const __half* stage_rmsnorm2() const { return rms2_.data<__half>(); }
  const __half* stage_gate() const { return gate_.data<__half>(); }
  const __half* stage_up() const { return up_.data<__half>(); }
  const __half* stage_silu_gate_mul_up() const {
    return sgmu_.data<__half>();
  }
  const __half* stage_down() const { return down_.data<__half>(); }
  const __half* stage_final_output() const {
    return final_.data<__half>();
  }

  KvCache& kv_cache() { return *kv_; }
  const KvCache& kv_cache() const { return *kv_; }
  const ModelConfig& config() const { return cfg_; }

 private:
  const BlockWeights* w_ = nullptr;
  ModelConfig cfg_{};
  std::unique_ptr<KvCache> kv_;  // KvCache is move-only (no default ctor)

  // Stage storage, in pipeline order.
  DeviceBuffer input_;  // [H]
  DeviceBuffer rms1_;   // [H]
  DeviceBuffer q_;      // [n_heads*hd]
  DeviceBuffer k_;      // [n_kv*hd]
  DeviceBuffer v_;      // [n_kv*hd]
  DeviceBuffer rope_q_; // [n_heads*hd]
  DeviceBuffer rope_k_; // [n_kv*hd]
  DeviceBuffer attn_;   // [n_heads*hd]
  DeviceBuffer oproj_;  // [H]
  DeviceBuffer res1_;   // [H]
  DeviceBuffer rms2_;   // [H]
  DeviceBuffer gate_;   // [inter]
  DeviceBuffer up_;     // [inter]
  DeviceBuffer sgmu_;   // [inter]
  DeviceBuffer down_;   // [H]
  DeviceBuffer final_;  // [H]

  DeviceBuffer attn_scratch_;  // fp32 [2 * n_heads * max_seq_len]
  DeviceBuffer positions_;     // int64 [n_heads] (all == position)
};

}  // namespace cudalm
