// CUDALM — Qwen3.5 Gated DeltaNet decoder layer wiring implementation.
// Math contract: docs/qwen35_architecture.md §8 (Gated DeltaNet) + §9 (decoder
// residual wiring) + §6.2 (Gated RMSNorm), pinned official source transformers
// fc91372 (torch fallback path). The persistent state (conv_state bf16,
// recurrent_state fp32) is owned here and updated IN PLACE each step.

#include "cudalm/qwen35_deltanet.h"

#include <cstddef>

#include "cudalm/cuda_check.h"
#include "cudalm/kernels/int4_gemv_bf16.h"
#include "cudalm/kernels/qwen35_deltanet_kernels.h"
#include "cudalm/kernels/qwen35_kernels.h"

namespace cudalm {

namespace {
std::size_t bf_bytes(std::size_t n) { return n * sizeof(__nv_bfloat16); }
std::size_t f32_bytes(std::size_t n) { return n * sizeof(float); }
}  // namespace

Qwen35DeltaNetLayer::Qwen35DeltaNetLayer(const Qwen35LayerWeights& weights,
                                         cudaStream_t stream)
    : w_(&weights), cfg_(weights.config()) {
  CUDALM_PRECONDITION(
      weights.config().is_linear_attention(weights.layer_idx()),
      "Qwen35DeltaNetLayer: weights must be a linear-attention (DeltaNet) "
      "layer");
  CUDALM_PRECONDITION(
      cfg_.lin_num_v_heads == cfg_.lin_num_k_heads,
      "Qwen35DeltaNetLayer: pinned 0.8B requires num_v_heads == "
      "num_k_heads (no repeat_interleave)");

  const std::size_t H = static_cast<std::size_t>(cfg_.hidden_size);
  const std::size_t conv_dim = static_cast<std::size_t>(cfg_.linear_conv_dim());
  const std::size_t key_dim = static_cast<std::size_t>(cfg_.linear_key_dim());
  const std::size_t value_dim =
      static_cast<std::size_t>(cfg_.linear_value_dim());
  const std::size_t n_heads = static_cast<std::size_t>(cfg_.lin_num_v_heads);
  const std::size_t hd = static_cast<std::size_t>(cfg_.lin_value_head_dim);
  const std::size_t inter = static_cast<std::size_t>(cfg_.intermediate_size);
  const std::size_t conv_state_len =
      static_cast<std::size_t>(cfg_.linear_conv_state_len());

  // Persistent state (zero-initialized; the first token starts from zero).
  conv_state_.allocate(bf_bytes(conv_dim * conv_state_len), stream);
  rec_state_.allocate(f32_bytes(n_heads * hd * hd), stream);

  // Stage buffers.
  input_.allocate(bf_bytes(H), stream);
  rms1_.allocate(bf_bytes(H), stream);
  mixed_.allocate(bf_bytes(conv_dim), stream);
  z_.allocate(bf_bytes(value_dim), stream);
  b_.allocate(bf_bytes(n_heads), stream);
  a_.allocate(bf_bytes(n_heads), stream);
  conv_out_.allocate(bf_bytes(conv_dim), stream);
  conv_silu_.allocate(bf_bytes(conv_dim), stream);
  q_.allocate(bf_bytes(key_dim), stream);
  k_.allocate(bf_bytes(key_dim), stream);
  v_.allocate(bf_bytes(value_dim), stream);
  beta_.allocate(bf_bytes(n_heads), stream);
  g_.allocate(f32_bytes(n_heads), stream);
  core_.allocate(bf_bytes(value_dim), stream);
  gated_.allocate(bf_bytes(value_dim), stream);
  out_proj_.allocate(bf_bytes(H), stream);
  res1_.allocate(bf_bytes(H), stream);
  rms2_.allocate(bf_bytes(H), stream);
  mlp_gate_.allocate(bf_bytes(inter), stream);
  mlp_up_.allocate(bf_bytes(inter), stream);
  silu_mul_.allocate(bf_bytes(inter), stream);
  mlp_down_.allocate(bf_bytes(H), stream);
  final_.allocate(bf_bytes(H), stream);

  reset_state(stream);
}

void Qwen35DeltaNetLayer::reset_state(cudaStream_t stream) {
  CUDA_CHECK(cudaMemsetAsync(conv_state_.data(), 0, conv_state_.bytes(),
                             stream));
  CUDA_CHECK(cudaMemsetAsync(rec_state_.data(), 0, rec_state_.bytes(), stream));
}

void Qwen35DeltaNetLayer::seed_state(const __nv_bfloat16* conv,
                                     const float* rec, cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(conv_state_.data(), conv, conv_state_.bytes(),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(rec_state_.data(), rec, rec_state_.bytes(),
                             cudaMemcpyHostToDevice, stream));
}

void Qwen35DeltaNetLayer::forward(int position, const __nv_bfloat16* x_in,
                                  cudaStream_t stream) {
  CUDALM_PRECONDITION(
      position >= 0 && position < cfg_.max_seq_len,
      "Qwen35DeltaNetLayer::forward: position out of bounds [0, max_seq_len)");

  const std::size_t H = static_cast<std::size_t>(cfg_.hidden_size);
  const std::size_t conv_dim = static_cast<std::size_t>(cfg_.linear_conv_dim());
  const std::size_t key_dim = static_cast<std::size_t>(cfg_.linear_key_dim());
  const std::size_t value_dim = static_cast<std::size_t>(cfg_.linear_value_dim());
  const std::size_t n_heads = static_cast<std::size_t>(cfg_.lin_num_v_heads);
  const std::size_t hd = static_cast<std::size_t>(cfg_.lin_value_head_dim);
  const std::size_t inter = static_cast<std::size_t>(cfg_.intermediate_size);
  const float eps = cfg_.eps;

  // 0) stage.input
  CUDA_CHECK(cudaMemcpyAsync(input_.data(), x_in, bf_bytes(H),
                             cudaMemcpyDeviceToDevice, stream));

  // 1) zero-centered RMSNorm 1 (input_layernorm)
  kernels::qwen35_rmsnorm_zc_bf16(
      input_.data<__nv_bfloat16>(),
      w_->input_layernorm.data<__nv_bfloat16>(), rms1_.data<__nv_bfloat16>(),
      1, static_cast<int>(H), eps, stream);

  // 2) projections (W4A16)
  int4_gemv_bf16(w_->in_proj_qkv.weight.data<std::uint8_t>(),
                 w_->in_proj_qkv.scale.data<__half>(),
                 rms1_.data<__nv_bfloat16>(), mixed_.data<__nv_bfloat16>(),
                 w_->in_proj_qkv.N, w_->in_proj_qkv.K, stream);
  int4_gemv_bf16(w_->in_proj_z.weight.data<std::uint8_t>(),
                 w_->in_proj_z.scale.data<__half>(),
                 rms1_.data<__nv_bfloat16>(), z_.data<__nv_bfloat16>(),
                 w_->in_proj_z.N, w_->in_proj_z.K, stream);
  int4_gemv_bf16(w_->in_proj_b.weight.data<std::uint8_t>(),
                 w_->in_proj_b.scale.data<__half>(),
                 rms1_.data<__nv_bfloat16>(), b_.data<__nv_bfloat16>(),
                 w_->in_proj_b.N, w_->in_proj_b.K, stream);
  int4_gemv_bf16(w_->in_proj_a.weight.data<std::uint8_t>(),
                 w_->in_proj_a.scale.data<__half>(),
                 rms1_.data<__nv_bfloat16>(), a_.data<__nv_bfloat16>(),
                 w_->in_proj_a.N, w_->in_proj_a.K, stream);

  // 3) depthwise causal conv1d decode update (in-place conv_state) + SiLU
  kernels::qwen35_deltanet_conv_decode_bf16(
      conv_state_.data<__nv_bfloat16>(), mixed_.data<__nv_bfloat16>(),
      w_->conv1d_weight.data<__nv_bfloat16>(), conv_out_.data<__nv_bfloat16>(),
      conv_silu_.data<__nv_bfloat16>(), static_cast<int>(conv_dim), stream);

  // 4) split conv_silu [conv_dim] -> q [key_dim] / k [key_dim] / v [value_dim]
  CUDA_CHECK(cudaMemcpyAsync(q_.data(), conv_silu_.data(), bf_bytes(key_dim),
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      k_.data(),
      static_cast<const char*>(conv_silu_.data()) + bf_bytes(key_dim),
      bf_bytes(key_dim), cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      v_.data(),
      static_cast<const char*>(conv_silu_.data()) + bf_bytes(2 * key_dim),
      bf_bytes(value_dim), cudaMemcpyDeviceToDevice, stream));

  // 5) g (fp32) + beta (bf16)
  kernels::qwen35_deltanet_gbeta_bf16(
      b_.data<__nv_bfloat16>(), a_.data<__nv_bfloat16>(),
      w_->A_log.data<float>(), w_->dt_bias.data<__nv_bfloat16>(),
      beta_.data<__nv_bfloat16>(), g_.data<float>(),
      static_cast<int>(n_heads), stream);

  // 6) gated delta-rule recurrent update (in-place recurrent_state, fp32)
  kernels::qwen35_deltanet_delta_rule_fp32(
      q_.data<__nv_bfloat16>(), k_.data<__nv_bfloat16>(),
      v_.data<__nv_bfloat16>(), g_.data<float>(),
      beta_.data<__nv_bfloat16>(), rec_state_.data<float>(),
      core_.data<__nv_bfloat16>(), static_cast<int>(n_heads),
      static_cast<int>(hd), eps, stream);

  // 7) gated RMSNorm (core, gate=z, w=linear_norm fp32)
  kernels::qwen35_rmsnorm_gated_bf16(
      core_.data<__nv_bfloat16>(), z_.data<__nv_bfloat16>(),
      w_->linear_norm.data<float>(), gated_.data<__nv_bfloat16>(),
      static_cast<int>(n_heads), static_cast<int>(hd), eps, stream);

  // 8) out_proj (W4A16) + residual 1
  int4_gemv_bf16(w_->out_proj.weight.data<std::uint8_t>(),
                 w_->out_proj.scale.data<__half>(),
                 gated_.data<__nv_bfloat16>(), out_proj_.data<__nv_bfloat16>(),
                 w_->out_proj.N, w_->out_proj.K, stream);
  kernels::qwen35_add_bf16(input_.data<__nv_bfloat16>(),
                           out_proj_.data<__nv_bfloat16>(),
                           res1_.data<__nv_bfloat16>(), H, stream);

  // 9) zero-centered RMSNorm 2 (post_attention_layernorm)
  kernels::qwen35_rmsnorm_zc_bf16(
      res1_.data<__nv_bfloat16>(),
      w_->post_attention_ln.data<__nv_bfloat16>(),
      rms2_.data<__nv_bfloat16>(), 1, static_cast<int>(H), eps, stream);

  // 10) SwiGLU MLP (gate/up W4A16, silu*mul, down W4A16) + residual 2
  int4_gemv_bf16(w_->gate_proj.weight.data<std::uint8_t>(),
                 w_->gate_proj.scale.data<__half>(),
                 rms2_.data<__nv_bfloat16>(), mlp_gate_.data<__nv_bfloat16>(),
                 w_->gate_proj.N, w_->gate_proj.K, stream);
  int4_gemv_bf16(w_->up_proj.weight.data<std::uint8_t>(),
                 w_->up_proj.scale.data<__half>(),
                 rms2_.data<__nv_bfloat16>(), mlp_up_.data<__nv_bfloat16>(),
                 w_->up_proj.N, w_->up_proj.K, stream);
  kernels::qwen35_silu_mul_bf16(mlp_gate_.data<__nv_bfloat16>(),
                                mlp_up_.data<__nv_bfloat16>(),
                                silu_mul_.data<__nv_bfloat16>(), inter,
                                stream);
  int4_gemv_bf16(w_->down_proj.weight.data<std::uint8_t>(),
                 w_->down_proj.scale.data<__half>(),
                 silu_mul_.data<__nv_bfloat16>(),
                 mlp_down_.data<__nv_bfloat16>(), w_->down_proj.N,
                 w_->down_proj.K, stream);
  kernels::qwen35_add_bf16(res1_.data<__nv_bfloat16>(),
                           mlp_down_.data<__nv_bfloat16>(),
                           final_.data<__nv_bfloat16>(), H, stream);
}

}  // namespace cudalm
