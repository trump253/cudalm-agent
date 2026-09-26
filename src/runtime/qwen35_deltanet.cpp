// CUDALM — Qwen3.5 Gated DeltaNet decoder layer wiring implementation.
// Math contract: docs/qwen35_architecture.md §8 (Gated DeltaNet) + §9 (decoder
// residual wiring) + §6.2 (Gated RMSNorm), pinned official source transformers
// fc91372 (torch fallback path). The persistent state (conv_state bf16,
// recurrent_state fp32) is owned here and updated IN PLACE each step.

#include "cudalm/qwen35_deltanet.h"

#include <cstddef>

#include "cudalm/cuda_check.h"
#include "cudalm/kernels/int4_gemv_bf16.h"
#include "cudalm/kernels/batch_decode.h"
#include "cudalm/kernels/qwen35_deltanet_kernels.h"
#include "cudalm/kernels/qwen35_kernels.h"

namespace cudalm {

using kernels::batch_int4_gemv_bf16;  // v0.6 Phase B batch GEMV (kernels namespace)

namespace {
std::size_t bf_bytes(std::size_t n) { return n * sizeof(__nv_bfloat16); }
std::size_t f32_bytes(std::size_t n) { return n * sizeof(float); }
}  // namespace

void qwen35_deltanet_require_supported_config(const Qwen35Config& c,
                                              int layer_idx) {
  CUDALM_PRECONDITION(
      c.is_linear_attention(layer_idx),
      "Qwen35DeltaNetLayer: layer must be linear-attention (Gated DeltaNet)");
  CUDALM_PRECONDITION(
      c.lin_conv_kernel_dim == 4,
      "Qwen35DeltaNetLayer: the conv kernel hardcodes a depthwise causal conv "
      "of kernel 4 (conv_dim*3 state, conv_dim*4 input buffer)");
  CUDALM_PRECONDITION(
      c.linear_conv_state_len() == 3,
      "Qwen35DeltaNetLayer: the conv state must hold kernel-1 == 3 tokens");
  CUDALM_PRECONDITION(
      c.lin_key_head_dim == c.lin_value_head_dim,
      "Qwen35DeltaNetLayer: key/value head dims must be equal (the kernels do "
      "not repeat_interleave)");
  CUDALM_PRECONDITION(
      c.lin_key_head_dim == 128,
      "Qwen35DeltaNetLayer: the delta-rule and gated-RMSNorm kernels hardcode "
      "head dim 128");
  CUDALM_PRECONDITION(
      c.lin_num_k_heads == c.lin_num_v_heads,
      "Qwen35DeltaNetLayer: num_k_heads must equal num_v_heads (no "
      "repeat_interleave)");
}

Qwen35DeltaNetLayer::Qwen35DeltaNetLayer(const Qwen35LayerWeights& weights,
                                         cudaStream_t stream)
    : w_(&weights), cfg_(weights.config()) {
  // Reject any config the Phase C kernels do not support before touching a
  // single buffer (the kernels hardcode kernel-4 / state-3 / head-dim-128 /
  // k==v heads; a silent OOB or semantic mismatch would be far worse than a
  // loud abort). See qwen35_deltanet_require_supported_config.
  qwen35_deltanet_require_supported_config(cfg_, weights.layer_idx());

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
  // Legacy owned-state path: the exact v0.2 pipeline over the layer's own
  // persistent state (behavior unchanged).
  forward_impl(position, x_in, conv_state_.data<__nv_bfloat16>(),
               rec_state_.data<float>(), stream);
}

void Qwen35DeltaNetLayer::forward_with_state(int position,
                                             const __nv_bfloat16* x_in,
                                             __nv_bfloat16* ext_conv,
                                             float* ext_rec,
                                             cudaStream_t stream) {
  // v0.5 Phase B external-state path: the SAME pipeline, state addressed
  // directly in the caller's buffers (e.g. a Qwen35DeltaStatePool slot) —
  // read + updated in place, no host roundtrip, layer-owned state untouched.
  forward_impl(position, x_in, ext_conv, ext_rec, stream);
}

void Qwen35DeltaNetLayer::forward_impl(int position,
                                       const __nv_bfloat16* x_in,
                                       __nv_bfloat16* conv_state,
                                       float* rec_state,
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
      conv_state, mixed_.data<__nv_bfloat16>(),
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
      beta_.data<__nv_bfloat16>(), rec_state,
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


// ---------------------------------------------------------------------------
// v0.6 Phase B: TRUE BATCHED decode (B rows, one launch per stage; per row
// BIT-IDENTICAL to forward_with_state() — see include/cudalm/kernels/
// batch_decode.h and docs §26).
// ---------------------------------------------------------------------------
void Qwen35DeltaNetLayer::grow_batch(int B, cudaStream_t stream) {
  if (B <= batch_cap_) return;
  batch_cap_ = B;
  const std::size_t b = static_cast<std::size_t>(B);
  const std::size_t H = static_cast<std::size_t>(cfg_.hidden_size);
  const std::size_t conv_dim = static_cast<std::size_t>(cfg_.linear_conv_dim());
  const std::size_t key_dim = static_cast<std::size_t>(cfg_.linear_key_dim());
  const std::size_t value_dim =
      static_cast<std::size_t>(cfg_.linear_value_dim());
  const std::size_t n_heads = static_cast<std::size_t>(cfg_.lin_num_v_heads);
  const std::size_t inter = static_cast<std::size_t>(cfg_.intermediate_size);
  b_input_.allocate(bf_bytes(b * H), stream);
  b_rms1_.allocate(bf_bytes(b * H), stream);
  b_mixed_.allocate(bf_bytes(b * conv_dim), stream);
  b_z_.allocate(bf_bytes(b * value_dim), stream);
  b_b_.allocate(bf_bytes(b * n_heads), stream);
  b_a_.allocate(bf_bytes(b * n_heads), stream);
  b_conv_out_.allocate(bf_bytes(b * conv_dim), stream);
  b_conv_silu_.allocate(bf_bytes(b * conv_dim), stream);
  b_q_.allocate(bf_bytes(b * key_dim), stream);
  b_k_.allocate(bf_bytes(b * key_dim), stream);
  b_v_.allocate(bf_bytes(b * value_dim), stream);
  b_beta_.allocate(bf_bytes(b * n_heads), stream);
  b_g_.allocate(f32_bytes(b * n_heads), stream);
  b_core_.allocate(bf_bytes(b * value_dim), stream);
  b_gated_.allocate(bf_bytes(b * value_dim), stream);
  b_out_proj_.allocate(bf_bytes(b * H), stream);
  b_res1_.allocate(bf_bytes(b * H), stream);
  b_rms2_.allocate(bf_bytes(b * H), stream);
  b_mlp_gate_.allocate(bf_bytes(b * inter), stream);
  b_mlp_up_.allocate(bf_bytes(b * inter), stream);
  b_silu_mul_.allocate(bf_bytes(b * inter), stream);
  b_mlp_down_.allocate(bf_bytes(b * H), stream);
  b_final_.allocate(bf_bytes(b * H), stream);
}

void Qwen35DeltaNetLayer::forward_batch_with_state(
    int B, const int* d_slots, const __nv_bfloat16* x_in,
    __nv_bfloat16* conv_base, float* rec_base, cudaStream_t stream) {
  CUDALM_PRECONDITION(B >= 1, "forward_batch_with_state: B >= 1");
  grow_batch(B, stream);
  const int H = cfg_.hidden_size;
  const int conv_dim = cfg_.linear_conv_dim();
  const int key_dim = cfg_.linear_key_dim();
  const int value_dim = cfg_.linear_value_dim();
  const int n_heads = cfg_.lin_num_v_heads;
  const int inter = cfg_.intermediate_size;
  const float eps = cfg_.eps;

  // 0) stage.input: plain D2D copy of [B][H] (bit-exact, like the frozen
  //    single path's stage.input).
  CUDA_CHECK(cudaMemcpyAsync(b_input_.data(), x_in,
                             bf_bytes(static_cast<std::size_t>(B) * H),
                             cudaMemcpyDeviceToDevice, stream));

  // 1) zero-centered RMSNorm 1 (frozen kernel, M = B rows).
  kernels::qwen35_rmsnorm_zc_bf16(
      b_input_.data<__nv_bfloat16>(),
      w_->input_layernorm.data<__nv_bfloat16>(),
      b_rms1_.data<__nv_bfloat16>(), B, H, eps, stream);

  // 2) projections (W4A16, batched GEMV — per row bit-identical to the
  //    frozen single-row int4_gemv_bf16).
  batch_int4_gemv_bf16(w_->in_proj_qkv.weight.data<std::uint8_t>(),
                       w_->in_proj_qkv.scale.data<__half>(),
                       b_rms1_.data<__nv_bfloat16>(),
                       b_mixed_.data<__nv_bfloat16>(), w_->in_proj_qkv.N,
                       w_->in_proj_qkv.K, B, stream);
  batch_int4_gemv_bf16(w_->in_proj_z.weight.data<std::uint8_t>(),
                       w_->in_proj_z.scale.data<__half>(),
                       b_rms1_.data<__nv_bfloat16>(), b_z_.data<__nv_bfloat16>(),
                       w_->in_proj_z.N, w_->in_proj_z.K, B, stream);
  batch_int4_gemv_bf16(w_->in_proj_b.weight.data<std::uint8_t>(),
                       w_->in_proj_b.scale.data<__half>(),
                       b_rms1_.data<__nv_bfloat16>(), b_b_.data<__nv_bfloat16>(),
                       w_->in_proj_b.N, w_->in_proj_b.K, B, stream);
  batch_int4_gemv_bf16(w_->in_proj_a.weight.data<std::uint8_t>(),
                       w_->in_proj_a.scale.data<__half>(),
                       b_rms1_.data<__nv_bfloat16>(), b_a_.data<__nv_bfloat16>(),
                       w_->in_proj_a.N, w_->in_proj_a.K, B, stream);

  // 3) depthwise causal conv1d decode update (per-row slot) + SiLU.
  kernels::batch_deltanet_conv_decode_bf16(
      conv_base, d_slots, b_mixed_.data<__nv_bfloat16>(),
      w_->conv1d_weight.data<__nv_bfloat16>(),
      b_conv_out_.data<__nv_bfloat16>(), b_conv_silu_.data<__nv_bfloat16>(),
      conv_dim, B, stream);

  // 4) split conv_silu -> q/k/v (one kernel; the frozen path's three D2D
  //    copies as a single batched pass).
  kernels::batch_deltanet_conv_split_bf16(
      b_conv_silu_.data<__nv_bfloat16>(), b_q_.data<__nv_bfloat16>(),
      b_k_.data<__nv_bfloat16>(), b_v_.data<__nv_bfloat16>(), key_dim,
      value_dim, B, stream);

  // 5) g (fp32) + beta (bf16) — batched.
  kernels::batch_deltanet_gbeta_bf16(
      b_b_.data<__nv_bfloat16>(), b_a_.data<__nv_bfloat16>(),
      w_->A_log.data<float>(), w_->dt_bias.data<__nv_bfloat16>(),
      b_beta_.data<__nv_bfloat16>(), b_g_.data<float>(), n_heads, B, stream);

  // 6) gated delta-rule recurrent decode update (per-row slot, in place).
  kernels::batch_deltanet_delta_rule_fp32(
      b_q_.data<__nv_bfloat16>(), b_k_.data<__nv_bfloat16>(),
      b_v_.data<__nv_bfloat16>(), b_g_.data<float>(),
      b_beta_.data<__nv_bfloat16>(), rec_base, d_slots,
      b_core_.data<__nv_bfloat16>(), n_heads, cfg_.lin_value_head_dim, eps, B,
      stream);

  // 7) gated RMSNorm (frozen kernel, n = B*n_heads rows).
  kernels::qwen35_rmsnorm_gated_bf16(
      b_core_.data<__nv_bfloat16>(), b_z_.data<__nv_bfloat16>(),
      w_->linear_norm.data<float>(), b_gated_.data<__nv_bfloat16>(),
      B * n_heads, cfg_.lin_value_head_dim, eps, stream);

  // 8) out_proj (W4A16, batched) + residual 1 (frozen add, flat [B][H]).
  batch_int4_gemv_bf16(w_->out_proj.weight.data<std::uint8_t>(),
                       w_->out_proj.scale.data<__half>(),
                       b_gated_.data<__nv_bfloat16>(),
                       b_out_proj_.data<__nv_bfloat16>(), w_->out_proj.N,
                       w_->out_proj.K, B, stream);
  kernels::qwen35_add_bf16(
      b_input_.data<__nv_bfloat16>(), b_out_proj_.data<__nv_bfloat16>(),
      b_res1_.data<__nv_bfloat16>(),
      static_cast<std::size_t>(B) * H, stream);

  // 9) zero-centered RMSNorm 2 (M = B rows).
  kernels::qwen35_rmsnorm_zc_bf16(
      b_res1_.data<__nv_bfloat16>(),
      w_->post_attention_ln.data<__nv_bfloat16>(),
      b_rms2_.data<__nv_bfloat16>(), B, H, eps, stream);

  // 10) SwiGLU MLP + residual 2.
  batch_int4_gemv_bf16(w_->gate_proj.weight.data<std::uint8_t>(),
                       w_->gate_proj.scale.data<__half>(),
                       b_rms2_.data<__nv_bfloat16>(),
                       b_mlp_gate_.data<__nv_bfloat16>(), w_->gate_proj.N,
                       w_->gate_proj.K, B, stream);
  batch_int4_gemv_bf16(w_->up_proj.weight.data<std::uint8_t>(),
                       w_->up_proj.scale.data<__half>(),
                       b_rms2_.data<__nv_bfloat16>(),
                       b_mlp_up_.data<__nv_bfloat16>(), w_->up_proj.N,
                       w_->up_proj.K, B, stream);
  kernels::qwen35_silu_mul_bf16(b_mlp_gate_.data<__nv_bfloat16>(),
                                b_mlp_up_.data<__nv_bfloat16>(),
                                b_silu_mul_.data<__nv_bfloat16>(),
                                static_cast<std::size_t>(B) * inter, stream);
  batch_int4_gemv_bf16(w_->down_proj.weight.data<std::uint8_t>(),
                       w_->down_proj.scale.data<__half>(),
                       b_silu_mul_.data<__nv_bfloat16>(),
                       b_mlp_down_.data<__nv_bfloat16>(), w_->down_proj.N,
                       w_->down_proj.K, B, stream);
  kernels::qwen35_add_bf16(b_res1_.data<__nv_bfloat16>(),
                           b_mlp_down_.data<__nv_bfloat16>(),
                           b_final_.data<__nv_bfloat16>(),
                           static_cast<std::size_t>(B) * H, stream);
}

}  // namespace cudalm
