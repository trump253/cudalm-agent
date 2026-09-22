// CUDALM — decoder block wiring implementation. CUDALM-native.
//
// Pipeline (docs/bootstrap_plan_v0.1.md §4), one decode step:
//   x -> RMSNorm -> Q/K/V (W4A16) -> RoPE(Q,K) -> KV write -> causal
//   attention -> O-proj -> residual -> RMSNorm -> gate/up (W4A16) ->
//   SiLU(gate)*up -> down (W4A16) -> residual.
//
// The K cache stores ROPE'd K rows and the V cache stores raw V rows (the
// golden generator's convention); attention reads [0..position] inclusive,
// so the KV write happens before the attention call.

#include "cudalm/decoder_block.h"

#include <vector>

#include "cudalm/cuda_check.h"
#include "cudalm/kernels/attention.h"
#include "cudalm/kernels/elementwise.h"
#include "cudalm/kernels/int4_gemv.h"
#include "cudalm/kernels/rmsnorm.h"
#include "cudalm/kernels/rope.h"

namespace cudalm {

namespace {
std::size_t halves_bytes(std::size_t n) { return n * sizeof(__half); }
}  // namespace

DecoderBlock::DecoderBlock(const BlockWeights& weights, cudaStream_t stream)
    : w_(&weights), cfg_(weights.config()),
      kv_(new KvCache(weights.config(), stream)) {
  const std::size_t H = static_cast<std::size_t>(cfg_.hidden_size);
  const std::size_t qo = static_cast<std::size_t>(cfg_.n_heads) * cfg_.head_dim;
  const std::size_t kv = static_cast<std::size_t>(cfg_.n_kv_heads) * cfg_.head_dim;
  const std::size_t inter = static_cast<std::size_t>(cfg_.intermediate_size);

  input_.allocate(halves_bytes(H), stream);
  rms1_.allocate(halves_bytes(H), stream);
  q_.allocate(halves_bytes(qo), stream);
  k_.allocate(halves_bytes(kv), stream);
  v_.allocate(halves_bytes(kv), stream);
  rope_q_.allocate(halves_bytes(qo), stream);
  rope_k_.allocate(halves_bytes(kv), stream);
  attn_.allocate(halves_bytes(qo), stream);
  oproj_.allocate(halves_bytes(H), stream);
  res1_.allocate(halves_bytes(H), stream);
  rms2_.allocate(halves_bytes(H), stream);
  gate_.allocate(halves_bytes(inter), stream);
  up_.allocate(halves_bytes(inter), stream);
  sgmu_.allocate(halves_bytes(inter), stream);
  down_.allocate(halves_bytes(H), stream);
  final_.allocate(halves_bytes(H), stream);
  attn_scratch_.allocate(2 * static_cast<std::size_t>(cfg_.n_heads) *
                             cfg_.max_seq_len * sizeof(float),
                         stream);
  positions_.allocate(static_cast<std::size_t>(cfg_.n_heads) *
                          sizeof(std::int64_t),
                      stream);
}

void DecoderBlock::forward(int position, const __half* x_in,
                           cudaStream_t stream) {
  CUDALM_PRECONDITION(
      position >= 0 && position < cfg_.max_seq_len,
      "DecoderBlock::forward: position out of bounds [0, max_seq_len)");

  const int H = cfg_.hidden_size;
  const int hd = cfg_.head_dim;
  const int n_heads = cfg_.n_heads;
  const int n_kv = cfg_.n_kv_heads;
  const int inter = cfg_.intermediate_size;

  // 0) stage.input
  CUDA_CHECK(cudaMemcpyAsync(input_.data(), x_in, halves_bytes(H),
                             cudaMemcpyDeviceToDevice, stream));

  // position table for the RoPE kernels (all rows share `position`).
  {
    std::vector<std::int64_t> host(n_heads, position);
    positions_.copy_from_host(host.data(), host.size() * sizeof(std::int64_t),
                              stream);
  }

  // 1) RMSNorm 1
  rmsnorm_fp16(input_.data<__half>(), w_->attn_norm.data<__half>(),
                        rms1_.data<__half>(), 1, H, cfg_.eps, stream);

  // 2) Q / K / V projections (W4A16 GEMV)
  int4_gemv(w_->q_proj.weight.data<std::uint8_t>(),
                     w_->q_proj.scale.data<__half>(), rms1_.data<__half>(),
                     q_.data<__half>(), w_->q_proj.N, w_->q_proj.K, stream);
  int4_gemv(w_->k_proj.weight.data<std::uint8_t>(),
                     w_->k_proj.scale.data<__half>(), rms1_.data<__half>(),
                     k_.data<__half>(), w_->k_proj.N, w_->k_proj.K, stream);
  int4_gemv(w_->v_proj.weight.data<std::uint8_t>(),
                     w_->v_proj.scale.data<__half>(), rms1_.data<__half>(),
                     v_.data<__half>(), w_->v_proj.N, w_->v_proj.K, stream);

  // 3) RoPE (interleaved pair; cos/sin row `position`)
  kernels::rope_fp16(q_.data<__half>(), positions_.data<std::int64_t>(),
                     w_->rope_cos.data<__half>(), w_->rope_sin.data<__half>(),
                     rope_q_.data<__half>(), n_heads, hd, stream);
  kernels::rope_fp16(k_.data<__half>(), positions_.data<std::int64_t>(),
                     w_->rope_cos.data<__half>(), w_->rope_sin.data<__half>(),
                     rope_k_.data<__half>(), n_kv, hd, stream);

  // 4) KV write at `position` (K cache <- rope_k, V cache <- v)
  kv_->write(position, rope_k_.data<__half>(), v_.data<__half>(), stream);

  // 5) causal attention over [0..position]
  kernels::attention_decode_fp16(
      rope_q_.data<__half>(), kv_->k(), kv_->v(), position,
      attn_.data<__half>(), n_heads, n_kv, hd, cfg_.max_seq_len,
      attn_scratch_.data<float>(), stream);

  // 6) O projection + residual 1
  int4_gemv(w_->o_proj.weight.data<std::uint8_t>(),
                     w_->o_proj.scale.data<__half>(), attn_.data<__half>(),
                     oproj_.data<__half>(), w_->o_proj.N, w_->o_proj.K, stream);
  add_fp16(input_.data<__half>(), oproj_.data<__half>(),
                    res1_.data<__half>(), H, stream);

  // 7) RMSNorm 2
  rmsnorm_fp16(res1_.data<__half>(), w_->ffn_norm.data<__half>(),
                        rms2_.data<__half>(), 1, H, cfg_.eps, stream);

  // 8) gate / up (W4A16 GEMV) + SiLU(gate) * up
  int4_gemv(w_->gate_proj.weight.data<std::uint8_t>(),
                     w_->gate_proj.scale.data<__half>(), rms2_.data<__half>(),
                     gate_.data<__half>(), w_->gate_proj.N, w_->gate_proj.K,
                     stream);
  int4_gemv(w_->up_proj.weight.data<std::uint8_t>(),
                     w_->up_proj.scale.data<__half>(), rms2_.data<__half>(),
                     up_.data<__half>(), w_->up_proj.N, w_->up_proj.K, stream);
  silu_mul_fp16(gate_.data<__half>(), up_.data<__half>(),
                         sgmu_.data<__half>(), inter, stream);

  // 9) down (W4A16 GEMV) + residual 2
  int4_gemv(w_->down_proj.weight.data<std::uint8_t>(),
                     w_->down_proj.scale.data<__half>(), sgmu_.data<__half>(),
                     down_.data<__half>(), w_->down_proj.N, w_->down_proj.K,
                     stream);
  add_fp16(res1_.data<__half>(), down_.data<__half>(),
                    final_.data<__half>(), H, stream);
}

}  // namespace cudalm
