// CUDALM — v0.5 Phase A: state-layout formulas gate (CPU-only).
//
// Pins that ALL pool sizing derives from the Qwen35Config hybrid schedule
// (no magic constants): the full/linear layer counts and the per-page /
// per-slot byte sizes, for the pinned 0.8B config (page_tokens 4/8/16) and
// the small synthetic config used by the CUDA state tests.

#include "../../tests/common/check.h"
#include "cudalm/qwen35_state_layout.h"

#include <cstddef>
#include <cstdio>
#include <cuda_bf16.h>

using namespace cudalm;

namespace {

// The small synthetic hybrid config used by the CUDA state-manager tests
// (must stay valid() and match tests/cuda/test_qwen35_state_manager.cpp).
Qwen35Config small_config() {
  Qwen35Config c;
  c.hidden_size = 256;
  c.num_hidden_layers = 8;
  c.intermediate_size = 512;
  c.vocab_size = 1024;
  c.n_heads = 4;
  c.n_kv_heads = 2;
  c.head_dim = 64;
  c.lin_num_k_heads = 4;
  c.lin_num_v_heads = 4;
  c.lin_key_head_dim = 32;
  c.lin_value_head_dim = 32;
  c.lin_conv_kernel_dim = 4;
  c.full_attention_interval = 4;
  c.group_size = 128;
  c.max_seq_len = 64;
  c.eps = 1e-6f;
  c.rope_theta = 1e7f;
  c.partial_rotary_factor = 0.25f;
  c.mrope_section[0] = 4;
  c.mrope_section[1] = 2;
  c.mrope_section[2] = 2;
  return c;
}

int test_08b_layer_counts() {
  const Qwen35Config c = Qwen35Config::qwen35_08b();
  CHECK(c.valid());
  // 24 layers, interval 4 -> full at 3,7,11,15,19,23 (6), linear 18.
  CHECK_EQ(qwen35_num_full_layers(c), 6);
  CHECK_EQ(qwen35_num_linear_layers(c), 18);
  CHECK_EQ(qwen35_num_full_layers(c) + qwen35_num_linear_layers(c),
           c.num_hidden_layers);
  std::printf("  [ok] 0.8B schedule: 6 full + 18 linear (from config, no "
              "hardcoding)\n");
  return 0;
}

int test_08b_kv_page_bytes() {
  const Qwen35Config c = Qwen35Config::qwen35_08b();
  // bytes_per_page = 2 (K+V) * 6 full * n_kv(2) * page_tokens * head_dim(256)
  //                  * 2B
  const std::size_t base = 2ull * 6 * 2 * 256 * 2;  // per page_token
  CHECK_EQ(qwen35_kv_page_bytes(c, 4), base * 4);
  CHECK_EQ(qwen35_kv_page_bytes(c, 8), base * 8);
  CHECK_EQ(qwen35_kv_page_bytes(c, 16), base * 16);
  // Hand-pinned: page_tokens = 16 -> 2*6*2*16*256*2 = 196,608 bytes.
  CHECK_EQ(qwen35_kv_page_bytes(c, 16), 196608ull);
  std::printf("  [ok] 0.8B KV page bytes for page_tokens 4/8/16 (16 -> "
              "196,608 B)\n");
  return 0;
}

int test_08b_delta_slot_bytes() {
  const Qwen35Config c = Qwen35Config::qwen35_08b();
  // per layer: conv bf16 [6144, 3] = 36,864 B;
  //            rec fp32 [16, 128, 128] = 1,048,576 B;
  // per slot:  18 * (36,864 + 1,048,576) = 19,537,920 B.
  CHECK_EQ(c.linear_conv_dim(), 6144);
  CHECK_EQ(c.linear_conv_state_len(), 3);
  CHECK_EQ(qwen35_delta_slot_bytes(c),
           18ull * (6144ull * 3 * sizeof(__nv_bfloat16) +
                    16ull * 128 * 128 * sizeof(float)));
  CHECK_EQ(qwen35_delta_slot_bytes(c), 19537920ull);
  std::printf("  [ok] 0.8B Delta slot bytes: 19,537,920 B per slot\n");
  return 0;
}

int test_small_config_formulas() {
  const Qwen35Config c = small_config();
  CHECK(c.valid());
  // 8 layers, interval 4 -> full at 3,7 (2), linear 6.
  CHECK_EQ(qwen35_num_full_layers(c), 2);
  CHECK_EQ(qwen35_num_linear_layers(c), 6);
  // KV page (page_tokens=4): 2 * 2 full * 2 kv * 4 * 64 * 2 = 4,096 B.
  CHECK_EQ(qwen35_kv_page_bytes(c, 4), 4096ull);
  CHECK_EQ(qwen35_kv_page_bytes(c, 8), 8192ull);
  // Delta slot: conv bf16 [384,3] = 2,304 B; rec fp32 [4,32,32] = 16,384 B;
  // slot = 6 * (2,304 + 16,384) = 112,128 B.
  CHECK_EQ(c.linear_conv_dim(), 384);
  CHECK_EQ(qwen35_delta_slot_bytes(c),
           6ull * (384ull * 3 * sizeof(__nv_bfloat16) +
                   4ull * 32 * 32 * sizeof(float)));
  CHECK_EQ(qwen35_delta_slot_bytes(c), 112128ull);
  std::printf("  [ok] small synthetic config: 2 full + 6 linear; page/slot "
              "bytes pinned\n");
  return 0;
}

}  // namespace

int main() {
  std::printf("test_state_pool_formulas: CUDALM v0.5 Phase A layout gate\n");
  int rc = 0;
  rc |= test_08b_layer_counts();
  rc |= test_08b_kv_page_bytes();
  rc |= test_08b_delta_slot_bytes();
  rc |= test_small_config_formulas();
  if (rc != 0) {
    std::fprintf(stderr, "test_state_pool_formulas: FAIL\n");
    return rc;
  }
  std::printf("test_state_pool_formulas: PASS\n");
  return 0;
}
