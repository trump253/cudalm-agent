// CUDALM — Qwen35Config contract test (CPU).
//
// Pins the pinned 0.8B text config (docs/qwen35_architecture.md §2), the
// derived dimension helpers, the hybrid layer schedule, validity rules, and
// the 88-byte blob layout (field offsets are part of the format; the
// cross-language blob check lives in test_cudalm_v2_format).

#include "../../tests/common/check.h"

#include "cudalm/qwen35_config.h"
#include "cudalm/weight_format_v2.h"

#include <cstdint>
#include <cstring>

using namespace cudalm;

namespace {

int test_pinned_08b() {
  const Qwen35Config c = Qwen35Config::qwen35_08b();
  CHECK(c.valid());
  // Pinned values (config.json @ dc7cdfe, docs §2).
  CHECK_EQ(c.hidden_size, 1024);
  CHECK_EQ(c.num_hidden_layers, 24);
  CHECK_EQ(c.intermediate_size, 3584);
  CHECK_EQ(c.vocab_size, 248320);
  CHECK_EQ(c.n_heads, 8);
  CHECK_EQ(c.n_kv_heads, 2);
  CHECK_EQ(c.head_dim, 256);
  CHECK_EQ(c.lin_num_k_heads, 16);
  CHECK_EQ(c.lin_num_v_heads, 16);
  CHECK_EQ(c.lin_key_head_dim, 128);
  CHECK_EQ(c.lin_value_head_dim, 128);
  CHECK_EQ(c.lin_conv_kernel_dim, 4);
  CHECK_EQ(c.full_attention_interval, 4);
  CHECK_EQ(c.group_size, 128);
  CHECK_EQ(c.max_seq_len, 262144);
  CHECK(c.eps == 1e-6f);
  CHECK(c.rope_theta == 1e7f);
  CHECK(c.partial_rotary_factor == 0.25f);
  CHECK_EQ(c.mrope_section[0], 11);
  CHECK_EQ(c.mrope_section[1], 11);
  CHECK_EQ(c.mrope_section[2], 10);
  // Derived dimensions.
  CHECK_EQ(c.q_proj_out(), 4096);     // per-head fused [q; gate]
  CHECK_EQ(c.kv_proj_out(), 512);
  CHECK_EQ(c.o_proj_in(), 2048);
  CHECK_EQ(c.gqa_group(), 4);
  CHECK_EQ(c.rotary_dim(), 64);
  CHECK_EQ(c.linear_key_dim(), 2048);
  CHECK_EQ(c.linear_value_dim(), 2048);
  CHECK_EQ(c.linear_conv_dim(), 6144);
  CHECK_EQ(c.linear_conv_state_len(), 3);
  // Hybrid schedule: full attention exactly at 3,7,11,15,19,23.
  int fa_count = 0;
  for (int i = 0; i < 24; ++i) {
    const bool fa = c.is_full_attention(i);
    if (fa) ++fa_count;
    CHECK(fa == ((i + 1) % 4 == 0));
  }
  CHECK_EQ(fa_count, 6);
  CHECK(!c.is_full_attention(24));    // out of range is false, not UB
  CHECK(!c.is_full_attention(-1));
  TEST_PASS("qwen35_config pinned 0.8B");
  return 0;
}

int test_blob_roundtrip() {
  const Qwen35Config c = Qwen35Config::qwen35_08b();
  std::uint8_t blob[wfmt2::kConfigBytes] = {0};
  wfmt2::config_to_blob(c, blob);
  // Field offsets are part of the format: spot-check a few.
  auto rd_i32 = [](const std::uint8_t* p) {
    std::int32_t v; std::memcpy(&v, p, 4); return v;
  };
  auto rd_f32 = [](const std::uint8_t* p) {
    float v; std::memcpy(&v, p, 4); return v;
  };
  CHECK_EQ(rd_i32(blob + 0), 1024);        // hidden_size
  CHECK_EQ(rd_i32(blob + 8), 3584);        // intermediate_size
  CHECK_EQ(rd_i32(blob + 12), 248320);     // vocab_size
  CHECK_EQ(rd_i32(blob + 48), 4);          // full_attention_interval
  CHECK_EQ(rd_f32(blob + 60), 1e-6f);      // eps
  CHECK_EQ(rd_f32(blob + 64), 1e7f);       // rope_theta
  CHECK_EQ(rd_i32(blob + 72), 11);         // mrope_section[0]
  CHECK_EQ(rd_i32(blob + 80), 10);         // mrope_section[2]
  CHECK_EQ(static_cast<int>(blob[84]), 0); // reserved
  const Qwen35Config back = wfmt2::config_from_blob(blob);
  CHECK(back == c);
  TEST_PASS("qwen35_config blob round-trip");
  return 0;
}

int test_validity_rules() {
  auto expect_invalid = [](const Qwen35Config& c, const char* why) {
    if (c.valid()) {
      std::fprintf(stderr, "expected invalid config accepted: %s\n", why);
      return 1;
    }
    return 0;
  };

  Qwen35Config g = Qwen35Config::qwen35_08b();
  CHECK(g.valid());

  g.n_kv_heads = 3;                        // GQA: heads % kv != 0
  if (expect_invalid(g, "kv not dividing heads")) return 1;

  g = Qwen35Config::qwen35_08b();
  g.mrope_section[0] = 12;                 // sum != rotary_dim/2
  if (expect_invalid(g, "mrope sum mismatch")) return 1;

  g = Qwen35Config::qwen35_08b();
  g.partial_rotary_factor = 0.3f;          // rotary_dim 77 -> odd
  if (expect_invalid(g, "odd rotary dim")) return 1;

  g = Qwen35Config::qwen35_08b();
  g.hidden_size = 1000;                    // H % G != 0
  if (expect_invalid(g, "H % group")) return 1;

  g = Qwen35Config::qwen35_08b();
  g.lin_value_head_dim = 131;              // 16*131 % 128 != 0
  if (expect_invalid(g, "lin value dim % group")) return 1;

  g = Qwen35Config::qwen35_08b();
  g.eps = 0.0f;
  if (expect_invalid(g, "eps <= 0")) return 1;

  g = Qwen35Config::qwen35_08b();
  g.full_attention_interval = 1;           // every layer full: not hybrid
  if (expect_invalid(g, "interval < 2")) return 1;

  g = Qwen35Config::qwen35_08b();
  g.lin_conv_kernel_dim = 1;
  if (expect_invalid(g, "conv kernel < 2")) return 1;

  g = Qwen35Config::qwen35_08b();
  g.lin_num_v_heads = 4;                   // 4 % 16 != 0 (v heads per k head)
  if (expect_invalid(g, "v heads % k heads")) return 1;

  g = Qwen35Config::qwen35_08b();
  g.hidden_size = 1024;
  g.intermediate_size = 3584;
  CHECK(g.valid());  // sanity: back to a valid config
  TEST_PASS("qwen35_config validity rules");
  return 0;
}

int test_operator_eq() {
  const Qwen35Config a = Qwen35Config::qwen35_08b();
  Qwen35Config b = a;
  CHECK(a == b);
  b.eps = 2e-6f;
  CHECK(!(a == b));
  TEST_PASS("qwen35_config operator==");
  return 0;
}

}  // namespace

int main() {
  int rc = 0;
  rc |= test_pinned_08b();
  rc |= test_blob_roundtrip();
  rc |= test_validity_rules();
  rc |= test_operator_eq();
  if (rc != 0) std::fprintf(stderr, "test_qwen35_config FAILED\n");
  return rc;
}
