// CUDALM — bootstrap smoke test (CPU-only): headers compile, value types
// behave. No CUDA calls.

#include "../../tests/common/check.h"

#include "cudalm/tensor.h"
#include "cudalm/model_config.h"

using namespace cudalm;

int main() {
  // Dtype metadata
  CHECK_EQ(dtype_element_bytes(Dtype::kFp16), std::size_t(2));
  CHECK_EQ(dtype_element_bytes(Dtype::kInt4Packed), std::size_t(1));
  CHECK_EQ(dtype_element_bytes(Dtype::kFp16Scale), std::size_t(2));
  CHECK_EQ(dtype_element_bytes(Dtype::kFp32), std::size_t(4));
  CHECK(std::string(dtype_name(Dtype::kInt4Packed)) == "int4_packed");

  // TensorView basics (non-owning, contiguous)
  int dummy = 0;
  TensorView v(&dummy, Dtype::kFp32, {2, 3});
  CHECK_EQ(v.ndim(), 2);
  CHECK_EQ(v.numel(), std::int64_t(6));
  CHECK_EQ(v.bytes(), std::size_t(24));
  CHECK(v.contiguous());
  CHECK_EQ(v.shape_string(), std::string("[2, 3]"));

  // Int4 packed: shape is packed bytes (N, K/2)
  TensorView w(&dummy, Dtype::kInt4Packed, {4, 512});  // K = 1024
  CHECK_EQ(w.bytes(), std::size_t(4 * 512));

  // ModelConfig defaults are valid and self-consistent
  const ModelConfig c = ModelConfig::v01_default();
  CHECK(c.valid());
  CHECK_EQ(c.q_proj_out(), c.hidden_size);            // n_heads*head_dim == H
  CHECK_EQ(c.kv_proj_out(), 4 * 128);
  CHECK_EQ(c.gqa_group(), 2);
  CHECK_EQ(c.kv_head_for_q(0), 0);
  CHECK_EQ(c.kv_head_for_q(3), 1);
  CHECK_EQ(c.rope_pairs(), 64);
  CHECK_EQ(c.intermediate_size % c.group_size, 0);

  // Invalid config rejected
  ModelConfig bad = c;
  bad.n_kv_heads = 3;  // 8 % 3 != 0
  CHECK(!bad.valid());

  TEST_PASS("bootstrap_cpu");
  return 0;
}
