// CUDALM — standalone greedy argmax hard gate (v0.4 Phase A; CPU-only).
//
// Proves the greedy argmax (greedy.h) on its OWN, not via a model run: the
// contract is "maximize the numeric BF16 logit value; tie -> lowest token
// id". Covers the normal case, all-negative logits, an exact-maximum tie
// (lowest id wins), a single element, and a bf16 rounding tie (two distinct
// floats that round to the SAME bf16 value must also tie -> lowest id).

#include "../../tests/common/check.h"

#include "cudalm/greedy.h"

#include <cuda_bf16.h>

#include <vector>

using namespace cudalm;

namespace {

// Build a host bf16 array from a list of floats (via the RNE bf16 conversion).
std::vector<__nv_bfloat16> mk(const std::vector<float>& xs) {
  std::vector<__nv_bfloat16> b(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) b[i] = __float2bfloat16(xs[i]);
  return b;
}

int test_normal() {
  auto b = mk({1.0f, 5.0f, 3.0f, 2.0f, -100.0f});
  CHECK_EQ(argmax_bf16(b.data(), static_cast<int>(b.size())), 1);  // 5.0
  return 0;
}

int test_negative() {
  // All-negative: the max is the LEAST negative.
  auto b = mk({-5.0f, -1.0f, -3.0f, -2.0f, -9.0f});
  CHECK_EQ(argmax_bf16(b.data(), static_cast<int>(b.size())), 1);  // -1.0
  return 0;
}

int test_exact_tie_lowest_id() {
  // Three exact maxima (3.0 at ids 0, 3, 4) -> the lowest id (0) wins.
  auto b = mk({3.0f, 1.0f, 2.0f, 3.0f, 3.0f});
  CHECK_EQ(argmax_bf16(b.data(), static_cast<int>(b.size())), 0);
  // Tie NOT at the front -> the lowest of the tied ids (2) wins.
  auto b2 = mk({1.0f, 0.5f, 2.5f, 2.5f, -1.0f});
  CHECK_EQ(argmax_bf16(b2.data(), static_cast<int>(b2.size())), 2);
  return 0;
}

int test_single_element() {
  auto b = mk({-7.5f});
  CHECK_EQ(argmax_bf16(b.data(), 1), 0);
  return 0;
}

int test_bf16_rounding_tie() {
  // Two distinct floats that round to the SAME bf16 value must tie (the
  // argmax compares the bf16 value, not the original float). 0.1f and
  // 0.1000001f both round to the nearest bf16 of ~0.1; verify they are equal
  // as bf16 and that the argmax picks the lowest id on the tie.
  const float a = 0.1f;
  const float b = 0.1f + 1e-7f;
  const __nv_bfloat16 ba = __float2bfloat16(a);
  const __nv_bfloat16 bb = __float2bfloat16(b);
  CHECK_EQ(__bfloat162float(ba), __bfloat162float(bb));  // same bf16 value
  std::vector<__nv_bfloat16> arr = {bb, ba};  // [b, a] -> both == ~0.1
  // The argmax over the bf16 values: a tie -> lowest id (0).
  CHECK_EQ(argmax_bf16(arr.data(), 2), 0);
  return 0;
}

}  // namespace

int main() {
  if (int rc = test_normal()) return rc;
  if (int rc = test_negative()) return rc;
  if (int rc = test_exact_tie_lowest_id()) return rc;
  if (int rc = test_single_element()) return rc;
  if (int rc = test_bf16_rounding_tie()) return rc;
  TEST_PASS("test_greedy_argmax (normal / negative / exact-tie / single / "
            "bf16-rounding-tie -> lowest id)");
  return 0;
}
