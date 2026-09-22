// CUDALM — CUDA tests for causal decode attention (3-kernel pipeline).
//
// CPU reference follows the golden math contract (fp32 math, one fp16 RNE at
// the stage boundary — docs/weight_format.md) and the golden generator's
// attention_ref exactly (GQA kh = h * n_kv / n_heads, scale = 1/sqrt(head_dim),
// max-subtracted softmax).
//
// Checks:
//   * p = 0 (T = 1): softmax over one element is exactly 1.0, so out_h ==
//     V[kh(h)][0] — pinned BIT-EXACT via memcmp;
//   * p > 0 (T = 8 and T = 65): out vs CPU reference under the stage
//     tolerance (atol = rtol = 1e-2) — fp32 FMA/re-association may differ by
//     ~1 ulp between host and device.

#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/attention.h"
#include "cudalm/kv_cache.h"
#include "cudalm/model_config.h"
#include "cudalm/stage_compare.h"

#include "../../tests/common/check.h"

using namespace cudalm;
using namespace cudalm::kernels;

namespace {

// Deterministic xorshift32 in [-1, 1).
struct Rng {
  std::uint32_t s;
  explicit Rng(std::uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
  float n01() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s * (2.0f / 4294967296.0f) - 1.0f;
  }
};

__half make_half(Rng& rng, float scale) {
  return __float2half_rn(rng.n01() * scale);
}

// CPU reference of the golden attention_ref (fp32 throughout, one fp16 RNE
// per output element).
//   q      : [n_heads][head_dim] fp16
//   k_cache: [n_kv][max_seq][head_dim] fp16 (rows 0..position)
//   v_cache: [n_kv][max_seq][head_dim] fp16
//   out    : [n_heads][head_dim] fp16
void ref_attention(const __half* q, const __half* k_cache,
                   const __half* v_cache, int position, __half* out,
                   int n_heads, int n_kv, int head_dim, int max_seq) {
  const int T = position + 1;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (int h = 0; h < n_heads; ++h) {
    const int kh = h * n_kv / n_heads;
    std::vector<float> s(T);
    float m = -FLT_MAX;
    for (int t = 0; t < T; ++t) {
      float acc = 0.0f;
      const __half* qh = q + static_cast<std::size_t>(h) * head_dim;
      const __half* kr =
          k_cache + (static_cast<std::size_t>(kh) * max_seq + t) * head_dim;
      for (int d = 0; d < head_dim; ++d)
        acc += __half2float(qh[d]) * __half2float(kr[d]);
      s[t] = acc * scale;
      m = std::max(m, s[t]);
    }
    std::vector<float> e(T);
    float sum = 0.0f;
    for (int t = 0; t < T; ++t) {
      e[t] = std::exp(s[t] - m);
      sum += e[t];
    }
    const __half* vb =
        v_cache + static_cast<std::size_t>(kh) * max_seq * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      float acc = 0.0f;
      for (int t = 0; t < T; ++t)
        acc += (e[t] / sum) * __half2float(vb[static_cast<std::size_t>(t) * head_dim + d]);
      out[static_cast<std::size_t>(h) * head_dim + d] = __float2half_rn(acc);
    }
  }
}

struct Case {
  int position;
  const char* label;
};

// Runs one attention case (fresh cache + q), compares vs the reference.
int run_case(const ModelConfig& cfg, const Case& c, cudaStream_t stream) {
  const int n_heads = cfg.n_heads, n_kv = cfg.n_kv_heads, hd = cfg.head_dim;
  const int max_seq = cfg.max_seq_len;
  const int T = c.position + 1;

  Rng qrng(0xa00 + c.position);
  std::vector<__half> q(static_cast<std::size_t>(n_heads) * hd);
  for (auto& v : q) v = make_half(qrng, 2.0f);

  // Fill the K/V cache at positions 0..c.position with deterministic data;
  // k_ref/v_ref mirror the SAME values in the cache layout [n_kv][max_seq][hd]
  // for the CPU reference.
  KvCache cache(cfg, stream);
  std::vector<__half> k_row(static_cast<std::size_t>(n_kv) * hd);
  std::vector<__half> v_row(static_cast<std::size_t>(n_kv) * hd);
  const std::size_t ref_n = static_cast<std::size_t>(n_kv) * max_seq * hd;
  std::vector<__half> k_ref(ref_n, __float2half(0.0f));
  std::vector<__half> v_ref(ref_n, __float2half(0.0f));
  DeviceBuffer dk_in(k_row.size() * sizeof(__half), stream);
  DeviceBuffer dv_in(v_row.size() * sizeof(__half), stream);
  for (int t = 0; t < T; ++t) {
    Rng krng(0xb00 * (t + 1) + c.position);
    Rng vrng(0xc00 * (t + 1) + c.position);
    for (int n = 0; n < n_kv; ++n) {
      for (int d = 0; d < hd; ++d) {
        k_row[static_cast<std::size_t>(n) * hd + d] = make_half(krng, 2.0f);
        k_ref[(static_cast<std::size_t>(n) * max_seq + t) * hd + d] =
            k_row[static_cast<std::size_t>(n) * hd + d];
        v_row[static_cast<std::size_t>(n) * hd + d] = make_half(vrng, 2.0f);
        v_ref[(static_cast<std::size_t>(n) * max_seq + t) * hd + d] =
            v_row[static_cast<std::size_t>(n) * hd + d];
      }
    }
    dk_in.copy_from_host(k_row.data(), dk_in.bytes(), stream);
    dv_in.copy_from_host(v_row.data(), dv_in.bytes(), stream);
    cache.write(t, dk_in.data<__half>(), dv_in.data<__half>(), stream);
  }

  DeviceBuffer dq(q.size() * sizeof(__half), stream);
  DeviceBuffer dout(static_cast<std::size_t>(n_heads) * hd * sizeof(__half),
                    stream);
  DeviceBuffer dscratch(
      2 * static_cast<std::size_t>(n_heads) * max_seq * sizeof(float), stream);
  dq.copy_from_host(q.data(), dq.bytes(), stream);

  attention_decode_fp16(dq.data<__half>(), cache.k(), cache.v(), c.position,
                        dout.data<__half>(), n_heads, n_kv, hd, max_seq,
                        dscratch.data<float>(), stream);

  std::vector<__half> out(static_cast<std::size_t>(n_heads) * hd);
  dout.copy_to_host(out.data(), dout.bytes(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__half> ref(out.size());
  ref_attention(q.data(), k_ref.data(), v_ref.data(), c.position, ref.data(),
                n_heads, n_kv, hd, max_seq);

  if (c.position == 0) {
    // T = 1: out_h must equal V[kh(h)][0] bit-for-bit.
    bool ok = true;
    for (int h = 0; h < n_heads; ++h) {
      const int kh = h * n_kv / n_heads;
      const __half* vrow =
          v_ref.data() + static_cast<std::size_t>(kh) * max_seq * hd;
      ok = ok && std::memcmp(out.data() + static_cast<std::size_t>(h) * hd,
                             vrow, hd * sizeof(__half)) == 0;
    }
    CHECK(ok);
    if (!ok) {
      std::fprintf(stderr, "  attention p0 (%s): V-row pin failed\n", c.label);
      return 1;
    }
    return 0;
  }

  StageCompareResult r = compare_fp16_stages(
      reinterpret_cast<const std::uint16_t*>(ref.data()),
      reinterpret_cast<const std::uint16_t*>(out.data()), ref.size());
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "  attention (%s, T=%d): max_abs_err=%.3e idx=%zu\n",
                 c.label, T, r.max_abs_err, r.max_abs_idx);
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  CHECK(n >= 1);
  CUDA_CHECK(cudaSetDevice(0));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const ModelConfig cfg = ModelConfig::v01_default();
  const Case cases[] = {
      {0, "p0-single"},   // T=1, bit-exact V-row pin
      {7, "p7"},          // T=8
      {64, "p64"},        // T=65, multi-block softmax
      {511, "p511-max"},  // T=512, full max_seq
  };
  int rc = 0;
  for (const Case& c : cases) rc |= run_case(cfg, c, stream);

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc == 0) TEST_PASS("test_attention");
  return rc;
}
