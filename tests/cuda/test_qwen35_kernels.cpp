// CUDALM — CUDA unit tests for the Qwen3.5 bf16 kernel family (Phase B).
//
// Every reference below mirrors the pinned official bf16 rounding sequence
// EXACTLY (docs/qwen35_architecture.md §5-§7) with host libm, so where the
// op is pure per-element with identical inputs the comparison is
// BIT-EXACT (rope, split, kv write, p=0 attention); where a reduction or a
// transcendent differs in fp32 association (rmsnorm sum, softmax sum,
// expf/cosf between host libm and CUDA) the comparison is
// compare_bf16_stages (1e-2), which still catches O(1) structural bugs.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/qwen35_kernels.h"
#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_kv_cache.h"
#include "cudalm/stage_compare.h"

#include "../../tests/common/check.h"

using namespace cudalm;

namespace {

struct Rng {
  std::uint32_t s;
  explicit Rng(std::uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
  std::uint32_t u32() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  // bf16 in [-0.5, 0.5): real hidden-state scale (the golden uses
  // 0.05*randn; the wider spread here stresses the rounding boundaries).
  __nv_bfloat16 bf() {
    const float f =
        static_cast<float>(u32()) * (2.0f / 4294967296.0f) - 1.0f;
    return __float2bfloat16_rn(0.5f * f);
  }
  float f01() { return static_cast<float>(u32()) * (1.0f / 4294967296.0f); }
};

std::vector<__nv_bfloat16> rand_bf(Rng& rng, std::size_t n) {
  std::vector<__nv_bfloat16> v(n);
  for (auto& x : v) x = rng.bf();
  return v;
}

int cmp_bits(const void* a, const void* b, std::size_t bytes) {
  if (std::memcmp(a, b, bytes) != 0) {
    std::fprintf(stderr, "  bit-exact mismatch (%zu bytes)\n", bytes);
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Zero-centered RMSNorm
// ---------------------------------------------------------------------------
void ref_rmsnorm_zc(const std::vector<__nv_bfloat16>& x,
                    const std::vector<__nv_bfloat16>& w, int M, int H,
                    float eps, std::vector<__nv_bfloat16>* y) {
  y->resize(static_cast<std::size_t>(M) * H);
  for (int m = 0; m < M; ++m) {
    float ss = 0.f;
    for (int d = 0; d < H; ++d) {
      const float v = __bfloat162float(x[static_cast<std::size_t>(m) * H + d]);
      ss += v * v;
    }
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(H) + eps);
    for (int d = 0; d < H; ++d) {
      const float xf = __bfloat162float(x[static_cast<std::size_t>(m) * H + d]);
      const float wf = 1.0f + __bfloat162float(w[d]);
      (*y)[static_cast<std::size_t>(m) * H + d] =
          __float2bfloat16_rn(xf * inv * wf);
    }
  }
}

int test_rmsnorm_zc(cudaStream_t stream) {
  const int cases[][2] = {{1, 1024}, {8, 256}, {2, 256}};
  for (const auto& c : cases) {
    const int M = c[0], H = c[1];
    Rng rng(0x2a2a + M * 1000 + H);
    std::vector<__nv_bfloat16> x = rand_bf(rng, M * H);
    std::vector<__nv_bfloat16> w = rand_bf(rng, H);
    const float eps = 1e-6f;

    DeviceBuffer dx(x.size() * 2, stream);
    DeviceBuffer dw(w.size() * 2, stream);
    DeviceBuffer dy(M * H * 2, stream);
    dx.copy_from_host(x.data(), dx.bytes(), stream);
    dw.copy_from_host(w.data(), dw.bytes(), stream);
    kernels::qwen35_rmsnorm_zc_bf16(dx.data<__nv_bfloat16>(),
                                    dw.data<__nv_bfloat16>(),
                                    dy.data<__nv_bfloat16>(), M, H, eps,
                                    stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__nv_bfloat16> ref;
    ref_rmsnorm_zc(x, w, M, H, eps, &ref);
    std::vector<__nv_bfloat16> act(M * H);
    dy.copy_to_host(act.data(), act.size() * 2, stream);
    const StageCompareResult r = compare_bf16_stages(
        reinterpret_cast<const std::uint16_t*>(ref.data()),
        reinterpret_cast<const std::uint16_t*>(act.data()),
        act.size());
    if (!r.ok) {
      std::fprintf(stderr, "  rmsnorm M=%d H=%d: FAIL max_abs_err=%.9g "
                           "max_rel_err=%.9g\n", M, H, r.max_abs_err,
                   r.max_rel_err);
      return 1;
    }
    std::fprintf(stderr, "  rmsnorm M=%d H=%d max_abs_err=%.9g OK\n", M, H,
                 r.max_abs_err);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Fused [q;gate] split
// ---------------------------------------------------------------------------
int test_split(cudaStream_t stream) {
  const int n_heads = 8, hd = 256;
  Rng rng(0x5eed);
  std::vector<__nv_bfloat16> fused = rand_bf(rng, n_heads * 2 * hd);

  DeviceBuffer df(fused.size() * 2, stream);
  DeviceBuffer dq(n_heads * hd * 2, stream);
  DeviceBuffer dg(n_heads * hd * 2, stream);
  df.copy_from_host(fused.data(), df.bytes(), stream);
  kernels::qwen35_split_q_gate_bf16(df.data<__nv_bfloat16>(),
                                    dq.data<__nv_bfloat16>(),
                                    dg.data<__nv_bfloat16>(), n_heads, hd,
                                    stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> q(n_heads * hd), g(n_heads * hd);
  dq.copy_to_host(q.data(), q.size() * 2, stream);
  dg.copy_to_host(g.data(), g.size() * 2, stream);
  for (int h = 0; h < n_heads; ++h)
    for (int d = 0; d < hd; ++d) {
      if (q[static_cast<std::size_t>(h) * hd + d] !=
          fused[static_cast<std::size_t>(h) * 2 * hd + d])
        return 1;
      if (g[static_cast<std::size_t>(h) * hd + d] !=
          fused[static_cast<std::size_t>(h) * 2 * hd + hd + d])
        return 1;
    }
  std::fprintf(stderr, "  split q_gate: bit-exact OK\n");
  return 0;
}

// ---------------------------------------------------------------------------
// Partial rotate-half RoPE
// ---------------------------------------------------------------------------
int test_rope(cudaStream_t stream) {
  const int hd = 256, rd = 64;
  const float theta = 1e7f;
  Rng rng(0x700700u);
  const int Ms[2] = {8, 2};
  for (const int M : Ms) {
    std::vector<__nv_bfloat16> x = rand_bf(rng, M * hd);
    const int position = (M == 8) ? 5 : 1;

    // Host cos/sin table — the same values the runtime layer computes
    // (host libm; the kernel only rounds them to bf16).
    std::vector<float> cosf_t(rd / 2), sinf_t(rd / 2);
    for (int j = 0; j < rd / 2; ++j) {
      const float inv =
          1.0f / powf(theta, (2.0f * j) / static_cast<float>(rd));
      const float freq = static_cast<float>(position) * inv;
      cosf_t[j] = cosf(freq);
      sinf_t[j] = sinf(freq);
    }

    DeviceBuffer dx(x.size() * 2, stream);
    DeviceBuffer dy(M * hd * 2, stream);
    DeviceBuffer dc(rd * 4, stream);  // cos[rd/2] | sin[rd/2], fp32
    std::vector<float> table(rd);
    for (int j = 0; j < rd / 2; ++j) {
      table[j] = cosf_t[j];
      table[rd / 2 + j] = sinf_t[j];
    }
    dx.copy_from_host(x.data(), dx.bytes(), stream);
    dc.copy_from_host(table.data(), table.size() * 4, stream);
    kernels::qwen35_partial_rope_bf16(
        dx.data<__nv_bfloat16>(), dy.data<__nv_bfloat16>(), M, hd, rd,
        dc.data<float>(), dc.data<float>() + rd / 2, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Exact-rounding CPU reference (identical op order to the kernel).
    std::vector<__nv_bfloat16> ref(M * hd);
    for (int m = 0; m < M; ++m) {
      for (int d = 0; d < hd; ++d) {
        if (d >= rd) {
          ref[static_cast<std::size_t>(m) * hd + d] =
              x[static_cast<std::size_t>(m) * hd + d];
          continue;
        }
        const int j = d % (rd / 2);  // replicated-freq partial RoPE
        const __nv_bfloat16 cb = __float2bfloat16_rn(cosf_t[j]);
        const __nv_bfloat16 sb = __float2bfloat16_rn(sinf_t[j]);
        const bool first = (d < rd / 2);
        const int pd = first ? (d + rd / 2) : (d - rd / 2);
        const float pv =
            __bfloat162float(x[static_cast<std::size_t>(m) * hd + pd]);
        const __nv_bfloat16 r1 = __float2bfloat16_rn(
            __bfloat162float(x[static_cast<std::size_t>(m) * hd + d]) *
            __bfloat162float(cb));
        const __nv_bfloat16 r2 =
            __float2bfloat16_rn((first ? -pv : pv) * __bfloat162float(sb));
        ref[static_cast<std::size_t>(m) * hd + d] =
            __float2bfloat16_rn(__bfloat162float(r1) + __bfloat162float(r2));
      }
    }
    std::vector<__nv_bfloat16> act(M * hd);
    dy.copy_to_host(act.data(), act.size() * 2, stream);
    // Same cos/sin, same per-element op order -> must be bit-exact.
    if (int rc = cmp_bits(act.data(), ref.data(), act.size() * 2)) return rc;
    std::fprintf(stderr, "  rope M=%d p=%d: bit-exact OK\n", M, position);
  }

  // p = 0: cos=1 / sin=0 are exact in bf16 -> identity (bit-exact).
  {
    const int M = 8;
    Rng rng(0x1000);
    std::vector<__nv_bfloat16> x = rand_bf(rng, M * hd);
    DeviceBuffer dx(x.size() * 2, stream);
    DeviceBuffer dy(M * hd * 2, stream);
    DeviceBuffer dc(rd * 4, stream);
    dx.copy_from_host(x.data(), dx.bytes(), stream);
    std::vector<float> table(rd);
    for (int j = 0; j < rd / 2; ++j) { table[j] = 1.0f; table[rd / 2 + j] = 0.0f; }
    dc.copy_from_host(table.data(), table.size() * 4, stream);
    kernels::qwen35_partial_rope_bf16(
        dx.data<__nv_bfloat16>(), dy.data<__nv_bfloat16>(), M, hd, rd,
        dc.data<float>(), dc.data<float>() + rd / 2, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<__nv_bfloat16> act(M * hd);
    dy.copy_to_host(act.data(), act.size() * 2, stream);
    if (int rc = cmp_bits(act.data(), x.data(), act.size() * 2)) return rc;
    std::fprintf(stderr, "  rope p=0 identity: bit-exact OK\n");
  }
  return 0;
}

// ---------------------------------------------------------------------------
// KV write (bit-exact copies)
// ---------------------------------------------------------------------------
int test_kv_write(cudaStream_t stream) {
  const int n_kv = 2, max_seq = 8, hd = 256;
  Rng rng(0x7476);
  std::vector<__nv_bfloat16> k = rand_bf(rng, n_kv * hd);
  std::vector<__nv_bfloat16> v = rand_bf(rng, n_kv * hd);
  DeviceBuffer dk(k.size() * 2, stream);
  DeviceBuffer dv(v.size() * 2, stream);
  DeviceBuffer dkc(static_cast<std::size_t>(n_kv) * max_seq * hd * 2, stream);
  DeviceBuffer dvc(static_cast<std::size_t>(n_kv) * max_seq * hd * 2, stream);
  dk.copy_from_host(k.data(), dk.bytes(), stream);
  dv.copy_from_host(v.data(), dv.bytes(), stream);

  const int pos = 3;
  kernels::qwen35_kv_write_bf16(dkc.data<__nv_bfloat16>(),
                                dvc.data<__nv_bfloat16>(),
                                dk.data<__nv_bfloat16>(),
                                dv.data<__nv_bfloat16>(), pos, n_kv, max_seq,
                                hd, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> kcache(n_kv * max_seq * hd), vcache(n_kv * max_seq * hd);
  dkc.copy_to_host(kcache.data(), kcache.size() * 2, stream);
  dvc.copy_to_host(vcache.data(), vcache.size() * 2, stream);
  for (int n = 0; n < n_kv; ++n)
    for (int t = 0; t < max_seq; ++t)
      for (int d = 0; d < hd; ++d) {
        const std::size_t idx =
            (static_cast<std::size_t>(n) * max_seq + t) * hd + d;
        const bool at_pos = (t == pos);
        if (at_pos) {
          if (kcache[idx] != k[static_cast<std::size_t>(n) * hd + d])
            return 1;
          if (vcache[idx] != v[static_cast<std::size_t>(n) * hd + d])
            return 1;
        } else {
          if (kcache[idx] != __nv_bfloat16(0.0f)) return 1;  // untouched
        }
      }
  std::fprintf(stderr, "  kv write: bit-exact + untouched rows OK\n");
  return 0;
}

// ---------------------------------------------------------------------------
// Decode attention
// ---------------------------------------------------------------------------
void ref_attention(const std::vector<__nv_bfloat16>& q,
                   const std::vector<__nv_bfloat16>& k_cache,
                   const std::vector<__nv_bfloat16>& v_cache, int position,
                   int n_heads, int n_kv, int hd, int max_seq,
                   std::vector<__nv_bfloat16>* out) {
  const int T = position + 1;
  const float scale = 1.0f / sqrtf(static_cast<float>(hd));
  out->resize(static_cast<std::size_t>(n_heads) * hd);
  for (int h = 0; h < n_heads; ++h) {
    const int kh = h * n_kv / n_heads;
    // 1) scores (exact bf16 rounding chain)
    std::vector<__nv_bfloat16> s2(T);
    for (int t = 0; t < T; ++t) {
      float acc = 0.f;
      for (int d = 0; d < hd; ++d) {
        acc += __bfloat162float(q[static_cast<std::size_t>(h) * hd + d]) *
               __bfloat162float(
                   k_cache[(static_cast<std::size_t>(kh) * max_seq + t) * hd + d]);
      }
      const __nv_bfloat16 s1 = __float2bfloat16_rn(acc);
      s2[t] = __float2bfloat16_rn(__bfloat162float(s1) * scale);
    }
    // 2) softmax in fp32 (max-subtracted), one bf16 RNE
    float m = -FLT_MAX;
    for (int t = 0; t < T; ++t) m = fmaxf(m, __bfloat162float(s2[t]));
    std::vector<float> e(T);
    float sum = 0.f;
    for (int t = 0; t < T; ++t) {
      e[t] = expf(__bfloat162float(s2[t]) - m);
      sum += e[t];
    }
    std::vector<__nv_bfloat16> p(T);
    for (int t = 0; t < T; ++t) p[t] = __float2bfloat16_rn(e[t] / sum);
    // 3) PV
    for (int d = 0; d < hd; ++d) {
      float acc = 0.f;
      for (int t = 0; t < T; ++t) {
        acc += __bfloat162float(p[t]) *
               __bfloat162float(
                   v_cache[(static_cast<std::size_t>(kh) * max_seq + t) * hd + d]);
      }
      (*out)[static_cast<std::size_t>(h) * hd + d] = __float2bfloat16_rn(acc);
    }
  }
}

int test_attention(cudaStream_t stream) {
  const int n_heads = 8, n_kv = 2, hd = 256, max_seq = 6;
  const int positions[2] = {0, 4};
  for (const int position : positions) {
    Rng rng(0xa770 + position);
    std::vector<__nv_bfloat16> q = rand_bf(rng, n_heads * hd);
    std::vector<__nv_bfloat16> kc = rand_bf(rng, n_kv * max_seq * hd);
    std::vector<__nv_bfloat16> vc = rand_bf(rng, n_kv * max_seq * hd);

    DeviceBuffer dq(q.size() * 2, stream);
    DeviceBuffer dk(kc.size() * 2, stream);
    DeviceBuffer dv(vc.size() * 2, stream);
    DeviceBuffer do_(n_heads * hd * 2, stream);
    DeviceBuffer ds(2 * static_cast<std::size_t>(n_heads) * max_seq * 2, stream);
    dq.copy_from_host(q.data(), dq.bytes(), stream);
    dk.copy_from_host(kc.data(), dk.bytes(), stream);
    dv.copy_from_host(vc.data(), dv.bytes(), stream);
    kernels::qwen35_attention_decode_bf16(
        dq.data<__nv_bfloat16>(), dk.data<__nv_bfloat16>(),
        dv.data<__nv_bfloat16>(), position, do_.data<__nv_bfloat16>(),
        n_heads, n_kv, hd, max_seq, ds.data<__nv_bfloat16>(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__nv_bfloat16> ref;
    ref_attention(q, kc, vc, position, n_heads, n_kv, hd, max_seq, &ref);
    std::vector<__nv_bfloat16> act(n_heads * hd);
    do_.copy_to_host(act.data(), act.size() * 2, stream);

    if (position == 0) {
      // Single key: probs = [1] exact -> out == V row (bit-exact).
      if (cmp_bits(act.data(), ref.data(), act.size() * 2)) {
        std::fprintf(stderr, "  attention p=0: p0 invariant failed\n");
        return 1;
      }
      for (int h = 0; h < n_heads; ++h) {
        const int kh = h * n_kv / n_heads;
        const std::size_t vrow = static_cast<std::size_t>(kh) * max_seq * hd;
        if (cmp_bits(reinterpret_cast<const std::uint8_t*>(act.data()) +
                         h * hd * 2,
                     reinterpret_cast<const std::uint8_t*>(vc.data()) + vrow * 2,
                     hd * 2))
          return 1;
      }
      std::fprintf(stderr, "  attention p=0: bit-exact (probs=[1]) OK\n");
    } else {
      const StageCompareResult r = compare_bf16_stages(
          reinterpret_cast<const std::uint16_t*>(ref.data()),
          reinterpret_cast<const std::uint16_t*>(act.data()), act.size());
      if (!r.ok) {
        std::fprintf(stderr, "  attention p=%d: FAIL max_abs_err=%.9g\n",
                     position, r.max_abs_err);
        return 1;
      }
      std::fprintf(stderr, "  attention p=%d: max_abs_err=%.9g OK\n", position,
                   r.max_abs_err);
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Elementwise: add / silu_mul / gate_mul
// ---------------------------------------------------------------------------
int test_elementwise(cudaStream_t stream) {
  const std::size_t n = 3584;  // the MLP width (n % 8 == 0)
  Rng rng(0xe100);
  std::vector<__nv_bfloat16> a = rand_bf(rng, n);
  std::vector<__nv_bfloat16> b = rand_bf(rng, n);

  DeviceBuffer da(a.size() * 2, stream);
  DeviceBuffer db(b.size() * 2, stream);
  DeviceBuffer dy(n * 2, stream);
  da.copy_from_host(a.data(), da.bytes(), stream);
  db.copy_from_host(b.data(), db.bytes(), stream);

  // add
  kernels::qwen35_add_bf16(da.data<__nv_bfloat16>(), db.data<__nv_bfloat16>(),
                           dy.data<__nv_bfloat16>(), n, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> act(n);
  dy.copy_to_host(act.data(), act.size() * 2, stream);
  for (std::size_t i = 0; i < n; ++i) {
    const __nv_bfloat16 ref =
        __float2bfloat16_rn(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    if (act[i] != ref) {
      std::fprintf(stderr, "  add: mismatch at %zu\n", i);
      return 1;
    }
  }
  std::fprintf(stderr, "  add: bit-exact OK\n");

  // silu_mul: s = bf16(g/(1+exp(-g))); y = bf16(f32(s)*f32(up))
  std::vector<__nv_bfloat16> act2(n);
  kernels::qwen35_silu_mul_bf16(da.data<__nv_bfloat16>(),
                                db.data<__nv_bfloat16>(),
                                dy.data<__nv_bfloat16>(), n, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  dy.copy_to_host(act2.data(), act2.size() * 2, stream);
  {
    StageCompareResult best;
    best.n = n;
    for (std::size_t i = 0; i < n; ++i) {
      const float g = __bfloat162float(a[i]);
      const __nv_bfloat16 s =
          __float2bfloat16_rn(g / (1.0f + expf(-g)));
      const __nv_bfloat16 ref = __float2bfloat16_rn(
          __bfloat162float(s) * __bfloat162float(b[i]));
      const std::uint16_t rb =
          *reinterpret_cast<const std::uint16_t*>(&ref);
      const std::uint16_t ab =
          *reinterpret_cast<const std::uint16_t*>(&act2[i]);
      const StageCompareResult r = compare_bf16_stages(&rb, &ab, 1);
      if (r.max_abs_err > best.max_abs_err) best.max_abs_err = r.max_abs_err;
      if (!r.ok) {
        std::fprintf(stderr, "  silu_mul: FAIL at %zu\n", i);
        return 1;
      }
    }
    std::fprintf(stderr, "  silu_mul: max_abs_err=%.9g OK\n",
                 best.max_abs_err);
  }

  // gate_mul: sg = bf16(1/(1+exp(-g))); y = bf16(f32(attn)*f32(sg))
  std::vector<__nv_bfloat16> act3(n);
  kernels::qwen35_gate_mul_bf16(da.data<__nv_bfloat16>(),
                                db.data<__nv_bfloat16>(),
                                dy.data<__nv_bfloat16>(), n, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  dy.copy_to_host(act3.data(), act3.size() * 2, stream);
  {
    StageCompareResult best;
    best.n = n;
    for (std::size_t i = 0; i < n; ++i) {
      const float g = __bfloat162float(b[i]);
      const __nv_bfloat16 sg =
          __float2bfloat16_rn(1.0f / (1.0f + expf(-g)));
      const __nv_bfloat16 ref = __float2bfloat16_rn(
          __bfloat162float(a[i]) * __bfloat162float(sg));
      const std::uint16_t rb =
          *reinterpret_cast<const std::uint16_t*>(&ref);
      const std::uint16_t ab =
          *reinterpret_cast<const std::uint16_t*>(&act3[i]);
      const StageCompareResult r = compare_bf16_stages(&rb, &ab, 1);
      if (r.max_abs_err > best.max_abs_err) best.max_abs_err = r.max_abs_err;
      if (!r.ok) {
        std::fprintf(stderr, "  gate_mul: FAIL at %zu\n", i);
        return 1;
      }
    }
    std::fprintf(stderr, "  gate_mul: max_abs_err=%.9g OK\n",
                 best.max_abs_err);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Qwen35KvCache RAII class
// ---------------------------------------------------------------------------
int test_kv_cache_class(cudaStream_t stream) {
  const Qwen35Config cfg = Qwen35Config::qwen35_08b();
  // The pinned config's max_seq_len (262144) would allocate ~536MB; the
  // class math is what is under test — use a local copy with a small
  // max_seq_len (the class only reads n_kv_heads/max_seq_len/head_dim).
  Qwen35Config c2 = cfg;
  c2.max_seq_len = 64;
  Qwen35KvCache cache(c2, stream);
  CHECK_EQ(cache.n_kv_heads(), 2);
  CHECK_EQ(cache.max_seq_len(), 64);
  CHECK_EQ(cache.head_dim(), 256);
  CHECK(cache.numel() == 2ull * 64 * 256);

  Rng rng(0x6342);
  std::vector<__nv_bfloat16> k = rand_bf(rng, 2 * 256);
  std::vector<__nv_bfloat16> v = rand_bf(rng, 2 * 256);
  DeviceBuffer dk(k.size() * 2, stream);
  DeviceBuffer dv(v.size() * 2, stream);
  dk.copy_from_host(k.data(), dk.bytes(), stream);
  dv.copy_from_host(v.data(), dv.bytes(), stream);

  cache.write(7, dk.data<__nv_bfloat16>(), dv.data<__nv_bfloat16>(), stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> kc(cache.numel()), vc(cache.numel());
  DeviceBuffer hk(kc.size() * 2, stream);
  DeviceBuffer hv(vc.size() * 2, stream);
  CUDA_CHECK(cudaMemcpyAsync(hk.data(), cache.k(), hk.bytes(),
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(hv.data(), cache.v(), hv.bytes(),
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  hk.copy_to_host(kc.data(), hk.bytes(), stream);
  hv.copy_to_host(vc.data(), hv.bytes(), stream);
  for (int n = 0; n < 2; ++n)
    for (int t = 0; t < 64; ++t) {
      const std::size_t idx = cache.row_offset(n, t);
      if (t == 7) {
        if (cmp_bits(reinterpret_cast<const std::uint8_t*>(kc.data()) + idx * 2,
                     reinterpret_cast<const std::uint8_t*>(k.data()) + n * 256 * 2,
                     256 * 2))
          return 1;
      } else if (kc[idx] != __nv_bfloat16(0.0f)) {
        return 1;
      }
    }
  std::fprintf(stderr, "  Qwen35KvCache: write/zero-init OK\n");
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

  if (int rc = test_rmsnorm_zc(stream)) return rc;
  if (int rc = test_split(stream)) return rc;
  if (int rc = test_rope(stream)) return rc;
  if (int rc = test_kv_write(stream)) return rc;
  if (int rc = test_attention(stream)) return rc;
  if (int rc = test_elementwise(stream)) return rc;
  if (int rc = test_kv_cache_class(stream)) return rc;

  CUDA_CHECK(cudaStreamDestroy(stream));
  TEST_PASS("test_qwen35_kernels");
  return 0;
}
