// CUDALM — v0.5 Phase B: paged-KV kernel standalone hard gate (CUDA).
//
// Proves the NEW paged pipeline (page indirection through a DEVICE block
// table) against the FROZEN contiguous pipeline (src/kernels/
// qwen35_kernels.cu) with SYNTHETIC data:
//
//   * paged KV write: the written physical row is BIT-EXACT (the source row)
//     for positions 0, page_tokens-1, page_tokens, page_tokens+1 and a
//     multi-token block; every unwritten row stays EXACTLY untouched;
//   * paged causal decode attention output is BIT-IDENTICAL to the frozen
//     contiguous attention for the same logical K/V rows and q, for a
//     boundary matrix of positions (0, pt-1, pt, pt+1, crossing several
//     pages) under TWO non-trivial physical mappings (NOT the identity
//     mapping): logical blocks 0..5 -> pages {3,0,5,1,4,2} and
//     {5,2,0,3,1,4};
//   * STALE BLOCK-TABLE PROOF: block-table entries beyond the current
//     (position/page_tokens)+1 are filled with the OUT-OF-RANGE sentinel
//     9999 — if the paged kernels ever dereference a stale entry, the
//     result is wrong AND compute-sanitizer flags the OOB read; the
//     bit-exact match against the contiguous reference proves the stale
//     region is never read.
//
// No checkpoint needed (pure synthetic data). No gather: the paged kernels
// address every K/V row through block_table[t / page_tokens] in-kernel.

#include "../../tests/common/check.h"
#include "cudalm/device_buffer.h"
#include "cudalm/kernels/paged_kv.h"
#include "cudalm/kernels/qwen35_kernels.h"

#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using namespace cudalm;

namespace {

// Fixed synthetic problem shape (GQA: 4 query heads over 2 kv heads).
const int kNumPages = 6;
const int kNkv = 2;
const int kNheads = 4;
const int kPt = 4;      // page_tokens
const int kHd = 8;      // head_dim
const int kMaxSeq = 24;  // logical max (6 blocks * 4)

// Two NON-trivial physical mappings (logical block -> physical page).
const int kMapA[6] = {3, 0, 5, 1, 4, 2};
const int kMapB[6] = {5, 2, 0, 3, 1, 4};

// Deterministic synthetic row value (diverse, signed, |v| <= ~6).
inline __nv_bfloat16 syn(int t, int n, int d) {
  const float v =
      (static_cast<float>((t * 131 + n * 17 + d * 7) % 97) - 48.0f) / 8.0f;
  return __float2bfloat16_rn(v);
}

// Fill a paged K/V page array with logical rows [0..T) under `map`,
// leaving every other row exactly zero.
void seed_paged(DeviceBuffer& k, DeviceBuffer& v, const int* map, int T,
                cudaStream_t s) {
  const std::size_t per_page =
      static_cast<std::size_t>(kNkv) * kPt * kHd;
  const std::size_t total = per_page * kNumPages;
  std::vector<__nv_bfloat16> kh(total, __float2bfloat16_rn(0.0f));
  std::vector<__nv_bfloat16> vh(total, __float2bfloat16_rn(0.0f));
  for (int t = 0; t < T; ++t) {
    const int page = map[t / kPt];
    const int off = t % kPt;
    for (int n = 0; n < kNkv; ++n)
      for (int d = 0; d < kHd; ++d) {
        const std::size_t idx =
            (static_cast<std::size_t>(page) * kNkv + n) * kPt * kHd +
            static_cast<std::size_t>(off) * kHd + d;
        kh[idx] = syn(t, n, d);
        vh[idx] = syn(t + 1000, n, d);  // K and V differ
      }
  }
  CUDA_CHECK(cudaMemcpyAsync(k.data(), kh.data(), total * 2,
                             cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(v.data(), vh.data(), total * 2,
                             cudaMemcpyHostToDevice, s));
}

// Build the FROZEN contiguous cache [n_kv][max_seq][hd] with the SAME
// logical rows [0..T).
void seed_contiguous(DeviceBuffer& kc, DeviceBuffer& vc, int T,
                     cudaStream_t s) {
  const std::size_t total = static_cast<std::size_t>(kNkv) * kMaxSeq * kHd;
  std::vector<__nv_bfloat16> kh(total, __float2bfloat16_rn(0.0f));
  std::vector<__nv_bfloat16> vh(total, __float2bfloat16_rn(0.0f));
  for (int t = 0; t < T; ++t)
    for (int n = 0; n < kNkv; ++n)
      for (int d = 0; d < kHd; ++d) {
        const std::size_t idx =
            (static_cast<std::size_t>(n) * kMaxSeq + t) * kHd + d;
        kh[idx] = syn(t, n, d);
        vh[idx] = syn(t + 1000, n, d);
      }
  CUDA_CHECK(cudaMemcpyAsync(kc.data(), kh.data(), total * 2,
                             cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(vc.data(), vh.data(), total * 2,
                             cudaMemcpyHostToDevice, s));
}

std::vector<__nv_bfloat16> d2h(const __nv_bfloat16* dev, std::size_t n,
                               cudaStream_t s) {
  std::vector<__nv_bfloat16> h(n);
  if (n)
    CUDA_CHECK(cudaMemcpyAsync(h.data(), dev, n * 2, cudaMemcpyDeviceToHost,
                               s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return h;
}

bool bits_equal(const std::vector<__nv_bfloat16>& a,
                const std::vector<__nv_bfloat16>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
}

// 1) Paged KV write: exact physical rows, untouched rows stay exact.
int test_paged_write() {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    DeviceBuffer k, v, kt, vt, bt;
    const std::size_t per_page = static_cast<std::size_t>(kNkv) * kPt * kHd;
    k.allocate(per_page * kNumPages * 2, s);
    v.allocate(per_page * kNumPages * 2, s);
    k.clear(per_page * kNumPages * 2, s);
    v.clear(per_page * kNumPages * 2, s);
    kt.allocate(kNkv * kHd * 2, s);
    vt.allocate(kNkv * kHd * 2, s);
    bt.allocate(sizeof(int) * 8, s);  // 8 slots: 6 valid + 2 STALE sentinels
    std::vector<int> map(kMapA, kMapA + 6);
    int stale[2] = {9999, 9999};
    CUDA_CHECK(cudaMemcpyAsync(bt.data(), map.data(), sizeof(int) * 6,
                               cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<int*>(bt.data()) + 6, stale,
                               sizeof(int) * 2, cudaMemcpyHostToDevice, s));

    // Write positions 0, pt-1, pt, pt+1 (0, 3, 4, 5) + one full block
    // (positions 0..7 already cover that; we write 0..5 explicitly).
    for (int p : {0, kPt - 1, kPt, kPt + 1}) {
      std::vector<__nv_bfloat16> ks(kNkv * kHd), vs(kNkv * kHd);
      for (int n = 0; n < kNkv; ++n)
        for (int d = 0; d < kHd; ++d) {
          ks[static_cast<std::size_t>(n) * kHd + d] = syn(p, n, d);
          vs[static_cast<std::size_t>(n) * kHd + d] = syn(p + 1000, n, d);
        }
      CUDA_CHECK(cudaMemcpyAsync(kt.data(), ks.data(), ks.size() * 2,
                                 cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(vt.data(), vs.data(), vs.size() * 2,
                                 cudaMemcpyHostToDevice, s));
      kernels::qwen35_paged_kv_write_bf16(
          kt.data<__nv_bfloat16>(), vt.data<__nv_bfloat16>(),
          k.data<__nv_bfloat16>(), v.data<__nv_bfloat16>(),
          static_cast<int*>(bt.data()), p, kPt, kNkv, kHd,
          static_cast<std::size_t>(kNkv) * kPt * kHd, s);  // adjacent pages
    }
    CUDA_CHECK(cudaStreamSynchronize(s));

    // Every written token's PHYSICAL row is bit-exact; every unwritten row
    // (all pages, incl. the stale-sentinel region's pages is irrelevant —
    // rows 6..15 logical are unwritten) stays exactly zero.
    const std::vector<__nv_bfloat16> kh = d2h(k.data<__nv_bfloat16>(),
                                              per_page * kNumPages, s);
    const std::vector<__nv_bfloat16> vh = d2h(v.data<__nv_bfloat16>(),
                                              per_page * kNumPages, s);
    // The written positions are exactly {0, pt-1, pt, pt+1}.
    auto is_written = [](int t) {
      return t == 0 || t == kPt - 1 || t == kPt || t == kPt + 1;
    };
    for (int t = 0; t < kMaxSeq; ++t) {
      const int page = map[t / kPt];
      const int off = t % kPt;
      const bool written = is_written(t);
      for (int n = 0; n < kNkv; ++n)
        for (int d = 0; d < kHd; ++d) {
          const std::size_t idx =
              (static_cast<std::size_t>(page) * kNkv + n) * kPt * kHd +
              static_cast<std::size_t>(off) * kHd + d;
          if (written) {
            CHECK(bits_equal(std::vector<__nv_bfloat16>{kh[idx]},
                             std::vector<__nv_bfloat16>{syn(t, n, d)}));
            CHECK(bits_equal(std::vector<__nv_bfloat16>{vh[idx]},
                             std::vector<__nv_bfloat16>{syn(t + 1000, n, d)}));
          } else {
            CHECK(bits_equal(std::vector<__nv_bfloat16>{kh[idx]},
                             std::vector<__nv_bfloat16>{__float2bfloat16_rn(0.0f)}));
            CHECK(bits_equal(std::vector<__nv_bfloat16>{vh[idx]},
                             std::vector<__nv_bfloat16>{__float2bfloat16_rn(0.0f)}));
          }
        }
    }
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  std::printf("  [ok] paged KV write: physical rows EXACT, unwritten rows "
              "untouched (stale sentinels present)\n");
  return 0;
}

// 2+3) Paged attention == frozen contiguous attention (bit-identical),
//      boundary positions, two non-trivial mappings, stale sentinels in the
//      block table beyond the read region.
int test_paged_attention_parity() {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    // Boundary matrix of logical lengths T (= position+1): covers position
    // 0, pt-1, pt, pt+1 and multi-page crossings.
    const int T_list[] = {1, 2, 4, 5, 6, 8, 9, 13};
    const int* maps[] = {kMapA, kMapB};
    DeviceBuffer k, v, kc, vc, q, out_p, out_c, scratch_p, scratch_c, bt;
    const std::size_t per_page = static_cast<std::size_t>(kNkv) * kPt * kHd;
    k.allocate(per_page * kNumPages * 2, s);
    v.allocate(per_page * kNumPages * 2, s);
    kc.allocate(static_cast<std::size_t>(kNkv) * kMaxSeq * kHd * 2, s);
    vc.allocate(static_cast<std::size_t>(kNkv) * kMaxSeq * kHd * 2, s);
    q.allocate(static_cast<std::size_t>(kNheads) * kHd * 2, s);
    out_p.allocate(static_cast<std::size_t>(kNheads) * kHd * 2, s);
    out_c.allocate(static_cast<std::size_t>(kNheads) * kHd * 2, s);
    scratch_p.allocate(static_cast<std::size_t>(2) * kNheads * kMaxSeq * 2, s);
    scratch_c.allocate(static_cast<std::size_t>(2) * kNheads * kMaxSeq * 2, s);
    bt.allocate(sizeof(int) * 8, s);  // valid blocks + stale sentinels

    // Deterministic q (diverse, signed).
    std::vector<__nv_bfloat16> qh(static_cast<std::size_t>(kNheads) * kHd);
    for (int h = 0; h < kNheads; ++h)
      for (int d = 0; d < kHd; ++d)
        qh[static_cast<std::size_t>(h) * kHd + d] =
            __float2bfloat16_rn((static_cast<float>((h * 53 + d * 11) % 89) -
                                 44.0f) / 7.0f);
    CUDA_CHECK(cudaMemcpyAsync(q.data(), qh.data(), qh.size() * 2,
                               cudaMemcpyHostToDevice, s));

    int checked = 0;
    for (const int* map : maps) {
      for (int T : T_list) {
        seed_paged(k, v, map, T, s);
        seed_contiguous(kc, vc, T, s);
        // Block table: T's blocks valid, the rest OUT-OF-RANGE sentinels.
        std::vector<int> bth(T > 0 ? (T + kPt - 1) / kPt : 0);
        for (int b = 0; b < static_cast<int>(bth.size()); ++b)
          bth[static_cast<std::size_t>(b)] = map[b];
        while (bth.size() < 8) bth.push_back(9999);  // STALE sentinels
        CUDA_CHECK(cudaMemcpyAsync(bt.data(), bth.data(), bth.size() * 4,
                                   cudaMemcpyHostToDevice, s));

        const int position = T - 1;
        kernels::qwen35_attention_decode_bf16(
            q.data<__nv_bfloat16>(), kc.data<__nv_bfloat16>(),
            vc.data<__nv_bfloat16>(), position,
            out_c.data<__nv_bfloat16>(), kNheads, kNkv, kHd, kMaxSeq,
            scratch_c.data<__nv_bfloat16>(), s);
        kernels::qwen35_paged_attention_decode_bf16(
            q.data<__nv_bfloat16>(), k.data<__nv_bfloat16>(),
            v.data<__nv_bfloat16>(), static_cast<int*>(bt.data()), position,
            kPt, out_p.data<__nv_bfloat16>(), kNheads, kNkv, kHd,
            static_cast<std::size_t>(kNkv) * kPt * kHd,
            scratch_p.data<__nv_bfloat16>(), s);
        CUDA_CHECK(cudaStreamSynchronize(s));

        const std::vector<__nv_bfloat16> oc =
            d2h(out_c.data<__nv_bfloat16>(),
                static_cast<std::size_t>(kNheads) * kHd, s);
        const std::vector<__nv_bfloat16> op =
            d2h(out_p.data<__nv_bfloat16>(),
                static_cast<std::size_t>(kNheads) * kHd, s);
        CHECK(bits_equal(op, oc));  // BIT-IDENTICAL
        ++checked;
      }
    }
    std::printf("  [ok] paged attention == frozen contiguous: %d "
                "position/mapping cases BIT-IDENTICAL (stale sentinels "
                "never read)\n",
                checked);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}

// ---------------------------------------------------------------------------
// Real-model-shape scenario: the exact Qwen3.5-0.8B full-attention dims
// (n_heads 16, n_kv 8, head_dim 64) with the small page_tokens = 2 used by
// the Phase B parity gate, TWO interleaved layer ordinals in ONE page array
// (the Phase-A pool layout: row-major [n_full][num_pages][n_kv][pt][hd],
// i.e. the pages of one ordinal are ADJACENT with stride
// n_kv*pt*hd, and ordinals are num_pages*n_kv*pt*hd apart), a
// non-trivial mapping
// {2,0,1}, and the boundary positions 1,2,3,4,5 (every position crosses a
// page boundary at pt=2 from position 2 on). Same gates: paged KV write
// physical rows EXACT; paged attention BIT-IDENTICAL to the frozen
// contiguous kernel; stale sentinels in the block table.
// ---------------------------------------------------------------------------
const int kRealNkv = 8;
const int kRealNheads = 16;
const int kRealHd = 64;  // 0.8B: hidden 1024 / 16 heads
const int kRealPt = 2;
const int kRealNumPages = 3;
const int kRealNfull = 2;
const int kRealMaxSeq = 12;  // 6 blocks at pt=2

int test_real_shape_parity() {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  {
    const int mapA[3] = {2, 0, 1};
    const int kMaxT = 6;
    const std::size_t page_elems =
        static_cast<std::size_t>(kRealNkv) * kRealPt * kRealHd;
    // True pool layout: pages of one ordinal are adjacent (stride
    // page_elems); ordinals are kRealNumPages * page_elems apart.
    const std::size_t page_stride = page_elems;
    const std::size_t ord_stride =
        static_cast<std::size_t>(kRealNumPages) * page_elems;
    const std::size_t per_ord = ord_stride * kRealNfull;
    DeviceBuffer k, v, kc, vc, q, out_p, out_c, scratch_p, scratch_c, bt;
    k.allocate(per_ord * 2, s);
    v.allocate(per_ord * 2, s);
    kc.allocate(static_cast<std::size_t>(kRealNkv) * kRealMaxSeq *
                    kRealHd * 2,
                s);
    vc.allocate(static_cast<std::size_t>(kRealNkv) * kRealMaxSeq *
                    kRealHd * 2,
                s);
    q.allocate(static_cast<std::size_t>(kRealNheads) * kRealHd * 2, s);
    out_p.allocate(static_cast<std::size_t>(kRealNheads) * kRealHd * 2, s);
    out_c.allocate(static_cast<std::size_t>(kRealNheads) * kRealHd * 2, s);
    scratch_p.allocate(
        static_cast<std::size_t>(2) * kRealNheads * kRealMaxSeq * 2, s);
    scratch_c.allocate(
        static_cast<std::size_t>(2) * kRealNheads * kRealMaxSeq * 2, s);
    bt.allocate(sizeof(int) * 8, s);

    // Host scratch reused for every seed (avoids repeated large host
    // allocations).
    std::vector<__nv_bfloat16> kh(per_ord, __float2bfloat16_rn(0.0f));
    std::vector<__nv_bfloat16> vh(per_ord, __float2bfloat16_rn(0.0f));
    // Seed one LAYER ORDINAL (ordinal 0 — base k.data()) with logical rows
    // [0..T) under the pool layout; every other row stays exactly zero.
    auto seed = [&](const int* map, int T, std::size_t ord_base) {
      for (int t = 0; t < T; ++t) {
        const int page = map[t / kRealPt];
        const int off = t % kRealPt;
        for (int n = 0; n < kRealNkv; ++n)
          for (int d = 0; d < kRealHd; ++d) {
            const std::size_t idx = ord_base +
                                    static_cast<std::size_t>(page) *
                                        page_stride +
                                    (static_cast<std::size_t>(n) * kRealPt +
                                     off) *
                                        kRealHd +
                                    d;
            kh[idx] = syn(t, n, d);
            vh[idx] = syn(t + 1000, n, d);
          }
      }
      CUDA_CHECK(cudaMemcpyAsync(k.data(), kh.data(), kh.size() * 2,
                                 cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(v.data(), vh.data(), vh.size() * 2,
                                 cudaMemcpyHostToDevice, s));
    };

    // Deterministic q for the 16 query heads.
    std::vector<__nv_bfloat16> qh(
        static_cast<std::size_t>(kRealNheads) * kRealHd);
    for (int h = 0; h < kRealNheads; ++h)
      for (int d = 0; d < kRealHd; ++d)
        qh[static_cast<std::size_t>(h) * kRealHd + d] =
            __float2bfloat16_rn(
                (static_cast<float>((h * 53 + d * 11) % 89) - 44.0f) / 7.0f);
    CUDA_CHECK(cudaMemcpyAsync(q.data(), qh.data(), qh.size() * 2,
                               cudaMemcpyHostToDevice, s));

    // Contiguous reference cache for all logical rows up to kMaxT (the
    // kernels read only [0..position]).
    {
      std::vector<__nv_bfloat16> kh(static_cast<std::size_t>(kRealNkv) *
                                        kRealMaxSeq * kRealHd,
                                    __float2bfloat16_rn(0.0f));
      std::vector<__nv_bfloat16> vh(static_cast<std::size_t>(kRealNkv) *
                                        kRealMaxSeq * kRealHd,
                                    __float2bfloat16_rn(0.0f));
      for (int t = 0; t < kMaxT; ++t)
        for (int n = 0; n < kRealNkv; ++n)
          for (int d = 0; d < kRealHd; ++d) {
            const std::size_t idx =
                (static_cast<std::size_t>(n) * kRealMaxSeq + t) * kRealHd +
                d;
            kh[idx] = syn(t, n, d);
            vh[idx] = syn(t + 1000, n, d);
          }
      CUDA_CHECK(cudaMemcpyAsync(kc.data(), kh.data(), kh.size() * 2,
                                 cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(vc.data(), vh.data(), vh.size() * 2,
                                 cudaMemcpyHostToDevice, s));
    }

    int checked = 0;
    for (int T : {2, 3, 4, 5, 6}) {
      seed(mapA, T, /*ord_base=*/0);  // ordinal 0 at the array base
      std::vector<int> bth((T + kRealPt - 1) / kRealPt);
      for (int b = 0; b < static_cast<int>(bth.size()); ++b)
        bth[static_cast<std::size_t>(b)] = mapA[b];
      while (bth.size() < 8) bth.push_back(9999);
      CUDA_CHECK(cudaMemcpyAsync(bt.data(), bth.data(), bth.size() * 4,
                                 cudaMemcpyHostToDevice, s));

      const int position = T - 1;
      kernels::qwen35_attention_decode_bf16(
          q.data<__nv_bfloat16>(), kc.data<__nv_bfloat16>(),
          vc.data<__nv_bfloat16>(), position, out_c.data<__nv_bfloat16>(),
          kRealNheads, kRealNkv, kRealHd, kRealMaxSeq,
          scratch_c.data<__nv_bfloat16>(), s);
      kernels::qwen35_paged_attention_decode_bf16(
          q.data<__nv_bfloat16>(), k.data<__nv_bfloat16>(),
          v.data<__nv_bfloat16>(), static_cast<int*>(bt.data()), position,
          kRealPt, out_p.data<__nv_bfloat16>(), kRealNheads, kRealNkv,
          kRealHd, page_stride, scratch_p.data<__nv_bfloat16>(), s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      const std::vector<__nv_bfloat16> oc = d2h(
          out_c.data<__nv_bfloat16>(),
          static_cast<std::size_t>(kRealNheads) * kRealHd, s);
      const std::vector<__nv_bfloat16> op = d2h(
          out_p.data<__nv_bfloat16>(),
          static_cast<std::size_t>(kRealNheads) * kRealHd, s);
      CHECK(bits_equal(op, oc));  // BIT-IDENTICAL
      ++checked;
    }
    std::printf("  [ok] real-shape (16/8/64, pt=2, pool layout, "
                "map {2,0,1}) attention BIT-IDENTICAL: %d cases\n",
                checked);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}

}  // namespace

int main() {
  std::printf("test_paged_kv: CUDALM v0.5 Phase B paged-KV hard gate "
              "(synthetic, vs frozen contiguous kernels)\n");
  int rc = 0;
  rc |= test_paged_write();
  rc |= test_paged_attention_parity();
  rc |= test_real_shape_parity();
  if (rc != 0) {
    std::fprintf(stderr, "test_paged_kv: FAIL\n");
    return rc;
  }
  std::printf("test_paged_kv: PASS\n");
  return 0;
}
