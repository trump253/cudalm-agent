// CUDALM — v0.6 Phase B: kernel-level row-parity hard gate for the TRUE
// BATCHED decode kernel family (tests/cuda/test_qwen35_batch_kernels.cpp).
//
// For EACH new batched kernel, this proves ROW-WISE parity with the frozen
// v0.5 single-token kernel: for every row b in a batch of B, the batched
// kernel's row-b output is BIT-IDENTICAL to the frozen single kernel run on
// row b's own inputs (same math, same accumulation order, same bf16
// rounding — the batch kernel only adds a batch axis). No CPU reference is
// used: the frozen single kernel IS the reference (the bit-exact contract).
//
// Coverage (the five Phase B kernel-level gates):
//   1. W4A16 GEMV        : batch_int4_gemv_bf16        vs int4_gemv_bf16
//   2. BF16 LM-head GEMV : batch_bf16_gemv             vs bf16_gemv
//   3. Paged KV write    : batch_paged_kv_write_bf16   vs qwen35_paged_kv_write_bf16
//                          (non-contiguous page mapping, heterogeneous pos)
//   4. Paged attention   : batch_paged_attention_decode_bf16
//                                        vs qwen35_paged_attention_decode_bf16
//                          (GQA, per-row block tables, heterogeneous pos)
//   5. Delta stateful    : batch_deltanet_conv_decode_bf16 vs
//                                            qwen35_deltanet_conv_decode_bf16
//                          batch_deltanet_gbeta_bf16          vs
//                                            qwen35_deltanet_gbeta_bf16
//                          batch_deltanet_delta_rule_fp32     vs
//                                            qwen35_deltanet_delta_rule_fp32
//                          (per-row Delta slots, in-place state update)
//
// Deterministic synthetic inputs (no checkpoint, no Python): the parity is
// a pure function of the (identical) per-row inputs, so any divergence is a
// real batch-indexing / math bug.

#include "../../tests/common/check.h"

#include "cudalm/device_buffer.h"
#include "cudalm/kernels/batch_decode.h"
#include "cudalm/kernels/bf16_gemv.h"
#include "cudalm/kernels/int4_gemv_bf16.h"
#include "cudalm/kernels/paged_kv.h"
#include "cudalm/kernels/qwen35_deltanet_kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cudalm;

namespace {

// Deterministic LCG in [-1, 1).
struct Rng {
  std::uint32_t s;
  explicit Rng(std::uint32_t seed) : s(seed ? seed : 1u) {}
  float next() {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(s >> 8) * (2.0f / static_cast<float>(1u << 24)) -
           1.0f;
  }
};

// Bit-exact buffer compare (bf16 or fp32 via sizeof(T)); prints on failure.
template <class T>
int cmp(const char* name, const T* ref, const T* act, std::size_t n,
        cudaStream_t st) {
  std::vector<T> r(n), a(n);
  if (n) {
    CUDA_CHECK(cudaMemcpyAsync(r.data(), ref, n * sizeof(T),
                               cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(a.data(), act, n * sizeof(T),
                               cudaMemcpyDeviceToHost, st));
  }
  const bool ok = std::memcmp(r.data(), a.data(), n * sizeof(T)) == 0;
  std::fprintf(stderr, "  %-40s %s (n=%zu)\n", name, ok ? "BIT-IDENTICAL"
                                                        : "FAIL",
               n);
  return ok ? 0 : 1;
}

// =====================================================================
// 1. W4A16 GEMV (batch_int4_gemv_bf16 vs int4_gemv_bf16), B rows.
// =====================================================================
int test_int4_gemv() {
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreate(&st));
  const int N = 512, K = 512, B = 3;
  Rng rng(101);
  // Weight: int4 packed [N, K] = N*K/2 bytes; scale [N, K/128] fp16.
  std::vector<uint8_t> W(static_cast<std::size_t>(N) * K / 2);
  std::vector<__half> scale(static_cast<std::size_t>(N) * (K / 128));
  for (auto& w : W) w = static_cast<uint8_t>(rng.next() * 127.0f + 127.0f);
  for (auto& sc : scale) sc = __float2half(0.5f * rng.next() + 0.5f);
  std::vector<__nv_bfloat16> x(static_cast<std::size_t>(B) * K);
  for (auto& v : x) v = __float2bfloat16_rn(rng.next());

  DeviceBuffer dW, dS, dx, dy, yb;
  dW.allocate(W.size(), st);
  dS.allocate(scale.size() * 2, st);
  dx.allocate(x.size() * 2, st);
  dy.allocate(static_cast<std::size_t>(B) * N * 2, st);
  yb.allocate(static_cast<std::size_t>(N) * 2, st);
  CUDA_CHECK(cudaMemcpyAsync(dW.data(), W.data(), W.size(),
                             cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dS.data(), scale.data(), scale.size() * 2,
                             cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dx.data(), x.data(), x.size() * 2,
                             cudaMemcpyHostToDevice, st));

  kernels::batch_int4_gemv_bf16(dW.data<uint8_t>(), dS.data<__half>(),
                                dx.data<__nv_bfloat16>(),
                                dy.data<__nv_bfloat16>(), N, K, B, st);
  int rc = 0;
  for (int b = 0; b < B; ++b) {
    int4_gemv_bf16(dW.data<uint8_t>(), dS.data<__half>(),
                   dx.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * K,
                   yb.data<__nv_bfloat16>(), N, K, st);
    char name[64];
    std::snprintf(name, sizeof(name), "INT4GEMV row%d vs single", b);
    rc |= cmp(name, dy.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * N,
              yb.data<__nv_bfloat16>(), static_cast<std::size_t>(N), st);
  }
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaStreamDestroy(st));
  return rc;
}

// =====================================================================
// 2. BF16 LM-head GEMV (batch_bf16_gemv vs bf16_gemv), B rows.
// =====================================================================
int test_bf16_gemv() {
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreate(&st));
  const int N = 256, K = 1024, B = 3;
  Rng rng(202);
  std::vector<__nv_bfloat16> W(static_cast<std::size_t>(N) * K),
      x(static_cast<std::size_t>(B) * K);
  for (auto& w : W) w = __float2bfloat16_rn(rng.next());
  for (auto& v : x) v = __float2bfloat16_rn(rng.next());
  DeviceBuffer dW, dx, dy, yb;
  dW.allocate(W.size() * 2, st);
  dx.allocate(x.size() * 2, st);
  dy.allocate(static_cast<std::size_t>(B) * N * 2, st);
  yb.allocate(static_cast<std::size_t>(N) * 2, st);
  CUDA_CHECK(cudaMemcpyAsync(dW.data(), W.data(), W.size() * 2,
                             cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dx.data(), x.data(), x.size() * 2,
                             cudaMemcpyHostToDevice, st));
  kernels::batch_bf16_gemv(dW.data<__nv_bfloat16>(), dx.data<__nv_bfloat16>(),
                           dy.data<__nv_bfloat16>(), N, K, B, st);
  int rc = 0;
  for (int b = 0; b < B; ++b) {
    bf16_gemv(dW.data<__nv_bfloat16>(),
              dx.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * K,
              yb.data<__nv_bfloat16>(), N, K, st);
    char name[64];
    std::snprintf(name, sizeof(name), "BF16GEMV row%d vs single", b);
    rc |= cmp(name, dy.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * N,
              yb.data<__nv_bfloat16>(), static_cast<std::size_t>(N), st);
  }
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaStreamDestroy(st));
  return rc;
}

// =====================================================================
// 5. Delta stateful kernels (per-row slot, in-place update).
// =====================================================================
int test_deltanet() {
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreate(&st));
  const int B = 2;
  const int conv_dim = 96, n_heads = 16, hd = 128, eps = 1e-6f;
  // The conv / gbeta / delta-rule kernels are independent; we drive them with
  // per-row tensors (conv_dim is the conv's channel count; the delta rule
  // uses the pinned head_dim == 128).
  Rng rng(303);
  const std::size_t conv_state_per =
      static_cast<std::size_t>(conv_dim) * 3;
  const std::size_t rec_per =
      static_cast<std::size_t>(n_heads) * hd * hd;
  // Initial conv state: 2 slots (slot b for row b).
  std::vector<__nv_bfloat16> conv0(2 * conv_state_per);
  for (auto& v : conv0) v = __float2bfloat16_rn(rng.next());
  // Initial rec state: 2 slots.
  std::vector<float> rec0(2 * rec_per);
  for (auto& v : rec0) v = 0.01f * rng.next();
  // new_mixed [B][conv_dim], conv_w [conv_dim][4].
  std::vector<__nv_bfloat16> mixed(static_cast<std::size_t>(B) * conv_dim),
      convw(static_cast<std::size_t>(conv_dim) * 4);
  for (auto& v : mixed) v = __float2bfloat16_rn(rng.next());
  for (auto& v : convw) v = __float2bfloat16_rn(rng.next());
  // b/a [B][n_heads], A_log [n_heads] f32, dt_bias [n_heads] bf16.
  std::vector<__nv_bfloat16> bh(static_cast<std::size_t>(B) * n_heads),
      ah(static_cast<std::size_t>(B) * n_heads), dtb(n_heads);
  std::vector<float> Alog(n_heads);
  for (auto& v : bh) v = __float2bfloat16_rn(rng.next());
  for (auto& v : ah) v = __float2bfloat16_rn(rng.next());
  for (auto& v : dtb) v = __float2bfloat16_rn(rng.next());
  for (auto& v : Alog) v = 0.1f * rng.next();
  // q/k/v [B][n_heads][hd], beta/g computed by the kernel.
  std::vector<__nv_bfloat16> qh(static_cast<std::size_t>(B) * n_heads * hd),
      kh(static_cast<std::size_t>(B) * n_heads * hd),
      vh(static_cast<std::size_t>(B) * n_heads * hd);
  for (auto& v : qh) v = __float2bfloat16_rn(rng.next());
  for (auto& v : kh) v = __float2bfloat16_rn(rng.next());
  for (auto& v : vh) v = __float2bfloat16_rn(rng.next());

  DeviceBuffer dconv, drec, dmixed, dconvw, db, da, dAlog, ddtb, dq, dk, dv,
      dbeta, dg, dcore, dconv_out, dconv_silu, dslots;
  DeviceBuffer scon, srec, sco, scs, sbeta, score;
  int rc = 0;

  // ---- conv (per-row slot) ----
  {
    dconv.allocate(2 * conv_state_per * 2, st);
    dmixed.allocate(mixed.size() * 2, st);
    dconvw.allocate(convw.size() * 2, st);
    dconv_out.allocate(static_cast<std::size_t>(B) * conv_dim * 2, st);
    dconv_silu.allocate(static_cast<std::size_t>(B) * conv_dim * 2, st);
    dslots.allocate(2 * 4, st);
    scon.allocate(2 * conv_state_per * 2, st);  // 2 slots (row b's slot)
    sco.allocate(conv_dim * 2, st);
    scs.allocate(conv_dim * 2, st);
    std::vector<int> sl = {0, 1};
    CUDA_CHECK(cudaMemcpyAsync(dconv.data(), conv0.data(),
                               conv0.size() * 2, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(dmixed.data(), mixed.data(),
                               mixed.size() * 2, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(dconvw.data(), convw.data(),
                               convw.size() * 2, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(dslots.data(), sl.data(), 8,
                               cudaMemcpyHostToDevice, st));
    kernels::batch_deltanet_conv_decode_bf16(dconv.data<__nv_bfloat16>(),
                                             dslots.data<int>(),
                                             dmixed.data<__nv_bfloat16>(),
                                             dconvw.data<__nv_bfloat16>(),
                                             dconv_out.data<__nv_bfloat16>(),
                                             dconv_silu.data<__nv_bfloat16>(),
                                             conv_dim, B, st);
    for (int b = 0; b < B; ++b) {
      CUDA_CHECK(cudaMemcpyAsync(scon.data(), conv0.data(),
                                 2 * conv_state_per * 2, cudaMemcpyHostToDevice,
                                 st));
      kernels::qwen35_deltanet_conv_decode_bf16(
          scon.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * conv_state_per,
          dmixed.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * conv_dim,
          dconvw.data<__nv_bfloat16>(), sco.data<__nv_bfloat16>(),
          scs.data<__nv_bfloat16>(), conv_dim, st);
      char nm[64];
      std::snprintf(nm, sizeof(nm), "DELTA.conv row%d out", b);
      rc |= cmp(nm, dconv_out.data<__nv_bfloat16>() +
                        static_cast<std::size_t>(b) * conv_dim,
                sco.data<__nv_bfloat16>(), static_cast<std::size_t>(conv_dim),
                st);
      std::snprintf(nm, sizeof(nm), "DELTA.conv row%d silu", b);
      rc |= cmp(nm, dconv_silu.data<__nv_bfloat16>() +
                        static_cast<std::size_t>(b) * conv_dim,
                scs.data<__nv_bfloat16>(), static_cast<std::size_t>(conv_dim),
                st);
      std::snprintf(nm, sizeof(nm), "DELTA.conv row%d state", b);
      rc |= cmp(nm, dconv.data<__nv_bfloat16>() +
                        static_cast<std::size_t>(b) * conv_state_per,
                scon.data<__nv_bfloat16>() +
                    static_cast<std::size_t>(b) * conv_state_per,
                conv_state_per, st);
    }
  }
  // ---- gbeta (per (b,h)) ----
  {
    db.allocate(bh.size() * 2, st);
    da.allocate(ah.size() * 2, st);
    dAlog.allocate(Alog.size() * 4, st);
    ddtb.allocate(dtb.size() * 2, st);
    dbeta.allocate(static_cast<std::size_t>(B) * n_heads * 2, st);
    dg.allocate(static_cast<std::size_t>(B) * n_heads * 4, st);
    sbeta.allocate(n_heads * 2, st);
    DeviceBuffer sg(n_heads * 4);
    sg.allocate(n_heads * 4, st);
    CUDA_CHECK(cudaMemcpyAsync(db.data(), bh.data(), bh.size() * 2,
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(da.data(), ah.data(), ah.size() * 2,
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(dAlog.data(), Alog.data(),
                               Alog.size() * 4, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ddtb.data(), dtb.data(), dtb.size() * 2,
                               cudaMemcpyHostToDevice, st));
    kernels::batch_deltanet_gbeta_bf16(db.data<__nv_bfloat16>(),
                                       da.data<__nv_bfloat16>(),
                                       dAlog.data<float>(),
                                       ddtb.data<__nv_bfloat16>(),
                                       dbeta.data<__nv_bfloat16>(),
                                       dg.data<float>(), n_heads, B, st);
    for (int b = 0; b < B; ++b) {
      kernels::qwen35_deltanet_gbeta_bf16(
          db.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * n_heads,
          da.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * n_heads,
          dAlog.data<float>(), ddtb.data<__nv_bfloat16>(),
          sbeta.data<__nv_bfloat16>(), sg.data<float>(), n_heads, st);
      char nm[64];
      std::snprintf(nm, sizeof(nm), "DELTA.gbeta row%d beta", b);
      rc |= cmp(nm, dbeta.data<__nv_bfloat16>() +
                        static_cast<std::size_t>(b) * n_heads,
                sbeta.data<__nv_bfloat16>(), static_cast<std::size_t>(n_heads),
                st);
      std::snprintf(nm, sizeof(nm), "DELTA.gbeta row%d g", b);
      rc |= cmp(nm, dg.data<float>() + static_cast<std::size_t>(b) * n_heads,
                sg.data<float>(), static_cast<std::size_t>(n_heads), st);
    }
  }
  // ---- delta rule (per-row rec slot, in-place) ----
  {
    dq.allocate(qh.size() * 2, st);
    dk.allocate(kh.size() * 2, st);
    dv.allocate(vh.size() * 2, st);
    drec.allocate(2 * rec_per * 4, st);
    dcore.allocate(static_cast<std::size_t>(B) * n_heads * hd * 2, st);
    score.allocate(n_heads * hd * 2, st);
    srec.allocate(rec_per * 4, st);
    // Reuse the g/beta buffers filled above (dbeta, dg).
    CUDA_CHECK(cudaMemcpyAsync(dq.data(), qh.data(), qh.size() * 2,
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(dk.data(), kh.data(), kh.size() * 2,
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(dv.data(), vh.data(), vh.size() * 2,
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(drec.data(), rec0.data(), rec0.size() * 4,
                               cudaMemcpyHostToDevice, st));
    // dslots already holds {0,1}.
    kernels::batch_deltanet_delta_rule_fp32(dq.data<__nv_bfloat16>(),
                                            dk.data<__nv_bfloat16>(),
                                            dv.data<__nv_bfloat16>(),
                                            dg.data<float>(),
                                            dbeta.data<__nv_bfloat16>(),
                                            drec.data<float>(),
                                            dslots.data<int>(),
                                            dcore.data<__nv_bfloat16>(),
                                            n_heads, hd, eps, B, st);
    for (int b = 0; b < B; ++b) {
      CUDA_CHECK(cudaMemcpyAsync(srec.data(),
                                 rec0.data() + static_cast<std::size_t>(b) *
                                                    rec_per,
                                 rec_per * 4, cudaMemcpyHostToDevice, st));
      kernels::qwen35_deltanet_delta_rule_fp32(
          dq.data<__nv_bfloat16>() +
              static_cast<std::size_t>(b) * n_heads * hd,
          dk.data<__nv_bfloat16>() +
              static_cast<std::size_t>(b) * n_heads * hd,
          dv.data<__nv_bfloat16>() +
              static_cast<std::size_t>(b) * n_heads * hd,
          dg.data<float>() + static_cast<std::size_t>(b) * n_heads,
          dbeta.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * n_heads,
          srec.data<float>(), score.data<__nv_bfloat16>(), n_heads, hd, eps,
          st);
      char nm[64];
      std::snprintf(nm, sizeof(nm), "DELTA.rule row%d core", b);
      rc |= cmp(nm, dcore.data<__nv_bfloat16>() +
                        static_cast<std::size_t>(b) * n_heads * hd,
                score.data<__nv_bfloat16>(),
                static_cast<std::size_t>(n_heads) * hd, st);
      std::snprintf(nm, sizeof(nm), "DELTA.rule row%d state", b);
      rc |= cmp(nm, drec.data<float>() + static_cast<std::size_t>(b) * rec_per,
                srec.data<float>(), rec_per, st);
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaStreamDestroy(st));
  return rc;
}

// Host-side bit-exact compare (both buffers already on host).
int cmp_host(const char* name, const std::vector<__nv_bfloat16>& ref,
             const std::vector<__nv_bfloat16>& act) {
  if (ref.size() != act.size()) {
    std::fprintf(stderr, "  %-40s FAIL size %zu != %zu\n", name, ref.size(),
                 act.size());
    return 1;
  }
  const bool ok = std::memcmp(ref.data(), act.data(), ref.size() * 2) == 0;
  std::fprintf(stderr, "  %-40s %s (n=%zu)\n", name,
               ok ? "BIT-IDENTICAL" : "FAIL", ref.size());
  return ok ? 0 : 1;
}

// =====================================================================
// 3 + 4. Paged KV write + paged attention (non-contiguous pages,
// heterogeneous positions, per-row block tables, GQA).
// =====================================================================
int test_paged() {
  cudaStream_t st = nullptr;
  CUDA_CHECK(cudaStreamCreate(&st));
  const int B = 2;
  const int n_heads = 4, n_kv = 2, hd = 16, pt = 2;
  const int row_stride = 8;   // block-table entries per row
  const int num_pages = 12;
  const std::size_t page_elems =
      static_cast<std::size_t>(n_kv) * pt * hd;  // bf16 per page
  const int T_max = 6;
  const int pos[B] = {2, 5};  // heterogeneous decode positions
  Rng rng(404);

  // Per-row block table (logical block -> physical page): DISTINCT and
  // NON-contiguous page sets per row (row 0 -> pages {0,1}, row 1 ->
  // pages {2,3,4} shuffled), so the shared page storage holds both rows'
  // prefixes with zero overlap.
  std::vector<int> bt(static_cast<std::size_t>(B) * row_stride, -1);
  {
    int page_counter = 0;
    for (int b = 0; b < B; ++b) {
      const int nblocks = (pos[b] + 1 + pt - 1) / pt;
      std::vector<int> pages;
      for (int i = 0; i < nblocks; ++i) pages.push_back(page_counter++ % num_pages);
      for (int i = 0; i < nblocks; ++i)
        bt[static_cast<std::size_t>(b) * row_stride + i] = pages[(i + b) % nblocks];
    }
  }

  // Shared page storage: fill every logical token t in [0..pos[b]] of row b
  // at (block_table_rowb[t/pt], t%pt) with deterministic K/V.
  std::vector<__nv_bfloat16> kshared(static_cast<std::size_t>(num_pages) * page_elems),
      vshared(static_cast<std::size_t>(num_pages) * page_elems);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t <= pos[b]; ++t) {
      const int page = bt[static_cast<std::size_t>(b) * row_stride + t / pt];
      const int off = t % pt;
      for (int n = 0; n < n_kv; ++n) {
        const std::size_t base = static_cast<std::size_t>(page) * page_elems +
                                 (static_cast<std::size_t>(n) * pt + off) * hd;
        for (int d = 0; d < hd; ++d) {
          kshared[base + d] = __float2bfloat16_rn(rng.next());
          vshared[base + d] = __float2bfloat16_rn(rng.next());
        }
      }
    }
  // The decode token's K/V (written by the KV write kernel) + query, per row.
  std::vector<__nv_bfloat16> ksrc(static_cast<std::size_t>(B) * n_kv * hd),
      vsrc(static_cast<std::size_t>(B) * n_kv * hd),
      q(static_cast<std::size_t>(B) * n_heads * hd);
  for (auto& w : ksrc) w = __float2bfloat16_rn(rng.next());
  for (auto& w : vsrc) w = __float2bfloat16_rn(rng.next());
  for (auto& w : q) w = __float2bfloat16_rn(rng.next());

  const std::size_t page_bytes = static_cast<std::size_t>(num_pages) * page_elems * 2;
  DeviceBuffer dpk_b, dpv_b, dpk_s, dpv_s, dbt, dbt_row, dpos, dksrc, dvsrc, dq,
      dout, dout_s, dscratch, dscratch_s;
  dpk_b.allocate(page_bytes, st);
  dpv_b.allocate(page_bytes, st);
  dpk_s.allocate(page_bytes, st);
  dpv_s.allocate(page_bytes, st);
  dbt.allocate(bt.size() * 4, st);
  dbt_row.allocate(row_stride * 4, st);
  dpos.allocate(B * 4, st);
  dksrc.allocate(ksrc.size() * 2, st);
  dvsrc.allocate(vsrc.size() * 2, st);
  dq.allocate(q.size() * 2, st);
  dout.allocate(static_cast<std::size_t>(B) * n_heads * hd * 2, st);
  dout_s.allocate(static_cast<std::size_t>(n_heads) * hd * 2, st);
  dscratch.allocate(2 * static_cast<std::size_t>(B) * n_heads * T_max * 2, st);
  dscratch_s.allocate(2 * static_cast<std::size_t>(n_heads) * T_max * 2, st);
  CUDA_CHECK(cudaMemcpyAsync(dbt.data(), bt.data(), bt.size() * 4,
                             cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dpos.data(), pos, B * 4, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dksrc.data(), ksrc.data(), ksrc.size() * 2,
                             cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dvsrc.data(), vsrc.data(), vsrc.size() * 2,
                             cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dq.data(), q.data(), q.size() * 2,
                             cudaMemcpyHostToDevice, st));
  int rc = 0;

  // ---------------- KV WRITE (batch vs frozen single, per row) ---------
  CUDA_CHECK(cudaMemcpyAsync(dpk_b.data(), kshared.data(), page_bytes,
                             cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dpv_b.data(), vshared.data(), page_bytes,
                             cudaMemcpyHostToDevice, st));
  kernels::batch_paged_kv_write_bf16(
      dksrc.data<__nv_bfloat16>(), dvsrc.data<__nv_bfloat16>(),
      dpk_b.data<__nv_bfloat16>(), dpv_b.data<__nv_bfloat16>(), dbt.data<int>(),
      dpos.data<int>(), row_stride, pt, n_kv, hd, page_elems, B, st);
  for (int b = 0; b < B; ++b) {
    // Frozen single reference: fresh identical page storage + row b's table.
    CUDA_CHECK(cudaMemcpyAsync(dpk_s.data(), kshared.data(), page_bytes,
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(dpv_s.data(), vshared.data(), page_bytes,
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(dbt_row.data(),
                               bt.data() + static_cast<std::size_t>(b) * row_stride,
                               row_stride * 4, cudaMemcpyHostToDevice, st));
    kernels::qwen35_paged_kv_write_bf16(
        dksrc.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * n_kv * hd,
        dvsrc.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * n_kv * hd,
        dpk_s.data<__nv_bfloat16>(), dpv_s.data<__nv_bfloat16>(),
        dbt_row.data<int>(), pos[b], pt, n_kv, hd, page_elems, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    // Compare the FULL page storage's row-b scatter slice.
    const int page = bt[static_cast<std::size_t>(b) * row_stride + pos[b] / pt];
    const int off = pos[b] % pt;
    std::vector<__nv_bfloat16> bk(page_bytes / 2), bs(page_bytes / 2),
        vk(page_bytes / 2), vs(page_bytes / 2);
    CUDA_CHECK(cudaMemcpyAsync(bk.data(), dpk_b.data(), page_bytes,
                               cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(bs.data(), dpk_s.data(), page_bytes,
                               cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(vk.data(), dpv_b.data(), page_bytes,
                               cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaMemcpyAsync(vs.data(), dpv_s.data(), page_bytes,
                               cudaMemcpyDeviceToHost, st));
    const std::size_t base =
        static_cast<std::size_t>(page) * page_elems + static_cast<std::size_t>(off) * hd;
    char nm[64];
    for (int n = 0; n < n_kv; ++n) {
      const std::size_t s0 = base + static_cast<std::size_t>(n) * pt * hd;
      const std::size_t s1 = base + static_cast<std::size_t>(n + 1) * pt * hd;
      std::snprintf(nm, sizeof(nm), "PAGED.KV row%d K@%d h%d", b, pos[b], n);
      rc |= cmp_host(nm,
                     std::vector<__nv_bfloat16>(bs.begin() + s0, bs.begin() + s1),
                     std::vector<__nv_bfloat16>(bk.begin() + s0, bk.begin() + s1));
      std::snprintf(nm, sizeof(nm), "PAGED.KV row%d V@%d h%d", b, pos[b], n);
      rc |= cmp_host(nm,
                     std::vector<__nv_bfloat16>(vs.begin() + s0, vs.begin() + s1),
                     std::vector<__nv_bfloat16>(vk.begin() + s0, vk.begin() + s1));
    }
  }

  // ---------------- ATTENTION (batch vs frozen single, per row) --------
  // Re-seed the shared pages + the decode writes (batch), then attend.
  CUDA_CHECK(cudaMemcpyAsync(dpk_b.data(), kshared.data(), page_bytes,
                             cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaMemcpyAsync(dpv_b.data(), vshared.data(), page_bytes,
                             cudaMemcpyHostToDevice, st));
  kernels::batch_paged_kv_write_bf16(
      dksrc.data<__nv_bfloat16>(), dvsrc.data<__nv_bfloat16>(),
      dpk_b.data<__nv_bfloat16>(), dpv_b.data<__nv_bfloat16>(), dbt.data<int>(),
      dpos.data<int>(), row_stride, pt, n_kv, hd, page_elems, B, st);
  kernels::batch_paged_attention_decode_bf16(
      dq.data<__nv_bfloat16>(), dpk_b.data<__nv_bfloat16>(),
      dpv_b.data<__nv_bfloat16>(), dbt.data<int>(), dpos.data<int>(), row_stride,
      pt, dout.data<__nv_bfloat16>(), n_heads, n_kv, hd, page_elems, T_max, B,
      dscratch.data<__nv_bfloat16>(), st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  std::vector<__nv_bfloat16> ob(static_cast<std::size_t>(B) * n_heads * hd);
  CUDA_CHECK(cudaMemcpyAsync(ob.data(), dout.data(), ob.size() * 2,
                             cudaMemcpyDeviceToHost, st));
  for (int b = 0; b < B; ++b) {
    CUDA_CHECK(cudaMemcpyAsync(dbt_row.data(),
                               bt.data() + static_cast<std::size_t>(b) * row_stride,
                               row_stride * 4, cudaMemcpyHostToDevice, st));
    kernels::qwen35_paged_attention_decode_bf16(
        dq.data<__nv_bfloat16>() + static_cast<std::size_t>(b) * n_heads * hd,
        dpk_b.data<__nv_bfloat16>(), dpv_b.data<__nv_bfloat16>(),
        dbt_row.data<int>(), pos[b], pt, dout_s.data<__nv_bfloat16>(), n_heads,
        n_kv, hd, page_elems, dscratch_s.data<__nv_bfloat16>(), st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<__nv_bfloat16> os(n_heads * hd);
    CUDA_CHECK(cudaMemcpyAsync(os.data(), dout_s.data(), os.size() * 2,
                               cudaMemcpyDeviceToHost, st));
    char nm[64];
    std::snprintf(nm, sizeof(nm), "PAGED.ATTN row%d vs single", b);
    rc |= cmp_host(nm,
                   std::vector<__nv_bfloat16>(
                       ob.begin() + static_cast<std::size_t>(b) * n_heads * hd,
                       ob.begin() + (static_cast<std::size_t>(b) + 1) * n_heads * hd),
                   os);
  }
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaStreamDestroy(st));
  return rc;
}

}  // namespace

int main() {
  int rc = 0;
  std::fprintf(stderr, "== W4A16 GEMV (batch vs single) ==\n");
  rc |= test_int4_gemv();
  std::fprintf(stderr, "== BF16 LM-head GEMV (batch vs single) ==\n");
  rc |= test_bf16_gemv();
  std::fprintf(stderr, "== Delta stateful (batch vs single) ==\n");
  rc |= test_deltanet();
  std::fprintf(stderr, "== Paged KV + attention (batch vs single) ==\n");
  rc |= test_paged();
  if (rc != 0) {
    std::fprintf(stderr, "test_qwen35_batch_kernels: FAIL\n");
    return rc;
  }
  std::printf("test_qwen35_batch_kernels: PASS (every batched kernel row is "
              "BIT-IDENTICAL to the frozen single kernel)\n");
  return 0;
}
