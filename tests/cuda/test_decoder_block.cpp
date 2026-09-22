// CUDALM — end-to-end decoder block golden test (M8).
//
// Runs one DecoderBlock step per CUDLMG01 golden file (position 0 and
// position 7) and compares stage-by-stage against the golden tensors with
// the pinned tolerance (|a-r| <= atol + rtol*|r|, atol = rtol = 1e-2 —
// docs/weight_format.md).
//
// Arguments: <block_v01.cudalm> <golden_p0.cudalm> <golden_p7.cudalm>
//
// Checks:
//   * the golden's embedded weights are byte-identical to the standalone
//     .cudalm weight file (same seed — makes the seed match explicit);
//   * all 16 stage tensors vs the golden (tolerance 1e-2);
//   * p = 0 invariants (BIT-EXACT): rope identity (rope_q == q,
//     rope_k == k, cos row 0 == 1 / sin row 0 == 0) and KV rows at
//     position 0 == stage.rope_k / stage.v;
//   * KV state: seeded history rows (t < p) BIT-EXACT vs the golden kv
//     state (pure copy round-trip, pins layout/mapping); the current
//     position row within the pinned tolerance (1e-2) vs the golden and
//     BIT-EXACT equal to the runtime's own stage.rope_k / stage.v rows.

#include <cuda_fp16.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cudalm/decoder_block.h"
#include "cudalm/golden_loader.h"
#include "cudalm/model_config.h"
#include "cudalm/stage_compare.h"
#include "cudalm/weight_loader.h"

#include "../../tests/common/check.h"

using namespace cudalm;

namespace {

struct StageSpec {
  const char* name;  // golden tensor name
  std::size_t elems; // fp16 element count
  const __half* (*get)(const DecoderBlock&);
};

const __half* block_input(const DecoderBlock& b) { return b.stage_input(); }
const __half* block_rms1(const DecoderBlock& b) { return b.stage_rmsnorm1(); }
const __half* block_q(const DecoderBlock& b) { return b.stage_q(); }
const __half* block_k(const DecoderBlock& b) { return b.stage_k(); }
const __half* block_v(const DecoderBlock& b) { return b.stage_v(); }
const __half* block_rq(const DecoderBlock& b) { return b.stage_rope_q(); }
const __half* block_rk(const DecoderBlock& b) { return b.stage_rope_k(); }
const __half* block_attn(const DecoderBlock& b) {
  return b.stage_attention_output();
}
const __half* block_oproj(const DecoderBlock& b) {
  return b.stage_output_projection();
}
const __half* block_res1(const DecoderBlock& b) { return b.stage_residual1(); }
const __half* block_rms2(const DecoderBlock& b) { return b.stage_rmsnorm2(); }
const __half* block_gate(const DecoderBlock& b) { return b.stage_gate(); }
const __half* block_up(const DecoderBlock& b) { return b.stage_up(); }
const __half* block_sgmuc(const DecoderBlock& b) {
  return b.stage_silu_gate_mul_up();
}
const __half* block_down(const DecoderBlock& b) { return b.stage_down(); }
const __half* block_final(const DecoderBlock& b) {
  return b.stage_final_output();
}

std::vector<StageSpec> stage_specs(const ModelConfig& c) {
  const std::size_t H = static_cast<std::size_t>(c.hidden_size);
  const std::size_t qo = static_cast<std::size_t>(c.n_heads) * c.head_dim;
  const std::size_t kv = static_cast<std::size_t>(c.n_kv_heads) * c.head_dim;
  const std::size_t inter = static_cast<std::size_t>(c.intermediate_size);
  return {
      {"stage.input", H, block_input},
      {"stage.rmsnorm1", H, block_rms1},
      {"stage.q", qo, block_q},
      {"stage.k", kv, block_k},
      {"stage.v", kv, block_v},
      {"stage.rope_q", qo, block_rq},
      {"stage.rope_k", kv, block_rk},
      {"stage.attention_output", qo, block_attn},
      {"stage.output_projection", H, block_oproj},
      {"stage.residual1", H, block_res1},
      {"stage.rmsnorm2", H, block_rms2},
      {"stage.gate", inter, block_gate},
      {"stage.up", inter, block_up},
      {"stage.silu_gate_mul_up", inter, block_sgmuc},
      {"stage.down", H, block_down},
      {"stage.final_output", H, block_final},
  };
}

// Copies a device stage buffer to the host and compares against the golden
// host bytes with the pinned tolerance.
int compare_stage(const char* name, const GoldenTensorInfo* g,
                  const __half* device, std::size_t elems,
                  cudaStream_t stream) {
  CHECK(g != nullptr);
  if (g == nullptr) {
    std::fprintf(stderr, "  %s: missing from golden file\n", name);
    return 1;
  }
  CHECK(g->byte_size == elems * 2);
  if (g->byte_size != elems * 2) {
    std::fprintf(stderr, "  %s: golden byte_size %llu != %zu*2\n", name,
                 (unsigned long long)g->byte_size, elems);
    return 1;
  }
  std::vector<__half> act(elems);
  CUDA_CHECK(cudaMemcpyAsync(act.data(), device, elems * sizeof(__half),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  StageCompareResult r = compare_fp16_stages(
      reinterpret_cast<const std::uint16_t*>(g->host_bytes),
      reinterpret_cast<const std::uint16_t*>(act.data()), elems);
  CHECK(r.ok);
  if (!r.ok) {
    std::fprintf(stderr, "  %s: FAIL max_abs_err=%.3e idx=%zu\n", name,
                 r.max_abs_err, r.max_abs_idx);
    return 1;
  }
  std::printf("  %-24s ok  (max_abs_err=%.3e)\n", name, r.max_abs_err);
  return 0;
}

// Seeds the runtime KV history (positions 0..position-1) from the golden
// kv state. Golden row order: (kv_head, position) -> row h*(position+1)+t.
int seed_history(const GoldenFile& g, DecoderBlock& block, cudaStream_t stream) {
  const ModelConfig& c = g.config();
  const int p = g.position();
  const int n_kv = c.n_kv_heads, hd = c.head_dim;
  const GoldenTensorInfo* ks = g.find("kv.k_state");
  const GoldenTensorInfo* vs = g.find("kv.v_state");
  CHECK(ks != nullptr && vs != nullptr);

  const __half* k_state =
      reinterpret_cast<const __half*>(ks->host_bytes);
  const __half* v_state =
      reinterpret_cast<const __half*>(vs->host_bytes);

  DeviceBuffer dk(static_cast<std::size_t>(n_kv) * hd * sizeof(__half), stream);
  DeviceBuffer dv(static_cast<std::size_t>(n_kv) * hd * sizeof(__half), stream);
  for (int t = 0; t < p; ++t) {
    std::vector<__half> kr(static_cast<std::size_t>(n_kv) * hd);
    std::vector<__half> vr(static_cast<std::size_t>(n_kv) * hd);
    for (int h = 0; h < n_kv; ++h) {
      const std::size_t src =
          (static_cast<std::size_t>(h) * (p + 1) + t) * hd;
      std::memcpy(kr.data() + static_cast<std::size_t>(h) * hd,
                  k_state + src, hd * sizeof(__half));
      std::memcpy(vr.data() + static_cast<std::size_t>(h) * hd,
                  v_state + src, hd * sizeof(__half));
    }
    dk.copy_from_host(kr.data(), dk.bytes(), stream);
    dv.copy_from_host(vr.data(), dv.bytes(), stream);
    block.kv_cache().write(t, dk.data<__half>(), dv.data<__half>(), stream);
  }
  return 0;
}

// Runtime KV rows 0..position vs the golden kv state rows:
//   * t < position (history seeded from the golden): BIT-EXACT - a pure
//     copy round-trip through kv_write, pinning the layout/mapping;
//   * t == position (written from the runtime stages): within the pinned
//     tolerance (1e-2) vs the golden - the stages legitimately carry a few
//     fp16 ulps of fp32 re-association / FMA-contract drift vs torch - and
//     BIT-EXACT equal to the runtime's own stage.rope_k / stage.v rows
//     (kv_write is a plain fp16 copy).
int check_kv_state(const GoldenFile& g, const DecoderBlock& block,
                   cudaStream_t stream) {
  const ModelConfig& c = g.config();
  const int p = g.position();
  const int n_kv = c.n_kv_heads, hd = c.head_dim, max_seq = c.max_seq_len;
  const GoldenTensorInfo* ks = g.find("kv.k_state");
  const GoldenTensorInfo* vs = g.find("kv.v_state");
  CHECK(ks != nullptr && vs != nullptr);

  const std::size_t cache_n =
      static_cast<std::size_t>(n_kv) * max_seq * hd;
  std::vector<__half> kdev(cache_n), vdev(cache_n);
  CUDA_CHECK(cudaMemcpyAsync(kdev.data(), block.kv_cache().k(),
                              cache_n * sizeof(__half), cudaMemcpyDeviceToHost,
                              stream));
  CUDA_CHECK(cudaMemcpyAsync(vdev.data(), block.kv_cache().v(),
                              cache_n * sizeof(__half), cudaMemcpyDeviceToHost,
                              stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const __half* k_state = reinterpret_cast<const __half*>(ks->host_bytes);
  const __half* v_state = reinterpret_cast<const __half*>(vs->host_bytes);

  // Runtime stage rows (the source of the position row).
  std::vector<__half> rk(static_cast<std::size_t>(n_kv) * hd),
      vst(static_cast<std::size_t>(n_kv) * hd);
  CUDA_CHECK(cudaMemcpyAsync(rk.data(), block.stage_rope_k(),
                              rk.size() * sizeof(__half),
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(vst.data(), block.stage_v(),
                              vst.size() * sizeof(__half),
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int rc = 0;
  for (int h = 0; h < n_kv; ++h) {
    for (int t = 0; t <= p; ++t) {
      const std::size_t rt =
          (static_cast<std::size_t>(h) * max_seq + t) * hd;
      const std::size_t gs =
          (static_cast<std::size_t>(h) * (p + 1) + t) * hd;
      if (t < p) {
        // Seeded history: bit-exact round-trip.
        const bool okk = std::memcmp(kdev.data() + rt, k_state + gs,
                                     hd * sizeof(__half)) == 0;
        const bool okv = std::memcmp(vdev.data() + rt, v_state + gs,
                                     hd * sizeof(__half)) == 0;
        if (!okk || !okv) {
          std::fprintf(stderr,
                       "  kv state: seeded row (h=%d,t=%d) bit-exact FAIL "
                       "(k=%d v=%d)\n",
                       h, t, (int)okk, (int)okv);
          rc |= 1;
        }
      } else {
        // Current position: tolerance vs golden + bit-exact vs the runtime
        // stage rows (plain-copy invariant of kv_write).
        const StageCompareResult rk_c = compare_fp16_stages(
            reinterpret_cast<const std::uint16_t*>(k_state + gs),
            reinterpret_cast<const std::uint16_t*>(kdev.data() + rt), hd);
        const StageCompareResult vs_c = compare_fp16_stages(
            reinterpret_cast<const std::uint16_t*>(v_state + gs),
            reinterpret_cast<const std::uint16_t*>(vdev.data() + rt), hd);
        const bool okk =
            rk_c.ok && std::memcmp(kdev.data() + rt,
                                   rk.data() + static_cast<std::size_t>(h) * hd,
                                   hd * sizeof(__half)) == 0;
        const bool okv =
            vs_c.ok && std::memcmp(vdev.data() + rt,
                                   vst.data() + static_cast<std::size_t>(h) * hd,
                                   hd * sizeof(__half)) == 0;
        if (!okk || !okv) {
          std::fprintf(stderr,
                       "  kv state: position row (h=%d) FAIL (k_tol=%d "
                       "v_tol=%d)\n",
                       h, (int)rk_c.ok, (int)vs_c.ok);
          rc |= 1;
        }
      }
    }
  }
  if (rc == 0)
    std::printf("  %-24s ok  (rows 0..%d: history bit-exact, position "
                "tolerance+copy)\n",
                "kv state", p);
  return rc;
}

int run_golden(const char* path, const BlockWeights& weights,
               cudaStream_t stream) {
  std::printf("[golden] %s\n", path);
  GoldenFile g;
  Status s = GoldenFile::load(path, &g);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "  load failed: %s\n", s.message.c_str());
    return 1;
  }
  const ModelConfig& c = g.config();
  const int p = g.position();
  CHECK(c.valid());

  DecoderBlock block(weights, stream);

  // Input: golden stage.input -> device.
  const GoldenTensorInfo* gin = g.find("stage.input");
  CHECK(gin != nullptr);
  DeviceBuffer dx(gin->byte_size, stream);
  dx.copy_from_host(gin->host_bytes, gin->byte_size, stream);

  int rc = 0;
  if (p > 0) rc |= seed_history(g, block, stream);
  block.forward(p, dx.data<__half>(), stream);

  for (const StageSpec& spec : stage_specs(c)) {
    rc |= compare_stage(spec.name, g.find(spec.name), spec.get(block),
                        spec.elems, stream);
  }

  // p = 0 invariants: rope identity (cos row 0 == 1, sin row 0 == 0).
  if (p == 0) {
    const std::size_t qo = static_cast<std::size_t>(c.n_heads) * c.head_dim;
    const std::size_t kv = static_cast<std::size_t>(c.n_kv_heads) * c.head_dim;
    std::vector<__half> rq(qo), rk(kv), q(qo), k(kv);
    CUDA_CHECK(cudaMemcpyAsync(rq.data(), block.stage_rope_q(),
                               qo * sizeof(__half), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(rk.data(), block.stage_rope_k(),
                               kv * sizeof(__half), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(q.data(), block.stage_q(),
                               qo * sizeof(__half), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(k.data(), block.stage_k(),
                               kv * sizeof(__half), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(std::memcmp(rq.data(), q.data(), qo * sizeof(__half)) == 0);
    CHECK(std::memcmp(rk.data(), k.data(), kv * sizeof(__half)) == 0);
    std::printf("  %-24s ok  (p=0 rope identity bit-exact)\n",
                "rope identity");
  }

  rc |= check_kv_state(g, block, stream);
  if (rc == 0) std::printf("[golden] PASS (position %d)\n", p);
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr,
                 "usage: %s <block_v01.cudalm> <golden_p0> <golden_p7>\n",
                 argv[0]);
    return 2;
  }
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  CHECK(n >= 1);
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Weights: standalone .cudalm + byte-equality cross-check against each
  // golden's embedded weights (same seed contract).
  WeightFile wf;
  Status sw = WeightFile::load(argv[1], &wf);
  CHECK(sw.ok);
  BlockWeights weights;
  Status sb = BlockWeights::load(argv[1], stream, &weights);
  CHECK(sb.ok);

  GoldenFile g0, g7;
  Status s0 = GoldenFile::load(argv[2], &g0);
  Status s7 = GoldenFile::load(argv[3], &g7);
  CHECK(s0.ok && s7.ok);

  int rc = 0;
  for (GoldenFile* g : {&g0, &g7}) {
    for (const WeightTensorInfo& t : wf.tensors()) {
      const GoldenTensorInfo* gt = g->find(t.name);
      CHECK(gt != nullptr);
      if (gt == nullptr) {
        std::fprintf(stderr, "  weight %s missing from golden\n",
                     t.name.c_str());
        rc |= 1;
        continue;
      }
      CHECK(gt->byte_size == t.byte_size &&
            std::memcmp(gt->host_bytes, t.host_bytes, t.byte_size) == 0);
    }
  }
  if (rc != 0) {
    std::fprintf(stderr, "weight cross-check FAILED\n");
    return rc;
  }
  std::printf("[weights] 18 tensors byte-identical in both goldens\n");

  rc |= run_golden(argv[2], weights, stream);
  rc |= run_golden(argv[3], weights, stream);

  CUDA_CHECK(cudaStreamDestroy(stream));
  if (rc == 0) TEST_PASS("test_decoder_block");
  return rc;
}
