// CUDALM — golden-reference container test (CPU, no CUDA).
//
// Loads the Python-generated CUDLMG01 golden files (positions 0 and 7) with
// the PyTorch-free C++ loader, validates the full tensor set, checks stage
// finiteness, and pins the exact invariants the generator guarantees
// (position-0 identity RoPE, softmax-over-one attention, KV-state rows).
// Also unit-checks the fp16 decode + stage-compare utilities and the
// loader's negative path (corrupted headers).
//
// Usage: test_golden_file [golden_p0.cudalm] [golden_p7.cudalm]
// (defaults: data/block_v01_golden_p0.cudalm, data/block_v01_golden_p7.cudalm)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cudalm/golden_loader.h"
#include "cudalm/model_config.h"
#include "cudalm/stage_compare.h"
#include "cudalm/tensor.h"

#include "../../tests/common/check.h"

using namespace cudalm;

namespace {

// CUDLMW01 / CUDLMG01 differ only at magic byte 5 ('W' vs 'G').
const std::uint8_t kByteW = 'W';
const std::uint8_t kByteG = 'G';

const char* const kStageNames[] = {
    "stage.input", "stage.rmsnorm1", "stage.q", "stage.k", "stage.v",
    "stage.rope_q", "stage.rope_k", "stage.attention_output",
    "stage.output_projection", "stage.residual1", "stage.rmsnorm2",
    "stage.gate", "stage.up", "stage.silu_gate_mul_up", "stage.down",
    "stage.final_output",
};
constexpr int kNumStages = 16;

bool file_bytes(const std::string& path, std::vector<std::uint8_t>* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  out->resize(static_cast<std::size_t>(n));
  if (n > 0 && !f.read(reinterpret_cast<char*>(out->data()), n)) return false;
  return true;
}

bool write_bytes(const std::string& path, const std::vector<std::uint8_t>& b) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  if (!b.empty()) f.write(reinterpret_cast<const char*>(b.data()),
                          static_cast<std::streamsize>(b.size()));
  return static_cast<bool>(f);
}

void le_i32(std::uint8_t* p, std::int32_t v) {
  std::uint32_t u = static_cast<std::uint32_t>(v);
  std::memcpy(p, &u, 4);
}
void le_u64(std::uint8_t* p, std::uint64_t v) { std::memcpy(p, &v, 8); }

// fp16 values are all finite.
int check_finite_fp16(const char* label, const std::uint8_t* bytes, std::size_t elems) {
  const std::uint16_t* h = reinterpret_cast<const std::uint16_t*>(bytes);
  for (std::size_t i = 0; i < elems; ++i) {
    const float v = fp16_bits_to_f32(h[i]);
    if (std::isnan(v) || std::isinf(v)) {
      std::fprintf(stderr, "  %s: non-finite fp16 at %zu\n", label, i);
      return 1;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Utility self-checks (fp16 decode + stage compare)
// ---------------------------------------------------------------------------
int test_fp16_decode() {
  // Bit patterns verified against Python's struct.pack('<e', x).
  struct Case { std::uint16_t bits; double expect; };
  const Case cases[] = {
      {0x0000u, 0.0},
      {0x3C00u, 1.0},
      {0x3C01u, 1.0009765625},
      {0x4000u, 2.0},
      {0x3BFEu, 0.9990234375},
      {0x5640u, 100.0},
      {0x3000u, 0.125},
      {0xC388u, -3.765625},
      {0x0400u, 6.103515625e-5},               // smallest normal, 2^-14
      {0x0001u, 5.960464477539063e-8},         // smallest subnormal, 2^-24
      {0x0011u, 17.0 / 16777216.0},            // subnormal 17 * 2^-24
      {0x4248u, 3.140625},                     // fp16-rounded pi
      {0x5FD0u, 500.0},
      {0x7BFFu, 65504.0},                      // max finite
      {0xFBFFu, -65504.0},
  };
  for (const Case& c : cases) {
    const double got = static_cast<double>(fp16_bits_to_f32(c.bits));
    CHECK_EQ(got, c.expect);
  }
  // signed zero, infinities, NaN
  CHECK_EQ(fp16_bits_to_f32(0x8000u), -0.0f);
  CHECK(std::signbit(fp16_bits_to_f32(0x8000u)));
  CHECK(std::isinf(fp16_bits_to_f32(0x7C00u)) && fp16_bits_to_f32(0x7C00u) > 0);
  CHECK(std::isinf(fp16_bits_to_f32(0xFC00u)) && fp16_bits_to_f32(0xFC00u) < 0);
  CHECK(std::isnan(fp16_bits_to_f32(0x7C01u)));
  CHECK(std::isnan(fp16_bits_to_f32(0x7E00u)));
  return 0;
}

int test_stage_compare() {
  // identical -> ok, zero error
  const std::uint16_t a[] = {0x3C00u, 0x3C00u, 0x4000u};
  StageCompareResult r = compare_fp16_stages(a, a, 3);
  CHECK(r.ok);
  CHECK_NEAR(r.max_abs_err, 0.0, 0.0);

  // one fp16 ulp off -> ok (9.77e-4 <= 1e-2 + 1e-2)
  const std::uint16_t b[] = {0x3C00u, 0x3C01u, 0x4000u};
  r = compare_fp16_stages(a, b, 3);
  CHECK(r.ok);
  CHECK_NEAR(r.max_abs_err, 0.0009765625, 1e-12);

  // 0.5 off -> fail
  const std::uint16_t c[] = {0x3C00u, 0x3E00u, 0x4000u};  // 1.5 vs 1.0
  r = compare_fp16_stages(a, c, 3);
  CHECK(!r.ok);
  CHECK(r.max_abs_idx == 1u);
  CHECK_NEAR(r.max_abs_err, 0.5, 1e-12);

  // NaN in act -> fail even though ref is finite
  const std::uint16_t d[] = {0x3C00u, 0x7C01u, 0x4000u};
  r = compare_fp16_stages(a, d, 3);
  CHECK(!r.ok);

  // zero-length -> trivially ok
  r = compare_fp16_stages(a, a, 0);
  CHECK(r.ok);
  return 0;
}

// ---------------------------------------------------------------------------
// Golden file checks
// ---------------------------------------------------------------------------
int check_golden_common(const GoldenFile& g, int expected_position) {
  CHECK(g.position() == expected_position);
  CHECK(g.num_tensors() == 36u);
  CHECK(g.config() == ModelConfig::v01_default());
  Status s = g.validate_block_tensors();
  CHECK(s.ok);
  if (!s.ok) { std::fprintf(stderr, "  block tensors: %s\n", s.message.c_str()); }
  s = g.validate_golden_tensors();
  CHECK(s.ok);
  if (!s.ok) { std::fprintf(stderr, "  golden tensors: %s\n", s.message.c_str()); }

  // All 18 golden tensors: finite fp16, and every stage nonzero.
  const ModelConfig& c = g.config();
  for (int i = 0; i < kNumStages; ++i) {
    const GoldenTensorInfo* t = g.find(kStageNames[i]);
    CHECK(t != nullptr);
    if (!t) return 1;
    const std::size_t elems =
        static_cast<std::size_t>(t->dims[1]) / 2;  // dims[0] == 1, fp16
    if (check_finite_fp16(kStageNames[i], t->host_bytes, elems)) return 1;
    bool any_nonzero = false;
    const std::uint16_t* h = reinterpret_cast<const std::uint16_t*>(t->host_bytes);
    for (std::size_t k = 0; k < elems; ++k) {
      if (h[k] != 0) { any_nonzero = true; break; }
    }
    CHECK(any_nonzero);  // seeded generator: no all-zero stage
  }
  for (const char* kv : {"kv.k_state", "kv.v_state"}) {
    const GoldenTensorInfo* t = g.find(kv);
    CHECK(t != nullptr);
    if (!t) return 1;
    const std::int64_t expect_rows =
        static_cast<std::int64_t>(c.n_kv_heads) * (expected_position + 1);
    CHECK(t->dims[0] == expect_rows);
    CHECK(t->dims[1] == c.head_dim);
    const std::size_t elems = static_cast<std::size_t>(t->dims[0] * t->dims[1]) / 2;
    if (check_finite_fp16(kv, t->host_bytes, elems)) return 1;
  }
  (void)c;
  return 0;
}

int check_position_zero(const GoldenFile& g) {
  const ModelConfig& c = g.config();
  const int hd = c.head_dim;

  // p=0: RoPE is the exact identity (cos row0 == 1, sin row0 == 0 in fp16)
  // -> stage.rope_q == stage.q bit-for-bit; same for k.
  const GoldenTensorInfo* q = g.find("stage.q");
  const GoldenTensorInfo* rq = g.find("stage.rope_q");
  const GoldenTensorInfo* k = g.find("stage.k");
  const GoldenTensorInfo* rk = g.find("stage.rope_k");
  CHECK(q && rq && k && rk);
  if (!q || !rq || !k || !rk) return 1;
  CHECK(std::memcmp(q->host_bytes, rq->host_bytes, q->byte_size) == 0);
  CHECK(std::memcmp(k->host_bytes, rk->host_bytes, k->byte_size) == 0);

  // p=0: softmax over one element is exactly 1 -> each query head h's
  // attention row is bit-for-bit its GQA kv head's v row.
  const GoldenTensorInfo* attn = g.find("stage.attention_output");
  const GoldenTensorInfo* v = g.find("stage.v");
  CHECK(attn && v);
  if (!attn || !v) return 1;
  for (int h = 0; h < c.n_heads; ++h) {
    const int kh = c.kv_head_for_q(h);
    const std::uint16_t* ah =
        reinterpret_cast<const std::uint16_t*>(attn->host_bytes) + h * hd;
    const std::uint16_t* vh =
        reinterpret_cast<const std::uint16_t*>(v->host_bytes) + kh * hd;
    CHECK(std::memcmp(ah, vh, hd * sizeof(std::uint16_t)) == 0);
  }
  return 0;
}

int check_kv_state_rows(const GoldenFile& g) {
  // For any position: the cache rows at `position` equal stage.rope_k /
  // stage.v bit-for-bit (the generator writes exactly those rows).
  const ModelConfig& c = g.config();
  const int hd = c.head_dim;
  const int nkv = c.n_kv_heads;
  const int p = g.position();

  const GoldenTensorInfo* ks = g.find("kv.k_state");
  const GoldenTensorInfo* vs = g.find("kv.v_state");
  const GoldenTensorInfo* rk = g.find("stage.rope_k");
  const GoldenTensorInfo* v = g.find("stage.v");
  CHECK(ks && vs && rk && v);
  if (!ks || !vs || !rk || !v) return 1;

  // Row order is (kv_head, position): head h's row at position p sits at
  // flat row h * (p + 1) + p.
  const std::size_t pos_row = static_cast<std::size_t>(p) * hd;
  for (int h = 0; h < nkv; ++h) {
    const std::uint16_t* cache_row = reinterpret_cast<const std::uint16_t*>(
        ks->host_bytes) + static_cast<std::size_t>(h) * (p + 1) * hd + pos_row;
    const std::uint16_t* stage_row =
        reinterpret_cast<const std::uint16_t*>(rk->host_bytes) + h * hd;
    CHECK(std::memcmp(cache_row, stage_row, hd * sizeof(std::uint16_t)) == 0);

    const std::uint16_t* v_cache_row = reinterpret_cast<const std::uint16_t*>(
        vs->host_bytes) + static_cast<std::size_t>(h) * (p + 1) * hd + pos_row;
    const std::uint16_t* v_stage_row =
        reinterpret_cast<const std::uint16_t*>(v->host_bytes) + h * hd;
    CHECK(std::memcmp(v_cache_row, v_stage_row, hd * sizeof(std::uint16_t)) == 0);
  }
  (void)c;
  return 0;
}

int check_position_seven(const GoldenFile& g) {
  const ModelConfig& c = g.config();
  const int hd = c.head_dim;

  // p=7: history must be used — attention head 0 differs from its kv
  // head's current v row.
  const GoldenTensorInfo* attn = g.find("stage.attention_output");
  const GoldenTensorInfo* v = g.find("stage.v");
  CHECK(attn && v);
  if (!attn || !v) return 1;
  const std::uint16_t* ah = reinterpret_cast<const std::uint16_t*>(attn->host_bytes);
  const std::uint16_t* vh = reinterpret_cast<const std::uint16_t*>(v->host_bytes);
  CHECK(std::memcmp(ah, vh, hd * sizeof(std::uint16_t)) != 0);

  // KV state: the 8 position rows of each head are not degenerate
  // (adjacent rows differ).
  const GoldenTensorInfo* ks = g.find("kv.k_state");
  CHECK(ks != nullptr);
  if (!ks) return 1;
  const std::uint16_t* base = reinterpret_cast<const std::uint16_t*>(ks->host_bytes);
  for (int h = 0; h < c.n_kv_heads; ++h) {
    const std::uint16_t* row0 = base + static_cast<std::size_t>(h) * 8 * hd;
    const std::uint16_t* row1 = row0 + hd;
    CHECK(std::memcmp(row0, row1, hd * sizeof(std::uint16_t)) != 0);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Negative tests (corrupted golden files)
// ---------------------------------------------------------------------------
int test_negative(const std::string& good_path, const std::string& tmp_path) {
  std::vector<std::uint8_t> raw;
  CHECK(file_bytes(good_path, &raw));
  if (raw.size() < gfmt::kHeaderBytes) return 1;

  struct Case {
    const char* label;
    std::vector<std::uint8_t> (*mutate)(std::vector<std::uint8_t>);
  };
  const Case cases[] = {
      {"bad magic (weight file)",
       [](std::vector<std::uint8_t> b) {
         b[5] = (b[5] == kByteG) ? kByteW : kByteG;  // magics differ at byte 5
         return b;
       }},
      {"bad version",
       [](std::vector<std::uint8_t> b) { b[8] = 2; return b; }},
      {"nonzero flags",
       [](std::vector<std::uint8_t> b) { b[13] = 1; return b; }},
      {"nonzero reserved header field",
       [](std::vector<std::uint8_t> b) { b[gfmt::kReservedOffset + 1] = 7; return b; }},
      {"position out of range",
       [](std::vector<std::uint8_t> b) {
         le_i32(b.data() + gfmt::kPositionOffset, 512);  // == max_seq_len
         return b;
       }},
      {"negative position",
       [](std::vector<std::uint8_t> b) {
         le_i32(b.data() + gfmt::kPositionOffset, -1);
         return b;
       }},
      {"bad table_offset",
       [](std::vector<std::uint8_t> b) {
         le_u64(b.data() + gfmt::kTableOffsetField, 72);
         return b;
       }},
      {"payload offset not aligned",
       [](std::vector<std::uint8_t> b) {
         std::uint64_t po = 0;
         std::memcpy(&po, b.data() + gfmt::kPayloadOffsetField, 8);
         le_u64(b.data() + gfmt::kPayloadOffsetField, po + 1);
         return b;
       }},
  };

  for (const Case& cse : cases) {
    std::vector<std::uint8_t> mutated = cse.mutate(raw);
    CHECK(write_bytes(tmp_path, mutated));
    GoldenFile g;
    Status s = GoldenFile::load(tmp_path, &g);
    CHECK(!s.ok);
    if (s.ok) {
      std::fprintf(stderr, "  accepted corrupted golden (%s)\n", cse.label);
      return 1;
    }
    std::fprintf(stderr, "  ok (rejected '%s'): %s\n", cse.label, s.message.c_str());
  }

  // truncated file (smaller than header)
  std::vector<std::uint8_t> tiny(raw.begin(), raw.begin() + 40);
  CHECK(write_bytes(tmp_path, tiny));
  {
    GoldenFile g;
    Status s = GoldenFile::load(tmp_path, &g);
    CHECK(!s.ok);
    if (s.ok) {
      std::fprintf(stderr, "  accepted truncated golden\n");
      return 1;
    }
  }

  // nonexistent path
  {
    GoldenFile g;
    Status s = GoldenFile::load("/nonexistent/cudalm/golden.cudalm", &g);
    CHECK(!s.ok);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string p0 = argc > 1 ? argv[1] : "data/block_v01_golden_p0.cudalm";
  const std::string p7 = argc > 2 ? argv[2] : "data/block_v01_golden_p7.cudalm";

  if (int rc = test_fp16_decode()) return rc;
  if (int rc = test_stage_compare()) return rc;

  {
    GoldenFile g;
    Status s = GoldenFile::load(p0, &g);
    CHECK(s.ok);
    if (!s.ok) { std::fprintf(stderr, "  load %s: %s\n", p0.c_str(), s.message.c_str()); return 1; }
    if (int rc = check_golden_common(g, 0)) return rc;
    if (int rc = check_position_zero(g)) return rc;
    if (int rc = check_kv_state_rows(g)) return rc;
  }

  {
    GoldenFile g;
    Status s = GoldenFile::load(p7, &g);
    CHECK(s.ok);
    if (!s.ok) { std::fprintf(stderr, "  load %s: %s\n", p7.c_str(), s.message.c_str()); return 1; }
    if (int rc = check_golden_common(g, 7)) return rc;
    if (int rc = check_position_seven(g)) return rc;
    if (int rc = check_kv_state_rows(g)) return rc;
  }

  if (int rc = test_negative(p0, "/tmp/cudalm_test_golden.cudalm")) return rc;
  std::remove("/tmp/cudalm_test_golden.cudalm");

  TEST_PASS("test_golden_file");
}
