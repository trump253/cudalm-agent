// CUDALM — cross-language weight-file test.
//
// Loads a .cudalm file produced by tools/convert_weights.py (Python/torch)
// and validates it entirely on the host, pinning the Python<->C++ contract:
//   * the file parses + passes validate_block_tensors()
//   * INT4 nibbles are in the quantization domain [-7, 7]
//   * group invariants (CUDALab contract):
//       scale > 0  ->  max|q| in the group == 7
//       scale == 0 ->  q == 0 throughout the group
//   * RoPE tables: row 0 is exactly (cos=1, sin=0); cos^2+sin^2 ~= 1 (fp16)
//
// The file must be generated first (ctest wires this as a fixture; manually:
//   python3 tools/convert_weights.py --out data/block_v01.cudalm)

#include "../../tests/common/check.h"

#include "cudalm/weight_loader.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace cudalm;

namespace {

// Decode one IEEE-754 half-precision value to float (host-only, CPU test).
float fp16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1u;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t mant = h & 0x3FFu;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign << 31;  // +-0
    } else {  // subnormal
      std::uint32_t e = 127 - 15 + 1;
      std::uint32_t m = mant;
      while ((m & 0x400u) == 0) { m <<= 1; --e; }
      m &= 0x3FFu;
      bits = (sign << 31) | (e << 23) | (m << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = (sign << 31) | 0x7F800000u | (mant << 13);  // Inf/NaN
  } else {
    bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

const std::uint16_t* as_fp16(const WeightTensorInfo* t) {
  return reinterpret_cast<const std::uint16_t*>(t->host_bytes);
}

// Check one W4A16 (weight, scale) pair against the quantization contract.
int check_proj(const WeightFile& wf, const char* name) {
  // `name` is the projection base ("attn.q_proj"); tensors are
  // "<name>.weight" (INT4_PACKED) and "<name>.scale" (FP16_SCALE).
  const WeightTensorInfo* wrec = wf.find(std::string(name) + ".weight");
  const WeightTensorInfo* srec = wf.find(std::string(name) + ".scale");
  CHECK(wrec != nullptr);
  CHECK(srec != nullptr);
  if (!wrec || !srec) return 1;

  const std::int64_t N = wrec->dims[0];
  const std::int64_t Khalf = wrec->dims[1];
  const std::int64_t K = 2 * Khalf;
  const std::int64_t G = K / 128;
  CHECK_EQ(K % 128, std::int64_t(0));
  CHECK_EQ(srec->dims[0], N);
  CHECK_EQ(srec->dims[1], G);

  const std::uint8_t* wp = wrec->host_bytes;
  const std::uint16_t* sp = as_fp16(srec);

  // Nibble domain over the whole weight, plus per-group invariants.
  for (std::int64_t n = 0; n < N; ++n) {
    for (std::int64_t g = 0; g < G; ++g) {
      const float scale = fp16_to_f32(sp[n * G + g]);
      int max_abs = 0;
      bool all_zero = true;
      for (std::int64_t b = 0; b < 64; ++b) {  // 128 int4 = 64 bytes/group
        const std::uint8_t byte = wp[n * Khalf + g * 64 + b];
        const int lo = int4::unpack_lo(byte);
        const int hi = int4::unpack_hi(byte);
        CHECK(lo >= -7 && lo <= 7);
        CHECK(hi >= -7 && hi <= 7);
        const int a = lo < 0 ? -lo : lo;
        const int c = hi < 0 ? -hi : hi;
        if (a > max_abs) max_abs = a;
        if (c > max_abs) max_abs = c;
        if (lo != 0 || hi != 0) all_zero = false;
      }
      if (scale > 0.0f) {
        if (max_abs != 7) {
          std::fprintf(stderr, "  %s group (%lld,%lld): scale=%.6g max|q|=%d\n",
                       name, (long long)n, (long long)g, scale, max_abs);
        }
        CHECK_EQ(max_abs, 7);
      } else {
        CHECK(all_zero);
      }
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1] : "data/block_v01.cudalm";

  WeightFile wf;
  Status s = WeightFile::load(path, &wf);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "  load: %s\n", s.message.c_str());
    return 1;
  }
  CHECK(wf.config().valid());
  CHECK(wf.validate_block_tensors().ok);

  const ModelConfig& cfg = wf.config();

  // The seven W4A16 projections, in block order.
  const char* projs[] = {
      "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
      "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
  };
  for (const char* p : projs) {
    if (check_proj(wf, p)) return 1;
  }

  // RoPE tables: exact row 0, unit circle elsewhere (fp16 tolerance).
  const WeightTensorInfo* cosr = wf.find("attn.rope_cos");
  const WeightTensorInfo* sinr = wf.find("attn.rope_sin");
  CHECK(cosr != nullptr);
  CHECK(sinr != nullptr);
  if (cosr && sinr) {
    const std::int64_t msl = cosr->dims[0];
    const std::int64_t P = cosr->dims[1];
    CHECK_EQ(msl, std::int64_t(cfg.max_seq_len));
    CHECK_EQ(P, std::int64_t(cfg.head_dim / 2));
    const std::uint16_t* c = as_fp16(cosr);
    const std::uint16_t* sn = as_fp16(sinr);
    for (std::int64_t i = 0; i < P; ++i) {
      CHECK_EQ(fp16_to_f32(c[i]), 1.0f);
      CHECK_EQ(fp16_to_f32(sn[i]), 0.0f);
    }
    for (std::int64_t p = 1; p < msl; ++p) {
      for (std::int64_t i = 0; i < P; ++i) {
        const float cv = fp16_to_f32(c[p * P + i]);
        const float sv = fp16_to_f32(sn[p * P + i]);
        const float resid = std::fabs(cv * cv + sv * sv - 1.0f);
        CHECK_NEAR(resid, 0.0, 1e-2);
      }
    }
  }

  // Sanity: norm weights are finite and nonzero (generator draws ~1 +/- 2%).
  for (const char* nm : {"attn_norm.weight", "ffn_norm.weight"}) {
    const WeightTensorInfo* t = wf.find(nm);
    CHECK(t != nullptr);
    if (!t) return 1;
    const std::uint16_t* v = as_fp16(t);
    bool any_nonzero = false;
    const std::int64_t nelem = static_cast<std::int64_t>(t->byte_size / 2);
    for (std::int64_t i = 0; i < nelem; ++i) {  // fp16 elements
      const float f = fp16_to_f32(v[i]);
      CHECK(std::isfinite(f));
      if (f != 0.0f) any_nonzero = true;
    }
    CHECK(any_nonzero);
  }

  TEST_PASS("weights_crosslang");
  return 0;
}
