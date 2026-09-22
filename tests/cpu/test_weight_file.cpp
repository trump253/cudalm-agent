// CUDALM — weight-file parser/loader CPU test.
//
// Builds a fully-valid decoder-block .cudalm file in memory with a reference
// writer (mirroring docs/weight_format.md), round-trips it through
// WeightFile::load, then exercises the negative/bounds path with targeted
// corruptions. Also pins the INT4 pack/unpack contract over the full domain.

#include "../../tests/common/check.h"

#include "cudalm/weight_loader.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

using namespace cudalm;
using wfmt::kHeaderBytes;
using wfmt::kPayloadAlign;

namespace {

// Deterministic pseudo-random byte stream (not cryptographic).
std::uint8_t prb(std::size_t i) {
  std::uint64_t x = (std::uint64_t)(i + 1) * 0x9E3779B97F4A7C15ull;
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDull;
  x ^= x >> 33;
  return static_cast<std::uint8_t>(x);
}

struct Rec {
  const char* name;
  Dtype dtype;
  std::vector<std::int64_t> dims;
};

// Reference writer: header + packed table + 16B-aligned payload.
std::vector<std::uint8_t> build_file(const ModelConfig& cfg,
                                     const std::vector<Rec>& recs) {
  // Lay out the payload first to know its size.
  std::vector<std::vector<std::uint8_t>> blobs(recs.size());
  std::uint64_t off = 0;
  for (std::size_t i = 0; i < recs.size(); ++i) {
    std::uint64_t sz = wfmt::expected_bytes(recs[i].dtype, recs[i].dims);
    // 16B-align the region start (payload start is 16B aligned by construction
    // of the header size; keep every region 16B aligned).
    while (off % kPayloadAlign != 0) { off += 1; }
    blobs[i].resize(sz);
    for (std::uint64_t b = 0; b < sz; ++b) blobs[i][b] = prb(i * 100000 + b);
    (void)off;
    off += sz;
  }
  // Build the table (need per-record offset; recompute deterministically).
  std::vector<std::uint64_t> offsets(recs.size());
  off = 0;
  for (std::size_t i = 0; i < recs.size(); ++i) {
    while (off % kPayloadAlign != 0) off += 1;
    offsets[i] = off;
    off += blobs[i].size();
  }
  const std::uint64_t payload_size = off;

  std::vector<std::uint8_t> table;
  auto wr32 = [&](std::uint32_t v) {
    for (int k = 0; k < 4; ++k) table.push_back((v >> (8 * k)) & 0xFF);  // LE
  };
  auto wr64 = [&](std::uint64_t v) {
    for (int k = 0; k < 8; ++k) table.push_back((v >> (8 * k)) & 0xFF);  // LE
  };
  for (std::size_t i = 0; i < recs.size(); ++i) {
    const auto& r = recs[i];
    wr32(static_cast<std::uint32_t>(std::strlen(r.name)));
    for (std::uint32_t c = 0; c < std::strlen(r.name); ++c) table.push_back(r.name[c]);
    table.push_back(static_cast<std::uint8_t>(r.dtype));
    table.push_back(static_cast<std::uint8_t>(r.dims.size()));
    for (auto d : r.dims) wr32(static_cast<std::uint32_t>(d));
    wr64(offsets[i]);
    wr64(blobs[i].size());
    table.push_back(16);  // align
    for (int r = 0; r < 7; ++r) table.push_back(0);  // pad
  }

  const std::uint64_t table_off = kHeaderBytes;
  std::uint64_t payload_off = table_off + table.size();
  while (payload_off % kPayloadAlign != 0) payload_off += 1;  // 16B-align

  std::vector<std::uint8_t> out;
  out.reserve(payload_off + payload_size);
  // Header
  for (int i = 0; i < 8; ++i) out.push_back(wfmt::kMagic[i]);
  for (int k = 0; k < 4; ++k) out.push_back((wfmt::kVersion >> (8 * k)) & 0xFF);
  for (int k = 0; k < 4; ++k) out.push_back(0);  // flags
  std::uint32_t nt = static_cast<std::uint32_t>(recs.size());
  for (int k = 0; k < 4; ++k) out.push_back((nt >> (8 * k)) & 0xFF);
  std::uint8_t cb[wfmt::kConfigBytes];
  wfmt::config_to_blob(cfg, cb);
  out.insert(out.end(), cb, cb + wfmt::kConfigBytes);
  for (int k = 0; k < 8; ++k) out.push_back((table_off >> (8 * k)) & 0xFF);
  for (int k = 0; k < 8; ++k) out.push_back((payload_off >> (8 * k)) & 0xFF);
  out.insert(out.end(), table.begin(), table.end());
  // Zero-fill the gap between table end and the 16B-aligned payload start.
  while (out.size() < payload_off) out.push_back(0);
  // Payload
  std::uint64_t cur = 0;
  for (std::size_t i = 0; i < blobs.size(); ++i) {
    while (cur % kPayloadAlign != 0) { out.push_back(0); cur += 1; }
    out.insert(out.end(), blobs[i].begin(), blobs[i].end());
    cur += blobs[i].size();
  }
  return out;
}

std::vector<Rec> block_recs(const ModelConfig& c) {
  const int H = c.hidden_size, hd = c.head_dim, nkv = c.n_kv_heads, inter = c.intermediate_size;
  return {
      {"attn_norm.weight",  Dtype::kFp16,        {H}},
      {"ffn_norm.weight",   Dtype::kFp16,        {H}},
      {"attn.rope_cos",     Dtype::kFp16,        {c.max_seq_len, hd / 2}},
      {"attn.rope_sin",     Dtype::kFp16,        {c.max_seq_len, hd / 2}},
      {"attn.q_proj.weight", Dtype::kInt4Packed, {H, H / 2}},
      {"attn.q_proj.scale",  Dtype::kFp16Scale,  {H, H / 128}},
      {"attn.k_proj.weight", Dtype::kInt4Packed, {nkv * hd, H / 2}},
      {"attn.k_proj.scale",  Dtype::kFp16Scale,  {nkv * hd, H / 128}},
      {"attn.v_proj.weight", Dtype::kInt4Packed, {nkv * hd, H / 2}},
      {"attn.v_proj.scale",  Dtype::kFp16Scale,  {nkv * hd, H / 128}},
      {"attn.o_proj.weight", Dtype::kInt4Packed, {H, H / 2}},
      {"attn.o_proj.scale",  Dtype::kFp16Scale,  {H, H / 128}},
      {"mlp.gate_proj.weight", Dtype::kInt4Packed, {inter, H / 2}},
      {"mlp.gate_proj.scale",  Dtype::kFp16Scale,  {inter, H / 128}},
      {"mlp.up_proj.weight",   Dtype::kInt4Packed, {inter, H / 2}},
      {"mlp.up_proj.scale",    Dtype::kFp16Scale,  {inter, H / 128}},
      {"mlp.down_proj.weight", Dtype::kInt4Packed, {H, inter / 2}},
      {"mlp.down_proj.scale",  Dtype::kFp16Scale,  {H, inter / 128}},
  };
}

void write_tmp(const char* path, const std::vector<std::uint8_t>& data) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data.data()), data.size());
}

}  // namespace

int main() {
  const char* path = "/tmp/cudalm_test_weights.cudalm";
  ModelConfig cfg = ModelConfig::v01_default();
  // Use a smaller config to keep the payload tiny; group stays 128.
  cfg.hidden_size = 128; cfg.n_heads = 2; cfg.n_kv_heads = 1; cfg.head_dim = 64;
  cfg.intermediate_size = 256; cfg.max_seq_len = 16;
  CHECK(cfg.valid());

  const std::vector<Rec> recs = block_recs(cfg);
  std::vector<std::uint8_t> good = build_file(cfg, recs);

  // ---- positive: full round-trip ----------------------------------------
  write_tmp(path, good);
  WeightFile wf;
  Status s = WeightFile::load(path, &wf);
  CHECK(s.ok);
  if (!s.ok) { std::fprintf(stderr, "  load: %s\n", s.message.c_str()); return 1; }
  CHECK_EQ(wf.num_tensors(), recs.size());
  CHECK(wf.config() == cfg);
  for (std::size_t i = 0; i < recs.size(); ++i) {
    const WeightTensorInfo* t = wf.find(recs[i].name);
    CHECK(t != nullptr);
    if (!t) return 1;
    CHECK(t->dtype == recs[i].dtype);
    CHECK(t->dims == recs[i].dims);
    std::uint64_t sz = wfmt::expected_bytes(recs[i].dtype, recs[i].dims);
    CHECK_EQ(t->byte_size, sz);
    // Content must match the deterministic prb fill.
    bool ok = true;
    for (std::uint64_t b = 0; b < sz; ++b)
      if (t->host_bytes[b] != prb(i * 100000 + b)) { ok = false; break; }
    CHECK(ok);
  }
  CHECK(wf.validate_block_tensors().ok);

  // ---- int4 pack/unpack contract over the full domain --------------------
  for (int lo = -8; lo <= 7; ++lo) {
    for (int hi = -8; hi <= 7; ++hi) {
      std::uint8_t b = int4::pack_byte(lo, hi);
      CHECK_EQ(int4::unpack_lo(b), lo);
      CHECK_EQ(int4::unpack_hi(b), hi);
    }
  }
  // Quantization domain [-7,7] subset (what the kernel actually sees).
  for (int v = -7; v <= 7; ++v) {
    std::uint8_t b = int4::pack_byte(v, v);
    CHECK_EQ(int4::unpack_lo(b), v);
    CHECK_EQ(int4::unpack_hi(b), v);
  }

  // ---- negative cases ----------------------------------------------------
  // NOTE: explicit `-> int` and a trailing `return 0;`. CHECK() expands to
  // `return 1;` inside a do-while; with the end-of-function path the deduced
  // return type would be int with a missing return = UB (harmless at -O0,
  // miscompiles at -O2).
  auto expect_fail = [&](const char* label, std::vector<std::uint8_t> data) -> int {
    write_tmp(path, data);
    WeightFile bad;
    Status bs = WeightFile::load(path, &bad);
    if (bs.ok) {
      std::fprintf(stderr, "  expected load failure for: %s\n", label);
    }
    CHECK(!bs.ok);
    return 0;
  };
  auto run_fail = [&](const char* label, std::vector<std::uint8_t> data) -> int {
    int rc = expect_fail(label, std::move(data));
    if (rc != 0) {
      std::fprintf(stderr, "  negative case failed: %s\n", label);
      return 1;
    }
    return 0;
  };

  // bad magic
  if (run_fail("magic", [&]{ auto d = good; d[0] = 'X'; return d; }())) return 1;
  // unsupported version
  if (run_fail("version", [&]{ auto d = good; d[8] = 2; return d; }())) return 1;
  // nonzero flags
  if (run_fail("flags", [&]{ auto d = good; d[12] = 1; return d; }())) return 1;
  // truncated file (smaller than header)
  if (run_fail("truncated-header", [&]{ auto d = good; d.resize(40); return d; }())) return 1;
  // truncated in payload (byte_size exceeds available)
  if (run_fail("truncated-payload", [&]{ auto d = good; d.resize(d.size() - 1); return d; }())) return 1;
  // duplicate tensor name: copy record 0 into record 1's slot by rewriting
  // is complex; instead craft a minimal 2-record file with same name.
  if (run_fail("duplicate-name", [&]{
          std::vector<Rec> r2 = {
              {"dup.weight", Dtype::kFp16, {128}},
              {"dup.weight", Dtype::kFp16, {128}},
          };
          return build_file(cfg, r2);
      }())) return 1;
  // bad dtype (4 = invalid)
  if (run_fail("bad-dtype", [&]{
          std::vector<Rec> r2 = {{"bad.dtype", Dtype::kFp32, {128}}};  // kFp32=4 not allowed
          return build_file(cfg, r2);
      }())) return 1;
  // payload_offset corrupted so payload regions run out of bounds
  if (run_fail("payload-offset-oor", [&]{ auto d = good; d[64] = 0xFF; return d; }())) return 1;
  // misaligned payload region: shift payload_offset by 1 (breaks 16B align)
  if (run_fail("misaligned", [&]{
          auto d = good;
          // payload_offset is at 64..71. Increment by 1 (LE).
          for (int k = 0; k < 8; ++k) {
            if (d[64 + k] == 0xFF) d[64 + k] = 0; else { d[64 + k] += 1; break; }
          }
          // table_offset must still be 72; payload moves by 1 -> region misaligned
          return d;
        }())) return 1;
  // reserved/pad nonzero: flip a pad byte in the first record.
  // First record layout (1-D, name "attn_norm.weight" = 16 chars):
  //   name_len(4) + name(16) + dtype(1) + ndim(1) + dims(4*1=4)
  //   + offset(8) + byte_size(8) + align(1) = 43 bytes,
  // so the 7 pad bytes start at table_off + 43 = 72 + 43 = 115.
  if (run_fail("pad-nonzero", [&]{ auto d = good; d[115] = 1; return d; }())) return 1;

  std::remove(path);
  TEST_PASS("weight_file_cpu");
  return 0;
}
