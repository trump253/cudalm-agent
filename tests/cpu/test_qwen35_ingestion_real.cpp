// CUDALM — real-checkpoint ingestion test (CPU; the one test that runs the
// full Python toolchain against the official pinned checkpoint).
//
// Pipeline (all offline):
//   1. tools/convert_qwen35.py   checkpoint -> .cudalm v2 (micro-stack
//      layers 0,1,2 Gated DeltaNet + 3 Full-Attention + model norm)
//   2. tools/verify_qwen35_ingestion.py  v2 file vs safetensors (bf16
//      bit-exact; W4A16 dequant bound; scale == fp16(amax/7); provenance)
//   3. this binary: C++ parse of the produced file (WeightFileV2), pinned
//      config + metadata, per-layer tensor-set validation, and a sidecar
//      cross-check (byte sizes + fp32 sums of every bf16/fp32 payload).
//
// Skips with exit 77 when the checkpoint is absent (the checkpoint lives in
// /root/models, never in the repo).

#include "../../tests/common/check.h"

#include "cudalm/qwen35_config.h"
#include "cudalm/weight_loader_v2.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace cudalm;

// Pinned provenance (docs/qwen35_architecture.md §1).
namespace pin {
constexpr const char* kRepo = "Qwen/Qwen3.5-0.8B-Base";
constexpr const char* kRevision = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68";
constexpr const char* kConfigSha =
    "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204";
constexpr const char* kCheckpointSha =
    "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c";
constexpr const char* kArch = "qwen3.5-text";
}  // namespace

namespace {

bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return static_cast<bool>(f);
}

int run_cmd(const std::string& cmd) {
  std::fprintf(stderr, "  $ %s\n", cmd.c_str());
  const int rc = std::system(cmd.c_str());
  if (rc != 0) std::fprintf(stderr, "  command failed (rc=%d)\n", rc);
  return rc;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::vector<std::uint8_t> out;
  std::ifstream f(path, std::ios::binary);
  if (!f) return out;
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  out.resize(static_cast<std::size_t>(n));
  if (n > 0) f.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

float fp32_payload_sum_f32(const std::uint8_t* p, std::uint64_t bytes) {
  float acc = 0.0f;
  for (std::uint64_t i = 0; i + 4 <= bytes; i += 4) {
    float v;
    std::memcpy(&v, p + i, 4);
    acc += v;
  }
  return acc;
}

float bf16_payload_sum_f32(const std::uint8_t* p, std::uint64_t bytes) {
  float acc = 0.0f;
  for (std::uint64_t i = 0; i + 1 < bytes; i += 2) {
    const std::uint16_t bits =
        static_cast<std::uint16_t>(p[i] | (static_cast<std::uint16_t>(p[i + 1]) << 8));
    const std::uint32_t fbits = static_cast<std::uint32_t>(bits) << 16;
    float v;
    std::memcpy(&v, &fbits, 4);
    acc += v;
  }
  return acc;
}

int test_cpp_side(const std::string& cudalm_path, const std::string& sidecar) {
  WeightFileV2 file;
  Status s = WeightFileV2::load(cudalm_path, &file);
  CHECK(s.ok);
  if (!s.ok) { std::fprintf(stderr, "load error: %s\n", s.message.c_str()); return 1; }

  // Pinned config, byte-for-byte from the Python writer.
  CHECK(file.config() == Qwen35Config::qwen35_08b());

  // Pinned provenance metadata.
  CHECK(file.meta("arch") != nullptr);
  CHECK_EQ(*file.meta("arch"), pin::kArch);
  CHECK(file.meta("model_repo") != nullptr);
  CHECK_EQ(*file.meta("model_repo"), pin::kRepo);
  CHECK(file.meta("model_revision") != nullptr);
  CHECK_EQ(*file.meta("model_revision"), pin::kRevision);
  CHECK(file.meta("config_sha256") != nullptr);
  CHECK_EQ(*file.meta("config_sha256"), pin::kConfigSha);
  CHECK(file.meta("checkpoint_sha256") != nullptr);
  CHECK_EQ(*file.meta("checkpoint_sha256"), pin::kCheckpointSha);

  // Micro-stack layer sets: 0,1,2 DeltaNet + 3 Full-Attention + model norm.
  CHECK(file.validate_layer(0).ok);
  CHECK(file.validate_layer(1).ok);
  CHECK(file.validate_layer(2).ok);
  CHECK(file.validate_layer(3).ok);
  CHECK(file.validate_model_norm().ok);
  // 3 DN layers x 22 tensors + 18 (FA) + 1 (model norm) = 85.
  CHECK_EQ(file.num_tensors(), 85u);

  // Sidecar cross-check (values written by the Python verifier).
  std::map<std::string, std::pair<std::uint64_t, float>> recs;
  {
    const std::vector<std::uint8_t> raw = read_file(sidecar);
    std::istringstream ss(std::string(raw.begin(), raw.end()));
    std::string line;
    while (std::getline(ss, line)) {
      if (line.empty()) continue;
      const std::size_t t1 = line.find('\t');
      const std::size_t t2 = line.find('\t', t1 + 1);
      CHECK(t1 != std::string::npos && t2 != std::string::npos);
      const std::string name = line.substr(0, t1);
      const std::uint64_t bytes = std::stoull(line.substr(t1 + 1, t2 - t1 - 1));
      const std::string sum_str = line.substr(t2 + 1);
      const float sum = sum_str == "-1" ? -1.0f : std::strtof(sum_str.c_str(), nullptr);
      recs[name] = {bytes, sum};
    }
  }
  CHECK_EQ(recs.size(), file.num_tensors());
  for (const auto& t : file.tensors()) {
    const auto it = recs.find(t.name);
    CHECK(it != recs.end());
    if (it == recs.end()) continue;
    CHECK_EQ(t.byte_size, it->second.first);
    if (it->second.second >= 0.0f) {  // bf16/fp32 tensor: compare fp32 sums
      const float got = (t.dtype == Dtype::kFp32)
                            ? fp32_payload_sum_f32(t.host_bytes, t.byte_size)
                            : bf16_payload_sum_f32(t.host_bytes, t.byte_size);
      CHECK_NEAR(got, it->second.second, 1e-3);
    }
  }
  TEST_PASS("qwen35_ingestion_real cpp side");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <out.cudalm> <checkpoint_dir> <python> <src_dir>\n",
                 argv[0]);
    return 2;
  }
  const std::string out = argv[1];
  const std::string ckpt = argv[2];
  const std::string py = argv[3];
  const std::string src = argv[4];

  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr,
                 "[SKIP] qwen35 ingestion: checkpoint not present at %s\n",
                 ckpt.c_str());
    return 77;
  }

  const std::string sidecar = out + ".sidecar";
  const std::string manifest = out + ".manifest.json";
  const std::string report = out + ".report.json";

  // 1. Convert (Python toolchain, offline).
  int rc = run_cmd(py + " " + src + "/tools/convert_qwen35.py" +
                   " --checkpoint-dir " + ckpt + " --out " + out +
                   " --layers 0,1,2,3 --manifest " + manifest);
  CHECK_EQ(rc, 0);

  // 2. Verify against the checkpoint (Python side).
  rc = run_cmd(py + " " + src + "/tools/verify_qwen35_ingestion.py" +
               " --cudalm " + out + " --checkpoint-dir " + ckpt +
               " --layers 0,1,2,3 --report " + report + " --sidecar " + sidecar);
  CHECK_EQ(rc, 0);

  // 3. C++ parse + pinned checks.
  rc = test_cpp_side(out, sidecar);
  if (rc != 0) std::fprintf(stderr, "test_qwen35_ingestion_real FAILED\n");
  return rc;
}
