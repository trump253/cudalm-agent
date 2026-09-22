// CUDALM — decoder-block latency breakdown benchmark (M9). CUDALM-native.
//
// Measures one DecoderBlock::forward step with CUDA events (one start/stop
// pair per pipeline stage via DecoderBlock::forwardTimed) and writes a JSON
// report with per-stage mean/min/max/p50 (microseconds) plus the total.
//
// Usage:
//   bench_decoder_block <block_v01.cudalm> <position> [iters] [out.json]
//
// It defaults: iters = 100, out.json = "-" (stdout). The KV cache is left
// at its zero-initialized state (benchmark measures kernel/launch cost at
// the given context depth; content does not change the code path).
//
// No PyTorch, no external deps. Provenance: CUDALM-native.

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cudalm/cuda_check.h"
#include "cudalm/decoder_block.h"
#include "cudalm/weight_loader.h"

using namespace cudalm;

namespace {

struct StageStats {
  std::string name;
  float mean_us = 0.f;
  float min_us = 0.f;
  float max_us = 0.f;
  float p50_us = 0.f;
};

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      default: out += c;
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::fprintf(stderr,
                 "usage: %s <block_v01.cudalm> <position> [iters] [out.json]\n",
                 argv[0]);
    return 2;
  }
  const std::string weights_path = argv[1];
  const int position = std::atoi(argv[2]);
  const int iters = argc >= 4 ? std::atoi(argv[3]) : 100;
  const std::string out_path = argc >= 5 ? argv[4] : "-";

  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  if (n < 1) {
    std::fprintf(stderr, "no CUDA device\n");
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  BlockWeights weights;
  Status sb = BlockWeights::load(weights_path, stream, &weights);
  if (!sb.ok) {
    std::fprintf(stderr, "weight load failed: %s\n", sb.message.c_str());
    return 1;
  }
  const ModelConfig& c = weights.config();
  if (position < 0 || position >= c.max_seq_len) {
    std::fprintf(stderr, "position %d out of range [0, %d)\n", position,
                 c.max_seq_len);
    return 1;
  }
  if (iters < 1) {
    std::fprintf(stderr, "iters must be >= 1\n");
    return 1;
  }

  // Deterministic random input (same seed for every run).
  const std::size_t H = static_cast<std::size_t>(c.hidden_size);
  std::vector<__half> x_in(H);
  {
    std::uint32_t seed = 20250922u;
    auto rnd = [&seed]() {
      seed = seed * 1664525u + 1013904223u;  // LCG, deterministic
      return static_cast<float>((seed >> 8) & 0xFFFFFF) /
             static_cast<float>(1u << 24) * 2.0f - 1.0f;
    };
    for (std::size_t i = 0; i < H; ++i) x_in[i] = __float2half_rn(rnd());
  }
  __half* dx;
  CUDA_CHECK(cudaMalloc(&dx, H * sizeof(__half)));
  CUDA_CHECK(cudaMemcpyAsync(dx, x_in.data(), H * sizeof(__half),
                             cudaMemcpyHostToDevice, stream));

  DecoderBlock block(weights, stream);

  // Events: 2 per stage.
  std::vector<cudaEvent_t> events(DecoderBlock::kNumStages * 2);
  for (auto& e : events) CUDA_CHECK(cudaEventCreate(&e));
  std::vector<float> stage_us(DecoderBlock::kNumStages);
  std::vector<std::vector<float>> samples(DecoderBlock::kNumStages,
                                          std::vector<float>(iters));

  const int warmup = 5;
  for (int i = 0; i < warmup; ++i) {
    block.forwardTimed(position, dx, stream, events.data(), stage_us.data());
  }
  for (int i = 0; i < iters; ++i) {
    block.forwardTimed(position, dx, stream, events.data(), stage_us.data());
    for (int s = 0; s < DecoderBlock::kNumStages; ++s)
      samples[s][i] = stage_us[s];
  }

  // Stats per stage.
  std::vector<StageStats> stats(DecoderBlock::kNumStages);
  double total_mean = 0.0, total_min = 1e30, total_max = 0.0;
  std::vector<float> totals(iters);
  for (int i = 0; i < iters; ++i) {
    float t = 0.f;
    for (int s = 0; s < DecoderBlock::kNumStages; ++s) t += samples[s][i];
    totals[i] = t;
  }
  std::sort(totals.begin(), totals.end());
  for (int i = 0; i < iters; ++i) {
    total_mean += totals[i];
    total_min = std::min(total_min, static_cast<double>(totals[i]));
    total_max = std::max(total_max, static_cast<double>(totals[i]));
  }
  total_mean /= iters;

  std::vector<std::string> lines;
  lines.push_back("{");
  lines.push_back("  \"cudalm\": \"v0.1 decoder block latency breakdown\",");
  lines.push_back("  \"device\": \"" + json_escape(prop.name) + "\",");
  {
    lines.push_back("  \"config\": {");
    lines.push_back("    \"hidden_size\": " + std::to_string(c.hidden_size) +
                    ",");
    lines.push_back("    \"n_heads\": " + std::to_string(c.n_heads) + ",");
    lines.push_back("    \"n_kv_heads\": " + std::to_string(c.n_kv_heads) +
                    ",");
    lines.push_back("    \"head_dim\": " + std::to_string(c.head_dim) + ",");
    lines.push_back("    \"intermediate_size\": " +
                    std::to_string(c.intermediate_size) + ",");
    lines.push_back("    \"group_size\": " + std::to_string(c.group_size) +
                    ",");
    lines.push_back("    \"max_seq_len\": " + std::to_string(c.max_seq_len) +
                    "\n  },");
  }
  lines.push_back("  \"position\": " + std::to_string(position) + ",");
  lines.push_back("  \"iters\": " + std::to_string(iters) + ",");
  lines.push_back("  \"warmup\": " + std::to_string(warmup) + ",");
  lines.push_back("  \"units\": \"microseconds (CUDA events)\",");
  lines.push_back("  \"stages\": [");
  const char* const* names = DecoderBlock::stage_names();
  for (int s = 0; s < DecoderBlock::kNumStages; ++s) {
    auto v = samples[s];
    std::sort(v.begin(), v.end());
    double mean = 0;
    for (float f : v) mean += f;
    mean /= v.size();
    stats[s].name = names[s];
    stats[s].mean_us = static_cast<float>(mean);
    stats[s].min_us = v.front();
    stats[s].max_us = v.back();
    stats[s].p50_us = v[v.size() / 2];
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "    {\"stage\": \"%s\", \"mean_us\": %.3f, \"min_us\": "
                  "%.3f, \"max_us\": %.3f, \"p50_us\": %.3f}%s",
                  stats[s].name.c_str(), stats[s].mean_us, stats[s].min_us,
                  stats[s].max_us, stats[s].p50_us,
                  s + 1 < DecoderBlock::kNumStages ? "," : "");
    lines.push_back(buf);
  }
  lines.push_back("  ],");
  {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "  \"total\": {\"mean_us\": %.3f, \"min_us\": %.3f, "
                  "\"max_us\": %.3f, \"p50_us\": %.3f},",
                  total_mean, total_min, total_max,
                  static_cast<double>(totals[iters / 2]));
    lines.push_back(buf);
  }
  {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "  \"tokens_per_second_mean\": %.1f",
                  1e6 / total_mean);
    lines.push_back(buf);
  }
  lines.push_back("}");

  std::string text;
  for (auto& l : lines) text += l + "\n";
  if (out_path == "-") {
    std::printf("%s", text.c_str());
  } else {
    std::ofstream f(out_path);
    if (!f) {
      std::fprintf(stderr, "cannot open %s\n", out_path.c_str());
      return 1;
    }
    f << text;
    std::fprintf(stderr, "wrote %s\n", out_path.c_str());
  }

  for (auto& e : events) CUDA_CHECK(cudaEventDestroy(e));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
