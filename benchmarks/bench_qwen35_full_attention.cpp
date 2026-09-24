// CUDALM — Qwen3.5 full-attention layer latency benchmark (v0.2 Phase B,
// schema mirrors the v0.1.1 bench_decoder_block report). CUDALM-native.
//
// Measures one Qwen35FullAttentionLayer::forward step (Qwen3.5-0.8B layer 3:
// H=1024, 8 Q / 2 KV heads, head_dim=256, SwiGLU 3584, bf16 + W4A16 G=128)
// with CUDA events via forwardTimed and writes a JSON report with three
// timing views (all microseconds):
//   * stages:            one start/stop event pair per pipeline stage (21)
//   * stage_sum_us:      sum of the 21 per-stage event durations per iter;
//                        EXCLUDES GPU work no stage covers (the RoPE-table
//                        H2D copy enqueued between the input and rmsnorm
//                        stages)
//   * whole_layer_gpu_us: one CUDA-event pair around the ENTIRE forward —
//                        a DEVICE-TIMELINE interval. It spans the GPU work
//                        enqueued within it (incl. the inter-stage RoPE-table
//                        H2D copy) and any device idle while the host
//                        enqueues work (e.g. host-side RoPE-table build).
//                        NOT a direct measurement of host CPU work.
//   * host_api_wall_us:  CPU wall-clock around the forwardTimed() call
//                        (includes the stream sync); this is where the
//                        actual CPU host time is attributed.
// One decode step of ONE decoder layer is NOT a model token: the rate
// metric is layer_steps_per_second_mean, based on whole_layer_gpu_us.
//
// Usage:
//   bench_qwen35_full_attention <qwen35_08b_l3.cudalm> <position> [iters]
//       [out.json]
//
// Defaults: iters = 100, out.json = "-" (stdout). The KV cache is left at
// its zero-initialized state (the benchmark measures kernel/launch cost at
// the given context depth; content does not change the code path).
//
// No PyTorch, no external deps. Provenance: CUDALM-native.

#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "cudalm/cuda_check.h"
#include "cudalm/qwen35_full_attention.h"
#include "cudalm/weight_loader_v2.h"

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
                 "usage: %s <qwen35_08b_l3.cudalm> <position> [iters] "
                 "[out.json]\n",
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

  WeightFileV2 file;
  Status sf = WeightFileV2::load(weights_path, &file);
  if (!sf.ok) {
    std::fprintf(stderr, "weight load failed: %s\n", sf.message.c_str());
    return 1;
  }
  const Qwen35Config& c = file.config();
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
  std::vector<__nv_bfloat16> x_in(H);
  {
    std::uint32_t seed = 20260209u;
    auto rnd = [&seed]() {
      seed = seed * 1664525u + 1013904223u;  // LCG, deterministic
      return static_cast<float>((seed >> 8) & 0xFFFFFF) /
             static_cast<float>(1u << 24) * 2.0f - 1.0f;
    };
    for (auto& v : x_in) v = __float2bfloat16_rn(0.05f * rnd());
  }
  DeviceBuffer d_in(x_in.size() * sizeof(__nv_bfloat16), stream);
  d_in.copy_from_host(x_in.data(), d_in.bytes(), stream);

  Qwen35LayerWeights weights;
  Status sw = Qwen35LayerWeights::load(file, 3, stream, &weights);
  if (!sw.ok) {
    std::fprintf(stderr, "layer 3 weight load failed: %s\n",
                 sw.message.c_str());
    return 1;
  }
  if (!weights.is_full_attention()) {
    std::fprintf(stderr, "layer 3 is not a full-attention layer\n");
    return 1;
  }
  Qwen35FullAttentionLayer layer(weights, stream);

  // Timing event array (kNumTimingEvents caller-created events).
  constexpr int kEvents = Qwen35FullAttentionLayer::kNumTimingEvents;
  std::vector<cudaEvent_t> events(kEvents);
  for (auto& e : events) CUDA_CHECK(cudaEventCreate(&e));
  std::vector<float> stage_us(Qwen35FullAttentionLayer::kNumStages);
  float whole_us = 0.f;

  const int warmup = 5;
  for (int i = 0; i < warmup; ++i) {
    layer.forwardTimed(position, d_in.data<__nv_bfloat16>(), stream,
                       events.data(), stage_us.data(), &whole_us);
  }

  constexpr int kNumStages = Qwen35FullAttentionLayer::kNumStages;
  std::vector<std::vector<float>> per_stage(kNumStages);
  std::vector<float> whole_hist;
  std::vector<double> wall_hist;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    layer.forwardTimed(position, d_in.data<__nv_bfloat16>(), stream,
                       events.data(), stage_us.data(), &whole_us);
    const auto t1 = std::chrono::steady_clock::now();
    wall_hist.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
    whole_hist.push_back(whole_us);
    for (int s = 0; s < Qwen35FullAttentionLayer::kNumStages; ++s)
      per_stage[s].push_back(stage_us[s]);
  }

  auto summarize = [](std::vector<float>& v) {
    StageStats st;
    const float sum = std::accumulate(v.begin(), v.end(), 0.f);
    st.mean_us = sum / static_cast<float>(v.size());
    st.min_us = *std::min_element(v.begin(), v.end());
    st.max_us = *std::max_element(v.begin(), v.end());
    std::vector<float> sorted = v;
    std::sort(sorted.begin(), sorted.end());
    st.p50_us = sorted[sorted.size() / 2];
    return st;
  };
  const char* const* names = Qwen35FullAttentionLayer::stage_names();
  std::string stages_json;
  double stage_sum_mean = 0.0, stage_sum_min = 1e30, stage_sum_max = 0.0;
  for (int s = 0; s < Qwen35FullAttentionLayer::kNumStages; ++s) {
    StageStats st = summarize(per_stage[s]);
    st.name = names[s];
    if (!stages_json.empty()) stages_json += ",\n";
    stages_json += "    {\"stage\": \"" + json_escape(st.name) +
                   "\", \"mean_us\": " + std::to_string(st.mean_us) +
                   ", \"min_us\": " + std::to_string(st.min_us) +
                   ", \"max_us\": " + std::to_string(st.max_us) +
                   ", \"p50_us\": " + std::to_string(st.p50_us) + "}";
    double sum = 0.0;
    for (float v : per_stage[s]) sum += v;
    stage_sum_mean += sum / iters;
    // per-iter stage sums for min/max
    (void)sum;
  }
  // Recompute per-iter stage sums for min/max (the stage event pairs are
  // non-overlapping by construction, so the per-iter sum is well defined).
  for (int i = 0; i < iters; ++i) {
    double sum = 0.0;
    for (int s = 0; s < Qwen35FullAttentionLayer::kNumStages; ++s)
      sum += per_stage[s][i];
    stage_sum_min = std::min(stage_sum_min, sum);
    stage_sum_max = std::max(stage_sum_max, sum);
  }
  stage_sum_mean /= 1.0;  // already averaged per stage above

  const StageStats whole = summarize(whole_hist);
  std::vector<float> wallf(wall_hist.begin(), wall_hist.end());
  const StageStats wall = summarize(wallf);

  std::string out;
  out += "{\n";
  out += "  \"cudalm\": \"v0.2 Phase B Qwen3.5 full-attention layer 3 "
         "latency breakdown\",\n";
  out += "  \"device\": \"" + std::string(prop.name) + "\",\n";
  out += "  \"config\": {\n";
  out += "    \"hidden_size\": " + std::to_string(c.hidden_size) + ",\n";
  out += "    \"n_heads\": " + std::to_string(c.n_heads) + ",\n";
  out += "    \"n_kv_heads\": " + std::to_string(c.n_kv_heads) + ",\n";
  out += "    \"head_dim\": " + std::to_string(c.head_dim) + ",\n";
  out += "    \"intermediate_size\": " + std::to_string(c.intermediate_size) +
         ",\n";
  out += "    \"group_size\": " + std::to_string(c.group_size) + ",\n";
  out += "    \"activation_dtype\": \"bf16\",\n";
  out += "    \"max_seq_len\": " + std::to_string(c.max_seq_len) + "\n";
  out += "  },\n";
  out += "  \"position\": " + std::to_string(position) + ",\n";
  out += "  \"iters\": " + std::to_string(iters) + ",\n";
  out += "  \"warmup\": " + std::to_string(warmup) + ",\n";
  out += "  \"units\": \"microseconds (stages + whole_layer_gpu: CUDA "
         "events; host_api_wall: CPU wall-clock)\",\n";
  out += "  \"stages\": [\n" + stages_json + "\n  ],\n";
  out += "  \"stage_sum_us\": {\"mean_us\": " + std::to_string(stage_sum_mean) +
         ", \"min_us\": " + std::to_string(stage_sum_min) +
         ", \"max_us\": " + std::to_string(stage_sum_max) + "},\n";
  out += "  \"whole_layer_gpu_us\": {\"mean_us\": " +
         std::to_string(whole.mean_us) + ", \"min_us\": " +
         std::to_string(whole.min_us) + ", \"max_us\": " +
         std::to_string(whole.max_us) + ", \"p50_us\": " +
         std::to_string(whole.p50_us) + "},\n";
  out += "  \"host_api_wall_us\": {\"mean_us\": " + std::to_string(wall.mean_us) +
         ", \"min_us\": " + std::to_string(wall.min_us) +
         ", \"max_us\": " + std::to_string(wall.max_us) + ", \"p50_us\": " +
         std::to_string(wall.p50_us) + "},\n";
  out += "  \"timing_semantics\": {\n";
  out += "    \"stage_sum_us\": \"sum of the 21 per-stage CUDA-event "
         "durations; excludes GPU work no stage covers (RoPE-table H2D "
         "copy), NOT a whole-layer latency\",\n";
  out += "    \"whole_layer_gpu_us\": \"one CUDA-event pair around the "
         "entire forward; a device-timeline interval covering the GPU work "
         "enqueued within it (incl. the inter-stage RoPE-table H2D copy) "
         "plus any device idle while the host enqueues work; NOT a direct "
         "measurement of host CPU work (see host_api_wall_us)\",\n";
  out += "    \"host_api_wall_us\": \"CPU wall-clock around the forwardTimed() "
         "call, including stream synchronization\"\n";
  out += "  },\n";
  out += "  \"layer_steps_per_second_mean\": " +
         std::to_string(1e6 / whole.mean_us) + "\n";
  out += "}\n";

  if (out_path == "-") {
    std::printf("%s", out.c_str());
  } else {
    std::ofstream f(out_path);
    f << out;
    std::fprintf(stderr, "wrote %s\n", out_path.c_str());
  }
  for (auto& e : events) CUDA_CHECK(cudaEventDestroy(e));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
