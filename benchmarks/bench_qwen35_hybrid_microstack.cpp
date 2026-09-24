// CUDALM — Qwen3.5 4-layer hybrid decoder micro-stack baseline latency
// benchmark (v0.2, Phase D). CUDALM-native. Unoptimized baseline only: no
// NCU, no kernel fusion, no CUDA Graphs. Mirrors the v0.1.1 / Phase B
// timing-report schema.
//
// Measures ONE Qwen35HybridMicroStack decode step (the real checkpoint layers
// 0-3: 3x Gated DeltaNet + 1x full attention, chained 0 -> 1 -> 2 -> 3) with
// CUDA events via forwardTimed and writes a JSON report with three timing
// views (all microseconds), for TWO cases:
//   * p=0             : fresh (zero) persistent state; the full-attention
//                       layer attends over 1 KV row.
//   * p=<deep> (e.g. 512): sequential state — the runtime first decodes
//                       positions 0..deep-1 (un-timed warmup) so every layer's
//                       persistent state is populated, then the step is timed
//                       at position deep (the full-attention layer attends over
//                       deep+1 KV rows).
//
// Timing views per case:
//   * stage_layer_sum_us:   sum of the 4 per-LAYER CUDA-event durations
//                           (the per-layer pipeline intervals; the 4 layers
//                           are non-overlapping by construction).
//   * whole_microstack_gpu_us: one CUDA-event pair around the ENTIRE 4-layer
//                           forward — a DEVICE-TIMELINE interval spanning the
//                           GPU work enqueued within it (all four layers) plus
//                           any device idle while the host enqueues; NOT a
//                           direct measurement of host CPU work.
//   * host_api_wall_us:     CPU wall-clock around the forwardTimed() call
//                           (includes the stream sync); where the actual CPU
//                           host time is attributed.
//   * per_layer_gpu_us:     the 4 individual per-layer CUDA-event durations
//                           (mean/min/max/p50 each).
//
// A micro-stack decode step runs four layers but is NOT a full-model token
// (the model has 24 layers). The rate metric is
// microstack_steps_per_second_mean, based on whole_microstack_gpu_us — never
// "tokens/s".
//
// Usage:
//   bench_qwen35_hybrid_microstack <out_ms.cudalm> [iters] [deep_pos] [out.json]
// Defaults: iters = 100, deep_pos = 512, out.json = "-" (stdout).
//
// No PyTorch, no external deps. Provenance: CUDALM-native.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "cudalm/cuda_check.h"
#include "cudalm/device_buffer.h"
#include "cudalm/qwen35_hybrid_microstack.h"
#include "cudalm/weight_loader_v2.h"

using namespace cudalm;

namespace {

struct StatBlock {
  float mean_us = 0.f;
  float min_us = 0.f;
  float max_us = 0.f;
  float p50_us = 0.f;
};

StatBlock summarize(std::vector<float> v) {
  StatBlock st;
  if (v.empty()) return st;
  st.mean_us = std::accumulate(v.begin(), v.end(), 0.f) /
               static_cast<float>(v.size());
  st.min_us = *std::min_element(v.begin(), v.end());
  st.max_us = *std::max_element(v.begin(), v.end());
  std::sort(v.begin(), v.end());
  st.p50_us = v[v.size() / 2];
  return st;
}

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

std::string stat_json(const char* key, const StatBlock& s) {
  return std::string("    \"") + key +
         "\": {\"mean_us\": " + std::to_string(s.mean_us) +
         ", \"min_us\": " + std::to_string(s.min_us) + ", \"max_us\": " +
         std::to_string(s.max_us) + ", \"p50_us\": " + std::to_string(s.p50_us) +
         "}";
}

struct CaseResult {
  int position = 0;
  std::string state;
  StatBlock layer_sum;
  StatBlock whole;
  StatBlock wall;
  std::vector<StatBlock> per_layer;  // 4 entries
};

// Every warmup + timed sample REPLAYS the identical pre-state (no snapshot/
// restore — Phase D just re-runs the prefix): reset to zero, then untimed
// forward(0..position-1) to rebuild the sequential pre-state, then the timed
// forward(position). So every sample measures the SAME decode step from the
// SAME pre-state (p0 would otherwise drift off its fresh zero state as the
// recurrent state accumulates across samples; p512 would drift off the
// 0..511 pre-state as each timed step advances the state):
//   p0:    reset_state() -> timed forward(0)
//   p512:  reset_state() -> untimed forward(0..511) -> timed forward(512)
// The pre-state rebuild is un-timed; the stream is synced before the wall-clock
// starts so the CPU window (and the CUDA events inside forwardTimed) wrap only
// the timed forward(position), not the untimed prefix.
CaseResult run_case(Qwen35HybridMicroStack& stack, const DeviceBuffer& xbuf,
                    int position, int iters, cudaStream_t stream,
                    cudaEvent_t* events, const char* state_label) {
  constexpr int kLayers = Qwen35HybridMicroStack::kNumLayers;
  float layer_us[kLayers];
  float whole_us = 0.f;
  std::vector<std::vector<float>> per_layer(kLayers);
  std::vector<float> whole_hist, sum_hist, wall_hist;
  const int warmup = 5;
  for (int it = 0; it < warmup + iters; ++it) {
    // Rebuild the identical pre-state from scratch (reset + untimed prefix).
    stack.reset_state(stream);
    for (int p = 0; p < position; ++p)
      stack.forward(p, xbuf.data<__nv_bfloat16>(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const bool timed = (it >= warmup);
    const auto t0 = std::chrono::steady_clock::now();
    stack.forwardTimed(position, xbuf.data<__nv_bfloat16>(), stream, events,
                       layer_us, &whole_us);
    const auto t1 = std::chrono::steady_clock::now();
    if (!timed) continue;
    wall_hist.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
    whole_hist.push_back(whole_us);
    float s = 0.f;
    for (int l = 0; l < kLayers; ++l) {
      per_layer[l].push_back(layer_us[l]);
      s += layer_us[l];
    }
    sum_hist.push_back(s);
  }
  CaseResult r;
  r.position = position;
  r.state = state_label;
  r.layer_sum = summarize(sum_hist);
  r.whole = summarize(whole_hist);
  r.wall = summarize(wall_hist);
  r.per_layer.resize(kLayers);
  for (int l = 0; l < kLayers; ++l) r.per_layer[l] = summarize(per_layer[l]);
  return r;
}

std::string case_json(const CaseResult& r, const std::string& label) {
  std::string out;
  out += "    {\n";
  out += "      \"label\": \"" + json_escape(label) + "\",\n";
  out += "      \"position\": " + std::to_string(r.position) + ",\n";
  out += "      \"state\": \"" + json_escape(r.state) + "\",\n";
  out += "      \"stage_layer_sum_us\": {\"mean_us\": " +
         std::to_string(r.layer_sum.mean_us) + ", \"min_us\": " +
         std::to_string(r.layer_sum.min_us) + ", \"max_us\": " +
         std::to_string(r.layer_sum.max_us) + ", \"p50_us\": " +
         std::to_string(r.layer_sum.p50_us) + "},\n";
  // per-layer breakdown.
  static const char* kLayerNames[4] = {"layer0_deltanet", "layer1_deltanet",
                                       "layer2_deltanet", "layer3_full_attn"};
  out += "      \"per_layer_gpu_us\": [\n";
  for (int l = 0; l < 4; ++l) {
    out += "        {\"layer\": \"" + std::string(kLayerNames[l]) +
           "\", \"mean_us\": " + std::to_string(r.per_layer[l].mean_us) +
           ", \"min_us\": " + std::to_string(r.per_layer[l].min_us) +
           ", \"max_us\": " + std::to_string(r.per_layer[l].max_us) +
           ", \"p50_us\": " + std::to_string(r.per_layer[l].p50_us) + "}" +
           (l < 3 ? "," : "") + "\n";
  }
  out += "      ],\n";
  out += stat_json("whole_microstack_gpu_us", r.whole) + ",\n";
  out += stat_json("host_api_wall_us", r.wall) + ",\n";
  out += "      \"microstack_steps_per_second_mean\": " +
         std::to_string(1e6 / r.whole.mean_us) + "\n";
  out += "    }";
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 5) {
    std::fprintf(stderr,
                 "usage: %s <out_ms.cudalm> [iters] [deep_pos] [out.json]\n",
                 argv[0]);
    return 2;
  }
  const std::string weights_path = argv[1];
  const int iters = argc >= 3 ? std::atoi(argv[2]) : 100;
  const int deep_pos = argc >= 4 ? std::atoi(argv[3]) : 512;
  const std::string out_path = argc >= 5 ? argv[4] : "-";
  if (iters < 1) {
    std::fprintf(stderr, "iters must be >= 1\n");
    return 1;
  }

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
  if (deep_pos < 0 || deep_pos >= c.max_seq_len) {
    std::fprintf(stderr, "deep_pos %d out of range [0, %d)\n", deep_pos,
                 c.max_seq_len);
    return 1;
  }

  // Deterministic random input (same LCG seed as the Phase B bench).
  const std::size_t H = static_cast<std::size_t>(c.hidden_size);
  std::vector<__nv_bfloat16> x_in(H);
  {
    std::uint32_t seed = 20260209u;
    auto rnd = [&seed]() {
      seed = seed * 1664525u + 1013904223u;
      return static_cast<float>((seed >> 8) & 0xFFFFFF) /
             static_cast<float>(1u << 24) * 2.0f - 1.0f;
    };
    for (auto& v : x_in) v = __float2bfloat16_rn(0.05f * rnd());
  }
  DeviceBuffer d_in(x_in.size() * sizeof(__nv_bfloat16), stream);
  d_in.copy_from_host(x_in.data(), d_in.bytes(), stream);

  Qwen35HybridMicroStack stack;
  Status sl = Qwen35HybridMicroStack::load(file, stream, &stack);
  if (!sl.ok) {
    std::fprintf(stderr, "micro-stack load failed: %s\n", sl.message.c_str());
    return 1;
  }

  constexpr int kEvents = Qwen35HybridMicroStack::kNumTimingEvents;
  std::vector<cudaEvent_t> events(kEvents);
  for (auto& e : events) CUDA_CHECK(cudaEventCreate(&e));

  // Global clock warmup: ramp the GPU to steady clock before the FIRST (p=0)
  // case so both cases are comparable. The RTX 2080 Ti idles at ~300 MHz and
  // ramps to ~1.85 GHz under sustained load; without a sustained warmup the
  // p=0 case (run first) would be inflated by clock ramp-up, which is not a
  // property of the code path. ~1500 decode steps is ~0.8 s of sustained GPU
  // load. The state reset below re-establishes the fresh-state precondition
  // for the p=0 case.
  {
    stack.reset_state(stream);
    for (int i = 0; i < 1500; ++i)
      stack.forward(0, d_in.data<__nv_bfloat16>(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stack.reset_state(stream);
  }

  CaseResult c0 = run_case(stack, d_in, 0, iters, stream, events.data(),
                           "fresh zero state (full-attention context depth 1)");
  CaseResult cd = run_case(
      stack, d_in, deep_pos, iters, stream, events.data(),
      "sequential (positions 0..deep-1 decoded; full-attention context depth "
      "deep+1)");

  std::string out;
  out += "{\n";
  out += "  \"cudalm\": \"v0.2 Phase D Qwen3.5 hybrid micro-stack (layers "
         "0-3) baseline decode-step latency\",\n";
  out += "  \"device\": \"" + std::string(prop.name) + "\",\n";
  out += "  \"config\": {\n";
  out += "    \"hidden_size\": " + std::to_string(c.hidden_size) + ",\n";
  out += "    \"num_layers\": " + std::to_string(Qwen35HybridMicroStack::kNumLayers) +
         ",\n";
  out += "    \"layer_pattern\": \"3x Gated DeltaNet (0,1,2) + 1x full "
         "attention (3)\",\n";
  out += "    \"activation_dtype\": \"bf16\",\n";
  out += "    \"max_seq_len\": " + std::to_string(c.max_seq_len) + "\n";
  out += "  },\n";
  out += "  \"iters\": " + std::to_string(iters) + ",\n";
  out += "  \"warmup\": 5,\n";
  out += "  \"units\": \"microseconds (per_layer/stage_layer_sum/"
         "whole_microstack_gpu: CUDA events; host_api_wall: CPU wall-clock)\",\n";
  out += "  \"cases\": [\n" + case_json(c0, "p0") + ",\n" +
         case_json(cd, "p" + std::to_string(deep_pos)) + "\n  ],\n";
  out += "  \"timing_semantics\": {\n";
  out += "    \"stage_layer_sum_us\": \"sum of the 4 per-layer CUDA-event "
         "durations (the layers are non-overlapping by construction); "
         "excludes inter-layer gaps and host enqueue time, NOT a whole-step "
         "latency\",\n";
  out += "    \"whole_microstack_gpu_us\": \"one CUDA-event pair around the "
         "entire 4-layer forward; a device-timeline interval covering the GPU "
         "work enqueued within it (all four layers) plus any device idle while "
         "the host enqueues; NOT a direct measurement of host CPU work (see "
         "host_api_wall_us)\",\n";
  out += "    \"host_api_wall_us\": \"CPU wall-clock around the forwardTimed() "
         "call, including stream synchronization\",\n";
  out += "    \"microstack_steps_per_second_mean\": \"1e6 / "
         "whole_microstack_gpu_us.mean_us; a micro-stack decode step is FOUR "
         "layers, NOT a full-model token (the model has 24 layers), so this is "
         "reported as steps, never tokens/s\"\n";
  out += "  }\n";
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
