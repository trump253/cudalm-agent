// CUDALM — v0.3 full-model test (real checkpoint; CPU structure gate + GPU
// ownership / state-lifecycle gate).
//
// Two phases:
//   (CPU) Structure / ingestion: convert the FULL Qwen3.5-0.8B-Base (all 24
//     decoder layers + embed_tokens + final norm + tie metadata) with
//     tools/convert_qwen35.py --full-model, parse it (WeightFileV2), and check
//     the pinned full-model contract — 24/24 layers validate, the
//     embedding/final-norm/LM-head tensor contract, the exact hybrid layer
//     schedule (full attention at 3,7,11,15,19,23), and the weight-tying flag.
//   (GPU) Ownership / state lifecycle: load the full Qwen35Model (device
//     memory), verify the model-level tensors (embedding, final norm, tied LM
//     head) and all 24 per-layer weight sets + runtimes, then prove
//     reset_state() covers EVERY layer: seed each layer's persistent state to a
//     non-zero sentinel, reset, and verify all 24 are back to zero.
//
// Skips with exit 77 when the checkpoint is absent (it lives in /root/models).

#include "../../tests/common/check.h"

#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_kv_cache.h"
#include "cudalm/qwen35_model.h"
#include "cudalm/weight_loader_v2.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace cudalm;

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

// The exact full-attention schedule for the pinned 24-layer model: full
// attention at layers 3,7,11,15,19,23; Gated DeltaNet at the other 18.
bool expect_full_attention(int i) { return (i == 3 || i == 7 || i == 11 ||
                                           i == 15 || i == 19 || i == 23); }

// (CPU) Validate the full-model file structure against the pinned contract.
int cpu_structure(const std::string& cudalm_path) {
  WeightFileV2 file;
  Status s = WeightFileV2::load(cudalm_path, &file);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "load error: %s\n", s.message.c_str());
    return 1;
  }

  // Pinned config.
  CHECK(file.config() == Qwen35Config::qwen35_08b());
  const Qwen35Config& c = file.config();
  CHECK_EQ(c.num_hidden_layers, 24);
  CHECK_EQ(c.vocab_size, 248320);
  CHECK_EQ(c.hidden_size, 1024);

  // Weight tying: the pinned 0.8B ties the LM head to the embedding.
  CHECK(file.meta("tie_word_embeddings") != nullptr);
  CHECK_EQ(*file.meta("tie_word_embeddings"), "true");

  // Full-model tensor set: embedding + final norm + every layer + tie.
  CHECK(file.validate_full_model().ok);
  // 24/24 layers validate individually.
  for (int i = 0; i < 24; ++i) {
    Status ls = file.validate_layer(i);
    CHECK(ls.ok);
    if (!ls.ok) {
      std::fprintf(stderr, "layer %d: %s\n", i, ls.message.c_str());
      return 1;
    }
  }
  // Embedding [vocab, hidden] bf16; final norm [hidden] bf16.
  const Qwen35TensorInfo* emb = file.find("embed_tokens.weight");
  CHECK(emb != nullptr);
  CHECK(emb->dtype == Dtype::kBf16);
  CHECK(emb->dims.size() == 2 && emb->dims[0] == 248320 && emb->dims[1] == 1024);
  const Qwen35TensorInfo* norm = file.find("norm.weight");
  CHECK(norm != nullptr);
  CHECK(norm->dtype == Dtype::kBf16);
  CHECK(norm->dims.size() == 1 && norm->dims[0] == 1024);
  // No separate lm_head tensor (tied to the embedding).
  CHECK(file.find("lm_head.weight") == nullptr);

  // Exact hybrid schedule: full attention at 3,7,11,15,19,23; DeltaNet at the
  // other 18.
  int n_full = 0, n_delta = 0;
  for (int i = 0; i < 24; ++i) {
    if (expect_full_attention(i)) {
      CHECK(file.config().is_full_attention(i));
      ++n_full;
    } else {
      CHECK(file.config().is_linear_attention(i));
      ++n_delta;
    }
  }
  CHECK_EQ(n_full, 6);
  CHECK_EQ(n_delta, 18);
  return 0;
}

// (GPU) Ownership + reset_state() covers all 24 layers.
int gpu_own_reset(const std::string& cudalm_path) {
  WeightFileV2 file;
  Status s = WeightFileV2::load(cudalm_path, &file);
  CHECK(s.ok);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Qwen35Model model;
  s = Qwen35Model::load(file, stream, &model);
  CHECK(s.ok);
  if (!s.ok) {
    std::fprintf(stderr, "Qwen35Model::load: %s\n", s.message.c_str());
    return 1;
  }
  CHECK(model.loaded());
  CHECK_EQ(model.num_layers(), 24);
  CHECK(model.tie_word_embeddings());

  // Model-level tensors: embedding [248320,1024] bf16, final norm [1024] bf16,
  // LM head ALIASES the embedding (weight tying → same device pointer).
  const TensorView& emb = model.embedding();
  CHECK(emb.dtype() == Dtype::kBf16);
  CHECK(emb.shape().size() == 2 && emb.shape()[0] == 248320 && emb.shape()[1] == 1024);
  const TensorView& norm = model.final_norm();
  CHECK(norm.dtype() == Dtype::kBf16);
  CHECK(norm.shape().size() == 1 && norm.shape()[0] == 1024);
  CHECK(model.lm_head().data() == emb.data());  // tied: same buffer

  // All 24 layers: the correct runtime type per the schedule + a weight set.
  for (int i = 0; i < 24; ++i) {
    CHECK(model.layer_weights(i) != nullptr);
    if (expect_full_attention(i)) {
      CHECK(model.attention(i) != nullptr);
      CHECK(model.delta(i) == nullptr);
    } else {
      CHECK(model.delta(i) != nullptr);
      CHECK(model.attention(i) == nullptr);
    }
  }
  // Boundary layers explicitly.
  CHECK(model.attention(3) != nullptr);
  CHECK(model.attention(23) != nullptr);
  CHECK(model.delta(0) != nullptr);
  CHECK(model.delta(22) != nullptr);

  // --- Prove reset_state() covers EVERY layer -----------------------------
  // Seed each layer's persistent state to a non-zero sentinel (so a no-op
  // reset would be caught), then reset and verify all 24 are back to zero.
  const Qwen35Config& cfg = file.config();
  std::vector<__nv_bfloat16> hconv(
      static_cast<std::size_t>(cfg.linear_conv_dim()) *
          static_cast<std::size_t>(cfg.linear_conv_state_len()),
      __float2bfloat16(1.0f));
  std::vector<float> hrec(16 * 128 * 128, 1.0f);
  for (int i = 0; i < 24; ++i) {
    if (model.is_linear_attention(i)) {
      model.delta(i)->seed_state(hconv.data(), hrec.data(), stream);
    } else {
      // Seed only the head of K/V (the region the check reads back); reset
      // zeroes the WHOLE KV, so a no-op reset would leave this head non-zero
      // and be caught. Seeding a small region (not the full ~256 MB tensor)
      // keeps the memcheck (byte-tracking) run fast.
      Qwen35KvCache& kv = model.attention(i)->kv_cache();
      CUDA_CHECK(cudaMemsetAsync(kv.k_mut(), 0x3f,
                                 64 * sizeof(__nv_bfloat16), stream));
      CUDA_CHECK(cudaMemsetAsync(kv.v_mut(), 0x3f,
                                 64 * sizeof(__nv_bfloat16), stream));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Sanity: the seed is non-zero.
  {
    float h[4];
    CUDA_CHECK(cudaMemcpy(h, model.delta(0)->recurrent_state(), sizeof(h),
                          cudaMemcpyDeviceToHost));
    CHECK(h[0] != 0.0f);
    __nv_bfloat16 hb[4];
    CUDA_CHECK(cudaMemcpy(hb, model.attention(3)->kv_cache().k(),
                          sizeof(hb), cudaMemcpyDeviceToHost));
    CHECK(__bfloat162float(hb[0]) != 0.0f);
  }

  // Reset the whole model (covers all 24 layers).
  model.reset_state(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Every layer's state must now be zero (DeltaNet recurrent_state; FA KV).
  for (int i = 0; i < 24; ++i) {
    if (model.is_linear_attention(i)) {
      float h[32];
      CUDA_CHECK(cudaMemcpy(h, model.delta(i)->recurrent_state(), sizeof(h),
                            cudaMemcpyDeviceToHost));
      for (int j = 0; j < 32; ++j) CHECK(h[j] == 0.0f);
    } else {
      __nv_bfloat16 h[64];
      CUDA_CHECK(cudaMemcpy(h, model.attention(i)->kv_cache().k(), sizeof(h),
                            cudaMemcpyDeviceToHost));
      for (int j = 0; j < 64; ++j) CHECK(__bfloat162float(h[j]) == 0.0f);
    }
  }

  // model is destroyed here (device memory released = "unload").
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <out.cudalm> <checkpoint_dir> <python> <src_dir>"
                 " [--no-gen]\n",
                 argv[0]);
    return 2;
  }
  const std::string out = argv[1];
  const std::string ckpt = argv[2];
  const std::string py = argv[3];
  const std::string src = argv[4];
  const bool no_gen = (argc >= 6 && std::string(argv[5]) == "--no-gen");

  if (!no_gen && !file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] qwen35 full model: checkpoint absent at %s\n",
                 ckpt.c_str());
    return 77;
  }

  if (!no_gen) {
    // 1. Convert the FULL model (all 24 layers + embedding + norm + tie).
    int rc = run_cmd(py + " " + src + "/tools/convert_qwen35.py" +
                     " --checkpoint-dir " + ckpt + " --out " + out +
                     " --full-model");
    CHECK_EQ(rc, 0);
  } else {
    CHECK(file_exists(out));
  }

  // 2. (CPU) full-model structure gate.
  int rc = cpu_structure(out);
  if (rc != 0) { std::fprintf(stderr, "cpu_structure FAILED\n"); return rc; }

  // 3. (GPU) ownership + reset_state covers all 24 layers.
  rc = gpu_own_reset(out);
  if (rc != 0) { std::fprintf(stderr, "gpu_own_reset FAILED\n"); return rc; }

  TEST_PASS("qwen35_full_model (structure + ownership + reset-all-24)");
  return 0;
}
