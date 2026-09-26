// CUDALM — `cudalm-generate`: the user-facing prompt -> text CLI
// (v0.4 Phase C).
//
//   raw UTF-8 prompt
//     -> native tokenizer encode   (CUDLMTK1 artifact; docs §18/§20)
//     -> Qwen35Generator           (serial prefill + greedy/sampling decode;
//                                   docs §19/§21; the tokenizer's pinned
//                                   real EOS is the stop token)
//     -> native tokenizer decode   (skip_special_tokens == false)
//     -> the generated text on stdout
//
// Single request, serial, no streaming / chat template / batching — v0.4
// scope (docs §21). No debug/logit output in normal mode.
//
// Exit codes: 0 = success; 1 = runtime failure (model / tokenizer load
// failure, generation contract violation); 2 = usage error (missing / bad
// arguments, invalid sampling config).

#include <cudalm/generate_cli.h>
#include <cudalm/qwen35_model.h>
#include <cudalm/qwen35_text_generator.h>
#include <cudalm/qwen35_tokenizer.h>
#include <cudalm/weight_loader_v2.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <string>

int main(int argc, char** argv) {
  using namespace cudalm;

  // ---- usage layer (exit 2) ----------------------------------------------
  GenerateCliOptions opts;
  std::string err;
  if (!parse_generate_cli_args(argc, argv, &opts, &err)) {
    std::fprintf(stderr, "cudalm-generate: %s\n%s\n", err.c_str(),
                 generate_cli_usage());
    return 2;
  }
  if (opts.help) {
    std::fputs(generate_cli_usage(), stdout);
    return 0;
  }
  const SamplingConfig sampling = opts.resolved_sampling();
  std::string verr;
  if (!validate_sampling_config(sampling, &verr)) {
    std::fprintf(stderr, "cudalm-generate: %s\n", verr.c_str());
    return 2;
  }

  // ---- runtime layer (exit 1) ---------------------------------------------
  cudaError_t cerr = cudaSetDevice(0);
  if (cerr != cudaSuccess) {
    std::fprintf(stderr, "cudalm-generate: CUDA unavailable: %s\n",
                 cudaGetErrorString(cerr));
    return 1;
  }
  cudaStream_t stream = nullptr;
  cerr = cudaStreamCreate(&stream);
  if (cerr != cudaSuccess) {
    std::fprintf(stderr, "cudalm-generate: cudaStreamCreate failed: %s\n",
                 cudaGetErrorString(cerr));
    return 1;
  }

  WeightFileV2 file;
  Status s = WeightFileV2::load(opts.model_path, &file);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-generate: model load failure: %s\n",
                 s.message.c_str());
    return 1;
  }
  Qwen35Model model;
  s = Qwen35Model::load(file, stream, &model);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-generate: model load failure: %s\n",
                 s.message.c_str());
    return 1;
  }

  std::unique_ptr<Qwen35Tokenizer> tokenizer;
  s = Qwen35Tokenizer::load(opts.tokenizer_path, &tokenizer);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-generate: tokenizer load failure: %s\n",
                 s.message.c_str());
    return 1;
  }

  Qwen35TextGenerator textgen(model, *tokenizer);
  const TextGenerationResult r =
      textgen.generate_text(opts.prompt, opts.max_new_tokens, sampling,
                            stream);
  cudaStreamDestroy(stream);
  if (!r.ok) {
    std::fprintf(stderr, "cudalm-generate: %s\n", r.error.c_str());
    return 1;
  }

  // The generated text (and nothing else) on stdout.
  std::fputs(r.generated_text.c_str(), stdout);
  std::fputc('\n', stdout);
  if (std::fflush(stdout) != 0) return 1;
  return 0;
}
