// CUDALM — `cudalm-generate` CLI argument parsing (v0.4 Phase C).
//
// CPU-only, model-free: the parsing + option-resolution logic is a pure
// function so it is unit-testable without a GPU / checkpoint (the CLI smoke
// test drives the real binary on top of this). The executable itself is
// tools/cudalm_generate.cpp.
//
// Flags (see the --help text):
//   --model <path>            required   the model .cudalm (CUDALMW02)
//   --tokenizer <path>        required   the tokenizer .cudaltk (CUDLMTK1)
//   --prompt <text>           required   the raw UTF-8 prompt
//   --max-new-tokens <N>      optional   default 64; N >= 0
//   --temperature <float>     optional   finite; <=0 -> greedy
//   --top-k <N>               optional   default 0 (disabled); N >= 0
//   --top-p <float>           optional   default 1.0 (disabled); (0,1]
//   --seed <uint64>           optional   default 0 (sampling mode only)
//   --greedy                  optional   force the frozen greedy path
//   --help / -h               print usage, exit 0
//
// Mode resolution (documented in --help):
//   * --greedy               -> greedy (any sampling flag alongside it is a
//                               usage error: the two are mutually exclusive);
//   * any of --temperature / --top-k / --top-p given -> sampling mode; an
//     omitted --temperature defaults to 1.0 (a temperature-0 value selects
//     greedy per the runtime contract);
//   * none of them given      -> greedy (the Phase A default path);
//   * --seed without a sampling flag is IGNORED (greedy consumes no RNG).
//
// Exit-code convention of the executable (not of this header): 0 = success,
// 1 = runtime failure (model / tokenizer load, generation contract),
// 2 = usage error (bad / missing arguments, invalid sampling config).

#pragma once

#include <cstdint>
#include <string>

#include "cudalm/sampling.h"

namespace cudalm {

struct GenerateCliOptions {
  bool help = false;
  std::string model_path;
  std::string tokenizer_path;
  std::string prompt;
  int max_new_tokens = 64;
  float temperature = 0.0f;  // the --temperature value (0.0 default)
  int top_k = 0;
  float top_p = 1.0f;
  std::uint64_t seed = 0;
  bool greedy = false;
  // true when any of --temperature / --top-k / --top-p appeared (even with
  // its default-equivalent value): this is what enables sampling mode.
  bool sampling_flag_given = false;
  // true only when --temperature itself appeared (an omitted temperature in
  // sampling mode defaults to 1.0).
  bool temperature_given = false;

  // The resolved SamplingConfig (see the file header for the rules).
  SamplingConfig resolved_sampling() const {
    if (greedy || !sampling_flag_given) return SamplingConfig::greedy();
    SamplingConfig c;
    c.temperature = temperature_given ? temperature : 1.0f;
    c.top_k = top_k;
    c.top_p = top_p;
    c.seed = seed;
    return c;
  }
};

// The --help / usage text (single source for the -h output + error hints).
const char* generate_cli_usage();

// Parse argv into `*out`. Returns true when parsing succeeded (a `--help`
// request counts as success with out->help == true). On failure: false,
// `*error` set (a clear message naming the offending argument), and `*out`
// left in a partial/undefined state (do not use it).
bool parse_generate_cli_args(int argc, char* const* argv,
                             GenerateCliOptions* out, std::string* error);

}  // namespace cudalm
