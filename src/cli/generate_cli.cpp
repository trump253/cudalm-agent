// CUDALM — `cudalm-generate` CLI argument parsing (v0.4 Phase C).
// See include/cudalm/generate_cli.h for the contract.

#include "cudalm/generate_cli.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>

namespace cudalm {

const char* generate_cli_usage() {
  return
      "usage: cudalm-generate --model <model.cudalm> --tokenizer "
      "<tokenizer.cudaltk> --prompt <text> [options]\n"
      "\n"
      "Options:\n"
      "  --model <path>            the model file (CUDALMW02); required\n"
      "  --tokenizer <path>        the tokenizer file (CUDLMTK1); required\n"
      "  --prompt <text>           the raw UTF-8 prompt; required\n"
      "  --max-new-tokens <N>      maximum tokens to generate (default 64)\n"
      "  --temperature <float>     sampling temperature (finite; <=0 = greedy)\n"
      "  --top-k <N>               keep only the best k candidates (default 0\n"
      "                            = disabled; larger than the vocab clamps)\n"
      "  --top-p <float>           keep the smallest set of top candidates\n"
      "                            whose cumulative probability reaches p\n"
      "                            (default 1.0 = disabled; range (0, 1])\n"
      "  --seed <uint64>           RNG seed (sampling mode only; greedy\n"
      "                            consumes no RNG)\n"
      "  --greedy                  force the frozen greedy path (mutually\n"
      "                            exclusive with --temperature/--top-k/--top-p)\n"
      "  --help, -h                show this help and exit\n"
      "\n"
      "Mode: greedy by default; passing --temperature, --top-k or --top-p\n"
      "enables sampling (an omitted --temperature then defaults to 1.0).\n"
      "Output: the generated text on stdout (errors go to stderr).\n";
}

namespace {

// Full-consumption float parse ("1.5x" is an error, not 1.5).
bool parse_float(const char* s, float* out) {
  if (s == nullptr || *s == '\0') return false;
  char* end = nullptr;
  const double v = std::strtod(s, &end);
  if (end == s || *end != '\0') return false;
  *out = static_cast<float>(v);
  return true;
}

// Full-consumption unsigned integer parse.
template <typename T>
bool parse_uint(const char* s, T* out) {
  if (s == nullptr || *s == '\0') return false;
  const char* p = s;
  while (*p != '\0') {
    if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
    ++p;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0' || errno == ERANGE) return false;
  if (v > static_cast<unsigned long long>(std::numeric_limits<T>::max()))
    return false;
  *out = static_cast<T>(v);
  return true;
}

}  // namespace

bool parse_generate_cli_args(int argc, char* const* argv,
                             GenerateCliOptions* out, std::string* error) {
  auto fail = [&](const std::string& m) {
    if (error != nullptr) *error = m;
    return false;
  };
  auto need_value = [&](int& i, const char* flag) -> const char* {
    if (i + 1 >= argc) {
      fail(std::string("missing value for ") + flag);
      return nullptr;
    }
    return argv[++i];
  };

  GenerateCliOptions o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      o.help = true;
      break;  // --help wins: nothing else is validated
    }
    const char* v = nullptr;
    if (a == "--model") {
      v = need_value(i, "--model");
      if (!v) return false;
      o.model_path = v;
    } else if (a == "--tokenizer") {
      v = need_value(i, "--tokenizer");
      if (!v) return false;
      o.tokenizer_path = v;
    } else if (a == "--prompt") {
      v = need_value(i, "--prompt");
      if (!v) return false;
      o.prompt = v;
    } else if (a == "--max-new-tokens") {
      v = need_value(i, "--max-new-tokens");
      if (!v) return false;
      unsigned long long n = 0;
      if (!parse_uint(v, &n) || n > static_cast<unsigned long long>(0x7FFFFFFF))
        return fail("--max-new-tokens must be a non-negative integer");
      o.max_new_tokens = static_cast<int>(n);
    } else if (a == "--temperature") {
      v = need_value(i, "--temperature");
      if (!v) return false;
      if (!parse_float(v, &o.temperature))
        return fail("--temperature must be a finite number");
      o.sampling_flag_given = true;
      o.temperature_given = true;
    } else if (a == "--top-k") {
      v = need_value(i, "--top-k");
      if (!v) return false;
      unsigned long long n = 0;
      if (!parse_uint(v, &n) ||
          n > static_cast<unsigned long long>(0x7FFFFFFF))
        return fail("--top-k must be a non-negative integer");
      o.top_k = static_cast<int>(n);
      o.sampling_flag_given = true;
    } else if (a == "--top-p") {
      v = need_value(i, "--top-p");
      if (!v) return false;
      if (!parse_float(v, &o.top_p))
        return fail("--top-p must be a finite number");
      o.sampling_flag_given = true;
    } else if (a == "--seed") {
      v = need_value(i, "--seed");
      if (!v) return false;
      if (!parse_uint(v, &o.seed))
        return fail("--seed must be a non-negative integer");
    } else if (a == "--greedy") {
      o.greedy = true;
    } else {
      return fail("unknown argument: " + a);
    }
  }

  if (o.help) {
    *out = o;
    return true;
  }

  if (o.greedy && o.sampling_flag_given)
    return fail("--greedy is mutually exclusive with --temperature/--top-k/"
                "--top-p");

  // Required arguments.
  if (o.model_path.empty())
    return fail("missing required argument --model");
  if (o.tokenizer_path.empty())
    return fail("missing required argument --tokenizer");
  if (o.prompt.empty())
    return fail("missing required argument --prompt");

  *out = o;
  return true;
}

}  // namespace cudalm
