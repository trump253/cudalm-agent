// CUDALM — v0.4 Phase C CLI argument-parsing gate (CPU-only, no model).
//
// Drives parse_generate_cli_args / GenerateCliOptions::resolved_sampling
// (cudalm/generate_cli.h) over the full argument matrix: --help, missing
// required args, unknown args, bad numeric values, mode resolution
// (greedy default / sampling flags / --greedy exclusivity), and the
// resolved SamplingConfig handed to the runtime gate.

#include "../../tests/common/check.h"
#include "cudalm/generate_cli.h"
#include "cudalm/sampling.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace cudalm;

namespace {

// Build a mutable argv (parse_generate_cli_args takes char* const*).
struct Args {
  std::vector<std::string> storage;
  std::vector<char*> ptrs;
  // reserve: the captured char* stay valid only if storage never reallocates.
  Args() {
    storage.reserve(32);
    ptrs.push_back(const_cast<char*>("cudalm-generate"));
  }
  Args& add(const std::string& s) {
    storage.push_back(s);
    ptrs.push_back(storage.back().data());
    return *this;
  }
  int argc() const { return static_cast<int>(ptrs.size()); }
  char* const* argv() const { return ptrs.data(); }
};

const char* kModel = "--model";
const char* kTok = "--tokenizer";
const char* kPrompt = "--prompt";

bool parse_ok(const Args& a, GenerateCliOptions* out, std::string* err) {
  return parse_generate_cli_args(a.argc(), a.argv(), out, err);
}

// A full valid argument set (the baseline that every error case breaks).
// NOTE: returns a NAMED variable — `return` of a call expression (e.g.
// `return Args().add(...)`) COPY-constructs here (no elision), which would
// leave the captured argv pointers dangling in the destroyed temporary.
Args base_args() {
  Args a;
  a.add(kModel).add("m.cudalm")
      .add(kTok).add("t.cudaltk")
      .add(kPrompt).add("The capital of France is");
  return a;
}

}  // namespace

int main() {
  std::printf("test_generate_cli_args: v0.4 Phase C CLI parsing gate\n");
  GenerateCliOptions o;
  std::string err;
  int rc = 0;

  // ---- --help / -h ---------------------------------------------------------
  {
    Args a;
    a.add("--help");
    CHECK(parse_ok(a, &o, &err));
    CHECK(o.help);
    Args b;
    b.add("-h").add("bogus");  // -h (first) wins: usage out, no validation
    CHECK(parse_ok(b, &o, &err));
    CHECK(o.help);
    Args c;
    c.add("bogus").add("-h");  // junk BEFORE -h is an error (no permuting)
    CHECK(!parse_ok(c, &o, &err));
    CHECK(err.find("bogus") != std::string::npos);
  }

  // ---- missing required args ------------------------------------------------
  {
    Args a;  // nothing at all
    CHECK(!parse_ok(a, &o, &err));
    CHECK(err.find("--model") != std::string::npos);
    a = Args().add(kModel).add("m.cudalm");
    CHECK(!parse_ok(a, &o, &err));
    CHECK(err.find("--tokenizer") != std::string::npos);
    a = Args().add(kModel).add("m.cudalm").add(kTok).add("t.cudaltk");
    CHECK(!parse_ok(a, &o, &err));
    CHECK(err.find("--prompt") != std::string::npos);
  }

  // ---- missing value for a flag ---------------------------------------------
  {
    Args a;
    a.add(kModel);  // value missing
    CHECK(!parse_ok(a, &o, &err));
    CHECK(err.find("--model") != std::string::npos);
  }

  // ---- unknown argument -------------------------------------------------------
  {
    Args a = base_args();
    a.add("--frobnicate");
    CHECK(!parse_ok(a, &o, &err));
    CHECK(err.find("--frobnicate") != std::string::npos);
  }

  // ---- bad numeric values ------------------------------------------------------
  {
    auto check_bad = [&](const std::string& flag, const std::string& value,
                         const std::string& needle) -> int {
      Args a = base_args();
      a.add(flag).add(value);
      if (parse_ok(a, &o, &err) || err.find(needle) == std::string::npos) {
        std::fprintf(stderr, "expected parse error with '%s' for %s %s, "
                             "got '%s'\n",
                     needle.c_str(), flag.c_str(), value.c_str(),
                     err.c_str());
        return 1;
      }
      return 0;
    };
    rc |= check_bad("--max-new-tokens", "abc", "--max-new-tokens");
    rc |= check_bad("--max-new-tokens", "-3", "--max-new-tokens");
    rc |= check_bad("--temperature", "abc", "--temperature");
    rc |= check_bad("--temperature", "1.5x", "--temperature");
    // --temperature float-range guarantees: text that does not round to a
    // representable float is a USAGE error — it must never silently become
    // inf (invalid config) or 0 (silent greedy).
    rc |= check_bad("--temperature", "1e40", "--temperature");      // float overflow -> inf
    rc |= check_bad("--temperature", "1e308", "--temperature");     // double-finite, float inf
    rc |= check_bad("--temperature", "inf", "--temperature");
    rc |= check_bad("--temperature", "nan", "--temperature");
    rc |= check_bad("--temperature", "1e-50", "--temperature");     // underflow -> 0 (no silent greedy)
    rc |= check_bad("--temperature", "7e-46", "--temperature");     // below half of the smallest denormal -> 0
    rc |= check_bad("--top-k", "-1", "--top-k");
    rc |= check_bad("--top-k", "1.5", "--top-k");
    rc |= check_bad("--top-p", "zz", "--top-p");
    rc |= check_bad("--seed", "-7", "--seed");
    rc |= check_bad("--seed", "99999999999999999999999999", "--seed");
  }
  // ---- representable extreme temperatures are LEGAL ----------------------------
  // (the float-range gate rejects only text that cannot round to a float;
  // denormals down to denorm_min and values up to FLT_MAX stay legal)
  {
    auto check_good_temp = [&](const std::string& value, float expect) -> int {
      Args a = base_args();
      a.add("--temperature").add(value);
      if (!parse_ok(a, &o, &err) || o.temperature != expect) {
        std::fprintf(stderr, "expected %s to parse to %g, got %g ('%s')\n",
                     value.c_str(), (double)expect, (double)o.temperature,
                     err.c_str());
        return 1;
      }
      return 0;
    };
    rc |= check_good_temp("0", 0.0f);                            // greedy spelling
    rc |= check_good_temp("-0", -0.0f);
    rc |= check_good_temp("1.4e-45", static_cast<float>(1.4e-45));   // denorm_min
    rc |= check_good_temp("1e-45", static_cast<float>(1e-45));       // rounds TO denorm_min (legal)
    rc |= check_good_temp("1.17549435e-38", static_cast<float>(1.17549435e-38));  // FLT_MIN
    rc |= check_good_temp("3.4e38", static_cast<float>(3.4e38));  // near FLT_MAX
  }

  // ---- valid full set: options land in the right fields -----------------------
  {
    Args a = base_args();
    a.add("--max-new-tokens").add("32");
    a.add("--temperature").add("0.8");
    a.add("--top-k").add("40");
    a.add("--top-p").add("0.95");
    a.add("--seed").add("42");
    CHECK(parse_ok(a, &o, &err));
    CHECK(!o.help);
    CHECK(o.model_path == "m.cudalm");
    CHECK(o.tokenizer_path == "t.cudaltk");
    CHECK(o.prompt == "The capital of France is");
    CHECK_EQ(o.max_new_tokens, 32);
    CHECK(o.temperature == 0.8f);
    CHECK_EQ(o.top_k, 40);
    CHECK(o.top_p == 0.95f);
    CHECK(o.seed == 42);
    CHECK(!o.greedy);
    CHECK(o.sampling_flag_given);
    const SamplingConfig c = o.resolved_sampling();
    CHECK(!c.is_greedy());
    CHECK(c.temperature == 0.8f);
    CHECK_EQ(c.top_k, 40);
    CHECK(c.top_p == 0.95f);
    CHECK(c.seed == 42);
    CHECK(validate_sampling_config(c, &err));
  }

  // ---- mode resolution ---------------------------------------------------------
  {
    // default: greedy (no sampling flags)
    {
      CHECK(parse_ok(base_args(), &o, &err));
      CHECK(!o.sampling_flag_given);
      const SamplingConfig c = o.resolved_sampling();
      CHECK(c.is_greedy());
    }
    // any sampling flag enables sampling; an omitted temperature -> 1.0
    {
      Args a = base_args();
      a.add("--top-k").add("40");
      CHECK(parse_ok(a, &o, &err));
      const SamplingConfig c = o.resolved_sampling();
      CHECK(!c.is_greedy());
      CHECK(c.temperature == 1.0f);
      CHECK_EQ(c.top_k, 40);
    }
    // --greedy alone
    {
      Args a = base_args();
      a.add("--greedy");
      CHECK(parse_ok(a, &o, &err));
      CHECK(o.greedy);
      CHECK(o.resolved_sampling().is_greedy());
    }
    // --greedy + a sampling flag -> usage error (mutually exclusive)
    {
      Args a = base_args();
      a.add("--greedy").add("--temperature").add("0.8");
      CHECK(!parse_ok(a, &o, &err));
      CHECK(err.find("--greedy") != std::string::npos);
    }
    // --temperature 0 -> sampling flag given but resolves to greedy
    {
      Args a = base_args();
      a.add("--temperature").add("0");
      CHECK(parse_ok(a, &o, &err));
      CHECK(o.sampling_flag_given);
      const SamplingConfig c = o.resolved_sampling();
      CHECK(c.is_greedy());  // temperature <= 0 selects the frozen path
    }
    // --seed alone does NOT enable sampling (greedy ignores the seed)
    {
      Args a = base_args();
      a.add("--seed").add("7");
      CHECK(parse_ok(a, &o, &err));
      CHECK(!o.sampling_flag_given);
      CHECK(o.resolved_sampling().is_greedy());
    }
  }

  // ---- values that parse but are semantically invalid --------------------------
  // (the single runtime gate validate_sampling_config catches them; the CLI
  // exits 2 before any CUDA work — checked by the CLI smoke test end to end)
  {
    Args a = base_args();
    a.add("--top-p").add("1.5");  // parses (finite) ...
    CHECK(parse_ok(a, &o, &err));
    // ... but the resolved config fails the unified gate
    CHECK(!validate_sampling_config(o.resolved_sampling(), &err));
    CHECK(err.find("top_p") != std::string::npos);
  }

  // ---- binary-safe stdout payload: embedded NUL bytes are NOT truncated --------
  // (the CLI used to fputs(c_str()): a legal generated text containing a
  // NUL byte — the native decode can produce one — would have been cut at
  // the first NUL. write_generated_text is length-aware.)
  {
    auto write_roundtrip = [&](const std::string& text,
                               const std::string& expect) -> int {
      std::FILE* f = std::tmpfile();
      if (f == nullptr) {
        std::fprintf(stderr, "tmpfile failed\n");
        return 1;
      }
      const bool ok = write_generated_text(text, f);
      std::fflush(f);
      std::fseek(f, 0, SEEK_SET);
      std::string got;
      char buf[64];
      std::size_t n = 0;
      while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
        got.append(buf, n);
      std::fclose(f);
      if (!ok || got != expect) {
        std::fprintf(stderr, "write roundtrip failed: wrote %zu bytes, "
                             "ok=%d, read %zu bytes (expected %zu)\n",
                     text.size(), (int)ok, got.size(), expect.size());
        return 1;
      }
      return 0;
    };
    // "ab\0cd" + the contract trailing newline: all 6 bytes must land.
    std::string text = "ab";
    text.push_back('\0');
    text.append("cd");
    std::string expect = "ab";
    expect.push_back('\0');
    expect.append("cd\n");
    rc |= write_roundtrip(text, expect);
    // text that IS a NUL byte (plus newline): 2 bytes.
    rc |= write_roundtrip(std::string(1, '\0'), std::string("\0\n", 2));
    // empty text: just the newline.
    rc |= write_roundtrip(std::string(), "\n");
  }

  if (rc != 0) {
    std::fprintf(stderr, "test_generate_cli_args: FAIL\n");
    return rc;
  }
  std::printf("test_generate_cli_args: PASS\n");
  return 0;
}
