// CUDALM — `cudalm-chat`: the v0.8 Phase D persistent TEXT-SESSION demo
// CLI.
//
//   one persistent session on the real Qwen3.5-0.8B-Base model
//     user turn N (raw UTF-8)
//       -> native tokenizer encode (verbatim — NO chat template, NO
//          special tokens, NO separator is added or stripped)
//       -> appended to the SAME persistent session (turn N only encodes
//          the new text; turns 1..N-1 are never re-encoded/re-forwarded)
//       -> scheduler session-bound turn (commit-then-stop; the response
//          is fully committed before it is shown)
//       -> native tokenizer decode -> the response text on stdout
//
// Scope (pinned, see README §v0.8): this is a persistent TEXT completion
// demo on a BASE model — NOT an instruct/chat-template serving API, NOT
// an HTTP/OpenAI API, no streaming. One session, one live turn at a
// time, single CUDA stream.
//
// Commands (a whole line): "reset" resets the session (state back to
// zero, same SessionId), "quit" / "exit" destroy it and leave.
//
// Exit codes: 0 = success; 1 = runtime failure (model / tokenizer /
// pool load failure, a turn that fails inside the frozen contracts);
// 2 = usage error (missing / bad arguments, invalid sampling config).

#include <cudalm/qwen35_model.h>
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/qwen35_tokenizer.h"
#include "cudalm/sampling.h"
#include "cudalm/scheduler.h"
#include "cudalm/session.h"
#include "cudalm/session_text_generator.h"
#include "cudalm/weight_loader_v2.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

using namespace cudalm;

namespace {

struct ChatOptions {
  bool help = false;
  std::string model_path;
  std::string tokenizer_path;
  int max_new_tokens = 64;
  int page_tokens = 64;
  int pages = 2048;
  int slots = 1;
  // Sampling (the same flag-presence rules as cudalm-generate):
  float temperature = 1.0f;
  bool temperature_given = false;
  int top_k = 0;
  float top_p = 1.0f;
  std::uint64_t seed = 0;
  bool greedy = false;
  bool sampling_flag_given = false;

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

const char* kUsage =
    "usage: cudalm-chat --model <full_model.cudalm> "
    "--tokenizer <tokenizer.cudaltk>\n"
    "                 [--max-new-tokens N]   (default 64)\n"
    "                 [--temperature T] [--top-k N] [--top-p P] [--seed N]\n"
    "                 [--greedy]\n"
    "                 [--page-tokens N] [--pages N] [--slots N]\n"
    "\n"
    "A persistent TEXT completion session on the BASE model: each user\n"
    "line is encoded verbatim (no chat template, no special tokens, no\n"
    "separator) and appended to the SAME persistent session — turn N\n"
    "never re-encodes or re-forwards turns 1..N-1. The model response is\n"
    "fully committed into the session's KV / Delta state before it is\n"
    "shown.\n"
    "\n"
    "REPL commands (whole line): reset — reset the session (same\n"
    "SessionId, state back to zero); quit / exit — destroy and leave.\n";

bool parse_double(const char* s, double* out);
bool parse_double(const char* s, double* out) {
  char* end = nullptr;
  const double v = std::strtod(s, &end);
  if (end == s || *end != '\0') return false;
  *out = v;
  return true;
}

bool parse_args(int argc, char** argv, ChatOptions* o, std::string* err) {
  auto next = [&](int& i, const char* name) -> const char* {
    if (i + 1 >= argc) {
      *err = std::string("missing value for ") + name;
      return nullptr;
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      o->help = true;
    } else if (a == "--model") {
      const char* v = next(i, a.c_str()); if (!v) return false;
      o->model_path = v;
    } else if (a == "--tokenizer") {
      const char* v = next(i, a.c_str()); if (!v) return false;
      o->tokenizer_path = v;
    } else if (a == "--max-new-tokens") {
      const char* v = next(i, a.c_str()); if (!v) return false;
      o->max_new_tokens = std::atoi(v);
    } else if (a == "--temperature") {
      const char* v = next(i, a.c_str()); if (!v) return false;
      double d; if (!parse_double(v, &d)) { *err = "bad --temperature"; return false; }
      o->temperature = static_cast<float>(d);
      o->temperature_given = true;
      o->sampling_flag_given = true;
    } else if (a == "--top-k") {
      const char* v = next(i, a.c_str()); if (!v) return false;
      o->top_k = std::atoi(v);
      o->sampling_flag_given = true;
    } else if (a == "--top-p") {
      const char* v = next(i, a.c_str()); if (!v) return false;
      double d; if (!parse_double(v, &d)) { *err = "bad --top-p"; return false; }
      o->top_p = static_cast<float>(d);
      o->sampling_flag_given = true;
    } else if (a == "--seed") {
      const char* v = next(i, a.c_str()); if (!v) return false;
      o->seed = std::strtoull(v, nullptr, 10);
      o->sampling_flag_given = true;
    } else if (a == "--greedy") {
      o->greedy = true;
    } else if (a == "--page-tokens") {
      const char* v = next(i, a.c_str()); if (!v) return false;
      o->page_tokens = std::atoi(v);
    } else if (a == "--pages") {
      const char* v = next(i, a.c_str()); if (!v) return false;
      o->pages = std::atoi(v);
    } else if (a == "--slots") {
      const char* v = next(i, a.c_str()); if (!v) return false;
      o->slots = std::atoi(v);
    } else {
      *err = "unknown argument: " + a;
      return false;
    }
  }
  if (!o->help) {
    if (o->model_path.empty()) { *err = "--model is required"; return false; }
    if (o->tokenizer_path.empty()) {
      *err = "--tokenizer is required";
      return false;
    }
  }
  return true;
}

std::string trim(const std::string& s) {
  const char* ws = " \t\r\n";
  const std::size_t b = s.find_first_not_of(ws);
  if (b == std::string::npos) return "";
  const std::size_t e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

}  // namespace

int main(int argc, char** argv) {
  // ---- usage layer (exit 2) ----------------------------------------------
  ChatOptions opts;
  std::string err;
  if (!parse_args(argc, argv, &opts, &err)) {
    std::fprintf(stderr, "cudalm-chat: %s\n%s\n", err.c_str(), kUsage);
    return 2;
  }
  if (opts.help) {
    std::fputs(kUsage, stdout);
    return 0;
  }
  if (opts.max_new_tokens < 0 || opts.page_tokens < 1 || opts.pages < 1 ||
      opts.slots < 1) {
    std::fprintf(stderr, "cudalm-chat: invalid pool/turn parameters\n%s\n",
                 kUsage);
    return 2;
  }
  const SamplingConfig sampling = opts.resolved_sampling();
  std::string verr;
  if (!validate_sampling_config(sampling, &verr)) {
    std::fprintf(stderr, "cudalm-chat: %s\n", verr.c_str());
    return 2;
  }

  // ---- runtime layer (exit 1) --------------------------------------------
  cudaError_t cerr = cudaSetDevice(0);
  if (cerr != cudaSuccess) {
    std::fprintf(stderr, "cudalm-chat: CUDA unavailable: %s\n",
                 cudaGetErrorString(cerr));
    return 1;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    std::fprintf(stderr, "cudalm-chat: cudaStreamCreate failed\n");
    return 1;
  }

  WeightFileV2 file;
  Status s = WeightFileV2::load(opts.model_path, &file);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-chat: model load failure: %s\n",
                 s.message.c_str());
    return 1;
  }
  Qwen35Model model;
  s = Qwen35Model::load(file, stream, &model);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-chat: model load failure: %s\n",
                 s.message.c_str());
    return 1;
  }
  std::unique_ptr<Qwen35Tokenizer> tokenizer;
  s = Qwen35Tokenizer::load(opts.tokenizer_path, &tokenizer);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-chat: tokenizer load failure: %s\n",
                 s.message.c_str());
    return 1;
  }
  Qwen35StateManager mgr(model.config(), opts.page_tokens, opts.pages,
                         opts.slots, stream);
  SessionManager sessions(mgr);
  ModelForwarder fwd(model);
  Qwen35SessionTextGenerator gen(fwd, *tokenizer, mgr, sessions, stream);

  SessionId sid = 0;
  s = gen.create_session(&sid);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-chat: create_session failed: %s\n",
                 s.message.c_str());
    return 1;
  }
  std::printf("cudalm-chat: v0.8 persistent text session (session %llu)\n",
              static_cast<unsigned long long>(sid));
  std::printf("  model: %s | sampling: %s | max_new_tokens: %d\n",
              opts.model_path.c_str(),
              sampling.is_greedy() ? "greedy" : "seeded",
              opts.max_new_tokens);
  std::printf("  RAW TEXT contract: your input is appended VERBATIM — no\n"
              "  chat template, no special tokens, no separator. This is a\n"
              "  base-model text-completion session, not a chat API.\n"
              "  Commands: reset | quit\n");

  std::string line;
  int turns = 0;
  for (;;) {
    std::printf("user> ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) break;  // EOF
    const std::string text = line;  // VERBATIM (the pinned contract)
    const std::string cmd = trim(text);
    if (cmd.empty()) continue;
    if (cmd == "quit" || cmd == "exit") break;
    if (cmd == "reset") {
      s = gen.reset_session(sid);
      std::printf(s.ok ? "session reset (state zero, same session id)\n"
                       : "reset failed: %s\n",
                   s.ok ? "" : s.message.c_str());
      continue;
    }
    ++turns;
    SessionTextTurnResult r =
        gen.generate_turn(sid, text, opts.max_new_tokens, sampling);
    if (r.ok) {
      std::printf("model> %s", r.generated_text.c_str());
      if (r.generated_text.empty() ||
          r.generated_text.back() != '\n') {
        std::printf("\n");
      }
      std::printf("  [turn %d: +%d input, +%d generated, stop %s, "
                  "context %d]\n",
                  turns, static_cast<int>(r.input_token_ids.size()),
                  static_cast<int>(r.generated_token_ids.size()),
                  r.stop_reason == FinishReason::Eos ? "eos"
                  : r.stop_reason == FinishReason::MaxNewTokens
                      ? "max_new_tokens"
                      : "?",
                  r.context_length);
    } else {
      std::fprintf(stderr, "model> [error] %s (session stays live at "
                           "context %d)\n",
                   r.error.c_str(), r.context_length);
    }
  }

  s = gen.destroy_session(sid);
  if (!s.ok) {
    std::fprintf(stderr, "cudalm-chat: destroy_session failed: %s\n",
                 s.message.c_str());
    return 1;
  }
  std::printf("session destroyed (state released)\n");
  cudaStreamDestroy(stream);
  return 0;
}
