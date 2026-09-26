// CUDALM — v0.4 Phase C `cudalm-generate` CLI smoke test (real binary).
//
// Runs the actual CLI executable as a child process (fork/exec, no shell —
// CJK prompts pass through untouched) and gates the user-facing contract:
//   * --help exits 0 with usage text;
//   * missing required args / unknown flag / bad numeric values -> non-zero
//     (exit 2) with a clear stderr line;
//   * bad model path -> "model load failure"; bad tokenizer path ->
//     "tokenizer load failure" (exit 1);
//   * invalid sampling config (--top-p 1.5) -> exit 2, no model work;
//   * greedy path: exit 0 + non-empty text on stdout;
//   * sampling path: exit 0 + non-empty text, and two runs with the same
//     seed produce BYTE-IDENTICAL stdout (the user-visible determinism);
//   * UTF-8 (CJK) prompt: exit 0 + non-empty text.
//
// Self-skips (77) when the model / tokenizer artifacts are absent.

#include "../../tests/common/check.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool file_exists(const std::string& p) {
  std::ifstream f(p);
  return static_cast<bool>(f);
}

struct CliRun {
  int exit_code = -1;
  std::string stdout_text;
  std::string stderr_text;
};

// Run the CLI with argv (child process; stdout/stderr captured in pipes).
CliRun run_cli(const std::string& binary,
               const std::vector<std::string>& args) {
  CliRun r;
  int out_pipe[2], err_pipe[2];
  if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
    std::fprintf(stderr, "CLI test: pipe() failed\n");
    return r;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    std::fprintf(stderr, "CLI test: fork() failed\n");
    return r;
  }
  if (pid == 0) {  // child
    dup2(out_pipe[1], STDOUT_FILENO);
    dup2(err_pipe[1], STDERR_FILENO);
    close(out_pipe[0]);
    close(out_pipe[1]);
    close(err_pipe[0]);
    close(err_pipe[1]);
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(binary.c_str()));
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    execv(binary.c_str(), argv.data());
    std::fprintf(stderr, "CLI test: execv failed\n");
    _exit(127);
  }
  // parent
  close(out_pipe[1]);
  close(err_pipe[1]);
  auto drain = [&](int fd, std::string* dst) {
    char buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0)
      dst->append(buf, static_cast<size_t>(n));
    close(fd);
  };
  drain(out_pipe[0], &r.stdout_text);
  drain(err_pipe[0], &r.stderr_text);
  int status = 0;
  waitpid(pid, &status, 0);
  if (WIFEXITED(status)) r.exit_code = WEXITSTATUS(status);
  return r;
}

// A base argument set pointing at the real artifacts.
std::vector<std::string> base_args(const std::string& model,
                                   const std::string& tokenizer) {
  return {"--model", model, "--tokenizer", tokenizer,
          "--prompt", "The capital of France is"};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: test_cudalm_generate_cli <cudalm-generate binary> "
                 "<full_model.cudalm> <tokenizer.cudaltk>\n");
    return 2;
  }
  const std::string binary = argv[1];
  const std::string model = argv[2];
  const std::string tokenizer = argv[3];
  if (!file_exists(binary) || !file_exists(model) ||
      !file_exists(tokenizer)) {
    std::fprintf(stderr,
                 "[SKIP] cudalm-generate CLI: binary/model/tokenizer absent\n");
    return 77;
  }

  std::printf("test_cudalm_generate_cli: v0.4 Phase C CLI smoke\n");
  int rc = 0;

  // ---- --help ----------------------------------------------------------------
  {
    const CliRun r = run_cli(binary, {"--help"});
    CHECK_EQ(r.exit_code, 0);
    CHECK(r.stdout_text.find("usage:") != std::string::npos);
    std::printf("  [ok] --help (exit 0 + usage)\n");
  }

  // ---- usage errors (exit 2, clear stderr) ------------------------------------
  {
    const CliRun none = run_cli(binary, {});
    CHECK(none.exit_code != 0);
    CHECK(!none.stderr_text.empty());
    CHECK(none.stderr_text.find("--model") != std::string::npos);

    auto usage_err = [&](const std::string& flag, const std::string& value,
                         const std::string& needle) -> int {
      std::vector<std::string> a = base_args(model, tokenizer);
      a.push_back(flag);
      a.push_back(value);
      const CliRun r = run_cli(binary, a);
      CHECK_EQ(r.exit_code, 2);
      CHECK(r.stderr_text.find(needle) != std::string::npos);
      return 0;
    };
    rc |= usage_err("--max-new-tokens", "-1", "--max-new-tokens");
    rc |= usage_err("--temperature", "abc", "--temperature");
    rc |= usage_err("--top-k", "-1", "--top-k");
    rc |= usage_err("--seed", "-1", "--seed");
    {
      std::vector<std::string> a = base_args(model, tokenizer);
      a.push_back("--no-such-flag");
      const CliRun r = run_cli(binary, a);
      CHECK_EQ(r.exit_code, 2);
      CHECK(r.stderr_text.find("--no-such-flag") != std::string::npos);
    }
    std::printf("  [ok] usage errors (missing args / bad values / unknown)\n");
  }

  // ---- invalid sampling config (exit 2, before any model work) ----------------
  {
    std::vector<std::string> a = base_args(model, tokenizer);
    a.push_back("--top-p");
    a.push_back("1.5");
    const CliRun r = run_cli(binary, a);
    CHECK_EQ(r.exit_code, 2);
    CHECK(r.stderr_text.find("top_p") != std::string::npos);
    std::printf("  [ok] invalid sampling config (--top-p 1.5 -> exit 2)\n");
  }

  // ---- bad model / tokenizer paths (exit 1, clear messages) -------------------
  {
    std::vector<std::string> a = base_args("/nonexistent/model.cudalm",
                                           tokenizer);
    const CliRun r = run_cli(binary, a);
    CHECK_EQ(r.exit_code, 1);
    CHECK(r.stderr_text.find("model load failure") != std::string::npos);

    std::vector<std::string> b = base_args(model, "/nonexistent/t.cudaltk");
    const CliRun r2 = run_cli(binary, b);
    CHECK_EQ(r2.exit_code, 1);
    CHECK(r2.stderr_text.find("tokenizer load failure") != std::string::npos);
    std::printf("  [ok] bad model / tokenizer paths (exit 1 + message)\n");
  }

  // ---- greedy path: exit 0 + non-empty text ------------------------------------
  {
    std::vector<std::string> a = base_args(model, tokenizer);
    a.push_back("--max-new-tokens");
    a.push_back("16");
    a.push_back("--greedy");
    const CliRun r = run_cli(binary, a);
    CHECK_EQ(r.exit_code, 0);
    CHECK(!r.stdout_text.empty());
    CHECK(r.stdout_text.find('\n') != std::string::npos);
    std::printf("  [ok] greedy path (exit 0, %.24s...)\n",
                r.stdout_text.substr(0, 24).c_str());
  }

  // ---- sampling path: exit 0 + same-seed determinism (byte-identical) ---------
  {
    auto sampling_args = [&]() {
      std::vector<std::string> a = base_args(model, tokenizer);
      a.push_back("--max-new-tokens");
      a.push_back("16");
      a.push_back("--temperature");
      a.push_back("0.8");
      a.push_back("--top-k");
      a.push_back("40");
      a.push_back("--top-p");
      a.push_back("0.95");
      a.push_back("--seed");
      a.push_back("42");
      return a;
    };
    const CliRun r1 = run_cli(binary, sampling_args());
    CHECK_EQ(r1.exit_code, 0);
    CHECK(!r1.stdout_text.empty());
    const CliRun r2 = run_cli(binary, sampling_args());
    CHECK_EQ(r2.exit_code, 0);
    CHECK(r1.stdout_text == r2.stdout_text);  // same seed -> same text, byte
    std::printf("  [ok] sampling path (exit 0; same seed byte-identical, "
                "%.24s...)\n",
                r1.stdout_text.substr(0, 24).c_str());
  }

  // ---- UTF-8 (CJK) prompt -------------------------------------------------------
  {
    std::vector<std::string> a = base_args(model, tokenizer);
    // "法国的首都是" (a CJK prompt; the raw UTF-8 bytes pass through the
    // child argv untouched — no shell involved).
    const std::string cjk_prompt = "\xE6\xB3\x95\xE5\x9B\xBD\xE7\x9A\x84"
                                   "\xE9\xA6\x96\xE9\x83\xBD\xE6\x98\xAF";
    a[5] = cjk_prompt;  // replace the --prompt value
    a.push_back("--max-new-tokens");
    a.push_back("8");
    const CliRun r = run_cli(binary, a);
    CHECK_EQ(r.exit_code, 0);
    CHECK(!r.stdout_text.empty());
    std::printf("  [ok] UTF-8 (CJK) prompt (exit 0, non-empty output)\n");
  }

  if (rc != 0) {
    std::fprintf(stderr, "test_cudalm_generate_cli: FAIL\n");
    return rc;
  }
  std::printf("test_cudalm_generate_cli: PASS\n");
  return 0;
}
