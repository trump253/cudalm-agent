// CUDALM — Qwen3.5 Gated DeltaNet supported-config contract test (CPU).
//
// The Phase C DeltaNet decode kernels hardcode the pinned Qwen3.5-0.8B shape
// (docs/qwen35_architecture.md §6.2/§8): depthwise causal conv kernel 4
// (conv_state len 3), an fp32 recurrent state of head dim 128, and no
// repeat_interleave (num_k_heads == num_v_heads). A config outside that
// contract would corrupt or go out of bounds, so the constructor gates on
// qwen35_deltanet_require_supported_config, which aborts on any violation
// (CUDALM has no error-returning path in v0.2). This test pins that gate:
//
//   * the pinned 0.8B config at a linear-attention layer must ACCEPT (called
//     directly in this process, so a spurious abort would crash the test);
//   * each individually-triggerable violation must REJECT with SIGABRT,
//     verified in a forked child (a precondition abort terminates the process,
//     so it cannot be observed in-process) — the same pattern as
//     tests/cuda/test_kv_cache.cpp.
//
// The contract is pure Qwen35Config logic (no CUDA calls), so this runs as a
// plain CPU test.

#include "../../tests/common/check.h"

#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_deltanet.h"

#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

using namespace cudalm;

namespace {

// Build a config that violates exactly one clause of the contract, derived
// from the pinned 0.8B config (so every other clause stays satisfied and the
// abort is attributed to the intended check). Returns the config; the caller
// picks the layer index (0 = linear-attention by the hybrid schedule).
Qwen35Config config_for_case(const char* which) {
  Qwen35Config c = Qwen35Config::qwen35_08b();
  if (std::strcmp(which, "conv_kernel") == 0) {
    c.lin_conv_kernel_dim = 5;  // kernel 5 -> state len 4 (fires the kernel-4
                                // clause, which precedes the state-len clause
                                // and is derived from the same field)
  } else if (std::strcmp(which, "key_ne_value") == 0) {
    c.lin_value_head_dim = 64;  // key 128 != value 64
  } else if (std::strcmp(which, "head_dim_not_128") == 0) {
    c.lin_key_head_dim = 64;    // equal, but not 128 (key==value passes, so
    c.lin_value_head_dim = 64;  // the head-dim-128 clause is the one that fires)
  } else if (std::strcmp(which, "num_heads_mismatch") == 0) {
    c.lin_num_v_heads = 8;  // num_v 8 != num_k 16
  }
  return c;
}

// Layer index for each case: every case uses layer 0 (linear-attention)
// except the full-attention case, which uses a (i+1)%4==0 layer so the
// is_linear_attention clause is the one that fires on an otherwise-valid config.
int layer_for_case(const char* which) {
  return (std::strcmp(which, "full_attention_layer") == 0) ? 3 : 0;
}

// Child entry: enforce the contract for one invalid case. Reaching the end
// means the precondition failed to fire, which is the test failure.
int abort_child_main(const char* which) {
  Qwen35Config c = config_for_case(which);
  qwen35_deltanet_require_supported_config(c, layer_for_case(which));
  std::fprintf(stderr,
               "  abort child (%s): survived an unsupported DeltaNet config\n",
               which);
  return 1;
}

// Fork one child per invalid case and require it to die with SIGABRT.
int check_rejections(const char* argv0) {
  const char* cases[] = {"conv_kernel", "key_ne_value", "head_dim_not_128",
                         "num_heads_mismatch", "full_attention_layer"};
  for (const char* c : cases) {
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
      const char* args[4] = {argv0, "--abort-check", c, nullptr};
      execvp(argv0, const_cast<char* const*>(args));
      _exit(127);  // exec failed
    }
    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    if (!(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT)) {
      std::fprintf(stderr,
                   "  config (%s): expected SIGABRT, status=0x%x\n", c, status);
      return 1;
    }
  }
  return 0;
}

// The pinned config at a linear-attention layer must be accepted (this runs in
// the parent process; a spurious abort would terminate the whole test).
int test_accepts_pinned_config() {
  const Qwen35Config c = Qwen35Config::qwen35_08b();
  qwen35_deltanet_require_supported_config(c, 0);
  // Every other linear-attention layer index is accepted too.
  for (int i = 0; i < c.num_hidden_layers; ++i) {
    if (c.is_linear_attention(i)) {
      qwen35_deltanet_require_supported_config(c, i);
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], "--abort-check") == 0) {
    return abort_child_main(argv[2]);
  }

  if (int rc = test_accepts_pinned_config()) return rc;
  if (int rc = check_rejections(argv[0])) return rc;

  std::fprintf(stderr, "[PASS] test_qwen35_deltanet_config\n");
  return 0;
}
