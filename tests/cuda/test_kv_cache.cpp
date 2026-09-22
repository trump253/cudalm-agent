// CUDALM — CUDA tests for the flat decode KV cache (KvCache + kv_write_fp16).
//
// Layout contract (docs/bootstrap_plan_v0.1.md §5): K, V each
// fp16 [n_kv_heads][max_seq_len][head_dim] row-major; flat offset of row
// (n, t) is ((n * max_seq_len) + t) * head_dim.
//
// Checks:
//   * p = 0 write/read round trip is BIT-EXACT (plain fp16 store, no math);
//   * untouched rows stay zero (constructor zero-inits);
//   * p > 0 with non-contiguous positions (0, 3, 7) pins the flat row
//     mapping (a wrong stride/position would land rows on the wrong rows);
//   * out-of-bounds positions (-1, max_seq_len) fatal-abort. CUDALM
//     preconditions abort the calling process, so those are verified in a
//     fresh child process (fork + exec of this same binary with
//     --abort-check <case>) that must die with SIGABRT.

#include <cuda_fp16.h>

#include <cstdio>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "cudalm/kv_cache.h"
#include "cudalm/model_config.h"

#include "../../tests/common/check.h"

using namespace cudalm;

namespace {

// Deterministic xorshift32 in [-1, 1).
struct Rng {
  std::uint32_t s;
  explicit Rng(std::uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
  float n01() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s * (2.0f / 4294967296.0f) - 1.0f;
  }
};

void fill_half(std::vector<__half>* out, std::size_t n, float scale,
               std::uint32_t seed) {
  Rng rng(seed);
  out->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    (*out)[i] = __float2half_rn(rng.n01() * scale);
  }
}

// Copies one cache tensor (n_kv*max_seq*hd fp16 elements) back to the host.
void read_back(const KvCache& cache, bool k_tensor,
               std::vector<std::uint16_t>* out, cudaStream_t stream) {
  const std::size_t n = cache.numel();
  out->resize(n);
  CUDA_CHECK(cudaMemcpyAsync(out->data(),
                             k_tensor ? cache.k() : cache.v(),
                             n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// All rows except those in `written` (written_count entries) must be zero
// (bit-exact 0x0000).
bool untouched_rows_zero(const std::vector<std::uint16_t>& c,
                         const KvCache& cache, const int* written,
                         int written_count) {
  for (int n = 0; n < cache.n_kv_heads(); ++n) {
    for (int t = 0; t < cache.max_seq_len(); ++t) {
      bool is_written = false;
      for (int i = 0; i < written_count; ++i)
        is_written = is_written || (written[i] == t);
      if (is_written) continue;
      const std::size_t off = cache.row_offset(n, t);
      for (int d = 0; d < cache.head_dim(); ++d) {
        if (c[off + d] != 0) return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// p = 0 write/read round trip (bit-exact) + untouched rows zero.
// ---------------------------------------------------------------------------
int test_roundtrip_p0(cudaStream_t stream) {
  const ModelConfig cfg = ModelConfig::v01_default();
  KvCache cache(cfg, stream);
  const int nkv = cfg.n_kv_heads, hd = cfg.head_dim;

  std::vector<__half> k, v;
  fill_half(&k, static_cast<std::size_t>(nkv) * hd, 2.0f, 0x11);
  fill_half(&v, static_cast<std::size_t>(nkv) * hd, 2.0f, 0x22);

  DeviceBuffer dk_in(k.size() * sizeof(__half), stream);
  DeviceBuffer dv_in(v.size() * sizeof(__half), stream);
  dk_in.copy_from_host(k.data(), dk_in.bytes(), stream);
  dv_in.copy_from_host(v.data(), dv_in.bytes(), stream);
  cache.write(0, dk_in.data<__half>(), dv_in.data<__half>(), stream);

  std::vector<std::uint16_t> dk, dv;
  read_back(cache, true, &dk, stream);
  read_back(cache, false, &dv, stream);

  // Compare each written row bit-for-bit.
  bool ok = true;
  for (int n = 0; n < nkv; ++n) {
    ok = ok &&
         std::memcmp(dk.data() + cache.row_offset(n, 0),
                     reinterpret_cast<const std::uint16_t*>(k.data() + n * hd),
                     static_cast<std::size_t>(hd) * 2) == 0;
    ok = ok &&
         std::memcmp(dv.data() + cache.row_offset(n, 0),
                     reinterpret_cast<const std::uint16_t*>(v.data() + n * hd),
                     static_cast<std::size_t>(hd) * 2) == 0;
  }
  CHECK(ok);
  if (!ok) {
    std::fprintf(stderr, "  kv p0: written row mismatch\n");
    return 1;
  }
  const int p0[] = {0};
  CHECK(untouched_rows_zero(dk, cache, p0, 1));
  CHECK(untouched_rows_zero(dv, cache, p0, 1));
  return 0;
}

// ---------------------------------------------------------------------------
// p > 0: non-contiguous history at positions 0, 3, 7.
// ---------------------------------------------------------------------------
int test_history_p_gt_0(cudaStream_t stream) {
  const ModelConfig cfg = ModelConfig::v01_default();
  KvCache cache(cfg, stream);
  const int nkv = cfg.n_kv_heads, hd = cfg.head_dim;
  const int positions[] = {0, 3, 7};

  std::vector<std::vector<__half>> ks, vs;
  for (int p = 0; p < 3; ++p) {
    fill_half(&ks.emplace_back(), static_cast<std::size_t>(nkv) * hd, 2.0f,
              0x30 + p);
    fill_half(&vs.emplace_back(), static_cast<std::size_t>(nkv) * hd, 2.0f,
              0x40 + p);
    DeviceBuffer dk_in(ks[p].size() * sizeof(__half), stream);
    DeviceBuffer dv_in(vs[p].size() * sizeof(__half), stream);
    dk_in.copy_from_host(ks[p].data(), dk_in.bytes(), stream);
    dv_in.copy_from_host(vs[p].data(), dv_in.bytes(), stream);
    cache.write(positions[p], dk_in.data<__half>(), dv_in.data<__half>(),
                stream);
  }

  std::vector<std::uint16_t> dk, dv;
  read_back(cache, true, &dk, stream);
  read_back(cache, false, &dv, stream);

  bool ok = true;
  for (int p = 0; p < 3; ++p) {
    for (int n = 0; n < nkv; ++n) {
      ok = ok &&
           std::memcmp(dk.data() + cache.row_offset(n, positions[p]),
                       reinterpret_cast<const std::uint16_t*>(
                           ks[p].data() + n * hd),
                       static_cast<std::size_t>(hd) * 2) == 0;
      ok = ok &&
           std::memcmp(dv.data() + cache.row_offset(n, positions[p]),
                       reinterpret_cast<const std::uint16_t*>(
                           vs[p].data() + n * hd),
                       static_cast<std::size_t>(hd) * 2) == 0;
    }
  }
  CHECK(ok);
  if (!ok) {
    std::fprintf(stderr, "  kv history: written row mismatch\n");
    return 1;
  }
  CHECK(untouched_rows_zero(dk, cache, positions, 3));
  CHECK(untouched_rows_zero(dv, cache, positions, 3));
  return 0;
}

// ---------------------------------------------------------------------------
// Out-of-bounds positions must fatal-abort (verified in a child process).
// ---------------------------------------------------------------------------
// Child entry: perform ONE bounds-violating write; reaching the end means
// the precondition failed to fire, which is the test failure.
int abort_child_main(const char* which) {
  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const ModelConfig cfg = ModelConfig::v01_default();
  KvCache cache(cfg, stream);
  std::vector<__half> k(static_cast<std::size_t>(cfg.n_kv_heads) *
                            cfg.head_dim,
                        __float2half_rn(0.5f));
  std::vector<__half> v(k.size(), __float2half_rn(0.5f));
  if (std::strcmp(which, "neg") == 0) {
    cache.write(-1, k.data(), v.data(), stream);
  } else {
    cache.write(cfg.max_seq_len, k.data(), v.data(), stream);
  }
  std::fprintf(stderr, "  abort child (%s): survived a bounds-violating write\n",
               which);
  return 1;
}

int check_aborts(const char* argv0) {
  const char* cases[] = {"neg", "over"};
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
      std::fprintf(stderr, "  bounds (%s): expected SIGABRT, status=0x%x\n",
                   c, status);
      return 1;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], "--abort-check") == 0) {
    return abort_child_main(argv[2]);
  }

  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  CHECK(n >= 1);
  CUDA_CHECK(cudaSetDevice(0));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  if (int rc = test_roundtrip_p0(stream)) return rc;
  if (int rc = test_history_p_gt_0(stream)) return rc;
  if (int rc = check_aborts(argv[0])) return rc;

  CUDA_CHECK(cudaStreamDestroy(stream));
  TEST_PASS("test_kv_cache");
  return 0;
}
