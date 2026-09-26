// CUDALM — v0.4 Phase C sampling correctness (CPU-only, synthetic logits).
//
// Mathematically pinned unit tests for include/cudalm/sampling.h (no model,
// no GPU, no checkpoint):
//   * greedy compatibility        : greedy config == the frozen argmax_bf16
//   * temperature scaling         : p(T) == softmax(logits / T)
//   * top-k filtering             : only the best k survive (ties -> lowest
//                                   id; top_k > vocab CLAMPS)
//   * top-p filtering             : smallest prefix reaching cumulative top_p
//   * top-k + top-p composition   : k first, p over the survivors
//   * seed reproducibility        : same seed -> same stream (per-request)
//   * different-seed behavior     : distinct seeds -> different streams
//   * invalid config              : fail loud (the single validation gate)
//   * extreme / all-negative      : max-subtracted softmax stays finite
//   * exact ties                  : deterministic tiebreaks end to end
//   * single remaining candidate  : top_k == 1 -> deterministic argmax
//   * extreme temperatures        : FLT_MIN / denorm_min / FLT_MAX stay a
//                                   legal distribution (double pipeline)

#include "../../tests/common/check.h"
#include "cudalm/greedy.h"
#include "cudalm/sampling.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace cudalm;

namespace {

std::vector<__nv_bfloat16> bf(const std::vector<float>& v) {
  std::vector<__nv_bfloat16> o;
  o.reserve(v.size());
  for (float x : v) o.push_back(__float2bfloat16(x));
  return o;
}

// Reference softmax in double precision (the test oracle for the fp32
// pipeline).
std::vector<double> ref_softmax(const std::vector<float>& logits) {
  double mx = -1e300;
  for (float x : logits) mx = std::max(mx, static_cast<double>(x));
  double sum = 0.0;
  for (float x : logits) sum += std::exp(static_cast<double>(x) - mx);
  std::vector<double> p(logits.size());
  for (size_t i = 0; i < p.size(); ++i)
    p[i] = std::exp(static_cast<double>(logits[i]) - mx) / sum;
  return p;
}

bool close_rel(double a, double b, double tol) {
  const double d = std::fabs(a - b);
  return d <= tol * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

// 1) Greedy compatibility: a greedy config is the frozen argmax, exactly,
//    and its effective distribution is one-hot at that pick.
int test_greedy_compatibility() {
  // (leading negatives are parenthesized: `}, {-x}` is a gcc parse
  // ambiguity between a new element and a brace-list subtraction)
  std::vector<std::vector<float>> cases = {
      {0.1f, 0.2f, 0.3f}, {(-5.0f), (-5.0f), (-2.0f), (-7.0f)},
      {3.0f, 3.0f, 3.0f}, {(-1.0e6f), 1.0e6f, 0.0f}, {7.5f},
      {(-0.25f), (-0.25f), (-0.25f), (-1.0f)},
  };
  SplitMix64 rng(1234);  // the greedy path must NOT advance the RNG
  const std::uint64_t state_before = rng.state;
  for (auto& c : cases) {
    const std::vector<__nv_bfloat16> l = bf(c);
    const int v = static_cast<int>(c.size());
    const SamplingConfig g = SamplingConfig::greedy();
    CHECK(g.is_greedy());
    const int pick = sample_token(l.data(), v, g, &rng);
    CHECK_EQ(pick, argmax_bf16(l.data(), v));
    const std::vector<float> p = effective_probabilities(l.data(), v, g);
    for (int i = 0; i < v; ++i)
      CHECK_EQ(p[i] > 0.0f, i == pick);
    CHECK(p[static_cast<size_t>(pick)] == 1.0f);
  }
  CHECK_EQ(rng.state, state_before);  // zero RNG consumption in greedy mode
  std::printf("  [ok] greedy compatibility (frozen argmax, zero RNG)\n");
  return 0;
}

// 2) Temperature scaling: p(T) == softmax(logits / T), exactly (fp32 vs
//    double reference within 1e-6 relative).
int test_temperature_scaling() {
  const std::vector<float> logits = {2.0f, 1.0f, 0.0f, -1.0f, -4.0f};
  const std::vector<__nv_bfloat16> l = bf(logits);
  for (const float T : {0.5f, 1.0f, 2.0f, 10.0f}) {
    SamplingConfig c;
    c.temperature = T;
    const std::vector<float> p = effective_probabilities(l.data(),
                                                         static_cast<int>(logits.size()), c);
    std::vector<float> scaled;
    for (float x : logits) scaled.push_back(x / T);
    const std::vector<double> ref = ref_softmax(scaled);
    for (size_t i = 0; i < ref.size(); ++i)
      CHECK(close_rel(static_cast<double>(p[i]), ref[i], 1e-6));
  }
  // T -> large flattens the distribution (the max probability shrinks).
  SamplingConfig t1;
  t1.temperature = 1.0f;
  const std::vector<float> p1 = effective_probabilities(l.data(), 5, t1);
  SamplingConfig t100;
  t100.temperature = 100.0f;
  const std::vector<float> p100 = effective_probabilities(l.data(), 5, t100);
  CHECK(p1[0] > p100[0]);
  std::printf("  [ok] temperature scaling (== softmax(logits/T))\n");
  return 0;
}

// 3) Top-k filtering: only the best k get probability; ties -> lowest id;
//    top_k > vocab clamps to the full softmax.
int test_top_k_filtering() {
  const std::vector<float> logits = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  const std::vector<__nv_bfloat16> l = bf(logits);
  {
    SamplingConfig c;
    c.temperature = 1.0f;
    c.top_k = 2;
    const std::vector<float> p = effective_probabilities(l.data(), 5, c);
    CHECK(p[0] > 0.0f);
    CHECK(p[1] > 0.0f);
    CHECK(p[2] == 0.0f);
    CHECK(p[3] == 0.0f);
    CHECK(p[4] == 0.0f);
    const std::vector<double> ref = ref_softmax({5.0f, 4.0f});
    CHECK(close_rel(p[0], ref[0], 1e-6));
    CHECK(close_rel(p[1], ref[1], 1e-6));
  }
  // Ties at the cut: values {3, 3, 2} top_k=1 -> the LOWEST id of the tied
  // maximum (id 0) is the only survivor.
  {
    const std::vector<float> tied = {3.0f, 3.0f, 2.0f};
    const std::vector<__nv_bfloat16> lt = bf(tied);
    SamplingConfig c;
    c.temperature = 1.0f;
    c.top_k = 1;
    const std::vector<float> p = effective_probabilities(lt.data(), 3, c);
    CHECK(p[0] == 1.0f);
    CHECK(p[1] == 0.0f);
    CHECK(p[2] == 0.0f);
  }
  // top_k > vocab: documented CLAMP to vocab (full softmax, no error).
  {
    SamplingConfig c;
    c.temperature = 1.0f;
    c.top_k = 1000;
    const std::vector<float> p = effective_probabilities(l.data(), 5, c);
    const std::vector<double> ref = ref_softmax(logits);
    for (int i = 0; i < 5; ++i) CHECK(close_rel(p[i], ref[i], 1e-6));
  }
  std::printf("  [ok] top-k filtering (ties lowest id; >vocab clamps)\n");
  return 0;
}

// 4) Top-p filtering: the smallest probability-sorted prefix whose cumulative
//    probability reaches top_p; renormalized; >= 1 token kept; top_p == 1
//    disables.
int test_top_p_filtering() {
  const std::vector<float> logits = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  const std::vector<__nv_bfloat16> l = bf(logits);
  // Full softmax (T=1) for the expected prefix computation.
  const std::vector<double> pfull = ref_softmax(logits);
  auto expect_prefix = [&](float top_p) {
    SamplingConfig c;
    c.temperature = 1.0f;
    c.top_p = top_p;
    const std::vector<float> p = effective_probabilities(l.data(), 5, c);
    // smallest prefix reaching top_p on the FULL distribution
    double cum = 0.0;
    int keep = 5;
    for (int i = 0; i < 5; ++i) {
      cum += pfull[i];
      if (cum >= static_cast<double>(top_p) - 1e-12 || i == 4) {
        keep = i + 1;
        break;
      }
    }
    for (int i = 0; i < 5; ++i) CHECK_EQ(p[i] > 0.0f, i < keep);
    double sum = 0.0;
    for (int i = 0; i < keep; ++i) sum += pfull[i];
    for (int i = 0; i < keep; ++i)
      CHECK(close_rel(static_cast<double>(p[i]), pfull[i] / sum, 1e-6));
    return keep;
  };
  // Full-softmax cumulative: 0.6364 / 0.8705 / 0.9566 / 0.9883 / 1.0.
  CHECK_EQ(expect_prefix(0.9f), 3);   // crosses at the 3rd token
  CHECK_EQ(expect_prefix(0.7f), 2);   // crosses at the 2nd
  CHECK_EQ(expect_prefix(0.63f), 1);  // the first token alone reaches 0.63
  CHECK_EQ(expect_prefix(0.99f), 5);  // only the full set reaches 0.99
  CHECK_EQ(expect_prefix(0.01f), 1);  // at least one token is always kept
  {
    SamplingConfig c;  // top_p == 1.0 -> disabled (full softmax)
    c.temperature = 1.0f;
    c.top_p = 1.0f;
    const std::vector<float> p = effective_probabilities(l.data(), 5, c);
    for (int i = 0; i < 5; ++i) CHECK(close_rel(p[i], pfull[i], 1e-6));
  }
  std::printf("  [ok] top-p filtering (minimal prefix; >=1 kept; 1 disables)\n");
  return 0;
}

// 5) Composition: temperature -> top-k -> top-p (p runs over the k
//    SURVIVORS, not the full distribution) -> normalize. The fixed order is
//    proven by a case where k-then-p keeps a DIFFERENT set than p alone.
int test_composition() {
  // 6 tokens: five tied at 3.0, one far lower at 0.0. T = 1.
  const std::vector<float> logits = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 0.0f};
  const std::vector<__nv_bfloat16> l = bf(logits);
  // top_k = 2: survivors {0,1} (the tied maximum's lowest ids). top_p = 0.8
  // over them: 0.5 < 0.8, 0.5 + 0.5 >= 0.8 -> both kept -> {0.5,0.5,0,0,0,0}.
  {
    SamplingConfig c;
    c.temperature = 1.0f;
    c.top_k = 2;
    c.top_p = 0.8f;
    const std::vector<float> p = effective_probabilities(l.data(), 6, c);
    CHECK(close_rel(p[0], 0.5, 1e-6));
    CHECK(close_rel(p[1], 0.5, 1e-6));
    CHECK(p[2] == 0.0f);
    CHECK(p[3] == 0.0f);
    CHECK(p[4] == 0.0f);
    CHECK(p[5] == 0.0f);
  }
  // The SAME top_p (0.8) with NO top-k keeps FIVE tokens {0..4} (cumulative
  // over the full distribution: 5 x 0.198 reaches 0.8 only at the 5th):
  // k-then-p and p-alone give different sets -> the order is pinned.
  {
    SamplingConfig c;
    c.temperature = 1.0f;
    c.top_p = 0.8f;
    const std::vector<float> p = effective_probabilities(l.data(), 6, c);
    for (int i = 0; i < 5; ++i) CHECK(p[i] > 0.0f);
    CHECK(p[5] == 0.0f);
  }
  // And top_p = 0.3 after k = 2: the first survivor (0.5) alone reaches 0.3
  // -> {1,0,0,0,0,0}.
  {
    SamplingConfig c;
    c.temperature = 1.0f;
    c.top_k = 2;
    c.top_p = 0.3f;
    const std::vector<float> p = effective_probabilities(l.data(), 6, c);
    CHECK(p[0] == 1.0f);
    for (int i = 1; i < 6; ++i) CHECK(p[i] == 0.0f);
  }
  std::printf("  [ok] top-k + top-p composition (fixed order proven)\n");
  return 0;
}

// A tiny deterministic "model": a sequence of fixed logit vectors (a
// degenerate 2- or 4-way choice at every step) for the stream tests.
std::vector<std::vector<float>> synthetic_steps() {
  return {{1.0f, 1.0f, 1.0f, 1.0f},   // 4-way tie -> uniform
          {2.0f, -2.0f, 0.5f},        // 3-way spread
          {0.0f, 0.0f},               // exact 2-way tie
          {-3.0f, -1.0f, -2.0f, -0.5f},  // all negative
          {4.0f, 4.0f, -8.0f, -8.0f}};  // 2-way tie on top
}

std::vector<int> run_stream(const std::vector<std::vector<float>>& steps,
                            const SamplingConfig& cfg) {
  std::vector<int> out;
  SplitMix64 rng(cfg.seed);
  for (const auto& s : steps) {
    const std::vector<__nv_bfloat16> l = bf(s);
    out.push_back(sample_token(l.data(), static_cast<int>(s.size()), cfg,
                               &rng));
  }
  return out;
}

// 6) Seed reproducibility: same config + seed -> the exact same stream, and
//    a fresh request sampler starts from the seed again (no global state, no
//    cross-request consumption).
int test_seed_reproducibility() {
  SamplingConfig c;
  c.temperature = 0.7f;
  c.top_k = 2;
  c.top_p = 0.9f;
  c.seed = 42;
  const std::vector<std::vector<float>> steps = synthetic_steps();
  const std::vector<int> a = run_stream(steps, c);
  const std::vector<int> b = run_stream(steps, c);  // fresh rng, same seed
  CHECK_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) CHECK_EQ(a[i], b[i]);
  // Per-request isolation: consume 100 draws, then a NEW sampler with the
  // same seed must reproduce the FIRST token of the original stream.
  {
    Sampler s1(c);
    const std::vector<__nv_bfloat16> l = bf(steps[0]);
    const int first = s1.sample(l.data(), 4);
    for (int i = 0; i < 100; ++i) (void)s1.sample(l.data(), 4);
    Sampler s2(c);  // new request, same seed
    const int first_again = s2.sample(l.data(), 4);
    CHECK_EQ(first, first_again);
  }
  std::printf("  [ok] seed reproducibility (same seed -> same stream)\n");
  return 0;
}

// 7) Different-seed behavior: on a uniform 4-way choice, distinct seeds must
//    produce different streams (and the stream is uniform over a long run).
int test_different_seeds() {
  SamplingConfig c;
  c.temperature = 1.0f;  // uniform on the tie
  c.seed = 1;
  const std::vector<__nv_bfloat16> l = bf({1.0f, 1.0f, 1.0f, 1.0f});
  std::vector<int> firsts;
  for (int s = 1; s <= 16; ++s) {
    c.seed = static_cast<std::uint64_t>(s);
    SplitMix64 rng(c.seed);
    firsts.push_back(sample_token(l.data(), 4, c, &rng));
  }
  // At least two distinct tokens across 16 seeds (deterministic: the SplitMix
  // values are fixed, this is a pinned fact, not a probability).
  bool seen1 = false, seen2 = false;
  for (int t : firsts) {
    if (t != firsts[0] && !seen2) seen2 = true;
    if (t == firsts[0]) seen1 = true;
  }
  CHECK(seen1);
  CHECK(seen2);
  // Uniformity: 8192 draws each cover all 4 tokens within [20%, 30%].
  {
    c.seed = 7;
    SplitMix64 rng(c.seed);
    std::vector<int> count(4, 0);
    for (int i = 0; i < 8192; ++i)
      count[static_cast<size_t>(sample_token(l.data(), 4, c, &rng))]++;
    for (int i = 0; i < 4; ++i)
      CHECK(count[i] > 1638 && count[i] < 2458);
  }
  std::printf("  [ok] different seeds -> different streams (uniform 4-way)\n");
  return 0;
}

// 8) Invalid configs fail loud through the single validation gate.
int test_invalid_config() {
  int rc = 0;
  auto bad = [&](SamplingConfig c, const std::string& needle) {
    std::string e;
    if (!validate_sampling_config(c, &e) &&
        e.find(needle) == std::string::npos) {
      std::fprintf(stderr, "expected failure message containing '%s', got "
                           "'%s'\n",
                   needle.c_str(), e.c_str());
      return 1;
    }
    std::string ok;
    if (validate_sampling_config(c, &ok)) {  // the no-error form must fail too
      std::fprintf(stderr, "no-error form accepted an invalid config\n");
      return 1;
    }
    return 0;
  };
  { SamplingConfig c; c.temperature = std::nanf(""); rc |= bad(c, "temperature"); }
  { SamplingConfig c; c.temperature = std::numeric_limits<float>::infinity();
    rc |= bad(c, "temperature"); }
  { SamplingConfig c; c.top_k = -1; rc |= bad(c, "top_k"); }
  { SamplingConfig c; c.top_p = 0.0f; rc |= bad(c, "top_p"); }
  { SamplingConfig c; c.top_p = 1.5f; rc |= bad(c, "top_p"); }
  { SamplingConfig c; c.top_p = std::nanf(""); rc |= bad(c, "top_p"); }
  // Valid edges: greedy (temperature <= 0, even negative), disabled k/p,
  // sampling mid-range.
  auto good = [](SamplingConfig c) -> int {
    std::string e;
    if (!validate_sampling_config(c, &e) || !e.empty()) {
      std::fprintf(stderr, "valid config rejected: %s\n", e.c_str());
      return 1;
    }
    return 0;
  };
  rc |= good(SamplingConfig::greedy());
  { SamplingConfig c; c.temperature = -3.0f; rc |= good(c);
    if (!c.is_greedy()) { std::fprintf(stderr, "T<0 must be greedy\n"); rc = 1; } }
  { SamplingConfig c; c.temperature = 0.0f; rc |= good(c);
    if (!c.is_greedy()) { std::fprintf(stderr, "T==0 must be greedy\n"); rc = 1; } }
  { SamplingConfig c; c.temperature = 0.8f; c.top_k = 40; c.top_p = 0.95f;
    rc |= good(c);
    if (c.is_greedy()) { std::fprintf(stderr, "T>0 must sample\n"); rc = 1; } }
  { SamplingConfig c; c.temperature = 1.0f; c.top_k = 0; c.top_p = 1.0f;
    rc |= good(c); }
  if (rc != 0) return rc;
  std::printf("  [ok] invalid config (fail loud via the single gate)\n");
  return 0;
}

// 9) Numerical stability: extreme logits (exp would overflow without the max
//    subtraction) and all-negative logits both produce a finite, normalized
//    distribution.
int test_extreme_logits() {
  {
    // bf16-exact inputs; / 0.1 -> f = {2000, 1000, -2000}: exp(2000) would
    // overflow fp32 without the max subtraction.
    const std::vector<float> logits = {200.0f, 100.0f, -200.0f};
    const std::vector<__nv_bfloat16> l = bf(logits);
    SamplingConfig c;
    c.temperature = 0.1f;
    const std::vector<float> p = effective_probabilities(l.data(), 3, c);
    CHECK(std::isfinite(p[0]));
    CHECK(std::isfinite(p[1]));
    CHECK(std::isfinite(p[2]));
    CHECK(p[0] > 0.9999);
    CHECK(p[1] == 0.0f);  // exp(-1000) underflows to 0 (a finite 0)
    double sum = 0.0;
    for (float x : p) sum += x;
    CHECK(close_rel(sum, 1.0, 1e-6));
    // Sampling must still work on that distribution (deterministic argmax).
    {
      SplitMix64 rng(0);
      CHECK_EQ(sample_token(l.data(), 3, c, &rng), 0);
    }
  }
  {
    const std::vector<float> logits = {-10.0f, -5.0f, -2.0f};
    const std::vector<__nv_bfloat16> l = bf(logits);
    SamplingConfig c;
    c.temperature = 1.0f;
    const std::vector<float> p = effective_probabilities(l.data(), 3, c);
    const std::vector<double> ref = ref_softmax(logits);
    for (int i = 0; i < 3; ++i) CHECK(close_rel(p[i], ref[i], 1e-6));
  }
  std::printf("  [ok] extreme / all-negative logits (finite, normalized)\n");
  return 0;
}

// 10) Exact ties end to end: greedy -> lowest id; sampling ties -> uniform;
//     top_k == 1 on a tied maximum -> deterministic lowest id for ANY seed.
int test_exact_ties() {
  const std::vector<float> logits = {5.0f, 5.0f, 5.0f, 5.0f};
  const std::vector<__nv_bfloat16> l = bf(logits);
  {
    SplitMix64 rng(0);
    const int g = sample_token(l.data(), 4, SamplingConfig::greedy(), &rng);
    CHECK_EQ(g, 0);
  }
  SamplingConfig c;
  c.temperature = 1.0f;
  const std::vector<float> p = effective_probabilities(l.data(), 4, c);
  for (int i = 0; i < 4; ++i) CHECK(close_rel(p[i], 0.25, 1e-6));
  SamplingConfig k1 = c;
  k1.top_k = 1;
  for (std::uint64_t s = 0; s < 8; ++s) {
    k1.seed = s;
    SplitMix64 rng(s);
    const int t = sample_token(l.data(), 4, k1, &rng);
    CHECK_EQ(t, 0);  // the single remaining candidate is deterministic
  }
  // Ties split by top-p: {2,2,1,1} -> p ~ {0.3655, 0.3655, 0.1345, 0.1345};
  // top_p 0.5: 0.3655 < 0.5 <= 0.3655 + 0.3655 -> keeps exactly {0,1}
  // (renormalized to 0.5/0.5).
  {
    const std::vector<__nv_bfloat16> lt = bf({2.0f, 2.0f, 1.0f, 1.0f});
    SamplingConfig t;
    t.temperature = 1.0f;
    t.top_p = 0.5f;
    const std::vector<float> pt = effective_probabilities(lt.data(), 4, t);
    CHECK(close_rel(pt[0], 0.5, 1e-6));
    CHECK(close_rel(pt[1], 0.5, 1e-6));
    CHECK(pt[2] == 0.0f);
    CHECK(pt[3] == 0.0f);
  }
  std::printf("  [ok] exact ties (greedy lowest id; uniform; k=1 pinned)\n");
  return 0;
}

// 11) Single remaining candidate: any config with top_k == 1 is a
//     deterministic argmax for every seed (no sampling variance at all).
int test_single_candidate() {
  const std::vector<float> logits = {-1.5f, 3.25f, -0.25f, 3.25f, -4.0f};
  const std::vector<__nv_bfloat16> l = bf(logits);
  SamplingConfig c;
  c.temperature = 2.0f;
  c.top_k = 1;
  c.top_p = 0.5f;
  for (std::uint64_t s = 0; s < 5; ++s) {
    c.seed = 1000 + s;
    SplitMix64 rng(1000 + s);
    const int t = sample_token(l.data(), 5, c, &rng);
    CHECK_EQ(t, 1);  // id 1: the tied maximum's LOWEST id
  }
  std::printf("  [ok] single remaining candidate (top_k=1 -> argmax)\n");
  return 0;
}

// 12) EXTREME TEMPERATURES (regression: the float-scaled pipeline overflowed
//     here — logit / denorm_min -> inf, then inf - max -> NaN, all
//     probabilities NaN, sample_token returned -1). The scaled/max/exp chain
//     now runs in double: for EVERY legal finite positive temperature a
//     finite bf16 logit vector must yield a legal distribution (all p
//     finite, sum ~= 1) and a sample_token result in [0, vocab).
int test_extreme_temperatures() {
  const std::vector<float> pos = {200.0f, 100.0f, -200.0f};      // bf16-exact
  const std::vector<float> neg = {-200.0f, -100.0f, 200.0f};     // bf16-exact
  const std::vector<float> tie = {200.0f, 200.0f, -200.0f};      // bf16-exact
  {
    // T = smallest NORMAL float: scaled ~1.7e40 (float: inf; double: fine).
    const std::vector<__nv_bfloat16> l = bf(pos);
    SamplingConfig c;
    c.temperature = std::numeric_limits<float>::min();
    const std::vector<float> p = effective_probabilities(l.data(), 3, c);
    for (int i = 0; i < 3; ++i) CHECK(std::isfinite(p[i]));
    CHECK(p[0] == 1.0f);
    CHECK(p[1] == 0.0f);
    CHECK(p[2] == 0.0f);
    double sum = 0.0;
    for (float x : p) sum += x;
    CHECK(close_rel(sum, 1.0, 1e-6));
    for (std::uint64_t s = 0; s < 16; ++s) {
      SplitMix64 rng(s);
      const int t = sample_token(l.data(), 3, c, &rng);
      CHECK(t >= 0 && t < 3);
      CHECK_EQ(t, 0);
    }
  }
  {
    // T = smallest DENORMAL float (denorm_min ~1.4e-45): scaled ~1.4e47 —
    // far outside float, inside double. Negative-leading extreme logits.
    const std::vector<__nv_bfloat16> l = bf(neg);
    SamplingConfig c;
    c.temperature = std::numeric_limits<float>::denorm_min();
    const std::vector<float> p = effective_probabilities(l.data(), 3, c);
    for (int i = 0; i < 3; ++i) CHECK(std::isfinite(p[i]));
    CHECK(p[0] == 0.0f);
    CHECK(p[1] == 0.0f);
    CHECK(p[2] == 1.0f);
    double sum = 0.0;
    for (float x : p) sum += x;
    CHECK(close_rel(sum, 1.0, 1e-6));
    for (std::uint64_t s = 0; s < 16; ++s) {
      SplitMix64 rng(s);
      const int t = sample_token(l.data(), 3, c, &rng);
      CHECK(t >= 0 && t < 3);
      CHECK_EQ(t, 2);
    }
  }
  {
    // T = denorm_min on a TIED maximum: p = {0.5, 0.5, 0}; every sample
    // stays in {0,1} for every seed.
    const std::vector<__nv_bfloat16> l = bf(tie);
    SamplingConfig c;
    c.temperature = std::numeric_limits<float>::denorm_min();
    const std::vector<float> p = effective_probabilities(l.data(), 3, c);
    for (int i = 0; i < 3; ++i) CHECK(std::isfinite(p[i]));
    CHECK(close_rel(p[0], 0.5, 1e-6));
    CHECK(close_rel(p[1], 0.5, 1e-6));
    CHECK(p[2] == 0.0f);
    double sum = 0.0;
    for (float x : p) sum += x;
    CHECK(close_rel(sum, 1.0, 1e-6));
    for (std::uint64_t s = 0; s < 16; ++s) {
      SplitMix64 rng(s);
      const int t = sample_token(l.data(), 3, c, &rng);
      CHECK(t >= 0 && t < 3);
      CHECK(t == 0 || t == 1);
    }
  }
  {
    // T = largest finite float (FLT_MAX): scaled ~5.9e-37 -> effectively
    // uniform; still a legal distribution (all finite, sum ~= 1, in range).
    const std::vector<__nv_bfloat16> l = bf(pos);
    SamplingConfig c;
    c.temperature = std::numeric_limits<float>::max();
    const std::vector<float> p = effective_probabilities(l.data(), 3, c);
    for (int i = 0; i < 3; ++i) CHECK(std::isfinite(p[i]));
    CHECK(close_rel(p[0], 1.0 / 3.0, 1e-6));
    CHECK(close_rel(p[1], 1.0 / 3.0, 1e-6));
    CHECK(close_rel(p[2], 1.0 / 3.0, 1e-6));
    double sum = 0.0;
    for (float x : p) sum += x;
    CHECK(close_rel(sum, 1.0, 1e-6));
    for (std::uint64_t s = 0; s < 16; ++s) {
      SplitMix64 rng(s);
      const int t = sample_token(l.data(), 3, c, &rng);
      CHECK(t >= 0 && t < 3);
    }
  }
  std::printf("  [ok] extreme temperatures (FLT_MIN / denorm_min / FLT_MAX: "
              "finite p, sum ~= 1, sample in [0, vocab))\n");
  return 0;
}

}  // namespace

int main() {
  std::printf("test_sampling: CUDALM v0.4 Phase C CPU sampler gate\n");
  int rc = 0;
  rc |= test_greedy_compatibility();
  rc |= test_temperature_scaling();
  rc |= test_top_k_filtering();
  rc |= test_top_p_filtering();
  rc |= test_composition();
  rc |= test_seed_reproducibility();
  rc |= test_different_seeds();
  rc |= test_invalid_config();
  rc |= test_extreme_logits();
  rc |= test_exact_ties();
  rc |= test_single_candidate();
  rc |= test_extreme_temperatures();
  if (rc != 0) {
    std::fprintf(stderr, "test_sampling: FAIL\n");
    return rc;
  }
  std::printf("test_sampling: PASS\n");
  return 0;
}
