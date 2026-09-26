// CUDALM — token sampling (v0.4 Phase C). CPU-only + deterministic.
//
// A minimal sampling abstraction ON TOP of the frozen Phase A greedy path:
// greedy / temperature / top-k / top-p / seed. No CUDA kernel, no model, no
// stream, no global random state. The sampler operates on the host bf16
// logits that Qwen35Generator already D2H's each step (the same scratch the
// frozen argmax reads).
//
// Mode selection: `temperature <= 0` (or SamplingConfig::greedy()) selects
// the FROZEN greedy path — the exact Phase A argmax_bf16 (maximize the
// numeric BF16 logit value; tie -> lowest token id), bit-for-bit, with ZERO
// RNG consumption. Any positive temperature enables sampling.
//
// Fixed pipeline order (never call-order dependent), sampling mode only:
//   1. temperature : f[i] = logits[i] / temperature        (T > 0)
//   2. top-k       : keep only the best k values (top_k == 0 disables;
//                    top_k < 0 is INVALID / fail loud;
//                    top_k > vocab_size CLAMPS to vocab_size; ties ->
//                    lowest token id wins the tiebreak)
//   3. top-p       : over the SURVIVORS (ascending probability = descending
//                    logit, tie -> lowest id), keep the SMALLEST prefix whose
//                    cumulative probability reaches top_p (0 < top_p <= 1;
//                    top_p == 1 disables); at least one token is always kept
//   4. normalize   : p[i] = exp(f[i] - max_f) / sum_survivors (max
//                    subtracted first -> no exp overflow; excluded tokens
//                    get p[i] == 0)
//   5. sample      : one RNG draw u in [0,1); walk the cumulative in
//                    ascending token-id order; the first index with u < cum
//                    wins; if u falls in a final rounding gap the LAST
//                    surviving token wins
//
// Determinism contract: the ONLY randomness is the per-request SplitMix64
// (a fully specified, portable 64-bit generator). A Sampler is constructed
// per generate() call with the request's seed; nothing random is global, so
// same (prompt, sampling config, seed, model) => the exact same token
// sequence, and a previous request can never leak RNG state into the next
// one (the Phase A/B state-contamination discipline applied to the sampler).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "cudalm/greedy.h"  // argmax_bf16 (the frozen Phase A greedy pick)

namespace cudalm {

// One request's sampling configuration.
struct SamplingConfig {
  float temperature = 0.0f;  // <=0 -> frozen greedy path; >0 -> sampling
  int top_k = 0;  // 0 -> disabled; <0 invalid (fail loud); >vocab_size clamps
  float top_p = 1.0f;  // (0,1]; 1.0 -> disabled
  std::uint64_t seed = 0;  // per-request RNG seed (ignored in greedy mode)

  // The canonical greedy config (== default-constructed; the Phase A path).
  static SamplingConfig greedy() { return SamplingConfig{}; }

  // Greedy iff temperature <= 0 (the frozen argmax_bf16 path, zero RNG).
  bool is_greedy() const { return temperature <= 0.0f; }
};

// Single validation gate for a sampling config (no side effects). Returns
// true when valid; on failure `*error` is set and false is returned.
// Invalid values FAIL LOUD (they are never silently clamped):
//   * temperature NaN/inf            (negative/zero are VALID -> greedy)
//   * top_k < 0                      (0 disables; >vocab clamps downstream)
//   * top_p NaN or <= 0 or > 1
inline bool validate_sampling_config(const SamplingConfig& c,
                                     std::string* error) {
  auto fail = [&](const char* m) {
    if (error != nullptr) *error = m;
    return false;
  };
  if (std::isnan(c.temperature) || std::isinf(c.temperature)) {
    return fail("sampling config: temperature must be finite "
                "(negative or zero selects greedy)");
  }
  if (c.top_k < 0) {
    return fail("sampling config: top_k must be >= 0 (0 disables top-k)");
  }
  if (std::isnan(c.top_p) || c.top_p <= 0.0f || c.top_p > 1.0f) {
    return fail("sampling config: top_p must be in (0, 1] "
                "(1.0 disables top-p)");
  }
  return true;
}

// SplitMix64 (Seeded). A fully specified 64-bit generator (constant +
// operations fixed here), so streams are deterministic across platforms and
// compiler versions. Per-REQUEST lifetime: one instance lives inside one
// Sampler; a new generate() constructs a new one from the request seed.
struct SplitMix64 {
  std::uint64_t state;
  explicit SplitMix64(std::uint64_t seed) : state(seed) {}
  // Next 64-bit value (the canonical SplitMix64 step).
  std::uint64_t next_u64() {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // Uniform double in [0,1) (the top 53 bits).
  double next_double() {
    return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
  }
};

// The effective probability vector after temperature -> top-k -> top-p ->
// normalize (stages 1-4; does NOT consume the RNG). p[i] == 0 for excluded
// tokens; the sum over survivors is 1 within fp32 rounding of the stored
// probabilities. Greedy mode: one-hot at argmax_bf16 (the frozen pick).
// Precondition: vocab >= 1, logits non-null, finite logits (the model
// produces finite bf16).
//
// NUMERICAL CONTRACT: the scaled logits / max / exp chain runs in DOUBLE.
// A finite bf16 logit (|x| <= ~3.4e38) divided by the SMALLEST legal
// positive float temperature (denorm_min ~1.4e-45) is ~2.4e83 — far inside
// the double range, so for EVERY valid finite positive temperature the
// result is a legal normalized distribution: all probabilities finite, at
// least one strictly positive (the max-scaled argmax contributes exp(0)),
// and no inf/NaN can ever be produced (a float-scaled pipeline overflows
// to inf for such temperatures and then inf - max -> NaN).
inline std::vector<float> effective_probabilities(const __nv_bfloat16* logits,
                                                  int vocab,
                                                  const SamplingConfig& cfg) {
  std::vector<float> p(static_cast<std::size_t>(vocab), 0.0f);
  if (vocab <= 0 || logits == nullptr) return p;
  if (cfg.is_greedy()) {
    p[static_cast<std::size_t>(argmax_bf16(logits, vocab))] = 1.0f;
    return p;
  }
  // Stage 1: temperature scaling in DOUBLE (see the numerical contract
  // above; T > 0 here, contract-gated upstream).
  std::vector<double> f(static_cast<std::size_t>(vocab));
  double max_f = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < vocab; ++i) {
    f[static_cast<std::size_t>(i)] =
        static_cast<double>(__bfloat162float(logits[i])) /
        static_cast<double>(cfg.temperature);
    max_f = std::max(max_f, f[static_cast<std::size_t>(i)]);
  }
  // The deterministic order used by both filters: value descending, tie ->
  // lowest token id.
  const auto by_value_then_id = [&](int a, int b) {
    if (f[static_cast<std::size_t>(a)] != f[static_cast<std::size_t>(b)])
      return f[static_cast<std::size_t>(a)] > f[static_cast<std::size_t>(b)];
    return a < b;
  };
  // Stage 2: top-k (ties -> lowest token id; k clamps to vocab).
  std::vector<char> keep(static_cast<std::size_t>(vocab), 1);
  if (cfg.top_k > 0) {
    const int k = cfg.top_k < vocab ? cfg.top_k : vocab;
    std::vector<int> idx(static_cast<std::size_t>(vocab));
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      by_value_then_id);
    for (int i = k; i < vocab; ++i) keep[static_cast<std::size_t>(idx[i])] = 0;
  }
  // Stage 3: top-p over the survivors (ascending probability order).
  std::vector<int> survivors;
  for (int i = 0; i < vocab; ++i)
    if (keep[static_cast<std::size_t>(i)])
      survivors.push_back(i);
  if (cfg.top_p < 1.0f && !survivors.empty()) {
    std::sort(survivors.begin(), survivors.end(), by_value_then_id);
    double total = 0.0;
    for (int i : survivors)
      total += std::exp(f[static_cast<std::size_t>(i)] - max_f);
    double cum = 0.0;
    int cut = static_cast<int>(survivors.size()) - 1;
    for (std::size_t j = 0; j < survivors.size(); ++j) {
      cum += std::exp(f[static_cast<std::size_t>(survivors[j])] - max_f);
      if (cum >= static_cast<double>(cfg.top_p) * total ||
          j + 1 == survivors.size()) {
        cut = static_cast<int>(j);
        break;
      }
    }
    for (std::size_t j = static_cast<std::size_t>(cut) + 1;
         j < survivors.size(); ++j)
      keep[static_cast<std::size_t>(survivors[j])] = 0;
  }
  // Stage 4: normalize over the final survivors (max already subtracted).
  double norm = 0.0;
  for (int i = 0; i < vocab; ++i)
    if (keep[static_cast<std::size_t>(i)])
      norm += std::exp(f[static_cast<std::size_t>(i)] - max_f);
  for (int i = 0; i < vocab; ++i)
    if (keep[static_cast<std::size_t>(i)])
      p[static_cast<std::size_t>(i)] =
          static_cast<float>(
              std::exp(f[static_cast<std::size_t>(i)] - max_f) / norm);
  return p;
}

// Pick one token from the host bf16 logits [vocab] under `cfg`. Greedy mode
// delegates to the frozen argmax_bf16 and consumes ZERO RNG. Sampling mode
// consumes exactly ONE rng draw (u in [0,1)) per call. Deterministic for
// fixed (logits, cfg, rng state).
//
// RETURN CONTRACT: for vocab >= 1 and finite logits the result is ALWAYS in
// [0, vocab) — the double pipeline (see effective_probabilities) guarantees
// at least one strictly positive probability, so the cumulative walk always
// finds a hit or falls back to the last survivor. The only -1 is the
// documented precondition failure (vocab <= 0 / null logits); callers
// (generator) additionally range-check defensively.
inline int sample_token(const __nv_bfloat16* logits, int vocab,
                        const SamplingConfig& cfg, SplitMix64* rng) {
  if (vocab <= 0 || logits == nullptr) return -1;
  if (cfg.is_greedy()) return argmax_bf16(logits, vocab);  // frozen, no RNG
  const std::vector<float> p = effective_probabilities(logits, vocab, cfg);
  const double u = rng->next_double();
  double cum = 0.0;
  int last_kept = -1;
  for (int i = 0; i < vocab; ++i) {
    if (p[static_cast<std::size_t>(i)] > 0.0f) {
      cum += static_cast<double>(p[static_cast<std::size_t>(i)]);
      last_kept = i;
      if (u < cum) return i;
    }
  }
  // Rounding gap at the very end -> last survivor (guaranteed >= 0 for
  // vocab >= 1 by the numerical contract; the argmax fallback is a
  // defensive last resort that can only fire on a logic bug).
  return last_kept >= 0 ? last_kept : argmax_bf16(logits, vocab);
}

// One request's sampler: the config + its per-request RNG. Construct one per
// generate() call (same seed -> same stream; a new request re-constructs, so
// RNG state never crosses requests). The config is validated by the caller
// (validate_sampling_config) before construction.
class Sampler {
 public:
  explicit Sampler(const SamplingConfig& cfg)
      : cfg_(cfg), rng_(cfg.seed) {}

  const SamplingConfig& config() const { return cfg_; }
  bool is_greedy() const { return cfg_.is_greedy(); }

  // The next token for the given host bf16 logits [vocab].
  int sample(const __nv_bfloat16* logits, int vocab) {
    return sample_token(logits, vocab, cfg_, &rng_);
  }

 private:
  SamplingConfig cfg_;
  SplitMix64 rng_;
};

}  // namespace cudalm
