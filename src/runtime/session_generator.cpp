// CUDALM — v0.8 Phase B: incremental multi-turn generation (impl).
//
// Drives the frozen v0.5 external-state forward (forward_token_with_state)
// on a session's bound sequence; host-side full-logits D2H + per-turn
// sampling (cudalm/sampling.h). The full contract — append-only
// incremental execution, the commit contract (every generated token,
// including the stop-triggering one, is committed before the turn
// returns), preflight zero-mutation, the context-overflow reject policy,
// and the no-rollback runtime-failure semantics — is pinned in
// session_generator.h and gated by tests/cpu/test_session_turn_contract.cpp
// (no checkpoint) + tests/cuda/test_qwen35_session_generation.cpp (real
// checkpoint, bit-identical vs the one-shot continuous reference).

#include "cudalm/session_generator.h"

#include <string>

#include "cudalm/cuda_check.h"

namespace cudalm {

TurnResult SessionGenerator::generate_turn(
    SessionId session_id, const std::vector<int>& new_input_tokens,
    int max_new_tokens, int eos_token_id, const SamplingConfig& sampling,
    cudaStream_t stream, const LogitsObserver* observer) {
  TurnResult result;
  Qwen35StateManager& mgr = sessions_.manager();  // forward writes the state
  const Qwen35Config& model_cfg = model_.config();
  const int vocab = model_cfg.vocab_size;
  const int N = static_cast<int>(new_input_tokens.size());

  auto fail = [&result](std::string msg) {
    result.ok = false;
    result.error = std::move(msg);
  };

  // ---- PREFLIGHT (zero mutation; checks 1-6 need no model vocab) ---------
  // 1. The session must be live (the bound sequence is live too — the
  //    Phase A 1:1 binding invariant).
  const Session* sess = sessions_.lookup(session_id);
  if (sess == nullptr) {
    fail("session turn: session " + std::to_string(session_id) +
         " is not live (never created or destroyed)");
    return result;
  }
  // 2-3. Cheap parameter gates.
  if (N == 0) {
    fail("session turn: empty new_input_tokens (a turn must append >= 1 "
         "token)");
    return result;
  }
  if (max_new_tokens < 0) {
    fail("session turn: max_new_tokens " + std::to_string(max_new_tokens) +
         " < 0");
    return result;
  }
  // 4. Sampling config (pure CPU validation, no side effects).
  std::string serr;
  if (!validate_sampling_config(sampling, &serr)) {
    fail("session turn: invalid sampling config: " + serr);
    return result;
  }
  // 5. CONTEXT OVERFLOW (the v0.8 policy): explicit reject, no eviction /
  //    truncation. The exact boundary (sum == max_seq_len) is ACCEPTED:
  //    the last committed token would sit at position max_seq_len - 1.
  const SequenceState* seq = mgr.lookup(sess->sequence_id);
  if (seq == nullptr) {
    fail("session turn: bound sequence " +
         std::to_string(sess->sequence_id) +
         " not live (internal invariant violation)");
    return result;
  }
  const int L = seq->length;  // the session's CURRENT logical position
  const int max_seq = mgr.config().max_seq_len;
  if (static_cast<long long>(L) + N + max_new_tokens > max_seq) {
    fail("session turn: context overflow (length " + std::to_string(L) +
         " + input " + std::to_string(N) + " + max_new_tokens " +
         std::to_string(max_new_tokens) + " > max_seq_len " +
         std::to_string(max_seq) +
         "); rejected (no eviction / truncation)");
    return result;
  }
  // 6. Single-stream contract (the v0.5 discipline, preflighted).
  if (stream != mgr.kv_pool().stream() || stream != mgr.delta_pool().stream()) {
    fail("session turn: stream mismatch — the v0.8 runtime is "
         "single-stream; the turn stream must equal the manager pool "
         "streams");
    return result;
  }
  // 7-8-9-10. Model-dependent gates (need the model's vocab / config).
  if (!model_.loaded()) {
    fail("session turn: model not loaded");
    return result;
  }
  if (eos_token_id != -1 && (eos_token_id < 0 || eos_token_id >= vocab)) {
    fail("session turn: invalid eos_token_id " + std::to_string(eos_token_id) +
         " (must be -1 = no gate, or in [0, " + std::to_string(vocab) +
         "))");
    return result;
  }
  for (int t : new_input_tokens) {
    if (t < 0 || t >= vocab) {
      fail("session turn: invalid input token id " + std::to_string(t) +
           " outside [0, " + std::to_string(vocab) + ")");
      return result;
    }
  }
  if (!(mgr.config() == model_cfg)) {
    fail("session turn: manager config does not match the model config");
    return result;
  }

  // ---- EXECUTION ----------------------------------------------------------
  const SequenceId sid = sess->sequence_id;
  result.ok = true;
  result.context_length = L;  // advances with every committed forward

  // A FRESH per-turn Sampler (seeded from `sampling.seed`) — per-turn RNG
  // isolation: a turn never sees another turn's RNG state. Greedy configs
  // run the frozen argmax inside sample() with zero RNG consumption.
  Sampler sampler(sampling);
  if (host_logits_.size() < static_cast<std::size_t>(vocab))
    host_logits_.resize(static_cast<std::size_t>(vocab));

  // Forward one token, then D2H its FULL logits into the scratch and
  // report them to the observer (step = the turn-local forward index).
  // On a forward failure NOTHING is D2H'd and no observer call happens;
  // the caller (the loop below) reports the runtime failure.
  auto commit_forward = [&](int token, int step) -> Status {
    Status s = model_.forward_token_with_state(token, sid, mgr, stream);
    if (!s.ok) return s;
    CUDA_CHECK(cudaMemcpyAsync(host_logits_.data(), model_.logits(),
                               static_cast<std::size_t>(vocab) *
                                   sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (observer) (*observer)(host_logits_.data(), step);
    return Status::ok_status();
  };

  // Defensive sampler-range check (the sampler's contract is [0, vocab);
  // a logic bug must never reach forward_token_with_state — fail loud).
  auto pick_next = [&](int* out) -> bool {
    const int t = sampler.sample(host_logits_.data(), vocab);
    if (t < 0 || t >= vocab) {
      fail("session turn: internal error: sampler returned token id " +
           std::to_string(t) + " outside [0, " + std::to_string(vocab) + ")");
      return false;
    }
    *out = t;
    return true;
  };

  // ---- input phase: positions L..L+N-1 (no sampling on input forwards) ----
  for (int i = 0; i < N; ++i) {
    Status s = commit_forward(new_input_tokens[i], i);
    if (!s.ok) {
      fail("session turn: forward failed at input token " +
           std::to_string(i) + " (token " +
           std::to_string(new_input_tokens[i]) + "): " + s.message +
           " — the " + std::to_string(i) +
           " input tokens before it stay committed (no rollback)");
      return result;  // inputs [0..i-1] committed; session at L+i
    }
    result.input_count = i + 1;
    result.forward_count = i + 1;
    result.context_length = L + i + 1;
  }

  // ---- max_new_tokens == 0: an input-only append (documented) -------------
  if (max_new_tokens == 0) {
    result.stop_reason = StopReason::MaxNewTokens;
    return result;
  }

  // ---- generation phase: positions L+N.. (commit-then-stop) --------------
  // The last input forward's logits predict g0.
  int next = 0;
  if (!pick_next(&next)) {
    return result;
  }
  for (int k = 0; k < max_new_tokens; ++k) {
    Status s = commit_forward(next, N + k);
    if (!s.ok) {
      fail("session turn: forward failed at generated token " +
           std::to_string(k) + " (token " + std::to_string(next) + "): " +
           s.message + " — the " + std::to_string(k) +
           " generated tokens before it stay committed (no rollback); this "
           "token is NOT committed");
      return result;  // generated [0..k-1] committed; `next` is not
    }
    result.generated.push_back(next);
    result.forward_count++;
    result.context_length = L + N + k + 1;
    if (eos_token_id != -1 && next == eos_token_id) {
      // EOS: the token IS committed (the commit contract); stop right
      // after it (EOS takes priority over max_new_tokens).
      result.stop_reason = StopReason::Eos;
      return result;
    }
    if (k + 1 == max_new_tokens) {
      // The LAST generated token of the turn IS committed (the commit
      // contract — deliberately different from the frozen v0.4 one-shot
      // request, which leaves its final token un-forwarded).
      result.stop_reason = StopReason::MaxNewTokens;
      return result;
    }
    if (!pick_next(&next)) {
      return result;
    }
  }
  // Unreachable: the loop returns on k+1 == max_new_tokens.
  result.stop_reason = StopReason::MaxNewTokens;
  return result;
}

}  // namespace cudalm
