// CUDALM — greedy stop-controller unit test (v0.4 Phase A; CPU-only).
//
// Deterministic proof of the greedy stop semantics, WITHOUT a model:
//   (a) the pure GreedyStopController decisions (EOS / MaxNewTokens /
//       MaxSeqLen / placeable / EOS-priority); and
//   (b) a faithful MIRROR of Qwen35Generator::generate's decode loop (the same
//       placeable -> place -> stop-check -> forward -> read order) driven by a
//       deterministic "next token" sequence, which proves:
//         * EOS selected -> stop immediately; the EOS token IS included in the
//           generated sequence; the EOS token is NOT forwarded (no forward
//           after EOS);
//         * max_new_tokens stop (the max_new_tokens-th token is included, not
//           forwarded);
//         * max_seq_len stop (prompt + generated reaches max_seq_len; the next
//           candidate is never placed, so no position >= max_seq_len is ever
//           forwarded).
// The real-model end-to-end behavior (forward_count + generated + stop_reason
// against the golden) is covered by test_qwen35_generation.

#include "../../tests/common/check.h"

#include "cudalm/greedy.h"

#include <functional>
#include <vector>

using namespace cudalm;

namespace {

// Mirror of the generator's decode loop (deterministic): `next_token(step)`
// returns the candidate the step-0 prefill-last logits / step-k decode logits
// would argmax to. forward_count counts the DECODE forwards (the prompt
// prefill forwards are a separate constant in the generator). The order is
// EXACTLY the generator's: placeable -> place -> stop-check -> (forward ->
// read next) only when NOT stopping.
struct SimResult {
  std::vector<int> generated;
  int forward_count = 0;  // decode forwards
  StopReason stop_reason = StopReason::MaxNewTokens;
};

SimResult simulate(const std::function<int(int)>& next_token,
                   const GreedyStopController& stop) {
  SimResult r;
  int next = next_token(0);  // prefill-last argmax -> t0
  for (int step = 0; step < stop.max_new_tokens; ++step) {
    if (!stop.placeable(step)) {
      r.stop_reason = StopReason::MaxSeqLen;
      return r;
    }
    r.generated.push_back(next);  // place (the EOS, if it is, is included)
    GreedyStopController::Decision d = stop.stop_after_token(next, step);
    if (d.stop) {
      r.stop_reason = d.reason;
      return r;  // NO forward after EOS / after the max_new_tokens-th token
    }
    r.forward_count++;  // forward(next)  (the EOS is never forwarded)
    next = next_token(step + 1);
  }
  r.stop_reason = StopReason::MaxNewTokens;
  return r;
}

// A deterministic "next token" sequence: `tokens[step]` (wraps past the end to
// a value distinct from eos so it does not accidentally hit EOS).
std::function<int(int)> seq(const std::vector<int>& tokens, int filler) {
  return [tokens, filler](int step) -> int {
    return step < static_cast<int>(tokens.size()) ? tokens[step] : filler;
  };
}

int test_eos_stop() {
  // EOS (id 5) is selected at step 2 (the 3rd generated token).
  GreedyStopController stop{5, 16, 3, 100};  // eos=5, max_new=16, N=3, seq=100
  auto r = simulate(seq({9, 8, 5, 7, 6}, 1), stop);
  CHECK(r.stop_reason == StopReason::Eos);
  CHECK_EQ(static_cast<int>(r.generated.size()), 3);
  CHECK(r.generated[0] == 9 && r.generated[1] == 8 && r.generated[2] == 5);
  CHECK_EQ(r.generated.back(), 5);          // EOS is INCLUDED in the sequence
  CHECK_EQ(r.forward_count, 2);             // t0,t1 forwarded; EOS (t2) NOT
  CHECK_EQ(r.forward_count,
           static_cast<int>(r.generated.size()) - 1);  // no forward after EOS
  return 0;
}

int test_max_new_tokens_stop() {
  // 5 non-EOS tokens requested -> all 5 generated, stop on MaxNewTokens, the
  // 5th token is included but not forwarded.
  GreedyStopController stop{999, 5, 4, 200};  // eos=999 (never hit), max=5
  auto r = simulate(seq({10, 11, 12, 13, 14, 15}, 16), stop);
  CHECK(r.stop_reason == StopReason::MaxNewTokens);
  CHECK_EQ(static_cast<int>(r.generated.size()), 5);
  CHECK_EQ(r.forward_count, 4);  // t0..t3 forwarded; t4 (the 5th) NOT
  return 0;
}

int test_max_seq_len_stop() {
  // prompt_len=3, max_seq_len=10 -> only 7 generated tokens fit (positions
  // 3..9); the 8th candidate (position 10) is never placed.
  GreedyStopController stop{999, 100, 3, 10};  // eos=999 (never), max=100
  auto r = simulate(seq({20, 21, 22, 23, 24, 25, 26, 27}, 28), stop);
  CHECK(r.stop_reason == StopReason::MaxSeqLen);
  CHECK_EQ(static_cast<int>(r.generated.size()), 7);  // 3 + 7 == 10 == max_seq
  // The last placed token (position 9) IS forwarded (to read the next), and
  // the next candidate (position 10) is not placed -> no position >= 10 fwd.
  CHECK_EQ(r.forward_count, 7);
  return 0;
}

int test_eos_priority_on_last_token() {
  // EOS is the LAST allowed token (step = max_new_tokens - 1): EOS must win
  // over MaxNewTokens (a natural stop).
  GreedyStopController stop{5, 4, 2, 100};  // eos=5, max_new=4
  auto r = simulate(seq({7, 8, 9, 5}, 10), stop);  // eos at step 3 (the 4th)
  CHECK(r.stop_reason == StopReason::Eos);
  CHECK_EQ(static_cast<int>(r.generated.size()), 4);
  CHECK_EQ(r.generated.back(), 5);
  CHECK_EQ(r.forward_count, 3);
  return 0;
}

int test_controller_pure() {
  GreedyStopController stop{5, 8, 3, 10};  // eos=5, max_new=8, N=3, seq=10
  // placeable: position_of(step) = 3+step < 10 -> steps 0..6 placeable, 7 not.
  for (int step = 0; step <= 6; ++step) CHECK(stop.placeable(step));
  CHECK(!stop.placeable(7));  // 3+7 == 10 == max_seq_len
  CHECK_EQ(stop.position_of(7), 10);
  // stop_after_token: EOS always stops (Eos).
  CHECK(stop.stop_after_token(5, 0).stop);
  CHECK(stop.stop_after_token(5, 0).reason == StopReason::Eos);
  // non-EOS before the last: continue.
  CHECK(!stop.stop_after_token(7, 0).stop);
  CHECK(!stop.stop_after_token(7, 6).stop);  // step 6 -> 7 tokens (< 8)
  // non-EOS on the last allowed token (step 7 -> 8 tokens): MaxNewTokens.
  auto d = stop.stop_after_token(7, 7);
  CHECK(d.stop);
  CHECK(d.reason == StopReason::MaxNewTokens);
  // EOS on the last token beats MaxNewTokens.
  auto d2 = stop.stop_after_token(5, 7);
  CHECK(d2.reason == StopReason::Eos);
  return 0;
}

}  // namespace

int main() {
  if (int rc = test_eos_stop()) return rc;
  if (int rc = test_max_new_tokens_stop()) return rc;
  if (int rc = test_max_seq_len_stop()) return rc;
  if (int rc = test_eos_priority_on_last_token()) return rc;
  if (int rc = test_controller_pure()) return rc;
  TEST_PASS("test_greedy_stop (EOS immediate+included+no-forward / "
            "max_new_tokens / max_seq_len / EOS priority / controller)");
  return 0;
}
