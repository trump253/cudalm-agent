// CUDALM — v0.8 Phase D: text-level session facade E2E HARD GATE (CUDA,
// REAL Qwen3.5-0.8B-Base checkpoint + REAL native tokenizer).
//
// Proves that the Phase D text facade does NOT break the frozen
// token/runtime contracts beneath it: two incremental TEXT turns driven
// through Qwen35SessionTextGenerator (encode -> session-bound scheduler
// turn -> decode) are compared, turn by turn, against the DIRECT
// token-level Phase C path (the same encoded ids through
// Scheduler::admit_session_turn on an independent session):
//
//   * ENCODED TOKEN IDS: the facade's input ids == the raw native encode
//     of ONLY that turn's text (turn 2 does not re-encode turn 1);
//   * GENERATED TOKEN IDS: IDENTICAL on both paths for both turns (the
//     text path CONTINUES from turn 1's committed state — a replay would
//     restart the positions and break the RoPE-dependent logits);
//   * DECODED TEXT: the facade's text == the native decode of the token
//     path's generated ids (the facade's decode is the same contract);
//   * FINAL CONTEXT LENGTH: IDENTICAL on both paths (== input1 + gen1 +
//     input2 + gen2 — exact arithmetic on the committed turns);
//
// plus, on the text path: RESET (state zero, next turn works fresh),
// CONTEXT OVERFLOW admission = zero mutation on a real session, and
// clean teardown (pool accounting back to zero).
//
// Self-skips (77) when the checkpoint / tokenizer artifact is absent.
//
// Provenance: CUDALM-native (v0.8 Phase D).

#include "../../tests/common/check.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/qwen35_tokenizer.h"
#include "cudalm/scheduler.h"
#include "cudalm/session.h"
#include "cudalm/session_text_generator.h"
#include "cudalm/weight_loader_v2.h"

using namespace cudalm;

namespace {

bool file_exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

const int kPageTokens = 2;
const int kPoolPages = 48;
const int kDeltaSlots = 1;

const char* kT1 = "The quick brown fox jumps over the lazy dog";
const char* kT2 = "The lazy dog barks at the mailman";
const char* kT3 = "After reset the stream starts fresh";
const int kT1_M = 12;
const int kT2_M = 12;
const int kT3_M = 4;

std::vector<int> to_ids(const std::vector<std::uint32_t>& v) {
  std::vector<int> out;
  out.reserve(v.size());
  for (std::uint32_t t : v) out.push_back(static_cast<int>(t));
  return out;
}

std::string decode_ids(const Qwen35Tokenizer& tok,
                       const std::vector<int>& ids) {
  std::string out;
  std::vector<std::uint32_t> u;
  for (int t : ids) u.push_back(static_cast<std::uint32_t>(t));
  Status s = tok.decode(u.data(), u.size(), false, &out);
  if (!s.ok) {
    std::fprintf(stderr, "  [FAIL] decode: %s\n", s.message.c_str());
    std::exit(1);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr,
                 "usage: %s <full_model.cudalm> <checkpoint_dir> <python> "
                 "<src_dir> <tokenizer.cudaltk> [--no-convert]\n",
                 argv[0]);
    return 2;
  }
  const std::string out = argv[1];
  const std::string ckpt = argv[2];
  const std::string py = argv[3];
  const std::string src = argv[4];
  const std::string tk = argv[5];
  const bool no_convert = argc >= 7 && std::string(argv[6]) == "--no-convert";

  if (no_convert && !file_exists(out)) {
    std::fprintf(stderr, "[SKIP] qwen35 session text e2e: no preconverted "
                         "model at %s and --no-convert given\n",
                 out.c_str());
    return 77;
  }
  if (!file_exists(ckpt + "/model.safetensors")) {
    std::fprintf(stderr, "[SKIP] qwen35 session text e2e: checkpoint absent "
                         "at %s\n",
                 ckpt.c_str());
    return 77;
  }
  if (!file_exists(tk)) {
    std::fprintf(stderr, "[SKIP] qwen35 session text e2e: tokenizer artifact "
                         "absent at %s\n",
                 tk.c_str());
    return 77;
  }
  if (!file_exists(out)) {
    CHECK_EQ(system(("python3 " + src + "/tools/convert_qwen35.py" +
                     " --full-model --checkpoint-dir " + ckpt +
                     " --out " + out).c_str()),
             0);
  }

  WeightFileV2 file;
  Status s = WeightFileV2::load(out, &file);
  CHECK(s.ok);
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Qwen35Model model;
  s = Qwen35Model::load(file, stream, &model);
  CHECK(s.ok);
  CHECK(model.loaded());
  const Qwen35Config& cfg = model.config();

  std::unique_ptr<Qwen35Tokenizer> tok;
  s = Qwen35Tokenizer::load(tk, &tok);
  CHECK(s.ok);
  const int eos = static_cast<int>(tok->eos_token_id());
  std::fprintf(stderr,
               "[text e2e] model loaded: %d layers, vocab %d, max_seq_len "
               "%d; tokenizer loaded (eos %d)\n",
               model.num_layers(), cfg.vocab_size, cfg.max_seq_len, eos);

  const SamplingConfig kGreedy = SamplingConfig::greedy();

  // =========================================================================
  // PATH A — DIRECT token-level Phase C path: the raw encoded ids through
  // Scheduler::admit_session_turn on session A (independent manager).
  // =========================================================================
  std::vector<std::uint32_t> e1u, e2u, e3u;
  CHECK(tok->encode(kT1, &e1u).ok);
  CHECK(tok->encode(kT2, &e2u).ok);
  CHECK(tok->encode(kT3, &e3u).ok);
  const std::vector<int> enc1 = to_ids(e1u);
  const std::vector<int> enc2 = to_ids(e2u);
  const std::vector<int> enc3 = to_ids(e3u);
  CHECK(!enc1.empty() && !enc2.empty() && !enc3.empty());

  Qwen35StateManager mgrA(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  SessionManager smA(mgrA);
  ModelForwarder fwdA(model);
  Scheduler schedA(fwdA, mgrA, stream, &smA);
  SessionId sidA = 0;
  CHECK(smA.create_session(&sidA).ok);
  const SequenceId seqA = smA.lookup(sidA)->sequence_id;

  // =========================================================================
  // PATH B — TEXT facade: the same texts through
  // Qwen35SessionTextGenerator::generate_turn (independent manager).
  // =========================================================================
  Qwen35StateManager mgrB(cfg, kPageTokens, kPoolPages, kDeltaSlots, stream);
  SessionManager smB(mgrB);
  ModelForwarder fwdB(model);
  Qwen35SessionTextGenerator genB(fwdB, *tok, mgrB, smB, stream);
  SessionId sidB = 0;
  CHECK(genB.create_session(&sidB).ok);
  const SequenceId seqB = smB.lookup(sidB)->sequence_id;

  auto parity_turn = [&](const char* name, const SessionTextTurnResult& rb,
                         const Request* ra, int lenA,
                         const std::vector<int>& enc, int max_new) {
    if (!rb.ok) {
      std::fprintf(stderr, "  [FAIL] %s: text turn failed: %s\n", name,
                   rb.error.c_str());
      return 1;
    }
    if (ra == nullptr || ra->status != RequestStatus::Finished) {
      std::fprintf(stderr, "  [FAIL] %s: token-path request not Finished\n",
                   name);
      return 1;
    }
    if (rb.input_token_ids != enc) {
      std::fprintf(stderr, "  [FAIL] %s: input ids != raw encode (the text "
                           "facade modified the input)\n",
                   name);
      return 1;
    }
    if (rb.generated_token_ids != ra->generated) {
      std::fprintf(stderr, "  [FAIL] %s: generated token IDs DIFFER between "
                           "the text path and the token-level Phase C path "
                           "(text %zu / token %zu)\n",
                   name, rb.generated_token_ids.size(), ra->generated.size());
      return 1;
    }
    if (rb.generated_text != decode_ids(*tok, ra->generated)) {
      std::fprintf(stderr, "  [FAIL] %s: decoded text differs\n", name);
      return 1;
    }
    if (rb.context_length != lenA) {
      std::fprintf(stderr, "  [FAIL] %s: context length differs (text %d / "
                           "token %d)\n",
                   name, rb.context_length, lenA);
      return 1;
    }
    if (rb.stop_reason != ra->finish_reason) {
      std::fprintf(stderr, "  [FAIL] %s: stop reason differs\n", name);
      return 1;
    }
    (void)max_new;
    std::printf("  [ok] %s: input %zu ids | generated %zu ids | text %zu "
                "chars | context %d | stop %s — text path == token path\n",
                name, rb.input_token_ids.size(),
                rb.generated_token_ids.size(), rb.generated_text.size(),
                rb.context_length,
                ra->finish_reason == FinishReason::Eos ? "eos"
                : ra->finish_reason == FinishReason::MaxNewTokens
                    ? "max_new_tokens"
                    : "?");
    return 0;
  };

  // =========================================================================
  // TURN 1 (both paths, same eos gate, greedy)
  // =========================================================================
  {
    RequestId ridA = 0;
    Status as = schedA.admit_session_turn(sidA, enc1, kT1_M, eos, kGreedy,
                                          &ridA);
    CHECK(as.ok);
    schedA.run();
    const Request* reqA = schedA.get(ridA);
    CHECK(reqA != nullptr);
    const int lenA = mgrA.lookup(seqA)->length;

    SessionTextTurnResult rb =
        genB.generate_turn(sidB, kT1, kT1_M, kGreedy);  // default eos gate
    CHECK(parity_turn("turn 1", rb, reqA, lenA, enc1, kT1_M) == 0);
  }

  // =========================================================================
  // TURN 2 (same sessions — CONTINUATION, not a restart)
  // =========================================================================
  {
    RequestId ridA = 0;
    Status as = schedA.admit_session_turn(sidA, enc2, kT2_M, eos, kGreedy,
                                          &ridA);
    CHECK(as.ok);
    schedA.run();
    const Request* reqA = schedA.get(ridA);
    CHECK(reqA != nullptr);
    const int lenA = mgrA.lookup(seqA)->length;

    SessionTextTurnResult rb =
        genB.generate_turn(sidB, kT2, kT2_M, kGreedy);
    CHECK(parity_turn("turn 2", rb, reqA, lenA, enc2, kT2_M) == 0);
    // EXACT ARITHMETIC: the final context == both turns' committed ids.
    const int n1 = static_cast<int>(enc1.size());
    const int n2 = static_cast<int>(enc2.size());
    const int m1 = static_cast<int>(schedA.get(1)->generated.size());
    const int m2 = static_cast<int>(reqA->generated.size());
    CHECK_EQ(rb.context_length, n1 + m1 + n2 + m2);
    std::printf("  [ok] final context %d == n1(%d) + m1(%d) + n2(%d) + "
                "m2(%d) — turn 2 CONTINUED from the committed state (no "
                "replay)\n",
                rb.context_length, n1, m1, n2, m2);
  }

  // =========================================================================
  // RESET (text path): state zero under the same SessionId; next turn
  // works fresh
  // =========================================================================
  {
    CHECK(genB.reset_session(sidB).ok);
    CHECK_EQ(mgrB.lookup(seqB)->length, 0);
    SessionTextTurnResult r3 = genB.generate_turn(sidB, kT3, kT3_M, kGreedy);
    CHECK(r3.ok);
    CHECK_EQ(r3.context_length,
             static_cast<int>(enc3.size()) + kT3_M);
    std::printf("  [ok] reset: length 0 under the same session, post-reset "
                "turn committed %d ids fresh\n",
                static_cast<int>(enc3.size()) + kT3_M);
  }

  // =========================================================================
  // CONTEXT OVERFLOW (text path, real session): zero mutation
  // =========================================================================
  {
    // A dedicated manager (its own single Delta slot — the main session
    // B still holds its manager's only one).
    Qwen35StateManager mgrX(cfg, kPageTokens, kPoolPages, kDeltaSlots,
                            stream);
    SessionManager smX(mgrX);
    ModelForwarder fwdX(model);
    Qwen35SessionTextGenerator genX(fwdX, *tok, mgrX, smX, stream);
    const int reqs = genX.scheduler().num_requests();
    SessionId sidX = 0;
    CHECK(genX.create_session(&sidX).ok);
    const SequenceId seqX = smX.lookup(sidX)->sequence_id;
    const int n1 = static_cast<int>(enc1.size());
    CHECK(mgrX.set_length(seqX, cfg.max_seq_len - n1).ok);
    const int lenX = mgrX.lookup(seqX)->length;
    SessionTextTurnResult r =
        genX.generate_turn(sidX, kT1, 1, kGreedy, -1);
    CHECK(!r.ok);
    CHECK(r.error.find("context overflow") != std::string::npos);
    CHECK_EQ(r.request_id, static_cast<RequestId>(0));
    CHECK_EQ(mgrX.lookup(seqX)->length, lenX);
    CHECK_EQ(genX.scheduler().num_requests(), reqs);
    CHECK(genX.destroy_session(sidX).ok);
    CHECK_EQ(mgrX.kv_pool().used_pages(), 0);
    CHECK_EQ(mgrX.delta_pool().used_slots(), 0);
    std::printf("  [ok] overflow on a real session: rejected, zero mutation\n");
  }

  // =========================================================================
  // CLEAN TEARDOWN: destroy everything; pool accounting back to zero
  // =========================================================================
  CHECK(genB.destroy_session(sidB).ok);
  CHECK(smA.destroy_session(sidA).ok);
  CHECK_EQ(mgrA.kv_pool().used_pages(), 0);
  CHECK_EQ(mgrA.delta_pool().used_slots(), 0);
  CHECK_EQ(mgrB.kv_pool().used_pages(), 0);
  CHECK_EQ(mgrB.delta_pool().used_slots(), 0);
  CHECK_EQ(mgrA.num_live_sequences(), 0);
  CHECK_EQ(mgrB.num_live_sequences(), 0);
  std::printf("  [ok] clean teardown: both managers back to zero\n");

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("test_qwen35_session_text_e2e: PASS\n");
  return 0;
}
