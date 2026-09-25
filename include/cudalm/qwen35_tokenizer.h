// CUDALM — Qwen3.5-0.8B-Base native tokenizer (PyTorch-free).
//
// Contract (docs/qwen35_architecture.md §18):
//   raw UTF-8 prompt
//     -> NFC normalize            (UAX #15; oracle-verified, see below)
//     -> added-token split        (33 pinned added tokens, matched first)
//     -> regex pre-tokenization   (leftmost-first, PCRE/Onig semantics)
//     -> ByteLevel BPE            (248044-token base vocab, 247587 merges)
//     -> token ids
//   decode: token ids -> COMPLETE byte stream (raw ByteLevel bytes for base
//   tokens, literal UTF-8 for added tokens, padding ids 248077..248319
//   contribute nothing — exactly like the pinned HuggingFace oracle) ->
//   lossy UTF-8 conversion of the WHOLE stream (each maximal invalid
//   subpart -> one U+FFFD; a legal multi-byte character may be split across
//   several base tokens and is still decoded as one character).  The output
//   is always valid UTF-8.
//
// The whole pipeline is transcribed offline by
// tools/convert_qwen35_tokenizer.py into the CUDLMTK1 artifact (vocab,
// merges, added tokens, NFC tables, \s/\pL/\pN/\pM range tables) which this
// class loads with full bounds-checking and fails loud on any violation.
//
// NFC oracle note: the pinned engine's NFC is an EXACT port of the
// Rust `unicode-normalization` streaming algorithm used by the tokenizers
// crate (Decompositions + Recompositions iterators):  (1) full canonical
// decomposition (recursive);  (2) canonical ordering in batches — a ccc == 0
// code point (or end of input) triggers a STABLE sort of the not-yet-emitted
// tail by ascending ccc (ccc == 0 is a starter / sequence boundary; jamo are
// ccc 0 in the tables, so jamo sequences are never reordered);  (3)
// composition integrated in the same pass — the composee composes with an
// incoming non-starter only while every buffered (delayed) mark has a
// STRICTLY SMALLER ccc, otherwise the mark is buffered and emitted later
// (exact composition table, exclusions included).  Observable consequence:
// "U+0391 U+0301 U+093C" (ccc 0, 230, 7) -> "U+0386 U+093C" (the 230 mark
// composes over the delayed 7 mark) — a case where pre-sorting the whole
// string first (as Python's unicodedata does) gives a DIFFERENT result; the
// pinned engine is the oracle, not Python.  Known data caveat: the tables are
// derived from Python U14 Unicode data while the engine's are Unicode 9.0
// (crate constant UNICODE_VERSION=(9,0,0)); differential validation found 98
// cps with U14 ccc > 0 but engine ccc 0 (e.g. U+1715) and one U13-added
// composition/decomposition pair (U+11935+U+11930 <-> U+11938, Dives Akuru)
// — reported as open blocker BLOCKER-D1 (docs/qwen35_architecture.md §20).
//
// No special/control token string literals appear in this file or its
// implementation: added tokens are carried by the artifact only.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cudalm/weight_format.h"

namespace cudalm {

class Qwen35Tokenizer {
 public:
  // Pinned contract constants (Qwen/Qwen3.5-0.8B-Base @ dc7cdfe).
  static constexpr std::uint32_t kBaseVocabSize = 248044;  // ids 0..248043
  static constexpr std::uint32_t kModelVocabSize = 248320; // config vocab_size
  static constexpr std::uint32_t kFirstAddedId = 248044;
  static constexpr std::uint32_t kNumAdded = 33;           // 248044..248076
  static constexpr std::uint32_t kEosTokenId = 248044;     // single EOS

  // Load the CUDLMTK1 artifact from disk.  Fails loud (Status.ok == false)
  // on any format/contract violation; never throws.
  static Status load(const std::string &path,
                     std::unique_ptr<Qwen35Tokenizer> *out);
  // Load from an in-memory blob (tests: corruption cases).
  static Status load_blob(const std::uint8_t *data, std::size_t len,
                          std::unique_ptr<Qwen35Tokenizer> *out);

  // raw UTF-8 text -> token ids (HF semantics: add_special_tokens=false).
  // Fails loud on invalid UTF-8.
  Status encode(const std::string &utf8_text,
                std::vector<std::uint32_t> *out) const;

  // token ids -> UTF-8 text.  skip_special_tokens matches the pinned HF
  // oracle (removes the 21 special added tokens; non-special added tokens
  // are always decoded).  ids outside [0, kModelVocabSize) fail loud.
  Status decode(const std::uint32_t *ids, std::size_t n,
                bool skip_special_tokens, std::string *out_utf8) const;

  std::uint32_t base_vocab_size() const { return base_vocab_size_; }
  std::uint32_t model_vocab_size() const { return model_vocab_size_; }
  std::uint32_t eos_token_id() const { return eos_token_id_; }
  bool is_special(std::uint32_t id) const;

  // ------------------------------------------------------------------
  // Pipeline stages, exposed for unit tests (corpus cross-checks).
  // ------------------------------------------------------------------
  // UTF-8 -> code points (fails loud on invalid sequences / overlongs).
  Status utf8_to_codepoints(const std::string &utf8,
                            std::vector<std::uint32_t> *out) const;
  static std::string codepoints_to_utf8(const std::uint32_t *cps,
                                        std::size_t n);
  // NFC (the oracle-verified algorithm above).  cps must be valid code
  // points (<= 0x10FFFF, no surrogates); fails loud otherwise.
  Status nfc_normalize(const std::uint32_t *cps, std::size_t n,
                       std::vector<std::uint32_t> *out) const;
  // Regex pre-tokenization of one added-token-free NFC chunk (fails loud on
  // invalid UTF-8; every valid code point is matchable, so a no-match is an
  // internal error, not a silent fallback).
  Status pretokenize(const std::string &nfc_utf8,
                     std::vector<std::string> *out) const;
  // BPE of one pre-token (UTF-8) -> base-vocab token ids.
  Status bpe_encode(const std::string &pre_token_utf8,
                    std::vector<std::uint32_t> *out) const;

 private:
  Qwen35Tokenizer() = default;
  Status init_from_blob(const std::uint8_t *data, std::size_t len);

  // --- artifact data ---------------------------------------------------
  std::uint32_t base_vocab_size_ = 0;
  std::uint32_t model_vocab_size_ = 0;
  std::uint32_t first_added_id_ = 0;
  std::uint32_t eos_token_id_ = 0;

  std::vector<std::string> base_vocab_;                 // [id] -> bytes
  std::vector<std::pair<std::uint32_t, std::uint32_t>> merges_;  // rank=i
  struct AddedToken {
    std::uint32_t id;
    bool special;
    std::string utf8;
  };
  std::vector<AddedToken> added_;                       // ascending id

  // NFC tables.
  std::vector<std::uint8_t> ccc_;                       // dense, size 0x110000
  struct DecompEntry {
    std::uint32_t cp;
    std::vector<std::uint32_t> parts;
  };
  std::vector<DecompEntry> decomp_;                     // ascending cp
  std::vector<std::pair<std::uint64_t, std::uint32_t>> comp_;  // (a<<21)|b -> c
  // Range tables (lo, hi inclusive), ascending, non-overlapping.
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ws_ranges_;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> letter_ranges_;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> number_ranges_;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> mark_ranges_;

  // --- derived (built in init_from_blob, immutable afterwards) ---------
  std::vector<bool> is_letter_, is_number_, is_mark_, is_ws_;  // 0x110000
  std::vector<std::uint8_t> byte_to_base_id_;    // 256 -> id (all present)
  std::unordered_map<std::string, std::uint32_t> bytes_to_id_;  // token bytes
  // (left_id, right_id) -> merge rank (parallel sorted arrays).
  std::vector<std::uint64_t> merge_rank_sorted_;
  std::vector<std::uint32_t> merge_rank_values_;

  Status nfc_impl(const std::uint32_t *cps, std::size_t n,
                  std::vector<std::uint32_t> *out) const;
  bool in_ranges(const std::vector<std::pair<std::uint32_t, std::uint32_t>> &r,
                 std::uint32_t cp) const;
};

}  // namespace cudalm
