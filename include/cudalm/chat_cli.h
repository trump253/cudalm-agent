// CUDALM — `cudalm-chat` CLI line contract (v0.8 Phase D review fix).
//
// Pinned REPL contract (the external-review CLI contract blocker):
//   * EXACT WHOLE-LINE command matching: ONLY the exact strings "quit"
//     and "exit" quit, and ONLY the exact string "reset" resets. A line
//     with any surrounding/interior whitespace (" reset ", "  reset")
//     is RAW TEXT, not a command — user input is NEVER trimmed.
//   * A truly EMPTY line ("" — nothing between the prompt and the
//     newline) is ignored (documented as "empty line is ignored").
//   * A whitespace-only NON-EMPTY line ("   ") is RAW TEXT (a normal
//     turn), exactly like any other text.
//   * Every non-command line is passed VERBATIM (byte-for-byte) to
//     Qwen35SessionTextGenerator::generate_turn — no spaces / newlines
//     / separators / special tokens are added or removed.
//   * BINARY-SAFE display: generated text may contain embedded NUL
//     bytes, so it must be written with a LENGTH-AWARE write (never
//     %s). The display-only newline the CLI appends when the response
//     does not end in '\n' is UX only — it never enters the session /
//     token history.

#pragma once

#include <cstddef>
#include <cstdio>
#include <string>

namespace cudalm {

// The classification of one REPL line under the pinned contract.
enum class ChatLineKind {
  Ignored,  // ONLY the truly empty line ""
  Reset,    // exactly "reset"
  Quit,     // exactly "quit" or "exit"
  Text,     // everything else — passed VERBATIM to generate_turn
};

inline ChatLineKind classify_chat_line(const std::string& line) {
  if (line.empty()) return ChatLineKind::Ignored;
  if (line == "reset") return ChatLineKind::Reset;
  if (line == "quit" || line == "exit") return ChatLineKind::Quit;
  return ChatLineKind::Text;
}

// Length-aware (binary-safe) write: writes EXACTLY `size` bytes (embedded
// NULs included). Never use %s on generated text.
inline void write_binary_safe(const char* data, std::size_t size,
                              std::FILE* out) {
  if (size == 0) return;
  std::fwrite(data, 1, size, out);
}

}  // namespace cudalm
