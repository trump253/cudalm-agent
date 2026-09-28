// CUDALM — v0.8 Phase D review fix: cudalm-chat CLI line contract gate
// (tiny CPU unit test — NO model, NO GPU; check.h only, no framework).
//
// Pins the CLI contract blocker fix (include/cudalm/chat_cli.h):
//   * EXACT WHOLE-LINE command matching: only the exact strings
//     "reset" / "quit" / "exit" are commands; " reset " (with
//     surrounding spaces), "reset ", " RESET" etc. are RAW TEXT —
//     user input is never trimmed;
//   * ONLY the truly empty line is ignored; a whitespace-only
//     NON-EMPTY line is RAW TEXT;
//   * BINARY-SAFE output: write_binary_safe writes the exact byte
//     count including embedded NULs.
//
// Provenance: CUDALM-native (v0.8 Phase D review fix).

#include "../../tests/common/check.h"

#include <cstdio>
#include <string>

#include "cudalm/chat_cli.h"

using namespace cudalm;

int main() {
  std::printf("test_cudalm_chat_cli: CLI line contract gate\n");

  // ---- EXACT WHOLE-LINE command matching --------------------------------
  CHECK(classify_chat_line("") == ChatLineKind::Ignored);  // only ""
  CHECK(classify_chat_line("reset") == ChatLineKind::Reset);
  CHECK(classify_chat_line("quit") == ChatLineKind::Quit);
  CHECK(classify_chat_line("exit") == ChatLineKind::Quit);
  // NOT commands — raw text (no trimming, no case folding):
  CHECK(classify_chat_line(" reset ") == ChatLineKind::Text);
  CHECK(classify_chat_line(" reset") == ChatLineKind::Text);
  CHECK(classify_chat_line("reset ") == ChatLineKind::Text);
  CHECK(classify_chat_line("  reset") == ChatLineKind::Text);
  CHECK(classify_chat_line(" RESET") == ChatLineKind::Text);
  CHECK(classify_chat_line("Reset") == ChatLineKind::Text);
  CHECK(classify_chat_line(" quit") == ChatLineKind::Text);
  CHECK(classify_chat_line("quit ") == ChatLineKind::Text);
  CHECK(classify_chat_line(" quit\n") == ChatLineKind::Text);
  // Whitespace-only NON-EMPTY lines are raw text (a turn):
  CHECK(classify_chat_line("   ") == ChatLineKind::Text);
  CHECK(classify_chat_line("\t") == ChatLineKind::Text);
  CHECK(classify_chat_line(" \t ") == ChatLineKind::Text);
  // Ordinary text:
  CHECK(classify_chat_line("hello") == ChatLineKind::Text);
  std::printf("  [ok] exact whole-line command matching\n");

  // ---- BINARY-SAFE output (embedded NULs byte-exact) ---------------------
  const char* const path = "/tmp/cudalm_chat_cli_binary_safe.out";
  FILE* f = std::fopen(path, "wb");
  CHECK(f != nullptr);
  const std::string blob("a\0b\0c", 5);  // 5 bytes, two embedded NULs
  write_binary_safe(blob.data(), blob.size(), f);
  std::fclose(f);
  FILE* g = std::fopen(path, "rb");
  CHECK(g != nullptr);
  unsigned char buf[16] = {};
  const std::size_t got = std::fread(buf, 1, sizeof(buf), g);
  std::fclose(g);
  std::remove(path);
  CHECK_EQ(static_cast<int>(got), 5);
  CHECK(buf[0] == 'a' && buf[1] == 0 && buf[2] == 'b' && buf[3] == 0 &&
        buf[4] == 'c');
  // Empty write: nothing written.
  FILE* h = std::fopen(path, "wb");
  CHECK(h != nullptr);
  write_binary_safe("ignored", 0, h);
  std::fclose(h);
  FILE* i = std::fopen(path, "rb");
  CHECK(i != nullptr);
  const int c = std::fgetc(i);
  std::fclose(i);
  std::remove(path);
  CHECK_EQ(c, EOF);
  std::printf("  [ok] binary-safe output (embedded NULs byte-exact)\n");

  std::printf("test_cudalm_chat_cli: PASS\n");
  return 0;
}
