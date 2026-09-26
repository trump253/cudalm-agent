// CUDALM — differential-validation dump tool for the native tokenizer.
//
// Loads a CUDLMTK1 artifact with the PyTorch-free C++ loader and dumps the
// native pipeline stages for hex-encoded inputs, so a shell/Python harness
// (tools/validate_qwen35_tokenizer.py) can compare each stage EXACTLY
// against the pinned tokenizers engine.
//
// Usage:
//   dump_qwen35_tokenizer <mode> <artifact.cudaltk> <input_file>
//
// Modes (one record per input line; output is line-aligned with the input):
//   E  <text_hex>            -> "<id,id,...>"      (final encode, EXACT)
//   X  <skip 0|1> <ids_csv>  -> "<hex>"            (decode, EXACT)
//   N  <cp,cp,...> (hex csv) -> "<cp,cp,...>"      (nfc_normalize stage)
//   P  <text_hex>            -> "hex\x1fhex\x1f..." (native pipeline order:
//                                 NFC first, then the pretokenize stage —
//                                 mirrors encode's internal stage order)
//
// The tool is deterministic and reads no assets beyond the artifact.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "cudalm/qwen35_tokenizer.h"

namespace {

std::string hex_of(const std::string& s) {
  static const char* d = "0123456789abcdef";
  std::string o;
  for (unsigned char c : s) {
    o += d[c >> 4];
    o += d[c & 15];
  }
  return o;
}

bool unhex(const std::string& h, std::string* out) {
  out->clear();
  auto val = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  if (h.size() % 2) return false;
  for (std::size_t i = 0; i + 1 < h.size(); i += 2) {
    int a = val(h[i]);
    int b = val(h[i + 1]);
    if (a < 0 || b < 0) return false;
    out->push_back(static_cast<char>((a << 4) | b));
  }
  return true;
}

bool parse_hex_csv(const std::string& line, std::vector<std::uint32_t>* out) {
  out->clear();
  std::size_t i = 0;
  while (i < line.size()) {
    std::size_t j = line.find(',', i);
    const std::string tok = line.substr(
        i, j == std::string::npos ? std::string::npos : j - i);
    if (!tok.empty()) out->push_back(std::stoul(tok, 0, 16));
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return true;
}

bool parse_dec_csv(const std::string& line, std::vector<std::uint32_t>* out) {
  out->clear();
  std::size_t i = 0;
  while (i < line.size()) {
    std::size_t j = line.find(',', i);
    const std::string tok = line.substr(
        i, j == std::string::npos ? std::string::npos : j - i);
    if (!tok.empty()) out->push_back(std::stoul(tok));
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <E|X|N|P> <artifact.cudaltk> <input_file>\n",
                 argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  const std::string artifact = argv[2];
  const std::string in_path = argv[3];
  std::vector<std::uint8_t> blob;
  {
    std::ifstream f(artifact, std::ios::binary);
    if (!f) return 3;
    blob.assign(std::istreambuf_iterator<char>(f),
                std::istreambuf_iterator<char>());
  }
  std::unique_ptr<cudalm::Qwen35Tokenizer> tk;
  if (!cudalm::Qwen35Tokenizer::load_blob(blob.data(), blob.size(), &tk).ok)
    return 3;

  std::ifstream in(in_path);
  if (!in) return 4;
  std::string line;
  int bad = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (mode == "E") {
      std::string text;
      if (!unhex(line, &text)) {
        ++bad;
        continue;
      }
      std::vector<std::uint32_t> ids;
      if (!tk->encode(text, &ids).ok) {
        ++bad;
        continue;
      }
      std::string o;
      for (auto id : ids) o += (o.empty() ? "" : ",") + std::to_string(id);
      std::printf("%s\n", o.c_str());
    } else if (mode == "N") {
      std::vector<std::uint32_t> cps, outc;
      parse_hex_csv(line, &cps);
      if (!tk->nfc_normalize(cps.data(), cps.size(), &outc).ok) {
        ++bad;
        continue;
      }
      std::string o;
      for (auto c : outc) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%x", c);
        if (!o.empty()) o += ",";
        o += buf;
      }
      std::printf("%s\n", o.c_str());
    } else if (mode == "P") {
      // Native pipeline order: NFC first, THEN pre-tokenization (the
      // pre-tokenizer stage inside encode always runs on NFC'd text).
      std::string text;
      if (!unhex(line, &text)) {
        ++bad;
        continue;
      }
      std::vector<std::uint32_t> cps, nfc;
      if (!tk->utf8_to_codepoints(text, &cps).ok ||
          !tk->nfc_normalize(cps.data(), cps.size(), &nfc).ok) {
        ++bad;
        continue;
      }
      const std::string nfc_text = cudalm::Qwen35Tokenizer::codepoints_to_utf8(
          nfc.data(), nfc.size());
      std::vector<std::string> pre;
      if (!tk->pretokenize(nfc_text, &pre).ok) {
        ++bad;
        continue;
      }
      std::string o;
      for (const auto& q : pre)
        o += (o.empty() ? "" : std::string("\x1f")) + hex_of(q);
      std::printf("%s\n", o.c_str());
    } else {  // X
      const auto sp = line.find(' ');
      const bool skip = (sp != std::string::npos &&
                         line.substr(0, sp) == "1");
      std::vector<std::uint32_t> ids;
      parse_dec_csv(sp == std::string::npos ? line : line.substr(sp + 1),
                    &ids);
      std::string out;
      if (!tk->decode(ids.data(), ids.size(), skip, &out).ok) {
        ++bad;
        continue;
      }
      std::printf("%s\n", hex_of(out).c_str());
    }
  }
  if (bad) {
    std::fprintf(stderr, "%d bad input lines\n", bad);
    return 1;
  }
  return 0;
}
