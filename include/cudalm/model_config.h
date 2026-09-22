// CUDALM — decoder-block model configuration.
//
// Fixed-width, trivially-serializable value type shared by the weight format
// (header), the loader, and the ops. Provenance: CUDALM-native.

#pragma once

#include <cstdint>

namespace cudalm {

struct ModelConfig {
  // Dimensions
  std::int32_t hidden_size = 0;       // H
  std::int32_t n_heads = 0;           // query heads
  std::int32_t n_kv_heads = 0;        // KV heads (GQA: n_heads % n_kv_heads == 0)
  std::int32_t head_dim = 0;          // d per head
  std::int32_t intermediate_size = 0; // SwiGLU hidden
  std::int32_t group_size = 128;      // W4A16 quant group (fixed 128 in v0.1)
  std::int32_t max_seq_len = 0;

  // Hyperparameters
  float eps = 1e-5f;        // RMSNorm
  float rope_theta = 10000.0f;

  // ---- Derived helpers -------------------------------------------------
  int q_proj_out() const { return n_heads * head_dim; }
  int kv_proj_out() const { return n_kv_heads * head_dim; }
  // Number of query heads that share one KV head (GQA group width).
  int gqa_group() const { return (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1; }
  // KV head used by query head h.
  int kv_head_for_q(int h) const { return (gqa_group() > 0) ? (h / gqa_group()) : h; }
  // RoPE table row length = head_dim / 2 (interleaved pairs).
  int rope_pairs() const { return head_dim / 2; }

  bool valid() const {
    if (hidden_size <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
      return false;
    if (intermediate_size <= 0 || max_seq_len <= 0) return false;
    if (n_heads % n_kv_heads != 0) return false;        // GQA must divide
    if (hidden_size % head_dim != 0) return false;
    if (hidden_size % group_size != 0) return false;    // W4A16 K multiple
    if (intermediate_size % group_size != 0) return false;
    if (group_size <= 0) return false;
    if (head_dim % 2 != 0) return false;                // interleaved pairs
    return true;
  }

  // The v0.1 fixed test config (see docs/bootstrap_plan_v0.1.md §3).
  static ModelConfig v01_default() {
    ModelConfig c;
    c.hidden_size = 1024;
    c.n_heads = 8;
    c.n_kv_heads = 4;
    c.head_dim = 128;
    c.intermediate_size = 2816;  // 22 * 128
    c.group_size = 128;
    c.max_seq_len = 512;
    c.eps = 1e-5f;
    c.rope_theta = 10000.0f;
    return c;
  }
};

}  // namespace cudalm
