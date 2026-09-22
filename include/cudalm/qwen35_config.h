// CUDALM — Qwen3.5 hybrid decoder configuration (v0.2).
//
// Fixed-width, trivially-serializable value type (blob layout in
// weight_format_v2.h), shared by the .cudalm v2 header, the loader, and the
// Qwen3.5 layer classes. Pinned contract: docs/qwen35_architecture.md
// (official config.json of Qwen/Qwen3.5-0.8B-Base @ dc7cdfe2 + transformers
// modeling @ fc9137225880).
//
// v0.1.1 ModelConfig is NOT modified; this is an independent type.

#pragma once

#include <cstdint>

namespace cudalm {

struct Qwen35Config {
  // ---- Core text-model dimensions (official text_config) ----------------
  std::int32_t hidden_size = 0;          // H
  std::int32_t num_hidden_layers = 0;    // 24
  std::int32_t intermediate_size = 0;    // SwiGLU hidden
  std::int32_t vocab_size = 0;           // 248320 (not used by v0.2 runtime)

  // ---- Full-attention dimensions (layers where (i+1) % interval == 0) ----
  std::int32_t n_heads = 0;              // query heads (8)
  std::int32_t n_kv_heads = 0;           // KV heads (2), GQA
  std::int32_t head_dim = 0;             // 256 (q_proj out per q-head; x2 for gate)

  // ---- Gated DeltaNet dimensions (the other layers) ----------------------
  std::int32_t lin_num_k_heads = 0;      // 16
  std::int32_t lin_num_v_heads = 0;      // 16
  std::int32_t lin_key_head_dim = 0;     // 128
  std::int32_t lin_value_head_dim = 0;   // 128
  std::int32_t lin_conv_kernel_dim = 0;  // 4 (depthwise causal conv)

  // ---- Hybrid schedule ----------------------------------------------------
  // layer i is full_attention iff (i + 1) % full_attention_interval == 0
  // (official: layer_types[i] = (i+1) % full_attention_interval ? linear : full)
  std::int32_t full_attention_interval = 0;  // 4

  // ---- Quantization / sequence -------------------------------------------
  std::int32_t group_size = 128;        // W4A16 quant group (CUDALM core contract)
  std::int32_t max_seq_len = 0;

  // ---- Hyperparameters ----------------------------------------------------
  float eps = 0.0f;                     // rms_norm_eps (1e-6)
  float rope_theta = 0.0f;              // 1e7
  float partial_rotary_factor = 0.0f;   // 0.25 -> rotary_dim = head_dim/4
  std::int32_t mrope_section[3] = {0, 0, 0};  // [11, 11, 10] (text: no-op, see doc §5)

  // ---- Derived helpers ----------------------------------------------------
  // q_proj output width: n_heads * head_dim * 2 (fused [query; gate] per head).
  // NOTE: NOT equal to hidden_size (8*256*2 = 4096 != 1024) — v0.2 keeps the
  // v0.1.1 generalization that Q width != H is legal.
  int q_proj_out() const { return n_heads * head_dim * 2; }
  int kv_proj_out() const { return n_kv_heads * head_dim; }
  int o_proj_in() const { return n_heads * head_dim; }
  int gqa_group() const { return (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1; }
  int kv_head_for_q(int h) const { return (gqa_group() > 0) ? (h / gqa_group()) : h; }

  // Partial rotary dimension applied to the FIRST rotary_dim channels of each
  // head (rotate-half layout; the remaining head_dim - rotary_dim pass through).
  int rotary_dim() const {
    return static_cast<int>(static_cast<float>(head_dim) * partial_rotary_factor + 0.5f);
  }

  // Gated DeltaNet dimensions.
  int linear_key_dim() const { return lin_num_k_heads * lin_key_head_dim; }   // 2048
  int linear_value_dim() const { return lin_num_v_heads * lin_value_head_dim; } // 2048
  int linear_conv_dim() const { return 2 * linear_key_dim() + linear_value_dim(); } // 6144
  // conv state holds the last (kernel - 1) tokens (official update math).
  int linear_conv_state_len() const { return lin_conv_kernel_dim - 1; }

  // Layer-type schedule (mirrors official Qwen3_5TextConfig default).
  bool is_full_attention(int i) const {
    return i >= 0 && i < num_hidden_layers &&
           full_attention_interval > 0 && (i + 1) % full_attention_interval == 0;
  }
  bool is_linear_attention(int i) const { return !is_full_attention(i); }

  bool valid() const {
    if (hidden_size <= 0 || num_hidden_layers <= 0 || intermediate_size <= 0)
      return false;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) return false;
    if (n_heads % n_kv_heads != 0) return false;             // GQA must divide
    if (lin_num_k_heads <= 0 || lin_num_v_heads <= 0) return false;
    if (lin_num_v_heads % lin_num_k_heads != 0) return false; // repeat_interleave
    if (lin_key_head_dim <= 0 || lin_value_head_dim <= 0) return false;
    if (lin_conv_kernel_dim < 2) return false;
    if (full_attention_interval < 2) return false;
    if (group_size <= 0 || max_seq_len <= 0) return false;
    if (eps <= 0.0f || rope_theta <= 0.0f) return false;
    // Partial rotary: even rotary dim, within head, and the mrope sections
    // (3 of them) must exactly tile rotary_dim/2 (official layout contract).
    const int rd = rotary_dim();
    if (rd <= 0 || rd > head_dim || rd % 2 != 0) return false;
    const int sec_sum = mrope_section[0] + mrope_section[1] + mrope_section[2];
    if (sec_sum != rd / 2) return false;
    // W4A16 GEMV: every GEMV K must be a multiple of group_size (128).
    // K values: H (all in-projs), o_proj_in, intermediate (down),
    // linear_value_dim (deltanet out_proj).
    if (hidden_size % group_size != 0) return false;
    if (o_proj_in() % group_size != 0) return false;
    if (intermediate_size % group_size != 0) return false;
    if (linear_value_dim() % group_size != 0) return false;
    return true;
  }

  bool operator==(const Qwen35Config& o) const {
    return hidden_size == o.hidden_size &&
           num_hidden_layers == o.num_hidden_layers &&
           intermediate_size == o.intermediate_size && vocab_size == o.vocab_size &&
           n_heads == o.n_heads && n_kv_heads == o.n_kv_heads && head_dim == o.head_dim &&
           lin_num_k_heads == o.lin_num_k_heads &&
           lin_num_v_heads == o.lin_num_v_heads &&
           lin_key_head_dim == o.lin_key_head_dim &&
           lin_value_head_dim == o.lin_value_head_dim &&
           lin_conv_kernel_dim == o.lin_conv_kernel_dim &&
           full_attention_interval == o.full_attention_interval &&
           group_size == o.group_size && max_seq_len == o.max_seq_len &&
           eps == o.eps && rope_theta == o.rope_theta &&
           partial_rotary_factor == o.partial_rotary_factor &&
           mrope_section[0] == o.mrope_section[0] &&
           mrope_section[1] == o.mrope_section[1] &&
           mrope_section[2] == o.mrope_section[2];
  }
  bool operator!=(const Qwen35Config& o) const { return !(*this == o); }

  // Pinned Qwen3.5-0.8B text config (docs/qwen35_architecture.md §2).
  static Qwen35Config qwen35_08b() {
    Qwen35Config c;
    c.hidden_size = 1024;
    c.num_hidden_layers = 24;
    c.intermediate_size = 3584;
    c.vocab_size = 248320;
    c.n_heads = 8;
    c.n_kv_heads = 2;
    c.head_dim = 256;
    c.lin_num_k_heads = 16;
    c.lin_num_v_heads = 16;
    c.lin_key_head_dim = 128;
    c.lin_value_head_dim = 128;
    c.lin_conv_kernel_dim = 4;
    c.full_attention_interval = 4;
    c.group_size = 128;
    c.max_seq_len = 262144;
    c.eps = 1e-6f;
    c.rope_theta = 1e7f;
    c.partial_rotary_factor = 0.25f;
    c.mrope_section[0] = 11;
    c.mrope_section[1] = 11;
    c.mrope_section[2] = 10;
    return c;
  }
};

}  // namespace cudalm
