// CUDALM — Qwen3.5 full model runtime implementation (v0.3: Phase A skeleton
// + Phase B full single-token forward).
// Reuses the frozen v0.2 single-layer runtimes (Qwen35FullAttentionLayer /
// Qwen35DeltaNetLayer); adds the model-level ownership (embedding, final
// norm, tied LM head), the per-layer dispatch over all 24 layers, the
// whole-model state reset, and the complete single-token forward
// (forward_token: embedding -> 24 layers -> final RMSNorm -> tied LM head ->
// logits, docs §18). The only NEW kernel is the tied-LM-head bf16 GEMV
// (kernels/bf16_gemv); no v0.2 single-layer kernel is copied.

#include "cudalm/qwen35_model.h"

#include <algorithm>
#include <string>

#include "cudalm/cuda_check.h"
#include "cudalm/kernels/batch_decode.h"
#include "cudalm/kernels/bf16_gemv.h"
#include "cudalm/kernels/qwen35_kernels.h"
#include "cudalm/qwen35_kv_cache.h"
#include "cudalm/qwen35_kv_page_pool.h"

namespace cudalm {

using kernels::batch_bf16_gemv;  // v0.6 Phase B batch GEMV (kernels namespace)

Status Qwen35Model::load(const WeightFileV2& file, cudaStream_t stream,
                         Qwen35Model* out) {
  out->cfg_ = file.config();
  out->loaded_ = false;
  out->tie_ = false;
  out->embed_ = TensorView();
  out->norm_ = TensorView();
  out->lm_head_ = TensorView();
  out->weights_.clear();
  out->layers_.clear();
  out->embed_buf_.reset();
  out->norm_buf_.reset();
  out->embed_out_.reset();
  out->norm_out_.reset();
  out->logits_buf_.reset();

  // Full-model tensor set: embedding + final norm + every layer + tie.
  Status s = file.validate_full_model();
  if (!s.ok) return s;
  // validate_full_model() guarantees tie_word_embeddings == "true", so the LM
  // head aliases the embedding (no separate tensor).
  out->tie_ = true;

  // Embedding: [vocab_size, hidden_size] bf16.
  {
    const Qwen35TensorInfo* info = file.find("embed_tokens.weight");
    out->embed_buf_.allocate(info->byte_size, stream);
    out->embed_buf_.copy_from_host(info->host_bytes, info->byte_size, stream);
    out->embed_ = TensorView(out->embed_buf_.data(), info->dtype, info->dims);
  }
  // Final norm: [hidden_size] bf16.
  {
    const Qwen35TensorInfo* info = file.find("norm.weight");
    out->norm_buf_.allocate(info->byte_size, stream);
    out->norm_buf_.copy_from_host(info->host_bytes, info->byte_size, stream);
    out->norm_ = TensorView(out->norm_buf_.data(), info->dtype, info->dims);
  }
  // LM head: TIED to the embedding — alias the same buffer (no separate
  // allocation). Used transposed: logits = hidden @ lm_head^T.
  out->lm_head_ = out->embed_;

  // Per-layer weight sets + runtimes, dispatched by the hybrid schedule.
  const int n = out->cfg_.num_hidden_layers;
  out->weights_.reserve(n);
  out->layers_.reserve(n);
  for (int i = 0; i < n; ++i) {
    auto w = std::make_unique<Qwen35LayerWeights>();
    s = Qwen35LayerWeights::load(file, i, stream, w.get());
    if (!s.ok) {
      s.message = "layer " + std::to_string(i) + ": " + s.message;
      return s;
    }
    out->weights_.push_back(std::move(w));
    // *weights_.back() is stable: the Qwen35LayerWeights object is heap
    // owned; push_back moves only the unique_ptr (and we reserved n).
    if (out->cfg_.is_linear_attention(i)) {
      out->layers_.emplace_back(
          std::unique_ptr<Qwen35DeltaNetLayer>(
              new Qwen35DeltaNetLayer(*out->weights_.back(), stream)));
    } else {
      out->layers_.emplace_back(
          std::unique_ptr<Qwen35FullAttentionLayer>(
              new Qwen35FullAttentionLayer(*out->weights_.back(), stream)));
    }
  }
  // Forward scratch + output buffers (sized by config): embedding output
  // [hidden], final-norm output [hidden], logits [vocab].
  const int H = out->cfg_.hidden_size;
  const int V = out->cfg_.vocab_size;
  const std::size_t bf16 = sizeof(__nv_bfloat16);
  out->embed_out_.allocate(static_cast<std::size_t>(H) * bf16, stream);
  out->norm_out_.allocate(static_cast<std::size_t>(H) * bf16, stream);
  out->logits_buf_.allocate(static_cast<std::size_t>(V) * bf16, stream);
  out->loaded_ = true;
  return Status::ok_status();
}

const Qwen35LayerWeights* Qwen35Model::layer_weights(int i) const {
  if (i < 0 || i >= num_layers()) return nullptr;
  return weights_[static_cast<std::size_t>(i)].get();
}

Qwen35DeltaNetLayer* Qwen35Model::delta(int i) const {
  if (i < 0 || i >= num_layers() || !is_linear_attention(i)) return nullptr;
  return std::get<0>(layers_[static_cast<std::size_t>(i)]).get();
}

Qwen35FullAttentionLayer* Qwen35Model::attention(int i) const {
  if (i < 0 || i >= num_layers() || !is_full_attention(i)) return nullptr;
  return std::get<1>(layers_[static_cast<std::size_t>(i)]).get();
}

void Qwen35Model::reset_state(cudaStream_t stream) {
  CUDALM_PRECONDITION(loaded_, "Qwen35Model: not loaded");
  for (int i = 0; i < num_layers(); ++i) {
    if (is_linear_attention(i)) {
      // DeltaNet: zero conv_state + recurrent_state (the layer owns them).
      delta(i)->reset_state(stream);
    } else {
      // Full attention: zero the KV cache (both tensors). The cache is
      // [n_kv][max_seq][head_dim]; bytes() is ONE tensor's size.
      Qwen35KvCache& kv = attention(i)->kv_cache();
      CUDA_CHECK(cudaMemsetAsync(kv.k_mut(), 0, kv.bytes(), stream));
      CUDA_CHECK(cudaMemsetAsync(kv.v_mut(), 0, kv.bytes(), stream));
    }
  }
}

void Qwen35Model::forward_token(int token_id, int position,
                                cudaStream_t stream) {
  CUDALM_PRECONDITION(loaded_, "Qwen35Model::forward_token: not loaded");
  CUDALM_PRECONDITION(
      token_id >= 0 && token_id < cfg_.vocab_size,
      "Qwen35Model::forward_token: token_id out of range");
  CUDALM_PRECONDITION(
      position >= 0 && position < cfg_.max_seq_len,
      "Qwen35Model::forward_token: position out of range");

  const int H = cfg_.hidden_size;

  // 1. Embedding lookup: embed_tokens.weight[token_id] (bf16 [hidden]).
  //    nn.Embedding is a pure index copy — no arithmetic, no rounding
  //    (docs §18) — so a D2D row copy is bit-exact.
  const __nv_bfloat16* emb_row =
      embed_.data<__nv_bfloat16>() +
      static_cast<std::size_t>(token_id) * static_cast<std::size_t>(H);
  CUDA_CHECK(cudaMemcpyAsync(
      embed_out_.data(), emb_row,
      static_cast<std::size_t>(H) * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToDevice, stream));
  const __nv_bfloat16* x = embed_out_.data<__nv_bfloat16>();

  // 2. Chain all 24 decoder layers (frozen v0.2 runtimes; each updates its own
  //    persistent state in place, so a sequential forward threads state).
  for (int i = 0; i < num_layers(); ++i) {
    if (is_linear_attention(i)) {
      delta(i)->forward(position, x, stream);
      x = delta(i)->stage_final_output();
    } else {
      attention(i)->forward(position, x, stream);
      x = attention(i)->stage_final_output();
    }
  }

  // 3. Final zero-centered RMSNorm (fp32 chain, one bf16 RNE; docs §18).
  kernels::qwen35_rmsnorm_zc_bf16(x, norm_.data<__nv_bfloat16>(),
                                  norm_out_.data<__nv_bfloat16>(),
                                  /*M=*/1, /*H=*/H, cfg_.eps, stream);
  const __nv_bfloat16* normed = norm_out_.data<__nv_bfloat16>();

  // 4. Tied LM head: logits = normed @ embedding^T (bf16 GEMV, fp32 acc; the
  //    LM head ALIASES the embedding weight — no separate copy).
  bf16_gemv(embed_.data<__nv_bfloat16>(), normed,
            logits_buf_.data<__nv_bfloat16>(), /*N=*/cfg_.vocab_size,
            /*K=*/H, stream);
}

Status Qwen35Model::forward_token_with_state(int token_id, SequenceId seq_id,
                                             Qwen35StateManager& mgr,
                                             cudaStream_t stream) {
  if (!loaded_)
    return Status::error("Qwen35Model::forward_token_with_state: not loaded");
  if (token_id < 0 || token_id >= cfg_.vocab_size)
    return Status::error("Qwen35Model::forward_token_with_state: token_id " +
                         std::to_string(token_id) + " out of range [0, " +
                         std::to_string(cfg_.vocab_size) + ")");
  // COMPATIBILITY GATE (before ANY state mutation — no KV allocation, no
  // Delta mutation, length unchanged, no layer forward): the manager's
  // config must be EXACTLY the model's config. A mismatched manager would
  // address pool pages / delta slots with the wrong layout, so it is
  // rejected fail-loud up front.
  if (!(mgr.config() == cfg_))
    return Status::error(
        "Qwen35Model::forward_token_with_state: manager config does not "
        "match the model config");
  // SINGLE-STREAM CONTRACT (Phase A/B; v0.5 is single-stream,
  // correctness-first — no cross-stream event machinery): every pool
  // access happens on the pools' streams, so the caller's stream must be
  // exactly the KV pool's and the Delta pool's stream.
  if (stream != mgr.kv_pool().stream() || stream != mgr.delta_pool().stream())
    return Status::error(
        "Qwen35Model::forward_token_with_state: stream mismatch — v0.5 "
        "is single-stream; the forward stream must equal the manager "
        "pool streams");
  const SequenceState* rec = mgr.lookup(seq_id);
  if (rec == nullptr)
    return Status::error(
        "Qwen35Model::forward_token_with_state: sequence id " +
        std::to_string(seq_id) + " is not a live sequence");
  // position is DERIVED from the sequence length (single source of truth).
  const int position = rec->length;
  if (position >= cfg_.max_seq_len)
    return Status::error(
        "Qwen35Model::forward_token_with_state: sequence length " +
        std::to_string(position) + " >= max_seq_len " +
        std::to_string(cfg_.max_seq_len));

  // OOM GATE FIRST: ensure the KV page for `position` BEFORE any model
  // state is touched. TRANSACTIONAL: on failure the block table, the pool
  // accounting, the DeltaNet slot, the KV pages, and `rec->length` are all
  // exactly unchanged, and no layer forward runs below.
  Status s = mgr.ensure_kv_capacity(seq_id, position);
  if (!s.ok) return s;

  // H2D the current block-table page IDs (per-token metadata; stream-ordered
  // before every paged kernel that reads them). Grow-only scratch sized to
  // the manager's max number of logical blocks.
  const int num_blocks = rec->block_table.num_blocks();
  const int max_blocks = rec->block_table.max_blocks();
  if (block_table_scratch_.bytes() <
      static_cast<std::size_t>(max_blocks) * sizeof(int)) {
    block_table_scratch_.allocate(
        static_cast<std::size_t>(max_blocks) * sizeof(int), stream);
  }
  CUDA_CHECK(cudaMemcpyAsync(
      block_table_scratch_.data(), rec->block_table.page_ids(),
      static_cast<std::size_t>(num_blocks) * sizeof(int),
      cudaMemcpyHostToDevice, stream));
  const int* d_block_table = static_cast<int*>(block_table_scratch_.data());

  Qwen35KvPagePool& kpool = mgr.kv_pool_mut();
  Qwen35DeltaStatePool& dpool = mgr.delta_pool_mut();
  const int slot = rec->delta_slot;
  const int pt = mgr.page_tokens();

  const int H = cfg_.hidden_size;
  // 1. Embedding lookup (bit-exact D2D row copy, same as forward_token).
  const __nv_bfloat16* emb_row =
      embed_.data<__nv_bfloat16>() +
      static_cast<std::size_t>(token_id) * static_cast<std::size_t>(H);
  CUDA_CHECK(cudaMemcpyAsync(
      embed_out_.data(), emb_row,
      static_cast<std::size_t>(H) * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToDevice, stream));
  const __nv_bfloat16* x = embed_out_.data<__nv_bfloat16>();

  // 2. Chain all 24 decoder layers with EXTERNAL state (v0.5 Phase B).
  for (int i = 0; i < num_layers(); ++i) {
    if (is_linear_attention(i)) {
      const int ord = dpool.linear_layer_ordinal(i);
      delta(i)->forward_with_state(position, x, dpool.conv_mut(ord, slot),
                                   dpool.recurrent_mut(ord, slot), stream);
      x = delta(i)->stage_final_output();
    } else {
      const int ord = kpool.full_layer_ordinal(i);
      Qwen35FullAttentionLayer::PagedStateRef ps;
      ps.k_pages = kpool.k_page_mut(ord, 0);
      ps.v_pages = kpool.v_page_mut(ord, 0);
      ps.block_table = d_block_table;
      ps.page_tokens = pt;
      ps.page_stride = kpool.page_stride_elems();
      attention(i)->forward_with_paged_state(position, x, ps, stream);
      x = attention(i)->stage_final_output();
    }
  }

  // 3. Final zero-centered RMSNorm (same as forward_token).
  kernels::qwen35_rmsnorm_zc_bf16(x, norm_.data<__nv_bfloat16>(),
                                  norm_out_.data<__nv_bfloat16>(),
                                  /*M=*/1, /*H=*/H, cfg_.eps, stream);
  const __nv_bfloat16* normed = norm_out_.data<__nv_bfloat16>();

  // 4. Tied LM head (same as forward_token).
  bf16_gemv(embed_.data<__nv_bfloat16>(), normed,
            logits_buf_.data<__nv_bfloat16>(), /*N=*/cfg_.vocab_size,
            /*K=*/H, stream);

  // 5. Commit: only AFTER the whole forward is enqueued do we advance the
  //    sequence length (pure metadata; cannot fail: position+1 <= max).
  s = mgr.advance(seq_id, 1);
  return s;
}

const __nv_bfloat16* Qwen35Model::layer_final_output(int i) const {
  if (i < 0 || i >= num_layers()) return nullptr;
  if (is_linear_attention(i)) return delta(i)->stage_final_output();
  return attention(i)->stage_final_output();
}


// ---------------------------------------------------------------------------
// v0.6 Phase B: TRUE BATCHED decode forward (B rows, one traversal).
// Per row BIT-IDENTICAL to forward_token_with_state() (row-parity).
// ---------------------------------------------------------------------------
Status Qwen35Model::forward_batch_with_state(const int* token_ids,
                                             const SequenceId* seq_ids, int B,
                                             Qwen35StateManager& mgr,
                                             cudaStream_t stream) {
  // ---------- PREFLIGHT (ZERO MUTATION — checked BEFORE anything) -------
  if (!loaded_)
    return Status::error("Qwen35Model::forward_batch_with_state: not loaded");
  if (B < 1)
    return Status::error("Qwen35Model::forward_batch_with_state: B >= 1");
  if (token_ids == nullptr || seq_ids == nullptr)
    return Status::error(
        "Qwen35Model::forward_batch_with_state: null token/sequence arrays");
  // COMPATIBILITY GATE (as in the single path): the manager's config must
  // be EXACTLY the model's config.
  if (!(mgr.config() == cfg_))
    return Status::error(
        "Qwen35Model::forward_batch_with_state: manager config does not "
        "match the model config");
  // SINGLE-STREAM CONTRACT (the batch path is single-stream like the rest
  // of v0.5/v0.6).
  if (stream != mgr.kv_pool().stream() || stream != mgr.delta_pool().stream())
    return Status::error(
        "Qwen35Model::forward_batch_with_state: stream mismatch — the batch "
        "forward stream must equal the manager pool streams");
  const int vocab = cfg_.vocab_size;
  // Per row: token validity, live sequence, UNIQUE ids, position in bounds.
  std::vector<const SequenceState*> recs(B);
  std::vector<SequenceId> ids(seq_ids, seq_ids + B);
  for (int b = 0; b < B; ++b) {
    if (token_ids[b] < 0 || token_ids[b] >= vocab)
      return Status::error(
          "Qwen35Model::forward_batch_with_state: row " + std::to_string(b) +
          " token_id " + std::to_string(token_ids[b]) + " out of range");
    const SequenceState* rec = mgr.lookup(seq_ids[b]);
    if (rec == nullptr)
      return Status::error(
          "Qwen35Model::forward_batch_with_state: row " + std::to_string(b) +
          " sequence id " + std::to_string(seq_ids[b]) + " is not a live "
          "sequence");
    if (rec->length >= cfg_.max_seq_len)
      return Status::error(
          "Qwen35Model::forward_batch_with_state: row " + std::to_string(b) +
          " sequence length " + std::to_string(rec->length) + " >= "
          "max_seq_len " + std::to_string(cfg_.max_seq_len));
    recs[b] = rec;
  }
  // UNiqueness of the sequence ids.
  {
    std::vector<SequenceId> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    for (int b = 1; b < B; ++b) {
      if (sorted[b] == sorted[b - 1])
        return Status::error(
            "Qwen35Model::forward_batch_with_state: duplicate sequence id " +
            std::to_string(sorted[b]));
    }
  }
  // AGGREGATE KV CAPACITY: the SUM of the NEW pages any row needs must fit
  // the pool's free pages (row order allocation below is then guaranteed to
  // succeed; on failure NOTHING was allocated/changed).
  {
    const int pt = mgr.page_tokens();
    long long pages_needed = 0;
    for (int b = 0; b < B; ++b) {
      const int position = recs[b]->length;  // the forward position
      const int need_blocks =
          (position + pt) / pt;  // blocks covering positions [0..position]
      const int have_blocks = recs[b]->block_table.num_blocks();
      if (need_blocks > have_blocks)
        pages_needed += (need_blocks - have_blocks);
    }
    if (pages_needed > mgr.kv_pool().free_pages())
      return Status::error(
          "Qwen35Model::forward_batch_with_state: aggregate KV capacity "
          "insufficient for the batch (need " +
          std::to_string(pages_needed) + " new pages, free " +
          std::to_string(mgr.kv_pool().free_pages()) +
          ") — the scheduler may fall back to the frozen serial path");
  }

  // ---------- GROW batch scratch to B (reallocate only on a larger B) ----
  if (B > batch_cap_) {
    batch_cap_ = B;
    const std::size_t b = static_cast<std::size_t>(B);
    const std::size_t H = static_cast<std::size_t>(cfg_.hidden_size);
    const std::size_t V = static_cast<std::size_t>(cfg_.vocab_size);
    embed_out_batch_.allocate(H * b * sizeof(__nv_bfloat16), stream);
    norm_out_batch_.allocate(H * b * sizeof(__nv_bfloat16), stream);
    logits_batch_.allocate(V * b * sizeof(__nv_bfloat16), stream);
    batch_token_ids_.allocate(b * sizeof(int), stream);
    batch_positions_.allocate(b * sizeof(int), stream);
    batch_slots_.allocate(b * sizeof(int), stream);
  }

  // ---------- COMMIT capacity (cannot fail: the aggregate check covers it)
  for (int b = 0; b < B; ++b) {
    const Status s = mgr.ensure_kv_capacity(seq_ids[b], recs[b]->length);
    CUDALM_PRECONDITION(
        s.ok,
        "forward_batch_with_state: post-preflight ensure_kv_capacity "
        "failure (logic bug — the aggregate capacity check must cover it)");
  }

  // ---------- H2D the per-batch metadata (tokens / positions / slots /
  // block tables). The block tables are re-read AFTER the ensure above so
  // they include the freshly allocated pages. Per-token metadata H2D,
  // stream-ordered — exactly the frozen single path's block-table H2D. ----
  const int max_blocks = recs[0]->block_table.max_blocks();
  if (batch_block_tables_.bytes() <
      static_cast<std::size_t>(B) * static_cast<std::size_t>(max_blocks) *
          sizeof(int)) {
    batch_block_tables_.allocate(
        static_cast<std::size_t>(B) * static_cast<std::size_t>(max_blocks) *
            sizeof(int),
        stream);
  }
  std::vector<int> host_bt(static_cast<std::size_t>(B) * max_blocks);
  std::vector<int> host_pos(B), host_slots(B);
  for (int b = 0; b < B; ++b) {
    const int* pids = recs[b]->block_table.page_ids();
    const int nb = recs[b]->block_table.num_blocks();
    for (int k = 0; k < max_blocks; ++k)
      host_bt[static_cast<std::size_t>(b) * max_blocks + k] =
          (k < nb) ? pids[k] : 0;  // stale tail never read (prefix contract)
    host_pos[b] = recs[b]->length;
    host_slots[b] = recs[b]->delta_slot;
  }
  CUDA_CHECK(cudaMemcpyAsync(batch_token_ids_.data(), token_ids,
                             static_cast<std::size_t>(B) * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(batch_positions_.data(), host_pos.data(),
                             static_cast<std::size_t>(B) * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(batch_slots_.data(), host_slots.data(),
                             static_cast<std::size_t>(B) * sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(batch_block_tables_.data(), host_bt.data(),
                             static_cast<std::size_t>(B) * max_blocks *
                                 sizeof(int),
                             cudaMemcpyHostToDevice, stream));

  const int* d_tokens = static_cast<int*>(batch_token_ids_.data());
  const int* d_positions = static_cast<int*>(batch_positions_.data());
  const int* d_slots = static_cast<int*>(batch_slots_.data());
  const int* d_block_tables = static_cast<int*>(batch_block_tables_.data());

  Qwen35KvPagePool& kpool = mgr.kv_pool_mut();
  Qwen35DeltaStatePool& dpool = mgr.delta_pool_mut();
  const int H = cfg_.hidden_size;
  const int pt = mgr.page_tokens();

  // 1. Batch embedding gather: embed_out_batch_[b] = embed[token_ids[b]].
  kernels::batch_embed_gather_bf16(embed_.data<__nv_bfloat16>(), d_tokens,
                                   embed_out_batch_.data<__nv_bfloat16>(), B,
                                   H, stream);
  const __nv_bfloat16* x = embed_out_batch_.data<__nv_bfloat16>();

  // 2. Chain all 24 decoder layers with EXTERNAL state, BATCHED (one
  //    traversal; row b addresses its own slot / block table / logical KV).
  for (int i = 0; i < num_layers(); ++i) {
    if (is_linear_attention(i)) {
      const int ord = dpool.linear_layer_ordinal(i);
      delta(i)->forward_batch_with_state(
          B, d_slots, x, dpool.conv_mut(ord, 0), dpool.recurrent_mut(ord, 0),
          stream);
      x = delta(i)->stage_final_output_batch();
    } else {
      const int ord = kpool.full_layer_ordinal(i);
      Qwen35FullAttentionLayer::PagedStateRefBatch psb;
      psb.k_pages = kpool.k_page_mut(ord, 0);
      psb.v_pages = kpool.v_page_mut(ord, 0);
      psb.block_table = d_block_tables;
      psb.row_stride = max_blocks;
      psb.d_positions = d_positions;
      psb.page_tokens = pt;
      psb.page_stride = kpool.page_stride_elems();
      attention(i)->forward_batch_with_paged_state(B, host_pos.data(), x,
                                                   psb, stream);
      x = attention(i)->stage_final_output_batch();
    }
  }

  // 3. Final zero-centered RMSNorm (M = B rows).
  kernels::qwen35_rmsnorm_zc_bf16(x, norm_.data<__nv_bfloat16>(),
                                  norm_out_batch_.data<__nv_bfloat16>(), B, H,
                                  cfg_.eps, stream);
  const __nv_bfloat16* normed = norm_out_batch_.data<__nv_bfloat16>();

  // 4. Tied LM head (batched bf16 GEMV): logits [B][vocab].
  batch_bf16_gemv(embed_.data<__nv_bfloat16>(), normed,
                  logits_batch_.data<__nv_bfloat16>(), cfg_.vocab_size, H, B,
                  stream);

  // 5. Commit: only AFTER the whole batch is enqueued do we advance each
  //    sequence length (pure metadata; cannot fail: position+1 <= max).
  for (int b = 0; b < B; ++b) {
    const Status s = mgr.advance(seq_ids[b], 1);
    CUDALM_PRECONDITION(
        s.ok,
        "forward_batch_with_state: post-forward advance failure (logic bug — "
        "the preflight checked every position < max_seq_len)");
  }
  return Status::ok_status();
}

const __nv_bfloat16* Qwen35Model::batch_layer_final_output(int i) const {
  if (i < 0 || i >= num_layers()) return nullptr;
  if (is_linear_attention(i))
    return delta(i)->stage_final_output_batch();
  return attention(i)->stage_final_output_batch();
}
}  // namespace cudalm
