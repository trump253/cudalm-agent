// CUDALM — Qwen3.5 unified state manager implementation (v0.5 Phase A).

#include "cudalm/qwen35_state_manager.h"

#include "cudalm/cuda_check.h"

namespace cudalm {

Qwen35StateManager::Qwen35StateManager(const Qwen35Config& cfg,
                                       int page_tokens,
                                       int kv_capacity_pages,
                                       int delta_capacity_slots,
                                       cudaStream_t stream)
    : cfg_(cfg),
      kv_(cfg, page_tokens, kv_capacity_pages, stream),
      delta_(cfg, delta_capacity_slots, stream) {
  CUDALM_PRECONDITION(cfg.valid(),
                      "Qwen35StateManager: config must be valid()");
}

Status Qwen35StateManager::not_live(SequenceId id) const {
  return Status::error("state manager: sequence " + std::to_string(id) +
                       " is not live (never created or retired)");
}

Status Qwen35StateManager::create_sequence(SequenceId* out_id) {
  if (out_id == nullptr)
    return Status::error("state manager: create_sequence: null output");
  // Delta slot first: on OOM nothing is registered (no id issued, no
  // half-record) — the transactional create contract.
  int slot = -1;
  Status s = delta_.acquire_slot(&slot);
  if (!s.ok) return s;
  SequenceState rec;
  rec.id = next_id_++;
  rec.delta_slot = slot;
  rec.block_table =
      KvBlockTable(cfg_.max_seq_len, kv_.page_tokens());
  rec.length = 0;
  sequences_.emplace(rec.id, std::move(rec));
  *out_id = rec.id;
  return Status::ok_status();
}

Status Qwen35StateManager::ensure_kv_capacity(SequenceId id, int position) {
  auto it = sequences_.find(id);
  if (it == sequences_.end()) return not_live(id);
  return it->second.block_table.ensure_capacity(position, kv_);
}

Status Qwen35StateManager::set_length(SequenceId id, int new_length) {
  auto it = sequences_.find(id);
  if (it == sequences_.end()) return not_live(id);
  if (new_length < 0 || new_length > cfg_.max_seq_len)
    return Status::error("state manager: set_length: new_length " +
                         std::to_string(new_length) + " outside [0, " +
                         std::to_string(cfg_.max_seq_len) + ")");
  it->second.length = new_length;
  return Status::ok_status();
}

Status Qwen35StateManager::advance(SequenceId id, int n_tokens) {
  auto it = sequences_.find(id);
  if (it == sequences_.end()) return not_live(id);
  if (n_tokens < 0)
    return Status::error("state manager: advance: n_tokens must be >= 0");
  const long long new_length =
      static_cast<long long>(it->second.length) + n_tokens;
  if (new_length > cfg_.max_seq_len)
    return Status::error("state manager: advance: new length " +
                         std::to_string(new_length) + " exceeds max_seq_len " +
                         std::to_string(cfg_.max_seq_len));
  it->second.length = static_cast<int>(new_length);
  return Status::ok_status();
}

Status Qwen35StateManager::reset_sequence(SequenceId id) {
  auto it = sequences_.find(id);
  if (it == sequences_.end()) return not_live(id);
  // Release every KV page (zeroed on release); the pool accounting returns
  // to pre-growth. clear() on an empty table is a no-op.
  Status s = it->second.block_table.clear(kv_);
  if (!s.ok) return s;
  // Zero the (still-owned) Delta slot in place — cannot OOM.
  s = delta_.reset_slot(it->second.delta_slot);
  if (!s.ok) return s;
  it->second.length = 0;
  return Status::ok_status();
}

Status Qwen35StateManager::retire_sequence(SequenceId id) {
  auto it = sequences_.find(id);
  if (it == sequences_.end()) return not_live(id);
  Status s = it->second.block_table.clear(kv_);
  if (s.ok) s = delta_.release_slot(it->second.delta_slot);
  if (!s.ok) return s;
  sequences_.erase(it);  // the id is never reissued (next_id_ only grows)
  return Status::ok_status();
}

const SequenceState* Qwen35StateManager::lookup(SequenceId id) const {
  const auto it = sequences_.find(id);
  return it == sequences_.end() ? nullptr : &it->second;
}

}  // namespace cudalm
