// CUDALM — Qwen3.5 paged-KV pool implementation (v0.5 Phase A).

#include "cudalm/qwen35_kv_page_pool.h"

#include "cudalm/cuda_check.h"

namespace cudalm {

Qwen35KvPagePool::Qwen35KvPagePool(const Qwen35Config& cfg, int page_tokens,
                                   int num_pages, cudaStream_t stream)
    : cfg_(cfg),
      page_tokens_(page_tokens),
      n_full_(qwen35_num_full_layers(cfg)),
      pool_stream_(stream),
      // num_pages is precondition-checked in the body (fatal abort for
      // negative); clamp here so member init itself is always safe.
      ids_(num_pages >= 0 ? num_pages : 0) {
  CUDALM_PRECONDITION(page_tokens_ >= 1,
                      "Qwen35KvPagePool: page_tokens must be >= 1");
  CUDALM_PRECONDITION(num_pages >= 0,
                      "Qwen35KvPagePool: num_pages must be >= 0");
  CUDALM_PRECONDITION(n_full_ >= 1 && cfg.n_kv_heads >= 1 &&
                          cfg.head_dim >= 1,
                      "Qwen35KvPagePool: config must have positive "
                      "full-attention KV dims");
  full_layer_idxs_.reserve(static_cast<std::size_t>(n_full_));
  for (int i = 0; i < cfg.num_hidden_layers; ++i)
    if (cfg.is_full_attention(i)) full_layer_idxs_.push_back(i);

  // Per-page slice: [n_kv][page_tokens][head_dim] bf16; per tensor
  // [n_full][num_pages] slices. Zero the WHOLE pool at construction (the
  // zero-on-release invariant then holds for every subsequent acquire).
  const std::size_t slice_bytes = page_elems() * sizeof(__nv_bfloat16);
  const std::size_t total =
      slice_bytes * static_cast<std::size_t>(n_full_) *
      static_cast<std::size_t>(num_pages);
  k_.allocate(total, stream);
  v_.allocate(total, stream);
  k_.clear(total, stream);
  v_.clear(total, stream);
}

Status Qwen35KvPagePool::allocate_page(int* out_page_id) {
  return ids_.acquire(out_page_id);
}

Status Qwen35KvPagePool::free_page(int page_id) {
  if (page_id < 0 || page_id >= ids_.capacity())
    return Status::error("kv page pool: page " + std::to_string(page_id) +
                         " out of range (capacity=" +
                         std::to_string(ids_.capacity()) + ")");
  if (!ids_.is_live(page_id))
    return Status::error("kv page pool: page " + std::to_string(page_id) +
                         " is not live (double free or invalid)");
  zero_page(page_id);  // ordered on the pool's stream (STREAM CONTRACT)
  return ids_.release(page_id);
}

void Qwen35KvPagePool::reset() {
  const std::vector<int> live = ids_.live_ids();
  for (int page_id : live) zero_page(page_id);
  ids_.reset();
}

void Qwen35KvPagePool::zero_page(int page_id) {
  const cudaStream_t stream = pool_stream_;
  const std::size_t slice_bytes = page_elems() * sizeof(__nv_bfloat16);
  for (int l = 0; l < n_full_; ++l) {
    const std::size_t off =
        (static_cast<std::size_t>(l) * ids_.capacity() +
         static_cast<std::size_t>(page_id)) * slice_bytes;
    CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(k_.data()) + off, 0,
                               slice_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(v_.data()) + off, 0,
                               slice_bytes, stream));
  }
}

int Qwen35KvPagePool::full_layer_ordinal(int layer_idx) const {
  if (!cfg_.is_full_attention(layer_idx)) return -1;
  for (int o = 0; o < static_cast<int>(full_layer_idxs_.size()); ++o)
    if (full_layer_idxs_[static_cast<std::size_t>(o)] == layer_idx) return o;
  return -1;  // unreachable
}

int Qwen35KvPagePool::layer_of_full_ordinal(int ordinal) const {
  if (ordinal < 0 || ordinal >= static_cast<int>(full_layer_idxs_.size()))
    return -1;
  return full_layer_idxs_[static_cast<std::size_t>(ordinal)];
}

namespace {
// Shared slice-pointer computation; range-checked once.
inline const __nv_bfloat16* page_ptr(const DeviceBuffer& buf, int n_full,
                                     int capacity_pages, int ordinal,
                                     int page_id, std::size_t page_elems) {
  CUDALM_PRECONDITION(
      ordinal >= 0 && ordinal < n_full && page_id >= 0 &&
          page_id < capacity_pages,
      "Qwen35KvPagePool: page slice (ordinal, page) out of range");
  const std::size_t off =
      (static_cast<std::size_t>(ordinal) * capacity_pages +
       static_cast<std::size_t>(page_id)) * page_elems;
  return buf.data<__nv_bfloat16>() + off;
}
}  // namespace

const __nv_bfloat16* Qwen35KvPagePool::k_page(int ordinal, int page_id) const {
  return page_ptr(k_, n_full_, ids_.capacity(), ordinal, page_id,
                  page_elems());
}

const __nv_bfloat16* Qwen35KvPagePool::v_page(int ordinal, int page_id) const {
  return page_ptr(v_, n_full_, ids_.capacity(), ordinal, page_id,
                  page_elems());
}

__nv_bfloat16* Qwen35KvPagePool::k_page_mut(int ordinal, int page_id) {
  return const_cast<__nv_bfloat16*>(k_page(ordinal, page_id));
}

__nv_bfloat16* Qwen35KvPagePool::v_page_mut(int ordinal, int page_id) {
  return const_cast<__nv_bfloat16*>(v_page(ordinal, page_id));
}

}  // namespace cudalm
