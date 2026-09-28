// CUDALM — v0.8 Phase B: session turn PREFLIGHT contract gate (CPU test;
// NO checkpoint, NO model — a default-constructed (NOT loaded) Qwen35Model
// + real device pools + a small synthetic valid() config, the same
// discipline as test_session_manager.cpp).
//
// The turn preflight is designed so that checks 1-6 (session liveness,
// non-empty input, max_new_tokens >= 0, sampling validity, CONTEXT
// OVERFLOW, stream contract) need NO model vocab — they are fully
// gateable here, on an unloaded model, where EVERY failing call must be
// ZERO-MUTATION (no forward, no KV allocation, no Delta mutation,
// length unchanged — asserted after each call against a pre-dirtied
// Delta slot + exact pool accounting).
//
// Gates:
//   * invalid SessionId        -> ok=false, error, zero mutation;
//   * empty new_input_tokens   -> ok=false, error, zero mutation;
//   * max_new_tokens < 0       -> ok=false, error, zero mutation;
//   * invalid sampling config  -> ok=false (validate_sampling_config
//     message), zero mutation;
//   * CONTEXT OVERFLOW: length + input + max_new_tokens > max_seq_len
//     -> ok=false with the explicit "context overflow" reject (the v0.8
//     policy: no eviction / truncation), zero mutation;
//   * EXACT BOUNDARY: length + input + max_new_tokens == max_seq_len is
//     ACCEPTED by the capacity gate (the call proceeds past the overflow
//     check and fails LATER at "model not loaded" — not an overflow
//     error), zero mutation;
//   * STREAM mismatch (a second stream != the pools' stream) -> ok=false
//     with the stream-contract error, zero mutation;
//   * model-not-loaded: a fully valid call on the unloaded model fails at
//     the model gate with ok=false + "model not loaded", zero mutation
//     (the model-dependent preflight checks — eos range, input-token
//     range, config match — are gated on the REAL model by
//     test_qwen35_session_generation.cpp).
//
// Provenance: CUDALM-native (v0.8 Phase B).

#include "../../tests/common/check.h"
#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/session.h"
#include "cudalm/session_generator.h"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using namespace cudalm;

namespace {

// Small synthetic hybrid config (valid(); MUST match
// test_session_manager.cpp): 8 layers, interval 4 -> 2 full (layers 3,7)
// + 6 linear; max_seq_len 64.
Qwen35Config small_config() {
  Qwen35Config c;
  c.hidden_size = 256;
  c.num_hidden_layers = 8;
  c.intermediate_size = 512;
  c.vocab_size = 1024;
  c.n_heads = 4;
  c.n_kv_heads = 2;
  c.head_dim = 64;
  c.lin_num_k_heads = 4;
  c.lin_num_v_heads = 4;
  c.lin_key_head_dim = 32;
  c.lin_value_head_dim = 32;
  c.lin_conv_kernel_dim = 4;
  c.full_attention_interval = 4;
  c.group_size = 128;
  c.max_seq_len = 64;
  c.eps = 1e-6f;
  c.rope_theta = 1e7f;
  c.partial_rotary_factor = 0.25f;
  c.mrope_section[0] = 4;
  c.mrope_section[1] = 2;
  c.mrope_section[2] = 2;
  return c;
}

void fill_bf16(__nv_bfloat16* dev, std::size_t n, float pattern,
               cudaStream_t s) {
  std::vector<__nv_bfloat16> h(n, __float2bfloat16(pattern));
  if (n > 0)
    CUDA_CHECK(cudaMemcpyAsync(dev, h.data(), n * sizeof(__nv_bfloat16),
                               cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
}

void fill_f32(float* dev, std::size_t n, float pattern, cudaStream_t s) {
  std::vector<float> h(n, pattern);
  if (n > 0)
    CUDA_CHECK(cudaMemcpyAsync(dev, h.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
}

// Write a non-zero pattern into EVERY layer's state for one slot (a marker
// that a zeroing/mutation would destroy).
int dirty_delta_slot(Qwen35DeltaStatePool& pool, int slot, cudaStream_t s) {
  for (int l = 0; l < pool.n_linear_layers(); ++l) {
    fill_bf16(pool.conv_mut(l, slot), pool.conv_elems(), 1.5f, s);
    fill_f32(pool.recurrent_mut(l, slot), pool.rec_elems(), 0.25f, s);
  }
  return 0;
}

// True iff the slot's pattern SURVIVED (non-zero somewhere in every layer
// — the inverse of "a stray reset zeroed it").
bool delta_slot_pattern_intact(const Qwen35DeltaStatePool& pool, int slot,
                               cudaStream_t s) {
  for (int l = 0; l < pool.n_linear_layers(); ++l) {
    bool any = false;
    const std::size_t cn = pool.conv_elems();
    std::vector<__nv_bfloat16> conv(cn);
    if (cn)
      CUDA_CHECK(cudaMemcpyAsync(conv.data(), pool.conv(l, slot),
                                 cn * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (__nv_bfloat16 x : conv)
      if (__bfloat162float(x) != 0.0f) any = true;
    if (!any) return false;
  }
  return true;
}

// The zero-mutation snapshot: length, KV table, pool accounting, and the
// Delta slot pattern. CHECK_ZERO_MUTATION asserts the whole snapshot is
// EXACTLY unchanged after a failing call.
struct Snapshot {
  int length = 0;
  int num_blocks = 0;
  int used_pages = 0;
  int used_slots = 0;
};

Snapshot take(const Qwen35StateManager& mgr, SequenceId sid) {
  const SequenceState* seq = mgr.lookup(sid);
  Snapshot sn;
  if (seq != nullptr) {
    sn.length = seq->length;
    sn.num_blocks = seq->block_table.num_blocks();
  }
  sn.used_pages = mgr.kv_pool().used_pages();
  sn.used_slots = mgr.delta_pool().used_slots();
  return sn;
}

int check_zero_mutation(const Qwen35StateManager& mgr, SequenceId sid,
                        const Snapshot& before, cudaStream_t s) {
  const Snapshot after = take(mgr, sid);
  CHECK_EQ(after.length, before.length);
  CHECK_EQ(after.num_blocks, before.num_blocks);
  CHECK_EQ(after.used_pages, before.used_pages);
  CHECK_EQ(after.used_slots, before.used_slots);
  CHECK(delta_slot_pattern_intact(mgr.delta_pool(),
                                  mgr.lookup(sid)->delta_slot, s));
  return 0;
}

}  // namespace

int main() {
  std::printf("test_session_turn_contract: CUDALM v0.8 Phase B turn "
              "preflight contract gate (unloaded model, real device pools, "
              "small synthetic config)\n");
  const Qwen35Config cfg = small_config();
  const int PT = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  cudaStream_t s_other;  // a WRONG stream (must be rejected)
  CUDA_CHECK(cudaStreamCreate(&s_other));
  {
    Qwen35StateManager mgr(cfg, PT, /*kv_pages=*/4, /*delta_slots=*/2, s);
    SessionManager sm(mgr);
    Qwen35Model model;  // NOT loaded: every preflight failure is reachable
    SessionGenerator gen(model, sm);
    const SamplingConfig greedy = SamplingConfig::greedy();

    SessionId a = 0;
    CHECK(sm.create_session(&a).ok);
    const SequenceId sid = sm.lookup(a)->sequence_id;
    // Pre-dirty A's Delta slot: any zeroing/mutation destroys the pattern.
    if (dirty_delta_slot(mgr.delta_pool_mut(), mgr.lookup(sid)->delta_slot, s))
      return 1;

    const auto expect_fail = [&](const char* name, const TurnResult& r,
                                 bool overflow_msg) -> int {
      CHECK(!r.ok);
      CHECK(!r.error.empty());
      if (overflow_msg) {
        CHECK(r.error.find("context overflow") != std::string::npos);
      } else {
        CHECK(r.error.find("context overflow") == std::string::npos);
      }
      CHECK(r.generated.empty());
      CHECK_EQ(r.input_count, 0);
      CHECK_EQ(r.forward_count, 0);
      CHECK_EQ(r.context_length, 0);
      std::printf("  [ok] %s: rejected, zero mutation\n", name);
      return 0;
    };

    // ---- 1. invalid SessionId -------------------------------------------
    {
      const Snapshot b = take(mgr, sid);
      TurnResult r = gen.generate_turn(999, {100}, 1, -1, greedy, s);
      CHECK(r.error.find("not live") != std::string::npos);
      if (expect_fail("invalid SessionId", r, /*overflow_msg=*/false)) return 1;
      if (check_zero_mutation(mgr, sid, b, s)) return 1;
    }
    // ---- 2. empty new_input_tokens ---------------------------------------
    {
      const Snapshot b = take(mgr, sid);
      TurnResult r = gen.generate_turn(a, {}, 1, -1, greedy, s);
      if (expect_fail("empty new_input_tokens", r, false)) return 1;
      if (check_zero_mutation(mgr, sid, b, s)) return 1;
    }
    // ---- 3. max_new_tokens < 0 --------------------------------------------
    {
      const Snapshot b = take(mgr, sid);
      TurnResult r = gen.generate_turn(a, {100}, -1, -1, greedy, s);
      if (expect_fail("max_new_tokens < 0", r, false)) return 1;
      if (check_zero_mutation(mgr, sid, b, s)) return 1;
    }
    // ---- 4. invalid sampling config ----------------------------------------
    {
      const Snapshot b = take(mgr, sid);
      SamplingConfig bad = SamplingConfig{};
      bad.temperature = 1.0f;
      bad.top_k = -1;  // invalid (fail loud, never clamped)
      TurnResult r = gen.generate_turn(a, {100}, 1, -1, bad, s);
      CHECK(r.error.find("sampling") != std::string::npos);
      if (expect_fail("invalid sampling config", r, false)) return 1;
      if (check_zero_mutation(mgr, sid, b, s)) return 1;
    }
    // ---- 5. CONTEXT OVERFLOW (length 62 + input 2 + max_new 1 = 65 > 64) ---
    {
      CHECK(mgr.set_length(sid, 62).ok);
      const Snapshot b = take(mgr, sid);
      TurnResult r = gen.generate_turn(a, {100, 200}, 1, -1, greedy, s);
      if (expect_fail("context overflow (62+2+1 > 64)", r,
                      /*overflow_msg=*/true))
        return 1;
      if (check_zero_mutation(mgr, sid, b, s)) return 1;  // length stays 62
    }
    // ---- 6. EXACT BOUNDARY ACCEPTED (61 + 2 + 1 == 64) ---------------------
    // The capacity gate must ACCEPT the exact boundary: the call proceeds
    // past the overflow check and fails LATER at the model gate (not an
    // overflow error).
    {
      CHECK(mgr.set_length(sid, 61).ok);
      const Snapshot b = take(mgr, sid);
      TurnResult r = gen.generate_turn(a, {100, 200}, 1, -1, greedy, s);
      CHECK(r.error.find("model not loaded") != std::string::npos);
      if (expect_fail("exact boundary (61+2+1 == 64) accepted by capacity "
                      "gate",
                      r, /*overflow_msg=*/false))
        return 1;
      if (check_zero_mutation(mgr, sid, b, s)) return 1;  // length stays 61
    }
    // ---- 7. STREAM mismatch -------------------------------------------------
    {
      CHECK(mgr.set_length(sid, 0).ok);
      const Snapshot b = take(mgr, sid);
      TurnResult r = gen.generate_turn(a, {100}, 1, -1, greedy, s_other);
      CHECK(r.error.find("stream") != std::string::npos);
      if (expect_fail("stream mismatch", r, false)) return 1;
      if (check_zero_mutation(mgr, sid, b, s)) return 1;
    }
    // ---- 8. model not loaded (a fully valid call) ---------------------------
    {
      const Snapshot b = take(mgr, sid);
      TurnResult r = gen.generate_turn(a, {100}, 1, -1, greedy, s);
      CHECK(r.error.find("model not loaded") != std::string::npos);
      if (expect_fail("model not loaded (fully valid call)", r, false))
        return 1;
      if (check_zero_mutation(mgr, sid, b, s)) return 1;
    }

    CHECK_EQ(sm.num_sessions(), 1);  // no half-session anywhere
    CHECK_EQ(mgr.num_live_sequences(), 1);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaStreamDestroy(s_other));
  std::printf("test_session_turn_contract: PASS\n");
  return 0;
}
