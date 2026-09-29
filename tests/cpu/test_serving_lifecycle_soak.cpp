// CUDALM — v0.9 Phase D: the in-process serving LIFECYCLE SOAK (CPU —
// NO sockets, NO checkpoint, NO model: the deterministic fake
// forwarder + real pools + real SessionManager / Scheduler + the
// ServingController with TTL + LRU-on-pressure + a FAKE clock, driven
// through the HTTP handler).
//
// A BOUNDED (fast) soak — a few hundred mixed operations (create /
// admit / drive / reset / destroy / TTL / LRU / create-reuse) that
// continuously verifies the POOL LIFECYCLE invariants:
//   * no stuck busy (live_requests returns to 0 after every group);
//   * no SessionId reuse (every issued id is globally unique —
//     monotonic);
//   * no RequestId reuse (ids are strictly monotonic);
//   * the session accounting stays consistent (the controller's
//     live_sessions == the manager's live count of the managed set);
//   * at the end: after destroying every session, the KV / Delta
//     pools are EXACTLY zero (no leaked pages / slots).
//
// Provenance: CUDALM-native (v0.9 Phase D).

#include "../../tests/common/check.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/qwen35_config.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/scheduler.h"
#include "cudalm/sampling.h"
#include "cudalm/session.h"
#include "cudalm/serving_controller.h"
#include "tools/common/http_protocol.h"
#include "tools/common/http_serving_handler.h"

using namespace cudalm;

namespace {

Qwen35Config small_config() {
  Qwen35Config c;
  c.hidden_size = 256;
  c.num_hidden_layers = 8;
  c.intermediate_size = 512;
  c.vocab_size = 512;
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

class FakeForwarder : public SequenceForwarder {
 public:
  int vocab = 512;

  Status forward_token(int /*token_id*/, SequenceId sequence_id,
                       Qwen35StateManager& mgr,
                       cudaStream_t /*stream*/) override {
    const SequenceState* st = mgr.lookup(sequence_id);
    if (st == nullptr) {
      return Status::error("fake forward: unknown sequence");
    }
    const int pos = st->length;
    Status s = mgr.ensure_kv_capacity(sequence_id, pos);
    if (!s.ok) return s;
    return mgr.set_length(sequence_id, pos + 1);
  }

  Status logits_to_host(std::vector<__nv_bfloat16>* out,
                        cudaStream_t /*stream*/) const override {
    out->assign(static_cast<std::size_t>(vocab),
                __float2bfloat16(-1.0f));
    (*out)[11] = __float2bfloat16(10.0f);
    return Status::ok_status();
  }

  int vocab_size() const override { return vocab; }
};

class FakeClock : public MonotonicClock {
 public:
  std::chrono::steady_clock::time_point t0 =
      std::chrono::steady_clock::now();
  int ms = 0;

  std::chrono::steady_clock::time_point now() const override {
    return t0 + std::chrono::milliseconds(ms);
  }
  void advance_ms(int n) { ms += n; }
};

class FakeCodec : public http::TextCodec {
 public:
  Status encode(const std::string& utf8,
                std::vector<int>* out) const override {
    out->clear();
    for (const unsigned char b : utf8) {
      out->push_back(static_cast<int>((b % 500u) + 2u));
    }
    return Status::ok_status();
  }

  Status decode(const std::vector<int>& ids,
                std::string* out) const override {
    out->clear();
    for (int id : ids) {
      out->push_back(static_cast<char>(id % 256));
    }
    return Status::ok_status();
  }

  // id 0 is never produced (codec >= 2; the forwarder always picks
  // 11) — EOS never fires:
  int eos_token_id() const override { return 0; }
};

// A deterministic 32-bit LCG (the soak pattern is reproducible).
struct Lcg {
  std::uint32_t s = 0xC0FFEE42u;
  std::uint32_t next() {
    s = s * 1664525u + 1013904223u;
    return s >> 8;
  }
  int range(int n) { return static_cast<int>(next() % static_cast<std::uint32_t>(n)); }
};

http::HttpRequest req(const std::string& method, const std::string& path,
                      const std::string& body = "") {
  http::HttpRequest r;
  r.method = method;
  const std::size_t q = path.find('?');
  r.path = q == std::string::npos ? path : path.substr(0, q);
  r.query = q == std::string::npos ? "" : path.substr(q + 1);
  r.body = body;
  return r;
}

}  // namespace

int main() {
  // ---- the runtime: max 3 sessions, TTL 2ms, LRU on pressure --------
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  Qwen35StateManager mgr(small_config(), 4, 64, 4, stream);
  SessionManager sm(mgr);
  FakeForwarder fwd;
  Scheduler sched(fwd, mgr, stream, &sm);
  FakeClock clk;
  const SessionEvictionPolicy pol =
      SessionEvictionPolicy().with_idle_ttl(std::chrono::milliseconds(2))
          .with_lru_on_session_pressure(true);
  ServingController ctrl(sched, sm, ServingLimits{3, -1, 0}, &clk, pol);
  FakeCodec codec;
  http::ServingHttpDeps deps{&ctrl, &sched, &sm, &mgr, &codec};
  http::ServingHttpApi api(deps);

  Lcg rnd;
  std::set<SessionId> issued;          // every id ever created
  std::vector<SessionId> managed;      // the ids we created (live or not)
  RequestId max_rid = 0;
  int total_turns = 0;

  auto is_live = [&sm](SessionId id) { return sm.lookup(id) != nullptr; };

  auto live_managed = [&]() {
    std::vector<SessionId> v;
    for (SessionId id : managed) {
      if (is_live(id)) v.push_back(id);
    }
    return v;
  };

  constexpr int kIterations = 240;
  for (int it = 0; it < kIterations; ++it) {
    const int op = rnd.range(10);
    if (op <= 3) {
      // ---- create (the TTL sweep / LRU pressure may fire) ----------
      SessionId sid = 0;
      Status s = ctrl.create_session(&sid);
      if (!s.ok) {
        // A no-candidate rejection (every session is busy /
        // terminal-undrained — rare here since we always drain):
        // destroy any live session to make room and retry ONCE:
        std::vector<SessionId> live = live_managed();
        CHECK(!live.empty());
        (void)ctrl.destroy_session(live[static_cast<std::size_t>(
            rnd.range(static_cast<int>(live.size())))]);
        s = ctrl.create_session(&sid);
        CHECK(s.ok);
      }
      // NO SessionId REUSE (the monotonic invariant):
      CHECK(issued.count(sid) == 0);
      issued.insert(sid);
      managed.push_back(sid);
      CHECK(ctrl.stats().live_sessions <= 3);
    } else if (op <= 6) {
      // ---- a turn on a random live managed session ------------------
      std::vector<SessionId> live = live_managed();
      if (live.empty()) continue;
      const SessionId sid =
          live[static_cast<std::size_t>(rnd.range(static_cast<int>(live.size())))];
      const int max_new = 1 + rnd.range(3);  // 1..3
      const http::HttpResponse r = api.handle(
          req("POST", "/v1/sessions/" + std::to_string(sid) +
                          "/turn?max_new_tokens=" + std::to_string(max_new),
              "soak " + std::to_string(it)));
      CHECK_EQ(r.status, 200);
      // the request id is STRICTLY monotonic (no reuse):
      long rid = 0;
      const std::string pat = "\"request_id\":";
      const std::size_t p = r.body.find(pat);
      CHECK(p != std::string::npos);
      rid = std::stol(r.body.substr(p + pat.size()));
      CHECK(rid > static_cast<long>(max_rid));
      max_rid = static_cast<RequestId>(rid);
      ++total_turns;
    } else if (op == 7) {
      // ---- reset a random live session -------------------------------
      std::vector<SessionId> live = live_managed();
      if (live.empty()) continue;
      const SessionId sid =
          live[static_cast<std::size_t>(rnd.range(static_cast<int>(live.size())))];
      const http::HttpResponse r =
          api.handle(req("POST", "/v1/sessions/" + std::to_string(sid) +
                                    "/reset"));
      CHECK_EQ(r.status, 200);
    } else if (op == 8) {
      // ---- destroy a random live session -----------------------------
      std::vector<SessionId> live = live_managed();
      if (live.empty()) continue;
      const SessionId sid =
          live[static_cast<std::size_t>(rnd.range(static_cast<int>(live.size())))];
      const http::HttpResponse r =
          api.handle(req("DELETE", "/v1/sessions/" + std::to_string(sid)));
      CHECK_EQ(r.status, 200);
      CHECK(!is_live(sid));
    } else {
      // ---- advance the fake clock (TTL expiries fire on the next
      //      create) -----------------------------------------------------
      clk.advance_ms(5);
    }

    // ---- the per-iteration invariants ----------------------------------
    const ServingStats st = ctrl.stats();
    // (1) no stuck busy: every group leaves the quota at zero:
    CHECK_EQ(st.live_requests, 0);
    // (2) the session accounting is consistent:
    CHECK_EQ(st.live_sessions,
             static_cast<int>(live_managed().size()));
    // (3) the manager agrees:
    CHECK_EQ(st.live_sessions, sm.num_sessions());
    // (4) the live-sequence count agrees (1 sequence per session):
    CHECK_EQ(mgr.num_live_sequences(), sm.num_sessions());
  }

  CHECK(total_turns > 0);

  // ---- teardown: destroy every remaining session; the pools must be
  //      EXACTLY zero (no leaked KV pages / Delta slots) ----------------
  for (SessionId sid : live_managed()) {
    (void)ctrl.destroy_session(sid);
  }
  const ServingStats fin = ctrl.stats();
  CHECK_EQ(fin.live_sessions, 0);
  CHECK_EQ(fin.live_requests, 0);
  CHECK_EQ(sm.num_sessions(), 0);
  CHECK_EQ(mgr.num_live_sequences(), 0);
  CHECK_EQ(static_cast<int>(mgr.delta_pool().used_slots()), 0);
  CHECK_EQ(mgr.used_state_bytes(), static_cast<std::size_t>(0));

  std::fprintf(stderr,
               "[soak] iterations=%d turns=%d created=%zu evicted_ttl=%llu "
               "evicted_lru=%llu\n",
               kIterations, total_turns, issued.size(),
               (unsigned long long)fin.evicted_sessions_ttl,
               (unsigned long long)fin.evicted_sessions_lru);
  TEST_PASS("test_serving_lifecycle_soak");
  (void)cudaStreamDestroy(stream);
  return 0;
}
