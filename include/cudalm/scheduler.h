// CUDALM — v0.6 Phase A/B: request scheduler / control plane.
//
// Sits ON TOP of the frozen v0.5 runtime — the scheduler does NOT own the
// model, the state manager, or the stream (non-owning references; all
// three outlive the scheduler):
//
//   Scheduler (this file)
//     -> SequenceForwarder    (single-sequence forward + logits access)
//     -> Qwen35StateManager   (sequence state ownership)
//
// PHASE A EXECUTION MODEL (pinned): one scheduler "iteration" (step())
// advances each eligible request by AT MOST ONE token. In Phase A every
// advance is ONE single-sequence forward_token_with_state() call on the
// SINGLE stream, proving
//     scheduler/control-plane semantics == independent execution
//     semantics
// for request lifecycle, admission, iteration order, prefill/decode
// progression, completion, retirement, and per-request sampling.
//
// PHASE B EXECUTION MODEL (pinned, additive): the SAME per-request
// semantics — each request still advances by AT MOST ONE token per
// iteration, FIFO snapshot order is preserved, prefill stays SERIAL (the
// last prompt token goes through the single path and produces g0) — but
// CONSECUTIVE decode-ready requests in the snapshot (a "decode cohort")
// are advanced by ONE TRUE BATCHED model traversal
// (Qwen35Model::forward_batch_with_state: a single layer chain with
// batched GEMVs / DeltaNet / paged attention over the whole cohort,
// heterogeneous positions and block tables) instead of N single forwards.
// A batch size of 1 keeps the single path; a batch PREFLIGHT failure
// (zero-mutation — e.g. insufficient aggregate KV capacity) falls back to
// the frozen serial path for that cohort (Phase A semantics). Per-row
// output is BIT-IDENTICAL to the frozen single path (row-parity
// contract).
//
// POLICY (deterministic FIFO / round-robin, pinned; NO priority or
// fairness heuristics):
//   * one iteration = a SNAPSHOT of all non-terminal request ids taken at
//     the START of the iteration, ordered by ascending RequestId (==
//     admission order), then each id is advanced by at most one token
//     (re-looked-up per id; nothing is admitted mid-iteration, so no
//     request can sneak into the running snapshot — a request admitted
//     between iterations first participates in the NEXT iteration);
//   * terminal requests are naturally skipped; a request admitted AFTER
//     the snapshot is never part of the current iteration.
//   * a fatal forward Status marks that request Failed (and retires its
//     sequence); the iteration CONTINUES with the remaining snapshot ids
//     (each request advances independently) and step() returns the first
//     error. The failed request is terminal and is NEVER advanced again.
//
// ADMISSION (transactional, pinned): validate the input -> create the
// SequenceState -> register the request -> issue the next RequestId. On
// ANY failure nothing is registered: no half-request, no leaked
// SequenceState, no consumed RequestId (the id is issued only AFTER a
// successful create_sequence).
//
// TERMINAL EXACTLY ONCE (pinned): a request transitions to a terminal
// status exactly once (EOS / max_new_tokens / cancel / fatal Status) and
// its sequence is retired EXACTLY ONCE on that transition. A stale
// request is never advanced again; its record stays inspectable via
// get() (terminal requests are kept, never erased).
//
// CANCELLATION (pinned contract — tested + documented): cancel(id) on a
// Waiting/Running request -> Cancelled + retire; cancel(id) on an
// ALREADY-TERMINAL request is IDEMPOTENT (returns ok, no state change);
// cancel(id) on an UNKNOWN id is a Status error (fail loud).
//
// The caller passes the single CUDA stream; it must equal the manager
// pools' stream (and the model's config must equal the manager's), which
// the v0.5 compatibility gate inside forward_token_with_state enforces
// fail-loud on the first forward — the scheduler adds no stream
// machinery of its own.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cudalm/qwen35_model.h"
#include "cudalm/qwen35_state_manager.h"
#include "cudalm/request.h"
#include "cudalm/sampling.h"
#include "cudalm/weight_format.h"  // Status

namespace cudalm {

// Single-sequence forward + last-logits access. The real implementation
// (ModelForwarder) wraps a loaded Qwen35Model; tests may substitute a
// CPU fake (no model, deterministic logits) so the control plane is
// testable without a checkpoint.
class SequenceForwarder {
 public:
  virtual ~SequenceForwarder() = default;

  // One single-token forward for `sequence_id` (stream-ordered; must be
  // the manager's single stream — enforced by the v0.5 gate).
  virtual Status forward_token(int token_id, SequenceId sequence_id,
                               Qwen35StateManager& mgr,
                               cudaStream_t stream) = 0;

  // The FULL logits [vocab] (bf16) produced by the LAST forward_token,
  // copied to host (the scheduler samples host-side, like the v0.4
  // generator).
  virtual Status logits_to_host(std::vector<__nv_bfloat16>* out,
                                cudaStream_t stream) const = 0;

  virtual int vocab_size() const = 0;

  // ---- v0.6 Phase B: TRUE BATCHED decode capability ---------------------
  // Capable forwarders (ModelForwarder) can run a whole DECODE cohort of
  // B sequences in ONE model traversal (forward_batch_with_state —
  // zero-mutation preflight; on failure NOTHING changed, so the scheduler
  // falls back to the frozen serial path). CPU fakes return false and the
  // scheduler keeps the Phase A single-forward semantics.
  virtual bool supports_batch() const { return false; }
  // One batched decode forward for B LIVE sequences (row b =
  // (token_ids[b], sids[b]); the caller guarantees all rows are
  // decode-ready — each forwards its last generated token at its current
  // sequence length).
  virtual Status forward_batch(const int* token_ids, const SequenceId* sids,
                               int B, Qwen35StateManager& mgr,
                               cudaStream_t stream) {
    (void)token_ids;
    (void)sids;
    (void)B;
    (void)mgr;
    (void)stream;
    return Status::error("batch forward not supported by this forwarder");
  }
  // The FULL logits [B][vocab] (bf16) of the LAST forward_batch — ONE
  // device-to-host copy of the whole batch (the scheduler then samples
  // each row with that request's own Sampler).
  virtual Status logits_batch_to_host(std::vector<__nv_bfloat16>* out, int B,
                                      cudaStream_t stream) const {
    (void)out;
    (void)B;
    (void)stream;
    return Status::error("batch logits not supported by this forwarder");
  }
};

// The real SequenceForwarder: a non-owning wrapper over a loaded
// Qwen35Model (delegates to forward_token_with_state / logits()).
class ModelForwarder : public SequenceForwarder {
 public:
  explicit ModelForwarder(Qwen35Model& model);

  Status forward_token(int token_id, SequenceId sequence_id,
                       Qwen35StateManager& mgr, cudaStream_t stream) override;
  Status logits_to_host(std::vector<__nv_bfloat16>* out,
                        cudaStream_t stream) const override;
  int vocab_size() const override;

  bool supports_batch() const override { return model_.loaded(); }
  Status forward_batch(const int* token_ids, const SequenceId* sids, int B,
                       Qwen35StateManager& mgr,
                       cudaStream_t stream) override;
  Status logits_batch_to_host(std::vector<__nv_bfloat16>* out, int B,
                              cudaStream_t stream) const override;

 private:
  Qwen35Model& model_;
  mutable std::vector<__nv_bfloat16> host_logits_;  // D2H scratch
};

// v0.6 Phase C: scheduler / runtime SERVING METRICS (observability ONLY —
// these counters never gate or alter correctness; the control-plane
// semantics are pinned by the tests independently of them).
//
// ATTEMPT vs COMMITTED semantics (pinned, unambiguous):
//  * "issued attempt" = a forward call that was sent to the forwarder
//    (counted BEFORE the call; a FAILED attempt still counts as an attempt
//    and as a traversal — the model did run once).
//  * "committed" = the forward SUCCEEDED and its progress was committed
//    (a failed single forward commits NOTHING: no logical token, no
//    Request.forward_count).
//  * A batched forward is only ever COMMITTED (the preflight + the
//    forward_batch call itself succeed before any batch counter moves; a
//    preflight failure is NOT a batch attempt — the cohort falls back to
//    the serial path, whose attempts are counted there).
//
// CONSEQUENT DEFINITIONS:
//  * single_forward_calls        = ISSUED single attempts (incl. failed)
//  * successful_single_forward_calls = COMMITTED (successful) single forwards
//  * batch_forward_calls         = COMMITTED batched forwards (each = ONE model
//                                  traversal, never B)
//  * model_traversal_calls       = single_forward_calls + batch_forward_calls
//                                  = ISSUED model traversals (a failed single
//                                  attempt IS one traversal; a committed batch
//                                  of B is ONE)
//  * logical_token_forwards      = successful_single_forward_calls +
//                                  batched_sequence_tokens = COMMITTED logical
//                                  tokens (one per TOKEN advanced; a failed
//                                  attempt contributes ZERO)
//
// KEY DISTINCTION (pinned): "logical sequence-token forwards" counts one
// per TOKEN committed (a successful single forward = 1 token; a committed
// batch of B = B tokens). "model traversal calls" counts one per model
// traversal ISSUED (a single attempt = 1 traversal, failed or not; a
// committed batch of B = ONE traversal, never B). A batch(B=4) is therefore
// 4 logical token-forwards but 1 model traversal. Confusing the two is the
// exact error this struct disambiguates.
struct SchedulerStats {
  // ---- request lifecycle ----------------------------------------------------
  int requests_admitted = 0;   // total requests ever admitted (incl. terminal)
  int requests_live = 0;       // non-terminal right now
  int requests_finished = 0;   // terminal Finished (EOS / max_new_tokens)
  int requests_cancelled = 0;  // terminal Cancelled
  int requests_failed = 0;     // terminal Failed (fatal forward / logits)

  // ---- forward accounting (attempt vs committed, see header above) ----------
  int single_forward_calls = 0;            // ISSUED single attempts (incl. failed)
  int successful_single_forward_calls = 0;  // COMMITTED (successful) single forwards
  int batch_forward_calls = 0;             // COMMITTED batched forwards (ONE each)
  int model_traversal_calls = 0;  // ISSUED traversals = single_forward_calls +
                                  // batch_forward_calls (failed single attempt
                                  // IS one traversal; committed batch = ONE)
  int batched_sequence_tokens = 0;  // sum of B over committed batch forwards
  int logical_token_forwards = 0;   // COMMITTED logical tokens =
                                    // successful_single_forward_calls +
                                    // batched_sequence_tokens (failed attempts
                                    // contribute ZERO)

  // ---- batch shape ------------------------------------------------------------
  int max_batch_size = 0;  // largest committed decode cohort (0 if none)
  double avg_committed_decode_batch_size = 0.0;  // batched tokens / batch calls
  // Cohort size B of every COMMITTED batched forward (B >= 2), in order.
  std::vector<int> batch_size_trace;
  // Size of EVERY decode cohort advanced (1 for a size-1 single decode, B
  // for a committed batch; prefill advances are NOT decode cohorts and are
  // not recorded). This is the trace used to evidence batch GROW / SHRINK.
  std::vector<int> decode_cohort_trace;
};

// The v0.6 Phase A request scheduler (control plane; deterministic FIFO).
class Scheduler {
 public:
  // One admission. All fields are validated (fail loud, nothing
  // registered on failure).
  struct Spec {
    std::vector<int> prompt;  // non-empty; every token in [0, vocab)
    int max_new_tokens = 1;   // >= 1
    int eos_token_id = -1;    // -1 = no EOS gate; else in [0, vocab)
    SamplingConfig sampling;  // per-request (seed inside; v0.4 contract)
  };

  // fwd/mgr/stream are NON-OWNING (they outlive the scheduler). `stream`
  // is the single stream and must equal the manager pools' stream (the
  // v0.5 compatibility gate inside forward_token_with_state enforces
  // this on the first forward, fail-loud).
  Scheduler(SequenceForwarder& fwd, Qwen35StateManager& mgr,
            cudaStream_t stream);

  // Transactional admission: validate -> create SequenceState -> register
  // -> issue the next (monotonic, never-reused) RequestId. On failure:
  // no RequestId issued, no half-request, no leaked sequence.
  Status admit(const Spec& spec, RequestId* out_request_id);

  // Pinned contract: Waiting/Running -> Cancelled + retire (resources
  // reclaimed; the request is never advanced again); already-terminal ->
  // ok (idempotent, no state change); unknown id -> Status error.
  Status cancel(RequestId request_id);

  // One iteration: snapshot the non-terminal ids (ascending == admission
  // order) at the start, then advance each by at most one token (prefill
  // or decode; the v0.4 progression semantics, one
  // forward_token_with_state per advance). A forward failure marks that
  // request Failed + retires its sequence; the remaining snapshot ids
  // still advance; the first error is returned.
  //
  // v0.6 Phase B: CONSECUTIVE decode-ready requests in the snapshot (a
  // "decode cohort") are advanced in ONE TRUE BATCHED forward when the
  // forwarder supports it (batch size >= 2; a size-1 cohort keeps the
  // frozen single path). The cohort is NOT reordered (snapshot order is
  // preserved), prefill requests in between stay on the serial path (the
  // LAST prompt token is always serial and produces g0; the next
  // iteration's g0 forward is what joins a decode cohort), and a batch
  // PREFLIGHT failure (e.g. insufficient aggregate KV capacity) falls
  // back to the frozen serial path for that cohort — Phase A semantics.
  // The observable per-request progression (tokens, logits, counts,
  // finish reasons, sampling streams) is IDENTICAL to Phase A.
  Status step();

  // ---- v0.6 Phase B instrumentation (pinned) ----------------------------
  // Number of batched forwards issued so far (each one advances a whole
  // decode cohort — a cohort of B costs EXACTLY ONE batch forward, not B
  // single forwards).
  int batch_forward_calls() const { return batch_forward_calls_; }
  // Number of single forwards ISSUED so far (Phase A path: prefill
  // advances, size-1 cohorts, and every serial fallback) — includes FAILED
  // attempts (the call was sent to the forwarder).
  int single_forward_calls() const { return single_forward_calls_; }
  // Number of single forwards that SUCCEEDED (committed) — a failed attempt
  // increments single_forward_calls() but NOT this counter (and commits no
  // logical token / no Request.forward_count).
  int successful_single_forward_calls() const {
    return successful_single_forward_calls_;
  }
  // Largest decode cohort (batch size) advanced so far (0 if none).
  int max_batch_size() const { return max_batch_size_; }
  // Number of batch attempts that FAILED preflight and fell back to the
  // frozen serial path.
  int batch_fallback_calls() const { return batch_fallback_calls_; }

  // ---- v0.6 Phase C: serving metrics (observability only) --------------------
  // A snapshot of the serving / runtime statistics (see SchedulerStats).
  // Pure observation: reading this never mutates scheduler state.
  SchedulerStats stats() const;

  // Run step() until EVERY request is terminal (deterministic; bounded —
  // fails loud if the bound is exceeded, which cannot happen for a
  // correctly advancing forwarder). FAILURE ISOLATION (pinned): a failing
  // step does NOT stop the other live requests — the failed request is
  // terminal (Failed + retired) and is never advanced again, while the
  // remaining requests keep advancing in the following iterations; the
  // first error encountered (or ok) is returned.
  Status run();

  // Inspect (nullptr if unknown). Terminal requests remain inspectable.
  const Request* get(RequestId request_id) const;
  int num_requests() const;  // all requests (incl. terminal)
  int num_live() const;      // non-terminal requests
  RequestId next_request_id() const;  // the next id admit() will issue

 private:
  // Advance `r` by EXACTLY one token: the next prompt token (prefill) or
  // the last generated token (decode). After the forward, sample the
  // next token ONLY when it is not yet known (after the LAST prompt
  // forward -> g0, or after a decode forward -> g_k); no sampling on
  // early prefill forwards. Completes the request (retire exactly once)
  // on EOS / max_new_tokens. A forward Status failure marks the request
  // Failed + retires and returns the error.
  Status advance_one(Request& r);

  // v0.6 Phase B: advance the maximal decode cohort runnable[i..j) in ONE
  // batched forward (zero-mutation preflight; fallback to the frozen
  // serial path on a preflight failure). `first_error` accumulates the
  // first failure. All cohort rows are decode-ready.
  void advance_batch_run(const std::vector<RequestId>& runnable,
                         std::size_t i, std::size_t j, Status* first_error);

  // Decode-ready: prefill complete AND at least one generated token (the
  // next forward is a DECODE of the last generated token).
  static bool is_decode_ready(const Request& r);

  // Terminal transition: set status/reason and retire the sequence
  // EXACTLY ONCE (internal invariant: the sequence is live, so a retire
  // failure is a logic bug -> fail loud).
  void finish(Request& r, RequestStatus status, FinishReason reason);

  SequenceForwarder& fwd_;
  Qwen35StateManager& mgr_;
  cudaStream_t stream_;
  std::map<RequestId, Request> requests_;  // ascending order == FIFO
  RequestId next_id_ = 1;
  // v0.6 Phase B instrumentation.
  int batch_forward_calls_ = 0;
  int single_forward_calls_ = 0;  // ISSUED single attempts (incl. failed)
  int max_batch_size_ = 0;
  int batch_fallback_calls_ = 0;
  // v0.6 Phase C: COMMITTED (successful) single forwards — the failed
  // attempt increments single_forward_calls_ but never this counter.
  int successful_single_forward_calls_ = 0;
  // v0.6 Phase C serving-metric accumulators (observability only).
  int batched_sequence_tokens_ = 0;          // sum of B over committed batches
  std::vector<int> batch_size_trace_;        // B of each committed batch
  std::vector<int> decode_cohort_trace_;     // size of every decode cohort
};

}  // namespace cudalm
