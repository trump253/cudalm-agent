// CUDALM — v0.6 Phase A: request scheduler / control plane.
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
// advances each eligible request by AT MOST ONE token, and every advance
// is ONE single-sequence forward_token_with_state() call on the SINGLE
// stream. The v0.5 runtime is reused AS-IS: NO batched CUDA compute, NO
// batched Qwen35Model forward, NO true continuous-batch GPU execution —
// that is Phase B. Phase A's goal is to prove
//     scheduler/control-plane semantics == independent execution
//     semantics
// for request lifecycle, admission, iteration order, prefill/decode
// progression, completion, retirement, and per-request sampling.
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

 private:
  Qwen35Model& model_;
  mutable std::vector<__nv_bfloat16> host_logits_;  // D2H scratch
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
  Status step();

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

  // Terminal transition: set status/reason and retire the sequence
  // EXACTLY ONCE (internal invariant: the sequence is live, so a retire
  // failure is a logic bug -> fail loud).
  void finish(Request& r, RequestStatus status, FinishReason reason);

  SequenceForwarder& fwd_;
  Qwen35StateManager& mgr_;
  cudaStream_t stream_;
  std::map<RequestId, Request> requests_;  // ascending order == FIFO
  RequestId next_id_ = 1;
};

}  // namespace cudalm
