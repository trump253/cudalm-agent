# CUDALM tests. Each test is a standalone executable returning non-zero on
# failure; assertions come from tests/common/check.h (no external framework).

# --- CPU tests ---------------------------------------------------------------
cudalm_add_test(test_bootstrap_cpu test_bootstrap_cpu cpu/test_bootstrap_cpu.cpp)
cudalm_add_test(test_weight_file_cpu test_weight_file_cpu cpu/test_weight_file.cpp)

# v0.2: Qwen3.5 hybrid-decoder configuration contract (pinned 0.8B values,
# derived dims, hybrid schedule, validity rules, 88-byte config blob).
cudalm_add_test(test_qwen35_config test_qwen35_config cpu/test_qwen35_config.cpp)

# v0.2 Phase C: Gated DeltaNet supported-config contract (the decode kernels
# hardcode kernel-4 / state-3 / head-dim-128 / k==v heads); the pinned config
# must be accepted and each violation must fatal-abort (child-process check).
cudalm_add_test(test_qwen35_deltanet_config test_qwen35_deltanet_config
                cpu/test_qwen35_deltanet_config.cpp)

# v0.4 Phase A: greedy generation helpers (CPU-only, no checkpoint).
#   * argmax: maximize the numeric BF16 logit value; tie -> lowest token id
#     (normal / negative / exact-tie / single / bf16-rounding-tie).
#   * stop controller: EOS (immediate + included + no forward after) /
#     max_new_tokens / max_seq_len / EOS-priority + the pure decisions.
cudalm_add_test(test_greedy_argmax test_greedy_argmax cpu/test_greedy_argmax.cpp)
cudalm_add_test(test_greedy_stop test_greedy_stop cpu/test_greedy_stop.cpp)

# v0.4 Phase C: CPU sampling (cudalm/sampling.h) — mathematically pinned
# synthetic-logits correctness: greedy compatibility (frozen argmax),
# temperature scaling (== softmax(logits/T)), top-k (ties -> lowest id,
# >vocab clamps), top-p (minimal prefix, >=1 kept), k+p composition (fixed
# order), seed reproducibility, different-seed streams, invalid config,
# extreme/all-negative logits, exact ties, single remaining candidate.
cudalm_add_test(test_sampling test_sampling cpu/test_sampling.cpp)

# v0.4 Phase C: CLI argument parsing (cudalm/generate_cli.h) — CPU-only, no
# model: --help, missing required args, unknown flags, bad numerics, mode
# resolution (greedy default / sampling flags / --greedy exclusivity).
cudalm_add_test(test_generate_cli_args test_generate_cli_args
                cpu/test_generate_cli_args.cpp)

# v0.5 Phase A: hybrid state-manager foundations (CPU metadata gates).
#   * test_fixed_id_pool: the shared allocator core — unique live ids,
#     LIFO reuse, OOM, double-free/invalid rejection, fixed-seed stress,
#     exact accounting.
#   * test_kv_block_table: logical block table vs a fake page source —
#     position decomposition boundary matrix, minimal prefix growth,
#     one page per block, TRANSACTIONAL OOM rollback (no leaked page),
#     bounds, clear, LIFO reuse identity.
#   * test_state_pool_formulas: ALL pool sizing derives from the
#     Qwen35Config hybrid schedule (0.8B: 6 full/18 linear; page/slot
#     byte sizes for page_tokens 4/8/16) — no magic constants.
cudalm_add_test(test_fixed_id_pool test_fixed_id_pool
                cpu/test_fixed_id_pool.cpp)
cudalm_add_test(test_kv_block_table test_kv_block_table
                cpu/test_kv_block_table.cpp)
cudalm_add_test(test_state_pool_formulas test_state_pool_formulas
                cpu/test_state_pool_formulas.cpp)

# v0.6 Phase A: request scheduler / control-plane gate (CPU test: NO
# checkpoint, NO model — a deterministic fake forwarder over the REAL
# Qwen35StateManager pools). Gates: request lifecycle (Waiting/Running/
# terminal records stay inspectable), monotonic never-reused RequestId,
# TRANSACTIONAL admission (capacity exhaustion + input validation: no
# leaked sequence, no half-request, no consumed RequestId; re-admission
# after retire/cancel), deterministic FIFO/round-robin iteration with
# snapshot semantics (admission order == RequestId order == iteration
# order; a request admitted between iterations joins the NEXT one),
# prefill/decode coexistence, EOS / max_new_tokens completion (exact
# generated stream + forward count N + m - 1), cancel + retire (idempotent
# re-cancel ok, unknown id fail-loud, stale request never advances),
# per-request SAMPLING RNG isolation (A(seed 42) interleaved == A alone;
# same prompt + same seed => identical streams regardless of
# interleaving), fatal Status (Failed + retire exactly once, the
# iteration's remaining snapshot ids still advance).
cudalm_add_test(test_qwen35_scheduler test_qwen35_scheduler
                cpu/test_qwen35_scheduler.cpp)

# --- CUDA tests ---------------------------------------------------------------
cudalm_add_test(test_cuda_smoke test_cuda_smoke cuda/test_cuda_smoke.cpp)
cudalm_add_test(test_elementwise_rmsnorm test_elementwise_rmsnorm cuda/test_elementwise_rmsnorm.cpp)
cudalm_add_test(test_int4_gemv test_int4_gemv cuda/test_int4_gemv.cpp)
# v0.2 Phase B: W4A16 GEMV bf16 specialization (Qwen3.5 path) + the bf16
# Qwen3.5 kernel family (zero-centered RMSNorm, fused q/gate split, partial
# rotate-half RoPE, bf16 KV write/decode attention, bf16 elementwise).
cudalm_add_test(test_int4_gemv_bf16 test_int4_gemv_bf16 cuda/test_int4_gemv_bf16.cpp)
# Plain BF16 GEMV (the tied-LM-head matmul, v0.3 Phase B) — standalone numeric
# hard gate: vec4 (K%8==0, aligned) + scalar (K%8!=0 / misaligned) + zero-row,
# vs a CPU FP32-accumulate -> BF16 RNE reference. No checkpoint needed.
cudalm_add_test(test_bf16_gemv test_bf16_gemv cuda/test_bf16_gemv.cpp)
cudalm_add_test(test_qwen35_kernels test_qwen35_kernels cuda/test_qwen35_kernels.cpp)
cudalm_add_test(test_rope test_rope cuda/test_rope.cpp)
cudalm_add_test(test_kv_cache test_kv_cache cuda/test_kv_cache.cpp)
cudalm_add_test(test_attention test_attention cuda/test_attention.cpp)
# v0.5 Phase A: hybrid state-manager DEVICE gate (no checkpoint): KV page
# pool + Delta state pool allocator/OOM/stress, REAL on-device reuse
# contamination (dirty pattern -> release -> reuse -> all zeros, conv+rec
# and K+V across every layer), block table on the real pool, sequence
# lifecycle (no aliasing, reset/retire, SequenceId monotone non-reuse),
# and OOM transactional behavior (table/accounting unchanged on failure).
cudalm_add_test(test_qwen35_state_manager test_qwen35_state_manager
                cuda/test_qwen35_state_manager.cpp)
# v0.5 Phase B: paged-KV kernel standalone hard gate (synthetic data, NO
# checkpoint): paged KV write (physical rows bit-exact, unwritten rows
# untouched) + paged causal decode attention BIT-IDENTICAL to the frozen
# contiguous attention under two NON-trivial physical mappings (not the
# identity) across the position boundary matrix (0, pt-1, pt, pt+1,
# multi-page crossings), with OUT-OF-RANGE sentinels in the block-table
# stale region proving stale entries are never read.
cudalm_add_test(test_paged_kv test_paged_kv cuda/test_paged_kv.cpp)
# v0.5 Phase B: single-sequence external-state parity hard gate (real
# checkpoint). A = frozen legacy owned-state forward_token (p0..p5, page
# boundaries at positions 2+4 with page_tokens=2) vs B = Qwen35StateManager
# sequence + forward_token_with_state (DeltaNet forward_with_state over the
# pool slot + paged attention over the pool + device block table). Gates:
# every step's embedding/24 layer finals/final-norm/FULL logits BIT-IDENTICAL,
# final hybrid state (18x delta conv+rec, 6x KV logical rows via block table)
# BIT-IDENTICAL, OOM-BEFORE-MUTATION (exhausted 3-page pool: 7th forward
# fails with length/block-table/pool/Delta/KV all unchanged), reset parity
# (reset_sequence + same tokens == first external run), fail-loud unknown id
# + invalid token. Self-skips (77) when the checkpoint is absent; the
# Phase B evidence environment MUST actually run it.
add_executable(test_qwen35_state_parity cuda/test_qwen35_state_parity.cpp)
target_link_libraries(test_qwen35_state_parity PRIVATE cudalm)
add_test(NAME test_qwen35_state_parity
         COMMAND test_qwen35_state_parity
                 ${CMAKE_BINARY_DIR}/data/qwen35_08b_full.cudalm
                 /root/models/Qwen3.5-0.8B-Base
                 ${CUDALM_TOOL_PYTHON3}
                 ${CMAKE_SOURCE_DIR})
set_tests_properties(test_qwen35_state_parity PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 1800)

# --- Cross-language weight-file tests (Python tools -> C++ loader) ------------
# tools/convert_weights.py (torch, offline) emits data/block_v01.cudalm into
# the build tree; test_weights_crosslang loads it with the PyTorch-free C++
# loader and pins the quantization/RoPE invariants. Python is a tool-only
# dependency; when it is absent the cross-language tests are skipped.
find_program(CUDALM_PYTHON3 NAMES python3)
# Tool python: prefers the standalone venv (torch + safetensors + numpy);
# falls back to system python3 (stdlib-only tools still work there).
find_program(CUDALM_TOOL_PYTHON3 NAMES python3 HINTS /root/py311/venv/bin)
if(NOT CUDALM_TOOL_PYTHON3)
  set(CUDALM_TOOL_PYTHON3 ${CUDALM_PYTHON3})
endif()
if(CUDALM_TOOL_PYTHON3)
  execute_process(
    COMMAND ${CUDALM_TOOL_PYTHON3} -c "import torch, safetensors, numpy"
    RESULT_VARIABLE CUDALM_TOOL_PY_ROI_OK
    OUTPUT_QUIET ERROR_QUIET)
else()
  set(CUDALM_TOOL_PY_ROI_OK 1)
endif()

# v0.5 Phase C: multi-sequence INTERLEAVED external-state hard gate (real
# checkpoint) — the v0.5 final sign-off test. Single model, single CUDA
# stream, one forward_token_with_state per step; the call order interleaves
# SequenceIds (no scheduler). Gates: A (6 tok) + B (5 tok) interleaved
# (A0 B0 A1 A2 B1 A3 B2 A4 B3 A5 B4) vs INDEPENDENT fresh-manager
# references — per step 24 layer finals + final norm + FULL logits and
# final hybrid state (18x delta conv/rec + 6x logical KV rows) BIT-IDENTICAL;
# cross-sequence ISOLATION (full hybrid state pre/post a key A and B step);
# RETIRE/reuse (A's id permanently invalid, A's 3 pages + delta slot
# reclaimed, B unchanged; C reuses A's slot + pages with a FRESH id and
# matches the fresh-C reference from fresh-zero state, B still unchanged);
# RESET-ONE-SEQUENCE isolation (reset B clears only B, live C unchanged).
# Self-skips (77) when the checkpoint is absent; the Phase C evidence
# environment MUST actually run it.
add_executable(test_qwen35_state_interleave cuda/test_qwen35_state_interleave.cpp)
target_link_libraries(test_qwen35_state_interleave PRIVATE cudalm)
add_test(NAME test_qwen35_state_interleave
         COMMAND test_qwen35_state_interleave
                 ${CMAKE_BINARY_DIR}/data/qwen35_08b_full.cudalm
                 /root/models/Qwen3.5-0.8B-Base
                 ${CUDALM_TOOL_PYTHON3}
                 ${CMAKE_SOURCE_DIR})
set_tests_properties(test_qwen35_state_interleave PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 1800)

# v0.6 Phase A: scheduler INTEGRATION hard gate (real checkpoint) — the
# Phase A sign-off test. Scheduler semantics == independent execution:
# A (3-tok prompt, max_new 3, seed 42) + B (7-tok prompt, max_new 2, seed
# 123) admitted, 2 iterations, then C (2-tok prompt, max_new 3, seed 7)
# admitted DYNAMICALLY, run to completion (deterministic FIFO/round-robin;
# one single-sequence forward_token_with_state per advance, single model +
# single stream — NO batched CUDA compute in Phase A). Gates: per request
# A/B/C the generated token IDs, the FULL logits[248320] that produced each
# generated token, the forward count, and the finish reason are EXACT vs
# INDEPENDENT fresh-manager references (direct forwards, not the
# scheduler); B's FULL hybrid state at length 4 (before A finishes) and
# length 5 (after A's finish + retire) == B-alone reference states
# (isolation: another request's finish/retire changes nothing in a
# still-live request's state); SAMPLING ISOLATION (same prompt + same seed
# => identical streams: X alone == Y alone == X+Y interleaved).
# Self-skips (77) when the checkpoint is absent; the Phase A evidence
# environment MUST actually run it.
add_executable(test_qwen35_scheduler_integration
               cuda/test_qwen35_scheduler_integration.cpp)
target_link_libraries(test_qwen35_scheduler_integration PRIVATE cudalm)
add_test(NAME test_qwen35_scheduler_integration
         COMMAND test_qwen35_scheduler_integration
                 ${CMAKE_BINARY_DIR}/data/qwen35_08b_full.cudalm
                 /root/models/Qwen3.5-0.8B-Base
                 ${CUDALM_TOOL_PYTHON3}
                 ${CMAKE_SOURCE_DIR})
set_tests_properties(test_qwen35_scheduler_integration PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 1800)

if(CUDALM_PYTHON3)
  set(CUDALM_CROSSLANG_WEIGHTS ${CMAKE_BINARY_DIR}/data/block_v01.cudalm)

  add_custom_command(
    OUTPUT ${CUDALM_CROSSLANG_WEIGHTS}
    COMMAND ${CUDALM_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/convert_weights.py
            --out ${CUDALM_CROSSLANG_WEIGHTS}
    DEPENDS ${CMAKE_SOURCE_DIR}/tools/convert_weights.py
            ${CMAKE_SOURCE_DIR}/tools/common/binfmt.py
    COMMENT "convert_weights: generating ${CUDALM_CROSSLANG_WEIGHTS}"
    VERBATIM)
  add_custom_target(gen_block_weights DEPENDS ${CUDALM_CROSSLANG_WEIGHTS})

  add_test(NAME gen_block_weights
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target gen_block_weights)
  set_tests_properties(gen_block_weights PROPERTIES FIXTURES_SETUP cudalm_weights)

  # Python-side contract selftest (quantization invariants, determinism,
  # file round-trip through the Python reader).
  add_test(NAME test_weights_python_selftest
           COMMAND ${CUDALM_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/convert_weights.py --selftest)

  add_executable(test_weights_crosslang cpu/test_weights_crosslang.cpp)
  target_link_libraries(test_weights_crosslang PRIVATE cudalm)
  add_test(NAME test_weights_crosslang
           COMMAND test_weights_crosslang ${CUDALM_CROSSLANG_WEIGHTS})
  set_tests_properties(test_weights_crosslang PROPERTIES
    FIXTURES_REQUIRED cudalm_weights)

  # --- Golden-reference cross-language tests ---------------------------------
  # tools/generate_golden.py (torch, offline) emits CUDLMG01 golden files for
  # position 0 and position 7 (non-trivial KV history); test_golden_file loads
  # them with the PyTorch-free C++ loader and pins the tensor set + invariants.
  set(CUDALM_GOLDEN_P0 ${CMAKE_BINARY_DIR}/data/block_v01_golden_p0.cudalm)
  set(CUDALM_GOLDEN_P7 ${CMAKE_BINARY_DIR}/data/block_v01_golden_p7.cudalm)

  add_custom_command(
    OUTPUT ${CUDALM_GOLDEN_P0}
    COMMAND ${CUDALM_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/generate_golden.py
            --out ${CUDALM_GOLDEN_P0} --position 0
    DEPENDS ${CMAKE_SOURCE_DIR}/tools/generate_golden.py
            ${CMAKE_SOURCE_DIR}/tools/convert_weights.py
            ${CMAKE_SOURCE_DIR}/tools/common/binfmt.py
    COMMENT "generate_golden: generating ${CUDALM_GOLDEN_P0}"
    VERBATIM)
  add_custom_command(
    OUTPUT ${CUDALM_GOLDEN_P7}
    COMMAND ${CUDALM_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/generate_golden.py
            --out ${CUDALM_GOLDEN_P7} --position 7
    DEPENDS ${CMAKE_SOURCE_DIR}/tools/generate_golden.py
            ${CMAKE_SOURCE_DIR}/tools/convert_weights.py
            ${CMAKE_SOURCE_DIR}/tools/common/binfmt.py
    COMMENT "generate_golden: generating ${CUDALM_GOLDEN_P7}"
    VERBATIM)
  add_custom_target(gen_golden_weights
    DEPENDS ${CUDALM_GOLDEN_P0} ${CUDALM_GOLDEN_P7})

  add_test(NAME gen_golden_weights
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target gen_golden_weights)
  set_tests_properties(gen_golden_weights PROPERTIES FIXTURES_SETUP cudalm_golden)

  # Python-side golden selftest (determinism, container round-trip,
  # p=0 identity invariants, history usage at p=7).
  add_test(NAME test_golden_python_selftest
            COMMAND ${CUDALM_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/generate_golden.py --selftest)

  add_executable(test_golden_file cpu/test_golden_file.cpp)
  target_link_libraries(test_golden_file PRIVATE cudalm)
  add_test(NAME test_golden_file
            COMMAND test_golden_file ${CUDALM_GOLDEN_P0} ${CUDALM_GOLDEN_P7})
  set_tests_properties(test_golden_file PROPERTIES
    FIXTURES_REQUIRED cudalm_golden)

  # --- End-to-end decoder block golden test (M8) ----------------------------
  # Runs DecoderBlock::forward at position 0 and position 7 and compares all
  # 16 stage tensors (tolerance 1e-2) + KV states (bit-exact) against the
  # CUDLMG01 goldens, plus the p=0 rope-identity invariants.
  add_executable(test_decoder_block cuda/test_decoder_block.cpp)
  target_link_libraries(test_decoder_block PRIVATE cudalm)
  add_test(NAME test_decoder_block
            COMMAND test_decoder_block ${CUDALM_CROSSLANG_WEIGHTS}
                    ${CUDALM_GOLDEN_P0} ${CUDALM_GOLDEN_P7})
  set_tests_properties(test_decoder_block PROPERTIES
    FIXTURES_REQUIRED "cudalm_weights;cudalm_golden")

  # --- Generalized projection-shape test (v0.1.1) --------------------------
  # H=1024, n_heads=16, head_dim=128 -> q_proj_out=2048 != H; n_kv_heads=8
  # -> kv_proj_out=1024. Proves the runtime, loaders, and tools no longer
  # assume Q width == hidden_size. Same binary as test_decoder_block: the
  # test is fully config-driven from the fixture files.
  set(CUDALM_GENERAL_CONFIG
      "{\"hidden_size\":1024,\"n_heads\":16,\"n_kv_heads\":8,\"head_dim\":128,\"intermediate_size\":2816,\"group_size\":128,\"max_seq_len\":512}")
  set(CUDALM_GENERAL_WEIGHTS ${CMAKE_BINARY_DIR}/data/block_v01_general.cudalm)
  set(CUDALM_GENERAL_GOLDEN_P0 ${CMAKE_BINARY_DIR}/data/block_v01_general_golden_p0.cudalm)
  set(CUDALM_GENERAL_GOLDEN_P7 ${CMAKE_BINARY_DIR}/data/block_v01_general_golden_p7.cudalm)

  add_custom_command(
    OUTPUT ${CUDALM_GENERAL_WEIGHTS}
    COMMAND ${CUDALM_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/convert_weights.py
            --out ${CUDALM_GENERAL_WEIGHTS} --config ${CUDALM_GENERAL_CONFIG}
    DEPENDS ${CMAKE_SOURCE_DIR}/tools/convert_weights.py
            ${CMAKE_SOURCE_DIR}/tools/common/binfmt.py
    COMMENT "convert_weights: generating ${CUDALM_GENERAL_WEIGHTS}"
    VERBATIM)
  add_custom_command(
    OUTPUT ${CUDALM_GENERAL_GOLDEN_P0}
    COMMAND ${CUDALM_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/generate_golden.py
            --out ${CUDALM_GENERAL_GOLDEN_P0} --position 0
            --config ${CUDALM_GENERAL_CONFIG}
    DEPENDS ${CUDALM_GENERAL_WEIGHTS}
    COMMENT "generate_golden: generating ${CUDALM_GENERAL_GOLDEN_P0}"
    VERBATIM)
  add_custom_command(
    OUTPUT ${CUDALM_GENERAL_GOLDEN_P7}
    COMMAND ${CUDALM_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/generate_golden.py
            --out ${CUDALM_GENERAL_GOLDEN_P7} --position 7
            --config ${CUDALM_GENERAL_CONFIG}
    DEPENDS ${CUDALM_GENERAL_WEIGHTS}
    COMMENT "generate_golden: generating ${CUDALM_GENERAL_GOLDEN_P7}"
    VERBATIM)
  add_custom_target(gen_general_weights
    DEPENDS ${CUDALM_GENERAL_WEIGHTS}
            ${CUDALM_GENERAL_GOLDEN_P0} ${CUDALM_GENERAL_GOLDEN_P7})

  add_test(NAME gen_general_weights
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target gen_general_weights)
  set_tests_properties(gen_general_weights PROPERTIES
    FIXTURES_SETUP cudalm_weights_general)

  add_test(NAME test_decoder_block_general
            COMMAND test_decoder_block ${CUDALM_GENERAL_WEIGHTS}
                    ${CUDALM_GENERAL_GOLDEN_P0} ${CUDALM_GENERAL_GOLDEN_P7})
  set_tests_properties(test_decoder_block_general PROPERTIES
    FIXTURES_REQUIRED cudalm_weights_general)

  # --- v0.2: .cudalm v2 container cross-language format test -----------------
  # tests/fixtures/gen_v2_format.py (stdlib Python, tools/common/cudalm_v2.py)
  # emits one fully valid synthetic file + 13 corruption cases; the C++ reader
  # must accept exactly the valid one. Stdlib-only, so system python3 suffices.
  set(CUDALM_V2_FORMAT_DIR ${CMAKE_BINARY_DIR}/data/v2format)
  add_custom_command(
    OUTPUT ${CUDALM_V2_FORMAT_DIR}/valid.cudalm
    COMMAND ${CUDALM_PYTHON3} ${CMAKE_SOURCE_DIR}/tests/fixtures/gen_v2_format.py
            ${CUDALM_V2_FORMAT_DIR}
    DEPENDS ${CMAKE_SOURCE_DIR}/tests/fixtures/gen_v2_format.py
            ${CMAKE_SOURCE_DIR}/tools/common/cudalm_v2.py
    COMMENT "gen_v2_format: generating ${CUDALM_V2_FORMAT_DIR}"
    VERBATIM)
  add_custom_target(gen_v2_format_fixture DEPENDS ${CUDALM_V2_FORMAT_DIR}/valid.cudalm)

  add_test(NAME gen_v2_format_fixture
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target gen_v2_format_fixture)
  set_tests_properties(gen_v2_format_fixture PROPERTIES FIXTURES_SETUP cudalm_v2_format)

  add_executable(test_cudalm_v2_format cpu/test_cudalm_v2_format.cpp)
  target_link_libraries(test_cudalm_v2_format PRIVATE cudalm)
  add_test(NAME test_cudalm_v2_format
           COMMAND test_cudalm_v2_format ${CUDALM_V2_FORMAT_DIR})
  set_tests_properties(test_cudalm_v2_format PROPERTIES
    FIXTURES_REQUIRED cudalm_v2_format)

  # --- v0.2: Qwen3.5 micro-stack --tokens argument validation (stdlib) -------
  # tools/common/token_seq.py is stdlib-only (no torch): it validates the
  # strictly-consecutive 0-anchored token sequence the hybrid micro-stack
  # golden generator (--microstack-prefix --tokens) accepts, rejecting gaps,
  # duplicates, non-0 starts and out-of-order entries. Runs under system
  # python3 (no venv / checkpoint needed).
  add_test(NAME test_qwen35_tokens_validation
           COMMAND ${CUDALM_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/common/token_seq.py)
  
  # --- v0.2: Qwen3.5 real-checkpoint ingestion (torch tool python) -----------
  # Requires the pinned checkpoint in /root/models (never committed) and a
  # tool python with torch + safetensors + numpy. The test self-skips (77)
  # when the checkpoint is absent.
  if(CUDALM_TOOL_PY_ROI_OK EQUAL 0)
    add_test(NAME test_convert_qwen35_selftest
             COMMAND ${CUDALM_TOOL_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/convert_qwen35.py
                     --selftest)

    add_executable(test_qwen35_ingestion_real cpu/test_qwen35_ingestion_real.cpp)
    target_link_libraries(test_qwen35_ingestion_real PRIVATE cudalm)
    add_test(NAME test_qwen35_ingestion_real
             COMMAND test_qwen35_ingestion_real
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_microstack.cudalm
                     /root/models/Qwen3.5-0.8B-Base
                     ${CUDALM_TOOL_PYTHON3}
                     ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_qwen35_ingestion_real PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 900)

    # --- v0.2 Phase B: Qwen3.5 full-attention golden gate (layer 3) ---------
    # tools/generate_qwen35_golden.py --selftest: synthetic-weights container
    # round-trip + reference-pipeline invariants (no checkpoint needed, but
    # requires the pinned-transformers tool python; self-skips 77 if the
    # installed transformers is not the pinned commit).
    add_test(NAME test_qwen35_golden_python_selftest
             COMMAND ${CUDALM_TOOL_PYTHON3} ${CMAKE_SOURCE_DIR}/tools/generate_qwen35_golden.py
                     --selftest)
    set_tests_properties(test_qwen35_golden_python_selftest PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 600)

    # HARD GATE: native runtime vs quantized PyTorch reference. The test
    # converts layer 3 from the real pinned checkpoint, generates CUDLMG02
    # goldens at p=0 and p=5 (seeded, non-zero KV history at p=5), runs
    # Qwen35FullAttentionLayer and compares all 21 stages + KV state.
    # Self-skips (77) when the checkpoint is absent.
    add_executable(test_qwen35_full_attention_golden
                   cuda/test_qwen35_full_attention_golden.cpp)
    target_link_libraries(test_qwen35_full_attention_golden PRIVATE cudalm)
    add_test(NAME test_qwen35_full_attention_golden
             COMMAND test_qwen35_full_attention_golden
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_l3.cudalm
                     /root/models/Qwen3.5-0.8B-Base
                     ${CMAKE_BINARY_DIR}/data/qwen35_golden_l3_p0.cudalm
                     ${CMAKE_BINARY_DIR}/data/qwen35_golden_l3_p5.cudalm
                     ${CUDALM_TOOL_PYTHON3}
                     ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_qwen35_full_attention_golden PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 1800)

    # --- v0.2 Phase C: Qwen3.5 Gated DeltaNet golden gate (layer 0) ---------
    # HARD GATE: native runtime vs quantized PyTorch reference. The test
    # converts layer 0 (a DeltaNet layer) from the real pinned checkpoint,
    # generates CUDLMG02 goldens for three state-transition scenarios
    # (A: first token / zero state; B: consecutive p0->p1 from the runtime's
    # own state; C: seeded non-zero conv+recurrent state), and compares all
    # 22 stages + stage.g (fp32) + the persistent state (conv_state bf16
    # [6144,3] + recurrent_state fp32 [16,128,128]) — the state is the Phase C
    # hard gate. Self-skips (77) when the checkpoint is absent.
    add_executable(test_qwen35_deltanet_golden
                   cuda/test_qwen35_deltanet_golden.cpp)
    target_link_libraries(test_qwen35_deltanet_golden PRIVATE cudalm)
    add_test(NAME test_qwen35_deltanet_golden
             COMMAND test_qwen35_deltanet_golden
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_l0.cudalm
                     /root/models/Qwen3.5-0.8B-Base
                     ${CMAKE_BINARY_DIR}/data/qwen35_golden_l0_A.cudalm
                     ${CMAKE_BINARY_DIR}/data/qwen35_golden_l0_B.cudalm
                     ${CMAKE_BINARY_DIR}/data/qwen35_golden_l0_C.cudalm
                     ${CUDALM_TOOL_PYTHON3}
                     ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_qwen35_deltanet_golden PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 1800)

    # --- v0.2 Phase D: Qwen3.5 4-layer hybrid micro-stack golden gate -------
    # HARD GATE: the native Qwen35HybridMicroStack (real checkpoint layers
    # 0-3: 3x Gated DeltaNet + 1x full attention, run as a 0->1->2->3 chain)
    # vs the pinned-transformers oracle running the SAME quantized layers in
    # order. The test converts layers 0-3, generates per-(token,layer)
    # CUDLMG02 goldens for scenario A (p=0) and scenario B (sequential
    # p=0,1,2 from the runtime's OWN prior-step state, no reset in between),
    # and compares every layer's bf16 stages + stage.g (fp32) + the persistent
    # state (DeltaNet conv_state+recurrent_state, full-attention K+V cache
    # rows) + the micro-stack final output. Self-skips (77) when the
    # checkpoint is absent. Distinct .cudalm output from the ingestion test.
    add_executable(test_qwen35_hybrid_microstack_golden
                   cuda/test_qwen35_hybrid_microstack_golden.cpp)
    target_link_libraries(test_qwen35_hybrid_microstack_golden PRIVATE cudalm)
    add_test(NAME test_qwen35_hybrid_microstack_golden
             COMMAND test_qwen35_hybrid_microstack_golden
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_microstack4.cudalm
                     /root/models/Qwen3.5-0.8B-Base
                     ${CMAKE_BINARY_DIR}/data/qwen35_msA
                     ${CMAKE_BINARY_DIR}/data/qwen35_msB
                     ${CUDALM_TOOL_PYTHON3}
                     ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_qwen35_hybrid_microstack_golden PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 1800)

    # --- v0.3 Phase A: Qwen3.5 full-model structure + ownership gate -------
    # HARD GATE: convert the COMPLETE Qwen3.5-0.8B-Base (all 24 decoder layers
    # + embed_tokens + final norm + tie metadata) in one shot, validate the
    # full-model tensor contract on the host (24/24 layers, embedding/final-
    # norm/LM-head shapes+dtype, exact hybrid schedule, weight tying), then
    # load the full Qwen35Model (device memory) and prove reset_state() covers
    # every one of the 24 layers (seed each layer's state non-zero, reset,
    # verify all zero). No forward/logits (that is v0.3 Phase B). Self-skips
    # (77) when the checkpoint is absent.
    add_executable(test_qwen35_full_model cuda/test_qwen35_full_model.cpp)
    target_link_libraries(test_qwen35_full_model PRIVATE cudalm)
    add_test(NAME test_qwen35_full_model
             COMMAND test_qwen35_full_model
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_full.cudalm
                     /root/models/Qwen3.5-0.8B-Base
                     ${CUDALM_TOOL_PYTHON3}
                     ${CMAKE_SOURCE_DIR})
    set_tests_properties(test_qwen35_full_model PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 1800)

    # v0.3 Phase B: full single-token forward golden hard gate. Runs the
    # COMPLETE model (embedding -> 24 layers -> final RMSNorm -> tied LM head
    # -> logits) on deterministic tokens (A: p0; B: p0->p1->p2, runtime threads
    # its own state) and compares embedding output + 24x layer final output +
    # final norm + FULL logits [vocab] + 18x DeltaNet conv/recurrent state +
    # 6x FullAttention KV rows 0..p against the pinned-quantized oracle.
    # Self-skips (77) when the checkpoint is absent.
    add_executable(test_qwen35_full_forward cuda/test_qwen35_full_forward.cpp)
    target_link_libraries(test_qwen35_full_forward PRIVATE cudalm)
    add_test(NAME test_qwen35_full_forward
             COMMAND test_qwen35_full_forward
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_full.cudalm
                     /root/models/Qwen3.5-0.8B-Base
                     ${CUDALM_TOOL_PYTHON3}
                     ${CMAKE_SOURCE_DIR}
                     ${CMAKE_BINARY_DIR}/data/qwen35_full)
    set_tests_properties(test_qwen35_full_forward PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 1800)
      # v0.4 Phase A: generation-level golden hard gate (single-request serial
    # prefill + greedy decode). Scenario A (short prompt) + B (longer prompt,
    # final persistent state) + C (repeat-generate contamination). Compares
    # EVERY generated token id + EVERY generation-step FULL logits [vocab] +
    # the final hybrid persistent state against the pinned-quantized oracle.
    # Self-skips (77) when the checkpoint is absent.
    add_executable(test_qwen35_generation cuda/test_qwen35_generation.cpp)
    target_link_libraries(test_qwen35_generation PRIVATE cudalm)
    add_test(NAME test_qwen35_generation
             COMMAND test_qwen35_generation
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_full.cudalm
                     /root/models/Qwen3.5-0.8B-Base
                     ${CUDALM_TOOL_PYTHON3}
                     ${CMAKE_SOURCE_DIR}
                     ${CMAKE_BINARY_DIR}/data/qwen35_gen)
    set_tests_properties(test_qwen35_generation PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 1800)
    # v0.4 Phase A: generator input-contract hardening. Verifies generate()'s
    # input / capacity contract (fail loud): empty prompt / invalid token /
    # invalid eos / max_new_tokens < 0 / prompt_len > max_seq_len all ->
    # ok == false, generated empty, forward_count == 0 (no prefill/decode);
    # max_new_tokens == 0 -> ok == true, empty, no decode. Self-skips (77) when
    # the checkpoint / full-model .cudalm is absent.
    add_executable(test_qwen35_generation_contract
                   cuda/test_qwen35_generation_contract.cpp)
    target_link_libraries(test_qwen35_generation_contract PRIVATE cudalm)
    add_test(NAME test_qwen35_generation_contract
             COMMAND test_qwen35_generation_contract
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_full.cudalm
                     /root/models/Qwen3.5-0.8B-Base
                     ${CUDALM_TOOL_PYTHON3}
                     ${CMAKE_SOURCE_DIR}
                     ${CMAKE_BINARY_DIR}/data/qwen35_gen)
    set_tests_properties(test_qwen35_generation_contract PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 600)
    # v0.4 Phase B: text-level E2E hard gate (raw UTF-8 in / out). Four
    # confident prompts (2 EN, 2 CJK; min top1-top2 logit gap > 0.4 at every
    # step, selected via tools/diag_gen_gaps.py): native encode == pinned HF
    # encode, greedy generation == pinned-quantized oracle, native decode ==
    # pinned HF decode — all EXACT — with the REAL tokenizer EOS (248044),
    # plus the text-level contamination repeat. Self-skips (77) when the
    # checkpoint / tokenizer files are absent.
    add_executable(test_qwen35_text_generation
                   cuda/test_qwen35_text_generation.cpp)
    target_link_libraries(test_qwen35_text_generation PRIVATE cudalm)
    add_test(NAME test_qwen35_text_generation
             COMMAND test_qwen35_text_generation
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_full.cudalm
                     ${CMAKE_BINARY_DIR}/data/qwen35_tokenizer.cudaltk
                     /root/models/Qwen3.5-0.8B-Base
                     ${CUDALM_TOOL_PYTHON3}
                     ${CMAKE_SOURCE_DIR}
                     ${CMAKE_BINARY_DIR}/data/qwen35_textgen)
    set_tests_properties(test_qwen35_text_generation PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 1800)

    # --- v0.4 Phase C: sampling integration (real checkpoint) ---------------
    # greedy EXACT (old API == greedy-config new API, ids/stop/forward),
    # seed determinism (A(42) == A(42)), the request-level contamination gate
    # A(42) -> B(123) -> A(42) at the id AND the text level (model-state reset
    # + per-request sampler RNG reset), sampling sanity + invalid-config
    # fail-loud. Self-skips (77) when the artifacts are absent.
    add_executable(test_qwen35_sampling cuda/test_qwen35_sampling.cpp)
    target_link_libraries(test_qwen35_sampling PRIVATE cudalm)
    add_test(NAME test_qwen35_sampling
             COMMAND test_qwen35_sampling
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_full.cudalm
                     ${CMAKE_BINARY_DIR}/data/qwen35_tokenizer.cudaltk)
    set_tests_properties(test_qwen35_sampling PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 1800)

    # --- v0.4 Phase C: `cudalm-generate` CLI smoke (the real binary) --------
    # --help / usage errors (exit 2 + message) / bad model+tokenizer paths
    # (exit 1 + message) / invalid sampling config (exit 2) / greedy path /
    # sampling path with same-seed byte-identical stdout / UTF-8 (CJK) prompt.
    add_executable(test_cudalm_generate_cli
                   cuda/test_cudalm_generate_cli.cpp)
    target_link_libraries(test_cudalm_generate_cli PRIVATE cudalm)
    add_test(NAME test_cudalm_generate_cli
             COMMAND test_cudalm_generate_cli
                     $<TARGET_FILE:cudalm-generate>
                     ${CMAKE_BINARY_DIR}/data/qwen35_08b_full.cudalm
                     ${CMAKE_BINARY_DIR}/data/qwen35_tokenizer.cudaltk)
    set_tests_properties(test_cudalm_generate_cli PROPERTIES
      SKIP_RETURN_CODE 77 TIMEOUT 1800)
  else()
    message(STATUS "tool python lacks torch/safetensors/numpy: "
                   "qwen35 ingestion tests disabled")
  endif()
else()
  message(STATUS "python3 not found: cross-language weight tests disabled")
endif()

# --- v0.4 Phase B: native tokenizer (CUDLMTK1) cross-language gate ----------
# tools/convert_qwen35_tokenizer.py (tokenizers lib, offline) transcribes the
# pinned Qwen3.5 tokenizer (tokenizer.json + tokenizer_config.json union) into
# the CUDLMTK1 artifact; tools/gen_tokenizer_refs.py emits the multilingual
# oracle corpus (E/D/D1/N/P lines). test_qwen35_tokenizer loads the artifact
# with the PyTorch-free C++ loader and gates: loader failure contract
# (corruption classes), NFC/pre-tokenization stage vectors, the decode
# contract (padding ids -> "", out-of-range fail loud, is_special exact), and
# EXACT cross-language equality on the whole corpus. Needs the `tokenizers`
# python lib (NOT torch) + the pinned local assets; self-skips (77) when the
# assets are absent.
if(CUDALM_TOOL_PYTHON3)
  execute_process(
    COMMAND ${CUDALM_TOOL_PYTHON3} -c "import tokenizers"
    RESULT_VARIABLE CUDALM_TOOL_PY_TOK_OK
    OUTPUT_QUIET ERROR_QUIET)
endif()
if(CUDALM_TOOL_PYTHON3 AND CUDALM_TOOL_PY_TOK_OK EQUAL 0)
  set(CUDALM_TK_ARTIFACT ${CMAKE_BINARY_DIR}/data/qwen35_tokenizer.cudaltk)
  set(CUDALM_TK_CORPUS ${CMAKE_BINARY_DIR}/data/qwen35_tokenizer_refs.txt)

  # Python-side converter selftest (determinism, round-trip, corruption
  # classes; self-skips 77 when the pinned assets are absent).
  add_test(NAME test_qwen35_tokenizer_python_selftest
           COMMAND ${CUDALM_TOOL_PYTHON3}
                   ${CMAKE_SOURCE_DIR}/tools/convert_qwen35_tokenizer.py
                   --selftest)
  set_tests_properties(test_qwen35_tokenizer_python_selftest PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 300)

  add_executable(test_qwen35_tokenizer cpu/test_qwen35_tokenizer.cpp)
  target_link_libraries(test_qwen35_tokenizer PRIVATE cudalm)
  add_test(NAME test_qwen35_tokenizer
           COMMAND test_qwen35_tokenizer ${CUDALM_TK_ARTIFACT}
                   /root/models/Qwen3.5-0.8B-Base
                   ${CUDALM_TOOL_PYTHON3}
                   ${CMAKE_SOURCE_DIR}
                   ${CUDALM_TK_CORPUS})
  set_tests_properties(test_qwen35_tokenizer PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 600)
else()
  message(STATUS "tokenizers lib not available: tokenizer tests disabled")
endif()
