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

# --- CUDA tests ---------------------------------------------------------------
cudalm_add_test(test_cuda_smoke test_cuda_smoke cuda/test_cuda_smoke.cpp)
cudalm_add_test(test_elementwise_rmsnorm test_elementwise_rmsnorm cuda/test_elementwise_rmsnorm.cpp)
cudalm_add_test(test_int4_gemv test_int4_gemv cuda/test_int4_gemv.cpp)
# v0.2 Phase B: W4A16 GEMV bf16 specialization (Qwen3.5 path) + the bf16
# Qwen3.5 kernel family (zero-centered RMSNorm, fused q/gate split, partial
# rotate-half RoPE, bf16 KV write/decode attention, bf16 elementwise).
cudalm_add_test(test_int4_gemv_bf16 test_int4_gemv_bf16 cuda/test_int4_gemv_bf16.cpp)
cudalm_add_test(test_qwen35_kernels test_qwen35_kernels cuda/test_qwen35_kernels.cpp)
cudalm_add_test(test_rope test_rope cuda/test_rope.cpp)
cudalm_add_test(test_kv_cache test_kv_cache cuda/test_kv_cache.cpp)
cudalm_add_test(test_attention test_attention cuda/test_attention.cpp)

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
  else()
    message(STATUS "tool python lacks torch/safetensors/numpy: "
                   "qwen35 ingestion tests disabled")
  endif()
else()
  message(STATUS "python3 not found: cross-language weight tests disabled")
endif()
