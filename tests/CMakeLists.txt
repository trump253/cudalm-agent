# CUDALM tests. Each test is a standalone executable returning non-zero on
# failure; assertions come from tests/common/check.h (no external framework).

# --- CPU tests ---------------------------------------------------------------
cudalm_add_test(test_bootstrap_cpu test_bootstrap_cpu cpu/test_bootstrap_cpu.cpp)
cudalm_add_test(test_weight_file_cpu test_weight_file_cpu cpu/test_weight_file.cpp)

# --- CUDA tests ---------------------------------------------------------------
cudalm_add_test(test_cuda_smoke test_cuda_smoke cuda/test_cuda_smoke.cpp)
cudalm_add_test(test_elementwise_rmsnorm test_elementwise_rmsnorm cuda/test_elementwise_rmsnorm.cpp)
cudalm_add_test(test_int4_gemv test_int4_gemv cuda/test_int4_gemv.cpp)
cudalm_add_test(test_rope test_rope cuda/test_rope.cpp)
cudalm_add_test(test_kv_cache test_kv_cache cuda/test_kv_cache.cpp)
cudalm_add_test(test_attention test_attention cuda/test_attention.cpp)

# --- Cross-language weight-file tests (Python tools -> C++ loader) ------------
# tools/convert_weights.py (torch, offline) emits data/block_v01.cudalm into
# the build tree; test_weights_crosslang loads it with the PyTorch-free C++
# loader and pins the quantization/RoPE invariants. Python is a tool-only
# dependency; when it is absent the cross-language tests are skipped.
find_program(CUDALM_PYTHON3 NAMES python3)

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
else()
  message(STATUS "python3 not found: cross-language weight tests disabled")
endif()
