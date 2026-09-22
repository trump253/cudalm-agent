# CUDALM tests. Each test is a standalone executable returning non-zero on
# failure; assertions come from tests/common/check.h (no external framework).

# --- CPU tests ---------------------------------------------------------------
cudalm_add_test(test_bootstrap_cpu test_bootstrap_cpu cpu/test_bootstrap_cpu.cpp)
cudalm_add_test(test_weight_file_cpu test_weight_file_cpu cpu/test_weight_file.cpp)

# --- CUDA tests ---------------------------------------------------------------
cudalm_add_test(test_cuda_smoke test_cuda_smoke cuda/test_cuda_smoke.cpp)
cudalm_add_test(test_elementwise_rmsnorm test_elementwise_rmsnorm cuda/test_elementwise_rmsnorm.cpp)
cudalm_add_test(test_int4_gemv test_int4_gemv cuda/test_int4_gemv.cpp)

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
else()
  message(STATUS "python3 not found: cross-language weight tests disabled")
endif()
