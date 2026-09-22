# CUDALM tests. Each test is a standalone executable returning non-zero on
# failure; assertions come from tests/common/check.h (no external framework).

# --- CPU tests ---------------------------------------------------------------
cudalm_add_test(test_bootstrap_cpu test_bootstrap_cpu cpu/test_bootstrap_cpu.cpp)
cudalm_add_test(test_weight_file_cpu test_weight_file_cpu cpu/test_weight_file.cpp)

# --- CUDA tests ---------------------------------------------------------------
cudalm_add_test(test_cuda_smoke test_cuda_smoke cuda/test_cuda_smoke.cpp)

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
else()
  message(STATUS "python3 not found: cross-language weight tests disabled")
endif()
