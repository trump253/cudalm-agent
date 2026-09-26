# CUDALM — Native C++/CUDA Quantized LLM Inference Engine
#
# v0.1 build system notes:
#   * Target: single GPU, Turing sm_75, CUDA 11.8, C++17.
#   * The system CMake on the build host is 3.16.3, which predates
#     `enable_language(CUDA)` (3.18+). We therefore compile .cu files with
#     explicit `nvcc` custom commands and link the resulting objects into the
#     targets. This keeps the build working on CMake >= 3.16 without any
#     CMake-only-for-newer features.
#   * The runtime library (`libcudalm`) must not depend on PyTorch/pybind in
#     any form. `scripts/check_no_torch.sh` (wired into ctest) enforces this
#     by scanning include/ and src/ for forbidden symbols.

cmake_minimum_required(VERSION 3.16)
project(cudalm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Keep the build loud: -Wreturn-type caught a real UB (CHECK() `return 1`
# gave a lambda a deduced `int` return type with a missing return on the
# happy path — benign at -O0, miscompiled at -O2).
add_compile_options(-Wall -Wextra -Werror)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --------------------------------------------------------------------------
# CUDA toolkit discovery
# --------------------------------------------------------------------------
# Note: predefining an EMPTY cache variable would make find_path skip its
# search entirely (the variable is already "defined"), so provide the real
# default directly and only fall back to a search if that path is absent.
set(CUDALM_CUDA_HOME "/usr/local/cuda-11.8" CACHE PATH "CUDA toolkit root (e.g. /usr/local/cuda-11.8)")
if(NOT EXISTS "${CUDALM_CUDA_HOME}/bin/nvcc")
  find_path(CUDALM_CUDA_HOME
            NAMES bin/nvcc
            PATHS /usr/local/cuda-11.8 /usr/local/cuda
            NO_DEFAULT_PATH)
endif()
if(NOT CUDALM_CUDA_HOME OR NOT EXISTS "${CUDALM_CUDA_HOME}/bin/nvcc")
  message(FATAL_ERROR "nvcc not found. Set -DCUDALM_CUDA_HOME=<cuda root>.")
endif()
message(STATUS "CUDALM CUDA_HOME = ${CUDALM_CUDA_HOME}")

set(CUDALM_ARCH "sm_75" CACHE STRING "CUDA compute architecture")
set(CUDA_NVCC_FLAGS
    -std=c++17
    --gpu-architecture=${CUDALM_ARCH}
    -O2
    -Xcompiler=-fPIC)

# Compile one .cu translation unit to an object file.
function(cudalm_add_cuda_object out_var src)
  get_filename_component(_name ${src} NAME_WE)
  set(_obj ${CMAKE_CURRENT_BINARY_DIR}/cuda_objs/${_name}.o)
  add_custom_command(
    OUTPUT ${_obj}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/cuda_objs
    COMMAND ${CUDALM_CUDA_HOME}/bin/nvcc ${CUDA_NVCC_FLAGS}
            -I${CMAKE_SOURCE_DIR}/include
            -c ${src} -o ${_obj}
    DEPENDS ${src}
    COMMENT "nvcc  ${src}"
    VERBATIM)
  set(${out_var} ${_obj} PARENT_SCOPE)
endfunction()

# Collect a list of .cu files into a list of object-file targets.
function(cudalm_cuda_objects objs_var)
  set(_objs)
  foreach(_src ${ARGN})
    cudalm_add_cuda_object(_o ${_src})
    list(APPEND _objs ${_o})
  endforeach()
  set(${objs_var} ${_objs} PARENT_SCOPE)
endfunction()

# --------------------------------------------------------------------------
# libcudalm
# --------------------------------------------------------------------------
file(GLOB_RECURSE CUDALM_CPP_SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE CUDALM_CU_SOURCES CONFIGURE_DEPENDS src/*.cu)
cudalm_cuda_objects(CUDALM_CUDA_OBJECTS ${CUDALM_CU_SOURCES})

add_library(cudalm STATIC
  ${CUDALM_CPP_SOURCES}
  ${CUDALM_CUDA_OBJECTS})
target_include_directories(cudalm PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CUDALM_CUDA_HOME}/include)
# CUDA 11.8 ships the runtime as libcudart.so (the static archive is named
# libcudart_static.a). The shared library keeps the link line dependency-free.
target_link_libraries(cudalm PUBLIC ${CUDALM_CUDA_HOME}/lib64/libcudart.so)

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
enable_testing()

# Helper: add a test executable that links libcudalm.
function(cudalm_add_test exe name)
  add_executable(${exe} ${ARGN})
  target_link_libraries(${exe} PRIVATE cudalm)
  add_test(NAME ${name} COMMAND ${exe})
endfunction()

# --------------------------------------------------------------------------
# Tools (differential validation against the pinned engine; NOT runtime code)
# --------------------------------------------------------------------------
# tools/dump_qwen35_tokenizer.cpp: dumps native encode/decode/NFC/pre-tokenize
# stages for hex inputs so tools/validate_qwen35_tokenizer.py can compare each
# stage EXACTLY with the pinned tokenizers engine (docs §20.5 validation).
add_executable(tool_qwen35_tokenizer_dump
               tools/dump_qwen35_tokenizer.cpp)
target_link_libraries(tool_qwen35_tokenizer_dump PRIVATE cudalm)

# --------------------------------------------------------------------------
# User-facing CLI (v0.4 Phase C): raw UTF-8 prompt -> generated text.
# Single request, serial, no streaming / chat template / batching (docs §21).
# --------------------------------------------------------------------------
add_executable(cudalm-generate tools/cudalm_generate.cpp)
target_link_libraries(cudalm-generate PRIVATE cudalm)

add_subdirectory(tests)
add_subdirectory(benchmarks)

# Forbidden-dependency guard: the runtime must stay PyTorch/pybind free.
add_test(NAME forbidden_deps_check
         COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/check_no_torch.sh ${CMAKE_SOURCE_DIR})
