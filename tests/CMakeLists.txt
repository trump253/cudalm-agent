# CUDALM tests. Each test is a standalone executable returning non-zero on
# failure; assertions come from tests/common/check.h (no external framework).

# --- CPU tests ---------------------------------------------------------------
cudalm_add_test(test_bootstrap_cpu test_bootstrap_cpu cpu/test_bootstrap_cpu.cpp)

# --- CUDA tests ---------------------------------------------------------------
cudalm_add_test(test_cuda_smoke test_cuda_smoke cuda/test_cuda_smoke.cpp)
