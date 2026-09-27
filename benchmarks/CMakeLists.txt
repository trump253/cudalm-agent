# CUDALM benchmarks. Standalone CUDA executables (no test framework, no
# PyTorch). Results are saved as JSON under benchmarks/results/ (committed
# evidence, M9).

add_executable(bench_decoder_block bench_decoder_block.cpp)
target_link_libraries(bench_decoder_block PRIVATE cudalm)

# v0.2 Phase B: Qwen3.5 full-attention layer 3 latency breakdown (bf16 +
# W4A16). Run with a converted .cudalm v2 (layer 3) file, e.g.
#   build/benchmarks/bench_qwen35_full_attention build/data/qwen35_08b_l3.cudalm 0 100 \
#       benchmarks/results/bench_qwen35_full_attention.json
add_executable(bench_qwen35_full_attention bench_qwen35_full_attention.cpp)
target_link_libraries(bench_qwen35_full_attention PRIVATE cudalm)

# v0.2 Phase D: Qwen3.5 4-layer hybrid micro-stack (layers 0-3) baseline
# decode-step latency. Run with a converted .cudalm v2 (layers 0-3) file, e.g.
#   build/benchmarks/bench_qwen35_hybrid_microstack build/data/qwen35_08b_microstack4.cudalm 100 512 \
#       benchmarks/results/bench_qwen35_hybrid_microstack.json
add_executable(bench_qwen35_hybrid_microstack bench_qwen35_hybrid_microstack.cpp)
target_link_libraries(bench_qwen35_hybrid_microstack PRIVATE cudalm)

# v0.6 Phase C: dynamic continuous-batching reproducible benchmark (real
# Qwen3.5-0.8B-Base). Compares the SAME four-request workload (dynamic
# arrival, true batched decode) executed INDEPENDENT/SERIAL (each request
# alone) vs SCHEDULER CONTINUOUS BATCHED. Warmup + multiple measured runs,
# each bracketed by cudaStreamSynchronize. NO performance pass threshold (the
# correctness-first batch kernels are untuned; tuning is v0.7). Run with a
# converted .cudalm v2 (full model) file, e.g.
#   build/benchmarks/bench_qwen35_continuous_batching \
#       build/data/qwen35_08b_full.cudalm /root/models/Qwen3.5-0.8B-Base \
#       <python> <src_dir> benchmarks/v06_continuous_batching.txt --no-convert
add_executable(bench_qwen35_continuous_batching bench_qwen35_continuous_batching.cpp)
target_link_libraries(bench_qwen35_continuous_batching PRIVATE cudalm)

# v0.7 Phase B: dedicated W4A16 GEMV microbenchmark over the Qwen3.5-0.8B
# production shape census (frozen R4 baseline vs R1/R2/R8 row-tile variants,
# B=1/2/3, CUDA-event timing, weighted production score). Synthetic
# deterministic fixtures — no checkpoint needed.
add_executable(bench_w4a16_qwen35_shapes bench_w4a16_qwen35_shapes.cpp)
target_link_libraries(bench_w4a16_qwen35_shapes PRIVATE cudalm)
