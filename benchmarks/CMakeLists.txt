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
