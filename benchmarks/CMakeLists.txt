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
