# CUDALM benchmarks. Standalone CUDA executables (no test framework, no
# PyTorch). Results are saved as JSON under benchmarks/results/ (committed
# evidence, M9).

add_executable(bench_decoder_block bench_decoder_block.cpp)
target_link_libraries(bench_decoder_block PRIVATE cudalm)
