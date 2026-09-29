# CUDALM v1.0 Release Validation

**Validated SHA**: `c005c84765854d2cebb88d7f5e9d12e6ba8e6b67`
（分支 `v1.0-portfolio-release`；validation 前 `git status --porcelain`
clean，`git rev-parse HEAD` = 该 SHA）

**Date**: 2026-07-08

## Environment

| item | value |
|---|---|
| GPU (device 0) | NVIDIA GeForce RTX 2080 Ti (11264 MiB, sm_75) |
| driver | 570.172.08 |
| CUDA toolkit / nvcc | 11.8, V11.8.89（`/usr/local/cuda-11.8/bin/nvcc`） |
| host compiler | gcc / g++ 9.4.0 (Ubuntu) |
| CMake / build type | 3.16.3 / Release |
| checkpoint | Qwen/Qwen3.5-0.8B-Base @ `/root/models/Qwen3.5-0.8B-Base`（model.safetensors sha256 `c2b1e5a1…`，converted `build/data/qwen35_08b_full.cudalm` sha256 `8637f987…`——与 Phase B 证据一致） |

## Hosted CI

Workflow `.github/workflows/ci.yml`（job `repository-checks`，
ubuntu-latest）承担 **repository / static guards only**：no-PyTorch
guard、tracked shell `bash -n`、tracked Python `py_compile`（syntax
only，不安装 torch/transformers/tokenizers/numpy）、
repository-relative Markdown 链接检查（`scripts/check_docs_links.py`，
stdlib only）、merge-conflict marker hygiene。

完整 build/runtime validation 依赖 CUDA toolkit、NVIDIA GPU 和 real
checkpoint，因此 hosted CI **不做** CUDA build / GPU test——full
GPU/checkpoint correctness validation 在下方 release NVIDIA 环境
执行。

**Run（SHA `c005c84…`）**: workflow `repository-checks`，run id
`36546380843`（[run URL](https://github.com/trump253/cudalm-agent/actions/runs/36546380843)）
—— **completed / success**。全部 step PASS：no-PyTorch runtime
guard、Shell syntax check (bash -n)、Python syntax check
(py_compile)、Markdown repository-relative link check、Repository
hygiene (merge-conflict markers)。

## Release build（local NVIDIA）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**PASS**（所有 target 构建成功；`cudalm-generate` / `cudalm-chat` /
`cudalm-server` 可执行文件生成）。

## Full ctest（local NVIDIA，real checkpoint present）

```bash
cd build && ctest --output-on-failure
```

**84 total / 84 passed / 0 failed / 0 skipped**（Total Test time
817.10 s）。

- 无 checkpoint-missing skip：checkpoint 在位，所有
  `SKIP_RETURN_CODE 77` 的 real-checkpoint 测试均**实际执行并 PASS**，
  skip list 为空；
- release-critical gates 全部实际执行（非 skip）：full-model /
  golden（`test_qwen35_full_model`、`test_qwen35_full_forward`、
  `test_qwen35_generation`、deltanet/full-attention/golden 族）、
  state-manager / scheduler-session integration
  （`test_qwen35_scheduler_session_integration` 等）、session
  generation（`test_qwen35_session_generation`）、multi-turn /
  continuous batching integration（`test_qwen35_continuous_batching`
  族）、HTTP E2E（`test_serving_http_e2e`）、HTTP disconnect
  （`test_serving_http_disconnect`）、bounded HTTP soak
  （`test_serving_http_soak`）。

## no-PyTorch guard

```bash
bash scripts/check_no_torch.sh .
```

**PASS**（`forbidden_deps_check OK`）。

## Compute Sanitizer（representative final gate）

选最终系统复杂度最高的 GPU integration hard gate
（scheduler + session + hybrid external state + 真 checkpoint）：

```bash
/usr/local/cuda-11.8/bin/compute-sanitizer --tool memcheck --leak-check full \
  ./build/tests/test_qwen35_scheduler_session_integration \
  build/data/qwen35_08b_full.cudalm /root/models/Qwen3.5-0.8B-Base \
  /root/py311/venv/bin/python3 /root/code/cudalm-agent
```

（argv 逐字取自 `ctest -N -V`；与 v08 sign-off 的 canonical 命令一致。）

**结果**：test PASS，**ERROR SUMMARY: 0 errors**，
**LEAK SUMMARY: 0 bytes leaked**。

## Performance

**NO benchmark rerun / NO Nsys / NO NCU**——Phase B frozen evidence
保持不变：`benchmarks/v10/`（raw reports + summary.json +
environment.txt）、`docs/v10_performance.md`（EN）/
`docs/v10_performance_zh.md`（CN），evidence SHA
`eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a`。

## Known limitations（unchanged）

- Qwen3.5-0.8B-Base only（单模型专用 runtime）
- single GPU、single CUDA stream、prefill 串行
- HTTP frontend single-threaded（one request at a time）
- raw-text completion——无官方 Qwen chat template、非
  OpenAI-compatible
- 无 tensor parallel / multi-GPU / speculative decoding / CUDA Graph /
  distributed serving
