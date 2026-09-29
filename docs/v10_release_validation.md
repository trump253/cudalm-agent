# CUDALM v1.0 Release Validation

**Status: PENDING** — this document is filled in by the Phase D
evidence commit with the real results of the local NVIDIA release
validation (release build, full ctest, no-PyTorch guard,
representative compute-sanitizer gate) executed in the release
NVIDIA environment. Do not treat any number in this file as final
until the evidence commit lands.

## Scope

- Validated SHA: _(filled by the evidence commit)_
- Environment: GPU / driver / CUDA / nvcc / compiler / CMake build
  type / checkpoint identity _(filled)_
- Hosted CI: `.github/workflows/ci.yml` → `repository-checks`
  (repository / static guards only — no CUDA build, no GPU tests on
  the hosted runner)
- Local NVIDIA release validation: full release build + full ctest
  (real-checkpoint GPU/integration + serving gates) + representative
  `compute-sanitizer` gate
- Performance: NO rerun — the frozen Phase B evidence stands
  (`benchmarks/v10/`, `docs/v10_performance.md`, evidence SHA
  `eeaef3e0b0cdd9b3dd808787e7b00f468f9b4b6a`)

## Known limitations (unchanged)

- Qwen3.5-0.8B-Base only
- single GPU, single CUDA stream, serial prefill
- single-threaded HTTP frontend (one request at a time)
- raw-text completion — no official Qwen chat template, not
  OpenAI-compatible
