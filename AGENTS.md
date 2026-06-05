# AGENTS

## Goal

- Keep `lib/Runtime` as the single runtime center for AscendC kernel
  compilation, CPU simulation/profiling, NPU execution wiring, task-graph
  execution, and future multi-task scheduling.
- Keep `runtime-session` as the only general runtime CLI entry point.
- Keep xvm CPU-simulation verification green while maintaining real NPU
  validation.
- Treat the runtime architecture task as complete except for real-device
  validation and follow-up fixes exposed by current hardware runs.

## Operating Mode

- Edit locally in this workspace, then verify on xvm under
  `/home/niu/code/Ascend-MLIR`.
- xvm is the authoritative development verification environment. Do not
  interpret stale macOS-side build artifacts as verification evidence.
- Follow `examples/dev-env.md` for xvm build/test/debug workflow.
- xvm currently uses CANN 9.1:
  - `/home/niu/Ascend/latest -> /home/niu/Ascend/cann-9.1.0`
  - set `ASCEND_HOME_PATH=/home/niu/Ascend/latest`, or source
    `/home/niu/Ascend/latest/set_env.sh`, before verification.
- Do not treat xvm CPU simulation as real NPU completion.
- The remote real-NPU host is run-only for this project and does not have the
  full LLVM development stack.
- Follow `examples/real-npu.md` for real-device debugging and operation.
- Use `scripts/real-npu-ci/` for shared x86/aarch64 real-NPU validation:
  developer machines trigger jobs; build and real-device execution stay on the
  aarch64 NPU host.
- Prebuild the real-NPU runner image once per dependency stack, preferably via
  `scripts/real-npu-ci/build-aarch64-image.sh --with-llvm`, and reuse the image
  for validation jobs.
- Keep `scripts/real-npu-ci/versions.env` as the version pin for automated
  aarch64 image rebuilds. Rebuild only when PyAsc, LLVM/MLIR, or system package
  requirements change.
- Use the real-NPU device id from current runner defaults or explicit user
  instructions. Check `scripts/sync-and-submit.sh --help` or
  `scripts/real-npu-ci/docker-run.sh --help` before relying on a default; pass
  `--device-id` explicitly when reproducing a prior result.

## Day-to-Day Ascend Development

- Current routine development is primarily under `lib/Conversion/Ascend`.
- Prefer the focused xvm build entry point for this work:
  `./scripts/build.sh --build-ascend --llvm-build-dir /home/niu/code/llvm-project/llvm/build`
- `--build-ascend` keeps the Ascend conversion/tool workflow available while
  skipping AFIR-only build surfaces by default:
  - builds `AscendConversion`, `ascend-mlir-opt`, `ascend-mlir-translate`,
    `ascend-debug`, `runtime-session`, `mix-compiler`, and
    `mix-tiling-helper`;
  - configures with `ASCEND_ENABLE_AFIR=OFF`, `ASCEND_ENABLE_TESTS=OFF`, and
    `AFIR_ENABLE_BINDING_PYTHON=OFF` unless explicitly overridden.
- For focused conversion regression after `lib/Conversion/Ascend` changes, use
  `ninja -C build-ascend-check check-ascend-conversion` from an xvm CMake
  configuration with `ASCEND_ENABLE_AFIR=OFF` and `ASCEND_ENABLE_TESTS=ON`.
- Do not treat `--build-ascend` or `check-ascend-conversion` as a substitute
  for full AFIR/runtime/real-NPU validation when the change touches AFIR-only
  code, runtime behavior, generated artifacts, or hardware execution.

## Runtime Architecture

- Runtime library layout:
  - `Artifact/`
  - `Execution/`
  - `Profile/`
  - `Mix/`
  - `Support/`
- `runtime-session` keeps CLI parsing, summary printing, and orchestration only.
- Runtime request assembly lives in
  `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h` and
  `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp`.
- Shared frontend behavior lives in
  `include/Runtime/Execution/RuntimeFrontendCore.h` and
  `lib/Runtime/Execution/RuntimeFrontendCore.cpp`.
- `runtime-session` and the C API use the shared frontend core for compile
  request assembly, single-task run preparation, normalized execution, and
  summary interpretation.
- `DefaultExecutionRunner` delegates to `NativeExecutionRunner`.
- The old active `Legacy/` runtime path has been removed from the current
  runtime execution path. Do not reopen it as a blocker unless a concrete
  dependency reappears.
- `NpuBackend` is wired through the runtime-native execution path and honors
  `ASCEND_DEVICE_ID`.
- `NativeExecutionRunner` must report real-device launch/synchronize failures
  instead of treating failed execution as success.
- Run-manifest-only `runtime-session` exists so the remote host can execute
  prebuilt artifacts without loading CANN compiler/simulator dependencies at
  process startup.

## Verification Gates

- Current CPU-simulation regression baselines:
  - `bash test/tools/runtime/run_runtime.sh`
  - `bash test/tools/runtime/run_simbackend_examples.sh`
  - `bash test/tools/examples/example_pipelines.sh`
- Before advancing any candidate fix, generated-kernel variant, or new runtime
  artifact to 910C real-device validation, run the matching xvm Ascend910B1
  simulation gate first and require:
  - `session.backend=sim`
  - `session.result=success`
  - `session.validation=pass`
- Keep original-demo xvm results separate from candidate-fix xvm results. An
  original-demo pass does not authorize taking an unverified candidate to the
  real NPU.
- Early-return checkpoints and intentionally incomplete diagnostic kernels may
  run on the real NPU only for localization. They are not proof that a fix is
  ready for board validation.
- Full transformer xvm runtime E2E is not a normal gate. Use transformer
  fragments plus full transformer compile/translate/artifact smoke as the xvm
  gate; treat full transformer runtime as explicit longrun/performance
  diagnostic only.
- Longrun transformer gates:
  - `test/tools/examples/transformer-runtime-e2e.mlir`
  - `test/tools/longrun/transformer-shape-matrix.mlir`
  - `test/tools/longrun/transformer-kernel-census.mlir`
- Set `AFIR_ENABLE_LONGRUN_TESTS=1` only when intentionally running long gates.
- Kernel DAG diagnostics live in
  `tools/ascend-debug/ascend_debug/kernel_dag.py` and are covered through the
  `ascend-debug` diagnostics CLI tests.

## Real-NPU Validation Flow

1. Generate artifacts, data, expected outputs, and run manifests on xvm.
2. Build run-manifest-only `runtime-session` on xvm with
   `ASCEND_RUNTIME_SESSION_RUN_ONLY=ON`.
3. Package only the runner, runtime artifacts, data, run manifest, and required
   runtime libraries.
4. Run on the remote host after sourcing the remote CANN/driver environment.

For containerized validation, prefer `scripts/sync-and-submit.sh` and the
`scripts/real-npu-ci/` runner. `--skip-sim` is a diagnostic shortcut only; it is
not a readiness gate for candidate kernel fixes.

Before quoting a real-NPU pass/fail, commit, job directory, device id, plog
summary, or transformer task/kernel count, re-check the current source:

- `examples/real-npu.md` for operational guidance and hardware notes.
- `scripts/real-npu-ci/README.md` for runner behavior and case coverage.
- `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md` for implementation status
  and verification records.
- Fresh xvm or real-NPU logs for the exact candidate under discussion.

## Decisions

- Keep request-building logic in the runtime library, not in CLI `main.cpp`.
- Preserve CLI behavior while refactoring internals; do not accept silent
  semantic drift.
- Use xvm for all normal build/test verification.
- For real-NPU debugging status, report real-device facts only. Do not present
  xvm simulation success as a project result.
- Use the real NPU host only for run-only validation and hardware-specific
  debugging.
- Treat any new real-NPU failure as a kernel/ABI/tiling investigation until
  evidence proves otherwise.
- If xvm reports stale or inconsistent build state, prefer a clean reconfigure
  before interpreting failures as CANN 9.1 regressions.
- Do not include unrelated dirty files in commits or reviews.
