# AGENTS

## Goal

- Keep `lib/Runtime` as the single runtime center for:
  - AscendC kernel compilation
  - CPU simulation execution with profiling
  - NPU execution path wiring
  - task-graph-based execution and future multi-task scheduling
- Keep `runtime-session` as the only general runtime CLI entry point.
- Keep xvm CPU-simulation verification green while finishing real NPU validation.
- Treat the original runtime task as architecturally complete except for real-device validation and follow-up fixes exposed by real hardware.

## Current Development Mode

- Edit locally in this workspace, then verify on xvm under `/home/niu/code/Ascend-MLIR`.
- xvm is the authoritative development verification environment.
- For xvm build/test/debug workflow, follow `examples/dev-env.md`.
- xvm currently uses CANN 9.1:
  - `/home/niu/Ascend/latest -> /home/niu/Ascend/cann-9.1.0`
  - Set `ASCEND_HOME_PATH=/home/niu/Ascend/latest` or source `/home/niu/Ascend/latest/set_env.sh` before verification.
- Do not treat xvm CPU simulation as real NPU completion.
- For shared x86/aarch64 developer workflows, use `scripts/real-npu-ci/` as
  the containerized real-NPU host runner. Developer machines trigger jobs; build
  and real-device execution stay on the aarch64 NPU host.
- Prebuild the real-NPU runner image once per dependency stack, preferably via
  `scripts/real-npu-ci/build-aarch64-image.sh --with-llvm` so pinned LLVM/MLIR
  is automatically built into `/opt/llvm/build`; validation jobs should reuse
  an existing image tag instead of rebuilding Docker images.
- Keep `scripts/real-npu-ci/versions.env` as the version pin for automated
  aarch64 image rebuilds. PyAsc follows the source repo/ref under validation;
  rebuild the image when PyAsc requires a new LLVM/MLIR stack or extra system
  packages.
- The generic aarch64 image may use Ubuntu 22.04 userland even if the real-NPU host
  OS differs, as long as build and real-device execution remain inside the
  container and driver/CANN/device nodes are mounted from the host.
- Before any candidate fix, generated-kernel variant, or new runtime artifact is
  advanced to 910C real-device validation, run the matching xvm Ascend910B1
  simulation gate first and require `session.backend=sim`,
  `session.result=success`, and `session.validation=pass`.
- Full transformer xvm runtime E2E is no longer a normal gate. It is too large
  for routine simulator validation after f32 matmul falls back to vec kernels.
  Use transformer fragments plus the full transformer compile/translate/artifact
  smoke as the xvm gate, and treat full transformer xvm runtime as explicit
  longrun/performance diagnostic only.
- Keep the xvm result for the original demo separate from the xvm result for a
  candidate fix. An original-demo xvm pass does not authorize taking an
  unverified candidate fix to the real NPU.
- Early-return checkpoints and other intentionally incomplete diagnostic kernels
  may be run on the real NPU only for localization. They are not proof that a
  fix is ready for board validation.
- The remote real-NPU host is run-only for this project. It does not have the full LLVM development stack.
- For real-device NPU debugging and operation, follow `examples/real-npu.md`.
- For remote NPU validation:
  1. Generate artifacts, data, expected outputs, and run manifests on xvm.
  2. Build a run-manifest-only `runtime-session` on xvm with `ASCEND_RUNTIME_SESSION_RUN_ONLY=ON`.
  3. Package only the runner, runtime artifacts, data, run manifest, and required runtime libraries.
  4. Run on the remote host after sourcing the remote CANN/driver environment.
- Use `ASCEND_DEVICE_ID=5` for the shared real-NPU host unless the user explicitly changes the device.

## Current Progress

- Runtime architecture has been reorganized around:
  - `Artifact/`
  - `Execution/`
  - `Profile/`
  - `Mix/`
  - `Support/`
- `runtime-session` keeps CLI parsing, summary printing, and execution orchestration only.
- Runtime request assembly lives in:
  - `include/Runtime/Artifact/RuntimeSessionRequestBuilder.h`
  - `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp`
- Shared runtime frontend behavior lives in:
  - `include/Runtime/Execution/RuntimeFrontendCore.h`
  - `lib/Runtime/Execution/RuntimeFrontendCore.cpp`
- `runtime-session` and the C API use the shared frontend core for compile request assembly, single-task run preparation, normalized execution, and summary interpretation.
- `DefaultExecutionRunner` delegates to `NativeExecutionRunner`.
- The old active `Legacy/` runtime implementation path has been removed from the current runtime execution path.
- `NpuBackend` is wired through the runtime-native execution path and honors `ASCEND_DEVICE_ID`.
- `NativeExecutionRunner` now reports `rtStreamSynchronize` failures instead of treating failed real-device execution as success.
- A run-manifest-only `runtime-session` build path exists so the remote host can execute prebuilt artifacts without loading CANN compiler/simulator dependencies at process startup.
- Remote host CANN 9.1 toolkit and A3 ops are installed under `/data/{username}/Ascend`.
- xvm CANN 9.1 toolkit and A3 ops are installed under `/home/niu/Ascend`, with top-level `latest` switched to 9.1.
- Current xvm 9.1 verification passes:
  - `bash test/tools/runtime/run_runtime.sh`
    - `RC=0`
    - runtime tests report `113 passed, 0 failed`
    - SimBackend vec/mix baseline passes
    - repeated mix simulation baseline passes
  - `bash test/tools/runtime/run_simbackend_examples.sh`
    - `RC=0`
    - 6 SimBackend examples pass
  - `bash test/tools/examples/example_pipelines.sh`
    - `RC=0`
    - 6 example pipelines pass
    - cross-session runtime-session smoke passes
- Transformer xvm coverage is split deliberately:
  - routine gate: `test/tools/examples/transformer-fragments.mlir`, which
    validates layernorm, QKV, QKV head projection, attention score, softmax,
    context, and composed attention block fragments with simulator
    `session.validation=pass`;
  - full graph smoke: `examples/transformer/run-mainline.sh --batch 1 --seq 1
    --log`, which checks normalize/kernelize/kernel split, Phase5 backend,
    CANN translation, and per-kernel artifact generation without running the
    full simulator runtime;
  - longrun diagnostic only:
    `test/tools/examples/transformer-runtime-e2e.mlir` with
    `AFIR_ENABLE_LONGRUN_TESTS=1`.
- Transformer shape matrix coverage is gated explicitly by
  `test/tools/longrun/transformer-shape-matrix.mlir`; set
  `AFIR_ENABLE_LONGRUN_TESTS=1` when intentionally running that long gate.
- Transformer kernel census diagnostics are gated explicitly by
  `test/tools/longrun/transformer-kernel-census.mlir`. The current
  `batch=1, seq=1` census is 53 kernels / 53 tasks / 47 graph edges, bucketed
  as 49 vec / 1 cube / 3 mix, with 26 root tasks and 25 weight/constant prepack
  candidates.
- Kernel DAG visualization lives in
  `test/tools/diagnostics/ascend_kernel_dag_viz.py`; it reads runtime manifest,
  run manifest, and kernelized IR to emit SVG plus summary JSON for op summary,
  shape/tile, critical path, prepack roots, and simple fusion candidates.
- Real NPU validation current state:
  - `examples/real-npu-microcases` passes on device 7 through the containerized
    run-only runner; the suite covers `const640`, copy variants, `relu_only`,
    and `broadcast_add`, with `session.result=success` and
    `session.validation=pass` for each case.
  - `scripts/sync-and-submit.sh --case all --ref 2ecc4d4-real-all --device-id 7 --jobs 6`
    passes on device 7. It covers `add-broadcast-concat`,
    `broadcast-add-reduce`, `gather-elementwise-fusion`,
    `matmul-add-leakyrelu`, `relu-broadcast-transpose`,
    `split-relu-brc-add-mul`, and `real-npu-multikernel`.
  - All real-NPU suite case logs report `session.result=success` and
    `session.validation=pass`, and collected plog summaries contain no new
    `errorStr`.
  - Full transformer real-NPU diagnostic passes on device 5 for commit
    `acbf997`: job
    `/data/nyh/real-npu-jobs/20260522-083037-3f9d754-f32vec-diag2-dev5-cmd`
    reports `session.backend=npu`, `session.result=success`,
    `session.validation=pass`, and no plog `errorStr`. Its matching full xvm
    simulator runtime is not a routine gate; the simulator keeps executing past
    the old 600s timeout and is treated as longrun/performance diagnostic.
  - Detailed real-NPU debugging experience, plog triage, and per-demo root-cause notes are maintained in `examples/real-npu.md`.

## Decisions

- Keep request-building logic in the runtime library, not in CLI `main.cpp`.
- Preserve CLI behavior while refactoring internals; do not accept silent semantic drift.
- Use xvm for all normal build/test verification.
- For real-NPU debugging status, do not present xvm simulation success as a
  project result. Treat xvm success only as the required gate before a
  candidate is allowed onto the real device.
- Use the real NPU host only for run-only validation and hardware-specific debugging.
- Do not run a proposed fix on the real NPU until that exact candidate has
  passed xvm Ascend910B1 simulation, except for explicitly labeled diagnostic
  checkpoint runs.
- Treat any new real-NPU failure as a kernel/ABI/tiling investigation until evidence proves otherwise.
- Do not reopen old `Legacy/` cleanup work as an active blocker unless a concrete current dependency reappears.
- Treat these as the current CPU-simulation regression baselines:
  - `test/tools/runtime/run_runtime.sh`
  - `test/tools/runtime/run_simbackend_examples.sh`
  - `test/tools/examples/example_pipelines.sh`

## Notes

- `examples/dev-env.md` is the operational guide for xvm build/test/debug workflow.
- `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md` remains the detailed
  implementation status and verification record; keep `AGENTS.md` focused on
  continuation rules, environment, and current next steps.
- `examples/real-npu.md` is the single operational guide for remote real-device NPU debugging, containerized real-NPU runner usage, run-only packaging, plog triage, and current real-NPU findings.
- If xvm reports stale or inconsistent build state, prefer a clean reconfigure before interpreting failures as CANN 9.1 regressions.
- Do not include unrelated dirty files in commits or reviews.
