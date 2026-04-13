# AGENTS

## Goal

- Keep `lib/Runtime` as the single runtime center for:
  - AscendC kernel compilation
  - CPU simulation execution with profiling
  - NPU execution path wiring
  - task-graph-based execution and future multi-task scheduling
- Remove architecture drift between old runtime entry points and the new runtime stack.
- Continue shrinking direct dependence on `lib/Runtime/Legacy` without breaking xvm verification.

## Progress

- Runtime architecture has been reorganized into:
  - `Artifact/`
  - `Execution/`
  - `Profile/`
  - `Mix/`
  - `Support/`
  - `Legacy/`
- `runtime-session` is now the single general CLI entry point.
- The previous `runtime-session` local request assembly logic has been moved into:
  - [include/Runtime/Artifact/RuntimeSessionRequestBuilder.h](/Volumes/GM9/code/Codex-Ascend-MLIR/include/Runtime/Artifact/RuntimeSessionRequestBuilder.h)
  - [lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp](/Volumes/GM9/code/Codex-Ascend-MLIR/lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp)
- `runtime_session_main.cpp` now keeps only:
  - CLI parsing
  - summary printing
  - execution orchestration
- Focused runtime verification currently passes on xvm through:
  - [test/tools/runtime/run_runtime.sh](/Volumes/GM9/code/Codex-Ascend-MLIR/test/tools/runtime/run_runtime.sh)
- The repeated mix simulation baseline is part of focused verification and currently passes.
- CLI regression coverage has been added for:
  - conflicting `--artifact-root` / `--kernel`
  - invalid `--kernel-kind` with missing kernel input
- Profiling/session-summary retention and mix-sim stability fixes are already landed and covered by focused runtime verification.
- A follow-up spec for `autotuner + Legacy` audit is written:
  - [docs/superpowers/specs/2026-04-13-autotuner-legacy-cleanup-design.md](/Volumes/GM9/code/Codex-Ascend-MLIR/docs/superpowers/specs/2026-04-13-autotuner-legacy-cleanup-design.md)

## Decisions

- Treat `runtime-session` as the only general runtime CLI.
- Keep request-building logic in the runtime library, not in CLI `main.cpp`.
- Preserve CLI behavior while refactoring internals; do not accept silent semantic drift.
- Use xvm as the authoritative verification environment.
- Treat `Legacy/` as mixed-status implementation code, not as uniformly dead code.
- Do not delete more `Legacy` code until:
  - dependency audit is complete
  - `autotuner` is re-audited against current runtime-native flows
  - cleanup candidates are grouped by prerequisite and risk
- Keep the untracked planning note below untouched:
  - `docs/superpowers/plans/2026-04-10-runtime-taskgraph-mix.md`

## TODO

- Produce the implementation plan for:
  - [docs/superpowers/specs/2026-04-13-autotuner-legacy-cleanup-design.md](/Volumes/GM9/code/Codex-Ascend-MLIR/docs/superpowers/specs/2026-04-13-autotuner-legacy-cleanup-design.md)
- Audit all remaining references to:
  - `Legacy/Compiler`
  - `Legacy/Executor`
  - `Legacy/SimValidator`
  - `Legacy/HostRunnerGen`
  - `Legacy/CompatRuntime`
- Re-audit `tools/autotuner/autotuner_main.cpp` to determine whether it still blocks further `Legacy` cleanup.
- Classify `Legacy` code into:
  - must keep for now
  - candidate for boundary shrink
  - deletable after migration
- After the audit, define the second-round cleanup sequence before deleting more runtime code.
