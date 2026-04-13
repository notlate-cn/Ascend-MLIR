# Autotuner Runtime Normalization Audit

## Scope

This audit covers `tools/autotuner/autotuner_main.cpp` against the current runtime stack. It records what is already runtime-native, what still reaches legacy implementation code indirectly, and whether autotuner is still a blocker for further `Legacy` cleanup.

Source scan used for this audit:
- `tools/autotuner/autotuner_main.cpp`
- `include/Runtime/Artifact/ArtifactCompiler.h`
- `lib/Runtime/Artifact/ArtifactCompiler.cpp`
- `include/Runtime/ExecutionSession.h`
- `lib/Runtime/Execution/ExecutionSession.cpp`
- `include/Runtime/TaskGraph.h`
- `include/Runtime/Profile/ProfileUtils.h`
- `lib/Runtime/Profile/ProfileUtils.cpp`

## Current Runtime-Native Pieces

- Autotuner now builds candidate work around runtime abstractions, not legacy tool flow. It includes `Runtime/ArtifactCompiler.h`, `Runtime/ExecutionSession.h`, `Runtime/ProfileTrace.h`, `Runtime/ProfileUtils.h`, and `Runtime/TaskGraph.h` directly in `tools/autotuner/autotuner_main.cpp`.
- Artifact preparation is routed through `ArtifactCompiler` in `prepareArtifact(...)`, and the autotuner no longer orchestrates compile steps itself. The tool either loads an existing artifact root or asks the runtime library to produce a `KernelArtifact`.
- Candidate execution is runtime-session based. `runSearch(...)` constructs a `TaskGraph`, creates `ExecutionSession(ExecutionBackendKind::Simulation)`, runs each candidate through `session.run(*graphOr)`, and consumes the returned `ProfileTrace`.
- Score extraction is runtime-profile based. `extractRuntimeScore(...)` reads `trace.profileArtifactPaths()`, parses the retained runtime profile JSON, and looks for score-like fields such as `score` and `cycle_count`.
- Profile retention is also runtime-native. Successful candidates call `retainProfileArtifactsForCli(...)` and record both the retained profile path and `session_summary_path` in the final JSON output.
- The tool’s request assembly uses runtime task structures. `buildCandidateGraph(...)` fills `RuntimeTask`, `ExecutionInvocation`, `TensorBinding`, and `TilingBinding` rather than any legacy CLI-specific execution payload.

## Remaining Legacy-Coupled Seams

| Seam | Evidence | Effect |
|---|---|---|
| Autotuner-local `--artifact-root` loader does not normalize `mix_resource_type` | `tools/autotuner/autotuner_main.cpp` loads manifests via `locateManifestPath(...)` and `loadArtifactFromRoot(...)`, but sets `artifact.mixResourceType = MixResourceType::Unknown` instead of parsing a manifest field. `lib/Runtime/Artifact/RuntimeSessionRequestBuilder.cpp` parses `mix_resource_type` in the canonical runtime loader. `ExecutionSession::canScheduleTask(...)` rejects `KernelKind::Mix` tasks when the mix resource type is `Unknown`. | `--artifact-root` loads can misclassify mix artifacts and fail scheduling even when the runtime-native loader would preserve the mix resource type. This is an autotuner-owned seam that still needs normalization. |
| `ArtifactCompiler` still depends on `Legacy/Compiler` for vec/cube source builds | `include/Runtime/Artifact/ArtifactCompiler.h` includes `Runtime/Legacy/Compiler.h`, and `lib/Runtime/Artifact/ArtifactCompiler.cpp` includes `Runtime/Compiler.h` before calling `Compiler::Compile(...)`. The same file routes `KernelKind::Mix` through `MixDirectBackend`. | Autotuner still inherits a legacy compile-path dependency for vec/cube source builds, but not for mix source builds. Because autotuner exposes `--kernel-kind mix`, the blocker is narrower than “all source builds.” |
| Runtime library still compiles legacy execution units | `lib/Runtime/CMakeLists.txt` still builds `Legacy/CompatRuntime.cpp`, `Legacy/Compiler.cpp`, `Legacy/Executor.cpp`, `Legacy/HostRunnerGen.cpp`, and `Legacy/SimValidator.cpp`. | Autotuner is not the direct reason those units remain, but its runtime path still rides on the runtime library that contains them. |
| Simulator execution still validates through legacy-backed backends | `ExecutionSession::run(...)` dispatches to `ExecutionBackend` as an orchestration layer. The direct legacy edges live in `lib/Runtime/Execution/SimBackend.cpp` and `lib/Runtime/Execution/NpuBackend.cpp`, which still include `Runtime/Executor.h` and `Runtime/SimValidator.h`. | Autotuner simulation still depends on legacy execution internals transitively, but the actual legacy coupling is in the backends, not in `ExecutionSession`. |

## Does Autotuner Still Block Legacy Cleanup?

Partially, but only in one narrow way.

Autotuner itself no longer directly depends on `Compiler`, `Executor`, `SimValidator`, `HostRunnerGen`, `msprof`, or `perf-report`. The current source scan did not find those references in `tools/autotuner/autotuner_main.cpp`, and the tool now uses `ArtifactCompiler`, `ExecutionSession`, `TaskGraph`, and runtime profile retention helpers instead.

The remaining autotuner-specific blockers are narrower than the original audit stated:

- The compile path through `ArtifactCompiler` still reaches `Legacy/Compiler` for vec/cube source builds.
- The local `--artifact-root` manifest loader still leaves `mixResourceType` as `Unknown`, which can break mix scheduling even though the canonical runtime loader already parses `mix_resource_type`.

That means autotuner still indirectly blocks `Legacy/Compiler` removal for vec/cube source builds, and it still needs a manifest-loading normalization pass before mix artifact-root loads are fully safe.

Autotuner does not appear to block `Legacy/HostRunnerGen` cleanup directly. It also does not own the `Legacy/Executor` or `Legacy/SimValidator` seams; the direct legacy edges are in `SimBackend` and `NpuBackend`, while `ExecutionSession` is only the dispatcher/orchestration layer.

## Follow-Up Tasks

- Narrow `ArtifactCompiler` so the autotuner vec/cube compile path no longer requires `Legacy/Compiler`, or introduce a runtime-native compile path for those source builds.
- Normalize the autotuner `--artifact-root` loader so it preserves `mix_resource_type` from the manifest instead of hardcoding `Unknown`.
- Keep `Legacy/HostRunnerGen` cleanup separate from autotuner work unless a new source scan shows a direct autotuner reference.
- Re-audit `SimBackend` and `NpuBackend` as the next blockers for `Legacy/Executor` and `Legacy/SimValidator`; autotuner only inherits those dependencies transitively today.
- If `ArtifactCompiler` is refactored, rerun the autotuner xvm smoke path and confirm that `--kernel` and `--artifact-root` still produce the same retained profile JSON and best-config summary.
