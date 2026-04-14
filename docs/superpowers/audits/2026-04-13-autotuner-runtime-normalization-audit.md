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
| Simulator execution now runs through runtime-native seams | `ExecutionSession::run(...)` dispatches to `ExecutionBackend`, and the execution path now flows through `DefaultExecutionRunner` -> `NativeExecutionRunner`. | Autotuner simulation no longer owns a direct legacy execution seam; remaining cleanup is about runtime-native normalization gaps, not legacy backend coupling. |

## Does Autotuner Still Block Legacy Cleanup?

Partially, but only in one narrow way.

Autotuner itself no longer directly depends on `Compiler`, `Executor`, `SimValidator`, `HostRunnerGen`, `msprof`, or `perf-report`. The current source scan did not find those references in `tools/autotuner/autotuner_main.cpp`, and the tool now uses `ArtifactCompiler`, `ExecutionSession`, `TaskGraph`, and runtime profile retention helpers instead.

The remaining autotuner-specific blocker is now narrower than the original audit stated:

- The local `--artifact-root` manifest loader still leaves `mixResourceType` as `Unknown`, which can break mix scheduling even though the canonical runtime loader already parses `mix_resource_type`.

That means autotuner no longer blocks `Legacy/Compiler` deletion, and the remaining autotuner-owned cleanup is the artifact-root normalization gap for mix scheduling.

Autotuner did not block `Legacy/HostRunnerGen` cleanup directly, and that unit has now been removed. It also no longer owns a direct `Legacy/Executor` or `Legacy/SimValidator` seam; the runtime execution path now goes through `NativeExecutionRunner`, while `ExecutionSession` remains only the dispatcher/orchestration layer.

## Follow-Up Tasks

- Normalize the autotuner `--artifact-root` loader so it preserves `mix_resource_type` from the manifest instead of hardcoding `Unknown`.
- Rerun the autotuner xvm smoke path after any artifact-loading change and confirm that `--kernel` and `--artifact-root` still produce the same retained profile JSON and best-config summary.
- Treat further work here as runtime-native normalization, not legacy cleanup.
