# Runtime Legacy Cleanup Candidates

## Preconditions

- The dependency inventory in `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md` is the source of truth for legacy references.
- The autotuner normalization audit in `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md` is the source of truth for autotuner-specific blockers.
- Cleanup candidates are classified from current source evidence only; this document does not propose code changes or deletions by itself.
- Any deletion candidate still needs the relevant runtime verification path on xvm before removal is attempted.

## Safe Now

- `Legacy/Executor`
  - No longer directly required by `lib/Runtime/Execution/SimBackend.cpp` or `lib/Runtime/Execution/NpuBackend.cpp`.
  - The runtime execution seam now sits behind `DefaultExecutionRunner`, so the next cleanup step should start from the adapter boundary rather than the backends.
- `Legacy/SimValidator`
  - No longer directly required by `lib/Runtime/Execution/SimBackend.cpp` or `lib/Runtime/Execution/NpuBackend.cpp`.
  - The compare-only validation seam has already moved to the runtime-native output comparator.

## Needs Migration First

- `Legacy/Compiler`
  - No longer blocks `ArtifactCompiler`; vec/cube source builds now route through `VecCubeArtifactBackend`.
  - Remaining cleanup status must now be driven by non-`ArtifactCompiler` consumers and whether they justify one more extraction pass or a much smaller retained legacy surface.
- `Legacy/CompatRuntime`
  - Still used by `lib/CAPI/Runtime/Runtime.cpp` and `test/tools/runtime/test_taskgraph_runtime.cpp`.
  - Must be migrated away from the compatibility adapter boundary before any cleanup attempt.

## Do Not Touch Yet

- `Legacy/HostRunnerGen`
  - Still has direct test coverage in `test/tools/runtime/test_runtime.cpp` and `test/tools/runner/test_runner_gen.cpp`.
  - Still has specialized workflow references and no runtime-native execution path depends on it yet.

## Verification Requirements

- Any change affecting the remaining `Legacy/Compiler` surface must rerun the autotuner xvm smoke path and confirm vec/cube `VecCubeArtifactBackend`-backed source builds still produce the same retained profile JSON and best-config summary.
- Any change affecting the autotuner `--artifact-root` path must verify mix artifact-root loads preserve `mix_resource_type` and do not regress task scheduling.
- Any change affecting the execution adapter boundary around `DefaultExecutionRunner` must rerun `bash test/tools/runtime/run_runtime.sh` on xvm.
- Any change affecting the runtime-native comparator or any remaining `Legacy/SimValidator` consumer must rerun `bash test/tools/runtime/run_runtime.sh` on xvm.
- Any change affecting `Legacy/CompatRuntime` must rerun the runtime C API and taskgraph runtime coverage that depends on the compat adapter boundary.
- Any proposed deletion of `Legacy/HostRunnerGen` must update or remove `test/tools/runtime/test_runtime.cpp` and `test/tools/runner/test_runner_gen.cpp` intentionally.
- No cleanup step is complete until the relevant audit entry has a matching source reference and a verified follow-up path.

## Next-Step Guidance

- The next cleanup slice should start from the remaining compiler-facing surface, not from runtime backend seams that have already been pushed behind adapters.
- `Legacy/Executor` cleanup should now proceed from the `DefaultExecutionRunner` boundary rather than from `SimBackend` or `NpuBackend`.
- `Legacy/SimValidator` cleanup should now proceed from any surviving non-backend consumers, because the direct runtime backend seam is already gone.
- Deeper physical deletion should wait until the remaining compiler surface and compatibility boundary are re-audited against current consumers.
