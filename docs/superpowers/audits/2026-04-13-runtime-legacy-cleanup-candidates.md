# Runtime Legacy Cleanup Candidates

## Preconditions

- The dependency inventory in `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md` is the source of truth for legacy references.
- The autotuner normalization audit in `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md` is the source of truth for autotuner-specific blockers.
- Cleanup candidates are classified from current source evidence only; this document does not propose code changes or deletions by itself.
- Any deletion candidate still needs the relevant runtime verification path on xvm before removal is attempted.

## Safe Now

- `Legacy/Executor`
  - Deleted after the runtime-native execution runner cutover.
  - Remaining execution cleanup work should focus on retained compatibility files, not executor seams.
- `Legacy/SimValidator`
  - No longer has any remaining in-repo consumers after the runtime-native output comparator cutover.
  - The retained legacy validator file and shim can be removed without touching runtime backend behavior.

## Needs Migration First

- `Legacy/Compiler`
  - No longer blocks `ArtifactCompiler`; vec/cube source builds now route through `VecCubeArtifactBackend`.
  - The remaining direct runtime-owned surface has been reduced to an explicit legacy mix compile test in `test/tools/runtime/test_runtime.cpp`, plus the retained implementation file itself.
  - Treat it as a retained compatibility unit until a self-contained runtime-native mix compile fixture exists.
- `Legacy/CompatRuntime`
  - Still used by `lib/CAPI/Runtime/Runtime.cpp` and `test/tools/runtime/test_taskgraph_runtime.cpp`.
  - Must be migrated away from the compatibility adapter boundary before any cleanup attempt.

## Do Not Touch Yet

- none currently

## Verification Requirements

- Any change affecting the remaining `Legacy/Compiler` surface must rerun the autotuner xvm smoke path and confirm vec/cube `VecCubeArtifactBackend`-backed source builds still produce the same retained profile JSON and best-config summary.
- Any change affecting the autotuner `--artifact-root` path must verify mix artifact-root loads preserve `mix_resource_type` and do not regress task scheduling.
- Any change affecting the runtime-native execution substrate must rerun `bash test/tools/runtime/run_runtime.sh` on xvm.
- Any change affecting the runtime-native comparator or any deletion of the legacy validator files must rerun `bash test/tools/runtime/run_runtime.sh` on xvm.
- Any change affecting `Legacy/CompatRuntime` must rerun the runtime C API and taskgraph runtime coverage that depends on the compat adapter boundary.
- No cleanup step is complete until the relevant audit entry has a matching source reference and a verified follow-up path.

## Next-Step Guidance

- The next cleanup slice should start from the retained compatibility files that still sit outside the runtime-native stack, not from runtime backend seams that have already been cut over.
- `Legacy/Executor` is no longer an active cleanup target because the file and shim have been deleted.
- `Legacy/SimValidator` can be removed directly because there are no surviving non-backend consumers left.
- `Legacy/HostRunnerGen` is no longer an active cleanup target because the file, shim, and dedicated tests have been deleted.
- `Legacy/Compiler` should now be treated as a small retained-surface decision, not as a main runtime migration blocker.
- Deeper physical deletion should wait until the remaining compatibility boundary and retained compiler test surface are re-audited against current consumers.
