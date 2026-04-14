# Runtime Legacy Cleanup Candidates

> Historical note: this document now serves primarily as a record of the
> legacy-cleanup sequence. The major legacy implementation units have already
> been removed, so active planning should now be framed in runtime-native
> terms rather than as additional legacy deletion work.

## Preconditions

- The dependency inventory in `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md` is the source of truth for legacy references.
- The autotuner normalization audit in `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md` is the source of truth for autotuner-specific blockers.
- Cleanup candidates are classified from current source evidence only; this document does not propose code changes or deletions by itself.
- Any deletion candidate still needs the relevant runtime verification path on xvm before removal is attempted.

## Completed Cleanup

- `Legacy/Executor`
  - Deleted after the runtime-native execution runner cutover.
  - Remaining execution cleanup work should focus on retained compatibility files, not executor seams.
- `Legacy/SimValidator`
  - No longer has any remaining in-repo consumers after the runtime-native output comparator cutover.
  - The retained legacy validator file and shim can be removed without touching runtime backend behavior.
- `Legacy/Compiler`
  - Deleted after the runtime-native vec/cube compile cutover and removal of the final retained legacy mix compile test.
  - No longer participates in the runtime library build, public headers, or focused runtime verification.

## Migration-Blocked Candidates

- none currently

## Deferred / Not Applicable

- none currently

## Verification Requirements

- Any change affecting the remaining `Legacy/Compiler` surface must rerun the autotuner xvm smoke path and confirm vec/cube `VecCubeArtifactBackend`-backed source builds still produce the same retained profile JSON and best-config summary.
- Any change affecting the autotuner `--artifact-root` path must verify mix artifact-root loads preserve `mix_resource_type` and do not regress task scheduling.
- Any change affecting the runtime-native execution substrate must rerun `bash test/tools/runtime/run_runtime.sh` on xvm.
- Any change affecting the runtime-native comparator or any deletion of the legacy validator files must rerun `bash test/tools/runtime/run_runtime.sh` on xvm.
- No cleanup step is complete until the relevant audit entry has a matching source reference and a verified follow-up path.

## Post-Legacy Guidance

- Legacy deletion is no longer the main runtime workstream.
- Follow-up work should move to runtime-native consolidation:
  - `runtime-session` / C API boundary cleanup
  - execution/profile contract tightening
  - task-graph / scheduler evolution
  - focused xvm verification maintenance
