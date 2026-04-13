# Runtime Legacy Cleanup Candidates

## Preconditions

- The dependency inventory in `docs/superpowers/audits/2026-04-13-runtime-legacy-dependency-audit.md` is the source of truth for legacy references.
- The autotuner normalization audit in `docs/superpowers/audits/2026-04-13-autotuner-runtime-normalization-audit.md` is the source of truth for autotuner-specific blockers.
- Cleanup candidates are classified from current source evidence only; this document does not propose code changes or deletions by itself.
- Any deletion candidate still needs the relevant runtime verification path on xvm before removal is attempted.

## Safe Now

- none

## Needs Migration First

- `Legacy/Compiler`
  - Required by `include/Runtime/Artifact/ArtifactCompiler.h` and `lib/Runtime/Artifact/ArtifactCompiler.cpp`.
  - Still blocks autotuner indirectly for vec/cube source builds through `ArtifactCompiler`.
  - Also remains on the cleanup-critical path until autotuner's local `--artifact-root` loader is normalized against the canonical runtime artifact loader.
- `Legacy/Executor`
  - Required by `lib/Runtime/Execution/SimBackend.cpp` and `lib/Runtime/Execution/NpuBackend.cpp`.
  - The direct legacy execution edges are in the backends; autotuner inherits them only transitively through runtime execution.
- `Legacy/SimValidator`
  - Required by `lib/Runtime/Execution/SimBackend.cpp` and `lib/Runtime/Execution/NpuBackend.cpp`.
  - The direct validation seam is in the backends; autotuner inherits it only transitively through runtime execution.
- `Legacy/CompatRuntime`
  - Still used by `lib/CAPI/Runtime/Runtime.cpp` and `test/tools/runtime/test_taskgraph_runtime.cpp`.
  - Must be migrated away from the compatibility adapter boundary before any cleanup attempt.

## Do Not Touch Yet

- `Legacy/HostRunnerGen`
  - Still has direct test coverage in `test/tools/runtime/test_runtime.cpp` and `test/tools/runner/test_runner_gen.cpp`.
  - Still has specialized workflow references and no runtime-native execution path depends on it yet.

## Verification Requirements

- Any change affecting `Legacy/Compiler` must rerun the autotuner xvm smoke path and confirm vec/cube `ArtifactCompiler`-backed source builds still produce the same retained profile JSON and best-config summary.
- Any change affecting the autotuner `--artifact-root` path must verify mix artifact-root loads preserve `mix_resource_type` and do not regress task scheduling.
- Any change affecting `Legacy/Executor` or `Legacy/SimValidator` must rerun `bash test/tools/runtime/run_runtime.sh` on xvm.
- Any change affecting `Legacy/CompatRuntime` must rerun the runtime C API and taskgraph runtime coverage that depends on the compat adapter boundary.
- Any proposed deletion of `Legacy/HostRunnerGen` must update or remove `test/tools/runtime/test_runtime.cpp` and `test/tools/runner/test_runner_gen.cpp` intentionally.
- No cleanup step is complete until the relevant audit entry has a matching source reference and a verified follow-up path.
