# ArtifactCompiler Vec/Cube Backend Design

> Date: 2026-04-13
> Status: Proposed

## Goal

Remove the `ArtifactCompiler -> Legacy/Compiler` dependency for `vec` and `cube` source builds by introducing a runtime-native vec/cube compile backend. Preserve the existing `KernelArtifact` / manifest contract and keep `mix` compilation on the current `MixDirectBackend` path.

## Non-Goals

- No change to `mix` compilation behavior.
- No new runner compatibility outputs.
- No broad compile-framework rewrite across all kernel kinds.
- No physical deletion of `Legacy/Compiler` in this slice.

## Current Problem

Today [ArtifactCompiler.cpp](/Volumes/GM9/code/Codex-Ascend-MLIR/lib/Runtime/Artifact/ArtifactCompiler.cpp) still directly uses [Legacy/Compiler.h](/Volumes/GM9/code/Codex-Ascend-MLIR/include/Runtime/Legacy/Compiler.h) for `vec` and `cube` builds. This blocks further `Legacy` cleanup even though the higher-level runtime stack (`runtime-session`, `autotuner`, C API, examples) is already centered on `KernelArtifact`, `TaskGraph`, `ExecutionSession`, and backend-driven execution.

That direct dependency keeps `ArtifactCompiler` tied to the old compile abstraction and prevents `Legacy/Compiler` from shrinking to only truly unported consumers.

## Recommended Approach

Introduce a dedicated runtime-native vec/cube compile backend and make [ArtifactCompiler](/Volumes/GM9/code/Codex-Ascend-MLIR/include/Runtime/Artifact/ArtifactCompiler.h) a dispatcher:

- `KernelKind::Vec` / `KernelKind::Cube` -> new vec/cube backend
- `KernelKind::Mix` -> existing `MixDirectBackend`

This makes the dependency break complete from the perspective of `ArtifactCompiler` without forcing a larger compile-framework redesign in the same change.

## Alternatives Considered

### 1. Inline vec/cube compilation directly into `ArtifactCompiler`

Rejected.

It would remove the legacy dependency, but it would also turn `ArtifactCompiler.cpp` into a growing implementation bucket with compile orchestration, command construction, artifact normalization, and manifest writing all in one place.

### 2. Full compile backend framework rewrite across vec/cube/mix

Rejected for this slice.

Architecturally attractive, but too broad for the immediate goal. It would expand verification surface area across the entire compile stack and slow down the `Legacy/Compiler` decoupling work.

## Architecture

### New Component

Add a new backend in the `Artifact` submodule, tentatively:

- `include/Runtime/Artifact/VecCubeArtifactBackend.h`
- `lib/Runtime/Artifact/VecCubeArtifactBackend.cpp`

Its responsibilities:

- compile AscendC `vec` / `cube` kernel source into the expected device binary output
- choose or derive the correct target arch defaults for `vec` vs `cube`
- return a normalized `KernelArtifact`
- write the canonical `out/manifest.txt`

It should not:

- know anything about `TaskGraph` or execution
- produce host runners
- implement `mix` behavior
- reintroduce legacy wrapper semantics

### ArtifactCompiler After The Change

[ArtifactCompiler.cpp](/Volumes/GM9/code/Codex-Ascend-MLIR/lib/Runtime/Artifact/ArtifactCompiler.cpp) becomes a thin dispatcher:

- validate `ArtifactCompileRequest`
- resolve SoC version once
- branch by `KernelKind`
- delegate to the appropriate backend
- return `KernelArtifact`

Crucially, `ArtifactCompiler` should no longer include or instantiate `Legacy/Compiler`.

## Output Contract

The vec/cube backend must preserve the current artifact contract already consumed by the runtime stack:

- `KernelArtifact.kernelName`
- `KernelArtifact.kernelKind`
- `KernelArtifact.socVersion`
- `KernelArtifact.artifactRoot`
- `KernelArtifact.deviceBinaryPath`
- `KernelArtifact.manifestPath`
- `KernelArtifact.mixResourceType = Unknown` for vec/cube

The emitted manifest remains:

- `out/manifest.txt`
- includes `kernel_name`
- includes `soc_version`
- includes `kernel_kind`
- includes `device_binary_path`
- includes `manifest_path`

No runner-related outputs are retained in this design.

## Compatibility Expectations

The following consumers should continue working without interface changes:

- [runtime-session](/Volumes/GM9/code/Codex-Ascend-MLIR/tools/runtime-session/runtime_session_main.cpp)
- [autotuner](/Volumes/GM9/code/Codex-Ascend-MLIR/tools/autotuner/autotuner_main.cpp)
- C API runtime compile entry points
- vec example pipelines
- mix pipelines (unchanged path)

The compatibility guarantee is behavioral, not implementation-based: callers still receive the same artifact shape and manifest semantics, but the vec/cube compilation implementation is no longer powered by `Legacy/Compiler`.

## Legacy Boundary After This Change

After this slice lands:

- `Legacy/Compiler` should no longer be a dependency of `ArtifactCompiler`
- `Legacy/Compiler` may still remain in the tree for other consumers
- the cleanup candidate audit can then reclassify `ArtifactCompiler -> Legacy/Compiler` as removed

This is the intended “complete” cut for this dependency seam.

## Testing And Verification

The implementation plan should require:

1. focused runtime verification on xvm:
   - `bash test/tools/runtime/run_runtime.sh`
2. autotuner vec smoke on xvm
3. at least one vec compile path and one cube compile path covered by focused runtime tests if cube coverage is practical in the existing test corpus
4. explicit verification that `ArtifactCompiler` no longer includes or links through `Legacy/Compiler` for vec/cube compilation

## Acceptance Criteria

This design is complete when all of the following are true:

- `ArtifactCompiler` no longer includes or instantiates `Legacy/Compiler`
- vec/cube builds use the new runtime-native backend
- `mix` compilation behavior is unchanged
- emitted vec/cube artifacts still satisfy current runtime consumers
- xvm runtime verification passes
- xvm autotuner vec smoke passes

## Scope Guardrails

This slice must not be expanded to include:

- `SimBackend` / `NpuBackend` legacy seam changes
- runner generation cleanup
- `mix` backend rewrites
- new CLI surface changes
- broad deletion of `Legacy/Compiler`
