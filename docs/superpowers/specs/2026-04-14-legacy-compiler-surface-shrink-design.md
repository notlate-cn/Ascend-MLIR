# Legacy Compiler Surface Shrink Design

> Date: 2026-04-14
> Status: Proposed

## Goal

Shrink `Legacy/Compiler` to only the minimum remaining non-runtime consumers by removing avoidable adapter and test-surface dependence now that `ArtifactCompiler` has already moved to `VecCubeArtifactBackend`.

## Non-Goals

- No broad deletion of `Legacy/Compiler` in this slice.
- No `SimBackend` / `NpuBackend` seam work.
- No C API redesign.
- No test framework rewrite.
- No `mix` compile-path changes.

## Current Problem

The major `ArtifactCompiler -> Legacy/Compiler` seam is already gone, but [Legacy/Compiler](/Volumes/GM9/code/Codex-Ascend-MLIR/include/Runtime/Legacy/Compiler.h) still survives as a public-facing implementation unit because of a much smaller remaining surface:

- C API compile entry comments and expectations still describe a `Compiler`-centric world.
- Focused runtime tests still instantiate `Compiler` directly for legacy mix fixture coverage.
- Forwarding headers keep `Runtime/Compiler.h` visible as if it were still a first-class runtime abstraction.

This keeps `Legacy/Compiler` looking larger and more central than it really is, and it blocks the next cleanup decision: whether it should remain as a narrow retained legacy implementation or be reduced further behind smaller helpers.

## Recommended Approach

Treat this slice as a surface-shrink and reclassification pass, not as a full compiler rewrite.

The implementation should:

- remove or reduce direct test dependence on `Compiler` where equivalent runtime-native paths already exist
- tighten C API and public-header positioning so `Legacy/Compiler` is no longer presented as a general runtime entry point
- leave any truly unavoidable remaining consumers explicit and isolated

The expected outcome is not “delete `Legacy/Compiler` now”, but “make the residual surface honest, small, and easy to audit”.

## Remaining Consumer Classes

Based on current repository state, the remaining consumer classes are:

### 1. C API compile entry points

[lib/CAPI/Runtime/Runtime.cpp](/Volumes/GM9/code/Codex-Ascend-MLIR/lib/CAPI/Runtime/Runtime.cpp) no longer instantiates `Compiler`; it already routes through:

- `CompatCompileOptions`
- `buildCompatCompileRequest(...)`
- `ArtifactCompiler`

That means the real remaining task here is not implementation migration. It is surface cleanup:

- ensure comments and naming do not imply the C API still wraps a `Compiler` object internally
- avoid using `Runtime/Compiler.h` as a conceptual dependency in public-facing documentation or adapter wording

### 2. Focused runtime tests

[test_runtime.cpp](/Volumes/GM9/code/Codex-Ascend-MLIR/test/tools/runtime/test_runtime.cpp) still directly instantiates `Compiler` for a legacy mix compile fixture.

This is the highest-value shrink target in this slice because it is:

- local
- low-risk
- runtime-owned

If an equivalent runtime-native compile path exists for the same behavior check, the test should move to that path instead of continuing to pin `Compiler` as a public test dependency.

### 3. Forwarding public header surface

[Runtime/Compiler.h](/Volumes/GM9/code/Codex-Ascend-MLIR/include/Runtime/Compiler.h) currently remains as a forwarding shim to the legacy header.

This should stay for now, but the cleanup slice should treat it as:

- compatibility surface
- not recommended runtime API

That means this slice may update comments, documentation, and references, but should not remove the shim unless every remaining consumer is already migrated.

## Proposed Changes

### A. Reduce direct runtime test dependence

Replace direct `Compiler` use in runtime-owned tests where practical with:

- `ArtifactCompiler`
- `VecCubeArtifactBackend`
- or existing runtime-native mix compile paths, depending on what the test is actually proving

The test intent must remain explicit:

- if the test is proving “mix artifact compilation works”, it should use the runtime-native compile path
- if the test is proving something uniquely legacy, it should be renamed and isolated as an explicit legacy test

### B. Reposition `Legacy/Compiler` as compatibility-only surface

Update comments and nearby documentation so they match current architecture:

- C API compile flow is runtime-native
- `Legacy/Compiler` is retained implementation, not the preferred runtime abstraction

This is important because architecture drift can survive in comments long after code migration is done.

### C. Re-audit the post-shrink residual set

After the shrink:

- list the remaining direct `Legacy/Compiler` consumers
- classify whether each is:
  - required legacy compatibility
  - removable test residue
  - future extraction target

That re-audit result will decide whether a later slice should:

- keep a very small retained `Legacy/Compiler`
- or extract the last remaining useful helpers and delete more of it

## Alternatives Considered

### 1. Delete `Legacy/Compiler` now

Rejected.

Too aggressive for the current state. The remaining surface is small, but not yet sufficiently isolated and reclassified to make deletion low-risk.

### 2. Leave `Legacy/Compiler` alone until much later

Rejected.

Now that the major seam is already gone, the remaining surface is small enough that postponing this would just keep stale architecture signals alive.

## Testing And Verification

The implementation plan should require:

1. focused runtime verification on xvm:
   - `bash test/tools/runtime/run_runtime.sh`
2. explicit verification of remaining `Legacy/Compiler` include/instantiation sites after the shrink:
   - repository search for `Runtime/Compiler.h`
   - repository search for `Compiler compiler(`
3. if a runtime-owned test stops using `Compiler`, ensure the replacement test still proves the same compile-path behavior

## Acceptance Criteria

This design is complete when:

- direct runtime-owned test dependence on `Compiler` is reduced or explicitly isolated
- C API/runtime comments no longer imply an internal `Compiler`-centric architecture
- the remaining `Legacy/Compiler` consumer set is smaller and easier to classify
- xvm focused runtime verification remains green

## Scope Guardrails

This slice must not expand into:

- `Legacy/Executor` deletion
- `SimBackend` / `NpuBackend` execution changes
- `HostRunnerGen` cleanup
- `CompatRuntime` redesign
- broad legacy file deletion without an updated residual-consumer audit
