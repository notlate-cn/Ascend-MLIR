# Runtime Directory Reorganization Design

**Date:** 2026-04-13

## Goal

Reorganize `include/Runtime` and `lib/Runtime` into explicit subdirectories that
match the current runtime architecture.

After this change:

- public headers and implementation files are grouped by responsibility
- `TaskGraph Runtime` code is visually separated from legacy runtime internals
- mix-specific code is isolated from general execution/runtime code
- future cleanup and ownership decisions become easier because file placement
  reflects architectural intent

This is a structural refactor, not a feature change.

## Non-Goals

- Changing runtime behavior
- Changing CLI behavior
- Changing C API behavior
- Renaming core public types such as `ExecutionSession`, `ArtifactCompiler`, or
  `ProfileTrace`
- Removing remaining legacy implementation classes in this phase
- Rewriting includes across the entire repo to non-`Runtime/...` style

## Current Problem

`include/Runtime` and `lib/Runtime` are both flat directories.

That was tolerable when the codebase was smaller, but it now hides important
boundaries:

- new runtime core vs legacy internals
- mix-specific toolchain code vs generic runtime execution
- artifact/session/profile logic vs low-level support helpers

As a result:

- ownership is unclear
- refactors touch too many unrelated files
- new contributors cannot see the architecture from the directory layout
- dead-code cleanup is harder because legacy and current code are interleaved

## Decision

Adopt a mirrored six-way directory split for both headers and implementation:

- `Runtime/Artifact`
- `Runtime/Execution`
- `Runtime/Profile`
- `Runtime/Mix`
- `Runtime/Support`
- `Runtime/Legacy`

The implementation side mirrors this exactly:

- `lib/Runtime/Artifact`
- `lib/Runtime/Execution`
- `lib/Runtime/Profile`
- `lib/Runtime/Mix`
- `lib/Runtime/Support`
- `lib/Runtime/Legacy`

## File Mapping

### Artifact

Headers:

- `ArtifactCompiler.h`
- `RunManifest.h`

Sources:

- `ArtifactCompiler.cpp`
- `RunManifest.cpp`

### Execution

Headers:

- `ExecutionBackend.h`
- `ExecutionSession.h`
- `TaskGraph.h`
- `SimBackend.h`
- `NpuBackend.h`

Sources:

- `ExecutionBackend.cpp`
- `ExecutionSession.cpp`
- `TaskGraph.cpp`
- `SimBackend.cpp`
- `NpuBackend.cpp`

### Profile

Headers:

- `ProfileTrace.h`
- `ProfileUtils.h`

Sources:

- `ProfileTrace.cpp`
- `ProfileUtils.cpp`

### Mix

Headers:

- `AscendCMixCompiler.h`
- `MixAbi.h`
- `MixAbiExtractor.h`
- `MixArtifact.h`
- `MixCommandBuilder.h`
- `MixDirectBackend.h`
- `MixSourceAnalyzer.h`
- `MixStubTemplate.h`

Sources:

- `AscendCMixCompiler.cpp`
- `MixAbi.cpp`
- `MixAbiExtractor.cpp`
- `MixCommandBuilder.cpp`
- `MixDirectBackend.cpp`
- `MixSourceAnalyzer.cpp`
- `MixStubTemplate.cpp`
- `AscendCannPaths.cmake`

### Support

Headers:

- `NpyIO.h`
- `PathUtils.h`
- `TilingPack.h`
- `TilingSchema.h`
- `Types.h`

Sources:

- `NpyIO.cpp`
- `PathUtils.cpp`
- `TilingPack.cpp`
- `TilingSchema.cpp`

### Legacy

Headers:

- `CompatRuntime.h`
- `Compiler.h`
- `Executor.h`
- `HostRunnerGen.h`
- `SimValidator.h`

Sources:

- `CompatRuntime.cpp`
- `Compiler.cpp`
- `Executor.cpp`
- `HostRunnerGen.cpp`
- `SimValidator.cpp`

## Include Strategy

This reorganization should preserve top-level `Runtime/...` include stability in
the short term.

### Stable include contract

Existing call sites should continue to compile with includes such as:

- `#include "Runtime/ExecutionSession.h"`
- `#include "Runtime/ProfileUtils.h"`

### Mechanism

For each moved public header:

1. Move the real header into its new subdirectory
2. Leave a thin forwarding header at the old top-level path
3. The forwarding header includes the new canonical header

Example:

- canonical: `include/Runtime/Execution/ExecutionSession.h`
- forwarding shim: `include/Runtime/ExecutionSession.h`

This gives us:

- clean physical organization now
- low-risk migration for existing includes
- freedom to remove forwarding shims later in a separate cleanup phase

## Build System Strategy

`lib/Runtime/CMakeLists.txt` should be updated so source lists reference the new
subdirectory paths.

No new library targets are introduced in this phase. Everything still builds
into the existing runtime library targets.

That keeps the change structural rather than architectural at the build target
level.

## Why Not Split Targets Too?

That would be a different project.

Changing physical layout is already a meaningful reorganization. Splitting CMake
targets at the same time would:

- expand the review surface
- create avoidable link-order and dependency churn
- make it harder to distinguish structure-only regressions from build-graph
  regressions

This phase should stop at directory and include reorganization.

## Migration Rules

To keep risk controlled, this reorg follows three rules:

1. No behavior changes mixed into the move
2. Keep forwarding headers at old public include paths
3. Finish by making the full runtime verification suite green before any
   further cleanup

## Risks

### Risk 1: partial include migration leaves confusing mixed style

Some files may start including subdirectory paths while others still use the
top-level forwarding headers.

Mitigation:

- allow both during this phase
- do not require global include-style cleanup yet

### Risk 2: CMake/source list drift

Moving files without updating every source list will break builds in ways that
look unrelated.

Mitigation:

- treat `lib/Runtime/CMakeLists.txt` as a first-class part of the change
- verify both focused runtime build and example pipelines on xvm

### Risk 3: forwarding headers accidentally diverge

If forwarding headers are edited independently later, they become a source of
confusion.

Mitigation:

- keep forwarding headers minimal: one include, no logic
- defer any header cleanup/removal to a later dedicated phase

## Acceptance Criteria

This reorganization is complete only when all of the following are true:

- `include/Runtime` and `lib/Runtime` are reorganized into the six agreed
  subdirectories
- old top-level public include paths still compile through forwarding headers
- `lib/Runtime/CMakeLists.txt` reflects the new layout
- `bash test/tools/runtime/run_runtime.sh` passes on xvm
- `bash test/tools/runtime/run_simbackend_examples.sh` passes on xvm
- no runtime behavior change is required to explain the result
