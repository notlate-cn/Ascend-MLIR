# Autotuner And Legacy Cleanup Design

**Goal:** Define a safe migration path for `autotuner` and a second-round cleanup strategy for `lib/Runtime/Legacy` without mixing migration and deletion work.

**Scope:** This spec covers dependency audit, target architecture, migration boundaries, and cleanup sequencing only. It does not implement behavior changes.

## Current State

The project runtime has already been reorganized around the newer library layers:
- `Artifact/`
- `Execution/`
- `Profile/`
- `Mix/`
- `Support/`
- `Legacy/`

`runtime-session`, examples, focused runtime verification, C API flows, and the newer profiling/session machinery now route through the new runtime abstractions. However, `autotuner` and several lower-level execution pieces still depend on `Legacy` classes directly or indirectly.

Current notable dependency edges:
- `tools/autotuner/autotuner_main.cpp`
  - uses `ArtifactCompiler`
  - uses runtime-native execution/profiling flow
  - still conceptually sits near older runtime seams and should be re-audited before more `Legacy` deletion
- `lib/Runtime/Artifact/ArtifactCompiler.cpp`
  - still relies on `Legacy/Compiler`
- `lib/Runtime/Execution/SimBackend.cpp`
  - still relies on `Legacy/Executor` and `Legacy/SimValidator`
- `lib/Runtime/Execution/NpuBackend.cpp`
  - still relies on `Legacy/Executor` and `Legacy/SimValidator`
- `lib/Runtime/Legacy/HostRunnerGen.cpp`
  - still has direct tests and may still matter to specialized workflows
- `lib/CAPI/Runtime/Runtime.cpp`
  - now routes through newer runtime-facing flows, but some lower-level execution behavior still depends on legacy implementations under the hood

## Problem Statement

At this point, deleting more `Legacy` code blindly is unsafe for two reasons:

1. Some files in `Legacy/` are still true implementation dependencies of new runtime modules, not merely compatibility leftovers.
2. `autotuner` has already been moved toward runtime-native execution, but its exact remaining relationship to `Legacy` must be made explicit before further deletion.

This means the next step should not be “delete more old code.” The next step should be a deliberate audit and migration design that classifies what is still required, what should be adapted, and what can be removed later.

## Design Principles

1. Do not mix `autotuner` migration and `Legacy` deletion in one implementation task.
2. Treat `Legacy` as an implementation bucket with mixed status, not as uniformly dead code.
3. Only delete code after a dependency class has been migrated or proven unused.
4. Prefer replacing upward-facing dependencies first (`autotuner`, tools, adapters) before attempting to remove deep execution substrate.
5. Keep each cleanup step independently verifiable with xvm runtime-focused tests.

## Target Classification Model

Each important `Legacy` class or module should be placed into exactly one of three buckets.

### Bucket A: Must Keep For Now
These are still active implementation dependencies of the new runtime stack and cannot be deleted yet.

Initial expected members:
- `Legacy/Compiler`
- `Legacy/Executor`
- `Legacy/SimValidator`
- parts of `Legacy/HostRunnerGen`

### Bucket B: Candidate For Adapter/Boundary Shrink
These still exist for real reasons, but should be moved further down or wrapped more tightly so fewer upper layers depend on them directly.

Initial expected members:
- `Legacy/HostRunnerGen`
- portions of `Legacy/CompatRuntime`
- helper seams used only because migration stopped halfway

### Bucket C: Delete After Migration
These are not removable today, but should become deletable once a specific dependency is removed or refactored.

Expected examples after follow-up work:
- old helper paths only used by `autotuner`
- compatibility-only glue that becomes unreachable once migration completes
- test fixtures that exist solely for removed legacy entry points

## Autotuner-Specific Direction

The `autotuner` should be treated as a runtime-native tool, not a special exception that keeps old architecture alive.

That means the audit should answer:
- Which runtime-facing abstractions `autotuner` already uses correctly
- Whether any remaining includes, helper assumptions, or execution/profiling paths still rely on `Legacy` semantics indirectly
- Which pieces of `Legacy` are only preserved because `autotuner` has not been fully normalized yet

The goal is not to redesign `autotuner` again. The goal is to determine whether `autotuner` still blocks `Legacy` cleanup, and if so, exactly where.

## Proposed Sequencing

### Phase 1: Audit
Produce an explicit dependency map covering:
- `autotuner` direct includes and runtime calls
- all references to `Legacy/Compiler`
- all references to `Legacy/Executor`
- all references to `Legacy/SimValidator`
- all references to `Legacy/HostRunnerGen`
- all references to `Legacy/CompatRuntime`

For each referenced item, record:
- who depends on it
- whether it is an implementation dependency or just compatibility glue
- whether there is already a new-runtime equivalent

### Phase 2: Autotuner Migration Plan
Based on the audit, define only the minimum follow-up work needed so `autotuner` no longer blocks additional cleanup.

This may include:
- narrowing headers
- moving remaining helper logic out of tool-local code
- replacing old adapter assumptions with existing runtime-native abstractions

This phase should end with a concrete answer to:
- “Which `Legacy` pieces remain required even after `autotuner` is fully aligned?”

### Phase 3: Legacy Cleanup Plan
Only after Phase 2 is understood should cleanup proceed.

Cleanup should happen in layers:
1. remove dead tests/helpers
2. remove compatibility-only glue proven unused
3. shrink adapter surfaces
4. only then consider deleting deeper legacy implementation files

## Verification Strategy

This design intentionally separates planning from deletion. Verification for later implementation phases must include, at minimum:
- `bash test/tools/runtime/run_runtime.sh`
- relevant `autotuner` smoke commands on xvm
- any focused tests that cover remaining `Legacy` consumers

Deletion is not considered safe unless the dependency map and these runtime checks agree.

## Deliverables For The Next Plan

The implementation plan derived from this spec should produce:
1. a concrete dependency audit artifact
2. an `autotuner` normalization task list if needed
3. a second-round `Legacy` cleanup candidate list, grouped by risk and prerequisite

## Non-Goals

This spec does not:
- remove any `Legacy` code
- redesign runtime execution internals
- change `autotuner` CLI behavior
- delete `Compiler`, `Executor`, `SimValidator`, or `HostRunnerGen` directly

Those belong to later implementation plans after the audit is complete.
