# xvm Default Verification Cleanup Design

## Goal

Stabilize the default xvm runtime verification entry points so runtime-focused verification does not fail or stall due to unrelated outer-tool build noise, host cache pollution, or missing simulator library path setup.

This is verification-environment cleanup, not runtime feature work.

## Scope

In scope:
- tighten the default xvm verification scripts under `test/tools/runtime/`
- reduce unnecessary outer-target build dependencies in default runtime verification
- centralize required runtime test environment setup for xvm
- preserve existing runtime-focused coverage and smoke behavior

Out of scope:
- scheduler changes
- profiling schema changes
- NPU real-device work
- broad CI redesign
- full project build policy changes outside runtime-focused verification

## Current Problem

The runtime stack is now runtime-native enough that focused verification should be able to run independently. But the current default entry points still have environment and build noise sources:
- `run_runtime.sh` builds `afir-opt` and `afir-translate` even when the runtime-focused checks do not need a fresh rebuild of both tools every time
- xvm verification can still be disrupted by polluted `build/` state or unrelated outer-target recompiles
- direct test invocations need explicit simulator/runtime `LD_LIBRARY_PATH` wiring that is not normalized into one reusable helper
- some verification commands are duplicated between scripts instead of sharing a runtime-focused baseline

That makes verification slower and less deterministic than it should be.

## Approaches

### Approach A: Introduce a shared runtime verification environment/helper layer (recommended)

Add a small shared shell helper for xvm runtime verification, then refactor `run_runtime.sh` and `run_simbackend_smoke.sh` / `run_simbackend_examples.sh` to use it.

The helper should centralize:
- Ascend/CANN env resolution
- LLVM env resolution
- xvm runtime-library path assembly
- stale `build/` detection / rebuild policy
- focused runtime build target list

Pros:
- keeps verification logic consistent
- avoids more ad hoc shell drift
- makes future default verification changes cheaper

Cons:
- requires touching multiple scripts at once

### Approach B: Patch `run_runtime.sh` only

Just reduce target list and inline more checks into the main script.

Pros:
- smaller change

Cons:
- preserves duplication and script drift
- likely needs another cleanup pass later

### Approach C: Move everything into one monolithic script

Collapse smoke/examples/focused verification into one large shell entry point.

Pros:
- superficially simple

Cons:
- too coarse; hurts reuse and makes maintenance worse

## Recommended Design

Use **Approach A**.

## Design

### 1. Add a shared runtime verification shell helper

Introduce a helper script under `test/tools/runtime/` that exports reusable functions for:
- sourcing Ascend and LLVM env helpers
- assembling runtime test `LD_LIBRARY_PATH`
- checking and recreating stale `build/` state
- configuring `cmake -S . -B build ...`
- building the focused runtime target set

### 2. Define a runtime-focused default target set

Default runtime verification should only rebuild what the runtime-focused checks actually consume.

The focused default set should include:
- `AscendCRuntime`
- `AFIRRuntimeCAPI`
- `runtime-session`
- `mix-compiler`
- any additional runtime-only support binary proven necessary by the current scripts

It should not rebuild unrelated outer-tool targets by default unless a specific runtime check requires them.

### 3. Normalize direct test execution environment

The shared helper should expose one canonical runtime test library path setup that covers:
- runtime shared libs
- LLVM libs
- Ascend lib64
- simulator libs
- device stub libs where needed

This should remove hand-assembled path drift between focused test invocations.

### 4. Preserve current verification coverage

The cleanup must keep these checks intact:
- runtime-session CLI checks
- runtime-session planning/error-path checks
- runtime-session positive vec and DAG simulation checks
- NPU mock / negative checks that are already part of focused runtime verification
- `test_taskgraph_runtime`
- `test_capi_runtime`
- `test_runtime`
- repeated mix simulation baseline
- SimBackend smoke baseline

### 5. Acceptance standard

The cleanup is complete when:
- `run_runtime.sh` uses the shared helper and no longer depends on unrelated outer-target rebuilds
- xvm focused runtime verification remains green
- the direct runtime test execution environment is normalized through one shell helper
- verification behavior is more deterministic without changing runtime feature semantics

## Verification

Minimum verification:
- `bash -n` on updated shell scripts
- xvm runtime-focused build using the reduced target list
- xvm `bash test/tools/runtime/run_runtime.sh`
- xvm `bash test/tools/runtime/run_simbackend_smoke.sh`

If a tool is intentionally removed from the default build target set, verification must prove it is still available where needed or no longer required.
