# Single Runtime CLI Design

**Date:** 2026-04-13

## Goal

Collapse the project onto a single runtime CLI architecture centered on
`runtime-session`.

After this change:

- `runtime-session` is the only general-purpose runtime CLI
- `compiler`, `validator`, and `mix-validator` are removed
- examples and tests stop depending on compatibility CLI wrappers

This is a deliberate architecture cutover, not a compatibility-preserving
migration. There are no active external users, so we should optimize for a
clean end state rather than transitional shells.

## Non-Goals

- Removing `mix-compiler` in this phase
- Migrating `autotuner` in this phase
- Removing low-level runtime implementation pieces such as `Compiler`,
  `Executor`, or `HostRunnerGen` if they are still used internally
- Changing MLIR lowering/codegen behavior in the examples
- Changing the public C API in this phase

## Current State

The runtime architecture is already centered on:

- `ArtifactCompiler`
- `ExecutionSession`
- `SimBackend`
- `NpuBackend`
- `RunManifest`
- `runtime-session`

At the same time, the repo still carries three top-level CLI wrappers:

- `tools/compiler`
- `tools/validator`
- `tools/mix-validator`

Those wrappers were useful during migration, but they now keep the project in a
split-brain state:

- users can still discover multiple overlapping CLIs
- tests and examples still encode compatibility behavior
- old CLI-specific helpers and output conventions remain artificially alive

## Decision

Adopt a single-center CLI architecture:

1. Keep `runtime-session`
2. Remove `compiler`
3. Remove `validator`
4. Remove `mix-validator`
5. Update examples, test harnesses, and docs to call `runtime-session` directly

`mix-compiler` remains temporarily because it still provides a dedicated mix
compile flow that has not yet been folded into `runtime-session`.

## CLI Boundary After Cutover

### Keep

- `runtime-session`
- `mix-compiler`
- `autotuner`

### Remove

- `compiler`
- `validator`
- `mix-validator`

### Rationale

`runtime-session` is already the only CLI that matches the new runtime object
model. It supports:

- artifact-oriented execution
- manifest-driven execution
- DAG execution
- backend selection
- runtime profiling output

The removed CLIs no longer represent distinct architecture. They only rephrase
pieces of the same runtime behavior.

## Example Strategy

The 6 example pipelines must continue to pass real CPU simulation on xvm, but
they should no longer do so through deleted wrappers.

### Vec examples

The 5 vec examples should:

- keep their existing MLIR/codegen stages
- stop calling `compiler`
- stop calling `validator`
- instead:
  - compile via `runtime-session` compile path or the normalized artifact flow
  - execute via `runtime-session --run-manifest --run`

### Mix example

`examples/matmul-add-leakyrelu` already uses a runtime-centered execution path.
This example should be normalized further only as needed to remove any remaining
dependency on deleted CLI wrappers.

## Testing Strategy

### Required verification

The architectural cutover is only acceptable if these pass:

1. `bash test/tools/runtime/run_runtime.sh`
2. `bash test/tools/runtime/run_simbackend_examples.sh`

These are the gating checks because they prove:

- focused runtime tests still pass
- C API smoke still passes
- real `SimBackend` CPU simulation still passes on xvm for all 6 examples

### Test updates

Tests that currently exercise `compiler`, `validator`, or `mix-validator`
directly must be either:

- removed, if they only verify deleted CLI behavior
- rewritten around `runtime-session`, if they still verify necessary runtime
  behavior

## Cleanup Scope

This phase removes:

- deleted CLI source files and their CMake targets
- example script references to deleted CLIs
- test references that only exist to verify deleted CLIs
- obsolete compatibility messages and CLI-only helpers where they become dead

This phase does not automatically remove every old low-level runtime class.
Those remain subject to separate dead-code auditing after the CLI cutover.

## Risks

### Risk 1: examples depend on deleted CLI side effects

Some examples may still rely on CLI-specific output conventions such as
particular artifact summaries or generated helper files.

Mitigation:

- migrate examples one by one to explicit `runtime-session` behavior
- use xvm `SimBackend` execution as the acceptance gate

### Risk 2: tests overfit deleted compatibility behavior

Some focused runtime tests currently validate wrapper semantics instead of core
runtime behavior.

Mitigation:

- delete wrapper-only assertions
- preserve only tests that validate runtime artifacts, manifests, sessions, and
  `SimBackend`/`NpuBackend` behavior

### Risk 3: `mix-compiler` and `autotuner` still anchor old implementation code

Even after removing the three compatibility CLIs, old implementation classes may
still remain because these specialized tools still depend on them.

Mitigation:

- explicitly defer that cleanup to the next phase
- do not block the CLI architecture cleanup on full internal dead-code removal

## Acceptance Criteria

This design is complete only when all of the following are true:

- `runtime-session` is the only general-purpose runtime CLI in `tools/`
- `compiler`, `validator`, and `mix-validator` are removed from source and build
- the 6 existing example pipelines still pass CPU simulation on xvm
- focused runtime verification still passes on xvm
- remaining old runtime code is limited to internal implementation dependencies,
  not competing top-level CLIs
