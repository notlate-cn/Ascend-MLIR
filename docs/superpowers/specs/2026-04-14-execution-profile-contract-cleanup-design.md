# Execution/Profile Contract Cleanup Design

## Goal

Tighten the runtime-native execution/profile contract after the shared frontend core cutover so execution results, validation state, and retained profile outputs are represented consistently across the runtime library, `runtime-session`, and the C API.

This is contract cleanup, not feature expansion.

## Scope

In scope:
- tighten the shared execution/profile result contract
- remove remaining duplication or ambiguity in success/error/validation/profile interpretation
- keep `runtime-session` and the C API thin and consistent on top of that contract

Out of scope:
- scheduler changes
- profiling schema v2
- new CLI flags
- new C API ABI
- NPU real-device feature work

## Current Problem

A shared frontend core now exists, but the execution/profile boundary is still only partially consolidated.

The codebase already shares:
- compile request assembly
- single-task run preparation
- frontend success/error summary interpretation

But there are still areas where the contract can be tightened further:
- profile retention and retained summary handling still surface through ad hoc strings and optional output paths
- validation success/failure is represented indirectly instead of as a first-class runtime-facing contract
- `runtime-session` and the C API still each infer some details from raw execution outcomes instead of consuming a fully normalized execution/profile result

That makes the system functionally correct but still looser than it should be after the boundary refactor.

## Approaches

### Approach A: Tighten the shared result model (recommended)

Strengthen the runtime-native execution/profile result model so it fully represents:
- success/failure
- backend kind
- validation status
- error stage/message
- retained task profile paths
- retained session summary path

Then make entry points consume that model directly.

Pros:
- keeps the cleanup at the right abstraction level
- reduces drift without changing public interfaces
- makes future scheduler/profile work easier

Cons:
- requires touching both runtime library and entry-point glue

### Approach B: Keep model thin, improve helper coverage only

Leave the current model mostly intact and just add more helpers/utilities.

Pros:
- smaller change

Cons:
- leaves ambiguity in the core contract
- likely causes another cleanup pass later

### Approach C: Push more logic into entry points

Avoid changing the core model and keep fixing CLI/C API wrappers individually.

Pros:
- locally simple

Cons:
- the opposite of the desired direction
- preserves drift

## Recommended Design

Use **Approach A**.

The cleanup should make the runtime-native result contract the single source of truth for execution/profile outcomes.

## Design

### 1. Strengthen the shared result contract

The shared execution/profile summary should represent, explicitly and directly:
- backend kind
- success/failure
- validation status (`not_run`, `passed`, `failed`)
- error stage
- user-facing error message
- retained task profile artifact paths
- retained session summary path

This should eliminate any remaining need for entry points to reconstruct those details from lower-level pieces.

### 2. Keep retention/path handling inside the contract layer

Retained profile artifacts and session summary paths should be treated as part of the normalized result contract, not as side channels hanging off individual call sites.

That means the contract layer should be the place that decides:
- whether retained summary exists
- which profile artifact paths are relevant to surface
- how validation status and profile outputs relate on success/failure paths

### 3. Keep entry points thin

`runtime-session` should only:
- parse args
- invoke runtime-native preparation/execution
- print the normalized result contract

The C API should only:
- translate ABI inputs
- invoke runtime-native preparation/execution
- copy normalized error/result text back to the caller

### 4. Preserve public behavior

This cleanup must not silently change:
- CLI validation/error precedence
- C API ABI behavior
- retained profile JSON schema
- current profiling output locations

## Verification

Minimum verification:
- focused runtime tests for the strengthened result contract
- xvm runtime-only rebuild of `AscendCRuntime`, `AFIRRuntimeCAPI`, and `runtime-session`
- xvm focused runtime/C API tests
- xvm `run_simbackend_smoke.sh`

If full `run_runtime.sh` is blocked by unrelated outer-tool noise, use the already-established direct runtime-focused xvm subset and record the blocker explicitly.

## Acceptance

The cleanup is complete when:
- execution/profile outcomes are represented by one normalized contract
- `runtime-session` and C API no longer infer contract details separately
- public behavior remains stable
- focused xvm verification stays green
