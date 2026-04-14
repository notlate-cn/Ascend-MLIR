# Runtime Frontend Core Design

## Goal

Unify `runtime-session` and the C API around a shared runtime-native frontend core so they no longer maintain separate request-assembly logic or separate run-result interpretation rules.

This is a boundary cleanup, not a feature expansion.

## Scope

In scope:
- shared runtime-native request assembly for the current `runtime-session` and C API use cases
- shared runtime-native run-result model for backend/result/stage/profile/validation reporting
- refactoring `runtime-session` and `lib/CAPI/Runtime/Runtime.cpp` to consume the shared core

Out of scope:
- changing public C API function names or ABI shape
- changing `runtime-session` text output format beyond what is needed to consume the shared result model
- scheduler changes
- profiling schema changes
- autotuner changes

## Current Problem

The codebase is already runtime-native in its core execution model, but the two external entry points still assemble and interpret runtime requests separately:
- `tools/runtime-session/runtime_session_main.cpp`
- `lib/CAPI/Runtime/Runtime.cpp`

This leaves duplicated boundary logic around:
- compile request construction
- artifact-root loading
- single-task run assembly
- tiling / output / expected-output binding rules
- success / error / validation / profile result interpretation

As long as those remain split, the runtime still has entry-point drift even after legacy deletion.

## Approaches

### Approach A: Shared frontend core (recommended)

Introduce a small shared runtime-native frontend layer that owns:
- request assembly helpers for compile and run flows
- a structured run-result summary model

`runtime-session` and the C API both use this layer, but continue to own their own presentation/ABI translation.

Pros:
- clean architectural boundary
- removes drift in one pass
- keeps CLI and C API thin without coupling them to each other

Cons:
- requires touching both entry points in one change

### Approach B: Reuse `RuntimeSessionRequestBuilder` directly from C API

Push the C API onto the existing builder and leave result interpretation mostly separate.

Pros:
- smaller change

Cons:
- builder stays semantically CLI-centered
- run-result drift remains
- likely needs another cleanup pass later

### Approach C: Keep separate assembly but share only result handling

Unify run-result interpretation but leave request-building separate.

Pros:
- least invasive

Cons:
- leaves the larger duplication problem in place
- not a real boundary cleanup

## Recommended Design

Use **Approach A**.

Add a small shared frontend core in the runtime library. The core should be runtime-native, not CLI-specific and not C-ABI-specific.

## Shared Frontend Core

### Request side

The shared core should provide helpers for:
- compile request creation from validated frontend inputs
- artifact-root loading
- single-task run request / graph assembly for the current CLI and C API needs

This should absorb logic that is currently split between:
- `RuntimeSessionRequestBuilder`
- direct request construction in `lib/CAPI/Runtime/Runtime.cpp`

The core should not know about:
- `llvm::cl`
- C ABI buffer-size conventions
- CLI text printing

### Result side

Add a structured frontend-facing run result summary that can represent:
- backend kind
- success / failure
- error stage
- validation status when present
- retained profile artifact path(s)
- retained session summary path when present

`runtime-session` should map this model to its current text output.

The C API should use the same model internally and translate failures/results into its existing ABI shape.

## Boundary ownership

### `runtime-session`

Should own only:
- CLI parsing and option validation
- invoking the shared frontend core
- text output formatting

### C API

Should own only:
- C ABI argument parsing / null-checking / buffer-size handling
- invoking the shared frontend core
- copying result/error text back through the C ABI

### Shared core

Should own:
- runtime-native request assembly
- runtime-native result interpretation

## Compatibility rules

- No public C API ABI break in this change.
- No intentional semantic drift in `runtime-session` error precedence or summary output.
- The change should reduce duplication, not rebrand it.

## Verification

Minimum verification:
- seam search showing duplicated request/result assembly code has been removed from `runtime-session` and C API entry points where intended
- `git diff --check`
- xvm focused runtime verification
- xvm `test_capi_runtime`

If full `run_runtime.sh` is blocked by unrelated outer-tool build noise, use the already-established fallback standard:
- direct runtime-focused xvm verification of affected test binaries and smoke paths
- explicitly note the blocker as external

## Acceptance

The change is complete when:
- `runtime-session` and C API both build their runtime-native requests through the shared frontend core
- run-result interpretation is shared rather than duplicated
- CLI and C API remain behaviorally compatible at their public boundary
- xvm focused verification stays green
