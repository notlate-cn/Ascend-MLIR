# Runtime Profile Retention Design

Date: 2026-04-13

## Goal

Stabilize retained simulator profiling artifacts under `/tmp/ascendc-runtime-profiles`
so runtime verification and autotuner runs remain inspectable without allowing the
directory to grow without bound.

This stage only covers retained profile cleanup policy. It does not change:

- scheduler behavior
- profile schema contents
- autotuner scoring logic
- NPU execution paths

## Current State

`runtime-session` now retains simulator profile artifacts under:

- `/tmp/ascendc-runtime-profiles/<session_id>/main-0.json`

This fixed the earlier leak in `/tmp/ascendc-runtime`, because session working
directories are cleaned before `_Exit(0)`.

However, retained profile directories are currently never pruned. Over time they
will accumulate, especially under repeated `run_runtime.sh`,
`run_simbackend_examples.sh`, and autotuner runs.

## Requirements

1. Keep retained profile artifacts available after successful simulator runs.
2. Bound disk usage with a deterministic retention policy.
3. Preserve current CLI behavior:
   - `session.profile[*]` still points to real retained files
   - no new mandatory CLI flags
4. Avoid changing simulator teardown behavior; `_Exit(0)` remains in place.
5. Keep implementation local to runtime/profile retention paths.

## Chosen Policy

Retain only the most recent `N` runtime-session profile directories.

Initial default:

- `N = 20`

Retention is applied under:

- `/tmp/ascendc-runtime-profiles`

Ordering is based on directory modification time.

This policy is preferred over TTL for the first implementation because:

- behavior is deterministic in tests
- it does not depend on wall-clock skew
- it is easy to reason about during repeated local/xvm verification

## Scope

### In Scope

- prune old retained profile session directories after a new one is created
- keep the newest `N` directories
- ignore non-directory entries safely
- ignore deletion failures best-effort unless they break creation of the current session
- focused tests for retention ordering and pruning behavior
- xvm verification via runtime scripts

### Out of Scope

- user-configurable retention flags
- TTL-based cleanup
- pruning of `/tmp/ascendc-runtime`
- session-level aggregated profile indexes
- schema changes

## Design

### Directory Layout

Retained profile artifacts continue to live under:

- `/tmp/ascendc-runtime-profiles/<session_id>/...`

No format change is introduced in this stage.

### Retention Hook

Retention runs in `runtime-session` on the simulator success path:

1. create retained directory for current `session_id`
2. copy profile artifacts into that directory
3. prune older retained session directories beyond the limit
4. destroy `ExecutionSession`
5. flush and `_Exit(0)`

Pruning runs only after the current session's retained artifacts are safely materialized.

### Pruning Rules

- collect direct child directories of `/tmp/ascendc-runtime-profiles`
- sort by last modification time, newest first
- keep the first `N`
- delete the remainder with `remove_all`
- ignore non-directories
- if metadata for an old directory cannot be read, treat it as oldest
- if deleting an old directory fails, continue and do not fail the current run

The current run only fails if:

- the retained root cannot be created
- the current session's profile artifacts cannot be copied

## Testing

### Unit / Focused Tests

Add focused coverage for:

- retaining the current profile artifact still works
- pruning removes directories older than the newest `N`
- pruning keeps the current session directory
- pruning ignores unrelated files in the retain root

### xvm Verification

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

And a repeated verification loop:

```bash
for i in 1 2 3; do
  bash test/tools/runtime/run_runtime.sh
done
```

Expected outcome:

- all runs pass
- retained profile paths remain valid in summaries
- `/tmp/ascendc-runtime-profiles` remains bounded instead of monotonically growing

## Risks

### Over-pruning

If the limit is too small, useful recent traces may be deleted sooner than desired.
This is acceptable for stage 1 because the policy is internal and can be tuned later.

### mtime Dependence

Ordering by modification time is adequate for local/xvm use. If future environments
show unstable ordering, the next fallback is to encode creation order in directory naming.

## Success Criteria

This stage is complete when:

1. simulator runs still emit valid retained profile paths
2. retained profile directories are bounded to the newest `N`
3. repeated `run_runtime.sh` no longer grows retained profile storage without bound
4. focused tests and xvm verification pass
