# Legacy HostRunnerGen Deletion Design

## Goal

Delete `Legacy/HostRunnerGen` and its public shims outright now that it no longer serves the runtime-native stack.

## Scope

In scope:
- Delete `include/Runtime/HostRunnerGen.h`
- Delete `include/Runtime/Legacy/HostRunnerGen.h`
- Delete `lib/Runtime/Legacy/HostRunnerGen.cpp`
- Delete or update tests that exist only to cover `HostRunnerGen`
- Update runtime build files, audits, and `AGENTS.md`

Out of scope:
- Replacing runner generation with a new runtime-native feature
- Changing `CompatRuntime` or C API behavior
- Reworking runtime CLI or execution behavior

## Current State

`HostRunnerGen` is no longer used by the runtime main path. Its only remaining in-repo consumers are tests:
- `test/tools/runtime/test_runtime.cpp`
- `test/tools/runner/test_runner_gen.cpp`

This makes it a clean physical deletion target.

## Design

### 1. Delete the legacy surface completely

Remove the full `HostRunnerGen` surface:
- public shim header
- legacy header
- implementation file

No forwarding shim or stub remains.

### 2. Remove HostRunnerGen-specific tests

Delete direct coverage that exists only for generated host runners:
- remove `test/tools/runner/test_runner_gen.cpp`
- remove the `HostRunnerGen` section from `test/tools/runtime/test_runtime.cpp`

Do not replace these with new runtime-native tests in this change. The feature itself is being removed.

### 3. Keep runtime-native verification unchanged

Focused runtime verification remains:
- `bash test/tools/runtime/run_runtime.sh`

The deletion is acceptable only if runtime-native compile/sim/profiling flows remain green.

### 4. Documentation and cleanup state

Update:
- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`
- `AGENTS.md`

`HostRunnerGen` should move from `Do Not Touch Yet` to deleted status.

## Risks

### Risk: hidden non-test consumers still exist
Mitigation:
- run source scan before deletion
- let build failures surface missed includes immediately

### Risk: test coverage drops unexpectedly beyond HostRunnerGen
Mitigation:
- restrict test deletions to HostRunnerGen-owned sections only
- keep all runtime-native verification intact

## Verification

Minimum verification:
- source scan confirms no remaining `HostRunnerGen` references in code
- `bash test/tools/runtime/run_runtime.sh` on xvm passes

Optional confirmation:
- `git grep`/`rg` shows no `Runtime/HostRunnerGen.h` or `HostRunnerGen` code references outside historical docs

## Expected Outcome

After this change:
- `HostRunnerGen` no longer exists in the codebase
- runtime-native verification remains green
- the remaining `Legacy/` cleanup focus narrows to `CompatRuntime` and the retained `Legacy/Compiler` unit
