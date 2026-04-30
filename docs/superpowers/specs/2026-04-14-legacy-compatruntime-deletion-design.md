# Legacy CompatRuntime Deletion Design

## Goal

Delete `Legacy/CompatRuntime` and its public shim by moving its remaining helper logic directly into runtime-native call sites.

## Scope

In scope:
- Delete `include/Runtime/CompatRuntime.h`
- Delete `include/Runtime/Legacy/CompatRuntime.h`
- Delete `lib/Runtime/Legacy/CompatRuntime.cpp`
- Replace `buildCompatCompileRequest(...)` call sites with direct `ArtifactCompileRequest` construction
- Replace `buildCompatSingleTaskRunManifest(...)` call sites with direct `RunManifestSpec` construction
- Replace `prepareValidatorTilingBinaryPath(...)` usage with runtime-native tiling-path preparation at the remaining call sites
- Update tests, build wiring, cleanup audit, and `AGENTS.md`

Out of scope:
- C API redesign
- `runtime-session` CLI redesign
- changes to execution backends or profiling behavior

## Current State

`CompatRuntime` remains only at the compatibility boundary:
- `lib/CAPI/Runtime/Runtime.cpp`
- `test/tools/runtime/test_taskgraph_runtime.cpp`

No runtime execution backend depends on it.

## Design

### 1. Remove the compat shim entirely

Delete the full compat surface:
- public shim header
- legacy header
- implementation file

Do not leave forwarding shims or stubs.

### 2. C API constructs runtime-native requests directly

`lib/CAPI/Runtime/Runtime.cpp` should stop calling compat helpers and instead:
- build `ArtifactCompileRequest` directly for compile paths
- build `TaskGraph` / `RunManifestSpec` inputs directly for execution paths
- materialize tiling binary paths directly where needed

The C API surface does not change; only the internal assembly path changes.

### 3. Tests move to runtime-native request construction

`test/tools/runtime/test_taskgraph_runtime.cpp` should stop covering compat helper wrappers and instead cover the equivalent runtime-native request construction or remaining helper behavior directly.

Delete compat-specific coverage rather than preserving it through new wrappers.

### 4. Verification remains runtime-focused

Primary verification remains:
- `bash test/tools/runtime/run_runtime.sh` on xvm

This must stay green after removing `CompatRuntime`.

## Risks

### Risk: C API request assembly drifts semantically
Mitigation:
- preserve existing compile/run behavior exactly
- keep focused runtime+CAPI verification green on xvm

### Risk: tiling-path preparation logic gets lost during deletion
Mitigation:
- identify the exact remaining call site(s)
- move only the minimal required logic
- keep negative-path coverage in taskgraph runtime tests

## Verification

Minimum verification:
- source scan shows no remaining `CompatRuntime` references outside docs
- `bash test/tools/runtime/run_runtime.sh` passes on xvm

Optional confirmation:
- direct seam scan for `buildCompatCompileRequest`, `buildCompatSingleTaskRunManifest`, and `prepareValidatorTilingBinaryPath` shows no in-repo code references outside historical docs

## Expected Outcome

After this change:
- `CompatRuntime` no longer exists in the codebase
- C API uses runtime-native request assembly directly
- the remaining `Legacy` focus narrows further to the retained `Legacy/Compiler` unit and any final cleanup around it
