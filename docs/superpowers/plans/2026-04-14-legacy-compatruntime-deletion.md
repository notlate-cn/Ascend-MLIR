# Legacy CompatRuntime Deletion Plan

## Objective

Delete `CompatRuntime` and move its remaining helper responsibilities directly into runtime-native call sites.

## Task 1: Remove compat-specific test coverage
- Update `test/tools/runtime/test_taskgraph_runtime.cpp`
- Remove coverage that exists only for `buildCompatCompileRequest(...)`
- Remove coverage that exists only for `buildCompatSingleTaskRunManifest(...)`
- Keep or replace only coverage for behavior that still exists after direct request construction
- Verification:
  - source scan narrows remaining `CompatRuntime` references to C API + implementation pending deletion

## Task 2: Cut C API off CompatRuntime
- Update `lib/CAPI/Runtime/Runtime.cpp`
- Replace compat compile-request construction with direct `ArtifactCompileRequest` assembly
- Replace compat run-manifest construction with direct runtime-native manifest/task assembly
- Inline or relocate minimal tiling-path preparation needed by C API
- Verification:
  - C API no longer includes `Runtime/CompatRuntime.h`

## Task 3: Delete CompatRuntime files and build entries
- Delete:
  - `include/Runtime/CompatRuntime.h`
  - `include/Runtime/Legacy/CompatRuntime.h`
  - `lib/Runtime/Legacy/CompatRuntime.cpp`
- Update `lib/Runtime/CMakeLists.txt`
- Verification:
  - source scan shows no remaining code references to `CompatRuntime` helpers

## Task 4: Sync docs and xvm verification
- Update cleanup audit and `AGENTS.md`
- Run `bash test/tools/runtime/run_runtime.sh` on xvm
- Verification:
  - report runtime, C API, and smoke results

## Notes
- Do not redesign the C API surface in this plan.
- Do not introduce a new compatibility helper layer under a different name.
- If direct request construction reveals shared logic worth reusing, move only that minimal logic into an existing runtime-native module.
