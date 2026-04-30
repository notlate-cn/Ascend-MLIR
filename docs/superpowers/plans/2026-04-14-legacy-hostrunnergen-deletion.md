# Legacy HostRunnerGen Deletion Plan

## Objective

Physically delete `HostRunnerGen` and its test-only consumers without changing runtime-native behavior.

## Task 1: Remove direct HostRunnerGen tests
- Delete `test/tools/runner/test_runner_gen.cpp`
- Remove the `HostRunnerGen` coverage block and includes from `test/tools/runtime/test_runtime.cpp`
- Keep unrelated runtime tests intact
- Verification:
  - source scan shows no remaining test-only `HostRunnerGen` consumers except headers/impl pending deletion

## Task 2: Delete HostRunnerGen code and build entries
- Delete:
  - `include/Runtime/HostRunnerGen.h`
  - `include/Runtime/Legacy/HostRunnerGen.h`
  - `lib/Runtime/Legacy/HostRunnerGen.cpp`
- Update `lib/Runtime/CMakeLists.txt`
- Verification:
  - source scan shows no remaining `HostRunnerGen` code references outside docs
  - runtime library config remains buildable

## Task 3: Sync docs and cleanup status
- Update cleanup audit
- Update `AGENTS.md`
- Mark `HostRunnerGen` as deleted, not retained
- Verification:
  - docs align with code reality

## Task 4: xvm focused verification
- Run `bash test/tools/runtime/run_runtime.sh` on xvm
- Confirm runtime-native verification remains green after deletion
- Verification:
  - report test counts and smoke status

## Notes
- Do not add a replacement runner generator in this plan.
- Do not modify `CompatRuntime` or C API in this plan.
- If hidden non-test consumers appear, stop and reassess rather than reintroducing shims.
