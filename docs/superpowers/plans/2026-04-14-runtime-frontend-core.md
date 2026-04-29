# Runtime Frontend Core Plan

## Objective

Unify `runtime-session` and the C API around a shared runtime-native frontend core for request assembly and run-result interpretation, while preserving current public behavior.

## Task 1: Introduce shared frontend-core API coverage

Files:
- `test/tools/runtime/test_taskgraph_runtime.cpp`
- possibly `test/tools/runtime/test_capi_runtime.cpp` if a focused seam test is needed

Steps:
1. Add focused tests for the new shared frontend-core API before implementation.
2. Cover request assembly for the current shared use cases:
   - compile request creation
   - artifact-root loading
   - single-task run assembly
3. Cover shared run-result summary behavior for:
   - success
   - validation failure
   - backend/stage/profile propagation

Verification:
- new focused tests fail first for missing API/behavior

## Task 2: Implement shared frontend core

Files:
- new runtime-native frontend-core header/implementation under `include/Runtime/Artifact` or `include/Runtime/Execution`
- matching `.cpp` in `lib/Runtime`
- `lib/Runtime/CMakeLists.txt`

Steps:
1. Add shared request-assembly helpers for compile and single-task run flows.
2. Add a structured run-result summary type.
3. Keep the API runtime-native; do not include CLI or C ABI concerns.

Verification:
- focused tests from Task 1 go green
- no behavior changes yet in `runtime-session` or C API

## Task 3: Move `runtime-session` onto shared frontend core

Files:
- `tools/runtime-session/runtime_session_main.cpp`

Steps:
1. Replace local request/result assembly with shared frontend-core calls.
2. Preserve current CLI validation order and text-summary semantics.
3. Keep printing local to the CLI.

Verification:
- `runtime-session` behavior stays stable in focused runtime verification
- no silent semantic drift in error precedence or summary lines

## Task 4: Move C API onto shared frontend core

Files:
- `lib/CAPI/Runtime/Runtime.cpp`
- related C API tests if needed

Steps:
1. Replace direct runtime-native request construction with shared frontend-core calls where applicable.
2. Replace duplicated run-result interpretation with shared result-summary handling.
3. Preserve public ABI behavior and error-buffer semantics.

Verification:
- `test_capi_runtime` passes on xvm

## Task 5: Focused xvm verification and status sync

Files:
- `AGENTS.md` if the internal boundary is materially changed

Commands:
- `bash test/tools/runtime/run_runtime.sh`

Fallback if unrelated outer-tool build noise blocks full script completion:
- direct xvm runtime-focused verification of the affected binaries/tests
- explicit note that the blocker is external to the frontend-core change

Acceptance:
- `runtime-session` and C API both use the shared frontend core
- duplicated boundary logic is reduced materially
- public behavior remains stable
- xvm focused verification stays green
