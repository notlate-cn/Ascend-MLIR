# Execution/Profile Contract Cleanup Plan

## Objective

Strengthen the shared execution/profile result contract so `runtime-session` and the C API consume one normalized runtime-native result model instead of reconstructing details independently.

## Task 1: Add focused contract tests

Files:
- `test/tools/runtime/test_taskgraph_runtime.cpp`
- optionally `test/tools/runtime/test_capi_runtime.cpp` if a direct seam check adds value

Steps:
1. Add tests for the strengthened result contract.
2. Cover explicit representation of:
   - backend kind
   - success/failure
   - validation status
   - error stage/message
   - retained task profile artifact paths
   - retained session summary path
3. Ensure tests fail before implementation where behavior/API is missing.

Verification:
- focused contract tests fail first for missing behavior

## Task 2: Tighten shared execution/profile result model

Files:
- shared frontend-core result model files under `include/Runtime/Execution` and `lib/Runtime/Execution`
- any shared profile/helper file that should now feed the normalized result directly

Steps:
1. Upgrade the normalized result model so it directly carries retained profile and validation information.
2. Remove any remaining ambiguity where entry points must infer contract details from raw pieces.
3. Keep the model runtime-native and free of CLI/C ABI presentation concerns.

Verification:
- Task 1 tests go green

## Task 3: Move `runtime-session` to the tightened contract

Files:
- `tools/runtime-session/runtime_session_main.cpp`

Steps:
1. Stop reconstructing result details locally where the normalized contract now provides them.
2. Keep CLI output behavior stable.
3. Ensure success/error/validation/profile printing comes from the normalized contract only.

Verification:
- focused runtime-session behavior stays stable on xvm

## Task 4: Move C API to the tightened contract

Files:
- `lib/CAPI/Runtime/Runtime.cpp`

Steps:
1. Stop reconstructing execution/profile result details locally where the normalized contract now provides them.
2. Preserve existing ABI/error-buffer behavior.
3. Keep the C API as a thin ABI translation layer over the normalized result contract.

Verification:
- `test_capi_runtime` passes on xvm

## Task 5: Focused xvm verification and status sync

Commands:
- runtime-only rebuild of `AscendCRuntime`, `AFIRRuntimeCAPI`, and `runtime-session`
- direct xvm execution of:
  - `test_taskgraph_runtime`
  - `test_capi_runtime`
  - `test_runtime`
  - `bash test/tools/runtime/run_simbackend_smoke.sh`

Fallback:
- if full `run_runtime.sh` is blocked by unrelated outer-tool noise, use the direct runtime-focused subset above and record the blocker explicitly

Acceptance:
- execution/profile outcomes are represented by one normalized contract
- `runtime-session` and C API no longer infer those details separately
- public behavior remains stable
- xvm focused verification stays green
