# xvm Default Verification Cleanup Plan

## Objective

Stabilize the default xvm runtime verification entry points so runtime-focused verification does not get blocked by unrelated outer-target rebuilds, duplicated environment setup, or inconsistent direct test execution shells.

## Task 1: Add shared runtime verification shell helpers

Files:
- `test/tools/runtime/` helper script(s)
- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_simbackend_smoke.sh`
- `test/tools/runtime/run_simbackend_examples.sh`

Steps:
1. Introduce a shared helper script under `test/tools/runtime/`.
2. Centralize:
   - Ascend env resolution
   - LLVM env resolution
   - runtime-focused `LD_LIBRARY_PATH` assembly
   - stale `build/` detection / rebuild policy
   - focused runtime build helper(s)
3. Keep helper responsibilities shell-only; do not move runtime feature logic into scripts.

Verification:
- `bash -n` on the new helper and the updated scripts
- script sourcing works without changing behavior

## Task 2: Reduce default runtime build target noise

Files:
- `test/tools/runtime/run_runtime.sh`
- optionally shared helper script introduced in Task 1

Steps:
1. Replace the current broad build step with the minimal runtime-focused target set.
2. Retain any target only if a downstream runtime verification step truly needs it.
3. Ensure the reduced build still covers:
   - runtime library
   - C API library
   - `runtime-session`
   - `mix-compiler`
   - any runtime-only support binary still proven necessary
4. Do not make this script depend on unrelated outer-tool rebuilds by default.

Verification:
- xvm focused build succeeds with the reduced target set
- `run_runtime.sh` still reaches all existing verification phases

## Task 3: Normalize direct runtime test execution environment

Files:
- shared helper script
- `test/tools/runtime/run_runtime.sh`

Steps:
1. Replace ad hoc direct test `LD_LIBRARY_PATH` assembly with one shared helper output.
2. Ensure the direct execution environment consistently covers:
   - `build/lib`
   - LLVM libs
   - Ascend lib64
   - simulator libs
   - device stub libs where required
3. Keep test commands themselves unchanged unless path plumbing must move.

Verification:
- xvm direct execution of `test_taskgraph_runtime`
- xvm direct execution of `test_capi_runtime`
- xvm direct execution of `test_runtime`

## Task 4: Keep coverage stable while switching scripts to helpers

Files:
- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_simbackend_smoke.sh`
- `test/tools/runtime/run_simbackend_examples.sh`

Steps:
1. Ensure the helper refactor does not drop:
   - runtime-session CLI checks
   - planning / negative-path checks
   - positive vec and DAG simulation checks
   - NPU mock / negative checks
   - repeated mix simulation baseline
   - SimBackend smoke summary output
2. Keep visible output stable where downstream grep checks already depend on it.
3. Avoid opportunistic changes unrelated to verification cleanup.

Verification:
- xvm `bash test/tools/runtime/run_runtime.sh`
- xvm `bash test/tools/runtime/run_simbackend_smoke.sh`

## Task 5: Sync status after cleanup

Files:
- `AGENTS.md`
- any actively maintained runtime verification status doc that now references the old noisy build path

Steps:
1. Record that default xvm runtime verification is now runtime-focused by design.
2. Remove outdated references to unnecessary outer-tool rebuild assumptions.
3. Keep TODO focused on the next real runtime work item after verification cleanup.

Verification:
- `git diff --check`
- final xvm verification evidence recorded before completion

## Acceptance

The cleanup is complete when:
- default runtime xvm verification uses shared env/build helpers
- `run_runtime.sh` no longer rebuilds unrelated outer-tool targets by default
- direct runtime test execution uses one normalized runtime library path setup
- xvm focused verification remains green
