# RuntimeMix Direct Packed Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `RuntimeMix` direct packed execution for `mix` kernels stable on xvm for the current `baremix` acceptance surface, so validator semantics align with existing `lib/Runtime`.

**Architecture:** First capture and prove the runner-vs-direct execution gap with minimal diagnostics, then fix only `RuntimeMix::Executor::RunPackedMixFile(...)` to match the working launch semantics. Keep ABI metadata, artifact layout, and current `baremix` acceptance surface unchanged.

**Tech Stack:** C++17, LLVM Support, RuntimeMix `Executor`, xvm simulator, bash-based validation, existing `examples/baremix-test` acceptance flow.

---

## File Structure

- Modify: `include/RuntimeMix/Executor.h`
  Purpose: expose only the minimal helper declarations needed to support a root-cause-driven fix in direct packed execution.

- Modify: `lib/RuntimeMix/Executor.cpp`
  Purpose: capture focused direct packed diagnostics, implement the minimal fix in `RunPackedMixFile(...)`, and keep the rest of runtime execution behavior unchanged.

- Modify: `tools/mix-validator/mix_validator_main.cpp`
  Purpose: only if strictly needed to add a narrow debug switch or to force the direct packed path during validation; do not mix in new ABI behavior.

- Test/Verify: `examples/baremix-test/run.sh`
  Purpose: normal runner-backed acceptance path must remain passing on xvm.

- Test/Verify: xvm direct fallback invocation via `tools/mix-validator`
  Purpose: prove `RunPackedMixFile(...)` is stable without `mix_runner`.

## Task 1: Reproduce and Capture the Direct Packed Failure

**Files:**
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Test: `examples/baremix-test/run.sh`

- [ ] **Step 1: Add the smallest possible debug forcing switch for direct packed validation**

Add a narrow boolean command-line flag in `tools/mix-validator/mix_validator_main.cpp` so xvm validation can force the direct packed path even when `mix_runner` exists. Keep it debug-scoped and avoid changing default behavior. The intended shape is:

```cpp
static cl::opt<bool> ForceDirectPacked(
    "force-direct-packed",
    cl::desc("Bypass mix_runner and validate through Executor::RunPackedMixFile"),
    cl::init(false));
```

Then gate the runner availability logic so:

```cpp
const bool canUseRunner = llvm::sys::fs::exists(runnerPath) && !ForceDirectPacked;
const bool canUseDirectPacked = !manifestKernelName.empty() &&
                                !manifestKernelSo.empty() &&
                                llvm::sys::fs::exists(manifestKernelSo) &&
                                (!canUseRunner);
```

- [ ] **Step 2: Rebuild validator and prove the forced direct path still fails on xvm before any fix**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-validator &&
  bash examples/baremix-test/run.sh >/tmp/runtime-mix-direct-baseline.log &&
  ./build/runtime-mix-bootstrap/bin/mix-validator \
    --artifact-root build/runtime-mix-baremix \
    --input-dir build/runtime-mix-baremix/testdata/input \
    --golden build/runtime-mix-baremix/testdata/output/golden.bin \
    --output-file build/runtime-mix-baremix/testdata/output/direct-actual.bin \
    --soc Ascend910B1 \
    --force-direct-packed
'
```

Expected:
- Normal `run.sh` path still succeeds first.
- The forced direct packed path fails with the current simulator/runtime symptom.

- [ ] **Step 3: Capture the exact failure evidence and localize it to `RunPackedMixFile(...)`**

Record the actual stderr from the forced direct packed path in notes or the task handoff. The evidence must include the exact error text, for example simulator `div by 0`, `invalid ldst addr`, or a launch/runtime return code.

The task is incomplete unless it states a concrete hypothesis in this form:

```text
I think direct packed fails because <specific launch/runtime difference>, because the runner-backed path does <X> while RunPackedMixFile currently does <Y>.
```

- [ ] **Step 4: Commit the repro-only instrumentation**

```bash
git add tools/mix-validator/mix_validator_main.cpp
git commit -m "test: add direct packed validator repro switch"
```

## Task 2: Diff Runner Semantics Against Direct Packed Launch

**Files:**
- Modify: `lib/RuntimeMix/Executor.cpp`
- Test: xvm direct packed validator repro

- [ ] **Step 1: Add narrow diagnostic output around `RunPackedMixFile(...)` without changing behavior yet**

In `lib/RuntimeMix/Executor.cpp`, add temporary debug prints guarded by an env var so the direct packed path can expose:
- shared library path
- launcher symbol name
- input/output byte sizes
- workspace byte size
- tiling byte size
- launch return code
- stream synchronize return code

The intended shape is:

```cpp
static bool runtimeMixDebugEnabled() {
  const char *v = std::getenv("RUNTIMEMIX_DEBUG");
  return v && std::string(v) != "0";
}
```

and then:

```cpp
if (runtimeMixDebugEnabled()) {
  llvm::errs() << "[RuntimeMix::RunPackedMixFile] workspace=" << args.workspace_size
               << " tiling=" << args.tiling.size() << " block_dim=" << args.block_dim
               << " so=" << shared_lib_path << " symbol=" << launch_name << "\n";
}
```

- [ ] **Step 2: Run the forced direct packed repro with debug enabled on xvm**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-validator &&
  RUNTIMEMIX_DEBUG=1 ./build/runtime-mix-bootstrap/bin/mix-validator \
    --artifact-root build/runtime-mix-baremix \
    --input-dir build/runtime-mix-baremix/testdata/input \
    --golden build/runtime-mix-baremix/testdata/output/golden.bin \
    --output-file build/runtime-mix-baremix/testdata/output/direct-actual.bin \
    --soc Ascend910B1 \
    --force-direct-packed
'
```

Expected:
- The same failure still reproduces.
- Diagnostics make the direct packed launch inputs explicit.

- [ ] **Step 3: Compare those diagnostics against the working runner-generated path**

Use the generated runner source and current artifact metadata to compare:
- `workspaceSize`
- `tilingFileSize`
- launch symbol signature
- launch argument order
- allocator/copy behavior

Run:

```bash
rg -n "workspaceSize|tilingFileSize|ACLRT_LAUNCH_KERNEL|aclrtlaunch_" \
  build/runtime-mix-baremix/work/main.cpp \
  build/runtime-mix-baremix/work/*tiling.cpp \
  lib/RuntimeMix/Executor.cpp
```

Expected:
- A concrete, written diff list with only the real behavioral gaps.

- [ ] **Step 4: Commit the diagnostic scaffolding if it is still needed for the fix**

If the diagnostics are still useful for the next task:

```bash
git add lib/RuntimeMix/Executor.cpp
git commit -m "debug: instrument direct packed mix execution"
```

If they are not needed, remove them before continuing and skip this commit.

## Task 3: Implement the Minimal Root-Cause Fix in `RunPackedMixFile(...)`

**Files:**
- Modify: `include/RuntimeMix/Executor.h`
- Modify: `lib/RuntimeMix/Executor.cpp`
- Test: xvm forced direct packed validator repro

- [ ] **Step 1: Write down the exact fix shape before editing code**

Create a short implementation note in the task handoff describing:
- the root cause
- why the chosen change fixes it
- why the change belongs in `RunPackedMixFile(...)` instead of validator or backend generation

This must be one explicit hypothesis, not a list of guesses.

- [ ] **Step 2: Apply only the minimal code change needed in `RunPackedMixFile(...)`**

Possible fix shapes are limited to direct execution semantics, for example:
- matching runner-compatible preload or `dlopen` visibility behavior
- matching workspace or tiling allocation/copy semantics
- matching launch-time auxiliary allocation or synchronization order

Do **not**:
- duplicate runner `main.cpp` logic wholesale
- add new ABI parameters
- special-case file names or shapes in `Executor`

- [ ] **Step 3: Re-run the forced direct packed validation on xvm**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-validator &&
  ./build/runtime-mix-bootstrap/bin/mix-validator \
    --artifact-root build/runtime-mix-baremix \
    --input-dir build/runtime-mix-baremix/testdata/input \
    --golden build/runtime-mix-baremix/testdata/output/golden.bin \
    --output-file build/runtime-mix-baremix/testdata/output/direct-actual.bin \
    --soc Ascend910B1 \
    --force-direct-packed
'
```

Expected:
- `PASS`
- no simulator crash symptom

- [ ] **Step 4: Check precision parity**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  python3 examples/baremix-test/scripts/verify_result.py \
    build/runtime-mix-baremix/testdata/output/direct-actual.bin \
    build/runtime-mix-baremix/testdata/output/golden.bin &&
  md5sum \
    build/runtime-mix-baremix/testdata/output/golden.bin \
    build/runtime-mix-baremix/testdata/output/direct-actual.bin
'
```

Expected:
- `test pass`
- identical md5 for golden and direct actual

- [ ] **Step 5: Commit the root-cause fix**

```bash
git add include/RuntimeMix/Executor.h lib/RuntimeMix/Executor.cpp
git commit -m "fix: stabilize direct packed mix execution"
```

## Task 4: Preserve Normal Acceptance and Document the Alignment

**Files:**
- Modify: `examples/baremix-test/README.md`
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Test: `examples/baremix-test/run.sh`

- [ ] **Step 1: Keep normal runner-backed acceptance green**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-compiler build/runtime-mix-bootstrap/bin/mix-validator &&
  bash examples/baremix-test/run.sh
'
```

Expected:
- `PASS`
- `test pass`

- [ ] **Step 2: Update README to state the new execution alignment clearly**

In `examples/baremix-test/README.md`, update the execution note so it explains:
- runtime validation semantics are now aligned toward direct packed execution
- runner still exists as an artifact/debug aid
- baremix remains the only accepted ABI surface in this phase

Use wording like:

```md
当前 `RuntimeMix` 的 `mix` 校验语义已经对齐现有 `lib/Runtime`：主执行模型是 direct packed mix execution。`mix_runner` 仍会随 artifact 生成，便于调试和对照，但不再是长期语义前提。
```

- [ ] **Step 3: If the debug forcing flag is no longer needed for future debugging, remove it**

If `--force-direct-packed` was only needed for diagnosis and no longer adds value, delete it now so the CLI surface stays clean. If you keep it, document that it is a debug-only switch and keep default behavior unchanged.

- [ ] **Step 4: Commit the acceptance/documentation cleanup**

```bash
git add tools/mix-validator/mix_validator_main.cpp examples/baremix-test/README.md
git commit -m "docs: align RuntimeMix mix validation with direct packed execution"
```

## Self-Review

- Spec coverage:
  - root-cause diff is covered by Task 1 and Task 2
  - minimal `RunPackedMixFile(...)` fix is covered by Task 3
  - xvm validation for both direct packed and normal acceptance is covered by Task 3 and Task 4
  - alignment note with existing `lib/Runtime` semantics is covered by Task 4

- Placeholder scan:
  - No `TBD` / `TODO`
  - All tasks include concrete file paths and exact commands

- Type consistency:
  - The plan consistently uses `RunPackedMixFile(...)`, `mix-validator`, `mix_runner`, and `--force-direct-packed`
