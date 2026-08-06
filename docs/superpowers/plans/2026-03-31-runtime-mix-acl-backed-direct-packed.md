# RuntimeMix ACL-Backed Direct Packed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `RuntimeMix::Executor::RunPackedMixFile(...)` stable for the current baremix acceptance surface by switching packed mix execution to the ACL/aclrt launch model already proven by the working runner path.

**Architecture:** First preserve the reproducible direct-packed failure with a narrow validator forcing switch, then implement a minimal ACL-backed execution path inside `RuntimeMix::Executor` for packed mix only. Finally, prove on xvm that direct packed no longer times out and that its output matches the runner-backed result.

**Tech Stack:** C++17, LLVM Support, RuntimeMix `Executor`, ACL/aclrt simulator runtime, xvm validation, existing baremix artifact/manifest flow.

---

## File Structure

- Modify: `tools/mix-validator/mix_validator_main.cpp`
  Purpose: keep the debug forcing switch that bypasses `mix_runner` so the direct packed path can be reproduced and verified explicitly.

- Modify: `include/RuntimeMix/Executor.h`
  Purpose: declare only the minimal ACL-backed helpers needed for packed mix execution.

- Modify: `lib/RuntimeMix/Executor.cpp`
  Purpose: implement the ACL-backed packed mix execution path and keep non-mix runtime behavior unchanged.

- Modify: `examples/baremix-test/README.md`
  Purpose: document the direct-packed debug verification path once the fix is in place.

- Test/Verify: `examples/baremix-test/run.sh`
  Purpose: normal acceptance must stay green on xvm.

## Task 1: Preserve the Reproducible Direct Packed Failure

**Files:**
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Test: xvm direct packed forcing command

- [ ] **Step 1: Keep the debug forcing switch explicit and narrow**

Ensure `tools/mix-validator/mix_validator_main.cpp` contains a debug-only switch:

```cpp
static llvm::cl::opt<bool> ForceDirectPacked(
    "force-direct-packed",
    llvm::cl::desc("Bypass mix_runner and validate through direct packed execution"),
    llvm::cl::init(false));
```

And the runner selection logic must stay:

```cpp
const bool canUseRunner = llvm::sys::fs::exists(runnerPath) &&
                          !ForceDirectPacked;
const bool canUseDirectPacked = !manifestKernelName.empty() &&
                                !manifestKernelSo.empty() &&
                                llvm::sys::fs::exists(manifestKernelSo) &&
                                !canUseRunner;
```

- [ ] **Step 2: Re-run the forced direct packed path on xvm to verify the baseline failure**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-validator &&
  bash examples/baremix-test/run.sh >/tmp/runtime-mix-acl-baseline.log &&
  timeout 25 ./build/runtime-mix-bootstrap/bin/mix-validator \
    --artifact-root build/runtime-mix-baremix \
    --input-dir build/runtime-mix-baremix/testdata/input \
    --golden build/runtime-mix-baremix/testdata/output/golden.bin \
    --output-file build/runtime-mix-baremix/testdata/output/direct-actual.bin \
    --soc Ascend910B1 \
    --force-direct-packed; \
  echo EXIT:$?
'
```

Expected:
- simulator emits the current failure signature
- command ends with `EXIT:124`

- [ ] **Step 3: Record the exact failure evidence in the task handoff**

The handoff must include the exact evidence:

```text
div by 0
invalid ldst addr
EXIT:124
```

and the specific root-cause hypothesis:

```text
The packed mix launcher is currently stable under ACL/aclrt execution semantics, but unstable under the existing RuntimeMix Executor rt* execution model.
```

- [ ] **Step 4: Commit the preserved repro hook**

```bash
git add tools/mix-validator/mix_validator_main.cpp
git commit -m "test: preserve direct packed mix repro switch"
```

## Task 2: Add Minimal ACL Entry Points to `RuntimeMix::Executor`

**Files:**
- Modify: `include/RuntimeMix/Executor.h`
- Modify: `lib/RuntimeMix/Executor.cpp`

- [ ] **Step 1: Declare only the minimal ACL function pointers needed for packed mix launch**

Extend `RuntimeMix::Executor` with ACL-backed function pointers needed for the packed mix path only. Keep the additions minimal and scoped to the current runner semantics. The intended declarations in `include/RuntimeMix/Executor.h` are:

```cpp
  int (*aclInit_)(const char*) = nullptr;
  int (*aclFinalize_)() = nullptr;
  int (*aclrtSetDevice_)(int32_t) = nullptr;
  int (*aclrtResetDevice_)(int32_t) = nullptr;
  int (*aclrtCreateStream_)(void**) = nullptr;
  int (*aclrtDestroyStream_)(void*) = nullptr;
  int (*aclrtMalloc_)(void**, uint64_t, uint32_t) = nullptr;
  int (*aclrtFree_)(void*) = nullptr;
  int (*aclrtMallocHost_)(void**) = nullptr;   // adjust signature to actual API used
  int (*aclrtFreeHost_)(void*) = nullptr;
  int (*aclrtMemcpy_)(void*, uint64_t, const void*, uint64_t, int32_t) = nullptr;
  int (*aclrtSynchronizeStream_)(void*) = nullptr;
```

If the exact signatures differ in the local ACL headers/runtime, match those actual signatures instead of forcing this sketch.

- [ ] **Step 2: Load those ACL symbols alongside the existing runtime symbols**

In `lib/RuntimeMix/Executor.cpp`, extend `LoadLib()` so the simulator runtime handle also resolves the ACL/aclrt symbols needed for the packed mix path. Use the same `LOAD(...)` style as the existing `rt*` symbols.

Keep the rest of the runtime loading logic unchanged.

- [ ] **Step 3: Commit the minimal ACL symbol surface**

```bash
git add include/RuntimeMix/Executor.h lib/RuntimeMix/Executor.cpp
git commit -m "refactor: add acl symbols for packed mix execution"
```

## Task 3: Implement ACL-Backed Packed Mix Execution

**Files:**
- Modify: `lib/RuntimeMix/Executor.cpp`
- Test: xvm forced direct packed command

- [ ] **Step 1: Write the failing verification command into the task handoff before editing**

Use this exact command as the failing verification target:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  timeout 25 ./build/runtime-mix-bootstrap/bin/mix-validator \
    --artifact-root build/runtime-mix-baremix \
    --input-dir build/runtime-mix-baremix/testdata/input \
    --golden build/runtime-mix-baremix/testdata/output/golden.bin \
    --output-file build/runtime-mix-baremix/testdata/output/direct-actual.bin \
    --soc Ascend910B1 \
    --force-direct-packed; \
  echo EXIT:$?
'
```

Expected before the fix:
- failure signature
- `EXIT:124`

- [ ] **Step 2: Replace the current packed mix launch sequence with the minimal ACL-backed equivalent**

Inside `RunPackedMixFile(...)` in `lib/RuntimeMix/Executor.cpp`, change only the packed mix execution path so it mirrors the stable runner semantics:

1. `aclInit(nullptr)`
2. `aclrtSetDevice(0)`
3. `aclrtCreateStream(...)`
4. allocate device buffers with `aclrtMalloc`
5. copy host inputs/tiling with `aclrtMemcpy`
6. call `aclrtlaunch_<kernel>`
7. `aclrtSynchronizeStream(...)`
8. copy outputs back with `aclrtMemcpy`
9. free ACL resources

Implementation constraints:
- do not rewrite `Executor::Initialize()` around ACL
- do not touch vec/cube execution paths
- do not special-case file names or tensor shapes here
- keep the existing 3-input / 1-output shape restriction unchanged

- [ ] **Step 3: Keep cleanup and failure handling local to the packed mix path**

If ACL-backed packed mix launch allocates its own stream or ACL resources, clean them up inside `RunPackedMixFile(...)` even on failure. Do not leak them into the general `Executor` lifecycle unless that is strictly required by the simulator API.

- [ ] **Step 4: Re-run the forced direct packed command on xvm**

Run:

```bash
sleep 4 && ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR &&
  rm -f build/runtime-mix-bootstrap/bin/mix-validator &&
  timeout 25 ./build/runtime-mix-bootstrap/bin/mix-validator \
    --artifact-root build/runtime-mix-baremix \
    --input-dir build/runtime-mix-baremix/testdata/input \
    --golden build/runtime-mix-baremix/testdata/output/golden.bin \
    --output-file build/runtime-mix-baremix/testdata/output/direct-actual.bin \
    --soc Ascend910B1 \
    --force-direct-packed; \
  echo EXIT:$?
'
```

Expected:
- no `div by 0`
- no `invalid ldst addr`
- no timeout
- `PASS`
- `EXIT:0`

- [ ] **Step 5: Verify precision parity**

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
- identical md5 values

- [ ] **Step 6: Commit the ACL-backed fix**

```bash
git add lib/RuntimeMix/Executor.cpp
git commit -m "fix: use acl launch model for packed mix execution"
```

## Task 4: Preserve Normal Acceptance and Update Docs

**Files:**
- Modify: `examples/baremix-test/README.md`
- Test: `examples/baremix-test/run.sh`

- [ ] **Step 1: Re-run the normal acceptance path on xvm**

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

- [ ] **Step 2: Update README to document the direct packed debug verification path**

Add a short note to `examples/baremix-test/README.md` describing:
- normal acceptance remains `bash examples/baremix-test/run.sh`
- `mix_runner` still exists for comparison/debug
- `--force-direct-packed` can be used to exercise `Executor::RunPackedMixFile(...)`

Use wording like:

```md
当前 `mix_runner` 仍然保留用于调试和对照；如果需要直接验证 packed mix runtime 路径，可单独执行 `mix-validator --force-direct-packed`。
```

- [ ] **Step 3: Commit the documentation cleanup**

```bash
git add examples/baremix-test/README.md
git commit -m "docs: document direct packed mix verification"
```

## Self-Review

- Spec coverage:
  - ACL-backed direct packed fix is covered by Task 2 and Task 3
  - reproducible failure preservation is covered by Task 1
  - xvm success validation is covered by Task 3 and Task 4
  - README clarification is covered by Task 4

- Placeholder scan:
  - No `TBD` / `TODO`
  - Exact files and commands are present

- Type consistency:
  - The plan consistently uses `RunPackedMixFile(...)`, `mix-validator`, `--force-direct-packed`, and ACL-backed packed mix execution terminology
