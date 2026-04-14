# Runtime Output Comparator Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the direct `Legacy/SimValidator` dependency from `SimBackend` and `NpuBackend` by extracting compare-only output validation into a shared runtime-native comparator.

**Architecture:** Add a small `OutputComparator` component under `Execution/` that compares actual and expected `NDArray` outputs and returns the same essential result shape currently used by both backends. Then switch `SimBackend` and `NpuBackend` to the new comparator while preserving stage-error behavior and xvm verification results.

**Tech Stack:** C++17, LLVM Support, runtime `NDArray`/`RunArgs`, xvm runtime verification, autotuner smoke

---

## File Map

**Create:**
- `include/Runtime/Execution/OutputComparator.h` — runtime-native output comparison API
- `include/Runtime/OutputComparator.h` — forwarding shim
- `lib/Runtime/Execution/OutputComparator.cpp` — compare-only implementation for `NDArray` outputs

**Modify:**
- `lib/Runtime/Execution/SimBackend.cpp` — replace direct `SimValidator` compare-only call with the new comparator
- `lib/Runtime/Execution/NpuBackend.cpp` — replace direct `SimValidator` compare-only call with the new comparator
- `lib/Runtime/CMakeLists.txt` — compile the new source
- `test/tools/runtime/test_taskgraph_runtime.cpp` — add focused comparator tests and backend validation-path regression coverage

**Reference Only:**
- `include/Runtime/Legacy/SimValidator.h`
- `lib/Runtime/Legacy/SimValidator.cpp`
- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_simbackend_examples.sh`

**Verification:**
- `ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'`
- `ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_simbackend_examples.sh'`
- `ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && cmake --build build --target autotuner -j2 && build/bin/autotuner --space examples/relu-broadcast-transpose/tiling_space.json --kernel examples/relu-broadcast-transpose/step8_kernel.cpp --kernel-kind vec --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy --expected examples/relu-broadcast-transpose/output_expected.npy --shape M=640,N=500 --output /tmp/autotuner-best.json'`
- `rg -n "Runtime/SimValidator.h" lib/Runtime/Execution/SimBackend.cpp lib/Runtime/Execution/NpuBackend.cpp`

---

### Task 1: Add Comparator API Skeleton And Failing Coverage

**Files:**
- Create: `include/Runtime/Execution/OutputComparator.h`
- Create: `include/Runtime/OutputComparator.h`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Add the minimal comparator public API**

Create `include/Runtime/Execution/OutputComparator.h` with a small result type and compare function:

```cpp
#pragma once

#include "Runtime/TaskGraph.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

struct OutputComparisonResult {
  bool passed = false;
  double maxAbsDiff = 0.0;
  double meanAbsDiff = 0.0;
  std::string errorMessage;
};

llvm::Expected<OutputComparisonResult>
compareRuntimeOutputs(llvm::ArrayRef<NDArray> actual,
                      llvm::ArrayRef<NDArray> expected,
                      double atol,
                      double rtol);

} // namespace mlir::runtime
```

Create the forwarding shim `include/Runtime/OutputComparator.h`:

```cpp
#pragma once

#include "Runtime/Execution/OutputComparator.h"
```

- [ ] **Step 2: Add focused failing tests for comparator behavior**

In `test/tools/runtime/test_taskgraph_runtime.cpp`, add focused tests that compile against the new API and lock the expected comparison contract.

Add tests for:
- exact match passes with zero diffs
- clear mismatch fails with non-empty error data
- mismatched shape or dtype returns an `llvm::Error`

Use existing `NDArray` fixture helpers and simple in-memory arrays. Example skeleton:

```cpp
static void testOutputComparatorPassesMatchingArrays() {
  NDArray actual;
  actual.shape = {2};
  actual.dtype = DType::F32;
  actual.allocate();
  static_cast<float *>(actual.data)[0] = 1.0f;
  static_cast<float *>(actual.data)[1] = 2.0f;

  NDArray expected;
  expected.shape = {2};
  expected.dtype = DType::F32;
  expected.allocate();
  static_cast<float *>(expected.data)[0] = 1.0f;
  static_cast<float *>(expected.data)[1] = 2.0f;

  auto resultOr = compareRuntimeOutputs({actual}, {expected}, 1.0, 1e-2);
  EXPECT((bool)resultOr, "output comparator returns result for matching arrays");
  if (!resultOr)
    return;
  EXPECT(resultOr->passed, "output comparator marks identical arrays as passing");
  EXPECT(resultOr->maxAbsDiff == 0.0, "output comparator reports zero max diff");
}
```

- [ ] **Step 3: Add a red backend-level regression test that references the comparator**

Add one focused test that proves the backend validation path will eventually consume the new comparator result shape, even if the implementation is not wired yet. Keep it narrow and local to the runtime test file.

- [ ] **Step 4: Run xvm verification and capture the red state**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- red because `compareRuntimeOutputs(...)` is not implemented / linked yet
- not due to unrelated runtime regressions

- [ ] **Step 5: Commit the API skeleton and failing coverage**

```bash
git add include/Runtime/Execution/OutputComparator.h include/Runtime/OutputComparator.h test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "test: cover runtime output comparator contract"
```

### Task 2: Implement The Runtime-Native Comparator

**Files:**
- Create: `lib/Runtime/Execution/OutputComparator.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`
- Reference: `include/Runtime/Legacy/SimValidator.h`
- Reference: `lib/Runtime/Legacy/SimValidator.cpp`

- [ ] **Step 1: Implement compareRuntimeOutputs(...)**

Create `lib/Runtime/Execution/OutputComparator.cpp` and implement:

- compare output count
- compare shape equality
- compare dtype equality
- compute per-element absolute diff
- apply `atol` / `rtol`
- accumulate:
  - `passed`
  - `maxAbsDiff`
  - `meanAbsDiff`
- return `llvm::Error` for structural mismatches (count/shape/dtype)
- return `OutputComparisonResult` for numeric comparison outcomes

The comparator must remain backend-agnostic and must not depend on `Executor` or `SimValidator`.

- [ ] **Step 2: Add the new source file to the runtime library build**

Update `lib/Runtime/CMakeLists.txt` to compile `Execution/OutputComparator.cpp`.

- [ ] **Step 3: Turn Task 1 tests green**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- the new comparator tests pass
- no existing runtime smoke regressions

- [ ] **Step 4: Commit the comparator implementation**

```bash
git add lib/Runtime/Execution/OutputComparator.cpp lib/Runtime/CMakeLists.txt
git commit -m "feat: add runtime output comparator"
```

### Task 3: Switch SimBackend And NpuBackend To The New Comparator

**Files:**
- Modify: `lib/Runtime/Execution/SimBackend.cpp`
- Modify: `lib/Runtime/Execution/NpuBackend.cpp`
- Reference: `include/Runtime/OutputComparator.h`

- [ ] **Step 1: Replace SimBackend compare-only validation**

In `SimBackend.cpp`:
- remove `#include "Runtime/SimValidator.h"`
- include `Runtime/OutputComparator.h`
- replace the `SimValidator validator; validator.CompareOnly(...)` block with `compareRuntimeOutputs(...)`
- preserve current stage-error behavior under `[sim:validate]`

- [ ] **Step 2: Replace NpuBackend compare-only validation**

In `NpuBackend.cpp`:
- remove `#include "Runtime/SimValidator.h"`
- include `Runtime/OutputComparator.h`
- replace the `SimValidator validator; validator.CompareOnly(...)` block with `compareRuntimeOutputs(...)`
- preserve current stage-error behavior under `[npu:validate]`

- [ ] **Step 3: Verify the direct SimValidator include seam is gone**

Run:

```bash
rg -n "Runtime/SimValidator.h" lib/Runtime/Execution/SimBackend.cpp lib/Runtime/Execution/NpuBackend.cpp
```

Expected:
- no matches

- [ ] **Step 4: Run focused xvm verification**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- full pass
- no runtime validation-path regressions

- [ ] **Step 5: Commit the backend cutover**

```bash
git add lib/Runtime/Execution/SimBackend.cpp lib/Runtime/Execution/NpuBackend.cpp
git commit -m "refactor: route backend validation through output comparator"
```

### Task 4: Verify Consumer Compatibility

**Files:**
- Verify: `test/tools/runtime/run_simbackend_examples.sh`
- Verify: `tools/autotuner/autotuner_main.cpp`
- Modify only if a narrow verification-driven fix is required

- [ ] **Step 1: Run xvm simulation example baseline**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_simbackend_examples.sh'`
```

Expected:
- all 6 examples pass
- no tolerance/validation regression

- [ ] **Step 2: Run xvm autotuner vec smoke**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && cmake --build build --target autotuner -j2 && build/bin/autotuner --space examples/relu-broadcast-transpose/tiling_space.json --kernel examples/relu-broadcast-transpose/step8_kernel.cpp --kernel-kind vec --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy --expected examples/relu-broadcast-transpose/output_expected.npy --shape M=640,N=500 --output /tmp/autotuner-best.json'`
```

Expected:
- success
- non-zero `score` / `cycle_count`

- [ ] **Step 3: If verification finds no regression, leave consumer code unchanged**

Do not change examples or autotuner unless a real compatibility regression is observed.

- [ ] **Step 4: Commit only if a small verification-driven compatibility fix was needed**

```bash
git add <only compatibility fix files>
git commit -m "fix: preserve runtime comparator consumer compatibility"
```

## Self-Review

- Spec coverage:
  - comparator extraction: covered by Tasks 1 and 2
  - backend cutover away from `SimValidator`: covered by Task 3
  - preserve validation behavior and xvm verification: covered by Tasks 3 and 4
  - no scope expansion into `Executor`: preserved
- Placeholder scan:
  - no `TODO`, `TBD`, or undefined “similar to” references remain
- Type consistency:
  - `OutputComparisonResult` and `compareRuntimeOutputs(...)` are named consistently across tasks
