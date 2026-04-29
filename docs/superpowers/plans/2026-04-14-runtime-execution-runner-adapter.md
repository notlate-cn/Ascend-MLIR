# Runtime Execution Runner Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the direct `Legacy/Executor` dependency from `SimBackend` and `NpuBackend` by introducing a runtime-native execution runner adapter, while preserving runtime behavior and xvm verification.

**Architecture:** Add a narrow `ExecutionRunner` abstraction under `Execution/`, implement a default adapter backed by `Legacy/Executor`, then switch both runtime backends to depend on the adapter instead of constructing `Executor` directly.

**Tech Stack:** C++17, LLVM Support, runtime backends, xvm runtime verification, autotuner vec smoke

---

## File Map

**Create:**
- `include/Runtime/Execution/ExecutionRunner.h`
- `include/Runtime/Execution/DefaultExecutionRunner.h`
- `lib/Runtime/Execution/DefaultExecutionRunner.cpp`

**Modify:**
- `lib/Runtime/Execution/SimBackend.cpp`
- `lib/Runtime/Execution/NpuBackend.cpp`
- `lib/Runtime/CMakeLists.txt`
- `test/tools/runtime/test_taskgraph_runtime.cpp`

**Reference Only:**
- `include/Runtime/Legacy/Executor.h`
- `lib/Runtime/Legacy/Executor.cpp`
- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_simbackend_smoke.sh`

**Verification:**
- `ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'`
- `ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_simbackend_smoke.sh'`
- `ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && cmake --build build --target autotuner -j2 && build/bin/autotuner --space examples/relu-broadcast-transpose/tiling_space.json --kernel examples/relu-broadcast-transpose/step8_kernel.cpp --kernel-kind vec --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy --expected examples/relu-broadcast-transpose/output_expected.npy --shape M=640,N=500 --output /tmp/autotuner-best.json'`
- `rg -n "Runtime/Executor.h" lib/Runtime/Execution/SimBackend.cpp lib/Runtime/Execution/NpuBackend.cpp`

---

### Task 1: Add Execution Runner API And Failing Coverage

**Files:**
- Create: `include/Runtime/Execution/ExecutionRunner.h`
- Create: `include/Runtime/Execution/DefaultExecutionRunner.h`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Add the runtime-native runner API**

Create a minimal runner-facing API that expresses only what `SimBackend` / `NpuBackend` need:

- runner mode enum
- launch request payload or narrow method set
- abstract interface for:
  - `initialize(deviceId)`
  - `runFile(...)`
  - `runPackedMixFile(...)`

Keep the contract runtime-native. Do not expose `Executor` in the API.

- [ ] **Step 2: Add focused failing tests for the adapter seam**

In `test/tools/runtime/test_taskgraph_runtime.cpp`, add focused tests that compile against the new API and lock these expectations:

- a fake execution runner can be defined in tests without including `Runtime/Executor.h`
- the new API can express both vec/cube file launches and packed mix launches
- the runtime-facing runner mode distinguishes simulation vs real-device intent

Keep the tests narrow and compile-time / small behavioral in scope.

- [ ] **Step 3: Capture the red state**

Run focused verification on xvm and confirm the red state is due to missing default adapter implementation, not unrelated regressions.

- [ ] **Step 4: Commit the API skeleton**

```bash
git add include/Runtime/Execution/ExecutionRunner.h include/Runtime/Execution/DefaultExecutionRunner.h test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "test: cover execution runner adapter contract"
```

### Task 2: Implement DefaultExecutionRunner On Top Of Legacy Executor

**Files:**
- Create: `lib/Runtime/Execution/DefaultExecutionRunner.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`

- [ ] **Step 1: Implement the default adapter**

Implement `DefaultExecutionRunner` as the only place in the runtime backend stack that directly uses `Legacy/Executor`.

Expected behavior:
- simulation mode maps to `Executor(BackendMode::Simulation)`
- real-device mode maps to `Executor(BackendMode::RealDevice)`
- initialize delegates to `Executor::Initialize(...)`
- vec/cube delegates to `Executor::RunFile(...)`
- mix delegates to `Executor::RunPackedMixFile(...)`

Keep the adapter thin. Do not move comparison or profiling logic into it.

- [ ] **Step 2: Add the new source to CMake**

Update `lib/Runtime/CMakeLists.txt` to compile the adapter.

- [ ] **Step 3: Turn Task 1 tests green**

Run focused verification and confirm the adapter contract tests now pass.

- [ ] **Step 4: Commit the adapter implementation**

```bash
git add lib/Runtime/Execution/DefaultExecutionRunner.cpp lib/Runtime/CMakeLists.txt
git commit -m "feat: add default execution runner adapter"
```

### Task 3: Switch SimBackend And NpuBackend To The Adapter

**Files:**
- Modify: `lib/Runtime/Execution/SimBackend.cpp`
- Modify: `lib/Runtime/Execution/NpuBackend.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Replace SimBackend direct Executor usage**

In `SimBackend.cpp`:
- remove `#include "Runtime/Executor.h"`
- include the new runner header instead
- replace direct `Executor` construction / initialization / launch calls with the adapter
- preserve existing stage names and error behavior

- [ ] **Step 2: Replace NpuBackend direct Executor usage**

In `NpuBackend.cpp`:
- remove `#include "Runtime/Executor.h"`
- include the new runner header instead
- replace direct `Executor` usage with the adapter
- preserve existing stage names and error behavior

- [ ] **Step 3: Add focused seam regression coverage**

Add one or two runtime tests that prove backend code can consume an injected runner-like abstraction without touching `Legacy/Executor` directly.

- [ ] **Step 4: Verify the direct include seam is gone**

Run:

```bash
rg -n "Runtime/Executor.h" lib/Runtime/Execution/SimBackend.cpp lib/Runtime/Execution/NpuBackend.cpp
```

Expected:
- no matches

- [ ] **Step 5: Commit the backend cutover**

```bash
git add lib/Runtime/Execution/SimBackend.cpp lib/Runtime/Execution/NpuBackend.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "refactor: use execution runner adapter in backends"
```

### Task 4: Verify Consumer Compatibility

**Files:**
- No planned code changes unless fallout requires a minimal fix

- [ ] **Step 1: Run focused xvm runtime verification**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

Expected:
- focused runtime verification remains green

- [ ] **Step 2: Run runtime smoke examples**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_simbackend_smoke.sh'
```

Expected:
- vec smoke passes
- mix smoke passes

- [ ] **Step 3: Run autotuner vec smoke**

Run:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && source examples/env.sh && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && cmake --build build --target autotuner -j2 && build/bin/autotuner --space examples/relu-broadcast-transpose/tiling_space.json --kernel examples/relu-broadcast-transpose/step8_kernel.cpp --kernel-kind vec --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy --expected examples/relu-broadcast-transpose/output_expected.npy --shape M=640,N=500 --output /tmp/autotuner-best.json'
```

Expected:
- smoke still passes
- non-zero `score` / `cycle_count`

- [ ] **Step 4: Summarize remaining legacy status**

Record the post-cutover status:
- `SimBackend` / `NpuBackend` no longer directly include `Runtime/Executor.h`
- `Legacy/Executor` remains only behind `DefaultExecutionRunner`

