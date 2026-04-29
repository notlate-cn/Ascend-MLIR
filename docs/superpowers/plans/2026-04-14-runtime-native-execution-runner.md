# Runtime-Native Execution Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the implementation behind `DefaultExecutionRunner` so the runtime main execution path no longer depends on `Legacy/Executor` while preserving current simulation, mix, and real-device wiring behavior.

**Architecture:** Keep the existing `ExecutionRunner` / `DefaultExecutionRunner` boundary stable for `SimBackend` and `NpuBackend`, but move the execution substrate itself into `Execution/`. Build the cutover in narrow TDD slices: first codify the current runner contract, then introduce a runtime-native implementation, then switch the default runner over, and finally re-verify xvm-focused runtime behavior.

**Tech Stack:** C++17, LLVM Support (`llvm::Error`, `StringRef`, `MemoryBuffer`, `DynamicLibrary`, `json` already in tree), existing runtime simulator/NPU launch glue, CMake, xvm runtime verification scripts.

---

## File Map

### New files
- `include/Runtime/Execution/NativeExecutionRunner.h` — runtime-native runner interface/implementation declaration backing the existing default runner.
- `lib/Runtime/Execution/NativeExecutionRunner.cpp` — runtime-native execution substrate for simulation and real-device launch.

### Modify files
- `include/Runtime/Execution/DefaultExecutionRunner.h` — replace `Legacy/Executor`-owned state with runtime-native runner state.
- `lib/Runtime/Execution/DefaultExecutionRunner.cpp` — delegate to the new runtime-native runner instead of constructing `Executor`.
- `lib/Runtime/CMakeLists.txt` — compile the new runner implementation and remove any now-unneeded legacy linkage from the default runner path.
- `test/tools/runtime/test_taskgraph_runtime.cpp` — add focused runner-adapter and backend contract tests.
- `test/tools/runtime/test_runtime.cpp` — update explicit legacy executor tests so they only cover retained legacy surface, and add direct runtime-native runner contract checks if needed.
- `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md` — reclassify `Legacy/Executor` after the cutover.
- `AGENTS.md` — update progress/decisions/TODO after successful cutover.

### Reference-only files
- `include/Runtime/Execution/ExecutionRunner.h`
- `lib/Runtime/Execution/SimBackend.cpp`
- `lib/Runtime/Execution/NpuBackend.cpp`
- `include/Runtime/Legacy/Executor.h`
- `lib/Runtime/Legacy/Executor.cpp`
- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_mix_repeat.sh`

### Expected final state
- `SimBackend` / `NpuBackend` still depend only on `ExecutionRunner` / `DefaultExecutionRunner`.
- `DefaultExecutionRunner` no longer includes or constructs `Legacy/Executor`.
- `Legacy/Executor` becomes a retained legacy-only unit or a dead file candidate for the next cleanup round.

---

### Task 1: Lock the current execution-runner contract with focused tests

**Files:**
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Modify: `test/tools/runtime/test_runtime.cpp`

- [ ] **Step 1: Add a failing test for the default runner contract in `test_taskgraph_runtime.cpp`**

Add a focused test near the existing execution-runner coverage that asserts a default runner can still surface the current staged errors through backend usage, without exposing `Legacy/Executor` types. Use the existing fake/driver-based backend patterns already in the file.

```cpp
TEST_CASE("DefaultExecutionRunner preserves backend-facing launch failures",
          "[runtime][execution-runner]") {
  using namespace mlir::runtime;

  TaskGraph graph;
  RuntimeTask task;
  task.taskId = "main";
  task.artifact.kernelName = "missing_kernel";
  task.artifact.kernelKind = KernelKind::Vec;
  task.artifact.deviceBinaryPath = "/tmp/definitely-missing.bin";
  task.invocation.outputBindings.push_back(
      TensorBinding::forOutput("out", "/tmp/out.npy", {1}, DType::F32));
  graph.addTask(task);

  auto backend = createExecutionBackend(ExecutionBackendKind::Simulation);
  REQUIRE(backend);

  auto result = backend->run(graph.tasks().front(), ExecutionContext{});
  REQUIRE_FALSE(result);
  CHECK_THAT(llvm::toString(result.takeError()),
             Catch::Matchers::ContainsSubstring("[sim:kernel_launch]"));
}
```

- [ ] **Step 2: Add a failing test in `test_runtime.cpp` that targets the retained legacy surface only**

Replace any future-facing expectation that would require `Legacy/Executor` on the main path with an explicit retained-legacy smoke check. Keep the missing packed mix library error assertion, but make the test name/state clearly legacy-only.

```cpp
SECTION("Legacy Executor retained packed mix error path") {
  Executor ex(BackendMode::Simulation);
  auto initErr = ex.Initialize(0);
  REQUIRE_FALSE(initErr);

  RunArgs args;
  auto err = ex.RunPackedMixFile("/tmp/missing.so", "fc_relu", args);
  REQUIRE(err);
  CHECK(llvm::toString(std::move(err)).find("packed mix") !=
        std::string::npos);
}
```

- [ ] **Step 3: Run the focused tests to verify they fail for the right reason**

Run:
```bash
cmake --build build --target test_taskgraph_runtime test_runtime -j2
ctest --test-dir build -R 'test_taskgraph_runtime|test_runtime' --output-on-failure
```

Expected:
- `test_taskgraph_runtime` fails because the new runner contract is not fully covered yet or stage text does not match.
- `test_runtime` remains buildable; if the new assertion passes immediately, keep it and proceed.

- [ ] **Step 4: Commit the red tests**

```bash
git add test/tools/runtime/test_taskgraph_runtime.cpp test/tools/runtime/test_runtime.cpp
git commit -m "test: lock runtime-native execution runner contract"
```

---

### Task 2: Introduce a runtime-native execution substrate under `Execution/`

**Files:**
- Create: `include/Runtime/Execution/NativeExecutionRunner.h`
- Create: `lib/Runtime/Execution/NativeExecutionRunner.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Declare the runtime-native runner in `NativeExecutionRunner.h`**

Create a focused class that owns mode-specific runtime library loading, initialization, and launch helpers. Keep the interface narrow and aligned to what `DefaultExecutionRunner` already needs.

```cpp
#pragma once

#include "Runtime/Execution/ExecutionRunner.h"
#include "Runtime/Support/Types.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

class NativeExecutionRunner {
public:
  explicit NativeExecutionRunner(ExecutionRunnerMode mode);
  ~NativeExecutionRunner();

  llvm::Error initialize(int deviceId);
  llvm::Error runBinary(const BinaryLaunchRequest &launch, RunArgs &args,
                        bool dumpBinaryData = false);
  llvm::Error runPackedMix(const PackedMixLaunchRequest &launch, RunArgs &args);

private:
  ExecutionRunnerMode mode_;
  bool initialized_ = false;
  // Add only the runtime-native state required for library handles, stream,
  // context, temporary allocations, and registered kernels.
};

} // namespace mlir::runtime
```

- [ ] **Step 2: Implement the runtime-native runner in `NativeExecutionRunner.cpp`**

Port the execution logic needed by the main runtime path from `Legacy/Executor.cpp` into the new file. Do not drag over unrelated helper surface. Keep internal helpers `static` or in an anonymous namespace.

Structure the file like this:

```cpp
#include "Runtime/Execution/NativeExecutionRunner.h"

#include "Runtime/Support/PathUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir::runtime {
namespace {

// Small private helpers only for this translation unit.
llvm::Error loadRuntimeLibraries(...);
llvm::Error registerBinary(...);
llvm::Error launchRegisteredKernel(...);
llvm::Error launchPackedMixKernel(...);

} // namespace

NativeExecutionRunner::NativeExecutionRunner(ExecutionRunnerMode mode)
    : mode_(mode) {}

NativeExecutionRunner::~NativeExecutionRunner() {
  // Preserve current teardown behavior.
}

llvm::Error NativeExecutionRunner::initialize(int deviceId) {
  // Runtime-native initialization for simulation or real-device mode.
}

llvm::Error NativeExecutionRunner::runBinary(const BinaryLaunchRequest &launch,
                                             RunArgs &args,
                                             bool dumpBinaryData) {
  // Read binary, register, launch, transfer buffers, preserve current errors.
}

llvm::Error NativeExecutionRunner::runPackedMix(
    const PackedMixLaunchRequest &launch, RunArgs &args) {
  // dlopen packed mix library and invoke exported launch function.
}

} // namespace mlir::runtime
```

Implementation rules:
- Preserve existing sim/real-device initialization semantics.
- Preserve current failure messages closely enough that backend stage wrappers stay valid.
- Do not expose `Legacy/Executor` in headers or implementation.

- [ ] **Step 3: Wire the new file into `lib/Runtime/CMakeLists.txt`**

Add the new source alongside the other `Execution/` sources.

```cmake
  Execution/DefaultExecutionRunner.cpp
  Execution/NativeExecutionRunner.cpp
```

Do not remove `Legacy/Executor.cpp` yet.

- [ ] **Step 4: Build the runtime targets to catch compile/link fallout early**

Run:
```bash
cmake --build build --target AscendCRuntime runtime-session autotuner -j2
```

Expected:
- Build fails only on unresolved implementation gaps in `NativeExecutionRunner.cpp`, not on unrelated targets.

- [ ] **Step 5: Commit the runtime-native runner scaffold**

```bash
git add include/Runtime/Execution/NativeExecutionRunner.h lib/Runtime/Execution/NativeExecutionRunner.cpp lib/Runtime/CMakeLists.txt
git commit -m "feat: add runtime-native execution runner"
```

---

### Task 3: Switch `DefaultExecutionRunner` over to the runtime-native substrate

**Files:**
- Modify: `include/Runtime/Execution/DefaultExecutionRunner.h`
- Modify: `lib/Runtime/Execution/DefaultExecutionRunner.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Replace `Legacy/Executor` state in `DefaultExecutionRunner.h`**

Update the header so it owns a `NativeExecutionRunner` instead of a legacy executor.

```cpp
#pragma once

#include "Runtime/Execution/ExecutionRunner.h"
#include "Runtime/Execution/NativeExecutionRunner.h"

namespace mlir::runtime {

class DefaultExecutionRunner final : public ExecutionRunner {
public:
  explicit DefaultExecutionRunner(ExecutionRunnerMode mode);
  llvm::Error initialize(int deviceId) override;
  llvm::Error runBinary(const BinaryLaunchRequest &launch, RunArgs &args,
                        bool dumpBinaryData) override;
  llvm::Error runPackedMix(const PackedMixLaunchRequest &launch,
                           RunArgs &args) override;

private:
  NativeExecutionRunner runner_;
};

} // namespace mlir::runtime
```

- [ ] **Step 2: Rewrite `DefaultExecutionRunner.cpp` as a thin delegate**

Remove `Runtime/Executor.h` and forward directly to the runtime-native runner.

```cpp
#include "Runtime/Execution/DefaultExecutionRunner.h"

namespace mlir::runtime {

DefaultExecutionRunner::DefaultExecutionRunner(ExecutionRunnerMode mode)
    : runner_(mode) {}

llvm::Error DefaultExecutionRunner::initialize(int deviceId) {
  return runner_.initialize(deviceId);
}

llvm::Error DefaultExecutionRunner::runBinary(const BinaryLaunchRequest &launch,
                                              RunArgs &args,
                                              bool dumpBinaryData) {
  return runner_.runBinary(launch, args, dumpBinaryData);
}

llvm::Error DefaultExecutionRunner::runPackedMix(
    const PackedMixLaunchRequest &launch, RunArgs &args) {
  return runner_.runPackedMix(launch, args);
}

} // namespace mlir::runtime
```

- [ ] **Step 3: Prove the seam is gone**

Run:
```bash
rg -n "Runtime/Executor.h" lib/Runtime/Execution/DefaultExecutionRunner.cpp
```

Expected:
- no matches

- [ ] **Step 4: Rebuild and run the focused runtime tests**

Run:
```bash
cmake --build build --target test_taskgraph_runtime test_runtime runtime-session autotuner -j2
ctest --test-dir build -R 'test_taskgraph_runtime|test_runtime' --output-on-failure
```

Expected:
- focused runtime tests pass
- no compile/link references to `Legacy/Executor` remain in `DefaultExecutionRunner`

- [ ] **Step 5: Commit the cutover**

```bash
git add include/Runtime/Execution/DefaultExecutionRunner.h lib/Runtime/Execution/DefaultExecutionRunner.cpp
git commit -m "refactor: route default runner through native execution substrate"
```

---

### Task 4: Re-verify xvm runtime behavior and update cleanup state

**Files:**
- Modify: `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Run xvm-focused runtime verification**

Run:
```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:
- `test_taskgraph_runtime` passes
- `test_capi_runtime` passes
- `test_runtime` passes
- vec/mix smoke passes
- repeated mix baseline passes

- [ ] **Step 2: Run the xvm autotuner vec smoke**

Run:
```bash
source examples/env.sh
cmake --build build --target autotuner -j2
build/bin/autotuner \
  --space examples/relu-broadcast-transpose/tiling_space.json \
  --kernel examples/relu-broadcast-transpose/step8_kernel.cpp \
  --kernel-kind vec \
  --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy \
  --expected examples/relu-broadcast-transpose/output_expected.npy \
  --shape M=640,N=500 \
  --output /tmp/autotuner-best.json
```

Expected:
- command succeeds
- output JSON contains non-zero `score` and `cycle_count`

- [ ] **Step 3: Update the cleanup audit and AGENTS state**

Revise the cleanup classification to reflect that the runtime main path no longer depends on `Legacy/Executor` directly or indirectly through the default runner.

Update language in `docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md` like this:

```md
- `Legacy/Executor`
  - main runtime execution path no longer depends on this file after the
    runtime-native execution runner cutover
  - remaining consumers, if any, are explicit retained legacy-only tests or
    compatibility surface
```

Update `AGENTS.md` `Progress` / `TODO` similarly.

- [ ] **Step 4: Commit verification-state updates**

```bash
git add docs/superpowers/audits/2026-04-13-runtime-legacy-cleanup-candidates.md AGENTS.md
git commit -m "docs: update executor cleanup state"
```

---

## Self-Review

- Spec coverage: the plan covers the runtime-native substrate, default runner cutover, verification, and cleanup-state updates; no spec requirement is left without a task.
- Placeholder scan: no `TBD` / `TODO` placeholders or vague “handle appropriately” steps remain.
- Type consistency: `ExecutionRunnerMode`, `BinaryLaunchRequest`, `PackedMixLaunchRequest`, and `RunArgs` are used consistently across tasks; the plan keeps `DefaultExecutionRunner` as the stable boundary while swapping its implementation.
