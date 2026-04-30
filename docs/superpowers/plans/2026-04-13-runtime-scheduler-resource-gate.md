# Runtime Scheduler Resource Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal resource-aware scheduler gate to `ExecutionSession` so unsupported mix resource types fail early without changing the current serial execution model.

**Architecture:** Keep `ExecutionSession` serial, but replace the placeholder scheduler gate with explicit task eligibility checks based on `KernelKind` and `MixResourceType`. The implementation remains local to the scheduler boundary and is validated with focused unit tests plus xvm simulator regressions.

**Tech Stack:** C++17, LLVM `Error`, existing `ExecutionSession`, `TaskGraph`, `MixResourceType`, xvm runtime verification scripts

---

### Task 1: Add failing scheduler-gate tests

**Files:**
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing tests**

Add a recording backend driver counter test block near the existing scheduler
tests:

```cpp
static RuntimeTask makeGateTask(const std::string &taskId, KernelKind kind,
                                MixResourceType mixType) {
  RuntimeTask task;
  task.taskId = taskId;
  task.artifact.kernelName = taskId + "_kernel";
  task.artifact.kernelKind = kind;
  task.artifact.mixResourceType = mixType;
  task.artifact.socVersion = "Ascend910B1";
  return task;
}

static void testExecutionSessionRejectsUnknownMixResourceType() {
  auto driver = std::make_shared<RecordingBackendDriver>();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  TaskGraph graph;
  graph.addTask(makeGateTask("mix_unknown", KernelKind::Mix,
                             MixResourceType::Unknown));

  auto traceOr = session.run(graph);
  EXPECT(!traceOr, "scheduler rejects unknown mix resource type");
  EXPECT(driver->invocations == 0,
         "backend is not invoked when scheduler gate rejects task");
  if (!traceOr) {
    const std::string message = llvm::toString(traceOr.takeError());
    EXPECT(message.find("unsupported mix resource type") != std::string::npos,
           "error mentions unsupported mix resource type");
    EXPECT(message.find("mix_unknown") != std::string::npos,
           "error mentions task id");
  }
}

static void testExecutionSessionAcceptsSupportedMixResourceTypes() {
  auto driver = std::make_shared<RecordingBackendDriver>();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  TaskGraph graph;
  graph.addTask(makeGateTask("mix_1c1v", KernelKind::Mix,
                             MixResourceType::Mix1C1V));
  graph.addTask(makeGateTask("mix_1c2v", KernelKind::Mix,
                             MixResourceType::Mix1C2V));

  auto traceOr = session.run(graph);
  EXPECT((bool)traceOr,
         "scheduler accepts supported mix resource types");
  EXPECT(driver->invocations == 2,
         "backend runs both supported mix tasks");
}

static void testExecutionSessionAcceptsVecAndCubeTasks() {
  auto driver = std::make_shared<RecordingBackendDriver>();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  TaskGraph graph;
  graph.addTask(makeGateTask("vec_task", KernelKind::Vec,
                             MixResourceType::Unknown));
  graph.addTask(makeGateTask("cube_task", KernelKind::Cube,
                             MixResourceType::Unknown));

  auto traceOr = session.run(graph);
  EXPECT((bool)traceOr, "scheduler accepts vec and cube tasks");
  EXPECT(driver->invocations == 2,
         "backend runs vec and cube tasks");
}
```

Register them in `main()` near other session/scheduler tests:

```cpp
  testExecutionSessionRejectsUnknownMixResourceType();
  testExecutionSessionAcceptsSupportedMixResourceTypes();
  testExecutionSessionAcceptsVecAndCubeTasks();
```

- [ ] **Step 2: Run test to verify it fails**

Run on xvm:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
source scripts/resolve_llvm_env.sh
LLVM_BUILD=$(require_llvm_build_dir)
LLVM_SOURCE_INCLUDE=$(cd "${LLVM_BUILD}/.." && pwd)/include
cmake --build build --target AscendCRuntime -j2
g++ -std=c++17 -I include/ -I "$LLVM_BUILD/include" -I "$LLVM_SOURCE_INCLUDE" \
  test/tools/runtime/test_taskgraph_runtime.cpp build/lib/libAscendCRuntime.a \
  $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) -ldl \
  -o /tmp/test_taskgraph_runtime.gate
/tmp/test_taskgraph_runtime.gate
```

Expected: FAIL because unknown mix resource type is still accepted and backend
invocation count is non-zero.

- [ ] **Step 3: Commit**

```bash
git add test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "test: add scheduler resource gate coverage"
```

### Task 2: Implement the scheduler gate

**Files:**
- Modify: `lib/Runtime/ExecutionSession.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Implement minimal resource eligibility logic**

Replace the current no-op gate:

```cpp
static llvm::Error canScheduleTask(const RuntimeTask &) {
  return llvm::Error::success();
}
```

with:

```cpp
static llvm::Error canScheduleTask(const RuntimeTask &task) {
  switch (task.artifact.kernelKind) {
  case KernelKind::Vec:
  case KernelKind::Cube:
    return llvm::Error::success();
  case KernelKind::Mix:
    switch (task.artifact.mixResourceType) {
    case MixResourceType::Mix1C1V:
    case MixResourceType::Mix1C2V:
      return llvm::Error::success();
    case MixResourceType::Unknown:
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "task %s requests unsupported mix resource type: unknown",
          task.taskId.c_str());
    default:
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "task %s requests unsupported mix resource type: %s",
          task.taskId.c_str(),
          stringifyMixResourceType(task.artifact.mixResourceType).c_str());
    }
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "task %s has unsupported kernel kind",
                                 task.taskId.c_str());
}
```

If `stringifyMixResourceType(...)` does not already exist in accessible runtime
code, add a small local helper in this file instead of widening scope.

- [ ] **Step 2: Run the focused test to verify it passes**

Run:

```bash
/tmp/test_taskgraph_runtime.gate
```

Expected:

- all tests pass
- new scheduler-gate tests pass

- [ ] **Step 3: Commit**

```bash
git add lib/Runtime/ExecutionSession.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: add scheduler resource eligibility gate"
```

### Task 3: Verify xvm simulator regressions stay green

**Files:**
- Modify: none
- Test: `test/tools/runtime/run_runtime.sh`
- Test: `test/tools/runtime/run_simbackend_examples.sh`

- [ ] **Step 1: Run focused runtime verification**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:

- `test_taskgraph_runtime` passes
- `test_capi_runtime` passes
- `test_runtime` passes
- vec and mix smoke examples pass

- [ ] **Step 2: Run full simulator example baseline**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_simbackend_examples.sh
```

Expected:

- all 6 examples pass
- summary remains green

- [ ] **Step 3: Commit verification notes**

No code change. Do not create a commit. Capture the exact xvm results in the
task handoff message.

---

## Self-Review

### Spec coverage

- explicit scheduler eligibility: covered by Tasks 1-2
- supported vec/cube/mix handling: covered by Tasks 1-2
- fail-fast unknown mix resource type: covered by Tasks 1-2
- xvm regression preservation: covered by Task 3

### Placeholder scan

- No `TODO`/`TBD`
- Each task includes explicit code or commands and expected outcomes

### Type consistency

- `canScheduleTask(const RuntimeTask &task)` stays the scheduler gate hook
- `Mix1C1V`, `Mix1C2V`, and `Unknown` are used consistently across tests and implementation
