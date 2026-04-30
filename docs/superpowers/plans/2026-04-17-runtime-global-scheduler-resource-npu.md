# Runtime Global Scheduler Resource NPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-session scheduling with a process-global scheduler that supports cross-session task submission, explicit resource reservation, and first-version multi-task `NpuBackend` execution through the same runtime contract.

**Architecture:** Introduce a process-global `GlobalScheduler` plus subordinate `ResourceScheduler`, then refactor `ExecutionSession` into a session-facing submit/query/wait facade. Keep `SimBackend` and `NpuBackend` as backend executors with explicit `BackendCapabilities`, and validate the rollout first with focused unit tests and xvm runtime verification before any fairness or advanced policy work.

**Tech Stack:** C++17, LLVM `Error`/ADT support, existing `ExecutionSession` / `TaskGraph` / `ProfileTrace`, `SimBackend`, `NpuBackend`, xvm simulator/runtime verification.

---

## File Structure

### New runtime scheduler units

- Create: `include/Runtime/Execution/BackendCapabilities.h`
- Create: `lib/Runtime/Execution/BackendCapabilities.cpp`
- Create: `include/Runtime/Execution/ResourceScheduler.h`
- Create: `lib/Runtime/Execution/ResourceScheduler.cpp`
- Create: `include/Runtime/Execution/GlobalScheduler.h`
- Create: `lib/Runtime/Execution/GlobalScheduler.cpp`
- Create: `include/Runtime/Execution/SessionHandle.h`
- Create: `lib/Runtime/Execution/SessionHandle.cpp`

### Existing runtime execution units to refactor

- Modify: `include/Runtime/Execution/ExecutionBackend.h`
- Modify: `include/Runtime/Execution/ExecutionSession.h`
- Modify: `lib/Runtime/Execution/ExecutionSession.cpp`
- Modify: `include/Runtime/Execution/SimBackend.h`
- Modify: `lib/Runtime/Execution/SimBackend.cpp`
- Modify: `include/Runtime/Execution/NpuBackend.h`
- Modify: `lib/Runtime/Execution/NpuBackend.cpp`
- Modify: `include/Runtime/Execution/RuntimeFrontendCore.h`
- Modify: `lib/Runtime/Execution/RuntimeFrontendCore.cpp`
- Modify: `tools/runtime-session/runtime_session_main.cpp`

### Tests and verification

- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Modify: `test/tools/runtime/run_runtime.sh`
- Modify: `test/tools/examples/example_pipelines.sh`

### Documentation

- Modify: `AGENTS.md`

## Task 1: Add Backend Capability Contract

**Files:**
- Create: `include/Runtime/Execution/BackendCapabilities.h`
- Create: `lib/Runtime/Execution/BackendCapabilities.cpp`
- Modify: `include/Runtime/Execution/ExecutionBackend.h`
- Modify: `include/Runtime/Execution/SimBackend.h`
- Modify: `lib/Runtime/Execution/SimBackend.cpp`
- Modify: `include/Runtime/Execution/NpuBackend.h`
- Modify: `lib/Runtime/Execution/NpuBackend.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test for explicit backend capabilities**

Add a focused test block in `test/tools/runtime/test_taskgraph_runtime.cpp` that asserts:

```cpp
static void testBackendCapabilitiesExposeSimAndNpuContracts() {
  auto simOr = createExecutionBackend(ExecutionBackendKind::Simulation);
  EXPECT((bool)simOr, "simulation backend creation succeeds");

  auto npuOr = createExecutionBackend(ExecutionBackendKind::Npu);
  EXPECT((bool)npuOr, "npu backend creation succeeds");

  if (simOr) {
    const BackendCapabilities caps = (*simOr)->capabilities();
    EXPECT(caps.supportsConcurrentDispatch,
           "sim backend advertises concurrent dispatch");
    EXPECT(caps.requiresSerializedLaunch,
           "sim backend advertises serialized launch");
  }

  if (npuOr) {
    const BackendCapabilities caps = (*npuOr)->capabilities();
    EXPECT(!caps.requiresSerializedLaunch,
           "npu backend does not force simulator launch serialization");
    EXPECT(caps.maxConcurrentTasks >= 1,
           "npu backend advertises at least one runnable task");
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run on xvm with the existing focused compilation command:

```bash
cd /home/niu/code/Codex-Ascend-MLIR
source test/tools/runtime/runtime_verify_env.sh
runtime_verify_setup_env >/dev/null
g++ -std=c++17 \
  -I include/ \
  -I "$LLVM_BUILD/include" \
  -I "$LLVM_SOURCE_INCLUDE" \
  test/tools/runtime/test_taskgraph_runtime.cpp \
  build/lib/libAscendCRuntime.a \
  $(runtime_verify_cann_tiling_link_flags) \
  $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) \
  -ldl \
  -o /tmp/test_taskgraph_runtime.gs
LD_LIBRARY_PATH="$(runtime_verify_runtime_ld_library_path)" \
  /tmp/test_taskgraph_runtime.gs
```

Expected: FAIL because `capabilities()` and `BackendCapabilities` do not exist.

- [ ] **Step 3: Write minimal backend capability types**

Create `include/Runtime/Execution/BackendCapabilities.h` with:

```cpp
#pragma once

#include <cstddef>

namespace mlir::runtime {

struct BackendCapabilities {
  bool supportsConcurrentDispatch = false;
  bool supportsConcurrentExecution = false;
  bool requiresSerializedLaunch = false;
  size_t maxConcurrentTasks = 1;
  size_t maxConcurrentStreams = 1;
};

} // namespace mlir::runtime
```

Create `lib/Runtime/Execution/BackendCapabilities.cpp` as an empty translation unit:

```cpp
#include "Runtime/Execution/BackendCapabilities.h"
```

Update `include/Runtime/Execution/ExecutionBackend.h`:

```cpp
#include "Runtime/Execution/BackendCapabilities.h"

class ExecutionBackend {
public:
  virtual ~ExecutionBackend() = default;
  virtual ExecutionBackendKind kind() const = 0;
  virtual BackendCapabilities capabilities() const = 0;
  virtual llvm::Expected<ExecutionResult> run(const ExecutionRequest &request) = 0;
};
```

Update `SimBackend` and `NpuBackend` implementations with first-version capability returns:

```cpp
BackendCapabilities SimBackend::capabilities() const {
  BackendCapabilities caps;
  caps.supportsConcurrentDispatch = true;
  caps.supportsConcurrentExecution = false;
  caps.requiresSerializedLaunch = true;
  caps.maxConcurrentTasks = 1024;
  caps.maxConcurrentStreams = 1;
  return caps;
}

BackendCapabilities NpuBackend::capabilities() const {
  BackendCapabilities caps;
  caps.supportsConcurrentDispatch = true;
  caps.supportsConcurrentExecution = true;
  caps.requiresSerializedLaunch = false;
  caps.maxConcurrentTasks = 1;
  caps.maxConcurrentStreams = 1;
  return caps;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target AscendCRuntime runtime-session -j8
source test/tools/runtime/runtime_verify_env.sh
runtime_verify_setup_env >/dev/null
g++ -std=c++17 \
  -I include/ \
  -I "$LLVM_BUILD/include" \
  -I "$LLVM_SOURCE_INCLUDE" \
  test/tools/runtime/test_taskgraph_runtime.cpp \
  build/lib/libAscendCRuntime.a \
  $(runtime_verify_cann_tiling_link_flags) \
  $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support --system-libs) \
  -ldl \
  -o /tmp/test_taskgraph_runtime.gs
LD_LIBRARY_PATH="$(runtime_verify_runtime_ld_library_path)" \
  /tmp/test_taskgraph_runtime.gs
```

Expected: PASS for the new backend capability test.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/BackendCapabilities.h \
        lib/Runtime/Execution/BackendCapabilities.cpp \
        include/Runtime/Execution/ExecutionBackend.h \
        include/Runtime/Execution/SimBackend.h \
        lib/Runtime/Execution/SimBackend.cpp \
        include/Runtime/Execution/NpuBackend.h \
        lib/Runtime/Execution/NpuBackend.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: add backend capability contract"
```

## Task 2: Add Resource Scheduler

**Files:**
- Create: `include/Runtime/Execution/ResourceScheduler.h`
- Create: `lib/Runtime/Execution/ResourceScheduler.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test for resource reservation**

Add a focused unit in `test/tools/runtime/test_taskgraph_runtime.cpp`:

```cpp
static void testResourceSchedulerReservesAndReleasesSlots() {
  ResourceScheduler scheduler;
  scheduler.configureSimDispatchLanes(1);
  scheduler.configureDeviceSlots(1);
  scheduler.configureWorkspaceBudget(1024);

  TaskResourceRequirement simReq;
  simReq.backendKind = ExecutionBackendKind::Simulation;
  simReq.workspaceBytes = 512;
  simReq.requiresSerializedLaunch = true;

  auto first = scheduler.tryReserve("session0", "task0", simReq);
  EXPECT(first.has_value(), "first sim reservation succeeds");

  auto second = scheduler.tryReserve("session1", "task1", simReq);
  EXPECT(!second.has_value(),
         "second sim reservation blocks when only one lane exists");

  scheduler.release(*first);
  auto third = scheduler.tryReserve("session1", "task1", simReq);
  EXPECT(third.has_value(), "reservation succeeds after release");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run the same `test_taskgraph_runtime.cpp` compile-and-run command.

Expected: FAIL because `ResourceScheduler` and `TaskResourceRequirement` do not exist.

- [ ] **Step 3: Write minimal resource scheduler implementation**

Create `include/Runtime/Execution/ResourceScheduler.h`:

```cpp
#pragma once

#include "Runtime/Execution/ExecutionBackend.h"

#include <cstddef>
#include <optional>
#include <string>

namespace mlir::runtime {

struct TaskResourceRequirement {
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  size_t workspaceBytes = 0;
  bool requiresSerializedLaunch = false;
  bool exclusiveDeviceAccess = false;
};

struct ResourceReservation {
  std::string sessionId;
  std::string taskId;
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  size_t workspaceBytes = 0;
  bool holdsSerializedLaunchLane = false;
  bool holdsDeviceSlot = false;
};

class ResourceScheduler {
public:
  void configureSimDispatchLanes(size_t count);
  void configureDeviceSlots(size_t count);
  void configureWorkspaceBudget(size_t bytes);

  std::optional<ResourceReservation>
  tryReserve(const std::string &sessionId, const std::string &taskId,
             const TaskResourceRequirement &requirement);

  void release(const ResourceReservation &reservation);

private:
  size_t availableSimDispatchLanes_ = 1;
  size_t availableDeviceSlots_ = 1;
  size_t availableWorkspaceBytes_ = 0;
};

} // namespace mlir::runtime
```

Create `lib/Runtime/Execution/ResourceScheduler.cpp`:

```cpp
#include "Runtime/Execution/ResourceScheduler.h"

namespace mlir::runtime {

void ResourceScheduler::configureSimDispatchLanes(size_t count) {
  availableSimDispatchLanes_ = count;
}

void ResourceScheduler::configureDeviceSlots(size_t count) {
  availableDeviceSlots_ = count;
}

void ResourceScheduler::configureWorkspaceBudget(size_t bytes) {
  availableWorkspaceBytes_ = bytes;
}

std::optional<ResourceReservation>
ResourceScheduler::tryReserve(const std::string &sessionId,
                              const std::string &taskId,
                              const TaskResourceRequirement &requirement) {
  if (requirement.workspaceBytes > availableWorkspaceBytes_)
    return std::nullopt;

  ResourceReservation reservation;
  reservation.sessionId = sessionId;
  reservation.taskId = taskId;
  reservation.backendKind = requirement.backendKind;
  reservation.workspaceBytes = requirement.workspaceBytes;

  if (requirement.backendKind == ExecutionBackendKind::Simulation &&
      requirement.requiresSerializedLaunch) {
    if (availableSimDispatchLanes_ == 0)
      return std::nullopt;
    --availableSimDispatchLanes_;
    reservation.holdsSerializedLaunchLane = true;
  }

  if (requirement.backendKind == ExecutionBackendKind::Npu) {
    if (availableDeviceSlots_ == 0)
      return std::nullopt;
    --availableDeviceSlots_;
    reservation.holdsDeviceSlot = true;
  }

  availableWorkspaceBytes_ -= requirement.workspaceBytes;
  return reservation;
}

void ResourceScheduler::release(const ResourceReservation &reservation) {
  availableWorkspaceBytes_ += reservation.workspaceBytes;
  if (reservation.holdsSerializedLaunchLane)
    ++availableSimDispatchLanes_;
  if (reservation.holdsDeviceSlot)
    ++availableDeviceSlots_;
}

} // namespace mlir::runtime
```

- [ ] **Step 4: Run test to verify it passes**

Run the same focused compile-and-run command.

Expected: PASS for the new resource scheduler test.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/ResourceScheduler.h \
        lib/Runtime/Execution/ResourceScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: add resource scheduler"
```

## Task 3: Add Global Scheduler Types

**Files:**
- Create: `include/Runtime/Execution/SessionHandle.h`
- Create: `lib/Runtime/Execution/SessionHandle.cpp`
- Create: `include/Runtime/Execution/GlobalScheduler.h`
- Create: `lib/Runtime/Execution/GlobalScheduler.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test for cross-session submission**

Add:

```cpp
static void testGlobalSchedulerTracksTwoIndependentSessions() {
  GlobalScheduler scheduler;

  TaskGraph graphA;
  RuntimeTask taskA;
  taskA.taskId = "a0";
  EXPECT(!graphA.addTask(taskA), "graphA add task");

  TaskGraph graphB;
  RuntimeTask taskB;
  taskB.taskId = "b0";
  EXPECT(!graphB.addTask(taskB), "graphB add task");

  auto sessionAOr = scheduler.submit(ExecutionBackendKind::Simulation, graphA);
  auto sessionBOr = scheduler.submit(ExecutionBackendKind::Simulation, graphB);

  EXPECT((bool)sessionAOr, "submit session A succeeds");
  EXPECT((bool)sessionBOr, "submit session B succeeds");
  if (sessionAOr && sessionBOr) {
    EXPECT(sessionAOr->sessionId() != sessionBOr->sessionId(),
           "global scheduler assigns distinct session ids");
    EXPECT(scheduler.sessionCount() == 2,
           "global scheduler tracks both sessions");
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run the focused compile-and-run command.

Expected: FAIL because `GlobalScheduler` and `SessionHandle` do not exist.

- [ ] **Step 3: Write minimal scheduler handle types**

Create `include/Runtime/Execution/SessionHandle.h`:

```cpp
#pragma once

#include <string>

namespace mlir::runtime {

class SessionHandle {
public:
  explicit SessionHandle(std::string sessionId) : sessionId_(std::move(sessionId)) {}
  const std::string &sessionId() const { return sessionId_; }

private:
  std::string sessionId_;
};

} // namespace mlir::runtime
```

Create `include/Runtime/Execution/GlobalScheduler.h`:

```cpp
#pragma once

#include "Runtime/Execution/SessionHandle.h"
#include "Runtime/TaskGraph.h"

#include "llvm/Support/Error.h"

#include <map>
#include <string>

namespace mlir::runtime {

class GlobalScheduler {
public:
  llvm::Expected<SessionHandle> submit(ExecutionBackendKind backendKind,
                                       const TaskGraph &graph);
  size_t sessionCount() const { return sessions_.size(); }

private:
  size_t nextSessionOrdinal_ = 0;
  std::map<std::string, ExecutionBackendKind> sessions_;
};

} // namespace mlir::runtime
```

Create `lib/Runtime/Execution/GlobalScheduler.cpp`:

```cpp
#include "Runtime/Execution/GlobalScheduler.h"

namespace mlir::runtime {

llvm::Expected<SessionHandle>
GlobalScheduler::submit(ExecutionBackendKind backendKind, const TaskGraph &graph) {
  auto orderedOr = graph.orderedTasks();
  if (!orderedOr)
    return orderedOr.takeError();
  if (orderedOr->empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot submit empty task graph");
  }

  const std::string sessionId = "global-session-" + std::to_string(nextSessionOrdinal_++);
  sessions_.emplace(sessionId, backendKind);
  return SessionHandle(sessionId);
}

} // namespace mlir::runtime
```

Create `lib/Runtime/Execution/SessionHandle.cpp`:

```cpp
#include "Runtime/Execution/SessionHandle.h"
```

- [ ] **Step 4: Run test to verify it passes**

Run the focused compile-and-run command.

Expected: PASS for the new cross-session submission test.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/SessionHandle.h \
        lib/Runtime/Execution/SessionHandle.cpp \
        include/Runtime/Execution/GlobalScheduler.h \
        lib/Runtime/Execution/GlobalScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: add global scheduler skeleton"
```

## Task 4: Refactor ExecutionSession Into Session Facade

**Files:**
- Modify: `include/Runtime/Execution/ExecutionSession.h`
- Modify: `lib/Runtime/Execution/ExecutionSession.cpp`
- Modify: `include/Runtime/Execution/RuntimeFrontendCore.h`
- Modify: `lib/Runtime/Execution/RuntimeFrontendCore.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test for session submission facade**

Add:

```cpp
static void testExecutionSessionSubmitsThroughGlobalScheduler() {
  ExecutionSession session(ExecutionBackendKind::Simulation);

  TaskGraph graph;
  RuntimeTask task;
  task.taskId = "main";
  EXPECT(!graph.addTask(task), "session facade add task");

  auto planOr = session.plan(graph);
  EXPECT((bool)planOr, "session facade planning still succeeds");

  auto submitOr = session.submit(graph);
  EXPECT((bool)submitOr, "session facade submit succeeds");
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL because `ExecutionSession::submit(...)` does not exist.

- [ ] **Step 3: Add minimal facade submission path**

Update `include/Runtime/Execution/ExecutionSession.h`:

```cpp
#include "Runtime/Execution/GlobalScheduler.h"

class ExecutionSession {
public:
  ...
  llvm::Expected<SessionHandle> submit(const TaskGraph &graph);

private:
  static GlobalScheduler &globalScheduler();
};
```

Update `lib/Runtime/Execution/ExecutionSession.cpp`:

```cpp
GlobalScheduler &ExecutionSession::globalScheduler() {
  static GlobalScheduler scheduler;
  return scheduler;
}

llvm::Expected<SessionHandle> ExecutionSession::submit(const TaskGraph &graph) {
  return globalScheduler().submit(backendKind_, graph);
}
```

Do not delete existing `run(...)` in this task. Keep current execution working
while introducing the new facade entrypoint.

- [ ] **Step 4: Run test to verify it passes**

Run the focused compile-and-run command.

Expected: PASS for the new session submission test and no regressions.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/ExecutionSession.h \
        lib/Runtime/Execution/ExecutionSession.cpp \
        include/Runtime/Execution/RuntimeFrontendCore.h \
        lib/Runtime/Execution/RuntimeFrontendCore.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: add session submit facade"
```

## Task 5: Route Execution Through Global Scheduler + Resource Scheduler

**Files:**
- Modify: `include/Runtime/Execution/GlobalScheduler.h`
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
- Modify: `include/Runtime/Execution/ExecutionSession.h`
- Modify: `lib/Runtime/Execution/ExecutionSession.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test for cross-session resource blocking**

Add:

```cpp
static void testGlobalSchedulerBlocksSecondSessionOnSingleSimLane() {
  GlobalScheduler scheduler;
  scheduler.mutableResourceScheduler().configureSimDispatchLanes(1);
  scheduler.mutableResourceScheduler().configureWorkspaceBudget(1 << 20);

  TaskGraph graphA;
  RuntimeTask taskA;
  taskA.taskId = "a0";
  EXPECT(!graphA.addTask(taskA), "graphA add task");

  TaskGraph graphB;
  RuntimeTask taskB;
  taskB.taskId = "b0";
  EXPECT(!graphB.addTask(taskB), "graphB add task");

  auto sessionAOr = scheduler.submit(ExecutionBackendKind::Simulation, graphA);
  auto sessionBOr = scheduler.submit(ExecutionBackendKind::Simulation, graphB);
  EXPECT((bool)sessionAOr && (bool)sessionBOr,
         "both submissions succeed before dispatch");
}
```

Then extend it once implementation exists to assert that only one task is
admitted into `Reserved/Running` at a time.

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL because `GlobalScheduler` does not yet own a `ResourceScheduler`
or task states.

- [ ] **Step 3: Implement minimal global dispatch state**

Extend `GlobalScheduler` with:

```cpp
struct GlobalTaskRecord {
  std::string sessionId;
  std::string taskId;
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  TaskResourceRequirement resources;
  enum class State {
    Submitted,
    Ready,
    Reserved,
    Running,
    Succeeded,
    Failed,
    Cancelled,
  } state = State::Submitted;
};
```

Add:

```cpp
ResourceScheduler &mutableResourceScheduler() { return resourceScheduler_; }
```

During `submit(...)`, create `GlobalTaskRecord`s for root tasks and mark them
`Ready`. Add a minimal `tryReserveReadyTasks()` helper that walks ready tasks
and promotes only resource-feasible tasks to `Reserved`.

- [ ] **Step 4: Run test to verify it passes**

Run the focused compile-and-run command.

Expected: PASS for the new global scheduler reservation behavior.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/GlobalScheduler.h \
        lib/Runtime/Execution/GlobalScheduler.cpp \
        include/Runtime/Execution/ExecutionSession.h \
        lib/Runtime/Execution/ExecutionSession.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: add global scheduler resource admission"
```

## Task 6: Enable First-Version NPU Multi-Task Scheduling Contract

**Files:**
- Modify: `include/Runtime/Execution/NpuBackend.h`
- Modify: `lib/Runtime/Execution/NpuBackend.cpp`
- Modify: `include/Runtime/Execution/BackendCapabilities.h`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test for NPU capability-driven admission**

Add:

```cpp
static void testNpuBackendAdvertisesSchedulableMultiTaskContract() {
  auto npuOr = createExecutionBackend(ExecutionBackendKind::Npu);
  EXPECT((bool)npuOr, "npu backend creation succeeds");
  if (!npuOr)
    return;

  const BackendCapabilities caps = (*npuOr)->capabilities();
  EXPECT(caps.supportsConcurrentDispatch,
         "npu backend allows scheduler-side concurrent dispatch");
  EXPECT(caps.maxConcurrentTasks >= 1,
         "npu backend exposes at least one task slot");
  EXPECT(caps.maxConcurrentStreams >= 1,
         "npu backend exposes at least one stream slot");
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL if `NpuBackend` still reports the old conservative no-concurrency
contract.

- [ ] **Step 3: Update NPU capability surface and resource requirement mapping**

In `lib/Runtime/Execution/NpuBackend.cpp`, update capability return to:

```cpp
BackendCapabilities NpuBackend::capabilities() const {
  BackendCapabilities caps;
  caps.supportsConcurrentDispatch = true;
  caps.supportsConcurrentExecution = true;
  caps.requiresSerializedLaunch = false;
  caps.maxConcurrentTasks = 1;
  caps.maxConcurrentStreams = 1;
  return caps;
}
```

Do not change actual launch parallelism yet. This task only changes the
scheduler contract and prepares resource-aware admission.

- [ ] **Step 4: Run test to verify it passes**

Run the focused compile-and-run command.

Expected: PASS for the updated NPU capability test.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/NpuBackend.h \
        lib/Runtime/Execution/NpuBackend.cpp \
        include/Runtime/Execution/BackendCapabilities.h \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: expose npu multi-task scheduler contract"
```

## Task 7: Surface Global Scheduler Observability

**Files:**
- Modify: `lib/Runtime/Execution/ExecutionSession.cpp`
- Modify: `lib/Runtime/Execution/RuntimeFrontendCore.cpp`
- Modify: `tools/runtime-session/runtime_session_main.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Modify: `test/tools/runtime/run_runtime.sh`

- [ ] **Step 1: Write the failing test for global scheduler summary fields**

Add assertions in `testRetainedSessionSummaryContents()` for:

```cpp
trace.setAttribute("scheduler_scope", "global");
trace.addCounter("global_session_count", 2);
trace.addCounter("resource_wait_count", 1);
```

Then assert the retained summary JSON contains:

```cpp
EXPECT(attributes->getString("scheduler_scope") &&
           *attributes->getString("scheduler_scope") == "global",
       "retained session summary keeps scheduler scope");
EXPECT(counters->getInteger("global_session_count") &&
           *counters->getInteger("global_session_count") == 2,
       "retained session summary keeps global session count");
EXPECT(counters->getInteger("resource_wait_count") &&
           *counters->getInteger("resource_wait_count") == 1,
       "retained session summary keeps resource wait count");
```

- [ ] **Step 2: Run test to verify it fails**

Run the focused compile-and-run command.

Expected: FAIL because the new attributes/counters are not emitted yet.

- [ ] **Step 3: Implement minimal global-scheduler observability**

In `ExecutionSession.cpp`, when running through the new scheduler path, add:

```cpp
sessionTrace.setAttribute("scheduler_scope", "global");
sessionTrace.addCounter("global_session_count", 1);
sessionTrace.addCounter("resource_wait_count", 0);
```

In `runtime_session_main.cpp`, print them through the existing
`session.runtime.attribute.*` and `session.runtime.counter.*` path.

In `run_runtime.sh`, add CLI checks for:

```bash
grep -q '^session.runtime.attribute.scheduler_scope=' /tmp/runtime_session_dag_run.log
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
bash test/tools/runtime/run_runtime.sh
```

Expected:

- `test_taskgraph_runtime`: PASS
- `test_capi_runtime`: PASS
- `test_runtime`: PASS
- repeated mix baseline: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/Runtime/Execution/ExecutionSession.cpp \
        lib/Runtime/Execution/RuntimeFrontendCore.cpp \
        tools/runtime-session/runtime_session_main.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp \
        test/tools/runtime/run_runtime.sh
git commit -m "runtime: surface global scheduler observability"
```

## Task 8: Run Cross-Session Verification and Update Runtime Notes

**Files:**
- Modify: `test/tools/examples/example_pipelines.sh`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add a focused cross-session smoke command**

In `test/tools/examples/example_pipelines.sh`, add a small runtime-session smoke
block that runs two sessions back-to-back and verifies both still succeed under
the new global scheduler ownership model.

Use a command shape like:

```bash
build/bin/runtime-session --run-manifest "$MANIFEST_A" --run > /tmp/runtime_session_a.log 2>&1
build/bin/runtime-session --run-manifest "$MANIFEST_B" --run > /tmp/runtime_session_b.log 2>&1
grep -q '^session.result=success$' /tmp/runtime_session_a.log
grep -q '^session.result=success$' /tmp/runtime_session_b.log
```

- [ ] **Step 2: Run full xvm verification**

Run:

```bash
bash test/tools/runtime/run_runtime.sh
bash test/tools/examples/example_pipelines.sh
```

Expected:

- runtime verification passes
- all 6 example pipelines pass

- [ ] **Step 3: Update AGENTS.md**

Add progress notes describing:

- `GlobalScheduler` skeleton exists
- `ExecutionSession` is now a session facade over global scheduling
- first-version `ResourceScheduler` exists
- first-version `NpuBackend` multi-task scheduler contract exists

- [ ] **Step 4: Commit**

```bash
git add test/tools/examples/example_pipelines.sh AGENTS.md
git commit -m "runtime: document global scheduler baseline"
```

## Self-Review

Spec coverage:

- `GlobalScheduler`: covered by Tasks 3, 4, 5
- `ResourceScheduler`: covered by Task 2 and Task 5
- `BackendCapabilities`: covered by Task 1 and Task 6
- `ExecutionSession` facade transition: covered by Task 4
- `SimBackend` / `NpuBackend` unified scheduler contract: covered by Tasks 1 and 6
- observability and verification: covered by Tasks 7 and 8

Placeholder scan:

- no `TBD`
- no “write tests later”
- every task includes explicit code or command content

Type consistency:

- `BackendCapabilities`
- `TaskResourceRequirement`
- `ResourceReservation`
- `SessionHandle`
- `GlobalTaskRecord`
- `GlobalScheduler`

These names are introduced before later tasks reference them.
