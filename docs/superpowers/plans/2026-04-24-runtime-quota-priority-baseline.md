# Runtime Quota And Priority Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add session admission quota and static session priority on top of the existing cross-session fairness baseline.

**Architecture:** Keep the stream/resource model and fairness baseline intact, then layer quota and priority only into `GlobalScheduler` session selection. Expose the result through the existing observability/summary path instead of adding a new reporting surface.

**Tech Stack:** C++17, runtime scheduler classes, `test_taskgraph_runtime`, `runtime-session`, xvm shell verification.

---

## File Structure

- Modify: `include/Runtime/Execution/GlobalScheduler.h`
  - add quota/priority scheduler types and session state
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
  - enforce quota and priority during admission
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
  - add focused quota and priority coverage
- Modify: `test/tools/runtime/run_runtime.sh`
  - assert quota/priority attributes on the concurrent DAG path
- Modify: `docs/runtime/README.md`
  - document the quota/priority baseline

## Task 1: Add Session Scheduling Options To GlobalScheduler

**Files:**
- Modify: `include/Runtime/Execution/GlobalScheduler.h`
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
static void testGlobalSchedulerEnforcesSessionAdmissionQuota() {
  GlobalScheduler scheduler;
  scheduler.configureResourceScheduler(/*simDispatchLanes=*/2, /*deviceSlots=*/2,
                                      /*workspaceBudget=*/4096,
                                      /*streamCapacity=*/2);

  BackendCapabilities caps;
  caps.supportsConcurrentDispatch = true;
  caps.supportsConcurrentExecution = true;
  caps.maxConcurrentTasks = 4;
  caps.maxConcurrentStreams = 2;

  SessionSchedulingOptions quotaOne;
  quotaOne.maxAdmittedTasks = 1;

  TaskGraph graphA;
  RuntimeTask a0;
  a0.taskId = "a0";
  a0.invocation.workspaceSize = 16;
  RuntimeTask a1;
  a1.taskId = "a1";
  a1.invocation.workspaceSize = 16;
  EXPECT(!graphA.addTask(a0), "add a0");
  EXPECT(!graphA.addTask(a1), "add a1");

  TaskGraph graphB;
  RuntimeTask b0;
  b0.taskId = "b0";
  b0.invocation.workspaceSize = 16;
  EXPECT(!graphB.addTask(b0), "add b0");

  auto sessionAOr =
      scheduler.submit(ExecutionBackendKind::Simulation, caps, graphA, quotaOne);
  auto sessionBOr =
      scheduler.submit(ExecutionBackendKind::Simulation, caps, graphB);
  EXPECT(static_cast<bool>(sessionAOr) && static_cast<bool>(sessionBOr),
         "submissions succeed");
  if (!sessionAOr || !sessionBOr)
    return;

  auto stats = scheduler.observabilitySnapshot();
  EXPECT(stats.counters.at("scheduler.task.reserved") == 2,
         "quota still allows another session to use spare capacity");
  EXPECT(stats.counters.at("scheduler.quota.blocked") == 1,
         "one ready task is blocked by session quota");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

- [ ] **Step 3: Write minimal implementation**

Add:

```cpp
enum class SessionPriorityClass { Low, Normal, High };

struct SessionSchedulingOptions {
  SessionPriorityClass priorityClass = SessionPriorityClass::Normal;
  size_t maxAdmittedTasks = 0;
};
```

Thread `SessionSchedulingOptions` through `submit(...)` and track `admittedTasks` in `GlobalSessionRecord`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/GlobalScheduler.h \
        lib/Runtime/Execution/GlobalScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: add scheduler quota baseline"
```

## Task 2: Add Static Session Priority On Top Of Fairness

**Files:**
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
static void testGlobalSchedulerPrefersHigherPrioritySessions() {
  GlobalScheduler scheduler;
  scheduler.configureResourceScheduler(/*simDispatchLanes=*/1, /*deviceSlots=*/1,
                                      /*workspaceBudget=*/4096,
                                      /*streamCapacity=*/1);

  BackendCapabilities caps;
  caps.supportsConcurrentDispatch = true;
  caps.supportsConcurrentExecution = true;
  caps.maxConcurrentTasks = 4;
  caps.maxConcurrentStreams = 1;

  SessionSchedulingOptions highPriority;
  highPriority.priorityClass = SessionPriorityClass::High;

  TaskGraph graphLow;
  RuntimeTask l0;
  l0.taskId = "l0";
  l0.invocation.workspaceSize = 16;
  RuntimeTask l1;
  l1.taskId = "l1";
  l1.invocation.workspaceSize = 16;
  EXPECT(!graphLow.addTask(l0), "add l0");
  EXPECT(!graphLow.addTask(l1), "add l1");

  TaskGraph graphHigh;
  RuntimeTask h0;
  h0.taskId = "h0";
  h0.invocation.workspaceSize = 16;
  EXPECT(!graphHigh.addTask(h0), "add h0");

  auto lowSessionOr =
      scheduler.submit(ExecutionBackendKind::Simulation, caps, graphLow);
  auto highSessionOr =
      scheduler.submit(ExecutionBackendKind::Simulation, caps, graphHigh, highPriority);
  EXPECT(static_cast<bool>(lowSessionOr) && static_cast<bool>(highSessionOr),
         "submissions succeed");
  if (!lowSessionOr || !highSessionOr)
    return;

  auto lowFirst = scheduler.waitAndAcquireTask(lowSessionOr->sessionId());
  EXPECT(static_cast<bool>(lowFirst) && lowFirst->has_value(), "low acquires first");
  if (!lowFirst || !lowFirst->has_value())
    return;
  EXPECT(lowFirst->value().taskId == "l0", "low gets initial turn");
  auto completeLow0 = scheduler.completeTask(lowSessionOr->sessionId(), "l0");
  EXPECT(static_cast<bool>(completeLow0), "complete l0");
  if (!completeLow0)
    return;

  auto highNext = scheduler.waitAndAcquireTask(highSessionOr->sessionId());
  EXPECT(static_cast<bool>(highNext) && highNext->has_value(),
         "high priority session gets the next turn");
  if (!highNext || !highNext->has_value())
    return;
  EXPECT(highNext->value().taskId == "h0", "high priority preempts low's second root");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

- [ ] **Step 3: Write minimal implementation**

Pick the highest ready-and-quota-eligible priority class first, then continue round-robin within that class only for the current pass.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

- [ ] **Step 5: Commit**

```bash
git add lib/Runtime/Execution/GlobalScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: add scheduler priority baseline"
```

## Task 3: Expose Quota And Priority Through Runtime Verification Surface

**Files:**
- Modify: `test/tools/runtime/run_runtime.sh`
- Modify: `docs/runtime/README.md`

- [ ] **Step 1: Add shell assertions and docs**

Add assertions:

```bash
grep -q '^session.runtime.attribute.scheduler_priority_policy=static_session_priority$' \
  /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.attribute.scheduler_quota_policy=session_admission_quota$' \
  /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.scheduler.quota.blocked=' \
  /tmp/runtime_session_dag_run.log
```

Update README to say quota/priority baseline now exists on top of fairness.

- [ ] **Step 2: Run xvm baselines**

Run:

```bash
ssh xvm@orb 'cd /Volumes/GM9/code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && ./scripts/build.sh --build-project --llvm-build-dir "$LLVM_BUILD_DIR"'
ssh xvm@orb 'cd /Volumes/GM9/code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
ssh xvm@orb 'cd /Volumes/GM9/code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_simbackend_examples.sh'
ssh xvm@orb 'cd /Volumes/GM9/code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/examples/example_pipelines.sh'
```

- [ ] **Step 3: Commit**

```bash
git add test/tools/runtime/run_runtime.sh docs/runtime/README.md
git commit -m "runtime: document quota and priority baseline"
```
