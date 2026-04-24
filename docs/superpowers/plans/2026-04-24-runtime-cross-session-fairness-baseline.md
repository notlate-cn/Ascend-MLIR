# Runtime Cross-Session Fairness Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a session round-robin fairness baseline to `GlobalScheduler` so multiple sessions share admission capacity more predictably without introducing quota or priority.

**Architecture:** Keep `ResourceScheduler` and the stream/resource model unchanged. Restrict this rollout to `GlobalScheduler` ready-task selection order plus shared fairness observability through the existing trace/frontend-summary surface.

**Tech Stack:** C++17, existing runtime scheduler classes, `test_taskgraph_runtime`, `runtime-session`, xvm verification scripts.

---

## File Structure

### Fairness core

- Modify: `include/Runtime/Execution/GlobalScheduler.h`
  - add fairness state and helper declarations
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
  - implement session round-robin admission and fairness counters

### Verification and docs

- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
  - add focused multi-session fairness coverage
- Modify: `test/tools/runtime/run_runtime.sh`
  - assert fairness attributes/counters on the concurrent DAG path
- Modify: `docs/runtime/README.md`
  - document the fairness baseline and clarify that quota/priority are still future work

## Task 1: Add Scheduler Fairness State And Observability

**Files:**
- Modify: `include/Runtime/Execution/GlobalScheduler.h`
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

Add a focused test that proves a later session is admitted after capacity returns instead of the first session taking two turns in a row:

```cpp
static void testGlobalSchedulerRoundRobinsAcrossSessions() {
  GlobalScheduler scheduler;
  scheduler.configureResourceScheduler(/*simDispatchLanes=*/1, /*deviceSlots=*/1,
                                      /*workspaceBudget=*/4096,
                                      /*streamCapacity=*/1);

  BackendCapabilities caps;
  caps.supportsConcurrentDispatch = true;
  caps.supportsConcurrentExecution = true;
  caps.maxConcurrentTasks = 3;
  caps.maxConcurrentStreams = 1;

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

  auto sessionAOr = scheduler.submit(ExecutionBackendKind::Simulation, caps, graphA);
  auto sessionBOr = scheduler.submit(ExecutionBackendKind::Simulation, caps, graphB);
  EXPECT(static_cast<bool>(sessionAOr) && static_cast<bool>(sessionBOr),
         "both sessions submit");
  if (!sessionAOr || !sessionBOr)
    return;

  auto firstA = scheduler.waitAndAcquireTask(sessionAOr->sessionId());
  EXPECT(static_cast<bool>(firstA) && firstA->has_value(), "session A acquires first task");
  if (!firstA || !firstA->has_value())
    return;
  EXPECT(firstA->value().taskId == "a0", "session A gets its first root");

  auto completeA0 = scheduler.completeTask(sessionAOr->sessionId(), "a0");
  EXPECT(static_cast<bool>(completeA0), "complete a0");
  if (!completeA0)
    return;

  auto firstB = scheduler.waitAndAcquireTask(sessionBOr->sessionId());
  EXPECT(static_cast<bool>(firstB) && firstB->has_value(),
         "session B is admitted before session A gets a second turn");
  if (!firstB || !firstB->has_value())
    return;
  EXPECT(firstB->value().taskId == "b0", "fairness gives session B the next turn");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: FAIL because the current scheduler still favors the older session when capacity returns.

- [ ] **Step 3: Write minimal implementation**

Add fairness state to `include/Runtime/Execution/GlobalScheduler.h`:

```cpp
  bool tryReserveOneReadyTaskForSessionLocked(llvm::StringRef sessionId);
  void noteFairnessSessionRemovalLocked(llvm::StringRef sessionId);

  size_t fairnessCursor_ = 0;
  int64_t fairnessRotationCount_ = 0;
  int64_t fairnessSessionSkipCount_ = 0;
  int64_t fairnessStarvationPreventedCount_ = 0;
  std::string lastAdmittedSessionId_;
  std::vector<std::string> sessionOrder_;
```

Change `submit()` / `releaseSession()` / `observabilitySnapshot()` / `tryReserveReadyTasksLocked()` in `lib/Runtime/Execution/GlobalScheduler.cpp` so that:

- new sessions are appended to `sessionOrder_`
- removed sessions are erased from `sessionOrder_` and the cursor is normalized
- the scheduler reports:

```cpp
  snapshot.attributes["scheduler_policy"] =
      "global_session_round_robin_baseline";
  snapshot.attributes["scheduler_fairness_policy"] =
      "session_round_robin";
  snapshot.counters["scheduler.fairness.cursor"] =
      static_cast<int64_t>(fairnessCursor_);
  snapshot.counters["scheduler.fairness.session_order_size"] =
      static_cast<int64_t>(sessionOrder_.size());
  snapshot.counters["scheduler.fairness.session_rotations_total"] =
      fairnessRotationCount_;
  snapshot.counters["scheduler.fairness.session_skips_total"] =
      fairnessSessionSkipCount_;
  snapshot.counters["scheduler.fairness.starvation_prevented_total"] =
      fairnessStarvationPreventedCount_;
```

Use a round-robin pass structure in `tryReserveReadyTasksLocked()`:

```cpp
void GlobalScheduler::tryReserveReadyTasksLocked() {
  while (!sessionOrder_.empty()) {
    const size_t sessionCount = sessionOrder_.size();
    const size_t start = fairnessCursor_ % sessionCount;
    bool admittedAny = false;

    for (size_t offset = 0; offset < sessionCount; ++offset) {
      const size_t index = (start + offset) % sessionCount;
      const std::string sessionId = sessionOrder_[index];
      if (tryReserveOneReadyTaskForSessionLocked(sessionId)) {
        fairnessCursor_ = sessionOrder_.empty() ? 0 : ((index + 1) % sessionOrder_.size());
        ++fairnessRotationCount_;
        admittedAny = true;
      }
    }

    if (!admittedAny)
      break;
  }
}
```

Within `tryReserveOneReadyTaskForSessionLocked(...)`, keep the existing blocked-reason accounting and continue scanning that session’s ready tasks until one reserves or none can.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: PASS with the new round-robin test.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/GlobalScheduler.h \
        lib/Runtime/Execution/GlobalScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: add cross-session fairness baseline"
```

## Task 2: Add Fairness Coverage For Mixed Session Contention

**Files:**
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`

- [ ] **Step 1: Write the failing tests**

Add one more focused test that proves a single session cannot consume all admission slots in one pass when another session is ready:

```cpp
static void testGlobalSchedulerAdmitsAtMostOneTaskPerSessionPerPass() {
  GlobalScheduler scheduler;
  scheduler.configureResourceScheduler(/*simDispatchLanes=*/2, /*deviceSlots=*/2,
                                      /*workspaceBudget=*/4096,
                                      /*streamCapacity=*/2);

  BackendCapabilities caps;
  caps.supportsConcurrentDispatch = true;
  caps.supportsConcurrentExecution = true;
  caps.maxConcurrentTasks = 4;
  caps.maxConcurrentStreams = 2;

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

  auto sessionAOr = scheduler.submit(ExecutionBackendKind::Simulation, caps, graphA);
  auto sessionBOr = scheduler.submit(ExecutionBackendKind::Simulation, caps, graphB);
  EXPECT(static_cast<bool>(sessionAOr) && static_cast<bool>(sessionBOr),
         "submissions succeed");
  if (!sessionAOr || !sessionBOr)
    return;

  auto stats = scheduler.observabilitySnapshot();
  EXPECT(stats.counters.at("scheduler.task.reserved") == 2,
         "only two tasks are reserved");

  auto acquiredA = scheduler.waitAndAcquireTask(sessionAOr->sessionId());
  auto acquiredB = scheduler.waitAndAcquireTask(sessionBOr->sessionId());
  EXPECT(static_cast<bool>(acquiredA) && acquiredA->has_value(), "session A acquires one task");
  EXPECT(static_cast<bool>(acquiredB) && acquiredB->has_value(), "session B acquires one task");
  if (!acquiredA || !acquiredA->has_value() || !acquiredB || !acquiredB->has_value())
    return;
  EXPECT(acquiredA->value().taskId == "a0", "session A gets one root");
  EXPECT(acquiredB->value().taskId == "b0", "session B gets one root");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: FAIL until one-per-session-per-pass fairness is enforced.

- [ ] **Step 3: Write minimal implementation**

If Task 1 did not already enforce the per-pass limit correctly, refine `tryReserveReadyTasksLocked()` so one session cannot succeed twice in the same pass. Keep the change contained to the fairness loop; do not touch `ResourceScheduler`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: PASS with both fairness-focused tests.

- [ ] **Step 5: Commit**

```bash
git add lib/Runtime/Execution/GlobalScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: verify fairness admission behavior"
```

## Task 3: Expose Fairness Baseline Through Runtime Verification Surface

**Files:**
- Modify: `test/tools/runtime/run_runtime.sh`
- Modify: `docs/runtime/README.md`

- [ ] **Step 1: Add shell assertions and docs**

Under the DAG simulation path in `test/tools/runtime/run_runtime.sh`, add:

```bash
grep -q '^session.runtime.attribute.scheduler_policy=global_session_round_robin_baseline$' \
  /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.attribute.scheduler_fairness_policy=session_round_robin$' \
  /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.scheduler.fairness.session_order_size=' \
  /tmp/runtime_session_dag_run.log
grep -q '^session.runtime.counter.scheduler.fairness.session_rotations_total=' \
  /tmp/runtime_session_dag_run.log
```

Update `docs/runtime/README.md` under the scheduler evolution section to say:

```md
### cross-session fairness baseline

`GlobalScheduler` now uses a first-version session round-robin fairness baseline.

- fairness currently applies only to cross-session admission ordering
- session-internal task ordering remains stable and unchanged
- quota / priority remain future work on top of this baseline
```

- [ ] **Step 2: Run focused verification**

Run:

```bash
bash test/tools/runtime/run_runtime.sh
```

Expected: PASS and the DAG path prints the fairness attributes/counters.

- [ ] **Step 3: Run documented xvm baselines**

Run:

```bash
ssh xvm@orb 'cd /Volumes/GM9/code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && ./scripts/build.sh --build-project --llvm-build-dir "$LLVM_BUILD_DIR"'
ssh xvm@orb 'cd /Volumes/GM9/code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
ssh xvm@orb 'cd /Volumes/GM9/code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_simbackend_examples.sh'
ssh xvm@orb 'cd /Volumes/GM9/code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/examples/example_pipelines.sh'
```

Expected:

- `run_runtime.sh`: `RC=0`
- `run_simbackend_examples.sh`: `RC=0`
- `example_pipelines.sh`: `RC=0`

- [ ] **Step 4: Commit**

```bash
git add test/tools/runtime/run_runtime.sh \
        docs/runtime/README.md
git commit -m "runtime: document fairness baseline"
```

## Self-Review

- Spec coverage:
  - round-robin cross-session fairness baseline: Task 1
  - one-per-session-per-pass admission behavior: Task 2
  - shared runtime verification surface and docs: Task 3
- Placeholder scan:
  - no `TODO` / `TBD` placeholders remain
  - each task has exact files, commands, and expected outcomes
- Type consistency:
  - `fairnessCursor_`, `sessionOrder_`, `scheduler_fairness_policy`, and `scheduler.fairness.*` names are used consistently across tasks
