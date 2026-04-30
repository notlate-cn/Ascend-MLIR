# Runtime Scheduler Accounting And Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the current global scheduler baseline by making resource-accounting lifecycle transitions explicit and exposing shared scheduler observability through the frontend/runtime-session contract.

**Architecture:** Keep `GlobalScheduler` and `ResourceScheduler` as the execution core, but make scheduler state transitions measurable instead of implicit. Add shared runtime-summary fields for scheduler/resource observability, then route `runtime-session` and focused runtime tests through that shared contract so future fairness/quota work has stable evidence surfaces.

**Tech Stack:** C++17, LLVM `Error`/ADT support, existing `ExecutionSession` / `GlobalScheduler` / `ResourceScheduler` / `RuntimeFrontendCore`, focused runtime tests and shell verification.

---

## File Structure

### Runtime scheduler core

- Modify: `include/Runtime/Execution/GlobalScheduler.h`
  - expose a focused runtime-observability snapshot API without leaking internal mutable state
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
  - maintain explicit lifecycle counters and explain blocked admission
- Modify: `include/Runtime/Execution/ExecutionSession.h`
  - keep the session-facing API unchanged unless a minimal helper is needed for observability handoff
- Modify: `lib/Runtime/Execution/ExecutionSession.cpp`
  - merge scheduler observability into the session `ProfileTrace` in one place

### Shared frontend/run contract

- Modify: `include/Runtime/Execution/RuntimeFrontendCore.h`
  - keep runtime observability explicit in `FrontendRunSummary`
- Modify: `lib/Runtime/Execution/RuntimeFrontendCore.cpp`
  - populate the new shared fields from session traces
- Modify: `tools/runtime-session/runtime_session_main.cpp`
  - print scheduler/resource observability only through the shared frontend summary

### Verification

- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
  - add focused scheduler-accounting and shared-summary assertions
- Modify: `test/tools/runtime/run_runtime.sh`
  - assert new runtime summary keys instead of inferring scheduler state indirectly

## Task 1: Add Global Scheduler Observability Snapshot

**Files:**
- Modify: `include/Runtime/Execution/GlobalScheduler.h`
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

Add a focused test block near the existing global scheduler tests in `test/tools/runtime/test_taskgraph_runtime.cpp`:

```cpp
static void testGlobalSchedulerReportsLifecycleCounters() {
  GlobalScheduler scheduler;
  scheduler.configureResourceScheduler(/*simDispatchLanes=*/1, /*deviceSlots=*/1,
                                      std::numeric_limits<size_t>::max());

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
         "both scheduler submissions succeed");

  auto stats = scheduler.observabilitySnapshot();
  EXPECT(stats.attributes.at("scheduler_policy") == "global_fifo_baseline",
         "scheduler snapshot reports the baseline policy");
  EXPECT(stats.counters.at("scheduler.session_count") == 2,
         "scheduler snapshot counts active sessions");
  EXPECT(stats.counters.at("scheduler.task.reserved") == 1,
         "scheduler snapshot counts reserved tasks");
  EXPECT(stats.counters.at("scheduler.task.ready") == 1,
         "scheduler snapshot counts ready tasks waiting on admission");
  EXPECT(stats.counters.at("scheduler.admission.resource_blocked") == 1,
         "scheduler snapshot counts resource-blocked admissions");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: FAIL with missing `observabilitySnapshot()` / missing scheduler counter fields.

- [ ] **Step 3: Write minimal implementation**

Update `include/Runtime/Execution/GlobalScheduler.h`:

```cpp
struct SchedulerObservabilitySnapshot {
  std::map<std::string, std::string> attributes;
  std::map<std::string, int64_t> counters;
};

class GlobalScheduler {
public:
  SchedulerObservabilitySnapshot observabilitySnapshot() const;
```

Add the implementation in `lib/Runtime/Execution/GlobalScheduler.cpp`:

```cpp
SchedulerObservabilitySnapshot GlobalScheduler::observabilitySnapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);

  SchedulerObservabilitySnapshot snapshot;
  snapshot.attributes["scheduler_policy"] = "global_fifo_baseline";
  snapshot.attributes["resource_model_version"] = "v1";
  snapshot.counters["scheduler.session_count"] =
      static_cast<int64_t>(sessions_.size());
  snapshot.counters["scheduler.task.submitted"] =
      static_cast<int64_t>(taskCountInState(GlobalTaskRecord::State::Submitted));
  snapshot.counters["scheduler.task.ready"] =
      static_cast<int64_t>(taskCountInState(GlobalTaskRecord::State::Ready));
  snapshot.counters["scheduler.task.reserved"] =
      static_cast<int64_t>(taskCountInState(GlobalTaskRecord::State::Reserved));
  snapshot.counters["scheduler.task.running"] =
      static_cast<int64_t>(taskCountInState(GlobalTaskRecord::State::Running));
  snapshot.counters["scheduler.task.succeeded"] =
      static_cast<int64_t>(taskCountInState(GlobalTaskRecord::State::Succeeded));
  snapshot.counters["scheduler.task.failed"] =
      static_cast<int64_t>(taskCountInState(GlobalTaskRecord::State::Failed));
  snapshot.counters["scheduler.task.cancelled"] =
      static_cast<int64_t>(taskCountInState(GlobalTaskRecord::State::Cancelled));

  int64_t resourceBlocked = 0;
  for (const auto &entry : tasks_) {
    const GlobalTaskRecord &record = entry.second;
    if (record.state != GlobalTaskRecord::State::Ready)
      continue;
    auto sessionIt = sessions_.find(record.sessionId);
    if (sessionIt == sessions_.end() || sessionIt->second.failed)
      continue;
    ++resourceBlocked;
  }
  snapshot.counters["scheduler.admission.resource_blocked"] = resourceBlocked;
  return snapshot;
}
```

Then refactor `taskCountInState()` into a private locked helper so `observabilitySnapshot()` does not recursively lock the mutex:

```cpp
size_t taskCountInStateLocked(GlobalTaskRecord::State state) const;
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: PASS with the new scheduler observability test succeeding.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/GlobalScheduler.h \
        lib/Runtime/Execution/GlobalScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: add scheduler observability snapshot"
```

## Task 2: Count Admission Outcomes Explicitly

**Files:**
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

Add a second focused test:

```cpp
static void testGlobalSchedulerTracksReleaseAndFailureCounters() {
  GlobalScheduler scheduler;
  scheduler.configureResourceScheduler(/*simDispatchLanes=*/1, /*deviceSlots=*/1,
                                      std::numeric_limits<size_t>::max());

  TaskGraph graph;
  RuntimeTask task;
  task.taskId = "main";
  EXPECT(!graph.addTask(task), "graph add task");

  auto sessionOr = scheduler.submit(ExecutionBackendKind::Simulation, graph);
  EXPECT((bool)sessionOr, "scheduler submission succeeds");
  if (!sessionOr)
    return;

  auto acquiredOr = scheduler.waitAndAcquireTask(sessionOr->sessionId());
  EXPECT((bool)acquiredOr && acquiredOr->has_value(),
         "scheduler acquires the runnable task");
  if (!acquiredOr || !acquiredOr->has_value())
    return;

  EXPECT(!scheduler.failTask(sessionOr->sessionId(), acquiredOr->value().taskId),
         "scheduler failTask succeeds");
  scheduler.releaseSession(sessionOr->sessionId());

  auto stats = scheduler.observabilitySnapshot();
  EXPECT(stats.counters.at("scheduler.transition.failed") == 1,
         "scheduler snapshot counts task failures");
  EXPECT(stats.counters.at("scheduler.transition.session_release") == 1,
         "scheduler snapshot counts session release");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: FAIL because the transition counters are not tracked yet.

- [ ] **Step 3: Write minimal implementation**

Add explicit cumulative counters to `GlobalScheduler` in `include/Runtime/Execution/GlobalScheduler.h`:

```cpp
  int64_t resourceBlockedAdmissionCount_ = 0;
  int64_t successfulReservationCount_ = 0;
  int64_t failedTaskCount_ = 0;
  int64_t completedTaskCount_ = 0;
  int64_t releasedSessionCount_ = 0;
```

Update `tryReserveReadyTasksLocked()` in `lib/Runtime/Execution/GlobalScheduler.cpp`:

```cpp
    auto reservationOr = resourceScheduler_.tryReserve(
        record.sessionId, record.taskId, record.resources);
    if (!reservationOr) {
      ++resourceBlockedAdmissionCount_;
      continue;
    }

    ++successfulReservationCount_;
```

Update `completeTask()` / `failTask()` / `releaseSession()`:

```cpp
  ++completedTaskCount_;
```

```cpp
  ++failedTaskCount_;
```

```cpp
  ++releasedSessionCount_;
```

Expose them from `observabilitySnapshot()`:

```cpp
  snapshot.counters["scheduler.admission.resource_blocked_total"] =
      resourceBlockedAdmissionCount_;
  snapshot.counters["scheduler.admission.reserved_total"] =
      successfulReservationCount_;
  snapshot.counters["scheduler.transition.completed"] = completedTaskCount_;
  snapshot.counters["scheduler.transition.failed"] = failedTaskCount_;
  snapshot.counters["scheduler.transition.session_release"] =
      releasedSessionCount_;
```

Keep the existing point-in-time blocked count (`scheduler.admission.resource_blocked`) as a snapshot value; the new counters are cumulative.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: PASS with explicit failure/release accounting exposed.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/GlobalScheduler.h \
        lib/Runtime/Execution/GlobalScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: count scheduler admission transitions"
```

## Task 3: Merge Scheduler Observability Into Session Trace

**Files:**
- Modify: `lib/Runtime/Execution/ExecutionSession.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

Extend the existing frontend/session contract coverage with a focused assertion:

```cpp
static void testExecutionSessionMergesSchedulerObservabilityIntoTrace() {
  TaskGraph graph;
  RuntimeTask task;
  task.taskId = "main";
  EXPECT(!graph.addTask(task), "graph add task");

  auto driver = std::make_shared<OrderedExecutionBackendDriver>();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);
  auto traceOr = session.run(graph);

  EXPECT((bool)traceOr, "session run succeeds");
  if (!traceOr)
    return;

  auto policy = traceOr->attributes.find("scheduler_policy");
  EXPECT(policy != traceOr->attributes.end() &&
             policy->second == "global_fifo_baseline",
         "session trace surfaces scheduler policy");
  EXPECT(traceOr->counters.find("scheduler.session_count") !=
             traceOr->counters.end(),
         "session trace surfaces scheduler snapshot counters");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: FAIL because `ExecutionSession` does not currently merge global scheduler observability into the session trace.

- [ ] **Step 3: Write minimal implementation**

In `lib/Runtime/Execution/ExecutionSession.cpp`, merge the snapshot before returning from both scheduler modes:

```cpp
static void mergeSchedulerObservability(ProfileTrace &trace,
                                        const SchedulerObservabilitySnapshot &snapshot) {
  for (const auto &[key, value] : snapshot.attributes)
    trace.setAttribute(key, value);
  for (const auto &[key, value] : snapshot.counters)
    trace.addCounter(key, value);
}
```

In the concurrent path, after `globalScheduler().releaseSession(globalSessionId);`:

```cpp
    mergeSchedulerObservability(sessionTrace,
                                globalScheduler().observabilitySnapshot());
```

In the serial path, before the final `return sessionTrace;`:

```cpp
  mergeSchedulerObservability(sessionTrace,
                              globalScheduler().observabilitySnapshot());
```

Use `addCounter()` rather than direct assignment so the merge remains compatible with existing trace accounting.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: PASS with session traces now carrying shared scheduler observability.

- [ ] **Step 5: Commit**

```bash
git add lib/Runtime/Execution/ExecutionSession.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: merge scheduler observability into session trace"
```

## Task 4: Route Shared Frontend Summary Through Explicit Runtime Fields

**Files:**
- Modify: `include/Runtime/Execution/RuntimeFrontendCore.h`
- Modify: `lib/Runtime/Execution/RuntimeFrontendCore.cpp`
- Modify: `tools/runtime-session/runtime_session_main.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

Extend the existing frontend summary tests:

```cpp
EXPECT(successSummary.runtimeAttributes.find("scheduler_mode") !=
           successSummary.runtimeAttributes.end(),
       "frontend summary surfaces runtime attributes explicitly");
EXPECT(successSummary.runtimeCounters.find("planned_task_count") !=
           successSummary.runtimeCounters.end(),
       "frontend summary surfaces runtime counters explicitly");
```

And add one runtime-session-oriented assertion in the shell test by checking the shared summary keys are still printed:

```bash
grep -q '^session.runtime.attribute.scheduler_policy=global_fifo_baseline$' /tmp/runtime_session_run.log
grep -q '^session.runtime.counter.scheduler.session_count=' /tmp/runtime_session_run.log
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime runtime-session -j2
./build/bin/test_taskgraph_runtime
bash test/tools/runtime/run_runtime.sh
```

Expected: FAIL because the frontend summary does not yet carry explicit runtime observability fields or runtime-session does not print them from the shared summary.

- [ ] **Step 3: Write minimal implementation**

Keep the explicit shared fields in `include/Runtime/Execution/RuntimeFrontendCore.h`:

```cpp
  std::map<std::string, std::string> runtimeAttributes;
  std::map<std::string, int64_t> runtimeCounters;
```

Populate them in `lib/Runtime/Execution/RuntimeFrontendCore.cpp`:

```cpp
  summary.runtimeAttributes = trace.attributes;
  summary.runtimeCounters = trace.counters;
```

Print only through the shared fields in `tools/runtime-session/runtime_session_main.cpp`:

```cpp
  for (const auto &[key, value] : summary.runtimeAttributes)
    llvm::outs() << "session.runtime.attribute." << key << "=" << value << "\n";
  for (const auto &[key, value] : summary.runtimeCounters)
    llvm::outs() << "session.runtime.counter." << key << "=" << value << "\n";
```

Do not re-introduce CLI-local interpretation of `ProfileTrace` internals.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target test_taskgraph_runtime runtime-session -j2
./build/bin/test_taskgraph_runtime
bash test/tools/runtime/run_runtime.sh
```

Expected: PASS with runtime-session printing the shared contract fields.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/RuntimeFrontendCore.h \
        lib/Runtime/Execution/RuntimeFrontendCore.cpp \
        tools/runtime-session/runtime_session_main.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp \
        test/tools/runtime/run_runtime.sh
git commit -m "runtime: expose scheduler observability through frontend summary"
```

## Task 5: Close Phase 1 Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-04-23-runtime-scheduler-fairness-staged-design.md`
- Modify: `docs/superpowers/plans/2026-04-23-runtime-scheduler-accounting-observability.md`

- [ ] **Step 1: Run focused verification**

Run:

```bash
cmake --build build --target AscendCRuntime runtime-session -j2
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
bash test/tools/runtime/run_runtime.sh
```

Expected:

- runtime targets build successfully
- `test_taskgraph_runtime` reports `0 failed`
- `run_runtime.sh` exits `RC=0`

- [ ] **Step 2: Update the staged design/spec with the landed Phase 1 outcome**

Append a short status note under the Phase 1 section in `docs/superpowers/specs/2026-04-23-runtime-scheduler-fairness-staged-design.md`:

```md
Phase 1 landed status:

- scheduler lifecycle counters are now exposed through a shared observability snapshot
- `ExecutionSession` merges scheduler observability into session traces
- `FrontendRunSummary` exposes runtime attributes/counters explicitly
- `runtime-session` prints scheduler observability only through the shared frontend contract
```

- [ ] **Step 3: Mark this implementation plan with final verification notes**

Add a short footer to this plan:

```md
## Verification Record

- `cmake --build build --target AscendCRuntime runtime-session -j2`
- `cmake --build build --target test_taskgraph_runtime -j2`
- `./build/bin/test_taskgraph_runtime`
- `bash test/tools/runtime/run_runtime.sh`
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-23-runtime-scheduler-fairness-staged-design.md \
        docs/superpowers/plans/2026-04-23-runtime-scheduler-accounting-observability.md
git commit -m "runtime: record phase1 scheduler verification"
```
