# Runtime Stream Resource Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current runtime scheduler/resource baseline so tasks can reserve stream-level execution capacity in addition to workspace, serialized launch lanes, and device slots.

**Architecture:** Keep the existing `ExecutionSession -> GlobalScheduler -> ResourceScheduler` execution path and make stream resources a new explicit part of `TaskResourceRequirement` / `ResourceReservation`. `ResourceScheduler` stays the single admission gate, `GlobalScheduler` stays the global DAG/session coordinator, and the frontend summary remains the only shared observability surface that consumers should read.

**Tech Stack:** C++17, LLVM `Error` / ADT helpers, existing runtime scheduler classes, `test_taskgraph_runtime`, `runtime-session`, xvm shell verification.

---

## File Structure

### Resource model and admission core

- Modify: `include/Runtime/Execution/ResourceScheduler.h`
  - add stream-level requirement, reservation, and blocked-reason types
- Modify: `lib/Runtime/Execution/ResourceScheduler.cpp`
  - reserve/release stream capacity and explain stream-related admission failures
- Modify: `include/Runtime/Execution/BackendCapabilities.h`
  - keep stream capacity visible to scheduler setup through the existing capability surface

### Global scheduler integration

- Modify: `include/Runtime/Execution/GlobalScheduler.h`
  - extend observability snapshot naming only if needed for new stream counters
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
  - derive stream requirements from capabilities, track stream-blocked admission, and expose stream counters/attributes

### Shared runtime summary surface

- Modify: `lib/Runtime/Execution/ExecutionSession.cpp`
  - merge new stream counters/attributes into the trace in the same place as Phase 1 counters
- Modify: `lib/Runtime/Execution/RuntimeFrontendCore.cpp`
  - keep `runtimeAttributes` / `runtimeCounters` populated without any direct scheduler knowledge in callers
- Modify: `tools/runtime-session/runtime_session_main.cpp`
  - keep printing through the shared summary contract only

### Verification

- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
  - add focused unit-style scheduler/resource tests for stream reservation and lifecycle cleanup
- Modify: `test/tools/runtime/run_runtime.sh`
  - assert the new stream summary keys on the serial path where values are stable

## Task 1: Add Stream Resource Fields To The Core Admission Types

**Files:**
- Modify: `include/Runtime/Execution/ResourceScheduler.h`
- Modify: `lib/Runtime/Execution/ResourceScheduler.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

Add this focused test near the existing `ResourceScheduler` / `GlobalScheduler` coverage in `test/tools/runtime/test_taskgraph_runtime.cpp`:

```cpp
static void testResourceSchedulerReservesStreamCapacity() {
  ResourceScheduler scheduler;
  scheduler.configureSimDispatchLanes(4);
  scheduler.configureDeviceSlots(4);
  scheduler.configureWorkspaceBudget(1024);
  scheduler.configureStreamCapacity(1);

  TaskResourceRequirement req;
  req.backendKind = ExecutionBackendKind::Simulation;
  req.workspaceBytes = 16;
  req.requiresStream = true;
  req.streamUnits = 1;

  auto first = scheduler.tryReserve("s0", "t0", req);
  EXPECT(static_cast<bool>(first), "first stream reservation succeeds");
  if (!first)
    return;
  EXPECT(first->holdsStreamSlot, "reservation records stream ownership");
  EXPECT(first->reservedStreamUnits == 1,
         "reservation records stream unit count");

  auto second = scheduler.tryReserve("s1", "t1", req);
  EXPECT(!second, "second reservation blocks when stream capacity is exhausted");

  scheduler.release(*first);
  auto third = scheduler.tryReserve("s2", "t2", req);
  EXPECT(static_cast<bool>(third),
         "stream capacity is returned after release");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: FAIL with missing `configureStreamCapacity`, missing `requiresStream` / `streamUnits`, and missing `holdsStreamSlot` fields.

- [ ] **Step 3: Write minimal implementation**

Update `include/Runtime/Execution/ResourceScheduler.h`:

```cpp
enum class ResourceBlockReason {
  None,
  Workspace,
  SerializedLaunch,
  DeviceCapacity,
  StreamCapacity,
  ExclusiveStreamConflict,
};

struct TaskResourceRequirement {
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  size_t workspaceBytes = 0;
  bool requiresSerializedLaunch = false;
  bool exclusiveDeviceAccess = false;
  bool requiresStream = false;
  size_t streamUnits = 0;
  bool exclusiveStreamAccess = false;
};

struct ResourceReservation {
  std::string sessionId;
  std::string taskId;
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  size_t workspaceBytes = 0;
  bool holdsSerializedLaunchLane = false;
  bool holdsDeviceSlot = false;
  bool holdsStreamSlot = false;
  size_t reservedStreamUnits = 0;
```

Add the new scheduler API and counters:

```cpp
class ResourceScheduler {
public:
  void configureStreamCapacity(size_t count);
  ResourceBlockReason lastBlockReason() const { return lastBlockReason_; }
```

Extend private storage:

```cpp
  struct ActiveReservation {
    std::string sessionId;
    std::string taskId;
    ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
    size_t workspaceBytes = 0;
    size_t deviceSlots = 0;
    bool holdsSerializedLaunchLane = false;
    bool holdsDeviceSlot = false;
    bool holdsStreamSlot = false;
    size_t streamUnits = 0;
    bool exclusiveStreamAccess = false;
  };

  size_t configuredStreamCapacity_ = 0;
  size_t reservedStreamUnits_ = 0;
  bool hasExclusiveStreamReservation_ = false;
  ResourceBlockReason lastBlockReason_ = ResourceBlockReason::None;
```

Implement the new logic in `lib/Runtime/Execution/ResourceScheduler.cpp`:

```cpp
void ResourceScheduler::configureStreamCapacity(size_t count) {
  configuredStreamCapacity_ = count;
}

std::optional<ResourceReservation>
ResourceScheduler::tryReserve(const std::string &sessionId,
                              const std::string &taskId,
                              const TaskResourceRequirement &requirement) {
  lastBlockReason_ = ResourceBlockReason::None;

  if (configuredWorkspaceBudget_ < reservedWorkspaceBytes_) {
    lastBlockReason_ = ResourceBlockReason::Workspace;
    return std::nullopt;
  }
  if (requirement.workspaceBytes >
      (configuredWorkspaceBudget_ - reservedWorkspaceBytes_)) {
    lastBlockReason_ = ResourceBlockReason::Workspace;
    return std::nullopt;
  }

  bool holdsSerializedLaunchLane = false;
  if (requirement.backendKind == ExecutionBackendKind::Simulation &&
      requirement.requiresSerializedLaunch) {
    if (configuredSimDispatchLanes_ <= reservedSimDispatchLanes_) {
      lastBlockReason_ = ResourceBlockReason::SerializedLaunch;
      return std::nullopt;
    }
    holdsSerializedLaunchLane = true;
  }

  bool holdsDeviceSlot = false;
  size_t deviceSlotsToReserve = 0;
  if (requirement.backendKind == ExecutionBackendKind::Npu) {
    deviceSlotsToReserve = requirement.exclusiveDeviceAccess
                               ? configuredDeviceSlots_
                               : 1;
    if (deviceSlotsToReserve == 0 ||
        configuredDeviceSlots_ < reservedDeviceSlots_ + deviceSlotsToReserve) {
      lastBlockReason_ = ResourceBlockReason::DeviceCapacity;
      return std::nullopt;
    }
    holdsDeviceSlot = true;
  }

  bool holdsStreamSlot = false;
  size_t streamUnitsToReserve = requirement.requiresStream
                                    ? std::max<size_t>(requirement.streamUnits, 1)
                                    : 0;
  if (streamUnitsToReserve > 0) {
    if (requirement.exclusiveStreamAccess &&
        (reservedStreamUnits_ > 0 || hasExclusiveStreamReservation_)) {
      lastBlockReason_ = ResourceBlockReason::ExclusiveStreamConflict;
      return std::nullopt;
    }
    if (!requirement.exclusiveStreamAccess && hasExclusiveStreamReservation_) {
      lastBlockReason_ = ResourceBlockReason::ExclusiveStreamConflict;
      return std::nullopt;
    }
    if (configuredStreamCapacity_ < reservedStreamUnits_ + streamUnitsToReserve) {
      lastBlockReason_ = ResourceBlockReason::StreamCapacity;
      return std::nullopt;
    }
    holdsStreamSlot = true;
  }

  ResourceReservation reservation;
  reservation.sessionId = sessionId;
  reservation.taskId = taskId;
  reservation.backendKind = requirement.backendKind;
  reservation.workspaceBytes = requirement.workspaceBytes;
  reservation.holdsSerializedLaunchLane = holdsSerializedLaunchLane;
  reservation.holdsDeviceSlot = holdsDeviceSlot;
  reservation.holdsStreamSlot = holdsStreamSlot;
  reservation.reservedStreamUnits = streamUnitsToReserve;
```

Record the active reservation and release it symmetrically:

```cpp
  activeReservations_.emplace(
      reservation.token_,
      ActiveReservation{reservation.sessionId,
                        reservation.taskId,
                        reservation.backendKind,
                        reservation.workspaceBytes,
                        deviceSlotsToReserve,
                        holdsSerializedLaunchLane,
                        holdsDeviceSlot,
                        holdsStreamSlot,
                        streamUnitsToReserve,
                        requirement.exclusiveStreamAccess});
  reservedWorkspaceBytes_ += requirement.workspaceBytes;
  if (holdsSerializedLaunchLane)
    ++reservedSimDispatchLanes_;
  reservedDeviceSlots_ += deviceSlotsToReserve;
  if (holdsStreamSlot) {
    reservedStreamUnits_ += streamUnitsToReserve;
    if (requirement.exclusiveStreamAccess)
      hasExclusiveStreamReservation_ = true;
  }
```

```cpp
  if (active.holdsStreamSlot) {
    reservedStreamUnits_ -= active.streamUnits;
    if (active.exclusiveStreamAccess)
      hasExclusiveStreamReservation_ = false;
  }
  lastBlockReason_ = ResourceBlockReason::None;
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: PASS with the new `ResourceScheduler` stream-capacity test succeeding.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/ResourceScheduler.h \
        lib/Runtime/Execution/ResourceScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: add stream resource reservations"
```

## Task 2: Derive Stream Requirements From Backend Capabilities In GlobalScheduler

**Files:**
- Modify: `include/Runtime/Execution/BackendCapabilities.h`
- Modify: `include/Runtime/Execution/GlobalScheduler.h`
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

Add this focused scheduler test:

```cpp
static void testGlobalSchedulerBlocksOnStreamCapacity() {
  GlobalScheduler scheduler;
  scheduler.configureResourceScheduler(/*simDispatchLanes=*/4,
                                      /*deviceSlots=*/4,
                                      /*workspaceBudget=*/4096,
                                      /*streamCapacity=*/1);

  BackendCapabilities caps;
  caps.supportsConcurrentDispatch = true;
  caps.supportsConcurrentExecution = true;
  caps.requiresSerializedLaunch = false;
  caps.maxConcurrentTasks = 4;
  caps.maxConcurrentStreams = 1;

  TaskGraph graph;
  RuntimeTask a;
  a.taskId = "a";
  RuntimeTask b;
  b.taskId = "b";
  EXPECT(!graph.addTask(a), "add task a");
  EXPECT(!graph.addTask(b), "add task b");

  auto sessionOr =
      scheduler.submit(ExecutionBackendKind::Simulation, caps, graph);
  EXPECT(static_cast<bool>(sessionOr), "submit succeeds");
  if (!sessionOr)
    return;

  auto stats = scheduler.observabilitySnapshot();
  EXPECT(stats.attributes.at("scheduler_stream_model") == "enabled",
         "stream model attribute is reported");
  EXPECT(stats.counters.at("scheduler.stream.capacity_total") == 1,
         "stream capacity total is reported");
  EXPECT(stats.counters.at("scheduler.stream.reserved") == 1,
         "one task reserves the only stream slot");
  EXPECT(stats.counters.at("scheduler.admission.stream_blocked") == 1,
         "second ready task is blocked on stream capacity");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: FAIL because `configureResourceScheduler()` does not accept a stream-capacity argument, and the snapshot has no stream counters yet.

- [ ] **Step 3: Write minimal implementation**

First extend the global scheduler configuration API in `include/Runtime/Execution/GlobalScheduler.h`:

```cpp
  void configureResourceScheduler(size_t simDispatchLanes, size_t deviceSlots,
                                  size_t workspaceBudget,
                                  size_t streamCapacity = 0);
```

In `lib/Runtime/Execution/GlobalScheduler.cpp`, wire the new argument through:

```cpp
GlobalScheduler::GlobalScheduler() {
  resourceScheduler_.configureSimDispatchLanes(1024);
  resourceScheduler_.configureDeviceSlots(1024);
  resourceScheduler_.configureWorkspaceBudget(
      std::numeric_limits<size_t>::max());
  resourceScheduler_.configureStreamCapacity(1024);
}

void GlobalScheduler::configureResourceScheduler(size_t simDispatchLanes,
                                                 size_t deviceSlots,
                                                 size_t workspaceBudget,
                                                 size_t streamCapacity) {
  std::lock_guard<std::mutex> lock(mutex_);
  resourceScheduler_.configureSimDispatchLanes(simDispatchLanes);
  resourceScheduler_.configureDeviceSlots(deviceSlots);
  resourceScheduler_.configureWorkspaceBudget(workspaceBudget);
  resourceScheduler_.configureStreamCapacity(streamCapacity);
  tryReserveReadyTasksLocked();
  schedulerCv_.notify_all();
}
```

When converting capabilities into requirements in `submit(...)`, derive stream usage conservatively:

```cpp
    record.resources.requiresStream = capabilities.maxConcurrentStreams > 0;
    record.resources.streamUnits =
        record.resources.requiresStream ? 1 : 0;
    record.resources.exclusiveStreamAccess =
        record.resources.requiresStream &&
        capabilities.maxConcurrentStreams <= 1 &&
        !capabilities.supportsConcurrentExecution;
```

Add new counters and attributes to `observabilitySnapshot()`:

```cpp
  snapshot.attributes["resource_model_version"] = "v2";
  snapshot.attributes["scheduler_stream_model"] = "enabled";
  snapshot.counters["scheduler.stream.capacity_total"] =
      static_cast<int64_t>(resourceScheduler_.configuredStreamCapacity());
  snapshot.counters["scheduler.stream.capacity_available"] =
      static_cast<int64_t>(resourceScheduler_.availableStreamCapacity());
  snapshot.counters["scheduler.stream.reserved"] =
      static_cast<int64_t>(resourceScheduler_.reservedStreamUnits());
```

Track stream-blocked admissions in `GlobalScheduler` state:

```cpp
  int64_t streamBlockedAdmissionCount_ = 0;
```

Then refine `tryReserveReadyTasksLocked()`:

```cpp
    auto reservationOr = resourceScheduler_.tryReserve(
        record.sessionId, record.taskId, record.resources);
    if (!reservationOr) {
      if (!record.waitingOnResources) {
        ++resourceBlockedAdmissionCount_;
        if (resourceScheduler_.lastBlockReason() ==
            ResourceBlockReason::StreamCapacity ||
            resourceScheduler_.lastBlockReason() ==
                ResourceBlockReason::ExclusiveStreamConflict)
          ++streamBlockedAdmissionCount_;
        record.waitingOnResources = true;
      }
      continue;
    }
```

Expose the counter:

```cpp
  int64_t streamBlocked = 0;
  for (const auto &entry : tasks_) {
    const GlobalTaskRecord &record = entry.second;
    if (record.state == GlobalTaskRecord::State::Ready &&
        record.waitingOnResources) {
      ++streamBlocked;
    }
  }
  snapshot.counters["scheduler.admission.stream_blocked"] = streamBlocked;
  snapshot.counters["scheduler.admission.stream_blocked_total"] =
      streamBlockedAdmissionCount_;
```

To support the snapshot, add the tiny query helpers to `ResourceScheduler` in `include/Runtime/Execution/ResourceScheduler.h`:

```cpp
  size_t configuredStreamCapacity() const { return configuredStreamCapacity_; }
  size_t reservedStreamUnits() const { return reservedStreamUnits_; }
  size_t availableStreamCapacity() const {
    return configuredStreamCapacity_ >= reservedStreamUnits_
               ? configuredStreamCapacity_ - reservedStreamUnits_
               : 0;
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: PASS with the stream-capacity global scheduler test succeeding.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/BackendCapabilities.h \
        include/Runtime/Execution/GlobalScheduler.h \
        include/Runtime/Execution/ResourceScheduler.h \
        lib/Runtime/Execution/GlobalScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: derive stream admission from capabilities"
```

## Task 3: Make Stream Block Reasons Stable And Lifecycle-Safe

**Files:**
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
- Modify: `lib/Runtime/Execution/ResourceScheduler.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

Add a lifecycle-focused test:

```cpp
static void testGlobalSchedulerReturnsStreamCapacityOnFailureAndRelease() {
  GlobalScheduler scheduler;
  scheduler.configureResourceScheduler(/*simDispatchLanes=*/4,
                                      /*deviceSlots=*/4,
                                      /*workspaceBudget=*/4096,
                                      /*streamCapacity=*/1);

  BackendCapabilities caps;
  caps.supportsConcurrentDispatch = true;
  caps.supportsConcurrentExecution = true;
  caps.maxConcurrentTasks = 2;
  caps.maxConcurrentStreams = 1;

  TaskGraph graph;
  RuntimeTask mainTask;
  mainTask.taskId = "main";
  EXPECT(!graph.addTask(mainTask), "add task");

  auto sessionOr =
      scheduler.submit(ExecutionBackendKind::Simulation, caps, graph);
  EXPECT(static_cast<bool>(sessionOr), "submit succeeds");
  if (!sessionOr)
    return;

  auto acquiredOr = scheduler.waitAndAcquireTask(sessionOr->sessionId());
  EXPECT(static_cast<bool>(acquiredOr) && acquiredOr->has_value(),
         "task acquires successfully");
  if (!acquiredOr || !acquiredOr->has_value())
    return;

  EXPECT(!scheduler.failTask(sessionOr->sessionId(), "main"),
         "failTask succeeds");
  scheduler.releaseSession(sessionOr->sessionId());

  auto stats = scheduler.observabilitySnapshot();
  EXPECT(stats.counters.at("scheduler.stream.reserved") == 0,
         "stream reservation count returns to zero");
  EXPECT(stats.counters.at("scheduler.stream.capacity_available") == 1,
         "stream capacity is fully returned");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: FAIL because stream counters are not being updated consistently on failure/release paths.

- [ ] **Step 3: Write minimal implementation**

Refine `GlobalTaskRecord` in `include/Runtime/Execution/GlobalScheduler.h` so the scheduler can distinguish generic resource blocking from stream blocking without reinterpreting all ready tasks:

```cpp
  ResourceBlockReason blockedReason = ResourceBlockReason::None;
```

When a reservation attempt fails in `tryReserveReadyTasksLocked()`, persist the reason:

```cpp
    if (!reservationOr) {
      record.blockedReason = resourceScheduler_.lastBlockReason();
      if (!record.waitingOnResources) {
        ++resourceBlockedAdmissionCount_;
        if (record.blockedReason == ResourceBlockReason::StreamCapacity ||
            record.blockedReason ==
                ResourceBlockReason::ExclusiveStreamConflict)
          ++streamBlockedAdmissionCount_;
        record.waitingOnResources = true;
      }
      continue;
    }

    record.blockedReason = ResourceBlockReason::None;
    record.waitingOnResources = false;
```

Clear the reason on every state transition that consumes or invalidates the wait:

```cpp
      record.state = GlobalTaskRecord::State::Running;
      record.blockedReason = ResourceBlockReason::None;
```

```cpp
    if (record.reservation) {
      resourceScheduler_.release(*record.reservation);
      record.reservation.reset();
    }
    record.blockedReason = ResourceBlockReason::None;
```

```cpp
  if (record->reservation) {
    resourceScheduler_.release(*record->reservation);
    record->reservation.reset();
  }
  record->blockedReason = ResourceBlockReason::None;
```

Use the stored blocked reason in the snapshot instead of counting every ready task as stream-blocked:

```cpp
  int64_t resourceBlocked = 0;
  int64_t streamBlocked = 0;
  for (const auto &entry : tasks_) {
    const GlobalTaskRecord &record = entry.second;
    if (!record.waitingOnResources)
      continue;
    ++resourceBlocked;
    if (record.blockedReason == ResourceBlockReason::StreamCapacity ||
        record.blockedReason == ResourceBlockReason::ExclusiveStreamConflict)
      ++streamBlocked;
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: PASS with failure/release paths returning stream capacity cleanly.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/Execution/GlobalScheduler.h \
        lib/Runtime/Execution/GlobalScheduler.cpp \
        lib/Runtime/Execution/ResourceScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: stabilize stream blocked admission accounting"
```

## Task 4: Expose Stream Observability Through The Shared Frontend Contract

**Files:**
- Modify: `lib/Runtime/Execution/ExecutionSession.cpp`
- Modify: `lib/Runtime/Execution/RuntimeFrontendCore.cpp`
- Modify: `tools/runtime-session/runtime_session_main.cpp`
- Modify: `test/tools/runtime/run_runtime.sh`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test**

Add a session-level test in `test/tools/runtime/test_taskgraph_runtime.cpp` next to the existing frontend-summary observability tests:

```cpp
static void testExecutionSessionPublishesStreamObservability() {
  auto driver = std::make_shared<ConcurrentRootOverlapBackendDriver>();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  TaskGraph graph;
  RuntimeTask a;
  a.taskId = "a";
  RuntimeTask b;
  b.taskId = "b";
  EXPECT(!graph.addTask(a), "add task a");
  EXPECT(!graph.addTask(b), "add task b");

  auto outcomeOr = session.run(graph);
  EXPECT(static_cast<bool>(outcomeOr), "session run succeeds");
  if (!outcomeOr)
    return;

  auto summary = summarizeFrontendRunSuccess(ExecutionBackendKind::Simulation,
                                             /*validationRan=*/true,
                                             outcomeOr->profileTrace);
  EXPECT(summary.runtimeAttributes.at("scheduler_stream_model") == "enabled",
         "frontend summary publishes stream model attribute");
  EXPECT(summary.runtimeCounters.count("scheduler.stream.capacity_total") == 1,
         "frontend summary publishes stream counters");
}
```

Add a shell assertion in `test/tools/runtime/run_runtime.sh` under the positive vec simulation path:

```bash
grep -q '^session.runtime.attribute.scheduler_stream_model=enabled$' \
  /tmp/runtime_session_run.log
grep -q '^session.runtime.counter.scheduler.stream.capacity_total=' \
  /tmp/runtime_session_run.log
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime runtime-session -j2
./build/bin/test_taskgraph_runtime
bash test/tools/runtime/run_runtime.sh
```

Expected: FAIL because the new stream counters/attributes are not reaching the session trace / runtime-session output yet.

- [ ] **Step 3: Write minimal implementation**

Keep the merge point centralized in `lib/Runtime/Execution/ExecutionSession.cpp`. Reuse the existing helper:

```cpp
static void
mergeSchedulerObservability(ProfileTrace &trace,
                            const SchedulerObservabilitySnapshot &snapshot) {
  for (const auto &[key, value] : snapshot.attributes)
    trace.setAttribute(key, value);
  for (const auto &[key, value] : snapshot.counters)
    trace.addCounter(key, value);
}
```

No new surface should be added to `FrontendRunSummary`; instead, ensure `RuntimeFrontendCore.cpp` keeps copying the merged trace fields into the already-existing shared maps:

```cpp
  summary.runtimeAttributes = trace.attributes();
  summary.runtimeCounters = trace.counters();
```

Keep `tools/runtime-session/runtime_session_main.cpp` printing only the shared summary fields, for example:

```cpp
for (const auto &[key, value] : summary.runtimeAttributes)
  llvm::outs() << "session.runtime.attribute." << key << "=" << value << "\n";
for (const auto &[key, value] : summary.runtimeCounters)
  llvm::outs() << "session.runtime.counter." << key << "=" << value << "\n";
```

If any direct trace reads remain in `runtime_session_main.cpp`, replace them with reads from `summary.runtimeAttributes` / `summary.runtimeCounters` in this step.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cmake --build build --target test_taskgraph_runtime runtime-session -j2
./build/bin/test_taskgraph_runtime
bash test/tools/runtime/run_runtime.sh
```

Expected: PASS with stream observability showing up through `FrontendRunSummary` and `runtime-session`.

- [ ] **Step 5: Commit**

```bash
git add lib/Runtime/Execution/ExecutionSession.cpp \
        lib/Runtime/Execution/RuntimeFrontendCore.cpp \
        tools/runtime-session/runtime_session_main.cpp \
        test/tools/runtime/run_runtime.sh \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: expose stream observability through frontend summary"
```

## Task 5: Add Focused Stream Contention Coverage And Re-Run xvm Baselines

**Files:**
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
- Modify: `docs/runtime/README.md`

- [ ] **Step 1: Write the failing test**

Add one final focused scenario that proves dependency advancement re-attempts a stream-blocked task after capacity returns:

```cpp
static void testGlobalSchedulerReleasesStreamBlockedDependent() {
  GlobalScheduler scheduler;
  scheduler.configureResourceScheduler(/*simDispatchLanes=*/4,
                                      /*deviceSlots=*/4,
                                      /*workspaceBudget=*/4096,
                                      /*streamCapacity=*/1);

  BackendCapabilities caps;
  caps.supportsConcurrentDispatch = true;
  caps.supportsConcurrentExecution = true;
  caps.maxConcurrentTasks = 3;
  caps.maxConcurrentStreams = 1;

  TaskGraph graph;
  RuntimeTask producer;
  producer.taskId = "producer";
  RuntimeTask peer;
  peer.taskId = "peer";
  RuntimeTask consumer;
  consumer.taskId = "consumer";
  consumer.dependencies = {"producer"};

  EXPECT(!graph.addTask(producer), "add producer");
  EXPECT(!graph.addTask(peer), "add peer");
  EXPECT(!graph.addTask(consumer), "add consumer");

  auto sessionOr =
      scheduler.submit(ExecutionBackendKind::Simulation, caps, graph);
  EXPECT(static_cast<bool>(sessionOr), "submit succeeds");
  if (!sessionOr)
    return;

  auto first = scheduler.waitAndAcquireTask(sessionOr->sessionId());
  EXPECT(static_cast<bool>(first) && first->has_value(),
         "first runnable task acquires");
  EXPECT(!scheduler.completeTask(sessionOr->sessionId(), first->value().taskId)
              .takeError(),
         "first task completes");

  auto stats = scheduler.observabilitySnapshot();
  EXPECT(stats.counters.at("scheduler.admission.stream_blocked_total") >= 1,
         "stream-blocked total is retained across retries");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

Expected: FAIL until the stream-blocked retry path and totals are stable.

- [ ] **Step 3: Write minimal implementation**

Adjust `tryReserveReadyTasksLocked()` and the dependency-advancement path in `lib/Runtime/Execution/GlobalScheduler.cpp` only as much as needed so that:

```cpp
    if (depRecord->remainingDependencies == 0) {
      depRecord->state = GlobalTaskRecord::State::Ready;
      depRecord->waitingOnResources = false;
      depRecord->blockedReason = ResourceBlockReason::None;
    }
```

and a task that returns from blocked-to-ready is retried through the same reservation path exactly once per scheduling pass.

Then add a short architecture note in `docs/runtime/README.md` under the scheduler/resource section:

```md
- Stream-level resource accounting is now part of the baseline resource model.
- Current policy remains backend-agnostic: stream consumption is expressed through task resource requirements, not through per-backend scheduling policy branches.
```

- [ ] **Step 4: Run verification to verify it passes**

Run local focused verification first:

```bash
cmake --build build --target test_taskgraph_runtime runtime-session -j2
./build/bin/test_taskgraph_runtime
bash test/tools/runtime/run_runtime.sh
```

Then run the documented xvm baselines:

```bash
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && ./scripts/build.sh --build-project --llvm-build-dir "$LLVM_BUILD_DIR"'
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_simbackend_examples.sh'
ssh xvm@orb 'cd /Users/niu/Code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/examples/example_pipelines.sh'
```

Expected:

- `test_taskgraph_runtime`: PASS
- `run_runtime.sh`: `RC=0`
- `run_simbackend_examples.sh`: `RC=0`
- `example_pipelines.sh`: `RC=0`

- [ ] **Step 5: Commit**

```bash
git add lib/Runtime/Execution/GlobalScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp \
        docs/runtime/README.md
git commit -m "runtime: verify stream resource model baseline"
```

## Self-Review

- Spec coverage:
  - stream resource data model: Task 1
  - global scheduler integration: Tasks 2-3
  - observability through shared summary: Task 4
  - focused verification and regression baselines: Task 5
- Placeholder scan:
  - no `TODO`/`TBD` placeholders remain
  - every task has exact files, commands, and concrete code snippets
- Type consistency:
  - `TaskResourceRequirement`, `ResourceReservation`, `ResourceBlockReason`, and `SchedulerObservabilitySnapshot` are used consistently across all tasks
