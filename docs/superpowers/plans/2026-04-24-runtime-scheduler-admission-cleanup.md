# Runtime Scheduler Admission Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `GlobalScheduler` admission-path complexity without changing the current scheduler policy baseline.

**Architecture:** Add one more focused characterization test, then extract small admission helpers for readiness, quota eligibility, quota-blocked marking, and backend default capability construction. Keep the public contract and observability surface unchanged.

**Tech Stack:** C++17, runtime scheduler code, `test_taskgraph_runtime`, xvm shell verification, runtime docs.

---

## File Structure

- Modify: `include/Runtime/Execution/GlobalScheduler.h`
  - declare small private helper methods for admission cleanup
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`
  - centralize repeated quota/ready/default-capability logic
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`
  - add focused characterization coverage for explicit scheduling vs default policy

## Task 1: Lock The Scheduling Override Contract

**Files:**
- Modify: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the characterization test**

Add a focused test that:

- configures a default `GlobalSchedulerPolicy` with `High` priority and
  `maxAdmittedTasks = 1`
- submits one session through the default path
- submits another session with explicit `SessionSchedulingOptions`
- verifies explicit per-session scheduling still wins over the configured
  default

- [ ] **Step 2: Run the focused runtime test target**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

- [ ] **Step 3: Keep the new test as the characterization guard**

Do not widen scope; this test exists to protect the cleanup that follows.

## Task 2: Extract Admission Helpers

**Files:**
- Modify: `include/Runtime/Execution/GlobalScheduler.h`
- Modify: `lib/Runtime/Execution/GlobalScheduler.cpp`

- [ ] **Step 1: Add small helper declarations**

Add helpers for:

- checking whether a session is quota-exhausted
- checking whether a session has ready tasks
- marking ready tasks as quota-blocked
- constructing default backend capabilities

- [ ] **Step 2: Rework the admission path around those helpers**

Update:

- `tryReserveReadyTasksLocked()`
- `tryReserveOneReadyTaskForSessionLocked()`
- the default `submit(...)` overload

Keep behavior unchanged.

- [ ] **Step 3: Re-run the focused runtime test target**

Run:

```bash
cmake --build build --target test_taskgraph_runtime -j2
./build/bin/test_taskgraph_runtime
```

## Task 3: Verify Full Runtime Baseline

**Files:**
- No code changes required

- [ ] **Step 1: Run xvm focused runtime verification**

Run:

```bash
ssh xvm@orb 'cd /Volumes/GM9/code/Codex-Ascend-MLIR && export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build && bash test/tools/runtime/run_runtime.sh'
```

- [ ] **Step 2: Commit cleanup**

Suggested commits:

```bash
git add include/Runtime/Execution/GlobalScheduler.h \
        lib/Runtime/Execution/GlobalScheduler.cpp \
        test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "runtime: simplify scheduler admission path"

git add docs/superpowers/specs/2026-04-24-runtime-scheduler-admission-cleanup-design.md \
        docs/superpowers/plans/2026-04-24-runtime-scheduler-admission-cleanup.md
git commit -m "docs: add scheduler admission cleanup design"
```
