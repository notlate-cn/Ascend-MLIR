# Runtime Global Scheduler Resource NPU Design

Date: 2026-04-17

## Goal

Turn runtime from a per-session executor into a global scheduling center that:

- accepts multiple concurrent sessions
- schedules tasks across sessions through one backend-aware scheduler
- manages shared runtime resources explicitly
- enables real `NpuBackend` multi-task execution through the same scheduler core

This stage does **not** try to solve every future policy problem. It establishes
the correct architecture so later fairness, quotas, and richer placement logic
extend one scheduler instead of fragmenting across `ExecutionSession`,
`SimBackend`, and `NpuBackend`.

## Current State

Runtime already has:

- `ExecutionSession` with DAG planning and execution
- conservative concurrent DAG dispatch for simulation
- explicit backend concurrency capability hooks
- `SimBackend` real-simulator safety through a dispatch-thread model
- `NpuBackend` runtime-native single-task execution path
- task/runtime observability through `ProfileTrace` and CLI summary fields

The current limitation is architectural:

- scheduling still lives inside one `ExecutionSession`
- resources are not modeled as first-class scheduler objects
- cross-session fairness and admission do not exist
- `NpuBackend` cannot participate in multi-task scheduling because there is no
  global coordinator

## Why This Stage Exists

The next runtime step is not “add more if-statements for NPU”. The runtime now
needs a single scheduler model that can:

- reason about many sessions at once
- separate scheduling from execution
- allocate scarce backend resources explicitly
- support both `sim` and `npu` with one contract

Without that split, multi-task NPU support would become a backend-local design,
which would block later fairness, prioritization, and resource policy work.

## Reference Systems

This design is intentionally based on established scheduler models instead of an
ad-hoc runtime-specific invention.

### Kubernetes Scheduler

Reference semantics:

- staged scheduling pipeline
- `filter -> score -> reserve -> permit -> bind`

Runtime mapping:

- `filter` -> dependency and feasibility checks
- `score` -> backend/resource placement choice
- `reserve` -> resource ticket reservation
- `permit` -> optional gang/barrier admission
- `bind` -> backend dispatch

### Ray

Reference semantics:

- one global scheduler
- explicit task resource requirements
- backend-neutral placement logic

Runtime mapping:

- one `GlobalScheduler`
- per-task `TaskResourceRequirement`
- backend/resource placement decided centrally

### Dask Distributed

Reference semantics:

- explicit DAG task state machine
- scheduler state separated from worker execution state

Runtime mapping:

- `GlobalTaskRecord` becomes the scheduler-owned task state unit
- backend execution remains outside scheduler core

### Slurm

Reference semantics:

- cross-job global queue
- resource-aware admission
- later fairness / quota / backfill extensions

Runtime mapping:

- cross-session queueing and admission are built into the scheduler shape now
- advanced queue policy is deferred, but the architecture leaves room for it

## Chosen Architecture

The runtime adopts:

- one process-global `GlobalScheduler`
- one subordinate `ResourceScheduler`
- backend-provided `BackendCapabilities`
- `ExecutionSession` as a submission/result handle, not as a scheduler

This is the chosen model because it keeps:

- scheduling centralized
- backend execution specialized
- session semantics stable
- resource policy explicit

## Core Objects

### `GlobalScheduler`

Object -> determined semantics -> concrete result

- `GlobalScheduler`
  - owns all submitted sessions and all schedulable tasks in the process
  - maintains global ready queues, in-flight state, completion state, and
    failure propagation
  - makes all scheduling decisions through dependency state and resource state
  - dispatches executable tasks to backend executors

Concrete result:

- `ExecutionSession` no longer contains the authoritative scheduler
- all backend task placement decisions become globally consistent

### `ExecutionSession`

- `ExecutionSession`
  - becomes a session-facing API object
  - submits a graph into `GlobalScheduler`
  - exposes wait/query/cancel/result/profile access
  - retains session-local summary and artifact retention responsibilities

Concrete result:

- session behavior stays user-facing
- scheduling authority moves out of the session object

### `SessionHandle`

- `SessionHandle`
  - stable identity for one submitted session
  - supports:
    - wait
    - cancel
    - query state
    - read summary/profile

Concrete result:

- runtime can support many live sessions without conflating submission with
  execution ownership

### `GlobalTaskRecord`

- `GlobalTaskRecord`
  - scheduler-owned task unit
  - contains:
    - `sessionId`
    - `taskId`
    - dependency state
    - backend selection intent
    - resource requirement
    - lifecycle state
    - timestamps / attempt / failure info

Concrete result:

- scheduler logic reasons over one uniform task object for both `sim` and `npu`

### `BackendCapabilities`

- `BackendCapabilities`
  - explicit backend capability declaration
  - replaces hardcoded scheduler assumptions
  - first version includes:
    - `supportsConcurrentDispatch`
    - `supportsConcurrentExecution`
    - `requiresSerializedLaunch`
    - `maxConcurrentTasks`
    - supported resource kinds

Concrete result:

- backend scheduling policy moves from implicit codepaths into explicit
  contracts

### `ResourceScheduler`

- `ResourceScheduler`
  - owns runtime-visible resource pools
  - grants and releases task execution tickets
  - enforces “dependency-ready + resource-ready” before dispatch

Concrete result:

- runtime stops treating every ready task as runnable

### `BackendExecutor`

- `BackendExecutor`
  - executes a task after scheduling is complete
  - does not own global queueing policy
  - does not decide cross-session ordering

Concrete result:

- `SimBackend` and `NpuBackend` remain execution components, not schedulers

## Global Scheduling Model

### Submission

Object -> determined semantics -> concrete result

- `ExecutionSession.submit(graph)`
  - normalizes one session graph into scheduler-owned task records
  - returns `SessionHandle`

- `GlobalScheduler`
  - assigns `sessionId`
  - registers dependency graph
  - publishes initial ready set into the global queue

Concrete result:

- many sessions can coexist in one scheduler without nesting schedulers

### Task Lifecycle

First-version task states:

- `Submitted`
- `WaitingDependencies`
- `Ready`
- `Reserved`
- `Dispatching`
- `Running`
- `Succeeded`
- `Failed`
- `Cancelled`

Concrete result:

- dependency resolution, resource reservation, dispatch, and backend execution
  become separately observable states

### Scheduling Pipeline

The scheduler uses a Kubernetes-style staged pipeline:

1. `preFilter`
- validate graph/task/session state
- reject malformed or impossible tasks early

2. `filter`
- check backend feasibility
- check resource feasibility
- check backend capability constraints

3. `score`
- choose among eligible backend/resource placements
- first version can stay rule-based

4. `reserve`
- acquire resource tickets
- mark task `Reserved`

5. `permit`
- optional synchronization gate for future group/gang semantics
- first version defaults to immediate permit

6. `bind`
- submit task to backend executor
- transition task to `Running`

Concrete result:

- scheduler behavior becomes explicit and extensible
- later queue policy changes do not require backend refactors

## Resource Model

### First-Version Resource Types

The first version models only runtime-visible resources that matter for `sim`
and `npu` execution:

- `SimDispatchLane`
- `DeviceSlot`
- `StreamSlot`
- `WorkspaceBudget`

### `TaskResourceRequirement`

Each task carries explicit scheduler-visible requirements:

- `backendKind`
- `workspaceBytes`
- `exclusiveDeviceAccess`
- `preferredStreamCount`
- `requiresSerializedLaunch`
- future-compatible optional placement hints

Concrete result:

- scheduler decisions stop depending on backend-local implicit assumptions

### Reservation Model

- resources are reserved before dispatch
- resources are released on completion, failure, or cancellation
- a task cannot enter backend execution without a valid ticket

Concrete result:

- overcommit becomes a scheduler policy decision instead of an accident

## Cross-Session Policy

### First-Version Policy

First version stays intentionally simple:

- one global ready queue
- FIFO within equal readiness
- resource-aware admission
- no user-visible priority classes yet
- no quota/backfill policy yet

Concrete result:

- the architecture supports future fairness work, but the first implementation
  remains debuggable

### Deferred Policy

Not in first version:

- session priority classes
- tenant quotas
- backfill
- preemption
- deadline-aware scheduling

These remain compatible future extensions because the global queue and resource
reservation model already exist.

## Sim Backend Integration

### `SimBackend`

- declares backend capabilities through `BackendCapabilities`
- continues to use dispatch-thread launch serialization where required
- scheduler sees this through:
  - `supportsConcurrentDispatch=true`
  - `supportsConcurrentExecution=false` or limited
  - `requiresSerializedLaunch=true`

Concrete result:

- scheduler can overlap prep/admission while still respecting simulator
  serialization constraints

## NPU Backend Integration

### First-Version NPU Contract

- `NpuBackend`
  - becomes schedulable through the same `GlobalScheduler`
  - declares device/stream concurrency through `BackendCapabilities`
  - executes only after explicit resource reservation

Concrete result:

- NPU multi-task support is introduced as scheduler participation, not as a
  special-case NPU-only subsystem

### `NpuExecutionContext`

The design requires explicit per-task execution context boundaries:

- device association
- stream ownership
- artifact launch context
- workspace ownership
- output lifecycle

Concrete result:

- NPU tasks can run concurrently only when scheduler-issued resources and
  execution context boundaries are compatible

### First-Version NPU Resource Policy

The initial policy stays conservative:

- start with one device
- explicit stream-slot accounting
- no hidden shared mutable launch state
- serialized or bounded stream launch depending on validated backend behavior

Concrete result:

- multi-task NPU support starts safe and measurable

## Failure, Cancellation, and Session Semantics

### Failure

- one task failure marks its task failed
- dependent tasks become unschedulable and are marked cancelled/blocked
- session result reflects partial completion plus failure cause
- global scheduler continues serving unrelated sessions

Concrete result:

- failure becomes session-local, not process-global

### Cancellation

- `SessionHandle.cancel()`
  - prevents unscheduled tasks from launching
  - requests cancellation propagation to in-flight tasks where supported

Concrete result:

- cross-session global scheduling remains compatible with explicit session
  teardown

## Observability

The global scheduler must extend the current runtime observability contract.

### Session-Level Metrics

Each session summary should eventually surface:

- scheduler mode
- queue wait time
- resource wait time
- in-flight task peak
- backend launch model
- per-session critical path estimate

### Global Metrics

The scheduler should expose:

- ready queue depth
- in-flight task count
- resource occupancy
- per-backend dispatch counts
- per-backend rejection counts

Concrete result:

- multi-session runtime behavior becomes measurable without reading internal
  scheduler code

## Scope

### In Scope

- define `GlobalScheduler`
- define `ResourceScheduler`
- define `BackendCapabilities`
- refactor `ExecutionSession` into submit/query/wait facade
- integrate `SimBackend` and `NpuBackend` into one scheduler contract
- support cross-session multi-task scheduling
- support first-version resource reservation and release
- define first-version NPU multi-task execution model

### Out of Scope

- distributed multi-process scheduler
- cross-host scheduling
- preemption
- tenant quotas
- advanced fairness heuristics
- autotuned cost-model placement
- final NPU performance tuning

## Verification

### Scheduler Verification

Add focused tests for:

- cross-session ready task ordering
- resource blocking across sessions
- release/unblock behavior
- failure isolation between sessions
- cancellation semantics

### Sim Verification

Run xvm simulation validation for:

- multiple sessions submitted concurrently
- session-local DAG plus cross-session coexistence
- scheduler/resource counters remain coherent

### NPU Verification

Run real-device validation for:

- two independent root tasks
- join task after upstream completion
- bounded stream resource contention
- one task failure without poisoning unrelated sessions

## Success Criteria

This design is complete when runtime can:

1. submit multiple sessions into one global scheduler
2. make dispatch decisions through dependency + resource state
3. run `sim` and `npu` through the same scheduling contract
4. support first-version NPU multi-task execution without backend-local
   scheduling logic
5. emit scheduler/resource observability that explains why tasks ran or waited
