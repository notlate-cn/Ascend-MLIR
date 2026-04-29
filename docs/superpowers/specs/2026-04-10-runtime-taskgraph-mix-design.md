# Runtime TaskGraph Mix Design

## Background

`lib/Runtime` in this repository already contains useful building blocks for AscendC compilation and execution:

- AscendC compilation through `Compiler`
- mix-oriented direct packaging through `MixDirectBackend`
- packed execution through `Executor`
- generated host runner and validator flows

However, the current center of gravity is still "single kernel tooling". The user requirement is larger:

1. Compile AscendC kernel code
2. Simulate AscendC kernel execution on CPU and collect profiling data
3. Execute on NPU
4. Leave a clean architectural path toward multi-task concurrent orchestration
5. Model mix kernels in a way that can grow into DAG-based scheduling later

The key constraint is that the next-stage runtime must support DAG task graphs rather than remain a collection of one-off compile/run helpers.

## Design Goal

Re-center `lib/Runtime` around a task-graph runtime model while preserving the current mix compilation and execution chain as backend implementation detail.

This means:

- keep current useful code paths
- stop exposing "single-kernel tooling" as the architectural center
- define stable runtime abstractions for artifacts, tasks, graphs, backends, sessions, and profiling
- support current single-task execution as a subset of task-graph execution
- leave room for future multi-task concurrent orchestration without redoing artifact formats and public APIs

## Non-Goals

This design intentionally does not attempt to:

- port the full `pypto` runtime scheduler
- reproduce `pypto` AICPU-driven dynamic machine runtime
- implement full wrap scheduling in the first version
- implement aggressive mix resource sharing in the first version
- design a new Python-first runtime layer in this phase

## External Reference Analysis

### `pyasc`

`pyasc` treats mix primarily as a compile-time and launcher-time concept.

Observed characteristics:

- mix detection is lightweight; the compiler marks kernels as mix based on IR structure
- mix semantics are expressed using AscendC/CANN kernel type selection
- execution remains launcher-centric
- testing focuses on whether launcher execution is triggered correctly

What is reusable from this approach:

- mix belongs in artifact metadata
- kernel type and resource ratio must be explicit
- generated launch artifacts are valid backend details

What is not enough for this repository:

- there is no strong DAG runtime abstraction
- it does not by itself solve future multi-task orchestration

### `pypto`

`pypto` treats mix primarily as a runtime scheduling problem.

Observed characteristics:

- mix is modeled explicitly as resource groups such as `1C1V` and `1C2V`
- scheduling binds multiple physical execution resources to one logical mix task
- task readiness, running state, pending queues, and resource pairing are runtime concerns
- profiling is runtime-native and task/core aware

What is reusable from this approach:

- resource model: mix needs first-class runtime semantics
- task metadata must include resource requirements
- profiling should be mapped back to runtime task identities

What should not be directly copied in phase one:

- the complete dynamic scheduler
- the AICPU/machine runtime coupling
- the full profiling transport and reporting stack

## Architectural Principle

The center of `lib/Runtime` must become a `TaskGraph Runtime`.

The following existing elements become backend implementation details:

- `MixDirectBackend`
- `Compiler`
- `Executor`
- packed `.so` execution
- generated host runner
- current validator binaries

Upper layers must not directly depend on:

- `packed.so`
- `aclrtlaunch_*`
- host runner file layout
- simulator library wiring

Those stay below the backend boundary.

## Recommended Architecture

### 1. Core Runtime Objects

Introduce stable runtime concepts:

- `KernelArtifact`
  - the canonical compiled kernel artifact
  - contains kernel name, kernel kind, executable forms, ABI, tiling description, workspace description, backend capabilities
- `RuntimeTask`
  - one executable node in a graph
  - references one `KernelArtifact`
  - binds actual inputs, outputs, constants, workspace policy, profiling options
- `TaskGraph`
  - explicit DAG of `RuntimeTask`
  - defines dependencies and dataflow
- `ExecutionBackend`
  - backend interface for running a task
  - hides sim vs NPU execution details
- `ExecutionSession`
  - owns graph execution state
  - performs dependency tracking, scheduling, execution, aggregation, and profiling collection
- `ProfileTrace`
  - normalized runtime profiling result associated with tasks and sessions

### 2. Mix Resource Model

Mix is modeled as runtime-visible artifact metadata, not as ad hoc backend flags.

Initial resource kinds:

- `AIVOnly`
- `AICOnly`
- `Mix1C1V`
- `Mix1C2V`

This metadata is derived from the compile result and stored in `KernelArtifact`.

The first version uses it for:

- validation
- execution backend selection
- profiling labeling
- future scheduler compatibility

The first version does not yet use it for sophisticated concurrent placement decisions.

### 3. Backends

Define backend implementations under a shared interface:

- `SimBackend`
  - responsible for simulator-based execution
  - owns environment setup, simulator library resolution, runner invocation or packed execution path, and profiling integration
- `NpuBackend`
  - responsible for real NPU execution
  - first phase focuses on correctness of code path and artifact execution wiring
  - hardware validation is allowed to remain deferred

Both backends consume the same `RuntimeTask` and `KernelArtifact` model.

### 4. Compile Path

Compilation must return `KernelArtifact`, not just a file path.

This unifies:

- vec/cube binaries
- mix packed shared objects
- ABI metadata
- tiling metadata
- runtime-manifest data

The compile path should preserve current implementation reuse:

- `Compiler` remains useful for vec/cube device compilation
- `MixDirectBackend` remains useful for mix packaging
- artifact assembly becomes a higher-level responsibility that normalizes outputs into `KernelArtifact`

### 5. Graph Execution Model

The first execution model is conservative DAG execution:

- explicit dependencies
- topological readiness tracking
- deterministic scheduling
- single-session ownership of temporary buffers and profile outputs

Phase-one scheduling policy:

- dependency-correct execution is mandatory
- serial or limited parallel execution is acceptable
- no `pypto`-style wrap scheduling is required yet

This is enough to avoid locking the system into single-task APIs while not overbuilding the scheduler in the first implementation.

## Why This Does Not Become "四不像"

This architecture only stays coherent if the public center shifts fully to task-graph objects.

It becomes incoherent if upper layers directly manipulate:

- `mix_runner`
- `packed.so`
- file-layout-specific manifests
- launcher-specific command construction

Therefore the boundary rule is:

Public runtime layers only interact with artifacts, tasks, graphs, sessions, backends, and profile traces.

All current compile/run details remain internal backend machinery.

## Data Model Sketch

### `KernelArtifact`

Responsibilities:

- identify the kernel
- describe execution form
- capture ABI
- record workspace and tiling contract
- declare mix resource type
- expose backend-ready paths

Expected fields:

- `kernelName`
- `kernelKind`
- `socVersion`
- `mixResourceType`
- `artifactRoot`
- `deviceBinaryPath` if vec/cube
- `packedSharedObjectPath` if mix
- `manifestPath`
- `abiMetadata`
- `tilingSpec`
- `workspaceSpec`
- `supportedBackends`

### `RuntimeTask`

Responsibilities:

- bind runtime values to an artifact
- define task-local execution options
- carry graph identity

Expected fields:

- `taskId`
- `artifactId` or direct artifact reference
- `inputs`
- `outputs`
- `attributes`
- `dependencyIds`
- `profileOptions`

### `TaskGraph`

Responsibilities:

- own task collection
- define DAG invariants
- support validation before execution

Expected capabilities:

- add task
- add dependency
- validate acyclic structure
- enumerate ready tasks

### `ExecutionSession`

Responsibilities:

- choose backend
- run validated graph
- manage task states
- collect results
- merge profiling outputs into one session view

Task states:

- `Pending`
- `Ready`
- `Running`
- `Succeeded`
- `Failed`

## Profiling Design

Profiling must be session- and task-oriented.

The first version should not attempt to reproduce `pypto`'s full profiling internals. Instead:

- simulator profiling is collected through `msprof op simulator` or equivalent simulator-supported output
- NPU profiling is collected through `msprof op` or equivalent real-device output
- raw backend-specific outputs are preserved
- runtime also emits normalized metadata mapping:
  - session id
  - task id
  - kernel name
  - backend
  - profile artifact locations

This gives immediate usability while keeping the door open to richer unified analysis later.

## Execution Phases

### Phase 1

Deliver:

- compile AscendC kernel into normalized `KernelArtifact`
- execute one `RuntimeTask` through `SimBackend`
- collect profiling output for simulator execution
- execute one `RuntimeTask` through `NpuBackend`
- support `TaskGraph` as public API even if scheduler is conservative

### Phase 2

Deliver:

- multiple tasks in one graph
- dependency-aware graph execution
- reusable intermediate buffers
- normalized profile aggregation across tasks

### Phase 3

Deliver:

- limited concurrent scheduling
- mix-resource-aware placement
- `Mix1C1V`/`Mix1C2V` admission control
- future path toward pypto-like orchestration without importing its full machine runtime

## File Organization Direction

The implementation should evolve toward clear separation:

- artifact modeling
- graph/task/session modeling
- backend abstraction
- sim backend
- NPU backend
- profiling normalization
- compile normalization

The current `lib/Runtime` files can be reused, but responsibility should migrate toward these units rather than continue growing a single compile/run utility surface.

## Risks

### Risk: leaking backend details upward

If task/session layers continue handling packed-library and runner-specific mechanics directly, the architecture will degrade quickly.

Mitigation:

- enforce backend boundary early
- normalize compile outputs into `KernelArtifact`

### Risk: overcommitting to scheduler complexity too early

If phase one attempts to copy `pypto` scheduling behavior, progress on the required compile/sim/NPU capabilities will likely stall.

Mitigation:

- phase one is DAG-correct, not scheduler-maximal

### Risk: profiling fragmentation

If sim and NPU profiling are exposed as unrelated outputs, users will have trouble scaling to graph execution.

Mitigation:

- define `ProfileTrace` now, even if it initially wraps backend-native outputs

## Decision

Adopt an artifact-first DAG runtime:

- preserve current compile and direct execution implementations
- make them internal backends
- define `KernelArtifact`, `RuntimeTask`, `TaskGraph`, `ExecutionBackend`, `ExecutionSession`, and `ProfileTrace` as the new architectural center
- support mix explicitly through resource metadata inspired by `pyasc` kernel typing and `pypto` mix resource modeling
- defer full pypto-style dynamic wrap scheduling to later phases

This provides a clean path from current single-kernel functionality to future multi-task concurrent orchestration without forcing a wholesale port of `pypto`.
