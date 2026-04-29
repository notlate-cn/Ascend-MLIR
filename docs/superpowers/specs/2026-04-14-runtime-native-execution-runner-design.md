# Runtime-Native Execution Runner Design

## Goal

Remove the remaining main-path dependence on `Legacy/Executor` by replacing the implementation behind:

- `include/Runtime/Execution/DefaultExecutionRunner.h`
- `lib/Runtime/Execution/DefaultExecutionRunner.cpp`

with a runtime-native execution substrate that preserves current runtime behavior for:

- vec/cube simulation launch
- mix packed simulation launch
- real-device path wiring

## Motivation

The previous runner-adapter cutover removed the direct `Runtime/Executor.h` seam from:

- `lib/Runtime/Execution/SimBackend.cpp`
- `lib/Runtime/Execution/NpuBackend.cpp`

That was a boundary cleanup, not an execution-substrate replacement. Today the main execution chain still reaches legacy code through:

- `DefaultExecutionRunner`
- `Legacy/Executor`

This means runtime execution still depends on:

- legacy library loading
- legacy device/simulator initialization
- legacy binary registration and launch glue
- legacy packed mix launch glue

If the goal is to shrink `Legacy/` beyond surface reclassification, this is the next real blocker. The correct next step is to move that substrate into `Execution/`, not to keep wrapping the old class.

## Non-Goals

- Do not redesign `RunArgs`.
- Do not change `runtime-session` CLI syntax or output.
- Do not change profile schema or retained-profile lifecycle.
- Do not change scheduler behavior.
- Do not remove `Legacy/Executor` in the same round unless it becomes obviously dead after the cutover.
- Do not add new NPU functionality beyond preserving current wiring and error staging.

## Approaches Considered

### 1. Replace the implementation behind `DefaultExecutionRunner` in place

Keep the current runner interface and rewrite only the implementation behind it.

Pros:
- smallest architectural blast radius
- preserves backend call sites
- directly removes the main-path legacy execution dependency

Cons:
- `DefaultExecutionRunner` may temporarily own more logic before a later split

### 2. Split immediately into `SimulationRunner` and `RealDeviceRunner`

Create two concrete runtime-native runners and make `DefaultExecutionRunner` a thin composition layer.

Pros:
- cleaner final structure
- mode boundaries are explicit from day one

Cons:
- larger interface churn
- broader verification surface for one round

### 3. Keep `Legacy/Executor` and just rename or copy it

Move the old implementation into `Execution/` with minimal semantic changes.

Pros:
- fastest on paper

Cons:
- not a real cutover
- preserves legacy assumptions and leaves cleanup debt in place

## Recommendation

Choose approach 1.

This round should perform a real substrate replacement while keeping the existing runtime-facing runner interface stable. That gives a meaningful legacy reduction without coupling the change to a larger mode-specific redesign.

## Design

### Runtime-native execution substrate

Implement the default runner entirely under `Execution/` and remove its reliance on `Legacy/Executor`.

The runtime-native implementation is responsible for:

- loading the required runtime libraries for simulation or real-device mode
- creating and tearing down device/simulator execution context
- handling temporary allocations and transfer buffers needed by `RunArgs`
- registering and launching vec/cube device binaries
- loading and launching packed mix shared objects

The runtime-facing interface does not change. `SimBackend` and `NpuBackend` continue to depend on `ExecutionRunner` and `DefaultExecutionRunner`.

### Mode handling

The current execution modes remain:

- simulation
- real device

But the runtime-native implementation should model them as runner concerns, not legacy `Executor` concerns.

The implementation should preserve current initialization semantics:

- simulation mode must continue to work with the xvm simulator environment
- real-device mode must preserve current failure staging when hardware or runtime pieces are unavailable

### Launch responsibilities

The runtime-native runner must support the same two launch paths used today:

- binary launch for vec/cube kernels
- packed shared-object launch for mix kernels

The backend-visible behavior must remain stable:

- `SimBackend` still decides whether a task launches as binary or packed mix
- `NpuBackend` still uses the same runtime task/request checks
- launch failures still surface through the current stage prefixes

### Error staging contract

The cutover must preserve these backend-facing stages:

- `[sim:executor_initialize]`
- `[sim:kernel_launch]`
- `[npu:executor_initialize]`
- `[npu:kernel_launch]`

If the new runner has finer-grained internal stages, they remain internal unless there is a clear need to surface them later.

### Legacy status after this round

The intended state after this round is:

- `SimBackend` and `NpuBackend` depend only on runtime-native execution abstractions
- `DefaultExecutionRunner` no longer depends on `Legacy/Executor`
- `Legacy/Executor` becomes either:
  - dead and removable in a follow-up round, or
  - a retained compatibility unit for explicit legacy-only tests if any still remain

That retained/removal decision is a follow-up cleanup choice, not part of this design.

## File Plan

### Modify

- `include/Runtime/Execution/DefaultExecutionRunner.h`
- `lib/Runtime/Execution/DefaultExecutionRunner.cpp`
- `lib/Runtime/CMakeLists.txt`
- `test/tools/runtime/test_taskgraph_runtime.cpp`
- `test/tools/runtime/test_runtime.cpp`

### Reference only

- `include/Runtime/Execution/ExecutionRunner.h`
- `lib/Runtime/Execution/SimBackend.cpp`
- `lib/Runtime/Execution/NpuBackend.cpp`
- `include/Runtime/Legacy/Executor.h`
- `lib/Runtime/Legacy/Executor.cpp`

## Verification

Primary verification remains xvm-focused:

- `bash test/tools/runtime/run_runtime.sh`
- `bash test/tools/runtime/run_mix_repeat.sh`
- autotuner vec smoke

Additionally, this round should explicitly prove:

- `rg -n "Runtime/Executor.h" lib/Runtime/Execution/DefaultExecutionRunner.cpp`
  returns no matches
- focused tests still cover:
  - missing packed mix library error
  - current NPU initialization error staging

## Acceptance Criteria

- `DefaultExecutionRunner` no longer includes or constructs `Legacy/Executor`
- `SimBackend` / `NpuBackend` behavior remains unchanged from the runtime API perspective
- vec/mix CPU simulation flows remain green on xvm
- repeated mix simulation baseline remains green
- real-device path still compiles and preserves current staged failure behavior
