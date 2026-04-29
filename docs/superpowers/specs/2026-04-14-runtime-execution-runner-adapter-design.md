# Runtime Execution Runner Adapter Design

## Goal

Remove the direct `Legacy/Executor` dependency from:
- `lib/Runtime/Execution/SimBackend.cpp`
- `lib/Runtime/Execution/NpuBackend.cpp`

without changing runtime execution semantics, CLI behavior, profiling flow, or xvm verification expectations.

## Motivation

The previous comparator extraction removed the direct `Legacy/SimValidator` seam from runtime backends. The next largest remaining execution-side legacy seam is the direct use of `Executor` from `SimBackend` and `NpuBackend`.

This seam currently blocks further shrinking of `Legacy/Executor`, because runtime backends still know:
- `BackendMode`
- `Executor` construction
- `Initialize(...)`
- `RunFile(...)`
- `RunPackedMixFile(...)`

The correct next step is not to rewrite the execution substrate. It is to insert a runtime-native adapter boundary so the backends stop depending on `Legacy/Executor` directly.

## Non-Goals

- Do not rewrite simulator or real-device launch logic.
- Do not change mix/vec/cube execution behavior.
- Do not change `runtime-session` CLI semantics.
- Do not remove `Legacy/Executor` in this round.
- Do not touch `test/tools/runtime/test_runtime.cpp` beyond minimal fallout fixes if required.

## Design

### New runtime-native boundary

Add a small execution-runner adapter under `Execution/`:

- `include/Runtime/Execution/ExecutionRunner.h`
- `include/Runtime/Execution/DefaultExecutionRunner.h`
- forwarding shims only if needed
- `lib/Runtime/Execution/DefaultExecutionRunner.cpp`

The public runtime-facing interface should express only what the backends need:

- initialize a simulation or real-device runner
- launch vec/cube binaries
- launch packed mix shared objects

It should not expose `Executor` itself.

### Minimal interface shape

The adapter contract should be intentionally narrow. At minimum:

- a runner mode enum matching runtime concerns, not legacy naming
- a small abstract runner interface
- one default implementation backed by `Legacy/Executor`

Expected responsibilities:

- `SimBackend` chooses simulation mode
- `NpuBackend` chooses real-device mode
- the default implementation maps those requests to `Executor`

### Backend cutover

After the adapter exists:

- `SimBackend.cpp` stops including `Runtime/Executor.h`
- `NpuBackend.cpp` stops including `Runtime/Executor.h`
- both include the new runtime-native runner header instead
- launch flow remains structurally identical
- stage errors remain unchanged:
  - `[sim:executor_initialize]`
  - `[sim:kernel_launch]`
  - `[npu:executor_initialize]`
  - `[npu:kernel_launch]`

### Legacy status after this round

`Legacy/Executor` remains in the tree, but only as the implementation detail behind `DefaultExecutionRunner`.

That yields the intended architectural state:

- runtime backends depend on runtime-native execution abstractions
- legacy execution code is pushed behind an adapter seam

## File Plan

### Create

- `include/Runtime/Execution/ExecutionRunner.h`
- `include/Runtime/Execution/DefaultExecutionRunner.h`
- `lib/Runtime/Execution/DefaultExecutionRunner.cpp`

### Modify

- `lib/Runtime/Execution/SimBackend.cpp`
- `lib/Runtime/Execution/NpuBackend.cpp`
- `lib/Runtime/CMakeLists.txt`
- `test/tools/runtime/test_taskgraph_runtime.cpp`

### Reference only

- `include/Runtime/Legacy/Executor.h`
- `lib/Runtime/Legacy/Executor.cpp`

## Verification

Primary verification remains xvm-focused runtime checks:

- `bash test/tools/runtime/run_runtime.sh`
- `bash test/tools/runtime/run_simbackend_smoke.sh`
- autotuner vec smoke

Additionally, the round should explicitly prove:

- `rg -n "Runtime/Executor.h" lib/Runtime/Execution/SimBackend.cpp lib/Runtime/Execution/NpuBackend.cpp`
  returns no matches

## Acceptance Criteria

- `SimBackend` and `NpuBackend` no longer directly include or construct `Executor`
- execution behavior remains unchanged for vec and mix CPU simulation flows
- NPU path wiring still compiles and preserves existing error staging
- focused xvm runtime verification stays green

