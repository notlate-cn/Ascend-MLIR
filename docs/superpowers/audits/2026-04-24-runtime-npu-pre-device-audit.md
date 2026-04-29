# Runtime NPU Pre-Device Audit

## Goal

Summarize what the current runtime already proves about the NPU path on xvm,
what remains intentionally unproven without hardware, and which runtime-owned
contracts are now enforced before the first real-device bring-up.

## Current Baseline

The current NPU path is split into two runtime modes:

- driver-backed `NpuBackend`: used for mock/driver tests and concurrent-session
  scheduler coverage
- bare `NpuBackend`: used for real-device execution-runner wiring and contract
  validation before hardware exists

This is enough to validate runtime wiring and request contracts on xvm, but not
enough to claim NPU completion.

## What Is Verified On xvm

Current xvm runtime verification proves:

- `NpuBackend` is a real backend in the unified runtime stack
- `ExecutionSession` can schedule NPU work through the same global scheduler
  path as simulation
- driver-backed NPU preserves injected capability contracts
- driver-backed NPU now still applies runtime-owned binding validation before
  delegating to the driver
- bare NPU request binding failures are stage-attributed as `[npu:bindings]`
- driver-backed failures are stage-attributed as `[npu:driver]`
- executor initialization failures are stage-attributed as
  `[npu:executor_initialize]`

Focused runtime results currently include:

- `test_taskgraph_runtime`: `964 passed, 0 failed`
- `test_capi_runtime`: `15 passed, 0 failed`
- `test_runtime`: `113 passed, 0 failed`

## Runtime-Owned NPU Contract

The runtime now enforces these NPU-side rules before driver/device execution:

- at least one output binding must exist
- all input/output/expected-output bindings on the NPU path must be
  `ExternalFile`
- all input/output/expected-output bindings on the NPU path must have a path
- when no expected outputs exist, output bindings must provide shape/dtype
- when expected outputs exist, output binding count must match expected-output
  count
- when expected outputs exist, provided output binding shape/dtype metadata must
  not contradict the loaded golden tensors

This contract is applied both to:

- bare `NpuBackend`
- driver-backed `NpuBackend`

That removes a previous ambiguity where mock/driver tests could accept requests
that the bare runtime path would reject.

## What Is Still Not Proven

Without real hardware, xvm still cannot prove:

- real device binary launch correctness
- real device dynamic-library launch correctness
- real device output correctness beyond runtime-owned validation
- true device concurrency semantics
- real device profile contents and performance counters
- real hardware failure taxonomy

These remain the real-device validation gap, not an architecture gap.

## Recommended Next Real-Device Entry Point

Use [docs/runtime/NPU-REAL-DEVICE-VALIDATION.md](/Volumes/GM9/code/Codex-Ascend-MLIR/docs/runtime/NPU-REAL-DEVICE-VALIDATION.md:1)
as the first bring-up checklist once hardware is available.
