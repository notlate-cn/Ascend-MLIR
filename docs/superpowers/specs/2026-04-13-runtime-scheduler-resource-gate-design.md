# Runtime Scheduler Resource Gate Design

Date: 2026-04-13

## Goal

Turn the current `ExecutionSession` scheduler skeleton into a resource-aware
serial scheduler by adding a minimal resource gate for kernel kinds and mix
resource types.

This stage does **not** introduce parallel execution. It only makes scheduling
decisions explicit and enforceable.

## Current State

`ExecutionSession` already has:

- dependency counting
- ready queue construction
- serial task dispatch
- a placeholder `canScheduleTask(const RuntimeTask &)` that always succeeds

This means the scheduler shape is present, but resource behavior is still
implicit. Nothing prevents the runtime from accepting unsupported task/resource
combinations until deep backend execution.

## Why This Stage Exists

You want the runtime to grow toward multi-task mix scheduling, but not by
rewriting `ExecutionSession` later. The next safe step is to make resource
eligibility part of the scheduler contract before introducing concurrency.

## Requirements

1. Keep execution serial.
2. Make unsupported resource requests fail early at the scheduler boundary.
3. Cover at least:
   - `vec`
   - `cube`
   - `mix`
   - `MixResourceType::Unknown`
   - supported mix resource types for this stage
4. Do not require NPU hardware.
5. Preserve current behavior for all existing 6 simulator examples.

## Chosen Scope

Stage 2 resource gate will support:

- `KernelKind::Vec`
- `KernelKind::Cube`
- `KernelKind::Mix` with:
  - `MixResourceType::Mix1C1V`
  - `MixResourceType::Mix1C2V`

It will reject:

- `KernelKind::Mix` with `MixResourceType::Unknown`
- any future unsupported mix resource type

It will not yet model:

- resource reservation across concurrent tasks
- capacity limits
- AIC/AIV occupancy accounting
- backend-specific queue partitioning

## Design

### Scheduler Gate Contract

Replace the current no-op scheduler hook with a resource eligibility check:

- input: `RuntimeTask`
- output: `llvm::Error` if the task cannot be scheduled at all

The gate is applied before backend execution, after the task becomes ready.

### Supported Resource Rules

Rules for this stage:

- `vec`: always schedulable
- `cube`: always schedulable
- `mix`:
  - schedulable only when `mixResourceType` is one of:
    - `Mix1C1V`
    - `Mix1C2V`

Fail-fast errors must include both:

- task id
- unsupported resource description

Example error shape:

- `task consumer requests unsupported mix resource type: unknown`

### Why Fail Here

This keeps resource validation at the scheduler boundary rather than leaking it
into:

- `SimBackend`
- `NpuBackend`
- example scripts

That makes later resource accounting an extension of one existing gate instead
of a cross-cutting refactor.

## Scope

### In Scope

- implement resource eligibility checks in scheduler gate
- expose clear failure strings
- add focused unit tests around supported and rejected task types
- verify existing simulator examples still pass

### Out of Scope

- parallel scheduling
- resource pool accounting
- fairness / prioritization
- dynamic capacity modeling
- DAG lane assignment

## Testing

### Focused Tests

Add tests covering:

- vec task passes gate
- cube task passes gate
- mix task with `Mix1C1V` passes gate
- mix task with `Mix1C2V` passes gate
- mix task with `Unknown` fails before backend execution

The failure test should prove the backend is not invoked.

### xvm Verification

Run:

```bash
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
```

Expected:

- all focused runtime tests pass
- all 6 example simulator pipelines still pass

## Risks

### Over-constraining Current Artifacts

If some existing mix artifacts still carry `Unknown`, this stage would reject
them too early. That is intentional: artifacts must carry usable mix metadata if
the scheduler is going to reason about them.

### False Sense of Full Scheduling

This stage does not add concurrency. It only establishes explicit resource
eligibility rules so the next scheduling stage has a stable place to grow.

## Success Criteria

This stage is complete when:

1. unsupported mix resource requests fail at scheduler gate
2. supported vec/cube/mix tasks still run
3. existing xvm simulator flows remain green
4. `ExecutionSession` is measurably more resource-aware without introducing
   parallelism
