# Runtime Output Comparator Extraction Design

> Date: 2026-04-14
> Status: Proposed

## Goal

Remove the direct `Legacy/SimValidator` dependency from `SimBackend` and `NpuBackend` by extracting the `CompareOnly(...)` behavior into a runtime-native output comparator shared by both backends.

## Non-Goals

- No change to `Executor` in this slice.
- No change to simulator launch, NPU launch, or backend selection behavior.
- No deletion of `Legacy/SimValidator` yet.
- No new CLI surface or manifest schema changes.

## Current Problem

The current direct legacy edges in the runtime execution layer are:

- [SimBackend.cpp](/Volumes/GM9/code/Codex-Ascend-MLIR/lib/Runtime/Execution/SimBackend.cpp)
- [NpuBackend.cpp](/Volumes/GM9/code/Codex-Ascend-MLIR/lib/Runtime/Execution/NpuBackend.cpp)

Both files still include:
- `Runtime/SimValidator.h`

And both call the same legacy compare-only path after execution succeeds:
- construct `SimValidator`
- call `CompareOnly(args, expectedOutputs, atol, rtol)`
- translate the result into runtime-stage errors

That means the new runtime backends still depend directly on a legacy validation utility even though they already own request binding, output writing, profiling, and stage-level error handling.

## Recommended Approach

Extract a runtime-native comparator into the `Execution` or `Support` portion of the runtime library and make both backends depend on it instead of `Legacy/SimValidator`.

Recommended component:

- `include/Runtime/Execution/OutputComparator.h`
- `lib/Runtime/Execution/OutputComparator.cpp`

This component should be narrow and data-oriented:
- take produced `RunArgs.outputs`
- take expected `NDArray` outputs
- apply `atol` / `rtol`
- return a small runtime result object equivalent to the current compare-only information:
  - `passed`
  - `maxAbsDiff`
  - `meanAbsDiff`
  - optional `errorMessage`

## Alternatives Considered

### 1. Leave comparison in `Legacy/SimValidator`

Rejected.

This preserves a direct legacy seam in the hottest execution code and blocks the second-round legacy cleanup from making meaningful progress.

### 2. Replace both `SimValidator` and `Executor` in one slice

Rejected for now.

That would mix a low-risk extraction with a high-risk execution refactor and make regression debugging much harder.

### 3. Duplicate compare logic directly into both backends

Rejected.

This would remove the legacy dependency but create copy-pasted comparison behavior and split tolerance semantics across files.

## Architecture

### New Comparator API

Introduce a runtime-native result type, for example:

```cpp
struct OutputComparisonResult {
  bool passed = false;
  double maxAbsDiff = 0.0;
  double meanAbsDiff = 0.0;
  std::string errorMessage;
};

llvm::Expected<OutputComparisonResult>
compareRuntimeOutputs(llvm::ArrayRef<NDArray> actual,
                      llvm::ArrayRef<NDArray> expected,
                      double atol,
                      double rtol);
```

This API should be:
- backend-agnostic
- free of `Executor`
- free of `SimValidator`
- usable by both `SimBackend` and `NpuBackend`

### Backend Integration

After extraction:

- [SimBackend.cpp](/Volumes/GM9/code/Codex-Ascend-MLIR/lib/Runtime/Execution/SimBackend.cpp)
  - removes `Runtime/SimValidator.h`
  - uses `compareRuntimeOutputs(...)`
  - preserves current stage error mapping under `[sim:validate]`

- [NpuBackend.cpp](/Volumes/GM9/code/Codex-Ascend-MLIR/lib/Runtime/Execution/NpuBackend.cpp)
  - removes `Runtime/SimValidator.h`
  - uses `compareRuntimeOutputs(...)`
  - preserves current stage error mapping under `[npu:validate]`

### Legacy Boundary After This Change

After this slice lands:

- `SimBackend` no longer directly depends on `Legacy/SimValidator`
- `NpuBackend` no longer directly depends on `Legacy/SimValidator`
- `Legacy/SimValidator` may still remain for other consumers, but it stops being a direct blocker in the runtime execution layer

This narrows the remaining direct legacy seam in execution to `Executor`.

## Behavioral Compatibility

Comparison behavior must remain compatible with the current runtime semantics:

- matching outputs still pass
- mismatching outputs still fail with stage `validate`
- `max_abs_diff` and `mean_abs_diff` remain available for formatted error messages
- tolerance behavior remains stable for existing examples and autotuner smoke

This slice must not silently change numeric comparison rules.

## Testing And Verification

The implementation plan should require:

1. focused unit coverage for the new comparator itself
2. regression coverage proving both backends now route through the runtime-native comparator path
3. xvm runtime verification:
   - `bash test/tools/runtime/run_runtime.sh`
4. xvm example simulation baseline:
   - `bash test/tools/runtime/run_simbackend_examples.sh`
5. xvm autotuner vec smoke

## Acceptance Criteria

This design is complete when all of the following are true:

- `SimBackend.cpp` no longer includes `Runtime/SimValidator.h`
- `NpuBackend.cpp` no longer includes `Runtime/SimValidator.h`
- both backends use the shared runtime comparator
- focused tests cover comparator pass/fail behavior
- xvm runtime verification passes
- xvm simulation example baseline passes
- xvm autotuner vec smoke passes

## Scope Guardrails

This slice must not be expanded to include:

- replacing `Executor`
- changing simulator or NPU launch behavior
- changing profile schema
- changing manifest format
- deleting `Legacy/SimValidator`
