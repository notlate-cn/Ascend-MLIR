# Autotuner Runtime Migration Design

## Goal

Replace `tools/autotuner`'s direct dependence on legacy `Compiler` / `Executor` / `SimValidator` / `HostRunnerGen` flow with a runtime-native implementation built on:

- `ArtifactCompiler`
- `ExecutionSession`
- `SimBackend`
- `ProfileTrace`

The migrated autotuner remains a single-kernel tiling search tool. It does not grow into a DAG autotuner in this round.

## Problem

`tools/autotuner/autotuner_main.cpp` is now the largest remaining top-level consumer of the old runtime-centered stack:

- It compiles kernels via `Compiler`
- Executes them via `Executor`
- Validates results via `SimValidator`
- Generates a separate runner via `HostRunnerGen`
- Uses ad hoc execution/profiling orchestration instead of the task-graph runtime

This blocks further cleanup of old runtime-centric code paths and keeps profiling/search behavior split away from the now-standard runtime execution model.

## Scope

This migration changes `autotuner` into a runtime-native CLI with a new interface.

Included:

- New autotuner CLI surface
- Runtime-native compile/run/validate/profile flow
- Search scoring based on runtime profiling outputs
- Best-config summary output

Not included:

- DAG autotuning
- NPU autotuning
- Backward-compatible legacy autotuner flags
- Host runner generation
- Separate `msprof` invocation path outside runtime

## Design Summary

### 1. CLI shape

`autotuner` becomes a single-task runtime search CLI with two entry forms:

- compile-and-search
- search-existing-artifact

Supported inputs:

- `--space <tiling_space.json>`
- one of:
  - `--kernel <step8_kernel.cpp>`
  - `--artifact-root <artifact_root>`
- `--kernel-kind <vec|cube|mix>` when compiling
- `--soc <soc_version>` when compiling
- `--inputs <comma-separated .npy files>`
- `--expected <expected_output.npy>`
- `--shape <KEY=VALUE,...>`
- optional:
  - `--atol`
  - `--rtol`
  - `--output <best_config.json>`
  - `--profile-out <dir>`

Removed from the model:

- runner generation options
- legacy perf-report / direct `msprof` shell path

### 2. Runtime-native execution model

For each candidate tiling configuration, autotuner constructs a single `RuntimeTask` and executes it through `ExecutionSession(ExecutionBackendKind::Simulation)`.

The execution request is built from:

- compiled or loaded `KernelArtifact`
- input `TensorBinding`s
- output `TensorBinding`
- `expectedOutputs`
- tiling bytes produced from the search configuration
- computed `block_dim`
- profiling enabled

This makes autotuner use the same execution path as `runtime-session`.

### 3. Compilation model

When `--kernel` is provided, autotuner uses `ArtifactCompiler`.

Compilation happens once per search run, not once per candidate, because tiling remains runtime input data rather than compiled-in specialization.

When `--artifact-root` is provided, autotuner loads the artifact through the same manifest/artifact loading utilities used by `runtime-session`.

### 4. Validation model

Correctness is defined by runtime execution success plus successful expected-output validation.

Autotuner no longer performs its own compare step outside runtime. Instead it relies on the runtime behavior already used elsewhere:

- success path emits `session.validation=pass`
- mismatch path emits `[sim:validate] ...`

Internally, autotuner should treat any candidate returning a runtime error as failed and exclude it from best-config selection.

### 5. Profiling and scoring

Autotuner no longer shells out to a separate profiler command.

Instead:

- runtime execution must first be extended to produce real simulator profile
  artifacts when `enableProfiling=true`
- `SimBackend` must surface those artifacts into `ProfileTrace`
- `ProfileTrace` exposes profile artifact paths
- autotuner parses the simulator profiling artifact and extracts a stable score

This is a required dependency, not an optional optimization. The current runtime
only carries the `enableProfiling` flag through the object model; it does not
yet guarantee real simulator profile artifact production on the sim path.

This round standardizes on a single scalar objective:

- lower is better
- if cycle count is available, use cycle count
- otherwise fail the candidate with a clear profiling parse error

This keeps scoring explicit and deterministic.

### 6. Output contract

Autotuner writes a machine-readable best-config file, for example:

```json
{
  "kernel_name": "relu_transpose_broadcast_add",
  "artifact_root": "/tmp/autotuner-artifact",
  "backend": "sim",
  "best": {
    "params": {
      "TB_M": 64,
      "TB_N": 64
    },
    "block_dim": 8,
    "score": 12345
  },
  "trials": {
    "total": 24,
    "passed": 21,
    "failed": 3
  }
}
```

The CLI may also print a concise human-readable summary, but JSON is the stable contract.

## File-Level Changes

Primary files expected to change:

- `tools/autotuner/autotuner_main.cpp`
- `tools/autotuner/CMakeLists.txt`
- `lib/Runtime/SimBackend.cpp`
- `lib/Runtime/Executor.cpp` if simulator profiling needs explicit runtime hook-up
- `include/Runtime/ArtifactCompiler.h` only if an existing helper is missing
- `include/Runtime/ProfileTrace.h` only if scoring utilities need a minimal accessor
- `lib/Runtime/ProfileUtils.*` if a reusable profile-score parser belongs there
- `test/tools/runtime/test_taskgraph_runtime.cpp` or a new focused autotuner test file
- `test/tools/runtime/run_runtime.sh` only if we add an autotuner smoke step

Potential cleanup after migration:

- remove `HostRunnerGen` dependency from autotuner
- remove `Compiler/Executor/SimValidator` direct dependency from autotuner

## Verification

Required verification after implementation:

1. Build `autotuner` on xvm
2. Run a real `SimBackend` autotune smoke case on xvm using a generated AscendC kernel
3. Confirm:
   - best-config JSON is written
   - runtime validation is enforced
   - runtime sim execution actually emits a profile artifact
   - runtime profiling artifacts are consumed for scoring
4. Re-run:
   - `bash test/tools/runtime/run_runtime.sh`
   - `bash test/tools/examples/example_pipelines.sh`

## Acceptance Criteria

The migration is complete when:

- `autotuner` no longer includes or directly orchestrates `Compiler`, `Executor`, `SimValidator`, or `HostRunnerGen`
- autotuner search runs through runtime-native execution
- candidate selection uses runtime profiling output
- a real CPU simulator autotune smoke case passes on xvm
- existing runtime and examples verification still passes
