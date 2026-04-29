# Runtime Compat Replacement Design

**Date:** 2026-04-11

## Goal

Replace the old top-level `lib/runtime` execution model with the new `lib/Runtime` task-graph runtime while preserving existing external entrypoints during migration. The replacement must cover three surfaces:

- `examples/*/run.sh`
- CLI tools: `compiler`, `validator`, `mix-validator`
- C API: `lib/CAPI/Runtime`

The migration target is not "keep two runtime architectures forever". The target is:

- old names may remain temporarily
- old top-level behavior is reimplemented as adapters
- new `ArtifactCompiler` / `ExecutionSession` / `SimBackend` / `NpuBackend` become the only runtime center

## Non-Goals

- Removing the low-level compiler/executor code immediately
- Changing MLIR lowering or codegen stages in examples
- Designing a brand new public C API object model in this phase
- Proving NPU real-hardware execution, since hardware is unavailable

## Current State

The new runtime already provides:

- `KernelArtifact`
- `RuntimeTask`
- `TaskGraph`
- `ExecutionBackend`
- `ExecutionSession`
- `ProfileTrace`
- `runtime-session`

It also has verified `SimBackend` CPU-simulation coverage for all 6 existing example pipelines.

The remaining architectural gap is that the public/top-level entrypoints still bind directly to old runtime objects:

- `tools/compiler/compiler_main.cpp`
- `tools/validator/validator_main.cpp`
- `tools/mix-validator/mix_validator_main.cpp`
- `lib/CAPI/Runtime/Runtime.cpp`
- `examples/*/run.sh`

## Design Summary

Adopt a compatibility-shell migration:

1. Keep old entrypoint names
2. Rewrite their internals as adapters to the new runtime
3. Move examples to default to the new runtime-backed path
4. Demote the old direct execution path to internal implementation detail only

This avoids a large flag-day cutover while still making the new runtime the single architectural center.

## Architecture

### 1. Runtime Center

The only top-level execution model after this phase is:

- compile through `ArtifactCompiler`
- execute through `ExecutionSession`
- select backend through `SimBackend` or `NpuBackend`
- represent execution data through bindings/invocations/manifests

Old objects such as `Compiler`, `Executor`, and `SimValidator` may still exist temporarily, but they must no longer be the public orchestration layer.

### 2. Compatibility Adapter Layer

Introduce a thin adapter layer that maps legacy interfaces onto the new runtime model.

Required adapter responsibilities:

- normalize legacy CLI arguments into artifact compile requests or run manifests
- keep legacy exit-code conventions where practical
- preserve legacy output files when scripts depend on them
- preserve current tolerance, tiling, input/output, and block-dim semantics

The adapter layer must stay thin. It must not grow a second execution model.

### 3. Example Pipeline Split

Example pipelines continue to own:

- MLIR optimization/lowering
- CANN signature canonicalization
- AscendC source generation
- test data generation

From `step8_kernel.cpp` onward, examples should use the new runtime-backed path.

That means:

- compile stage produces a normalized artifact root
- execution stage uses runtime manifest + `ExecutionSession`
- result checking continues to validate real simulator outputs

## Replacement Plan By Surface

### CLI Replacement

#### `compiler`

`compiler` becomes a legacy front-end over `ArtifactCompiler`.

Behavioral requirements:

- continue to accept current vec/cube-focused arguments
- produce familiar output files where existing scripts expect them
- write or preserve `manifest.txt` as the source of truth
- stop owning the compile pipeline directly

If runner generation is still needed by old scripts, it becomes optional compatibility output, not the architectural center.

#### `validator`

`validator` becomes a single-task simulation adapter over the new runtime.

Behavioral requirements:

- accept current arguments such as:
  - `--bin`
  - `--name`
  - `--inputs`
  - `--expected`
  - `--tiling-*`
  - `--block-dim`
  - `--atol`
  - `--rtol`
- construct a temporary artifact-backed run manifest
- execute through `ExecutionSession` with `SimBackend`
- preserve existing result-checking behavior and dump options

The important change is internal: `validator` no longer directly orchestrates `Executor + SimValidator`.

#### `mix-validator`

`mix-validator` becomes a mix-specific manifest adapter, not an independent execution stack.

Behavioral requirements:

- continue to accept existing artifact-root/data-root style arguments
- resolve mix artifact metadata from normalized manifests
- execute through the same `ExecutionSession` model as vec/cube
- preserve current golden checking flow

This unifies vec and mix validation around the same runtime center.

### C API Replacement

The public C ABI should remain stable in this phase.

The exported names remain:

- `afirt_compiler_*`
- `afirt_executor_*`

Internally:

- compiler handles are backed by `ArtifactCompiler`-oriented state
- executor handles are backed by new-runtime execution state
- `run`/`run_file` build bindings/invocations instead of calling old orchestration paths directly

This keeps downstream native users stable while replacing the implementation core.

### Example Replacement

Examples should migrate in two waves.

#### Wave 1: vec examples

Migrate the 5 vec examples to use the new runtime-backed compile and simulation path while preserving the existing front half of the scripts.

Acceptance requirement:

- all 5 vec examples still pass real `SimBackend` CPU simulation

#### Wave 2: mix example

Migrate `examples/matmul-add-leakyrelu` to the same runtime-centered flow.

Acceptance requirement:

- the mix example still passes real `SimBackend` CPU simulation

## Error Handling

The replacement must preserve practical debuggability.

Requirements:

- compatibility CLIs must keep stable, readable error messages
- runtime-stage failures should still expose backend/stage context
- examples must fail at the same conceptual phase they do today:
  - codegen
  - compile
  - runtime execution
  - output validation

Where exact old wording is not realistic to preserve, phase clarity matters more than verbatim string compatibility.

## Testing Strategy

Validation must distinguish unit tests from simulation execution.

### Unit / focused runtime tests

Keep extending:

- `test/tools/runtime/test_taskgraph_runtime.cpp`
- other focused CLI/C API tests as needed

These validate argument adaptation, manifest construction, error routing, and compatibility behavior.

### xvm development regression

Keep:

- `test/tools/runtime/run_runtime.sh`

This remains the focused xvm regression entrypoint for runtime changes.

### SimBackend real execution baselines

This is the main acceptance gate for replacement work.

Keep and extend:

- `test/tools/runtime/run_simbackend_smoke.sh`
- `test/tools/runtime/run_simbackend_examples.sh`

Success criteria for this phase:

- all 6 examples pass through the new runtime-backed default path

### NPU validation

No real-hardware validation is required in this phase. The existing NPU mock/negative validation remains sufficient.

## Migration Stages

### Stage 1: CLI internals

Replace internals of `compiler`, `validator`, and `mix-validator` with new-runtime adapters while preserving existing command-line surfaces.

### Stage 2: examples default path

Update example scripts so their default compile/validate path uses the new runtime-backed entrypoints.

### Stage 3: C API internals

Retarget C API implementation to the new runtime while keeping the exported ABI stable.

### Stage 4: old path demotion

Once all examples and compatibility surfaces pass, explicitly demote old direct orchestration code to internal-only status. At this point it should no longer be required by any default top-level flow.

## Completion Criteria

This replacement phase is complete only when all of the following are true:

1. All 6 examples run through the new runtime-backed default path and pass real `SimBackend` CPU simulation
2. `compiler`, `validator`, and `mix-validator` are compatibility adapters over the new runtime, not independent execution centers
3. `lib/CAPI/Runtime` routes compile/run through the new runtime implementation
4. The old direct orchestration path is no longer used by default examples or default public entrypoints

## Risks

### Legacy output compatibility drift

Some scripts may depend on exact file names or side effects. This must be discovered and preserved where it affects real workflows.

### Adapter bloat

If CLI/C API shims start re-implementing runtime logic, the migration fails architecturally. The adapters must remain translation layers only.

### Mix-specific drift

The mix path has more custom artifact/data conventions than vec. That migration should happen after vec compatibility is stable, not before.

## Recommendation

Proceed with the compatibility-shell migration in this order:

1. CLI internals
2. vec examples
3. mix example
4. C API internals

This order gives the fastest proof that the new runtime can actually replace the old public workflow without forcing a high-risk flag day.
