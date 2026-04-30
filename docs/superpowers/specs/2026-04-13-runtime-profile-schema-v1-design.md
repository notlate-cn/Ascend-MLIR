# Runtime Profile Schema V1 Design

## Goal

Upgrade the runtime simulator profiling artifact from a minimal score-only JSON
into a stable task-level profile schema that is still lightweight, still easy
for `runtime-session` and `autotuner` to consume, and compatible with future
DAG/session-level profiling.

This round does not introduce a full profiler event system. It standardizes the
contents of the existing per-task `trace.json`.

## Problem

The current simulator profile artifact is intentionally minimal:

```json
{
  "backend": "simulation",
  "cycle_count": 1498485,
  "elapsed": 1498485,
  "score": 1498485,
  "task_id": "main"
}
```

This is enough to unblock:

- `runtime-session` profile artifact surfacing
- `autotuner` score extraction

But it is not enough for:

- debugging runtime execution context
- explaining which kernel/config produced a score
- correlating profile output with tensor metadata
- growing into DAG/session profiling later without another incompatible rewrite

## Scope

Included:

- a richer task-level simulator profile JSON schema
- runtime emission of the richer schema from `SimBackend`
- backward-compatible score extraction for autotuner
- focused tests for required fields and compatibility

Not included:

- session summary JSON
- timeline/event-stream profiling
- NPU real-device profiling schema changes
- detailed hardware counters beyond the current scalar score
- profiling UI or visualization work

## Design Summary

### 1. Keep the current file location and artifact model

The simulator profile artifact remains a single JSON file at:

`<working-dir>/opprof/simulator/trace.json`

This preserves:

- `ProfileUtils::normalizeSimulatorProfileTrace(...)`
- `ProfileTrace::profileArtifactPaths()`
- `runtime-session` profile artifact reporting
- current test and script expectations

Only the JSON contents become richer.

### 2. Standardize a task-level schema

The profile artifact becomes a versioned task profile document:

```json
{
  "schema_version": 1,
  "backend": "simulation",
  "session_id": "runtime-session--e67db6",
  "task_id": "main",
  "kernel_name": "relu_transpose_broadcast_add",
  "kernel_kind": "vec",
  "soc_version": "Ascend910B1",
  "block_dim": 8,
  "workspace_size": 8192,
  "cycle_count": 1498485,
  "elapsed_us": 1498485,
  "score": 1498485,
  "validation_passed": true,
  "artifact_root": "/tmp/autotuner_build-00648f",
  "inputs": [
    { "name": "input0", "shape": [640, 1], "dtype": "f16" },
    { "name": "input1", "shape": [500, 640], "dtype": "f16" }
  ],
  "outputs": [
    { "name": "output", "shape": [500, 640], "dtype": "f16" }
  ],
  "tiling": {
    "present": true,
    "binary_path": "/tmp/.../tiling.bin",
    "bytes": 24
  }
}
```

### 3. Required fields

The following fields are required in schema v1:

- `schema_version`
- `backend`
- `session_id`
- `task_id`
- `kernel_name`
- `kernel_kind`
- `soc_version`
- `block_dim`
- `workspace_size`
- `cycle_count`
- `elapsed_us`
- `score`
- `validation_passed`
- `artifact_root`
- `inputs`
- `outputs`
- `tiling`

Rationale:

- these fields are available from the runtime execution request/result
- they identify the execution context well enough for tuning/debugging
- they avoid another schema churn for the next round

### 4. Tensor metadata encoding

Each tensor entry in `inputs` and `outputs` uses:

```json
{
  "name": "input0",
  "shape": [640, 1],
  "dtype": "f16"
}
```

Rules:

- `name` is required
- `shape` is always emitted
- `dtype` is always emitted using the runtime short-name style already used in
  manifests and tests (`f16`, `bf16`, `f32`, `i8`, `i32`, `i64`)

This keeps the profile readable and directly useful for debugging and tuning.

### 5. Tiling metadata encoding

Schema v1 includes a compact tiling object:

```json
{
  "present": true,
  "binary_path": "/tmp/.../tiling.bin",
  "bytes": 24
}
```

Rules:

- `present` is always emitted
- when no tiling binding exists:
  - `present=false`
  - `binary_path=""`
  - `bytes=0`
- when tiling exists:
  - `binary_path` is the runtime materialized tiling binary path
  - `bytes` is the final serialized byte size

This is enough for traceability without dumping raw tiling contents.

### 6. Score semantics

Schema v1 keeps a single scalar objective:

- `score`
- `cycle_count`
- `elapsed_us`

For simulator runtime in this round:

- all three fields may currently carry the same measured scalar value

This is acceptable for v1 because the main value is the richer execution
context, not differentiated timing sources. The schema leaves room for future
divergence without changing consumer contracts.

### 7. Validation semantics

`validation_passed` is a required boolean.

For the current runtime behavior:

- emit `true` for successful runs that also passed validation
- emit `false` only if we deliberately extend runtime to persist a profile
  artifact for failed-but-profiled runs in a future round

In this round, successful simulator profile artifacts should still correspond to
successful validated runs, so the immediate emitted value is expected to be
`true`.

The field still belongs in v1 because consumers should not infer validation
state from file existence.

### 8. Backward compatibility for consumers

`autotuner` must not assume only the new schema exists. Its score extraction
should remain backward-compatible with:

- old minimal profile JSON
- new schema v1 JSON

The parser should continue to prefer:

- explicit `score`
- otherwise `cycle_count`
- otherwise other existing score-like integer fallbacks

This avoids coupling rollout order between runtime and autotuner.

## File-Level Changes

Primary files expected to change:

- `lib/Runtime/SimBackend.cpp`
- `tools/autotuner/autotuner_main.cpp`
- `test/tools/runtime/test_taskgraph_runtime.cpp`

Optional only if useful:

- `include/Runtime/ProfileUtils.h`
- `lib/Runtime/ProfileUtils.cpp`

No schema v1 change should require modifying:

- `ProfileTrace` storage model
- `runtime-session` CLI summary format
- `ExecutionSession` scheduling model

## Verification

Required verification:

1. Build runtime targets on xvm
2. Run `bash test/tools/runtime/run_runtime.sh`
3. Inspect a real generated simulator `trace.json`
4. Run autotuner vec smoke on xvm
5. Confirm:
   - `trace.json` contains the schema v1 required fields
   - runtime-session still reports the artifact path
   - autotuner still produces non-zero `score/cycle_count`

## Acceptance Criteria

The schema upgrade is complete when:

- simulator runtime emits schema v1 task profile JSON
- `run_runtime.sh` still passes on xvm
- `autotuner` still scores candidates from runtime profile artifacts
- a real xvm `trace.json` clearly shows task metadata, tensor metadata, and
  tiling metadata in addition to score fields
