# Runtime Session Profile Summary Design

## Goal

Extend runtime profiling from a single task-level `trace.json` into a stable
session-level profiling contract that remains lightweight, remains compatible
with the current schema v1 task trace, and becomes the canonical profiling
surface for downstream tools such as `autotuner`.

This round does not introduce timeline events, per-core hardware counters, or a
visualization layer. It standardizes session/task summary outputs and makes
`autotuner` consume them instead of inventing a parallel profiling summary.

## Problem

Current runtime profiling is task-local and minimal in structure:

- a retained per-task JSON artifact exists
- `runtime-session` surfaces task profile paths
- `autotuner` reads per-task score/cycle data

This is enough for basic scoring, but it is not a stable session-level
contract. It leaves gaps in three places:

- no canonical `session_summary.json`
- no stable mapping from session to task profile artifacts beyond ad hoc path
  enumeration
- `autotuner` still has to infer run-level meaning from individual task traces

This becomes more problematic as runtime moves toward DAG execution and richer
scheduler behavior. Session-level profiling structure should come from runtime,
not be recreated by each consumer.

## Scope

Included:

- runtime emission of `session_summary.json`
- stable retained profile layout for session + task artifacts
- task profile artifact relocation into a task-oriented directory structure
- `runtime-session` surfacing of session summary path
- `autotuner` consumption of runtime session/task profile outputs
- focused tests and xvm verification

Not included:

- timeline/event-stream profiling
- NPU real-device profile schema changes
- profile UI or reporting dashboards
- per-core/per-stage simulator counters
- DAG critical-path analysis beyond simple aggregate totals

## Design Summary

### 1. Retained profile layout becomes session-oriented

The retained profile root remains:

`/tmp/ascendc-runtime-profiles/<session_id>/`

Inside that directory, runtime writes:

- `session_summary.json`
- `tasks/<task_id>.json`

The current task-level JSON schema v1 stays valid. The change is that retained
profiles now use stable names and directories instead of anonymous `main-0.json`
style files.

Example:

```text
/tmp/ascendc-runtime-profiles/runtime-session--abcd12/
  session_summary.json
  tasks/
    main.json
    producer.json
    consumer.json
```

This layout is deterministic, human-readable, and fits both single-task and DAG
sessions.

### 2. Keep task profile schema v1 as the task contract

Each task profile file remains a schema v1 task profile document. No breaking
field rename is introduced in this round.

That means downstream code can continue to rely on:

- `schema_version`
- `task_id`
- `score`
- `cycle_count`
- `inputs`
- `outputs`
- `tiling`

Only the retained file naming/layout changes.

### 3. Add `session_summary.json` as the canonical run-level contract

Runtime emits a new session summary file:

```json
{
  "schema_version": 1,
  "session_id": "runtime-session--abcd12",
  "backend": "simulation",
  "task_count": 2,
  "successful_task_count": 2,
  "failed_task_count": 0,
  "tasks": [
    {
      "task_id": "producer",
      "profile_path": "/tmp/ascendc-runtime-profiles/runtime-session--abcd12/tasks/producer.json",
      "score": 100,
      "cycle_count": 100
    },
    {
      "task_id": "consumer",
      "profile_path": "/tmp/ascendc-runtime-profiles/runtime-session--abcd12/tasks/consumer.json",
      "score": 120,
      "cycle_count": 120
    }
  ],
  "total_score": 220,
  "total_cycle_count": 220
}
```

Required fields:

- `schema_version`
- `session_id`
- `backend`
- `task_count`
- `successful_task_count`
- `failed_task_count`
- `tasks`
- `total_score`
- `total_cycle_count`

Rules:

- `tasks` preserves runtime execution order
- each task entry includes `task_id`, `profile_path`, `score`, `cycle_count`
- `total_score` and `total_cycle_count` are simple sums across successful task
  profiles in this round
- failed task counts are reserved for future extension; successful retained sim
  runs in this round should still normally produce zero failed tasks

### 4. Runtime-session surfaces both session and task profile paths

`runtime-session --run` keeps its current task profile output lines:

- `session.profile.count=...`
- `session.profile[i]=...`

It additionally prints:

- `session.profile.summary=<path-to-session_summary.json>`

This preserves existing script compatibility and adds a stable run-level handle.

### 5. Profile retention and pruning remain session-based

Existing retention behavior remains:

- retain profiles under `/tmp/ascendc-runtime-profiles`
- prune to recent session directories

The new files simply live inside each retained session directory. No change to
retention policy is required in this round.

### 6. Autotuner consumes runtime session/task profiling instead of inventing its own

`autotuner` should keep its current best-config JSON, but it must source
profiling context from runtime outputs.

Enhancements:

- record the chosen task profile path
- record the session summary path
- continue to use task-level `score` / `cycle_count` as the ranking value
- expose trial totals and failed-candidate reasons without inventing a new
  profile schema

Example best-config excerpt:

```json
{
  "kernel_name": "relu_transpose_broadcast_add",
  "backend": "sim",
  "best": {
    "params": {
      "TB_M": 64,
      "TB_N": 64
    },
    "block_dim": 8,
    "score": 12345,
    "cycle_count": 12345,
    "profile_path": "/tmp/ascendc-runtime-profiles/runtime-session--abcd12/tasks/main.json",
    "session_summary_path": "/tmp/ascendc-runtime-profiles/runtime-session--abcd12/session_summary.json"
  },
  "trials": {
    "total": 24,
    "passed": 21,
    "failed": 3
  }
}
```

This keeps autotuner thin: runtime owns profiling structure; autotuner only
points at the relevant artifacts and reports search outcomes.

## File-Level Changes

Primary files expected to change:

- `include/Runtime/ProfileTrace.h`
- `lib/Runtime/ProfileTrace.cpp` if helper logic belongs there
- `lib/Runtime/ProfileUtils.cpp`
- `tools/runtime-session/runtime_session_main.cpp`
- `tools/autotuner/autotuner_main.cpp`
- `test/tools/runtime/test_taskgraph_runtime.cpp`
- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_simbackend_examples.sh` only if summary extraction
  should surface session summary paths

## Verification

Required verification after implementation:

1. xvm: `bash test/tools/runtime/run_runtime.sh`
2. xvm: `bash test/tools/runtime/run_simbackend_examples.sh`
3. xvm: autotuner vec smoke
4. Confirm:
   - runtime emits `session_summary.json`
   - retained task profiles live under `tasks/<task_id>.json`
   - `runtime-session` prints `session.profile.summary=...`
   - autotuner best-config JSON references both task profile and session
     summary paths
   - existing vec/mix sim smoke still passes

## Acceptance Criteria

This round is complete when:

- runtime retained profiles include `session_summary.json`
- retained task profiles use stable task-based paths
- `runtime-session` exposes the session summary path
- `autotuner` consumes runtime profile outputs and records the referenced paths
- xvm runtime and autotuner smoke verification passes without regressing the
  existing profile schema v1 task files
