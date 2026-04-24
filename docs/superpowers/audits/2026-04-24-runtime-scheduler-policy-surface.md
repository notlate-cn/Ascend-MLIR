# Runtime Scheduler Policy Surface Audit

## Scope

This audit records the current implemented scheduler policy surface in
`lib/Runtime/Execution`, after the landing of:

- stream-level resource accounting
- cross-session fairness baseline
- session admission quota baseline
- static session priority baseline
- internal default-policy surface

It is a state-of-the-code audit, not a new design proposal. The goal is to
make the current scheduler baseline easier to reason about before any further
policy work.

## Implemented Baseline

The current scheduler policy stack is:

1. resource admission
2. stream resource accounting
3. cross-session round-robin fairness
4. static session priority
5. session admission quota
6. internal default policy selection

These layers all execute inside `GlobalScheduler`, with `ResourceScheduler`
remaining the shared resource arbiter.

## Contract Surface

### Global Defaults

The scheduler now has an explicit internal default-policy holder:

- `GlobalSchedulerPolicy`
- `defaultSessionScheduling`

This default policy currently feeds the `submit(...)` overloads that do not
pass explicit `SessionSchedulingOptions`.

Current default semantics:

- `priorityClass = Normal`
- `maxAdmittedTasks = 0`

`0` for `maxAdmittedTasks` still means "unlimited".

### Per-Session Policy

The currently active per-session scheduling knobs are:

- `SessionSchedulingOptions.priorityClass`
- `SessionSchedulingOptions.maxAdmittedTasks`

Current priority classes:

- `Low`
- `Normal`
- `High`

Current quota semantics:

- quota applies to `Reserved + Running`
- quota does not apply to `Ready` or `Submitted`
- a quota-exhausted session is skipped for further admission until capacity is
  returned

### Fairness Baseline

Cross-session fairness is currently defined as:

- stable session order
- round-robin admission cursor
- fairness only across sessions
- stable task ordering inside a single session

This means the scheduler currently answers:

- "which session should be considered next?"

but does not yet attempt to solve:

- weighted fairness
- dynamic aging
- multi-tenant policy composition

### Priority And Backfill

Priority currently sits above fairness:

- the highest ready-and-eligible priority class is chosen first
- round-robin only applies within that class
- a lower-priority session can still backfill if the higher-priority session is
  quota-blocked or otherwise ineligible

This is an important current property: the baseline is not "strict priority at
all costs"; it is "priority among eligible sessions".

## Explicit Non-Goals In Current Code

The current implementation does not provide:

- backend-specific scheduler policy branches
- weighted or dynamic priority
- CLI or environment-level scheduler policy configuration
- persisted scheduler policy in runtime artifacts
- tenant-wide or cluster-wide quota semantics

The current config surface is runtime-internal and test-oriented.

## Observability Surface

The shared runtime summary now exposes enough scheduler policy state to make
the baseline auditable from `runtime-session` and frontend-core consumers.

Current attributes include:

- `scheduler_policy=global_session_round_robin_baseline`
- `scheduler_fairness_policy=session_round_robin`
- `scheduler_priority_policy=static_session_priority`
- `scheduler_quota_policy=session_admission_quota`
- `scheduler_default_priority_class=<low|normal|high>`
- `scheduler_stream_model=enabled`
- `resource_model_version=v2`

Current counters include:

- `scheduler.fairness.cursor`
- `scheduler.fairness.session_order_size`
- `scheduler.fairness.session_rotations_total`
- `scheduler.fairness.session_skips_total`
- `scheduler.fairness.starvation_prevented_total`
- `scheduler.quota.blocked`
- `scheduler.quota.blocked_total`
- `scheduler.priority.high_ready`
- `scheduler.priority.normal_ready`
- `scheduler.priority.low_ready`
- `scheduler.policy.default_max_admitted_tasks`

This is enough to answer:

- which baseline policy is active
- what the internal defaults are
- whether quota is actively blocking admission
- whether fairness is actually rotating across sessions
- whether lower-priority backfill is being exercised

## Current Behavioral Boundaries

### What Is Stable

- global scheduling is real on the concurrent path
- stream accounting is part of admission
- fairness/quota/priority all participate in the same admission loop
- default session policy is explicit rather than hidden in local overload logic

### What Is Still Intentionally Conservative

- scheduler policy is still backend-agnostic
- stream usage is expressed through resource requirements, not through
  per-backend policy branches
- priority is static, not workload-adaptive
- quota is session-local and admission-scoped only

### What Still Needs Hardware Closure

- all of the above has been validated on xvm CPU-simulation baselines
- none of it should be treated as final NPU policy completion before
  real-device validation exists

## Verification State

Fresh xvm verification for the current policy surface passes:

- `./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/llvm/build`
- `bash test/tools/runtime/run_runtime.sh`
- `bash test/tools/runtime/run_simbackend_examples.sh`
- `bash test/tools/examples/example_pipelines.sh`

Focused runtime results currently include:

- `test_taskgraph_runtime`: `955 passed, 0 failed`
- `test_capi_runtime`: `15 passed, 0 failed`
- `test_runtime`: `113 passed, 0 failed`

## Recommended Next Step

Before adding more scheduler behavior, the most useful next step is a narrow
policy audit / cleanup pass in code and docs:

- keep the current baseline names and semantics stable
- avoid introducing new policy branches without matching observability
- treat any future weighted priority or richer quota work as a separate design
  stage, not as a small extension of the current baseline

That keeps the current scheduler stack understandable and preserves the value
of the verification surface that now exists.
