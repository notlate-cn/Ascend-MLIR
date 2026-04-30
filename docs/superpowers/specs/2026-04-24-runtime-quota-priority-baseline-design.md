# Runtime Quota And Priority Baseline Design

## Goal

在现有 cross-session round-robin fairness baseline 之上，引入最小可验证的：

- session admission quota
- static session priority

本轮目标不是做复杂 policy 系统，而是让 `GlobalScheduler` 具备“受控的不公平”能力，并让这些策略有统一 observability surface。

## Current State

当前 runtime 已有：

- global scheduler baseline
- stream-level resource accounting
- cross-session session round-robin fairness
- shared frontend summary observability

仍缺：

- 限制某个 session 同时吃掉多少 admission capacity 的 quota 机制
- 在多个 session 都 ready 时，显式表达“谁应该先被考虑”的 priority 机制

## Non-Goals

本轮不做：

- dynamic priority
- tenant/cluster-wide quotas
- weighted fair sharing
- backend-specific quota policy
- policy config surface 扩展到 CLI/env/user-facing manifest

## Chosen Baseline

### Session Admission Quota

定义为：单个 session 在任意时刻最多允许占用多少个已 admission task。

这里的“已 admission task”包括：

- `Reserved`
- `Running`

不包括：

- `Ready`
- `Submitted`
- `Succeeded`
- `Failed`
- `Cancelled`

默认值：`0`，表示不限制。

### Static Session Priority

定义固定 priority class：

- `Low`
- `Normal`
- `High`

默认值：`Normal`。

调度语义：

- scheduler 先找当前 still-eligible 的最高 priority class
- 在该 priority class 内继续使用现有 round-robin fairness
- 只有当更高 priority class 没有 ready-and-eligible session 时，才考虑更低 priority class

这意味着：

- priority 覆盖跨 class 的 admission 顺序
- fairness 只负责同 class 内的默认公平

## Scheduler Changes

新增 `SessionSchedulingOptions`，由 `GlobalScheduler::submit(...)` 接收。

包含：

- `priorityClass`
- `maxAdmittedTasks`

`GlobalSessionRecord` 新增：

- `scheduling`
- `admittedTasks`

行为：

- reservation 成功时 `admittedTasks += 1`
- reservation 释放时 `admittedTasks -= 1`
- 若 `maxAdmittedTasks > 0` 且 `admittedTasks >= maxAdmittedTasks`，则该 session 在本轮 admission 中视为 quota-blocked

priority 选择逻辑：

- 先扫描 session order，找出存在 ready task 且未被 quota 拦住的最高 priority
- admission pass 仅在这一 priority class 中做 round-robin

## Observability

新增 attributes：

- `scheduler_priority_policy=static_session_priority`
- `scheduler_quota_policy=session_admission_quota`

新增 counters：

- `scheduler.quota.blocked`
- `scheduler.quota.blocked_total`
- `scheduler.priority.high_ready`
- `scheduler.priority.normal_ready`
- `scheduler.priority.low_ready`

语义：

- `quota.blocked`: 当前因 quota 被挡住的 ready task 数
- `quota.blocked_total`: 累计首次进入 quota-blocked 状态的 task 次数
- `priority.*_ready`: 当前各 priority class 中 ready-and-eligible 的 session 数

## Verification

focused tests 至少覆盖：

1. quota 限制单个 session 只能同时占一个 admission slot
2. high priority session 在 lower priority session 之前获得下一次 admission 机会
3. quota 释放后，原先 quota-blocked 的 session 会重新参与调度

xvm 基线保持：

- `./scripts/build.sh --build-project --llvm-build-dir ...`
- `bash test/tools/runtime/run_runtime.sh`
- `bash test/tools/runtime/run_simbackend_examples.sh`
- `bash test/tools/examples/example_pipelines.sh`

## Expected Outcome

完成后，runtime 会具备：

- fairness baseline 之上的最小 quota 能力
- fairness baseline 之上的最小 priority 能力
- 可供未来 policy/config surface 扩展的 scheduler contract
