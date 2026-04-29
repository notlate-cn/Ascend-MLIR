# Runtime Cross-Session Fairness Baseline Design

## Goal

在当前 `ExecutionSession -> GlobalScheduler -> ResourceScheduler` 主链不变的前提下，为 `GlobalScheduler` 增加一个可验证的 cross-session fairness baseline。

本轮目标不是引入 quota / priority，而是先消除当前按全局 key 顺序扫描 ready task 带来的明显偏置，让多个 session 在共享 admission capacity 时具备稳定、可解释的默认公平语义。

## Current State

当前调度器已经具备：

- 真实的 global DAG admission 路径
- session 级 reservation / release 生命周期
- stream-level resource accounting
- shared frontend summary observability

但 ready task 的 admission 顺序仍然基本取决于 `tasks_` 的全局 key 顺序。这会带来一个直接问题：

- 如果一个较早提交的 session 持续拥有 ready frontier，它可能在资源一旦归还时再次优先被 admission
- 后提交但同样 ready 的 session 会反复留在 `Ready` 状态
- 当前行为虽然正确，但默认公平语义过弱，不适合作为后续 quota / priority 的基础

## Non-Goals

本轮明确不做：

- quota
- priority class
- weighted fairness
- backend-specific fairness policy
- stream-specific priority policy
- broader scheduler optimization

也就是说，本轮只定义一个简单、稳定、可验证的 fairness baseline。

## Chosen Policy

选择 `session round-robin baseline`。

具体语义：

- 调度器维护一个稳定的 session 顺序
- 每次 admission pass 从 fairness cursor 指向的 session 开始
- 在单轮 pass 中，每个 session 最多成功 admission 一个 task
- 如果资源仍有余量，调度器继续下一轮 pass，并从上次成功 admission 的下一个 session 开始
- session 内部多个 ready task 仍按现有稳定顺序挑选；本轮不改变单 session 内部 policy

这样可以保证：

- 不会因为一个 session 有更多 ready roots 就在单轮 admission 中吃满全部容量
- 资源归还后，下一个更早应该被考虑的是“下一个 session”，不是“同一个 session 的下一个 task”
- 现有 resource model 和 stream model 都保持不变，只改变 ready task 的尝试顺序

## Why This Policy

相对 age-based 或 deficit-based 方案，round-robin baseline 更适合当前阶段：

- 实现面小
- 调试面清晰
- 不需要先定义 quota / credit / weight 语义
- 与当前 baseline 差异足够小，适合先在 xvm 上收敛

它不是最终 fairness 设计，但适合作为后续 quota / priority 的稳定底座。

## Scheduler Changes

`GlobalScheduler` 需要新增一组 fairness state：

- stable session order
- fairness cursor
- last admitted session
- fairness observability counters

核心变化只发生在 `tryReserveReadyTasksLocked()` 的 ready-task 选择逻辑：

1. 不再直接按全局 `tasks_` 顺序扫全部 ready task
2. 改为按 session 轮转
3. 每个 session 在单轮 pass 中至多成功 reservation 一个 task
4. 若一个 session 的第一个 ready task 资源不足，仍允许继续尝试该 session 后续 ready task，避免 fairness baseline 退化成“被单个大 task 卡死”

这保持了两条原则：

- fairness baseline 不应破坏现有 resource backfill 能力
- fairness baseline 只决定“谁先被尝试”，不重写 resource admission contract

## Observability

本轮需要把 fairness 行为纳入现有 shared runtime summary surface。

建议暴露：

- attributes
  - `scheduler_policy=global_session_round_robin_baseline`
  - `scheduler_fairness_policy=session_round_robin`
- counters
  - `scheduler.fairness.cursor`
  - `scheduler.fairness.session_order_size`
  - `scheduler.fairness.session_rotations_total`
  - `scheduler.fairness.session_skips_total`
  - `scheduler.fairness.starvation_prevented_total`

字段语义：

- `cursor`: 下一轮 admission pass 的起始 session 下标
- `session_order_size`: 当前参与 fairness 轮转的 session 数
- `session_rotations_total`: 因成功 admission 而推进 fairness cursor 的累计次数
- `session_skips_total`: 某个 session 在有 ready task 的情况下，本轮未成功 admission 的累计次数
- `starvation_prevented_total`: 与上一次成功 admission 相比，调度器成功切换到另一个 session 的累计次数

这些字段不是完整 fairness telemetry，但足够支撑 baseline 验证和后续 quota / priority 扩展。

## Verification

focused tests 至少覆盖：

1. 单 lane / 单 stream 下，`session A` 有多个 ready task，`session B` 有一个 ready task 时，资源归还后应先放行 `session B`
2. 多 session 混跑时，capacity 足够支持多任务 admission，单轮 pass 不应被一个 session 独占
3. `releaseSession()` / `failTask()` 后，fairness cursor 和 session order 仍保持一致，不出现悬空状态

runtime verification 继续以这三条 xvm baseline 为准：

- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_simbackend_examples.sh`
- `test/tools/examples/example_pipelines.sh`

## Risks

### Risk 1: Fairness Baseline Changes Existing Admission Order

这是预期变化，但必须限制在 cross-session 范围，不能打破单 session 内部语义和依赖正确性。

缓解：

- session 内部仍保留稳定顺序
- 只改变跨 session 的尝试顺序
- focused tests 明确覆盖 mixed-session contention

### Risk 2: Fairness Observability Becomes Misleading

如果 counters 与实际 cursor/session order 漂移，后续 quota / priority 会建立在错误证据上。

缓解：

- fairness state 只在 `GlobalScheduler` 内维护
- session submit / release / admission success 时同步更新
- focused tests 检查 counters 和行为一致

## Expected Outcome

完成后，runtime 会具备：

- 一个简单但真实的 cross-session fairness baseline
- 不依赖 backend-specific policy 的共享 admission 规则
- 可供 quota / priority 增量叠加的 observability surface

后续阶段再做：

- quota
- priority
- fairness + quota interaction
- more advanced backfill heuristics
