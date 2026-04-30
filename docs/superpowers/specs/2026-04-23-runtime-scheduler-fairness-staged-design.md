# Runtime Scheduler Fairness Staged Design

## Goal

在不引入 backend-specific policy 分叉的前提下，继续推进 `lib/Runtime` 的 scheduler/resource 能力，完成从 first-version baseline 到可验证、可演进的公平调度基础设施过渡。

本设计明确排除：

- per-backend slot policy
- 重新定义 `Simulation` / `NPU` 的独立调度框架
- 在缺少 NPU 真机的前提下做过深的 backend 行为假设

当前约束保持不变：

- `Simulation` 和 `NPU` 继续共用同一套 scheduler policy 框架
- backend 差异只通过 capability 和 resource requirement 表达
- 当前 global scheduler/resource 路径继续作为 baseline，不做架构重置

## Current State

当前 runtime 已经具备：

- `ExecutionSession -> GlobalScheduler -> ResourceScheduler -> BackendCapabilities -> Sim/NpuBackend` 主链
- 跨 session 的 global DAG admission baseline
- first-version resource reservation
- truthful `scheduler_scope=global` / `scheduler_mode=concurrent`
- session release 后的 admission capacity 恢复

但仍存在三个明显缺口：

1. 资源账虽然可用，但还没有形成更细粒度的、可审计的 accounting 语义。
2. 调度器 observability 仍偏弱，难以支撑后续 fairness / quota / priority 演进。
3. 缺少 stream-level 资源表达，也缺少跨 session 的明确公平策略。

## Non-Goals

本轮设计不包含：

- per-backend slot policy 设计与实现
- 真机 NPU policy 调优
- 彻底重写 `GlobalScheduler`
- broad scheduler optimization
- 与当前阶段无关的 `Legacy/` 清理

## Architecture Direction

整体方向分为三个阶段：

1. 先把 resource accounting 和 scheduler observability 做扎实。
2. 再把资源模型从“任务/设备”细化到“任务/stream/设备”。
3. 最后在可信资源账和可观测性之上，引入 cross-session fairness、quota 和 priority。

这三步必须保持顺序，不能反过来。原因很简单：如果没有可信 accounting 和统一 observability，后续 fairness/quota 只会变成难以验证的策略黑盒。

## Phase 1: Resource Accounting And Observability

### Objectives

- 明确 reservation 生命周期
- 让 fail / cancel / release 的资源返还行为可验证
- 暴露 scheduler 内部状态变化的统一 observability contract

### Design

`ResourceScheduler` 和 `GlobalScheduler` 需要继续保持现有抽象边界，但语义上要更清楚地区分：

- task readiness
- task admission
- resource reservation
- task running
- task completion / failure / cancellation
- session teardown

这一步不改变默认调度策略，只要求：

- 任何资源占用都能在生命周期结束时被严格返还
- 任何资源未被返还的情况都能通过 test 暴露
- scheduler 对“为什么任务没被放行”给出统一可见的解释

建议新增或统一以下 observability 面：

- runtime attributes:
  - `scheduler_scope`
  - `scheduler_mode`
  - `scheduler_policy`
  - `resource_model_version`
- runtime counters:
  - submitted / ready / reserved / running / completed / failed / cancelled
  - reservation attempts / successful reservations / blocked reservations
  - session release count
  - dependency wait count
  - resource wait count

这些字段应通过 shared frontend contract 暴露，避免 CLI 或测试绕过 frontend summary 直接解释内部 trace。

### Expected Outcome

Phase 1 完成后，scheduler 仍是当前 baseline 策略，但：

- 资源账可信
- summary/profile/CLI 对调度内部状态有统一表达
- 后续 stream/fairness 改动可以基于这些字段验证行为

## Phase 2: Stream-Level Resource Model

### Objectives

- 引入 stream 抽象
- 不改变“统一策略框架”前提下的 admission 模型
- 为后续 fairness/quota 提供更真实的资源粒度

### Design

当前资源模型更接近：

- workspace
- serialized launch lane
- device slot

Phase 2 需要在这个基础上新增 stream-level 表达，例如：

- task 是否需要 stream
- task 是否可与其他 task 共享 device 但不可共享 stream
- stream reservation 生命周期

这里的重点不是做复杂 stream scheduler，而是让 runtime 能表达：

- 同设备多执行通道
- stream 内顺序约束
- stream 间并行机会

保持约束：

- 不做 per-backend stream policy 分叉
- 不把 stream-level 设计绑死在某个 driver 行为上
- 继续让 capability/resource contract 决定是否可并发，而不是硬编码猜测

### Expected Outcome

Phase 2 完成后，resource model 会从“任务争设备”细化为“任务争设备上的执行资源”，但默认 admission policy 仍然保持统一。

## Phase 3: Fairness, Quota, And Priority

### Objectives

- 先定义 cross-session fairness baseline
- 再叠加 quota / priority
- 避免某个 session 长时间吃满全局 admission capacity

### Design

这一阶段应拆成两个子阶段：

#### Phase 3A: Cross-Session Fairness

先定义默认公平规则，例如：

- session 轮转式 admission
- 避免单 session 长时间独占 ready frontier
- 在不破坏依赖正确性的前提下支持简单 backfill

目标不是做到“最优公平”，而是先消除明显饥饿。

#### Phase 3B: Quota And Priority

在 fairness baseline 稳定之后，再增加：

- session quota
- optional priority class
- quota 与 fairness 的组合规则
- priority 与 backfill 的组合规则

这里必须坚持：

- quota / priority 是在 fairness baseline 之上的增量能力
- 默认行为应保持简单和可预测
- 不允许策略系统变成隐藏在代码常量里的黑盒

### Expected Outcome

Phase 3 完成后：

- 多 session 混跑具备明确公平语义
- scheduler 可以表达受控的不公平
- 后续真机验证时可以直接检查 quota / priority 行为是否符合预期

## Verification Strategy

验证必须随阶段同步扩展。

### Phase 1 Verification

- unit tests:
  - reservation/release
  - fail/cancel/session release resource return
  - counters/attributes surface correctness
- focused runtime tests:
  - current `test_taskgraph_runtime`
  - `runtime-session` summary assertions

### Phase 2 Verification

- stream reservation lifecycle tests
- stream contention tests
- stream-level admission order tests

### Phase 3 Verification

- multi-session fairness tests
- starvation-prevention tests
- quota enforcement tests
- priority ordering tests
- fairness + quota interaction tests

所有阶段都必须继续保持以下 baseline 为绿色：

- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_simbackend_examples.sh`
- `test/tools/examples/example_pipelines.sh`

## Risks

### Risk 1: Over-design Before Hardware Validation

在没有 NPU 真机的情况下，策略层很容易过拟合模拟环境。

缓解：

- 本轮不做 per-backend policy 分叉
- 不做 backend-specific advanced heuristics
- 先把 accounting / observability 做扎实

### Risk 2: Policy Complexity Outruns Debuggability

如果 observability 不先补，fairness / quota 会难以排查。

缓解：

- 固定先做 Phase 1
- 任何新增策略都必须同时新增 runtime counters/attributes

### Risk 3: Resource Model Drift

stream-level 设计如果过早绑定某个 backend，会导致 future refactor 成本升高。

缓解：

- stream-level contract 只表达资源语义
- backend-specific meaning 继续通过 capability/resource requirement 注入

## Recommended Execution Order

建议执行顺序如下：

1. Phase 1: resource accounting refinement
2. Phase 1: scheduler observability contract
3. Phase 2: stream-level resource model
4. Phase 3A: cross-session fairness baseline
5. Phase 3B: quota / priority
6. policy/config surface 收尾
7. verification matrix 补全

## Success Criteria

本设计视为成功的条件是：

- scheduler/resource 演进不引入 backend-specific policy 分叉
- runtime summary/profile 对 scheduler 行为具备统一可观测性
- 多 session admission 有明确 fairness baseline
- quota / priority 能以增量方式叠加在 fairness 之上
- 当前 CPU-simulation baseline 继续保持绿色
