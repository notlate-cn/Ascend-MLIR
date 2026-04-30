# Runtime Stream Resource Model Design

## Goal

在不引入 backend-specific policy 分叉的前提下，为当前 `lib/Runtime` 的 scheduler/resource baseline 增加 stream-level 资源表达。

本轮设计的目标不是实现复杂 stream scheduler，而是把资源模型从“任务/设备”细化到“任务/stream/设备”，为后续 fairness / quota / priority 奠定可信基础。

当前约束保持不变：

- `Simulation` 和 `NPU` 继续共用同一套 scheduler policy 框架
- backend 差异只通过 capability 和资源需求表达
- 当前 `GlobalScheduler -> ResourceScheduler` 架构继续作为 baseline
- 不做 per-backend policy 分叉

## Current State

当前 runtime 已经具备：

- `ExecutionSession -> GlobalScheduler -> ResourceScheduler -> BackendCapabilities` 主链
- first-version resource admission
- 显式 scheduler observability snapshot / counters / frontend summary surface
- focused xvm runtime verification baseline

当前资源模型主要表达三类约束：

- workspace bytes
- serialized launch lane
- device slot / exclusive device access

这套模型已经足够支撑 current baseline，但粒度仍然偏粗。它无法清楚表达：

- 同一设备上的多个执行通道
- “共享设备但不共享 stream”的约束
- stream 生命周期上的 reservation / release / contention

这会直接限制后续 fairness / quota / priority 的语义精度。

## Non-Goals

本轮明确不做：

- per-backend stream policy
- 绑定某个 driver/NPU 真机实现细节的 stream 数量推断
- broader scheduler optimization
- cross-session fairness / quota / priority 本身
- 重新定义 `BackendCapabilities` 的整体模型
- 架构重置或 `GlobalScheduler` 重写

## Problem Statement

如果 scheduler 只知道“任务占设备”或“任务独占设备”，那么后续策略层只能建立在过粗的 admission 粒度上：

- fairness 只能看到 session 是否占了 device slot，看不到是否只是占了一个 stream
- quota 无法表达“限制并发 stream 消费”这种更真实的资源边界
- observability 也无法解释“设备未满，但为什么任务仍未被放行”

因此，下一阶段的关键不是做更多策略，而是先把资源语言扩充到能表达 stream。

## Design Direction

整体方向是：

1. 在当前 resource model 上增加 stream-level reservation 语义
2. 保持统一 policy，不做 backend-specific policy 分叉
3. 让 `GlobalScheduler` / `ResourceScheduler` 能基于 stream 资源做 admission、release、failure cleanup
4. 把 stream observability 纳入现有 frontend/runtime summary contract

换句话说，本轮是 **resource model refinement**，不是 **policy expansion**。

## Resource Model Changes

### New Resource Concept

在现有资源模型中新增统一 stream 资源表达。最小语义应能回答两个问题：

1. 一个 task 是否需要占用 stream resource
2. 该 task 是否要求独占 stream，还是只要求某类 stream capacity

这里不要求 scheduler 理解具体 backend 的物理 stream 拓扑，只要求它理解“有限个 execution channels”这一类资源。

### Representation

建议把 stream 作为 `TaskResourceRequirements` / `ReservedResources` 的显式部分，而不是继续通过现有 `exclusiveDeviceAccess` 间接表达。

表达上保持保守：

- `requiresStream`: task 是否需要 stream admission
- `streamUnits`: task 需要占用多少个 stream 单位，baseline 默认为 `0` 或 `1`
- `exclusiveStreamAccess`: 是否要求独占 stream domain

其中：

- `streamUnits` 表示可计量的 stream capacity consumption
- `exclusiveStreamAccess` 表示不能和其他 stream consumers 并发

这样可以兼容后续扩展，而不必现在就引入复杂 stream class / stream affinity。

### Relationship With Existing Device-Level Resources

stream 资源不是 device resource 的替代品，而是补充：

- `device slot` 继续表达 device-level capacity
- `exclusiveDeviceAccess` 继续表达设备级独占
- `streamUnits` / `exclusiveStreamAccess` 表达设备内部执行通道约束

task admission 需要同时满足：

- workspace constraints
- serialized launch constraints
- device constraints
- stream constraints

因此 admission 语义会变成多维资源同时满足，而不是“有 device slot 就能跑”。

## Scheduler Semantics

### Reservation Lifecycle

stream resource 必须纳入和现有 reservation 一致的生命周期：

1. task 进入 ready frontier
2. scheduler 尝试 reserve device + stream + other resources
3. reservation 成功后 task 进入 reserved/running
4. task 成功、失败、取消、session release 时统一返还 stream resource

这里要保持和 Phase 1 accounting 一致：

- 返还必须是严格的
- 同一个 task 不能重复返还
- blocked admission 不得制造虚假占用

### Contention Semantics

在 baseline 阶段，stream contention 只需要表达两种情况：

- stream capacity 不足
- stream domain 被独占占用

不引入：

- stream affinity
- stream priority
- backend-specific stream classes

这样能确保当前实现仍保持简单、保守、可验证。

### Interaction With Existing Serialized Launch

`serialized launch` 和 `stream` 不是同一概念。

- serialized launch 表达 launch path 本身是否必须串行
- stream 表达设备执行通道是否可被多个 task 占用

因此一个 backend/task 可能同时具备：

- launch 必须串行
- 但执行 stream 可并发

或者：

- launch 可并发
- 但 stream capacity 只有 1

这两者必须在资源模型里独立存在，不能互相替代。

## ResourceScheduler Changes

`ResourceScheduler` 需要从 current baseline 的三类资源扩充到包含 stream 资源。

### Required Responsibilities

- 保存全局 stream capacity
- 判断 reservation 是否因 stream capacity 被阻塞
- 在 reservation 成功时记录 stream consumption
- 在 release/fail/cancel/session teardown 时返还 stream consumption
- 对 exclusive stream access 做冲突判断

### Blocking Explanation

为了配合已有 observability，`ResourceScheduler` 应能区分 blocked reason：

- blocked by workspace
- blocked by serialized launch
- blocked by device capacity
- blocked by stream capacity
- blocked by exclusive stream conflict

不要求这一步就做完整 rich enum surface，但至少要能让 scheduler counters/attributes 区分出 stream-blocked admission。

## GlobalScheduler Changes

`GlobalScheduler` 不需要变成 stream-aware policy engine，但要能消费更细粒度的 resource-admission 结果。

### Required Responsibilities

- 在 root admission / dependency advancement 时继续调用统一 reservation API
- 保持 current global scheduling path 不变
- 记录 stream-related blocked / reserved / released transitions
- 在 observability snapshot 中暴露 stream state

### Session Semantics

stream resource 是 global scheduler 级共享资源，不是 session-local 资源。

因此：

- 一个 session 占用 stream capacity 会影响其他 session 的 admission
- session release 必须把所有被其保留的 stream resource 一次性返还

这也是后续 fairness / quota 的前提。

## Backend Contract Changes

本轮不引入 backend-specific policy，但需要让 backend capability/resource contract 能提供 stream-level requirement。

方向上应保持保守：

- backend 可以声明 task 是否消耗 stream resource
- backend 可以声明默认 stream consumption 或 exclusive stream requirement
- scheduler 不反向猜 backend 的 stream 行为

这意味着变化主要落在 resource requirement derivation，而不是 policy branch。

## Observability

stream-level 资源一旦进入 admission，就必须进入已有 observability contract。

### Attributes

建议新增或统一以下 runtime attributes：

- `resource_model_version=v2`
- `scheduler_stream_model=enabled`

### Counters

建议新增以下 counters：

- `scheduler.stream.capacity_total`
- `scheduler.stream.capacity_available`
- `scheduler.stream.reserved`
- `scheduler.admission.stream_blocked`
- `scheduler.admission.stream_blocked_total`
- `scheduler.transition.stream_release`

这些字段应继续通过：

- `ExecutionSession` trace merge
- `RuntimeFrontendCore` shared summary
- `runtime-session` summary printing

统一暴露，而不是让 CLI 或 tests 直接解释 scheduler internals。

## Verification Strategy

### Unit-Level Verification

需要新增 focused tests 覆盖：

- stream reservation success
- stream capacity exhaustion
- exclusive stream conflict
- fail/cancel/session release 后的 stream return
- stream counters / summary surface correctness

### Runtime-Focused Verification

需要在 `test_taskgraph_runtime` 中补以下场景：

- 两个 task 可共享 device，但第二个 task 因 stream exhaustion 阻塞
- 第一个 task 完成后第二个 task 被放行
- session release 能清理已保留 stream resource

### Regression Baseline

仍必须保持以下 baseline 为绿色：

- `test/tools/runtime/run_runtime.sh`
- `test/tools/runtime/run_simbackend_examples.sh`
- `test/tools/examples/example_pipelines.sh`

## Risks

### Risk 1: Stream Model Overfitting

如果现在把 stream 设计得过细，会隐式绑定特定 backend 或 driver 行为。

缓解：

- 只实现最小 stream capacity / exclusivity 语义
- 不做 stream affinity / stream class
- 不做 per-backend stream policy

### Risk 2: Double-Counting Resources

stream 引入后，task 可能同时占 device slot 和 stream，如果 release 路径不严，会引入 accounting bug。

缓解：

- 严格复用 Phase 1 reservation lifecycle discipline
- 把 stream reservation 纳入相同 release path
- 强制补 fail/cancel/session release tests

### Risk 3: Observability Drift

如果 stream state 没同步进 frontend summary，后续 fairness 工作会再次失去证据面。

缓解：

- stream counters/attributes 必须和实现一起落
- 不接受“先实现、后补 summary”的拆分方式

## Expected Outcome

Phase 2 完成后，runtime 会获得这些新能力：

- resource model 能显式表达 stream consumption
- admission 能区分 device-blocked 和 stream-blocked
- session-level/global observability 能解释 stream contention
- 后续 fairness / quota / priority 可以建立在更真实的资源粒度之上

但本轮仍不会提供：

- per-backend stream policy
- stream priority
- quota / priority / fairness 本身
- backend-specific stream topology optimization

## Recommended Execution Order

建议实现顺序如下：

1. 扩展 resource requirement / reservation data model，纳入 stream 字段
2. 扩展 `ResourceScheduler` admission / release 逻辑
3. 扩展 `GlobalScheduler` observability snapshot
4. 扩展 `ExecutionSession` / `RuntimeFrontendCore` summary surface
5. 补 `test_taskgraph_runtime` focused cases
6. 跑 runtime-focused xvm baseline
