## 5. 第四层：Realize

第四层将第三层确定的 schedule 结果落实为显式 buffer 语义和内存实现。本层不重新搜索 tile、reorder、pipeline 或 cache 策略，只消费第三层已确定的结构事实，并将其写回 MLIR。

**三条架构边界：**
- IR carrier 只使用已有 dialect、`memref.memory_space`、`memref.copy` 和 attributes
- on-chip place 通过 `memory_space` 表达，跨 place movement 通过 `memref.copy` 表达
- 对内允许持续增强本地 analysis、planner、cost model 和 verifier

### 5.1 整体流水线与核心类

```
Bufferization → Placement → Static Memory Planning → Data Movement → Materialization
```

| 类 | 职责 | 核心方法 |
|---|---|---|
| `BufferizationDriver` | 复用 `one-shot-bufferize`，并在 bufferized IR 上补做事实收集与一致性检查 | `runOneShotBufferize()`、`collectBufferFacts()`、`verifyPostBufferization()` |
| `PlacementPlanner` | 为每个 buffer 选择 memory place | `buildCandidatePlaces()`、`filterIllegalPlaces()`、`selectPlaces()` |
| `StaticMemoryPlanner` | 计算 live range、buffer reuse 和 workspace packing | `buildLiveIntervals()`、`planWorkspaceSlots()`、`verifyCapacity()` |
| `MovementPlanner` | 为跨 place producer-consumer 关系生成 movement 计划 | `buildMovements()`、`selectPath()`、`eliminateRedundantCopies()` |
| `MemoryRealizationDriver` | 把 placement / workspace / movement 结果写回普通 MLIR | `materializeAlloc()`、`materializeWorkspace()`、`materializeCopies()`、`verify()` |

这些类均为第四层内部的 planning / realization 组件，不是新的 IR 层，实现上可以先以 pass 内局部类或 analysis helper 形式存在。

### 阶段间数据流

| 阶段 | 产出 | 消费方 | 消费字段 |
|---|---|---|---|
| Bufferization | `BufferizedKernelIR` | Placement | `bufferValues`、`aliasInfo`、`guardBindings`、`bufferRoles` |
| Bufferization | `BufferizedKernelIR` | Static Memory Planning | `loopScopes`、`readPoints`、`writePoints` |
| Bufferization | `BufferizedKernelIR` | Data Movement | `aliasInfo`、`readPoints`、`writePoints` |
| Placement | `PlacementPlan` | Static Memory Planning | `selectedPlace`（按 place 分组 local buffer）|
| Placement | `PlacementPlan` | Data Movement | `selectedPlace`（识别跨 place 边）|
| Placement | `PlacementPlan` | Materialization | `selectedPlace`（写入 `memory_space`）|
| Static Memory Planning | `StaticMemoryPlan` | Data Movement | `workspaceSlots`（复用已有 on-chip slot）|
| Static Memory Planning | `StaticMemoryPlan` | Materialization | `workspaceSlots`（生成共享 workspace）、`peakUsagePerPlace`（容量 verifier）|
| Data Movement | `MovementPlan` | Materialization | `movements`、`selectedPath`（插入 `memref.copy`）|
| Materialization | `MemoryRealizationPlan` | 第五层 Translate | `resolvedPlacement`、`workspaceLayout`、`resolvedMovements`（只读消费）|

---

### 5.2 输入与输出

### 输入

第四层以 `ScheduleDecisionSet` 和第三层 `Structured Lowering` 后的结构化 module 为唯一真相来源，分四类：

| 来源 | 内容 | 访问方式 |
|---|---|---|
| 第三层决策结果 | `scheduleContract`、`promotionHints`、`cachePlan`、`unitAssignment`、`decisionGuards`、`pipelineDepthExpr`、`enableDoubleBuffer` | 从 `func` attribute `AscendScheduleDecisionSetAttr` 反序列化为 `ScheduleDecisionSet` 对象，以只读方式注入各 planner |
| 第三层 IR 结构 | loop 骨架、indexing 关系，以及 `Structured Lowering` 写入的内存意图标记 | 直接从 IR attribute 读取：`CacheReadMarker` / `CacheWriteMarker`（附加在 loop op 或计算 op）、`PipelineMarker`（附加在最外层 pipeline loop）、`DoubleBufferMarker`（附加在 movement loop）、`PromotionHintAttr`（附加在对应 op 或 loop） |
| target 查询接口 | `TargetMemoryModel`、`TargetIntrinsicModel`、`TargetCostModel` | 编译器初始化阶段构造，以只读引用注入，不通过 IR attribute 传递 |
| 结构化 tensor IR | 本体 | 当前 pass 的 `ModuleOp` |

**`decisionGuards` 的双重来源：** `decisionGuards` 同时存在于两处，两者必须一致：
- `ScheduleDecisionSet` 中每个 `ScheduleDecision.decisionGuards` 字段（纯数据对象）
- IR 上 guarded region 的 `AscendGuardAttr`，由 `Structured Lowering` 在生成 loop 骨架时写入

第四层以 `AscendGuardAttr` 作为 guard 结构的 IR 载体；若两者出现不一致，`BufferizationDriver` 的前置校验应报错拒绝进入后续规划。一致性检查为双向：IR 上每个 `AscendGuardAttr` 的 guard 表达式必须能在 `ScheduleDecisionSet.decisionGuards` 中找到对应条目（IR→数据方向）；同时，`ScheduleDecisionSet` 中每个 `decisionGuard` 条目必须能在 IR 上找到对应的 `AscendGuardAttr` guarded region（数据→IR 方向）。任一方向不一致均报错。

### 输出

输出为带显式 buffer、place、workspace 和 movement 语义的普通 MLIR，满足如下约定：

| 语义 | IR carrier |
|---|---|
| memory place | `memref` type 的 `memory_space` |
| 跨 place movement | `memref.copy` |
| workspace | `memref.alloc` + `memref.subview` |
| guard / unit / schedule 信息 | 现有 op attribute |
| function boundary | `func.func` 的 `memref` 参数与结果 |

### 主边界对象

`BufferizedKernelIR`、`PlacementPlan`、`StaticMemoryPlan`、`MovementPlan`、`MemoryRealizationPlan`

---

### 5.3 Bufferization

### 职责

把第三层 tensor IR 落成 buffer IR，并在 bufferized IR 上补做第四层所需的事实收集与合法性检查。

**边界约定：**
- 主转换复用 upstream `one-shot-bufferize`，不重写其主逻辑
- 不依赖 upstream pass 的私有临时状态；拿不到的分析信息统一从 bufferized IR 二次扫描回填
- 不在本节决定最终 memory place，不生成 target-specific movement op

### 输出：`BufferizedKernelIR`

| 字段 | 类型 | 含义 |
|---|---|---|
| `bufferValues` | `SmallVector<Value>` | 当前 kernel 中所有关键 buffer 值 |
| `aliasInfo` | `AliasInfoView` | alias / subview / view-like 关系 |
| `readPoints` | `DenseMap<Value, SmallVector<Operation *>>` | 每个 buffer 的读点 |
| `writePoints` | `DenseMap<Value, SmallVector<Operation *>>` | 每个 buffer 的写点 |
| `loopScopes` | `DenseMap<Value, LoopRegion>` | 每个 buffer 的主要生存区间（从定义点、最后使用点和 loop 骨架推导）|
| `guardBindings` | `DenseMap<Value, SmallVector<GuardExpr>>` | 每个 buffer 关联的 guard 条件集合（从第三层 `decisionGuards` 回填）|
| `bufferRoles` | `DenseMap<Value, BufferRole>` | 输入、输出、临时、cache、workspace 等角色 |

`BufferRole` 枚举：`InputBuffer`、`OutputBuffer`、`TemporaryBuffer`、`CacheBuffer`、`WorkspaceBuffer`

`GuardedBufferKey` = `(baseBuffer, guardExpr)`，同一底层 buffer 在某个 guard 上下文中的独立规划单元；单 guard 时退化为单条记录。

### 实现

**执行步骤：**

1. 在本地 wrapper pass 中执行前置校验（见下）
2. 调用 upstream `one-shot-bufferize`
3. 在 bufferized IR 上回填 buffer facts（五步，见下）
4. 执行后置校验：确认 bufferization 前后 loop、索引、guard 结构无漂移
5. 输出 `BufferizedKernelIR`

**前置校验最小条件列表（步骤 1）：**

以下任一条件不满足，应立即报诊断并中止，不进入 `one-shot-bufferize`：

| 校验项 | 条件 |
|---|---|
| `AscendScheduleDecisionSetAttr` 存在 | `func` attribute 中必须可解析出 `ScheduleDecisionSet` |
| `AscendGuardAttr` 与 `decisionGuards` 双向一致 | IR 上每个 guarded region 的 guard 表达式必须能在 `ScheduleDecision.decisionGuards` 中找到对应条目（IR→数据）；且 `ScheduleDecisionSet` 中每个 `decisionGuard` 条目必须能在 IR 上找到对应的 `AscendGuardAttr` guarded region（数据→IR）；任一方向不满足均报错 |
| 内存意图标记完整性 | 每个 `CacheReadMarker` 必须指向 IR 中存在的 `Value`，其 `place` 字段必须是 `TargetMemoryModel` 承认的合法 place；`PipelineMarker` 的 `depthExpr` 不得为空 |
| `PromotionHintAttr` 可解析 | 每个 `PromotionHintAttr` 的 `isBinding`、`reuseScope`、`preferredUnit` 字段必须完整，不允许存在 unknown 枚举值 |
| loop 骨架轴数与 `outerInnerMapping` 一致 | `ScheduleDecision.outerInnerMapping` 中记录的每个 `(outer, inner)` 轴对，在 IR loop 中必须能找到对应的嵌套层 |
| `unitAssignment` 非空 | 每个 `ScheduleDecision` 必须有明确的 `Cube` / `Vector` / `Both` 分配 |

**后置校验最小条件列表（步骤 4）：**

以下任一条件不满足，应报诊断并中止，不进入 Placement：

| 校验项 | 条件 |
|---|---|
| loop 层次数不变 | bufferization 不得新增或删除 loop 层次 |
| tile 索引可追溯 | 每个 `memref` 的 indexing 表达式仍可追溯到 `outerInnerMapping` 中的某个 `(outer, inner)` 轴对 |
| guarded region 边界不越界 | 原 guarded region 内的 op 集合不得因 bufferization 发生越界或合并 |
| `CacheReadMarker` / `PromotionHintAttr` 仍附着在对应 op 或 loop | bufferization 不得丢失这些 attr |
| 所有 `memref` 来源可追溯 | 每个 `memref` 值必须能追溯到 `memref.alloc`、function argument 或 loop carried block argument；"来源不明"指定义 op 既不是 `memref.alloc`，也不是已知的 view-like op（`memref.subview`、`memref.cast`、`memref.reinterpret_cast`、`memref.collapse_shape`、`memref.expand_shape`），且来源链在追溯过程中中断（出现无法识别的 definingOp）；此类 buffer 无法安全进入后续规划，必须报错 |

**回填五步：**

**① alias / subview 回填**

对每个关键 memref 值递归追溯其来源，形成两类事实：

- `baseBuffer(v)`：`v` 对应的底层 owning buffer / function argument / 根 block argument
- `aliasClass(base)`：所有 `baseBuffer` 相同的值构成同一 alias class

终止条件：定义 op 是 `memref.alloc`（owning buffer）、值是 function argument（ABI buffer）或无法继续追溯的 loop carried block argument。

**② read / write 回填**

优先复用 MLIR 已有接口（`MemoryEffectOpInterface`、`DestinationStyleOpInterface`）收集访问事实，所有访问折算到 `baseBuffer`。对未实现标准接口的未知 memref op，第一版保守记为同时读写；无法安全判定时直接报诊断阻止进入后续规划。

**③ loop scope 回填**

找定义点与最后 use 的最近公共 loop region 记为 `loopScope(baseBuffer)`。function argument 默认 function scope；若 use 泄漏到外层，scope 提升到覆盖所有 use 的最近公共 loop。

**④ guard 绑定回填**

`decisionGuards` 从 IR 上的 `AscendGuardAttr` 读取（而非重新解析 `ScheduleDecisionSet`），原因是 `Structured Lowering` 已把 guard 结构化地写入 IR，此处以 IR 为权威来源。绑定规则如下：

- 若 buffer 的定义点位于某个带 `AscendGuardAttr` 的 region 内，则继承该 guard
- 若其主要读写区间进一步落在更内层 guarded region 内，则收紧到该 guard
- 同一 `baseBuffer` 出现在多个不同 guard 下时，全部记录进 `guardBindings[baseBuffer]`
- 不生成新的 schedule 级 guard；若发现 IR 中出现 `ScheduleDecisionSet` 里不存在的 `AscendGuardAttr`，应在前置校验阶段已报错（见步骤 1）

**loop carried block argument 的跨 guard 归属规则：**

pipeline loop 中的 loop carried block argument 的定义点（loop 入口 block argument）和使用点可能分属不同 guarded region。归属规则如下：

| 场景 | 归属策略 |
|---|---|
| 定义点在 guard A 内，所有使用点也在 guard A 内 | 继承 guard A |
| 定义点在 guard A 内，部分使用点在 guard B 内 | 同时记录 guard A 和 guard B，在 `guardBindings[baseBuffer]` 中保留两条记录；后续各阶段按 `GuardedBufferKey` 展开独立规划 |
| 定义点在无 guard 的 region 内，使用点在 guard A 内 | 保守处理：不继承任何 guard，记为无 guard buffer；后续规划以最宽松约束处理 |
| 定义点在 guard A 内，使用点全部在无 guard 的 region 内 | 继承 guard A（定义点决定归属） |

loop carried block argument 不得被归并为单一 guard 后再分裂——若首次扫描发现跨 guard，直接记录多条，不做后置分裂。

后续各阶段仅在确实需要分支差异时，再把 `baseBuffer` 按 `GuardedBufferKey` 展开独立规划，避免无意义分裂。

**guard 分裂结果的下游传递约定：**

`BufferizedKernelIR` 是只读数据结构，`PlacementPlanner` 不得回写它。guard 分裂的结果通过 `PlacementPlan` 传递给下游：

- `PlacementPlan.selectedPlace` 以 `GuardedBufferKey = (baseBuffer, guardExpr)` 为 key；同一 `baseBuffer` 若在不同 guard 下被分裂，则在 `selectedPlace` 中有多条记录
- `StaticMemoryPlanner` 和 `MovementPlanner` 均以 `GuardedBufferKey` 为规划单元，通过查询 `PlacementPlan.selectedPlace` 感知分裂结果，无需再访问 `BufferizedKernelIR.guardBindings`
- 若某 `baseBuffer` 在 `PlacementPlan` 中只有一条记录（未分裂），下游以该单条记录处理；若有多条记录（已分裂），下游对每条 `GuardedBufferKey` 独立规划
- `PlacementPlanner` 在完成分裂决定后，必须在 `PlacementPlan.fallbackReason` 中为每个分裂的 key 记录分裂原因，供 debug 和 verifier 使用

**⑤ bufferRole 回填**

| 条件 | role | 说明 |
|---|---|---|
| function argument | `InputBuffer` | |
| ABI 可见输出 | `OutputBuffer` | |
| 对应 op 或 loop 上有 `CacheReadMarker` / `CacheWriteMarker` 指向该 value | `CacheBuffer` | 从 IR attribute 直接读取，不从 `cachePlan` 重新推导 |
| 由 `MovementPlanner` 在规划阶段创建的 copy 目标 buffer，或对应 op 上有 `PackingMarker` / `TransposeMarker` | `WorkspaceBuffer` | 此类 buffer 在 bufferization 完成时尚未存在，由 Data Movement 阶段在 `reuseOrCreateTransferBuffer` 中创建；`bufferRole` 回填阶段遇到此类 buffer 时，以 buffer 是否由 movement 规划新建为判断依据，而非基于计算语义 |
| 其余局部中间值（`memref.alloc` 来源，无上述任何 marker，非 movement 新建） | `TemporaryBuffer` | |

**`WorkspaceBuffer` 与 `TemporaryBuffer` 的区分依据：**

`bufferRole` 回填（步骤⑤）在 bufferization 完成时执行，此时 `MovementPlanner` 尚未运行，movement 新建的 buffer 还不存在。因此步骤⑤**不负责**识别 `WorkspaceBuffer`——所有由 `MovementPlanner` 新建的 copy 目标 buffer，在 `reuseOrCreateTransferBuffer` 中创建时直接标记为 `WorkspaceBuffer`，不经过步骤⑤的回填流程。步骤⑤只需区分 `InputBuffer`、`OutputBuffer`、`CacheBuffer` 和 `TemporaryBuffer` 四类。

`CacheBuffer` 的识别必须以 IR 上的 `CacheReadMarker` / `CacheWriteMarker` 为准，不允许重新从 `cachePlan` 推导——`Structured Lowering` 已把 `cachePlan` 物化为这两种 attr，第四层不得绕过它们直接解释 `cachePlan`。

**示例：`broadcast + add + reduce`**

| 字段 | 示例 |
|---|---|
| `bufferValues` | `%x_buf`、`%b_buf`、`%acc_buf` |
| `aliasInfo` | `%acc_buf` 与对应 `subview` 共享底层 buffer |
| `readPoints[%b_buf]` | `linalg.generic` 的输入 use |
| `writePoints[%acc_buf]` | `linalg.generic` 的 destination use |
| `loopScopes[%b_buf]` | 当前 tile loop |
| `bufferRoles[%b_buf]` | `CacheBuffer` 候选 |

**代码接口：**

```cpp
class BufferizationDriver {
public:
  FailureOr<BufferizedKernelIR>
  build(ModuleOp module, const ScheduleDecisionSet &schedule,
        DiagnosticEmitter &diag) const;

private:
  LogicalResult runOneShotBufferize(ModuleOp module, DiagnosticEmitter &diag) const;
  BufferizedKernelIR collectBufferFacts(ModuleOp module, DiagnosticEmitter &diag) const;
};
```

---

### 5.4 Placement

### 职责

决定每个 buffer 应落在哪个 memory place，输出 `PlacementPlan`。

**目标：** 为每个关键 buffer 生成合法 place 候选集合，结合第三层语义和 target memory hierarchy 选择最终 place，并在选择阶段处理容量、对齐、可见性和回退规则。

**非目标：** 不直接生成 target-specific movement op，不做 workspace packing。

### 输出：`PlacementPlan`

| 字段 | 类型 | 含义 |
|---|---|---|
| `selectedPlace` | `DenseMap<GuardedBufferKey, MemoryPlace>` | 每个规划单元的最终 place |
| `candidatePlaces` | `DenseMap<GuardedBufferKey, SmallVector<MemoryPlace>>` | 每个规划单元的合法候选集合 |
| `fallbackReason` | `DenseMap<GuardedBufferKey, StringRef>` | 发生降级或回退时的原因 |

最小 place 集合：`GM`（全局内存）、`VECIN`（向量输入）、`VECCALC`（向量计算）、`VECOUT`（向量输出）、`A1/B1`（matmul 输入上层）、`A2/B2`（matmul 输入下层）、`CO1`（matmul 累加输出）

### 实现

**执行步骤：**

1. 读取 `ScheduleDecisionSet` 中的 `unitAssignment`、`promotionHints`（含 `isBinding`）、`cachePlan`，以及 IR 上的 `PromotionHintAttr` 和 `CacheReadMarker`，构造候选 place 集合
2. 用 `TargetMemoryModel` 的容量、对齐、可见性和合法 path 过滤非法候选
3. 用 `TargetCostModel` 按复用收益、movement 成本和执行单元邻近性做稳定排序
4. 若同一 `baseBuffer` 在多个 guard 下需要不同 place，则按 `GuardedBufferKey` 分裂规划（否则保持共享）
5. 选出最终 place，记录回退原因，输出 `PlacementPlan`

**`unitAssignment` 到候选 place 的映射规则（步骤 1 候选构造的核心依据）：**

`unitAssignment` 决定了候选 place 的初始范围，是候选构造的第一优先依据，优先级高于 `promotionHints`：

| `unitAssignment` | buffer role | 候选 place 初始集合 |
|---|---|---|
| `Vector` | 输入 tile | `{VECIN, GM}` |
| `Vector` | 中间计算 | `{VECCALC}` |
| `Vector` | 输出 tile | `{VECOUT, GM}` |
| `Cube` | lhs tile | `{A1, A2, GM}` |
| `Cube` | rhs tile | `{B1, B2, GM}` |
| `Cube` | accumulator | `{CO1}` |
| `Both`（Cube+Vector epilogue）| matmul 输出中间值 | `{CO1}`（固定选 CO1；CO1→VECIN movement 由 Data Movement 阶段负责，Placement 只需将该 buffer 的 place 定为 CO1，其 consumer 在 VECIN，Data Movement 自动检测跨 place 边并插入 `memref.copy`）|
| 任意 | function 输入 / 输出 | `{GM}`（固定，不受 `unitAssignment` 影响）|

`promotionHints` 在候选集合基础上进一步约束（不扩展集合，只做收紧或排序调整）：若 hint 的 `preferredUnit` 与 `unitAssignment` 矛盾，以 `unitAssignment` 为准，hint 降级为建议。

**`isBinding` 强制约束处理：**

`promotionHints.isBinding = true` 来自第三层 `scheduleContract.mustKeepOnChipValues`，第四层必须严格遵守：

| 场景 | 处理方式 |
|---|---|
| `isBinding=true`，容量充足 | 按 `preferredUnit` / `unitAssignment` 选择最优片上 place |
| `isBinding=true`，容量不足 | **报编译错误**，不允许静默降级到 `GM`；错误信息必须包含：buffer 名、所需容量、place 当前剩余容量 |
| `isBinding=true`，path 不合法 | **报编译错误**，不允许绕过 path legality 检查 |
| `isBinding=false`，容量不足 | 允许降级并输出 warning；降级路径见默认 place 与回退规则表 |
| `isBinding=false`，path 不合法 | 允许回退到更保守合法路径，输出 warning |

**默认 place 与回退规则：**

| 场景 | 默认候选 | 回退 |
|---|---|---|
| function 输入 / 输出 | `GM` | 不回退 |
| vector 输入 tile | `VECIN` | 容量不足或 path 不合法时回退到 `GM` 直读 |
| vector 中间计算结果 | `VECCALC` | 容量不足时回退为更保守的 local alloc |
| vector 输出 tile | `VECOUT` | 容量不足时退回更保守的 local 组织，但必须仍可回写 `GM` |
| matmul lhs tile | `A1 -> A2` | 任一层不合法时回退到更短路径或禁用对应提升 |
| matmul rhs tile | `B1 -> B2` | 任一层不合法时回退到更短路径或禁用对应提升 |
| matmul accumulator | `CO1` | `CO1` 不可用时回退到更保守的累加组织方式 |
| cache / workspace buffer | 由 `cachePlan` 和主要 consumer 决定 | 容量或路径不满足时按显式规则降级 |

**guard 分裂判定：**

| 条件 | 是否必须分裂 |
|---|---|
| 不同 guard 下 `selectedPlace` 不同 | 必须 |
| 同一候选 place 在某个 guard 下合法、另一个不合法 | 必须 |
| 不同 guard 下需要的 movement `pathKind` 不同 | 必须 |
| 一个 guard 需要独立 local buffer，另一个可以共享 | 必须 |
| `selectedPlace`、path legality、容量结论完全相同 | 不必 |

**落地约束：** place legality、path legality、容量和对齐判断必须直接查询 `TargetMemoryModel`；结果最终只通过 `memory_space` 落地，不引入新的 placement op。

**`HorizontalFusionCandidate` 的 `sharedInputs` 去重规则：**

当 `ScheduleDecisionSet` 对应的原始候选为 `HorizontalFusionCandidate` 时（识别方式：`ScheduleDecision` 对象持有 `candidateKind` 字段，值为 `HorizontalFusion`；该字段由第三层 `ScheduleDecisionBuilder` 在构造 `ScheduleDecision` 时从原始 `KernelCandidate` 的 `primitives` 字段中提取并写入，第四层直接读取 `candidateKind`，不重新访问 `primitives`），同一 `HorizontalFusionCandidate` 的多个 sibling group 可能共享相同的输入 buffer（即 `sharedInputs` 字段所列的 buffer）。`PlacementPlanner` 在处理这类候选时必须遵守以下规则：

| 规则 | 说明 |
|---|---|
| 识别 `sharedInputs` | 从 `HorizontalFusionCandidate.sharedInputs` 中取出所有跨 sibling group 共享的输入 buffer；这些 buffer 在 `BufferizedKernelIR` 中可能对应同一 `baseBuffer` 的多个 use |
| 统一分配 place | 同一 `baseBuffer` 的 `sharedInputs` 只能分配一个 place；不允许同一底层 buffer 在不同 sibling group 中被分配到不同 place |
| 容量计入一次 | `sharedInputs` buffer 的容量只计入一次，不因 sibling group 数量放大；各 sibling group 的 consumer 通过 alias / subview 共享同一 place 上的 buffer |
| 非共享 buffer 独立规划 | 每个 sibling group 私有的输入、输出和中间 buffer 按各自 `unitAssignment` 独立进行 place 选择，不受其他 group 的影响 |
| `isBinding` 约束仍适用 | 若 `sharedInputs` 中某个 buffer 的 `promotionHints.isBinding = true`，则统一 place 必须满足片上约束；容量不足时按 `isBinding` 规则报编译错误 |

**规划伪代码：**

```text
for each buffer in bufferFacts:
  planningKeys = expandByGuardIfNeeded(buffer, guardBindings)
  for each key in planningKeys:
    candidates = buildCandidatePlaces(key, promotionHints, cachePlan, unitAssignment)
    candidates = filterByTargetLegality(candidates, targetMemoryModel)
    candidates = filterByCapacityAndAlignment(candidates, currentMemoryUsage)
    if candidates is empty:
      candidates = fallbackPlaces(key)
    selected = stableBest(candidates, targetCostModel, reuseBenefit, movementCost)
    assign key -> selected
    reserveCapacity(selected, key, pipelineDepthExpr, enableDoubleBuffer)
```

**示例 A：vector 通路**

| buffer | role | 候选 place | 最终 place |
|---|---|---|---|
| `%src_tile` | vector 输入 | `GM`, `VECIN` | `VECIN` |
| `%tmp_tile` | vector 中间值 | `VECCALC` | `VECCALC` |
| `%dst_tile` | vector 输出 | `VECOUT` | `VECOUT` |

**示例 B：matmul 通路**

| buffer | role | 候选 place | 最终 place |
|---|---|---|---|
| `%lhs_tile` | matmul lhs | `GM`, `A1`, `A2` | `A1/A2` |
| `%rhs_tile` | matmul rhs | `GM`, `B1`, `B2` | `B1/B2` |
| `%acc_tile` | matmul accumulator | `CO1` | `CO1` |

**代码接口：**

```cpp
class PlacementPlanner {
public:
  FailureOr<PlacementPlan>
  build(const BufferizedKernelIR &bufferizedIR,
        const ScheduleDecisionSet &schedule,
        const TargetMemoryModel &memoryModel,
        const TargetCostModel &costModel,
        DiagnosticEmitter &diag) const;
};
```

---

### 5.5 Static Memory Planning

### 职责

在 placement 确定之后，优化片上内存占用和工作区组织方式：计算 live range、识别可复用 local buffer、在条件满足时把多个 temporary / cache buffer 折叠到共享 workspace，并将 `enableDoubleBuffer`、`pipelineDepthExpr` 带来的额外占用纳入容量检查。

**算法策略（先保守、后增强）：**

| 层次 | 算法 | 适用阶段 |
|---|---|---|
| 第一层 | live interval reuse + 线性 packing | 第一版实现 |
| 第二层 | 带对齐和容量约束的 interval coloring / packing | 后续增强 |
| 回退层 | 禁用部分 promotion / 退回独立 alloc | 容量冲突无法化解时 |

### 输出：`StaticMemoryPlan`

| 字段 | 类型 | 含义 |
|---|---|---|
| `liveIntervals` | `DenseMap<GuardedBufferKey, LiveInterval>` | 每个规划单元的生存区间 |
| `workspaceSlots` | `DenseMap<GuardedBufferKey, WorkspaceSlot>` | 每个可复用规划单元对应的 workspace slot（第一版可为空）|
| `peakUsagePerPlace` | `DenseMap<MemoryPlace, int64_t>` | 每个 place 的峰值占用 |

**与 MLIR upstream 的边界：** upstream 只提供 IR carrier 和局部 pass 基础设施，不能替代 target-aware 的静态内存规划能力，本节 planner 仍需本地实现。

| upstream 能力 | 复用方式 | 不能替代的部分 |
|---|---|---|
| `one-shot-bufferize` | 提供 tensor->memref 和 in-place/out-of-place 基础结果 | 不负责 on-chip place 级静态内存规划 |
| ownership-based buffer deallocation | 处理 buffer 生命周期结束后的释放语义 | 不负责片上 workspace reuse |
| alloc / buffer hoisting | 把 alloc 外提到更合适的支配点 | 不决定哪些 buffer 应共享 slot |
| `memref.subview` | 表达共享 workspace 的切片结果 | 不负责 slot 规划和容量回退 |

### 实现

**执行步骤：**

1. 按 memory place 分组收集 local buffer
2. 根据定义点、最后使用点和 loop scope 计算 live range
3. 从 `ScheduleDecisionSet` 读取 `pipelineDepthExpr`、`enableDoubleBuffer`，生成容量放大系数；若 `pipelineDepthExpr` 含动态符号（运行时变量），按以下策略处理：
   - 若 `TargetMemoryModel` 为该 `pipelineDepthExpr` 提供了静态上界（`maxPipelineDepth`），则以该上界作为放大系数进行悲观容量检查
   - 若无静态上界可用，则跳过该 place 的容量检查，输出 warning（不报编译错误），并在 `StaticMemoryPlan.peakUsagePerPlace` 中将该 place 的峰值标记为 `kUnknown`
   - 不允许将含动态符号的 `pipelineDepthExpr` 直接当作编译期常数使用
4. 对 live range 不重叠且 size/alignment 兼容的 buffer 做 reuse 判定
5. 启用共享 workspace 时生成 workspace slot，并统计峰值占用
6. 超过 place 容量时触发回退或禁用部分 promotion

**最小实现规则：**

| 规则 | 说明 |
|---|---|
| 按 memory place 分组 | 只在同一 place 内做 reuse 和 packing，不跨 place 复用 |
| 按 live range 判定复用 | 生命周期重叠的 buffer 不能共享同一段 workspace |
| workspace 优先服务临时 buffer | function boundary buffer 不参与片上 workspace 复用 |
| `enableDoubleBuffer` 单独计入容量 | 默认按两份 local buffer 计算占用 |
| `pipelineDepthExpr` 参与容量估算 | pipeline stage 增加的并发必须计入容量检查 |
| 第一版允许退化 | 可退化为每个 local buffer 单独 alloc，但容量检查框架必须保留 |

**规划伪代码：**

```text
for each place in onChipPlaces:
  buffers = collectBuffersAssignedTo(place)
  factor = capacityMultiplier(schedule.pipelineDepthExpr,
                              schedule.enableDoubleBuffer, place)
  intervals = buildLiveIntervals(buffers)
  sort intervals by startPoint

  for each interval in intervals:
    slot = findReusableSlot(interval, existingSlots)
    if slot exists:
      assign interval -> slot
    else:
      slot = createNewSlot(interval.size, interval.alignment)
      assign interval -> slot

  if totalSize(existingSlots) * factor > place.capacity:
    triggerFallbackOrDisablePromotion(place)
```

**示例：两个生命周期不重叠的 local temporary**

| buffer | place | live range | 结果 |
|---|---|---|---|
| `%tmp0` | `VECIN` | `[L1, L3]` | 复用 slot0 |
| `%tmp1` | `VECIN` | `[L4, L6]` | 复用 slot0 |

启用共享 workspace 后的 materialized 结果：

```mlir
%workspace = memref.alloc(...) : memref<..., 9 : i32>
%tmp0 = memref.subview %workspace[...] : ...
%tmp1 = memref.subview %workspace[...] : ...
```

**与 Data Movement 的衔接：**

`StaticMemoryPlan` 中的 `workspaceSlots` 在 Data Movement 阶段有两处直接消费：

| 消费场景 | 消费字段 | 消费逻辑 |
|---|---|---|
| 判断是否可复用已有 local buffer | `workspaceSlots[key].place`、`.size`、`.alignment` | 若 consumer 所需 place / size / alignment 与已有 slot 匹配，且 live range 不重叠，则直接复用该 slot，不创建新 copy 目标 buffer |
| 跨 place movement 的 dst buffer 选择 | `workspaceSlots[key]` | 若 dst place 上已有 slot 可覆盖 consumer，则 `reuseOrCreateTransferBuffer` 应返回该 slot 对应的 `memref.subview`，而非新建 `memref.alloc` |

因此 Data Movement 在构造 movement 之前，必须先查询 `StaticMemoryPlan.workspaceSlots`，确认是否存在可复用的 on-chip slot，再决定是否插入新的 copy 目标。

**第一版 `workspaceSlots` 为空时的退化行为：** 第一版 `StaticMemoryPlanner` 允许输出空的 `workspaceSlots`。此时 Data Movement 的 `reuseOrCreateTransferBuffer` 将始终走"创建新 copy 目标"路径，不做 slot 复用。这是可接受的保守行为：所有跨 place 边各自新建独立 local buffer，不共享 slot，内存占用偏大但语义正确。第一版实现中不需要因 `workspaceSlots` 为空而报错或 warning。

**代码接口：**

```cpp
class StaticMemoryPlanner {
public:
  FailureOr<StaticMemoryPlan>
  build(const BufferizedKernelIR &bufferizedIR,
        const PlacementPlan &placement,
        const ScheduleDecisionSet &schedule,
        const TargetMemoryModel &memoryModel,
        DiagnosticEmitter &diag) const;
};
```

---

### 5.6 Data Movement

### 职责

在 placement 和静态内存规划结果的基础上，把跨 place 的 producer-consumer 关系显式化，并以 `memref.copy` 写回 MLIR。

**目标：** 识别所有跨 place 边 → 选择合法 path → 消除冗余 copy → 输出 `MovementPlan`。

### 输出：`MovementPlan`

| 字段 | 类型 | 含义 |
|---|---|---|
| `movements` | `SmallVector<MovementStep>` | 显式 movement 列表 |
| `selectedPath` | `DenseMap<MovementId, MemoryPath>` | 每个 movement 的合法路径 |
| `guardBinding` | `DenseMap<MovementId, GuardExpr>` | movement 绑定的 guard |

**第四层 movement carrier 与 target 语义映射：**

| `memref.copy` 路径 | `pathKind`（对应 V2-8.3 路径分类） | 后续 lowering 推导的 target 语义 |
|---|---|---|
| `GM -> VECIN` | `DirectCopy` | `data_copy_l2` |
| `GM -> A1/B1`（无 transpose） | `Load2D` | `data_copy_nd2nz` |
| `GM -> B1`（带 transpose variant） | `Load2DTranspose` | `data_copy_nd2nz` 的 transpose variant |
| `A1 -> A2` | `DirectCopy` | `load_data_l0` |
| `B1 -> B2` | `DirectCopy` | `load_data_with_transpose` |
| `CO1 -> VECIN` | `QueueTransfer` | `QueueTransfer` / `DataCopyCO12DstOp` |
| `CO1 -> GM`（fixpipe） | `FixPipe` | `fix_pipe_l0c2out` |
| `VECOUT -> GM` | `DirectCopy` | `data_copy_l2` |

**统一 carrier 约定**：第四层不为 `Load2DTranspose` 与 `FixPipe` 引入额外 op，所有跨 place 搬运（含 transpose load 与 fixpipe）均以 `memref.copy` 表达；具体路径区分由 `MovementPlan.selectedPath.pathKind` 持有，第五层 `ComputeLoweringDriver` 在 6.3 阶段按 `pathKind` 选择对应 backend intrinsic（如 `Load2DTranspose` → `data_copy_nd2nz` 的 transpose variant，`FixPipe` → `fix_pipe_l0c2out`）。`MovementPlanner` 选路时通过 `TargetMemoryModel.findPaths(src, dst)` 获取所有合法 path variant，再按 `TargetCostModel.getPathCost` 排序选最优；同一 src/dst pair 存在多条 path variant 时（如 `GM → B1` 的 `Load2D` 与 `Load2DTranspose`），由 `selectLegalPath` 根据 `scheduleContract.layoutConstraints` 决定采用哪条 variant。

### 实现

**执行步骤：**

1. 根据 producer place 和 consumer place 识别跨层传递
2. 对每条跨层边用 target memory / intrinsic / cost model 选择合法 path
3. 优先复用已有 local / workspace buffer，再决定是否创建新的 copy 目标
4. 先做冗余 copy 消除，再做相邻 copy 合并
5. 输出 `MovementPlan`

**最小实现规则：**

| 规则 | 说明 |
|---|---|
| 跨 place 必须显式化 | producer 和 consumer 不在同一 place 时，必须插入显式 movement |
| 同 place 默认不新增 copy | source / destination 已在同一 place 时，不额外插 movement |
| 优先复用现有 local buffer | 若已有合法 local buffer 能覆盖 consumer 需求，优先复用 |
| copy 消除优先于 copy 合并 | 先删冗余 movement，再考虑相邻 tile copy 合并 |
| path 不合法时必须回退 | 不能生成非法 `src -> dst` 组合，必须回退到合法路径 |
| dynamic guard 不改变原边界 | movement 可绑定已有 guard，但不能新增 schedule 级分支 |

**冗余 copy 的最小判断条件：**

以下任一条件成立，则该 copy 为冗余，应在 `eliminateRedundantCopies` 阶段删除：

| 条件 | 说明 |
|---|---|
| `src` 和 `dst` 经 alias 分析指向同一 `baseBuffer`，且在同一 place | 无需 copy，直接使用原 buffer |
| `dst` buffer 在 copy 之后的所有 use 都被覆盖（即 `dst` 只被该 copy 写、下一步立刻被另一个 copy 覆盖） | dead store，copy 结果从未被消费 |
| 存在两条连续 copy `A -> B -> C`，且 `B` 除这两条 copy 外没有其他 reader / writer | 若 `A -> C` 是合法 path，则消除中间 `B`，合并为 `A -> C`；若 `A -> C` 不是合法 path（target 不支持直接跨越），则**保留原 `A -> B -> C` 路径**，不做消除 |
| `src` 的 `baseBuffer` 和 `dst` 的 `baseBuffer` 在 `StaticMemoryPlan.workspaceSlots` 中指向同一 slot | 同 slot 内部不需要 copy |

注意：冗余消除只删除确定无用的 copy；对于不能静态证明冗余的情况，保守保留，不允许猜测性删除。

**相邻 copy 合并规则（`mergeAdjacentCopiesIfLegal`）：**

相邻 copy 合并与冗余消除不同：冗余消除删除无用 copy，合并则是将多个相邻 tile copy 折叠为粒度更大的单次 copy 以减少 movement 开销。第一版**不实现**相邻 copy 合并，伪代码中 `mergeAdjacentCopiesIfLegal` 为空操作（no-op），直接返回原 `movementPlan`。

后续版本若实现，最小合并条件如下：

| 条件 | 说明 |
|---|---|
| 相邻 tile copy 的 `src` 和 `dst` place 完全相同 | 只合并同一 place 对之间的连续 copy |
| 合并后的 copy 覆盖范围不超过 `TargetMemoryModel` 的单次 movement 粒度上限 | 超出上限时不合并 |
| 合并后的 dst buffer 在静态内存规划中有连续可用 slot | 不允许为合并创建额外碎片 |
| `src` 的 tile 在 IR 中是静态可分析的连续布局 | 动态布局或非连续布局不参与合并 |
| 合并不跨越 guarded region 边界 | 不同 guard 下的 copy 不合并 |

**规划伪代码：**

```text
movementPlan = []
for each consumer in bufferFacts:
  srcPlace = getProducerPlace(consumer)
  dstPlace = getConsumerPlace(consumer)
  if srcPlace == dstPlace: continue

  path = selectLegalPath(srcPlace, dstPlace,
                         targetMemoryModel, targetIntrinsicModel, targetCostModel)
  dstBuffer = reuseOrCreateTransferBuffer(consumer, dstPlace)
  movement = createMovement(srcBuffer, dstBuffer, path, guard)
  append movement

movementPlan = eliminateRedundantCopies(movementPlan)
movementPlan = mergeAdjacentCopiesIfLegal(movementPlan)
```

**示例 A：vector 通路**

| source | target | path | materialize |
|---|---|---|---|
| `%src_gm` | `%src_vecin` | `GM -> VECIN` | `memref.copy %src_gm, %src_vecin` |
| `%dst_vecout` | `%dst_gm` | `VECOUT -> GM` | `memref.copy %dst_vecout, %dst_gm` |

**示例 B：matmul 通路**

| source | target | path | materialize |
|---|---|---|---|
| `%lhs_gm` | `%lhs_a1` | `GM -> A1` | `memref.copy` |
| `%lhs_a1` | `%lhs_a2` | `A1 -> A2` | `memref.copy` |
| `%acc_co1` | `%acc_vecin` | `CO1 -> VECIN` | `memref.copy` |

**落地路径：** 第一版先复用 `memref.copy + memory_space` 通路；后续 backend lowering 再根据 `src/dst` place 选择具体 target-specific movement 实现；movement planning 增强时只改规划逻辑，不改 IR carrier。

**代码接口：**

```cpp
class MovementPlanner {
public:
  FailureOr<MovementPlan>
  build(const BufferizedKernelIR &bufferizedIR,
        const PlacementPlan &placement,
        const StaticMemoryPlan &staticMemory,
        const TargetMemoryModel &memoryModel,
        const TargetIntrinsicModel &intrinsicModel,
        const TargetCostModel &costModel,
        DiagnosticEmitter &diag) const;
};
```

---

### 5.7 Materialization 与 Verifier

将 Placement、Static Memory Planning 和 Data Movement 的结果统一写回普通 MLIR（5.7.1），并验证输出 IR 满足第三层约束和 target memory 约束（5.7.2）。两者由同一个 `MemoryRealizationDriver.materialize()` 调用完成，共享 `MemoryRealizationPlan` 输出结构。

### 输出：`MemoryRealizationPlan`

| 字段 | 类型 | 含义 |
|---|---|---|
| `resolvedPlacement` | `DenseMap<GuardedBufferKey, MemoryPlace>` | 最终冻结的 `(buffer, guard) -> place` 结果 |
| `workspaceLayout` | `DenseMap<GuardedBufferKey, WorkspaceSlot>` | 最终采用的 workspace slot 布局（未启用共享时可为空）|
| `resolvedMovements` | `SmallVector<MovementStep>` | 最终保留的 movement 及 path/guard 绑定 |
| `materializedAllocs` | `SmallVector<memref::AllocOp>` | 新生成的 local / workspace alloc |
| `materializedCopies` | `SmallVector<memref::CopyOp>` | 新生成的显式 movement |
| `diagnostics` | `SmallVector<StringRef>` | verifier 输出的诊断信息 |

**`MemoryRealizationPlan` 的冻结语义：**

`MemoryRealizationPlan` 是第四层唯一对外输出的稳定规划结果，第五层（Translate）以只读方式消费它。冻结规则如下：

| 字段 | 冻结时机 | 第五层消费方式 |
|---|---|---|
| `resolvedPlacement` | Materialization 完成后不可再修改 | `ComputeLoweringDriver` 用它确认每条 movement 的 `src/dst place`，验证 `Backend Compute IR` 的搬运路径与 place 对齐 |
| `workspaceLayout` | 同上 | `ComputeLoweringDriver` 用它确认 workspace 的 alloc 起始地址和 size，生成 `tbuf` 初始化 |
| `resolvedMovements` | 同上 | `ComputeLoweringDriver` 用 `pathKind` 选择对应的 target movement op（如 `data_copy_nd2nz`、`QueueTransfer`）|
| `materializedAllocs` / `materializedCopies` | 同上 | 为 debug / verifier 保留，不被 Translate 直接消费 |

一旦 `MemoryRealizationDriver.materialize()` 返回 success，`MemoryRealizationPlan` 进入冻结状态，不允许任何阶段（包括第五层内部）对其写入或修改。

**`materialize()` 失败时的 IR 状态约定：**

`materialize()` 按步骤 1→2→3→4→5 顺序执行，步骤 1–4 会修改 IR（插入 alloc、copy，改写 operand）。若步骤 5（verifier）失败，IR 已处于部分物化状态。约定如下：

| 场景 | IR 状态 | 处理方式 |
|---|---|---|
| 步骤 1–4 成功，步骤 5 verifier 失败 | IR 已被修改（含新增 alloc / copy），处于部分物化状态 | 保留修改后的 IR 供诊断使用，**不回滚**；调用方（编译 pipeline）在收到 failure 后应终止后续 pass，不将该 IR 传递给第五层 |
| 步骤 1–4 中途失败（如 `isBinding` error 中止）| IR 可能处于不完整的中间状态 | 同上，保留 IR 供诊断，不回滚，调用方终止 pipeline |
| `MemoryRealizationPlan` 的冻结状态 | `materialize()` 返回 failure 时，plan **不进入冻结状态**，第五层不得消费 | — |

不实现 IR 回滚的原因：MLIR pass 基础设施不提供事务性 IR 修改，回滚代价高且引入额外复杂度；编译失败时保留中间 IR 有助于诊断。

**代码接口：**

```cpp
class MemoryRealizationDriver {
public:
  // 成功返回时，plan 进入冻结状态，第五层以只读方式消费
  FailureOr<MemoryRealizationPlan>
  materialize(ModuleOp module,
              const PlacementPlan &placement,
              const StaticMemoryPlan &staticMemory,
              const MovementPlan &movement,
              const ScheduleDecisionSet &schedule,
              DiagnosticEmitter &diag) const;
};
```

---

#### 5.7.1 Materialization

**职责：** 将 Placement、Static Memory Planning 和 Data Movement 的规划结果写回 IR，生成可被后续 lowering 消费的普通 MLIR。

**执行步骤：**

1. 为 `PlacementPlan` 中的 local / workspace buffer 创建 alloc 或 subview，并写入 `memory_space`
2. 为 `StaticMemoryPlan` 中的 workspace slot 创建共享 workspace
3. 为 `MovementPlan` 中的 movement 插入 `memref.copy`，并改写相关 operand / subview
4. 删除已证明冗余的 alloc / copy

**实现原理：**

| 步骤 | 实现方式 | 落地要求 |
|---|---|---|
| alloc / workspace 落地 | 创建 local alloc 或共享 workspace，写入 `memory_space` | 不改变计算语义 |
| movement 落地 | 插入 `memref.copy`，改写相关 operand / subview | 只输出普通 MLIR |
| 清理与回写 | 删除冗余 alloc / copy，补充必要 attrs | 输出 IR 仍可被后续 lowering 消费 |

**落地路径：** 先把当前 `ascendc-buffer-placement` 的"标 `memory_space` + 插 `memref.copy`"能力整理成通用 materialization 框架，再逐步把 verifier 从零散检查收敛成一组稳定规则。

**伪代码：**

```text
// Phase 1: Alloc & workspace materialization
for each placement in placementPlan:
  materializeAllocOrSubview(placement)
  writeMemorySpace(placement)
  if placement.isBinding and not isOnChip(placement.place):
    emitError("isBinding buffer must be on-chip: " + placement.buffer + " -> " + placement.place)

for each workspaceSlot in staticMemoryPlan:
  materializeWorkspaceAndSubviews(workspaceSlot)

// Phase 2: Movement materialization
for each movement in movementPlan:
  insertMemrefCopy(movement)
  rewriteOperandsAndViews(movement)

// Phase 3: Cleanup
cleanupRedundantAllocAndCopy()
```

**示例：vector 通路最终落地**

```mlir
%src_vecin = memref.alloc(...) : memref<..., 9 : i32>
memref.copy %src_gm, %src_vecin : ...
%dst_vecout = memref.alloc(...) : memref<..., 10 : i32>
memref.copy %dst_vecout, %dst_gm : ...
```

---

#### 5.7.2 Verifier

**职责：** 在 Materialization 写回 IR 之后，验证输出 IR 满足第三层约束和 target memory 约束。Verifier 是 `materialize()` 步骤5，不单独暴露接口。

**检查顺序与最小诊断字段：**

Verifier 按以下顺序执行，前一项失败时后续项仍继续检查（以收集完整诊断），但整体返回 failure。例外：若检查项 1（Place legality）中存在 `isBinding=true` 的 buffer 失败，该 buffer 记录为 error 级别诊断后**立即中止后续 verifier**，原因是非法 place 会导致后续容量检查和 path legality 检查在无意义数据上产生级联错误，中止可避免误导性诊断。其余 warning 级别失败继续执行后续检查。

| 顺序 | 检查项 | 检查条件 | 失败时诊断必须包含 |
|---|---|---|---|
| 1 | Place legality | 每个 `memory_space` 值必须是 `TargetMemoryModel` 承认的合法 place；function boundary buffer 必须在 `GM` | buffer 名、非法 `memory_space` 值、`TargetMemoryModel` 支持的合法 place 列表 |
| 2 | Movement path legality | 每条 `memref.copy` 的 `src memory_space -> dst memory_space` 组合必须是 `TargetMemoryModel` 承认的合法 path | copy op 位置、`src place`、`dst place`、合法 path 列表 |
| 3 | 容量约束 | 每个 on-chip place 的 `peakUsagePerPlace` 不超过 `TargetMemoryModel` 的容量上限（含 `enableDoubleBuffer` 和 `pipelineDepthExpr` 的放大系数） | place 名、峰值用量、容量上限、放大系数来源 |
| 4 | Guard 一致性 | 每个 `GuardedBufferKey` 的 guard 表达式必须能在 `ScheduleDecisionSet.decisionGuards` 中找到对应条目；不同 guard 下的 `resolvedPlacement` 不得出现冲突的 `memory_space` | 冲突的 guard 对、buffer 名、对应的 place 差异 |
| 5 | MLIR 内置 verifier | 运行 `mlir::verify(module)` | MLIR 自身的诊断输出 |

`isBinding=true` 的 buffer 在 place legality 检查失败时，必须单独标注为编译错误（error 级别），不得与普通 warning 混淆。

**伪代码：**

```text
// Phase 4: Verifier（顺序执行；isBinding error 立即中止，其余失败继续收集诊断）
verifyPlaceLegality()          // 检查 memory_space 合法性
verifyMovementLegality()       // 检查 src->dst path 合法性
verifyCapacityConstraints()    // 检查峰值容量（含双缓冲/流水放大）
verifyGuardConsistency()       // 检查 GuardedBufferKey 与 decisionGuards 一致性
runMlirVerifier()              // MLIR 内置 verifier
```

Verifier 确认示例（vector 通路）：`9` 和 `10` 是合法 place；`GM -> VECIN`、`VECOUT -> GM` 是合法 path；当前 place 容量和对齐约束满足。