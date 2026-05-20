## 7. E2E Debug 与可观测性

本节定义贯穿五层编译流程的统一调试规范，目标是：在不污染正式 IR 契约的前提下，让开发者能够在每一层观察主边界对象、在层间转换时定位对象失真或失效。

### 7.1 Debug 输出机制

统一提供三种输出方式，彼此独立：

| 方式               | 用途                                                     | 是否进入正式 IR           |
| ------------------ | -------------------------------------------------------- | ------------------------- |
| `stage dump`       | 导出当前阶段的 IR、对象摘要和 verifier 结果              | 否                        |
| `debug annotation` | 在 IR 上临时标注关键分析结果，辅助 `mlir-opt` / 打印阅读 | 否，默认仅 debug 模式启用 |
| `debug report`     | 结构化导出分析结果、候选列表、决策和过滤原因             | 否                        |

**约束：**

- 正式编译链路不依赖任何 debug annotation
- 阶段内对象（`OpSemanticSummary`、`KernelPatternCandidate[]` 等）默认仅存在于 analysis cache 或 report 中，不写入 IR
- debug annotation 是只读观察视图，后续 pass 不得以其内容作为真相来源

### 7.2 各层可观测对象

| 层                | 应可观测的主边界对象                                         |
| ----------------- | ------------------------------------------------------------ |
| 第一层：Normalize | 规范化后的入口 module、被拒绝的方言 / 结构、保留的结构属性   |
| 第二层：Kernelize | `ProducerConsumerIndex`、`OpSemanticSummary`、结构标记属性、`OpRoleMap`、`FusionCandidate[]`、`KernelPatternCandidate[]`、最终 `KernelPattern[]` |
| 第三层：Schedule  | `CoalescedAxisInfo`、`ScheduleProblem`、`scheduleTemplate`、`scheduleSearchSpace`、过滤原因、`ScheduleDecisionSet` |
| 第四层：Realize   | `BufferizedKernelIR`、`PlacementPlan`、`StaticMemoryPlan`、`MovementPlan`、`MemoryRealizationPlan` |
| 第五层：Translate | `Backend Compute IR`、`AscendC Kernel MLIR`、`Host Tiling`、可选 `Runtime Manifest`、`ScheduleEntry[]` |

### 7.3 各层 Debug 重点

#### 7.3.1 第二层：Kernelize

重点回答：某个 op 为何被标记为某个 `OpRole`、某个候选为何能形成或被裁掉、最终 `KernelPattern` 为何拆成当前这组 region。

**最少输出内容：**

| 对象                         | 最少输出字段                                                 |
| ---------------------------- | ------------------------------------------------------------ |
| `OpSemanticSummary`          | `resultShape`、`indexingMaps`、`iteratorTypes`、`accessPatternKind`、`semanticAttrs` |
| `FusionCandidate`            | `seed`、内部 op 集合、外部输入 / 输出、合法性结果、收益分数  |
| `HorizontalFusionCandidate`  | `siblingCandidates`、`sharedInputs`、`perGroupContracts`、合法性结果（互不可达校验、组内候选数）、合并决策原因 |
| `KernelPatternCandidate`     | `primaryOps`、`roles`、`primitives`、边界输入输出、拆分原因  |

**`OpSemanticSummary` 生命周期：**

- 覆盖第二层全程，可选延伸至第三层 `ScheduleProblemBuilder`
- 一旦有 pass 修改 `resultShape`、`indexingMaps`、`iteratorTypes`、结构属性或 region 边界，旧 summary 立即失效
- 进入 `Structured Lowering` 前不再复用旧 summary

#### 7.3.2 第三层：Schedule

重点回答：为何选择了某个 `scheduleTemplate`、某些 `ScheduleInstance` 为何被保留或裁掉、最终 `ScheduleDecisionSet` 的 guard 和参数从何而来。

**最少输出内容：**

| 对象                  | 最少输出字段                                                 |
| --------------------- | ------------------------------------------------------------ |
| `CoalescedAxisInfo`   | 逻辑轴、原始轴来源、extent、barrier 原因                     |
| `ScheduleProblem`     | 轴集合、四类约束摘要、关键 shape 关系                        |
| `scheduleSearchSpace` | 候选总数、每个候选的 primitive 组合、`loadOrder`、`computeOrder`、cache 选择 |
| 过滤结果              | `StructuralFilter`、`Shape/HardwareFilter`、`MemoryFilter`、`Dedup`、`DominancePrune` 各自裁掉的候选和原因 |
| `ScheduleDecisionSet` | `decisionId`、guard 表达式、`scheduleInstance`、tiling 参数、`cachePlan`、`pipelineDepthExpr` |

#### 7.3.3 第四层：Realize

重点回答：某个值为何被放到某个 `MemoryPlace`、某条搬运路径为何存在、`cachePlan` / `promotionPlan` / `placement` 三者是否一致。

**最少输出内容：**

| 对象                        | 最少输出字段                                                 |
| --------------------------- | ------------------------------------------------------------ |
| `MemoryRealizationPlan`     | `resolvedPlacement`、`workspaceLayout`、`resolvedMovements`、materialized alloc/copy 列表 |
| `resolvedMovements`         | `src/dst`、`pathKind`、`loopRegion`、`guard`                 |
| `TargetMemoryModel`（参照） | path 可达性、容量规则、对齐约束                              |

#### 7.3.4 第五层：Translate

重点回答：backend lowering 是否忠实保留第三、四层决策；host / kernel / runtime 的 schema 是否一致；动态 shape 下 runtime 如何最终选中某个 decision。

**最少输出内容：**

| 对象                       | 最少输出字段                                                |
| -------------------------- | ----------------------------------------------------------- |
| `AscendC Kernel MLIR`      | kernel 参数列表、buffer 顺序、tiling params、guard 绑定情况 |
| `Host Tiling`              | 每个 tiling 参数的来源表达式                                |
| `Runtime Manifest`（可选） | `shapeBucketKey`、`guardSet`、`scheduleEntries`、`cacheKey` |

### 7.4 Diagnostics 规范

所有阶段的 diagnostics 必须包含以下最小字段集：

| 字段            | 含义                                                         |
| --------------- | ------------------------------------------------------------ |
| `stage`         | 所属阶段，取值为 `Normalize`、`Kernelize`、`Schedule`、`Realize`、`Translate` 之一 |
| `objectId`      | 绑定的最小对象标识，例如 `opName`、`candidateId`、`variantId`、`GuardedBufferKey`、`kernelName` |
| `reasonKind`    | 失败或回退原因类别，必须来自稳定枚举集合，不允许使用自由文本 |
| `message`       | 面向人类阅读的简短原因摘要                                   |
| `isRecoverable` | 是否允许当前阶段回退并继续主链                               |
| `fallbackTaken` | 若可回退，实际采用的回退路径；不可回退时为空                 |

**附加约束：**

- 若与动态 shape 或 guard 相关，额外记录 `guardExpr` 或 `shapeBucketKey`
- 若由 target 限制触发，额外记录最小 target 上下文（`memoryPlace`、`pathKind` 或 `intrinsicName`）
- 回退成功时也必须保留 diagnostics，供 debug 和缓存负结果使用

**当前版本最小 `reasonKind` 枚举集合：**

| `reasonKind`                  | 触发场景                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| `TargetProfileMissingField`   | 构造期（Target 建模）：SoC config 缺少必要字段，fail-fast    |
| `TargetPathIntrinsicMissing`  | 构造期（Target 建模）：必需搬运路径（`GM→A1`、`GM→B1`、`CO1→VECIN`、`VECOUT→GM` 等）缺少对应 intrinsic，fail-fast |
| `DialectRejected`             | 第一层：遇到白名单之外的 dialect 或 op                       |
| `StructuralBarrier`           | 第二层：存在阻断融合的结构性 barrier                         |
| `ScheduleFamilyNotSupported`  | 第三层：当前 kernel 无法匹配任何已注册 schedule family       |
| `NoValidScheduleInstance`     | 第三层：搜索后所有候选均被过滤                               |
| `CapacityOverflow`            | 第三/四层：tile 或 buffer 超过片上容量                       |
| `AlignmentViolation`          | 第四层：buffer 或 stride 不满足对齐约束                      |
| `NoValidPath`                 | 第四层：src → dst 之间无合法搬运路径                         |
| `GuardSplit`                  | 第三/四层：shape 条件导致需要分支决策                        |
| `WorkspacePackingDisabled`    | 第四层：静态 workspace 打包未启用，退化为独立 alloc           |
| `KernelABIMismatch`           | 第五层：kernel 签名与 host/runtime 协议不一致                |

### 7.5 Debug 开关

| 开关                                 | 含义                                           |
| ------------------------------------ | ---------------------------------------------- |
| `--ascend-debug-stage=<stage>`       | 只 dump 指定层或指定任务的产物                 |
| `--ascend-debug-report-dir=<dir>`    | 将结构化 report 写入目录                       |
| `--ascend-debug-annotate-ir`         | 在 IR 上启用 debug annotation                  |
| `--ascend-debug-keep-all-candidates` | 第三层保留全部候选，关闭 `topK` 裁剪，便于对比 |

### 7.6 Debug 与 Verifier 的分工

- **Verifier** 回答"当前 IR / 对象是否合法"
- **Debug 输出** 回答"为什么变成现在这个状态"

每一层均应同时输出：verifier 是否通过；若未通过，失败点在哪；若通过，关键对象摘要和裁剪 / 选择理由。

### 7.7 推荐问题定位路径

按以下顺序逐层缩小范围：

1. **入口问题**：检查第一层输出是否已满足白名单和符号化 shape 约束
2. **融合 / 划分问题**：检查第二层 `FusionCandidate[]`、`KernelPatternCandidate[]`、最终 `KernelPattern[]`
3. **调度问题**：检查第三层 `scheduleSearchSpace`、过滤原因和 `ScheduleDecisionSet`
4. **内存问题**：检查第四层 `MemoryRealizationPlan` 和 `resolvedMovements`
5. **Backend / Runtime 问题**：检查第五层 `AscendC Kernel MLIR`、`Host Tiling` 和可选 `Runtime Manifest`