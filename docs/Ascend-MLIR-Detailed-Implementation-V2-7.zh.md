## 7. E2E Debug 与可观测性

本节定义贯穿五层编译流程和 runtime 执行链路的统一调试规范。目标是：在不污染正式 IR 契约、不把 runtime 责任塞回 Conversion 的前提下，让开发者能按执行顺序查看编译产物、解释每个 kernel 的来源和决策、比较 sim / NPU / CPU reference 的数值结果，并快速定位第一个出错的 kernel 或语义边界。

本节的统一用户入口设计为 `ascend-debug`。`ascend-debug` 是 CLI-first 的调试编排工具；后续 CLI 仿真界面或 Web UI 只能组织参数、生成命令、展示产物，不应绕过 CLI 形成另一套后端语义。

### 7.1 职责边界

`lib/Conversion/Ascend` 不直接承担“跑数据、比较精度、回放 NPU”的职责。它应输出足够完整、稳定、可机器消费的编译期可追踪元数据：

- `kernel_id`
- source op / source value id
- source location
- schedule decision
- tile / tail plan
- memory place
- workspace slot
- emitted function
- cross-kernel edge

真正的 tensor dump、sim / NPU replay、CPU reference 对比、first-bad-kernel 定位执行，应放在 `lib/Runtime` / `runtime-session`，由 `ascend-debug` 统一编排。

| 模块 | 职责 |
| --- | --- |
| `ascend-mlir-opt` | 作为 Ascend pass pipeline 的编译入口，产出编号后的 stage MLIR、pass report、diagnostic、provenance 原始信息 |
| `lib/Conversion/Ascend` | 解释编译决策：kernelize、schedule、realize、translate/preemit 的选择、过滤、回退和映射 |
| `lib/Runtime` / `runtime-session` | 执行 sim / NPU、保存 actual outputs、读取 expected outputs、比较 tensor、保留 profile |
| 统一 diagnostics / debug 子系统 | 集中承载 debug graph、kernel DAG、stage graph、memory、diff、locate 等调试能力；具体目录名可随工程组织调整，但不应分散为多套脚本入口 |
| `ascend-debug` | 用户面对的唯一通用调试入口，负责编排 collect / open / diff / locate / graph workspace |

### 7.2 Debug 输出机制

统一提供五类输出方式，彼此独立：

| 方式 | 用途 | 是否进入正式 IR |
| --- | --- | --- |
| `stage dump` | 按执行顺序导出每个阶段的输入 / 输出 IR、对象摘要和 verifier 结果 | 否 |
| `debug report` | 结构化导出分析结果、候选列表、决策和过滤原因 | 否 |
| `debug annotation` | 在 IR 上临时标注关键分析结果，辅助 `ascend-mlir-opt` / 打印阅读 | 否，默认仅 debug 模式启用 |
| `provenance` | 记录 source value / semantic boundary / kernel / runtime task / checkpoint 的跨阶段映射 | 否，作为外部 artifact |
| `runtime evidence` | 保存 actual tensor、expected tensor、diff summary、profile 和 runtime log | 否，属于 runtime artifact |

**约束：**

- 正式编译链路不依赖任何 debug annotation。
- debug annotation 是只读观察视图，后续 pass 不得以其内容作为真相来源。
- 阶段内对象（`OpSemanticSummary`、`KernelPatternCandidate[]` 等）默认仅存在于 analysis cache 或 report 中，不写入正式 IR。
- `provenance` 是跨工具调试契约，不是优化决策输入；优化 pass 不应读取 `provenance` 反向影响编译结果。
- runtime tensor dump 只由 runtime 侧产生；Conversion 只能声明哪些 boundary 可作为 checkpoint。

### 7.3 统一 CLI：`ascend-debug`

`ascend-debug` 第一版只暴露四个主命令：

| 命令 | 目的 | 核心输出 |
| --- | --- | --- |
| `collect` | 收集一次 debug run 的编译期和可选运行期证据 | `debug-run/` 目录 |
| `open` | 打开或生成只读 dashboard | `index.html`、图和 summary |
| `diff` | 比较最终输出或 checkpoint tensor | `diff-summary.json` / 文本摘要 |
| `locate` | 根据 DAG 和 diff 结果定位 first-bad-kernel / first-bad-boundary | `locate-summary.json` / 文本摘要 |

推荐默认路径：

```bash
ascend-debug collect model.mlir --out debug-run/
ascend-debug open debug-run/
ascend-debug diff debug-run/ --actual npu --expected cpu
ascend-debug locate debug-run/ --strategy topo
```

`collect` 不再拆更多子命令，而是用 preset 控制深度：

| preset | 行为 |
| --- | --- |
| `quick` | 默认。只收编译期 artifacts：stage MLIR、report、provenance、manifest、图数据。不跑 NPU，不 dump 中间 tensor |
| `accuracy` | 在 `quick` 基础上运行最终输出比较，保留 actual / expected tensor 和 diff summary |
| `deep` | 在 `accuracy` 基础上开启 per-kernel checkpoint，保留 first-bad-kernel 定位所需数据 |

常用参数应保持少量：

| 参数 | 含义 |
| --- | --- |
| `--out <dir>` | debug run 输出目录 |
| `--preset quick|accuracy|deep` | debug 深度 |
| `--backend sim|npu` | runtime 后端 |
| `--reference cpu|sim` | diff 参考来源 |
| `--device-id <id>` | 本次 NPU run 使用的单个 device，优先级高于 `ASCEND_DEVICE_ID` |
| `--atol <value>` / `--rtol <value>` | 数值比较容忍度 |

高级参数允许存在，但不应成为新手路径。例如：`--checkpoint none|final|kernel|boundary|anchor`、`--focus kernel_0007`、`--pipeline <pipeline>`、`--keep-temp`。

### 7.4 Stage Artifact 目录与编号

`collect` 产出的 artifacts 必须有稳定目录结构，并且 stage MLIR 文件必须按执行顺序编号。编号采用零填充和预留号段，方便人类按文件名排序阅读，也方便后续插入子阶段。

推荐目录：

```text
debug-run/
  manifest.json
  provenance.json
  index.html
  stages/
    000-source.mlir
    010-normalize-in.mlir
    019-normalize-out.mlir
    020-kernelize-in.mlir
    029-kernelize-out.mlir
    030-schedule-in.mlir
    039-schedule-out.mlir
    040-realize-in.mlir
    049-realize-out.mlir
    050-translate-in.mlir
    059-preemit-out.mlir
  reports/
    020-kernelize.report.txt
    030-schedule.report.txt
    040-realize.report.txt
    050-translate.report.txt
  graphs/
    020-kernelize.ir.svg
    060-kernel-dag.svg
    061-memory-timeline.svg
  tensors/
    final/
    checkpoints/
  profiles/
  summaries/
    diff-summary.json
    locate-summary.json
```

`manifest.json` 必须保存同一份 stage 顺序，工具不得依赖文件系统排序作为唯一真相。

推荐最小字段：

```json
{
  "schema_version": 1,
  "tool": "ascend-debug",
  "input": "model.mlir",
  "preset": "deep",
  "backend": "npu",
  "device_id": 5,
  "device_scope": "single_run_single_device",
  "stages": [
    {"order": 0, "name": "source", "path": "stages/000-source.mlir"},
    {"order": 10, "name": "normalize-in", "path": "stages/010-normalize-in.mlir"},
    {"order": 19, "name": "normalize-out", "path": "stages/019-normalize-out.mlir"}
  ]
}
```

### 7.5 Provenance 与 Semantic Boundary

`provenance` 的核心抽象是 `semantic boundary`。它不是默认把原图物理拆成多个小图，而是记录一个可由 CPU reference 和 NPU / sim 共同理解的语义对比点。

一个 boundary 必须回答：

- 这个 boundary 覆盖哪些 source ops / source values？
- 它对应哪个 kernel 或 debug anchor？
- 它的 runtime task 是什么？
- 它的输出 checkpoint 在哪里？
- 它对应的 schedule、tile/tail、memory、workspace 决策在哪里？

示例：

```text
source graph:
  op12: add
  op13: relu
  op14: mul
  op15: reduce

kernelized:
  kernel_0007 = fuse(op12, op13, op14)
  kernel_0008 = op15
```

对应 boundary：

```json
{
  "id": "boundary.kernel_0007.out0",
  "source_ops": ["op12", "op13", "op14"],
  "source_values": ["%14"],
  "kernel_id": "kernel_0007",
  "runtime_task": "kernel_0007",
  "emitted_function": "kernel_0007_func",
  "checkpoint": "tensors/checkpoints/kernel_0007.out0.npy",
  "schedule": "reports/kernel_0007.schedule.json",
  "memory": "reports/kernel_0007.memory.json"
}
```

这样即使 CPU reference 和 NPU kernel 数量不同，`locate` 也不要求两边 kernel 一一对应。它只比较同一个 semantic boundary 上的 tensor 是否一致。

推荐 boundary 粒度：

| 粒度 | 用途 | 默认开启 |
| --- | --- | --- |
| kernel group boundary | 每个 fused kernel 覆盖的一组 source ops 的输出 | `deep` 开启 |
| source op boundary | 单个 source op 输出 | 否，数据量大 |
| debug anchor boundary | 失败 fusion group 内部强制物化的中间值 | 仅 `locate --expand` |

### 7.6 精度 Diff 与 First-Bad 定位

`diff` 分为四层能力：

| 层级 | 对比对象 | 难度 | 默认行为 |
| --- | --- | --- | --- |
| L0 final output | 最终输出 tensor | 低 | `accuracy` 默认 |
| L1 kernel checkpoint | runtime task / kernel 输出 | 中 | `deep` 默认 |
| L2 semantic group | fused kernel 覆盖的 source op group 输出 | 高 | 需要 provenance 和 CPU semantic runner |
| L3 debug anchor | fusion group 内部强制物化中间值 | 高 | 仅失败后按需重编译 |

NPU 与 CPU 对比不应由 NPU path 直接跑 CPU。`ascend-debug collect` 应用同一组输入生成三类可比较证据：

- NPU actual tensor
- sim actual tensor
- CPU reference tensor

`diff` 只消费 tensor 文件和 metadata，输出 max abs diff、mean abs diff、top offenders、shape/dtype mismatch 和 tolerance 结果。

`locate` 的默认策略：

1. 先检查最终输出是否失败。
2. 若失败且存在 per-kernel checkpoint，按 kernel DAG 拓扑顺序扫描。
3. 对每个 boundary，若所有前驱通过而当前 boundary 失败，则标记为 first-bad-boundary / first-bad-kernel。
4. 若失败 kernel 是 fusion group，输出 source ops、schedule、tile/tail、memory place、workspace slot、runtime task。
5. 若需要进一步缩小到 fusion 内部，用户显式运行 `locate --expand kernel_0007`，触发 debug materialization 或禁用局部 fusion 的二次 `collect`。

`locate` 不应默认全图插入内部 anchor。全量 anchor 会显著改变编译结果、数据量和性能，只应作为失败区域的二次定位手段。

### 7.7 可视化能力

可视化分三类：

| 视图 | 数据来源 | 是否可复用 MLIR 基础能力 |
| --- | --- | --- |
| stage IR graph | `stages/*.mlir` | 是。可复用 MLIR Graphviz / op graph 思路，但应接入 `ascend-mlir-opt` 工作流 |
| unified debug graph / kernel DAG | runtime manifest、run manifest、provenance、kernelized IR、stage graph、diff / locate / memory summary | 否。需要 Ascend 语义字段，由统一 diagnostics / debug 子系统内部实现 |
| memory timeline | Realize report、memory plan、workspace slot、movement edge | 部分。可借鉴 liveness / bufferization 思路，但 Ascend memory place 和 workspace reuse 需要自研 |

`ascend_kernel_dag_viz` 不再作为调试入口或底层依赖。kernel DAG 能力应由统一 diagnostics / debug 子系统承载，并由 `ascend-debug collect/open` 统一调用；旧独立脚本应删除，避免形成多套 debug 工具。

统一 debug graph 应继续增强：

- node 显示 kernel id、kernel kind、op summary、output shape、symbolic tile params、workspace size
- edge 显示 value / boundary / checkpoint 信息
- 标注 critical path、prepack root、simple fusion hint
- 接入 diff summary，高亮 failed / first-bad boundary
- 接入 memory summary，高亮 workspace reuse 和 movement edge

`*.mlir` 可视化属于通用 IR 结构视图，不能替代 Ascend kernel DAG 视图。IR graph 回答“IR 中 op/value 如何连接”，kernel DAG 回答“最终哪些 runtime kernel 如何依赖、为何这样划分、哪里开始错”。

### 7.8 各层可观测对象

| 层 | 应可观测的主边界对象 |
| --- | --- |
| 第一层：Normalize | 规范化后的入口 module、被拒绝的方言 / 结构、保留的结构属性 |
| 第二层：Kernelize | `ProducerConsumerIndex`、`OpSemanticSummary`、结构标记属性、`OpRoleMap`、`FusionCandidate[]`、`KernelPatternCandidate[]`、最终 `KernelPattern[]`、kernel DAG edge |
| 第三层：Schedule | `CoalescedAxisInfo`、`ScheduleProblem`、`scheduleTemplate`、`scheduleSearchSpace`、过滤原因、`ScheduleDecisionSet`、tile/tail plan |
| 第四层：Realize | `BufferizedKernelIR`、`PlacementPlan`、`StaticMemoryPlan`、`MovementPlan`、`MemoryRealizationPlan`、workspace slot、live interval |
| 第五层：Translate / PreEmit | `Backend Compute IR`、`AscendC Kernel MLIR`、`Host Tiling`、`Runtime Manifest`、emitted function、ABI、artifact path、runtime task mapping |
| Runtime | run manifest、task graph、actual outputs、expected outputs、profile、launch log、validation result |

### 7.9 各层 Debug 重点

#### 7.9.1 第二层：Kernelize

重点回答：某个 op 为何被标记为某个 `OpRole`、某个候选为何能形成或被裁掉、最终 `KernelPattern` 为何拆成当前这组 region。

**最少输出内容：**

| 对象 | 最少输出字段 |
| --- | --- |
| `OpSemanticSummary` | `resultShape`、`indexingMaps`、`iteratorTypes`、`accessPatternKind`、`semanticAttrs` |
| `FusionCandidate` | `seed`、内部 op 集合、外部输入 / 输出、合法性结果、收益分数 |
| `HorizontalFusionCandidate` | `siblingCandidates`、`sharedInputs`、`perGroupContracts`、合法性结果（互不可达校验、组内候选数）、合并决策原因 |
| `KernelPatternCandidate` | `primaryOps`、`roles`、`primitives`、边界输入输出、拆分原因 |
| `KernelPattern` | `kernel_id`、覆盖的 source ops / values、输入输出边界、DAG 前驱 / 后继 |

**`OpSemanticSummary` 生命周期：**

- 覆盖第二层全程，可选延伸至第三层 `ScheduleProblemBuilder`。
- 一旦有 pass 修改 `resultShape`、`indexingMaps`、`iteratorTypes`、结构属性或 region 边界，旧 summary 立即失效。
- 进入 `Structured Lowering` 前不再复用旧 summary。

#### 7.9.2 第三层：Schedule

重点回答：为何选择了某个 `scheduleTemplate`、某些 `ScheduleInstance` 为何被保留或裁掉、最终 `ScheduleDecisionSet` 的 guard 和参数从何而来。

**最少输出内容：**

| 对象 | 最少输出字段 |
| --- | --- |
| `CoalescedAxisInfo` | 逻辑轴、原始轴来源、extent、barrier 原因 |
| `ScheduleProblem` | 轴集合、四类约束摘要、关键 shape 关系 |
| `scheduleSearchSpace` | 候选总数、每个候选的 primitive 组合、`loadOrder`、`computeOrder`、cache 选择 |
| 过滤结果 | `StructuralFilter`、`Shape/HardwareFilter`、`MemoryFilter`、`Dedup`、`DominancePrune` 各自裁掉的候选和原因 |
| `ScheduleDecisionSet` | `decisionId`、guard 表达式、`scheduleInstance`、tiling 参数、tail policy、`cachePlan`、`pipelineDepthExpr` |

#### 7.9.3 第四层：Realize

重点回答：某个值为何被放到某个 `MemoryPlace`、某条搬运路径为何存在、`cachePlan` / `promotionPlan` / `placement` 三者是否一致、workspace 如何复用。

**最少输出内容：**

| 对象 | 最少输出字段 |
| --- | --- |
| `MemoryRealizationPlan` | `resolvedPlacement`、`workspaceLayout`、`resolvedMovements`、materialized alloc/copy 列表 |
| `resolvedMovements` | `src/dst`、`pathKind`、`loopRegion`、`guard` |
| `StaticMemoryPlan` | slot id、live interval、size、alignment、reuse group、peak usage |
| `TargetMemoryModel`（参照） | path 可达性、容量规则、对齐约束 |

`ascend-debug open` 读取 `reports/040-realize.report.txt` 时，应优先使用 `StaticMemoryPlan` 的 `live_interval[]`、`workspace_slot[]` 与 `MovementPlan` 的 `movement_step[]` 明细生成 `summaries/memory.json`。该 summary 至少包含：

- 每个 kernel 的 workspace slot、value live range、physical offset、memory place、byte size。
- 按 `(place, offset)` 聚合出的 workspace reuse group。
- 按 live interval 扫描得到的 peak timeline 和 peak workspace bytes。
- movement edge 摘要，说明 GM 到片上 workspace 的数据搬运路径。

只有当 Realize report 不含 slot/lifetime 明细时，debug dashboard 才回退到从 kernel DAG 读取 `workspace_size` 的粗粒度 overview。

#### 7.9.4 第五层：Translate / PreEmit

重点回答：backend lowering 是否忠实保留第三、四层决策；host / kernel / runtime 的 schema 是否一致；动态 shape 下 runtime 如何最终选中某个 decision。

**最少输出内容：**

| 对象 | 最少输出字段 |
| --- | --- |
| `AscendC Kernel MLIR` | kernel 参数列表、buffer 顺序、tiling params、guard 绑定情况 |
| `Host Tiling` | 每个 tiling 参数的来源表达式 |
| `Runtime Manifest` | `kernel_id`、artifact path、emitted function、ABI、shape/tile policy、workspace size |
| `Run Manifest` | task id、dependencies、inputs、outputs、expected outputs、backend、device id |
| `TranslateReport` | lowering decision、unsupported fallback、emitted artifact、runtime task mapping |

当前 Normalize / Kernelize / Schedule / Realize 已有较多 report 基础；Translate / PreEmit 需要补齐同等级别的 lowering decision report，避免末端仅依赖 IR 和 error 定位。

### 7.10 Diagnostics 规范

所有阶段的 diagnostics 必须包含以下最小字段集：

| 字段 | 含义 |
| --- | --- |
| `stage` | 所属阶段，取值为 `Normalize`、`Kernelize`、`Schedule`、`Realize`、`Translate`、`Runtime` 之一 |
| `objectId` | 绑定的最小对象标识，例如 `opName`、`candidateId`、`variantId`、`GuardedBufferKey`、`kernelName`、`boundaryId` |
| `reasonKind` | 失败或回退原因类别，必须来自稳定枚举集合，不允许使用自由文本 |
| `message` | 面向人类阅读的简短原因摘要 |
| `isRecoverable` | 是否允许当前阶段回退并继续主链 |
| `fallbackTaken` | 若可回退，实际采用的回退路径；不可回退时为空 |

**附加约束：**

- 若与动态 shape 或 guard 相关，额外记录 `guardExpr` 或 `shapeBucketKey`。
- 若由 target 限制触发，额外记录最小 target 上下文（`memoryPlace`、`pathKind` 或 `intrinsicName`）。
- 若与精度定位相关，额外记录 `boundaryId`、`kernel_id`、`runtime_task` 和 checkpoint path。
- 回退成功时也必须保留 diagnostics，供 debug 和缓存负结果使用。

**当前版本最小 `reasonKind` 枚举集合：**

| `reasonKind` | 触发场景 |
| --- | --- |
| `TargetProfileMissingField` | 构造期（Target 建模）：SoC config 缺少必要字段，fail-fast |
| `TargetPathIntrinsicMissing` | 构造期（Target 建模）：必需搬运路径（`GM→A1`、`GM→B1`、`CO1→VECIN`、`VECOUT→GM` 等）缺少对应 intrinsic，fail-fast |
| `DialectRejected` | 第一层：遇到白名单之外的 dialect 或 op |
| `StructuralBarrier` | 第二层：存在阻断融合的结构性 barrier |
| `ScheduleFamilyNotSupported` | 第三层：当前 kernel 无法匹配任何已注册 schedule family |
| `NoValidScheduleInstance` | 第三层：搜索后所有候选均被过滤 |
| `CapacityOverflow` | 第三/四层：tile 或 buffer 超过片上容量 |
| `AlignmentViolation` | 第四层：buffer 或 stride 不满足对齐约束 |
| `NoValidPath` | 第四层：src → dst 之间无合法搬运路径 |
| `GuardSplit` | 第三/四层：shape 条件导致需要分支决策 |
| `WorkspacePackingDisabled` | 第四层：静态 workspace 打包未启用，退化为独立 alloc |
| `KernelABIMismatch` | 第五层：kernel 签名与 host/runtime 协议不一致 |
| `RuntimeValidationFailed` | Runtime：actual tensor 与 expected/reference tensor 超出 tolerance |
| `CheckpointUnavailable` | Runtime：请求的 boundary / kernel 无法提供 checkpoint |

### 7.11 Pass 级 Debug 开关

`ascend-debug` 是用户入口；pass 级开关仍应保留给开发者和 lit 测试使用。现有和推荐 pass 级能力包括：

| 开关 / 选项 | 含义 |
| --- | --- |
| `debug-stage=<stage>` | 在具体 pass 内选择 debug stage |
| `dump-report=true` | 将该 pass 的结构化 report 打印或写出 |
| `--mlir-print-ir-before/after` | 复用 MLIR 通用 IR 打印能力 |
| `--mlir-print-ir-after-all` | 复用 MLIR 通用 pass 后 IR dump 能力 |
| pass-specific tuning/debug options | 例如保留候选、限制 top-k、指定 target tile policy |

`ascend-debug collect` 可以编排这些 pass 级能力，但不应把所有底层 pass 参数直接暴露成新手路径。必要时通过 `--pipeline`、`--extra-opt-arg` 或 debug profile 文件传递。

### 7.12 MLIR 可复用能力与自研能力

可复用 MLIR 的能力：

- IR 打印和 pass 前后 dump。
- pass pipeline、pass timing、pass statistics。
- diagnostic、source location、verification。
- Graphviz / op graph 类通用 IR 可视化思路。
- liveness / bufferization 的基础分析思想。

必须自研的 Ascend 语义调试能力：

- stage artifact 目录、编号和 manifest 约定。
- source op / semantic boundary / kernel / runtime task 的 provenance map。
- schedule / tile / tail 决策解释。
- Ascend memory place、workspace slot、movement、reuse 的生命周期可视化。
- runtime artifact / run manifest / task graph 的统一浏览。
- per-kernel checkpoint、sim / NPU / CPU reference diff。
- first-bad-kernel 和 fusion group 内部二次定位。

### 7.13 单 Device 约束

`ascend-debug` 第一版只支持单次 run 选择单个 device：

```bash
ascend-debug collect model.mlir --backend npu --device-id 5 --out debug-run/
```

语义固定为：

- one debug run
- one process invocation
- one selected NPU device
- one run manifest
- one provenance graph

不在第一版支持：

- `--devices 5,7`
- per-task device placement
- cross-device task graph
- device-to-device tensor movement
- multi-device context / stream pool

如果要比较不同 device 的结果，应由外层 CI 或脚本分别启动多次独立 `ascend-debug collect`，每次绑定一个 `--device-id`，再用 `ascend-debug diff` 比较两个 run 目录。

### 7.14 Debug 与 Verifier 的分工

- **Verifier** 回答“当前 IR / 对象是否合法”。
- **Debug 输出** 回答“为什么变成现在这个状态”。
- **Diff** 回答“数值是否一致”。
- **Locate** 回答“第一个可观测错误 boundary / kernel 在哪里”。

每一层均应同时输出：verifier 是否通过；若未通过，失败点在哪；若通过，关键对象摘要和裁剪 / 选择理由。

### 7.15 推荐问题定位路径

按以下顺序逐层缩小范围：

1. **入口问题**：检查第一层输出是否已满足白名单和符号化 shape 约束。
2. **融合 / 划分问题**：检查第二层 `FusionCandidate[]`、`KernelPatternCandidate[]`、最终 `KernelPattern[]` 和 kernel DAG edge。
3. **调度问题**：检查第三层 `scheduleSearchSpace`、过滤原因、`ScheduleDecisionSet`、tile/tail plan。
4. **内存问题**：检查第四层 `MemoryRealizationPlan`、`resolvedMovements`、workspace slot timeline、peak usage。
5. **Backend / Runtime 问题**：检查第五层 `AscendC Kernel MLIR`、`Host Tiling`、`Runtime Manifest`、run manifest、ABI 和 artifact。
6. **最终精度问题**：运行 `ascend-debug diff` 比较 final output。
7. **中间精度问题**：运行 `ascend-debug collect --preset deep` 生成 checkpoint，再用 `ascend-debug locate` 找 first-bad-kernel。
8. **融合内部问题**：对失败 kernel 运行 `locate --expand <kernel_id>`，通过 debug anchor 或局部禁用 fusion 做二次定位。
