# Ascend Kernelize Op Interface Design

## Goal

把当前内部 `KernelizeOpRegistry` 演进为正式、可扩展、可 review 的 Kernelize op 语义契约，使新增 op / dialect / view-like 形态不再需要修改 Kernelize 主干分析代码。

本设计只定义 Kernelize 入口层的 op semantic contract。它不把 `KernelPatternGraph`、candidate closure、fusion primitive、partition 决策等内部结构公开成稳定 API。

## Current State

当前代码已经完成第一轮去硬编码：

- `DependencyAnalysis` 不再按 `linalg.matmul` / `linalg.batch_matmul` / `linalg.transpose` 等具体 op name 收集目标 op。
- `KernelizeOpRegistry::buildDefault()` 默认注册一个 `"linalg"` model。
- `KernelizeOpRegistry` 通过 `matchLinalgOp + populateLinalgSemanticSummary` 生成 `OpSemanticSummary`。
- linalg 语义主要由 indexing maps、iterator kinds、DPS inputs/inits 和 result rank 推导。
- dependency traversal 已能跳过无 region 的 tensor-typed 中间 op，避免 view-like op 直接切断 producer/consumer 关系。

仍然存在的问题：

- `KernelizeOpRegistry` 是 `lib/Conversion/Ascend/Kernelize` 私有 function table，不是正式扩展契约。
- `OpSemanticSummary` 定义在 `DependencyAnalysis.h`，生命周期和字段形状是阶段内部实现，不适合作为 public API。
- 默认 model 仍只能从 Kernelize 主干源码注册；新增 dialect/op model 仍要改 `KernelizeOpRegistry.cpp`。
- unknown op / unsupported op 的 report 语义不够稳定，扩面后容易继续出现“静默不参与”。
- tensor view、layout transform、handwritten marker 等语义应该通过 trait / interface 表达，而不是散落在各分析器中。

## Design Decision

采用“三层契约”：

1. **Public semantic interface**
   - 新增薄 public header：`include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`。
   - 只暴露扩展方需要填写的 semantic info，不暴露 Kernelize 内部 graph/candidate 类型。

2. **External trait model registry**
   - 新增 public registry：`KernelizeOpModelRegistry`。
   - 用于给不能修改源码的 op 注册 external model，例如 upstream `linalg::LinalgOp`。
   - 默认 registry 在 `lib/Conversion/Ascend/Kernelize` 内注册 linalg structured model 和 tensor view model。

3. **Internal adapter**
   - `DependencyAnalysis` 只依赖 resolver 输出的 public semantic info。
   - 内部 adapter 把 public semantic info 转为当前 `OpSemanticSummary`。
   - 后续内部 candidate / role / schedule contract 可以继续重构，不破坏 public contract。

## Public Contract

新增 public 语义对象：

```c++
enum class KernelizeParticipationKind {
  Ignore,
  Analyze,
  Transparent,
  Unsupported,
};

enum class KernelizeSemanticTrait {
  Unknown,
  Structured,
  TensorView,
  LayoutTransform,
  HandwrittenGroup,
};

struct KernelizeOpSemanticInfo {
  KernelizeParticipationKind participation = KernelizeParticipationKind::Unsupported;
  AccessPatternKind accessPattern = AccessPatternKind::Unknown;
  SmallVector<IteratorKind, 4> iteratorKinds;
  SmallVector<AffineMap, 4> indexingMaps;
  SmallVector<unsigned, 2> resultRanks;
  SmallVector<KernelizeSemanticTrait, 4> traits;
  StringRef modelName = "unknown";
};
```

语义约定：

- `Analyze`：该 op 是 Kernelize 目标 op，会进入 `orderedOps`、role classification、candidate analysis。
- `Transparent`：该 op 不成为 kernel op，但 dependency traversal 可穿透它，例如 tensor view chain。
- `Ignore`：该 op 与 Kernelize 无关，既不报错也不进入分析。
- `Unsupported`：该 op 位于 Kernelize 相关 tensor dataflow 上，但没有可用 semantic model；必须进入 report/diagnostic，不允许静默吞掉。

`AccessPatternKind`、`IteratorKind` 继续复用当前 Kernelize 内部枚举，但要迁到公共 contract header 或一个 `Common` header 中。这样 Schedule / Realize 仍可复用同一语义词表，不再复制字符串。

## Resolver Order

Kernelize 语义解析按固定顺序执行：

1. op 自身实现的 `KernelizeOpInterface`
2. external `KernelizeOpModelRegistry`
3. builtin fallback model
4. unsupported / ignore classification

优先级必须稳定：

- 如果 op 实现了 interface，interface 结果优先。
- external model 只用于无法修改源码的 op，例如 upstream linalg、tensor、arith/memref view-like op。
- fallback 只做保守分类，不做复杂 pattern 推导。

## Interface Shape

首轮不强制引入 TableGen OpInterface。原因是当前目标主要是 upstream linalg / tensor view，无法直接给 upstream op 加 trait；external model registry 是最小可落地路径。

但 public API 要为后续 TableGen interface 预留同形函数：

```c++
class KernelizeOpInterface {
public:
  LogicalResult getKernelizeSemanticInfo(KernelizeOpSemanticInfo &info);
};
```

如果后续需要 AFIR 自有 dialect op 直接声明 interface，可以新增 `KernelizeOpInterfaces.td`，生成的 interface 方法填充同一个 `KernelizeOpSemanticInfo`。Resolver 层无需重写。

## Default Models

### Structured Linalg Model

默认 linalg model 保留当前能力：

- 通过 `linalg::LinalgOp` interface 匹配。
- 读取 indexing maps 和 iterator kinds。
- 根据 reduction iterator、DPS inputs/inits、result maps 判断 contraction / reduction。
- 根据 parallel indexing maps 判断 elementwise / broadcast / layout transform。
- 不按具体 `linalg.matmul` / `batch_matmul` / `transpose` / `fill` op name 分支。

修正要求：

- result rank 不再只取第一个 ranked result；要记录全部 ranked result ranks。
- 多 result rank 不一致时，semantic info 不能悄悄降级；要明确标记为 unsupported 或在 report 中说明。
- zero-rank maps 继续合法跳过。

### Tensor View Model

默认 tensor view model 覆盖：

- `tensor.extract_slice`
- `tensor.cast`
- `tensor.expand_shape`
- `tensor.collapse_shape`

该 model 输出：

- `participation = Transparent`
- `traits = [TensorView]`
- `accessPattern = LayoutTransform` 或 `Unknown`，不作为 schedulable role 种子。

该 model 的唯一职责是让 dependency traversal 以统一方式穿透 view-like op，不把 view-like 支持写散到各分析器里。

### Handwritten Group Model

`ascend.kernelize.handwritten_group` 保留为 marker-based model：

- marker op 或 marker attrs 通过 model 输出 `HandwrittenGroup` trait。
- model 不绕过 closure / contract / partition。
- MustCoLocate / MustSeparate 边仍由 pattern/partition 主链路消费。

## Dependency Analysis Changes

`DependencyAnalyzer` 改为两阶段：

1. resolve semantic info for all tensor/dataflow-relevant ops
2. build analyzed op index and producer/consumer graph

producer traversal 规则：

- 遇到 `Analyze` producer：作为 producer 记录。
- 遇到 `Transparent` producer：沿其 tensor operands 继续追溯。
- 遇到 `Ignore` producer：停止。
- 遇到 `Unsupported` producer 且其 result 被 Analyze op 消费：记录 unsupported diagnostic。

这样可以区分：

- “没有 linalg 上游”
- “上游只是 tensor.empty / arith.constant”
- “上游是未建模 tensor producer”

这三类当前容易混在一起。

## Reporting And Diagnostics

`emitDependencyAnalysisReport` 增加稳定字段：

- `participation`
- `model`
- `traits`
- `access`
- `result_ranks`
- `unsupported_reason`

诊断策略：

- 默认 pass 不因为无关 unknown op 失败。
- 如果 unknown/unsupported op 处在 Analyze op 的 tensor producer 链上，Kernelize 必须 fail-closed 或至少 emit error，首轮选择 fail-closed。
- Diagnostic anchor 优先使用具体 unsupported op；如果只知道 consumer，则 anchor consumer 并打印 producer chain summary。

## Public Header Boundary

新增 public 文件仅限：

- `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- 可选：`include/Conversion/Ascend/Kernelize/KernelizeOpModelRegistry.h`

保持不公开：

- `DependencyAnalysis.h`
- `KernelizeTypes.h`
- `KernelPattern.h`
- candidate / closure / merge / partition 内部类型

如果 registry 类型可以完全隐藏在 pass pipeline 内，public header 只暴露 registration hook：

```c++
void registerDefaultKernelizeOpModels(KernelizeOpModelRegistry &registry);
```

## CMake And Dialect Registration

首轮不新增 MLIR dialect 或 ODS 生成目标。

如果后续引入 TableGen interface：

- 新增 `include/Conversion/Ascend/Kernelize/KernelizeOpInterfaces.td`
- 新增 interface incgen target
- `AscendConversion` 依赖该 incgen target
- AFIR 自有 op 可选择实现该 interface

本轮优先使用 C++ external model registry，降低 build system 改动和 generated header 风险。

## Testing

### Unit Tests

新增或扩展 `AscendKernelPatternTest` / 新建 `AscendKernelizeOpInterfaceTest`：

- interface result 优先于 external registry。
- linalg external model 输出 `Structured` trait、iterator kinds、indexing maps。
- tensor view model 输出 `Transparent + TensorView`。
- unsupported tensor producer 被消费时产生 unsupported result。

### LIT

新增 focused LIT：

- `ascend-kernelize-op-interface-linalg.mlir`
  - generic contraction 不按 op name 分类。
  - multi-result rank mismatch 报明确 diagnostic。

- `ascend-kernelize-op-interface-tensor-view.mlir`
  - `tensor.extract_slice/cast/expand_shape/collapse_shape` 链可穿透。
  - view op 不被单独标成 kernel op。

- `ascend-kernelize-op-interface-unsupported.mlir`
  - 未建模 tensor producer 被 linalg consumer 使用时 fail-closed。
  - diagnostic 含 producer op name 和 consumer kernel context。

### Regression

继续跑：

- Kernelize focused LIT
- Schedule / Realize / full-pipeline focused LIT
- Ascend ctest
- `test/tools/check_ascend_no_v2_code_naming.sh`
- examples mainline smoke，至少 transformer 与一个普通 vector demo

## Migration Plan

1. 新增 public semantic info 和 internal resolver。
2. 把当前 `KernelizeOpRegistry` 改成 resolver 的 external model backend。
3. 移植 linalg model 到 default external model。
4. 新增 tensor view transparent model。
5. 改 `DependencyAnalysis` 消费 resolver 输出。
6. 更新 report 与 diagnostics。
7. 删除或降级旧 `KernelizeOpRegistry` 私有 API。

每一步都必须有 RED/GREEN 测试，不在同一提交中混入 candidate / schedule / realize 行为重构。

## Non-Goals

- 不在本阶段重写 candidate closure、fusion candidate、partitioner。
- 不把 Schedule / Realize 的 role enum 一并改完；只保证 Kernelize 输出 contract 可稳定扩展。
- 不修改 MLIR upstream op 源码。
- 不承诺完整 FlashAttention lowering；HandwrittenPattern 只保留 model/trait 入口。
- 不把 registry 做成跨动态库插件系统；首轮只支持编译期注册。

## Acceptance Criteria

本设计完成后的代码应满足：

- 新增 target op / view op model 可以通过注册 model 完成，不修改 `DependencyAnalysis.cpp` 主流程。
- linalg 默认 model 仍不按具体 op name 分支。
- tensor view traversal 由 model 驱动，而不是 hard-coded scattered checks。
- unsupported tensor producer 不再和普通 no-producer case 混淆。
- public header boundary 仍通过现有 boundary test。
- 现有 examples mainline 和 transformer smoke 不回退。
