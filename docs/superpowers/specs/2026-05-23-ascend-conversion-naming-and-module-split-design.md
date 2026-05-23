# Ascend Conversion Naming And Module Split Design

## Goal

收敛 `lib/Conversion/Ascend` 当前主线代码的内部命名，并拆分已经明显过大的实现文件，使第五层 `Translate` 的代码接口继续稳定、但实现局部性更好。

本设计保持所有 public pass / CLI / LIT RUN surface 不变。`--ascend-compute-lower`、`--ascend-parallelize`、`--ascend-prepare-for-emit`、`--ascend-canonicalize-cann-signature` 仍是外部入口；本轮只调整内部文件、类型、函数、注释和文档中的当前主线命名。

## Current State

当前顶层目录已经基本匹配主线五层流程：

- `Normalize`
- `Kernelize`
- `Schedule`
- `Realize`
- `Translate`
- `Debug`

近期代码已经把原先的 backend layer 移到 `Translate` 下，外部入口也已经收敛到 versionless `ascend-*` pass。剩余问题集中在三个方面：

1. `Phase5` 仍作为当前代码词汇出现，但它只表达历史开发阶段，不表达行为。
2. `Translate/KernelIR` 内部命名与实际职责存在偏差；当前 pass 名是 `ascend-compute-lower`，实现实际构造 AscendC / EmitAsc / SCF 形式，而不是一个清晰命名的独立 `KernelIR` dialect。
3. 部分实现文件过大，维护者需要在单文件中同时理解多个 op family 或 bridge 规则。

当前最大的大文件为：

- `lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp`
- `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- `lib/Conversion/Ascend/Translate/PreEmit/PrepareForEmit.cpp`
- `lib/Conversion/Ascend/Translate/KernelIR/DataMovementConversion.cpp`
- `lib/Conversion/Ascend/Schedule/ScheduleSearch.cpp`
- `lib/Conversion/Ascend/Kernelize/KernelPattern.cpp`

本轮优先处理前两个。`PrepareForEmit.cpp` 体量较大，但它当前仍是连续 ABI transform；`ScheduleSearch.cpp` 和 `KernelPattern.cpp` 职责清楚，暂不作为首轮拆分目标。

## Decision

采用“内部模块化，外部入口不变”的策略。

### Keep Stable

以下 surface 本轮不改：

- pass 名和 CLI 参数
- test RUN 命令
- public pass factory 名称
- `include/Conversion/Ascend/Passes.td` 中 pass argument
- runtime artifact 文件名和 example script 输出名，除非只是文档说明

### Rename Current Code Vocabulary

`Phase5` 不再作为当前代码命名使用。历史 specs、plans、tracking 中用于记录开发批次的 `Phase 5` / `Phase 5C` / `Phase 5C+` 可以保留。

当前代码和当前状态文档中的行为词汇使用以下替换方向：

| Old wording | New wording | Meaning |
|---|---|---|
| `Phase5Bridge` | `TranslateMemoryBridge` | Realize 产物到 Translate/ComputeLower 可消费内存形态的桥接 |
| `Phase5BridgeMaterializationCounts` | `TranslateBridgeMaterializationCounts` | bridge 实际插入 alloc/copy 的计数 |
| `materializePhase5Bridge` | `materializeTranslateMemoryBridge` | 物化 Translate 可消费的 output / movement bridge |
| `classifyPhase5ReductionBody` | `classifyBackendReductionBody` | 判断 reduction body 对 backend lowering 的 compute kind |
| `isSupportedPhase5VectorOutput` | `isSupportedBackendVectorOutput` | 判断 backend lowering 可消费的 final vector output |
| `isSupportedPhase5GatherOutput` | `isSupportedBackendGatherOutput` | 判断 backend lowering 可消费的 final gather output |
| `isSupportedPhase5FinalOutput` | `isSupportedBackendFinalOutput` | 判断 backend lowering 可消费的 final output |
| `Phase 5 compute lowering` | `Translate compute lowering` or `AscendC compute lowering` | 第五层 compute lowering 行为 |

`Phase` 一词可继续用于单个函数内部的步骤注释，例如 `LinalgToKernelIR.cpp` 中 `Phase 0: Build Buffer Context`、`Phase 1: Movement Lowering`。这些是局部算法步骤，不是历史开发批次。

## Compute Lowering Split

保留外部接口：

```c++
LogicalResult convertCompute(func::FuncOp funcOp, AscendCBufferContext &ctx);
```

`ComputeOpConversion.cpp` 变成 compute lowering 编排模块，不再承载所有 op family 的具体实现。

新增内部头：

```text
lib/Conversion/Ascend/Translate/KernelIR/ComputeLoweringInternal.h
```

该头只在 `Translate/KernelIR` 内部使用，暴露 family-level helpers，不成为 public include surface。

首轮拆分建议：

```text
ComputeOpConversion.cpp
ComputeElementwiseLowering.cpp
ComputeReductionLowering.cpp
ComputeMatmulLowering.cpp
ComputeFillLowering.cpp
ComputeGatherLowering.cpp
ComputeScalarFallbackLowering.cpp
```

如果实现中某个 family 仍高度耦合，允许先拆成较少文件，但必须让 `ComputeOpConversion.cpp` 明显瘦身，并让每个文件名对应明确 lowering family。

### Compute Split Interface

内部 helper 采用 `lower*` 动词命名，按 family 接收同一个 lowering context：

```c++
LogicalResult lowerElementwiseComputes(func::FuncOp funcOp,
                                       AscendCBufferContext &ctx);
LogicalResult lowerReductionComputes(func::FuncOp funcOp,
                                     AscendCBufferContext &ctx);
LogicalResult lowerMatmulComputes(func::FuncOp funcOp,
                                  AscendCBufferContext &ctx);
LogicalResult lowerFillComputes(func::FuncOp funcOp,
                                AscendCBufferContext &ctx);
LogicalResult lowerGatherComputes(func::FuncOp funcOp,
                                  AscendCBufferContext &ctx);
LogicalResult lowerScalarFallbackComputes(func::FuncOp funcOp,
                                          AscendCBufferContext &ctx);
```

首轮不要求构造新的 class hierarchy。现有 code path 以 MLIR `func.walk` 和 in-place IR mutation 为主，函数模块比抽象基类更符合当前实现。

## Realize Bridge Split

保留 `MemoryRealizationDriver` 作为 Realize materialization 编排接口。

从 `MemoryRealizationDriver.cpp` 拆出 Translate memory bridge：

```text
lib/Conversion/Ascend/Realize/TranslateMemoryBridge.h
lib/Conversion/Ascend/Realize/TranslateMemoryBridge.cpp
```

`TranslateMemoryBridge` 负责：

- 识别 backend final output 是否需要 VECOUT bridge
- 识别 cube-to-vector bridge 形态
- 进行 preflight，避免半插入 IR
- 插入 bridge alloc/copy
- 返回 per-kernel `TranslateBridgeMaterializationCounts`

`MemoryRealizationDriver` 负责：

- plan 校验
- memory-space annotation
- selected movement materialization
- 调用 `TranslateMemoryBridge`
- 汇总并回写 `MemoryRealizationPlan`

这样 `MemoryRealizationDriver` 的外部 interface 仍然深，但 bridge 规则的实现局部性更强。

## Translate / KernelIR Naming

本轮不移动 `Translate/KernelIR` 目录。

原因：

- 最近已有一次 `Move Ascend backend layer under translate`，继续移动目录会产生大量 include path diff。
- public pass 名已经比目录名更稳定：`ascend-compute-lower` 表达了用户可见行为。
- 当前更大的维护问题是大文件和 `Phase5` 词汇漂移，而不是目录本身。

本轮做轻量收敛：

- 文件 banner / 注释中把 `KernelIR` 解释为 internal AscendC kernel construction surface。
- public 或 near-public 名称优先使用 `ComputeLowering`、`BackendSupportMatrix`、`PreEmit`。
- 不新增新的 `KernelIR*` class 名，除非它确实表示该内部 surface 的 shared utility。

如果后续仍认为目录名阻碍理解，再单独设计 `Translate/ComputeLower` 或 `Translate/BackendCompute` 目录迁移。

## Normalize Documentation Fix

`NormalizePass` 当前已经执行输入 dialect gate 并打 `ascend.normalized`。当前 `Passes.td` 仍有 skeleton / MVP 描述。

本轮更新当前 pass 描述：

- 从 “skeleton pass” 改为 “validates supported entry dialects and stamps normalized marker”
- 保持 pass 行为不变

## Non-Goals

本轮不做：

- 修改 pass CLI 名称
- 删除历史 `Phase 5` spec / plan 文件
- 大规模重写 tracking 文档历史段落
- 移动 `Translate/KernelIR` 目录
- 重写 compute lowering 算法
- 引入新的 polymorphic lowering class hierarchy
- 改 runtime-session、artifact schema 或 real-NPU workflow

## Migration Order

按低风险到高风险推进：

1. 更新当前代码注释和 `Passes.td` 描述，去掉当前代码中的历史 `Phase5` 表述。
2. 重命名 `LinalgBodyClassifier` 当前代码中的 `Phase5*` 函数，并同步 Realize / ComputeLower call sites。
3. 拆出 `TranslateMemoryBridge`，保持 `MemoryRealizationDriver::materialize(...)` 外部行为不变。
4. 拆 `ComputeOpConversion.cpp` 的 least-coupled family，先从 elementwise / fill / matmul 或 reduction 中最容易独立的一组开始。
5. 继续拆剩余 compute family，直到 `ComputeOpConversion.cpp` 只承担 orchestration 和少量 shared dispatch。
6. 根据实际 diff 更新当前 tracking 行中的现役命名，历史计划链接保持原名。

每一步都应保持 buildable；不要把所有文件搬迁和所有函数重命名压成一个不可 review 的改动。

## Testing

首轮验证分层执行。

Focused build/lit:

```text
ninja -C build afir-opt AscendBackendSupportMatrixTest
ctest --test-dir build -R 'AscendBackendSupportMatrix|Ascend.*Registry' --output-on-failure
llvm-lit -v build/test/Conversion/ascend-compute-lower*.mlir
llvm-lit -v build/test/Conversion/ascend-realize*.mlir
```

Pipeline smoke:

```text
llvm-lit -v build/test/Conversion/ascend-full-pipeline*.mlir
llvm-lit -v build/test/Target/cann-translate*.mlir
```

Repository baseline, when the implementation touches shared conversion behavior:

```text
bash test/tools/runtime/run_runtime.sh
bash test/tools/runtime/run_simbackend_examples.sh
bash test/tools/examples/example_pipelines.sh
```

All authoritative verification remains on xvm under `/home/niu/code/Ascend-MLIR`.

## Review Checklist

- No public pass argument changed.
- No test RUN command changed only because of internal naming.
- `Phase5` no longer appears in current `include/Conversion/Ascend` or `lib/Conversion/Ascend` symbol names.
- `ComputeOpConversion.cpp` is no longer the sole home for every compute family.
- `MemoryRealizationDriver.cpp` no longer owns Translate memory bridge pattern matching details.
- New internal file names describe behavior, not historical milestone names.
- Any doc update distinguishes current code vocabulary from historical plan names.
