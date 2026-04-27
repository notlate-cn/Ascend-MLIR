# Vector Plan Generation — Phase 1 实施方案（修订版）

**目标：** 在当前 Ascend-MLIR 管线中落地一个**自洽、可验证、可逐步扩展**的 Vector Phase 1 实现方案：

- 先覆盖 **静态 shape + 经过预处理后已基本规整的 linalg 输入**；
- 正确完成 **chain analysis → golden axis 分析 → IR 级 collapse → tile+fuse realization → metadata emission**；
- 让产物能直接进入现有 Phase 2 lowering pipeline；
- 对完整 spec 中最重的部分（完整 Step 4 轴归一化、kernel outline、多 plan 枚举）明确延后到后续版本，而不是在 v1 中模糊承诺。

**Spec：** `docs/superpowers/specs/2026-04-09-mlir-ai-compiler-generalization-design.md`

---

## 1. 本方案的范围与边界

## 1.1 本版要做什么

这版实施方案覆盖以下能力：

1. **Step 1 预处理**
   - `--eliminate-cf-assert`
   - `--canonicalize-extract-broadcast`
   - 复用现有 `--linalg-fold-unit-extent-dims`
   - 复用现有 `--linalg-fuse-elementwise-ops`

2. **Step 2 chain analysis**
   - root-driven backward partition
   - 支持 pointwise multi-use
   - 切断 cube op / reduction multi-use / 无有效 TilingInterface / 非可兼容轴结构

3. **Steps 3–5 计划分析**
   - 基于 root 构造 golden axis
   - 只接受当前 v1 可兼容的 member 进入 chain
   - 对已证明可 collapse 的相邻 parallel dims 做**真实 IR 变换**

4. **Step 8 realization**
   - 使用 `tileConsumerAndFuseProducersUsingSCF`
   - 让 chain 内 producer fuse 到 tile body 中
   - 外层 loop 打 `ascendc.parallel = true`

5. **Spec §3 元数据**
   - stamp `tiling.tiles`
   - stamp `tiling.shapes`

6. **端到端测试**
   - 简单 bias+relu / bias-add 链
   - LayerNorm 静态 shape 链

---

## 1.2 本版明确不做什么

这些内容**不在本版闭环中承诺完成**：

1. **完整 Step 4 轴归一化**
   - 不实现“任意 chain 内 op 自动插入 broadcast / transpose / expand_shape / collapse_shape 来统一 golden axis”的完整版。
   - v1 只接受经过 Step 1 预处理后已落在“可兼容子集”内的 op。

2. **kernel function outline**
   - 不在本版中把每个 plan outline 成独立 `func.func @..._chain<i>_plan<n>`。
   - realization 先在原 `func.func` 内完成，保证现有 Phase 2 pipeline 可直接消费。

3. **多 plan 结构枚举**
   - `default` plan 是本版必须项。
   - `sub_split` / `flat` 只有在 API 可行性验证通过后再实现；本版不预先承诺。

4. **动态 shape 端到端 correctness 证明**
   - 本版允许保留最小 `DimExpr` / `tiling.shapes` 机制，但集成验证只覆盖静态 shape。

---

## 1.3 本版成功标准

满足以下条件即视为本版落地成功：

1. `afir-opt` 能注册并运行两类新 pass：
   - `--eliminate-cf-assert`
   - `--canonicalize-extract-broadcast`
   - `--vector-plan-generation`

2. 简单 pointwise chain（如 bias+relu）能被：
   - 分析为一条 chain
   - 真实 tile + fuse
   - 生成 `scf.for`
   - 外层 loop 带 `ascendc.parallel = true`

3. LayerNorm 静态 shape case 能：
   - 形成单条 Vector chain
   - 保留 pointwise multi-use 语义
   - 生成 tiled loop

4. module 上能看到：
   - `tiling.tiles`
   - `tiling.shapes`

---

## 2. 方案的核心取舍

## 2.1 关于 Step 4 轴归一化

Spec 中 Step 4 是 load-bearing step，但它也是本项目当前实现成本最高的部分。

因此本版采用以下取舍：

- **不在 v1 中实现完整归一化变换**；
- **在 chain analysis 中提前做兼容性筛选**；
- 只有那些在当前 v1 规则下能被视为“与 root 共享同一未来 golden axis 空间”的 producer 才允许进入 chain；
- 不能兼容的 producer 直接作为 `NotNormalizable` boundary 切断。

这让后续的 golden axis / collapse / realization 建立在一个更小但可控的输入子集上。

---

## 2.2 关于 collapse

本版统一采用以下结论：

- **不是只做 collapse 分析；而是做真实 IR 级 collapse。**
- 但仅对那些已经证明满足 upstream precondition 的相邻 parallel dims 调用：

```cpp
linalg::collapseOpIterationDims(...)
```

如果某条 chain 无法 collapse，不生成额外 plan，直接带原 golden dims 继续后续流程。

---

## 2.3 关于 `XBLOCK / XBLOCK_SUB / RBLOCK`

本版把这三个名字视为**目标接口**，但实现上分两层处理：

1. **必须落地：**
   - `XBLOCK` 对应外层并行 tile

2. **编码前必须先做 API 可行性验证：**
   - `XBLOCK_SUB` 是否能在当前 upstream SCF tiling API 上自然表示为第二层同轴 tile
   - `RBLOCK` 在 full reduction 策略下如何进入 realization 接口

因此本版计划里专门加一个 **API feasibility spike**。只有验证通过后，才把 `XBLOCK_SUB` / `RBLOCK` 写死进 realization 和测试断言。

---

## 2.4 关于 outline

Spec 原义要求每个 plan 生成独立 kernel function。当前项目的现有 Phase 2 pipeline 仍以单个 `func.func` 内 IR 为主要输入，因此：

- 本版先在原函数内完成 realization；
- outline 明确放到后续版本；
- 文档和测试里不再把 outline 当作本版交付要求。

---

## 3. 文件结构

### 新建文件

```text
lib/Conversion/CanonicalizeExtractBroadcast/
  CanonicalizeExtractBroadcastPass.cpp
  CMakeLists.txt

include/Conversion/CanonicalizeExtractBroadcast/
  CanonicalizeExtractBroadcastPass.h

lib/Conversion/VectorPlanGeneration/
  ChainAnalysis.h
  ChainAnalysis.cpp
  PlanGeneration.h
  PlanGeneration.cpp
  PlanRealization.h
  PlanRealization.cpp
  VectorPlanGenerationPass.cpp
  CMakeLists.txt

include/Conversion/VectorPlanGeneration/
  VectorPlanGenerationPass.h

test/Conversion/
  eliminate-cf-assert.mlir
  canonicalize-extract-broadcast.mlir
  vector-plan-chain-analysis.mlir
  vector-plan-generation.mlir
  vector-plan-layernorm.mlir
```

### 修改文件

```text
include/Conversion/Passes.td
include/Conversion/Passes.h
lib/Conversion/CMakeLists.txt
tools/afir-opt/CMakeLists.txt
```

---

## 4. Task 分解

## Task 1：集成 `EliminateCfAssert`

**目标：** 让现有 pass 被 TableGen/CMake/afir-opt 正确注册，并提供 lit 测试。

**交付：**
- `Passes.td` 注册 `eliminate-cf-assert`
- `Passes.h` 增加 include
- `lib/Conversion/CMakeLists.txt` 添加子目录
- `tools/afir-opt/CMakeLists.txt` 链接新库
- `test/Conversion/eliminate-cf-assert.mlir`

**验收：**
- `afir-opt --eliminate-cf-assert` 可运行
- 测试能证明 `cf.assert` 被删除，其他正常 op 保留

---

## Task 2：实现 `CanonicalizeExtractBroadcast`

**目标：** 严格按 Appendix A 识别 torch-mlir 的 extract-select broadcast 模式。

**必须坚持的 matcher 规则：**
1. body 中恰好一个 `tensor.extract`
2. 无其他 memory-reading op
3. 所有 iterator 都是 `parallel`
4. 输出 indexing map 是 identity
5. `tensor.extract` 的每个 index 只能是：
   - `arith.constant`
   - `arith.select(arith.cmpi eq %dim %c1, %c0, linalg.index %k)`

**注意：**
- 不允许把裸 `linalg.index` 当成 broadcast 模式的一部分。
- 本版重写目标统一为：
  - 新的 `linalg.generic` + 投影 `indexing_map`
- 不在此 task 中混入别的 broadcast canonicalization 策略。

**交付：**
- pass 头文件、实现、CMake、注册
- `test/Conversion/canonicalize-extract-broadcast.mlir`

**验收：**
- 正例中 `tensor.extract` / `arith.select` 消失
- 负例不被误改写

---

## Task 3：Chain Analysis

**目标：** 实现 deterministic、root-driven backward partition。

### 数据结构

```cpp
enum class BoundaryReason {
  CubeOp,
  NoTilingInterface,
  SideEffect,
  ReductionMultiUse,
  AlreadyClaimed,
  NotNormalizable,
  BlockArgument,
};

struct BoundaryRecord {
  Operation *producer;
  Operation *consumer;
  BoundaryReason reason;
};

struct VectorChain {
  Operation *root;
  SmallVector<Operation *> members;
  SmallVector<Value> externalInputs;
  SmallVector<Operation *> externalUsers;
  SmallVector<BoundaryRecord> boundaries;
};
```

### 关键约束

- 遍历顺序使用 **block program order**，不要用 `func.walk()` 充当 program order。
- pointwise multi-use 允许；reduction multi-use 切断。
- `NotNormalizable` 在本 task 中就判定：
  - 它是一个 **chain inclusion legality rule**；
  - 不是后面再重复判断的“第二事实源”。

### 交付

- `ChainAnalysis.h/.cpp`
- `vector-plan-chain-analysis.mlir`

**测试覆盖：**
- 单 chain pointwise
- cube boundary
- pointwise multi-use allowed
- reduction multi-use cut
- 至少一个 `NotNormalizable` cut case（若当前很难构造，先留 TODO）

---

## Task 4：Pass 骨架 + 诊断 remark

**目标：** 把 chain analysis 接进 `--vector-plan-generation`，先建立可观察性。

**交付：**
- `VectorPlanGenerationPass.h/.cpp`
- `VectorPlanGeneration` CMake
- pass 注册
- `--emit-remarks` 选项

**remark 至少包含：**
- chain root
- member 数量
- boundary reason

**验收：**
- `vector-plan-chain-analysis.mlir` 能稳定通过

---

## Task 5：Golden Axis + Collapse Analysis + IR Collapse

**目标：** 在 v1 子集里完成：
- root-driven golden axis 分析
- compatibility-preserving collapsed dims 计算
- 对满足前提的 dims 调用 upstream `collapseOpIterationDims`

### 规则

1. `buildGoldenAxis` 只做当前 v1 子集可支持的分析：
   - 从 root 的 iterator_types 与 primitive dim provenance 出发
   - 不再承担完整 Step 4 归一化

2. `computeCollapse` 只负责找出：
   - 相邻 parallel dims
   - 且在 chain 内所有 relevant operand/indexing maps 上都满足 contiguity precondition

3. `materializeCollapse` 负责真实变换 IR：
   - 使用 `linalg::collapseOpIterationDims`
   - 失败则保守 fallback，不中断整个 pass

### 说明

这里建议把原来混在一起的内容拆成三个函数：

```cpp
GoldenAxisLayout buildGoldenAxis(...);
LogicalResult computeCollapse(...);
LogicalResult materializeCollapse(...);
```

这样可以避免“只分析不变换”的语义混乱。

---

## Task 6：Tiling API 可行性验证（Spike）

**目标：** 在正式写 realization 前，先回答一件关键事情：

> 当前 upstream `tileConsumerAndFuseProducersUsingSCF` + `SCFTileAndFuseOptions`，能否直接表达本方案中的 `XBLOCK / XBLOCK_SUB / RBLOCK` 语义？

### 这个 spike 必须回答的问题

1. 是否能在当前 API 下自然得到：
   - 外层 `XBLOCK`
   - 同轴内层 `XBLOCK_SUB`

2. reduction 维在 full reduction 策略下：
   - `RBLOCK` 是不是显式 tile 参数
   - 还是当前 realization 中直接固定为 full extent

3. 若 API 不能自然表达两层同轴 tiling：
   - 本版 realization 是否先只落 `XBLOCK`
   - `XBLOCK_SUB` 延后到后续版本

### 产物

输出一个简短结论，写入本计划对应实现注释或开发记录中：

- **结论 A：** 可以表达 → Task 7 落全量
- **结论 B：** 不能完整表达 → Task 7 只落最小可行 realization，并同步收缩测试断言

**注意：** 这个 spike 不需要单独新 pass，但它必须发生在 Task 7 之前。

---

## Task 7：Plan Realization（核心）

**目标：** 在原 `func.func` 内，使用正确 API 完成 tile+fuse realization。

### 必须使用的 API

- `scf::SCFTileAndFuseOptions`
- `scf::tileConsumerAndFuseProducersUsingSCF`

### realization 流程

1. 根据 Task 5 的结果，先做真实 IR collapse（若存在）
2. 构造 tile 参数与 tiling options
3. `fusionControlFn` 只允许 chain 内 member 的 producer 被 fuse
4. 调 `tileConsumerAndFuseProducersUsingSCF`
5. 用返回的 `replacements` 替换原 root uses
6. 给外层 loop 加 `ascendc.parallel = true`
7. 跑必要的 cleanup（canonicalize/CSE/DCE，按项目现有方式接入）

### 这一步不要做什么

- 不做 outline
- 不做 alternate plan function 生成
- 不在此处承担完整 Step 4 normalization

### 测试要求

`vector-plan-generation.mlir` 至少要验证：
- 生成了 `scf.for`
- 外层 loop 带 `ascendc.parallel = true`
- producer body 确实被 fuse 进 tile body（而不是只 tile root）
- 若 Task 6 证明 `XBLOCK_SUB` 尚未落地，则不要在测试中提前断言它已存在

---

## Task 8：LayerNorm 集成测试

**目标：** 用 worked example 风格的静态 shape LayerNorm 验证整条最小闭环。

### 测试流程

```text
afir-opt --eliminate-cf-assert \
         --canonicalize-extract-broadcast \
         --linalg-fold-unit-extent-dims \
         --linalg-fuse-elementwise-ops \
         --vector-plan-generation="emit-remarks=true"
```

### 至少检查

- 只有 1 条 chain
- root 是 `+beta` 对应的最终 consumer
- 有 `scf.for`
- outer loop 带 `ascendc.parallel = true`
- tile body 内还能看到 LayerNorm 关键算子痕迹（如 `arith.addf` / `arith.mulf` / `math.rsqrt`）
- pointwise multi-use 没被错误切断

---

## Task 9：Module 级 metadata

**目标：** stamp `tiling.tiles` 和 `tiling.shapes`。

### 规则

1. `tiling.tiles`
   - 从本版 realization 真正支持的 tile 参数列表生成
   - 不要把尚未落地的参数提前写进去

2. `tiling.shapes`
   - **从原始 kernel tensor 参数的 primitive dims 收集**
   - 不要从 `layout.dims` 直接反推
   - 不要把 composite dim（如 `B*S`）直接写进 `tiling.shapes`

### 说明

`tiling.shapes` 的来源应当是“原始 tensor 参数 + dim index”的稳定绑定，而不是 golden axis 的中间分析结果。

---

## 5. 依赖关系

```text
Task 1 (EliminateCfAssert 集成) ─┐
Task 2 (CanonicalizeExtractBroadcast) ─┤── 可并行
Task 3 (Chain Analysis) ──────────────┘
         │
         ▼
Task 4 (Pass 骨架 + 诊断)
         │
         ▼
Task 5 (Golden Axis + Collapse)
         │
         ▼
Task 6 (Tiling API Spike)
         │
         ▼
Task 7 (Plan Realization)
         │
         ├──► Task 8 (LayerNorm 集成测试)
         └──► Task 9 (Module metadata)
```

---

## 6. 本版结束后，后续版本再做什么

以下内容明确留到后续版本：

1. **完整 Step 4 轴归一化**
   - 自动插入 broadcast / transpose / reshape 以统一 golden axis

2. **kernel outline**
   - 每个 plan 变成独立 `func.func @..._chain<i>_plan<n>`

3. **多 plan 枚举**
   - `sub_split`
   - `flat`
   - benchmark-time winner selection

4. **动态 shape 端到端测试**

---

## 7. 一句话总结

这版实施方案的核心思想是：

> 先用预处理和 chain legality 规则把输入收缩到“静态 shape + 已规整的 Vector 子集”，在这个子集上正确实现真实的 IR collapse 与 `tileConsumerAndFuseProducersUsingSCF` realization，形成一个可验证、可进入 Phase 2 的最小闭环；剩余最重的归一化、outline、多 plan 枚举留到后续版本。
