# Vector Plan Generation — Phase 1 实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标:** 实现 generalized Phase 1 Vector track pipeline，将任意 linalg 程序自动编译为 tiled+fused Ascend NPU kernel，无需手写 transform 脚本。

**架构:** 4 个预处理 pass（Step 1）+ 1 个编排 pass `--vector-plan-generation`（Steps 2–8）。编排 pass 内部按 chain 粒度执行：chain 分析 → golden axis → 轴归一化 → 维度 collapse → split-tiling → plan 生成 → plan 实现。最终调用上游 `tileConsumerAndFuseProducersUsingSCF` API 生成 tiled+fused 的 kernel 函数。

**技术栈:** MLIR C++（upstream linalg/SCF/tensor dialects），TableGen pass 注册，lit+FileCheck 测试。

**Spec:** `docs/superpowers/specs/2026-04-09-mlir-ai-compiler-generalization-design.md`

---

## 英文方案的问题检视

在写中文方案之前，列出英文方案中发现的问题：

### 问题 1：Step 8 用错了 API（严重）

英文方案 Task 7 调用 `tileUsingSCF()`，只 tile root 不 fuse producer。
Spec 明确要求调用 `tileConsumerAndFuseProducersUsingSCF()`，tile root 的同时
将 chain 内所有 producer fuse 到 tile body 中。这是整个方案的核心 API。

**修正:** 必须用 `tileConsumerAndFuseProducersUsingSCF` + `SCFTileAndFuseOptions`，
并通过 `fusionControlFn` 控制哪些 producer 可以 fuse（答案：chain members 内的全部允许）。

### 问题 2：缺少 IR 层面的 collapse（严重）

英文方案 Task 5 只计算了哪些维度可以 collapse，但从未调用
`linalg::collapseOpIterationDims()` 实际变换 IR。Spec Step 5 要求在 tiling
之前先 collapse 维度，使迭代空间降维。

**修正:** 在 `realizePlan` 中，tiling 之前必须对 chain 中每个 op 调用
`collapseOpIterationDims`。

### 问题 3：轴归一化被跳过（严重）

英文方案 Task 5 的 `GoldenAxis.cpp` 没有实现 Spec Step 4 的归一化优先级：
1. iterator/indexing normalization first
2. broadcast/reshape normalization second
3. explicit transpose third
4. otherwise cut the chain

Spec 说这是 "load-bearing step"。没有它，后续的 collapse 和 tiling 都不正确。

**修正:** v1 先实现简化版——只处理已经 canonical 的情况（预处理后大部分 op 已归一化）。
对无法归一化的 op，在 chain analysis 阶段就 cut（`cannotAbsorbProducer` 返回
`NotNormalizable`）。完整归一化推迟到 v1.1。

### 问题 4：缺少 kernel 函数 outline（中等）

Spec 要求每个 plan 生成独立的 `func.func @<name>_chain<i>_plan<n>`。
英文方案直接在原函数内 tile，没有 outline 成独立函数。

**修正:** v1 先不做 outline（在原函数内 tile）。outline 推迟到 v1.1，
因为当前 Phase 2 pipeline（bufferize → buffer-placement → linalg-to-ascendc）
本来就期望在单个 func.func 上操作。先让 tiling 跑通，再做函数分拆。

### 问题 5：chain analysis 遍历顺序不严谨（轻微）

英文方案用 `func.walk()` + `std::reverse()` 得到反向程序序。
`walk()` 不保证 program order（它递归进 region）。
应该直接遍历 block 的 operation list。

**修正:** 用 `func.getBody().front().getOperations()` 得到 program order，
然后 reverse iterate。

### 问题 6：tiling.shapes 元数据缺失（轻微）

Task 9 只 stamp `tiling.tiles`，缺少 `tiling.shapes`。

**修正:** 一并 stamp。

---

## 文件结构

### 新建文件

```
lib/Conversion/CanonicalizeExtractBroadcast/
  CanonicalizeExtractBroadcastPass.cpp   ← Step 1.2: extract-select → 投影 indexing_map
  CMakeLists.txt

include/Conversion/CanonicalizeExtractBroadcast/
  CanonicalizeExtractBroadcastPass.h

lib/Conversion/VectorPlanGeneration/
  ChainAnalysis.h/.cpp       ← Step 2: chain 数据结构 + 分区算法
  PlanGeneration.h/.cpp      ← Steps 3–7: golden axis + collapse + split-tiling + plan 记录
  PlanRealization.h/.cpp     ← Step 8: tileConsumerAndFuseProducersUsingSCF 封装
  VectorPlanGenerationPass.cpp  ← 编排 pass
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

```
include/Conversion/Passes.td         ← 新增 3 个 pass 定义
include/Conversion/Passes.h          ← 新增 2 个 include
lib/Conversion/CMakeLists.txt        ← 新增 2 个子目录
tools/afir-opt/CMakeLists.txt        ← 链接 2 个新库
```

---

## Task 1: 集成 EliminateCfAssert 到构建系统

代码已存在于 `lib/Conversion/EliminateCfAssert/`，但未注册到 TableGen、CMake、afir-opt。

**文件:**
- 修改: `include/Conversion/Passes.td`
- 修改: `include/Conversion/Passes.h`
- 修改: `lib/Conversion/CMakeLists.txt`
- 修改: `tools/afir-opt/CMakeLists.txt`
- 新建: `test/Conversion/eliminate-cf-assert.mlir`

- [ ] **Step 1: 添加 TableGen 定义**

在 `include/Conversion/Passes.td` 的 `#endif` 之前添加：

```tablegen
def EliminateCfAssertPass : Pass<"eliminate-cf-assert", "mlir::func::FuncOp"> {
  let summary = "Remove cf.assert operations inserted by torch-mlir";
  let description = [{
    torch-mlir inserts cf.assert ops for dynamic shape broadcast validation.
    These are unnecessary in our pipeline and block downstream passes that
    do not support cf dialect. This pass removes them and DCEs their conditions.
  }];
  let constructor = "mlir::afir::createEliminateCfAssertPass()";
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect"
  ];
}
```

- [ ] **Step 2: 注册到 Passes.h / CMake / afir-opt**

`include/Conversion/Passes.h` 添加：
```cpp
#include "Conversion/EliminateCfAssert/EliminateCfAssertPass.h"
```

`lib/Conversion/CMakeLists.txt` 添加：
```cmake
add_subdirectory(EliminateCfAssert)
```

`tools/afir-opt/CMakeLists.txt` 添加：
```cmake
    EliminateCfAssertConversion
```

- [ ] **Step 3: 编写 lit 测试**

新建 `test/Conversion/eliminate-cf-assert.mlir`：

```mlir
// RUN: afir-opt --eliminate-cf-assert %s | FileCheck %s

// CHECK-LABEL: func.func @remove_broadcast_assert
// CHECK-NOT: cf.assert
// CHECK-NOT: arith.cmpi
func.func @remove_broadcast_assert(%arg0: tensor<?x?x128xf32>) -> tensor<?x?x128xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x128xf32>
  %0 = arith.cmpi eq, %dim, %dim : index
  cf.assert %0, "mismatched size for broadcast"
  return %arg0 : tensor<?x?x128xf32>
}

// CHECK-LABEL: func.func @preserve_non_assert_ops
// CHECK: arith.addi
func.func @preserve_non_assert_ops(%arg0: index, %arg1: index) -> index {
  %0 = arith.cmpi eq, %arg0, %arg1 : index
  cf.assert %0, "test"
  %1 = arith.addi %arg0, %arg1 : index
  return %1 : index
}
```

- [ ] **Step 4: 构建并验证**

```bash
cmake --build build --target afir-opt && \
build/bin/llvm-lit test/Conversion/eliminate-cf-assert.mlir -v
```

- [ ] **Step 5: 提交**

```bash
git add include/Conversion/Passes.td include/Conversion/Passes.h \
        include/Conversion/EliminateCfAssert/ \
        lib/Conversion/EliminateCfAssert/ \
        lib/Conversion/CMakeLists.txt tools/afir-opt/CMakeLists.txt \
        test/Conversion/eliminate-cf-assert.mlir
git commit -m "feat: integrate --eliminate-cf-assert pass into build system"
```

---

## Task 2: CanonicalizeExtractBroadcast Pass

实现 Spec Appendix A。将 torch-mlir 的 `extract-select` 动态 broadcast 模式
重写为投影 `indexing_map`。

**核心识别规则：**
1. body 中恰好有 1 个 `tensor.extract`，无其他内存读取 op
2. 所有 iterator 都是 `parallel`
3. `tensor.extract` 的每个 index 操作数是以下之一：
   - `arith.constant`
   - `arith.select(arith.cmpi eq %dim %c1, %c0, linalg.index %k)`
4. 输出 indexing map 是 identity

**重写目标：** 将该 generic 替换为 `ins(%src) outs(%dst)` 的新 generic，
其中 src 的 indexing_map 在 broadcast 维度上映射到常量 0。

**文件:**
- 新建: `include/Conversion/CanonicalizeExtractBroadcast/CanonicalizeExtractBroadcastPass.h`
- 新建: `lib/Conversion/CanonicalizeExtractBroadcast/CanonicalizeExtractBroadcastPass.cpp`
- 新建: `lib/Conversion/CanonicalizeExtractBroadcast/CMakeLists.txt`
- 修改: `include/Conversion/Passes.td`、`Passes.h`、两个 `CMakeLists.txt`
- 新建: `test/Conversion/canonicalize-extract-broadcast.mlir`

- [ ] **Step 1: 编写 failing 测试**

```mlir
// RUN: afir-opt --canonicalize-extract-broadcast %s | FileCheck %s

#map_identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @broadcast_from_reduction
// CHECK-NOT: tensor.extract
// CHECK-NOT: arith.select
// CHECK: linalg.generic
// CHECK-SAME: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
func.func @broadcast_from_reduction(
    %src: tensor<?x?x1xf32>, %dst: tensor<?x?x128xf32>,
    %dim0: index, %dim1: index) -> tensor<?x?x128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = linalg.generic {
      indexing_maps = [#map_identity],
      iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%dst : tensor<?x?x128xf32>) {
    ^bb0(%out: f32):
      %i0 = linalg.index 0 : index
      %i1 = linalg.index 1 : index
      %p0 = arith.cmpi eq, %dim0, %c1 : index
      %s0 = arith.select %p0, %c0, %i0 : index
      %p1 = arith.cmpi eq, %dim1, %c1 : index
      %s1 = arith.select %p1, %c0, %i1 : index
      %e = tensor.extract %src[%s0, %s1, %c0] : tensor<?x?x1xf32>
      linalg.yield %e : f32
  } -> tensor<?x?x128xf32>
  return %result : tensor<?x?x128xf32>
}

// 负例：不匹配的 pattern 不应被改写
// CHECK-LABEL: func.func @not_a_broadcast
// CHECK: tensor.extract
func.func @not_a_broadcast(%src: tensor<?x?xf32>, %dst: tensor<?x?xf32>,
                           %idx: tensor<?xi32>) -> tensor<?x?xf32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%idx : tensor<?xi32>) outs(%dst : tensor<?x?xf32>) {
    ^bb0(%in: i32, %out: f32):
      %i = arith.index_cast %in : i32 to index
      %j = linalg.index 1 : index
      %e = tensor.extract %src[%i, %j] : tensor<?x?xf32>
      linalg.yield %e : f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
```

- [ ] **Step 2: 添加 TableGen + header + CMake**

TableGen 定义（`Passes.td`）：
```tablegen
def CanonicalizeExtractBroadcastPass
    : Pass<"canonicalize-extract-broadcast", "mlir::func::FuncOp"> {
  let summary = "Rewrite extract-select broadcast pattern to projection indexing_map";
  let constructor = "mlir::afir::createCanonicalizeExtractBroadcastPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];
}
```

CMakeLists.txt:
```cmake
add_mlir_library(CanonicalizeExtractBroadcastConversion
  CanonicalizeExtractBroadcastPass.cpp
  ADDITIONAL_HEADER_DIRS ${CMAKE_SOURCE_DIR}/include/Conversion
  DEPENDS AFIRConversionPassIncGen
  LINK_LIBS PUBLIC
  MLIRArithDialect MLIRLinalgDialect MLIRTensorDialect MLIRFuncDialect MLIRTransforms
)
```

- [ ] **Step 3: 实现 pass**

核心逻辑（详见英文方案 Task 2 Step 5，代码可复用）：
- `classifyExtractIndex(Value) → optional<AffineExpr>`：识别每个 index 操作数
- `tryRewriteExtractBroadcast(GenericOp) → bool`：模式匹配 + 重写
- pass `runOnOperation`：collect candidates → try rewrite each

- [ ] **Step 4: 注册 + 构建 + 验证 + 提交**

---

## Task 3: Chain Analysis 基础设施（Step 2）

实现 root-driven backward chain partitioning 算法。
这个 task 只构建数据结构和算法，不变换 IR。

**文件:**
- 新建: `lib/Conversion/VectorPlanGeneration/ChainAnalysis.h`
- 新建: `lib/Conversion/VectorPlanGeneration/ChainAnalysis.cpp`

### 数据结构

```cpp
/// chain 切分原因
enum class BoundaryReason {
  CubeOp, NoTilingInterface, SideEffect,
  ReductionMultiUse, AlreadyClaimed, NotNormalizable, BlockArgument
};

struct BoundaryRecord { Operation *producer, *consumer; BoundaryReason reason; };

struct VectorChain {
  Operation *root;
  SmallVector<Operation *> members;       // 拓扑序
  SmallVector<Value> externalInputs;
  SmallVector<Operation *> externalUsers;
  SmallVector<BoundaryRecord> boundaries;
};
```

### 关键算法

```cpp
/// 判断：是否 Cube op（matmul, batch_matmul, conv）
bool isCubeOp(Operation *op);

/// 判断：是否 Vector chain 候选（linalg structured + TilingInterface，排除 fill）
bool isVectorCandidate(Operation *op);

/// 判断：是否可作为 chain root（Spec §4 Step 2.2）
///   - Vector 候选 + 未被 claimed
///   - 至少一个 result 满足边界条件（返回值、被 Cube 消费、无 consumer 等）
bool isRootCandidate(Operation *op, const DenseSet<Operation *> &claimed);

/// 从 root 反向构建 chain（Spec §4 Step 2.3）
///   - worklist 反向遍历 producer
///   - canAbsorbProducer 检查 5 个条件（Spec §4 Step 2.4）
///   - fanout rule：pointwise 多用允许，reduction 多用切断（Spec §4 Step 2.5）
VectorChain buildChainFromRoot(Operation *root, const DenseSet<Operation *> &claimed);

/// 全函数 chain 分析（Spec §4 Step 2.1）
///   ⚠️ 遍历顺序：用 block.getOperations() 的 reverse iterator
///   不要用 func.walk()（不保证 program order）
SmallVector<VectorChain> analyzeChains(func::FuncOp func);
```

- [ ] **Step 1: 实现数据结构和算法**
- [ ] **Step 2: 提交**

---

## Task 4: VectorPlanGeneration Pass 骨架 + Chain 诊断

将 chain analysis 接入 pass 框架。pass 接受 `--emit-remarks` 选项，
输出 chain 信息作为 remark 供测试验证。

**文件:**
- 新建: `include/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.h`
- 新建: `lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp`
- 新建: `lib/Conversion/VectorPlanGeneration/CMakeLists.txt`
- 修改: `include/Conversion/Passes.td`、`Passes.h`、两个 `CMakeLists.txt`
- 新建: `test/Conversion/vector-plan-chain-analysis.mlir`

- [ ] **Step 1: 添加 TableGen + header + CMake**

```tablegen
def VectorPlanGenerationPass
    : Pass<"vector-plan-generation", "mlir::func::FuncOp"> {
  let summary = "Generate tiled+fused Vector kernels from linalg programs";
  let constructor = "mlir::afir::createVectorPlanGenerationPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect", "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect", "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
  let options = [
    Option<"emitRemarks", "emit-remarks", "bool", "false",
           "Emit diagnostic remarks for chain analysis (for testing)">,
  ];
}
```

- [ ] **Step 2: 实现 pass 骨架**

`runOnOperation` 中：
1. 调用 `analyzeChains(func)` 得到 chains
2. 如果 `emitRemarks`，对每个 chain emit remark（root name、members 数、boundary reason）
3. 后续 Steps 3-8 的调用点留 TODO 注释

- [ ] **Step 3: 编写 lit 测试**

测试用例覆盖：
- 纯 pointwise chain → 单 chain
- Cube boundary → chain 在 matmul 处切断
- pointwise multi-use → chain 不切断
- reduction multi-use → chain 切断

- [ ] **Step 4: 构建 + 验证 + 提交**

---

## Task 5: Plan Generation（Steps 3–7）

合并原英文方案的 Task 5 (Golden Axis) 和 Task 6 (SplitTiling)。
这些都是 plan 的元数据计算，不涉及 IR 变换。

**文件:**
- 新建: `lib/Conversion/VectorPlanGeneration/PlanGeneration.h`
- 新建: `lib/Conversion/VectorPlanGeneration/PlanGeneration.cpp`
- 修改: `lib/Conversion/VectorPlanGeneration/CMakeLists.txt`

### 数据结构

```cpp
/// 符号维度表达式（Spec §4 Step 3.1）
struct DimExpr {
  enum Kind { Const, ArgDim, Mul };
  Kind kind;
  int64_t constVal;         // Const
  unsigned argNum, dimNum;  // ArgDim
  SmallVector<DimExpr> factors; // Mul
};

/// Golden 维度角色
enum class DimRole { Parallel, Reduction, Mixed };

/// Golden 维度
struct GoldenDim { DimRole role; DimExpr expr; };

/// Golden 轴布局
struct GoldenAxisLayout {
  SmallVector<GoldenDim> dims;           // 归一化后 golden 维度
  SmallVector<GoldenDim> collapsedDims;  // collapse 后
  SmallVector<SmallVector<unsigned>> collapseGroups; // 哪些维度被合并
};

/// Tiling 方案
enum class PlanKind { Default, SubSplit, Flat };

struct TilingPlan {
  PlanKind kind;
  unsigned splitAxisIdx;                  // 外层并行维度 index
  SmallVector<unsigned> tilingAxes;       // 内层 tile 维度
  SmallVector<unsigned> reductionAxes;    // reduction 维度
  SmallVector<std::string> tileNames;     // ["XBLOCK", "XBLOCK_SUB", "RBLOCK"]
};
```

### 关键函数

```cpp
/// Step 3: 构建 golden axis（从 root 的 iterator_types 出发）
///   v1 简化版：直接取 root 的迭代空间，检查其他 member 的兼容性
///   不做完整的轴归一化（Step 4），不兼容的 op 在 chain analysis 阶段已被切掉
GoldenAxisLayout buildGoldenAxis(const VectorChain &chain, func::FuncOp func);

/// Step 5: 计算哪些相邻 parallel dims 可以 collapse
///   规则：相邻 parallel dims 在每个 operand 上都连续
///   Reduction 不与 parallel 合并
void computeCollapse(GoldenAxisLayout &layout, const VectorChain &chain);

/// Steps 6-7: 根据 collapsedDims 生成 tiling plans
///   - split_axis = 最外层 parallel dim
///   - tiling_axis = 所有 dims
///   - default plan: XBLOCK + XBLOCK_SUB [+ RBLOCK]
///   - 有 reduction → RBLOCK = full extent
SmallVector<TilingPlan> generatePlans(const GoldenAxisLayout &layout);
```

- [ ] **Step 1: 实现数据结构和函数**
- [ ] **Step 2: 在 pass 中调用并输出 remark**

```cpp
// VectorPlanGenerationPass::runOnOperation() 中追加：
GoldenAxisLayout layout = buildGoldenAxis(chain, func);
SmallVector<TilingPlan> plans = generatePlans(layout);
if (emitRemarks) {
  chain.root->emitRemark() << "chain[" << i << "] tiles: ["
    << llvm::join(plans[0].tileNames, ", ") << "]";
}
```

- [ ] **Step 3: 验证 + 提交**

---

## Task 6: Plan Realization（Step 8）— 核心 Task

这是全方案最关键的 task：实际变换 IR。

### 与英文方案的关键差异

| 方面 | 英文方案（错误） | 中文方案（正确） |
|------|-----------------|-----------------|
| API | `tileUsingSCF` | `tileConsumerAndFuseProducersUsingSCF` |
| Fusion | 无 | `fusionControlFn` 允许 chain members |
| Collapse | 只计算不变换 | 调用 `collapseOpIterationDims` 变换 IR |
| Outline | 无 | v1 暂不做，留到 v1.1 |

**文件:**
- 新建: `lib/Conversion/VectorPlanGeneration/PlanRealization.h`
- 新建: `lib/Conversion/VectorPlanGeneration/PlanRealization.cpp`
- 修改: `lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp`
- 修改: `lib/Conversion/VectorPlanGeneration/CMakeLists.txt`
- 新建: `test/Conversion/vector-plan-generation.mlir`

- [ ] **Step 1: 编写 failing 测试**

```mlir
// RUN: afir-opt --vector-plan-generation %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @bias_relu
// 验证：tile args 被追加到函数签名
// CHECK-SAME: %{{.*}}: index
// 验证：生成了 scf.for 循环
// CHECK: scf.for
// 验证：外层循环有 ascendc.parallel 属性
// CHECK-SAME: ascendc.parallel = true
// 验证：producer (relu) 被 fuse 到了 tile body 内
// CHECK: arith.maximumf
// CHECK: arith.addf
func.func @bias_relu(%arg0: tensor<128x64xf32>,
                     %bias: tensor<64xf32>) -> tensor<128x64xf32> {
  %empty = tensor.empty() : tensor<128x64xf32>
  %relu = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<128x64xf32>) outs(%empty : tensor<128x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 0.0 : f32
      %r = arith.maximumf %in, %cst : f32
      linalg.yield %r : f32
  } -> tensor<128x64xf32>
  %result = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%relu, %bias : tensor<128x64xf32>, tensor<64xf32>)
      outs(%empty : tensor<128x64xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %r = arith.addf %in0, %in1 : f32
      linalg.yield %r : f32
  } -> tensor<128x64xf32>
  return %result : tensor<128x64xf32>
}
```

**关键 CHECK：** `arith.maximumf` 和 `arith.addf` 都在 `scf.for` body 内 →
证明 producer 被 fuse 进来了，不只是 tile root。

- [ ] **Step 2: 实现 realizePlan 函数**

```cpp
LogicalResult realizePlan(func::FuncOp func,
                          VectorChain &chain,
                          const GoldenAxisLayout &layout,
                          const TilingPlan &plan,
                          IRRewriter &rewriter) {
  auto tilingInterface = dyn_cast<TilingInterface>(chain.root);
  if (!tilingInterface)
    return failure();

  // 1. 追加 tile index 参数到函数签名
  SmallVector<Value> tileArgs = appendTileArgs(func, plan, rewriter);

  // 2. 构建 tile sizes（SSA index values，不是常量）
  unsigned numLoops = tilingInterface.getLoopIteratorTypes().size();
  SmallVector<OpFoldResult> tileSizes(numLoops, rewriter.getIndexAttr(0));
  // XBLOCK → split axis
  if (tileArgs.size() >= 1)
    tileSizes[plan.splitAxisIdx] = tileArgs[0];

  // 3. ⚠️ 关键：构建 SCFTileAndFuseOptions
  scf::SCFTileAndFuseOptions opts;
  opts.tilingOptions.setTileSizes(tileSizes);

  // 4. ⚠️ 关键：fusionControlFn 允许 chain members
  DenseSet<Operation *> chainMembers(chain.members.begin(),
                                      chain.members.end());
  opts.fusionControlFn =
      [&](tensor::ExtractSliceOp sliceOp,
          OpResult originalProducer,
          bool isDestinationOperand)
          -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    Operation *producerOp = originalProducer.getOwner();
    if (chainMembers.contains(producerOp))
      return scf::SCFTileAndFuseOptions::ControlFnResult{
          /*yieldProducerReplacement=*/false};
    return std::nullopt; // 不 fuse chain 外的 producer
  };

  // 5. ⚠️ 关键：调用 tileConsumerAndFuseProducersUsingSCF
  rewriter.setInsertionPoint(chain.root);
  auto result = scf::tileConsumerAndFuseProducersUsingSCF(
      rewriter, tilingInterface, opts);
  if (failed(result))
    return chain.root->emitError("tileConsumerAndFuseProducersUsingSCF failed");

  // 6. 替换原始 root 的 uses
  for (auto [origVal, replacement] : result->replacements)
    rewriter.replaceAllUsesWith(origVal, replacement);

  // 7. 标注外层循环 ascendc.parallel
  if (!result->loops.empty()) {
    result->loops[0]->setAttr("ascendc.parallel",
                               rewriter.getBoolAttr(true));
  }

  // 8. 清理被 fuse 的原始 ops（已被 tiled 副本替代）
  // tileConsumerAndFuseProducersUsingSCF 通常会自动处理，
  // 但如果有 dead ops 残留，跑一遍 canonicalize + DCE。

  return success();
}
```

- [ ] **Step 3: 接入 pass 主循环**

```cpp
// VectorPlanGenerationPass::runOnOperation() 中：
IRRewriter rewriter(&getContext());
for (size_t i = 0; i < chains.size(); ++i) {
  auto &chain = chains[i];
  GoldenAxisLayout layout = buildGoldenAxis(chain, func);
  SmallVector<TilingPlan> plans = generatePlans(layout);
  if (plans.empty()) continue;

  if (failed(realizePlan(func, chain, layout, plans[0], rewriter))) {
    chain.root->emitWarning("failed to realize plan for chain");
  }
}
```

- [ ] **Step 4: 构建 + 验证**

```bash
cmake --build build --target afir-opt && \
build/bin/llvm-lit test/Conversion/vector-plan-generation.mlir -v
```

- [ ] **Step 5: 提交**

---

## Task 7: LayerNorm 集成测试

端到端测试：跑完整个 Step 1 预处理 + vector-plan-generation，
验证 LayerNorm 模式（Spec §5 worked example）。

**文件:**
- 新建: `test/Conversion/vector-plan-layernorm.mlir`

- [ ] **Step 1: 编写集成测试**

用静态 shape `2x8x128` 的 LayerNorm 模式（residual add → mean → sub →
var → rsqrt → normalize → gamma → beta），作为 input 依次跑：

```
afir-opt --eliminate-cf-assert \
         --canonicalize-extract-broadcast \
         --linalg-fold-unit-extent-dims \
         --linalg-fuse-elementwise-ops \
         --vector-plan-generation="emit-remarks=true"
```

CHECK 要求：
- 检测到 1 个 chain（root = +beta 的 addf）
- 生成了 scf.for 循环
- pointwise multi-use (%add, %sub) 被正确处理

- [ ] **Step 2: 运行 + debug + 提交**

---

## Task 8: Module 级别 Tiling 元数据（Spec §3）

Stamp `tiling.tiles` 和 `tiling.shapes` 到 module 属性上。

- [ ] **Step 1: 在 realizePlan 成功后 stamp metadata**

```cpp
auto moduleOp = func->getParentOfType<ModuleOp>();
if (moduleOp) {
  // tiling.tiles = ["XBLOCK", "XBLOCK_SUB", "RBLOCK"]
  SmallVector<Attribute> tileNames;
  for (auto &name : plan.tileNames)
    tileNames.push_back(rewriter.getStringAttr(name));
  moduleOp->setAttr("tiling.tiles", rewriter.getArrayAttr(tileNames));

  // tiling.shapes = [{name="d_B", from_arg=0, dim=0}, ...]
  // 从 golden axis 的 DimExpr 中提取 ArgDim 信息
  SmallVector<Attribute> shapes;
  for (auto &dim : layout.dims) {
    if (dim.expr.kind == DimExpr::ArgDim) {
      auto dict = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr("name",
          rewriter.getStringAttr("d_" + std::to_string(dim.expr.dimNum))),
        rewriter.getNamedAttr("from_arg",
          rewriter.getI64IntegerAttr(dim.expr.argNum)),
        rewriter.getNamedAttr("dim",
          rewriter.getI64IntegerAttr(dim.expr.dimNum)),
      });
      shapes.push_back(dict);
    }
  }
  moduleOp->setAttr("tiling.shapes", rewriter.getArrayAttr(shapes));
}
```

- [ ] **Step 2: 更新测试 + 验证 + 提交**

---

## 依赖图

```
Task 1 (EliminateCfAssert 集成)  ─┐
Task 2 (CanonicalizeExtractBroadcast) ─┤── 可并行
Task 3 (Chain Analysis)          ─┘
         │
         ▼
Task 4 (Pass 骨架 + 诊断)   ← 依赖 Task 3
         │
         ▼
Task 5 (Plan Generation)    ← 依赖 Task 3
         │
         ▼
Task 6 (Plan Realization)   ← 依赖 Task 4, 5（核心 task）
         │
         ▼
Task 7 (LayerNorm 集成测试) ← 依赖 Task 1, 2, 6
         │
         ▼
Task 8 (Module 元数据)      ← 依赖 Task 6
```

## v1 范围边界

明确哪些在 v1 做，哪些推迟：

| 特性 | v1 | 推迟到 |
|------|----|----|
| chain analysis + root selection | ✅ | — |
| golden axis（简化版，从 root 推导） | ✅ | — |
| 轴归一化（Step 4 完整版） | ❌ | v1.1 |
| dimension collapse（分析 + IR 变换） | ✅ 分析 / ❌ IR 变换 | v1.1 调用 `collapseOpIterationDims` |
| split-tiling rules | ✅ | — |
| tileConsumerAndFuseProducersUsingSCF | ✅ | — |
| kernel function outline | ❌ | v1.1 |
| tiling.tiles / tiling.shapes 元数据 | ✅ | — |
| 静态 shape 测试 | ✅ | — |
| 动态 shape 测试 | ❌ | v1.1 |

**v1 可验证的端到端效果：**
对静态 shape 的 LayerNorm / FFN bias+relu 等 pattern，
`--vector-plan-generation` 能自动生成带 `scf.for` 的 tiled+fused IR，
外层循环标注 `ascendc.parallel`，可直接进入 Phase 2 pipeline。
