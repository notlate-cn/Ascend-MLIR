# impl-00: 基础设施 — 数据结构 + Pass 骨架

**设计依据**: `docs/superpowers/specs/2026-04-14-vector-plan-unified-design.md` §6  
**被依赖于**: impl-01 / impl-02 / impl-03 / impl-04（所有后续模块）

---

## 定位

本模块建立整个 vector-plan 系统的数据契约（header 文件）和编译基础设施（pass 注册 + CMake），
不包含任何算法实现。后续所有 pass 的 `.cpp` 文件都 `#include` 这里的 header，
不再各自重复定义数据结构。

**接口边界**：只写 header（`.h` / `.td`）和空壳 `.cpp`，`runOnOperation()` 直接 return。

---

## 目录结构

```
include/Conversion/VectorPlan/
  Passes.td
  Passes.h
  GroupInfo.h
  TilePlan.h
  TileInfo.h

lib/Conversion/VectorPlan/
  CMakeLists.txt
  GroupAnalysis/
    GroupAnalysisPass.cpp        # 空壳
  GroupOutline/
    GroupOutlinePass.cpp         # 空壳
  TileFuse/
    TileFusePass.cpp             # 空壳
  TileInfo/
    TilePlanToTileInfo.cpp       # 空壳

test/Conversion/VectorPlan/
  (impl-01 起添加测试文件)
```

参考模式：`include/Conversion/Passes.td`、`lib/Conversion/MarkStructuredOps/`。

---

## Phase 1: GroupInfo.h

文件路径：`include/Conversion/VectorPlan/GroupInfo.h`

```cpp
#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vector_plan {

// 轴在 group canonical 迭代空间中的语义类型
enum class AxisRole : uint8_t {
  Parallel,   // 所有成员均为 parallel 或 absent → 可 tile
  Reduction,  // 至少一个成员为 reduction → 不做并行切分
};

struct AxisInfo {
  llvm::StringRef name;       // 可选语义名，如 "B", "S", "H"；可为空
  int64_t         staticSize; // ShapedType::kDynamic 表示运行时动态
  AxisRole        role;
};

// group 内所有 linalg op 的集合描述
struct GroupInfo {
  enum class Kind : uint8_t { Vector, Cube };

  Kind                               kind;
  llvm::SmallVector<linalg::LinalgOp> topoMembers;   // 按 topo_index 升序
  llvm::SmallVector<linalg::LinalgOp> sinks;         // group 内无消费者的末端 op
  llvm::SmallVector<AxisInfo>         canonicalAxes; // 上确界推导所得
  llvm::SmallVector<Value>            boundaryIn;    // 来自 group 外的输入 tensor
  llvm::SmallVector<Value>            boundaryOut;   // 被 group 外消费的输出 tensor
};

// Collapse 之后的扩展（仅 VectorGroup，CubeGroup 为恒等映射）
struct CollapsedGroupInfo : GroupInfo {
  llvm::SmallVector<AxisInfo> collapsedAxes;
  llvm::SmallVector<int>      axisMap;    // 原轴 idx → collapsed 轴 idx
  bool                        hasB2;      // 是否存在 B2 input（需生成两份 Variant）
  bool                        noCollapse; // 含 no_collapse op → 走原始 G-axes loop
};

// CubeGroup 专用扩展
struct CubeGroupInfo : GroupInfo {
  linalg::LinalgOp matmul;
  linalg::LinalgOp epilogueAnchor; // 最末 epilogue op；无 epilogue 时 == matmul
};

} // namespace mlir::vector_plan
```

### 不变量
- `topoMembers` 按 `topo_index` 严格升序，GroupEmitter 按此顺序发射
- `canonicalAxes` 由所有 member 的 `iterator_types` 按上确界规则推导（§3.2）
- `CollapsedGroupInfo.axisMap[i]` 若 axis i 参与 collapse，则映射到新的 collapsed 轴下标；否则 = -1

---

## Phase 2: TilePlan.h

文件路径：`include/Conversion/VectorPlan/TilePlan.h`

```cpp
#pragma once

#include "GroupInfo.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::vector_plan {

enum class TileLevel : uint8_t {
  Outer,  // 外层分核：XBLOCK / BM / BN
  Inner,  // 内层 tile：XBLOCK_SUB / Tb_M / Tb_N / RBLOCK_sub / t_K
  Full,   // 不切：RBLOCK（enableReductionSplit=false 时）
};

enum class TileFieldKind : uint8_t {
  TunableTile, // autotuner 搜索的 tile 参数
  FixedTile,   // 固定值或 full dim（不搜索）
  ShapeDim,    // 运行时 shape 透传到 TilingData
  Derived,     // 由其他字段/shape 推导，不进 TilingData bytes
};

struct TileParam {
  llvm::StringRef name;         // "XBLOCK" / "XBLOCK_SUB" / "RBLOCK_0" / "BM" ...
  Value           ssa;          // pass 插入的 func index arg（loop step）
  OpFoldResult    defaultValue; // 静态 shape → IntegerAttr；动态 → Value
  int32_t         axisIdx;      // 对应 collapsedAxes 的下标
  TileLevel       level;
  AxisRole        role;
};

struct TilePlan {
  const CollapsedGroupInfo *group; // 不拥有，指向 Pass 2 内的 GroupInfo

  // VectorGroup：
  //   tileable[0] = {XBLOCK(Outer), XBLOCK_SUB(Inner)}
  //   tileable[i>0] = {XBLOCK_SUB_i(Inner)}   （其余 parallel 轴，默认 full）
  //   full[j] = {RBLOCK_j(Full 或 Inner，取决于 enableReductionSplit)}
  // CubeGroup：
  //   tileable[M] = {BM(Outer), Tb_M(Inner)}
  //   tileable[N] = {BN(Outer), Tb_N(Inner)}
  //   full[K]     = {t_K(Inner)}
  //   tileable[batch_i] = {XBLOCK_i(Outer), XBLOCK_SUB_i(Inner)}（若有 batch 轴）
  llvm::SmallVector<llvm::SmallVector<TileParam>> tileable;
  llvm::SmallVector<TileParam>                    full;

  // VectorGroup：1 个元素 = ceildiv(tileable[0].extent, XBLOCK)
  // CubeGroup：2 个元素 = {ceildiv(M, BM), ceildiv(N, BN)}
  llvm::SmallVector<OpFoldResult, 2> blockDimExprs;
};

} // namespace mlir::vector_plan
```

---

## Phase 3: TileInfo.h

文件路径：`include/Conversion/VectorPlan/TileInfo.h`

```cpp
#pragma once

#include "GroupInfo.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <string>

namespace mlir::vector_plan {

// 可序列化的表达式树，不含 MLIR Value 指针
struct ValueExpr {
  enum class Kind {
    Const,    // int64_t 常量
    ShapeDim, // tensor arg[argIndex].dim[dimIndex]
    FieldRef, // 引用另一 TileFieldSpec.fieldId
    Mul,
    Add,      // 预留
    CeilDiv,
    Min,      // 预留
    Max,      // 预留
  } kind = Kind::Const;

  int64_t     constValue = 0;
  int32_t     argIndex   = -1;
  int32_t     dimIndex   = -1;
  std::string fieldId;
  std::shared_ptr<ValueExpr> lhs, rhs;

  static ValueExpr makeConst(int64_t v);
  static ValueExpr makeShapeDim(int32_t arg, int32_t dim);
  static ValueExpr makeFieldRef(llvm::StringRef id);
  static ValueExpr makeMul(ValueExpr lhs, ValueExpr rhs);
  static ValueExpr makeCeilDiv(ValueExpr lhs, ValueExpr rhs);
};

// v1 必须支持：Const / ShapeDim / FieldRef / Mul / CeilDiv

struct ShapeRef { int32_t argIndex, dimIndex; };

struct TileAxisInfo {
  int32_t     axisIndex;
  std::string axisName;
  AxisRole    role;
  ValueExpr   extentExpr; // 可以是 B*S 复合表达式
};

struct SearchSpace {
  bool                     enabled = false;
  llvm::SmallVector<int64_t> candidates;
  std::optional<int64_t>   alignment;   // cube BK: K 粒度（FP16=16, INT8=32）
  std::optional<ValueExpr> upperBound;  // cube BK: L0A/L0B 容量（单位 element 数）
};

struct TileFieldSpec {
  std::string              fieldId;   // 稳定语义 ID，如 "tile.xblock"
  std::string              abiName;   // TilingData 字段名，如 "XBLOCK"
  std::string              abiType;   // v1 统一 "i64"
  std::optional<int32_t>   abiIndex;  // TilingData struct 顺序；Derived 为 nullopt

  TileFieldKind            kind;
  std::optional<int32_t>   axisIndex;
  std::optional<TileLevel> level;
  std::optional<ShapeRef>  shapeBinding;  // kind=ShapeDim 时必填
  std::optional<ValueExpr> defaultExpr;   // TunableTile / FixedTile 时填写
  std::optional<SearchSpace> search;      // TunableTile 时填写
};

struct TileInfo {
  std::string kernelId; // "group0_plan0"
  int32_t     groupId;
  int32_t     planId;

  llvm::SmallVector<TileAxisInfo>  axes;
  llvm::SmallVector<TileFieldSpec> fields;

  // VectorGroup：1 个；CubeGroup：2 个（grid_y, grid_x）
  llvm::SmallVector<ValueExpr, 2>  blockDimExprs;
};

// TileInfo 不变量（见 §7.4）：
// 1. 每个 TileInfo 只对应一个 (groupId, planId)
// 2. abiIndex 是 TilingData 字段顺序的唯一权威；Derived.abiIndex == nullopt
// 3. ShapeDim 只引用原始 tensor 参数的 primitive dim
// 4. TileInfo 不持有 MLIR SSA Value
// 5. AutoTuner 只搜索 TunableTile；blockDimExprs 必须显式写出（不靠反推）

} // namespace mlir::vector_plan
```

---

## Phase 4: Passes.td + 空壳 cpp + CMake 接入

### Passes.td 追加（`include/Conversion/Passes.td` 末尾）

```tablegen
def VectorPlanGroupAnalysis : Pass<"vector-plan-group-analysis", "func::FuncOp"> {
  let summary = "Analyze linalg ops and annotate fusion groups";
  let options = [
    Option<"enableReductionSplit",     "enable-reduction-split",     "bool",    "false", "">,
    Option<"maxReduceEpilogueOps",     "max-reduce-epilogue-ops",    "int32_t", "3",     "">,
    Option<"maxHorizontalExtraInputs", "max-horizontal-extra-inputs","int32_t", "4",     "">,
  ];
}

def VectorPlanGroupOutline : Pass<"vector-plan-group-outline", "ModuleOp"> {
  let summary = "Outline annotated fusion groups into kernel funcs";
  let options = [
    Option<"kernelFuncPrefix", "kernel-func-prefix", "std::string",
           "\"kernel_group\"", "">,
    Option<"outputDir", "output-dir", "std::string", "\"\"", "">,
  ];
}

def VectorPlanTileFuse : Pass<"vector-plan-tile-fuse", "func::FuncOp"> {
  let summary = "Tile and fuse kernel group func";
  let options = [
    Option<"enableCollapse",       "enable-collapse",        "bool",    "true",  "">,
    Option<"enableReductionSplit", "enable-reduction-split", "bool",    "false", "">,
    Option<"hwMatmulMAlign",       "hw-matmul-m-align",      "int64_t", "16",    "">,
    Option<"hwMatmulNAlign",       "hw-matmul-n-align",      "int64_t", "16",    "">,
    Option<"hwMatmulKAlign",       "hw-matmul-k-align",      "int64_t", "16",    "">,
    Option<"l0aCapacityBytes",     "l0a-capacity-bytes",     "int64_t", "65536", "">,
  ];
}
```

### 空壳 cpp（三个 pass 格式相同）

`lib/Conversion/VectorPlan/GroupAnalysis/GroupAnalysisPass.cpp`:

```cpp
#include "Conversion/VectorPlan/GroupInfo.h"
#include "Conversion/VectorPlan/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace {
struct VectorPlanGroupAnalysisPass
    : public PassWrapper<VectorPlanGroupAnalysisPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorPlanGroupAnalysisPass)

  void runOnOperation() override {
    // TODO: impl-01
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createVectorPlanGroupAnalysisPass() {
  return std::make_unique<VectorPlanGroupAnalysisPass>();
}
```

### CMakeLists.txt

`lib/Conversion/VectorPlan/CMakeLists.txt`:

```cmake
add_mlir_library(MLIRVectorPlan
  GroupAnalysis/GroupAnalysisPass.cpp
  GroupOutline/GroupOutlinePass.cpp
  TileFuse/TileFusePass.cpp
  TileInfo/TilePlanToTileInfo.cpp

  DEPENDS
  MLIRVectorPlanPassIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRPass
)
```

在 `lib/Conversion/CMakeLists.txt` 末尾追加：

```cmake
add_subdirectory(VectorPlan)
```

---

## 验收测试

```bash
# Phase 4 完成后：三个 pass 均可注册，不报错
mlir-opt --vector-plan-group-analysis /dev/null
mlir-opt --vector-plan-group-outline /dev/null
mlir-opt --vector-plan-tile-fuse /dev/null

# 预期：无 error，输出空 module
```

```mlir
// test/Conversion/VectorPlan/foundation-smoke.mlir
// RUN: mlir-opt --vector-plan-group-analysis %s | FileCheck %s
// CHECK: module

module {}
```
