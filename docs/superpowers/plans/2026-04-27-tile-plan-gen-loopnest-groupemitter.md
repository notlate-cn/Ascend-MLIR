# TilePlanGen + LoopNest + GroupEmitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 2 (TilePlanGen), Phase 3 (LoopNestBuilder + SliceComputer + GroupEmitter), and Phase 4 (Reduction Split + BCast escape) of `vector-plan-tile-fuse`.

**Architecture:** Four new files per component; TileFusePass.cpp wires phases together; all axis indices are post-collapse (`CollapsedGroupInfo::collapsedAxes` numbering); static and dynamic shapes both supported.

**Tech Stack:** MLIR (linalg, scf, tensor, arith, func dialects); LLVM ADT; C++17; afir-opt + FileCheck for tests.

**Spec:** `docs/superpowers/specs/2026-04-27-tile-plan-gen-design.md`

**Build:** `cd build && ninja afir-opt 2>&1 | tail -20`

**Run single test:** `afir-opt test/Conversion/Collapse/<file>.mlir --vector-plan-tile-fuse | FileCheck test/Conversion/Collapse/<file>.mlir`

---

## File Map

| Action | Path |
|--------|------|
| Modify | `include/Conversion/VectorPlan/TilePlan.h` — `name: StringRef` → `std::string` |
| Modify | `include/Conversion/Passes.td` — add `maxFullLoopIters` option |
| Modify | `lib/Conversion/VectorPlan/CMakeLists.txt` — add 5 new source files |
| Create | `lib/Conversion/VectorPlan/TileFuse/TileFuseUtils.h` — shared helpers |
| Create | `lib/Conversion/VectorPlan/TileFuse/TileFuseUtils.cpp` |
| Create | `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.h` |
| Create | `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` |
| Create | `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.h` |
| Create | `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp` |
| Create | `lib/Conversion/VectorPlan/TileFuse/SliceComputer.h` |
| Create | `lib/Conversion/VectorPlan/TileFuse/SliceComputer.cpp` |
| Create | `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.h` |
| Create | `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp` |
| Modify | `lib/Conversion/VectorPlan/TileFuse/TileFusePass.cpp` — wire Phase 2/3/4 |
| Create | `test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir` |
| Create | `test/Conversion/Collapse/tile-fuse-vector-bcast.mlir` |
| Create | `test/Conversion/Collapse/tile-fuse-vector-bcast-escape.mlir` |
| Create | `test/Conversion/Collapse/tile-fuse-vector-reduce.mlir` |
| Create | `test/Conversion/Collapse/tile-fuse-vector-reduce-split.mlir` |

---

## Task 1: Infrastructure Setup

**Files:**
- Modify: `include/Conversion/VectorPlan/TilePlan.h`
- Modify: `include/Conversion/Passes.td`
- Modify: `lib/Conversion/VectorPlan/CMakeLists.txt`
- Create: `lib/Conversion/VectorPlan/TileFuse/TileFuseUtils.h` (stub)
- Create: `lib/Conversion/VectorPlan/TileFuse/TileFuseUtils.cpp` (stub)
- Create: `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.h` (stub)
- Create: `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` (stub)
- Create: `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.h` (stub)
- Create: `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp` (stub)
- Create: `lib/Conversion/VectorPlan/TileFuse/SliceComputer.h` (stub)
- Create: `lib/Conversion/VectorPlan/TileFuse/SliceComputer.cpp` (stub)
- Create: `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.h` (stub)
- Create: `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp` (stub)

- [ ] **Step 1: Change `TileParam::name` from `StringRef` to `std::string`**

In `include/Conversion/VectorPlan/TilePlan.h`, change:
```cpp
struct TileParam {
  std::string  name;        // was: llvm::StringRef name
  Value        ssa;
  OpFoldResult defaultValue;
  int32_t      axisIdx;
  TileLevel    level;
  AxisRole     role;
};
```

- [ ] **Step 2: Add `maxFullLoopIters` option to `Passes.td`**

In `include/Conversion/Passes.td`, find the `VectorPlanTileFuse` pass options block and add after `enableReductionSplit`:
```tablegen
Option<"maxFullLoopIters", "max-full-loop-iters", "int64_t", "2048",
       "BCast axes larger than this threshold escape to tileable Inner">,
```

- [ ] **Step 3: Add new source files to `CMakeLists.txt`**

In `lib/Conversion/VectorPlan/CMakeLists.txt`, add after `TileFuse/TileFusePass.cpp`:
```cmake
  TileFuse/TileFuseUtils.cpp
  TileFuse/TilePlanGen.cpp
  TileFuse/LoopNestBuilder.cpp
  TileFuse/SliceComputer.cpp
  TileFuse/GroupEmitter.cpp
```

Also add to `LINK_LIBS PUBLIC`:
```cmake
  MLIRAffineDialect
  MLIRArithDialect
```
(check if MLIRArithDialect already present; MLIRAffineDialect is new)

- [ ] **Step 4: Create `TileFuseUtils.h` stub**

```cpp
// lib/Conversion/VectorPlan/TileFuse/TileFuseUtils.h
#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir::afir {

// Cast any integer-typed value to index. No-op if already index.
mlir::Value castToIndex(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v);

// Return an index-typed Value holding the extent of post-collapse axis axisIdx.
// Static extent → arith.constant index; dynamic → tensor.dim on first matching operand.
mlir::Value getAxisExtentValue(mlir::OpBuilder &b, mlir::Location loc,
                                const mlir::vector_plan::CollapsedGroupInfo &info,
                                int axisIdx);

} // namespace mlir::afir
```

- [ ] **Step 5: Create `TileFuseUtils.cpp` stub**

```cpp
// lib/Conversion/VectorPlan/TileFuse/TileFuseUtils.cpp
#include "TileFuseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

Value castToIndex(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex()) return v;
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

Value getAxisExtentValue(OpBuilder &b, Location loc,
                          const CollapsedGroupInfo &info, int axisIdx) {
  // TODO Task 2
  return b.create<arith::ConstantIndexOp>(loc, 0);
}

} // namespace mlir::afir
```

- [ ] **Step 6: Create `TilePlanGen.h` stub**

```cpp
// lib/Conversion/VectorPlan/TileFuse/TilePlanGen.h
#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::afir {

mlir::vector_plan::TilePlan
genVectorTilePlan(mlir::func::FuncOp func,
                  const mlir::vector_plan::CollapsedGroupInfo &info,
                  mlir::OpBuilder &builder,
                  mlir::Location loc,
                  bool enableReductionSplit,
                  int64_t maxFullLoopIters);

} // namespace mlir::afir
```

- [ ] **Step 7: Create `TilePlanGen.cpp` stub**

```cpp
// lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp
#include "TilePlanGen.h"
#include "TileFuseUtils.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

TilePlan genVectorTilePlan(func::FuncOp func,
                            const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc,
                            bool enableReductionSplit,
                            int64_t maxFullLoopIters) {
  TilePlan plan;
  plan.group = &info;
  return plan; // TODO Task 2
}

} // namespace mlir::afir
```

- [ ] **Step 8: Create `LoopNestBuilder.h` stub**

```cpp
// lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.h
#pragma once
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

struct LoopNestResult {
  llvm::DenseMap<int, mlir::Value>     loopIVs;       // post-collapse axis idx → IV (composedIV for Outer+Inner pairs)
  llvm::SmallVector<mlir::scf::ForOp>  allForOps;     // outermost to innermost
  llvm::SmallVector<mlir::scf::ForOp>  bcastForOps;   // BCast Full loops (subset of allForOps, for hoist)
  mlir::Block                         *innermostBody = nullptr;
  llvm::SmallVector<mlir::Value>       iterArgs;      // innermost-level iter args
};

LoopNestResult buildLoopNest(mlir::OpBuilder &builder,
                              mlir::Location loc,
                              const mlir::vector_plan::TilePlan &plan,
                              mlir::ValueRange initTensors);

} // namespace mlir::afir
```

- [ ] **Step 9: Create `LoopNestBuilder.cpp` stub**

```cpp
// lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp
#include "LoopNestBuilder.h"
#include "TileFuseUtils.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

LoopNestResult buildLoopNest(OpBuilder &builder, Location loc,
                              const TilePlan &plan, ValueRange initTensors) {
  LoopNestResult result;
  result.innermostBody = builder.getInsertionBlock();
  result.iterArgs = SmallVector<Value>(initTensors);
  return result; // TODO Task 3
}

} // namespace mlir::afir
```

- [ ] **Step 10: Create `SliceComputer.h` stub**

```cpp
// lib/Conversion/VectorPlan/TileFuse/SliceComputer.h
#pragma once
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpFoldResult.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

struct SliceParams {
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides; // all 1
};

// Compute slice params for `tensor` whose linalg indexing map is `indexingMap`.
// loopIVs maps post-collapse axis idx → current loop IV (composedIV for paired axes).
SliceParams computeSlice(mlir::AffineMap indexingMap,
                          const llvm::DenseMap<int, mlir::Value> &loopIVs,
                          const mlir::vector_plan::TilePlan &plan,
                          mlir::Value tensor,
                          mlir::OpBuilder &builder,
                          mlir::Location loc);

} // namespace mlir::afir
```

- [ ] **Step 11: Create `SliceComputer.cpp` stub**

```cpp
// lib/Conversion/VectorPlan/TileFuse/SliceComputer.cpp
#include "SliceComputer.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

SliceParams computeSlice(AffineMap indexingMap,
                          const DenseMap<int, Value> &loopIVs,
                          const TilePlan &plan,
                          Value tensor,
                          OpBuilder &builder, Location loc) {
  return {}; // TODO Task 3
}

} // namespace mlir::afir
```

- [ ] **Step 12: Create `GroupEmitter.h` stub**

```cpp
// lib/Conversion/VectorPlan/TileFuse/GroupEmitter.h
#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "LoopNestBuilder.h"
#include "mlir/IR/Builders.h"

namespace mlir::afir {

// Emit tiled ops into loopNest.innermostBody and propagate scf.yield up the loop chain.
// Returns the outermost for-op results that replace the original op results.
llvm::SmallVector<mlir::Value>
emitGroup(mlir::OpBuilder &builder,
          mlir::Location loc,
          const mlir::vector_plan::CollapsedGroupInfo &info,
          const mlir::vector_plan::TilePlan &plan,
          const LoopNestResult &loopNest);

} // namespace mlir::afir
```

- [ ] **Step 13: Create `GroupEmitter.cpp` stub**

```cpp
// lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp
#include "GroupEmitter.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

SmallVector<Value> emitGroup(OpBuilder &builder, Location loc,
                              const CollapsedGroupInfo &info,
                              const TilePlan &plan,
                              const LoopNestResult &loopNest) {
  return {}; // TODO Task 3
}

} // namespace mlir::afir
```

- [ ] **Step 14: Verify build compiles with stubs**

```bash
cd build && ninja afir-opt 2>&1 | grep -E "error:|warning:" | head -20
```
Expected: no errors (stubs compile cleanly).

- [ ] **Step 15: Commit**

```bash
git add include/Conversion/VectorPlan/TilePlan.h \
        include/Conversion/Passes.td \
        lib/Conversion/VectorPlan/CMakeLists.txt \
        lib/Conversion/VectorPlan/TileFuse/TileFuseUtils.h \
        lib/Conversion/VectorPlan/TileFuse/TileFuseUtils.cpp \
        lib/Conversion/VectorPlan/TileFuse/TilePlanGen.h \
        lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp \
        lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.h \
        lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp \
        lib/Conversion/VectorPlan/TileFuse/SliceComputer.h \
        lib/Conversion/VectorPlan/TileFuse/SliceComputer.cpp \
        lib/Conversion/VectorPlan/TileFuse/GroupEmitter.h \
        lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp
git commit -m "feat(vector-plan): add TilePlanGen+LoopNest+Emitter stub infrastructure"
```

---

## Task 2: TilePlanGen — Parallel Axis Assignment

**Files:**
- Implement: `lib/Conversion/VectorPlan/TileFuse/TileFuseUtils.cpp`
- Implement: `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp`
- Update: `lib/Conversion/VectorPlan/TileFuse/TileFusePass.cpp` (call Phase 2)
- Create: `test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir`

**Axis assignment rules implemented here (parallel only; BCast and reduction in Tasks 4–6):**
| Axis | TileLevel | Name |
|------|-----------|------|
| 1st non-BCast parallel | Outer + Inner | `XBLOCK` + `XBLOCK_SUB` |
| 2nd+ non-BCast parallel | Inner only | `XBLOCK_SUB_1`, `_2`, … |

- [ ] **Step 1: Write failing test `tile-fuse-vector-pointwise.mlir`**

```mlir
// test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir
// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// 1-D pointwise (d0 only, no collapse). After Phase 2, func gains XBLOCK + XBLOCK_SUB args.

// CHECK: func.func @pointwise(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128 : i64}
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16 : i64}

func.func @pointwise(%a: tensor<32768xf32>, %b: tensor<32768xf32>,
                     %c: tensor<32768xf32>) -> tensor<32768xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%a, %b : tensor<32768xf32>, tensor<32768xf32>)
    outs(%c : tensor<32768xf32>) {
  ^bb0(%a0: f32, %b0: f32, %c0: f32):
    %add = arith.addf %a0, %b0 : f32
    linalg.yield %add : f32
  } -> tensor<32768xf32>
  return %result : tensor<32768xf32>
}
```

- [ ] **Step 2: Run test to verify it currently fails (no func args added yet)**

```bash
afir-opt test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir \
  --vector-plan-tile-fuse 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir
```
Expected: FileCheck fails (no `vector_plan.default_tile_size` attrs in output).

- [ ] **Step 3: Implement `getAxisExtentValue` in `TileFuseUtils.cpp`**

```cpp
Value getAxisExtentValue(OpBuilder &b, Location loc,
                          const CollapsedGroupInfo &info, int axisIdx) {
  int64_t staticSize = info.collapsedAxes[axisIdx].staticSize;
  if (staticSize != ShapedType::kDynamic)
    return b.create<arith::ConstantIndexOp>(loc, staticSize);

  // Dynamic: find first operand with this axis in its indexing map.
  for (LinalgOp op : info.topoMembers) {
    auto maps     = op.getIndexingMapsArray();
    auto operands = op->getOperands();
    for (auto [operand, map] : llvm::zip(operands, maps)) {
      if (!isa<RankedTensorType>(operand.getType())) continue;
      for (auto [dimPos, expr] : llvm::enumerate(map.getResults())) {
        auto d = dyn_cast<AffineDimExpr>(expr);
        if (d && (int)d.getPosition() == axisIdx)
          return b.create<tensor::DimOp>(loc, operand, (int64_t)dimPos);
      }
    }
  }
  llvm_unreachable("axis not found in any operand map");
}
```

Required includes in TileFuseUtils.cpp:
```cpp
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
```

- [ ] **Step 4: Implement `insertFuncArg` and `genVectorTilePlan` in `TilePlanGen.cpp`**

```cpp
// TilePlanGen.cpp — full implementation
#include "TilePlanGen.h"
#include "TileFuseUtils.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

static Value insertFuncArg(func::FuncOp func, OpBuilder &builder,
                            Location loc, int64_t defaultVal,
                            StringRef paramName) {
  unsigned idx = func.getNumArguments();
  SmallVector<Type> argTypes(func.getFunctionType().getInputs());
  argTypes.push_back(builder.getIndexType());
  func.setType(FunctionType::get(builder.getContext(), argTypes,
                                  func.getFunctionType().getResults()));
  Value newArg =
      func.getBody().front().addArgument(builder.getIndexType(), loc);
  func.setArgAttrs(idx,
                   {NamedAttribute(
                       builder.getStringAttr("vector_plan.default_tile_size"),
                       builder.getI64IntegerAttr(defaultVal))});
  return newArg;
}

TilePlan genVectorTilePlan(func::FuncOp func,
                            const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc,
                            bool enableReductionSplit,
                            int64_t maxFullLoopIters) {
  TilePlan plan;
  plan.group = &info;

  DenseSet<int> bcastSet(info.broadcastAxes.begin(), info.broadcastAxes.end());
  int parallelCount = 0, bcastCount = 0, bcastTileCount = 0, rblockCount = 0;

  for (int i = 0; i < (int)info.collapsedAxes.size(); ++i) {
    const AxisInfo &ax = info.collapsedAxes[i];
    Value ext = getAxisExtentValue(builder, loc, info, i);

    if (ax.role == AxisRole::Parallel && !bcastSet.count(i)) {
      // Non-BCast parallel axis.
      if (parallelCount == 0) {
        // First parallel: Outer XBLOCK + Inner XBLOCK_SUB pair.
        Value xblock    = insertFuncArg(func, builder, loc, 128, "XBLOCK");
        Value xblockSub = insertFuncArg(func, builder, loc, 16,  "XBLOCK_SUB");
        SmallVector<TileParam> group;
        group.push_back({"XBLOCK",     xblock,    OpFoldResult(ext),    i,
                          TileLevel::Outer, AxisRole::Parallel});
        group.push_back({"XBLOCK_SUB", xblockSub, OpFoldResult(xblock), i,
                          TileLevel::Inner, AxisRole::Parallel});
        plan.tileable.push_back(std::move(group));
        // blockDimExprs: ceildivsi(extent, XBLOCK).
        Value blockCount =
            builder.create<arith::CeilDivSIOp>(loc, ext, xblock);
        plan.blockDimExprs.push_back(OpFoldResult(blockCount));
      } else {
        // Additional parallel: Inner only.
        std::string name =
            llvm::formatv("XBLOCK_SUB_{0}", parallelCount - 1).str();
        Value param = insertFuncArg(func, builder, loc, 16, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i,
                          TileLevel::Inner, AxisRole::Parallel});
        plan.tileable.push_back(std::move(group));
      }
      ++parallelCount;

    } else if (bcastSet.count(i)) {
      // BCast axis — handled in Task 4 (BCast Full) and Task 5 (escape).
      // For now emit a Full placeholder so builds don't assert.
      std::string name = llvm::formatv("BCAST_{0}", bcastCount).str();
      Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
      plan.full.push_back({name, step, OpFoldResult(ext), i,
                            TileLevel::Full, AxisRole::Parallel});
      ++bcastCount;

    } else {
      // Reduction axis — handled in Task 6.
      std::string name = llvm::formatv("RBLOCK_{0}", rblockCount++).str();
      plan.full.push_back({name, ext, OpFoldResult(ext), i,
                            TileLevel::Full, AxisRole::Reduction});
    }
  }
  return plan;
}

} // namespace mlir::afir
```

- [ ] **Step 5: Update `TileFusePass.cpp` to call Phase 2**

In `TileFusePass.cpp` `runOnOperation`, after `(void)collapsedInfo;` remove that line and add:

```cpp
    // Phase 2: TilePlanGen.
    auto plan = genVectorTilePlan(func, collapsedInfo, builder, func.getLoc(),
                                  enableReductionSplit, maxFullLoopIters);
    (void)plan; // consumed by Phase 3 — TODO
```

Add includes at top of TileFusePass.cpp:
```cpp
#include "TilePlanGen.h"
```

- [ ] **Step 6: Build**

```bash
cd build && ninja afir-opt 2>&1 | grep -E "error:" | head -20
```
Expected: no errors.

- [ ] **Step 7: Run test to verify it passes**

```bash
afir-opt test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir \
  --vector-plan-tile-fuse 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir
```
Expected: FileCheck passes — func has XBLOCK and XBLOCK_SUB args with `vector_plan.default_tile_size` attrs.

- [ ] **Step 8: Commit**

```bash
git add lib/Conversion/VectorPlan/TileFuse/TileFuseUtils.cpp \
        lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp \
        lib/Conversion/VectorPlan/TileFuse/TileFusePass.cpp \
        test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir
git commit -m "feat(vector-plan): implement TilePlanGen parallel-axis assignment"
```

---

## Task 3: LoopNestBuilder + SliceComputer + GroupEmitter — Pointwise End-to-End

**Files:**
- Implement: `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp`
- Implement: `lib/Conversion/VectorPlan/TileFuse/SliceComputer.cpp`
- Implement: `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp`
- Update: `lib/Conversion/VectorPlan/TileFuse/TileFusePass.cpp` (wire Phase 3)

**Design recap:**
- LoopNestBuilder emits Outer loops ({ascendc.parallel}) then Inner loops; leaves innermost body empty.
- SliceComputer maps axis IVs + tile sizes → `tensor.extract_slice` params.
- GroupEmitter emits ops + `tensor.insert_slice` + `scf.yield` chain up all loops.

- [ ] **Step 1: Update test to check full tiled structure**

Add to `tile-fuse-vector-pointwise.mlir` (after existing CHECKs):

```mlir
// CHECK: scf.for %[[OUTER:[^ ]*]] = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK-SAME: {ascendc.parallel}
// CHECK: scf.for %[[INNER:[^ ]*]] = %{{.*}} to %[[XBLOCK]] step %[[XBLOCK_SUB]]
// CHECK: %[[IV:[^ ]*]] = arith.addi %[[OUTER]], %[[INNER]]
// CHECK: tensor.extract_slice
// CHECK: linalg.generic
// CHECK: tensor.insert_slice
// CHECK: scf.yield
```

- [ ] **Step 2: Run test to verify new CHECKs fail**

```bash
afir-opt test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir \
  --vector-plan-tile-fuse 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir
```
Expected: fails at the `scf.for` check.

- [ ] **Step 3: Implement `LoopNestBuilder.cpp`**

Required includes:
```cpp
#include "LoopNestBuilder.h"
#include "TileFuseUtils.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
```

Implementation:
```cpp
using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

LoopNestResult buildLoopNest(OpBuilder &builder, Location loc,
                              const TilePlan &plan, ValueRange initTensors) {
  LoopNestResult result;
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> iterArgs(initTensors);
  DenseMap<int, Value> outerIVs; // axisIdx → outer loop IV

  auto emitFor = [&](Value lb, Value ub, Value step,
                     bool isParallelOuter) -> scf::ForOp {
    auto forOp = builder.create<scf::ForOp>(loc, lb, ub, step, iterArgs);
    if (isParallelOuter)
      forOp->setAttr("ascendc.parallel", builder.getUnitAttr());
    result.allForOps.push_back(forOp);
    builder.setInsertionPointToStart(forOp.getBody());
    iterArgs = SmallVector<Value>(forOp.getRegionIterArgs());
    return forOp;
  };

  // 1. Outer loops (TileLevel::Outer).
  for (auto &group : plan.tileable) {
    for (const auto &tp : group) {
      if (tp.level != TileLevel::Outer) continue;
      Value ub = getAxisExtentValue(builder, loc, *plan.group, tp.axisIdx);
      auto forOp = emitFor(c0, ub, tp.ssa, /*isParallelOuter=*/true);
      outerIVs[tp.axisIdx] = forOp.getInductionVar();
    }
  }

  // 2. BCast Full loops (plan.full, AxisRole::Parallel, TileLevel::Full).
  for (const auto &tp : plan.full) {
    if (tp.role != AxisRole::Parallel) continue;
    Value ub = getAxisExtentValue(builder, loc, *plan.group, tp.axisIdx);
    auto forOp = emitFor(c0, ub, tp.ssa, /*isParallelOuter=*/false);
    result.bcastForOps.push_back(forOp);
    result.loopIVs[tp.axisIdx] = forOp.getInductionVar();
  }

  // 3. Inner loops (TileLevel::Inner), sorted by axisIdx.
  SmallVector<const TileParam *> innerParams;
  for (auto &group : plan.tileable)
    for (const auto &tp : group)
      if (tp.level == TileLevel::Inner)
        innerParams.push_back(&tp);
  llvm::sort(innerParams, [](const TileParam *a, const TileParam *b) {
    return a->axisIdx < b->axisIdx;
  });

  for (const TileParam *tp : innerParams) {
    // UB = Outer tile size if paired, else full extent.
    Value ub;
    if (outerIVs.count(tp->axisIdx)) {
      for (auto &group : plan.tileable)
        for (const auto &op : group)
          if (op.level == TileLevel::Outer && op.axisIdx == tp->axisIdx)
            ub = op.ssa;
    }
    if (!ub)
      ub = getAxisExtentValue(builder, loc, *plan.group, tp->axisIdx);
    auto forOp = emitFor(c0, ub, tp->ssa, /*isParallelOuter=*/false);

    // Build composedIV = outerIV + innerIV for paired axes.
    if (outerIVs.count(tp->axisIdx)) {
      Value composed = builder.create<arith::AddIOp>(
          loc, outerIVs[tp->axisIdx], forOp.getInductionVar());
      result.loopIVs[tp->axisIdx] = composed;
    } else {
      result.loopIVs[tp->axisIdx] = forOp.getInductionVar();
    }
  }

  result.innermostBody = builder.getInsertionBlock();
  result.iterArgs = iterArgs;
  return result;
}

} // namespace mlir::afir
```

- [ ] **Step 4: Implement `SliceComputer.cpp`**

Key logic: for each result dimension of `indexingMap`, if the axis has a loop IV → (offset=IV, size=tileSize), else → (offset=0, size=full dim).

```cpp
#include "SliceComputer.h"
#include "TileFuseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

// Return the tile size Value for axis axisIdx from plan.
// BCast Full axes return constant 1. Reduction Full axes return full extent.
// Inner axes return their ssa (XBLOCK_SUB etc.).
static Value getTileSizeForAxis(const TilePlan &plan, int axisIdx,
                                 OpBuilder &b, Location loc) {
  // Check plan.full first (Full-level axes).
  for (const auto &tp : plan.full) {
    if (tp.axisIdx != axisIdx) continue;
    if (tp.role == AxisRole::Parallel) // BCast Full: step=1
      return b.create<arith::ConstantIndexOp>(loc, 1);
    // Reduction Full: full extent (but this axis has no IV so this branch
    // is only reached from SliceComputer's "no IV" path, which uses tensor.dim).
    // Return extent Value stored in tp.ssa.
    return tp.ssa;
  }
  // Check tileable: return innermost param's ssa for this axis.
  for (auto &group : plan.tileable) {
    for (const auto &tp : group) {
      if (tp.axisIdx != axisIdx) continue;
      if (tp.level == TileLevel::Inner)
        return tp.ssa;
    }
  }
  return b.create<arith::ConstantIndexOp>(loc, 0); // unreachable
}

SliceParams computeSlice(AffineMap indexingMap,
                          const DenseMap<int, Value> &loopIVs,
                          const TilePlan &plan,
                          Value tensor,
                          OpBuilder &builder, Location loc) {
  SliceParams sp;
  Value c0   = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1   = builder.create<arith::ConstantIndexOp>(loc, 1);
  int numDims = (int)indexingMap.getNumResults();

  for (int dimPos = 0; dimPos < numDims; ++dimPos) {
    AffineExpr expr = indexingMap.getResult(dimPos);
    auto d = dyn_cast<AffineDimExpr>(expr);
    if (d && loopIVs.count((int)d.getPosition())) {
      // This axis has a loop IV.
      int axisIdx = (int)d.getPosition();
      sp.offsets.push_back(OpFoldResult(loopIVs.lookup(axisIdx)));
      sp.sizes.push_back(
          OpFoldResult(getTileSizeForAxis(plan, axisIdx, builder, loc)));
    } else {
      // No IV (Reduction Full or non-dim expr): full dimension.
      sp.offsets.push_back(OpFoldResult(c0));
      Value dimSize =
          builder.create<tensor::DimOp>(loc, tensor, (int64_t)dimPos);
      sp.sizes.push_back(OpFoldResult(dimSize));
    }
    sp.strides.push_back(OpFoldResult(c1));
  }
  return sp;
}

} // namespace mlir::afir
```

- [ ] **Step 5: Implement `GroupEmitter.cpp` (baseline — parallel only, no BCast hoist)**

```cpp
#include "GroupEmitter.h"
#include "SliceComputer.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

SmallVector<Value> emitGroup(OpBuilder &builder, Location loc,
                              const CollapsedGroupInfo &info,
                              const TilePlan &plan,
                              const LoopNestResult &loopNest) {
  // Collect all boundary outputs and map them to current iter args.
  // Boundary outs are the DPS inits of all topoMembers that come from outside the group.
  DenseSet<Value> groupResults;
  for (LinalgOp op : info.topoMembers)
    for (Value r : op->getResults())
      groupResults.insert(r);

  // allOuts: DPS inits of all ops (in topo order, deduped).
  SmallVector<Value> allOuts;
  DenseSet<Value> seenOuts;
  for (LinalgOp op : info.topoMembers)
    for (Value out : op.getDpsInits())
      if (seenOuts.insert(out).second)
        allOuts.push_back(out);

  // Map original out tensor → current iter arg.
  DenseMap<Value, Value> outToIterArg;
  for (auto [out, iterArg] : llvm::zip(allOuts, loopNest.iterArgs))
    outToIterArg[out] = iterArg;

  // tiledValues: original SSA value → tiled SSA value (for interior producers).
  DenseMap<Value, Value> tiledValues;
  SmallVector<Value> insertedOuts; // updated out tensors to yield

  builder.setInsertionPointToStart(loopNest.innermostBody);

  for (LinalgOp op : info.topoMembers) {
    auto maps    = op.getIndexingMapsArray();
    auto operands = op->getOperands();
    int numInputs = op.getNumDpsInputs();

    SmallVector<Value> newOperands;

    // Inputs.
    for (auto [idx, operand] : llvm::enumerate(operands)) {
      if ((int)idx >= (int)op.getNumDpsInputs() + (int)op.getNumDpsInits())
        break;
      bool isInit = ((int)idx >= numInputs);
      AffineMap map = maps[idx];

      if (isInit) {
        // DPS init: use iter arg (possibly updated by previous op).
        Value iterArg = outToIterArg.lookup(operand);
        if (!iterArg) iterArg = operand; // fallback
        newOperands.push_back(iterArg);
        continue;
      }

      // Input: check if interior (produced by previous tiled op).
      if (tiledValues.count(operand)) {
        newOperands.push_back(tiledValues[operand]);
        continue;
      }

      // Boundary input: emit extract_slice.
      auto sp = computeSlice(map, loopNest.loopIVs, plan, operand, builder, loc);
      auto slicedType = tensor::ExtractSliceOp::inferResultType(
          cast<RankedTensorType>(operand.getType()), sp.offsets, sp.sizes,
          sp.strides);
      Value sliced = builder.create<tensor::ExtractSliceOp>(
          loc, slicedType, operand, sp.offsets, sp.sizes, sp.strides);
      newOperands.push_back(sliced);
    }

    // Clone op, set new operands.
    Operation *cloned = builder.clone(*op.getOperation());
    for (auto [idx, val] : llvm::enumerate(newOperands))
      cloned->setOperand((unsigned)idx, val);
    // Map original results to cloned results.
    for (auto [origRes, newRes] :
         llvm::zip(op->getResults(), cloned->getResults()))
      tiledValues[origRes] = newRes;
  }

  // Emit insert_slice for each boundary out and collect updated tensors.
  SmallVector<Value> yieldVals;
  for (auto [origOut, iterArg] : llvm::zip(allOuts, loopNest.iterArgs)) {
    // Find the tiled result for origOut (the result of the op that wrote it).
    Value tiledOut;
    for (LinalgOp op : info.topoMembers) {
      for (auto [outOperand, result] :
           llvm::zip(op.getDpsInits(), op->getResults())) {
        if (outOperand == origOut && tiledValues.count(result)) {
          tiledOut = tiledValues[result];
          break;
        }
      }
      if (tiledOut) break;
    }
    if (!tiledOut) { yieldVals.push_back(iterArg); continue; }

    // Compute slice for the output tensor.
    // Find the output's indexing map from the last op writing it.
    AffineMap outMap;
    for (LinalgOp op : info.topoMembers) {
      auto maps = op.getIndexingMapsArray();
      int numIn = op.getNumDpsInputs();
      for (auto [i, out] : llvm::enumerate(op.getDpsInits())) {
        if (out == origOut) { outMap = maps[numIn + (int)i]; break; }
      }
      if (outMap) break;
    }
    auto sp = computeSlice(outMap, loopNest.loopIVs, plan, iterArg,
                            builder, loc);
    Value inserted = builder.create<tensor::InsertSliceOp>(
        loc, tiledOut, iterArg, sp.offsets, sp.sizes, sp.strides);
    yieldVals.push_back(inserted);
  }

  // Emit scf.yield in innermost body.
  builder.create<scf::YieldOp>(loc, yieldVals);

  // Propagate yields up through enclosing loops (inner to outer).
  for (int i = (int)loopNest.allForOps.size() - 2; i >= 0; --i) {
    scf::ForOp inner = loopNest.allForOps[i + 1];
    scf::ForOp outer = loopNest.allForOps[i];
    builder.setInsertionPointToEnd(outer.getBody());
    builder.create<scf::YieldOp>(loc, inner.getResults());
  }

  // Return outermost for-op results (replace original op results outside).
  if (loopNest.allForOps.empty()) return yieldVals;
  return SmallVector<Value>(loopNest.allForOps.front().getResults());
}

} // namespace mlir::afir
```

- [ ] **Step 6: Update `TileFusePass.cpp` to wire full Phase 3**

Replace `(void)plan;` with:

```cpp
    // Collect original op results and init tensors BEFORE modification.
    SmallVector<Value> originalResults;
    SmallVector<Value> initTensors;
    DenseSet<Value> seenInits;
    for (LinalgOp op : collapsedInfo.topoMembers) {
      for (Value r : op->getResults())
        originalResults.push_back(r);
      for (Value out : op.getDpsInits())
        if (seenInits.insert(out).second)
          initTensors.push_back(out);
    }

    // Phase 3a: LoopNestBuilder.
    builder.setInsertionPoint(collapsedInfo.topoMembers.front());
    auto loopNest = buildLoopNest(builder, func.getLoc(), plan, initTensors);

    // Phase 3b+c: GroupEmitter.
    builder.setInsertionPointToStart(loopNest.innermostBody);
    auto loopResults = emitGroup(builder, func.getLoc(), collapsedInfo, plan, loopNest);

    // Replace original results with loop results, then erase original ops.
    for (auto [origRes, loopRes] : llvm::zip(originalResults, loopResults))
      origRes.replaceAllUsesWith(loopRes);
    for (LinalgOp op : collapsedInfo.topoMembers)
      op->erase();
```

Add includes at top of TileFusePass.cpp:
```cpp
#include "GroupEmitter.h"
#include "LoopNestBuilder.h"
#include "SliceComputer.h"
```

- [ ] **Step 7: Build**

```bash
cd build && ninja afir-opt 2>&1 | grep -E "error:" | head -20
```

- [ ] **Step 8: Run pointwise test**

```bash
afir-opt test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir \
  --vector-plan-tile-fuse 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir
```
Expected: all CHECKs pass — outer scf.for with `{ascendc.parallel}`, inner scf.for, composedIV, extract_slice, linalg.generic, insert_slice, scf.yield.

- [ ] **Step 9: Commit**

```bash
git add lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp \
        lib/Conversion/VectorPlan/TileFuse/SliceComputer.cpp \
        lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp \
        lib/Conversion/VectorPlan/TileFuse/TileFusePass.cpp
git commit -m "feat(vector-plan): implement LoopNestBuilder + SliceComputer + GroupEmitter baseline"
```

---

## Task 4: BCast Full Loops + Load Hoist

**Files:**
- Update: `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` (replace BCast placeholder)
- Update: `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp` (add load hoist)
- Create: `test/Conversion/Collapse/tile-fuse-vector-bcast.mlir`

**BCast test structure:** 3 parallel axes (d0=1024, d1=512, d2=16). Input `b` reads only `(d0,d1)` (missing d2) so d2 is BCast. Collapse group = [d0,d1] → collapsed to d0'=524288. Post-collapse: axis 0 = d0'(524288), axis 1 = d2(16, BCast).

- [ ] **Step 1: Create `tile-fuse-vector-bcast.mlir`**

```mlir
// test/Conversion/Collapse/tile-fuse-vector-bcast.mlir
// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// 3-D pointwise: d0(1024),d1(512) collapse; d2(16) is BCast (b misses d2).
// Expected post-collapse axes: [d0'(524288), d2(16,BCast)].
// Expected loops: XBLOCK(Outer) → BCAST_0(step=1) → XBLOCK_SUB(Inner).
// Extract_slice of %b must be hoisted before the BCAST loop.

// CHECK: func.func @bcast_op(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16

// CHECK: scf.for %[[OUT:[^ ]*]] = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK-SAME: {ascendc.parallel}
// CHECK: %[[B_SLICE:[^ ]*]] = tensor.extract_slice %{{.*}}[%[[OUT]]]
// CHECK-NOT: scf.for
// CHECK: scf.for %[[BCAST:[^ ]*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: scf.for %[[INNER:[^ ]*]] = %{{.*}} to %[[XBLOCK]] step %[[XBLOCK_SUB]]

func.func @bcast_op(%a: tensor<1024x512x16xf32>,
                    %b: tensor<1024x512xf32>,
                    %c: tensor<1024x512x16xf32>) -> tensor<1024x512x16xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%a, %b : tensor<1024x512x16xf32>, tensor<1024x512xf32>)
    outs(%c : tensor<1024x512x16xf32>) {
  ^bb0(%a0: f32, %b0: f32, %c0: f32):
    %mul = arith.mulf %a0, %b0 : f32
    linalg.yield %mul : f32
  } -> tensor<1024x512x16xf32>
  return %result : tensor<1024x512x16xf32>
}
```

- [ ] **Step 2: Run test to verify it currently fails**

```bash
afir-opt test/Conversion/Collapse/tile-fuse-vector-bcast.mlir \
  --vector-plan-tile-fuse 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-bcast.mlir
```
Expected: fails (BCast loop appears in wrong order or extract_slice not hoisted).

- [ ] **Step 3: Update `TilePlanGen.cpp` BCast branch (replace placeholder)**

In the `bcastSet.count(i)` branch, replace the placeholder with:

```cpp
    } else if (bcastSet.count(i)) {
      bool escape = (ax.staticSize == ShapedType::kDynamic) ||
                    (ax.staticSize > maxFullLoopIters);
      if (!escape) {
        // BCast Full: step=1, iterate full extent.
        std::string name = llvm::formatv("BCAST_{0}", bcastCount).str();
        Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
        plan.full.push_back({name, step, OpFoldResult(ext), i,
                              TileLevel::Full, AxisRole::Parallel});
      } else {
        // BCast escape: treat as tileable Inner (handled in Task 5).
        std::string name = llvm::formatv("BCAST_TILE_{0}", bcastTileCount++).str();
        Value param = insertFuncArg(func, builder, loc, 16, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i,
                          TileLevel::Inner, AxisRole::Parallel});
        plan.tileable.push_back(std::move(group));
      }
      ++bcastCount;
```

- [ ] **Step 4: Add load hoist to `GroupEmitter.cpp`**

Add `computeHoistPoint` helper before the `emitGroup` function:

```cpp
// Compute insertion point for an extract_slice by hoisting it past BCast loops
// whose IVs don't appear in `offsets`. Returns a {block, iterator} pair.
static std::pair<Block *, Block::iterator>
computeHoistPoint(const SmallVector<scf::ForOp> &bcastForOps,
                   const SmallVector<OpFoldResult> &offsets) {
  // Collect all Values referenced in offsets.
  DenseSet<Value> offsetVals;
  for (OpFoldResult ofr : offsets)
    if (auto v = dyn_cast<Value>(ofr))
      offsetVals.insert(v);

  // Start from innermost BCast loop and hoist past loops whose IV is absent.
  Block *hoistBlock = nullptr;
  Block::iterator hoistIt;
  for (auto it = bcastForOps.rbegin(); it != bcastForOps.rend(); ++it) {
    scf::ForOp bcastFor = *it;
    if (offsetVals.count(bcastFor.getInductionVar())) break; // IV used — stop
    hoistBlock = bcastFor->getBlock();
    hoistIt    = Block::iterator(bcastFor);
  }
  return {hoistBlock, hoistIt};
}
```

In the main `emitGroup` boundary-input section, replace the plain `emitFor` with hoist-aware emission:

```cpp
      // Boundary input: compute hoist point, then emit extract_slice.
      auto sp = computeSlice(map, loopNest.loopIVs, plan, operand, builder, loc);
      auto [hoistBlock, hoistIt] =
          computeHoistPoint(loopNest.bcastForOps, sp.offsets);
      OpBuilder::InsertionGuard guard(builder);
      if (hoistBlock) builder.setInsertionPoint(hoistBlock, hoistIt);
      auto slicedType = tensor::ExtractSliceOp::inferResultType(
          cast<RankedTensorType>(operand.getType()), sp.offsets, sp.sizes,
          sp.strides);
      Value sliced = builder.create<tensor::ExtractSliceOp>(
          loc, slicedType, operand, sp.offsets, sp.sizes, sp.strides);
      newOperands.push_back(sliced);
```

- [ ] **Step 5: Build**

```bash
cd build && ninja afir-opt 2>&1 | grep -E "error:" | head -20
```

- [ ] **Step 6: Run BCast test**

```bash
afir-opt test/Conversion/Collapse/tile-fuse-vector-bcast.mlir \
  --vector-plan-tile-fuse 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-bcast.mlir
```
Expected: passes — BCast loop nested between Outer and Inner; `%b` extract_slice hoisted before BCast loop.

- [ ] **Step 7: Re-run pointwise test to confirm no regression**

```bash
afir-opt test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir \
  --vector-plan-tile-fuse 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir
```

- [ ] **Step 8: Commit**

```bash
git add lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp \
        lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp \
        test/Conversion/Collapse/tile-fuse-vector-bcast.mlir
git commit -m "feat(vector-plan): BCast Full loops + load hoist in GroupEmitter"
```

---

## Task 5: BCast Escape to Tileable Inner

**Files:**
- Create: `test/Conversion/Collapse/tile-fuse-vector-bcast-escape.mlir`

**The BCast escape path is already wired in Task 4's TilePlanGen update** (the `else` branch emits `BCAST_TILE_0` as Inner). This task adds the test to confirm it works.

- [ ] **Step 1: Create `tile-fuse-vector-bcast-escape.mlir`**

Same structure as bcast.mlir but change the BCast axis from size 16 to size 32, and pass `max-full-loop-iters=8` so 32 > 8 triggers escape.

```mlir
// test/Conversion/Collapse/tile-fuse-vector-bcast-escape.mlir
// RUN: afir-opt %s --vector-plan-tile-fuse="max-full-loop-iters=8" 2>&1 | FileCheck %s
//
// BCast axis d2 has size 32 > max-full-loop-iters=8 → escapes to tileable Inner.
// Expected: NO BCast Full loop; instead d2 gets BCAST_TILE_0 Inner tiling.
// The func gains XBLOCK, XBLOCK_SUB, and BCAST_TILE_0 args.

// CHECK: func.func @bcast_escape(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16
// CHECK-SAME: %[[BCAST_TILE:[^ ,)]*]]: index {vector_plan.default_tile_size = 16

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK-SAME: {ascendc.parallel}
// CHECK-NOT: scf.for %{{.*}} to %c32
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[XBLOCK]] step %[[XBLOCK_SUB]]
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[BCAST_TILE]]

func.func @bcast_escape(%a: tensor<1024x512x32xf32>,
                         %b: tensor<1024x512xf32>,
                         %c: tensor<1024x512x32xf32>) -> tensor<1024x512x32xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%a, %b : tensor<1024x512x32xf32>, tensor<1024x512xf32>)
    outs(%c : tensor<1024x512x32xf32>) {
  ^bb0(%a0: f32, %b0: f32, %c0: f32):
    %mul = arith.mulf %a0, %b0 : f32
    linalg.yield %mul : f32
  } -> tensor<1024x512x32xf32>
  return %result : tensor<1024x512x32xf32>
}
```

- [ ] **Step 2: Run escape test**

```bash
afir-opt test/Conversion/Collapse/tile-fuse-vector-bcast-escape.mlir \
  "--vector-plan-tile-fuse=max-full-loop-iters=8" 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-bcast-escape.mlir
```
Expected: passes — BCAST_TILE_0 arg present, no BCast Full scf.for.

- [ ] **Step 3: Commit**

```bash
git add test/Conversion/Collapse/tile-fuse-vector-bcast-escape.mlir
git commit -m "test(vector-plan): add BCast escape test (max-full-loop-iters)"
```

---

## Task 6: Reduction Full + Phase 4 Reduction Split

**Files:**
- Update: `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` (Reduction Inner path)
- Update: `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp` (Phase 4 split)
- Create: `test/Conversion/Collapse/tile-fuse-vector-reduce.mlir`
- Create: `test/Conversion/Collapse/tile-fuse-vector-reduce-split.mlir`

**Reduction Full (no split):** reduction axis has no loop IV; SliceComputer emits full-dim slice (offset=0, size=full). No extra logic needed in LoopNestBuilder since `plan.full` Reduction entries are simply skipped (no loop emitted).

**Phase 4 Split:** when a `TileParam` in `plan.tileable` has `AxisRole::Reduction`, GroupEmitter must:
1. Init accumulator via `tensor.empty` + `linalg.fill(0)`
2. Build an inner `scf.for RBLOCK(0..extent step rblock_size)` with accumulator as iter arg
3. Emit the reduction op inside RBLOCK
4. After RBLOCK: any epilogue ops use the accumulated result
5. Insert final result and yield

- [ ] **Step 1: Create `tile-fuse-vector-reduce.mlir`**

```mlir
// test/Conversion/Collapse/tile-fuse-vector-reduce.mlir
// RUN: afir-opt %s --vector-plan-tile-fuse 2>&1 | FileCheck %s
//
// 2-D reduction (d0 parallel, d1 reduction). No collapse (different roles).
// Reduction axis d1 is Full (no RBLOCK loop). The extract_slice of %a takes full d1 extent.

// CHECK: func.func @reduce(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK-SAME: {ascendc.parallel}
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[XBLOCK]] step %[[XBLOCK_SUB]]
// CHECK: tensor.extract_slice %{{.*}}[%{{.*}}, 0] [%[[XBLOCK_SUB]], 512]
// CHECK: linalg.generic
// CHECK: tensor.insert_slice

func.func @reduce(%a: tensor<1024x512xf32>,
                  %c: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%a : tensor<1024x512xf32>)
    outs(%c : tensor<1024xf32>) {
  ^bb0(%a0: f32, %acc: f32):
    %add = arith.addf %a0, %acc : f32
    linalg.yield %add : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
```

- [ ] **Step 2: Create `tile-fuse-vector-reduce-split.mlir`**

```mlir
// test/Conversion/Collapse/tile-fuse-vector-reduce-split.mlir
// RUN: afir-opt %s --vector-plan-tile-fuse="enable-reduction-split=true" 2>&1 | FileCheck %s
//
// Phase 4: reduction split. d1 becomes RBLOCK_0 Inner (tunable).
// Expected: linalg.fill(0), scf.for RBLOCK, linalg.generic inside, insert_slice after.

// CHECK: func.func @reduce_split(
// CHECK-SAME: %[[XBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 128
// CHECK-SAME: %[[XBLOCK_SUB:[^ ,)]*]]: index {vector_plan.default_tile_size = 16
// CHECK-SAME: %[[RBLOCK:[^ ,)]*]]: index {vector_plan.default_tile_size = 64

// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[XBLOCK]]
// CHECK-SAME: {ascendc.parallel}
// CHECK: scf.for %{{.*}} = %{{.*}} to %[[XBLOCK]] step %[[XBLOCK_SUB]]
// CHECK: linalg.fill
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[RBLOCK]]
// CHECK: linalg.generic
// CHECK: scf.yield
// CHECK: tensor.insert_slice

func.func @reduce_split(%a: tensor<1024x512xf32>,
                         %c: tensor<1024xf32>) -> tensor<1024xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%a : tensor<1024x512xf32>)
    outs(%c : tensor<1024xf32>) {
  ^bb0(%a0: f32, %acc: f32):
    %add = arith.addf %a0, %acc : f32
    linalg.yield %add : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}
```

- [ ] **Step 3: Run both tests to verify they currently fail**

```bash
afir-opt test/Conversion/Collapse/tile-fuse-vector-reduce.mlir \
  --vector-plan-tile-fuse 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-reduce.mlir

afir-opt test/Conversion/Collapse/tile-fuse-vector-reduce-split.mlir \
  "--vector-plan-tile-fuse=enable-reduction-split=true" 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-reduce-split.mlir
```
Expected: both fail.

- [ ] **Step 4: Update `TilePlanGen.cpp` Reduction branch**

Replace the reduction `else` branch (currently always emits Full):

```cpp
    } else {
      // Reduction axis.
      std::string name = llvm::formatv("RBLOCK_{0}", rblockCount++).str();
      if (!enableReductionSplit) {
        // Full: single iteration covering the entire reduction extent.
        // ssa = extent (step = extent → one iteration).
        plan.full.push_back({name, ext, OpFoldResult(ext), i,
                              TileLevel::Full, AxisRole::Reduction});
      } else {
        // Phase 4: tiled reduction via RBLOCK Inner parameter.
        Value param = insertFuncArg(func, builder, loc, 64, name);
        SmallVector<TileParam> group;
        group.push_back({name, param, OpFoldResult(ext), i,
                          TileLevel::Inner, AxisRole::Reduction});
        plan.tileable.push_back(std::move(group));
      }
    }
```

- [ ] **Step 5: Update `GroupEmitter.cpp` to detect and handle Phase 4**

Add `hasReductionInner` helper at the top of GroupEmitter.cpp:

```cpp
static bool hasReductionInner(const TilePlan &plan) {
  for (auto &group : plan.tileable)
    for (const auto &tp : group)
      if (tp.level == TileLevel::Inner && tp.role == AxisRole::Reduction)
        return true;
  return false;
}
```

Add `emitGroupWithReductionSplit` below:

```cpp
static SmallVector<Value>
emitGroupWithReductionSplit(OpBuilder &builder, Location loc,
                             const CollapsedGroupInfo &info,
                             const TilePlan &plan,
                             const LoopNestResult &loopNest) {
  // Find the RBLOCK TileParam.
  const TileParam *rblockParam = nullptr;
  for (auto &group : plan.tileable)
    for (const auto &tp : group)
      if (tp.role == AxisRole::Reduction) { rblockParam = &tp; break; }
  assert(rblockParam && "reduction split requires RBLOCK param");

  // Collect boundary outs and map to iter args.
  SmallVector<Value> allOuts;
  DenseSet<Value> seenOuts;
  for (LinalgOp op : info.topoMembers)
    for (Value out : op.getDpsInits())
      if (seenOuts.insert(out).second)
        allOuts.push_back(out);
  DenseMap<Value, Value> outToIterArg;
  for (auto [out, iterArg] : llvm::zip(allOuts, loopNest.iterArgs))
    outToIterArg[out] = iterArg;

  builder.setInsertionPointToStart(loopNest.innermostBody);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);

  // Emit accumulator: tensor.empty + linalg.fill(0).
  // Shape matches the output (the iter arg slice shape).
  Value iterArgOut = loopNest.iterArgs.front();
  auto outType = cast<RankedTensorType>(iterArgOut.getType());
  // Compute sizes for the output slice (from SliceComputer via outer ivs).
  auto maps = info.topoMembers.back().getIndexingMapsArray();
  int numIn = info.topoMembers.back().getNumDpsInputs();
  AffineMap outMap = maps[numIn];
  auto sp = computeSlice(outMap, loopNest.loopIVs, plan, iterArgOut,
                          builder, loc);
  Value accEmpty = builder.create<tensor::EmptyOp>(loc, sp.sizes,
                                                    outType.getElementType());
  Value zero = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(outType.getElementType()));
  Value accZero = builder.create<linalg::FillOp>(loc, zero, accEmpty)
                      .getResult(0);

  // RBLOCK scf.for.
  Value reductionExtent =
      getAxisExtentValue(builder, loc, info, rblockParam->axisIdx);
  auto rForOp = builder.create<scf::ForOp>(
      loc, c0, reductionExtent, rblockParam->ssa,
      SmallVector<Value>{accZero});
  builder.setInsertionPointToStart(rForOp.getBody());

  // Build loopIVs with the RBLOCK IV.
  DenseMap<int, Value> innerIVs(loopNest.loopIVs);
  innerIVs[rblockParam->axisIdx] = rForOp.getInductionVar();

  // Emit the reduction op inside RBLOCK.
  DenseMap<Value, Value> tiledValues;
  Value accIterArg = rForOp.getRegionIterArgs().front();
  // Update outToIterArg to point to the rblock iter arg.
  DenseMap<Value, Value> rblockOutMap;
  rblockOutMap[allOuts.front()] = accIterArg;

  for (LinalgOp op : info.topoMembers) {
    auto opMaps     = op.getIndexingMapsArray();
    int numInputs   = op.getNumDpsInputs();
    SmallVector<Value> newOperands;

    for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
      bool isInit = ((int)idx >= numInputs);
      if (isInit) {
        newOperands.push_back(rblockOutMap.count(operand)
                                  ? rblockOutMap[operand]
                                  : accIterArg);
        continue;
      }
      if (tiledValues.count(operand)) {
        newOperands.push_back(tiledValues[operand]);
        continue;
      }
      AffineMap map = opMaps[idx];
      auto sp2 = computeSlice(map, innerIVs, plan, operand, builder, loc);
      auto slicedType = tensor::ExtractSliceOp::inferResultType(
          cast<RankedTensorType>(operand.getType()), sp2.offsets,
          sp2.sizes, sp2.strides);
      Value sliced = builder.create<tensor::ExtractSliceOp>(
          loc, slicedType, operand, sp2.offsets, sp2.sizes, sp2.strides);
      newOperands.push_back(sliced);
    }
    Operation *cloned = builder.clone(*op.getOperation());
    for (auto [idx, val] : llvm::enumerate(newOperands))
      cloned->setOperand((unsigned)idx, val);
    for (auto [orig, newR] : llvm::zip(op->getResults(), cloned->getResults()))
      tiledValues[orig] = newR;
  }

  // Yield accumulator result from RBLOCK body.
  Value reductionResult = tiledValues[info.topoMembers.back()->getResult(0)];
  builder.create<scf::YieldOp>(loc, SmallVector<Value>{reductionResult});

  // After RBLOCK: insert the accumulated result into the output tensor.
  builder.setInsertionPointAfter(rForOp);
  Value accFinal = rForOp.getResult(0);
  Value inserted = builder.create<tensor::InsertSliceOp>(
      loc, accFinal, iterArgOut, sp.offsets, sp.sizes, sp.strides);

  // Yield from innermost parallel body.
  builder.create<scf::YieldOp>(loc, SmallVector<Value>{inserted});

  // Propagate yields up parallel loop chain (excluding RBLOCK).
  SmallVector<scf::ForOp> parallelOps;
  for (scf::ForOp fo : loopNest.allForOps)
    parallelOps.push_back(fo);
  for (int i = (int)parallelOps.size() - 2; i >= 0; --i) {
    builder.setInsertionPointToEnd(parallelOps[i].getBody());
    builder.create<scf::YieldOp>(loc, parallelOps[i + 1].getResults());
  }

  return SmallVector<Value>(loopNest.allForOps.front().getResults());
}
```

At the top of `emitGroup`, add the dispatch:

```cpp
SmallVector<Value> emitGroup(OpBuilder &builder, Location loc,
                              const CollapsedGroupInfo &info,
                              const TilePlan &plan,
                              const LoopNestResult &loopNest) {
  if (hasReductionInner(plan))
    return emitGroupWithReductionSplit(builder, loc, info, plan, loopNest);
  // ... (existing baseline path below)
```

Also add required includes in GroupEmitter.cpp:
```cpp
#include "mlir/Dialect/Linalg/IR/Linalg.h"
```

- [ ] **Step 6: Build**

```bash
cd build && ninja afir-opt 2>&1 | grep -E "error:" | head -20
```

- [ ] **Step 7: Run both reduction tests**

```bash
afir-opt test/Conversion/Collapse/tile-fuse-vector-reduce.mlir \
  --vector-plan-tile-fuse 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-reduce.mlir

afir-opt test/Conversion/Collapse/tile-fuse-vector-reduce-split.mlir \
  "--vector-plan-tile-fuse=enable-reduction-split=true" 2>&1 | FileCheck \
  test/Conversion/Collapse/tile-fuse-vector-reduce-split.mlir
```
Expected: both pass.

- [ ] **Step 8: Run full regression**

```bash
bash test/Conversion/Collapse/run_tests.sh --all
```
Expected: all 5 new tests + existing 7 pass.

- [ ] **Step 9: Commit**

```bash
git add lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp \
        lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp \
        test/Conversion/Collapse/tile-fuse-vector-reduce.mlir \
        test/Conversion/Collapse/tile-fuse-vector-reduce-split.mlir
git commit -m "feat(vector-plan): Phase 4 reduction split + reduction Full axis support"
```
