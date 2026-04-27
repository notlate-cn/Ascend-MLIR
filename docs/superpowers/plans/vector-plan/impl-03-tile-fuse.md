# impl-03: Pass 2 — Tile Fuse

**设计依据**: `docs/superpowers/specs/2026-04-14-vector-plan-unified-design.md` §4 §5 §6  
**前置**: impl-00（数据结构）；Pass 2 独立于 impl-01/02，可并行开发，但需要 kernel_group{N}.mlir 作为输入

---

## 定位

`vector-plan-tile-fuse` 是 func-level pass，对每个 `kernel_group{N}.mlir` 独立运行。
**不复用** `linalg::tileUsingForOp` / `tileAndFuseProducerOfSlice`，自行建 loop nest。
统一发射策略让 horizontal fusion（无 SSA 边的 sibling）天然工作。

内部三阶段：
```
Collapse  →  TilePlanGen  →  LoopNest + Emit
```

**关键约束**：
- Pass 2 不感知 `group_id`；func 内所有 linalg op 属于同一 group
- CubeGroup 跳过 Collapse（恒等映射）

---

## Phase 1: Collapse — 简化迭代空间

文件：`lib/Conversion/VectorPlan/TileFuse/Collapse.cpp`

### 候选 collapse 组 G

从 canonical axes（上确界推导）中找**连续同类型**轴（连续 Parallel 或连续 Reduction）：

```cpp
SmallVector<SmallVector<int>> findCandidateGroups(ArrayRef<AxisInfo> canonicalAxes) {
  SmallVector<SmallVector<int>> groups;
  SmallVector<int> current;
  for (auto [i, ax] : enumerate(canonicalAxes)) {
    if (!current.empty() && ax.role != canonicalAxes[current.back()].role) {
      if (current.size() > 1) groups.push_back(current);
      current.clear();
    }
    current.push_back(i);
  }
  if (current.size() > 1) groups.push_back(current);
  return groups;
}
```

### Per-input 分类（A / B1 / B2 / C）

```cpp
enum class InputClass { A, B1, B2, C };

InputClass classifyInput(AffineMap map, ArrayRef<int> G) {
  // 找 map results 中出现的 G 内轴
  DenseSet<int> resultG;
  for (auto expr : map.getResults())
    if (auto dim = dyn_cast<AffineDimExpr>(expr))
      if (llvm::is_contained(G, (int)dim.getPosition()))
        resultG.insert(dim.getPosition());

  if (resultG.empty()) return InputClass::A;    // G 完全缺失
  if (resultG.size() < G.size()) return InputClass::B1; // 部分缺失

  // 所有 G 轴均在 result 中，检查是否连续同序
  SmallVector<int> positions;
  for (int g : G)
    for (auto [i, expr] : enumerate(map.getResults()))
      if (auto dim = dyn_cast<AffineDimExpr>(expr))
        if ((int)dim.getPosition() == g) { positions.push_back(i); break; }

  for (int i = 1; i < (int)positions.size(); ++i)
    if (positions[i] != positions[i - 1] + 1) return InputClass::B2;

  return InputClass::C;
}
```

### Pre-Check

```cpp
struct CollapsePreCheck {
  SmallVector<int>                G;
  DenseMap<Value, InputClass>     inputClasses; // boundary input → 最严格的 class
  bool hasB1 = false, hasB2 = false;
};

CollapsePreCheck preCheck(const GroupInfo &info, ArrayRef<int> G) {
  CollapsePreCheck result;
  result.G = SmallVector<int>(G);

  for (auto op : info.topoMembers) {
    auto lop = cast<linalg::LinalgOp>(op);
    for (auto [operand, map] :
         llvm::zip(lop.getInputs(), lop.getIndexingMapsArray())) {
      if (!isBoundaryInput(operand, info)) continue;
      auto cls = classifyInput(map, G);
      auto &stored = result.inputClasses[operand];
      // B2 优先级最高（严格排序：A < C < B1 < B2）
      if (classRank(cls) > classRank(stored)) stored = cls;
      if (cls == InputClass::B1) result.hasB1 = true;
      if (cls == InputClass::B2) result.hasB2 = true;
    }
  }
  return result;
}
```

### B1 处理 — 补全缺失维

```cpp
void fixupB1(OpBuilder &builder, GroupInfo &info,
              const CollapsePreCheck &check) {
  for (auto &[boundary, cls] : check.inputClasses) {
    if (cls != InputClass::B1) continue;

    for (auto op : info.topoMembers) {
      auto lop = cast<linalg::LinalgOp>(op);
      for (auto [idx, operand] : enumerate(lop.getInputs())) {
        if (operand != boundary) continue;
        if (classifyInput(lop.getIndexingMapsArray()[idx], check.G)
            != InputClass::B1) continue;

        // 找缺失的 G 内轴
        auto missing = getMissingDims(lop.getIndexingMapsArray()[idx], check.G);
        builder.setInsertionPoint(op);
        // linalg.broadcast 补展开
        auto broadcastOp = builder.create<linalg::BroadcastOp>(
            op->getLoc(), operand,
            /*init=*/createExpandedInit(builder, operand, missing),
            /*dimensions=*/missing);
        broadcastOp->setAttr("vector_plan.no_collapse", builder.getUnitAttr());
        lop->setOperand(idx, broadcastOp.getResult());
        // 插入 topoMembers（在 op 之前）
        insertBefore(info.topoMembers, op, broadcastOp);
      }
    }
  }
}
```

### B2 处理 — 生成两个 Variant

**Variant 1（Preserve）— 插 linalg.transpose，no_collapse barrier**:

```cpp
GroupInfo fixupB2Variant1(OpBuilder &builder, GroupInfo info,
                            const CollapsePreCheck &check) {
  for (auto &[boundary, cls] : check.inputClasses) {
    if (cls != InputClass::B2) continue;
    for (auto op : info.topoMembers) {
      auto lop = cast<linalg::LinalgOp>(op);
      for (auto [idx, operand] : enumerate(lop.getInputs())) {
        if (operand != boundary) continue;
        if (classifyInput(lop.getIndexingMapsArray()[idx], check.G)
            != InputClass::B2) continue;
        auto perm = computePermToCanonical(lop.getIndexingMapsArray()[idx], check.G);
        builder.setInsertionPoint(op);
        auto transposeOp = builder.create<linalg::TransposeOp>(
            op->getLoc(), operand,
            createTransposedInit(builder, operand, perm), perm);
        transposeOp->setAttr("vector_plan.no_collapse", builder.getUnitAttr());
        lop->setOperand(idx, transposeOp.getResult());
        insertBefore(info.topoMembers, op, transposeOp);
      }
    }
  }
  // 标记：含 no_collapse op，loop nest 走原始 G-axes 路径
  info.noCollapse = true;
  return info;
}
```

**Variant 2（Eliminate）— 改 consumer map + load_with_transpose 标记**:

```cpp
GroupInfo fixupB2Variant2(OpBuilder &builder, GroupInfo info,
                            const CollapsePreCheck &check) {
  DenseSet<Value> marked;
  for (auto &[boundary, cls] : check.inputClasses) {
    if (cls != InputClass::B2) continue;
    for (auto op : info.topoMembers) {
      auto lop = cast<linalg::LinalgOp>(op);
      for (auto [idx, operand] : enumerate(lop.getInputs())) {
        if (operand != boundary) continue;
        // 把 consumer map 改为 canonical（B2 → C）
        setIndexingMapToCanonical(lop, idx, check.G);
      }
    }
    // boundary input 打标（每个只打一次）
    if (!marked.count(boundary)) {
      // boundary 通常是 BlockArgument；在 func arg 上设 attr
      if (auto blockArg = dyn_cast<BlockArgument>(boundary))
        blockArg.getOwner()->getParentOp()->setAttr(
            "vector_plan.load_with_transpose", builder.getUnitAttr());
      marked.insert(boundary);
    }
  }
  return info;
}
```

### 测试用例（Phase 1）

```mlir
// test/Conversion/VectorPlan/tile-fuse-collapse-b1.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s

// LayerNorm [B, S, H] → 候选 G = {d_B, d_S}
// scale/bias [H] → Case A；其余 → Case C
func.func @kernel_group0(%input: tensor<4x8x16xf16>,
                          %scale: tensor<16xf16>,
                          %bias: tensor<16xf16>) -> tensor<4x8x16xf16> {
  // ... linalg ops
}
// CHECK: tensor.collapse_shape {{.*}} into tensor<32x16xf16>
// CHECK: scf.for

// test/Conversion/VectorPlan/tile-fuse-collapse-b2.mlir
// B2 case：transpose+pointwise，input [S, B] 乱序
// CHECK: linalg.broadcast  ← B1 补缺（若有）
// CHECK: linalg.transpose  ← B2 Variant 1
//  或
// CHECK-NOT: linalg.transpose  ← B2 Variant 2（改 map + load_with_transpose）
```

---

## Phase 2: TilePlanGen — VectorGroup 基础路径

文件：`lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp`

**适用条件**：`enableReductionSplit=false`，无 B2。

```cpp
TilePlan genVectorTilePlan(const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc) {
  TilePlan plan;
  plan.group = &info;

  SmallVector<int> tileableIdx, reductionIdx;
  for (auto [i, ax] : enumerate(info.collapsedAxes)) {
    if (ax.role == AxisRole::Parallel)  tileableIdx.push_back(i);
    if (ax.role == AxisRole::Reduction) reductionIdx.push_back(i);
  }
  assert(!tileableIdx.empty() && "VectorGroup must have at least one tileable axis");

  // 第一个 tileable 轴：两级 (XBLOCK Outer + XBLOCK_SUB Inner)
  {
    int axIdx = tileableIdx[0];
    Value extent = getAxisExtentValue(info.collapsedAxes[axIdx], builder, loc);
    Value xblock    = insertFuncArg(builder, "XBLOCK", 256);
    Value xblockSub = insertFuncArg(builder, "XBLOCK_SUB", 64);
    plan.tileable.push_back({
      TileParam{"XBLOCK",     xblock,    IntegerAttr::get(i64, 256),
                axIdx, TileLevel::Outer, AxisRole::Parallel},
      TileParam{"XBLOCK_SUB", xblockSub, IntegerAttr::get(i64, 64),
                axIdx, TileLevel::Inner, AxisRole::Parallel},
    });
    // blockDimExprs[0] = ceildiv(extent, XBLOCK)
    plan.blockDimExprs.push_back(
        builder.create<arith::CeilDivSIOp>(loc, extent, xblock).getResult());
  }

  // 其余 tileable 轴：一级 Inner，defaultValue = full dim
  for (int i = 1; i < (int)tileableIdx.size(); ++i) {
    int axIdx = tileableIdx[i];
    Value extent = getAxisExtentValue(info.collapsedAxes[axIdx], builder, loc);
    plan.tileable.push_back({
      TileParam{formatv("XBLOCK_SUB_{0}", i), extent,
                /*default=*/getShapeDimAttr(builder, axIdx),
                axIdx, TileLevel::Inner, AxisRole::Parallel},
    });
  }

  // Reduction 轴：Full（不切）
  for (auto [j, axIdx] : enumerate(reductionIdx)) {
    Value extent = getAxisExtentValue(info.collapsedAxes[axIdx], builder, loc);
    plan.full.push_back(
      TileParam{formatv("RBLOCK_{0}", j), extent,
                getShapeDimAttr(builder, axIdx),
                (int)axIdx, TileLevel::Full, AxisRole::Reduction}
    );
  }

  return plan;
}
```

**enableReductionSplit=true 修改**（Phase 4 实现，此处列出接口）：

```cpp
// full[j] 改为：
TileParam{formatv("RBLOCK_{0}", j),
          insertFuncArg(builder, formatv("RBLOCK_{0}", j), defaultVal),
          defaultVal,
          (int)axIdx, TileLevel::Inner,  // ← Full → Inner
          AxisRole::Reduction}
```

---

## Phase 3: LoopNestBuilder + SliceComputer + GroupEmitter

### LoopNestBuilder

文件：`lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp`

```cpp
struct LoopNestResult {
  DenseMap<int, Value> loopIVs;      // collapsed axis idx → scf.for IV
  Block               *innermostBody;
  SmallVector<Value>   iterArgs;     // group boundary outputs 的 iter arg
};

LoopNestResult buildLoopNest(OpBuilder &builder, Location loc,
                               const TilePlan &plan,
                               ArrayRef<Value> initTensors) {
  LoopNestResult result;

  // loop order: Outer tileable → Inner tileable（Reduction 轴 Full 时不建 loop）
  SmallVector<TileParam> loopParams;
  for (auto &tileVec : plan.tileable)
    for (auto &tp : tileVec)
      if (tp.level == TileLevel::Outer || tp.level == TileLevel::Inner)
        loopParams.push_back(tp);

  SmallVector<Value> currentIterArgs(initTensors.begin(), initTensors.end());

  for (auto &tp : loopParams) {
    Value extent = getAxisExtentValue(*plan.group, tp.axisIdx, builder, loc);
    Value step   = tp.ssa; // tile size 即为 step

    auto forOp = builder.create<scf::ForOp>(
        loc,
        builder.create<arith::ConstantIndexOp>(loc, 0),
        castToIndex(extent, builder, loc),
        castToIndex(step, builder, loc),
        currentIterArgs);

    if (tp.level == TileLevel::Outer)
      forOp->setAttr("ascendc.parallel", builder.getUnitAttr());

    result.loopIVs[tp.axisIdx] = forOp.getInductionVar();
    builder.setInsertionPointToStart(forOp.getBody());
    currentIterArgs = SmallVector<Value>(forOp.getRegionIterArgs());
  }

  result.innermostBody = builder.getInsertionBlock();
  result.iterArgs      = currentIterArgs;
  return result;
}
```

### SliceComputer

文件：`lib/Conversion/VectorPlan/TileFuse/SliceComputer.cpp`

```cpp
struct SliceParams {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides; // 全为 1
};

SliceParams computeSlice(OpBuilder &builder, Location loc,
                          AffineMap indexingMap,
                          const DenseMap<int, Value> &loopIVs,
                          const TilePlan &plan,
                          Value tensor) {
  SliceParams params;
  auto zero = builder.getIndexAttr(0);
  auto tensorType = cast<RankedTensorType>(tensor.getType());

  for (auto [dimIdx, expr] : enumerate(indexingMap.getResults())) {
    params.strides.push_back(builder.getIndexAttr(1));

    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      int g = dimExpr.getPosition();
      if (loopIVs.count(g)) {
        // 该 G-axis 在当前 loop 层有 IV → tile
        params.offsets.push_back(loopIVs.at(g));
        params.sizes.push_back(
            castToIndex(getTileSizeForAxis(plan, g), builder, loc));
      } else {
        // Reduction 轴或未参与当前 loop 的轴 → full dim
        params.offsets.push_back(zero);
        params.sizes.push_back(
            builder.create<tensor::DimOp>(loc, tensor, dimIdx).getResult());
      }
    } else {
      // 常量 0 或其他 affine expr（如 broadcast 维）→ full dim
      params.offsets.push_back(zero);
      params.sizes.push_back(
          builder.create<tensor::DimOp>(loc, tensor, dimIdx).getResult());
    }
  }
  return params;
}
```

### GroupEmitter

文件：`lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp`

```cpp
using TiledValueMap = DenseMap<Value, Value>;

void emitGroup(OpBuilder &builder, Location loc,
               const CollapsedGroupInfo &info,
               const TilePlan &plan,
               const LoopNestResult &loopNest) {
  TiledValueMap tiledValues;

  for (linalg::LinalgOp op : info.topoMembers) {
    SmallVector<Value> newOperands;

    for (auto [idx, operand] : enumerate(op->getOperands())) {
      if (tiledValues.count(operand)) {
        // interior value：前序 op 的 tiled 结果，直接使用
        newOperands.push_back(tiledValues[operand]);
      } else if (isBoundaryInput(operand, info)) {
        // boundary input：emit extract_slice
        auto map = op.getIndexingMapsArray()[idx];
        auto slice = computeSlice(builder, loc, map,
                                   loopNest.loopIVs, plan, operand);
        auto extractSlice = builder.create<tensor::ExtractSliceOp>(
            loc, operand, slice.offsets, slice.sizes, slice.strides);
        newOperands.push_back(extractSlice);
      } else {
        newOperands.push_back(operand); // 常量、scalar
      }
    }

    // emit tiled op
    auto *tiledOp = builder.clone(*op);
    for (auto [i, v] : enumerate(newOperands))
      tiledOp->setOperand(i, v);
    for (auto [orig, tiled] : llvm::zip(op->getResults(), tiledOp->getResults()))
      tiledValues[orig] = tiled;
  }

  // emit insert_slice + scf.yield for boundary outputs
  SmallVector<Value> yieldVals;
  for (auto [outVal, iterArg] :
       llvm::zip(info.boundaryOut, loopNest.iterArgs)) {
    // output indexing map 是 identity → offsets 同 XBLOCK/XBLOCK_SUB loop IVs
    auto insertSlice = builder.create<tensor::InsertSliceOp>(
        loc, tiledValues[outVal], iterArg,
        getOutputOffsets(loopNest, plan),
        getOutputSizes(plan),
        /*strides=*/ones(builder, loc, info.collapsedAxes.size()));
    yieldVals.push_back(insertSlice);
  }
  builder.create<scf::YieldOp>(loc, yieldVals);
}
```

### 测试用例（Phase 3）

```mlir
// test/Conversion/VectorPlan/tile-fuse-vector-pointwise.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s

func.func @kernel_group0(%x: tensor<?xf16>, %y: tensor<?xf16>) -> tensor<?xf16> {
  %out = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%x, %y) outs(%init: tensor<?xf16>) { ... }
  return %out
}
// CHECK: scf.for %[[XB:.*]] = %c0 to {{.*}} step {{.*}} {ascendc.parallel
// CHECK:   scf.for %[[XS:.*]] = %c0 to {{.*}} step {{.*}} {
// CHECK:     %[[S1:.*]] = tensor.extract_slice %x[%[[XB]]]
// CHECK:     %[[S2:.*]] = tensor.extract_slice %y[%[[XB]]]
// CHECK:     linalg.generic ins(%[[S1]], %[[S2]])
// CHECK:     tensor.insert_slice

// test/Conversion/VectorPlan/tile-fuse-vector-reduce-pointwise.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
func.func @kernel_group0(%in: tensor<?x?xf16>) -> tensor<?xf16> {
  %r = linalg.reduce { arith.addf } ins(%in) outs(...) dimensions = [1]
  %out = linalg.generic { ... } ins(%r) outs(...)  // epilogue
  return %out
}
// CHECK: scf.for %[[XB:.*]]
// CHECK:   scf.for %[[XS:.*]]
// CHECK:     %[[slice:.*]] = tensor.extract_slice %in[%[[XB]], 0]
// CHECK:     %[[r:.*]] = linalg.reduce
// CHECK:     linalg.generic ins(%[[r]])  ← epilogue 使用 reduce 结果（无 extract_slice）
// CHECK:     tensor.insert_slice

// test/Conversion/VectorPlan/tile-fuse-vector-sibling.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
// 3 个 sibling：共享 boundary input x
func.func @kernel_group0(%x: tensor<?xf16>, %a: tensor<?xf16>,
                          %b: tensor<?xf16>, %c: tensor<?xf16>)
    -> (tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) {
  %s1 = linalg.generic { ... } ins(%x, %a) outs(...)
  %s2 = linalg.generic { ... } ins(%x, %b) outs(...)
  %s3 = linalg.generic { ... } ins(%x, %c) outs(...)
  return %s1, %s2, %s3
}
// 同一 loop body 内，x 只 extract_slice 一次
// CHECK: scf.for %[[XB:.*]]
// CHECK:   scf.for %[[XS:.*]]
// CHECK:     tensor.extract_slice %x[%[[XB]]]
// CHECK-COUNT-3: linalg.generic
// CHECK-COUNT-3: tensor.insert_slice
```

---

## Phase 4: Reduce Split（accumulator pattern）

`enableReductionSplit=true` 时，对含 reduction 的 VectorGroup 启用。

```cpp
void emitGroupWithReductionSplit(OpBuilder &builder, Location loc,
                                  const CollapsedGroupInfo &info,
                                  const TilePlan &plan,
                                  const LoopNestResult &parallelLoopNest) {
  // 把 topoMembers 分成两段：pre-reduction（含 reduce op）+ epilogue
  auto [preReduction, epilogue] = splitAtReductionBoundary(info.topoMembers);

  // 1. 在 parallel loop body 内 emit linalg.fill（init accumulator）
  auto accType = getReductionAccType(info);
  auto empty   = builder.create<tensor::EmptyOp>(loc, accType, ValueRange{});
  Value acc    = builder.create<linalg::FillOp>(
      loc, TypeRange{accType}, ValueRange{zero, empty}).getResult(0);

  // 2. 建 RBLOCK scf.for
  const TileParam &rblock = plan.full[0]; // RBLOCK_0
  Value rExtent = getAxisExtentValue(info, rblock.axisIdx, builder, loc);
  auto rForOp = builder.create<scf::ForOp>(
      loc,
      builder.create<arith::ConstantIndexOp>(loc, 0),
      castToIndex(rExtent, builder, loc),
      castToIndex(rblock.ssa, builder, loc),
      ValueRange{acc});

  // 3. pre-reduction ops → emit 进 RBLOCK body
  builder.setInsertionPointToStart(rForOp.getBody());
  TiledValueMap tiledValues;
  tiledValues[getAccInit(preReduction)] = rForOp.getRegionIterArgs()[0];
  LoopNestResult rblockLoopNest = parallelLoopNest;
  rblockLoopNest.loopIVs[rblock.axisIdx] = rForOp.getInductionVar();
  emitOps(builder, loc, preReduction, plan, rblockLoopNest, tiledValues);
  builder.create<scf::YieldOp>(loc, tiledValues[getReductionResult(preReduction)]);

  // 4. epilogue ops → emit 在 RBLOCK loop 之后
  builder.setInsertionPointAfter(rForOp);
  tiledValues[getReductionResult(preReduction)] = rForOp.getResult(0);
  emitOps(builder, loc, epilogue, plan, parallelLoopNest, tiledValues);
}
```

### 测试用例（Phase 4）

```mlir
// test/Conversion/VectorPlan/tile-fuse-vector-softmax.mlir
// RUN: mlir-opt --vector-plan-tile-fuse="enable-reduction-split=true" %s \
// RUN:   | FileCheck %s

func.func @kernel_group0(%in: tensor<?x?xf16>) -> tensor<?x?xf16> {
  // max reduce → sub → exp → sum reduce → div
}
// CHECK: scf.for %[[XB:.*]] {ascendc.parallel
// CHECK:   scf.for %[[XS:.*]]
// CHECK:     linalg.fill  ← acc init（在 RBLOCK loop 之前）
// CHECK:     scf.for %[[RB:.*]] = %c0 to {{.*}} step {{.*}} {  ← RBLOCK loop
// CHECK:       linalg.reduce
// CHECK:     }
// CHECK:     linalg.generic  ← epilogue（在 RBLOCK loop 之后）
```

---

## Phase 5: B2 Variant 1（no_collapse transpose 路径）

B2 input 存在时，TileFusePass 生成两份独立的 IR（Variant 1 / Variant 2），
由 Autotuner 在运行时选优。两份 IR 分别对应 `kernel_groupN_v1.mlir` /
`kernel_groupN_v2.mlir`，各自走完整的 TilePlan + LoopNest 流程。

```cpp
// TileFusePass.cpp 顶层，Collapse 之后：
if (check.hasB2) {
  // 生成两个 CollapsedGroupInfo，分别走完整的 TilePlan + LoopNest
  GroupInfo v1Info = fixupB2Variant1(builder, cloneGroupInfo(info), check);
  GroupInfo v2Info = fixupB2Variant2(builder, cloneGroupInfo(info), check);
  // 对 v1 执行后续步骤 → 产出 Variant 1 IR
  // 对 v2 执行后续步骤 → 产出 Variant 2 IR
} else {
  // 无 B2：单一路径
}

// Variant 1 路径识别（loop nest 建立时使用）：
bool useOriginalAxes = llvm::any_of(info.topoMembers, [](linalg::LinalgOp op) {
  return op->hasAttr("vector_plan.no_collapse");
});
// useOriginalAxes=true → LoopNestBuilder 用原始 G-axes（各轴独立 IV）
// SliceComputer 对 B2 input 用原始 B2 map，对 output 用 canonical map
```

### 测试用例（Phase 5）

```mlir
// test/Conversion/VectorPlan/tile-fuse-vector-b2-v1.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s

// input T [S, B]，consumer map: (d0,d1)→(d1,d0)（B2）
// Variant 1：插 linalg.transpose [S,B]→[B,S]
func.func @kernel_group0(%T: tensor<?x?xf16>, %side: tensor<?x?xf16>)
    -> tensor<?x?xf16> { ... }
// CHECK: linalg.transpose %T perm = [1, 0]  ← no_collapse barrier
// CHECK: scf.for %[[IV0:.*]]  ← 原始 d0 loop
// CHECK:   scf.for %[[IV1:.*]]  ← 原始 d1 loop
// CHECK:     tensor.extract_slice %T[%[[IV1]], %[[IV0]]]  ← B2 map offset
```

---

## Phase 6: B2 Variant 2（load_with_transpose 路径）

```cpp
// GroupEmitter 中，boundary input 处理追加判断：
if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
  auto parentFunc = cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
  if (parentFunc->hasAttr("vector_plan.load_with_transpose")) {
    // 生成 ConfusionTranspose 占位 op（GM→UB 搬运时完成重排）
    auto confOp = builder.create<ConfusionTransposeOp>(loc, operand, ...);
    newOperands.push_back(confOp);
    continue;
  }
}
```

### 测试用例（Phase 6）

```mlir
// test/Conversion/VectorPlan/tile-fuse-vector-b2-v2.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s

// Variant 2：consumer map 改为 canonical，boundary input 打 load_with_transpose
func.func @kernel_group0(%T: tensor<?x?xf16>, %side: tensor<?x?xf16>)
    -> tensor<?x?xf16> { ... }
// CHECK-NOT: linalg.transpose  ← 无 transpose node
// CHECK: tensor.collapse_shape  ← 整体 collapse，无 barrier
// CHECK: scf.for %[[XB:.*]]  ← collapsed loop
// CHECK:   confusion_transpose %T  ← Load 侧重排
```

---

## Phase 7: CubeGroup — 层次化 Loop Nest

### TilePlanGen（CubeGroup 分支）

```cpp
TilePlan genCubeTilePlan(const CubeGroupInfo &info,
                          OpBuilder &builder, Location loc) {
  TilePlan plan;
  auto matmul = info.matmul;
  Value lhs = matmul.getInputs()[0]; // [M, K]
  Value rhs = matmul.getInputs()[1]; // [K, N]

  int mRank = cast<RankedTensorType>(lhs.getType()).getRank();
  Value Mext = builder.create<tensor::DimOp>(loc, lhs, mRank - 2);
  Value Next = builder.create<tensor::DimOp>(loc, rhs,
      cast<RankedTensorType>(rhs.getType()).getRank() - 1);
  Value Kext = builder.create<tensor::DimOp>(loc, lhs, mRank - 1);

  // M 轴：两级
  plan.tileable.push_back({
    TileParam{"BM",   insertFuncArg(builder, "BM",   128), Mext,
              mAxisIdx, TileLevel::Outer, AxisRole::Parallel},
    TileParam{"Tb_M", insertFuncArg(builder, "Tb_M",  64), Mext,
              mAxisIdx, TileLevel::Inner, AxisRole::Parallel},
  });
  // N 轴：两级
  plan.tileable.push_back({
    TileParam{"BN",   insertFuncArg(builder, "BN",   128), Next,
              nAxisIdx, TileLevel::Outer, AxisRole::Parallel},
    TileParam{"Tb_N", insertFuncArg(builder, "Tb_N", 128), Next,
              nAxisIdx, TileLevel::Inner, AxisRole::Parallel},
  });
  // K 轴：Inner（必须切，非 Full）
  plan.full.push_back(
    TileParam{"t_K", insertFuncArg(builder, "t_K", 64), Kext,
              kAxisIdx, TileLevel::Inner, AxisRole::Reduction}
  );

  // 2D blockDimExprs
  plan.blockDimExprs.push_back(
      builder.create<arith::CeilDivSIOp>(loc, Mext,
          plan.tileable[0][0].ssa).getResult()); // grid_y
  plan.blockDimExprs.push_back(
      builder.create<arith::CeilDivSIOp>(loc, Next,
          plan.tileable[1][0].ssa).getResult()); // grid_x

  return plan;
}
```

### CubeGroup 发射

```cpp
void emitCubeGroup(OpBuilder &builder, Location loc,
                    const CubeGroupInfo &info, const TilePlan &plan) {
  // 4 层 loop：BM(Outer) → BN(Outer) → Tb_M(Inner) → Tb_N(Inner)
  auto outerLoopNest = buildCubeOuterLoops(builder, loc, plan);

  // innermost body（Tb_N 内）：accumulator pattern
  auto accType = getMatmulOutputType(info.matmul);
  auto empty   = builder.create<tensor::EmptyOp>(loc, accType, ValueRange{});
  Value acc    = builder.create<linalg::FillOp>(
      loc, TypeRange{accType}, ValueRange{zero, empty}).getResult(0);

  // t_K loop
  const TileParam &tK = plan.full[0];
  Value Kext = getAxisExtentValue(info, tK.axisIdx, builder, loc);
  auto tkForOp = builder.create<scf::ForOp>(
      loc,
      builder.create<arith::ConstantIndexOp>(loc, 0),
      castToIndex(Kext, builder, loc),
      castToIndex(tK.ssa, builder, loc),
      ValueRange{acc});
  tkForOp->setAttr("ascendc.prologue",
                    builder.getStringAttr("lhs:A1->A2,rhs:B1->B2"));

  // t_K body：只发射 matmul
  builder.setInsertionPointToStart(tkForOp.getBody());
  TiledValueMap tiledValues;
  tiledValues[info.matmul.getInputs()[0]] =
      getASlice(builder, loc, info, outerLoopNest, tkForOp.getInductionVar(), plan);
  tiledValues[info.matmul.getInputs()[1]] =
      getBSlice(builder, loc, info, outerLoopNest, tkForOp.getInductionVar(), plan);
  tiledValues[getAccInit(info.matmul)] = tkForOp.getRegionIterArgs()[0];
  emitOp(builder, loc, info.matmul, tiledValues, plan, outerLoopNest);
  builder.create<scf::YieldOp>(loc, tiledValues[info.matmul->getResult(0)]);

  // epilogue ops after t_K loop
  builder.setInsertionPointAfter(tkForOp);
  tiledValues[info.matmul->getResult(0)] = tkForOp.getResult(0);
  tiledValues[info.matmul->getResult(0)]
      ->getDefiningOp()
      ->setAttr("ascendc.unit", builder.getStringAttr("AiCore.Cube"));

  for (linalg::LinalgOp epi : getEpilogueOps(info))
    emitOp(builder, loc, epi, tiledValues, plan, outerLoopNest);
}
```

### 测试用例（Phase 7）

```mlir
// test/Conversion/VectorPlan/tile-fuse-cube-matmul-bias-relu.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s

func.func @kernel_group1(%A: tensor<?x?xf16>, %B: tensor<?x?xf16>,
                          %bias: tensor<?xf16>) -> tensor<?x?xf16> {
  %mm  = linalg.matmul ins(%A, %B) outs(...)
  %add = linalg.generic { ... } ins(%mm, %bias) outs(...)  // bias add
  %relu = linalg.generic { ... } ins(%add) outs(...)       // relu
  return %relu
}
// CHECK:      scf.for %[[BM:.*]] {{.*}} {ascendc.parallel
// CHECK-NEXT:   scf.for %[[BN:.*]] {{.*}} {ascendc.parallel
// CHECK:          scf.for %[[TBM:.*]]
// CHECK:            scf.for %[[TBN:.*]]
// CHECK:              linalg.fill  ← acc init
// CHECK:              scf.for %[[TK:.*]] {{.*}} ascendc.prologue
// CHECK:                linalg.matmul
// CHECK:              linalg.generic  ← bias add，AiCore.Vector
// CHECK:              linalg.generic  ← relu，AiCore.Vector

// test/Conversion/VectorPlan/tile-fuse-cube-layernorm-matmul.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
// （LayerNorm VectorGroup + matmul+epilogue CubeGroup 各自独立文件，分别验证）
```
