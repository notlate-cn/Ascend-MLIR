# TilePlanGen + LoopNest + GroupEmitter Design

**Date**: 2026-04-27  
**Scope**: VectorPlan TileFuse Phase 2 (TilePlanGen) + Phase 3 (LoopNestBuilder / SliceComputer / GroupEmitter) + Phase 4 (Reduction Split) + BCast escape  
**Prerequisite**: Phase 1 Collapse complete (`CollapsedGroupInfo` produced)

---

## Goal

Implement the three phases of `vector-plan-tile-fuse` that follow Collapse:

1. **TilePlanGen** — translate `CollapsedGroupInfo` into a `TilePlan` (axis → tile parameter assignment)
2. **LoopNestBuilder + SliceComputer + GroupEmitter** — lower `TilePlan` to a tiled `scf.for` loop nest with `tensor.extract_slice` / `tensor.insert_slice`
3. **Reduction Split (Phase 4)** — when `enable-reduction-split=true`, tile the reduction axis with an `RBLOCK` loop and emit accumulator init + epilogue pattern

All axis indices are **post-collapse** (`CollapsedGroupInfo::collapsedAxes` numbering). Static and dynamic shapes both supported.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` | Phase 2: CollapsedGroupInfo → TilePlan |
| Create | `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.h` | Public entry point `genVectorTilePlan()` |
| Create | `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp` | Phase 3a: TilePlan → scf.for loop nest |
| Create | `lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.h` | `LoopNestResult`, `buildLoopNest()` |
| Create | `lib/Conversion/VectorPlan/TileFuse/SliceComputer.cpp` | Phase 3b: loopIVs + indexingMap → SliceParams |
| Create | `lib/Conversion/VectorPlan/TileFuse/SliceComputer.h` | `SliceParams`, `computeSlice()` |
| Create | `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp` | Phase 3c: topo-order emit + Load hoist + Phase 4 |
| Create | `lib/Conversion/VectorPlan/TileFuse/GroupEmitter.h` | `emitGroup()` |
| Modify | `lib/Conversion/VectorPlan/TileFuse/TileFusePass.cpp` | Wire Phase 2/3/4; add pass options |
| Modify | `lib/Conversion/VectorPlan/CMakeLists.txt` | Add new source files |
| Modify | `include/Conversion/Passes.td` | Add `enableReductionSplit`, `maxFullLoopIters` options |
| Create | `test/Conversion/Collapse/tile-fuse-vector-pointwise.mlir` | Pointwise, no BCast, no reduction |
| Create | `test/Conversion/Collapse/tile-fuse-vector-bcast.mlir` | BCast axis Full loop + Load hoist |
| Create | `test/Conversion/Collapse/tile-fuse-vector-bcast-escape.mlir` | BCast axis escape to tileable |
| Create | `test/Conversion/Collapse/tile-fuse-vector-reduce.mlir` | Reduction Full (no split) |
| Create | `test/Conversion/Collapse/tile-fuse-vector-reduce-split.mlir` | Reduction split (Phase 4) |

---

## Component Designs

### 1. TilePlanGen

**Entry point**:
```cpp
// TilePlanGen.h
mlir::vector_plan::TilePlan
genVectorTilePlan(const mlir::vector_plan::CollapsedGroupInfo &info,
                  mlir::OpBuilder &builder,
                  mlir::Location loc,
                  bool enableReductionSplit,
                  int64_t maxFullLoopIters);
```

**Axis assignment rules**:

| Axis type | Condition | TileLevel | Name pattern |
|-----------|-----------|-----------|--------------|
| 1st Parallel axis | — | Outer + Inner | `XBLOCK` + `XBLOCK_SUB` |
| 2nd+ Parallel axis | — | Inner | `XBLOCK_SUB_1`, `_2`, … |
| BCast axis | staticSize ≤ maxFullLoopIters | Full, step=1 | `BCAST_0`, `_1`, … |
| BCast axis | staticSize > maxFullLoopIters (escape) | Inner | `BCAST_TILE_0`, … |
| Reduction axis | enableReductionSplit=false | Full, step=extent | `RBLOCK_0`, `_1`, … |
| Reduction axis | enableReductionSplit=true | Inner | `RBLOCK_0`, `_1`, … |

**BCast axis detection**: axes whose post-collapse index is in `info.broadcastAxes`. Their extents come from `info.broadcastAxisExtents` (for dynamic shapes) or from the static size in `info.collapsedAxes`.

**`insertFuncArg`**: appends an `index`-typed argument to the `func.func` with a `vector_plan.default_tile_size` integer attr (the default value). Returns the new `BlockArgument` as a `Value`.

**`blockDimExprs`**: `[ceildiv(extent_of_first_parallel_axis, XBLOCK)]` — 1D for VectorGroup.

**`getAxisExtentValue`**: for static size returns `arith.constant`; for dynamic returns `tensor.dim` on the first operand that has that axis in its indexing map.

---

### 2. LoopNestBuilder

**Data structures**:
```cpp
// LoopNestBuilder.h
struct LoopNestResult {
  llvm::DenseMap<int, mlir::Value> loopIVs;        // axis idx → composedIV
  mlir::Block                     *innermostBody;
  llvm::SmallVector<mlir::Value>   iterArgs;        // current innermost iter args
  llvm::SmallVector<mlir::scf::ForOp> bcastForOps; // BCast Full loops (for hoist)
};

LoopNestResult buildLoopNest(mlir::OpBuilder &builder,
                              mlir::Location loc,
                              const mlir::vector_plan::TilePlan &plan,
                              mlir::ValueRange initTensors);
```

**Loop construction order** (D1 / BAII L2):
1. Outer tileable → emit `scf.for`, set `ascendc.parallel` attr, record `outerTileSize[axisIdx]`
2. BCast Full → emit `scf.for` (upper=extent, step=1), append to `bcastForOps`
3. Inner tileable → emit `scf.for` (upper=`outerTileSize[axisIdx]` if Outer exists, else extent, step=tile size)
4. Reduction Inner (Phase 4 only) → handled inside GroupEmitter, not here

**composedIV**: when Inner loop is emitted for an axis that already has an Outer IV:
```
composedIV = outerIV + innerIV
loopIVs[axisIdx] = composedIV
```

**Reduction Full axes** are not emitted as loops; they are absent from `loopIVs` — SliceComputer handles missing axes as full-dim.

**iter args**: each `scf.for` is constructed with the current `initTensors` as iter args. At each nesting level the iter args are the `regionIterArgs` of the innermost `scf.for`.

---

### 3. SliceComputer

```cpp
// SliceComputer.h
struct SliceParams {
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides; // all 1
};

SliceParams computeSlice(mlir::AffineMap indexingMap,
                          const llvm::DenseMap<int, mlir::Value> &loopIVs,
                          const mlir::vector_plan::TilePlan &plan,
                          mlir::Value tensor,
                          mlir::OpBuilder &builder,
                          mlir::Location loc);
```

**Per result-dim rules**:

| Map result | loopIVs has axis? | offset | size |
|------------|-------------------|--------|------|
| `AffineDimExpr(g)` | yes | `loopIVs[g]` | `getTileSizeForAxis(plan, g)` |
| `AffineDimExpr(g)` | no (Reduction Full / unused) | 0 | `tensor.dim(tensor, dimIdx)` |
| other expr | — | 0 | `tensor.dim(tensor, dimIdx)` |

**`getTileSizeForAxis`**: searches `plan.tileable` and `plan.full` for a `TileParam` with matching `axisIdx`. For BCast Full axes (step=1) returns constant 1. For Outer axes returns `XBLOCK` ssa. For Inner returns `XBLOCK_SUB` ssa.

---

### 4. GroupEmitter

```cpp
// GroupEmitter.h
void emitGroup(mlir::OpBuilder &builder,
               mlir::Location loc,
               const mlir::vector_plan::CollapsedGroupInfo &info,
               const mlir::vector_plan::TilePlan &plan,
               const LoopNestResult &loopNest);
```

**Operand classification** (per op per operand):
- **boundary input**: defined outside the group (func arg or pre-group value) → emit `tensor.extract_slice`
- **interior value**: result of a previous topo member → look up `tiledValues` map
- **scalar / constant**: pass through unchanged

**Load hoist algorithm** (`computeHoistPoint`):
For each `extract_slice`, iterate `bcastForOps` from innermost to outermost. If the slice's `offsets` do not reference a BCast loop's IV, the insertion point is moved to before that `scf.for`. The first BCast loop whose IV appears in `offsets` stops the traversal.

**Emit sequence**:
1. For each op in `info.topoMembers` (topo order):
   a. Compute `newOperands` per above classification
   b. `builder.clone(*op)`, set operands, record `tiledValues[origResult] = tiledResult`
2. After all ops: for each boundary output, emit `tensor.insert_slice` + collect yield values
3. Emit `scf.yield(yieldVals)` in innermost body

---

### 5. Phase 4: Reduction Split

**Trigger**: `plan.full` contains a `TileParam` with `TileLevel::Inner` and `AxisRole::Reduction`.

**`emitGroup` branching**: detect trigger at entry; if true, call `emitGroupWithReductionSplit` instead of the baseline path.

**`emitGroupWithReductionSplit` sequence**:
1. Split `info.topoMembers` at first reduction op → `preReduction` + `epilogue`
2. In parallel loop body: emit `tensor.empty` + `linalg.fill(zero)` → `acc`
3. Build `scf.for RBLOCK(0 to reductionExtent step rblockTileSize)` with `acc` as iter arg
4. Inside RBLOCK body: emit `preReduction` ops (using `rForOp.getInductionVar()` as the reduction axis IV), `scf.yield(reductionResult)`
5. After RBLOCK loop: emit `epilogue` ops using `rForOp.getResult(0)` as the reduction result
6. Emit `tensor.insert_slice` + `scf.yield` for boundary outputs

**`splitAtReductionBoundary`**: walks `topoMembers` in topo order; the split point is after the last op whose `iteratorTypes` contains `reduction`. All ops up to and including that op are `preReduction`; the rest are `epilogue`.

---

## Pass Options (additions to `Passes.td`)

```tablegen
let options = [
  Option<"enableReductionSplit", "enable-reduction-split", "bool",
         /*default=*/"false",
         "Tile reduction axes with RBLOCK loop instead of full-dim iteration">,
  Option<"maxFullLoopIters", "max-full-loop-iters", "int64_t",
         /*default=*/"2048",
         "BCast axes larger than this are escaped to tileable Inner">,
];
```

---

## TileFusePass wiring

```cpp
void runOnOperation() override {
  // Phase 0: BroadcastAbsorb (existing)
  // Phase 1: Collapse (existing) → collapsedInfo
  // Phase 2: TilePlanGen
  auto plan = genVectorTilePlan(collapsedInfo, builder, func.getLoc(),
                                enableReductionSplit, maxFullLoopIters);
  // Phase 3: LoopNest
  SmallVector<Value> initTensors = getInitTensors(func, collapsedInfo);
  auto loopNest = buildLoopNest(builder, func.getLoc(), plan, initTensors);
  // Phase 3c / Phase 4: GroupEmitter
  builder.setInsertionPoint(loopNest.innermostBody, loopNest.innermostBody->begin());
  emitGroup(builder, func.getLoc(), collapsedInfo, plan, loopNest);
  // Remove original linalg ops (now replaced by tiled loop nest)
  eraseOriginalOps(collapsedInfo);
}
```

---

## Test Scenarios

| Test file | Pass options | Key checks |
|-----------|-------------|------------|
| `tile-fuse-vector-pointwise.mlir` | default | `scf.for XBLOCK {ascendc.parallel}`, inner `scf.for XBLOCK_SUB`, `extract_slice`, `insert_slice` |
| `tile-fuse-vector-bcast.mlir` | default | BCast loop between Outer and Inner, `extract_slice %mask` hoisted before BCast loop |
| `tile-fuse-vector-bcast-escape.mlir` | `max-full-loop-iters=8` | BCast axis with size>8 becomes tileable Inner (no BCast Full loop) |
| `tile-fuse-vector-reduce.mlir` | default | No RBLOCK loop; reduction axis walked full-dim inside `extract_slice` |
| `tile-fuse-vector-reduce-split.mlir` | `enable-reduction-split=true` | `linalg.fill`, `scf.for RBLOCK`, `linalg.reduce` inside, epilogue after |

---

## Key Invariants

- All axis indices are post-collapse throughout (no mixing with canonical indices).
- `loopIVs` never contains a Reduction Full axis — SliceComputer's "no IV → full dim" path handles it.
- BCast Full axes always have step=1 in the loop; `getTileSizeForAxis` returns constant 1 for them so `extract_slice` sizes for BCast dimensions are 1.
- `emitGroup` never modifies `TilePlan` or `LoopNestResult` — it is a pure emitter.
- Phase 4 only fires when `plan` contains a Reduction Inner axis; the baseline `emitGroup` path is unaffected.
