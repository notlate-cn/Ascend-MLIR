# VectorPlan Phase 1 Collapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 1 (BroadcastAbsorb + Collapse) of `vector-plan-tile-fuse` so that `kernel_groupN` funcs get their canonical axes collapsed in the IR, producing `CollapsedGroupInfo` for subsequent Phase 2/3.

**Architecture:** Three components: (1) `VectorPlanBroadcastAbsorb` pass (new, separate) folds `linalg.broadcast` into downstream `linalg.generic` indexing maps by dropping broadcast dimensions; (2) `Collapse.cpp` (new, internal to TileFuse) computes BCast axes, prunes candidate collapse groups, classifies inputs as A/B2/C, and for surviving Case-C groups transforms the IR — inserting `tensor.collapse_shape` for C-type operands and `tensor.expand_shape` for results, rewriting indexing maps to post-collapse numbering; (3) `TileFusePass` runs BroadcastAbsorb patterns then calls `collapseGroup()` — Phase 2/3 remain TODO stubs. Target: static shapes, single `linalg.generic` per kernel func (the common case after GroupOutline).

**Tech Stack:** MLIR (linalg, tensor, arith, affine, func, scf dialects), C++17, lit + FileCheck for tests.

**Key invariant (BAII):** An axis is in BCast(G) iff some boundary input has *partial* G-axis coverage — at least one G-axis present in the map AND at least one G-axis absent. Inputs with ALL G-axes absent (Case A) do **not** contribute to BCast. BCast axes are pruned from collapse candidates; residual sub-groups of size ≥ 2 are eligible to collapse.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `include/Conversion/VectorPlan/GroupInfo.h` | Add `broadcastAxes` + `broadcastAxisExtents` to `CollapsedGroupInfo` |
| Modify | `include/Conversion/Passes.td` | Register `VectorPlanBroadcastAbsorb` pass |
| Modify | `include/Conversion/VectorPlan/VectorPlanPasses.h` | Declare `createVectorPlanBroadcastAbsorbPass()` |
| Modify | `lib/Conversion/VectorPlan/CMakeLists.txt` | Add `TileFuse/BroadcastAbsorb.cpp`, `TileFuse/Collapse.cpp` |
| Create | `lib/Conversion/VectorPlan/TileFuse/BroadcastAbsorb.cpp` | BroadcastAbsorb pass + pattern |
| Create | `lib/Conversion/VectorPlan/TileFuse/Collapse.h` | `collapseGroup()` declaration |
| Create | `lib/Conversion/VectorPlan/TileFuse/Collapse.cpp` | Collapse analysis + IR transformation |
| Modify | `lib/Conversion/VectorPlan/TileFuse/TileFusePass.cpp` | Call Phase 1 |
| Create | `test/Conversion/VectorPlan/broadcast-absorb-basic.mlir` | Absorption happy path |
| Create | `test/Conversion/VectorPlan/broadcast-absorb-fallback.mlir` | Non-generic consumer → preserve |
| Create | `test/Conversion/VectorPlan/collapse-prune-middle-axis.mlir` | BCast in middle → no collapse |
| Create | `test/Conversion/VectorPlan/collapse-prune-tail-axis.mlir` | BCast at tail → collapse [d0,d1] |
| Create | `test/Conversion/VectorPlan/tile-fuse-collapse-c.mlir` | Case A scale/bias + Case C input → collapse |

---

## Task 1: Data model + infrastructure

**Files:**
- Modify: `include/Conversion/VectorPlan/GroupInfo.h`
- Modify: `include/Conversion/Passes.td`
- Modify: `include/Conversion/VectorPlan/VectorPlanPasses.h`
- Modify: `lib/Conversion/VectorPlan/CMakeLists.txt`

- [ ] **Step 1: Add `broadcastAxes` / `broadcastAxisExtents` to `CollapsedGroupInfo`**

In `include/Conversion/VectorPlan/GroupInfo.h`, replace the `CollapsedGroupInfo` struct:

```cpp
struct CollapsedGroupInfo : GroupInfo {
  llvm::SmallVector<AxisInfo> collapsedAxes;
  llvm::SmallVector<int>      axisMap;   // original axis idx -> post-collapse idx; -1 if absorbed
  bool                        hasB2      = false;
  bool                        noCollapse = false;
  // Post-collapse broadcast axis indices (ascending). Filled by Phase 1 Collapse.
  // These are axes where some boundary input partially broadcasts (B1 scenario).
  llvm::SmallVector<int>            broadcastAxes;
  // Extent SSA values for each broadcastAxes entry (needed for dynamic shapes in Phase 2).
  llvm::DenseMap<int, mlir::Value>  broadcastAxisExtents;
};
```

- [ ] **Step 2: Register `VectorPlanBroadcastAbsorb` in `Passes.td`**

In `include/Conversion/Passes.td`, append before the final `#endif`:

```tablegen
//===----------------------------------------------------------------------===//
// BroadcastAbsorb
//===----------------------------------------------------------------------===//

def VectorPlanBroadcastAbsorb : Pass<"vector-plan-broadcast-absorb", "mlir::func::FuncOp"> {
  let summary = "Absorb linalg.broadcast into downstream linalg.generic indexing maps";
  let description = [{
    For each linalg.broadcast whose result is consumed by exactly one
    linalg.generic, replaces the generic's operand with the broadcast's
    original input and drops the broadcast dimensions from the operand's
    indexing map. Broadcasts with other consumers are left unchanged.
  }];
  let constructor = "mlir::afir::createVectorPlanBroadcastAbsorbPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}
```

- [ ] **Step 3: Declare factory in `VectorPlanPasses.h`**

```cpp
std::unique_ptr<Pass> createVectorPlanBroadcastAbsorbPass();
```

- [ ] **Step 4: Add source files to `lib/Conversion/VectorPlan/CMakeLists.txt`**

```cmake
add_mlir_library(MLIRVectorPlan
  GroupAnalysis/AxisLattice.cpp
  GroupAnalysis/CanFuse.cpp
  GroupAnalysis/GroupAnalysisPass.cpp
  GroupOutline/GroupOutlinePass.cpp
  TileFuse/BroadcastAbsorb.cpp   # new
  TileFuse/Collapse.cpp           # new
  TileFuse/TileFusePass.cpp
  TileInfo/TilePlanToTileInfo.cpp
  Pipeline.cpp
  ...
```

- [ ] **Step 5: Build (expect link error on missing .cpp — that is correct)**

```bash
cd build && ninja MLIRVectorPlan 2>&1 | grep -E "error:|BroadcastAbsorb|Collapse" | head -10
```
Expected: tablegen succeeds, link error for missing object files (files not yet created).

- [ ] **Step 6: Commit**

```bash
git add include/Conversion/VectorPlan/GroupInfo.h \
        include/Conversion/Passes.td \
        include/Conversion/VectorPlan/VectorPlanPasses.h \
        lib/Conversion/VectorPlan/CMakeLists.txt
git commit -m "feat(vector-plan): data model + pass infra for Phase 1 Collapse"
```

---

## Task 2: BroadcastAbsorb pass

**Files:**
- Create: `lib/Conversion/VectorPlan/TileFuse/BroadcastAbsorb.cpp`
- Create: `test/Conversion/VectorPlan/broadcast-absorb-basic.mlir`
- Create: `test/Conversion/VectorPlan/broadcast-absorb-fallback.mlir`

- [ ] **Step 1: Write the failing basic test**

Create `test/Conversion/VectorPlan/broadcast-absorb-basic.mlir`:

```mlir
// RUN: mlir-opt --vector-plan-broadcast-absorb %s | FileCheck %s

// CHECK-LABEL: func.func @absorb_into_generic
func.func @absorb_into_generic(
    %x: tensor<4x8xf16>,
    %y: tensor<4x3x8xf16>) -> tensor<4x3x8xf16> {
  %init = tensor.empty() : tensor<4x3x8xf16>
  // broadcast %x[4,8] → [4,3,8] over dimension 1 (the S=3 axis)
  %b = linalg.broadcast ins(%x : tensor<4x8xf16>)
                        outs(%init : tensor<4x3x8xf16>)
                        dimensions = [1]
  %r = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%b, %y : tensor<4x3x8xf16>, tensor<4x3x8xf16>)
    outs(%init : tensor<4x3x8xf16>) {
    ^bb0(%a: f16, %bv: f16, %o: f16):
      %add = arith.addf %a, %bv : f16
      linalg.yield %add : f16
  } -> tensor<4x3x8xf16>
  return %r : tensor<4x3x8xf16>
}
// After absorption: no linalg.broadcast; first operand uses %x directly
// with d1 dropped from its map (d0,d1,d2)->(d0,d1,d2) → (d0,d1,d2)->(d0,d2)
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic
// CHECK-SAME: affine_map<(d0, d1, d2) -> (d0, d2)>
```

- [ ] **Step 2: Write the failing fallback test**

Create `test/Conversion/VectorPlan/broadcast-absorb-fallback.mlir`:

```mlir
// RUN: mlir-opt --vector-plan-broadcast-absorb %s | FileCheck %s

// broadcast result is returned directly — no linalg.generic consumer → preserve
// CHECK-LABEL: func.func @fallback_no_generic_consumer
func.func @fallback_no_generic_consumer(%x: tensor<4x8xf16>) -> tensor<4x3x8xf16> {
  %init = tensor.empty() : tensor<4x3x8xf16>
  %b = linalg.broadcast ins(%x : tensor<4x8xf16>)
                        outs(%init : tensor<4x3x8xf16>)
                        dimensions = [1]
  return %b : tensor<4x3x8xf16>
}
// CHECK: linalg.broadcast
```

- [ ] **Step 3: Run tests (expect FAIL — pass not yet compiled)**

```bash
cd build && ninja mlir-opt 2>&1 | tail -3
```
Expected: build error on missing BroadcastAbsorb.cpp.

- [ ] **Step 4: Create `BroadcastAbsorb.cpp`**

Create `lib/Conversion/VectorPlan/TileFuse/BroadcastAbsorb.cpp`:

```cpp
#include "Conversion/VectorPlan/VectorPlanPasses.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DECL_VECTORPLANBROADCASTABSORB
#define GEN_PASS_DEF_VECTORPLANBROADCASTABSORB
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir::afir {

namespace {

// Remove results at the given sorted positions from an AffineMap.
// E.g., dropResultsAt((d0,d1,d2)->(d0,d1,d2), {1}) → (d0,d1,d2)->(d0,d2)
static AffineMap dropResultsAt(AffineMap map,
                                ArrayRef<int64_t> sortedPositions) {
  DenseSet<int64_t> posSet(sortedPositions.begin(), sortedPositions.end());
  SmallVector<AffineExpr> kept;
  for (auto [i, expr] : llvm::enumerate(map.getResults()))
    if (!posSet.count((int64_t)i))
      kept.push_back(expr);
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                        kept, map.getContext());
}

// Pattern: absorb linalg.broadcast into a single linalg.generic consumer.
// Fires when the broadcast result has exactly one use and that use is GenericOp.
struct AbsorbBroadcastIntoGeneric : OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp bcast,
                                PatternRewriter &rewriter) const override {
    if (!bcast.getResult().hasOneUse())
      return failure();
    auto *user = *bcast.getResult().getUsers().begin();
    auto generic = dyn_cast<GenericOp>(user);
    if (!generic)
      return failure();

    // Find which input operand uses the broadcast result.
    int32_t opIdx = -1;
    for (auto [i, inp] : llvm::enumerate(generic.getInputs())) {
      if (inp == bcast.getResult()) {
        opIdx = (int32_t)i;
        break;
      }
    }
    if (opIdx < 0)
      return failure();

    // Broadcast dimensions: which result dims of the broadcast output are new.
    SmallVector<int64_t> dims =
        llvm::to_vector(bcast.getDimensions().getAsValueRange<IntegerAttr,
                                                               int64_t>());
    llvm::sort(dims);

    // Drop those positions from the consumer's map for this operand.
    SmallVector<AffineMap> maps(generic.getIndexingMapsArray());
    maps[opIdx] = dropResultsAt(maps[opIdx], dims);

    // Update the operand and maps in-place.
    rewriter.modifyOpInPlace(generic, [&]() {
      generic.setOperand(opIdx, bcast.getInput());
      generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    });

    if (bcast.getResult().use_empty())
      rewriter.eraseOp(bcast);

    return success();
  }
};

struct VectorPlanBroadcastAbsorbPass
    : public ::impl::VectorPlanBroadcastAbsorbBase<
          VectorPlanBroadcastAbsorbPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AbsorbBroadcastIntoGeneric>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createVectorPlanBroadcastAbsorbPass() {
  return std::make_unique<VectorPlanBroadcastAbsorbPass>();
}

} // namespace mlir::afir
```

- [ ] **Step 5: Build and run both tests**

```bash
cd build && ninja mlir-opt && \
  mlir-opt --vector-plan-broadcast-absorb \
    ../test/Conversion/VectorPlan/broadcast-absorb-basic.mlir | \
  FileCheck ../test/Conversion/VectorPlan/broadcast-absorb-basic.mlir && \
  mlir-opt --vector-plan-broadcast-absorb \
    ../test/Conversion/VectorPlan/broadcast-absorb-fallback.mlir | \
  FileCheck ../test/Conversion/VectorPlan/broadcast-absorb-fallback.mlir
```
Expected: both FileCheck runs exit 0.

- [ ] **Step 6: Commit**

```bash
git add lib/Conversion/VectorPlan/TileFuse/BroadcastAbsorb.cpp \
        test/Conversion/VectorPlan/broadcast-absorb-basic.mlir \
        test/Conversion/VectorPlan/broadcast-absorb-fallback.mlir
git commit -m "feat(vector-plan): BroadcastAbsorb pass (absorb linalg.broadcast into indexing_maps)"
```

---

## Task 3: Collapse analysis (BCast, pruning, classification)

**Files:**
- Create: `lib/Conversion/VectorPlan/TileFuse/Collapse.h`
- Create: `lib/Conversion/VectorPlan/TileFuse/Collapse.cpp` (analysis section only — IR transform in Task 4)
- Create: `test/Conversion/VectorPlan/collapse-prune-middle-axis.mlir`
- Create: `test/Conversion/VectorPlan/collapse-prune-tail-axis.mlir`

The two tests drive pruning logic. They run `--vector-plan-tile-fuse`; after Task 3 the pass runs Phase 1 analysis, and in Task 4 the IR transformation fires. After Task 3, the prune-middle test will pass (`CHECK-NOT: tensor.collapse_shape`) and the tail test will still fail (no collapse emitted yet). Full green after Task 4.

- [ ] **Step 1: Write `collapse-prune-middle-axis` test**

Create `test/Conversion/VectorPlan/collapse-prune-middle-axis.mlir`:

```mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
//
// 3 parallel axes [d0,d1,d2]. Input %x: tensor<4x8xf16> has map
// (d0,d1,d2)->(d0,d2) — d1 is missing but d0 and d2 are present → partial
// coverage → d1 is a BCast axis sitting in the middle of candidate {d0,d1,d2}.
// Pruning splits into [d0](size 1, discarded) + [d2](size 1, discarded).
// Result: no collapse.
//
// CHECK-LABEL: func.func @kernel_group0
func.func @kernel_group0(
    %in: tensor<4x3x8xf16>,
    %x:  tensor<4x8xf16>) -> tensor<4x3x8xf16> {
  %init = tensor.empty() : tensor<4x3x8xf16>
  %r = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%in, %x : tensor<4x3x8xf16>, tensor<4x8xf16>)
    outs(%init : tensor<4x3x8xf16>) {
    ^bb0(%a: f16, %b: f16, %o: f16):
      %add = arith.addf %a, %b : f16
      linalg.yield %add : f16
  } -> tensor<4x3x8xf16>
  return %r : tensor<4x3x8xf16>
}
// d1 BCast → prune → [d0](size 1)+[d2](size 1) both discarded → no collapse
// CHECK-NOT: tensor.collapse_shape
```

- [ ] **Step 2: Write `collapse-prune-tail-axis` test**

Create `test/Conversion/VectorPlan/collapse-prune-tail-axis.mlir`:

```mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
//
// 3 parallel axes [d0,d1,d2]. Input %x: tensor<4x8xf16> has map
// (d0,d1,d2)->(d0,d1) — d2 is missing but d0 and d1 are present → d2 BCast
// at the tail. Pruning removes d2 leaving [d0,d1] (size 2, keep).
// Result: tensor.collapse_shape [[0,1],[2]] → tensor<32x16xf16>.
//
// CHECK-LABEL: func.func @kernel_group0
func.func @kernel_group0(
    %in: tensor<4x8x16xf16>,
    %x:  tensor<4x8xf16>) -> tensor<4x8x16xf16> {
  %init = tensor.empty() : tensor<4x8x16xf16>
  %r = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%in, %x : tensor<4x8x16xf16>, tensor<4x8xf16>)
    outs(%init : tensor<4x8x16xf16>) {
    ^bb0(%a: f16, %b: f16, %o: f16):
      %add = arith.addf %a, %b : f16
      linalg.yield %add : f16
  } -> tensor<4x8x16xf16>
  return %r : tensor<4x8x16xf16>
}
// d2 BCast → prune to [d0,d1] (size 2) → collapse d0*d1
// CHECK: tensor.collapse_shape
// CHECK-SAME: into tensor<32x16xf16>
```

- [ ] **Step 3: Create `Collapse.h`**

Create `lib/Conversion/VectorPlan/TileFuse/Collapse.h`:

```cpp
#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::afir {

/// Analyse all linalg ops in `func`, compute collapse groups, and transform
/// the IR (insert tensor.collapse_shape / expand_shape, rewrite indexing maps).
///
/// Returns a CollapsedGroupInfo describing the post-collapse iteration space.
/// When no collapse is profitable (BCast pruning leaves no group of size ≥ 2,
/// or B2 detected, or multi-op func), returns an identity mapping
/// (collapsedAxes == canonicalAxes, axisMap[i] == i).
///
/// v1 limitation: only transforms a func with exactly one linalg.generic.
/// Static and dynamic shapes both supported (dynamic → collapsed dim is ?).
mlir::vector_plan::CollapsedGroupInfo
collapseGroup(mlir::OpBuilder &builder, mlir::func::FuncOp func);

} // namespace mlir::afir
```

- [ ] **Step 4: Create `Collapse.cpp` (analysis helpers — IR transform shell in Task 4)**

Create `lib/Conversion/VectorPlan/TileFuse/Collapse.cpp` with the analysis helpers and a stub for the IR transform:

```cpp
#include "Collapse.h"
#include "AxisLattice.h"
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::vector_plan;
using namespace mlir::linalg;

namespace mlir::afir {

//===----------------------------------------------------------------------===//
// Analysis helpers
//===----------------------------------------------------------------------===//

// Returns consecutive same-role sub-sequences of size ≥ 2 as candidate groups.
// E.g., axes = [par, par, red, par] → [[0,1]] (the single-element [3] is dropped)
static SmallVector<SmallVector<int>>
findCandidateGroups(ArrayRef<AxisInfo> axes) {
  SmallVector<SmallVector<int>> groups;
  SmallVector<int> current;
  for (auto [i, ax] : llvm::enumerate(axes)) {
    if (!current.empty() && ax.role != axes[current.back()].role) {
      if (current.size() > 1)
        groups.push_back(current);
      current.clear();
    }
    current.push_back((int)i);
  }
  if (current.size() > 1)
    groups.push_back(current);
  return groups;
}

// Compute BCast axes for a candidate group G.
// An axis a_i ∈ G is broadcast iff some boundary input has PARTIAL G coverage:
//   ≥ 1 G-axis present in map results  AND  a_i absent.
// Case A (all G-axes absent) does NOT contribute to BCast.
static DenseSet<int>
computeBCast(ArrayRef<int> G,
             ArrayRef<linalg::LinalgOp> members,
             ArrayRef<Value> boundaryIn) {
  DenseSet<Value> bInSet(boundaryIn.begin(), boundaryIn.end());
  DenseSet<int> gSet(G.begin(), G.end());
  DenseSet<int> bcast;

  for (linalg::LinalgOp op : members) {
    for (auto [operand, map] :
         llvm::zip(op.getInputs(), op.getIndexingMapsArray())) {
      if (!bInSet.count(operand))
        continue;
      // Find G-axes present in this map's results.
      DenseSet<int> presentInG;
      for (AffineExpr expr : map.getResults()) {
        if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
          int pos = (int)dim.getPosition();
          if (gSet.count(pos))
            presentInG.insert(pos);
        }
      }
      if (presentInG.empty() ||                   // Case A: no contribution
          (int)presentInG.size() == (int)G.size()) // Case C/B2: all present
        continue;
      // B1 (partial): missing G-axes become BCast.
      for (int g : G)
        if (!presentInG.count(g))
          bcast.insert(g);
    }
  }
  return bcast;
}

// Split candidate group G by BCast; keep sub-groups of size ≥ 2.
static SmallVector<SmallVector<int>>
pruneBCastAxes(ArrayRef<int> G, const DenseSet<int> &bcast) {
  SmallVector<SmallVector<int>> out;
  SmallVector<int> current;
  for (int d : G) {
    if (bcast.count(d)) {
      if (current.size() >= 2)
        out.push_back(current);
      current.clear();
    } else {
      current.push_back(d);
    }
  }
  if (current.size() >= 2)
    out.push_back(current);
  return out;
}

enum class InputClass { A, B2, C };

// Classify a boundary input's map with respect to a pruned (BCast-free) group.
static InputClass classifyInput(AffineMap map, ArrayRef<int> G) {
  DenseSet<int> gSet(G.begin(), G.end());
  DenseSet<int> presentInG;
  for (AffineExpr expr : map.getResults()) {
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      int pos = (int)dim.getPosition();
      if (gSet.count(pos))
        presentInG.insert(pos);
    }
  }
  if (presentInG.empty())
    return InputClass::A;

  assert((int)presentInG.size() == (int)G.size() &&
         "B1 must have been removed by BCast pruning before classifyInput");

  // Check contiguous ascending order of result positions for G-axes.
  SmallVector<int> resultPositions;
  for (int g : G) {
    for (auto [i, expr] : llvm::enumerate(map.getResults())) {
      if (auto dim = dyn_cast<AffineDimExpr>(expr))
        if ((int)dim.getPosition() == g) {
          resultPositions.push_back((int)i);
          break;
        }
    }
  }
  for (int k = 1; k < (int)resultPositions.size(); ++k)
    if (resultPositions[k] != resultPositions[k - 1] + 1)
      return InputClass::B2;
  return InputClass::C;
}

// Return true if any boundary input classified as B2 for the given group.
static bool hasAnyB2(ArrayRef<int> G,
                     ArrayRef<linalg::LinalgOp> members,
                     ArrayRef<Value> boundaryIn) {
  DenseSet<Value> bInSet(boundaryIn.begin(), boundaryIn.end());
  for (linalg::LinalgOp op : members) {
    for (auto [operand, map] :
         llvm::zip(op.getInputs(), op.getIndexingMapsArray())) {
      if (bInSet.count(operand) &&
          classifyInput(map, G) == InputClass::B2)
        return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// IR transformation helpers
//===----------------------------------------------------------------------===//

// Build axisMap and collapsedAxes given the canonical axes and the chosen
// collapse group (a pruned sub-sequence of consecutive canonical-axis indices).
// axisMap[origIdx] = post-collapse idx, or -1 if the axis is absorbed into the
// leader of its collapse sub-group.
static void buildAxisMap(ArrayRef<AxisInfo> canonicalAxes,
                         ArrayRef<int> collapseGroup,
                         SmallVector<int> &axisMapOut,
                         SmallVector<AxisInfo> &collapsedAxesOut) {
  int numOrig = (int)canonicalAxes.size();
  DenseSet<int> cgSet(collapseGroup.begin(), collapseGroup.end());
  axisMapOut.assign(numOrig, 0);

  int postIdx = 0;
  for (int origIdx = 0; origIdx < numOrig; ++origIdx) {
    if (!cgSet.count(origIdx)) {
      // Not in collapse group: keep as individual post-collapse axis.
      axisMapOut[origIdx] = postIdx++;
      collapsedAxesOut.push_back(canonicalAxes[origIdx]);
    } else if (origIdx == collapseGroup.front()) {
      // Leader of collapse group: compute product size.
      int64_t prod = 1;
      for (int g : collapseGroup) {
        int64_t s = canonicalAxes[g].staticSize;
        if (s == ShapedType::kDynamic || prod == ShapedType::kDynamic)
          prod = ShapedType::kDynamic;
        else
          prod *= s;
      }
      axisMapOut[origIdx] = postIdx++;
      collapsedAxesOut.push_back({"", prod, canonicalAxes[origIdx].role});
    } else {
      // Non-leader member of collapse group: absorbed into leader.
      axisMapOut[origIdx] = -1;
    }
  }
}

// Rewrite an indexing map for the post-collapse axis numbering.
// Results for absorbed axes (-1) are dropped; remaining dim positions are
// renumbered according to axisMap.
static AffineMap rewriteMap(AffineMap map,
                             ArrayRef<int> axisMap,
                             int numPostDims) {
  SmallVector<AffineExpr> newResults;
  DenseSet<int> seenPost;
  for (AffineExpr expr : map.getResults()) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim) { newResults.push_back(expr); continue; }
    int origIdx = (int)dim.getPosition();
    int postIdx = (origIdx < (int)axisMap.size()) ? axisMap[origIdx] : origIdx;
    if (postIdx < 0) continue; // absorbed: skip
    if (!seenPost.count(postIdx)) {
      newResults.push_back(getAffineDimExpr(postIdx, map.getContext()));
      seenPost.insert(postIdx);
    }
  }
  return AffineMap::get(numPostDims, 0, newResults, map.getContext());
}

// Build tensor.collapse_shape reassociation for an operand whose map results
// correspond 1-1 to its tensor dimensions. Groups consecutive result positions
// that map to the same post-collapse axis index.
// E.g., map = (d0,d1,d2)->(d0,d1,d2), axisMap = [0,-1,1]:
//   dim 0 → orig d0 → post 0,  dim 1 → orig d1 → post -1 (absorbed → same as 0)
//   dim 2 → orig d2 → post 1
//   → reassoc = [[0,1],[2]]
static SmallVector<ReassociationIndices>
buildReassociation(AffineMap map, ArrayRef<int> axisMap) {
  SmallVector<ReassociationIndices> reassoc;
  int prevPost = -99;
  for (auto [i, expr] : llvm::enumerate(map.getResults())) {
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim) continue;
    int origIdx = (int)dim.getPosition();
    int postIdx = (origIdx < (int)axisMap.size()) ? axisMap[origIdx] : origIdx;

    if (postIdx < 0) {
      // Absorbed: append tensor dim i to the last group.
      assert(!reassoc.empty());
      reassoc.back().push_back((int64_t)i);
    } else if (postIdx == prevPost) {
      // Same post-axis as previous (shouldn't happen for simple linalg maps).
      reassoc.back().push_back((int64_t)i);
    } else {
      reassoc.push_back({(int64_t)i});
      prevPost = postIdx;
    }
  }
  return reassoc;
}

// Return the collapsed tensor type (product of grouped dims; ? if any dynamic).
static RankedTensorType collapseType(RankedTensorType orig,
                                      ArrayRef<ReassociationIndices> reassoc) {
  SmallVector<int64_t> shape;
  for (auto &group : reassoc) {
    int64_t size = 1;
    for (int64_t dimPos : group) {
      int64_t d = orig.getDimSize((int)dimPos);
      size = (d == ShapedType::kDynamic || size == ShapedType::kDynamic)
                 ? ShapedType::kDynamic : size * d;
    }
    shape.push_back(size);
  }
  return RankedTensorType::get(shape, orig.getElementType());
}

// Emit tensor.collapse_shape for an operand whose map is Case C.
// Returns the collapsed Value (or the original if Case A).
static Value emitCollapseIfNeeded(OpBuilder &builder, Location loc,
                                   Value operand, AffineMap map,
                                   ArrayRef<int> collapseGroup,
                                   ArrayRef<int> axisMap) {
  if (classifyInput(map, collapseGroup) != InputClass::C)
    return operand; // Case A: no collapse
  auto reassoc = buildReassociation(map, axisMap);
  auto oldType = cast<RankedTensorType>(operand.getType());
  auto newType = collapseType(oldType, reassoc);
  return builder.create<tensor::CollapseShapeOp>(loc, newType, operand,
                                                  reassoc);
}

//===----------------------------------------------------------------------===//
// applyCollapseForSingleGeneric — IR transformation
//===----------------------------------------------------------------------===//

// Performs the Phase 1 IR transformation for a func containing exactly one
// linalg.generic. Inserts collapse_shape for C-type operands and outs, rewrites
// the linalg.generic with post-collapse maps and shapes, and inserts expand_shape
// so the func return types remain unchanged. Also populates `info` in-place.
static void
applyIRTransform(OpBuilder &builder, func::FuncOp func,
                 linalg::GenericOp lop,
                 ArrayRef<int> collapseGroup,
                 const SmallVector<int> &axisMap,
                 int numPostDims) {
  Location loc = lop.getLoc();
  builder.setInsertionPoint(lop);

  // --- Inputs ---
  SmallVector<Value> newInputs;
  SmallVector<AffineMap> newMaps;

  for (auto [operand, map] :
       llvm::zip(lop.getInputs(), lop.getIndexingMapsArray())) {
    newInputs.push_back(
        emitCollapseIfNeeded(builder, loc, operand, map, collapseGroup, axisMap));
    newMaps.push_back(rewriteMap(map, axisMap, numPostDims));
  }

  // --- Outputs (outs/init tensors) ---
  SmallVector<Value> newOuts;
  int numInputs = lop.getNumDpsInputs();
  for (auto [outVal, outMap] :
       llvm::zip(lop.getOutputs(),
                 lop.getIndexingMapsArray().drop_front(numInputs))) {
    newOuts.push_back(
        emitCollapseIfNeeded(builder, loc, outVal, outMap, collapseGroup, axisMap));
    newMaps.push_back(rewriteMap(outMap, axisMap, numPostDims));
  }

  // --- New iterator types (drop absorbed axes) ---
  SmallVector<utils::IteratorType> newIterTypes;
  for (auto [origIdx, ax] : llvm::enumerate(lop.getIteratorTypesArray())) {
    int postIdx = ((int)origIdx < (int)axisMap.size()) ? axisMap[origIdx] : (int)origIdx;
    if (postIdx < 0) continue; // absorbed
    newIterTypes.push_back(ax);
  }

  // --- Result types ---
  SmallVector<Type> resultTypes;
  for (Value out : newOuts)
    resultTypes.push_back(out.getType());

  // --- Create new linalg.generic ---
  auto newGeneric = builder.create<GenericOp>(
      loc, resultTypes, newInputs, newOuts,
      builder.getAffineMapArrayAttr(newMaps),
      builder.getStrArrayAttr(llvm::to_vector(llvm::map_range(
          newIterTypes, [](utils::IteratorType it) {
            return it == utils::IteratorType::reduction ? "reduction" : "parallel";
          }))));

  // Clone the body region (scalar body types are unchanged).
  IRMapping mapping;
  lop.getRegion().cloneInto(&newGeneric.getRegion(), mapping);

  // --- expand_shape for outputs → restore original result types ---
  builder.setInsertionPointAfter(newGeneric);
  for (auto [oldRes, newRes] :
       llvm::zip(lop.getResults(), newGeneric.getResults())) {
    // Build expand reassociation = inverse of collapse reassociation.
    int outMapIdx = numInputs; // first output map index
    AffineMap outMap = lop.getIndexingMapsArray()[outMapIdx];
    auto reassoc = buildReassociation(outMap, axisMap);
    auto origType = cast<RankedTensorType>(oldRes.getType());
    Value expanded = builder.create<tensor::ExpandShapeOp>(
        loc, origType, newRes, reassoc);
    oldRes.replaceAllUsesWith(expanded);
  }

  lop.erase();
}

//===----------------------------------------------------------------------===//
// collapseGroup — public entry point
//===----------------------------------------------------------------------===//

CollapsedGroupInfo collapseGroup(OpBuilder &builder, func::FuncOp func) {
  // Collect linalg ops in program order.
  SmallVector<linalg::LinalgOp> members;
  func.walk([&](linalg::LinalgOp op) { members.push_back(op); });

  CollapsedGroupInfo result;
  result.kind = GroupInfo::Kind::Vector;

  if (members.empty())
    return result;

  auto canonicalAxes = computeCanonicalAxes(llvm::ArrayRef(members));
  result.canonicalAxes = canonicalAxes;
  result.topoMembers   = members;

  // Boundary inputs = used func arguments.
  SmallVector<Value> boundaryIn;
  for (Value arg : func.getArguments())
    if (!arg.use_empty())
      boundaryIn.push_back(arg);
  result.boundaryIn = boundaryIn;

  // Boundary outputs = return operands.
  auto retOp = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  result.boundaryOut = SmallVector<Value>(retOp.getOperands());

  // Identity mapping default.
  result.collapsedAxes = canonicalAxes;
  for (int i = 0; i < (int)canonicalAxes.size(); ++i)
    result.axisMap.push_back(i);

  // Find first candidate group that survives BCast pruning.
  auto candidates = findCandidateGroups(canonicalAxes);
  SmallVector<int> chosenGroup;
  DenseSet<int> chosenBCast;
  for (auto &cand : candidates) {
    auto bcast  = computeBCast(cand, members, boundaryIn);
    auto pruned = pruneBCastAxes(cand, bcast);
    if (!pruned.empty()) {
      chosenGroup = pruned.front();
      chosenBCast = bcast;
      break;
    }
  }
  if (chosenGroup.empty())
    return result; // nothing to collapse

  // B2 check (v1: fall back to noCollapse).
  if (hasAnyB2(chosenGroup, members, boundaryIn)) {
    result.hasB2      = true;
    result.noCollapse = true;
    return result;
  }

  // Build axisMap and collapsedAxes.
  SmallVector<int> newAxisMap;
  SmallVector<AxisInfo> newCollapsedAxes;
  buildAxisMap(canonicalAxes, chosenGroup, newAxisMap, newCollapsedAxes);
  result.axisMap       = newAxisMap;
  result.collapsedAxes = newCollapsedAxes;
  int numPost = (int)newCollapsedAxes.size();

  // Fill broadcastAxes (post-collapse idx for each BCast axis).
  for (int origIdx : chosenBCast) {
    int pm = (origIdx < (int)newAxisMap.size()) ? newAxisMap[origIdx] : origIdx;
    if (pm >= 0 && !llvm::is_contained(result.broadcastAxes, pm))
      result.broadcastAxes.push_back(pm);
  }
  llvm::sort(result.broadcastAxes);

  // IR transformation (v1: single generic only).
  if (members.size() == 1) {
    auto lop = dyn_cast<linalg::GenericOp>(members.front().getOperation());
    if (lop)
      applyIRTransform(builder, func, lop, chosenGroup, newAxisMap, numPost);
  }
  // Multi-op case: analysis done, IR transform deferred to future work.

  return result;
}

} // namespace mlir::afir
```

**Note on `getStrArrayAttr` for iterator types:** MLIR's `linalg.generic` historically uses `StringAttr` array ("parallel"/"reduction"). In recent MLIR versions the canonical form uses `linalg.iterator_type<...>`. If the `getStrArrayAttr` line fails to compile because `GenericOp` requires `ArrayAttr` of `IteratorTypeAttr`, replace it with:

```cpp
builder.getArrayAttr(llvm::to_vector(llvm::map_range(
    newIterTypes, [&](utils::IteratorType it) -> Attribute {
      return linalg::IteratorTypeAttr::get(builder.getContext(), it);
    })))
```

- [ ] **Step 5: Update `TileFusePass.cpp` to call Phase 1 (no-op on Phase 2/3)**

Replace the entire body of `runOnOperation()` in `TileFusePass.cpp`:

```cpp
void runOnOperation() override {
  func::FuncOp func = getOperation();
  OpBuilder builder(func.getContext());

  // Phase 0: absorb linalg.broadcast into downstream linalg.generic.
  {
    RewritePatternSet patterns(&getContext());
    patterns.add<mlir::afir::AbsorbBroadcastIntoGeneric>(&getContext());
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }

  // Phase 1: Collapse — transforms IR, returns CollapsedGroupInfo.
  auto collapsedInfo = mlir::afir::collapseGroup(builder, func);
  (void)collapsedInfo; // Phase 2/3 consume this — TODO

  // Phase 2: TilePlanGen  — TODO (impl-03 Phase 2)
  // Phase 3: LoopNestBuilder + GroupEmitter — TODO (impl-03 Phase 3)
}
```

Add the required includes to `TileFusePass.cpp`:

```cpp
#include "Collapse.h"
#include "Conversion/VectorPlan/VectorPlanPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
```

Also forward-declare `AbsorbBroadcastIntoGeneric` or include the pattern via a shared header. The simplest approach: move the pattern struct declaration to a `BroadcastAbsorb.h` internal header, include it from both `BroadcastAbsorb.cpp` and `TileFusePass.cpp`.

Alternatively, just call `createVectorPlanBroadcastAbsorbPass` and apply it via `PassManager`, or factor the pattern into a helper function:

```cpp
// In BroadcastAbsorb.cpp, add a non-pattern public helper:
namespace mlir::afir {
void populateBroadcastAbsorbPatterns(RewritePatternSet &patterns) {
  patterns.add<AbsorbBroadcastIntoGeneric>(patterns.getContext());
}
} // namespace

// In VectorPlanPasses.h:
void populateBroadcastAbsorbPatterns(mlir::RewritePatternSet &patterns);
```

Use `populateBroadcastAbsorbPatterns` in `TileFusePass.cpp`.

- [ ] **Step 6: Build**

```bash
cd build && ninja mlir-opt 2>&1 | grep error | head -20
```
Expected: build succeeds (fix any MLIR API compilation errors — see note on iterator types above and `GenericOp` constructor signature).

- [ ] **Step 7: Run prune-middle test**

```bash
mlir-opt --vector-plan-tile-fuse \
  ../test/Conversion/VectorPlan/collapse-prune-middle-axis.mlir | \
FileCheck ../test/Conversion/VectorPlan/collapse-prune-middle-axis.mlir
```
Expected: PASS (no tensor.collapse_shape emitted).

- [ ] **Step 8: Run prune-tail test**

```bash
mlir-opt --vector-plan-tile-fuse \
  ../test/Conversion/VectorPlan/collapse-prune-tail-axis.mlir | \
FileCheck ../test/Conversion/VectorPlan/collapse-prune-tail-axis.mlir
```
Expected: PASS (tensor.collapse_shape ... into tensor<32x16xf16> present).

- [ ] **Step 9: Commit**

```bash
git add lib/Conversion/VectorPlan/TileFuse/Collapse.h \
        lib/Conversion/VectorPlan/TileFuse/Collapse.cpp \
        lib/Conversion/VectorPlan/TileFuse/TileFusePass.cpp \
        test/Conversion/VectorPlan/collapse-prune-middle-axis.mlir \
        test/Conversion/VectorPlan/collapse-prune-tail-axis.mlir
git commit -m "feat(vector-plan): Phase 1 Collapse (BCast pruning + A/C IR transformation)"
```

---

## Task 4: End-to-end collapse test (Case A + Case C)

**Files:**
- Create: `test/Conversion/VectorPlan/tile-fuse-collapse-c.mlir`

This test covers the LayerNorm-like scenario from the spec: `scale[H]` and `bias[H]` are Case A (complete absence of d0/d1 from their maps → no BCast contribution), `input[B,S,H]` is Case C → collapse d0*d1.

- [ ] **Step 1: Write `tile-fuse-collapse-c` test**

Create `test/Conversion/VectorPlan/tile-fuse-collapse-c.mlir`:

```mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
//
// LayerNorm-style scale+bias:
//   input[4,8,16] — full (d0,d1,d2)→(d0,d1,d2)  Case C
//   scale[16]     — (d0,d1,d2)→(d2)              Case A: d0,d1 fully absent
//   bias[16]      — (d0,d1,d2)→(d2)              Case A
//
// BCast analysis for candidate G={d0,d1}:
//   scale: d0 and d1 BOTH absent → ALL of G absent → Case A → no BCast contribution.
//   No partial absence → BCast(G) = {}.
//   After pruning: G={d0,d1} unchanged (size 2, keep) → collapse!
//
// Expected: tensor.collapse_shape [[0,1],[2]] → tensor<32x16xf16>
//           tensor.expand_shape to restore return type tensor<4x8x16xf16>
//
// CHECK-LABEL: func.func @kernel_group0
func.func @kernel_group0(
    %input: tensor<4x8x16xf16>,
    %scale: tensor<16xf16>,
    %bias:  tensor<16xf16>) -> tensor<4x8x16xf16> {
  %init = tensor.empty() : tensor<4x8x16xf16>
  %r = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,  // input: Case C
      affine_map<(d0, d1, d2) -> (d2)>,            // scale: Case A
      affine_map<(d0, d1, d2) -> (d2)>,            // bias:  Case A
      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],  // output: Case C
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input, %scale, %bias : tensor<4x8x16xf16>, tensor<16xf16>, tensor<16xf16>)
    outs(%init : tensor<4x8x16xf16>) {
    ^bb0(%a: f16, %s: f16, %b: f16, %o: f16):
      %mul = arith.mulf %a, %s : f16
      %add = arith.addf %mul, %b : f16
      linalg.yield %add : f16
  } -> tensor<4x8x16xf16>
  return %r : tensor<4x8x16xf16>
}
// After collapse: tensor.collapse_shape collapses d0*d1 = 32
// CHECK: tensor.collapse_shape
// CHECK-SAME: into tensor<32x16xf16>
// The return type is preserved via expand_shape
// CHECK: tensor.expand_shape
// CHECK-SAME: into tensor<4x8x16xf16>
// linalg.generic uses 2D iteration (d01, d2)
// CHECK: linalg.generic
// CHECK-SAME: (d0, d1) -> (d0, d1)
// CHECK-SAME: (d0, d1) -> (d1)
```

- [ ] **Step 2: Run the test**

```bash
cd build && mlir-opt --vector-plan-tile-fuse \
  ../test/Conversion/VectorPlan/tile-fuse-collapse-c.mlir | \
FileCheck ../test/Conversion/VectorPlan/tile-fuse-collapse-c.mlir
```
Expected: PASS.

- [ ] **Step 3: Run the existing VectorPlan regression tests**

```bash
cd build && ninja check-mlir-afir 2>&1 | tail -20
```
Or run lit directly:
```bash
python3 -m pytest ../test/Conversion/VectorPlan/ -x 2>&1 | tail -20
```
Expected: all previously passing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add test/Conversion/VectorPlan/tile-fuse-collapse-c.mlir
git commit -m "test(vector-plan): add end-to-end collapse Case-A+C test (tile-fuse-collapse-c)"
```

---

## Self-Review

**Spec coverage:**
- BroadcastAbsorb: covered in Task 2 ✓
- `findCandidateGroups`: covered in Task 3 (Collapse.cpp) ✓
- `pruneBCastAxes` (BAII L1 落地): covered in Task 3 ✓
- BCast partial-only definition (Case A does not contribute): captured in the BCast definition note and `computeBCast` implementation ✓
- `classifyInput` A/C/B2: covered in Task 3 ✓
- B2 → `noCollapse=true` (v1 fallback): covered in Task 3 ✓
- `CollapsedGroupInfo::broadcastAxes` filled: covered in Task 3 ✓
- IR transformation (collapse_shape + rewrite maps + expand_shape): covered in Task 3 ✓
- `tile-fuse-collapse-c` spec test scenario: covered in Task 4 ✓
- `collapse-prune-middle-axis` / `collapse-prune-tail-axis`: covered in Task 3 ✓

**Not in scope (future):**
- Multi-op group IR transform (analysis works; IR transform skipped for `members.size() > 1`)
- B2 Variant 1/2 (Phase 5/6)
- Dynamic shape extent tracking for `broadcastAxisExtents`
- Phase 2 TilePlanGen, Phase 3 LoopNestBuilder + GroupEmitter
- BAII L1/L2 verifier

**Known MLIR API details to verify during implementation:**
1. `BroadcastOp::getDimensions()` return type — may be `DenseI64ArrayAttr` or `ArrayAttr`; adjust `getAsValueRange` call accordingly.
2. `GenericOp` iterator type attribute: may require `linalg::IteratorTypeAttr::get(ctx, it)` array rather than string array depending on MLIR version. See note in Task 3 Step 4.
3. `GenericOp` body region: after `cloneInto`, verify no duplicate entry block. If GenericOp creates an empty entry block on construction, use `cloneInto` with `mapping` to append instead.
4. `tensor.expand_shape` reassociation must match the `tensor.collapse_shape` exactly (same `ReassociationIndices`).