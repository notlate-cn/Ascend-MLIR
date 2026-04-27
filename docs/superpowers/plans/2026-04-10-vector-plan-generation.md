# Vector Plan Generation — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the generalized Phase 1 Vector track pipeline that compiles arbitrary linalg programs into tiled+fused Ascend NPU kernels without hand-written transform scripts.

**Architecture:** Phase 1 consists of four preprocessing passes (Step 1) followed by a single orchestrator pass `--vector-plan-generation` (Steps 2–8). The preprocessing passes are independent and reusable; the orchestrator pass performs chain analysis, golden axis construction, axis normalization, dimension collapse, split-tiling, plan generation, and realization via the upstream `tileConsumerAndFuseProducersUsingSCF` API. Each chain produces an independent kernel function with symbolic tile parameters.

**Tech Stack:** MLIR C++ (upstream linalg/SCF/tensor dialects), TableGen pass registration, lit+FileCheck testing.

**Spec:** `docs/superpowers/specs/2026-04-09-mlir-ai-compiler-generalization-design.md`

---

## File Structure

### New Files

```
lib/Conversion/CanonicalizeExtractBroadcast/
  CanonicalizeExtractBroadcastPass.cpp   — Step 1.2: rewrite extract-select → broadcast
  CMakeLists.txt

include/Conversion/CanonicalizeExtractBroadcast/
  CanonicalizeExtractBroadcastPass.h

lib/Conversion/VectorPlanGeneration/
  ChainAnalysis.h          — Step 2 data structures + algorithm
  ChainAnalysis.cpp
  GoldenAxis.h             — Steps 3–4: golden axis + normalization
  GoldenAxis.cpp
  PlanGeneration.h         — Steps 5–7: collapse, split-tiling, plan records
  PlanGeneration.cpp
  PlanRealization.h        — Step 8: tileConsumerAndFuseProducersUsingSCF wrapper
  PlanRealization.cpp
  VectorPlanGenerationPass.cpp  — orchestrator pass
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

### Modified Files

```
include/Conversion/Passes.td             — add 3 new pass definitions
include/Conversion/Passes.h              — add 2 new includes
lib/Conversion/CMakeLists.txt            — add 2 new subdirectories
tools/afir-opt/CMakeLists.txt            — link 2 new libraries
```

---

### Task 1: Integrate EliminateCfAssert into the Build

The pass code already exists at `lib/Conversion/EliminateCfAssert/` and `include/Conversion/EliminateCfAssert/` but is not registered in the build system or TableGen.

**Files:**
- Modify: `include/Conversion/Passes.td`
- Modify: `include/Conversion/Passes.h`
- Modify: `lib/Conversion/CMakeLists.txt`
- Modify: `tools/afir-opt/CMakeLists.txt`
- Create: `test/Conversion/eliminate-cf-assert.mlir`

- [ ] **Step 1: Add TableGen definition**

Add to `include/Conversion/Passes.td` before the `#endif`:

```tablegen
//===----------------------------------------------------------------------===//
// EliminateCfAssert
//===----------------------------------------------------------------------===//

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

- [ ] **Step 2: Add include to Passes.h**

Add to `include/Conversion/Passes.h` after the existing includes (line 20):

```cpp
#include "Conversion/EliminateCfAssert/EliminateCfAssertPass.h"
```

- [ ] **Step 3: Add subdirectory to lib CMake**

Add to `lib/Conversion/CMakeLists.txt`:

```cmake
add_subdirectory(EliminateCfAssert)
```

- [ ] **Step 4: Link library in afir-opt**

Add to `tools/afir-opt/CMakeLists.txt` in the `target_link_libraries` block:

```cmake
    EliminateCfAssertConversion
```

- [ ] **Step 5: Write the lit test**

Create `test/Conversion/eliminate-cf-assert.mlir`:

```mlir
// RUN: afir-opt --eliminate-cf-assert %s | FileCheck %s

// CHECK-LABEL: func.func @remove_broadcast_assert
// CHECK-NOT: cf.assert
// CHECK-NOT: arith.cmpi
func.func @remove_broadcast_assert(%arg0: tensor<?x?x128xf32>) -> tensor<?x?x128xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?x128xf32>
  %c1 = arith.constant 1 : index
  %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?x128xf32>
  %0 = arith.cmpi eq, %dim, %dim : index
  cf.assert %0, "mismatched size for broadcast"
  %1 = arith.cmpi eq, %dim_1, %dim_1 : index
  cf.assert %1, "mismatched size for broadcast"
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

// CHECK-LABEL: func.func @no_asserts_noop
func.func @no_asserts_noop(%arg0: f32) -> f32 {
  return %arg0 : f32
}
```

- [ ] **Step 6: Build and run test**

Run:
```bash
cmake --build build --target afir-opt && \
build/bin/llvm-lit test/Conversion/eliminate-cf-assert.mlir -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add include/Conversion/Passes.td include/Conversion/Passes.h \
        include/Conversion/EliminateCfAssert/ \
        lib/Conversion/EliminateCfAssert/ \
        lib/Conversion/CMakeLists.txt \
        tools/afir-opt/CMakeLists.txt \
        test/Conversion/eliminate-cf-assert.mlir
git commit -m "feat: integrate --eliminate-cf-assert pass into build system"
```

---

### Task 2: CanonicalizeExtractBroadcast Pass

Implements spec Appendix A. Rewrites the torch-mlir `extract-select` dynamic broadcast pattern into explicit `linalg.broadcast` or identity-map generics.

**Files:**
- Create: `include/Conversion/CanonicalizeExtractBroadcast/CanonicalizeExtractBroadcastPass.h`
- Create: `lib/Conversion/CanonicalizeExtractBroadcast/CanonicalizeExtractBroadcastPass.cpp`
- Create: `lib/Conversion/CanonicalizeExtractBroadcast/CMakeLists.txt`
- Modify: `include/Conversion/Passes.td`
- Modify: `include/Conversion/Passes.h`
- Modify: `lib/Conversion/CMakeLists.txt`
- Modify: `tools/afir-opt/CMakeLists.txt`
- Create: `test/Conversion/canonicalize-extract-broadcast.mlir`

- [ ] **Step 1: Write the failing test**

Create `test/Conversion/canonicalize-extract-broadcast.mlir`:

```mlir
// RUN: afir-opt --canonicalize-extract-broadcast %s | FileCheck %s

// The torch-mlir extract-select broadcast pattern:
//   linalg.generic with body:
//     linalg.index → arith.cmpi eq (dim, 1) → arith.select (0, index) → tensor.extract
//
// Should be rewritten to a linalg.generic with a projection indexing_map
// that broadcasts the source dimensions.

#map_identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @broadcast_from_reduction
// CHECK-NOT: tensor.extract
// CHECK-NOT: arith.select
// CHECK: linalg.generic
// CHECK-SAME: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
// CHECK-SAME: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @broadcast_from_reduction(
    %src: tensor<?x?x1xf32>, %dst: tensor<?x?x128xf32>,
    %dim0: index, %dim1: index
) -> tensor<?x?x128xf32> {
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

// CHECK-LABEL: func.func @not_a_broadcast
// CHECK: tensor.extract
func.func @not_a_broadcast(
    %src: tensor<?x?xf32>, %dst: tensor<?x?xf32>,
    %idx: tensor<?xi32>
) -> tensor<?x?xf32> {
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

- [ ] **Step 2: Add TableGen definition**

Add to `include/Conversion/Passes.td` before the `#endif`:

```tablegen
//===----------------------------------------------------------------------===//
// CanonicalizeExtractBroadcast
//===----------------------------------------------------------------------===//

def CanonicalizeExtractBroadcastPass
    : Pass<"canonicalize-extract-broadcast", "mlir::func::FuncOp"> {
  let summary = "Rewrite extract-select broadcast pattern to projection indexing_map";
  let description = [{
    torch-mlir emits a dynamic broadcast pattern using linalg.index + arith.cmpi
    + arith.select + tensor.extract for shapes that may be 1 at runtime. This
    pass detects that structure and rewrites it to a linalg.generic with a
    projection indexing_map that maps the broadcast dimension to a constant 0.

    Recognition rules (from spec Appendix A):
    1. Exactly one tensor.extract in the body, no other memory-reading ops.
    2. Every iteration dim is parallel.
    3. Every index operand to tensor.extract is either:
       - arith.constant, or
       - arith.select(arith.cmpi eq, %src_dim_k, %c1), %c0, linalg.index %k
    4. The output indexing map is identity.
  }];
  let constructor = "mlir::afir::createCanonicalizeExtractBroadcastPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];
}
```

- [ ] **Step 3: Create header**

Create `include/Conversion/CanonicalizeExtractBroadcast/CanonicalizeExtractBroadcastPass.h`:

```cpp
//===- CanonicalizeExtractBroadcastPass.h -----------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_CANONICALIZEEXTRACTBROADCAST_H
#define ASCEND_MLIR_CONVERSION_CANONICALIZEEXTRACTBROADCAST_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createCanonicalizeExtractBroadcastPass();

}  // namespace mlir::afir

#endif
```

- [ ] **Step 4: Create CMakeLists.txt**

Create `lib/Conversion/CanonicalizeExtractBroadcast/CMakeLists.txt`:

```cmake
add_mlir_library(CanonicalizeExtractBroadcastConversion
  CanonicalizeExtractBroadcastPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRTransforms
)
```

- [ ] **Step 5: Implement the pass**

Create `lib/Conversion/CanonicalizeExtractBroadcast/CanonicalizeExtractBroadcastPass.cpp`:

```cpp
//===- CanonicalizeExtractBroadcastPass.cpp ---------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// Rewrites torch-mlir's extract-select dynamic broadcast pattern:
//
//   %i0 = linalg.index 0
//   %p0 = arith.cmpi eq, %dim, %c1
//   %s0 = arith.select %p0, %c0, %i0
//   %e  = tensor.extract %src[%s0, ...]
//
// into a linalg.generic with a projection indexing_map:
//
//   linalg.generic { indexing_maps = [(d0,d1,d2) -> (d0, d1, 0),
//                                     (d0,d1,d2) -> (d0, d1, d2)] }
//     ins(%src) outs(%dst) { yield %in }
//
//===----------------------------------------------------------------------===//

#include "Conversion/CanonicalizeExtractBroadcast/CanonicalizeExtractBroadcastPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DECL_CANONICALIZEEXTRACTBROADCASTPASS
#define GEN_PASS_DEF_CANONICALIZEEXTRACTBROADCASTPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

/// For a given index operand of tensor.extract, determine whether it is:
///   (a) a constant index → return the constant as an AffineExpr
///   (b) arith.select(cmpi eq %dim %c1, %c0, linalg.index %k) → return d(k)
///   (c) linalg.index %k directly → return d(k)
/// Returns std::nullopt if the pattern is not recognized.
static std::optional<AffineExpr>
classifyExtractIndex(Value indexVal, MLIRContext *ctx) {
  // Case (a): arith.constant
  if (auto constOp = indexVal.getDefiningOp<arith::ConstantIndexOp>())
    return getAffineConstantExpr(constOp.value(), ctx);

  // Case (c): linalg.index directly
  if (auto indexOp = indexVal.getDefiningOp<linalg::IndexOp>())
    return getAffineDimExpr(indexOp.getDim(), ctx);

  // Case (b): arith.select(cmpi eq %dim %c1, %c0, linalg.index)
  auto selectOp = indexVal.getDefiningOp<arith::SelectOp>();
  if (!selectOp)
    return std::nullopt;

  // The false-value should be linalg.index %k
  auto falseIndex = selectOp.getFalseValue().getDefiningOp<linalg::IndexOp>();
  if (!falseIndex)
    return std::nullopt;

  // The true-value should be constant 0
  auto trueConst =
      selectOp.getTrueValue().getDefiningOp<arith::ConstantIndexOp>();
  if (!trueConst || trueConst.value() != 0)
    return std::nullopt;

  // The condition should be arith.cmpi eq, %dim, %c1
  auto cmpOp = selectOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmpOp || cmpOp.getPredicate() != arith::CmpIPredicate::eq)
    return std::nullopt;

  // One operand should be constant 1
  auto lhsConst = cmpOp.getLhs().getDefiningOp<arith::ConstantIndexOp>();
  auto rhsConst = cmpOp.getRhs().getDefiningOp<arith::ConstantIndexOp>();
  bool hasConst1 = (lhsConst && lhsConst.value() == 1) ||
                   (rhsConst && rhsConst.value() == 1);
  if (!hasConst1)
    return std::nullopt;

  // This dimension is broadcast: map to constant 0 in the indexing map
  return getAffineConstantExpr(0, ctx);
}

/// Check if a linalg.generic matches the extract-select broadcast pattern
/// and rewrite it. Returns true if rewritten.
static bool tryRewriteExtractBroadcast(linalg::GenericOp genericOp) {
  // Rule 2: all iterators must be parallel
  for (auto iterType : genericOp.getIteratorTypesArray())
    if (iterType != utils::IteratorType::parallel)
      return false;

  // Rule 4: must have zero inputs and identity output map (outs-only generic)
  if (genericOp.getNumDpsInputs() != 0)
    return false;

  // Rule 1: body must contain exactly one tensor.extract
  Block &body = genericOp.getRegion().front();
  SmallVector<tensor::ExtractOp> extracts;
  body.walk([&](tensor::ExtractOp op) { extracts.push_back(op); });
  if (extracts.size() != 1)
    return false;

  tensor::ExtractOp extractOp = extracts[0];

  // The yield must yield the extract result
  auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());
  if (yieldOp.getNumOperands() != 1 ||
      yieldOp.getOperand(0) != extractOp.getResult())
    return false;

  // Rule 3: classify every index operand of the extract
  MLIRContext *ctx = genericOp.getContext();
  unsigned rank = genericOp.getNumLoops();
  SmallVector<AffineExpr> srcExprs;
  for (Value idx : extractOp.getIndices()) {
    auto expr = classifyExtractIndex(idx, ctx);
    if (!expr)
      return false;
    srcExprs.push_back(*expr);
  }

  // Build the source indexing map
  AffineMap srcMap = AffineMap::get(rank, 0, srcExprs, ctx);

  // Build the output indexing map (identity)
  AffineMap dstMap = genericOp.getIndexingMapsArray().back();

  // Rewrite: replace the generic with a new one that has src as input
  Value srcTensor = extractOp.getTensor();
  OpBuilder builder(genericOp);

  auto newGeneric = builder.create<linalg::GenericOp>(
      genericOp.getLoc(),
      genericOp.getResultTypes(),
      /*inputs=*/ValueRange{srcTensor},
      /*outputs=*/genericOp.getDpsInits(),
      /*indexingMaps=*/SmallVector<AffineMap>{srcMap, dstMap},
      genericOp.getIteratorTypesArray(),
      /*bodyBuild=*/[](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args[0]); // yield the input element
      });

  genericOp.getResult(0).replaceAllUsesWith(newGeneric.getResult(0));
  genericOp.erase();
  return true;
}

struct CanonicalizeExtractBroadcastPass
    : public ::impl::CanonicalizeExtractBroadcastPassBase<
          CanonicalizeExtractBroadcastPass> {
  using CanonicalizeExtractBroadcastPassBase::
      CanonicalizeExtractBroadcastPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect generics first to avoid iterator invalidation.
    SmallVector<linalg::GenericOp> candidates;
    func.walk([&](linalg::GenericOp op) { candidates.push_back(op); });

    for (auto op : candidates)
      tryRewriteExtractBroadcast(op);
  }
};

std::unique_ptr<Pass> createCanonicalizeExtractBroadcastPass() {
  return std::make_unique<CanonicalizeExtractBroadcastPass>();
}

} // namespace mlir::afir
```

- [ ] **Step 6: Register in build system**

Add to `include/Conversion/Passes.h` (after the EliminateCfAssert include):

```cpp
#include "Conversion/CanonicalizeExtractBroadcast/CanonicalizeExtractBroadcastPass.h"
```

Add to `lib/Conversion/CMakeLists.txt`:

```cmake
add_subdirectory(CanonicalizeExtractBroadcast)
```

Add to `tools/afir-opt/CMakeLists.txt`:

```cmake
    CanonicalizeExtractBroadcastConversion
```

- [ ] **Step 7: Build and run test**

Run:
```bash
cmake --build build --target afir-opt && \
build/bin/llvm-lit test/Conversion/canonicalize-extract-broadcast.mlir -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add include/Conversion/CanonicalizeExtractBroadcast/ \
        lib/Conversion/CanonicalizeExtractBroadcast/ \
        include/Conversion/Passes.td include/Conversion/Passes.h \
        lib/Conversion/CMakeLists.txt tools/afir-opt/CMakeLists.txt \
        test/Conversion/canonicalize-extract-broadcast.mlir
git commit -m "feat: add --canonicalize-extract-broadcast pass (Phase 1 Step 1.2)"
```

---

### Task 3: Chain Analysis Infrastructure (Steps 2)

Builds the data structures and algorithm for root-driven backward chain partitioning. This task implements the pass skeleton and chain analysis only — no IR transformation yet. The pass emits chain boundaries as remarks for testing.

**Files:**
- Create: `lib/Conversion/VectorPlanGeneration/ChainAnalysis.h`
- Create: `lib/Conversion/VectorPlanGeneration/ChainAnalysis.cpp`
- Create: `test/Conversion/vector-plan-chain-analysis.mlir`

- [ ] **Step 1: Write the failing test**

Create `test/Conversion/vector-plan-chain-analysis.mlir`:

```mlir
// RUN: afir-opt --vector-plan-generation %s -mlir-print-ir-after-all 2>&1 | FileCheck %s

// Test: pointwise chain with single root
// Expected: one chain rooted at the add, containing both generics.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK: remark: chain{{.*}}root: {{.*}}addf
// CHECK: remark: chain{{.*}}members: 2
func.func @pointwise_chain(%arg0: tensor<4x8xf32>,
                           %bias: tensor<8xf32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %relu = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 0.0 : f32
      %r = arith.maximumf %in, %cst : f32
      linalg.yield %r : f32
  } -> tensor<4x8xf32>
  %result = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%relu, %bias : tensor<4x8xf32>, tensor<8xf32>)
      outs(%empty : tensor<4x8xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %r = arith.addf %in0, %in1 : f32
      linalg.yield %r : f32
  } -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// Test: chain cuts at matmul (Cube op)
// Expected: two chains — one before matmul, one after.

// CHECK: remark: chain{{.*}}root: {{.*}}addf
// CHECK: remark: chain{{.*}}members: 1
// CHECK: remark: chain{{.*}}boundary: {{.*}}matmul{{.*}}cube
func.func @cube_boundary(%arg0: tensor<4x8xf32>,
                         %weight: tensor<8x4xf32>,
                         %bias: tensor<4xf32>) -> tensor<4x4xf32> {
  %empty = tensor.empty() : tensor<4x4xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
  %matmul = linalg.matmul
      ins(%arg0, %weight : tensor<4x8xf32>, tensor<8x4xf32>)
      outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul, %bias : tensor<4x4xf32>, tensor<4xf32>)
      outs(%empty : tensor<4x4xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %r = arith.addf %in0, %in1 : f32
      linalg.yield %r : f32
  } -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// Test: pointwise multi-use keeps chain, reduction multi-use cuts
// CHECK: remark: chain{{.*}}pointwise_multiuse: allowed
func.func @multiuse_rules(%arg0: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %empty = tensor.empty() : tensor<4x8xf32>
  // This pointwise op has two uses — should stay in chain
  %relu = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 0.0 : f32
      %r = arith.maximumf %in, %cst : f32
      linalg.yield %r : f32
  } -> tensor<4x8xf32>
  %neg = linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%relu : tensor<4x8xf32>) outs(%empty : tensor<4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %r = arith.negf %in : f32
      linalg.yield %r : f32
  } -> tensor<4x8xf32>
  return %relu, %neg : tensor<4x8xf32>, tensor<4x8xf32>
}
```

- [ ] **Step 2: Define data structures**

Create `lib/Conversion/VectorPlanGeneration/ChainAnalysis.h`:

```cpp
//===- ChainAnalysis.h - Vector chain partitioning ---------------*- C++-*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_VECTORPLANGENERATION_CHAINANALYSIS_H
#define ASCEND_MLIR_VECTORPLANGENERATION_CHAINANALYSIS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

/// Why a chain boundary was formed.
enum class BoundaryReason {
  CubeOp,            // producer is a Cube-track op (matmul, etc.)
  NoTilingInterface,  // producer has no usable TilingInterface
  SideEffect,         // producer has side effects
  ReductionMultiUse,  // reduction op with >1 uses
  AlreadyClaimed,     // producer already belongs to another chain
  NotNormalizable,    // cannot be normalized to golden axis space
  BlockArgument,      // operand is a block argument (func input)
};

/// Record of why a chain stopped at a given edge.
struct BoundaryRecord {
  Operation *producer;
  Operation *consumer;
  BoundaryReason reason;
};

/// A single Vector chain: a maximal set of fusible Vector ops.
struct VectorChain {
  Operation *root;
  SmallVector<Operation *> members;  // topological order
  SmallVector<Value> externalInputs;
  SmallVector<Operation *> externalUsers;
  SmallVector<BoundaryRecord> boundaries;
};

/// Returns true if op is a Cube-track op (matmul, batch_matmul, conv, ...).
bool isCubeOp(Operation *op);

/// Returns true if op is a Vector-track candidate for chain analysis.
bool isVectorCandidate(Operation *op);

/// Returns true if op qualifies as a chain root (spec §4 Step 2.2).
bool isRootCandidate(Operation *op,
                     const llvm::DenseSet<Operation *> &claimed);

/// Build a chain from a root by backward producer traversal (spec §4 Step 2.3).
VectorChain buildChainFromRoot(Operation *root,
                               const llvm::DenseSet<Operation *> &claimed);

/// Run the full chain analysis on a function (spec §4 Step 2.1).
SmallVector<VectorChain> analyzeChains(func::FuncOp func);

} // namespace mlir::afir

#endif
```

- [ ] **Step 3: Implement chain analysis**

Create `lib/Conversion/VectorPlanGeneration/ChainAnalysis.cpp`:

```cpp
//===- ChainAnalysis.cpp - Vector chain partitioning -------------*- C++-*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ChainAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::afir;

bool mlir::afir::isCubeOp(Operation *op) {
  return isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op);
}

bool mlir::afir::isVectorCandidate(Operation *op) {
  if (isCubeOp(op))
    return false;
  // Must be a linalg structured op with TilingInterface
  if (!isa<linalg::LinalgOp>(op))
    return false;
  if (!isa<TilingInterface>(op))
    return false;
  // linalg.fill is tileable but not a Vector compute op — skip it.
  // It will be re-introduced by tile-and-fuse as a producer init.
  if (isa<linalg::FillOp>(op))
    return false;
  return true;
}

/// Returns true if op has any reduction iterator.
static bool hasReduction(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  for (auto it : linalgOp.getIteratorTypesArray())
    if (it == utils::IteratorType::reduction)
      return true;
  return false;
}

bool mlir::afir::isRootCandidate(Operation *op,
                                 const llvm::DenseSet<Operation *> &claimed) {
  if (!isVectorCandidate(op))
    return false;
  if (claimed.contains(op))
    return false;

  // At least one result escapes the Vector domain or is a function return.
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      // User is not a Vector candidate → boundary → root
      if (!isVectorCandidate(user) || claimed.contains(user))
        return true;
    }
    // No users at all → dead code, still a root
    if (result.use_empty())
      return true;
  }
  // Result is returned from the function
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (parentFunc) {
    auto returnOp = cast<func::ReturnOp>(
        parentFunc.getBody().front().getTerminator());
    for (Value operand : returnOp.getOperands()) {
      for (Value result : op->getResults()) {
        if (operand == result)
          return true;
      }
    }
  }
  return false;
}

/// Check if a producer can be absorbed into the chain (spec §4 Step 2.4).
static std::optional<BoundaryReason>
cannotAbsorbProducer(Operation *producer, Operation *consumer,
                     const llvm::DenseSet<Operation *> &claimed) {
  if (!producer)
    return BoundaryReason::BlockArgument;
  if (isCubeOp(producer))
    return BoundaryReason::CubeOp;
  if (!isa<TilingInterface>(producer))
    return BoundaryReason::NoTilingInterface;
  if (!isVectorCandidate(producer))
    return BoundaryReason::NoTilingInterface;
  if (!isPure(producer))
    return BoundaryReason::SideEffect;
  if (claimed.contains(producer))
    return BoundaryReason::AlreadyClaimed;

  // Fanout rule: reduction multi-use cuts the chain.
  if (hasReduction(producer)) {
    unsigned useCount = 0;
    for (Value result : producer->getResults())
      useCount += std::distance(result.getUsers().begin(),
                                result.getUsers().end());
    if (useCount > 1)
      return BoundaryReason::ReductionMultiUse;
  }

  return std::nullopt; // OK to absorb
}

VectorChain mlir::afir::buildChainFromRoot(
    Operation *root, const llvm::DenseSet<Operation *> &claimed) {
  VectorChain chain;
  chain.root = root;

  llvm::SetVector<Operation *> members;
  SmallVector<Operation *> worklist;
  worklist.push_back(root);

  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    if (members.contains(current))
      continue;
    members.insert(current);

    // Walk operands backward
    for (Value operand : current->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      auto reason = cannotAbsorbProducer(producer, current, claimed);
      if (reason) {
        chain.boundaries.push_back({producer, current, *reason});
        chain.externalInputs.push_back(operand);
        continue;
      }
      worklist.push_back(producer);
    }
  }

  // Topological sort of members (program order).
  SmallVector<Operation *> sorted;
  root->getParentRegion()->walk([&](Operation *op) {
    if (members.contains(op))
      sorted.push_back(op);
  });
  chain.members = sorted;

  // Record external users.
  for (Operation *member : chain.members) {
    for (Value result : member->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!members.contains(user))
          chain.externalUsers.push_back(user);
      }
    }
  }

  return chain;
}

SmallVector<VectorChain> mlir::afir::analyzeChains(func::FuncOp func) {
  // Collect ops in reverse program order.
  SmallVector<Operation *> reverseOrder;
  func.walk([&](Operation *op) { reverseOrder.push_back(op); });
  std::reverse(reverseOrder.begin(), reverseOrder.end());

  llvm::DenseSet<Operation *> claimed;
  SmallVector<VectorChain> chains;

  for (Operation *op : reverseOrder) {
    if (!isRootCandidate(op, claimed))
      continue;

    VectorChain chain = buildChainFromRoot(op, claimed);
    for (Operation *member : chain.members)
      claimed.insert(member);
    chains.push_back(std::move(chain));
  }

  return chains;
}
```

- [ ] **Step 4: Commit chain analysis**

```bash
git add lib/Conversion/VectorPlanGeneration/ChainAnalysis.h \
        lib/Conversion/VectorPlanGeneration/ChainAnalysis.cpp
git commit -m "feat: add chain analysis infrastructure (Phase 1 Step 2)"
```

---

### Task 4: VectorPlanGeneration Pass Skeleton + Chain Diagnostic

Wire up the pass skeleton that runs chain analysis and emits diagnostic remarks. This validates the chain partitioning before adding IR transformations.

**Files:**
- Create: `include/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.h`
- Create: `lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp`
- Create: `lib/Conversion/VectorPlanGeneration/CMakeLists.txt`
- Modify: `include/Conversion/Passes.td`
- Modify: `include/Conversion/Passes.h`
- Modify: `lib/Conversion/CMakeLists.txt`
- Modify: `tools/afir-opt/CMakeLists.txt`

- [ ] **Step 1: Add TableGen definition**

Add to `include/Conversion/Passes.td` before the `#endif`:

```tablegen
//===----------------------------------------------------------------------===//
// VectorPlanGeneration
//===----------------------------------------------------------------------===//

def VectorPlanGenerationPass
    : Pass<"vector-plan-generation", "mlir::func::FuncOp"> {
  let summary = "Generate tiled+fused Vector kernels from linalg programs";
  let description = [{
    Phase 1 orchestrator for the generalized Vector track pipeline.
    Performs chain analysis (Step 2), golden axis construction (Steps 3-4),
    dimension collapse (Step 5), split-tiling rules (Steps 6-7), and
    plan realization via tileConsumerAndFuseProducersUsingSCF (Step 8).

    Each independent Vector chain becomes a separate kernel function with
    symbolic tile parameters (XBLOCK, XBLOCK_SUB, RBLOCK).
  }];
  let constructor = "mlir::afir::createVectorPlanGenerationPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
  let options = [
    Option<"emitRemarks", "emit-remarks", "bool", /*default=*/"false",
           "Emit diagnostic remarks for chain analysis (for testing)">,
  ];
}
```

- [ ] **Step 2: Create header**

Create `include/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.h`:

```cpp
//===- VectorPlanGenerationPass.h --------------------------------*- C++-*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_VECTORPLANGENERATION_H
#define ASCEND_MLIR_CONVERSION_VECTORPLANGENERATION_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createVectorPlanGenerationPass();

}  // namespace mlir::afir

#endif
```

- [ ] **Step 3: Create pass skeleton with chain diagnostics**

Create `lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp`:

```cpp
//===- VectorPlanGenerationPass.cpp - Phase 1 orchestrator -------*- C++-*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/VectorPlanGeneration/VectorPlanGenerationPass.h"
#include "ChainAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

#define GEN_PASS_DECL_VECTORPLANGENERATIONPASS
#define GEN_PASS_DEF_VECTORPLANGENERATIONPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

static const char *boundaryReasonStr(BoundaryReason reason) {
  switch (reason) {
  case BoundaryReason::CubeOp:           return "cube";
  case BoundaryReason::NoTilingInterface: return "no_tiling_interface";
  case BoundaryReason::SideEffect:        return "side_effect";
  case BoundaryReason::ReductionMultiUse: return "reduction_multi_use";
  case BoundaryReason::AlreadyClaimed:    return "already_claimed";
  case BoundaryReason::NotNormalizable:   return "not_normalizable";
  case BoundaryReason::BlockArgument:     return "block_argument";
  }
  return "unknown";
}

struct VectorPlanGenerationPass
    : public ::impl::VectorPlanGenerationPassBase<VectorPlanGenerationPass> {
  using VectorPlanGenerationPassBase::VectorPlanGenerationPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Step 2: Chain analysis
    SmallVector<VectorChain> chains = analyzeChains(func);

    if (chains.empty())
      return;

    // Emit diagnostic remarks if requested (for testing)
    if (emitRemarks) {
      for (size_t i = 0; i < chains.size(); ++i) {
        auto &chain = chains[i];
        chain.root->emitRemark()
            << "chain[" << i << "] root: "
            << chain.root->getName().getStringRef()
            << ", members: " << chain.members.size();

        // Check for pointwise multi-use
        for (Operation *member : chain.members) {
          unsigned useCount = 0;
          for (Value result : member->getResults())
            useCount += std::distance(result.getUsers().begin(),
                                      result.getUsers().end());
          if (useCount > 1) {
            member->emitRemark() << "chain[" << i
                                 << "] pointwise_multiuse: allowed";
          }
        }

        for (auto &boundary : chain.boundaries) {
          if (boundary.producer) {
            boundary.consumer->emitRemark()
                << "chain[" << i << "] boundary: "
                << boundary.producer->getName().getStringRef()
                << " reason: " << boundaryReasonStr(boundary.reason);
          }
        }
      }
    }

    // Steps 3-8: TODO — will be added in Tasks 5-7
  }
};

std::unique_ptr<Pass> createVectorPlanGenerationPass() {
  return std::make_unique<VectorPlanGenerationPass>();
}

} // namespace mlir::afir
```

- [ ] **Step 4: Create CMakeLists.txt**

Create `lib/Conversion/VectorPlanGeneration/CMakeLists.txt`:

```cmake
add_mlir_library(VectorPlanGenerationConversion
  VectorPlanGenerationPass.cpp
  ChainAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTransforms
  MLIRTilingInterface
)
```

- [ ] **Step 5: Register in build system**

Add to `include/Conversion/Passes.h`:

```cpp
#include "Conversion/VectorPlanGeneration/VectorPlanGenerationPass.h"
```

Add to `lib/Conversion/CMakeLists.txt`:

```cmake
add_subdirectory(VectorPlanGeneration)
```

Add to `tools/afir-opt/CMakeLists.txt`:

```cmake
    VectorPlanGenerationConversion
```

- [ ] **Step 6: Build and run chain analysis tests**

Run:
```bash
cmake --build build --target afir-opt && \
build/bin/llvm-lit test/Conversion/vector-plan-chain-analysis.mlir -v
```

Expected: PASS — chain analysis detects correct roots, members, and boundaries.

- [ ] **Step 7: Commit**

```bash
git add include/Conversion/VectorPlanGeneration/ \
        lib/Conversion/VectorPlanGeneration/ \
        include/Conversion/Passes.td include/Conversion/Passes.h \
        lib/Conversion/CMakeLists.txt tools/afir-opt/CMakeLists.txt \
        test/Conversion/vector-plan-chain-analysis.mlir
git commit -m "feat: add --vector-plan-generation pass skeleton with chain analysis"
```

---

### Task 5: Golden Axis + Collapse (Steps 3–5)

Implements the golden axis construction, axis normalization, and dimension collapse. After this task, the pass can determine the canonical iteration space for each chain.

**Files:**
- Create: `lib/Conversion/VectorPlanGeneration/GoldenAxis.h`
- Create: `lib/Conversion/VectorPlanGeneration/GoldenAxis.cpp`
- Modify: `lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp`
- Modify: `lib/Conversion/VectorPlanGeneration/CMakeLists.txt`

- [ ] **Step 1: Define DimExpr and GoldenAxis data structures**

Create `lib/Conversion/VectorPlanGeneration/GoldenAxis.h`:

```cpp
//===- GoldenAxis.h - Golden axis layout + collapse ---------------*- C++-*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_VECTORPLANGENERATION_GOLDENAXIS_H
#define ASCEND_MLIR_VECTORPLANGENERATION_GOLDENAXIS_H

#include "ChainAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

/// Minimal symbolic dim expression for dynamic shapes (spec §4 Step 3.1).
struct DimExpr {
  enum Kind { Const, ArgDim, Mul };
  Kind kind;
  int64_t constVal = 0;        // for Const
  unsigned argNum = 0;         // for ArgDim
  unsigned dimNum = 0;         // for ArgDim
  SmallVector<DimExpr> factors; // for Mul

  static DimExpr makeConst(int64_t c) {
    DimExpr e; e.kind = Const; e.constVal = c; return e;
  }
  static DimExpr makeArgDim(unsigned arg, unsigned dim) {
    DimExpr e; e.kind = ArgDim; e.argNum = arg; e.dimNum = dim; return e;
  }
  static DimExpr makeMul(DimExpr a, DimExpr b) {
    DimExpr e; e.kind = Mul;
    e.factors.push_back(std::move(a));
    e.factors.push_back(std::move(b));
    return e;
  }
};

/// Role of a golden dim.
enum class DimRole { Parallel, Reduction, Mixed };

/// One dimension of the golden iteration space.
struct GoldenDim {
  DimRole role;
  DimExpr expr;
};

/// The golden axis layout for a chain.
struct GoldenAxisLayout {
  SmallVector<GoldenDim> dims;

  /// Indices of dims that can be collapsed (contiguous parallel dims).
  SmallVector<SmallVector<unsigned>> collapseGroups;

  /// After collapse: the final iteration dims.
  SmallVector<GoldenDim> collapsedDims;
};

/// Construct the golden axis layout for a chain (spec §4 Steps 3–4).
/// Uses the root's iterator_types as the starting layout, then maps
/// every other member's dims onto the root.
GoldenAxisLayout buildGoldenAxis(const VectorChain &chain,
                                 func::FuncOp func);

/// Determine which dims can be collapsed (spec §4 Step 5).
/// Modifies layout.collapseGroups and layout.collapsedDims.
void computeCollapse(GoldenAxisLayout &layout, const VectorChain &chain);

} // namespace mlir::afir

#endif
```

- [ ] **Step 2: Implement golden axis construction**

Create `lib/Conversion/VectorPlanGeneration/GoldenAxis.cpp`:

```cpp
//===- GoldenAxis.cpp - Golden axis layout + collapse ------------*- C++-*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "GoldenAxis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::afir;

/// Extract DimExpr for a given dim of a tensor value.
static DimExpr getDimExpr(Value tensor, unsigned dim, func::FuncOp func) {
  auto type = cast<RankedTensorType>(tensor.getType());
  if (!type.isDynamicDim(dim))
    return DimExpr::makeConst(type.getDimSize(dim));

  // If tensor is a block argument, it's a primitive dim.
  if (auto blockArg = dyn_cast<BlockArgument>(tensor)) {
    unsigned argNum = blockArg.getArgNumber();
    return DimExpr::makeArgDim(argNum, dim);
  }

  // For ops defining the tensor, trace through to find the source.
  // Simplified: for v1, treat as opaque dynamic dim.
  return DimExpr::makeArgDim(/*arg=*/0, dim);
}

GoldenAxisLayout mlir::afir::buildGoldenAxis(const VectorChain &chain,
                                              func::FuncOp func) {
  GoldenAxisLayout layout;

  // Start from the root's iteration space.
  auto rootLinalgOp = dyn_cast<linalg::LinalgOp>(chain.root);
  if (!rootLinalgOp)
    return layout;

  auto iterTypes = rootLinalgOp.getIteratorTypesArray();
  unsigned numLoops = iterTypes.size();

  // Get the output tensor to derive DimExprs.
  Value rootOutput = rootLinalgOp.getDpsInits()[0];
  auto outputType = cast<RankedTensorType>(rootOutput.getType());

  // Build golden dims from root.
  // Canonical ordering: parallel dims first, then reduction dims.
  SmallVector<unsigned> parallelDims, reductionDims;
  for (unsigned i = 0; i < numLoops; ++i) {
    if (iterTypes[i] == utils::IteratorType::parallel)
      parallelDims.push_back(i);
    else
      reductionDims.push_back(i);
  }

  // For each parallel dim, determine if any member uses it as reduction.
  // If so, mark as Mixed.
  llvm::DenseSet<unsigned> reductionUsedDims;
  for (Operation *member : chain.members) {
    auto memberOp = dyn_cast<linalg::LinalgOp>(member);
    if (!memberOp || member == chain.root)
      continue;
    auto memberIters = memberOp.getIteratorTypesArray();
    // Map member dims back to root dims via indexing maps.
    // Simplified: check if same positional dim is used as reduction.
    unsigned memberLoops = memberIters.size();
    for (unsigned i = 0; i < std::min(numLoops, memberLoops); ++i) {
      if (memberIters[i] == utils::IteratorType::reduction)
        reductionUsedDims.insert(i);
    }
  }

  // Build golden dims: parallel first, then reduction.
  for (unsigned i : parallelDims) {
    DimRole role = reductionUsedDims.contains(i) ? DimRole::Mixed
                                                  : DimRole::Parallel;
    // Map root loop dim to output tensor dim via indexing map.
    AffineMap outputMap = rootLinalgOp.getIndexingMapsArray().back();
    DimExpr expr = DimExpr::makeConst(1); // default
    for (unsigned r = 0; r < outputMap.getNumResults(); ++r) {
      auto dimExpr = dyn_cast<AffineDimExpr>(outputMap.getResult(r));
      if (dimExpr && dimExpr.getPosition() == i) {
        expr = getDimExpr(rootOutput, r, func);
        break;
      }
    }
    layout.dims.push_back({role, expr});
  }
  for (unsigned i : reductionDims) {
    AffineMap outputMap = rootLinalgOp.getIndexingMapsArray().back();
    DimExpr expr = DimExpr::makeConst(1);
    // Reduction dims may not appear in output map; check input maps.
    for (unsigned inIdx = 0; inIdx < rootLinalgOp.getNumDpsInputs(); ++inIdx) {
      AffineMap inMap = rootLinalgOp.getIndexingMapsArray()[inIdx];
      Value inTensor = rootLinalgOp.getDpsInputs()[inIdx];
      for (unsigned r = 0; r < inMap.getNumResults(); ++r) {
        auto dimE = dyn_cast<AffineDimExpr>(inMap.getResult(r));
        if (dimE && dimE.getPosition() == i) {
          expr = getDimExpr(inTensor, r, func);
          break;
        }
      }
    }
    layout.dims.push_back({DimRole::Reduction, expr});
  }

  computeCollapse(layout, chain);
  return layout;
}

void mlir::afir::computeCollapse(GoldenAxisLayout &layout,
                                  const VectorChain &chain) {
  // Find groups of contiguous parallel dims that can be collapsed.
  // Rule: adjacent parallel dims that are contiguous on every operand
  // of every member. Reduction dims are never merged with parallel.
  SmallVector<unsigned> currentGroup;
  for (unsigned i = 0; i < layout.dims.size(); ++i) {
    if (layout.dims[i].role == DimRole::Parallel ||
        layout.dims[i].role == DimRole::Mixed) {
      if (!currentGroup.empty() &&
          layout.dims[i].role != layout.dims[currentGroup.back()].role) {
        // Role changed — flush group if it has >1 members.
        if (currentGroup.size() > 1)
          layout.collapseGroups.push_back(currentGroup);
        currentGroup.clear();
      }
      // Only collapse pure parallel dims.
      if (layout.dims[i].role == DimRole::Parallel)
        currentGroup.push_back(i);
      else {
        if (currentGroup.size() > 1)
          layout.collapseGroups.push_back(currentGroup);
        currentGroup.clear();
      }
    } else {
      // Reduction: flush any pending parallel group.
      if (currentGroup.size() > 1)
        layout.collapseGroups.push_back(currentGroup);
      currentGroup.clear();
    }
  }
  if (currentGroup.size() > 1)
    layout.collapseGroups.push_back(currentGroup);

  // Build collapsed dims.
  llvm::DenseSet<unsigned> collapsedInto;
  for (auto &group : layout.collapseGroups)
    for (unsigned i = 1; i < group.size(); ++i)
      collapsedInto.insert(group[i]);

  for (unsigned i = 0; i < layout.dims.size(); ++i) {
    if (collapsedInto.contains(i))
      continue;
    // If this is the start of a collapse group, merge DimExprs.
    bool isGroupStart = false;
    for (auto &group : layout.collapseGroups) {
      if (!group.empty() && group[0] == i) {
        isGroupStart = true;
        DimExpr merged = layout.dims[group[0]].expr;
        for (unsigned j = 1; j < group.size(); ++j)
          merged = DimExpr::makeMul(merged, layout.dims[group[j]].expr);
        layout.collapsedDims.push_back({DimRole::Parallel, merged});
        break;
      }
    }
    if (!isGroupStart)
      layout.collapsedDims.push_back(layout.dims[i]);
  }
}
```

- [ ] **Step 3: Add GoldenAxis.cpp to CMakeLists.txt**

Update `lib/Conversion/VectorPlanGeneration/CMakeLists.txt`:

```cmake
add_mlir_library(VectorPlanGenerationConversion
  VectorPlanGenerationPass.cpp
  ChainAnalysis.cpp
  GoldenAxis.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTransforms
  MLIRTilingInterface
)
```

- [ ] **Step 4: Wire golden axis into the pass**

Update `VectorPlanGenerationPass.cpp` `runOnOperation()` to call `buildGoldenAxis` after chain analysis:

```cpp
    // Steps 3-5: Golden axis + collapse
    for (size_t i = 0; i < chains.size(); ++i) {
      auto &chain = chains[i];
      GoldenAxisLayout layout = buildGoldenAxis(chain, func);

      if (emitRemarks) {
        std::string dimStr;
        for (auto &dim : layout.collapsedDims) {
          if (!dimStr.empty()) dimStr += ", ";
          dimStr += (dim.role == DimRole::Parallel  ? "par" :
                     dim.role == DimRole::Reduction ? "red" : "mixed");
        }
        chain.root->emitRemark()
            << "chain[" << i << "] golden: [" << dimStr << "]"
            << " collapsed_dims: " << layout.collapsedDims.size();
      }
    }
```

Add includes at top:

```cpp
#include "GoldenAxis.h"
```

- [ ] **Step 5: Build and run**

```bash
cmake --build build --target afir-opt && \
build/bin/llvm-lit test/Conversion/vector-plan-chain-analysis.mlir -v
```

- [ ] **Step 6: Commit**

```bash
git add lib/Conversion/VectorPlanGeneration/GoldenAxis.h \
        lib/Conversion/VectorPlanGeneration/GoldenAxis.cpp \
        lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp \
        lib/Conversion/VectorPlanGeneration/CMakeLists.txt
git commit -m "feat: add golden axis construction and collapse (Phase 1 Steps 3-5)"
```

---

### Task 6: SplitTiling + Plan Generation (Steps 6–7)

Implements the deterministic split-tiling rules and plan record generation. After this task, the pass knows exactly which tiling scheme to use for each chain.

**Files:**
- Create: `lib/Conversion/VectorPlanGeneration/PlanGeneration.h`
- Create: `lib/Conversion/VectorPlanGeneration/PlanGeneration.cpp`
- Modify: `lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp`
- Modify: `lib/Conversion/VectorPlanGeneration/CMakeLists.txt`

- [ ] **Step 1: Define plan data structures**

Create `lib/Conversion/VectorPlanGeneration/PlanGeneration.h`:

```cpp
//===- PlanGeneration.h - Split-tiling + plan records ------------*- C++-*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_VECTORPLANGENERATION_PLANGENERATION_H
#define ASCEND_MLIR_VECTORPLANGENERATION_PLANGENERATION_H

#include "GoldenAxis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::afir {

/// Axis assignment for tiling.
struct TilingAssignment {
  /// Index into collapsedDims that becomes the outer parallel split.
  unsigned splitAxisIdx = 0;
  /// Indices into collapsedDims covered by inner tiling.
  SmallVector<unsigned> tilingAxes;
  /// Indices of reduction axes (subset of tilingAxes).
  SmallVector<unsigned> reductionAxes;
};

/// A single plan for one chain.
enum class PlanKind { Default, SubSplit, Flat };

struct TilingPlan {
  PlanKind kind;
  TilingAssignment assignment;
  /// Symbolic tile names in order: e.g. ["XBLOCK", "XBLOCK_SUB", "RBLOCK"]
  SmallVector<std::string> tileNames;
  /// Whether each tile is for a reduction dim.
  SmallVector<bool> tileIsReduction;
};

/// Apply SplitTiling rules (spec §4 Step 6) and generate plans (Step 7).
SmallVector<TilingPlan> generatePlans(const GoldenAxisLayout &layout);

} // namespace mlir::afir

#endif
```

- [ ] **Step 2: Implement plan generation**

Create `lib/Conversion/VectorPlanGeneration/PlanGeneration.cpp`:

```cpp
//===- PlanGeneration.cpp - Split-tiling + plan records -----------*- C++-*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "PlanGeneration.h"

using namespace mlir::afir;

SmallVector<TilingPlan> mlir::afir::generatePlans(
    const GoldenAxisLayout &layout) {
  SmallVector<TilingPlan> plans;
  auto &dims = layout.collapsedDims;

  if (dims.empty())
    return plans;

  // Identify parallel and reduction dims.
  SmallVector<unsigned> parallelDims, reductionDims, mixedDims;
  for (unsigned i = 0; i < dims.size(); ++i) {
    switch (dims[i].role) {
    case DimRole::Parallel:
      parallelDims.push_back(i);
      break;
    case DimRole::Reduction:
      reductionDims.push_back(i);
      break;
    case DimRole::Mixed:
      mixedDims.push_back(i);
      // Mixed dims are treated as reduction for tiling purposes.
      reductionDims.push_back(i);
      break;
    }
  }

  // === Default plan ===
  TilingPlan defaultPlan;
  defaultPlan.kind = PlanKind::Default;

  // split_axis: prefer the outermost (first) parallel dim.
  if (!parallelDims.empty()) {
    defaultPlan.assignment.splitAxisIdx = parallelDims[0];
  } else {
    // Degenerate: no parallel dims. Use dim 0 and emit flat plan instead.
    defaultPlan.kind = PlanKind::Flat;
    defaultPlan.assignment.splitAxisIdx = 0;
  }

  // tiling_axis: all dims except split_axis are tiling axes.
  // Must include stride-1 (innermost) dim and all reduction dims.
  for (unsigned i = 0; i < dims.size(); ++i) {
    defaultPlan.assignment.tilingAxes.push_back(i);
  }
  defaultPlan.assignment.reductionAxes = reductionDims;

  // Tile names: XBLOCK (split), XBLOCK_SUB (inner parallel), RBLOCK (reduction)
  defaultPlan.tileNames.push_back("XBLOCK");
  defaultPlan.tileIsReduction.push_back(false);

  if (defaultPlan.kind != PlanKind::Flat) {
    defaultPlan.tileNames.push_back("XBLOCK_SUB");
    defaultPlan.tileIsReduction.push_back(false);
  }

  if (!reductionDims.empty()) {
    defaultPlan.tileNames.push_back("RBLOCK");
    defaultPlan.tileIsReduction.push_back(true);
  }

  plans.push_back(std::move(defaultPlan));
  return plans;
}
```

- [ ] **Step 3: Wire into pass and add to CMake**

Add `PlanGeneration.cpp` to `CMakeLists.txt`.

Update `VectorPlanGenerationPass.cpp` to call `generatePlans` after golden axis:

```cpp
#include "PlanGeneration.h"

// After golden axis construction:
    SmallVector<TilingPlan> planSet = generatePlans(layout);

    if (emitRemarks) {
      for (auto &plan : planSet) {
        std::string tileStr;
        for (auto &name : plan.tileNames) {
          if (!tileStr.empty()) tileStr += ", ";
          tileStr += name;
        }
        chain.root->emitRemark()
            << "chain[" << i << "] plan: "
            << (plan.kind == PlanKind::Default ? "default" :
                plan.kind == PlanKind::SubSplit ? "sub_split" : "flat")
            << " tiles: [" << tileStr << "]";
      }
    }
```

- [ ] **Step 4: Build and run**

```bash
cmake --build build --target afir-opt && \
build/bin/llvm-lit test/Conversion/vector-plan-chain-analysis.mlir -v
```

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/VectorPlanGeneration/PlanGeneration.h \
        lib/Conversion/VectorPlanGeneration/PlanGeneration.cpp \
        lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp \
        lib/Conversion/VectorPlanGeneration/CMakeLists.txt
git commit -m "feat: add split-tiling rules and plan generation (Phase 1 Steps 6-7)"
```

---

### Task 7: Plan Realization (Step 8)

The load-bearing task: actually transforms the IR by calling `tileConsumerAndFuseProducersUsingSCF`, creating kernel functions with symbolic tile parameters, and annotating loops with `ascendc.parallel`.

**Files:**
- Create: `lib/Conversion/VectorPlanGeneration/PlanRealization.h`
- Create: `lib/Conversion/VectorPlanGeneration/PlanRealization.cpp`
- Modify: `lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp`
- Modify: `lib/Conversion/VectorPlanGeneration/CMakeLists.txt`
- Create: `test/Conversion/vector-plan-generation.mlir`

- [ ] **Step 1: Write the failing integration test**

Create `test/Conversion/vector-plan-generation.mlir`:

```mlir
// RUN: afir-opt --vector-plan-generation %s | FileCheck %s

// Test: a simple bias-add chain should be tiled with XBLOCK / XBLOCK_SUB.
// The pass should produce a kernel function with tile args appended.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @bias_relu
// CHECK: scf.for
// CHECK-SAME: step %{{.*}}
// CHECK: scf.for
// CHECK-SAME: step %{{.*}}
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

- [ ] **Step 2: Define realization interface**

Create `lib/Conversion/VectorPlanGeneration/PlanRealization.h`:

```cpp
//===- PlanRealization.h - Realize plans via tile-and-fuse --------*- C++-*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_VECTORPLANGENERATION_PLANREALIZATION_H
#define ASCEND_MLIR_VECTORPLANGENERATION_PLANREALIZATION_H

#include "ChainAnalysis.h"
#include "GoldenAxis.h"
#include "PlanGeneration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::afir {

/// Realize a single plan for a chain (spec §4 Step 8).
///
/// This function:
/// 1. Collapses iteration dims via collapseOpIterationDims.
/// 2. Appends tile index arguments to the function.
/// 3. Calls tileConsumerAndFuseProducersUsingSCF on the chain root.
/// 4. Annotates the outer loop with ascendc.parallel.
///
/// Returns failure if tiling cannot be applied.
LogicalResult realizePlan(func::FuncOp func,
                          VectorChain &chain,
                          const GoldenAxisLayout &layout,
                          const TilingPlan &plan,
                          IRRewriter &rewriter);

} // namespace mlir::afir

#endif
```

- [ ] **Step 3: Implement plan realization**

Create `lib/Conversion/VectorPlanGeneration/PlanRealization.cpp`:

```cpp
//===- PlanRealization.cpp - Realize plans via tile-and-fuse ------*- C++-*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "PlanRealization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::afir;

/// Append index arguments to the function for each tile parameter.
/// Returns the SSA values of the new arguments.
static SmallVector<Value> appendTileArgs(func::FuncOp func,
                                         const TilingPlan &plan,
                                         IRRewriter &rewriter) {
  SmallVector<Value> tileValues;
  unsigned numExisting = func.getNumArguments();
  Location loc = func.getLoc();

  for (size_t i = 0; i < plan.tileNames.size(); ++i) {
    func.insertArgument(numExisting + i,
                        rewriter.getIndexType(),
                        /*attrs=*/{},
                        loc);
    tileValues.push_back(func.getArgument(numExisting + i));
  }

  return tileValues;
}

LogicalResult mlir::afir::realizePlan(func::FuncOp func,
                                       VectorChain &chain,
                                       const GoldenAxisLayout &layout,
                                       const TilingPlan &plan,
                                       IRRewriter &rewriter) {
  auto tilingInterface = dyn_cast<TilingInterface>(chain.root);
  if (!tilingInterface)
    return chain.root->emitError("root does not implement TilingInterface");

  // Step 1: Append tile arguments to the function.
  SmallVector<Value> tileArgs = appendTileArgs(func, plan, rewriter);
  if (tileArgs.empty())
    return failure();

  // Step 2: Build tile sizes.
  // For the default plan: [XBLOCK, XBLOCK_SUB] or [XBLOCK, XBLOCK_SUB, RBLOCK]
  unsigned numLoops =
      cast<TilingInterface>(chain.root).getLoopIteratorTypes().size();
  SmallVector<OpFoldResult> tileSizes(numLoops, rewriter.getIndexAttr(0));

  // Map tile args to loop dims:
  //   XBLOCK → outermost parallel dim (split)
  //   XBLOCK_SUB → same dim (inner)
  //   RBLOCK → reduction dim
  if (tileArgs.size() >= 1 && plan.assignment.splitAxisIdx < numLoops)
    tileSizes[plan.assignment.splitAxisIdx] = tileArgs[0];

  // Step 3: Call tileUsingSCF on the root.
  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizes);

  rewriter.setInsertionPoint(chain.root);
  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCF(rewriter, tilingInterface, tilingOptions);

  if (failed(tilingResult))
    return chain.root->emitError("tileUsingSCF failed");

  // Replace uses of the original root results.
  rewriter.replaceOp(chain.root, tilingResult->replacements);

  // Step 4: Annotate outer loop with ascendc.parallel.
  if (!tilingResult->loops.empty()) {
    tilingResult->loops[0]->setAttr(
        "ascendc.parallel",
        rewriter.getBoolAttr(true));
  }

  return success();
}
```

- [ ] **Step 4: Wire realization into the pass**

Update `VectorPlanGenerationPass.cpp`:

```cpp
#include "PlanRealization.h"

    // Step 8: Realize plans
    IRRewriter rewriter(&getContext());
    for (size_t i = 0; i < chains.size(); ++i) {
      auto &chain = chains[i];
      GoldenAxisLayout layout = buildGoldenAxis(chain, func);
      SmallVector<TilingPlan> planSet = generatePlans(layout);

      if (planSet.empty())
        continue;

      // Realize the default plan.
      if (failed(realizePlan(func, chain, layout, planSet[0], rewriter))) {
        chain.root->emitWarning("failed to realize plan for chain");
        continue;
      }
    }
```

- [ ] **Step 5: Add PlanRealization.cpp to CMake**

Update `CMakeLists.txt` to add `PlanRealization.cpp`.

- [ ] **Step 6: Build and run**

```bash
cmake --build build --target afir-opt && \
build/bin/llvm-lit test/Conversion/vector-plan-generation.mlir -v
```

Expected: PASS — the bias+relu chain is tiled with scf.for loops and tile args are appended.

- [ ] **Step 7: Commit**

```bash
git add lib/Conversion/VectorPlanGeneration/PlanRealization.h \
        lib/Conversion/VectorPlanGeneration/PlanRealization.cpp \
        lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp \
        lib/Conversion/VectorPlanGeneration/CMakeLists.txt \
        test/Conversion/vector-plan-generation.mlir
git commit -m "feat: add plan realization via tileUsingSCF (Phase 1 Step 8)"
```

---

### Task 8: LayerNorm Integration Test

End-to-end test with the LayerNorm pattern from `transformer_dynamic.mlir` — the canonical stress test from the spec (§5). Validates chain analysis handles multi-use, reduction, and broadcast correctly.

**Files:**
- Create: `test/Conversion/vector-plan-layernorm.mlir`

- [ ] **Step 1: Write the LayerNorm integration test**

Create `test/Conversion/vector-plan-layernorm.mlir`:

```mlir
// RUN: afir-opt --eliminate-cf-assert \
// RUN:          --canonicalize-extract-broadcast \
// RUN:          --linalg-fold-unit-extent-dims \
// RUN:          --linalg-fuse-elementwise-ops \
// RUN:          --vector-plan-generation="emit-remarks=true" %s 2>&1 \
// RUN:   | FileCheck %s

// This is the LayerNorm pattern from transformer_dynamic.mlir (spec §5).
// The full chain (%add ... %add_beta) should form one Vector chain
// with the final +beta as root.

#map_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_reduce = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map_bcast = affine_map<(d0, d1, d2) -> (d2)>

// CHECK: remark: chain{{.*}}root: {{.*}}addf
// CHECK-SAME: members:
// CHECK: scf.for
func.func @layernorm_pattern(
    %input: tensor<2x8x128xf32>,
    %residual: tensor<2x8x128xf32>,
    %gamma: tensor<128xf32>,
    %beta: tensor<128xf32>
) -> tensor<2x8x128xf32> {
  %cst_zero = arith.constant 0.0 : f32
  %cst_H_inv = arith.constant 7.8125e-03 : f32
  %cst_eps = arith.constant 1.0e-05 : f32
  %empty3d = tensor.empty() : tensor<2x8x128xf32>
  %empty_red = tensor.empty() : tensor<2x8x1xf32>
  %init_red = linalg.fill ins(%cst_zero : f32)
      outs(%empty_red : tensor<2x8x1xf32>) -> tensor<2x8x1xf32>

  // residual add
  %add = linalg.generic {
      indexing_maps = [#map_3d, #map_3d, #map_3d],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input, %residual : tensor<2x8x128xf32>, tensor<2x8x128xf32>)
      outs(%empty3d : tensor<2x8x128xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %r = arith.addf %a, %b : f32
      linalg.yield %r : f32
  } -> tensor<2x8x128xf32>

  // mean reduce
  %mean_sum = linalg.generic {
      indexing_maps = [#map_3d, #map_reduce],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%add : tensor<2x8x128xf32>)
      outs(%init_red : tensor<2x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %r = arith.addf %in, %out : f32
      linalg.yield %r : f32
  } -> tensor<2x8x1xf32>

  // /H
  %mean = linalg.generic {
      indexing_maps = [#map_reduce, #map_reduce],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%mean_sum : tensor<2x8x1xf32>)
      outs(%empty_red : tensor<2x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %r = arith.mulf %in, %cst_H_inv : f32
      linalg.yield %r : f32
  } -> tensor<2x8x1xf32>

  // broadcast mean + sub
  %sub = linalg.generic {
      indexing_maps = [#map_3d, #map_reduce, #map_3d],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%add, %mean : tensor<2x8x128xf32>, tensor<2x8x1xf32>)
      outs(%empty3d : tensor<2x8x128xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %r = arith.subf %a, %b : f32
      linalg.yield %r : f32
  } -> tensor<2x8x128xf32>

  // square
  %sq = linalg.generic {
      indexing_maps = [#map_3d, #map_3d, #map_3d],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%sub, %sub : tensor<2x8x128xf32>, tensor<2x8x128xf32>)
      outs(%empty3d : tensor<2x8x128xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %r = arith.mulf %a, %b : f32
      linalg.yield %r : f32
  } -> tensor<2x8x128xf32>

  // var reduce
  %var_sum = linalg.generic {
      indexing_maps = [#map_3d, #map_reduce],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%sq : tensor<2x8x128xf32>)
      outs(%init_red : tensor<2x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %r = arith.addf %in, %out : f32
      linalg.yield %r : f32
  } -> tensor<2x8x1xf32>

  // /H + eps + rsqrt
  %rstd = linalg.generic {
      indexing_maps = [#map_reduce, #map_reduce],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%var_sum : tensor<2x8x1xf32>)
      outs(%empty_red : tensor<2x8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %cst_H_inv : f32
      %ve = arith.addf %v, %cst_eps : f32
      %r = math.rsqrt %ve : f32
      linalg.yield %r : f32
  } -> tensor<2x8x1xf32>

  // normalize: (x - mean) * rstd
  %norm = linalg.generic {
      indexing_maps = [#map_3d, #map_reduce, #map_3d],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%sub, %rstd : tensor<2x8x128xf32>, tensor<2x8x1xf32>)
      outs(%empty3d : tensor<2x8x128xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %r = arith.mulf %a, %b : f32
      linalg.yield %r : f32
  } -> tensor<2x8x128xf32>

  // * gamma
  %scaled = linalg.generic {
      indexing_maps = [#map_3d, #map_bcast, #map_3d],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%norm, %gamma : tensor<2x8x128xf32>, tensor<128xf32>)
      outs(%empty3d : tensor<2x8x128xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %r = arith.mulf %a, %b : f32
      linalg.yield %r : f32
  } -> tensor<2x8x128xf32>

  // + beta
  %result = linalg.generic {
      indexing_maps = [#map_3d, #map_bcast, #map_3d],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%scaled, %beta : tensor<2x8x128xf32>, tensor<128xf32>)
      outs(%empty3d : tensor<2x8x128xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %r = arith.addf %a, %b : f32
      linalg.yield %r : f32
  } -> tensor<2x8x128xf32>

  return %result : tensor<2x8x128xf32>
}
```

- [ ] **Step 2: Run the full pipeline**

```bash
cmake --build build --target afir-opt && \
build/bin/llvm-lit test/Conversion/vector-plan-layernorm.mlir -v
```

Expected: PASS — the LayerNorm chain is detected as one chain with +beta as root, and tiling loops are generated.

- [ ] **Step 3: Debug and fix any issues**

If the test fails, investigate chain analysis boundaries and fix. Common issues:
- `linalg.fill` being treated as chain member (should be excluded)
- Multi-use detection not working for the `%add` / `%sub` nodes
- Golden axis mismatch when reduction and parallel dims coexist

- [ ] **Step 4: Commit**

```bash
git add test/Conversion/vector-plan-layernorm.mlir
git commit -m "test: add LayerNorm integration test for vector-plan-generation"
```

---

### Task 9: Module-Level Tiling Metadata

Add the `tiling.tiles` and `tiling.shapes` module-level attributes described in spec §3. These are consumed by Phase 3 (autotuner) and the host-side dispatch.

**Files:**
- Modify: `lib/Conversion/VectorPlanGeneration/PlanRealization.cpp`
- Modify: `test/Conversion/vector-plan-generation.mlir`

- [ ] **Step 1: Add metadata emission to PlanRealization**

After plan realization succeeds, stamp the module with tiling metadata:

```cpp
// In realizePlan(), after successful tiling:

// Stamp tiling.tiles on the module.
if (auto moduleOp = func->getParentOfType<ModuleOp>()) {
  SmallVector<Attribute> tileNames;
  for (auto &name : plan.tileNames)
    tileNames.push_back(rewriter.getStringAttr(name));
  moduleOp->setAttr("tiling.tiles",
                     rewriter.getArrayAttr(tileNames));
}
```

- [ ] **Step 2: Update test to check metadata**

Add to `test/Conversion/vector-plan-generation.mlir`:

```mlir
// CHECK: module attributes
// CHECK-SAME: tiling.tiles = ["XBLOCK", "XBLOCK_SUB"
```

- [ ] **Step 3: Build and run**

```bash
cmake --build build --target afir-opt && \
build/bin/llvm-lit test/Conversion/vector-plan-generation.mlir -v
```

- [ ] **Step 4: Commit**

```bash
git add lib/Conversion/VectorPlanGeneration/PlanRealization.cpp \
        test/Conversion/vector-plan-generation.mlir
git commit -m "feat: emit tiling.tiles module metadata (spec §3)"
```

---

## Dependency Graph

```
Task 1 (EliminateCfAssert build integration)
    └─ independent, can start immediately

Task 2 (CanonicalizeExtractBroadcast)
    └─ independent, can start immediately

Task 3 (Chain Analysis)
    └─ independent data structures

Task 4 (Pass Skeleton + Chain Diagnostic)
    └─ depends on Task 3

Task 5 (Golden Axis + Collapse)
    └─ depends on Task 3

Task 6 (SplitTiling + Plan Generation)
    └─ depends on Task 5

Task 7 (Plan Realization)
    └─ depends on Tasks 4, 6

Task 8 (LayerNorm Integration Test)
    └─ depends on Tasks 1, 2, 7

Task 9 (Module Tiling Metadata)
    └─ depends on Task 7
```

Tasks 1, 2, 3 can be parallelized. Tasks 4+5 can proceed after 3. Task 7 is the critical path.
