# Ascend MLIR V2 Kernelize Candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `--ascend-kernelize` from MVP single-op marking to the V2-3 Kernelize candidate analysis pipeline that produces deterministic `KernelPattern` annotations.

**Architecture:** Keep all implementation under local `Conversion/AscendV2/Kernelize` files and do not modify upstream MLIR. Split the current monolithic pass into analysis data types, dependency analysis, structural marking, role classification, candidate generation, merge analysis, horizontal fusion, pattern graph construction, and final partitioning. Preserve the existing MVP attrs consumed by `--ascend-schedule` while adding richer debug output and verifier hooks.

**Tech Stack:** C++17, MLIR IR/Linalg APIs, LLVM ADT, lit/FileCheck, xvm/docker build path `/home/niu/code/Ascend-MLIR`.

---

## Scope

This plan implements Phase 1 from `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`.

In scope:

- V2-3 `DependencyAnalyzer`, `StructuralMarker`, `OpRoleClassifier`.
- Primitive-driven single-primary candidate generation.
- `CandidateClosure`, legality/profitability filtering, deterministic ordering.
- `CandidateMergeAnalyzer`, `HorizontalFusionAnalyzer`.
- `KernelPatternGraph` and `KernelPartitioner`.
- Debug reports and focused lit tests.

Out of scope:

- Schedule search internals from V2-4.
- Target cost model from V2-Target/V2-8.
- Realize/Translate lowering from V2-5/V2-6.
- Upstream MLIR source changes.

## File Structure

- Create: `include/Conversion/AscendV2/Kernelize/KernelizeTypes.h`
  - Shared enums, attr names, IDs, candidate structs, graph structs, and formatting helpers.
- Create: `include/Conversion/AscendV2/Kernelize/DependencyAnalysis.h`
  - Public interface for building `ProducerConsumerIndex`, `OpSemanticSummary`, and stable op order.
- Create: `lib/Conversion/AscendV2/Kernelize/DependencyAnalysis.cpp`
  - Use-def graph scan, linalg semantic extraction, op numbering, one-hop producer/consumer edges.
- Create: `include/Conversion/AscendV2/Kernelize/StructuralMarking.h`
  - Public interface for branch/merge and handwritten-pattern structural marks.
- Create: `lib/Conversion/AscendV2/Kernelize/StructuralMarking.cpp`
  - Attribute attachment for structural groups with deterministic IDs.
- Create: `include/Conversion/AscendV2/Kernelize/OpRoleClassification.h`
  - Public interface for `OpRoleMap` construction.
- Create: `lib/Conversion/AscendV2/Kernelize/OpRoleClassification.cpp`
  - Full V2 role derivation from summaries and structural attrs.
- Create: `include/Conversion/AscendV2/Kernelize/FusionCandidateAnalysis.h`
  - Primitive registry, candidate config, `FusionCandidateAnalyzer`.
- Create: `lib/Conversion/AscendV2/Kernelize/FusionCandidateAnalysis.cpp`
  - `ElementwiseChain`, `ConsumerIntoPrimary`, `ReductionInlining`, `FallbackSingleOp` primitives.
- Create: `include/Conversion/AscendV2/Kernelize/CandidateClosure.h`
  - Closure recomputation and validation API.
- Create: `lib/Conversion/AscendV2/Kernelize/CandidateClosure.cpp`
  - Boundary value extraction, closedness checks, candidate failure reasons.
- Create: `include/Conversion/AscendV2/Kernelize/CandidateMergeAnalysis.h`
  - Adjacency index and merged candidate API.
- Create: `lib/Conversion/AscendV2/Kernelize/CandidateMergeAnalysis.cpp`
  - Candidate adjacency, merge prefilters, contract-compatible merge.
- Create: `include/Conversion/AscendV2/Kernelize/HorizontalFusionAnalysis.h`
  - Horizontal fusion candidate API.
- Create: `lib/Conversion/AscendV2/Kernelize/HorizontalFusionAnalysis.cpp`
  - Shared-input sibling grouping and V2-3 initial constraints.
- Create: `include/Conversion/AscendV2/Kernelize/KernelPattern.h`
  - `KernelPatternCandidate`, `KernelPatternGraph`, and final `KernelPattern` API.
- Create: `lib/Conversion/AscendV2/Kernelize/KernelPattern.cpp`
  - Candidate graph construction, overlap map, dependency edges, partitioner.
- Modify: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
  - Orchestrate the seven V2-3 stages and emit attrs/report.
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
  - Add the new source files.
- Modify: `test/Conversion/ascend-kernelize-mvp.mlir`
  - Keep MVP behavior stable where downstream schedule tests depend on it.
- Create: `test/Conversion/ascend-kernelize-dependency.mlir`
  - Dependency and semantic summary coverage.
- Create: `test/Conversion/ascend-kernelize-roles.mlir`
  - Role classification and structural marker coverage.
- Create: `test/Conversion/ascend-kernelize-candidates.mlir`
  - Primitive seed/expand, closure, filtering, and stable ordering coverage.
- Create: `test/Conversion/ascend-kernelize-merge-horizontal.mlir`
  - Candidate merge and horizontal fusion coverage.
- Create: `test/Conversion/ascend-kernelize-patterns.mlir`
  - Kernel pattern graph and partition coverage.
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
  - Mark Phase 1 as in progress when execution begins and record verification.

## Common Commands

Host static checks:

```bash
git diff --check
```

xvm/docker build and focused tests:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v \
  build-v2-verify/test/Conversion/ascend-kernelize-dependency.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-roles.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-candidates.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-merge-horizontal.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-patterns.mlir \
  build-v2-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'
```

Regression:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target check-afir -j10'
```

## Task 1: Shared Kernelize Data Model

**Files:**

- Create: `include/Conversion/AscendV2/Kernelize/KernelizeTypes.h`
- Modify: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
- Test: existing `test/Conversion/ascend-kernelize-mvp.mlir`

- [ ] **Step 1: Move shared attr names and enums into `KernelizeTypes.h`**

Add this header:

```cpp
//===- KernelizeTypes.h - Ascend V2 kernelize data model --------*- C++ -*-===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_TYPES_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_TYPES_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringLiteral.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::afir::ascend::v2::kernelize {

inline constexpr llvm::StringLiteral kNormalizedAttr = "ascend.v2.normalized";
inline constexpr llvm::StringLiteral kOpRoleAttr = "ascend.v2.op_role";
inline constexpr llvm::StringLiteral kOpRolesAttr = "ascend.v2.op_roles";
inline constexpr llvm::StringLiteral kKernelAttr = "ascend.v2.kernel";
inline constexpr llvm::StringLiteral kPrimaryAttr = "ascend.v2.primary";
inline constexpr llvm::StringLiteral kBranchRootAttr = "ascend.v2.branch_root";
inline constexpr llvm::StringLiteral kBranchGroupAttr = "ascend.v2.branch_group";
inline constexpr llvm::StringLiteral kMergeRootAttr = "ascend.v2.merge_root";
inline constexpr llvm::StringLiteral kMergeGroupAttr = "ascend.v2.merge_group";

enum class AccessPatternKind {
  NotApplicable,
  Elementwise,
  Broadcast,
  Reduction,
  Contraction,
  Gather,
  Scatter,
  LayoutTransform,
  Unknown
};

enum class OpRole {
  Primary,
  Cube,
  Vector,
  Reduction,
  Injective,
  Indexing,
  LayoutTransform,
  Branch,
  Merge,
  Barrier,
  Unsupported
};

enum class CandidateKind {
  Fusion,
  Merged,
  HorizontalFusion,
  FallbackSingleOp,
  HandwrittenPattern
};

enum class KernelPatternEdgeKind {
  DataDependency,
  Overlap,
  MustCoLocate,
  MustSeparate,
  ScheduleBarrier
};

struct OperationId {
  unsigned value = 0;
};

struct KernelizeConfig {
  unsigned maxPrimitivePerOp = 4;
  unsigned maxOpsPerCandidate = 32;
  unsigned maxBranchesPerCandidate = 4;
  unsigned maxPrimaryRolesPerCandidate = 2;
  unsigned maxHorizontalFusionGroupSize = 8;
  unsigned localTopKPerPrimaryOpNeighborhood = 8;
};

inline llvm::StringRef stringifyAccessPattern(AccessPatternKind kind) {
  switch (kind) {
  case AccessPatternKind::NotApplicable:
    return "NotApplicable";
  case AccessPatternKind::Elementwise:
    return "Elementwise";
  case AccessPatternKind::Broadcast:
    return "Broadcast";
  case AccessPatternKind::Reduction:
    return "Reduction";
  case AccessPatternKind::Contraction:
    return "Contraction";
  case AccessPatternKind::Gather:
    return "Gather";
  case AccessPatternKind::Scatter:
    return "Scatter";
  case AccessPatternKind::LayoutTransform:
    return "LayoutTransform";
  case AccessPatternKind::Unknown:
    return "Unknown";
  }
  return "Unknown";
}

inline llvm::StringRef stringifyOpRole(OpRole role) {
  switch (role) {
  case OpRole::Primary:
    return "Primary";
  case OpRole::Cube:
    return "Cube";
  case OpRole::Vector:
    return "Vector";
  case OpRole::Reduction:
    return "Reduction";
  case OpRole::Injective:
    return "Injective";
  case OpRole::Indexing:
    return "Indexing";
  case OpRole::LayoutTransform:
    return "LayoutTransform";
  case OpRole::Branch:
    return "Branch";
  case OpRole::Merge:
    return "Merge";
  case OpRole::Barrier:
    return "Barrier";
  case OpRole::Unsupported:
    return "Unsupported";
  }
  return "Unsupported";
}

inline llvm::StringRef stringifyCandidateKind(CandidateKind kind) {
  switch (kind) {
  case CandidateKind::Fusion:
    return "Fusion";
  case CandidateKind::Merged:
    return "Merged";
  case CandidateKind::HorizontalFusion:
    return "HorizontalFusion";
  case CandidateKind::FallbackSingleOp:
    return "FallbackSingleOp";
  case CandidateKind::HandwrittenPattern:
    return "HandwrittenPattern";
  }
  return "Fusion";
}

inline llvm::StringRef stringifyKernelPatternEdgeKind(KernelPatternEdgeKind kind) {
  switch (kind) {
  case KernelPatternEdgeKind::DataDependency:
    return "DataDependency";
  case KernelPatternEdgeKind::Overlap:
    return "Overlap";
  case KernelPatternEdgeKind::MustCoLocate:
    return "MustCoLocate";
  case KernelPatternEdgeKind::MustSeparate:
    return "MustSeparate";
  case KernelPatternEdgeKind::ScheduleBarrier:
    return "ScheduleBarrier";
  }
  return "DataDependency";
}

} // namespace mlir::afir::ascend::v2::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_TYPES_H
```

- [ ] **Step 2: Update `KernelizePass.cpp` to include the shared header**

Replace local constants with:

```cpp
#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"

using namespace mlir::afir::ascend::v2::kernelize;
```

All references to `kNormalizedAttr`, `kOpRoleAttr`, `kKernelAttr`, and `kPrimaryAttr` should compile through the shared header.

- [ ] **Step 3: Run focused MVP test to confirm no behavior change**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-verify/test/Conversion/ascend-kernelize-mvp.mlir'
```

Expected: `ascend-kernelize-mvp.mlir` passes and still emits `ascend.v2.op_role`, `ascend.v2.kernel`, and `ascend.v2.primary`.

- [ ] **Step 4: Commit**

```bash
git add include/Conversion/AscendV2/Kernelize/KernelizeTypes.h \
  lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp
git commit -m "conversion: add Ascend V2 kernelize data model"
```

## Task 2: Dependency Analysis and Semantic Summary

**Files:**

- Create: `include/Conversion/AscendV2/Kernelize/DependencyAnalysis.h`
- Create: `lib/Conversion/AscendV2/Kernelize/DependencyAnalysis.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
- Modify: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
- Create: `test/Conversion/ascend-kernelize-dependency.mlir`

- [ ] **Step 1: Write failing dependency lit test**

Create `test/Conversion/ascend-kernelize-dependency.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @dependency_chain(%a: tensor<16xf32>, %b: tensor<16xf32>, %c: tensor<16xf32>)
    -> tensor<16xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
  %1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%0 : tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      %scale = arith.mulf %x, %x : f32
      linalg.yield %scale : f32
    } -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// CHECK: Kernelize report
// CHECK: DependencyAnalysis
// CHECK: op_id = 0
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "Elementwise"
// CHECK-SAME: producers = 0
// CHECK-SAME: consumers = 1
// CHECK: op_id = 1
// CHECK-SAME: op = "linalg.generic"
// CHECK-SAME: access = "Elementwise"
// CHECK-SAME: producers = 1
// CHECK-SAME: consumers = 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-verify/test/Conversion/ascend-kernelize-dependency.mlir'
```

Expected: FAIL because `DependencyAnalysis` report does not exist.

- [ ] **Step 3: Add dependency analysis interfaces**

Add `DependencyAnalysis.h`:

```cpp
//===- DependencyAnalysis.h - Ascend V2 dependency analysis -----*- C++ -*-===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_DEPENDENCY_ANALYSIS_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_DEPENDENCY_ANALYSIS_H

#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::v2::kernelize {

struct OpSemanticSummary {
  Operation *op = nullptr;
  OperationId opId;
  AccessPatternKind accessPattern = AccessPatternKind::Unknown;
  SmallVector<StringRef> iteratorTypes;
  unsigned resultRank = 0;
  bool hasReductionIterator = false;
  bool hasOnlyParallelIterators = false;
};

struct ProducerConsumerIndex {
  SmallVector<Operation *> orderedOps;
  DenseMap<Operation *, OperationId> opIds;
  DenseMap<Operation *, SmallVector<Operation *>> producers;
  DenseMap<Operation *, SmallVector<Operation *>> consumers;
};

struct DependencyAnalysisResult {
  ProducerConsumerIndex index;
  DenseMap<Operation *, OpSemanticSummary> summaries;
};

class DependencyAnalyzer {
public:
  FailureOr<DependencyAnalysisResult> analyze(ModuleOp module) const;
};

void emitDependencyAnalysisReport(raw_ostream &os,
                                  const DependencyAnalysisResult &result);

} // namespace mlir::afir::ascend::v2::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_DEPENDENCY_ANALYSIS_H
```

- [ ] **Step 4: Implement dependency analysis**

Implement these rules in `DependencyAnalysis.cpp`:

```cpp
FailureOr<DependencyAnalysisResult> DependencyAnalyzer::analyze(ModuleOp module) const {
  DependencyAnalysisResult result;

  module.walk([&](Operation *op) {
    if (!isKernelizeTargetOp(op))
      return;
    OperationId id{static_cast<unsigned>(result.index.orderedOps.size())};
    result.index.orderedOps.push_back(op);
    result.index.opIds.try_emplace(op, id);
    result.summaries.try_emplace(op, buildSemanticSummary(op, id));
  });

  for (Operation *consumer : result.index.orderedOps) {
    for (Value operand : consumer->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (!producer || !result.index.opIds.contains(producer))
        continue;
      result.index.producers[consumer].push_back(producer);
      result.index.consumers[producer].push_back(consumer);
    }
  }

  sortAndUniqueEdges(result.index);
  return result;
}
```

`isKernelizeTargetOp` must accept `linalg.generic`, `linalg.matmul`, and `linalg.batch_matmul`, and must reject `linalg.yield` and `linalg.index`.

`buildSemanticSummary` must classify:

- `linalg.matmul`, `linalg.batch_matmul` as `Contraction`.
- `linalg.generic` with only parallel iterators and identity maps as `Elementwise`.
- `linalg.generic` with at least one reduction iterator as `Reduction`.
- `linalg.generic` with projected/broadcast indexing as `Broadcast`.
- Any other supported linalg op as `Unknown`.

- [ ] **Step 5: Wire report into `KernelizePass.cpp`**

At the start of the pass after normalized checks:

```cpp
FailureOr<DependencyAnalysisResult> depResult =
    DependencyAnalyzer().analyze(module);
if (failed(depResult)) {
  signalPassFailure();
  return;
}

if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Kernelize))
  emitDependencyAnalysisReport(llvm::errs(), *depResult);
```

- [ ] **Step 6: Build and run dependency test**

Run the xvm focused command from Step 2.

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add include/Conversion/AscendV2/Kernelize/DependencyAnalysis.h \
  lib/Conversion/AscendV2/Kernelize/DependencyAnalysis.cpp \
  lib/Conversion/AscendV2/CMakeLists.txt \
  lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp \
  test/Conversion/ascend-kernelize-dependency.mlir
git commit -m "conversion: add Ascend V2 dependency analysis"
```

## Task 3: Structural Marking

**Files:**

- Create: `include/Conversion/AscendV2/Kernelize/StructuralMarking.h`
- Create: `lib/Conversion/AscendV2/Kernelize/StructuralMarking.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
- Modify: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
- Create: `test/Conversion/ascend-kernelize-roles.mlir`

- [ ] **Step 1: Write failing structural marker coverage**

Create the first half of `test/Conversion/ascend-kernelize-roles.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @branch_merge(%a: tensor<16xf32>, %b: tensor<16xf32>, %c: tensor<16xf32>)
    -> tensor<16xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%a : tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      linalg.yield %x : f32
    } -> tensor<16xf32>
  %1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%0 : tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      %v = arith.addf %x, %x : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  %2 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%0 : tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      %v = arith.mulf %x, %x : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  %3 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%1, %2 : tensor<16xf32>, tensor<16xf32>)
      outs(%b : tensor<16xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %v = arith.addf %x, %y : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  return %3 : tensor<16xf32>
}

// CHECK: StructuralMarking
// CHECK: op_id = 0
// CHECK-SAME: branch_root = true
// CHECK-SAME: branch_group = 0
// CHECK: op_id = 3
// CHECK-SAME: merge_root = true
// CHECK-SAME: merge_group = 0
```

- [ ] **Step 2: Add structural marker interface**

Add `StructuralMarking.h`:

```cpp
class StructuralMarker {
public:
  LogicalResult mark(ModuleOp module, const DependencyAnalysisResult &deps) const;
};

void emitStructuralMarkingReport(raw_ostream &os,
                                 const DependencyAnalysisResult &deps);
```

- [ ] **Step 3: Implement deterministic branch/merge marks**

Rules:

- Branch root: an analyzed op with at least two analyzed direct consumers.
- Branch group: assigned from branch roots in `orderedOps` order.
- Merge root: an analyzed op with at least two analyzed direct producers.
- Merge group: assigned from merge roots in `orderedOps` order.
- Attach `BoolAttr` and `I64IntegerAttr` using attr names from `KernelizeTypes.h`.

Implementation shape:

```cpp
LogicalResult StructuralMarker::mark(ModuleOp module,
                                     const DependencyAnalysisResult &deps) const {
  MLIRContext *ctx = module.getContext();
  unsigned nextBranchGroup = 0;
  unsigned nextMergeGroup = 0;

  for (Operation *op : deps.index.orderedOps) {
    if (deps.index.consumers.lookup(op).size() >= 2) {
      op->setAttr(kBranchRootAttr, BoolAttr::get(ctx, true));
      op->setAttr(kBranchGroupAttr,
                  IntegerAttr::get(IntegerType::get(ctx, 64), nextBranchGroup++));
    }
    if (deps.index.producers.lookup(op).size() >= 2) {
      op->setAttr(kMergeRootAttr, BoolAttr::get(ctx, true));
      op->setAttr(kMergeGroupAttr,
                  IntegerAttr::get(IntegerType::get(ctx, 64), nextMergeGroup++));
    }
  }
  return success();
}
```

- [ ] **Step 4: Wire marker after dependency analysis**

In `KernelizePass.cpp`:

```cpp
if (failed(StructuralMarker().mark(module, *depResult))) {
  signalPassFailure();
  return;
}
if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Kernelize))
  emitStructuralMarkingReport(llvm::errs(), *depResult);
```

- [ ] **Step 5: Run role test to verify structural section passes**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-verify/test/Conversion/ascend-kernelize-roles.mlir'
```

Expected: PASS for structural marker checks.

- [ ] **Step 6: Commit**

```bash
git add include/Conversion/AscendV2/Kernelize/StructuralMarking.h \
  lib/Conversion/AscendV2/Kernelize/StructuralMarking.cpp \
  lib/Conversion/AscendV2/CMakeLists.txt \
  lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp \
  test/Conversion/ascend-kernelize-roles.mlir
git commit -m "conversion: add Ascend V2 structural marking"
```

## Task 4: Full OpRole Classification

**Files:**

- Create: `include/Conversion/AscendV2/Kernelize/OpRoleClassification.h`
- Create: `lib/Conversion/AscendV2/Kernelize/OpRoleClassification.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
- Modify: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
- Modify: `test/Conversion/ascend-kernelize-roles.mlir`
- Modify: `test/Conversion/ascend-kernelize-mvp.mlir`

- [ ] **Step 1: Extend role lit test**

Append checks to `test/Conversion/ascend-kernelize-roles.mlir`:

```mlir
// CHECK: OpRoleClassification
// CHECK: op_id = 0
// CHECK-SAME: roles = ["Primary", "Vector", "Injective", "Branch"]
// CHECK: op_id = 3
// CHECK-SAME: roles = ["Primary", "Vector", "Injective", "Merge"]
```

Add a second function for reduction:

```mlir
func.func @reduction(%a: tensor<16xf32>, %init: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
      ins(%a : tensor<16xf32>)
      outs(%init : tensor<f32>) {
    ^bb0(%x: f32, %acc: f32):
      %sum = arith.addf %x, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK: roles = ["Primary", "Reduction"]
```

- [ ] **Step 2: Add role classifier interface**

Add `OpRoleClassification.h`:

```cpp
using OpRoleList = SmallVector<OpRole, 4>;
using OpRoleMap = DenseMap<Operation *, OpRoleList>;

class OpRoleClassifier {
public:
  FailureOr<OpRoleMap> classify(const DependencyAnalysisResult &deps) const;
};

void attachRoleAttributes(ModuleOp module, const OpRoleMap &roleMap);
void emitOpRoleClassificationReport(raw_ostream &os,
                                     const DependencyAnalysisResult &deps,
                                     const OpRoleMap &roleMap);
```

- [ ] **Step 3: Implement role derivation**

Rules:

- `Contraction` summary -> `Primary`, `Cube`.
- `Reduction` summary -> `Primary`, `Reduction`.
- `Elementwise` or `Broadcast` summary -> `Primary`, `Vector`, `Injective`.
- `Gather` summary -> `Indexing`.
- `LayoutTransform` summary -> `LayoutTransform`.
- `branch_root` attr -> append `Branch`.
- `merge_root` attr -> append `Merge`.
- `Unknown` summary -> `Unsupported`.

Maintain deterministic order using this priority:

```cpp
static constexpr OpRole kRolePriority[] = {
    OpRole::Primary, OpRole::Cube, OpRole::Vector, OpRole::Reduction,
    OpRole::Injective, OpRole::Indexing, OpRole::LayoutTransform,
    OpRole::Branch, OpRole::Merge, OpRole::Barrier, OpRole::Unsupported};
```

Attach attrs:

- `ascend.v2.op_roles` as an array of string attrs.
- `ascend.v2.op_role` as the MVP-compatible schedule role:
  - `cube` for roles containing `Cube`.
  - `reduction` for roles containing `Reduction`.
  - `vector` for roles containing `Vector`.
  - `unsupported` otherwise.

- [ ] **Step 4: Replace MVP role logic in `KernelizePass.cpp`**

After structural marking:

```cpp
FailureOr<OpRoleMap> roleMap = OpRoleClassifier().classify(*depResult);
if (failed(roleMap)) {
  signalPassFailure();
  return;
}
attachRoleAttributes(module, *roleMap);
if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Kernelize))
  emitOpRoleClassificationReport(llvm::errs(), *depResult, *roleMap);
```

Remove `classifyLinalgOp` from `KernelizePass.cpp` after tests pass.

- [ ] **Step 5: Run role and MVP tests**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v \
  build-v2-verify/test/Conversion/ascend-kernelize-roles.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-mvp.mlir'
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add include/Conversion/AscendV2/Kernelize/OpRoleClassification.h \
  lib/Conversion/AscendV2/Kernelize/OpRoleClassification.cpp \
  lib/Conversion/AscendV2/CMakeLists.txt \
  lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp \
  test/Conversion/ascend-kernelize-roles.mlir \
  test/Conversion/ascend-kernelize-mvp.mlir
git commit -m "conversion: implement Ascend V2 op role classification"
```

## Task 5: Primitive Candidate Analysis and CandidateClosure

**Files:**

- Create: `include/Conversion/AscendV2/Kernelize/CandidateClosure.h`
- Create: `lib/Conversion/AscendV2/Kernelize/CandidateClosure.cpp`
- Create: `include/Conversion/AscendV2/Kernelize/FusionCandidateAnalysis.h`
- Create: `lib/Conversion/AscendV2/Kernelize/FusionCandidateAnalysis.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
- Modify: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
- Create: `test/Conversion/ascend-kernelize-candidates.mlir`

- [ ] **Step 1: Write failing candidate lit test**

Create `test/Conversion/ascend-kernelize-candidates.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @elementwise_chain(%a: tensor<16xf32>, %b: tensor<16xf32>, %c: tensor<16xf32>)
    -> tensor<16xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
  %1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%0 : tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      %scale = arith.mulf %x, %x : f32
      linalg.yield %scale : f32
    } -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// CHECK: FusionCandidateAnalysis
// CHECK: candidate_id = 0
// CHECK-SAME: kind = "Fusion"
// CHECK-SAME: primitive = "ElementwiseChain"
// CHECK-SAME: primary_ops = [0]
// CHECK-SAME: internal_ops = [0, 1]
// CHECK-SAME: closed = true
// CHECK-SAME: benefit =
```

- [ ] **Step 2: Add closure API**

Add `CandidateClosure.h`:

```cpp
struct CandidateClosure {
  SmallVector<Operation *> internalOps;
  SmallVector<Value> externalInputs;
  SmallVector<Value> externalOutputs;
  SmallVector<Value> escapingValues;
  bool isClosed = false;
  std::string failureReason;
};

CandidateClosure computeCandidateClosure(ArrayRef<Operation *> internalOps,
                                          const ProducerConsumerIndex &index);
```

- [ ] **Step 3: Implement closure computation**

Rules:

- `internalOps` sorted by `OperationId`.
- `externalInputs`: operands whose defining op is outside the candidate or block arguments.
- `externalOutputs`: results used outside the candidate or returned from the function.
- `escapingValues`: result values with outside users not represented by `externalOutputs`.
- `isClosed = true` when all internal producer/consumer edges are contained and escaping values are either candidate outputs or allowed function returns.

Core loop:

```cpp
for (Operation *op : internalOps) {
  for (Value operand : op->getOperands()) {
    Operation *producer = operand.getDefiningOp();
    if (!producer || !internalSet.contains(producer))
      appendUnique(closure.externalInputs, operand);
  }
  for (Value result : op->getResults()) {
    bool usedOutside = false;
    for (Operation *user : result.getUsers()) {
      if (!internalSet.contains(user)) {
        usedOutside = true;
        break;
      }
    }
    if (usedOutside)
      appendUnique(closure.externalOutputs, result);
  }
}
closure.isClosed = closure.escapingValues.empty();
```

- [ ] **Step 4: Add primitive candidate API**

Add `FusionCandidateAnalysis.h`:

```cpp
struct ScheduleContract {
  SmallVector<StringRef> templateFamilies;
};

struct FusionCandidate {
  unsigned candidateId = 0;
  CandidateKind kind = CandidateKind::Fusion;
  std::string primitive;
  SmallVector<Operation *> primaryOps;
  SmallVector<Operation *> internalOps;
  CandidateClosure closure;
  ScheduleContract scheduleContract;
  int64_t benefitScore = 0;
  bool legal = false;
  std::string rejectionReason;
};

class FusionCandidateAnalyzer {
public:
  SmallVector<FusionCandidate> analyze(const DependencyAnalysisResult &deps,
                                       const OpRoleMap &roleMap,
                                       const KernelizeConfig &config) const;
};

void emitFusionCandidateReport(raw_ostream &os,
                               ArrayRef<FusionCandidate> candidates,
                               const ProducerConsumerIndex &index);
```

- [ ] **Step 5: Implement first primitive set**

Implement four primitives:

- `ElementwiseChain`: seed on `Vector + Injective`, expand through direct single-consumer injective users.
- `ConsumerIntoPrimary`: seed on `Cube` or `Reduction`, absorb direct injective consumers.
- `ReductionInlining`: seed on `Reduction`, absorb direct injective producer if there is no branch.
- `FallbackSingleOp`: seed on every supported op and create a one-op candidate.

Budget checks:

- Reject if `internalOps.size() > config.maxOpsPerCandidate`.
- Reject if primary op count exceeds `maxPrimaryRolesPerCandidate`.
- Reject if closure is not closed.

Schedule contract families:

- `Cube` -> `cube`.
- `Reduction` -> `reduction`.
- `Vector` -> `vector`.
- `FallbackSingleOp` copies the seed family.

Benefit:

- `ElementwiseChain`: `10 * (internalOps.size() - 1)`.
- `ConsumerIntoPrimary`: `20 + 5 * absorbedConsumerCount`.
- `ReductionInlining`: `10`.
- `FallbackSingleOp`: `1`.

- [ ] **Step 6: Wire candidate analysis into pass**

After role classification:

```cpp
KernelizeConfig config;
SmallVector<FusionCandidate> fusionCandidates =
    FusionCandidateAnalyzer().analyze(*depResult, *roleMap, config);
if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Kernelize))
  emitFusionCandidateReport(llvm::errs(), fusionCandidates, depResult->index);
```

- [ ] **Step 7: Run candidate test**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-verify/test/Conversion/ascend-kernelize-candidates.mlir'
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add include/Conversion/AscendV2/Kernelize/CandidateClosure.h \
  lib/Conversion/AscendV2/Kernelize/CandidateClosure.cpp \
  include/Conversion/AscendV2/Kernelize/FusionCandidateAnalysis.h \
  lib/Conversion/AscendV2/Kernelize/FusionCandidateAnalysis.cpp \
  lib/Conversion/AscendV2/CMakeLists.txt \
  lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp \
  test/Conversion/ascend-kernelize-candidates.mlir
git commit -m "conversion: add Ascend V2 fusion candidate analysis"
```

## Task 6: Candidate Merge Analysis

**Files:**

- Create: `include/Conversion/AscendV2/Kernelize/CandidateMergeAnalysis.h`
- Create: `lib/Conversion/AscendV2/Kernelize/CandidateMergeAnalysis.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
- Modify: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
- Create: `test/Conversion/ascend-kernelize-merge-horizontal.mlir`

- [ ] **Step 1: Write failing merge test**

Create the first part of `test/Conversion/ascend-kernelize-merge-horizontal.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @merge_vector_reduce(%a: tensor<16xf32>, %b: tensor<16xf32>, %init: tensor<f32>)
    -> tensor<f32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
      outs(%a : tensor<16xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
  %1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
      ins(%0 : tensor<16xf32>)
      outs(%init : tensor<f32>) {
    ^bb0(%x: f32, %acc: f32):
      %sum = arith.addf %x, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK: CandidateMergeAnalysis
// CHECK: merged_candidate_id = 0
// CHECK-SAME: source_candidates =
// CHECK-SAME: primary_ops = [0, 1]
// CHECK-SAME: primitive_combo = ["ElementwiseChain", "ReductionInlining"]
// CHECK-SAME: closed = true
```

- [ ] **Step 2: Add merge API**

Add `CandidateMergeAnalysis.h`:

```cpp
struct MergedCandidate {
  unsigned mergedCandidateId = 0;
  SmallVector<unsigned> sourceCandidateIds;
  SmallVector<Operation *> primaryOps;
  SmallVector<Operation *> internalOps;
  SmallVector<std::string> primitiveCombo;
  CandidateClosure closure;
  ScheduleContract scheduleContract;
  int64_t benefitScore = 0;
  bool legal = false;
  std::string rejectionReason;
};

class CandidateMergeAnalyzer {
public:
  SmallVector<MergedCandidate> analyze(ArrayRef<FusionCandidate> candidates,
                                       const DependencyAnalysisResult &deps,
                                       const KernelizeConfig &config) const;
};

void emitCandidateMergeReport(raw_ostream &os,
                              ArrayRef<MergedCandidate> merged,
                              const ProducerConsumerIndex &index);
```

- [ ] **Step 3: Implement adjacency and prefilters**

Rules:

- Build `op -> covering candidate ids` from legal fusion candidates.
- Two candidates are adjacent when a result from an op in candidate A is consumed by an op in candidate B.
- Attempt only one-hop pairs.
- Reject if:
  - union op count exceeds `maxOpsPerCandidate`.
  - primary op count exceeds `maxPrimaryRolesPerCandidate`.
  - template family set is empty after table lookup or fallback intersection.
  - recomputed closure is not closed.

Initial combo table:

```cpp
("vector", "reduction") -> ["reduction"]
("cube", "vector") -> ["cube"]
("vector", "vector") -> ["vector"]
```

- [ ] **Step 4: Wire merge analysis**

After fusion candidate analysis:

```cpp
SmallVector<MergedCandidate> mergedCandidates =
    CandidateMergeAnalyzer().analyze(fusionCandidates, *depResult, config);
if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Kernelize))
  emitCandidateMergeReport(llvm::errs(), mergedCandidates, depResult->index);
```

- [ ] **Step 5: Run merge test**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-verify/test/Conversion/ascend-kernelize-merge-horizontal.mlir'
```

Expected: PASS for merge checks.

- [ ] **Step 6: Commit**

```bash
git add include/Conversion/AscendV2/Kernelize/CandidateMergeAnalysis.h \
  lib/Conversion/AscendV2/Kernelize/CandidateMergeAnalysis.cpp \
  lib/Conversion/AscendV2/CMakeLists.txt \
  lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp \
  test/Conversion/ascend-kernelize-merge-horizontal.mlir
git commit -m "conversion: add Ascend V2 candidate merge analysis"
```

## Task 7: Horizontal Fusion Analysis

**Files:**

- Create: `include/Conversion/AscendV2/Kernelize/HorizontalFusionAnalysis.h`
- Create: `lib/Conversion/AscendV2/Kernelize/HorizontalFusionAnalysis.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
- Modify: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
- Modify: `test/Conversion/ascend-kernelize-merge-horizontal.mlir`

- [ ] **Step 1: Extend horizontal fusion test**

Append a sibling-candidate function to `test/Conversion/ascend-kernelize-merge-horizontal.mlir`:

```mlir
func.func @horizontal_siblings(%a: tensor<16xf32>, %out0: tensor<16xf32>, %out1: tensor<16xf32>)
    -> (tensor<16xf32>, tensor<16xf32>) {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%a : tensor<16xf32>)
      outs(%out0 : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      %v = arith.addf %x, %x : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  %1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%a : tensor<16xf32>)
      outs(%out1 : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      %v = arith.mulf %x, %x : f32
      linalg.yield %v : f32
    } -> tensor<16xf32>
  return %0, %1 : tensor<16xf32>, tensor<16xf32>
}

// CHECK: HorizontalFusionAnalysis
// CHECK: horizontal_candidate_id = 0
// CHECK-SAME: sibling_candidates =
// CHECK-SAME: shared_inputs = 1
// CHECK-SAME: per_group_contracts = 2
```

- [ ] **Step 2: Add horizontal fusion API**

Add `HorizontalFusionAnalysis.h`:

```cpp
struct HorizontalFusionCandidate {
  unsigned horizontalCandidateId = 0;
  SmallVector<unsigned> siblingCandidateIds;
  SmallVector<Value> sharedInputs;
  SmallVector<ScheduleContract> perGroupContracts;
  int64_t benefitScore = 0;
  bool legal = false;
  std::string rejectionReason;
};

class HorizontalFusionAnalyzer {
public:
  SmallVector<HorizontalFusionCandidate> analyze(
      ArrayRef<FusionCandidate> fusionCandidates,
      ArrayRef<MergedCandidate> mergedCandidates,
      const DependencyAnalysisResult &deps,
      const KernelizeConfig &config) const;
};

void emitHorizontalFusionReport(raw_ostream &os,
                                ArrayRef<HorizontalFusionCandidate> horizontal);
```

- [ ] **Step 3: Implement sibling grouping**

Initial rules:

- Candidate source set: legal single-primary `FusionCandidate` plus legal `MergedCandidate`.
- Build `Value -> candidate ids` from each candidate closure's `externalInputs`.
- A group is eligible when:
  - it has at least two candidates sharing at least one input.
  - all candidates are mutually unreachable in the producer/consumer graph.
  - group size is at most `maxHorizontalFusionGroupSize`.
  - each source candidate is closed.
- Benefit: `15 * (groupSize - 1)`.
- `perGroupContracts` copies each source candidate contract without merging.

- [ ] **Step 4: Wire horizontal fusion after merge analysis**

```cpp
SmallVector<HorizontalFusionCandidate> horizontalCandidates =
    HorizontalFusionAnalyzer().analyze(fusionCandidates, mergedCandidates,
                                       *depResult, config);
if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Kernelize))
  emitHorizontalFusionReport(llvm::errs(), horizontalCandidates);
```

- [ ] **Step 5: Run horizontal test**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-verify/test/Conversion/ascend-kernelize-merge-horizontal.mlir'
```

Expected: PASS for merge and horizontal checks.

- [ ] **Step 6: Commit**

```bash
git add include/Conversion/AscendV2/Kernelize/HorizontalFusionAnalysis.h \
  lib/Conversion/AscendV2/Kernelize/HorizontalFusionAnalysis.cpp \
  lib/Conversion/AscendV2/CMakeLists.txt \
  lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp \
  test/Conversion/ascend-kernelize-merge-horizontal.mlir
git commit -m "conversion: add Ascend V2 horizontal fusion analysis"
```

## Task 8: KernelPattern Graph and Partitioner

**Files:**

- Create: `include/Conversion/AscendV2/Kernelize/KernelPattern.h`
- Create: `lib/Conversion/AscendV2/Kernelize/KernelPattern.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`
- Modify: `lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp`
- Create: `test/Conversion/ascend-kernelize-patterns.mlir`
- Modify: `test/Conversion/ascend-v2-pipeline-mvp.mlir`

- [ ] **Step 1: Write failing pattern graph test**

Create `test/Conversion/ascend-kernelize-patterns.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='debug-stage=kernelize dump-report=true' 2>&1 | FileCheck %s

func.func @pattern_partition(%a: tensor<16xf32>, %b: tensor<16xf32>, %c: tensor<16xf32>)
    -> tensor<16xf32> {
  %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<16xf32>
  %1 = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%0 : tensor<16xf32>)
      outs(%c : tensor<16xf32>) {
    ^bb0(%x: f32, %out: f32):
      %scale = arith.mulf %x, %x : f32
      linalg.yield %scale : f32
    } -> tensor<16xf32>
  return %1 : tensor<16xf32>
}

// CHECK: KernelPatternGraph
// CHECK: pattern_candidate_id = 0
// CHECK-SAME: source = "Fusion"
// CHECK-SAME: internal_ops = [0, 1]
// CHECK: KernelPartition
// CHECK: kernel_pattern = "kernel_0"
// CHECK-SAME: internal_ops = [0, 1]
```

- [ ] **Step 2: Add pattern graph API**

Add `KernelPattern.h`:

```cpp
struct KernelPatternCandidate {
  unsigned candidateId = 0;
  CandidateKind sourceKind = CandidateKind::Fusion;
  SmallVector<Operation *> internalOps;
  SmallVector<Operation *> primaryOps;
  CandidateClosure closure;
  ScheduleContract scheduleContract;
  int64_t benefitScore = 0;
};

struct KernelPatternEdge {
  unsigned from = 0;
  unsigned to = 0;
  KernelPatternEdgeKind kind = KernelPatternEdgeKind::DataDependency;
  Value carriedValue;
};

struct KernelPatternGraph {
  SmallVector<KernelPatternCandidate> nodes;
  DenseMap<Operation *, SmallVector<unsigned>> coveringMap;
  SmallVector<KernelPatternEdge> edges;
};

struct KernelPattern {
  unsigned patternId = 0;
  std::string kernelName;
  SmallVector<Operation *> internalOps;
  SmallVector<Operation *> primaryOps;
  ScheduleContract scheduleContract;
};

class KernelPatternBuilder {
public:
  KernelPatternGraph build(ArrayRef<FusionCandidate> fusionCandidates,
                           ArrayRef<MergedCandidate> mergedCandidates,
                           ArrayRef<HorizontalFusionCandidate> horizontalCandidates,
                           const DependencyAnalysisResult &deps) const;
};

class KernelPartitioner {
public:
  SmallVector<KernelPattern> partition(const KernelPatternGraph &graph,
                                       const DependencyAnalysisResult &deps) const;
};

void attachKernelPatternAttributes(ModuleOp module,
                                   ArrayRef<KernelPattern> patterns);
void emitKernelPatternGraphReport(raw_ostream &os,
                                  const KernelPatternGraph &graph,
                                  const ProducerConsumerIndex &index);
void emitKernelPartitionReport(raw_ostream &os,
                               ArrayRef<KernelPattern> patterns,
                               const ProducerConsumerIndex &index);
```

- [ ] **Step 3: Implement graph construction**

Rules:

- Add legal fusion candidates, merged candidates, and horizontal candidates as nodes.
- For horizontal candidates, internal ops are the union of sibling candidate ops.
- Build `coveringMap` by appending node ids for each internal op.
- Add `Overlap` edges for candidates sharing at least one op.
- Add `DataDependency` edges when a result from a candidate op is consumed by another candidate op.
- Sort nodes by descending `benefitScore`, then ascending first primary op id, then ascending candidate id.

- [ ] **Step 4: Implement constrained greedy partitioner**

Rules:

- Visit connected overlap components independently.
- Select highest sorted candidate that does not overlap already selected ops.
- After candidate selection, create `FallbackSingleOp` patterns for uncovered analyzed ops.
- Final patterns sorted by minimum contained op id.
- Attach attrs to each internal op:
  - `ascend.v2.kernel = "kernel_N"`.
  - `ascend.v2.primary = true` only for primary ops in that pattern.
  - Preserve `ascend.v2.op_role` from Task 4.

- [ ] **Step 5: Wire graph and partition into pass**

Replace the old per-op kernel assignment with:

```cpp
KernelPatternGraph graph = KernelPatternBuilder().build(
    fusionCandidates, mergedCandidates, horizontalCandidates, *depResult);
SmallVector<KernelPattern> patterns =
    KernelPartitioner().partition(graph, *depResult);
attachKernelPatternAttributes(module, patterns);
if (ascend::v2::shouldDump(options, ascend::v2::DebugStage::Kernelize)) {
  emitKernelPatternGraphReport(llvm::errs(), graph, depResult->index);
  emitKernelPartitionReport(llvm::errs(), patterns, depResult->index);
}
```

- [ ] **Step 6: Run pattern and pipeline tests**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v \
  build-v2-verify/test/Conversion/ascend-kernelize-patterns.mlir \
  build-v2-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'
```

Expected: both tests pass. The pipeline test proves `--ascend-schedule` still consumes `ascend.v2.kernel` and `ascend.v2.op_role`.

- [ ] **Step 7: Commit**

```bash
git add include/Conversion/AscendV2/Kernelize/KernelPattern.h \
  lib/Conversion/AscendV2/Kernelize/KernelPattern.cpp \
  lib/Conversion/AscendV2/CMakeLists.txt \
  lib/Conversion/AscendV2/Kernelize/KernelizePass.cpp \
  test/Conversion/ascend-kernelize-patterns.mlir \
  test/Conversion/ascend-v2-pipeline-mvp.mlir
git commit -m "conversion: build Ascend V2 kernel patterns"
```

## Task 9: Phase 1 Verification and Tracking Update

**Files:**

- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Run host static check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 2: Run focused xvm build and lit**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v \
  build-v2-verify/test/Conversion/ascend-kernelize-dependency.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-roles.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-candidates.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-merge-horizontal.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-patterns.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-mvp.mlir \
  build-v2-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'
```

Expected: all listed tests pass.

- [ ] **Step 3: Run regression**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target check-afir -j10'
```

Expected: `check-afir` passes with the known unsupported tests unchanged.

- [ ] **Step 4: Update tracking board**

Update `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`:

- Phase 1 status: `Done`.
- Add verification commands and results under a Phase 1 verification record.
- Add commit range for Phase 1.
- Change current next step to Phase 2 Schedule full search.

- [ ] **Step 5: Commit tracking update**

```bash
git add docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
git commit -m "docs: update Ascend MLIR V2 Phase 1 tracking"
```

## Review Checklist

- V2-3 stage order is preserved: Dependency Analysis -> Structural Marking -> OpRole Classification -> Fusion Candidate Analysis -> Candidate Merge Analysis -> KernelPattern Construction -> Kernel Partition Decision.
- Existing MVP pipeline still works: Normalize -> Kernelize -> Schedule.
- `ascend.v2.op_role`, `ascend.v2.kernel`, and `ascend.v2.primary` remain available for Schedule MVP.
- Candidate output uses `primaryOps` terminology, not anchor terminology.
- No upstream MLIR source is modified.
- No Layer 2 pass consumes Schedule or Target model internals.
- All candidate IDs and pattern IDs are deterministic for identical IR.
- Fallback single-op coverage guarantees every first-layer-supported linalg op receives a kernel pattern.
