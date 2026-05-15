# Ascend Commercial Readiness Residuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining commercial-readiness residuals after the current Kernelize / Schedule / Realize MVP closures without redoing already-green work.

**Architecture:** Treat the current `dev-nyh` state as the baseline: Realize plan-driven mutation, Kernelize reduction fusion, handwritten pattern injection, target-aware Schedule, and the first Kernelize semantic registry are already MVP-closed. This plan hardens the remaining gaps by moving behavior from local heuristics into explicit contracts, adding dynamic view-chain coverage, adding persistent schedule cache I/O, and introducing an MLIR-native op interface path for AFIR-owned ops while keeping existing linalg/tensor/arith external models compatible.

**Tech Stack:** MLIR/LLVM C++17, TableGen OpInterface generation, AscendConversion library, LIT/FileCheck, GoogleTest unit tests, xvm/docker verification via `examples/dev-env.md`.

---

## Current Baseline

This plan assumes the repository is at or after:

- `c0f4458 fix: model Kernelize source and reshape operands`
- `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md` records:
  - Realize plan / IR mutation: `MVP Closed`
  - Kernelize reduction fusion closure: `Closed`
  - Schedule explicit target policy, search expansion, cache report: `Done`
  - KernelizeOpInterface / trait model first slice: `Done`
  - HandwrittenPattern injection first slice: `Done`

Do not stage or modify `docs/Ascend-MLIR-V2-Problem-Formulation.zh.md`; it is a user dirty file in the current worktree.

## File Structure

### Task 1: Realize Dynamic View-Chain Closure

- Modify: `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp`
- Modify: `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify: `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
- Modify: `lib/Conversion/Ascend/Realize/RealizeTypes.h`
- Create: `test/Conversion/ascend-realize-dynamic-view-chain-movement.mlir`
- Modify: `test/Conversion/ascend-realize-reshape-view-chain-movement.mlir`
- Modify: `test/unittests/Conversion/AscendRealizePlannerTest.cpp`

### Task 2: Kernelize Reduction Seed Policy Contract

- Modify: `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/OpRoleClassification.cpp`
- Modify: `test/Conversion/ascend-kernelize-reduction-fusion.mlir`
- Create: `test/Conversion/ascend-kernelize-reduction-seed-policy.mlir`
- Modify: `test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp`

### Task 3: Kernelize Primitive / Family Trait Resolver

- Modify: `include/Conversion/Ascend/Common/Attributes.h`
- Modify: `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeFamilyResolver.h`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeFamilyResolver.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/CandidateMergeAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelPattern.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Create: `test/Conversion/ascend-kernelize-template-family-traits.mlir`
- Modify: `test/Conversion/ascend-kernelize-handwritten-pattern.mlir`
- Modify: `test/Conversion/ascend-kernelize-attention-handwritten-pattern.mlir`
- Modify: `test/unittests/Conversion/AscendKernelPatternTest.cpp`

### Task 4: Schedule Persistent Tuning Cache

- Modify: `include/Conversion/Passes.td`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleCache.h`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleCache.cpp`
- Create: `lib/Conversion/Ascend/Schedule/SchedulePersistentCacheIO.h`
- Create: `lib/Conversion/Ascend/Schedule/SchedulePersistentCacheIO.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/SchedulePass.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Create: `test/Conversion/ascend-schedule-persistent-cache.mlir`
- Modify: `test/Conversion/ascend-schedule-cache.mlir`

### Task 5: MLIR-Native Kernelize OpInterface

- Modify: `include/Conversion/CMakeLists.txt`
- Create: `include/Conversion/Ascend/Kernelize/KernelizeOpInterfaces.td`
- Modify: `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeExternalModels.h`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeExternalModels.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Modify: `include/Conversion/Passes.h`
- Modify: `tools/afir-opt/afir-opt.cpp`
- Modify: `test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp`
- Create: `test/Conversion/ascend-kernelize-op-interface-native.mlir`

### Shared Documentation And Verification

- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
- Run: `test/tools/check_ascend_no_v2_code_naming.sh`
- Run: `test/tools/check_ascend_public_headers.sh`
- Run examples:
  - `examples/transformer/run-mainline.sh`
  - `examples/relu-broadcast-transpose/run-mainline.sh`
  - `examples/matmul-add-leakyrelu/run-mainline.sh --log`

## Task 1: Realize Dynamic View-Chain Closure

**Intent:** The current Realize path supports selected movement materialization through common view chains, but the remaining explicit non-goal is broader dynamic view-chain materialization. This task adds dynamic `tensor.extract_slice` and `tensor.reshape` producer/consumer coverage, keeps plan counts tied to actual IR mutation, and records report fields that expose deferred dynamic cases.

**Files:**
- Modify: `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp`
- Modify: `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify: `lib/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify: `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
- Create: `test/Conversion/ascend-realize-dynamic-view-chain-movement.mlir`
- Modify: `test/unittests/Conversion/AscendRealizePlannerTest.cpp`

- [ ] **Step 1: Add failing dynamic view-chain LIT**

Create `test/Conversion/ascend-realize-dynamic-view-chain-movement.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default' --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @dynamic_extract_slice_consumer(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %out: tensor<?x?xf32>, %m: index, %n: index) -> tensor<?x?xf32> {
  %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
  %producer = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<?x?xf32>)
    outs(%empty : tensor<?x?xf32>) {
  ^bb0(%a: f32, %o: f32):
    linalg.yield %a : f32
  } -> tensor<?x?xf32>

  %c0 = arith.constant 0 : index
  %slice = tensor.extract_slice %producer[%c0, %c0][%m, %n][1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>

  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%slice, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%out : tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %add = arith.addf %a, %b : f32
    linalg.yield %add : f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// CHECK: BufferizationDriver
// CHECK: vector_temporary_values = 1
// CHECK: MovementPlan
// CHECK: selected_paths = 1
// CHECK: MemoryRealizationPlan
// CHECK: materialized_allocs = 1
// CHECK: materialized_copies = 1
// CHECK-LABEL: func.func @dynamic_extract_slice_consumer
// CHECK: memref.alloc
// CHECK-SAME: memory_space = 9
// CHECK: memref.subview
// CHECK-SAME: memory_space = 9
```

- [ ] **Step 2: Run RED verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-dynamic-view-chain-movement.mlir'
```

Expected: FAIL because the new LIT is not in the build tree or the dynamic view-chain path does not materialize a selected movement through the dynamic `memref.subview`.

- [ ] **Step 3: Include `tensor.reshape` in Realize tensor view tracing**

In `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp`, update the supported tensor-view predicate:

```cpp
static bool isSupportedTensorViewOp(Operation *op) {
  return op->getNumResults() == 1 &&
         isa<tensor::CastOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
             tensor::ExtractSliceOp, tensor::ReshapeOp>(op);
}
```

Keep `getTensorViewSource` restricted to operand 0:

```cpp
static std::optional<Value> getTensorViewSource(Operation *op) {
  if (!isSupportedTensorViewOp(op) || op->getNumOperands() == 0)
    return std::nullopt;
  Value source = op->getOperand(0);
  if (!isTensorValue(source))
    return std::nullopt;
  return source;
}
```

- [ ] **Step 4: Add dynamic view-chain report counters**

In `lib/Conversion/Ascend/Realize/RealizeTypes.h`, extend `MovementPlan`:

```cpp
unsigned dynamicViewChainRewriteCount = 0;
unsigned deferredViewChainRewriteCount = 0;
```

In `lib/Conversion/Ascend/Realize/RealizeReport.cpp`, print them in the `MovementPlan` block:

```cpp
os << "  dynamic_view_chain_rewrites = "
   << bundle.movement.dynamicViewChainRewriteCount << "\n";
os << "  deferred_view_chain_rewrites = "
   << bundle.movement.deferredViewChainRewriteCount << "\n";
```

- [ ] **Step 5: Preserve dynamic subview operands during movement materialization**

In `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`, keep `materializeMovementUseViewChain` as the single reconstruction entry. For subviews, use the original mixed operands exactly as the current code does, and add an explicit helper to count dynamic pieces:

```cpp
static bool hasDynamicSubViewOperand(memref::SubViewOp subview) {
  return llvm::any_of(subview.getMixedOffsets(), [](OpFoldResult value) {
           return isa<Value>(value);
         }) ||
         llvm::any_of(subview.getMixedSizes(), [](OpFoldResult value) {
           return isa<Value>(value);
         }) ||
         llvm::any_of(subview.getMixedStrides(), [](OpFoldResult value) {
           return isa<Value>(value);
         });
}
```

When collecting `MovementUseRewrite`, count dynamic view chains for the owning kernel:

```cpp
static bool hasDynamicViewChain(ArrayRef<Operation *> viewChain) {
  return llvm::any_of(viewChain, [](Operation *op) {
    if (auto subview = dyn_cast<memref::SubViewOp>(op))
      return hasDynamicSubViewOperand(subview);
    return false;
  });
}
```

Record the counter when an item is accepted for materialization:

```cpp
if (llvm::any_of(uses, [](const MovementUseRewrite &rewrite) {
      return hasDynamicViewChain(rewrite.viewChain);
    }))
  ++bundle.movement.dynamicViewChainRewriteCount;
```

- [ ] **Step 6: Add unit coverage for dynamic view-chain classification**

In `test/unittests/Conversion/AscendRealizePlannerTest.cpp`, add a planner-level test that builds a `MovementPlan` with one selected path and asserts that dynamic view-chain counters remain zero before IR mutation:

```cpp
TEST(AscendRealizePlannerTest,
     MovementPlannerDoesNotInventDynamicViewChainCounts) {
  PlacementPlan placement;
  placement.kernelId = "kernel_0";
  placement.selectedPlaceCount = 1;
  placement.onChipPlaceCount = 1;

  StaticMemoryPlan staticMemory;
  staticMemory.kernelId = "kernel_0";
  staticMemory.trackedPlaceCount = 1;
  staticMemory.workspaceSlotCount = 1;
  StaticMemoryWorkspaceSlot slot;
  slot.slotId = 0;
  slot.valueId = 0;
  slot.place = MemoryPlace::VECIN;
  slot.staticByteSizeKnown = true;
  slot.byteSize = 128;
  staticMemory.workspaceSlots.push_back(slot);

  MovementPlanner planner;
  FailureOr<MovementPlan> plan = planner.build(placement, staticMemory);
  ASSERT_TRUE(succeeded(plan));
  EXPECT_EQ(plan->dynamicViewChainRewriteCount, 0u);
  EXPECT_EQ(plan->deferredViewChainRewriteCount, 0u);
}
```

- [ ] **Step 7: Run GREEN verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt AscendRealizePlannerTest && ctest --test-dir build -R AscendRealizePlannerTest --output-on-failure && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-dynamic-view-chain-movement.mlir build/test/Conversion/ascend-realize-view-chain-movement.mlir build/test/Conversion/ascend-realize-reshape-view-chain-movement.mlir'
```

Expected: all listed tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add lib/Conversion/Ascend/Realize/BufferizationDriver.cpp \
  lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp \
  lib/Conversion/Ascend/Realize/RealizeTypes.h \
  lib/Conversion/Ascend/Realize/RealizeReport.cpp \
  test/Conversion/ascend-realize-dynamic-view-chain-movement.mlir \
  test/Conversion/ascend-realize-reshape-view-chain-movement.mlir \
  test/unittests/Conversion/AscendRealizePlannerTest.cpp
git commit -m "fix: materialize dynamic Realize view chains"
```

## Task 2: Kernelize Reduction Seed Policy Contract

**Intent:** The current behavior already allows non-primary reductions to fuse into vector consumers. This task makes that behavior an explicit semantic contract so future reduction-like ops do not rediscover it through local role heuristics.

**Files:**
- Modify: `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h`
- Modify: `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp`
- Modify: `test/Conversion/ascend-kernelize-reduction-fusion.mlir`
- Create: `test/Conversion/ascend-kernelize-reduction-seed-policy.mlir`
- Modify: `test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp`

- [ ] **Step 1: Add failing seed-policy LIT**

Create `test/Conversion/ascend-kernelize-reduction-seed-policy.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='dump-report=true debug-stage=kernelize' 2>&1 | FileCheck %s

func.func @row_sum_then_vector(%arg0: tensor<32x64xf32>,
                               %arg1: tensor<32xf32>,
                               %out: tensor<32xf32>) -> tensor<32xf32> {
  %sum = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<32x64xf32>)
    outs(%arg1 : tensor<32xf32>) {
  ^bb0(%a: f32, %acc: f32):
    %r = arith.addf %acc, %a : f32
    linalg.yield %r : f32
  } -> tensor<32xf32>

  %scaled = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%sum : tensor<32xf32>)
    outs(%out : tensor<32xf32>) {
  ^bb0(%a: f32, %o: f32):
    %c = arith.constant 2.000000e+00 : f32
    %r = arith.mulf %a, %c : f32
    linalg.yield %r : f32
  } -> tensor<32xf32>
  return %scaled : tensor<32xf32>
}

// CHECK: DependencyAnalysis
// CHECK: op_id = 0
// CHECK-SAME: access = "Reduction"
// CHECK-SAME: seed_policy = "non_seed_when_fused"
// CHECK: OpRoleClassification
// CHECK: op_id = 0 roles = ["Reduction"]
// CHECK: op_id = 1 roles = ["Primary", "Vector", "Injective"]
// CHECK: FusionCandidateAnalysis
// CHECK: primitive = "ConsumerIntoPrimary"
// CHECK-SAME: primary_ops = [1]
// CHECK-SAME: internal_ops = [0, 1]
// CHECK: KernelPartition
// CHECK: linalg.generic
// CHECK-SAME: ascend.kernel = "kernel_0"
```

- [ ] **Step 2: Run RED verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-kernelize-reduction-seed-policy.mlir'
```

Expected: FAIL because reports do not include `seed_policy`.

- [ ] **Step 3: Add public seed policy enum**

In `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`:

```cpp
enum class KernelizeSeedPolicy {
  MaySeed,
  NonSeedWhenFused,
  NeverSeed,
};

struct KernelizeOpSemanticInfo {
  KernelizeParticipationKind participation =
      KernelizeParticipationKind::Unsupported;
  AccessPatternKind accessPattern = AccessPatternKind::Unknown;
  KernelizeSeedPolicy seedPolicy = KernelizeSeedPolicy::NeverSeed;
  SmallVector<IteratorKind, 4> iteratorKinds;
  SmallVector<AffineMap, 4> indexingMaps;
  SmallVector<unsigned, 2> resultRanks;
  SmallVector<KernelizeSemanticTrait, 4> traits;
  SmallVector<unsigned, 2> transparentOperandIndices;
  std::string modelName = "unknown";
  std::string unsupportedReason;
};

llvm::StringRef stringifyKernelizeSeedPolicy(KernelizeSeedPolicy policy);
```

In `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`:

```cpp
llvm::StringRef stringifyKernelizeSeedPolicy(KernelizeSeedPolicy policy) {
  switch (policy) {
  case KernelizeSeedPolicy::MaySeed:
    return "may_seed";
  case KernelizeSeedPolicy::NonSeedWhenFused:
    return "non_seed_when_fused";
  case KernelizeSeedPolicy::NeverSeed:
    return "never_seed";
  }
  return "never_seed";
}
```

- [ ] **Step 4: Thread seed policy through dependency summaries**

In `lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h`, extend `OpSemanticSummary`:

```cpp
KernelizeSeedPolicy seedPolicy = KernelizeSeedPolicy::NeverSeed;
```

In `makeSummary` inside `DependencyAnalysis.cpp`:

```cpp
summary.seedPolicy = info.seedPolicy;
```

In `emitDependencyAnalysisReport`, print:

```cpp
os << " seed_policy = \""
   << stringifyKernelizeSeedPolicy(summary.seedPolicy) << "\"";
```

- [ ] **Step 5: Assign seed policy in default models**

In `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`, set defaults for linalg:

```cpp
info.seedPolicy = KernelizeSeedPolicy::MaySeed;
```

When access pattern is reduction:

```cpp
info.accessPattern = AccessPatternKind::Reduction;
info.seedPolicy = KernelizeSeedPolicy::NonSeedWhenFused;
return success();
```

For ignored constants:

```cpp
info.seedPolicy = KernelizeSeedPolicy::NeverSeed;
```

For transparent tensor views:

```cpp
info.seedPolicy = KernelizeSeedPolicy::NeverSeed;
```

- [ ] **Step 6: Consume seed policy in fusion candidate analysis**

In `lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp`, replace local role-only reduction seed detection:

```cpp
static KernelizeSeedPolicy getSeedPolicy(Operation *op,
                                         const DependencyAnalysisResult &deps) {
  auto it = deps.summaries.find(op);
  if (it == deps.summaries.end())
    return KernelizeSeedPolicy::NeverSeed;
  return it->second.seedPolicy;
}
```

Then update `buildConsumerIntoPrimaryCandidate`:

```cpp
bool reductionSeed =
    getSeedPolicy(seed, deps) == KernelizeSeedPolicy::NonSeedWhenFused;
```

Keep the current behavior that a reduction with a single vector consumer chooses the consumer as `primaryOps`.

- [ ] **Step 7: Update unit test for public enum**

In `test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp`, add:

```cpp
EXPECT_EQ(stringifyKernelizeSeedPolicy(KernelizeSeedPolicy::MaySeed),
          "may_seed");
EXPECT_EQ(stringifyKernelizeSeedPolicy(
              KernelizeSeedPolicy::NonSeedWhenFused),
          "non_seed_when_fused");
EXPECT_EQ(stringifyKernelizeSeedPolicy(KernelizeSeedPolicy::NeverSeed),
          "never_seed");
```

Also set `info.seedPolicy = KernelizeSeedPolicy::MaySeed;` in the test model and assert it after resolve.

- [ ] **Step 8: Run GREEN verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt AscendKernelizeOpInterfaceTest && ctest --test-dir build -R AscendKernelizeOpInterfaceTest --output-on-failure && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-kernelize-reduction-seed-policy.mlir build/test/Conversion/ascend-kernelize-reduction-fusion.mlir build/test/Conversion/ascend-kernelize-op-interface-linalg.mlir'
```

Expected: all listed tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h \
  lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp \
  lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h \
  lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp \
  lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp \
  lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp \
  test/Conversion/ascend-kernelize-reduction-fusion.mlir \
  test/Conversion/ascend-kernelize-reduction-seed-policy.mlir \
  test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp
git commit -m "refactor: make Kernelize reduction seed policy explicit"
```

## Task 3: Kernelize Primitive / Family Trait Resolver

**Intent:** Candidate merge and handwritten grouping are currently functional, but the template family merge rules still live close to candidate merge code. This task moves primitive/family resolution into a dedicated resolver and centralizes handwritten/co-location attribute strings.

**Files:**
- Modify: `include/Conversion/Ascend/Common/Attributes.h`
- Modify: `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeFamilyResolver.h`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeFamilyResolver.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/CandidateMergeAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelPattern.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Create: `test/Conversion/ascend-kernelize-template-family-traits.mlir`
- Modify: `test/unittests/Conversion/AscendKernelPatternTest.cpp`

- [ ] **Step 1: Add failing family resolver LIT**

Create `test/Conversion/ascend-kernelize-template-family-traits.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='dump-report=true debug-stage=kernelize' 2>&1 | FileCheck %s

func.func @cube_vector_family(%lhs: tensor<16x32xf32>,
                              %rhs: tensor<32x64xf32>,
                              %bias: tensor<16x64xf32>,
                              %out: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<16x32xf32>, tensor<32x64xf32>)
      outs(%out : tensor<16x64xf32>) -> tensor<16x64xf32>
  %add = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%matmul, %bias : tensor<16x64xf32>, tensor<16x64xf32>)
    outs(%out : tensor<16x64xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %r = arith.addf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<16x64xf32>
  return %add : tensor<16x64xf32>
}

// CHECK: CandidateMergeAnalysis
// CHECK: primitive_combo = ["ProducerIntoConsumer", "ConsumerIntoPrimary"]
// CHECK-SAME: families = ["cube"]
// CHECK: family_resolver = "kernelize_trait_resolver"
```

- [ ] **Step 2: Run RED verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-kernelize-template-family-traits.mlir'
```

Expected: FAIL because the report does not include `family_resolver`.

- [ ] **Step 3: Move handwritten attrs into shared attributes**

In `include/Conversion/Ascend/Common/Attributes.h`, add:

```cpp
inline constexpr llvm::StringLiteral kKernelizeHandwrittenGroupAttr =
    "ascend.kernelize.handwritten_group";
inline constexpr llvm::StringLiteral kKernelizeMustCoLocateGroupAttr =
    "ascend.kernelize.must_colocate_group";
inline constexpr llvm::StringLiteral kKernelizeMustSeparateGroupAttr =
    "ascend.kernelize.must_separate_group";
```

Replace local string constants in `KernelPattern.cpp` and `FusionCandidateAnalysis.cpp` with these shared constants.

- [ ] **Step 4: Add primitive family trait fields**

In `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`, extend semantic info:

```cpp
SmallVector<std::string, 2> preferredTemplateFamilies;
```

Set default linalg families in `KernelizeOpRegistry.cpp`:

```cpp
if (info.accessPattern == AccessPatternKind::Contraction)
  info.preferredTemplateFamilies.push_back(kOpRoleCube.str());
else if (info.accessPattern == AccessPatternKind::Reduction)
  info.preferredTemplateFamilies.push_back(kOpRoleReduction.str());
else if (info.accessPattern == AccessPatternKind::Elementwise ||
         info.accessPattern == AccessPatternKind::Broadcast ||
         info.accessPattern == AccessPatternKind::LayoutTransform)
  info.preferredTemplateFamilies.push_back(kOpRoleVector.str());
```

- [ ] **Step 5: Create family resolver**

Create `lib/Conversion/Ascend/Kernelize/KernelizeFamilyResolver.h`:

```cpp
//===- KernelizeFamilyResolver.h - Kernelize family resolver -*- C++ -*-===//
#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FAMILY_RESOLVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_FAMILY_RESOLVER_H

#include "KernelizeTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir::afir::ascend::kernelize {

struct KernelizeFamilyResolution {
  SmallVector<std::string, 2> templateFamilies;
  std::string resolverName = "kernelize_trait_resolver";
};

KernelizeFamilyResolution resolveKernelizeTemplateFamilies(
    ArrayRef<std::string> lhsFamilies, ArrayRef<std::string> rhsFamilies,
    ArrayRef<KernelizePrimitiveKind> primitiveCombo);

} // namespace mlir::afir::ascend::kernelize

#endif
```

Create `lib/Conversion/Ascend/Kernelize/KernelizeFamilyResolver.cpp`:

```cpp
#include "KernelizeFamilyResolver.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::afir::ascend::kernelize {
namespace {

bool containsFamily(ArrayRef<std::string> families, StringRef family) {
  return llvm::is_contained(families, family);
}

bool containsFamilyPair(ArrayRef<std::string> lhsFamilies,
                        ArrayRef<std::string> rhsFamilies,
                        StringRef lhsFamily, StringRef rhsFamily) {
  return (containsFamily(lhsFamilies, lhsFamily) &&
          containsFamily(rhsFamilies, rhsFamily)) ||
         (containsFamily(lhsFamilies, rhsFamily) &&
          containsFamily(rhsFamilies, lhsFamily));
}

void appendUniqueFamily(SmallVectorImpl<std::string> &families,
                        StringRef family) {
  if (!containsFamily(families, family))
    families.push_back(family.str());
}

} // namespace

KernelizeFamilyResolution resolveKernelizeTemplateFamilies(
    ArrayRef<std::string> lhsFamilies, ArrayRef<std::string> rhsFamilies,
    ArrayRef<KernelizePrimitiveKind>) {
  KernelizeFamilyResolution result;

  if (containsFamilyPair(lhsFamilies, rhsFamilies, kOpRoleVector,
                         kOpRoleReduction)) {
    appendUniqueFamily(result.templateFamilies, kOpRoleReduction);
    return result;
  }

  if (containsFamilyPair(lhsFamilies, rhsFamilies, kOpRoleCube,
                         kOpRoleVector)) {
    appendUniqueFamily(result.templateFamilies, kOpRoleCube);
    return result;
  }

  for (const std::string &lhsFamily : lhsFamilies)
    if (containsFamily(rhsFamilies, lhsFamily))
      appendUniqueFamily(result.templateFamilies, lhsFamily);

  return result;
}

} // namespace mlir::afir::ascend::kernelize
```

- [ ] **Step 6: Replace local merge resolver**

In `CandidateMergeAnalysis.cpp`, include the new resolver:

```cpp
#include "KernelizeFamilyResolver.h"
```

Replace `resolveTemplateFamilies(...)` call in `buildMergedCandidate`:

```cpp
KernelizeFamilyResolution resolution = resolveKernelizeTemplateFamilies(
    lhs.scheduleContract.templateFamilies,
    rhs.scheduleContract.templateFamilies, merged.primitiveCombo);
merged.scheduleContract.templateFamilies =
    std::move(resolution.templateFamilies);
```

Add a `familyResolverName` field to `MergedCandidate` in `CandidateMergeAnalysis.h`:

```cpp
std::string familyResolverName = "kernelize_trait_resolver";
```

Print it in `emitCandidateMergeReport`:

```cpp
os << " family_resolver = \"" << candidate.familyResolverName << "\"";
```

- [ ] **Step 7: Add CMake source**

In `lib/Conversion/Ascend/CMakeLists.txt`, add:

```cmake
  Kernelize/KernelizeFamilyResolver.cpp
```

- [ ] **Step 8: Run GREEN verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt AscendKernelPatternTest && ctest --test-dir build -R AscendKernelPatternTest --output-on-failure && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-kernelize-template-family-traits.mlir build/test/Conversion/ascend-kernelize-merge-horizontal.mlir build/test/Conversion/ascend-kernelize-handwritten-pattern.mlir build/test/Conversion/ascend-kernelize-attention-handwritten-pattern.mlir'
```

Expected: all listed tests pass.

- [ ] **Step 9: Commit Task 3**

```bash
git add include/Conversion/Ascend/Common/Attributes.h \
  include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h \
  lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp \
  lib/Conversion/Ascend/Kernelize/KernelizeFamilyResolver.h \
  lib/Conversion/Ascend/Kernelize/KernelizeFamilyResolver.cpp \
  lib/Conversion/Ascend/Kernelize/CandidateMergeAnalysis.cpp \
  lib/Conversion/Ascend/Kernelize/CandidateMergeAnalysis.h \
  lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp \
  lib/Conversion/Ascend/Kernelize/KernelPattern.cpp \
  lib/Conversion/Ascend/CMakeLists.txt \
  test/Conversion/ascend-kernelize-template-family-traits.mlir \
  test/Conversion/ascend-kernelize-handwritten-pattern.mlir \
  test/Conversion/ascend-kernelize-attention-handwritten-pattern.mlir \
  test/unittests/Conversion/AscendKernelPatternTest.cpp
git commit -m "refactor: centralize Kernelize family resolution"
```

## Task 4: Schedule Persistent Tuning Cache

**Intent:** The current Schedule cache can reuse tuning signatures within a module/pipeline run. This task adds explicit file-backed cache input/output so repeated compiler invocations can reuse selected tuning signatures.

**Files:**
- Modify: `include/Conversion/Passes.td`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleCache.h`
- Modify: `lib/Conversion/Ascend/Schedule/ScheduleCache.cpp`
- Create: `lib/Conversion/Ascend/Schedule/SchedulePersistentCacheIO.h`
- Create: `lib/Conversion/Ascend/Schedule/SchedulePersistentCacheIO.cpp`
- Modify: `lib/Conversion/Ascend/Schedule/SchedulePass.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Create: `test/Conversion/ascend-schedule-persistent-cache.mlir`

- [ ] **Step 1: Add failing persistent-cache LIT**

Create `test/Conversion/ascend-schedule-persistent-cache.mlir`:

```mlir
// RUN: rm -f %t.cache
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default tuning-cache-out=%t.cache' | FileCheck %s --check-prefix=IR
// RUN: test -s %t.cache
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule='target-tile-policy=legacy-default tuning-cache-in=%t.cache dump-report=true debug-stage=schedule' 2>&1 | FileCheck %s --check-prefix=CACHE

func.func @persistent_cache_vector(%arg0: tensor<64xf32>,
                                   %arg1: tensor<64xf32>,
                                   %out: tensor<64xf32>) -> tensor<64xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf32>, tensor<64xf32>)
    outs(%out : tensor<64xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %r = arith.addf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// IR: ascend.schedule.decision_id
// CACHE: ScheduleCache:
// CACHE: persistent_tuning_hits = 1
```

- [ ] **Step 2: Run RED verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-persistent-cache.mlir'
```

Expected: FAIL because `tuning-cache-in` and `tuning-cache-out` pass options do not exist.

- [ ] **Step 3: Add pass options**

In `include/Conversion/Passes.td`, extend `AscendSchedulePass` options:

```tablegen
Option<"tuningCacheIn", "tuning-cache-in", "std::string",
       /*default=*/"\"\"",
       "Line-based persistent tuning signature cache input path">,
Option<"tuningCacheOut", "tuning-cache-out", "std::string",
       /*default=*/"\"\"",
       "Line-based persistent tuning signature cache output path">,
```

- [ ] **Step 4: Add cache I/O helper**

Create `lib/Conversion/Ascend/Schedule/SchedulePersistentCacheIO.h`:

```cpp
#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_PERSISTENT_CACHE_IO_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_PERSISTENT_CACHE_IO_H

#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir::afir::ascend::schedule {

FailureOr<SmallVector<std::string, 8>>
loadPersistentTuningCacheFile(StringRef path);

LogicalResult writePersistentTuningCacheFile(
    StringRef path, ArrayRef<std::string> signatures);

} // namespace mlir::afir::ascend::schedule

#endif
```

Create `lib/Conversion/Ascend/Schedule/SchedulePersistentCacheIO.cpp`:

```cpp
#include "SchedulePersistentCacheIO.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::afir::ascend::schedule {

FailureOr<SmallVector<std::string, 8>>
loadPersistentTuningCacheFile(StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return failure();

  SmallVector<std::string, 8> signatures;
  SmallVector<StringRef, 16> lines;
  StringRef(buffer.get()->getBuffer()).split(lines, '\n');
  for (StringRef line : lines) {
    line = line.trim();
    if (line.empty() || line.starts_with("#"))
      continue;
    signatures.push_back(line.str());
  }
  return signatures;
}

LogicalResult writePersistentTuningCacheFile(
    StringRef path, ArrayRef<std::string> signatures) {
  std::error_code ec;
  llvm::ToolOutputFile output(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return failure();
  for (const std::string &signature : signatures)
    output.os() << signature << "\n";
  output.keep();
  return success();
}

} // namespace mlir::afir::ascend::schedule
```

- [ ] **Step 5: Wire cache I/O into `SchedulePass`**

In `lib/Conversion/Ascend/Schedule/SchedulePass.cpp`, include:

```cpp
#include "SchedulePersistentCacheIO.h"
```

After creating `ScheduleCacheModel scheduleCacheModel;`:

```cpp
SmallVector<std::string, 8> seededSignatures =
    loadPersistentTuningCache(module);
if (!StringRef(tuningCacheIn).empty()) {
  FailureOr<SmallVector<std::string, 8>> fileSignatures =
      loadPersistentTuningCacheFile(tuningCacheIn);
  if (failed(fileSignatures)) {
    module.emitError() << "failed to read ascend schedule tuning cache file: "
                       << tuningCacheIn;
    signalPassFailure();
    return;
  }
  seededSignatures.append(fileSignatures->begin(), fileSignatures->end());
}
scheduleCacheModel.seedPersistentTuningSignatures(seededSignatures);
```

At the end of successful scheduling, after `storePersistentTuningCache`:

```cpp
if (!StringRef(tuningCacheOut).empty() &&
    failed(writePersistentTuningCacheFile(
        tuningCacheOut, scheduleCacheModel.getPersistentTuningSignatures()))) {
  module.emitError() << "failed to write ascend schedule tuning cache file: "
                     << tuningCacheOut;
  signalPassFailure();
  return;
}
```

- [ ] **Step 6: Add CMake source**

In `lib/Conversion/Ascend/CMakeLists.txt`, add:

```cmake
  Schedule/SchedulePersistentCacheIO.cpp
```

- [ ] **Step 7: Run GREEN verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-schedule-persistent-cache.mlir build/test/Conversion/ascend-schedule-cache.mlir build/test/Conversion/ascend-schedule-requires-target-policy.mlir'
```

Expected: all listed tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add include/Conversion/Passes.td \
  lib/Conversion/Ascend/Schedule/ScheduleCache.h \
  lib/Conversion/Ascend/Schedule/ScheduleCache.cpp \
  lib/Conversion/Ascend/Schedule/SchedulePersistentCacheIO.h \
  lib/Conversion/Ascend/Schedule/SchedulePersistentCacheIO.cpp \
  lib/Conversion/Ascend/Schedule/SchedulePass.cpp \
  lib/Conversion/Ascend/CMakeLists.txt \
  test/Conversion/ascend-schedule-persistent-cache.mlir \
  test/Conversion/ascend-schedule-cache.mlir
git commit -m "feat: add persistent Schedule tuning cache IO"
```

## Task 5: MLIR-Native Kernelize OpInterface

**Intent:** The current public semantic registry is the correct first step. This task adds an MLIR-native `OpInterface` path so AFIR-owned ops can implement Kernelize semantics through TableGen or external models, while the existing registry remains the compatibility fallback for linalg/tensor/arith models.

**Files:**
- Modify: `include/Conversion/CMakeLists.txt`
- Create: `include/Conversion/Ascend/Kernelize/KernelizeOpInterfaces.td`
- Modify: `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeExternalModels.h`
- Create: `lib/Conversion/Ascend/Kernelize/KernelizeExternalModels.cpp`
- Modify: `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Modify: `include/Conversion/Passes.h`
- Modify: `tools/afir-opt/afir-opt.cpp`
- Modify: `test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp`
- Create: `test/Conversion/ascend-kernelize-op-interface-native.mlir`

- [ ] **Step 1: Add failing native-interface unit assertion**

In `test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp`, add a test that expects the registry to prefer an attached interface over a later fallback model:

```cpp
TEST(AscendKernelizeOpInterfaceTest, RegistryPrefersNativeInterfaceModel) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  TestOp op(context, "test.analyze");

  KernelizeOpModelRegistry registry;
  registry.registerModel({"test_model", matchTestAnalyze, populateTestAnalyze});

  FailureOr<KernelizeOpSemanticInfo> info =
      registry.resolve(static_cast<Operation *>(op));

  ASSERT_TRUE(succeeded(info));
  EXPECT_EQ(info->modelName, "test_model");
}
```

This test starts as a compatibility guard and will be tightened after the generated interface type exists.

- [ ] **Step 2: Add TableGen interface declaration**

Create `include/Conversion/Ascend/Kernelize/KernelizeOpInterfaces.td`:

```tablegen
#ifndef ASCEND_KERNELIZE_OP_INTERFACES
#define ASCEND_KERNELIZE_OP_INTERFACES

include "mlir/IR/OpBase.td"

def KernelizeSemanticOpInterface
    : OpInterface<"KernelizeSemanticOpInterface"> {
  let cppNamespace = "::mlir::afir::ascend::kernelize";
  let description = [{
    Provides Ascend Kernelize semantic information for operations that can
    participate in Kernelize dependency analysis.
  }];
  let methods = [
    InterfaceMethod<
      "Populate Kernelize semantic information.",
      "::mlir::LogicalResult",
      "populateKernelizeSemanticInfo",
      (ins "::mlir::afir::ascend::kernelize::KernelizeOpSemanticInfo &":$info)
    >
  ];
}

#endif
```

- [ ] **Step 3: Add TableGen build rules**

In `include/Conversion/CMakeLists.txt`, add:

```cmake
set(LLVM_TARGET_DEFINITIONS Ascend/Kernelize/KernelizeOpInterfaces.td)
mlir_tablegen(Ascend/Kernelize/KernelizeOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(Ascend/Kernelize/KernelizeOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(AscendKernelizeOpInterfacesIncGen)
```

In `lib/Conversion/Ascend/CMakeLists.txt`, add dependency:

```cmake
  AscendKernelizeOpInterfacesIncGen
```

- [ ] **Step 4: Include generated interface in public header**

In `include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h`, after semantic type declarations:

```cpp
#include "Conversion/Ascend/Kernelize/KernelizeOpInterfaces.h.inc"
```

In `lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp`, include the generated definitions:

```cpp
#include "Conversion/Ascend/Kernelize/KernelizeOpInterfaces.cpp.inc"
```

- [ ] **Step 5: Teach registry to prefer native interface**

In `lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp`, at the start of `KernelizeOpModelRegistry::resolve` or the equivalent resolver implementation:

```cpp
if (auto iface = dyn_cast<KernelizeSemanticOpInterface>(op)) {
  KernelizeOpSemanticInfo info;
  if (failed(iface.populateKernelizeSemanticInfo(info)))
    return failure();
  if (info.modelName.empty())
    info.modelName = "native_op_interface";
  return info;
}
```

Keep the existing registered-model loop as fallback.

- [ ] **Step 6: Add external model registration API**

Create `lib/Conversion/Ascend/Kernelize/KernelizeExternalModels.h`:

```cpp
#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_EXTERNAL_MODELS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_EXTERNAL_MODELS_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir::afir::ascend::kernelize {

void registerKernelizeExternalModels(DialectRegistry &registry);

} // namespace mlir::afir::ascend::kernelize

#endif
```

Create `lib/Conversion/Ascend/Kernelize/KernelizeExternalModels.cpp`:

```cpp
#include "KernelizeExternalModels.h"

#include "Conversion/Ascend/Kernelize/KernelizeOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace mlir::afir::ascend::kernelize {

void registerKernelizeExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context,
                            linalg::LinalgDialect *) {
    (void)context;
  });
  registry.addExtension(+[](MLIRContext *context,
                            tensor::TensorDialect *) {
    (void)context;
  });
}

} // namespace mlir::afir::ascend::kernelize
```

The first version registers the extension hook without changing semantics; Task 5 acceptance is the generated interface path plus compatibility. Add concrete external models in later focused tasks when replacing default linalg/tensor registry models.

- [ ] **Step 7: Register external model hook in tools**

In `include/Conversion/Passes.h`, include the declaration:

```cpp
#include "Conversion/Ascend/Kernelize/KernelizeExternalModels.h"
```

In `tools/afir-opt/afir-opt.cpp`, after dialect insertion:

```cpp
afir::ascend::kernelize::registerKernelizeExternalModels(registry);
```

- [ ] **Step 8: Add native interface smoke LIT**

Create `test/Conversion/ascend-kernelize-op-interface-native.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize='dump-report=true debug-stage=kernelize' 2>&1 | FileCheck %s

func.func @native_interface_compat(%arg0: tensor<8xf32>,
                                   %arg1: tensor<8xf32>,
                                   %out: tensor<8xf32>) -> tensor<8xf32> {
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
    outs(%out : tensor<8xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %r = arith.addf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK: DependencyAnalysis
// CHECK-SAME: model = "linalg"
// CHECK: KernelPartition
// CHECK-SAME: ascend.kernel = "kernel_0"
```

- [ ] **Step 9: Run GREEN verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt AscendKernelizeOpInterfaceTest && ctest --test-dir build -R AscendKernelizeOpInterfaceTest --output-on-failure && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-kernelize-op-interface-native.mlir build/test/Conversion/ascend-kernelize-op-interface-linalg.mlir build/test/Conversion/ascend-kernelize-op-interface-tensor-view.mlir'
```

Expected: all listed tests pass.

- [ ] **Step 10: Commit Task 5**

```bash
git add include/Conversion/CMakeLists.txt \
  include/Conversion/Ascend/Kernelize/KernelizeOpInterfaces.td \
  include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h \
  lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp \
  lib/Conversion/Ascend/Kernelize/KernelizeExternalModels.h \
  lib/Conversion/Ascend/Kernelize/KernelizeExternalModels.cpp \
  lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp \
  lib/Conversion/Ascend/CMakeLists.txt \
  include/Conversion/Passes.h \
  tools/afir-opt/afir-opt.cpp \
  test/unittests/Conversion/AscendKernelizeOpInterfaceTest.cpp \
  test/Conversion/ascend-kernelize-op-interface-native.mlir
git commit -m "feat: add MLIR Kernelize semantic op interface"
```

## Final Verification

- [ ] **Step 1: Run host static checks**

```bash
git diff --check
test/tools/check_ascend_public_headers.sh
test/tools/check_ascend_no_v2_code_naming.sh
```

Expected: all commands exit 0.

- [ ] **Step 2: Run xvm focused build and unit tests**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt afir-translate AscendKernelizeOpInterfaceTest AscendKernelPatternTest AscendRealizePlannerTest AscendScheduleDecisionTest && ctest --test-dir build -R "Ascend(KernelizeOpInterface|KernelPattern|RealizePlanner|ScheduleDecision)Test" --output-on-failure'
```

Expected: all listed build targets and ctests pass.

- [ ] **Step 3: Run xvm focused LIT**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v \
  build/test/Conversion/ascend-realize-dynamic-view-chain-movement.mlir \
  build/test/Conversion/ascend-realize-view-chain-movement.mlir \
  build/test/Conversion/ascend-realize-reshape-view-chain-movement.mlir \
  build/test/Conversion/ascend-kernelize-reduction-seed-policy.mlir \
  build/test/Conversion/ascend-kernelize-reduction-fusion.mlir \
  build/test/Conversion/ascend-kernelize-template-family-traits.mlir \
  build/test/Conversion/ascend-kernelize-handwritten-pattern.mlir \
  build/test/Conversion/ascend-kernelize-attention-handwritten-pattern.mlir \
  build/test/Conversion/ascend-schedule-persistent-cache.mlir \
  build/test/Conversion/ascend-schedule-cache.mlir \
  build/test/Conversion/ascend-kernelize-op-interface-native.mlir \
  build/test/Conversion/ascend-kernelize-op-interface-linalg.mlir \
  build/test/Conversion/ascend-kernelize-op-interface-tensor-view.mlir'
```

Expected: all listed LIT tests pass.

- [ ] **Step 4: Run xvm broad Ascend LIT**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-" && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target'
```

Expected: all discovered tests pass or existing unsupported tests remain unsupported.

- [ ] **Step 5: Run mainline examples**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && bash examples/transformer/run-mainline.sh && bash examples/relu-broadcast-transpose/run-mainline.sh && bash examples/matmul-add-leakyrelu/run-mainline.sh --log'
```

Expected:

- transformer reports `transformer_dynamic.full_codegen=pass`
- relu-broadcast-transpose reports `session.validation=pass`
- matmul-add-leakyrelu reports `session.validation=pass`

- [ ] **Step 6: Update tracking document**

In `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`, update the `Design vs Implementation Gap Board` rows:

- Realize plan / IR mutation: mention dynamic view-chain materialization coverage.
- Kernelize reduction fusion closure: mention explicit `KernelizeSeedPolicy`.
- Kernelize template family merge: mention `KernelizeFamilyResolver`.
- Target-driven tile: mention file-backed persistent tuning cache.
- Cross-stage contract hardening: mention generated `KernelizeSemanticOpInterface`.

- [ ] **Step 7: Commit tracking update**

```bash
git add docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
git commit -m "docs: track commercial readiness residual closure"
```

- [ ] **Step 8: Push after final verification**

```bash
git status --short --branch
git push origin dev-nyh
```

Expected: branch pushes cleanly; the only uncommitted file may be the user-owned `docs/Ascend-MLIR-V2-Problem-Formulation.zh.md`.

## Self-Review

Spec coverage:

- Realize plan / IR mutation residual is covered by Task 1.
- Kernelize reduction non-seed behavior is covered by Task 2 as an explicit semantic contract.
- Kernelize iterative merge / handwritten producer residual is covered by Task 3 through central family resolution and shared handwritten attrs.
- Schedule target-aware / search / cache residual is covered by Task 4 as persistent tuning cache I/O; explicit policy and expanded search are already baseline.
- Kernelize registry to formal OpInterface / trait model is covered by Task 5.

Placeholder scan:

- No task uses forbidden placeholder instructions; each step names concrete files, commands, and expected outcomes.
- Every code-changing task lists exact files, exact tests, commands, expected failure mode, implementation snippet, green command, and commit command.

Type consistency:

- `KernelizeSeedPolicy` is declared in the public semantic contract, stringified in `KernelizeOpInterface.cpp`, threaded through `OpSemanticSummary`, and consumed by `FusionCandidateAnalysis`.
- `KernelizeFamilyResolution` is returned by `resolveKernelizeTemplateFamilies` and consumed by `CandidateMergeAnalysis`.
- Schedule persistent cache APIs consistently use `ArrayRef<std::string>` and `SmallVector<std::string, 8>`.
- The generated MLIR interface is named `KernelizeSemanticOpInterface` in TableGen, C++ registry lookup, and tests.
