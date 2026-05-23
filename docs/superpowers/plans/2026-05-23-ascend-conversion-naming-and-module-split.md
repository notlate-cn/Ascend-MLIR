# Ascend Conversion Naming And Module Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove historical `Phase5` wording from current Ascend conversion code and split the largest conversion implementation files without changing public pass, CLI, or test RUN surfaces.

**Architecture:** Keep `lib/Conversion/Ascend`'s current pipeline directories and public pass arguments stable. First make mechanical behavior-neutral renames, then extract the Realize-to-Translate memory bridge into its own module, then split compute lowering around an internal lowering context and focused family modules.

**Tech Stack:** C++17, MLIR dialect conversion/lowering code, TableGen pass definitions, CMake, llvm-lit, ctest, xvm CANN 9.1 verification.

---

## File Structure

Create:

- `lib/Conversion/Ascend/Realize/TranslateMemoryBridge.h`
  - Owns `TranslateBridgeMaterializationCounts`.
  - Declares `materializeTranslateMemoryBridge(ModuleOp module)`.
  - Stays internal to `lib/Conversion/Ascend`; do not add it under `include/`.
- `lib/Conversion/Ascend/Realize/TranslateMemoryBridge.cpp`
  - Owns final-output VECOUT bridge and cube-to-vector bridge implementation.
  - Contains bridge-local pattern structs and preflight helpers.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeLoweringInternal.h`
  - Internal header for compute lowering split.
  - Owns `ComputeLoweringContext` and family-level `lower*Computes` declarations.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeSelectedTileLowering.cpp`
  - Owns selected-tile materialization functions currently declared in `KernelIRUtils.h`.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeScalarFallbackLowering.cpp`
  - Owns GM scalar-loop fallback and local scalar load/store fallback.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeTransposeLowering.cpp`
  - Owns named `linalg.transpose` lowering.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeFillLowering.cpp`
  - Owns `linalg.fill` lowering.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeMatmulLowering.cpp`
  - Owns `linalg.matmul` and `linalg.batch_matmul` lowering.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeElementwiseLowering.cpp`
  - Owns named `linalg.elementwise` lowering.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeReductionLowering.cpp`
  - Owns reduction `linalg.generic` lowering.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeGatherLowering.cpp`
  - Owns gather-flavored all-parallel `linalg.generic` helper logic used by
    the all-parallel coordinator.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeAllParallelLowering.cpp`
  - Owns the all-parallel `linalg.generic` coordinator so gather pre/post-op
    ordering remains identical to the current single-file implementation.

Modify:

- `include/Conversion/Ascend/Passes.td`
  - Update stale Normalize description.
  - Replace current `Phase 5` pass descriptions with behavior terms.
- `include/Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h`
  - Rename current `Phase5*` API symbols to backend vocabulary.
- `lib/Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.cpp`
  - Rename definitions and internal calls.
- `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.h`
  - Include `TranslateMemoryBridge.h`.
  - Use `TranslateBridgeMaterializationCounts`.
  - Stop exposing `materializePhase5Bridge`.
- `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
  - Keep plan validation, memory-space annotation, movement materialization, aggregation, and report updates.
  - Call `materializeTranslateMemoryBridge(module)`.
- `lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp`
  - Becomes orchestration plus only shared code that has not yet been extracted.
  - Must end with a small `convertCompute` that calls family helpers in the current order.
- `lib/Conversion/Ascend/CMakeLists.txt`
  - Add the new Realize and Compute source files to the existing source groups.
- `test/unittests/Conversion/AscendLinalgBodyClassifierTest.cpp`
  - Rename function calls and test names that say `Phase5`.
- `test/unittests/Conversion/AscendRealizePlannerTest.cpp`
  - Include `TranslateMemoryBridge.h`.
  - Replace direct bridge method tests with `materializeTranslateMemoryBridge`.
- `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
  - Only update current-state rows that describe active code vocabulary; keep historical plan links and phase labels unchanged.

Do not touch unrelated dirty files. In the current workspace, `examples/real-npu.md` may already be modified; do not stage it unless the user explicitly asks.

---

## Task 0: Preflight And Baseline

**Files:**
- Read: `docs/superpowers/specs/2026-05-23-ascend-conversion-naming-and-module-split-design.md`
- Read: `examples/dev-env.md`
- Read: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Confirm the worktree before editing**

Run:

```bash
git status --short
```

Expected:

```text
 M examples/real-npu.md
```

If more files are dirty, inspect them with `git diff --stat` and do not stage unrelated changes.

- [ ] **Step 2: Confirm the design still matches the checked-out source**

Run:

```bash
rg -n "Phase5|phase5|Phase 5|materializePhase5|classifyPhase5|isSupportedPhase5" include/Conversion/Ascend lib/Conversion/Ascend test/unittests/Conversion docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
```

Expected: matches in `Passes.td`, `LinalgBodyClassifier`, `MemoryRealizationDriver`, `AscendLinalgBodyClassifierTest`, `AscendRealizePlannerTest`, and historical/current tracking text. Do not edit generated artifacts or example artifact filenames that contain `phase5_`.

- [ ] **Step 3: Build a focused baseline if `build/` exists locally**

Run:

```bash
test -d build && ninja -C build afir-opt AscendLinalgBodyClassifierTest AscendRealizePlannerTest || true
test -d build && ctest --test-dir build -R 'AscendLinalgBodyClassifier|AscendRealizePlanner' --output-on-failure || true
```

Expected when local build is configured: both targets build and both ctest entries pass. If `build/` is absent locally, skip local baseline and rely on xvm in later verification tasks.

---

## Task 1: Rename Current `Phase5` Code Vocabulary

**Files:**
- Modify: `include/Conversion/Ascend/Passes.td`
- Modify: `include/Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h`
- Modify: `lib/Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.cpp`
- Modify: `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.h`
- Modify: `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify: `lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp`
- Modify: `test/unittests/Conversion/AscendLinalgBodyClassifierTest.cpp`
- Modify: `test/unittests/Conversion/AscendRealizePlannerTest.cpp`

- [ ] **Step 1: Apply the symbol rename mechanically**

Run:

```bash
perl -0pi -e 's/classifyPhase5ReductionBody/classifyBackendReductionBody/g; s/isSupportedPhase5VectorOutput/isSupportedBackendVectorOutput/g; s/isSupportedPhase5GatherOutput/isSupportedBackendGatherOutput/g; s/isSupportedPhase5FinalOutput/isSupportedBackendFinalOutput/g; s/Phase5BridgeMaterializationCounts/TranslateBridgeMaterializationCounts/g; s/materializePhase5Bridge/materializeTranslateMemoryBridge/g; s/phase5BridgeCounts/translateBridgeCounts/g; s/phase5Bridge/translateBridge/g; s/Phase5BridgeOutput/TranslateBridgeOutput/g; s/Phase5CubeBridge/TranslateCubeBridge/g' \
  include/Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h \
  lib/Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.cpp \
  lib/Conversion/Ascend/Realize/MemoryRealizationDriver.h \
  lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp \
  test/unittests/Conversion/AscendLinalgBodyClassifierTest.cpp \
  test/unittests/Conversion/AscendRealizePlannerTest.cpp
```

Expected: no command output.

- [ ] **Step 2: Update test names that still contain `Phase5`**

Replace these exact test names in `test/unittests/Conversion/AscendRealizePlannerTest.cpp`:

```text
Phase5BridgeFailureDoesNotLeavePartialVecOutAlloc -> TranslateBridgeFailureDoesNotLeavePartialVecOutAlloc
Phase5CubeBridgeDominatesNestedVectorUse -> TranslateCubeBridgeDominatesNestedVectorUse
Phase5CubeBridgeSupportsBatchMatmul -> TranslateCubeBridgeSupportsBatchMatmul
```

Replace this exact test name in `test/unittests/Conversion/AscendLinalgBodyClassifierTest.cpp`:

```text
ClassifiesGmScalarGenericButRejectsPhase5Consumers -> ClassifiesGmScalarGenericButRejectsBackendConsumers
```

- [ ] **Step 3: Update current pass descriptions**

Patch `include/Conversion/Ascend/Passes.td` so the affected descriptions read:

```tablegen
def AscendNormalizePass : Pass<"ascend-normalize", "mlir::ModuleOp"> {
  let summary = "Normalize supported entry IR for the Ascend MLIR pipeline";
  let description = [{
    Validates supported entry dialects for the Ascend pipeline, preserves the
    normalized IR shape, stamps the module with the normalized marker, and emits
    optional debug-stage reporting.
  }];
```

```tablegen
def AscendComputeLowerPass : Pass<"ascend-compute-lower", "mlir::func::FuncOp"> {
  let summary = "Lower supported memory-realized linalg and copies to AscendC";
  let description = [{
    Translate compute lowering. It annotates kernel kind and mix matmul ABI
    semantics when needed, lowers supported linalg compute operations and
    memref.copy movement to AscendC, and fails closed on support-matrix
    diagnostics.
  }];
```

```tablegen
def AscendParallelizePass : Pass<"ascend-parallelize", "mlir::ModuleOp"> {
  let summary = "Lower Ascend kernel dispatch to explicit parallel loops";
  let description = [{
    Lowers Ascend kernel dispatch structure to explicit parallel loops for the
    internal AscendC kernel construction surface.
  }];
```

```tablegen
def AscendPrepareForEmitPass
    : Pass<"ascend-prepare-for-emit", "mlir::ModuleOp"> {
  let summary = "Prepare AscendC kernel ABI data for emission";
```

```tablegen
def AscendCanonicalizeCannSignaturePass
    : Pass<"ascend-canonicalize-cann-signature", "mlir::ModuleOp"> {
  let summary = "Canonicalize Ascend kernel signatures to CANN ABI";
  let description = [{
    Canonicalizes Ascend kernel signatures to the CANN runtime ABI.
  }];
```

- [ ] **Step 4: Confirm no current conversion symbol still says `Phase5`**

Run:

```bash
rg -n "Phase5|phase5|Phase 5" include/Conversion/Ascend lib/Conversion/Ascend test/unittests/Conversion
```

Expected: no matches except intentional historical text if a nearby comment explicitly says it is historical. Prefer no matches in these directories.

- [ ] **Step 5: Build and run focused tests**

Run:

```bash
ninja -C build afir-opt AscendLinalgBodyClassifierTest AscendRealizePlannerTest
ctest --test-dir build -R 'AscendLinalgBodyClassifier|AscendRealizePlanner' --output-on-failure
llvm-lit -v build/test/Conversion/ascend-realize*.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower*.mlir
```

Expected: build succeeds; ctest reports both tests passed; lit reports all selected tests passed.

- [ ] **Step 6: Commit the pure rename**

Run:

```bash
git status --short
git add include/Conversion/Ascend/Passes.td \
  include/Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h \
  lib/Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.cpp \
  lib/Conversion/Ascend/Realize/MemoryRealizationDriver.h \
  lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp \
  test/unittests/Conversion/AscendLinalgBodyClassifierTest.cpp \
  test/unittests/Conversion/AscendRealizePlannerTest.cpp
git commit -m "refactor: rename Ascend backend lowering vocabulary"
```

Expected: commit does not include `examples/real-npu.md`.

---

## Task 2: Extract `TranslateMemoryBridge`

**Files:**
- Create: `lib/Conversion/Ascend/Realize/TranslateMemoryBridge.h`
- Create: `lib/Conversion/Ascend/Realize/TranslateMemoryBridge.cpp`
- Modify: `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.h`
- Modify: `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`
- Modify: `test/unittests/Conversion/AscendRealizePlannerTest.cpp`

- [ ] **Step 1: Create the internal bridge header**

Create `lib/Conversion/Ascend/Realize/TranslateMemoryBridge.h` with this exact interface:

```c++
//===- TranslateMemoryBridge.h - Ascend Translate memory bridge -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TRANSLATEMEMORYBRIDGE_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TRANSLATEMEMORYBRIDGE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::afir::ascend::realize {

struct TranslateBridgeMaterializationCounts {
  unsigned materializedAllocCount = 0;
  unsigned materializedCopyCount = 0;
};

FailureOr<llvm::StringMap<TranslateBridgeMaterializationCounts>>
materializeTranslateMemoryBridge(ModuleOp module);

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TRANSLATEMEMORYBRIDGE_H
```

- [ ] **Step 2: Move bridge implementation out of the driver**

Create `lib/Conversion/Ascend/Realize/TranslateMemoryBridge.cpp`.

Move these definitions from `MemoryRealizationDriver.cpp` into the new file, preserving function bodies except for the `Phase5` names already changed in Task 1:

```text
getKernelId
opRolesAttrHasRole
hasRole
isVectorOp
isCubeOp
withMemorySpace
getMemorySpaceAttr
annotateAscendCUnits
isReductionInitFillForWriter
isAllowedExternalOutputUse
isFinalKernelOutput
TranslateBridgeOutput
TranslateCubeBridge
isConstantOpFoldResult
hasReturnUse(Value, DenseSet<Operation *> &)
hasReturnUse(Value)
isSupportedConcatTargetSubview
findSupportedConcatCopyUse
dynamicSizeForDim
replaceAllocDimUses
verifyReplaceableAllocDimUses
buildDynamicSizes
createMemorySpaceAllocLike
isBridgeableCubeCompute
isDpsInputOperand
collectSafeCubeVectorUses
materializeTranslateMemoryBridge
```

Use these includes at the top of `TranslateMemoryBridge.cpp`:

```c++
#include "TranslateMemoryBridge.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h"
#include "Target/Ascend/TargetProfile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
```

Keep these definitions in `MemoryRealizationDriver.cpp`:

```text
AnnotatableAlloc
MovementUseRewrite
MovementMaterializationItem
addMaterializationCounts
getMemorySpaceValue
hasMemorySpace
isGmMemref
hasDynamicSubViewOperand
hasDynamicViewChain
appendUniqueGmMovementSource
collectGmMovementSourcesByValueId
collectViewChainToSource
collectMovementSourceUses
isStaticIdentityMemRef
getElementByteWidth
getElementOffset
getPackedWorkspaceElementCount
getIdentityStrides
canShareMovementWorkspace
createPackedMovementWorkspaceAlloc
createPackedMovementWorkspaceView
materializeMovementUseViewChain
preflightMovementUseViewChain
preflightSingleMovementItem
preflightMovementWorkspaceGroup
materializeSingleMovementItem
materializeMovementWorkspaceGroup
countDynamicViewChainRewrites
lookupWorkspaceSlot
MemoryRealizationDriver methods
```

If `MemoryRealizationDriver.cpp` still needs `getKernelId`, `isVectorOp`, `withMemorySpace`, `getMemorySpaceAttr`, `buildDynamicSizes`, or `createMemorySpaceAllocLike`, keep private copies there rather than creating a new broad utility header in this task. The duplication is local and avoids widening the internal interface during the extraction.

- [ ] **Step 3: Update the driver header**

Patch `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.h`:

```c++
#include "RealizeTypes.h"
#include "TranslateMemoryBridge.h"
```

Remove the old count struct from this header. Keep these public methods:

```c++
  FailureOr<llvm::StringMap<TranslateBridgeMaterializationCounts>>
  materializeMovementSteps(ModuleOp module,
                           MutableArrayRef<RealizePlanBundle> bundles) const;
  void markMemorySpaceMaterialized(
      MemoryRealizationPlan &plan, unsigned annotationCount,
      const TranslateBridgeMaterializationCounts &materializationCounts) const;
```

Do not keep a `MemoryRealizationDriver::materializeTranslateMemoryBridge` method. Direct bridge tests should call the free function from `TranslateMemoryBridge.h`.

- [ ] **Step 4: Update the driver implementation**

In `MemoryRealizationDriver.cpp`, add:

```c++
#include "TranslateMemoryBridge.h"
```

Replace bridge calls inside `MemoryRealizationDriver::materialize` with:

```c++
  FailureOr<llvm::StringMap<TranslateBridgeMaterializationCounts>>
      translateBridgeCounts = materializeTranslateMemoryBridge(module);
  if (failed(translateBridgeCounts))
    return failure();
```

Aggregate with:

```c++
    TranslateBridgeMaterializationCounts materializationCount;
    auto bridgeIt = translateBridgeCounts->find(bundle.kernel.kernelId);
    if (bridgeIt != translateBridgeCounts->end())
      materializationCount = bridgeIt->second;
```

- [ ] **Step 5: Update bridge unit tests**

In `test/unittests/Conversion/AscendRealizePlannerTest.cpp`, include:

```c++
#include "Conversion/Ascend/Realize/TranslateMemoryBridge.h"
```

Replace direct calls:

```c++
driver.materializeTranslateMemoryBridge(*module)
```

with:

```c++
materializeTranslateMemoryBridge(*module)
```

Keep `MemoryRealizationDriver driver;` only in tests that still call driver methods.

- [ ] **Step 6: Wire CMake**

Add the new source under `ASCEND_REALIZE_SOURCES` in `lib/Conversion/Ascend/CMakeLists.txt`:

```cmake
  Realize/TranslateMemoryBridge.cpp
```

Place it near `Realize/MemoryRealizationDriver.cpp`.

- [ ] **Step 7: Build and run Realize-focused verification**

Run:

```bash
ninja -C build afir-opt AscendRealizePlannerTest
ctest --test-dir build -R 'AscendRealizePlanner' --output-on-failure
llvm-lit -v build/test/Conversion/ascend-realize*.mlir
llvm-lit -v build/test/Conversion/ascend-full-pipeline-ordinary-smoke.mlir
llvm-lit -v build/test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit bridge extraction**

Run:

```bash
git add lib/Conversion/Ascend/Realize/TranslateMemoryBridge.h \
  lib/Conversion/Ascend/Realize/TranslateMemoryBridge.cpp \
  lib/Conversion/Ascend/Realize/MemoryRealizationDriver.h \
  lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp \
  lib/Conversion/Ascend/CMakeLists.txt \
  test/unittests/Conversion/AscendRealizePlannerTest.cpp
git commit -m "refactor: extract Ascend translate memory bridge"
```

Expected: commit contains no pass CLI or test RUN command changes.

---

## Task 3: Introduce Compute Lowering Internal Context

**Files:**
- Create: `lib/Conversion/Ascend/Translate/KernelIR/ComputeLoweringInternal.h`
- Modify: `lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Create the internal compute header**

Create `lib/Conversion/Ascend/Translate/KernelIR/ComputeLoweringInternal.h` with this initial content:

```c++
//===- ComputeLoweringInternal.h - Ascend compute lowering internals -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTELOWERINGINTERNAL_H
#define ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTELOWERINGINTERNAL_H

#include "Conversion/Ascend/Translate/KernelIR/KernelIRUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir {

struct ComputeLoweringContext {
  func::FuncOp funcOp;
  AscendCBufferContext &ctx;
  MLIRContext *mlirCtx;
  OpBuilder builder;

  ComputeLoweringContext(func::FuncOp funcOp, AscendCBufferContext &ctx)
      : funcOp(funcOp), ctx(ctx), mlirCtx(funcOp.getContext()),
        builder(mlirCtx) {}

  void copyAscendCUnitAttr(Operation *src, Operation *dst) const;
  Value getEnclosingLoopStepBound(Value value, Operation *anchor) const;
  Value getSubviewSizeValue(OpBuilder &builder, Location loc, Value memref,
                            unsigned dim) const;
  Value computeProduct(OpBuilder &builder, Location loc,
                       ArrayRef<Value> dims) const;
  Value dequeTensor(OpBuilder &builder, Location loc, Value queue,
                    Type elemType) const;
  Value allocTensor(OpBuilder &builder, Location loc, Value queue,
                    Type elemType) const;
  Value tbufTensor(OpBuilder &builder, Location loc, int64_t memorySpace,
                   Type elemType) const;
  Value subviewByteOffset(OpBuilder &builder, Location loc, Value memref) const;
  Value tbufSlice(OpBuilder &builder, Location loc, Value memref,
                  Value sizeElems, Value offsetBytes) const;
  Value readTensor(OpBuilder &builder, Location loc, Value memref) const;
  Value writeTensor(OpBuilder &builder, Location loc, Value memref) const;
  scf::ForOp getEnclosingFor(Operation *op) const;
  std::pair<Value, scf::ForOp> allocHoisted(Operation *op, Value queue,
                                            Type elemType, Location loc);
};

LogicalResult lowerScalarFallbackComputes(ComputeLoweringContext &lowering);
LogicalResult lowerTransposeComputes(ComputeLoweringContext &lowering);
LogicalResult lowerReductionComputes(ComputeLoweringContext &lowering);
LogicalResult lowerParallelGenericComputes(ComputeLoweringContext &lowering);
LogicalResult lowerGatherCompute(ComputeLoweringContext &lowering,
                                 linalg::GenericOp genOp,
                                 ArrayRef<linalg::GenericOp> parallelGenericOps);
LogicalResult lowerMatmulComputes(ComputeLoweringContext &lowering);
LogicalResult lowerElementwiseComputes(ComputeLoweringContext &lowering);
LogicalResult lowerFillComputes(ComputeLoweringContext &lowering);
LogicalResult lowerLocalScalarFallbackComputes(ComputeLoweringContext &lowering);

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTELOWERINGINTERNAL_H
```

- [ ] **Step 2: Move local lambdas into `ComputeLoweringContext` methods**

In `ComputeOpConversion.cpp`, include:

```c++
#include "ComputeLoweringInternal.h"
```

Move the bodies of these `convertCompute` lambdas into the corresponding `ComputeLoweringContext` methods:

```text
copyAscendCUnitAttr
getEnclosingLoopStepBound
getSubviewSizeValue
computeProduct
dequeTensor
allocTensor
tbufTensor
subviewByteOffset
tbufSlice
readTensor
writeTensor
getEnclosingFor
allocHoisted
```

Use `this->ctx` where the lambda used `ctx`, and use `this->mlirCtx` where the lambda used `mlirCtx`.

- [ ] **Step 3: Keep `convertCompute` behavior unchanged**

After moving the methods, start `convertCompute` like this:

```c++
LogicalResult convertCompute(func::FuncOp funcOp, AscendCBufferContext &ctx) {
  ComputeLoweringContext lowering(funcOp, ctx);

  if (failed(lowerTransposeComputes(lowering)))
    return failure();
  if (failed(lowerScalarFallbackComputes(lowering)))
    return failure();
  if (failed(lowerReductionComputes(lowering)))
    return failure();
  if (failed(lowerParallelGenericComputes(lowering)))
    return failure();
  if (failed(lowerMatmulComputes(lowering)))
    return failure();
  if (failed(lowerElementwiseComputes(lowering)))
    return failure();
  if (failed(lowerFillComputes(lowering)))
    return failure();
  if (failed(lowerLocalScalarFallbackComputes(lowering)))
    return failure();
  return success();
}
```

During this task, keep the `lower*Computes` function bodies in `ComputeOpConversion.cpp` as temporary functions. The next tasks move them to dedicated files.

- [ ] **Step 4: Add no new CMake sources yet**

Do not modify `lib/Conversion/Ascend/CMakeLists.txt` in this task unless a compiler requires a separate source for the context methods. Prefer keeping methods in `ComputeOpConversion.cpp` until the first family source is created.

- [ ] **Step 5: Build and run compute-focused tests**

Run:

```bash
ninja -C build afir-opt
llvm-lit -v build/test/Conversion/ascend-compute-lower*.mlir
```

Expected: all selected compute-lower tests pass.

- [ ] **Step 6: Commit the context extraction**

Run:

```bash
git add lib/Conversion/Ascend/Translate/KernelIR/ComputeLoweringInternal.h \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp
git commit -m "refactor: introduce Ascend compute lowering context"
```

Expected: behavior-only diff is limited to moving local helper bodies.

---

## Task 4: Move Selected-Tile Lowering Out Of `ComputeOpConversion.cpp`

**Files:**
- Create: `lib/Conversion/Ascend/Translate/KernelIR/ComputeSelectedTileLowering.cpp`
- Modify: `lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Move selected-tile helper definitions**

Move these existing definitions unchanged from `ComputeOpConversion.cpp` into `ComputeSelectedTileLowering.cpp`:

```text
isSupportedRank2Reduction
isSupportedRank2AllParallel
lowerTransposeToLoops
isPureYieldGeneric
isGmAllParallelGeneric
isSimpleDimOrConstantMap
isGmScalarLoopGeneric
lowerPureYieldGenericToLoops
isContiguousRank2View
getRank2RowStrideValue
materializeIndexValue
isKnownZeroIndex
lowerRank2GmTransposeToLocalDataCopy
getSingleDimProjection
isRank2IdentityMap
isRank2SwapPermutation
getStaticIndexValue
isRank2BroadcastTransposeMap
isSupportedSelectedTileMap
isSupportedSelectedAllParallelTileMap
hasSupportedSelectedAllParallelTileMaps
getStaticReductionExtent
validateSelectedReductionTile
validateSelectedAllParallelTile
validateSelectedTileMaps
createRank2TileAlloc
buildTiledOperandSubview
buildTiledAllParallelOperandSubview
rootMemref
inputMemrefRoots
rootIntersects
rootEquals
isBenignShapeOrViewOp
mayWriteAnyRoot
mayReadRoot
touchesAnyRoot
canMoveSelectedTileLoopBeforeWriteback
selectedTileInsertionPoint
isZeroScalarConstant
findLatestZeroFillBefore
materializeSelectedReductionTiles
materializeSelectedTransposeTiles
materializeSelectedAllParallelTiles
```

If a helper is also needed by later compute family files, declare it in `ComputeLoweringInternal.h`; otherwise keep it in the anonymous namespace of `ComputeSelectedTileLowering.cpp`.

- [ ] **Step 2: Include the same dialect headers used by the moved code**

Start `ComputeSelectedTileLowering.cpp` with:

```c++
#include "ComputeLoweringInternal.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
```

- [ ] **Step 3: Wire CMake**

Add:

```cmake
  Translate/KernelIR/ComputeSelectedTileLowering.cpp
```

under `ASCEND_TRANSLATE_KERNELIR_SOURCES`, before `Translate/KernelIR/ComputeOpConversion.cpp`.

- [ ] **Step 4: Build selected-tile tests**

Run:

```bash
ninja -C build afir-opt
llvm-lit -v build/test/Conversion/ascend-compute-lower-selected-tile-materializes-loop.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower-selected-tile-rejects-partial-reduction.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower-selected-tile-rejects-unsupported-map.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower-selected-all-parallel-tile-materializes-loop.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower-selected-all-parallel-tile-fallback.mlir
```

Expected: all selected-tile lit tests pass.

- [ ] **Step 5: Commit selected-tile split**

Run:

```bash
git add lib/Conversion/Ascend/Translate/KernelIR/ComputeSelectedTileLowering.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp \
  lib/Conversion/Ascend/CMakeLists.txt
git commit -m "refactor: split Ascend selected-tile compute lowering"
```

---

## Task 5: Split Scalar Fallback, Transpose, Fill, Matmul, And Elementwise Lowering

**Files:**
- Create: `lib/Conversion/Ascend/Translate/KernelIR/ComputeScalarFallbackLowering.cpp`
- Create: `lib/Conversion/Ascend/Translate/KernelIR/ComputeTransposeLowering.cpp`
- Create: `lib/Conversion/Ascend/Translate/KernelIR/ComputeFillLowering.cpp`
- Create: `lib/Conversion/Ascend/Translate/KernelIR/ComputeMatmulLowering.cpp`
- Create: `lib/Conversion/Ascend/Translate/KernelIR/ComputeElementwiseLowering.cpp`
- Modify: `lib/Conversion/Ascend/Translate/KernelIR/ComputeLoweringInternal.h`
- Modify: `lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Move scalar-loop fallback**

Move these existing definitions and logic to `ComputeScalarFallbackLowering.cpp`:

```text
getDimValue
getVerbatimScalarTypeName
emitStridedGmToLocalCopy
emitLocalToLocalScalarCopy
emitLocalTensorZeroPad
ceilToMultipleIndex
getRootAllocOp
getMemRefDimWithoutDimOp
computeContiguousFlatIndex
lowerProjectedSuffixCopyToSegmentDataCopy
lowerAllParallelGenericToLoops
lowerGmGenericToScalarLoops
lowerMatmulToLoops
lowerBatchMatmulToLoops
lowerFillToScalarLoops
the first linalg.generic GM scalar-loop walk from convertCompute
the final memref.load/memref.store local tensor fallback walks from convertCompute
```

Implement the two exported functions in that file:

```c++
LogicalResult lowerScalarFallbackComputes(ComputeLoweringContext &lowering);
LogicalResult lowerLocalScalarFallbackComputes(ComputeLoweringContext &lowering);
```

- [ ] **Step 2: Move named transpose lowering**

Move the initial `SmallVector<linalg::TransposeOp> transposeOps` block to
`ComputeTransposeLowering.cpp`.

Implement:

```c++
LogicalResult lowerTransposeComputes(ComputeLoweringContext &lowering);
```

Use `lowering.readTensor`, `lowering.writeTensor`, and
`lowering.copyAscendCUnitAttr`. If `lowerRank2GmTransposeToLocalDataCopy` or
`lowerTransposeToLoops` were moved to `ComputeSelectedTileLowering.cpp`, declare
them in `ComputeLoweringInternal.h` with the same signatures and keep their
implementations in one source file only.

- [ ] **Step 3: Move fill lowering**

Move the `SmallVector<linalg::FillOp> fillOps` block and its directly required helpers to `ComputeFillLowering.cpp`.

Implement:

```c++
LogicalResult lowerFillComputes(ComputeLoweringContext &lowering);
```

Use `lowering.writeTensor`, `lowering.copyAscendCUnitAttr`, and helper functions from `ComputeScalarFallbackLowering.cpp` only if they are declared in `ComputeLoweringInternal.h`. If a helper is only used by fill, keep it file-local.

- [ ] **Step 4: Move matmul lowering**

Move the `batchMatmulOps` block and `matmulOps` block to `ComputeMatmulLowering.cpp`.

Implement:

```c++
LogicalResult lowerMatmulComputes(ComputeLoweringContext &lowering);
```

Keep the helper lambdas `toI16`, `getDim`, `buildMmadParams`, `matrixElementCount`, and `batchMatrixByteOffset` local to this file unless another family needs them.

- [ ] **Step 5: Move named elementwise lowering**

Move the `SmallVector<linalg::ElementwiseOp> ewOps` block to `ComputeElementwiseLowering.cpp`.

Implement:

```c++
LogicalResult lowerElementwiseComputes(ComputeLoweringContext &lowering);
```

Keep the supported named kinds unchanged:

```text
add
mul
max_signed
sub
div
min_signed
```

- [ ] **Step 6: Wire CMake**

Add these sources under `ASCEND_TRANSLATE_KERNELIR_SOURCES`, before `ComputeOpConversion.cpp`:

```cmake
  Translate/KernelIR/ComputeScalarFallbackLowering.cpp
  Translate/KernelIR/ComputeTransposeLowering.cpp
  Translate/KernelIR/ComputeFillLowering.cpp
  Translate/KernelIR/ComputeMatmulLowering.cpp
  Translate/KernelIR/ComputeElementwiseLowering.cpp
```

- [ ] **Step 7: Build focused compute tests**

Run:

```bash
ninja -C build afir-opt
llvm-lit -v build/test/Conversion/ascend-compute-lower-transpose.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower-transpose-unsupported.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower-generic-scalar-loop.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower-gm-fill-scalar-loop.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower-matmul-gm.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower-batch-matmul-gm.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower.mlir
llvm-lit -v build/test/Conversion/ascend-compute-lower-fused-generic.mlir
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit first compute family split**

Run:

```bash
git add lib/Conversion/Ascend/Translate/KernelIR/ComputeScalarFallbackLowering.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeTransposeLowering.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeFillLowering.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeMatmulLowering.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeElementwiseLowering.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeLoweringInternal.h \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp \
  lib/Conversion/Ascend/CMakeLists.txt
git commit -m "refactor: split Ascend compute lowering families"
```

---

## Task 6: Split Generic Reduction, Gather, And All-Parallel Lowering

**Files:**
- Create: `lib/Conversion/Ascend/Translate/KernelIR/ComputeReductionLowering.cpp`
- Create: `lib/Conversion/Ascend/Translate/KernelIR/ComputeGatherLowering.cpp`
- Create: `lib/Conversion/Ascend/Translate/KernelIR/ComputeAllParallelLowering.cpp`
- Modify: `lib/Conversion/Ascend/Translate/KernelIR/ComputeLoweringInternal.h`
- Modify: `lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Move reduction generic lowering**

Move the reduction `linalg.generic` block to `ComputeReductionLowering.cpp`.

Implement:

```c++
LogicalResult lowerReductionComputes(ComputeLoweringContext &lowering);
```

Keep reduction support behavior unchanged:

```text
ReductionAdd -> ReduceSum2DL2Op
ReductionMax -> ReduceMax2DL2Op
ReductionMin -> ReduceMin2DL2Op
ReductionMul -> ReduceProd2DL2Op
```

Use `classifyBackendReductionBody`, not the old `classifyPhase5ReductionBody` name.

- [ ] **Step 2: Move gather all-parallel generic helper logic**

Move the `isIndexSelectGeneric` path and its fused pre/post-op handling to `ComputeGatherLowering.cpp`.

Implement:

```c++
LogicalResult lowerGatherCompute(ComputeLoweringContext &lowering,
                                 linalg::GenericOp genOp,
                                 ArrayRef<linalg::GenericOp> parallelGenericOps);
```

Keep the gather attributes unchanged:

```text
ascend::kGatherDimAttr
ascend::kEmbeddingDimAttr
```

Keep `isEmbeddingGeneric` reserved but unused if the current code already reserves it.

- [ ] **Step 3: Move the all-parallel generic coordinator**

Move the all-parallel `linalg.generic` collection and per-op loop to
`ComputeAllParallelLowering.cpp`. Preserve the current order inside that loop:

```text
GM pure-yield copy fallback
GM all-parallel scalar loop fallback
index-select gather path, including pre/post-op erasure
transpose generic path
remaining all-parallel vector path
```

Implement:

```c++
LogicalResult lowerParallelGenericComputes(ComputeLoweringContext &lowering);
```

Keep these local classifiers in this file unless needed by gather:

```text
IndexingMapAnalysis
analyzeIndexingMap
isBroadcastMap
isTransposeGeneric
```

- [ ] **Step 4: Wire CMake**

Add:

```cmake
  Translate/KernelIR/ComputeReductionLowering.cpp
  Translate/KernelIR/ComputeGatherLowering.cpp
  Translate/KernelIR/ComputeAllParallelLowering.cpp
```

under `ASCEND_TRANSLATE_KERNELIR_SOURCES`, before `ComputeOpConversion.cpp`.

- [ ] **Step 5: Verify `ComputeOpConversion.cpp` is now orchestration**

Run:

```bash
wc -l lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp
rg -n "SmallVector<linalg::(GenericOp|MatmulOp|BatchMatmulOp|ElementwiseOp|FillOp)|for \\(linalg::" lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp
```

Expected: line count is substantially below the original 4385 lines, and no family-level lowering loop remains in `ComputeOpConversion.cpp`. It may still contain `convertCompute` and shared context method definitions.

- [ ] **Step 6: Run broad compute and pipeline lit**

Run:

```bash
ninja -C build afir-opt
llvm-lit -v build/test/Conversion/ascend-compute-lower*.mlir
llvm-lit -v build/test/Conversion/ascend-full-pipeline*.mlir
llvm-lit -v build/test/Target/cann-translate*.mlir
```

Expected: all selected lit tests pass.

- [ ] **Step 7: Commit remaining compute split**

Run:

```bash
git add lib/Conversion/Ascend/Translate/KernelIR/ComputeReductionLowering.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeGatherLowering.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeAllParallelLowering.cpp \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeLoweringInternal.h \
  lib/Conversion/Ascend/Translate/KernelIR/ComputeOpConversion.cpp \
  lib/Conversion/Ascend/CMakeLists.txt
git commit -m "refactor: split Ascend generic compute lowering"
```

---

## Task 7: Update Current Documentation And Run xvm Baseline

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Update only current-state vocabulary**

In `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`, update active current-state rows so they use:

```text
Translate memory bridge
backend body classifier
Translate compute lowering
CANN ABI
```

Keep historical plan links and section labels such as `Phase 5`, `Phase 5C`, and `Phase 5C+` unchanged when they refer to completed milestone records.

- [ ] **Step 2: Check no current conversion code uses `Phase5`**

Run:

```bash
rg -n "Phase5|phase5|Phase 5" include/Conversion/Ascend lib/Conversion/Ascend test/unittests/Conversion
```

Expected: no matches.

- [ ] **Step 3: Check external pass surface is unchanged**

Run:

```bash
git diff HEAD~4..HEAD -- include/Conversion/Ascend/Passes.td test/Conversion | rg -n "ascend-compute-lower|ascend-parallelize|ascend-prepare-for-emit|ascend-canonicalize-cann-signature|RUN:"
```

Expected: pass arguments are unchanged; any `RUN:` lines shown are context-only and not altered for internal naming.

- [ ] **Step 4: Run repository formatting checks**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Run authoritative xvm focused verification**

On xvm under `/home/niu/code/Ascend-MLIR`, after syncing the branch and sourcing CANN 9.1:

```bash
source /home/niu/Ascend/latest/set_env.sh
ninja -C build afir-opt AscendLinalgBodyClassifierTest AscendRealizePlannerTest AscendBackendSupportMatrixTest
ctest --test-dir build -R 'AscendLinalgBodyClassifier|AscendRealizePlanner|AscendBackendSupportMatrix|Ascend.*Registry' --output-on-failure
llvm-lit -v build/test/Conversion/ascend-compute-lower*.mlir
llvm-lit -v build/test/Conversion/ascend-realize*.mlir
llvm-lit -v build/test/Conversion/ascend-full-pipeline*.mlir
llvm-lit -v build/test/Target/cann-translate*.mlir
```

Expected: all commands return `RC=0`.

- [ ] **Step 6: Run xvm CPU-simulation baselines**

On xvm:

```bash
source /home/niu/Ascend/latest/set_env.sh
bash test/tools/runtime/run_runtime.sh
bash test/tools/runtime/run_simbackend_examples.sh
bash test/tools/examples/example_pipelines.sh
```

Expected:

```text
run_runtime.sh: RC=0 and runtime tests report 113 passed, 0 failed
run_simbackend_examples.sh: RC=0 and 6 SimBackend examples pass
example_pipelines.sh: RC=0 and 6 example pipelines pass
```

- [ ] **Step 7: Commit current documentation update**

Run:

```bash
git add docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
git commit -m "docs: update Ascend conversion current vocabulary"
```

Expected: commit only contains tracking-doc wording updates.

---

## Final Review Checklist

- [ ] `rg -n "Phase5|phase5|Phase 5" include/Conversion/Ascend lib/Conversion/Ascend test/unittests/Conversion` has no current-code matches.
- [ ] `include/Conversion/Ascend/Passes.td` keeps all pass arguments unchanged.
- [ ] `test/Conversion` RUN lines are unchanged except context from unrelated local changes.
- [ ] `MemoryRealizationDriver.cpp` no longer contains Translate bridge pattern-matching details.
- [ ] `TranslateMemoryBridge.cpp` owns VECOUT final-output bridge and cube-to-vector bridge implementation.
- [ ] `ComputeOpConversion.cpp` is no longer the sole home for compute family lowering.
- [ ] New compute files map to behavior families, not historical milestone names.
- [ ] Local or xvm focused lit passes.
- [ ] xvm runtime baselines pass before claiming the refactor complete.
