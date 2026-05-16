# Phase 3: Realize IR Instantiation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `materialization-mode=tiled-linalg` and `materialization-mode=full-realize` to `AscendRealizePass`, driving `tileUsingSCF` and `BufferizationDriver` from the `ScheduleDecision` tile shape already encoded on IR as `DenseI64ArrayAttr`.

**Architecture:** Two-step. Step A (`tiled-linalg`): read `ascend.schedule.selected_tile_shape` from each primary op, call `scf::tileUsingSCF`, fuse epilogue producers via `scf::tileAndFuseProducerOfSlice`. Step B (`full-realize`): run existing `BufferizationDriver::runOneShotBufferize` on the tiled-tensor IR, then annotate memory spaces. Existing `plan-only`/`one-shot-bufferize`/`memory-space-annotate` modes are untouched.

**Tech Stack:** MLIR `scf::tileUsingSCF` + `scf::tileAndFuseProducerOfSlice` (from `mlir/Dialect/SCF/Transforms/TileUsingInterface.h`); existing `BufferizationDriver`; existing `MemoryRealizationDriver`; MLIR `IRRewriter`.

---

## File Map

| Action | File |
|--------|------|
| Create | `lib/Conversion/Ascend/Realize/TilingRealizationDriver.h` |
| Create | `lib/Conversion/Ascend/Realize/TilingRealizationDriver.cpp` |
| Modify | `lib/Conversion/Ascend/Realize/RealizePass.cpp` |
| Modify | `include/Conversion/Passes.td` |
| Create | `test/unittests/Conversion/AscendRealizePassTiledLinalgTest.cpp` |
| Create | `test/lit/Conversion/AscendRealize/tiled-linalg-matmul.mlir` |
| Create | `test/lit/Conversion/AscendRealize/full-realize-matmul.mlir` |

---

### Task 1: TilingRealizationDriver header

**Files:**
- Create: `lib/Conversion/Ascend/Realize/TilingRealizationDriver.h`

- [ ] **Step 1: Write header**

```cpp
//===- TilingRealizationDriver.h - Ascend tiling realization ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TILINGREALIZATIONDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TILINGREALIZATIONDRIVER_H

#include "RealizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::ascend::realize {

/// Drives Schedule-decision-based tiling of linalg ops in a module using
/// scf::tileUsingSCF. Operates on tensor IR (pre-bufferize).
class TilingRealizationDriver {
public:
  /// Tile every linalg.generic / linalg.matmul that carries
  /// ascend.schedule.selected_tile_shape. Replaces the op in-place with a
  /// scf.for nest. Returns failure if any tile shape is malformed.
  LogicalResult tileModule(ModuleOp module) const;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_TILINGREALIZATIONDRIVER_H
```

- [ ] **Step 2: Commit**

```bash
git add lib/Conversion/Ascend/Realize/TilingRealizationDriver.h
git commit -m "feat(realize): add TilingRealizationDriver header"
```

---

### Task 2: TilingRealizationDriver implementation

**Files:**
- Create: `lib/Conversion/Ascend/Realize/TilingRealizationDriver.cpp`

The driver walks the module, collects linalg ops that carry `ascend.schedule.selected_tile_shape`, calls `scf::tileUsingSCF`, then walks the generated slices and calls `scf::tileAndFuseProducerOfSlice` for each.

- [ ] **Step 1: Write failing smoke test (see Task 3) first — TDD gate**

Skip ahead to Task 3 Step 1, come back here once the test file exists and fails to link.

- [ ] **Step 2: Write TilingRealizationDriver.cpp**

```cpp
//===- TilingRealizationDriver.cpp - Ascend tiling realization --------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "TilingRealizationDriver.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::afir::ascend;
using namespace mlir::afir::ascend::realize;

namespace {

// Collect ops-to-tile in program order so we tile outer-most first.
SmallVector<Operation *> collectTileableOps(ModuleOp module) {
  SmallVector<Operation *> ops;
  module.walk([&](Operation *op) {
    if (!op->getAttrOfType<DenseI64ArrayAttr>(kScheduleSelectedTileShapeAttr))
      return;
    if (!isa<TilingInterface>(op))
      return;
    ops.push_back(op);
  });
  return ops;
}

FailureOr<SmallVector<OpFoldResult>>
buildTileSizes(MLIRContext *ctx, DenseI64ArrayAttr tileShapeAttr) {
  SmallVector<OpFoldResult> sizes;
  for (int64_t sz : tileShapeAttr.asArrayRef()) {
    if (sz == 0 || ShapedType::isDynamic(sz))
      sizes.push_back(OpFoldResult(IntegerAttr::get(
          IntegerType::get(ctx, 64), 0)));
    else
      sizes.push_back(OpFoldResult(IntegerAttr::get(
          IntegerType::get(ctx, 64), sz)));
  }
  return sizes;
}

} // namespace

namespace mlir::afir::ascend::realize {

LogicalResult TilingRealizationDriver::tileModule(ModuleOp module) const {
  IRRewriter rewriter(module.getContext());

  SmallVector<Operation *> toTile = collectTileableOps(module);
  for (Operation *op : toTile) {
    auto tileShapeAttr =
        op->getAttrOfType<DenseI64ArrayAttr>(kScheduleSelectedTileShapeAttr);

    FailureOr<SmallVector<OpFoldResult>> tileSizes =
        buildTileSizes(module.getContext(), tileShapeAttr);
    if (failed(tileSizes))
      return op->emitError("TilingRealizationDriver: invalid tile shape attr");

    scf::SCFTilingOptions opts;
    opts.setTileSizes(*tileSizes);

    rewriter.setInsertionPoint(op);
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, cast<TilingInterface>(op), opts);
    if (failed(tilingResult))
      return op->emitError(
          "TilingRealizationDriver: scf::tileUsingSCF failed");

    // Fuse producer slices generated during tiling.
    for (Operation *sliceOp : tilingResult->generatedSlices) {
      auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(sliceOp);
      if (!extractSlice)
        continue;
      scf::tileAndFuseProducerOfSlice(rewriter, extractSlice,
                                      tilingResult->loops);
    }

    rewriter.replaceOp(op, tilingResult->replacements);
  }
  return success();
}

} // namespace mlir::afir::ascend::realize
```

- [ ] **Step 3: Verify it compiles on xvm**

```bash
ssh xvm@orb "cd /mnt/mac/Volumes/GM9/code/Ascend-MLIR && ninja -C build lib/Conversion/Ascend/Realize/CMakeFiles/AscendRealize.dir/TilingRealizationDriver.cpp.o -j4 2>&1 | tail -30"
```

Expected: no errors (may have warnings; fix any errors).

- [ ] **Step 4: Commit**

```bash
git add lib/Conversion/Ascend/Realize/TilingRealizationDriver.cpp
git commit -m "feat(realize): implement TilingRealizationDriver using scf::tileUsingSCF"
```

---

### Task 3: Unit test for TilingRealizationDriver

**Files:**
- Create: `test/unittests/Conversion/AscendRealizePassTiledLinalgTest.cpp`

This test builds a small `linalg.matmul` with `ascend.schedule.selected_tile_shape = [4, 4, 4]` attached, runs `TilingRealizationDriver::tileModule`, and checks that `scf.for` loops are generated.

- [ ] **Step 1: Write failing test**

```cpp
//===- AscendRealizePassTiledLinalgTest.cpp - TilingRealizationDriver tests ===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/TilingRealizationDriver.h"
#include "Conversion/Ascend/Common/Attributes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::afir::ascend;
using namespace mlir::afir::ascend::realize;

namespace {

class TilingRealizationDriverTest : public ::testing::Test {
protected:
  MLIRContext context;
  OpBuilder builder{&context};

  void SetUp() override {
    context.loadDialect<func::FuncDialect, linalg::LinalgDialect,
                        arith::ArithDialect, tensor::TensorDialect,
                        scf::SCFDialect>();
  }
};

// Build:
//   func @matmul(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>,
//                %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
//     %0 = linalg.matmul ins(%A, %B) outs(%C)  [tile_shape=[4,4,4]]
//     return %0
//   }
ModuleOp buildMatmulModule(MLIRContext &ctx, OpBuilder &b) {
  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  b.setInsertionPointToEnd(module.getBody());

  auto f32 = b.getF32Type();
  auto tensorTy = RankedTensorType::get({8, 8}, f32);
  auto funcTy = FunctionType::get(&ctx, {tensorTy, tensorTy, tensorTy},
                                  {tensorTy});
  auto funcOp = b.create<func::FuncOp>(UnknownLoc::get(&ctx), "matmul", funcTy);
  auto *block = funcOp.addEntryBlock();
  b.setInsertionPointToStart(block);

  Value A = block->getArgument(0);
  Value B = block->getArgument(1);
  Value C = block->getArgument(2);

  auto matmul = b.create<linalg::MatmulOp>(
      UnknownLoc::get(&ctx), TypeRange{tensorTy},
      ValueRange{A, B}, ValueRange{C});

  // Attach tile shape [4, 4, 4]
  matmul->setAttr(kScheduleSelectedTileShapeAttr,
                  b.getDenseI64ArrayAttr({4, 4, 4}));

  b.create<func::ReturnOp>(UnknownLoc::get(&ctx),
                           matmul.getResult(0));
  return module;
}

TEST_F(TilingRealizationDriverTest, TilesMatmulIntoScfForNest) {
  ModuleOp module = buildMatmulModule(context, builder);

  TilingRealizationDriver driver;
  ASSERT_TRUE(succeeded(driver.tileModule(module)));

  // After tiling: linalg.matmul should no longer exist at module top-level.
  // At least one scf.for must be present inside the function.
  bool hasScfFor = false;
  bool hasLinalgMatmulAtTopLevel = false;
  module.walk([&](Operation *op) {
    if (isa<scf::ForOp>(op))
      hasScfFor = true;
    if (isa<linalg::MatmulOp>(op) &&
        isa<func::FuncOp>(op->getParentOp()))
      hasLinalgMatmulAtTopLevel = true;
  });

  EXPECT_TRUE(hasScfFor) << "Expected at least one scf.for after tiling";
  EXPECT_FALSE(hasLinalgMatmulAtTopLevel)
      << "linalg.matmul should be inside scf.for loops after tiling";
}

TEST_F(TilingRealizationDriverTest, SkipsOpsWithoutTileShapeAttr) {
  ModuleOp module = buildMatmulModule(context, builder);
  // Remove the tile shape attr so no tiling should happen.
  module.walk([](linalg::MatmulOp op) {
    op->removeAttr(kScheduleSelectedTileShapeAttr);
  });

  TilingRealizationDriver driver;
  ASSERT_TRUE(succeeded(driver.tileModule(module)));

  bool hasScfFor = false;
  module.walk([&](scf::ForOp) { hasScfFor = true; });
  EXPECT_FALSE(hasScfFor) << "No scf.for expected when tile shape attr absent";
}

} // namespace
```

- [ ] **Step 2: Run to verify compile fails (missing header/symbol)**

```bash
ssh xvm@orb "cd /mnt/mac/Volumes/GM9/code/Ascend-MLIR && ninja -C build test/unittests/Conversion/CMakeFiles/AscendConversionTests.dir/AscendRealizePassTiledLinalgTest.cpp.o 2>&1 | tail -20"
```

Expected: compilation failure — `TilingRealizationDriver.h` not yet in CMake.

- [ ] **Step 3: Wire test into CMakeLists.txt**

Add the test file to the existing `test/unittests/Conversion/CMakeLists.txt`:

```cmake
AscendRealizePassTiledLinalgTest.cpp
```

Also add these link libraries if not already present:

```cmake
MLIRLinalgDialect
MLIRSCFTransforms
MLIRTensorDialect
```

- [ ] **Step 4: Run test — expect PASS**

```bash
ssh xvm@orb "cd /mnt/mac/Volumes/GM9/code/Ascend-MLIR && ninja -C build -j4 AscendConversionTests && ctest --test-dir build -R 'AscendRealizePassTiledLinalg' -V 2>&1 | tail -30"
```

Expected: 2 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add test/unittests/Conversion/AscendRealizePassTiledLinalgTest.cpp \
        test/unittests/Conversion/CMakeLists.txt
git commit -m "test(realize): add TilingRealizationDriver unit tests"
```

---

### Task 4: Wire tiled-linalg mode into RealizePass

**Files:**
- Modify: `include/Conversion/Passes.td`
- Modify: `lib/Conversion/Ascend/Realize/RealizePass.cpp`

- [ ] **Step 1: Add mode constant and validation in RealizePass.cpp**

In `RealizePass.cpp`, add at the top of the anonymous namespace (after `kMemorySpaceAnnotateMaterializationMode`):

```cpp
constexpr llvm::StringLiteral kTiledLinalgMaterializationMode = "tiled-linalg";
constexpr llvm::StringLiteral kFullRealizeMaterializationMode = "full-realize";
```

Update `isSupportedMaterializationMode`:

```cpp
static bool isSupportedMaterializationMode(StringRef mode) {
  return mode == kPlanOnlyMaterializationMode ||
         mode == kOneShotBufferizeMaterializationMode ||
         mode == kMemorySpaceAnnotateMaterializationMode ||
         mode == kTiledLinalgMaterializationMode ||
         mode == kFullRealizeMaterializationMode;
}
```

- [ ] **Step 2: Add TilingRealizationDriver include**

At the top of `RealizePass.cpp` includes block:

```cpp
#include "TilingRealizationDriver.h"
```

- [ ] **Step 3: Add tiled-linalg execution branch in runOnOperation()**

Insert after the existing `buildMVPRealizePlans` call block (around line 275, just before the `if (materializationMode == kOneShotBufferizeMaterializationMode...)` block):

```cpp
    if (materializationMode == kTiledLinalgMaterializationMode ||
        materializationMode == kFullRealizeMaterializationMode) {
      TilingRealizationDriver tilingDriver;
      if (failed(tilingDriver.tileModule(getOperation()))) {
        getOperation()->emitError()
            << "ascend-realize tiled-linalg: tiling failed";
        signalPassFailure();
        return;
      }
    }
```

- [ ] **Step 4: Add full-realize branch (bufferize + memory space)**

Replace the existing `kOneShotBufferizeMaterializationMode` block (lines ~275-284) with:

```cpp
    if (materializationMode == kOneShotBufferizeMaterializationMode ||
        materializationMode == kMemorySpaceAnnotateMaterializationMode ||
        materializationMode == kFullRealizeMaterializationMode) {
      BufferizationDriver bufferizationDriver;
      if (failed(bufferizationDriver.runOneShotBufferize(getOperation()))) {
        getOperation()->emitError()
            << "ascend-realize failed to run one-shot bufferize";
        signalPassFailure();
        return;
      }
    }

    if (materializationMode == kMemorySpaceAnnotateMaterializationMode ||
        materializationMode == kFullRealizeMaterializationMode) {
      MemoryRealizationDriver memoryRealizationDriver;
      if (failed(memoryRealizationDriver.materialize(
              getOperation(), *bundles,
              MemoryRealizationMode::MemorySpaceAnnotate))) {
        getOperation()->emitError()
            << "ascend-realize failed to materialize Phase 5 memory bridge";
        signalPassFailure();
        return;
      }
    }
```

(This replaces two separate blocks with a single combined set.)

- [ ] **Step 5: Update Passes.td description**

Find the `AscendRealizePass` description field in `include/Conversion/Passes.td` and update the `materialization-mode` option description:

```tablegen
    Option<"materializationMode", "materialization-mode", "std::string",
           /*default=*/"\"plan-only\"",
           "Realize materialization mode: plan-only, one-shot-bufferize, memory-space-annotate, tiled-linalg, or full-realize">,
```

- [ ] **Step 6: Verify build**

```bash
ssh xvm@orb "cd /mnt/mac/Volumes/GM9/code/Ascend-MLIR && ninja -C build -j4 ascend-mlir-opt 2>&1 | tail -20"
```

Expected: builds cleanly.

- [ ] **Step 7: Commit**

```bash
git add include/Conversion/Passes.td \
        lib/Conversion/Ascend/Realize/RealizePass.cpp
git commit -m "feat(realize): add tiled-linalg and full-realize materialization modes"
```

---

### Task 5: LIT test — tiled-linalg mode

**Files:**
- Create: `test/lit/Conversion/AscendRealize/tiled-linalg-matmul.mlir`

- [ ] **Step 1: Find existing LIT test directory**

```bash
ls /Volumes/GM9/code/Ascend-MLIR/test/lit/Conversion/AscendRealize/
```

Use whichever directory already holds `AscendRealize` LIT tests. If none exists, check `test/lit/Conversion/` and create the directory.

- [ ] **Step 2: Write failing LIT test**

```mlir
// RUN: ascend-mlir-opt %s \
// RUN:   --ascend-realize="materialization-mode=tiled-linalg" \
// RUN: | FileCheck %s

// CHECK-LABEL: func @matmul_tiled
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               linalg.matmul

func.func @matmul_tiled(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>,
                         %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.matmul {ascend.kernel = "k0",
                       ascend.schedule.decision_id = "d0",
                       ascend.schedule.structured_lowering = "loop_skeleton_v0",
                       ascend.schedule.selected_tile_shape = dense<[4, 4, 4]> : tensor<3xi64>}
       ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>)
       outs(%C : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

> **Note:** `ascend.schedule.selected_tile_shape` is a `DenseI64ArrayAttr`. In MLIR textual IR, `DenseI64ArrayAttr` is spelled as `array<i64: 4, 4, 4>`. Adjust the test syntax to match how `mlir-opt` parses it; run `ascend-mlir-opt --print-ir-before-all` to observe the concrete serialization from a real SchedulePass run and mirror it exactly.

- [ ] **Step 3: Run and iterate until CHECK lines pass**

```bash
ssh xvm@orb "cd /mnt/mac/Volumes/GM9/code/Ascend-MLIR && ctest --test-dir build -R 'tiled-linalg-matmul' -V 2>&1 | tail -40"
```

Common failure: `DenseI64ArrayAttr` textual format. Fix CHECK lines to match actual output.

- [ ] **Step 4: Commit**

```bash
git add test/lit/Conversion/AscendRealize/tiled-linalg-matmul.mlir
git commit -m "test(realize): add tiled-linalg LIT test for matmul tile shape"
```

---

### Task 6: LIT test — full-realize mode

**Files:**
- Create: `test/lit/Conversion/AscendRealize/full-realize-matmul.mlir`

- [ ] **Step 1: Write test**

```mlir
// RUN: ascend-mlir-opt %s \
// RUN:   --ascend-realize="materialization-mode=full-realize" \
// RUN: | FileCheck %s

// After full-realize: tensor IR must be gone, memref must appear.
// CHECK-LABEL: func @matmul_full_realize
// CHECK-NOT:     tensor<
// CHECK:         memref<

func.func @matmul_full_realize(%A: tensor<8x8xf32>, %B: tensor<8x8xf32>,
                                %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.matmul {ascend.kernel = "k0",
                       ascend.schedule.decision_id = "d0",
                       ascend.schedule.structured_lowering = "loop_skeleton_v0",
                       ascend.schedule.selected_tile_shape = array<i64: 4, 4, 4>}
       ins(%A, %B : tensor<8x8xf32>, tensor<8x8xf32>)
       outs(%C : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
```

- [ ] **Step 2: Run, fix CHECK patterns to match actual output**

```bash
ssh xvm@orb "cd /mnt/mac/Volumes/GM9/code/Ascend-MLIR && ctest --test-dir build -R 'full-realize-matmul' -V 2>&1 | tail -40"
```

- [ ] **Step 3: Commit**

```bash
git add test/lit/Conversion/AscendRealize/full-realize-matmul.mlir
git commit -m "test(realize): add full-realize LIT test for memref output"
```

---

### Task 7: Regression guard — existing tests still pass

- [ ] **Step 1: Run all Ascend tests**

```bash
ssh xvm@orb "cd /mnt/mac/Volumes/GM9/code/Ascend-MLIR && ctest --test-dir build -R 'Ascend' -j4 2>&1 | tail -30"
```

Expected: all previously passing tests still pass.

- [ ] **Step 2: If any fail, investigate and fix before proceeding**

Look at the failure output, identify whether the tiled-linalg/full-realize changes accidentally affected the `buildMVPRealizePlans` path (they should not — the tiling step only runs for the two new modes). Common issue: import ordering in `CMakeLists.txt` causing link failures.

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix(realize): resolve regression in existing realize tests"
```

---

### Task 8: CMakeLists.txt — add TilingRealizationDriver to build

**Files:**
- Modify: `lib/Conversion/Ascend/Realize/CMakeLists.txt`

- [ ] **Step 1: Read current CMakeLists.txt**

```bash
cat /Volumes/GM9/code/Ascend-MLIR/lib/Conversion/Ascend/Realize/CMakeLists.txt
```

- [ ] **Step 2: Add TilingRealizationDriver.cpp and required link deps**

Find the `add_mlir_library` (or `add_library`) call and add:

```cmake
  TilingRealizationDriver.cpp
```

Add these to `LINK_LIBS` if not already present:

```cmake
  MLIRSCFTransforms
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTilingInterface
```

- [ ] **Step 3: Full rebuild and all-test run**

```bash
ssh xvm@orb "cd /mnt/mac/Volumes/GM9/code/Ascend-MLIR && ninja -C build -j6 ascend-mlir-opt AscendConversionTests && ctest --test-dir build -R 'Ascend' -j4 2>&1 | tail -30"
```

Expected: all PASSED.

- [ ] **Step 4: Commit**

```bash
git add lib/Conversion/Ascend/Realize/CMakeLists.txt
git commit -m "build: wire TilingRealizationDriver into Realize CMakeLists"
```

---

## Phase 3 Acceptance Checklist

- [ ] `materialization-mode=tiled-linalg`: a matmul with tile shape `[4,4,4]` produces 3 `scf.for` loops; LIT FileCheck passes
- [ ] `materialization-mode=full-realize`: same matmul produces memref IR with no tensor types; LIT FileCheck passes
- [ ] `TilingRealizationDriver` unit tests: 2/2 PASS
- [ ] All previously passing Ascend LIT and unit tests: no regression
- [ ] `plan-only`, `one-shot-bufferize`, `memory-space-annotate` modes: behavior unchanged

---

## Implementation Order Notes

Do tasks in this order to avoid blocking on CMake:

1. **Task 8 first** (CMakeLists) — must exist so Tasks 2 and 3 can compile
2. Task 1 (header)
3. Task 2 (implementation) — iterate with Task 3 in TDD order
4. Task 3 (unit test)
5. Task 4 (wire into RealizePass)
6. Task 5 (tiled-linalg LIT)
7. Task 6 (full-realize LIT)
8. Task 7 (regression guard)
