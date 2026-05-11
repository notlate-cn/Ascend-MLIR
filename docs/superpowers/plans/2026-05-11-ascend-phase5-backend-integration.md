# Ascend Phase 5 Backend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 5 Translate / Runtime Artifact MVP by adding official Ascend backend entry points, explicit support-matrix validation, and generated runtime artifacts while reusing the existing AscendC / CANN backend implementation.

**Architecture:** Add `Conversion/Ascend/Backend` as the formal Phase 5 boundary. The new passes wrap or share existing `LinalgToAscendC`, `AscendCParallelize`, `AscendCPrepareForEmit`, `CanonicalizeCannSignature`, and `CannKernel` translation code, then add fail-closed verifiers and artifact emitters around those reused implementations.

**Tech Stack:** MLIR/LLVM C++ pass infrastructure, AscendC dialect, EmitAsc dialect, LIT/FileCheck, GoogleTest, `afir-opt`, `afir-translate`, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Create `include/Conversion/Ascend/Backend/BackendSupportMatrix.h`
- Create `include/Conversion/Ascend/Backend/ComputeLoweringPass.h`
- Create `include/Conversion/Ascend/Backend/BackendWrapperPasses.h`
- Create `lib/Conversion/Ascend/Backend/BackendSupportMatrix.cpp`
- Create `lib/Conversion/Ascend/Backend/ComputeLoweringPass.cpp`
- Create `lib/Conversion/Ascend/Backend/BackendWrapperPasses.cpp`
- Modify `include/Conversion/Passes.h`
- Modify `include/Conversion/Passes.td`
- Modify `include/Conversion/LinalgToAscendC/LinalgToAscendCUtils.h`
- Modify `lib/Conversion/LinalgToAscendC/LinalgToAscendCPass.cpp`
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
- Modify `tools/afir-opt/CMakeLists.txt` only if new link dependencies are required after local build.
- Create `test/unittests/Conversion/AscendBackendSupportMatrixTest.cpp`
- Modify `test/unittests/Conversion/CMakeLists.txt`
- Create `test/Conversion/ascend-compute-lower.mlir`
- Create `test/Conversion/ascend-backend-abi-wrappers.mlir`
- Create `include/Target/CannKernel/CannRuntimeArtifacts.h`
- Create `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`
- Modify `include/Target/CannKernel/CannTranslation.h`
- Modify `lib/Target/CannKernel/CannTranslation.cpp`
- Modify `lib/Target/CannKernel/CMakeLists.txt`
- Modify `tools/afir-translate/afir-translate.cpp`
- Create `test/Target/cann-translate-runtime-artifacts.mlir`
- Create `test/Target/cann-translate-runtime-artifacts-unsupported.mlir`
- Create `test/Conversion/ascend-phase5-transformer-dynamic-smoke.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains local user instructions and unrelated working-tree changes.

## Design Constraints

- Do not copy an entire backend implementation into `Conversion/Ascend`.
- Keep old pass entry points available until a later deletion plan.
- New Phase 5 entry points must be fail-closed: unsupported combinations produce diagnostics and pass failure.
- Phase 5 must not introduce new schedule, placement, or memory-planning decisions.
- The code surface must remain versionless: no code namespace, source path, CMake target, IR attr, or test target may contain `v2` / `V2`.
- `schema_version = "2.0"` is allowed only as the runtime JSON schema value.

## Task 1: Backend Support Matrix

**Files:**
- Create `include/Conversion/Ascend/Backend/BackendSupportMatrix.h`
- Create `lib/Conversion/Ascend/Backend/BackendSupportMatrix.cpp`
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
- Create `test/unittests/Conversion/AscendBackendSupportMatrixTest.cpp`
- Modify `test/unittests/Conversion/CMakeLists.txt`

- [ ] **Step 1: Write the failing unit test**

Create `test/unittests/Conversion/AscendBackendSupportMatrixTest.cpp` with these checks:

```cpp
//===- AscendBackendSupportMatrixTest.cpp - Ascend backend tests ---===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/BackendSupportMatrix.h"

#include "gtest/gtest.h"

using namespace mlir::afir::ascend::backend;

TEST(AscendBackendSupportMatrixTest, SupportsKnownMovementPaths) {
  AscendBackendSupportMatrix matrix;
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::GM, MemorySpace::A1));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::GM, MemorySpace::B1));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::GM,
                                             MemorySpace::VECIN));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::A1, MemorySpace::A2));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::B1, MemorySpace::B2));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::CO1,
                                             MemorySpace::VECIN));
  EXPECT_TRUE(matrix.isSupportedMovementPath(MemorySpace::VECOUT,
                                             MemorySpace::GM));
}

TEST(AscendBackendSupportMatrixTest, RejectsUnknownMovementPaths) {
  AscendBackendSupportMatrix matrix;
  EXPECT_FALSE(matrix.isSupportedMovementPath(MemorySpace::GM,
                                              MemorySpace::VECCALC));
  EXPECT_FALSE(matrix.isSupportedMovementPath(MemorySpace::VECCALC,
                                              MemorySpace::GM));

  UnsupportedReason reason =
      matrix.explainMovementPath(MemorySpace::VECCALC, MemorySpace::GM);
  EXPECT_EQ(reason.category, "movement");
  EXPECT_EQ(reason.detail, "unsupported movement path VECCALC -> GM");
}

TEST(AscendBackendSupportMatrixTest, ConvertsIntegerMemorySpaces) {
  EXPECT_EQ(parseMemorySpace(0), MemorySpace::GM);
  EXPECT_EQ(parseMemorySpace(1), MemorySpace::A1);
  EXPECT_EQ(parseMemorySpace(2), MemorySpace::A2);
  EXPECT_EQ(parseMemorySpace(3), MemorySpace::B1);
  EXPECT_EQ(parseMemorySpace(4), MemorySpace::B2);
  EXPECT_EQ(parseMemorySpace(7), MemorySpace::CO1);
  EXPECT_EQ(parseMemorySpace(9), MemorySpace::VECIN);
  EXPECT_EQ(parseMemorySpace(10), MemorySpace::VECOUT);
  EXPECT_EQ(parseMemorySpace(11), MemorySpace::VECCALC);
  EXPECT_EQ(parseMemorySpace(99), MemorySpace::Unknown);
}

TEST(AscendBackendSupportMatrixTest, SupportsKnownComputeKinds) {
  AscendBackendSupportMatrix matrix;
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::Matmul));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::Fill));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseAdd));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ElementwiseMax));
  EXPECT_TRUE(matrix.isSupportedComputeKind(ComputeKind::ReductionAdd));
  EXPECT_FALSE(matrix.isSupportedComputeKind(ComputeKind::Unknown));
}
```

Register the test in `test/unittests/Conversion/CMakeLists.txt`:

```cmake
add_executable(AscendBackendSupportMatrixTest
  AscendBackendSupportMatrixTest.cpp
)

target_link_libraries(AscendBackendSupportMatrixTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendBackendSupportMatrixTest)

add_test(NAME AscendBackendSupportMatrixTest
  COMMAND $<TARGET_FILE:AscendBackendSupportMatrixTest>
)
```

- [ ] **Step 2: Run RED verification in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build AscendBackendSupportMatrixTest'
```

Expected: build fails because `Conversion/Ascend/Backend/BackendSupportMatrix.h` does not exist.

- [ ] **Step 3: Add the support matrix API**

Create `include/Conversion/Ascend/Backend/BackendSupportMatrix.h`:

```cpp
//===- BackendSupportMatrix.h - Ascend backend support matrix ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_SUPPORT_MATRIX_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_SUPPORT_MATRIX_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace mlir::afir::ascend::backend {

enum class MemorySpace : int64_t {
  Unknown = -1,
  GM = 0,
  A1 = 1,
  A2 = 2,
  B1 = 3,
  B2 = 4,
  CO1 = 7,
  VECIN = 9,
  VECOUT = 10,
  VECCALC = 11,
};

enum class ComputeKind {
  Unknown,
  Matmul,
  Fill,
  ElementwiseAdd,
  ElementwiseMax,
  ReductionAdd,
};

struct UnsupportedReason {
  std::string category;
  std::string detail;
};

MemorySpace parseMemorySpace(int64_t value);
llvm::StringRef stringifyMemorySpace(MemorySpace space);
llvm::StringRef stringifyComputeKind(ComputeKind kind);

class AscendBackendSupportMatrix {
public:
  bool isSupportedMovementPath(MemorySpace source,
                               MemorySpace target) const;
  UnsupportedReason explainMovementPath(MemorySpace source,
                                        MemorySpace target) const;

  bool isSupportedComputeKind(ComputeKind kind) const;
  UnsupportedReason explainComputeKind(ComputeKind kind) const;
};

} // namespace mlir::afir::ascend::backend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_SUPPORT_MATRIX_H
```

- [ ] **Step 4: Add the support matrix implementation**

Create `lib/Conversion/Ascend/Backend/BackendSupportMatrix.cpp`:

```cpp
//===- BackendSupportMatrix.cpp - Ascend backend support matrix -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/BackendSupportMatrix.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir::afir::ascend::backend {

MemorySpace parseMemorySpace(int64_t value) {
  switch (value) {
  case 0:
    return MemorySpace::GM;
  case 1:
    return MemorySpace::A1;
  case 2:
    return MemorySpace::A2;
  case 3:
    return MemorySpace::B1;
  case 4:
    return MemorySpace::B2;
  case 7:
    return MemorySpace::CO1;
  case 9:
    return MemorySpace::VECIN;
  case 10:
    return MemorySpace::VECOUT;
  case 11:
    return MemorySpace::VECCALC;
  default:
    return MemorySpace::Unknown;
  }
}

llvm::StringRef stringifyMemorySpace(MemorySpace space) {
  switch (space) {
  case MemorySpace::GM:
    return "GM";
  case MemorySpace::A1:
    return "A1";
  case MemorySpace::A2:
    return "A2";
  case MemorySpace::B1:
    return "B1";
  case MemorySpace::B2:
    return "B2";
  case MemorySpace::CO1:
    return "CO1";
  case MemorySpace::VECIN:
    return "VECIN";
  case MemorySpace::VECOUT:
    return "VECOUT";
  case MemorySpace::VECCALC:
    return "VECCALC";
  case MemorySpace::Unknown:
    return "Unknown";
  }
  return "Unknown";
}

llvm::StringRef stringifyComputeKind(ComputeKind kind) {
  switch (kind) {
  case ComputeKind::Matmul:
    return "matmul";
  case ComputeKind::Fill:
    return "fill";
  case ComputeKind::ElementwiseAdd:
    return "elementwise_add";
  case ComputeKind::ElementwiseMax:
    return "elementwise_max";
  case ComputeKind::ReductionAdd:
    return "reduction_add";
  case ComputeKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

bool AscendBackendSupportMatrix::isSupportedMovementPath(
    MemorySpace source, MemorySpace target) const {
  return (source == MemorySpace::GM &&
          (target == MemorySpace::A1 || target == MemorySpace::B1 ||
           target == MemorySpace::VECIN)) ||
         (source == MemorySpace::A1 && target == MemorySpace::A2) ||
         (source == MemorySpace::B1 && target == MemorySpace::B2) ||
         (source == MemorySpace::CO1 && target == MemorySpace::VECIN) ||
         (source == MemorySpace::VECOUT && target == MemorySpace::GM);
}

UnsupportedReason AscendBackendSupportMatrix::explainMovementPath(
    MemorySpace source, MemorySpace target) const {
  if (isSupportedMovementPath(source, target))
    return {"movement", ""};
  return {"movement",
          llvm::formatv("unsupported movement path {0} -> {1}",
                        stringifyMemorySpace(source),
                        stringifyMemorySpace(target))
              .str()};
}

bool AscendBackendSupportMatrix::isSupportedComputeKind(
    ComputeKind kind) const {
  switch (kind) {
  case ComputeKind::Matmul:
  case ComputeKind::Fill:
  case ComputeKind::ElementwiseAdd:
  case ComputeKind::ElementwiseMax:
  case ComputeKind::ReductionAdd:
    return true;
  case ComputeKind::Unknown:
    return false;
  }
  return false;
}

UnsupportedReason AscendBackendSupportMatrix::explainComputeKind(
    ComputeKind kind) const {
  if (isSupportedComputeKind(kind))
    return {"compute", ""};
  return {"compute",
          llvm::formatv("unsupported compute kind {0}",
                        stringifyComputeKind(kind))
              .str()};
}

} // namespace mlir::afir::ascend::backend
```

- [ ] **Step 5: Register the new source**

Modify `lib/Conversion/Ascend/CMakeLists.txt` and add the new file near other Ascend conversion sources:

```cmake
  Backend/BackendSupportMatrix.cpp
```

- [ ] **Step 6: Run GREEN verification in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build AscendBackendSupportMatrixTest && ctest --test-dir build -R AscendBackendSupportMatrixTest --output-on-failure'
```

Expected: `AscendBackendSupportMatrixTest` builds and passes.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git diff --check -- . ':!AGENTS.md'
git add include/Conversion/Ascend/Backend/BackendSupportMatrix.h \
        lib/Conversion/Ascend/Backend/BackendSupportMatrix.cpp \
        lib/Conversion/Ascend/CMakeLists.txt \
        test/unittests/Conversion/AscendBackendSupportMatrixTest.cpp \
        test/unittests/Conversion/CMakeLists.txt
git commit -m "feat: add Ascend backend support matrix"
```

## Task 2: `--ascend-compute-lower` Wrapper With Fail-Closed Verification

**Files:**
- Create `include/Conversion/Ascend/Backend/ComputeLoweringPass.h`
- Create `lib/Conversion/Ascend/Backend/ComputeLoweringPass.cpp`
- Modify `include/Conversion/Passes.h`
- Modify `include/Conversion/Passes.td`
- Modify `include/Conversion/LinalgToAscendC/LinalgToAscendCUtils.h`
- Modify `lib/Conversion/LinalgToAscendC/LinalgToAscendCPass.cpp`
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
- Create `test/Conversion/ascend-compute-lower.mlir`
- Create `test/Conversion/ascend-compute-lower-unsupported.mlir`

- [ ] **Step 1: Write the failing LIT test**

Create `test/Conversion/ascend-compute-lower.mlir`:

```mlir
// RUN: afir-opt %s --ascend-compute-lower | FileCheck %s

#map_par_reduce_lhs = affine_map<(d0, d1) -> (d0)>
#map_par_reduce_rhs = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @copy_gm_to_vecin
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: memref.copy
func.func @copy_gm_to_vecin(%src: memref<?x?xf32>) {
  %dst = memref.alloc() : memref<16x16xf32, 9 : i32>
  memref.copy %src, %dst : memref<?x?xf32> to memref<16x16xf32, 9 : i32>
  return
}

// CHECK-LABEL: func.func @vector_add
// CHECK: ascendc.add_l2
// CHECK-NOT: linalg.elementwise
func.func @vector_add() {
  %src0 = memref.alloc() : memref<32x32xf32, 9 : i32>
  %src1 = memref.alloc() : memref<32x32xf32, 9 : i32>
  %dst = memref.alloc() : memref<32x32xf32, 11 : i32>
  linalg.elementwise kind=#linalg.elementwise_kind<add>
      {ascendc.unit = "AiCore.Vector"}
      ins(%src0, %src1 : memref<32x32xf32, 9 : i32>,
                       memref<32x32xf32, 9 : i32>)
      outs(%dst : memref<32x32xf32, 11 : i32>)
  return
}
```

Create `test/Conversion/ascend-compute-lower-unsupported.mlir`:

```mlir
// RUN: not afir-opt %s --ascend-compute-lower 2>&1 | FileCheck %s

// CHECK: unsupported movement path VECCALC -> GM
func.func @unsupported_copy(%dst: memref<?x?xf32>) {
  %src = memref.alloc() : memref<16x16xf32, 11 : i32>
  memref.copy %src, %dst : memref<16x16xf32, 11 : i32> to memref<?x?xf32>
  return
}
```

- [ ] **Step 2: Run RED verification in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-compute-lower.mlir build/test/Conversion/ascend-compute-lower-unsupported.mlir'
```

Expected: LIT fails because `--ascend-compute-lower` is not registered.

- [ ] **Step 3: Expose shared legacy lowering**

Modify `include/Conversion/LinalgToAscendC/LinalgToAscendCUtils.h` and add:

```cpp
/// Run the existing LinalgToAscendC lowering implementation on one function.
/// This is shared by the legacy --linalg-to-ascendc pass and the Phase 5
/// --ascend-compute-lower wrapper.
LogicalResult lowerLinalgToAscendC(func::FuncOp funcOp);
```

Modify `lib/Conversion/LinalgToAscendC/LinalgToAscendCPass.cpp`:

```cpp
LogicalResult lowerLinalgToAscendC(func::FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  OpBuilder builder(ctx);

  Block &entryBlock = funcOp.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);
  Value pipe = builder.create<PipeOp>(funcOp.getLoc(), PipeType::get(ctx));

  Value lastQueueInserted = pipe;
  AscendCBufferContext bufCtx;
  bufCtx.pipe = pipe;

  funcOp.walk([&](memref::AllocOp allocOp) {
    int64_t ms = getMemorySpace(allocOp.getType());
    if (ms <= 0)
      return;

    auto pos = static_cast<TPosition>(ms);
    builder.setInsertionPointAfterValue(lastQueueInserted);
    Value queue = builder.create<QueueOp>(allocOp.getLoc(),
                                           QueueType::get(ctx, pos, 1));
    lastQueueInserted = queue;

    builder.setInsertionPoint(allocOp);
    Value tbuf = builder.create<TBufOp>(allocOp.getLoc(),
                                         TBufType::get(ctx, pos));
    Value len = computeAllocByteCount(builder, allocOp.getLoc(), allocOp);
    builder.create<TPipeInitBufferOp>(allocOp.getLoc(), pipe, tbuf, len);
    Value depth = builder.create<arith::ConstantOp>(
        allocOp.getLoc(), builder.getI32IntegerAttr(1));
    builder.create<TPipeInitQueueOp>(allocOp.getLoc(), pipe, queue, depth,
                                     len);

    bufCtx.allocToQueue[allocOp.getResult()] = queue;
    bufCtx.allocToTBuf[allocOp.getResult()] = tbuf;
  });

  if (failed(convertDataMove(funcOp, bufCtx)))
    return failure();
  if (failed(convertCompute(funcOp, bufCtx)))
    return failure();

  RewritePatternSet hoistPatterns(ctx);
  hoistPatterns.add<
      ascendc::HoistOpPattern<ascendc::QueueOp>,
      ascendc::HoistOpPattern<ascendc::TBufOp>,
      ascendc::HoistOpPattern<ascendc::TPipeInitBufferOp>,
      ascendc::HoistOpPattern<ascendc::TPipeInitQueueOp>>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(hoistPatterns))))
    return failure();

  return success();
}
```

Then replace the body of legacy `LinalgToAscendCPass::runOnOperation()` with:

```cpp
void runOnOperation() override {
  if (failed(lowerLinalgToAscendC(getOperation()))) {
    signalPassFailure();
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "=== After LinalgToAscendCPass ===\n");
  LLVM_DEBUG(getOperation().print(llvm::dbgs()));
}
```

Keep `computeAllocByteCount` in the same translation unit; do not change its semantics.

- [ ] **Step 4: Add the new pass declaration**

Create `include/Conversion/Ascend/Backend/ComputeLoweringPass.h`:

```cpp
//===- ComputeLoweringPass.h - Ascend compute lowering pass -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_COMPUTE_LOWERING_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_COMPUTE_LOWERING_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {

std::unique_ptr<Pass> createAscendComputeLowerPass();

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_COMPUTE_LOWERING_PASS_H
```

Modify `include/Conversion/Passes.h`:

```cpp
#include "Conversion/Ascend/Backend/ComputeLoweringPass.h"
```

Modify `include/Conversion/Passes.td` and add near the Ascend pipeline passes:

```tablegen
def AscendComputeLowerPass : Pass<"ascend-compute-lower", "mlir::func::FuncOp"> {
  let summary = "Lower supported memory-realized linalg and copies to AscendC";
  let description = [{
    Phase 5 compute lowering wrapper. It reuses the existing LinalgToAscendC
    lowering implementation and adds fail-closed support-matrix diagnostics.
  }];
  let constructor = "mlir::afir::createAscendComputeLowerPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect"
  ];
}
```

- [ ] **Step 5: Implement fail-closed verification and wrapper lowering**

Create `lib/Conversion/Ascend/Backend/ComputeLoweringPass.cpp`:

```cpp
//===- ComputeLoweringPass.cpp - Ascend compute lowering wrapper ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/ComputeLoweringPass.h"
#include "Conversion/Ascend/Backend/BackendSupportMatrix.h"
#include "Conversion/LinalgToAscendC/LinalgToAscendCUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#define GEN_PASS_DEF_ASCENDCOMPUTELOWERPASS
#include "Conversion/Passes.h.inc"

namespace mlir::afir {
namespace {

using backend::AscendBackendSupportMatrix;
using backend::ComputeKind;
using backend::MemorySpace;

MemorySpace memorySpaceOf(Type type) {
  return backend::parseMemorySpace(getMemorySpace(type));
}

ComputeKind classifyLinalgOp(Operation *op) {
  if (isa<linalg::MatmulOp>(op))
    return ComputeKind::Matmul;
  if (isa<linalg::FillOp>(op))
    return ComputeKind::Fill;
  if (auto elementwise = dyn_cast<linalg::ElementwiseOp>(op)) {
    auto kind = elementwise.getKind();
    if (kind == linalg::ElementwiseKind::add)
      return ComputeKind::ElementwiseAdd;
    if (kind == linalg::ElementwiseKind::max_signed)
      return ComputeKind::ElementwiseMax;
  }
  if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
    if (llvm::is_contained(generic.getIteratorTypesArray(),
                           utils::IteratorType::reduction))
      return ComputeKind::ReductionAdd;
  }
  return ComputeKind::Unknown;
}

LogicalResult verifySupportedInputs(func::FuncOp funcOp,
                                    const AscendBackendSupportMatrix &matrix) {
  WalkResult result = funcOp.walk([&](memref::CopyOp copyOp) {
    MemorySpace src = memorySpaceOf(copyOp.getSource().getType());
    MemorySpace dst = memorySpaceOf(copyOp.getTarget().getType());
    if (matrix.isSupportedMovementPath(src, dst))
      return WalkResult::advance();
    backend::UnsupportedReason reason = matrix.explainMovementPath(src, dst);
    copyOp.emitError(reason.detail);
    return WalkResult::interrupt();
  });
  if (result.wasInterrupted())
    return failure();

  result = funcOp.walk([&](Operation *op) {
    if (!isa<linalg::LinalgOp>(op))
      return WalkResult::advance();
    ComputeKind kind = classifyLinalgOp(op);
    if (matrix.isSupportedComputeKind(kind))
      return WalkResult::advance();
    backend::UnsupportedReason reason = matrix.explainComputeKind(kind);
    op->emitError(reason.detail);
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult verifyNoResidualLowerableOps(func::FuncOp funcOp) {
  WalkResult result = funcOp.walk([&](Operation *op) {
    if (isa<memref::CopyOp>(op) || isa<linalg::LinalgOp>(op)) {
      op->emitError("ascend-compute-lower left a lowerable operation behind");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

struct AscendComputeLowerPass
    : public impl::AscendComputeLowerPassBase<AscendComputeLowerPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    AscendBackendSupportMatrix matrix;

    if (failed(verifySupportedInputs(funcOp, matrix))) {
      signalPassFailure();
      return;
    }
    if (failed(lowerLinalgToAscendC(funcOp))) {
      signalPassFailure();
      return;
    }
    if (failed(verifyNoResidualLowerableOps(funcOp))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAscendComputeLowerPass() {
  return std::make_unique<AscendComputeLowerPass>();
}

} // namespace mlir::afir
```

If the exact MLIR API for `linalg::ElementwiseKind` or reduction iterator names differs in this checkout, use the existing patterns in `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp` and keep the public `ComputeKind` results unchanged.

- [ ] **Step 6: Register sources and link dependencies**

Modify `lib/Conversion/Ascend/CMakeLists.txt`:

```cmake
  Backend/ComputeLoweringPass.cpp
```

Add these link libraries if the xvm build reports missing symbols:

```cmake
  MLIRAsc
  MLIRAscUtils
  LinalgToAscendCConversion
```

If linking `AscendConversion` to `LinalgToAscendCConversion` creates a cycle, move only the reusable `lowerLinalgToAscendC` helper to a new small library in `lib/Conversion/LinalgToAscendC/CMakeLists.txt` and link both pass libraries to that helper library.

- [ ] **Step 7: Run GREEN verification in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt AscendBackendSupportMatrixTest && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-compute-lower.mlir build/test/Conversion/ascend-compute-lower-unsupported.mlir && ctest --test-dir build -R AscendBackendSupportMatrixTest --output-on-failure'
```

Expected: build succeeds, the supported lowering checks pass, and the unsupported copy fails with `unsupported movement path VECCALC -> GM`.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git diff --check -- . ':!AGENTS.md'
git add include/Conversion/Ascend/Backend/ComputeLoweringPass.h \
        lib/Conversion/Ascend/Backend/ComputeLoweringPass.cpp \
        include/Conversion/Passes.h \
        include/Conversion/Passes.td \
        include/Conversion/LinalgToAscendC/LinalgToAscendCUtils.h \
        lib/Conversion/LinalgToAscendC/LinalgToAscendCPass.cpp \
        lib/Conversion/Ascend/CMakeLists.txt \
        test/Conversion/ascend-compute-lower.mlir \
        test/Conversion/ascend-compute-lower-unsupported.mlir
git commit -m "feat: add Ascend compute lowering entry"
```

## Task 3: Ascend ABI Wrapper Passes

**Files:**
- Create `include/Conversion/Ascend/Backend/BackendWrapperPasses.h`
- Create `lib/Conversion/Ascend/Backend/BackendWrapperPasses.cpp`
- Modify `include/Conversion/Passes.h`
- Modify `include/Conversion/Passes.td`
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
- Create `test/Conversion/ascend-backend-abi-wrappers.mlir`

- [ ] **Step 1: Write the failing LIT test**

Create `test/Conversion/ascend-backend-abi-wrappers.mlir`:

```mlir
// RUN: afir-opt %s --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s

// CHECK-LABEL: func.func @broadcast_add_reducesum
// CHECK-SAME: %[[A:[a-z0-9]+]]: memref<?xf16>
// CHECK-SAME: %[[B:[a-z0-9]+]]: memref<?x?xf16>
// CHECK-SAME: %[[OUT:[a-z0-9]+]]: memref<?xf16
// CHECK-SAME: %[[WS:[a-z0-9]+]]: memref<ui8>
// CHECK-SAME: %[[TILING:[a-z0-9]+]]: !emitasc.py_struct<"TilingData"
// CHECK-SAME: cann.num_inputs = 2
// CHECK-NOT: emitasc.copy_struct
// CHECK: emitasc.member %[[TILING]] "TB_M"

module {
  func.func @broadcast_add_reducesum(
      %input_a: memref<?xf16>,
      %input_b: memref<?x?xf16>,
      %dim_arg0_0: i64,
      %dim_arg1_1: i64,
      %output: memref<?xf16>
  ) attributes {ascendc.aicore, ascendc.global, ascendc.kernel_kind = "mix"} {
    %tb_m = arith.constant 16 : i64
    %tb_n = arith.constant 4 : i64
    func.return
  }
}
```

- [ ] **Step 2: Run RED verification in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-backend-abi-wrappers.mlir'
```

Expected: LIT fails because `--ascend-prepare-for-emit` and `--ascend-canonicalize-cann-signature` are not registered.

- [ ] **Step 3: Add wrapper pass declarations**

Create `include/Conversion/Ascend/Backend/BackendWrapperPasses.h`:

```cpp
//===- BackendWrapperPasses.h - Ascend backend wrapper passes ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_WRAPPER_PASSES_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_WRAPPER_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {

std::unique_ptr<Pass> createAscendParallelizePass();
std::unique_ptr<Pass> createAscendPrepareForEmitPass();
std::unique_ptr<Pass> createAscendCanonicalizeCannSignaturePass();

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_WRAPPER_PASSES_H
```

Modify `include/Conversion/Passes.h`:

```cpp
#include "Conversion/Ascend/Backend/BackendWrapperPasses.h"
```

Modify `include/Conversion/Passes.td`:

```tablegen
def AscendParallelizePass : Pass<"ascend-parallelize", "mlir::func::FuncOp"> {
  let summary = "Run Phase 5 Ascend kernel dispatch lowering";
  let description = [{
    Versionless Phase 5 wrapper around the existing AscendC parallelization
    implementation.
  }];
  let constructor = "mlir::afir::createAscendParallelizePass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
}

def AscendPrepareForEmitPass : Pass<"ascend-prepare-for-emit", "mlir::func::FuncOp"> {
  let summary = "Prepare Phase 5 AscendC kernel ABI data";
  let description = [{
    Versionless Phase 5 wrapper around the existing AscendC prepare-for-emit
    implementation.
  }];
  let constructor = "mlir::afir::createAscendPrepareForEmitPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def AscendCanonicalizeCannSignaturePass
    : Pass<"ascend-canonicalize-cann-signature", "mlir::ModuleOp"> {
  let summary = "Canonicalize Phase 5 Ascend kernel signatures to CANN ABI";
  let description = [{
    Versionless Phase 5 wrapper around the existing CANN signature
    canonicalization implementation.
  }];
  let constructor = "mlir::afir::createAscendCanonicalizeCannSignaturePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::memref::MemRefDialect"
  ];
}
```

- [ ] **Step 4: Implement wrappers through nested pipelines**

Create `lib/Conversion/Ascend/Backend/BackendWrapperPasses.cpp`:

```cpp
//===- BackendWrapperPasses.cpp - Ascend backend wrapper passes -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/BackendWrapperPasses.h"
#include "Conversion/AscendCParallelize/AscendCParallelizePass.h"
#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"
#include "Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#define GEN_PASS_DEF_ASCENDPARALLELIZEPASS
#define GEN_PASS_DEF_ASCENDPREPAREFOREMITPASS
#define GEN_PASS_DEF_ASCENDCANONICALIZECANNSIGNATUREPASS
#include "Conversion/Passes.h.inc"

namespace mlir::afir {
namespace {

struct AscendParallelizePass
    : public impl::AscendParallelizePassBase<AscendParallelizePass> {
  void runOnOperation() override {
    OpPassManager pm("func.func");
    pm.addPass(createAscendCParallelizePass());
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

struct AscendPrepareForEmitPass
    : public impl::AscendPrepareForEmitPassBase<AscendPrepareForEmitPass> {
  void runOnOperation() override {
    OpPassManager pm("func.func");
    pm.addPass(createAscendCPrepareForEmitPass());
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

struct AscendCanonicalizeCannSignaturePass
    : public impl::AscendCanonicalizeCannSignaturePassBase<
          AscendCanonicalizeCannSignaturePass> {
  void runOnOperation() override {
    OpPassManager pm("builtin.module");
    pm.addPass(createCanonicalizeCannSignaturePass());
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createAscendParallelizePass() {
  return std::make_unique<AscendParallelizePass>();
}

std::unique_ptr<Pass> createAscendPrepareForEmitPass() {
  return std::make_unique<AscendPrepareForEmitPass>();
}

std::unique_ptr<Pass> createAscendCanonicalizeCannSignaturePass() {
  return std::make_unique<AscendCanonicalizeCannSignaturePass>();
}

} // namespace mlir::afir
```

If `runPipeline` is not available for this pass base in the local MLIR version, replace each wrapper with a shared helper extraction from the legacy pass implementation. Do not duplicate whole pass bodies.

- [ ] **Step 5: Register sources and link dependencies**

Modify `lib/Conversion/Ascend/CMakeLists.txt`:

```cmake
  Backend/BackendWrapperPasses.cpp
```

Add link libraries if required by the xvm linker:

```cmake
  AscendCParallelizeConversion
  AscendCPrepareForEmitConversion
  CanonicalizeCannSignatureConversion
```

- [ ] **Step 6: Run GREEN verification in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-backend-abi-wrappers.mlir build/test/Conversion/canonicalize-cann-signature.mlir'
```

Expected: new wrapper LIT passes and the old canonicalization LIT still passes.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git diff --check -- . ':!AGENTS.md'
git add include/Conversion/Ascend/Backend/BackendWrapperPasses.h \
        lib/Conversion/Ascend/Backend/BackendWrapperPasses.cpp \
        include/Conversion/Passes.h \
        include/Conversion/Passes.td \
        lib/Conversion/Ascend/CMakeLists.txt \
        test/Conversion/ascend-backend-abi-wrappers.mlir
git commit -m "feat: add Ascend backend ABI wrappers"
```

## Task 4: Runtime Artifact Emitters

**Files:**
- Create `include/Target/CannKernel/CannRuntimeArtifacts.h`
- Create `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`
- Modify `include/Target/CannKernel/CannTranslation.h`
- Modify `lib/Target/CannKernel/CannTranslation.cpp`
- Modify `lib/Target/CannKernel/CMakeLists.txt`
- Modify `tools/afir-translate/afir-translate.cpp`
- Create `test/Target/cann-translate-runtime-artifacts.mlir`
- Create `test/Target/cann-translate-runtime-artifacts-unsupported.mlir`

- [ ] **Step 1: Write artifact LIT tests**

Create `test/Target/cann-translate-runtime-artifacts.mlir`:

```mlir
// RUN: rm -f %t.cpp %t.tiling.json %t.manifest.json %t.host.cpp
// RUN: afir-translate -mlir-to-cann %s --tiling-space-out=%t.tiling.json --runtime-manifest-out=%t.manifest.json --host-tiling-out=%t.host.cpp --cann-soc=Ascend910B2 > %t.cpp
// RUN: FileCheck %s --input-file=%t.tiling.json --check-prefix=TILING
// RUN: FileCheck %s --input-file=%t.manifest.json --check-prefix=MANIFEST
// RUN: FileCheck %s --input-file=%t.host.cpp --check-prefix=HOST

// TILING: "schema_version": "2.0"
// TILING: "kernel": "broadcast_add_reducesum"
// TILING: "soc": "Ascend910B2"
// TILING: "workspace_size_expr": "0"
// TILING: "name": "TB_M"
// TILING: "fixed": false
// TILING: "name": "dim_arg0_0"
// TILING: "fixed": true
// TILING: "shape_key": "arg0_dim0"

// MANIFEST: "kernelName": "broadcast_add_reducesum"
// MANIFEST: "guardSet": []
// MANIFEST: "workspaceSizeBytes": 0
// MANIFEST: "shapeArgOrder"
// MANIFEST: "name": "dim_arg0_0"
// MANIFEST: "kernelGraph"

// HOST: struct TilingData
// HOST: extern "C"
// HOST: int32_t broadcast_add_reducesum_GetTilingSize(void)
// HOST: int32_t broadcast_add_reducesum_GetTiling(const int64_t* shape_args, int32_t shape_count, void* tiling_out)
// HOST: int64_t broadcast_add_reducesum_GetBlockDim(const int64_t* shape_args, int32_t shape_count)
// HOST: int64_t broadcast_add_reducesum_GetWorkspaceSize(const int64_t* shape_args, int32_t shape_count)

module {
  func.func @broadcast_add_reducesum(
      %a: memref<?xf16>,
      %b: memref<?x?xf16>,
      %out: memref<?xf16>,
      %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData",
          [i64, i64, i64, i64],
          ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 2 : i32} {
    %tb_m = emitasc.member %tiling "TB_M"
        : !emitasc.py_struct<"TilingData",
              [i64, i64, i64, i64],
              ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>,
          i64
    func.return
  }
}
```

Create `test/Target/cann-translate-runtime-artifacts-unsupported.mlir`:

```mlir
// RUN: not afir-translate -mlir-to-cann %s --runtime-manifest-out=%t.manifest.json 2>&1 | FileCheck %s

// CHECK: runtime manifest MVP supports exactly one global kernel

module {
  func.func @kernel_a(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    func.return
  }

  func.func @kernel_b(
      %a: memref<?xf16>, %out: memref<?xf16>, %ws: memref<ui8>,
      %tiling: !emitasc.py_struct<"TilingData", [i64], ["TB_M"]>
  ) attributes {ascendc.aicore, ascendc.global, cann.num_inputs = 1 : i32} {
    func.return
  }
}
```

- [ ] **Step 2: Run RED verification in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-translate && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target/cann-translate-runtime-artifacts.mlir build/test/Target/cann-translate-runtime-artifacts-unsupported.mlir'
```

Expected: LIT fails because `--runtime-manifest-out`, `--host-tiling-out`, and `--cann-soc` are not registered and current `tiling_space.json` lacks `schema_version`.

- [ ] **Step 3: Add artifact API**

Create `include/Target/CannKernel/CannRuntimeArtifacts.h`:

```cpp
//===- CannRuntimeArtifacts.h - CANN runtime artifact emission --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TARGET_CANNKERNEL_RUNTIME_ARTIFACTS_H
#define AFIR_TARGET_CANNKERNEL_RUNTIME_ARTIFACTS_H

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::cann {

struct CannRuntimeArtifactOptions {
  StringRef kernelFile;
  StringRef soc = "Ascend910B1";
};

LogicalResult emitTilingSpaceJson(ModuleOp module, StringRef outPath,
                                  const CannRuntimeArtifactOptions &options);
LogicalResult emitRuntimeManifestJson(ModuleOp module, StringRef outPath,
                                      const CannRuntimeArtifactOptions &options);
LogicalResult emitHostTilingCpp(ModuleOp module, StringRef outPath,
                                const CannRuntimeArtifactOptions &options);

} // namespace mlir::afir::cann

#endif // AFIR_TARGET_CANNKERNEL_RUNTIME_ARTIFACTS_H
```

- [ ] **Step 4: Implement ABI extraction and artifact emission**

Create `lib/Target/CannKernel/CannRuntimeArtifacts.cpp`. Include these required behaviors:

```cpp
static SmallVector<func::FuncOp> collectGlobalKernels(ModuleOp module) {
  SmallVector<func::FuncOp> kernels;
  for (Operation &child : module.getBody()->getOperations())
    if (auto funcOp = dyn_cast<func::FuncOp>(child))
      if (funcOp->hasAttr(ascendc::attr::global))
        kernels.push_back(funcOp);
  return kernels;
}
```

Use a single helper to validate MVP scope:

```cpp
static FailureOr<func::FuncOp> getSingleGlobalKernel(ModuleOp module,
                                                     StringRef artifactName) {
  SmallVector<func::FuncOp> kernels = collectGlobalKernels(module);
  if (kernels.size() != 1) {
    module.emitError() << artifactName
                       << " MVP supports exactly one global kernel";
    return failure();
  }
  return kernels.front();
}
```

Extract tiling fields from the final argument:

```cpp
static FailureOr<emitasc::PyStructType>
getTilingType(func::FuncOp funcOp) {
  if (funcOp.getNumArguments() < 2) {
    funcOp.emitError("CANN ABI requires workspace and tiling arguments");
    return failure();
  }
  Type tilingType = funcOp.getArgument(funcOp.getNumArguments() - 1).getType();
  auto pyStruct = dyn_cast<emitasc::PyStructType>(tilingType);
  if (!pyStruct) {
    funcOp.emitError("last CANN ABI argument must be !emitasc.py_struct");
    return failure();
  }
  return pyStruct;
}
```

Generate shape keys exactly as the existing skeleton did:

```cpp
static bool isShapeField(StringRef name) {
  return name.starts_with("dim_arg");
}

static std::string makeShapeKey(StringRef name) {
  StringRef rest = name.drop_front(4);
  auto pos = rest.rfind('_');
  if (pos == StringRef::npos)
    return rest.str();
  return rest.substr(0, pos).str() + "_dim" + rest.substr(pos + 1).str();
}
```

`emitTilingSpaceJson` must write:

```json
{
  "schema_version": "2.0",
  "kernel": "<func-name>",
  "kernel_file": "<options.kernelFile>",
  "soc": "<options.soc>",
  "block_dim_expr": "20",
  "workspace_size_expr": "0",
  "tiling_params": [...]
}
```

For each tiling field:

- `name`: field name
- `type`: `"int64"`
- `fixed`: `true` only for `dim_arg*`
- `shape_key`: generated only for fixed fields
- `values`: empty array only for non-fixed fields

`emitRuntimeManifestJson` must write static single-kernel MVP fields:

```json
{
  "kernelName": "<func-name>",
  "shapeBucketKey": "static",
  "guardSet": [],
  "tilingSchema": [...],
  "scheduleEntries": [
    {
      "decisionId": "static_0",
      "guard": "true",
      "tilingParams": {}
    }
  ],
  "abiSignature": "<func-name>:cann_static",
  "cacheKey": "<func-name>:static:<soc>",
  "workspaceSizeExpr": "0",
  "workspaceSizeBytes": 0,
  "shapeArgOrder": [...],
  "kernelGraph": {
    "nodes": [{"name": "<func-name>"}],
    "edges": []
  }
}
```

`shapeArgOrder` entries must include:

```json
{"name": "<field-name>", "shapeKey": "<shape-key>", "abiPosition": <field-index>}
```

`emitHostTilingCpp` must write valid C++ source with:

```cpp
#include <cstdint>
#include <cstring>

struct TilingData {
  int64_t TB_M;
  int64_t TB_N;
  int64_t dim_arg0_0;
  int64_t dim_arg1_1;
};

extern "C" {

int32_t broadcast_add_reducesum_GetTilingSize(void) {
  return static_cast<int32_t>(sizeof(TilingData));
}

int32_t broadcast_add_reducesum_GetTiling(const int64_t* shape_args,
                                          int32_t shape_count,
                                          void* tiling_out) {
  if (shape_count != 2 || tiling_out == nullptr)
    return 1;
  TilingData data{};
  data.TB_M = 0;
  data.TB_N = 0;
  data.dim_arg0_0 = shape_args[0];
  data.dim_arg1_1 = shape_args[1];
  std::memcpy(tiling_out, &data, sizeof(TilingData));
  return 0;
}

int64_t broadcast_add_reducesum_GetBlockDim(const int64_t* shape_args,
                                            int32_t shape_count) {
  return shape_count == 2 ? 20 : -1;
}

int64_t broadcast_add_reducesum_GetWorkspaceSize(const int64_t* shape_args,
                                                 int32_t shape_count) {
  return shape_count == 2 ? 0 : -1;
}

} // extern "C"
```

Use the actual kernel name and tiling fields; non-shape fields get `0` in this MVP because no autotuner result is passed to the translator yet.

- [ ] **Step 5: Wire translation options**

Modify `include/Target/CannKernel/CannTranslation.h`:

```cpp
struct CannTranslationOptions {
  StringRef tilingSpaceOutPath;
  StringRef runtimeManifestOutPath;
  StringRef hostTilingOutPath;
  StringRef kernelFile;
  StringRef soc = "Ascend910B1";
};

LogicalResult translateToCannKernel(Operation *op, raw_ostream &os,
                                    const CannTranslationOptions &options);
LogicalResult translateToCannKernel(Operation *op, raw_ostream &os,
                                    StringRef tilingSpaceOutPath = "",
                                    StringRef kernelFile = "");
```

Modify `lib/Target/CannKernel/CannTranslation.cpp`:

- Include `Target/CannKernel/CannRuntimeArtifacts.h`.
- Keep the old overload and have it construct `CannTranslationOptions`.
- Remove or stop using the old local `emitTilingSpaceJson` helper.
- After source emission succeeds, call artifact emitters for non-empty paths:

```cpp
afir::cann::CannRuntimeArtifactOptions artifactOptions;
artifactOptions.kernelFile = options.kernelFile;
artifactOptions.soc = options.soc;

if (!options.tilingSpaceOutPath.empty() &&
    failed(afir::cann::emitTilingSpaceJson(moduleOp,
                                           options.tilingSpaceOutPath,
                                           artifactOptions)))
  return failure();
if (!options.runtimeManifestOutPath.empty() &&
    failed(afir::cann::emitRuntimeManifestJson(moduleOp,
                                               options.runtimeManifestOutPath,
                                               artifactOptions)))
  return failure();
if (!options.hostTilingOutPath.empty() &&
    failed(afir::cann::emitHostTilingCpp(moduleOp,
                                         options.hostTilingOutPath,
                                         artifactOptions)))
  return failure();
```

Ensure artifact failures return `failure()` instead of warning-only behavior.

- [ ] **Step 6: Add CLI options**

Modify `tools/afir-translate/afir-translate.cpp`:

```cpp
static cl::opt<std::string> RuntimeManifestOut(
    "runtime-manifest-out",
    cl::desc("Write runtime_manifest.json to this path"),
    cl::init(""));

static cl::opt<std::string> HostTilingOut(
    "host-tiling-out",
    cl::desc("Write host tiling C ABI source to this path"),
    cl::init(""));

static cl::opt<std::string> CannSoc(
    "cann-soc",
    cl::desc("CANN SoC string used in generated runtime artifacts"),
    cl::init("Ascend910B1"));
```

Pass options to the translator:

```cpp
CannTranslationOptions options;
options.tilingSpaceOutPath = TilingSpaceOut;
options.runtimeManifestOutPath = RuntimeManifestOut;
options.hostTilingOutPath = HostTilingOut;
options.soc = CannSoc;
return translateToCannKernel(op, os, options);
```

- [ ] **Step 7: Register the new source**

Modify `lib/Target/CannKernel/CMakeLists.txt`:

```cmake
  CannRuntimeArtifacts.cpp
```

- [ ] **Step 8: Run GREEN verification in xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-translate && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target/cann-translate-runtime-artifacts.mlir build/test/Target/cann-translate-runtime-artifacts-unsupported.mlir build/test/Target/cann-translate.mlir'
```

Expected: new artifact tests pass, unsupported multi-kernel manifest fails with the explicit MVP diagnostic, and existing CANN translation still passes.

- [ ] **Step 9: Commit Task 4**

Run:

```bash
git diff --check -- . ':!AGENTS.md'
git add include/Target/CannKernel/CannRuntimeArtifacts.h \
        lib/Target/CannKernel/CannRuntimeArtifacts.cpp \
        include/Target/CannKernel/CannTranslation.h \
        lib/Target/CannKernel/CannTranslation.cpp \
        lib/Target/CannKernel/CMakeLists.txt \
        tools/afir-translate/afir-translate.cpp \
        test/Target/cann-translate-runtime-artifacts.mlir \
        test/Target/cann-translate-runtime-artifacts-unsupported.mlir
git commit -m "feat: emit Ascend runtime artifacts"
```

## Task 5: Phase 5 Tracking, Transformer Smoke, Review, And Verification

**Files:**
- Create `test/Conversion/ascend-phase5-transformer-dynamic-smoke.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Add transformer dynamic smoke test**

Create `test/Conversion/ascend-phase5-transformer-dynamic-smoke.mlir`:

```mlir
// RUN: not afir-opt %S/../../examples/transformer/transformer_dynamic.mlir --ascend-compute-lower 2>&1 | FileCheck %s

// CHECK: unsupported
```

This test intentionally accepts an unsupported diagnostic for the current transformer graph. Its purpose is to prove Phase 5 fails explicitly on the final acceptance scenario until the support matrix is expanded.

- [ ] **Step 2: Update tracking**

Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`:

```markdown
| `ComputeLoweringDriver` 对齐 | `Done` | 新增 `--ascend-compute-lower` 正式入口，复用现有 LinalgToAscendC lowering，并通过 support matrix 对 unsupported op / movement path fail-closed | `ascend-compute-lower.mlir`；`AscendBackendSupportMatrixTest` |
| ABI lowering 对齐 | `Done` | 新增 `--ascend-parallelize`、`--ascend-prepare-for-emit`、`--ascend-canonicalize-cann-signature` 正式入口，旧原型入口保留 | `ascend-backend-abi-wrappers.mlir` |
| `HostTilingEmitter` | `Done` | `afir-translate --host-tiling-out` 输出静态 shape / 单 kernel C ABI source | `cann-translate-runtime-artifacts.mlir` |
| `RuntimeManifestBuilder` | `Done` | `afir-translate --runtime-manifest-out` 输出静态 shape / 单 kernel manifest；多 global kernel 显式 unsupported | `cann-translate-runtime-artifacts-unsupported.mlir` |
| `tiling_space.json` export | `Done` | `--tiling-space-out` 升级为 `schema_version = "2.0"`，包含 workspace/block dim/schema fields | `cann-translate-runtime-artifacts.mlir` |
```

Add a Phase 5 verification record with the exact commands run in Step 4.

- [ ] **Step 3: Run focused xvm verification**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt afir-translate AscendBackendSupportMatrixTest && ctest --test-dir build -R AscendBackendSupportMatrixTest --output-on-failure && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-compute-lower.mlir build/test/Conversion/ascend-compute-lower-unsupported.mlir build/test/Conversion/ascend-backend-abi-wrappers.mlir build/test/Conversion/ascend-phase5-transformer-dynamic-smoke.mlir build/test/Target/cann-translate-runtime-artifacts.mlir build/test/Target/cann-translate-runtime-artifacts-unsupported.mlir'
```

Expected:

- `AscendBackendSupportMatrixTest` passes.
- `ascend-compute-lower.mlir` passes.
- `ascend-compute-lower-unsupported.mlir` passes by observing explicit unsupported diagnostic.
- `ascend-backend-abi-wrappers.mlir` passes.
- `ascend-phase5-transformer-dynamic-smoke.mlir` passes by observing explicit unsupported diagnostic.
- Both runtime artifact target tests pass.

- [ ] **Step 4: Run regression and guards**

Run:

```bash
git diff --check -- . ':!AGENTS.md'
test/tools/check_ascend_no_v2_code_naming.sh
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && test/tools/check_ascend_no_v2_code_naming.sh && ninja -C build afir-opt afir-translate AscendBackendSupportMatrixTest AscendCommonAttributesTest AscendKernelPatternTest AscendRealizePlannerTest AscendTargetMemoryModelTest AscendTargetIntrinsicModelTest AscendTargetCostModelTest AscendTargetModelVerifierTest && ctest --test-dir build -R "Ascend(CommonAttributes|KernelPattern|RealizePlanner|BackendSupportMatrix|TargetMemoryModel|TargetIntrinsicModel|TargetCostModel|TargetModelVerifier)Test" --output-on-failure && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-" && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Target'
```

Expected:

- Host and xvm code naming guard pass.
- Focused Ascend unit tests pass.
- Ascend conversion LIT pass.
- Target LIT pass.

- [ ] **Step 5: Run required reviews**

Dispatch two review subagents:

Spec compliance reviewer:

```text
Review the Phase 5 implementation against:
- docs/superpowers/specs/2026-05-11-ascend-phase5-backend-integration-design.md
- docs/superpowers/plans/2026-05-11-ascend-phase5-backend-integration.md
- docs/Ascend-MLIR-Detailed-Implementation-V2-6.zh.md
- docs/Ascend-MLIR-Detailed-Implementation-V2-9.zh.md

Check that the implementation creates official Ascend Phase 5 entry points, reuses existing backend logic instead of copying it, adds fail-closed support-matrix diagnostics, emits runtime artifacts, preserves old prototype entry points, and does not add schedule/placement/memory decisions.
Return PASS or concrete issues with file/line evidence.
```

Code quality reviewer:

```text
Review the Phase 5 code changes for correctness, build risk, MLIR pass API usage, ownership/lifetime issues, diagnostics, test robustness, and code naming rules.
Focus on the new Ascend backend support matrix, compute lowering wrapper, ABI wrapper passes, runtime artifact emitters, and translator CLI options.
Return APPROVED or concrete issues with file/line evidence.
```

Fix any required issues and re-run the affected focused tests.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git diff --check -- . ':!AGENTS.md'
git add test/Conversion/ascend-phase5-transformer-dynamic-smoke.mlir \
        docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
git commit -m "docs: complete Ascend phase5 tracking"
```

- [ ] **Step 7: Push after final verification**

Run:

```bash
git status --short
git push
```

Expected: only `AGENTS.md` may remain modified before push; pushed branch contains all Phase 5 commits.

## Completion Criteria

Phase 5 MVP is complete when all of these are true:

- `--ascend-compute-lower` exists and uses fail-closed support validation.
- `--ascend-parallelize`, `--ascend-prepare-for-emit`, and `--ascend-canonicalize-cann-signature` exist as official Phase 5 entry points.
- Existing prototype entry points still work.
- `afir-translate -mlir-to-cann` can emit AscendC source, `tiling_space.json`, `runtime_manifest.json`, and host tiling C ABI source.
- `tiling_space.json` contains `schema_version = "2.0"`.
- Unsupported transformer dynamic lowering fails with a clear unsupported diagnostic.
- xvm/docker build and focused tests pass.
- Code naming guard passes.
- Spec compliance review passes.
- Code quality review approves.
- Tracking board marks Phase 5 task rows `Done`.
