# AscendCPrepareForEmit Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the monolithic `AscendCPrepareForEmitPass` into three focused passes (`AscendCFlattenGMPtrPass`, `AscendCPackTilingDataPass`, `AscendCFinalizeKernelPass`) with a clean Phase-B-only tiling info path, while keeping the original pass intact for backward compatibility.

**Architecture:** Three new passes run in sequence: `FlattenGMPtr` resolves subview chains before any TilingData exists (uses `memref.dim` directly); `PackTilingData` packs the now-clean IR into a TilingData struct reading only from `vector_plan.tiling_infos` (no Phase A fallback); `FinalizeKernel` stamps attributes and lowers residual affine ops. The existing `AscendCPrepareForEmitPass` is left untouched. Pipeline.cpp is updated to call the three new passes in place of the old one.

**Tech Stack:** MLIR TableGen pass registration, MLIR C++ pass API, FileCheck lit tests, `afir-opt` tool.

---

## File Structure

**New files:**
- `lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp` — promotes GM allocs to func args, flattens subview→set_global_buffer to flat pointer+offset, handles GM→GM copy→memmove
- `lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp` — reads `vector_plan.tiling_infos`, scans `memref.dim` ops, builds TilingData struct, replaces all uses, no Phase A fallback
- `lib/Conversion/AscendCPrepareForEmit/FinalizeKernelPass.cpp` — sets `ascendc.aicore`/`ascendc.global`, strips return operands, lowers `affine.min`→`arith.minsi`

**New test files:**
- `test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir`
- `test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir`
- `test/Conversion/AscendCPrepareForEmit/finalize-kernel.mlir`

**Modified files:**
- `include/Conversion/Passes.td` — add three new pass defs after `AscendCPrepareForEmitPass`
- `include/Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h` — add three factory function declarations
- `lib/Conversion/AscendCPrepareForEmit/CMakeLists.txt` — add three new source files
- `lib/Conversion/VectorPlan/Pipeline.cpp` — replace `createAscendCPrepareForEmitPass()` with the three new passes in sequence

---

## Task 1: Pass registration scaffold

Add the three new pass definitions to `Passes.td` and the three factory declarations to the header. No implementation yet — just enough to make the build pass.

**Files:**
- Modify: `include/Conversion/Passes.td:207-230`
- Modify: `include/Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h`
- Modify: `lib/Conversion/AscendCPrepareForEmit/CMakeLists.txt`
- Create: `lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp`
- Create: `lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp`
- Create: `lib/Conversion/AscendCPrepareForEmit/FinalizeKernelPass.cpp`

- [ ] **Step 1: Add three pass defs to Passes.td**

Insert after the closing `}` of `AscendCPrepareForEmitPass` (line ~230):

```tablegen
//===----------------------------------------------------------------------===//
// AscendC Prepare For Emit — refactored sub-passes
//===----------------------------------------------------------------------===//

def AscendCFlattenGMPtrPass : Pass<"ascendc-flatten-gm-ptr", "mlir::func::FuncOp"> {
  let summary = "Promote GM allocs to func args and flatten subview chains for set_global_buffer";
  let description = [{
    Runs before PackTilingData so memref.dim values are still available as
    plain IR ops (not yet packed into TilingData).

    Transformations:
      1. Promote every top-level memref.alloc with memory_space=0 (GM) to a
         new function block argument.
      2. Replace each ascendc.global_tensor.set_global_buffer whose buffer is a
         memref.subview chain with a flat-pointer + integer offset form using
         emitasc.reinterpret_cast + emitasc.ptr_offset.
      3. Replace GM->GM memref.copy ops with a verbatim memmove call.
  }];
  let constructor = "mlir::afir::createAscendCFlattenGMPtrPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def AscendCPackTilingDataPass : Pass<"ascendc-pack-tiling-data", "mlir::func::FuncOp"> {
  let summary = "Pack index/i64 tiling args and memref.dim uses into a TilingData GM struct";
  let description = [{
    Reads vector_plan.tiling_infos on the parent ModuleOp to discover named
    tiling parameters (e.g. XBLOCK, XBLOCK_SUB).  Fails if tiling_infos is
    absent — no Phase A positional fallback.

    Transformations:
      1. Collect tiling arg names/indices from vector_plan.tiling_infos.
      2. Scan for memref.dim %blockArg, %cI ops; each unique (arg, dim) pair
         becomes a dim_argN_D field in TilingData.
      3. Inject a new TilingData GM pointer block argument, emit
         emitasc.copy_struct + emitasc.member to extract every field.
      4. Replace all tiling arg uses and memref.dim uses with the extracted
         values; erase the original args.
      5. Emit emitasc.declare_py_struct at module scope.
  }];
  let constructor = "mlir::afir::createAscendCPackTilingDataPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect"
  ];
}

def AscendCFinalizeKernelPass : Pass<"ascendc-finalize-kernel", "mlir::func::FuncOp"> {
  let summary = "Stamp kernel attributes, strip return values, lower affine.min";
  let description = [{
    Final cleanup before ascir-translate:
      1. Set {ascendc.aicore, ascendc.global} attributes.
      2. Replace func.return with operands by a void func.return.
      3. Lower affine.min -> arith.minsi (ascir-translate has no affine support).
  }];
  let constructor = "mlir::afir::createAscendCFinalizeKernelPass()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}
```

- [ ] **Step 2: Add factory declarations to header**

In `include/Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h`, add after the existing declaration:

```cpp
std::unique_ptr<Pass> createAscendCFlattenGMPtrPass();
std::unique_ptr<Pass> createAscendCPackTilingDataPass();
std::unique_ptr<Pass> createAscendCFinalizeKernelPass();
```

- [ ] **Step 3: Add source files to CMakeLists.txt**

```cmake
add_mlir_library(AscendCPrepareForEmitConversion
  AscendCPrepareForEmitPass.cpp
  FlattenGMPtrPass.cpp
  PackTilingDataPass.cpp
  FinalizeKernelPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRAsc
  MLIREmitAsc
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTransforms
)
```

- [ ] **Step 4: Create stub source files**

`lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp`:
```cpp
#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"
#include "mlir/Pass/Pass.h"
#define GEN_PASS_DECL_ASCENDCFLATTENGMPTRPASS
#define GEN_PASS_DEF_ASCENDCFLATTENGMPTRPASS
#include "Conversion/Passes.h.inc"
using namespace mlir;
namespace mlir::afir {
struct AscendCFlattenGMPtrPass
    : public ::impl::AscendCFlattenGMPtrPassBase<AscendCFlattenGMPtrPass> {
  void runOnOperation() override {}
};
std::unique_ptr<Pass> createAscendCFlattenGMPtrPass() {
  return std::make_unique<AscendCFlattenGMPtrPass>();
}
} // namespace mlir::afir
```

`lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp`:
```cpp
#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"
#include "mlir/Pass/Pass.h"
#define GEN_PASS_DECL_ASCENDCPACKTILINGDATAPASS
#define GEN_PASS_DEF_ASCENDCPACKTILINGDATAPASS
#include "Conversion/Passes.h.inc"
using namespace mlir;
namespace mlir::afir {
struct AscendCPackTilingDataPass
    : public ::impl::AscendCPackTilingDataPassBase<AscendCPackTilingDataPass> {
  void runOnOperation() override {}
};
std::unique_ptr<Pass> createAscendCPackTilingDataPass() {
  return std::make_unique<AscendCPackTilingDataPass>();
}
} // namespace mlir::afir
```

`lib/Conversion/AscendCPrepareForEmit/FinalizeKernelPass.cpp`:
```cpp
#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"
#include "mlir/Pass/Pass.h"
#define GEN_PASS_DECL_ASCENDCFINALIZEKERNELPASS
#define GEN_PASS_DEF_ASCENDCFINALIZEKERNELPASS
#include "Conversion/Passes.h.inc"
using namespace mlir;
namespace mlir::afir {
struct AscendCFinalizeKernelPass
    : public ::impl::AscendCFinalizeKernelPassBase<AscendCFinalizeKernelPass> {
  void runOnOperation() override {}
};
std::unique_ptr<Pass> createAscendCFinalizeKernelPass() {
  return std::make_unique<AscendCFinalizeKernelPass>();
}
} // namespace mlir::afir
```

- [ ] **Step 5: Build to verify registration compiles**

```bash
cd build && ninja afir-opt -j$(nproc) 2>&1 | tail -5
```
Expected: build succeeds; `bin/afir-opt --help 2>&1 | grep -E "flatten-gm|pack-tiling|finalize-kernel"` shows three new passes.

- [ ] **Step 6: Commit**

```bash
git add include/Conversion/Passes.td \
        include/Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h \
        lib/Conversion/AscendCPrepareForEmit/CMakeLists.txt \
        lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp \
        lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp \
        lib/Conversion/AscendCPrepareForEmit/FinalizeKernelPass.cpp
git commit -m "refactor(prepare-for-emit): register three refactored sub-passes (stubs)"
```

---

## Task 2: Implement `AscendCFinalizeKernelPass`

The simplest of the three — sets attributes, drops return operands, lowers `affine.min`. Extract directly from `AscendCPrepareForEmitPass.cpp` steps 10, 11, 12.

**Files:**
- Modify: `lib/Conversion/AscendCPrepareForEmit/FinalizeKernelPass.cpp`
- Create: `test/Conversion/AscendCPrepareForEmit/finalize-kernel.mlir`

- [ ] **Step 1: Write the test**

`test/Conversion/AscendCPrepareForEmit/finalize-kernel.mlir`:
```mlir
// RUN: afir-opt %s --ascendc-finalize-kernel 2>&1 | FileCheck %s

// CHECK: func.func @my_kernel(
// CHECK-SAME: ascendc.aicore
// CHECK-SAME: ascendc.global
// CHECK-NOT: return %

// CHECK: affine.min is lowered:
// CHECK: arith.minsi

func.func @my_kernel(%a: memref<f32>) -> memref<f32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %min = affine.min affine_map<(d0)[s0] -> (d0, s0)>(%c0)[%c8]
  return %a : memref<f32>
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd build && bin/afir-opt ../test/Conversion/AscendCPrepareForEmit/finalize-kernel.mlir \
  --ascendc-finalize-kernel 2>&1 | head -5
```
Expected: output contains `func.func @my_kernel` unchanged (stub does nothing).

- [ ] **Step 3: Implement FinalizeKernelPass**

Full content of `lib/Conversion/AscendCPrepareForEmit/FinalizeKernelPass.cpp`:

```cpp
#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_ASCENDCFINALIZEKERNELPASS
#define GEN_PASS_DEF_ASCENDCFINALIZEKERNELPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

static void finalizeFunc(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();

  // 1. Set kernel attributes.
  func->setAttr("ascendc.aicore", UnitAttr::get(ctx));
  func->setAttr("ascendc.global", UnitAttr::get(ctx));

  // 2. Strip return operands (kernel returns void).
  func.walk([&](func::ReturnOp ret) {
    if (ret.getNumOperands() > 0) {
      OpBuilder b(ret);
      b.create<func::ReturnOp>(ret.getLoc());
      ret.erase();
    }
  });

  // Update function type to match (no results).
  SmallVector<Type> argTypes;
  for (BlockArgument arg : func.getBody().front().getArguments())
    argTypes.push_back(arg.getType());
  func.setFunctionType(FunctionType::get(ctx, argTypes, /*results=*/{}));

  // 3. Lower affine.min -> arith.minsi.
  SmallVector<affine::AffineMinOp> minOps;
  func.walk([&](affine::AffineMinOp op) { minOps.push_back(op); });
  for (affine::AffineMinOp minOp : minOps) {
    OpBuilder b(minOp);
    Location loc = minOp.getLoc();
    AffineMap map = minOp.getAffineMap();
    ValueRange operands = minOp.getOperands();
    SmallVector<Value> results;
    for (AffineExpr expr : map.getResults())
      results.push_back(mlir::affine::expandAffineExpr(
          b, loc, expr,
          operands.take_front(map.getNumDims()),
          operands.drop_front(map.getNumDims())));
    Value minVal = results[0];
    for (unsigned i = 1; i < results.size(); ++i)
      minVal = b.create<arith::MinSIOp>(loc, minVal, results[i]);
    minOp.replaceAllUsesWith(minVal);
    minOp.erase();
  }
}

struct AscendCFinalizeKernelPass
    : public ::impl::AscendCFinalizeKernelPassBase<AscendCFinalizeKernelPass> {
  void runOnOperation() override { finalizeFunc(getOperation()); }
};

std::unique_ptr<Pass> createAscendCFinalizeKernelPass() {
  return std::make_unique<AscendCFinalizeKernelPass>();
}

} // namespace mlir::afir
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd build && ninja afir-opt -j$(nproc) 2>&1 | tail -3
bin/afir-opt ../test/Conversion/AscendCPrepareForEmit/finalize-kernel.mlir \
  --ascendc-finalize-kernel 2>&1 | FileCheck ../test/Conversion/AscendCPrepareForEmit/finalize-kernel.mlir
```
Expected: FileCheck passes.

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/AscendCPrepareForEmit/FinalizeKernelPass.cpp \
        test/Conversion/AscendCPrepareForEmit/finalize-kernel.mlir
git commit -m "feat(prepare-for-emit): implement AscendCFinalizeKernelPass"
```

---

## Task 3: Implement `AscendCFlattenGMPtrPass`

Extract the GM alloc promotion (step 7a), subview flattening (step 7b), and GM→GM copy lowering (step 7c) from `AscendCPrepareForEmitPass.cpp`. At this stage, `memref.dim` ops are still in the IR — use them directly for stride values.

**Files:**
- Modify: `lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp`
- Create: `test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir`

- [ ] **Step 1: Write the test**

`test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir`:
```mlir
// RUN: afir-opt %s --ascendc-flatten-gm-ptr 2>&1 | FileCheck %s

// 1D subview: set_global_buffer with subview offset should become flat ptr + offset.
// CHECK-LABEL: func.func @test_1d
// CHECK: emitasc.reinterpret_cast
// CHECK: emitasc.ptr_offset
// CHECK-NOT: memref.subview

// GM alloc should be promoted to a func arg.
// CHECK-LABEL: func.func @test_alloc_promoted
// CHECK-SAME: memref<16xf32>
// CHECK-NOT: memref.alloc

func.func @test_1d(%arg0: memref<1024xf32>) {
  %gt = ascendc.global_tensor : !ascendc.global_tensor<*xf32>
  %c8 = arith.constant 8 : index
  %sub = memref.subview %arg0[%c8][16][1]
    : memref<1024xf32> to memref<16xf32, strided<[1], offset: ?>>
  %cast = memref.cast %sub
    : memref<16xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
  ascendc.global_tensor.set_global_buffer %gt, %cast
    : !ascendc.global_tensor<*xf32>, memref<?xf32, strided<[?], offset: ?>>
  return
}

func.func @test_alloc_promoted() {
  %alloc = memref.alloc() : memref<16xf32>
  %cst = arith.constant 1.0 : f32
  linalg.fill ins(%cst : f32) outs(%alloc : memref<16xf32>)
  return
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd build && bin/afir-opt ../test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir \
  --ascendc-flatten-gm-ptr 2>&1 | head -10
```
Expected: subview still present (stub does nothing).

- [ ] **Step 3: Implement FlattenGMPtrPass**

Full content of `lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp`. This is a direct extraction of steps 7a/7b/7c from `AscendCPrepareForEmitPass.cpp`, with one key simplification: stride values are read from `memref.dim` ops already in the IR (no TilingData lookup needed).

```cpp
#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define GEN_PASS_DECL_ASCENDCFLATTENGMPTRPASS
#define GEN_PASS_DEF_ASCENDCFLATTENGMPTRPASS
#include "Conversion/Passes.h.inc"

#define DEBUG_TYPE "ascendc-flatten-gm-ptr"

using namespace mlir;
using namespace mlir::ascendc;
using namespace mlir::emitasc;

namespace mlir::afir {

static constexpr int64_t kGMSpace = 22;

// Materialize an OpFoldResult as an index Value.
static Value materializeOffset(OpBuilder &b, Location loc, OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    int64_t v = cast<IntegerAttr>(attr).getValue().getSExtValue();
    return b.create<arith::ConstantIndexOp>(loc, v);
  }
  return ofr.get<Value>();
}

// Walk a 1D subview chain to a root BlockArgument, accumulating flat offset.
static std::pair<BlockArgument, Value>
resolveSubviewChain1D(memref::SubViewOp leaf, OpBuilder &b, Location loc) {
  Value accOffset = b.create<arith::ConstantIndexOp>(loc, 0);
  Value cur = leaf.getResult();
  while (true) {
    auto sv = cur.getDefiningOp<memref::SubViewOp>();
    if (!sv)
      return {BlockArgument{}, Value{}};
    SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
    if (offs.size() != 1)
      return {BlockArgument{}, Value{}};
    Value off = materializeOffset(b, loc, offs[0]);
    accOffset = b.create<arith::AddIOp>(loc, accOffset, off);
    Value src = sv.getSource();
    if (auto castOp = src.getDefiningOp<memref::CastOp>())
      src = castOp.getSource();
    if (auto ba = dyn_cast<BlockArgument>(src))
      return {ba, accOffset};
    cur = src;
  }
}

// Get dim[1] of a memref block argument by inserting a memref.dim op.
// Used for 2D row-major stride: stride = dim[1].
static Value getDim1(OpBuilder &b, Location loc, BlockArgument ba) {
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  return b.create<memref::DimOp>(loc, ba, c1);
}

// Walk a value to its root BlockArgument + flat index offset (handles both
// 1D and 2D subview chains).
static std::pair<BlockArgument, Value>
resolveGMChain(Value start, OpBuilder &b, Location loc) {
  Value cur = start;
  Value acc = b.create<arith::ConstantIndexOp>(loc, 0);
  while (true) {
    if (auto sv = cur.getDefiningOp<memref::SubViewOp>()) {
      SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
      if (offs.size() == 1) {
        acc = b.create<arith::AddIOp>(loc, acc, materializeOffset(b, loc, offs[0]));
        cur = sv.getSource();
        continue;
      }
      if (offs.size() >= 2) {
        Value src = sv.getSource();
        if (auto castOp = src.getDefiningOp<memref::CastOp>())
          src = castOp.getSource();
        auto ba = dyn_cast<BlockArgument>(src);
        if (!ba)
          return {BlockArgument{}, Value{}};
        Value colStride = getDim1(b, loc, ba);
        Value row = materializeOffset(b, loc, offs[0]);
        Value col = materializeOffset(b, loc, offs[1]);
        Value flat = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, row, colStride), col);
        acc = b.create<arith::AddIOp>(loc, acc, flat);
        return {ba, acc};
      }
      return {BlockArgument{}, Value{}};
    }
    if (auto castOp = cur.getDefiningOp<memref::CastOp>()) {
      cur = castOp.getSource();
      continue;
    }
    if (auto ba = dyn_cast<BlockArgument>(cur))
      return {ba, acc};
    return {BlockArgument{}, Value{}};
  }
}

static void flattenGMPtr(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  Block &entry = func.getBody().front();
  Type indexTy = IndexType::get(ctx);
  Type i32Ty = IntegerType::get(ctx, 32);

  // ── 1. Promote top-level GM allocs to func args ───────────────────────────
  // Track dynamic sizes captured before erasing each alloc, indexed by the
  // new block arg number.
  DenseMap<unsigned, SmallVector<Value>> promotedArgDynSizes;
  {
    SmallVector<memref::AllocOp> gmAllocs;
    for (Operation &op : entry.without_terminator()) {
      auto allocOp = dyn_cast<memref::AllocOp>(&op);
      if (!allocOp)
        continue;
      if (cast<MemRefType>(allocOp.getResult().getType()).getMemorySpaceAsInt() == 0)
        gmAllocs.push_back(allocOp);
    }
    for (memref::AllocOp allocOp : gmAllocs) {
      auto origTy = cast<MemRefType>(allocOp.getResult().getType());
      SmallVector<int64_t> strides(origTy.getRank(), 1);
      auto stridedLayout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
      auto stridedTy = MemRefType::get(origTy.getShape(), origTy.getElementType(),
                                       stridedLayout);
      SmallVector<Value> dynSizes(allocOp.getDynamicSizes());
      BlockArgument newArg = entry.addArgument(stridedTy, allocOp.getLoc());
      promotedArgDynSizes[newArg.getArgNumber()] = dynSizes;
      OpBuilder b(allocOp);
      Value casted = b.create<memref::CastOp>(allocOp.getLoc(), origTy, newArg);
      allocOp.replaceAllUsesWith(casted);
      allocOp.erase();
    }
  }

  // ── 2. Flatten subview+set_global_buffer to flat pointer + offset ─────────
  auto mkFlatTy = [&](Type elem) {
    return MemRefType::get({ShapedType::kDynamic}, elem,
        MemRefLayoutAttrInterface{},
        IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));
  };

  SmallVector<GlobalTensorSetGlobalBufferOp> setGlobalBufferOps;
  func.walk([&](GlobalTensorSetGlobalBufferOp op) {
    if (op.getBuffer().getDefiningOp() &&
        isa<memref::SubViewOp>(op.getBuffer().getDefiningOp()))
      setGlobalBufferOps.push_back(op);
  });

  for (GlobalTensorSetGlobalBufferOp sgbOp : setGlobalBufferOps) {
    auto subview = sgbOp.getBuffer().getDefiningOp<memref::SubViewOp>();
    if (!subview)
      continue;
    OpBuilder b(sgbOp);
    Location loc = sgbOp.getLoc();
    BlockArgument baseArg;
    Value flatOffset;
    SmallVector<OpFoldResult> mixedOffsets = subview.getMixedOffsets();

    if (mixedOffsets.size() >= 2) {
      // 2D: accumulate row/col offsets, use dim[1] as col stride.
      Value accRow = b.create<arith::ConstantIndexOp>(loc, 0);
      Value accCol = b.create<arith::ConstantIndexOp>(loc, 0);
      Value cur = subview.getResult();
      bool ok = true;
      while (true) {
        auto sv = cur.getDefiningOp<memref::SubViewOp>();
        if (!sv) { ok = false; break; }
        SmallVector<OpFoldResult> offs = sv.getMixedOffsets();
        if (offs.size() < 2) { ok = false; break; }
        accRow = b.create<arith::AddIOp>(loc, accRow, materializeOffset(b, loc, offs[0]));
        accCol = b.create<arith::AddIOp>(loc, accCol, materializeOffset(b, loc, offs[1]));
        Value src = sv.getSource();
        if (auto castOp = src.getDefiningOp<memref::CastOp>())
          src = castOp.getSource();
        if (auto ba = dyn_cast<BlockArgument>(src)) { baseArg = ba; break; }
        cur = src;
      }
      if (!ok || !baseArg)
        continue;
      Value colStride = getDim1(b, loc, baseArg);
      flatOffset = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, accRow, colStride), accCol);
    } else {
      auto [ba, acc] = resolveSubviewChain1D(subview, b, loc);
      if (!ba)
        continue;
      baseArg = ba;
      flatOffset = acc;
    }

    Value flatOffsetI32 = b.create<arith::IndexCastOp>(loc, i32Ty, flatOffset);
    Value flatBase = b.create<emitasc::ReinterpretCastOp>(
        loc, mkFlatTy(cast<MemRefType>(baseArg.getType()).getElementType()), baseArg);
    b.create<GlobalTensorSetGlobalBufferOp>(loc, sgbOp.getTensor(), flatBase, flatOffsetI32);
    sgbOp.erase();
    if (subview.use_empty())
      subview.erase();
  }

  // Erase dead subview chains.
  {
    SmallVector<memref::SubViewOp> dead;
    func.walk([&](memref::SubViewOp sv) {
      if (sv.use_empty()) dead.push_back(sv);
    });
    for (auto sv : dead) sv.erase();
  }

  // ── 3. GM→GM memref.copy → memmove verbatim ──────────────────────────────
  {
    SmallVector<memref::CopyOp> gmCopies;
    func.walk([&](memref::CopyOp op) {
      if (cast<MemRefType>(op.getSource().getType()).getMemorySpaceAsInt() == 0 &&
          cast<MemRefType>(op.getTarget().getType()).getMemorySpaceAsInt() == 0)
        gmCopies.push_back(op);
    });
    for (memref::CopyOp copyOp : gmCopies) {
      OpBuilder b(copyOp);
      Location loc = copyOp.getLoc();
      auto [srcArg, srcOff] = resolveGMChain(copyOp.getSource(), b, loc);
      auto [dstArg, dstOff] = resolveGMChain(copyOp.getTarget(), b, loc);
      if (!srcArg || !dstArg) {
        LLVM_DEBUG(llvm::dbgs() << "[flatten-gm-ptr] unresolved GM copy\n");
        continue;
      }
      Type srcElem = cast<MemRefType>(srcArg.getType()).getElementType();
      Type dstElem = cast<MemRefType>(dstArg.getType()).getElementType();
      Value srcBase = b.create<emitasc::ReinterpretCastOp>(loc, mkFlatTy(srcElem), srcArg);
      Value dstBase = b.create<emitasc::ReinterpretCastOp>(loc, mkFlatTy(dstElem), dstArg);
      Value srcPtr = b.create<emitasc::PtrOffsetOp>(
          loc, mkFlatTy(srcElem), srcBase, IntegerAttr{}, srcOff);
      Value dstPtr = b.create<emitasc::PtrOffsetOp>(
          loc, mkFlatTy(dstElem), dstBase, IntegerAttr{}, dstOff);
      // Byte count from promoted alloc's captured dynamic sizes.
      Value byteCount;
      {
        auto it = promotedArgDynSizes.find(srcArg.getArgNumber());
        if (it != promotedArgDynSizes.end() && !it->second.empty()) {
          Value count = b.create<arith::ConstantIndexOp>(loc, 1);
          for (Value sz : it->second)
            count = b.create<arith::MulIOp>(loc, count, sz);
          int64_t elemBytes = srcElem.getIntOrFloatBitWidth() / 8;
          byteCount = b.create<arith::MulIOp>(
              loc, count, b.create<arith::ConstantIndexOp>(loc, elemBytes));
        } else {
          auto mrt = cast<MemRefType>(copyOp.getSource().getType());
          int64_t staticElems = 1;
          bool ok = true;
          for (int64_t d : mrt.getShape()) {
            if (d == ShapedType::kDynamic) { ok = false; break; }
            staticElems *= d;
          }
          if (!ok) continue;
          byteCount = b.create<arith::ConstantIndexOp>(
              loc, staticElems * (srcElem.getIntOrFloatBitWidth() / 8));
        }
      }
      b.create<emitasc::VerbatimOp>(loc, b.getStringAttr("memmove($1, $2, $3)"),
                                    ValueRange{dstPtr, srcPtr, byteCount});
      copyOp.erase();
    }
    SmallVector<Operation *> dead;
    func.walk([&](Operation *op) {
      if (op->use_empty() && isa<memref::SubViewOp, memref::CastOp>(op))
        dead.push_back(op);
    });
    for (Operation *op : dead) op->erase();
  }

  // Update function type.
  SmallVector<Type> newArgTypes;
  for (BlockArgument arg : entry.getArguments())
    newArgTypes.push_back(arg.getType());
  func.setFunctionType(FunctionType::get(ctx, newArgTypes,
                                         func.getFunctionType().getResults()));
}

struct AscendCFlattenGMPtrPass
    : public ::impl::AscendCFlattenGMPtrPassBase<AscendCFlattenGMPtrPass> {
  void runOnOperation() override { flattenGMPtr(getOperation()); }
};

std::unique_ptr<Pass> createAscendCFlattenGMPtrPass() {
  return std::make_unique<AscendCFlattenGMPtrPass>();
}

} // namespace mlir::afir
```

- [ ] **Step 4: Build and run test**

```bash
cd build && ninja afir-opt -j$(nproc) 2>&1 | tail -3
bin/afir-opt ../test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir \
  --ascendc-flatten-gm-ptr 2>&1 | \
  FileCheck ../test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir
```
Expected: FileCheck passes.

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/AscendCPrepareForEmit/FlattenGMPtrPass.cpp \
        test/Conversion/AscendCPrepareForEmit/flatten-gm-ptr.mlir
git commit -m "feat(prepare-for-emit): implement AscendCFlattenGMPtrPass"
```

---

## Task 4: Implement `AscendCPackTilingDataPass`

Extract TilingData building (steps 1-8 of the original pass) with Phase A removed. The pass reads only `vector_plan.tiling_infos`; if absent, it emits an error and fails.

**Files:**
- Modify: `lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp`
- Create: `test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir`

- [ ] **Step 1: Write the test**

`test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir`:
```mlir
// RUN: afir-opt %s --ascendc-pack-tiling-data 2>&1 | FileCheck %s
//
// Verify:
//  - Tiling args replaced by emitasc.member reads
//  - memref.dim replaced by emitasc.member read
//  - dim_arg0_0 field present (from memref.dim %arg0, 0 used in body)
//  - No i64/index tiling block args remain

// CHECK: func.func @relu(
// CHECK-SAME: memref<1024xf32>
// CHECK-SAME: memref<1024xf32>
// CHECK-SAME: !emitasc.py_struct<"TilingData"
// CHECK-NOT: index)
// CHECK: emitasc.member {{.*}} "XBLOCK"
// CHECK: emitasc.member {{.*}} "XBLOCK_SUB"
// CHECK: emitasc.member {{.*}} "dim_arg0_0"
// CHECK: emitasc.member {{.*}} "dim_arg1_0"
// CHECK-NOT: TB_M

module attributes {
  vector_plan.tiling_infos = [{
    fields = [
      {abi_index = 0 : i32, arg_index = 2 : i32, default_value = 128 : i64,
       kind = "tunable", name = "XBLOCK"},
      {abi_index = 1 : i32, arg_index = 3 : i32, default_value = 16 : i64,
       kind = "tunable", name = "XBLOCK_SUB"}
    ],
    kernel_id = "relu"
  }]
} {
  func.func @relu(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>,
                  %xblock: index, %xblock_sub: index) {
    %c0 = arith.constant 0 : index
    %d0 = memref.dim %arg0, %c0 : memref<1024xf32>
    %d1 = memref.dim %arg1, %c0 : memref<1024xf32>
    return
  }
}
```

- [ ] **Step 2: Run to verify fails**

```bash
cd build && bin/afir-opt ../test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir \
  --ascendc-pack-tiling-data 2>&1 | head -10
```
Expected: unchanged output (stub).

- [ ] **Step 3: Implement PackTilingDataPass**

Full content of `lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp`:

```cpp
#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_ASCENDCPACKTILINGDATAPASS
#define GEN_PASS_DEF_ASCENDCPACKTILINGDATAPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::emitasc;

namespace mlir::afir {

static constexpr int64_t kGMSpace = 22;

struct DimKey {
  unsigned argNumber;
  int64_t dimIndex;
  bool operator==(const DimKey &o) const {
    return argNumber == o.argNumber && dimIndex == o.dimIndex;
  }
};

static emitasc::PyStructType buildTilingDataType(MLIRContext *ctx,
                                                 ArrayRef<StringRef> names) {
  SmallVector<Attribute> typeAttrs, nameAttrs;
  Type i64 = IntegerType::get(ctx, 64);
  for (auto name : names) {
    typeAttrs.push_back(TypeAttr::get(i64));
    nameAttrs.push_back(StringAttr::get(ctx, name));
  }
  return emitasc::PyStructType::get(
      ctx, StringAttr::get(ctx, "TilingData"),
      ArrayAttr::get(ctx, typeAttrs), ArrayAttr::get(ctx, nameAttrs));
}

static LogicalResult packTilingData(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  Block &entry = func.getBody().front();
  Type indexTy = IndexType::get(ctx);
  Type i64Ty = IntegerType::get(ctx, 64);

  // ── 1. Read tiling args from vector_plan.tiling_infos ────────────────────
  SmallVector<BlockArgument> tilingArgs;
  SmallVector<std::string> tilingArgNames;

  auto moduleOp = func->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return func.emitError("PackTilingData: func not inside a module");

  auto tilingInfosAttr = moduleOp->getAttrOfType<ArrayAttr>("vector_plan.tiling_infos");
  if (!tilingInfosAttr)
    return func.emitError("PackTilingData: vector_plan.tiling_infos not found on module; "
                          "ensure TilePlanGen ran before this pass");

  for (Attribute infoAttr : tilingInfosAttr) {
    auto info = dyn_cast<DictionaryAttr>(infoAttr);
    if (!info) continue;
    auto kid = dyn_cast_or_null<StringAttr>(info.get("kernel_id"));
    if (!kid || kid.getValue() != func.getName()) continue;
    auto fieldsAttr = dyn_cast_or_null<ArrayAttr>(info.get("fields"));
    if (!fieldsAttr) break;
    for (Attribute fa : fieldsAttr) {
      auto field = dyn_cast<DictionaryAttr>(fa);
      if (!field) continue;
      unsigned argIdx = (unsigned)cast<IntegerAttr>(field.get("arg_index"))
                            .getValue().getSExtValue();
      if (argIdx >= entry.getNumArguments())
        return func.emitError("PackTilingData: arg_index ") << argIdx << " out of range";
      tilingArgs.push_back(cast<BlockArgument>(entry.getArgument(argIdx)));
      tilingArgNames.push_back(cast<StringAttr>(field.get("name")).getValue().str());
    }
    break;
  }

  // ── 2. Collect memref.dim uses on block args ──────────────────────────────
  SmallVector<DimKey> dimKeys;
  SmallVector<memref::DimOp> dimOps;

  auto addDimKey = [&](unsigned argNum, int64_t dimIdx) {
    DimKey key{argNum, dimIdx};
    if (llvm::none_of(dimKeys, [&](const DimKey &k) { return k == key; }))
      dimKeys.push_back(key);
  };

  func.walk([&](memref::DimOp dimOp) {
    auto arg = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!arg) return;
    auto constOp = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
    if (!constOp) return;
    auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
    if (!intAttr) return;
    addDimKey(arg.getArgNumber(), intAttr.getValue().getSExtValue());
    dimOps.push_back(dimOp);
  });

  // ── 3. Build TilingData field names ──────────────────────────────────────
  SmallVector<std::string> tilingNameStorage;
  for (const std::string &n : tilingArgNames) tilingNameStorage.push_back(n);
  for (const DimKey &k : dimKeys)
    tilingNameStorage.push_back("dim_arg" + std::to_string(k.argNumber) +
                                "_" + std::to_string(k.dimIndex));
  SmallVector<StringRef> tilingNames;
  for (const std::string &s : tilingNameStorage) tilingNames.push_back(s);

  if (tilingNames.empty())
    return success(); // nothing to pack

  // ── 4. Build TilingData type and GM pointer arg ───────────────────────────
  auto tilingStructTy = buildTilingDataType(ctx, tilingNames);
  auto tilingArgTy = MemRefType::get(
      {ShapedType::kDynamic}, tilingStructTy, MemRefLayoutAttrInterface{},
      IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));

  // ── 5. Inject TilingData arg and extract fields ───────────────────────────
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&entry);
  BlockArgument tilingArg = entry.addArgument(tilingArgTy, func.getLoc());
  Value localStruct = builder.create<emitasc::CopyStructOp>(
      func.getLoc(), tilingStructTy, tilingArg);

  SmallVector<Value> tilingFieldVals;
  for (unsigned i = 0; i < tilingNames.size(); ++i)
    tilingFieldVals.push_back(builder.create<emitasc::MemberOp>(
        func.getLoc(), i64Ty, localStruct, builder.getStringAttr(tilingNames[i])));

  // ── 6. Replace tiling arg uses (index-cast i64 → index) ──────────────────
  for (unsigned i = 0; i < tilingArgs.size(); ++i) {
    Value val = tilingFieldVals[i];
    if (tilingArgs[i].getType().isIndex())
      val = builder.create<arith::IndexCastOp>(func.getLoc(), indexTy, val);
    tilingArgs[i].replaceAllUsesWith(val);
  }

  // ── 7. Replace memref.dim uses with index-cast of tiling fields ───────────
  unsigned dimFieldBase = tilingArgs.size();
  for (memref::DimOp dimOp : dimOps) {
    auto arg = cast<BlockArgument>(dimOp.getSource());
    auto constOp = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
    int64_t dimIdxVal = cast<IntegerAttr>(constOp.getValue()).getValue().getSExtValue();
    DimKey key{arg.getArgNumber(), dimIdxVal};
    unsigned k = llvm::find_if(dimKeys, [&](const DimKey &d) { return d == key; })
                 - dimKeys.begin();
    Value i64Val = tilingFieldVals[dimFieldBase + k];
    OpBuilder b(dimOp);
    Value idxVal = b.create<arith::IndexCastOp>(dimOp.getLoc(), indexTy, i64Val);
    dimOp.replaceAllUsesWith(idxVal);
    dimOp.erase();
  }

  // ── 8. Erase tiling block args (reverse order) ────────────────────────────
  SmallVector<unsigned> toErase;
  for (BlockArgument a : tilingArgs) toErase.push_back(a.getArgNumber());
  llvm::sort(toErase, std::greater<unsigned>());
  for (unsigned idx : toErase) entry.eraseArgument(idx);
  if (func->getAttr("arg_attrs")) func->removeAttr("arg_attrs");

  // ── 9. Update function type ───────────────────────────────────────────────
  SmallVector<Type> newArgTypes;
  for (BlockArgument a : entry.getArguments()) newArgTypes.push_back(a.getType());
  func.setFunctionType(FunctionType::get(ctx, newArgTypes,
                                         func.getFunctionType().getResults()));

  // ── 10. Declare TilingData struct at module level ─────────────────────────
  OpBuilder modBuilder(ctx);
  modBuilder.setInsertionPointToStart(moduleOp.getBody());
  modBuilder.create<emitasc::DeclarePyStructOp>(func.getLoc(),
                                                TypeAttr::get(tilingStructTy));

  return success();
}

struct AscendCPackTilingDataPass
    : public ::impl::AscendCPackTilingDataPassBase<AscendCPackTilingDataPass> {
  void runOnOperation() override {
    if (failed(packTilingData(getOperation())))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createAscendCPackTilingDataPass() {
  return std::make_unique<AscendCPackTilingDataPass>();
}

} // namespace mlir::afir
```

- [ ] **Step 4: Build and run test**

```bash
cd build && ninja afir-opt -j$(nproc) 2>&1 | tail -3
bin/afir-opt ../test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir \
  --ascendc-pack-tiling-data 2>&1 | \
  FileCheck ../test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir
```
Expected: FileCheck passes.

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp \
        test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir
git commit -m "feat(prepare-for-emit): implement AscendCPackTilingDataPass (Phase B only)"
```

---

## Task 5: Wire new passes into Pipeline.cpp and verify E2E

Replace `createAscendCPrepareForEmitPass()` in `--vector-plan-codegen` with the three new passes in order: `FlattenGMPtr → PackTilingData → FinalizeKernel`.

**Files:**
- Modify: `lib/Conversion/VectorPlan/Pipeline.cpp`

- [ ] **Step 1: Update Pipeline.cpp**

In `lib/Conversion/VectorPlan/Pipeline.cpp`, replace:
```cpp
pm.addNestedPass<func::FuncOp>(createAscendCPrepareForEmitPass());
```
with:
```cpp
pm.addNestedPass<func::FuncOp>(createAscendCFlattenGMPtrPass());
pm.addNestedPass<func::FuncOp>(createAscendCPackTilingDataPass());
pm.addNestedPass<func::FuncOp>(createAscendCFinalizeKernelPass());
```

- [ ] **Step 2: Build**

```bash
cd build && ninja afir-opt afir-translate -j$(nproc) 2>&1 | tail -3
```
Expected: build succeeds.

- [ ] **Step 3: Run existing E2E test**

```bash
cd build && bin/afir-opt ../test/Conversion/VectorPlanCodegen/codegen-relu-e2e.mlir \
  --vector-plan-codegen 2>&1 | \
  FileCheck ../test/Conversion/VectorPlanCodegen/codegen-relu-e2e.mlir --check-prefix=MLIR
bin/afir-opt ../test/Conversion/VectorPlanCodegen/codegen-relu-e2e.mlir \
  --vector-plan-codegen 2>&1 | \
  bin/afir-translate -mlir-to-cann 2>&1 | \
  FileCheck ../test/Conversion/VectorPlanCodegen/codegen-relu-e2e.mlir --check-prefix=CANN
```
Expected: both FileCheck runs pass.

- [ ] **Step 4: Run simulator E2E**

```bash
source ../examples/env.sh
bash ../examples/relu-e2e/run.sh 2>&1 | grep "session.validation"
```
Expected: `session.validation=pass`

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/VectorPlan/Pipeline.cpp
git commit -m "refactor(pipeline): switch vector-plan-codegen to three new sub-passes"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ Split into 3 passes: Tasks 2, 3, 4
- ✅ No Phase A fallback in PackTilingData: Task 4 step 3 — hard error if no tiling_infos
- ✅ Existing pass untouched: Pipeline.cpp is the only change; original pass file not modified
- ✅ E2E verification: Task 5 steps 3-4

**2. Placeholder scan:** None found.

**3. Type consistency:**
- `createAscendCFlattenGMPtrPass()` declared in Task 1, used in Task 5 ✅
- `createAscendCPackTilingDataPass()` declared in Task 1, used in Task 5 ✅
- `createAscendCFinalizeKernelPass()` declared in Task 1, used in Task 5 ✅
- `kGMSpace = 22` defined locally in both FlattenGMPtr and PackTilingData (acceptable duplication; would be a shared header concern for a later cleanup pass)
