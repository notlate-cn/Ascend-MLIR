# CanonicalizeCannSignaturePass + cann-translate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `--canonicalize-cann-signature` MLIR pass and `afir-opt -mlir-to-cann` translation so the pipeline produces kernel.cpp directly compatible with the Compiler/Validator/Autotuner toolchain.

**Architecture:** A new MLIR pass (`CanonicalizeCannSignaturePass`) transforms the func.func signature from PyAsc internal format `(inputs, tiling_ptr, outputs)` to CANN standard `(inputs, outputs, workspace, tiling_byvalue)`. A new translation (`translateToCannKernel`) registered as `mlir-to-cann` in `afir-opt` emits CANN-style C++ (`GM_ADDR` params, by-value tiling struct) by handling `func::FuncOp` specially and delegating all body ops to PyAsc's existing `emitOperation()`.

**Tech Stack:** C++17, MLIR (TableGen pass registration, FuncOp rewriting, BlockArgument replacement), PyAsc `CodeEmitter` + `emitOperation()` from `Common.h`, LLVM `MlirTranslateMain`.

**Reference spec:** `docs/superpowers/specs/2026-03-21-canonicalize-cann-signature-design.md`

**Test example:** `examples/broadcast-add-reduce/` (2 inputs, 1 output, 4×`int64_t` TilingData)

---

## File Map

### New files
| File | Responsibility |
|------|----------------|
| `include/Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h` | Declares `createCanonicalizeCannSignaturePass()` |
| `lib/Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.cpp` | Pass implementation: reorder args, remove copy_struct, add workspace+tiling-by-value |
| `lib/Conversion/CanonicalizeCannSignature/CMakeLists.txt` | `add_mlir_library(CanonicalizeCannSignatureConversion ...)` |
| `include/Target/CannKernel/CannTranslation.h` | Declares `translateToCannKernel(Operation*, raw_ostream&)` |
| `lib/Target/CannKernel/CannTranslation.cpp` | Emits CANN-style C++: struct decl + `GM_ADDR` signature + body via `emitOperation()` |
| `lib/Target/CannKernel/CMakeLists.txt` | `add_mlir_library(CannKernelTranslation ...)` |
| `test/Conversion/canonicalize-cann-signature.mlir` | Lit FileCheck test for the pass |
| `test/Target/cann-translate.mlir` | Lit FileCheck test for the translator |

### Modified files
| File | Change |
|------|--------|
| `include/Conversion/Passes.td` | Add `CanonicalizeCannSignaturePass` entry |
| `include/Conversion/Passes.h` | Include new pass header |
| `lib/Conversion/CMakeLists.txt` | `add_subdirectory(CanonicalizeCannSignature)` |
| `tools/afir-opt/afir-opt.cpp` | Include `CannTranslation.h`, register `mlir-to-cann` via `TranslateFromMLIRRegistration`, switch to `MlirTranslateMain` or add translation alongside `MlirOptMain` |
| `tools/afir-opt/CMakeLists.txt` | Link `CanonicalizeCannSignatureConversion`, `CannKernelTranslation`, `MLIRTargetAsc`, `MLIRTranslation` |
| `examples/broadcast-add-reduce/run.sh` | Add Stage 7b (pass) and update Stage 8 (use `-mlir-to-cann`) |

---

## Task 1: Add Pass TableGen entry and header

**Files:**
- Modify: `include/Conversion/Passes.td`
- Create: `include/Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h`
- Modify: `include/Conversion/Passes.h`

- [ ] **Step 1: Add pass entry to Passes.td**

Append before `#endif // AFIR_CONVERSION_PASSES` in `include/Conversion/Passes.td`:

```tablegen
//===----------------------------------------------------------------------===//
// CanonicalizeCannSignature
//===----------------------------------------------------------------------===//

def CanonicalizeCannSignaturePass : Pass<"canonicalize-cann-signature", "mlir::ModuleOp"> {
  let summary = "Canonicalize aicore kernel signature to CANN calling convention";
  let description = [{
    Transforms func.func ops with {ascendc.aicore, ascendc.global} attributes from
    PyAsc internal format (inputs..., tiling_memref, outputs...) to CANN standard:
    (inputs..., outputs..., workspace:memref<ui8>, tiling:PyStructType).

    Also:
    - Removes the emitasc.copy_struct op (tiling is now by-value).
    - Adds cann.num_inputs = N : i32 attribute to the function.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::memref::MemRefDialect"
  ];
}
```

- [ ] **Step 2: Create pass header**

Create `include/Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h`:

```cpp
//===- CanonicalizeCannSignaturePass.h ----------------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_CANONICALIZEANNSIGNATURE_PASS_H
#define AFIR_CONVERSION_CANONICALIZEANNSIGNATURE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::afir {
std::unique_ptr<Pass> createCanonicalizeCannSignaturePass();
}  // namespace mlir::afir

#endif // AFIR_CONVERSION_CANONICALIZEANNSIGNATURE_PASS_H
```

- [ ] **Step 3: Include in Passes.h**

In `include/Conversion/Passes.h`, add after the last existing `#include "Conversion/..."` line:

```cpp
#include "Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h"
```

- [ ] **Step 4: Verify TableGen compiles**

```bash
cd /home/niu/code/Ascend-MLIR
orb -m xvm bash -c "cd /home/niu/code/Ascend-MLIR && cmake --build build --target AFIRConversionPassIncGen 2>&1 | tail -5"
```

Expected: build succeeds, no errors.

- [ ] **Step 5: Commit**

```bash
git add include/Conversion/Passes.td \
        include/Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h \
        include/Conversion/Passes.h
git commit -m "feat(cann): add CanonicalizeCannSignaturePass TableGen entry and header"
```

---

## Task 2: Implement CanonicalizeCannSignaturePass

**Files:**
- Create: `lib/Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.cpp`
- Create: `lib/Conversion/CanonicalizeCannSignature/CMakeLists.txt`
- Modify: `lib/Conversion/CMakeLists.txt`

- [ ] **Step 1: Write the failing lit test**

Create `test/Conversion/canonicalize-cann-signature.mlir`:

```mlir
// RUN: afir-opt --canonicalize-cann-signature %s | FileCheck %s

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
      %tiling_data: memref<?x!emitasc.py_struct<"TilingData",
          [i64, i64, i64, i64],
          ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, 22 : i32>,
      %output: memref<?xf16>
  ) attributes {ascendc.aicore, ascendc.global} {
    %local_tiling = emitasc.copy_struct %tiling_data
        : memref<?x!emitasc.py_struct<"TilingData",
              [i64, i64, i64, i64],
              ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>, 22 : i32>,
          !emitasc.py_struct<"TilingData",
              [i64, i64, i64, i64],
              ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>
    %tb_m = emitasc.member %local_tiling "TB_M"
        : !emitasc.py_struct<"TilingData",
              [i64, i64, i64, i64],
              ["TB_M", "TB_N", "dim_arg0_0", "dim_arg1_1"]>,
          i64
    func.return
  }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source examples/env.sh
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  afir-opt --canonicalize-cann-signature \
  /home/niu/code/Ascend-MLIR/test/Conversion/canonicalize-cann-signature.mlir 2>&1 | head -5"
```

Expected: error `unknown pass name 'canonicalize-cann-signature'`

- [ ] **Step 3: Create CMakeLists.txt**

Create `lib/Conversion/CanonicalizeCannSignature/CMakeLists.txt`:

```cmake
add_mlir_library(CanonicalizeCannSignatureConversion
  CanonicalizeCannSignaturePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIREmitAsc
  MLIRMemRefDialect
  MLIRTransforms
  MLIRIR
)
```

- [ ] **Step 4: Wire into lib/Conversion/CMakeLists.txt**

Append to `lib/Conversion/CMakeLists.txt`:

```cmake
add_subdirectory(CanonicalizeCannSignature)
```

- [ ] **Step 5: Implement the pass**

Create `lib/Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.cpp`:

```cpp
//===- CanonicalizeCannSignaturePass.cpp - CANN signature canonicalization -===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// Transforms aicore kernel func.func from PyAsc internal format:
//   (inputs..., memref<?x!emitasc.py_struct<...>, 22:i32>, outputs...)
// to CANN standard format:
//   (inputs..., outputs..., memref<ui8>, !emitasc.py_struct<...>)
//
// Also removes emitasc.copy_struct and adds cann.num_inputs attribute.
//
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_CANONICALIZECANNSIGNATUREPASS
#define GEN_PASS_DEF_CANONICALIZECANNSIGNATUREPASS
#include "Conversion/Passes.h.inc"

#include "Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

// Returns true if `type` is the tiling memref: memref<?x!emitasc.py_struct<...>, 22:i32>
static bool isTilingMemref(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType) return false;
  if (!isa<emitasc::PyStructType>(memrefType.getElementType())) return false;
  auto memSpace = memrefType.getMemorySpace();
  if (!memSpace) return false;
  auto intAttr = dyn_cast<IntegerAttr>(memSpace);
  return intAttr && intAttr.getInt() == 22;
}

// Returns the PyStructType extracted from a tiling memref argument.
static emitasc::PyStructType getTilingStructType(Type tilingMemrefType) {
  return cast<emitasc::PyStructType>(
      cast<MemRefType>(tilingMemrefType).getElementType());
}

static LogicalResult canonicalizeFuncOp(func::FuncOp funcOp,
                                        IRRewriter &rewriter) {
  // Only process aicore global kernels.
  if (!funcOp->hasAttr(ascendc::attr::global) ||
      !funcOp->hasAttr(ascendc::attr::aicore))
    return success();

  auto args = funcOp.getArguments();
  int tilingIdx = -1;
  for (int i = 0, e = args.size(); i < e; ++i) {
    if (isTilingMemref(args[i].getType())) {
      if (tilingIdx != -1)
        return funcOp.emitOpError("has multiple tiling memref arguments");
      tilingIdx = i;
    }
  }
  if (tilingIdx == -1)
    return funcOp.emitOpError("has no tiling memref argument "
                               "(memref<?x!emitasc.py_struct<...>, 22:i32>)");

  // Find the copy_struct op that uses the tiling arg.
  emitasc::CopyStructOp copyOp;
  for (Operation *user : args[tilingIdx].getUsers()) {
    if (auto op = dyn_cast<emitasc::CopyStructOp>(user)) {
      if (copyOp)
        return funcOp.emitOpError("tiling arg has multiple copy_struct users");
      copyOp = op;
    }
  }
  if (!copyOp)
    return funcOp.emitOpError("tiling arg has no emitasc.copy_struct user");

  // Verify copy_struct result is only used by emitasc.member ops.
  for (Operation *user : copyOp.getResult().getUsers()) {
    if (!isa<emitasc::MemberOp>(user))
      return funcOp.emitOpError("copy_struct result has non-member user: ")
             << user->getName();
  }

  int numInputs = tilingIdx;
  // outputs = args after tilingIdx (original outputs)
  emitasc::PyStructType tilingStructType =
      getTilingStructType(args[tilingIdx].getType());
  MLIRContext *ctx = funcOp.getContext();

  // Build new arg types: inputs..., outputs..., memref<ui8>, PyStructType
  SmallVector<Type> newArgTypes;
  // inputs
  for (int i = 0; i < tilingIdx; ++i)
    newArgTypes.push_back(args[i].getType());
  // outputs (args after tiling)
  for (int i = tilingIdx + 1, e = args.size(); i < e; ++i)
    newArgTypes.push_back(args[i].getType());
  // workspace
  Type workspaceType = MemRefType::get({}, IntegerType::get(ctx, 8, IntegerType::Unsigned));
  newArgTypes.push_back(workspaceType);
  // tiling by-value
  newArgTypes.push_back(tilingStructType);

  // Rebuild function type (void return).
  auto newFuncType = FunctionType::get(ctx, newArgTypes, {});

  rewriter.setInsertionPoint(funcOp);

  // Build location list for new args.
  SmallVector<Location> newArgLocs;
  for (int i = 0; i < tilingIdx; ++i)
    newArgLocs.push_back(args[i].getLoc());
  for (int i = tilingIdx + 1, e = args.size(); i < e; ++i)
    newArgLocs.push_back(args[i].getLoc());
  newArgLocs.push_back(funcOp.getLoc()); // workspace
  newArgLocs.push_back(funcOp.getLoc()); // tiling

  // Insert new args at end of block (we'll move them and erase old ones).
  Block &entryBlock = funcOp.getBody().front();

  // Add workspace and tiling as new block args.
  BlockArgument wsArg = entryBlock.addArgument(workspaceType, funcOp.getLoc());
  BlockArgument tilingArg =
      entryBlock.addArgument(tilingStructType, funcOp.getLoc());

  // Replace copy_struct result uses with new tiling arg.
  rewriter.replaceAllUsesWith(copyOp.getResult(), tilingArg);
  rewriter.eraseOp(copyOp);

  // Move outputs after inputs (shift tiling arg out).
  // Currently: inputs(0..tilingIdx-1), tiling(tilingIdx), outputs(tilingIdx+1..N-1),
  //            ws(N), tilingByVal(N+1)
  // Target:    inputs(0..tilingIdx-1), outputs(tilingIdx+1..N-1), ws(N), tilingByVal(N+1)
  // Just erase the old tiling memref arg (tilingIdx). Its only user (copy_struct) was erased.
  entryBlock.eraseArgument(tilingIdx);

  // Update function type and cann.num_inputs attribute.
  funcOp.setType(newFuncType);
  funcOp->setAttr("cann.num_inputs",
                  IntegerAttr::get(IntegerType::get(ctx, 32), numInputs));

  return success();
}

struct CanonicalizeCannSignaturePass
    : public ::impl::CanonicalizeCannSignaturePassBase<
          CanonicalizeCannSignaturePass> {
  using CanonicalizeCannSignaturePassBase::CanonicalizeCannSignaturePassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(module.getContext());
    WalkResult result = module.walk([&](func::FuncOp funcOp) {
      if (failed(canonicalizeFuncOp(funcOp, rewriter)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

namespace mlir::afir {
std::unique_ptr<Pass> createCanonicalizeCannSignaturePass() {
  return std::make_unique<CanonicalizeCannSignaturePass>();
}
} // namespace mlir::afir
```

- [ ] **Step 6: Link into afir-opt CMakeLists.txt**

Add `CanonicalizeCannSignatureConversion` to the `target_link_libraries` in `tools/afir-opt/CMakeLists.txt`:

```cmake
    CanonicalizeCannSignatureConversion
```

(After the existing `AscendCPrepareForEmitConversion` line.)

- [ ] **Step 7: Build**

```bash
orb -m xvm bash -c "cd /home/niu/code/Ascend-MLIR && \
  ./scripts/build.sh --build-project 2>&1 | tail -20"
```

Expected: build succeeds with `afir-opt` linked.

- [ ] **Step 8: Run the lit test**

```bash
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  /home/niu/code/llvm-project/build/bin/FileCheck \
  /home/niu/code/Ascend-MLIR/test/Conversion/canonicalize-cann-signature.mlir \
  <<< \$(afir-opt --canonicalize-cann-signature \
  /home/niu/code/Ascend-MLIR/test/Conversion/canonicalize-cann-signature.mlir)"
```

Expected: FileCheck passes, no errors.

- [ ] **Step 9: Commit**

```bash
git add lib/Conversion/CanonicalizeCannSignature/ \
        lib/Conversion/CMakeLists.txt \
        tools/afir-opt/CMakeLists.txt \
        test/Conversion/canonicalize-cann-signature.mlir
git commit -m "feat(cann): implement CanonicalizeCannSignaturePass"
```

---

## Task 3: Implement cann-translate (translateToCannKernel)

**Files:**
- Create: `include/Target/CannKernel/CannTranslation.h`
- Create: `lib/Target/CannKernel/CannTranslation.cpp`
- Create: `lib/Target/CannKernel/CMakeLists.txt`
- Create: `lib/Target/CMakeLists.txt`

- [ ] **Step 1: Write the failing lit test**

Create `test/Target/cann-translate.mlir`:

```mlir
// RUN: afir-opt -mlir-to-cann %s | FileCheck %s

// CHECK: struct TilingData {
// CHECK-NEXT: int64_t TB_M;
// CHECK-NEXT: int64_t TB_N;
// CHECK-NEXT: int64_t dim_arg0_0;
// CHECK-NEXT: int64_t dim_arg1_1;
// CHECK: extern "C" __global__ __aicore__ void broadcast_add_reducesum(
// CHECK-NEXT: GM_ADDR
// CHECK-NEXT: GM_ADDR
// CHECK-NEXT: GM_ADDR
// CHECK-NEXT: GM_ADDR
// CHECK-NEXT: TilingData
// CHECK: ) {
// CHECK-NOT: copy_struct
// CHECK: .TB_M

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

- [ ] **Step 2: Run test to verify it fails**

```bash
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  afir-opt -mlir-to-cann /home/niu/code/Ascend-MLIR/test/Target/cann-translate.mlir 2>&1 | head -5"
```

Expected: error (unknown flag or unregistered translation).

- [ ] **Step 3: Create header**

Create `include/Target/CannKernel/CannTranslation.h`:

```cpp
//===- CannTranslation.h - CANN kernel C++ translation ----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_TARGET_CANNKERNEL_CANNTRANSLATION_H
#define AFIR_TARGET_CANNKERNEL_CANNTRANSLATION_H

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
/// Translates a module containing CANN-signature aicore functions to C++.
/// Expects func.func args in order: inputs, outputs, workspace:memref<ui8>,
/// tiling:!emitasc.py_struct<...>, with cann.num_inputs attr.
LogicalResult translateToCannKernel(Operation *op, raw_ostream &os);
} // namespace mlir

#endif // AFIR_TARGET_CANNKERNEL_CANNTRANSLATION_H
```

- [ ] **Step 4: Implement CannTranslation.cpp**

Create `lib/Target/CannKernel/CannTranslation.cpp`:

```cpp
//===- CannTranslation.cpp - CANN kernel C++ translation ------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// Translates MLIR with CANN-signature aicore func.func to C++:
//
//   extern "C" __global__ __aicore__ void kernel(
//       GM_ADDR in0, GM_ADDR in1,   // inputs
//       GM_ADDR out0,                // outputs
//       GM_ADDR workspace,           // workspace (unused)
//       TilingData tiling            // by-value struct
//   ) { <body> }
//
// All body ops are emitted by PyAsc's emitOperation().
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannTranslation.h"
#include "ascir/Dialect/Asc/Utils/Attributes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/CodeEmitter.h"
#include "ascir/Target/Asc/Common.h"     // emitOperation(), needsSemicolon()

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// Emit the TilingData struct declaration from a PyStructType.
// Example output:
//   struct TilingData {
//       int64_t TB_M;
//       int64_t TB_N;
//   };
static LogicalResult emitTilingStructDecl(CodeEmitter &emitter,
                                          emitasc::PyStructType pyType) {
  auto &os = emitter.ostream();
  os << "struct " << pyType.getNameAttr().getValue() << " {\n";
  os.indent();

  auto types = pyType.getTypesAttr().getValue();
  auto names = pyType.getNamesAttr().getValue();
  assert(types.size() == names.size() && "PyStruct types/names mismatch");

  for (size_t i = 0; i < types.size(); ++i) {
    Type fieldType = cast<TypeAttr>(types[i]).getValue();
    StringRef fieldName = cast<StringAttr>(names[i]).getValue();
    // Use CodeEmitter's type emission (i64 -> int64_t, i32 -> int32_t, etc.)
    if (failed(emitter.emitType(UnknownLoc::get(pyType.getContext()), fieldType)))
      return failure();
    os << " " << fieldName << ";\n";
  }

  os.unindent() << "};\n\n";
  return success();
}

// Emit the CANN-standard aicore function.
static LogicalResult printCannFuncOp(CodeEmitter &emitter,
                                     func::FuncOp funcOp) {
  CodeEmitter::Scope scope(emitter);
  auto &os = emitter.ostream();
  auto args = funcOp.getArguments();
  int N = args.size();

  // cann.num_inputs attribute must exist (set by CanonicalizeCannSignaturePass).
  auto numInputsAttr = funcOp->getAttrOfType<IntegerAttr>("cann.num_inputs");
  if (!numInputsAttr)
    return funcOp.emitOpError("missing cann.num_inputs attribute; "
                               "run --canonicalize-cann-signature first");
  int numInputs = numInputsAttr.getInt();

  // Args layout: inputs[0..numInputs-1], outputs[numInputs..N-3],
  //              workspace[N-2], tiling[N-1]
  if (N < numInputs + 2)
    return funcOp.emitOpError("too few arguments for CANN layout");

  auto tilingArg = args[N - 1];
  auto pyType = dyn_cast<emitasc::PyStructType>(tilingArg.getType());
  if (!pyType)
    return funcOp.emitOpError("last argument must be !emitasc.py_struct");

  // 1. Emit TilingData struct declaration.
  if (failed(emitTilingStructDecl(emitter, pyType)))
    return failure();

  // 2. Emit function signature.
  os << "extern \"C\" __global__ __aicore__ void " << funcOp.getName() << "(\n";
  os.indent();

  // inputs
  for (int i = 0; i < numInputs; ++i) {
    os << "GM_ADDR " << emitter.getOrCreateName(args[i]);
    os << ",\n";
  }
  // outputs
  for (int i = numInputs; i < N - 2; ++i) {
    os << "GM_ADDR " << emitter.getOrCreateName(args[i]);
    os << ",\n";
  }
  // workspace
  os << "GM_ADDR " << emitter.getOrCreateName(args[N - 2]) << ",\n";
  // tiling by-value
  os << pyType.getNameAttr().getValue() << " "
     << emitter.getOrCreateName(tilingArg) << "\n";

  os.unindent() << ") {\n";
  os.indent();

  // 3. Emit body ops using PyAsc's emitOperation().
  for (Operation &op : funcOp.getBody().front()) {
    if (failed(emitOperation(emitter, op, needsSemicolon(op))))
      return failure();
  }

  os.unindent() << "}\n";
  return success();
}

} // namespace

LogicalResult mlir::translateToCannKernel(Operation *op, raw_ostream &os) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp)
    return op->emitOpError("expected ModuleOp");

  CodeEmitter emitter(os);
  CodeEmitter::Scope scope(emitter);

  // Emit kernel_operator.h include first.
  os << "#include \"kernel_operator.h\"\n\n";

  for (Operation &child : *moduleOp.getBody()) {
    if (auto funcOp = dyn_cast<func::FuncOp>(child)) {
      if (funcOp->hasAttr(ascendc::attr::global)) {
        if (failed(printCannFuncOp(emitter, funcOp)))
          return failure();
        continue;
      }
    }
    // Non-aicore ops: use PyAsc emitter as-is.
    if (failed(emitOperation(emitter, child, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}
```

- [ ] **Step 5: Create lib/Target/CannKernel/CMakeLists.txt**

```cmake
add_mlir_library(CannKernelTranslation
  CannTranslation.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIREmitAsc
  MLIRAsc
  MLIRTargetAsc
  MLIRIR
  LLVMSupport
)
```

- [ ] **Step 6: Create lib/Target/CMakeLists.txt**

```cmake
add_subdirectory(CannKernel)
```

- [ ] **Step 7: Add lib/Target and tools/afir-translate to root CMakeLists**

In the root `CMakeLists.txt`, the lib and tools subdirs are listed individually (lines ~89–98). Add the two new entries:

After `add_subdirectory(lib/Utils)` (line 93), add:
```cmake
add_subdirectory(lib/Target)
```

After `add_subdirectory(tools/validator)` (line 98), add:
```cmake
add_subdirectory(tools/afir-translate)
```

- [ ] **Step 8: Wire translation into afir-opt**

`afir-opt` currently uses `MlirOptMain`, which doesn't support `TranslateFromMLIRRegistration`. We need to register the translation alongside the opt functionality.

The cleanest approach: add a new standalone `afir-translate` tool (mirrors `ascir-translate`).

Create `tools/afir-translate/afir-translate.cpp`:

```cpp
//===- afir-translate.cpp - AFIR translation driver ---------------*- C++ -*-===//
//
// Supports -mlir-to-cann translation for CANN-standard kernel emission.
//
//===----------------------------------------------------------------------===//

#include "Target/CannKernel/CannTranslation.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllTranslations();

  TranslateFromMLIRRegistration cannReg(
      "mlir-to-cann", "translate MLIR to CANN-standard AscendC kernel",
      [](Operation *op, raw_ostream &os) {
        return translateToCannKernel(op, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, ascendc::AscendCDialect,
                        emitasc::EmitAscDialect, emitc::EmitCDialect,
                        func::FuncDialect, LLVM::LLVMDialect, math::MathDialect,
                        memref::MemRefDialect, scf::SCFDialect>();
        ascendc::registerExternalModels(registry);
        emitasc::registerExternalModels(registry);
      });

  return failed(mlirTranslateMain(argc, argv, "AFIR translation tool"));
}
```

Create `tools/afir-translate/CMakeLists.txt`:

```cmake
set(LLVM_LINK_COMPONENTS Core Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(afir-translate
  afir-translate.cpp
)

target_link_libraries(afir-translate
  PRIVATE
    ${dialect_libs}
    MLIRTranslateLib
    MLIRParser
    MLIRSupport
    MLIRIR
    MLIRAsc
    MLIREmitAsc
    MLIRTargetAsc
    CannKernelTranslation
)

mlir_check_all_link_libraries(afir-translate)
```

The `tools/afir-translate` subdirectory is wired via the root `CMakeLists.txt` in Step 7 above — no separate `tools/CMakeLists.txt` exists.

- [ ] **Step 9: Build**

```bash
orb -m xvm bash -c "cd /home/niu/code/Ascend-MLIR && \
  ./scripts/build.sh --build-project 2>&1 | tail -20"
```

Expected: `afir-translate` binary built.

- [ ] **Step 10: Run the lit test**

```bash
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  afir-translate -mlir-to-cann \
  /home/niu/code/Ascend-MLIR/test/Target/cann-translate.mlir 2>&1"
```

Pipe through FileCheck:
```bash
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  afir-translate -mlir-to-cann \
  /home/niu/code/Ascend-MLIR/test/Target/cann-translate.mlir | \
  /home/niu/code/llvm-project/build/bin/FileCheck \
  /home/niu/code/Ascend-MLIR/test/Target/cann-translate.mlir"
```

Expected: FileCheck passes.

- [ ] **Step 11: Add afir-translate to test suite depends**

In `test/CMakeLists.txt`, add `afir-translate` to `AFIR_TEST_DEPENDS` so it is built before `check-afir` runs:

```cmake
set(AFIR_TEST_DEPENDS
  afir-opt
  afir-translate
  FileCheck
  count
  not
)
```

- [ ] **Step 12: Commit**

```bash
git add include/Target/ lib/Target/ tools/afir-translate/ \
        test/Target/cann-translate.mlir test/CMakeLists.txt
git commit -m "feat(cann): implement cann-translate (mlir-to-cann translation)"
```

---

## Task 4: Wire up broadcast-add-reduce example end-to-end

**Files:**
- Modify: `examples/broadcast-add-reduce/run.sh`

- [ ] **Step 1: Run existing pipeline through step7**

```bash
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  bash /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/run.sh 2>&1 | tail -10"
```

Expected: stages 0–7 complete, `step7_kernel.mlir` produced.

- [ ] **Step 2: Run pass manually to verify step7 → step7_cann**

```bash
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  afir-opt --canonicalize-cann-signature \
  /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/step7_kernel.mlir \
  -o /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/step7_cann.mlir && \
  grep -E 'func.func|cann.num_inputs|py_struct|memref<ui8>' \
  /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/step7_cann.mlir | head -10"
```

Expected: signature shows `memref<ui8>` and `!emitasc.py_struct<...>` as last two args, `cann.num_inputs = 2`.

- [ ] **Step 3: Run translator manually to verify step7_cann → step8_kernel.cpp**

```bash
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  afir-translate -mlir-to-cann \
  /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/step7_cann.mlir \
  -o /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/step8_cann_kernel.cpp && \
  head -35 /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/step8_cann_kernel.cpp"
```

Expected output starts with:
```cpp
#include "kernel_operator.h"

struct TilingData {
    int64_t TB_M;
    ...
};

extern "C" __global__ __aicore__ void broadcast_add_reducesum(
    GM_ADDR v1,
    GM_ADDR v2,
    GM_ADDR v3,
    GM_ADDR v4,
    TilingData v5
) {
```

- [ ] **Step 4: Run compiler CLI on generated kernel**

```bash
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  mkdir -p /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/build && \
  compiler \
    --kernel /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/step8_cann_kernel.cpp \
    --output /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/build \
    --num-inputs 2 --num-outputs 1 \
    --tiling-layout 'int64,int64,int64,int64' \
    --kernel-type vec 2>&1"
```

Expected: `Compiled: .../broadcast_add_reducesum.bin` and `Runner: .../runner`.

- [ ] **Step 5: Update tiling_space.json to point at new kernel file**

In `examples/broadcast-add-reduce/tiling_space.json`, change `kernel_file`:

```json
"kernel_file": "step8_kernel.cpp",
```

(was `step8_kernel-adjust.cpp`)

- [ ] **Step 6: Run autotuner to validate end-to-end**

```bash
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  cd /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce && \
  autotuner \
    --space tiling_space.json \
    --inputs input_a.npy,input_b.npy \
    --expected output_c.npy \
    --output build \
    --shape M=32,N=32 2>&1"
```

Expected: at least one config passes accuracy check; best config printed.

- [ ] **Step 7: Update run.sh**

In `examples/broadcast-add-reduce/run.sh`, replace the Stage 8 block with:

```bash
# ── STAGE 7b: Canonicalize CANN signature ──────────────────────────────────
echo ""
echo "==================== [STAGE 7b] CANN Signature：--canonicalize-cann-signature ===================="
$AFIR_OPT --canonicalize-cann-signature \
  "$DIR/step7_kernel.mlir" \
  -o "$DIR/step7_cann.mlir" 2>&1
log "  ✓ CANN 签名规范化成功，输出: step7_cann.mlir"
log "$(grep -E 'func.func|cann.num_inputs|py_struct|memref<ui8>' "$DIR/step7_cann.mlir" | head -6)"

# ── STAGE 8: AscendC C++ Code Generation (CANN standard) ───────────────────
echo ""
echo "==================== [STAGE 8] Codegen：afir-translate -mlir-to-cann ===================="
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/step7_cann.mlir" -o "$DIR/step8_kernel.cpp" 2>&1
log "  ✓ Codegen 成功，输出: step8_kernel.cpp"
log "$(head -20 "$DIR/step8_kernel.cpp")"
```

- [ ] **Step 8: Run the updated run.sh end-to-end**

```bash
orb -m xvm bash -c "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  bash /home/niu/code/Ascend-MLIR/examples/broadcast-add-reduce/run.sh --log 2>&1 | tail -20"
```

Expected: all stages 0–8 pass, `step8_kernel.cpp` has CANN-standard signature.

- [ ] **Step 9: Commit**

```bash
git add examples/broadcast-add-reduce/run.sh \
        examples/broadcast-add-reduce/tiling_space.json \
        examples/broadcast-add-reduce/step7_cann.mlir \
        examples/broadcast-add-reduce/step8_kernel.cpp
git commit -m "feat(example): wire broadcast-add-reduce to CANN toolchain via canonicalize-cann-signature + afir-translate"
```

---

## Notes for Implementer

- **`afir-translate` vs `afir-opt`**: `afir-opt` uses `MlirOptMain` which doesn't support `TranslateFromMLIRRegistration` (that requires `MlirTranslateMain`). Hence a new `afir-translate` binary mirrors the pattern of the existing `ascir-translate`.
- **`lib/CMakeLists.txt`**: Check whether it already has `add_subdirectory(Target)`. If not, add it. If `lib/Target/` already exists for a different reason, integrate `CannKernel` into the existing `lib/Target/CMakeLists.txt`.
- **`tiling_space.json`** for broadcast-add-reduce already has `"kernel_file": "step8_kernel-adjust.cpp"`. Update to `"step8_kernel.cpp"` after the pipeline generates the CANN-compliant version, or keep both files and point the autotuner at the new one during testing.
- **`needsSemicolon`** is a template in `ascir/Target/Asc/Common.h:65` — include `Common.h` directly.
- **Memory space 22** = `ascendc::AddressSpace::gm` (value 22 in the `AddressSpace` enum).
