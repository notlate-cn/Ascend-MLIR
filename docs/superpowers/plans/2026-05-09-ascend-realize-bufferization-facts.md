# Ascend Realize Bufferization Facts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first concrete `BufferizationDriver` slice for Ascend Realize by collecting per-kernel tensor facts and reporting input/output/temporary value counts.

**Architecture:** Keep `--ascend-realize` read-only for this slice. The new driver scans scheduled structured ops, classifies tensor values by kernel, and fills `BufferizedKernelIR` without mutating IR or claiming one-shot bufferization is complete. Later one-shot-bufferize integration will replace the internal fact source while preserving this driver/report interface.

**Tech Stack:** MLIR C++ pass infrastructure, Linalg DPS interfaces, LLVM ADT containers, LIT/FileCheck, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Create `include/Conversion/Ascend/Realize/BufferizationDriver.h`
  - Declares the read-only `BufferizationDriver` interface.
- Create `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp`
  - Implements per-kernel tensor fact collection.
- Modify `include/Conversion/Ascend/Realize/RealizeTypes.h`
  - Extends `BufferizedKernelIR` with role counts.
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
  - Calls `BufferizationDriver` while building `RealizePlanBundle`.
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
  - Prints the new fact counters.
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
  - Adds the new implementation file.
- Modify `test/Conversion/ascend-realize-mvp.mlir`
  - Updates the existing Realize smoke test to expect non-zero facts.
- Add `test/Conversion/ascend-realize-bufferization-facts.mlir`
  - Covers a two-op same-kernel pipeline with one internal temporary value.
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
  - Records the new Phase 3 subtask and verification result.

## Task 1: Red Tests For Bufferization Facts

**Files:**
- Modify: `test/Conversion/ascend-realize-mvp.mlir`
- Add: `test/Conversion/ascend-realize-bufferization-facts.mlir`

- [ ] **Step 1: Update the existing Realize smoke expectation**

Change the `BufferizedKernelIR` checks in `test/Conversion/ascend-realize-mvp.mlir` from the old placeholder output:

```mlir
// CHECK-NEXT:   mode = "gm_only"
// CHECK-NEXT:   buffer_values = 0
```

to the new fact output:

```mlir
// CHECK-NEXT:   mode = "tensor_facts"
// CHECK-NEXT:   buffer_values = 3
// CHECK-NEXT:   input_values = 2
// CHECK-NEXT:   output_values = 1
// CHECK-NEXT:   temporary_values = 0
```

Reasoning: the single `linalg.generic` has two DPS tensor inputs and one tensor result. Its `tensor.empty` init is not counted as an input value.

- [ ] **Step 2: Add a focused two-op fact test**

Create `test/Conversion/ascend-realize-bufferization-facts.mlir` with this content:

```mlir
// RUN: afir-opt %s --ascend-realize='dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @scheduled_two_op_kernel(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> attributes {ascend.normalized = true} {
  %empty0 = tensor.empty() : tensor<64xf16>
  %mid = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty0 : tensor<64xf16>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  %empty1 = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%mid : tensor<64xf16>)
    outs(%empty1 : tensor<64xf16>)
    attrs = {
      ascend.kernel = "kernel_0",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %o: f16):
    %v = arith.negf %x : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>

  return %out : tensor<64xf16>
}

// CHECK: Realize report
// CHECK-NEXT:   kernels = 1
// CHECK: BufferizedKernelIR:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "tensor_facts"
// CHECK-NEXT:   buffer_values = 4
// CHECK-NEXT:   input_values = 2
// CHECK-NEXT:   output_values = 1
// CHECK-NEXT:   temporary_values = 1
```

Reasoning: `%arg0` and `%arg1` are external inputs, `%out` escapes the kernel, and `%mid` is produced and consumed inside the same kernel.

- [ ] **Step 3: Run RED verification in xvm/docker**

Run from the host:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && llvm-lit -v test/Conversion/ascend-realize-mvp.mlir test/Conversion/ascend-realize-bufferization-facts.mlir'
```

Expected: FAIL. The current implementation still prints `mode = "gm_only"` and `buffer_values = 0`, and the new focused test cannot pass until the driver exists.

## Task 2: BufferizationDriver Data Model And Implementation

**Files:**
- Add: `include/Conversion/Ascend/Realize/BufferizationDriver.h`
- Add: `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp`
- Modify: `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Extend `BufferizedKernelIR`**

Add these fields to `BufferizedKernelIR` in `include/Conversion/Ascend/Realize/RealizeTypes.h`:

```cpp
  unsigned inputValueCount = 0;
  unsigned outputValueCount = 0;
  unsigned temporaryValueCount = 0;
```

Keep the default mode as `"gm_only"` in the type. The driver sets `"tensor_facts"` only when facts are collected.

- [ ] **Step 2: Declare the driver interface**

Create `include/Conversion/Ascend/Realize/BufferizationDriver.h`:

```cpp
//===- BufferizationDriver.h - Ascend realize buffer facts -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_BUFFERIZATIONDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_BUFFERIZATIONDRIVER_H

#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::ascend::realize {

class BufferizationDriver {
public:
  FailureOr<SmallVector<BufferizedKernelIR, 4>>
  collectTensorFacts(ModuleOp module) const;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_BUFFERIZATIONDRIVER_H
```

- [ ] **Step 3: Implement fact collection**

Create `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp` with this behavior:

```cpp
//===- BufferizationDriver.cpp - Ascend realize buffer facts -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/BufferizationDriver.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir::afir::ascend::realize {
namespace {

struct KernelTensorFacts {
  llvm::SetVector<Value> inputValues;
  llvm::SetVector<Value> outputValues;
  llvm::SetVector<Value> temporaryValues;
};

static bool isTensorValue(Value value) {
  return llvm::isa<TensorType>(value.getType());
}

static StringRef getKernelId(Operation *op) {
  auto kernelAttr = op->getAttrOfType<StringAttr>(kKernelAttr);
  return kernelAttr ? kernelAttr.getValue() : StringRef();
}

static bool isProducedByKernel(Value value, StringRef kernelId) {
  Operation *def = value.getDefiningOp();
  return def && getKernelId(def) == kernelId;
}

static bool hasUseOutsideKernel(Value value, StringRef kernelId) {
  for (Operation *user : value.getUsers()) {
    if (getKernelId(user) != kernelId)
      return true;
  }
  return value.use_empty();
}

static void collectLinalgFacts(linalg::LinalgOp linalgOp, StringRef kernelId,
                               KernelTensorFacts &facts) {
  for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
    Value input = operand->get();
    if (!isTensorValue(input))
      continue;
    if (isProducedByKernel(input, kernelId))
      facts.temporaryValues.insert(input);
    else
      facts.inputValues.insert(input);
  }

  for (Value result : linalgOp->getResults()) {
    if (!isTensorValue(result))
      continue;
    if (hasUseOutsideKernel(result, kernelId))
      facts.outputValues.insert(result);
    else
      facts.temporaryValues.insert(result);
  }
}

static BufferizedKernelIR buildIR(StringRef kernelId,
                                  const KernelTensorFacts &facts) {
  BufferizedKernelIR ir;
  ir.kernelId = kernelId.str();
  ir.mode = "tensor_facts";
  ir.inputValueCount = facts.inputValues.size();
  ir.outputValueCount = facts.outputValues.size();
  ir.temporaryValueCount = facts.temporaryValues.size();
  ir.bufferValueCount =
      ir.inputValueCount + ir.outputValueCount + ir.temporaryValueCount;
  return ir;
}

} // namespace

FailureOr<SmallVector<BufferizedKernelIR, 4>>
BufferizationDriver::collectTensorFacts(ModuleOp module) const {
  llvm::StringMap<KernelTensorFacts> factsByKernel;

  module.walk([&](Operation *op) {
    StringRef kernelId = getKernelId(op);
    if (kernelId.empty())
      return;

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
      collectLinalgFacts(linalgOp, kernelId, factsByKernel[kernelId]);
  });

  SmallVector<StringRef, 4> kernelIds;
  for (const auto &entry : factsByKernel)
    kernelIds.push_back(entry.getKey());
  llvm::sort(kernelIds);

  SmallVector<BufferizedKernelIR, 4> result;
  for (StringRef kernelId : kernelIds)
    result.push_back(buildIR(kernelId, factsByKernel[kernelId]));
  return result;
}

} // namespace mlir::afir::ascend::realize
```

If the local LLVM/MLIR version needs a different include for `TensorType`, use the existing project pattern for `mlir/IR/BuiltinTypes.h`.

- [ ] **Step 4: Add the source to CMake**

Add `Realize/BufferizationDriver.cpp` before `Realize/RealizePass.cpp` in `lib/Conversion/Ascend/CMakeLists.txt`:

```cmake
  Realize/BufferizationDriver.cpp
  Realize/RealizePass.cpp
```

## Task 3: Wire Driver Into Realize And Report

**Files:**
- Modify: `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Modify: `lib/Conversion/Ascend/Realize/RealizeReport.cpp`

- [ ] **Step 1: Include the driver**

In `RealizePass.cpp`, add:

```cpp
#include "Conversion/Ascend/Realize/BufferizationDriver.h"
```

- [ ] **Step 2: Join collected facts by kernel**

After the existing attribute-consistency walk succeeds and before constructing bundles, call:

```cpp
  BufferizationDriver bufferizationDriver;
  FailureOr<SmallVector<BufferizedKernelIR, 4>> bufferized =
      bufferizationDriver.collectTensorFacts(module);
  if (failed(bufferized))
    return failure();

  llvm::StringMap<BufferizedKernelIR> bufferizedByKernel;
  for (BufferizedKernelIR &ir : *bufferized)
    bufferizedByKernel[ir.kernelId] = std::move(ir);
```

Then, when building each `RealizePlanBundle`, replace the current empty `bundle.bufferizedIR.kernelId = ...` assignment with:

```cpp
    auto bufferizedIt = bufferizedByKernel.find(bundle.kernel.kernelId);
    if (bufferizedIt != bufferizedByKernel.end())
      bundle.bufferizedIR = std::move(bufferizedIt->second);
    else
      bundle.bufferizedIR.kernelId = bundle.kernel.kernelId;
```

Keep the old fallback so a future non-linalg scheduled op does not crash the read-only Realize pass.

- [ ] **Step 3: Print role counts**

In `RealizeReport.cpp`, print these lines immediately after `buffer_values`:

```cpp
    os << "  input_values = " << bundle.bufferizedIR.inputValueCount << "\n";
    os << "  output_values = " << bundle.bufferizedIR.outputValueCount << "\n";
    os << "  temporary_values = " << bundle.bufferizedIR.temporaryValueCount
       << "\n";
```

- [ ] **Step 4: Run focused GREEN verification in xvm/docker**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt && llvm-lit -v test/Conversion/ascend-realize-mvp.mlir test/Conversion/ascend-realize-bufferization-facts.mlir'
```

Expected: PASS for both LIT tests.

## Task 4: Tracking, Guard, And Broader Verification

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Update the Phase 3 board**

Change the Phase 3 `BufferizationDriver 对接` row from `Planned` to `In Progress` or `Done` only after verification succeeds. The evidence text must include the exact xvm commands and pass counts.

- [ ] **Step 2: Run code naming guard on host**

Run:

```bash
test/tools/check_ascend_no_v2_code_naming.sh
```

Expected: PASS with no source-level `V2/v2` naming regressions.

- [ ] **Step 3: Run focused Ascend Conversion suite in xvm/docker**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt AscendCommonAttributesTest AscendKernelPatternTest && ./build/unittests/Conversion/AscendCommonAttributesTest && ./build/unittests/Conversion/AscendKernelPatternTest && llvm-lit -v test/Conversion/ascend-*.mlir'
```

Expected: all focused unit tests and `test/Conversion/ascend-*.mlir` pass.

- [ ] **Step 4: Commit**

Commit only after all verification commands have passed:

```bash
git add include/Conversion/Ascend/Realize/BufferizationDriver.h \
        include/Conversion/Ascend/Realize/RealizeTypes.h \
        lib/Conversion/Ascend/Realize/BufferizationDriver.cpp \
        lib/Conversion/Ascend/Realize/RealizePass.cpp \
        lib/Conversion/Ascend/Realize/RealizeReport.cpp \
        lib/Conversion/Ascend/CMakeLists.txt \
        test/Conversion/ascend-realize-mvp.mlir \
        test/Conversion/ascend-realize-bufferization-facts.mlir \
        docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md \
        docs/superpowers/plans/2026-05-09-ascend-realize-bufferization-facts.md
git commit -m "feat: collect Ascend realize bufferization facts"
```

## Self-Review

- Spec coverage: This plan implements only the first `BufferizationDriver` slice from V2-5: fact collection and report wiring. It intentionally does not run upstream one-shot-bufferize, choose memory places, allocate static workspace, insert movements, or materialize memory.
- Placeholder scan: No `TBD`, `TODO`, or vague implementation-only instructions remain.
- Type consistency: `BufferizedKernelIR` owns the new counters; `BufferizationDriver` fills them; `RealizePass` copies them into bundles; `RealizeReport` prints them.
- Testing: The RED/GREEN path is explicit and xvm/docker commands are included.
