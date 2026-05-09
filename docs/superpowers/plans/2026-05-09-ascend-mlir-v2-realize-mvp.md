# Ascend MLIR V2 Realize MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first fourth-layer Realize implementation slice: a stable `--ascend-realize` pass, Realize plan object skeletons, and MVP debug reports over the scheduled IR.

**Architecture:** This MVP is intentionally non-materializing. It consumes the current Schedule/StructuredLowering attributes already written on `linalg` ops, reconstructs a minimal read-only Realize view, emits empty/GM-only plan reports, and preserves IR unchanged. Real bufferization, placement selection, workspace packing, and `memref.copy` insertion remain follow-up tasks.

**Tech Stack:** C++17, MLIR ModulePass/TableGen, LLVM ADT, `func`/`linalg`/`memref` dialect deps, lit/FileCheck, xvm/docker verification.

---

## Scope

Implement now:

- `--ascend-realize` pass registration.
- `debug-stage=realize` support.
- MVP plan objects:
  - `BufferizedKernelIR`
  - `PlacementPlan`
  - `StaticMemoryPlan`
  - `MovementPlan`
  - `MemoryRealizationPlan`
- Realize report printer.
- Schedule/StructuredLowering attr validation from existing op attrs.
- Focused lit coverage and tracking-board update.

Do not implement now:

- `one-shot-bufferize`
- real `memref.alloc` / `memref.copy` insertion
- target-aware placement
- static memory packing
- movement path selection
- Translate-layer consumption

## File Map

- Create `include/Conversion/AscendV2/Realize/RealizePass.h`
  - Declares `createAscendRealizePass()`.
- Create `include/Conversion/AscendV2/Realize/RealizeTypes.h`
  - Defines MVP Realize data objects and attr constants.
- Create `include/Conversion/AscendV2/Realize/RealizeReport.h`
  - Declares report printers.
- Create `lib/Conversion/AscendV2/Realize/RealizePass.cpp`
  - Implements pass, validation, and MVP object construction.
- Create `lib/Conversion/AscendV2/Realize/RealizeReport.cpp`
  - Implements stable debug report formatting.
- Modify `include/Conversion/Passes.td`
  - Adds `AscendRealizePass`.
- Modify `include/Conversion/Passes.h`
  - Includes `RealizePass.h`.
- Modify `include/Conversion/AscendV2/Debug/DebugOptions.h`
  - Adds `DebugStage::Realize`.
- Modify `lib/Conversion/AscendV2/Debug/DebugOptions.cpp`
  - Parses and prints `realize`.
- Modify `lib/Conversion/AscendV2/CMakeLists.txt`
  - Builds Realize sources and links `MLIRMemRefDialect`.
- Create `test/Conversion/ascend-realize-mvp.mlir`
  - Covers full Normalize -> Kernelize -> Schedule -> Realize pipeline and report.
- Create `test/Conversion/ascend-realize-requires-schedule.mlir`
  - Covers failure when Realize runs before Schedule.
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
  - Adds Phase 3 Task 0/1 tracking entries.

---

## Task 0: Save Plan

**Files:**

- Create: `docs/superpowers/plans/2026-05-09-ascend-mlir-v2-realize-mvp.md`

- [ ] **Step 1: Run static check**

Run:

```bash
git diff --check
```

Expected: no output, exit code 0.

- [ ] **Step 2: Commit plan**

Run:

```bash
git add docs/superpowers/plans/2026-05-09-ascend-mlir-v2-realize-mvp.md
git commit -m "docs: plan Ascend V2 realize MVP"
```

Expected: one docs commit.

---

## Task 1: RED Lit For Realize MVP

**Files:**

- Create: `test/Conversion/ascend-realize-mvp.mlir`

- [ ] **Step 1: Write failing lit test**

Create `test/Conversion/ascend-realize-mvp.mlir`:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @elementwise(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
  %empty = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}

// CHECK: Ascend V2 realize report (ascend-realize)
// CHECK: Realize report
// CHECK-NEXT:   kernels = 1
// CHECK: BufferizedKernelIR:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "gm_only"
// CHECK-NEXT:   buffer_values = 0
// CHECK: PlacementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   selected_places = 0
// CHECK: StaticMemoryPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   workspace_slots = 0
// CHECK: MovementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   movements = 0
// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   frozen = true
// CHECK: linalg.generic
// CHECK-SAME: ascend.v2.schedule.decision_id = "kernel_0.decision.0"
// CHECK-SAME: ascend.v2.schedule.structured_lowering = "loop_skeleton_v0"
```

- [ ] **Step 2: Run RED**

Run on xvm:

```bash
rsync -av --relative test/Conversion/ascend-realize-mvp.mlir xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir'
```

Expected: FAIL because `--ascend-realize` is not registered.

---

## Task 2: Realize Pass Skeleton And Debug Stage

**Files:**

- Create: `include/Conversion/AscendV2/Realize/RealizePass.h`
- Create: `lib/Conversion/AscendV2/Realize/RealizePass.cpp`
- Modify: `include/Conversion/Passes.td`
- Modify: `include/Conversion/Passes.h`
- Modify: `include/Conversion/AscendV2/Debug/DebugOptions.h`
- Modify: `lib/Conversion/AscendV2/Debug/DebugOptions.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`

- [ ] **Step 1: Add pass declaration header**

Create `include/Conversion/AscendV2/Realize/RealizePass.h`:

```cpp
//===- RealizePass.h - Ascend V2 realize pass ------------------*- C++ -*-===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_PASS_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::afir {

std::unique_ptr<Pass> createAscendRealizePass();

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_PASS_H
```

- [ ] **Step 2: Add TableGen pass**

In `include/Conversion/Passes.td`, add after `AscendSchedulePass`:

```td
def AscendRealizePass : Pass<"ascend-realize", "mlir::ModuleOp"> {
  let summary = "Construct Ascend MLIR V2 memory realization plans";
  let description = [{
    MVP pass for the V2 Realize stage. It preserves IR unchanged, validates
    that Schedule/StructuredLowering attributes are present, and emits an
    optional debug report with empty GM-only plan objects.
  }];
  let constructor = "mlir::afir::createAscendRealizePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"debugStage", "debug-stage", "std::string", /*default=*/"\"none\"",
           "V2 debug stage: none, normalize, kernelize, schedule, realize, all">,
    Option<"dumpReport", "dump-report", "bool", /*default=*/"false",
           "Dump V2 analysis report to stderr">,
  ];
}
```

- [ ] **Step 3: Include pass header**

In `include/Conversion/Passes.h`, add:

```cpp
#include "Conversion/AscendV2/Realize/RealizePass.h"
```

- [ ] **Step 4: Extend debug stage**

In `include/Conversion/AscendV2/Debug/DebugOptions.h`, change enum to:

```cpp
enum class DebugStage { None, Normalize, Kernelize, Schedule, Realize, All };
```

In `lib/Conversion/AscendV2/Debug/DebugOptions.cpp`, add parsing:

```cpp
if (trimmed.equals_insensitive("realize"))
  return DebugStage::Realize;
```

And add emit switch case:

```cpp
case DebugStage::Realize:
  stageName = "realize";
  break;
```

- [ ] **Step 5: Add minimal pass implementation**

Create `lib/Conversion/AscendV2/Realize/RealizePass.cpp`:

```cpp
//===- RealizePass.cpp - Ascend V2 realize pass --------------------------===//

#include "Conversion/AscendV2/Realize/RealizePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DECL_ASCENDREALIZEPASS
#define GEN_PASS_DEF_ASCENDREALIZEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

struct AscendRealizePass
    : public ::impl::AscendRealizePassBase<AscendRealizePass> {
  using AscendRealizePassBase::AscendRealizePassBase;

  void runOnOperation() override {
    ::mlir::ascend::v2::DebugOptions options{
        ::mlir::ascend::v2::parseDebugStage(debugStage), dumpReport};
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Realize))
      ::mlir::ascend::v2::emitStageHeader(
          llvm::errs(), ::mlir::ascend::v2::DebugStage::Realize,
          getArgument());
  }
};

std::unique_ptr<Pass> createAscendRealizePass() {
  return std::make_unique<AscendRealizePass>();
}

} // namespace mlir::afir
```

- [ ] **Step 6: Add CMake sources and deps**

In `lib/Conversion/AscendV2/CMakeLists.txt`, add sources:

```cmake
  Realize/RealizePass.cpp
```

Add link dependency:

```cmake
  MLIRMemRefDialect
```

- [ ] **Step 7: Build and run RED-to-partial-GREEN**

Run on xvm:

```bash
rsync -av --relative include/Conversion/Passes.td include/Conversion/Passes.h include/Conversion/AscendV2/Debug/DebugOptions.h lib/Conversion/AscendV2/Debug/DebugOptions.cpp include/Conversion/AscendV2/Realize/RealizePass.h lib/Conversion/AscendV2/Realize/RealizePass.cpp lib/Conversion/AscendV2/CMakeLists.txt xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir'
```

Expected: pass is recognized and stage header prints; test still fails until report objects are added.

---

## Task 3: Realize Plan Objects And Report Printer

**Files:**

- Create: `include/Conversion/AscendV2/Realize/RealizeTypes.h`
- Create: `include/Conversion/AscendV2/Realize/RealizeReport.h`
- Create: `lib/Conversion/AscendV2/Realize/RealizeReport.cpp`
- Modify: `lib/Conversion/AscendV2/Realize/RealizePass.cpp`
- Modify: `lib/Conversion/AscendV2/CMakeLists.txt`

- [ ] **Step 1: Add MVP types**

Create `include/Conversion/AscendV2/Realize/RealizeTypes.h`:

```cpp
//===- RealizeTypes.h - Ascend V2 realize data model -----------*- C++ -*-===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZETYPES_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZETYPES_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::afir::ascend::v2::realize {

inline constexpr llvm::StringLiteral kScheduleDecisionIdAttr =
    "ascend.v2.schedule.decision_id";
inline constexpr llvm::StringLiteral kStructuredLoweringAttr =
    "ascend.v2.schedule.structured_lowering";
inline constexpr llvm::StringLiteral kKernelAttr = "ascend.v2.kernel";

enum class MemoryPlace { GM, VECIN, VECCALC, VECOUT, A1, B1, A2, B2, CO1 };

struct RealizeKernelView {
  std::string kernelId;
  std::string decisionId;
  std::string structuredLowering;
  unsigned scheduledOps = 0;
};

struct BufferizedKernelIR {
  std::string kernelId;
  std::string mode = "gm_only";
  unsigned bufferValueCount = 0;
};

struct PlacementPlan {
  std::string kernelId;
  unsigned selectedPlaceCount = 0;
};

struct StaticMemoryPlan {
  std::string kernelId;
  unsigned workspaceSlotCount = 0;
};

struct MovementPlan {
  std::string kernelId;
  unsigned movementCount = 0;
};

struct MemoryRealizationPlan {
  std::string kernelId;
  bool frozen = false;
  unsigned materializedAllocCount = 0;
  unsigned materializedCopyCount = 0;
};

struct RealizePlanBundle {
  RealizeKernelView kernel;
  BufferizedKernelIR bufferizedIR;
  PlacementPlan placement;
  StaticMemoryPlan staticMemory;
  MovementPlan movement;
  MemoryRealizationPlan realization;
};

} // namespace mlir::afir::ascend::v2::realize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZETYPES_H
```

- [ ] **Step 2: Add report API**

Create `include/Conversion/AscendV2/Realize/RealizeReport.h`:

```cpp
//===- RealizeReport.h - Ascend V2 realize reports ------------*- C++ -*-===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZEREPORT_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZEREPORT_H

#include "Conversion/AscendV2/Realize/RealizeTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::v2::realize {

void printRealizeReport(llvm::ArrayRef<RealizePlanBundle> bundles,
                        llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::v2::realize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZEREPORT_H
```

- [ ] **Step 3: Add stable report implementation**

Create `lib/Conversion/AscendV2/Realize/RealizeReport.cpp`:

```cpp
//===- RealizeReport.cpp - Ascend V2 realize reports ---------------------===//

#include "Conversion/AscendV2/Realize/RealizeReport.h"

namespace mlir::afir::ascend::v2::realize {

void printRealizeReport(llvm::ArrayRef<RealizePlanBundle> bundles,
                        llvm::raw_ostream &os) {
  os << "Realize report\n";
  os << "  kernels = " << bundles.size() << "\n";
  for (const RealizePlanBundle &bundle : bundles) {
    os << "BufferizedKernelIR:\n";
    os << "  kernel = " << bundle.bufferizedIR.kernelId << "\n";
    os << "  mode = \"" << bundle.bufferizedIR.mode << "\"\n";
    os << "  buffer_values = " << bundle.bufferizedIR.bufferValueCount << "\n";
    os << "PlacementPlan:\n";
    os << "  kernel = " << bundle.placement.kernelId << "\n";
    os << "  selected_places = " << bundle.placement.selectedPlaceCount << "\n";
    os << "StaticMemoryPlan:\n";
    os << "  kernel = " << bundle.staticMemory.kernelId << "\n";
    os << "  workspace_slots = " << bundle.staticMemory.workspaceSlotCount << "\n";
    os << "MovementPlan:\n";
    os << "  kernel = " << bundle.movement.kernelId << "\n";
    os << "  movements = " << bundle.movement.movementCount << "\n";
    os << "MemoryRealizationPlan:\n";
    os << "  kernel = " << bundle.realization.kernelId << "\n";
    os << "  frozen = " << (bundle.realization.frozen ? "true" : "false") << "\n";
  }
}

} // namespace mlir::afir::ascend::v2::realize
```

- [ ] **Step 4: Wire report into CMake**

In `lib/Conversion/AscendV2/CMakeLists.txt`, add:

```cmake
  Realize/RealizeReport.cpp
```

- [ ] **Step 5: Run focused lit**

Run on xvm:

```bash
rsync -av --relative include/Conversion/AscendV2/Realize/RealizeTypes.h include/Conversion/AscendV2/Realize/RealizeReport.h lib/Conversion/AscendV2/Realize/RealizeReport.cpp lib/Conversion/AscendV2/CMakeLists.txt xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir'
```

Expected: still fails until pass constructs bundles.

---

## Task 4: Schedule Attr Validation And MVP Bundle Construction

**Files:**

- Modify: `lib/Conversion/AscendV2/Realize/RealizePass.cpp`
- Create: `test/Conversion/ascend-realize-requires-schedule.mlir`

- [ ] **Step 1: Add failure lit**

Create `test/Conversion/ascend-realize-requires-schedule.mlir`:

```mlir
// RUN: not afir-opt %s --ascend-normalize --ascend-kernelize --ascend-realize 2>&1 | FileCheck %s

func.func @elementwise(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
  %empty = tensor.empty() : tensor<64xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty : tensor<64xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return %out : tensor<64xf16>
}

// CHECK: error: ascend-realize requires scheduled structured lowering attributes
```

- [ ] **Step 2: Run RED**

Run on xvm:

```bash
rsync -av --relative test/Conversion/ascend-realize-requires-schedule.mlir xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-requires-schedule.mlir'
```

Expected: FAIL until validation exists.

- [ ] **Step 3: Implement validation and bundles**

In `lib/Conversion/AscendV2/Realize/RealizePass.cpp`, add helpers:

```cpp
static FailureOr<SmallVector<RealizePlanBundle>>
buildMVPRealizePlans(ModuleOp module) {
  DenseMap<StringRef, unsigned> scheduledOpsByKernel;
  DenseMap<StringRef, std::string> decisionByKernel;
  DenseMap<StringRef, std::string> skeletonByKernel;

  module.walk([&](Operation *op) {
    auto kernelAttr = op->getAttrOfType<StringAttr>(kKernelAttr);
    auto decisionAttr = op->getAttrOfType<StringAttr>(kScheduleDecisionIdAttr);
    auto skeletonAttr = op->getAttrOfType<StringAttr>(kStructuredLoweringAttr);
    if (!kernelAttr && !decisionAttr && !skeletonAttr)
      return;
    if (!kernelAttr || !decisionAttr || !skeletonAttr)
      return;
    StringRef kernel = kernelAttr.getValue();
    ++scheduledOpsByKernel[kernel];
    decisionByKernel[kernel] = decisionAttr.getValue().str();
    skeletonByKernel[kernel] = skeletonAttr.getValue().str();
  });

  if (scheduledOpsByKernel.empty())
    return failure();

  SmallVector<StringRef> kernels;
  for (const auto &entry : scheduledOpsByKernel)
    kernels.push_back(entry.first);
  llvm::sort(kernels);

  SmallVector<RealizePlanBundle> bundles;
  for (StringRef kernel : kernels) {
    RealizePlanBundle bundle;
    bundle.kernel.kernelId = kernel.str();
    bundle.kernel.decisionId = decisionByKernel[kernel];
    bundle.kernel.structuredLowering = skeletonByKernel[kernel];
    bundle.kernel.scheduledOps = scheduledOpsByKernel[kernel];
    bundle.bufferizedIR.kernelId = bundle.kernel.kernelId;
    bundle.placement.kernelId = bundle.kernel.kernelId;
    bundle.staticMemory.kernelId = bundle.kernel.kernelId;
    bundle.movement.kernelId = bundle.kernel.kernelId;
    bundle.realization.kernelId = bundle.kernel.kernelId;
    bundle.realization.frozen = true;
    bundles.push_back(std::move(bundle));
  }
  return bundles;
}
```

Then in `runOnOperation()`:

```cpp
FailureOr<SmallVector<RealizePlanBundle>> bundles =
    buildMVPRealizePlans(getOperation());
if (failed(bundles)) {
  getOperation()->emitError()
      << "ascend-realize requires scheduled structured lowering attributes";
  signalPassFailure();
  return;
}
if (::mlir::ascend::v2::shouldDump(
        options, ::mlir::ascend::v2::DebugStage::Realize))
  printRealizeReport(*bundles, llvm::errs());
```

Include:

```cpp
#include "Conversion/AscendV2/Realize/RealizeReport.h"
#include "Conversion/AscendV2/Realize/RealizeTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
```

- [ ] **Step 4: Run GREEN**

Run on xvm:

```bash
rsync -av --relative lib/Conversion/AscendV2/Realize/RealizePass.cpp test/Conversion/ascend-realize-mvp.mlir test/Conversion/ascend-realize-requires-schedule.mlir xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir build/test/Conversion/ascend-realize-requires-schedule.mlir'
```

Expected: 2/2 passed.

---

## Task 5: Tracking And Focused Verification

**Files:**

- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Update Phase 3 tracking**

In `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`, change Phase 3 overview from `Planned` to `In Progress` and add a Phase 3 execution table:

```markdown
## Phase 3：Realize plan objects

目标：建立第四层 `--ascend-realize` pass、稳定 plan object 和 MVP debug report，后续再接入真实 bufferization、placement、static memory、movement 和 materialization。

| 任务 | 状态 | 说明 | 验收 |
|---|---|---|---|
| Task 0: Realize MVP 计划 | `Done` | 拆分第一批 Realize 实现范围 | 计划文件已提交 |
| Task 1: Realize pass skeleton and plan reports | `Done` | `--ascend-realize`、`debug-stage=realize`、MVP plan objects | focused lit 2/2 passed |
```

- [ ] **Step 2: Run final focused verification**

Run on xvm:

```bash
rsync -av --relative docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md xvm@orb:/home/niu/code/Ascend-MLIR/
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ninja -C build afir-opt && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir build/test/Conversion/ascend-realize-requires-schedule.mlir build/test/Conversion/ascend-v2-pipeline-mvp.mlir'
```

Expected: focused lit passed. If `ascend-v2-pipeline-mvp.mlir` is not changed to include Realize, it should still pass as regression coverage.

- [ ] **Step 3: Run broader check**

Run on xvm:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && cmake --build build --target check-afir -j10'
```

Expected: no new Realize failures. Existing unrelated example pipeline manifest failures may remain and must be recorded separately.

- [ ] **Step 4: Commit**

Run:

```bash
git add include/Conversion/Passes.td include/Conversion/Passes.h include/Conversion/AscendV2/Debug/DebugOptions.h lib/Conversion/AscendV2/Debug/DebugOptions.cpp include/Conversion/AscendV2/Realize lib/Conversion/AscendV2/Realize lib/Conversion/AscendV2/CMakeLists.txt test/Conversion/ascend-realize-mvp.mlir test/Conversion/ascend-realize-requires-schedule.mlir docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md
git commit -m "realize: add Ascend V2 MVP plan reports"
```

Expected: one implementation commit. Do not stage `AGENTS.md`.

---

## Self-Review

- Spec coverage: This plan covers V2-5 section 5.1 main objects and a minimal 5.7 frozen `MemoryRealizationPlan` report. It intentionally defers 5.3-5.6 real algorithms to later tasks.
- Placeholder scan: no placeholder tasks; every task has concrete files and commands.
- Type consistency: all MVP types live under `mlir::afir::ascend::v2::realize`; report and pass use the same `RealizePlanBundle` object.
- Scope boundary: no real IR materialization in this batch; this keeps the first Realize slice reviewable and independently testable.
