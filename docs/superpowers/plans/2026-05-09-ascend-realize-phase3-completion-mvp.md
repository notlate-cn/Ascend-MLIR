# Ascend Realize Phase 3 Completion MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Phase 3 Realize plan-object chain by adding read-only MVP planners for static memory, movement, and memory realization.

**Architecture:** Keep `--ascend-realize` read-only. The existing Bufferization facts and GM-default PlacementPlan feed three new components: `StaticMemoryPlanner`, `MovementPlanner`, and `MemoryRealizationDriver`. These components report conservative no-workspace/no-movement/read-only-freeze plans so the full Phase 3 object chain is explicit and testable before real one-shot bufferization, target-aware placement, workspace allocation, data movement, and IR materialization are implemented.

**Tech Stack:** MLIR C++ pass infrastructure, LLVM ADT containers, LIT/FileCheck, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Create `include/Conversion/Ascend/Realize/StaticMemoryPlanner.h`
- Create `lib/Conversion/Ascend/Realize/StaticMemoryPlanner.cpp`
- Create `include/Conversion/Ascend/Realize/MovementPlanner.h`
- Create `lib/Conversion/Ascend/Realize/MovementPlanner.cpp`
- Create `include/Conversion/Ascend/Realize/MemoryRealizationDriver.h`
- Create `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
- Modify `test/Conversion/ascend-realize-mvp.mlir`
- Add `test/Conversion/ascend-realize-phase3-completion.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains unrelated local user edits.

## Task 1: Red Tests For Full Phase 3 Plan Chain

**Files:**
- Modify: `test/Conversion/ascend-realize-mvp.mlir`
- Add: `test/Conversion/ascend-realize-phase3-completion.mlir`

- [ ] **Step 1: Update the existing MVP test expectations**

In `test/Conversion/ascend-realize-mvp.mlir`, update the `StaticMemoryPlan`, `MovementPlan`, and `MemoryRealizationPlan` checks:

```mlir
// CHECK: StaticMemoryPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "empty_workspace"
// CHECK-NEXT:   tracked_places = 3
// CHECK-NEXT:   workspace_slots = 0
// CHECK-NEXT:   peak_usage_known = false
// CHECK: MovementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "gm_noop"
// CHECK-NEXT:   cross_place_edges = 0
// CHECK-NEXT:   movements = 0
// CHECK-NEXT:   redundant_movements = 0
// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "read_only_freeze"
// CHECK-NEXT:   frozen = true
// CHECK-NEXT:   verified = true
// CHECK-NEXT:   materialized_allocs = 0
// CHECK-NEXT:   materialized_copies = 0
```

- [ ] **Step 2: Add a dedicated completion test**

Create `test/Conversion/ascend-realize-phase3-completion.mlir`:

```mlir
// RUN: afir-opt %s --split-input-file --ascend-realize='dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

func.func @two_op_kernel(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> attributes {ascend.normalized = true} {
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

// CHECK-LABEL: Realize report
// CHECK-NEXT:   kernels = 1
// CHECK: BufferizedKernelIR:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "tensor_facts"
// CHECK-NEXT:   buffer_values = 4
// CHECK-NEXT:   input_values = 2
// CHECK-NEXT:   output_values = 1
// CHECK-NEXT:   temporary_values = 1
// CHECK: PlacementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "gm_default"
// CHECK-NEXT:   selected_places = 4
// CHECK-NEXT:   gm_places = 4
// CHECK-NEXT:   on_chip_places = 0
// CHECK-NEXT:   deferred_local_places = 1
// CHECK: StaticMemoryPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "empty_workspace"
// CHECK-NEXT:   tracked_places = 4
// CHECK-NEXT:   workspace_slots = 0
// CHECK-NEXT:   peak_usage_known = false
// CHECK: MovementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "gm_noop"
// CHECK-NEXT:   cross_place_edges = 0
// CHECK-NEXT:   movements = 0
// CHECK-NEXT:   redundant_movements = 0
// CHECK: MemoryRealizationPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "read_only_freeze"
// CHECK-NEXT:   frozen = true
// CHECK-NEXT:   verified = true
// CHECK-NEXT:   materialized_allocs = 0
// CHECK-NEXT:   materialized_copies = 0
```

- [ ] **Step 3: Run RED verification in xvm/docker**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir build/test/Conversion/ascend-realize-phase3-completion.mlir'
```

Expected: FAIL because current reports do not print modes or counters for static memory, movement, and realization beyond the previous placeholders.

## Task 2: Add Planner/Driver Data Models

**Files:**
- Modify: `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Create: `include/Conversion/Ascend/Realize/StaticMemoryPlanner.h`
- Create: `include/Conversion/Ascend/Realize/MovementPlanner.h`
- Create: `include/Conversion/Ascend/Realize/MemoryRealizationDriver.h`

- [ ] **Step 1: Extend `StaticMemoryPlan`**

```cpp
  std::string mode = "none";
  unsigned trackedPlaceCount = 0;
  bool peakUsageKnown = false;
```

- [ ] **Step 2: Extend `MovementPlan`**

```cpp
  std::string mode = "none";
  unsigned crossPlaceEdgeCount = 0;
  unsigned redundantMovementCount = 0;
```

- [ ] **Step 3: Extend `MemoryRealizationPlan`**

```cpp
  std::string mode = "none";
  bool verified = false;
```

- [ ] **Step 4: Create `StaticMemoryPlanner.h`**

```cpp
//===- StaticMemoryPlanner.h - Ascend static memory plan ------*- C++ -*-===//
#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_STATICMEMORYPLANNER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_STATICMEMORYPLANNER_H

#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::ascend::realize {

class StaticMemoryPlanner {
public:
  FailureOr<StaticMemoryPlan> build(const PlacementPlan &placement) const;
};

} // namespace mlir::afir::ascend::realize

#endif
```

- [ ] **Step 5: Create `MovementPlanner.h`**

```cpp
//===- MovementPlanner.h - Ascend movement plan -------------*- C++ -*-===//
#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MOVEMENTPLANNER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MOVEMENTPLANNER_H

#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::ascend::realize {

class MovementPlanner {
public:
  FailureOr<MovementPlan> build(const PlacementPlan &placement,
                                const StaticMemoryPlan &staticMemory) const;
};

} // namespace mlir::afir::ascend::realize

#endif
```

- [ ] **Step 6: Create `MemoryRealizationDriver.h`**

```cpp
//===- MemoryRealizationDriver.h - Ascend memory realization --*- C++ -*-===//
#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_MEMORYREALIZATIONDRIVER_H

#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::ascend::realize {

class MemoryRealizationDriver {
public:
  FailureOr<MemoryRealizationPlan>
  materialize(const PlacementPlan &placement,
              const StaticMemoryPlan &staticMemory,
              const MovementPlan &movement) const;
};

} // namespace mlir::afir::ascend::realize

#endif
```

## Task 3: Implement Read-Only MVP Planners

**Files:**
- Create: `lib/Conversion/Ascend/Realize/StaticMemoryPlanner.cpp`
- Create: `lib/Conversion/Ascend/Realize/MovementPlanner.cpp`
- Create: `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Implement `StaticMemoryPlanner.cpp`**

```cpp
//===- StaticMemoryPlanner.cpp - Ascend static memory plan ---------------===//
#include "Conversion/Ascend/Realize/StaticMemoryPlanner.h"

namespace mlir::afir::ascend::realize {

FailureOr<StaticMemoryPlan>
StaticMemoryPlanner::build(const PlacementPlan &placement) const {
  StaticMemoryPlan plan;
  plan.kernelId = placement.kernelId;
  plan.mode = "empty_workspace";
  plan.trackedPlaceCount = placement.selectedPlaceCount;
  plan.workspaceSlotCount = 0;
  plan.peakUsageKnown = false;
  return plan;
}

} // namespace mlir::afir::ascend::realize
```

- [ ] **Step 2: Implement `MovementPlanner.cpp`**

```cpp
//===- MovementPlanner.cpp - Ascend movement plan ------------------------===//
#include "Conversion/Ascend/Realize/MovementPlanner.h"

namespace mlir::afir::ascend::realize {

FailureOr<MovementPlan>
MovementPlanner::build(const PlacementPlan &placement,
                       const StaticMemoryPlan &staticMemory) const {
  MovementPlan plan;
  plan.kernelId = placement.kernelId;
  plan.mode = "gm_noop";
  plan.crossPlaceEdgeCount = 0;
  plan.movementCount = 0;
  plan.redundantMovementCount = 0;
  (void)staticMemory;
  return plan;
}

} // namespace mlir::afir::ascend::realize
```

- [ ] **Step 3: Implement `MemoryRealizationDriver.cpp`**

```cpp
//===- MemoryRealizationDriver.cpp - Ascend memory realization -----------===//
#include "Conversion/Ascend/Realize/MemoryRealizationDriver.h"

namespace mlir::afir::ascend::realize {

FailureOr<MemoryRealizationPlan>
MemoryRealizationDriver::materialize(const PlacementPlan &placement,
                                     const StaticMemoryPlan &staticMemory,
                                     const MovementPlan &movement) const {
  MemoryRealizationPlan plan;
  plan.kernelId = placement.kernelId;
  plan.mode = "read_only_freeze";
  plan.frozen = true;
  plan.verified = staticMemory.kernelId == placement.kernelId &&
                  movement.kernelId == placement.kernelId;
  plan.materializedAllocCount = 0;
  plan.materializedCopyCount = 0;
  return plan;
}

} // namespace mlir::afir::ascend::realize
```

- [ ] **Step 4: Add sources to CMake**

```cmake
  Realize/MemoryRealizationDriver.cpp
  Realize/MovementPlanner.cpp
  Realize/PlacementPlanner.cpp
  Realize/RealizePass.cpp
  Realize/RealizeReport.cpp
  Realize/StaticMemoryPlanner.cpp
```

Keep the list sorted by component name within the Realize block if possible.

## Task 4: Wire The Full Plan Chain Into Realize

**Files:**
- Modify: `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Modify: `lib/Conversion/Ascend/Realize/RealizeReport.cpp`

- [ ] **Step 1: Add includes**

```cpp
#include "Conversion/Ascend/Realize/MemoryRealizationDriver.h"
#include "Conversion/Ascend/Realize/MovementPlanner.h"
#include "Conversion/Ascend/Realize/StaticMemoryPlanner.h"
```

- [ ] **Step 2: Instantiate planners before the bundle loop**

```cpp
  StaticMemoryPlanner staticMemoryPlanner;
  MovementPlanner movementPlanner;
  MemoryRealizationDriver memoryRealizationDriver;
```

- [ ] **Step 3: Replace placeholders in the bundle loop**

After placement is built:

```cpp
    FailureOr<StaticMemoryPlan> staticMemory =
        staticMemoryPlanner.build(bundle.placement);
    if (failed(staticMemory))
      return failure();
    bundle.staticMemory = std::move(*staticMemory);

    FailureOr<MovementPlan> movement =
        movementPlanner.build(bundle.placement, bundle.staticMemory);
    if (failed(movement))
      return failure();
    bundle.movement = std::move(*movement);

    FailureOr<MemoryRealizationPlan> realization =
        memoryRealizationDriver.materialize(bundle.placement,
                                            bundle.staticMemory,
                                            bundle.movement);
    if (failed(realization))
      return failure();
    bundle.realization = std::move(*realization);
```

Remove the old direct assignments for `staticMemory.kernelId`, `movement.kernelId`, `realization.kernelId`, and `realization.frozen`.

- [ ] **Step 4: Extend report output**

Print the new fields in `RealizeReport.cpp`:

```cpp
    os << "  mode = \"" << bundle.staticMemory.mode << "\"\n";
    os << "  tracked_places = " << bundle.staticMemory.trackedPlaceCount << "\n";
    os << "  peak_usage_known = "
       << (bundle.staticMemory.peakUsageKnown ? "true" : "false") << "\n";
```

```cpp
    os << "  mode = \"" << bundle.movement.mode << "\"\n";
    os << "  cross_place_edges = " << bundle.movement.crossPlaceEdgeCount
       << "\n";
    os << "  redundant_movements = "
       << bundle.movement.redundantMovementCount << "\n";
```

```cpp
    os << "  mode = \"" << bundle.realization.mode << "\"\n";
    os << "  verified = "
       << (bundle.realization.verified ? "true" : "false") << "\n";
    os << "  materialized_allocs = "
       << bundle.realization.materializedAllocCount << "\n";
    os << "  materialized_copies = "
       << bundle.realization.materializedCopyCount << "\n";
```

## Task 5: Verification, Review, Tracking, Commit

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Run focused GREEN verification**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir build/test/Conversion/ascend-realize-phase3-completion.mlir'
```

Expected: 2/2 passed.

- [ ] **Step 2: Update tracking board**

Set `StaticMemoryPlan`, `MovementPlan`, and `MemoryRealizationPlan` to `Done` as read-only MVPs. Add verification rows for RED/GREEN, spec review, code quality review, xvm focused tests, and code naming guard.

- [ ] **Step 3: Run final focused verification**

```bash
test/tools/check_ascend_no_v2_code_naming.sh
git diff --check
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt AscendCommonAttributesTest AscendKernelPatternTest && ./build/bin/AscendCommonAttributesTest && ./build/bin/AscendKernelPatternTest && ctest --test-dir build -R "Ascend(CommonAttributes|KernelPattern)Test" --output-on-failure && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-"'
```

Expected: guard and diff checks pass; unit binaries pass; ctest 2/2 passes; Ascend-filtered LIT passes.

- [ ] **Step 4: Commit without `AGENTS.md`**

```bash
git add docs/superpowers/plans/2026-05-09-ascend-realize-phase3-completion-mvp.md \
        docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md \
        include/Conversion/Ascend/Realize/StaticMemoryPlanner.h \
        include/Conversion/Ascend/Realize/MovementPlanner.h \
        include/Conversion/Ascend/Realize/MemoryRealizationDriver.h \
        include/Conversion/Ascend/Realize/RealizeTypes.h \
        lib/Conversion/Ascend/Realize/StaticMemoryPlanner.cpp \
        lib/Conversion/Ascend/Realize/MovementPlanner.cpp \
        lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp \
        lib/Conversion/Ascend/Realize/RealizePass.cpp \
        lib/Conversion/Ascend/Realize/RealizeReport.cpp \
        lib/Conversion/Ascend/CMakeLists.txt \
        test/Conversion/ascend-realize-mvp.mlir \
        test/Conversion/ascend-realize-phase3-completion.mlir
git commit -m "feat: complete Ascend realize phase 3 MVP"
git push
```

## Self-Review

- This completes Phase 3 at the plan-object MVP level: all five Realize plan objects have explicit builders/drivers and report output.
- It intentionally does not claim full one-shot bufferization, target-aware placement, workspace packing, data movement insertion, or memory materialization.
- It keeps implementation read-only and makes deferred work visible through modes and counters.
