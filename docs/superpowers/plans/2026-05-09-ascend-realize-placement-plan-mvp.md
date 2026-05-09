# Ascend Realize PlacementPlan MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first concrete `PlacementPlan` slice for Ascend Realize by assigning collected buffer facts to conservative `GM` placement and reporting the result.

**Architecture:** Keep `--ascend-realize` read-only for this slice. A new `PlacementPlanner` consumes `BufferizedKernelIR` and builds a `PlacementPlan` with all known buffer facts assigned to `GM`, while recording how many local/on-chip placements are deferred. Later `TargetMemoryModel` integration will replace this GM-default policy while preserving the planner/report interface.

**Tech Stack:** MLIR C++ pass infrastructure, LLVM ADT containers, LIT/FileCheck, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Create `include/Conversion/Ascend/Realize/PlacementPlanner.h`
  - Declares the read-only `PlacementPlanner` interface.
- Create `lib/Conversion/Ascend/Realize/PlacementPlanner.cpp`
  - Implements conservative GM-default placement planning from `BufferizedKernelIR`.
- Modify `include/Conversion/Ascend/Realize/RealizeTypes.h`
  - Extends `PlacementPlan` with placement mode and counters.
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
  - Calls `PlacementPlanner` while building each `RealizePlanBundle`.
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
  - Prints the new placement counters.
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
  - Adds the new implementation file.
- Modify `test/Conversion/ascend-realize-mvp.mlir`
  - Updates the existing smoke test to expect GM-default placement.
- Add `test/Conversion/ascend-realize-placement-plan.mlir`
  - Covers placement report behavior for one-op, two-op, and dead-result facts.
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
  - Records the new Phase 3 placement MVP and verification result.

## Task 1: Red Tests For PlacementPlan MVP

**Files:**
- Modify: `test/Conversion/ascend-realize-mvp.mlir`
- Add: `test/Conversion/ascend-realize-placement-plan.mlir`

- [ ] **Step 1: Update the existing Realize smoke placement expectation**

Change the `PlacementPlan` checks in `test/Conversion/ascend-realize-mvp.mlir` from:

```mlir
// CHECK: PlacementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   selected_places = 0
```

to:

```mlir
// CHECK: PlacementPlan:
// CHECK-NEXT:   kernel = kernel_0
// CHECK-NEXT:   mode = "gm_default"
// CHECK-NEXT:   selected_places = 3
// CHECK-NEXT:   gm_places = 3
// CHECK-NEXT:   on_chip_places = 0
// CHECK-NEXT:   deferred_local_places = 0
```

Reasoning: the single elementwise kernel has two external inputs and one output, all conservatively assigned to `GM`; there are no internal temporary values.

- [ ] **Step 2: Add a focused placement report test**

Create `test/Conversion/ascend-realize-placement-plan.mlir` with this content:

```mlir
// RUN: afir-opt %s --split-input-file --ascend-realize='dump-report=true debug-stage=realize' 2>&1 | FileCheck %s

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

// -----

func.func @scheduled_dead_result_kernel(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) attributes {ascend.normalized = true} {
  %empty0 = tensor.empty() : tensor<64xf16>
  %dead = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
    outs(%empty0 : tensor<64xf16>)
    attrs = {
      ascend.kernel = "kernel_1",
      ascend.op_role = "vector",
      ascend.schedule.decision_id = "kernel_1.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    } {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<64xf16>
  return
}

// CHECK: BufferizedKernelIR:
// CHECK-NEXT:   kernel = kernel_1
// CHECK-NEXT:   mode = "tensor_facts"
// CHECK-NEXT:   buffer_values = 2
// CHECK-NEXT:   input_values = 2
// CHECK-NEXT:   output_values = 0
// CHECK-NEXT:   temporary_values = 0
// CHECK: PlacementPlan:
// CHECK-NEXT:   kernel = kernel_1
// CHECK-NEXT:   mode = "gm_default"
// CHECK-NEXT:   selected_places = 2
// CHECK-NEXT:   gm_places = 2
// CHECK-NEXT:   on_chip_places = 0
// CHECK-NEXT:   deferred_local_places = 0
```

- [ ] **Step 3: Run RED verification in xvm/docker**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir build/test/Conversion/ascend-realize-placement-plan.mlir'
```

Expected: FAIL because the current Realize report still prints `selected_places = 0` and does not print placement mode, GM count, on-chip count, or deferred-local count.

## Task 2: PlacementPlanner Data Model And Implementation

**Files:**
- Add: `include/Conversion/Ascend/Realize/PlacementPlanner.h`
- Add: `lib/Conversion/Ascend/Realize/PlacementPlanner.cpp`
- Modify: `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify: `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] **Step 1: Extend `PlacementPlan`**

Add these fields to `PlacementPlan` in `include/Conversion/Ascend/Realize/RealizeTypes.h`:

```cpp
  std::string mode = "none";
  unsigned gmPlaceCount = 0;
  unsigned onChipPlaceCount = 0;
  unsigned deferredLocalPlaceCount = 0;
```

Keep `selectedPlaceCount` as the total number of classified planning units. `gmPlaceCount + onChipPlaceCount` must equal `selectedPlaceCount` for this MVP.

- [ ] **Step 2: Declare the planner interface**

Create `include/Conversion/Ascend/Realize/PlacementPlanner.h`:

```cpp
//===- PlacementPlanner.h - Ascend realize placement plan -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PLACEMENTPLANNER_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PLACEMENTPLANNER_H

#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::afir::ascend::realize {

class PlacementPlanner {
public:
  FailureOr<PlacementPlan> build(const BufferizedKernelIR &bufferizedIR) const;
};

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_PLACEMENTPLANNER_H
```

- [ ] **Step 3: Implement GM-default placement**

Create `lib/Conversion/Ascend/Realize/PlacementPlanner.cpp`:

```cpp
//===- PlacementPlanner.cpp - Ascend realize placement plan --------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/PlacementPlanner.h"

namespace mlir::afir::ascend::realize {

FailureOr<PlacementPlan>
PlacementPlanner::build(const BufferizedKernelIR &bufferizedIR) const {
  PlacementPlan plan;
  plan.kernelId = bufferizedIR.kernelId;
  plan.mode = "gm_default";
  plan.selectedPlaceCount = bufferizedIR.bufferValueCount;
  plan.gmPlaceCount = bufferizedIR.bufferValueCount;
  plan.onChipPlaceCount = 0;
  plan.deferredLocalPlaceCount = bufferizedIR.temporaryValueCount;
  return plan;
}

} // namespace mlir::afir::ascend::realize
```

This MVP deliberately assigns every known buffer fact to `GM`. `temporaryValueCount` is reported as deferred local placement so later target-aware planning has an explicit work item count.

- [ ] **Step 4: Add the source to CMake**

Add `Realize/PlacementPlanner.cpp` after `Realize/BufferizationDriver.cpp` in `lib/Conversion/Ascend/CMakeLists.txt`:

```cmake
  Realize/BufferizationDriver.cpp
  Realize/PlacementPlanner.cpp
  Realize/RealizePass.cpp
```

## Task 3: Wire PlacementPlanner Into Realize And Report

**Files:**
- Modify: `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Modify: `lib/Conversion/Ascend/Realize/RealizeReport.cpp`

- [ ] **Step 1: Include the planner**

In `RealizePass.cpp`, add:

```cpp
#include "Conversion/Ascend/Realize/PlacementPlanner.h"
```

- [ ] **Step 2: Build placement for each bundle**

Before the kernel loop, construct the planner:

```cpp
  PlacementPlanner placementPlanner;
```

After `bundle.bufferizedIR` has been assigned, replace the current placeholder `bundle.placement.kernelId = bundle.kernel.kernelId;` with:

```cpp
    FailureOr<PlacementPlan> placement =
        placementPlanner.build(bundle.bufferizedIR);
    if (failed(placement))
      return failure();
    bundle.placement = std::move(*placement);
```

Leave `StaticMemoryPlan`, `MovementPlan`, and `MemoryRealizationPlan` as placeholders.

- [ ] **Step 3: Print placement counters**

In `RealizeReport.cpp`, print these lines after `PlacementPlan.kernel` and before `selected_places`:

```cpp
    os << "  mode = \"" << bundle.placement.mode << "\"\n";
```

Then print these lines after `selected_places`:

```cpp
    os << "  gm_places = " << bundle.placement.gmPlaceCount << "\n";
    os << "  on_chip_places = " << bundle.placement.onChipPlaceCount << "\n";
    os << "  deferred_local_places = "
       << bundle.placement.deferredLocalPlaceCount << "\n";
```

- [ ] **Step 4: Run focused GREEN verification in xvm/docker**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir build/test/Conversion/ascend-realize-placement-plan.mlir'
```

Expected: PASS for both LIT tests.

## Task 4: Tracking, Review, Guard, And Broader Verification

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Update the Phase 3 board**

Change the Phase 3 `PlacementPlan` row from `Planned` to `Done` only after focused and broader verification succeeds. The description must state this is a GM-default MVP and not target-aware placement.

- [ ] **Step 2: Run code naming guard on host**

Run:

```bash
test/tools/check_ascend_no_v2_code_naming.sh
```

Expected: PASS with no source-level `V2/v2` naming regressions.

- [ ] **Step 3: Run focused Ascend Conversion suite in xvm/docker**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt AscendCommonAttributesTest AscendKernelPatternTest && ./build/bin/AscendCommonAttributesTest && ./build/bin/AscendKernelPatternTest && ctest --test-dir build -R "Ascend(CommonAttributes|KernelPattern)Test" --output-on-failure && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-"'
```

Expected: build passes, both unit binaries pass, ctest reports 2/2 passed, and Ascend-filtered LIT reports all selected tests passed.

- [ ] **Step 4: Commit**

Commit only after all review and verification commands have passed. Do not include unrelated `AGENTS.md` edits:

```bash
git add docs/superpowers/plans/2026-05-09-ascend-realize-placement-plan-mvp.md \
        docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md \
        include/Conversion/Ascend/Realize/PlacementPlanner.h \
        include/Conversion/Ascend/Realize/RealizeTypes.h \
        lib/Conversion/Ascend/Realize/PlacementPlanner.cpp \
        lib/Conversion/Ascend/Realize/RealizePass.cpp \
        lib/Conversion/Ascend/Realize/RealizeReport.cpp \
        lib/Conversion/Ascend/CMakeLists.txt \
        test/Conversion/ascend-realize-mvp.mlir \
        test/Conversion/ascend-realize-placement-plan.mlir
git commit -m "feat: add Ascend realize placement plan MVP"
```

## Self-Review

- Spec coverage: This plan implements only the first `PlacementPlan` slice from V2-5: a conservative GM-default placement plan report from already collected buffer facts. It does not query `TargetMemoryModel`, select on-chip places, check capacity/path legality, write `memory_space`, create workspace, insert movement, or materialize memory.
- Placeholder scan: No incomplete implementation requirements remain.
- Type consistency: `PlacementPlan` owns the new counters; `PlacementPlanner` fills them; `RealizePass` copies them into bundles; `RealizeReport` prints them.
- Testing: RED/GREEN LIT flow, host naming guard, xvm build/unit/ctest, and Ascend-filtered LIT are all specified.
