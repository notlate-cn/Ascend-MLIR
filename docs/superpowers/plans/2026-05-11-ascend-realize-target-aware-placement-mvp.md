# Ascend Realize Target-Aware Placement MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue Phase 3B by adding an opt-in target-aware placement planner slice that consumes `TargetMemoryModel`.

**Architecture:** Keep the default `gm-default` placement unchanged. Add a `placement-mode=target-aware` path that loads a CANN target profile, builds `TargetMemoryModel`, and lets `PlacementPlanner` place vector temporary buffers on `VECCALC` when the target model says the place exists, has capacity, and is visible to the Vector unit. This remains a planning/report slice; it does not write `memory_space`, allocate workspace, insert movement copies, or consume `TargetCostModel` yet.

**Tech Stack:** MLIR/LLVM C++ pass infrastructure, Ascend target model library, LIT/FileCheck, GoogleTest unit tests, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Modify `include/Conversion/Passes.td`
- Modify `include/Conversion/Ascend/Realize/PlacementPlanner.h`
- Modify `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp`
- Modify `lib/Conversion/Ascend/Realize/PlacementPlanner.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
- Modify `test/unittests/Conversion/AscendRealizePlannerTest.cpp`
- Create `test/Conversion/Inputs/ascend-target-aware-placement-cann/aarch64-linux/data/platform_config/SyntheticSoC.ini`
- Create `test/Conversion/ascend-realize-target-aware-placement.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains unrelated local user edits.

## Design Decisions

- Pass options:

```text
--ascend-realize='placement-mode=gm-default'
--ascend-realize='placement-mode=target-aware cann-root=<root> soc=<soc>'
```

- `gm-default` preserves current placement behavior.
- `target-aware` requires a valid `cann-root`; `soc` defaults to `Ascend910B2`
  and may be overridden. Invalid mode or missing target profile fails before
  placement planning.
- The first target-aware rule is intentionally narrow: function inputs/outputs stay in `GM`; vector temporaries move from deferred GM fallback to `VECCALC` when legal.
- Legal `VECCALC` means:
  - `TargetMemoryModel::supportsMemoryPlace(VECCALC)` is true
  - `TargetMemoryModel::getCapacity(VECCALC)` succeeds and capacity is positive
  - `TargetMemoryModel::isPlaceVisibleTo(VECCALC, Vector)` is true
- If `VECCALC` is not legal, the planner falls back to the existing GM-default counts.
- This slice does not write `memory_space` or insert `memref.copy`.

## Task 1: RED Tests

**Files:**
- Modify `test/unittests/Conversion/AscendRealizePlannerTest.cpp`
- Create `test/Conversion/Inputs/ascend-target-aware-placement-cann/aarch64-linux/data/platform_config/SyntheticSoC.ini`
- Create `test/Conversion/ascend-realize-target-aware-placement.mlir`

- [ ] Add a unit test that builds a complete synthetic `TargetMemoryModel` and verifies `PlacementPlanner::build(bufferizedIR, memoryModel)` sets `mode = "target_aware"`, `gmPlaceCount = inputs + outputs`, `onChipPlaceCount = temporaries`, and `deferredLocalPlaceCount = 0`.
- [ ] Add a LIT test that runs `ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC dump-report=true debug-stage=realize'`.
- [ ] Check the report shows `PlacementPlan mode = "target_aware"`, `gm_places = 3`, `on_chip_places = 1`, and `deferred_local_places = 0`.
- [ ] Add a negative LIT RUN for unknown `placement-mode` and check the diagnostic.
- [ ] Run focused xvm tests before implementation. Expected: compile/option failure because target-aware placement API/options do not exist.

## Task 2: Minimal Implementation

**Files:**
- Modify `include/Conversion/Passes.td`
- Modify `include/Conversion/Ascend/Realize/PlacementPlanner.h`
- Modify `lib/Conversion/Ascend/Realize/PlacementPlanner.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
- Modify `tools/afir-opt/CMakeLists.txt`

- [ ] Add `placementMode`, `soc`, and `cannRoot` pass options to `AscendRealizePass`.
- [ ] Extend `BufferizedKernelIR` facts with vector temporary counts derived
      from `ascend.op_role`.
- [ ] Add `PlacementPlanner::build(const BufferizedKernelIR &, const ::mlir::ascend::TargetMemoryModel &)`.
- [ ] Implement the VECCALC legality helper in `PlacementPlanner.cpp`.
- [ ] In `RealizePass`, reject unknown placement modes before plan construction.
- [ ] In `RealizePass`, load `CannTargetProfileLoader`, build `TargetMemoryModel`, and pass it to `PlacementPlanner` only for `target-aware`.
- [ ] Add link dependencies needed for the conversion library.
- [ ] Run the focused tests and keep existing `gm-default` report tests passing.

## Task 3: Tracking, Review, And Verification

**Files:**
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] Mark Phase 3B target-aware placement planning MVP as `Done`.
- [ ] State that `TargetCostModel` sorting, workspace layout, explicit movement, and `memory_space` materialization remain planned.
- [ ] Run code naming guard.
- [ ] Run `git diff --check -- . ':!AGENTS.md'`.
- [ ] Run xvm build for `afir-opt` and focused Realize/Target unit tests.
- [ ] Run xvm focused LIT and Ascend Conversion LIT regression.
- [ ] Request spec compliance review, then code quality review.
- [ ] Fix any Critical/Important review findings before commit.
