# Ascend Realize Data Movement Plan MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Continue Phase 3B by upgrading `MovementPlanner` from GM-only noop to an explicit data movement planning/report MVP.

**Architecture:** Keep zero-on-chip placement on the existing `gm_noop` path. When placement and static memory show on-chip workspace demand, emit a report-only `movement_planning` plan that records movement demand, workspace reuse candidates, and deferred path selection. This slice does not insert `memref.copy`, does not write `memory_space`, and does not claim legal target path selection because value-level source/destination places are not modeled yet.

**Tech Stack:** MLIR/LLVM C++ pass infrastructure, Ascend Realize planners, LIT/FileCheck, GoogleTest unit tests, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Modify `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify `lib/Conversion/Ascend/Realize/MovementPlanner.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
- Modify `test/unittests/Conversion/AscendRealizePlannerTest.cpp`
- Create `test/Conversion/ascend-realize-data-movement-plan.mlir`
- Modify `test/Conversion/ascend-realize-mvp.mlir`
- Modify `test/Conversion/ascend-realize-phase3-completion.mlir`
- Modify `test/Conversion/ascend-realize-workspace-layout.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains unrelated local user edits.

## Design Decisions

- `MovementPlanner` remains report-only and does not mutate IR.
- `movement_planning` is entered only when `placement.onChipPlaceCount > 0`.
- `crossPlaceEdgeCount` is a conservative count of movement demand and equals `placement.onChipPlaceCount`.
- `movementDemandCount` also equals `placement.onChipPlaceCount`.
- `selectedPathCount = 0` in this MVP because value-level source/destination places are not modeled.
- `pathSelectionDeferredCount = movementDemandCount`.
- `workspaceReuseCandidateCount = staticMemory.workspaceSlotCount`.
- `movementCount = 0` and `materializationDeferred = true` so the report cannot be confused with inserted `memref.copy` operations.
- Existing `gm_noop` behavior remains unchanged for zero on-chip placement.

## Task 1: RED Tests

**Files:**
- Modify `test/unittests/Conversion/AscendRealizePlannerTest.cpp`
- Create `test/Conversion/ascend-realize-data-movement-plan.mlir`

- [x] Add a unit test `MovementPlannerBuildsPlanningForOnChipWorkspace`.
- [x] In the unit test, create target-aware placement with `onChipPlaceCount = 1` and a `workspace_layout` static memory plan with `workspaceSlotCount = 1`.
- [x] Check `mode = "movement_planning"`, `crossPlaceEdgeCount = 1`, `movementDemandCount = 1`, `selectedPathCount = 0`, `pathSelectionDeferredCount = 1`, `workspaceReuseCandidateCount = 1`, `movementCount = 0`, and `materializationDeferred = true`.
- [x] Add a LIT test that runs target-aware placement on a vector temporary kernel using the existing synthetic CANN profile.
- [x] Check the report shows `MovementPlan mode = "movement_planning"` with deferred path selection and zero materialized movements.
- [x] Check the same LIT still reports `MemoryRealizationPlan mode = "read_only_freeze"` and `materialized_copies = 0`.
- [x] Run focused xvm tests before implementation. Expected: compile/FileCheck failure because the new fields and movement planning mode do not exist yet.

## Task 2: Minimal Implementation

**Files:**
- Modify `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify `lib/Conversion/Ascend/Realize/MovementPlanner.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`

- [x] Add `movementDemandCount`, `selectedPathCount`, `pathSelectionDeferredCount`, `workspaceReuseCandidateCount`, and `materializationDeferred` to `MovementPlan`.
- [x] Preserve existing validation: kernel id must match and tracked place count must match selected place count.
- [x] Preserve `gm_noop` when `placement.onChipPlaceCount == 0`.
- [x] Emit `movement_planning` with deferred path selection when `placement.onChipPlaceCount > 0`.
- [x] Print the new movement fields after existing movement counters in `RealizeReport`.
- [x] Keep `MemoryRealizationDriver` unchanged so materialization remains read-only.

## Task 3: Tracking, Review, And Verification

**Files:**
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [x] Mark `explicit data movement plan MVP` as `Done`.
- [x] State that legal path selection, value-level movement steps, `memref.copy`, `memory_space`, and alloc/copy materialization remain planned.
- [x] Run `git diff --check -- . ':!AGENTS.md'`.
- [x] Run `test/tools/check_ascend_no_v2_code_naming.sh`.
- [x] Run xvm build for `afir-opt` and `AscendRealizePlannerTest`.
- [x] Run xvm focused LIT for data movement, workspace layout, and phase3 completion.
- [x] Request spec compliance review, then code quality review.
- [x] Run final xvm regression before commit and push.
