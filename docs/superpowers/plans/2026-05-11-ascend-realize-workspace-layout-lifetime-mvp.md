# Ascend Realize Workspace Layout Lifetime MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Continue Phase 3B by adding a conservative workspace layout / lifetime planning slice to `StaticMemoryPlanner`.

**Architecture:** Keep zero-on-chip placement on the existing `empty_workspace` path. When placement contains on-chip buffers, build a report-only `workspace_layout` plan with one live interval and one workspace slot per on-chip buffer. This slice does not materialize allocs/subviews, does not insert movement, and does not perform byte-level capacity verification because value-level size/alignment is not modeled yet.

**Tech Stack:** MLIR/LLVM C++ pass infrastructure, Ascend Realize planners, LIT/FileCheck, GoogleTest unit tests, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Modify `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify `lib/Conversion/Ascend/Realize/StaticMemoryPlanner.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
- Modify `test/unittests/Conversion/AscendRealizePlannerTest.cpp`
- Create `test/Conversion/ascend-realize-workspace-layout.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains unrelated local user edits.

## Design Decisions

- `StaticMemoryPlanner` remains report-only and does not mutate IR.
- `PlacementPlan::onChipPlaceCount` is the only current signal for local workspace demand.
- For zero on-chip buffers:
  - preserve `mode = "empty_workspace"`
  - preserve `workspaceSlotCount = 0`
  - preserve `peakUsageKnown = false`
- For positive on-chip buffers:
  - set `mode = "workspace_layout"`
  - set `localBufferCount = onChipPlaceCount`
  - set `liveIntervalCount = onChipPlaceCount`
  - set `workspaceSlotCount = onChipPlaceCount`
  - set `peakUsageKnown = true`
  - set `peakUsageUnitCount = workspaceSlotCount`
  - set `capacityCheckDeferred = true`
- The first layout is conservative: no slot reuse yet, so lifetime computation is represented as stable interval counts, not real interval coloring.
- Byte-level capacity checking remains planned until value-level buffer sizes, alignment, and selected places are modeled.

## Task 1: RED Tests

**Files:**
- Modify `test/unittests/Conversion/AscendRealizePlannerTest.cpp`
- Create `test/Conversion/ascend-realize-workspace-layout.mlir`

- [x] Add a unit test `StaticMemoryPlannerBuildsWorkspaceLayoutForOnChipPlaces`.
- [x] In the unit test, create a `PlacementPlan` with `selectedPlaceCount = 4`, `gmPlaceCount = 3`, `onChipPlaceCount = 1`, `deferredLocalPlaceCount = 0`.
- [x] Check `mode = "workspace_layout"`, `trackedPlaceCount = 4`, `localBufferCount = 1`, `liveIntervalCount = 1`, `workspaceSlotCount = 1`, `peakUsageKnown = true`, `peakUsageUnitCount = 1`, and `capacityCheckDeferred = true`.
- [x] Add a LIT test that runs target-aware placement on a two-op vector temporary kernel using the existing synthetic CANN profile.
- [x] Check the report shows `StaticMemoryPlan mode = "workspace_layout"`, one live interval, one workspace slot, known peak usage, one peak usage unit, and deferred byte-capacity check.
- [x] Run focused xvm tests before implementation. Expected: compile/FileCheck failure because the new fields and report values do not exist yet.

## Task 2: Minimal Implementation

**Files:**
- Modify `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify `lib/Conversion/Ascend/Realize/StaticMemoryPlanner.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`

- [x] Add `localBufferCount`, `liveIntervalCount`, `peakUsageUnitCount`, and `capacityCheckDeferred` to `StaticMemoryPlan`.
- [x] Preserve the existing empty workspace behavior when `placement.onChipPlaceCount == 0`.
- [x] For positive `onChipPlaceCount`, build the conservative `workspace_layout` plan described above.
- [x] Print the new fields in `RealizeReport`.
- [x] Keep `MovementPlanner` and `MemoryRealizationDriver` behavior unchanged.

## Task 3: Tracking, Review, And Verification

**Files:**
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [x] Mark `workspace layout / lifetime MVP` as `Done`.
- [x] State that slot reuse, value-level selected place, byte-level capacity verification, explicit movement, and `memory_space` materialization remain planned.
- [x] Run `git diff --check -- . ':!AGENTS.md'`.
- [x] Run `test/tools/check_ascend_no_v2_code_naming.sh`.
- [x] Run xvm build for `afir-opt` and `AscendRealizePlannerTest`.
- [x] Run xvm focused LIT for workspace layout, target-aware placement, and phase3 completion.
- [x] Request spec compliance review, then code quality review.
- [x] Run final xvm regression before commit and push.
