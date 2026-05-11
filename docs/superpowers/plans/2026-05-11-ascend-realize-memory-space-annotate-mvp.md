# Ascend Realize Memory Space Annotate MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Phase 3B with an opt-in memory-space annotation MVP that performs conservative IR mutation after One-Shot Bufferize.

**Architecture:** Add `materialization-mode=memory-space-annotate`. This mode first runs upstream One-Shot Bufferize, then annotates only value-level vector temporary allocs that can be proven from scheduled linalg ops with `VECCALC` memory space. It updates `MemoryRealizationPlan` reports, but still does not create new workspace allocs, subviews, or `memref.copy`.

**Tech Stack:** MLIR/LLVM C++ pass infrastructure, One-Shot Bufferize, memref/linalg/func dialect APIs, LIT/FileCheck, GoogleTest unit tests, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Modify `include/Conversion/Passes.td`
- Modify `include/Conversion/Ascend/Realize/MemoryRealizationDriver.h`
- Modify `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`
- Modify `test/unittests/Conversion/AscendRealizePlannerTest.cpp`
- Create `test/Conversion/ascend-realize-memory-space-annotate.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains unrelated local user edits.

## Design Decisions

- `memory-space-annotate` is opt-in; default `plan-only` and existing `one-shot-bufferize` remain unchanged.
- The mode runs One-Shot Bufferize before memory-space annotation.
- The first mutation only updates existing `memref.alloc` result types for proven vector temporary buffers.
- A proven vector temporary buffer is a `memref.alloc` used as a vector linalg op DPS init, whose users are all in the same `ascend.kernel`, and that has no outside-kernel use such as function return or cast.
- Proven vector temporary buffers are annotated with integer memory space `11`, matching `VECCALC`.
- Function boundary GM annotation is deferred because explicit GM memory space `0` is printed as the default memref type and is not useful as visible acceptance evidence.
- The pass does not create new workspace allocs, subviews, or `memref.copy`.
- The pass updates the report to `mode = "memory_space_annotate"` with `memory_space_annotations > 0`, while keeping `materialized_allocs = 0` and `materialized_copies = 0`.

## Task 1: RED Tests

**Files:**
- Modify `test/unittests/Conversion/AscendRealizePlannerTest.cpp`
- Create `test/Conversion/ascend-realize-memory-space-annotate.mlir`

- [x] Add a unit test `MemoryRealizationAnnotatesPlanForMemorySpaces`.
- [x] In the unit test, call a new report helper on a `MemoryRealizationPlan` with annotation count `1`.
- [x] Check `mode = "memory_space_annotate"`, `frozen = true`, `verificationScope = "memory_space_annotation"`, `memorySpaceAnnotationCount = 1`, `materializedAllocCount = 0`, and `materializedCopyCount = 0`.
- [x] Add a LIT test that runs `--ascend-realize='materialization-mode=memory-space-annotate dump-report=true debug-stage=realize'` on a two-op vector temporary kernel.
- [x] Check the output IR contains a temporary `memref.alloc` with `11 : i32` memory space.
- [x] Check the report shows `MemoryRealizationPlan mode = "memory_space_annotate"`, `memory_space_annotations = 1`, `materialized_allocs = 0`, and `materialized_copies = 0`.
- [x] Add `CHECK-NOT: memref.copy` to prove this MVP did not materialize movement copies.
- [x] Run focused xvm tests before implementation. Expected: option/report/FileCheck failure because the new mode and fields do not exist yet.

## Task 2: Minimal Implementation

**Files:**
- Modify `include/Conversion/Passes.td`
- Modify `include/Conversion/Ascend/Realize/MemoryRealizationDriver.h`
- Modify `include/Conversion/Ascend/Realize/RealizeTypes.h`
- Modify `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizeReport.cpp`

- [x] Add `memory-space-annotate` as a supported `materialization-mode`.
- [x] Add `memorySpaceAnnotationCount` to `MemoryRealizationPlan`.
- [x] Add `MemoryRealizationDriver::annotateMemorySpaces(ModuleOp module)`.
- [x] In `annotateMemorySpaces`, collect proven vector temporary `memref.alloc` values after One-Shot Bufferize.
- [x] Replace each collected alloc with the same memref type carrying memory space `11 : i32`.
- [x] Add a helper that updates each bundle realization plan to `memory_space_annotate`.
- [x] Print `memory_space_annotations` in `RealizeReport`.
- [x] Preserve `one-shot-bufferize` behavior without annotation.

## Task 3: Tracking, Review, And Verification

**Files:**
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [x] Mark `memory_space annotation MVP` as `Done`.
- [x] State that real workspace alloc/subview, legal path selection, and `memref.copy` materialization remain planned.
- [x] Run `git diff --check -- . ':!AGENTS.md'`.
- [x] Run `test/tools/check_ascend_no_v2_code_naming.sh`.
- [x] Run xvm build for `afir-opt` and `AscendRealizePlannerTest`.
- [x] Run xvm focused LIT for memory-space annotation, one-shot bufferize, data movement plan, and workspace layout.
- [x] Request spec compliance review, then code quality review.
- [x] Run final xvm regression before commit and push.
