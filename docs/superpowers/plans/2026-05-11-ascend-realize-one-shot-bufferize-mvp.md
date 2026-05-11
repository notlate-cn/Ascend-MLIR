# Ascend Realize One-Shot Bufferize MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start Phase 3B by adding an opt-in `one-shot-bufferize` materialization path to `ascend-realize`.

**Architecture:** Keep the default Realize behavior plan-only and stable. Add a `materialization-mode` pass option with `plan-only` as the default and `one-shot-bufferize` as the first mutating mode. The mutating mode reuses MLIR One-Shot Bufferize after Realize has validated scheduled structured-lowering attrs and collected current plan facts; placement, workspace layout, explicit movement, and target-aware memory spaces remain future Phase 3B tasks.

**Tech Stack:** MLIR/LLVM C++ pass infrastructure, MLIR Bufferization transforms, LIT/FileCheck, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Modify `include/Conversion/Passes.td`
- Modify `include/Conversion/Ascend/Realize/BufferizationDriver.h`
- Modify `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Modify `lib/Conversion/Ascend/CMakeLists.txt`
- Create `test/Conversion/ascend-realize-one-shot-bufferize.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains unrelated local user edits.

## Design Decisions

- Pass option:

```text
--ascend-realize='materialization-mode=plan-only'
--ascend-realize='materialization-mode=one-shot-bufferize'
```

- `plan-only` preserves current behavior and current report text.
- `one-shot-bufferize` runs upstream MLIR One-Shot Bufferize with function boundary bufferization enabled.
- Realize still validates `ascend.kernel`, `ascend.schedule.decision_id`, and `ascend.schedule.structured_lowering` before any IR mutation.
- Realize still builds the existing read-only plan objects from tensor facts before mutation.
- The mutating mode does not yet assign target `memory_space`, does not insert target-specific movement, does not build workspace slots, and does not consume `TargetMemoryModel`.
- Unknown `materialization-mode` values must fail with a clear diagnostic before any IR mutation.

## Task 1: RED Tests

**Files:**
- Create `test/Conversion/ascend-realize-one-shot-bufferize.mlir`

- [ ] Add a scheduled tensor-level linalg function.
- [ ] Run `ascend-realize` with `materialization-mode=one-shot-bufferize`.
- [ ] Check the output function boundary is memref-based.
- [ ] Check a `memref.alloc` is present.
- [ ] Check `tensor.empty` is gone.
- [ ] Add a split-file case for an invalid materialization mode and check the diagnostic.
- [ ] Run the focused LIT test in xvm before implementation. Expected: fail because `materialization-mode` is not a recognized option.

## Task 2: Minimal Implementation

**Files:**
- Modify `include/Conversion/Passes.td`
- Modify `include/Conversion/Ascend/Realize/BufferizationDriver.h`
- Modify `lib/Conversion/Ascend/Realize/BufferizationDriver.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Modify `lib/Conversion/Ascend/CMakeLists.txt`

- [ ] Add the `materializationMode` pass option.
- [ ] Add `BufferizationDriver::runOneShotBufferize(ModuleOp module)`.
- [ ] Implement the method with `bufferization::OneShotBufferizationOptions`, `bufferizeFunctionBoundaries = true`, and `bufferization::runOneShotModuleBufferize`.
- [ ] In `RealizePass`, reject unknown materialization modes.
- [ ] In `RealizePass`, run One-Shot Bufferize only when the option is `one-shot-bufferize`.
- [ ] Add required Bufferization CMake link libraries.
- [ ] Run the focused LIT test in xvm. Expected: pass.

## Task 3: Tracking, Review, And Verification

**Files:**
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] Mark the Phase 3B One-Shot Bufferize row as `In Progress` or `Done` for this opt-in MVP slice.
- [ ] State that target-aware placement, workspace layout, explicit movement, and target memory-space mutation remain planned.
- [ ] Run code naming guard.
- [ ] Run `git diff --check -- . ':!AGENTS.md'`.
- [ ] Run xvm build for `afir-opt` and focused Realize/Target unit tests.
- [ ] Run xvm focused LIT for the new test and Ascend Conversion LIT regression.
- [ ] Request spec compliance review, then code quality review.
- [ ] Fix any Critical/Important review findings before commit.
