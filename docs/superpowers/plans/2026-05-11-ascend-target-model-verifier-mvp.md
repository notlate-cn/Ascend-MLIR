# Ascend Target Model Verifier MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Phase 4 by adding a read-only `TargetModelVerifier` that checks `TargetProfile`, `TargetMemoryModel`, `TargetIntrinsicModel`, and `TargetCostModel` are mutually closed.

**Architecture:** Keep each target model query-only. The verifier consumes already-built models and reports fail-fast diagnostics through `raw_ostream`; it does not rebuild models, mutate IR, or feed scheduling/realization yet.

**Tech Stack:** MLIR/LLVM C++ support types, LLVM ADT containers, GoogleTest unit tests, xvm/docker verification via `examples/dev-env.md`.

---

## File Structure

- Create `include/Target/Ascend/TargetModelVerifier.h`
- Create `lib/Target/Ascend/TargetModelVerifier.cpp`
- Modify `lib/Target/Ascend/CMakeLists.txt`
- Create `test/unittests/Target/AscendTargetModelVerifierTest.cpp`
- Modify `test/unittests/Target/CMakeLists.txt`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains unrelated local user edits.

## Design Decisions

- Public API:

```cpp
class TargetModelVerifier {
public:
  LogicalResult verify(const TargetProfile &profile,
                       const TargetMemoryModel &memoryModel,
                       const TargetIntrinsicModel &intrinsicModel,
                       const TargetCostModel &costModel,
                       raw_ostream &os) const;
};
```

- The verifier first calls `verifyTargetProfile`.
- Every required place must be present in `TargetMemoryModel`, and its capacity must be positive.
- The required direct MVP paths must exist with the expected `PathKind`.
- Every direct path exposed by `TargetMemoryModel` must reference supported places, have a path kind, and have a `TargetCostModel` path cost.
- `DirectCopy`, `Load2D`, `Load2DTranspose`, and `FixPipe` paths require a corresponding intrinsic class in `TargetIntrinsicModel`.
- `QueueTransfer` is a model-only execution-unit handoff in this MVP and does not require a movement intrinsic.
- `supportFixpipe=true` requires both a `FixPipe` path and a concrete fixpipe path intrinsic. A concrete fixpipe path intrinsic is not allowed when `supportFixpipe=false`.
- Diagnostics name the failed component and the relevant place/path/kind.

## Task 1: Red Tests For TargetModelVerifier

**Files:**
- Create `test/unittests/Target/AscendTargetModelVerifierTest.cpp`
- Modify `test/unittests/Target/CMakeLists.txt`

- [ ] Add a test target named `AscendTargetModelVerifierTest`.
- [ ] Add a complete synthetic profile with required capacities, movement intrinsics, fixpipe intrinsic, and memory rates.
- [ ] Add `AcceptsClosedTargetModel` to prove a complete profile/model set verifies.
- [ ] Add `RejectsMissingMovementIntrinsic` by removing all `Load2D` intrinsics and checking the diagnostic mentions `TargetPathIntrinsicMissing` and `Load2D`.
- [ ] Add `RejectsMissingPathCost` with an empty `TargetCostModel` and check the diagnostic mentions `TargetPathCostMissing` and the first missing direct path.
- [ ] Run the new test target build in xvm before production code exists. Expected result: compile failure because `TargetModelVerifier.h` is missing.

## Task 2: Implement TargetModelVerifier

**Files:**
- Create `include/Target/Ascend/TargetModelVerifier.h`
- Create `lib/Target/Ascend/TargetModelVerifier.cpp`
- Modify `lib/Target/Ascend/CMakeLists.txt`

- [ ] Define `TargetModelVerifier::verify`.
- [ ] Add local helpers for path printing, path-kind names, required place checks, required path checks, intrinsic-kind checks, and path-cost checks.
- [ ] Keep all helpers private to the `.cpp`.
- [ ] Run `AscendTargetModelVerifierTest` in xvm. Expected result: pass.

## Task 3: Tracking And Verification

**Files:**
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] Mark `profile verifier` as `Done` and rename the row to `TargetModelVerifier MVP`.
- [ ] Record that QueueTransfer is intentionally intrinsic-free in the MVP.
- [ ] Run the code naming guard.
- [ ] Run `git diff --check -- . ':!AGENTS.md'`.
- [ ] Build and run focused target model tests in xvm.
- [ ] Run target profile lit and Ascend conversion lit regression in xvm.
- [ ] Request spec compliance review, then code quality review.
- [ ] Fix any Critical/Important findings before commit.
