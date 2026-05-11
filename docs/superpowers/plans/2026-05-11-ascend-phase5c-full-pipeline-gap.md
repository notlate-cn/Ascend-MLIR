# Ascend Phase 5C Full Pipeline Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the current ordinary `Phase 0 -> Phase 5` full-pipeline acceptance gap as a reproducible expected-fail smoke and document the exact bridge needed to turn it positive.

**Architecture:** Do not change lowering behavior in this task. Add one LIT that runs the real pass chain through `ascend-realize` and `ascend-compute-lower`, plus a gap report and tracking update that classify the failure as the Realize-to-Phase-5 materialization bridge.

**Tech Stack:** MLIR pass pipeline, LIT/FileCheck, `afir-opt`, xvm/docker verification via `examples/dev-env.md`.

---

## Superseded

This plan captured the expected-fail gap state before the Realize-to-Phase-5
bridge. The bridge follow-up is tracked in
`docs/superpowers/plans/2026-05-11-ascend-realize-phase5-bridge.md`; the active
positive smoke is now
`test/Conversion/ascend-full-pipeline-ordinary-smoke.mlir`.

## File Structure

- Create `test/Conversion/ascend-full-pipeline-gap-smoke.mlir`
- Create `docs/Ascend-MLIR-Phase5C-Full-Pipeline-Gap-Report.zh.md`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

Do not modify or stage `AGENTS.md`; it contains local user instructions and unrelated working-tree changes.

## Task 1: Full-Pipeline Expected-Fail Smoke

**Files:**
- Create `test/Conversion/ascend-full-pipeline-gap-smoke.mlir`

- [x] **Step 1: Add ordinary tensor/linalg input**

Use a single `linalg.generic` vector add on tensor inputs:

```mlir
func.func @ordinary_elementwise(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16> {
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
```

- [x] **Step 2: Add one-shot full-pipeline RUN line**

```mlir
// RUN: not afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='materialization-mode=one-shot-bufferize' --ascend-compute-lower 2>&1 | FileCheck %s
```

Expected diagnostic:

```text
error: ascend-compute-lower left a lowerable operation behind
```

- [x] **Step 3: Add target-aware memory-space annotate RUN line**

```mlir
// RUN: not afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC materialization-mode=memory-space-annotate' --ascend-compute-lower 2>&1 | FileCheck %s --check-prefix=TARGET-AWARE
```

Expected diagnostic:

```text
error: ascend-compute-lower left a lowerable operation behind
```

## Task 2: Gap Report And Tracking

**Files:**
- Create `docs/Ascend-MLIR-Phase5C-Full-Pipeline-Gap-Report.zh.md`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [x] **Step 1: Document the bridge gap**

Record that current full-pipeline failure is expected because Realize produces bufferized GM memrefs but has not materialized Phase 5 visible on-chip buffers and copies.

- [x] **Step 2: Update tracking**

Add the plan/report links, the expected-fail smoke task, and a verification row. Keep Phase 5 backend integration marked done, but explicitly state that full ordinary `Phase 0 -> Phase 5` positive acceptance is blocked on the Realize-to-Phase-5 value-level materialization bridge.

## Task 3: Verification

**Files:**
- Test only; no source edits.

- [x] **Step 1: Run whitespace check**

```bash
git diff --check -- . ':!AGENTS.md'
```

Expected: no output, exit 0.

- [x] **Step 2: Run code naming guard**

```bash
test/tools/check_ascend_no_v2_code_naming.sh
```

Expected: exit 0.

- [x] **Step 3: Run xvm focused LIT**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test --filter="ascend-full-pipeline-gap-smoke"'
```

Expected: 1/1 passed.

- [x] **Step 4: Run xvm Ascend Conversion regression**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-"'
```

Expected: all discovered Ascend conversion tests pass.

Actual:

- host `git diff --check -- . ':!AGENTS.md'` passed.
- host `test/tools/check_ascend_no_v2_code_naming.sh` passed.
- xvm `test/tools/check_ascend_no_v2_code_naming.sh` passed.
- xvm focused LIT passed: `ascend-full-pipeline-gap-smoke.mlir` 1/1.
- xvm Ascend Conversion filtered regression passed: 39/39.
