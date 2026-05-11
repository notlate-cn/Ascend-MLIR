# Ascend Realize To Phase 5 Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the minimum ordinary tensor/linalg full pipeline from expected-fail into a positive `Phase 0 -> Phase 5` smoke by materializing the Realize output shape required by Phase 5.

**Architecture:** Keep the bridge inside `MemoryRealizationDriver` and enable it only in `materialization-mode=memory-space-annotate`. The bridge preserves the GM result buffer, creates a Phase 5 visible `VECOUT` buffer for supported final vector outputs, rewires the linalg output to `VECOUT`, and inserts a `VECOUT -> GM` `memref.copy` epilogue that existing Phase 5 data-move lowering already supports.

**Tech Stack:** MLIR C++ rewrite utilities, linalg/memref dialects, Ascend Realize pass, AscendC Phase 5 lowering, LIT/FileCheck, xvm/docker verification.

---

## File Structure

- Modify `include/Conversion/Ascend/Realize/MemoryRealizationDriver.h`
- Modify `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`
- Rename and modify `test/Conversion/ascend-full-pipeline-ordinary-smoke.mlir`
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
- Create `docs/Ascend-MLIR-Phase5C-Full-Pipeline-Bridge-Report.zh.md`

Do not modify or stage `AGENTS.md`; it contains local user instructions and unrelated working-tree changes.

## Task 1: Positive Full-Pipeline Smoke

**Files:**
- Modify `test/Conversion/ascend-full-pipeline-ordinary-smoke.mlir`

- [x] **Step 1: Convert expected-fail to positive**

Use the existing ordinary tensor/linalg add input and run:

```mlir
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='materialization-mode=memory-space-annotate dump-report=true debug-stage=realize' --ascend-compute-lower 2>&1 | FileCheck %s
// RUN: afir-opt %s --ascend-normalize --ascend-kernelize --ascend-schedule --ascend-realize='placement-mode=target-aware cann-root=%S/Inputs/ascend-target-aware-placement-cann soc=SyntheticSoC materialization-mode=memory-space-annotate' --ascend-compute-lower --ascend-parallelize --ascend-prepare-for-emit --ascend-canonicalize-cann-signature | FileCheck %s --check-prefix=ABI
```

Check for:

```mlir
// CHECK: mode = "memory_space_materialize"
// CHECK: materialized_allocs = 1
// CHECK-NEXT: materialized_copies = 1
// CHECK: ascendc.add_l2
// CHECK: ascendc.data_copy_l2
// CHECK-NOT: linalg.generic
```

- [x] **Step 2: Verify RED**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test --filter="ascend-full-pipeline-ordinary-smoke"'
```

Expected before implementation: FAIL because `MemoryRealizationPlan` still reports `memory_space_annotate` and `materialized_allocs = 0`.

## Task 2: Realize Bridge Implementation

**Files:**
- Modify `include/Conversion/Ascend/Realize/MemoryRealizationDriver.h`
- Modify `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`
- Modify `lib/Conversion/Ascend/Realize/RealizePass.cpp`

- [x] **Step 1: Add bridge count model**

Add:

```cpp
struct Phase5BridgeMaterializationCounts {
  unsigned materializedAllocCount = 0;
  unsigned materializedCopyCount = 0;
};
```

- [x] **Step 2: Add conservative bridge matcher**

Match only:

- vector role ops
- all-parallel iterator types
- `linalg.generic` body with supported Phase 5 arithmetic ops, or `linalg.elementwise` add/mul/max
- default GM `memref.alloc` output
- no same-kernel or different-kernel consumer except the writer; external return/cast users are allowed

- [x] **Step 3: Materialize VECOUT output and GM epilogue**

For each matched output:

```cpp
auto vecOutAlloc = rewriter.create<memref::AllocOp>(
    loc, vecOutType, gmAlloc.getDynamicSizes(),
    gmAlloc.getSymbolOperands(), gmAlloc.getAlignmentAttr());
initOperand->set(vecOutAlloc.getResult());
rewriter.setInsertionPointAfter(linalgOp);
rewriter.create<memref::CopyOp>(loc, vecOutAlloc.getResult(),
                                gmAlloc.getResult());
```

- [x] **Step 4: Report materialization counts**

Report `mode = "memory_space_materialize"` and count materialized alloc/copy per kernel when the bridge fires.

## Task 3: Verification

**Files:**
- Test and docs only after implementation.

- [x] **Step 1: Build `afir-opt` in xvm**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt'
```

Expected: passed.

- [x] **Step 2: Run focused GREEN**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test --filter="ascend-full-pipeline-ordinary-smoke"'
```

Expected: passed after implementation.

## Task 4: Follow-Up Verification And Review

**Files:**
- Modify `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
- Create `docs/Ascend-MLIR-Phase5C-Full-Pipeline-Bridge-Report.zh.md`

- [x] **Step 1: Run full focused regression**

```bash
git diff --check -- . ':!AGENTS.md'
test/tools/check_ascend_no_v2_code_naming.sh
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-"'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && test/tools/check_ascend_no_v2_code_naming.sh'
```

Expected: all pass.

- [x] **Step 2: Run subagent reviews**

Request:

- spec compliance review for bridge scope and tracking
- code quality review for conservative matcher, IR mutation safety, and tests

Actual:

- Spec compliance review passed with no required fixes.
- Code quality review initially requested output-map identity and view/cast-mediated downstream-user safeguards.
- Re-review approved after adding output identity checks, recursive cast-to-return validation, and negative LIT coverage.

Actual verification so far:

- xvm `ninja -C build afir-opt` passed.
- xvm focused `ascend-full-pipeline-ordinary-smoke` passed.
- host `git diff --check -- . ':!AGENTS.md'` passed.
- host and xvm `test/tools/check_ascend_no_v2_code_naming.sh` passed.
- xvm `llvm-lit -v build/test/Conversion --filter="ascend-"` passed 39/39 after clearing a stale deleted test artifact from the xvm synced tree.
- xvm focused unit regression passed 8/8:
  `AscendCommonAttributesTest`, `AscendKernelPatternTest`,
  `AscendRealizePlannerTest`, `AscendBackendSupportMatrixTest`,
  `AscendTargetMemoryModelTest`, `AscendTargetIntrinsicModelTest`,
  `AscendTargetCostModelTest`, `AscendTargetModelVerifierTest`.
- xvm Target LIT passed 11/11.
- Code review follow-up added negative coverage for non-identity output maps,
  tensor extract/subview-mediated downstream users, and cast-mediated downstream
  users; focused smoke and Ascend Conversion regression still passed.
