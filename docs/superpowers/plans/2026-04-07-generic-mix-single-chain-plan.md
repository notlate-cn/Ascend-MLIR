# Generic Mix Single-Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the mix translator use a generic region-driven primary path for single-chain `cube -> boundary -> vector` kernels, while preserving current end-to-end correctness.

**Architecture:** Preserve `kernel_kind` and `ascendc.unit` as the analysis source of truth, build an executable `MixPartitionPlan`, validate that the plan is a supported single chain, and route eligible kernels through a region-driven mix emitter. Keep the current specialized supported-mix path only as transitional fallback until the new path is verified.

**Tech Stack:** MLIR, AscendC lowering passes, CANN translation, xvm-based end-to-end simulator validation.

---

### Task 1: Lock the Current Single-Chain Shape in Tests

**Files:**
- Modify: `test/Target/cann-translate-mix.mlir`
- Modify: `test/Target/cann-translate-mix-relu.mlir`
- Modify: `test/Target/cann-translate-mix-no-bias.mlir`
- Test: `test/Target/cann-translate-mix*.mlir`

- [ ] **Step 1: Add comments documenting the expected single-chain structure**

Add comments that explicitly identify the intended shape as one cube region, one boundary transfer, and one vector region.

- [ ] **Step 2: Tighten the translation checks around shell and branch structure**

Ensure the checks still verify mix shell structure, AIC branch, AIV branch, and boundary synchronization rather than sample-specific naming.

- [ ] **Step 3: Run translator checks manually**

Run: `./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix.mlir`
Expected: output still contains mix shell, AIC, boundary sync, and AIV structure.

- [ ] **Step 4: Commit**

```bash
git add test/Target/cann-translate-mix.mlir test/Target/cann-translate-mix-relu.mlir test/Target/cann-translate-mix-no-bias.mlir
git commit -m "Tighten generic mix single-chain translator coverage"
```

### Task 2: Make Single-Chain Eligibility Explicit in Analysis

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `test/Target/cann-translate-mix.mlir`

- [ ] **Step 1: Add a helper that validates a `MixPartitionPlan` as a single executable chain**

The helper should require exactly one cube region, exactly one vector region, and at least one boundary crossing connecting them.

- [ ] **Step 2: Make the helper produce structured failure reasons**

Represent reasons such as multiple cube regions, multiple vector regions, missing boundary, or invalid ordering.

- [ ] **Step 3: Route current generic-mix eligibility through that helper**

Replace implicit acceptance logic with one explicit single-chain validator.

- [ ] **Step 4: Re-run translator output on the current mix fixture**

Run: `./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix.mlir`
Expected: same translated shell as before.

- [ ] **Step 5: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp test/Target/cann-translate-mix.mlir
git commit -m "Validate generic mix single-chain eligibility explicitly"
```

### Task 3: Split Boundary Emission into a First-Class Layer

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `test/Target/cann-translate-mix.mlir`

- [ ] **Step 1: Extract a dedicated boundary emission entry point**

Separate boundary transfer and synchronization emission from the current cube/vector branch helpers.

- [ ] **Step 2: Make boundary emission consume explicit `MixBoundaryValue` data**

Do not infer transfer structure from the old supported-kernel template inside the boundary helper.

- [ ] **Step 3: Re-run the translation fixture**

Run: `./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix.mlir`
Expected: boundary synchronization is still emitted with unchanged visible structure.

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp test/Target/cann-translate-mix.mlir
git commit -m "Make mix boundary emission explicit"
```

### Task 4: Route the Primary Mix Path Through Region-Driven Emission

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `test/Target/cann-translate-mix.mlir`
- Test: `test/Target/cann-translate-mix-relu.mlir`
- Test: `test/Target/cann-translate-mix-no-bias.mlir`

- [ ] **Step 1: Add a generic single-chain emission entry point**

Create a top-level path that emits kernel shell, cube region, boundary layer, and vector region from `MixPartitionPlan`.

- [ ] **Step 2: Make eligible plans use that path as the primary route**

Retain the current supported-mix implementation only as fallback.

- [ ] **Step 3: Verify all three translation fixtures still translate correctly**

Run:
```bash
./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix.mlir
./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix-relu.mlir
./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix-no-bias.mlir
```
Expected: all still emit mix shell plus the expected relu/leaky-relu/no-bias forms.

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp test/Target/cann-translate-mix.mlir test/Target/cann-translate-mix-relu.mlir test/Target/cann-translate-mix-no-bias.mlir
git commit -m "Use generic mix single-chain emission as primary path"
```

### Task 5: Verify End-to-End on xvm

**Files:**
- Modify: `docs/superpowers/plans/2026-04-07-generic-mix-single-chain-plan.md`
- Test: `examples/matmul-add-leakyrelu/run.sh`
- Test: `examples/add-broadcast-concat/run.sh`

- [ ] **Step 1: Rebuild the translator on xvm**

Run the normal xvm build command used for this repo.
Expected: translator and runtime binaries rebuild cleanly.

- [ ] **Step 2: Re-run the mix example on xvm**

Run: `bash -lc 'source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log'`
Expected: `PASS` with the current zero-diff result.

- [ ] **Step 3: Re-run one non-mix sanity example on xvm**

Run: `bash -lc 'source examples/env.sh && bash examples/add-broadcast-concat/run.sh --log'`
Expected: `PASS` with current plain-path behavior unchanged.

- [ ] **Step 4: Record the verification result directly in the plan or a linked note**

Summarize commands run, commit under test, and observed results.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-04-07-generic-mix-single-chain-plan.md
git commit -m "Record generic mix single-chain xvm verification"
```
