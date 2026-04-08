# Generic Mix Single-Chain Op-Driven Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the single-chain mix translator emit cube/vector/boundary bodies through region-local op dispatch instead of relying on fixed supported-mix body templates.

**Architecture:** Keep `MixPartitionPlan` and current single-chain validation as the routing boundary. Preserve the current shell/ABI/task-kind constraints, but replace region body emission with op-driven dispatch that walks region-local ops and emits supported ones explicitly. Unsupported region-local ops must fail with targeted diagnostics.

**Tech Stack:** MLIR, C++, AFIR translator tests, xvm end-to-end example validation

---

### Task 1: Define Region-Op Dispatch Boundary

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `test/Target/cann-translate-mix.mlir`
- Test: `test/Target/cann-translate-mix-no-bias.mlir`

- [ ] **Step 1: Add explicit region-op dispatch entry points**

Introduce file-local helpers for op-driven region emission, conceptually:

- `emitMixCubeRegionOps(...)`
- `emitMixBoundaryRegionOps(...)`
- `emitMixVectorRegionOps(...)`

These helpers should accept a `MixRegionPlan` and dispatch over its ops instead of assuming a semantic pattern.

- [ ] **Step 2: Keep the current shell helpers separate**

Ensure shell helpers remain responsible only for:

- kernel signature
- task kind spelling
- shell prologue/epilogue
- shared shell declarations

Do not let region-op helpers pull shell responsibilities back in.

- [ ] **Step 3: Run translator checks**

Run:

```bash
./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix-input.mlir | FileCheck test/Target/cann-translate-mix.mlir
./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix-no-bias-input.mlir | FileCheck test/Target/cann-translate-mix-no-bias.mlir
```

Expected:

- both pass
- visible shell structure remains unchanged

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp test/Target/cann-translate-mix.mlir test/Target/cann-translate-mix-no-bias.mlir
git commit -m "Introduce op-driven mix region emission boundary"
```

### Task 2: Migrate Boundary Body Emission to Op Dispatch

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `test/Target/cann-translate-mix.mlir`
- Test: `test/Target/cann-translate-mix-island.mlir`

- [ ] **Step 1: Route boundary emission through region-local ops**

Make boundary region emission consume the boundary region’s op list and selected payload rather than calling a fixed boundary template directly.

- [ ] **Step 2: Preserve current explicit payload checks**

Do not weaken:

- selected crossing retention
- payload feasibility
- explicit synchronization behavior

- [ ] **Step 3: Verify structural negative coverage still holds**

Run:

```bash
not ./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix-island-input.mlir 2>&1 | FileCheck test/Target/cann-translate-mix-island.mlir
```

Expected:

- failure remains explicit
- diagnostic still reports chain-external vector ops

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp test/Target/cann-translate-mix.mlir test/Target/cann-translate-mix-island.mlir
git commit -m "Drive mix boundary emission from region-local ops"
```

### Task 3: Migrate Vector Region to Op-Driven Dispatch

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `test/Target/cann-translate-mix.mlir`
- Test: `test/Target/cann-translate-mix-no-bias.mlir`
- Create: `test/Target/cann-translate-mix-unsupported-vector.mlir`
- Create: `test/Target/cann-translate-mix-unsupported-vector-input.mlir`

- [ ] **Step 1: Implement vector region op dispatch**

Replace the remaining fixed vector-body template assumptions with iteration over region-local ops.

The dispatcher should emit supported ops using existing emit capabilities and reject unsupported ones explicitly.

- [ ] **Step 2: Add targeted unsupported-vector negative coverage**

Create a fixture whose structure is valid single-chain mix, but whose vector region contains an op that the new dispatcher intentionally does not yet support.

Expected diagnostic style:

- unsupported vector op in single-chain mix emitter

- [ ] **Step 3: Verify positive and negative translator behavior**

Run:

```bash
./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix-input.mlir | FileCheck test/Target/cann-translate-mix.mlir
./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix-no-bias-input.mlir | FileCheck test/Target/cann-translate-mix-no-bias.mlir
not ./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix-unsupported-vector-input.mlir 2>&1 | FileCheck test/Target/cann-translate-mix-unsupported-vector.mlir
```

Expected:

- positive fixtures pass
- unsupported-vector fixture fails with the explicit emission-time diagnostic

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp test/Target/cann-translate-mix.mlir test/Target/cann-translate-mix-no-bias.mlir test/Target/cann-translate-mix-unsupported-vector.mlir test/Target/cann-translate-mix-unsupported-vector-input.mlir
git commit -m "Add op-driven vector mix emission"
```

### Task 4: Migrate Cube Region to Op-Driven Dispatch

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Create: `test/Target/cann-translate-mix-unsupported-cube.mlir`
- Create: `test/Target/cann-translate-mix-unsupported-cube-input.mlir`

- [ ] **Step 1: Implement cube region op dispatch**

Replace the remaining fixed cube-body template assumptions with region-op dispatch for cube-local ops.

- [ ] **Step 2: Add unsupported-cube negative coverage**

Create a fixture whose structure is valid single-chain mix, but whose cube region contains an op intentionally unsupported by the new dispatcher.

Expected diagnostic style:

- unsupported cube op in single-chain mix emitter

- [ ] **Step 3: Verify translator behavior**

Run:

```bash
./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix-input.mlir | FileCheck test/Target/cann-translate-mix.mlir
not ./build/bin/afir-translate -mlir-to-cann test/Target/cann-translate-mix-unsupported-cube-input.mlir 2>&1 | FileCheck test/Target/cann-translate-mix-unsupported-cube.mlir
```

Expected:

- baseline mix fixture still passes
- unsupported-cube fixture fails with the explicit emission-time diagnostic

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp test/Target/cann-translate-mix-unsupported-cube.mlir test/Target/cann-translate-mix-unsupported-cube-input.mlir
git commit -m "Add op-driven cube mix emission"
```

### Task 5: End-to-End Verification on xvm

**Files:**
- Modify: `docs/superpowers/plans/2026-04-08-generic-mix-single-chain-op-driven-plan.md`

- [ ] **Step 1: Rebuild affected tools**

Run on xvm:

```bash
cd /home/niu/code/Ascend-MLIR
cmake --build build --target afir-translate check-afir -j10
```

- [ ] **Step 2: Re-run mix end-to-end example**

Run on xvm:

```bash
cd /home/niu/code/Ascend-MLIR
bash -lc 'source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log'
```

Expected:

- `PASS`
- `max_abs_diff=0.000000e+00`
- `mean_abs_diff=0.000000e+00`

- [ ] **Step 3: Re-run unified example harness**

Run on xvm:

```bash
cd /home/niu/code/Ascend-MLIR
bash scripts/build.sh --build-tests
```

Expected:

- lit suite passes
- example pipeline harness still passes

- [ ] **Step 4: Record verification note**

Append a short note to this plan documenting:

- translator command results
- xvm end-to-end result
- whether any unsupported-op negatives were added

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-04-08-generic-mix-single-chain-op-driven-plan.md
git commit -m "Record op-driven mix emission verification"
```

### Verification Note

- 2026-04-08 xvm rebuild:
  - `cmake --build build --target check-afir -j10`
  - result: `Total Discovered Tests: 19`, `Passed: 19 (100.00%)`
- 2026-04-08 xvm mix e2e:
  - `bash -lc 'source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log'`
  - result: `PASS`
  - `max_abs_diff=0.000000e+00`
  - `mean_abs_diff=0.000000e+00`
- 2026-04-08 xvm full test entry:
  - `bash scripts/build.sh --build-tests`
  - result: exit code `0`
  - lit suite passed; tool integration stage reported `0 passed, 0 failed`
- Added unsupported-op negative coverage:
  - `test/Target/cann-translate-mix-unsupported-vector.mlir`
  - `test/Target/cann-translate-mix-unsupported-cube.mlir`
