# Mix Emitter Endgame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining supported-mix translator work so the mix path is cleanly separated into analysis, configuration, kernel shell, and region emitters while preserving the currently passing xvm runtime path.

**Architecture:** Keep supported semantics intentionally narrow, but make the implementation replaceable. The work proceeds by extracting the remaining local template logic into explicit helpers, then introducing explicit region-emission boundaries for cube, vector, and synchronization paths. Every step is validated by translator-output inspection and xvm end-to-end execution.

**Tech Stack:** MLIR/LLVM C++, `afir-translate`, AscendC mix kernel codegen, xvm/orb simulator validation, git

---

### Task 1: Finish Supported-Mix Structural Decomposition

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `examples/matmul-add-leakyrelu/run.sh`

- [ ] **Step 1: Extract remaining AIV local lambdas into file-level helpers**

Move the remaining AIV-local helpers in `emitSupportedMixAivRegion(...)` into file-level static helpers adjacent to the existing AIC helpers:

```c++
static void emitSupportedMixVectorCountDecl(raw_ostream &os,
                                            const SupportedMixKernelConfig &config);
static void emitSupportedMixAivQueueSetup(raw_ostream &os,
                                          const SupportedMixKernelConfig &config);
static void emitSupportedMixAivInputCopy(raw_ostream &os);
static void emitSupportedMixAivOutputCopy(raw_ostream &os);
```

Update `emitSupportedMixAivRegion(...)` so it becomes simple orchestration:

```c++
static void emitSupportedMixAivRegion(raw_ostream &os,
                                      const SupportedMixKernelConfig &config) {
  os << "  if ASCEND_IS_AIV {\\n";
  emitSupportedMixAivQueueSetup(os, config);
  emitSupportedMixAivInputCopy(os);
  emitSupportedMixVectorEpilogue(os, config);
  emitSupportedMixAivOutputCopy(os);
  os << "  }\\n";
}
```

- [ ] **Step 2: Rebuild translator and inspect output**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build --target afir-translate -j4'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/afir-translate -mlir-to-cann examples/matmul-add-leakyrelu/step7_cann.mlir | grep -n "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)\\|mm.SetBias\\|CrossCoreSetFlag<0x2, PIPE_FIX>(3)\\|CrossCoreWaitFlag(3)\\|LeakyRelu\\|uint32_t count = static_cast<uint32_t>(tiling.singleCoreM \\* tiling.singleCoreN / 2)"'
```

Expected:
- output contains all six lines

- [ ] **Step 3: Run xvm end-to-end validation**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log | tail -n 20'
```

Expected:
- final lines include `max_abs_diff=0.000000e+00`
- final lines include `mean_abs_diff=0.000000e+00`
- final lines include `PASS`

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp
git commit -m "Extract supported mix AIV emit helpers"
```

### Task 2: Introduce Explicit Region-Emission Boundaries

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `examples/matmul-add-leakyrelu/run.sh`

- [ ] **Step 1: Add explicit region-emission entry points**

Add file-level helpers that represent region-level responsibilities:

```c++
static void emitSupportedMixBoundarySync(raw_ostream &os,
                                         const SupportedMixKernelConfig &config);
static void emitSupportedMixCubeRegion(raw_ostream &os,
                                       const SupportedMixKernelConfig &config);
static void emitSupportedMixVectorRegion(raw_ostream &os,
                                         const SupportedMixKernelConfig &config);
```

Implementation rule:
- `emitSupportedMixCubeRegion(...)` should delegate to current AIC helpers
- `emitSupportedMixBoundarySync(...)` should own cross-core synchronization emission
- `emitSupportedMixVectorRegion(...)` should delegate to current AIV helpers

This is a structural relocation only. Do not change emitted semantics.

- [ ] **Step 2: Update top-level supported mix emission to use region helpers**

Make `emitSupportedMixKernel(...)` read like this:

```c++
static void emitSupportedMixKernel(raw_ostream &os, func::FuncOp funcOp,
                                   const SupportedMixKernelConfig &config) {
  emitSupportedMixKernelPrologue(os, funcOp.getName(), config);
  emitSupportedMixCubeRegion(os, config);
  emitSupportedMixBoundarySync(os, config);
  emitSupportedMixVectorRegion(os, config);
  os << "}\\n";
}
```

Adjust AIC/AIV helpers so cross-core set/wait logic lives under the boundary helper rather than being embedded in the region body.

- [ ] **Step 3: Rebuild translator and inspect output**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build --target afir-translate -j4'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/afir-translate -mlir-to-cann examples/matmul-add-leakyrelu/step7_cann.mlir | grep -n "CrossCoreSetFlag<0x2, PIPE_FIX>(3)\\|CrossCoreWaitFlag(3)\\|LeakyRelu\\|uint32_t count = static_cast<uint32_t>(tiling.singleCoreM \\* tiling.singleCoreN / 2)"'
```

Expected:
- all four lines still appear

- [ ] **Step 4: Run xvm end-to-end validation**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log | tail -n 20'
```

Expected:
- final lines include `max_abs_diff=0.000000e+00`
- final lines include `mean_abs_diff=0.000000e+00`
- final lines include `PASS`

- [ ] **Step 5: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp
git commit -m "Introduce supported mix region emit boundaries"
```

### Task 3: Align Analysis Outputs with Region Inputs

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `examples/matmul-add-leakyrelu/run.sh`

- [ ] **Step 1: Make region helpers consume only explicit config facts**

Audit helper signatures and ensure region helpers rely only on:

```c++
SupportedMixKernelConfig
MixTaskKindDescriptor
```

No region helper should recompute analysis facts by scanning MLIR again.

If any helper still derives facts internally, move that logic into config inference helpers and pass only the resulting facts into emission.

- [ ] **Step 2: Add file-level comments marking the analysis/config/emission layers**

Insert short comments above the major sections in `CannTranslation.cpp`:

```c++  
// Supported mix analysis helpers.
// Supported mix configuration inference.
// Supported mix emission helpers.
```

The comments should be brief and only used to lock in the new structure.

- [ ] **Step 3: Rebuild translator and inspect output**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build --target afir-translate -j4'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/afir-translate -mlir-to-cann examples/matmul-add-leakyrelu/step7_cann.mlir | grep -n "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)\\|mm.SetBias\\|CrossCoreSetFlag<0x2, PIPE_FIX>(3)\\|CrossCoreWaitFlag(3)\\|LeakyRelu"'
```

Expected:
- all five lines still appear

- [ ] **Step 4: Run xvm end-to-end validation**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log | tail -n 20'
```

Expected:
- final lines include `max_abs_diff=0.000000e+00`
- final lines include `mean_abs_diff=0.000000e+00`
- final lines include `PASS`

- [ ] **Step 5: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp
git commit -m "Align supported mix analysis and region inputs"
```

### Task 4: Reduce Supported-Mix Logic to Capability Gate + Emission

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `examples/matmul-add-leakyrelu/run.sh`

- [ ] **Step 1: Make the top-level supported-mix path read as capability check plus emission**

Refactor the translator flow so the supported path is structured like:

```c++
MixPartitionSummary summary = buildMixPartitionSummary(funcOp);
if (!isSupportedCurrentMixEmission(funcOp, summary))
  return failure();

FailureOr<SupportedMixKernelConfig> config =
    inferSupportedMixKernelConfig(funcOp, summary);
if (failed(config))
  return failure();

emitSupportedMixKernel(os, funcOp, *config);
```

If this structure already exists, tighten naming and surrounding helper boundaries so that no unrelated logic is interleaved with it.

- [ ] **Step 2: Rebuild translator and inspect output**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build --target afir-translate -j4'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./build/bin/afir-translate -mlir-to-cann examples/matmul-add-leakyrelu/step7_cann.mlir | grep -n "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)\\|mm.SetBias\\|CrossCoreSetFlag<0x2, PIPE_FIX>(3)\\|CrossCoreWaitFlag(3)\\|LeakyRelu\\|uint32_t count = static_cast<uint32_t>(tiling.singleCoreM \\* tiling.singleCoreN / 2)"'
```

Expected:
- all six lines still appear

- [ ] **Step 3: Run xvm end-to-end validation**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log | tail -n 20'
```

Expected:
- final lines include `max_abs_diff=0.000000e+00`
- final lines include `mean_abs_diff=0.000000e+00`
- final lines include `PASS`

- [ ] **Step 4: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp
git commit -m "Finalize supported mix emission structure"
```

### Task 5: Final Documentation and Clean Review

**Files:**
- Create: `docs/superpowers/specs/2026-04-07-mix-emitter-endgame-design.md`
- Create: `docs/superpowers/plans/2026-04-07-mix-emitter-endgame-plan.md`
- Test: `examples/matmul-add-leakyrelu/run.sh`

- [ ] **Step 1: Verify docs are present and clean**

Check that both files exist:

```bash
ls docs/superpowers/specs/2026-04-07-mix-emitter-endgame-design.md
ls docs/superpowers/plans/2026-04-07-mix-emitter-endgame-plan.md
```

Expected:
- both files are listed

- [ ] **Step 2: Final xvm confirmation after the last code task**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log | tail -n 20'
```

Expected:
- final lines include `max_abs_diff=0.000000e+00`
- final lines include `mean_abs_diff=0.000000e+00`
- final lines include `PASS`

- [ ] **Step 3: Commit docs if they changed after review**

```bash
git add docs/superpowers/specs/2026-04-07-mix-emitter-endgame-design.md docs/superpowers/plans/2026-04-07-mix-emitter-endgame-plan.md
git commit -m "Add mix emitter endgame design and plan"
```
