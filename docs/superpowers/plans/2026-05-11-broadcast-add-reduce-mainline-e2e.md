# Broadcast Add Reduce Mainline E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `examples/broadcast-add-reduce/step0_input.mlir` run through the new Ascend mainline to generated CANN kernel code and runtime-session simulator validation for the current no-real-tiling acceptance slice.

**Architecture:** First close the compiler IR contract with focused LIT coverage, then add an example script that uses the new Ascend pass sequence and Phase 5 artifacts. Keep old transform tiling, `ascendc-buffer-placement`, and `linalg-to-ascendc` out of the new script.

**Tech Stack:** MLIR LIT/FileCheck, Ascend Normalize/Kernelize/Schedule/Realize passes, Ascend Phase 5 wrappers, `afir-translate -mlir-to-cann`, `runtime-session` sim backend.

---

### Task 1: Mainline IR Contract for Broadcast Add Reduce

**Files:**
- Create: `test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir`
- Modify: `lib/Conversion/Ascend/Schedule/KernelPatternView.cpp`
- Modify: `lib/Conversion/Ascend/Realize/MemoryRealizationDriver.cpp`

- [ ] **Step 1: Write the failing LIT**

Create a self-contained broadcast-add-reduce test based on `examples/broadcast-add-reduce/step0_input.mlir`. The positive RUN must use:

```bash
afir-opt %s \
  --linalg-fuse-elementwise-ops \
  --ascend-normalize \
  --ascend-kernelize \
  --ascend-schedule \
  --ascend-realize='materialization-mode=memory-space-annotate' \
  --ascend-compute-lower \
  --ascend-parallelize \
  --ascend-prepare-for-emit \
  --ascend-canonicalize-cann-signature
```

Check for `ascendc.reduce_sum_2d_l2`, `ascendc.data_copy_l2`, CANN ABI `memref<ui8>`, and no residual `linalg.generic`.

- [ ] **Step 2: Verify RED**

Run in xvm:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion/ascend-full-pipeline-broadcast-add-reduce.mlir'
```

Expected: FAIL with `requires ascend.kernel` on `linalg.fill`.

- [ ] **Step 3: Make Schedule tolerate helper fill ops**

Update `buildKernelPatternViews` so unmarked `linalg.fill` ops are not treated as missing kernel patterns. They are helper initializers consumed by the scheduled reduction and are handled by Phase 5 compute lowering.

- [ ] **Step 4: Materialize final reduction outputs**

Extend the Phase 5 bridge in `MemoryRealizationDriver` to recognize supported final reduction outputs in addition to supported final vector outputs. The conservative pattern is a `linalg.generic` with at least one reduction iterator and a body made only from `arith.addf` / `arith.constant`, yielding the final add result. For final outputs, allocate VECOUT and insert `memref.copy VECOUT -> GM`.

- [ ] **Step 5: Verify GREEN**

Run the focused LIT from Step 2. Expected: PASS.

### Task 2: New Mainline Example Script

**Files:**
- Create: `examples/broadcast-add-reduce/run-mainline.sh`
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [ ] **Step 1: Write the script**

Create a new script that starts from `step0_input.mlir`, writes intermediate files into `build_mainline/`, and uses only the new mainline after frontend linalg canonicalization. The default shape is `M=64,N=15000`, because the current Schedule stage emits schedule attrs/report but does not yet materialize M-axis tiling loops. The original legacy demo shape `M=640,N=15000` remains a tracked follow-up until Schedule emits real tiling/block dispatch.

```bash
--linalg-fuse-elementwise-ops
--ascend-normalize
--ascend-kernelize
--ascend-schedule
--ascend-realize='materialization-mode=memory-space-annotate'
--ascend-compute-lower
--ascend-parallelize
--ascend-prepare-for-emit
--ascend-canonicalize-cann-signature
```

Then run `afir-translate -mlir-to-cann` with `--tiling-space-out`, `--runtime-manifest-out`, and `--host-tiling-out`.

- [ ] **Step 2: Generate data and runtime manifest**

Reuse `gen_data.py`. Generate a runtime-session run manifest that consumes the Phase 5 tiling schema and fills all required tiling params for the script shape. Default: `M=64`, `N=15000`; the script also accepts `--m` / `--n` for probing larger shapes.

- [ ] **Step 3: Verify simulator**

Run in xvm:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && AFIR_OPT=$PWD/build/bin/afir-opt AFIR_TRANSLATE=$PWD/build/bin/afir-translate RUNTIME_SESSION=$PWD/build/bin/runtime-session bash examples/broadcast-add-reduce/run-mainline.sh --log'
```

Expected: `session.result=success` and `session.validation=pass`.

Also probe the legacy large shape:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && AFIR_OPT=$PWD/build/bin/afir-opt AFIR_TRANSLATE=$PWD/build/bin/afir-translate RUNTIME_SESSION=$PWD/build/bin/runtime-session bash examples/broadcast-add-reduce/run-mainline.sh --m 640 --n 15000 --log'
```

Expected for this task: document the result. It is allowed to fail until Schedule materializes real tiling loops.

- [ ] **Step 4: Update tracking**

Record the new mainline example status, the exact verification command, the original large-shape result, and the remaining gap that the old `run.sh` still exists as legacy until all examples migrate.

### Task 3: Regression Sweep and Review

**Files:**
- Test-only unless review finds issues.

- [ ] **Step 1: Run focused conversion tests**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh >/tmp/ascend_env.log && ninja -C build afir-opt afir-translate runtime-session && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/Conversion --filter="ascend-"'
```

- [ ] **Step 2: Run runtime example**

Run `examples/broadcast-add-reduce/run-mainline.sh --log` again after the full conversion sweep.

- [ ] **Step 3: Run code naming guard**

```bash
test/tools/check_ascend_no_v2_code_naming.sh
```

- [ ] **Step 4: Review**

Use two-stage review: spec compliance first, then code quality. Fix only issues relevant to this migration.
