# Non-Mix Pipeline Regression Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the five non-mix example pipelines on xvm after the mix-mode work, using shared mechanism fixes rather than sample-specific patches.

**Architecture:** Build a common regression matrix for the five examples, diagnose failures at shared routing boundaries, then repair MLIR/translator/runtime separation so non-mix kernels stay on the plain vector path while the validated mix path remains intact.

**Tech Stack:** MLIR/LLVM C++, shell scripts, `afir-opt`, `afir-translate`, `compiler`, `validator`, xvm/orb runtime environment, git

---

### Task 1: Build the xvm Regression Matrix

**Files:**
- Create: `docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md`
- Test: `examples/add-broadcast-concat/run.sh`
- Test: `examples/broadcast-add-reduce/run.sh`
- Test: `examples/gather-elementwise-fusion/run.sh`
- Test: `examples/relu-broadcast-transpose/run.sh`
- Test: `examples/split-relu-brc-add-mul/run.sh`

- [ ] **Step 1: Create the empty results matrix document**

Create `docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md` with this table skeleton:

```md
# Non-Mix Pipeline Regression Results

| Example | Expected Kind | Last Good Stage | First Failing Stage | Compile | Runtime | Accuracy | Root Cause Bucket | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| add-broadcast-concat | vec | | | | | | | |
| broadcast-add-reduce | vec | | | | | | | |
| gather-elementwise-fusion | vec | | | | | | | |
| relu-broadcast-transpose | vec | | | | | | | |
| split-relu-brc-add-mul | vec | | | | | | | |
```

- [ ] **Step 2: Run all five examples on xvm and capture raw outcomes**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash -lc "
source examples/env.sh
bash examples/add-broadcast-concat/run.sh --log > /tmp/add-broadcast-concat.log 2>&1 || true
bash examples/broadcast-add-reduce/run.sh --log > /tmp/broadcast-add-reduce.log 2>&1 || true
bash examples/gather-elementwise-fusion/run.sh --log > /tmp/gather-elementwise-fusion.log 2>&1 || true
bash examples/relu-broadcast-transpose/run.sh --log > /tmp/relu-broadcast-transpose.log 2>&1 || true
bash examples/split-relu-brc-add-mul/run.sh --log > /tmp/split-relu-brc-add-mul.log 2>&1 || true
"'
```

Expected:
- each example produces a complete log file
- failures are preserved instead of aborting the batch

- [ ] **Step 3: Summarize the five outcomes into the matrix**

For each log, record:
- highest completed stage
- first failing stage
- whether `.bin` exists
- whether runtime initialized
- whether accuracy passed
- whether the failure is environment, compile, runtime, or numerical

Use commands like:

```bash
ssh xvm@orb 'for f in /tmp/add-broadcast-concat.log /tmp/broadcast-add-reduce.log /tmp/gather-elementwise-fusion.log /tmp/relu-broadcast-transpose.log /tmp/split-relu-brc-add-mul.log; do echo "==== $f"; tail -n 60 "$f"; done'
```

- [ ] **Step 4: Commit the initial matrix**

```bash
git add docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md
git commit -m "Add non-mix pipeline regression matrix"
```

### Task 2: Verify Non-Mix Classification and Translator Routing

**Files:**
- Modify: `docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md`
- Modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Test: `examples/add-broadcast-concat/step7_cann.mlir`
- Test: `examples/broadcast-add-reduce/step7_cann.mlir`
- Test: `examples/gather-elementwise-fusion/step7_cann.mlir`
- Test: `examples/relu-broadcast-transpose/step7_cann.mlir`
- Test: `examples/split-relu-brc-add-mul/step7_cann.mlir`

- [ ] **Step 1: Inspect kernel-kind semantics in the five pipelines**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash -lc "
source examples/env.sh
for f in \
examples/add-broadcast-concat/step7_cann.mlir \
examples/broadcast-add-reduce/step7_cann.mlir \
examples/gather-elementwise-fusion/step7_cann.mlir \
examples/relu-broadcast-transpose/step7_cann.mlir \
examples/split-relu-brc-add-mul/step7_cann.mlir; do
  echo ==== \$f
  grep -n \"ascendc.kernel_kind\\|ascendc.unit\" \$f || true
done
"'
```

Expected:
- none of the five examples should require `kernel_kind=mix`
- any unexpected `AiCore.Cube` evidence is a primary suspect

- [ ] **Step 2: Inspect generated C++ for accidental mix emission**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash -lc "
source examples/env.sh
for f in \
examples/add-broadcast-concat/step8_kernel.cpp \
examples/broadcast-add-reduce/step8_kernel.cpp \
examples/gather-elementwise-fusion/step8_kernel.cpp \
examples/relu-broadcast-transpose/step8_kernel.cpp \
examples/split-relu-brc-add-mul/step8_kernel.cpp; do
  echo ==== \$f
  grep -n \"KERNEL_TYPE_MIX\\|ASCEND_IS_AIC\\|ASCEND_IS_AIV\\|CrossCoreSetFlag\\|CrossCoreWaitFlag\" \$f || true
done
"'
```

Expected:
- no mix shell or cross-core synchronization appears in these non-mix kernels

- [ ] **Step 3: If routing leakage exists, tighten translator entry conditions**

Adjust [`CannTranslation.cpp`](/Volumes/GM9/code/Ascend-MLIR/lib/Target/CannKernel/CannTranslation.cpp) only if inspection proves a non-mix kernel is entering mix-specific logic.

The fix must:
- guard mix lowering strictly on semantic classification
- leave vector lowering untouched for non-mix kernels
- avoid checking example names or generated symbol names

- [ ] **Step 4: Rebuild translator and spot-check one passing and one failing example**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build --target afir-translate -j4'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash -lc "source examples/env.sh && bash examples/relu-broadcast-transpose/run.sh --log"'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash -lc "source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log"'
```

Expected:
- the non-mix spot check improves or remains correct
- the mix example still passes

- [ ] **Step 5: Commit translator-side routing fixes**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md
git commit -m "Restore non-mix translator routing boundaries"
```

### Task 3: Verify Runtime and Validator Separation

**Files:**
- Modify: `docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md`
- Modify: `lib/Runtime/Compiler.cpp`
- Modify: `lib/Runtime/Executor.cpp`
- Modify: `tools/validator/validator_main.cpp`

- [ ] **Step 1: Confirm the five examples stay on the plain non-mix execution path**

Inspect compile and run behavior by checking:
- whether `.bin` generation is plain or packed/mix-specific
- whether validator requires mix metadata
- whether executor enters mix-specific code for these examples

Use targeted searches:

```bash
cd /Volumes/GM9/code/Ascend-MLIR
rg -n "mix|packed|launch info|manifest|RunPackedMix|Mix" lib/Runtime/Compiler.cpp lib/Runtime/Executor.cpp tools/validator/validator_main.cpp
```

- [ ] **Step 2: Fix only proven shared runtime leakage**

If the regression matrix shows non-mix artifacts are incorrectly consumed by mix runtime behavior, repair the shared boundary so that:
- plain `.bin` stays on the plain execution route
- mix execution only activates when the artifact and ABI actually require it

Do not add sample-name conditions.

- [ ] **Step 3: Rebuild runtime tools and rerun the five examples on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build --target compiler validator -j4'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash -lc "
source examples/env.sh
for ex in \
add-broadcast-concat \
broadcast-add-reduce \
gather-elementwise-fusion \
relu-broadcast-transpose \
split-relu-brc-add-mul; do
  echo ===== \$ex
  bash examples/\$ex/run.sh --log || true
done
"'
```

Expected:
- examples either recover or fail for reasons now isolated away from runtime leakage

- [ ] **Step 4: Commit runtime-side separation fixes**

```bash
git add lib/Runtime/Compiler.cpp lib/Runtime/Executor.cpp tools/validator/validator_main.cpp docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md
git commit -m "Restore non-mix runtime execution boundaries"
```

### Task 4: Resolve Shared Environment Breakages Separately

**Files:**
- Modify: `docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md`
- Modify: `examples/env.sh`

- [ ] **Step 1: Classify environment-only failures explicitly**

Examples include:
- missing `numpy`
- missing `libascend_hal.so`
- wrong `PATH`
- wrong `LD_LIBRARY_PATH`

Document these in the results matrix rather than misclassifying them as codegen regressions.

- [ ] **Step 2: Fix only common environment setup when the matrix proves a shared setup regression**

If the matrix shows that multiple examples are blocked by the same setup regression, repair [`examples/env.sh`](/Volumes/GM9/code/Ascend-MLIR/examples/env.sh) as the shared setup entry point.

The fix must be:
- common across examples
- independent of sample name
- limited to restoring the prior environment contract

- [ ] **Step 3: Re-run only the environment-blocked examples**

Run the previously blocked examples again on xvm and update the matrix.

- [ ] **Step 4: Commit environment-contract fixes**

```bash
git add examples/env.sh docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md
git commit -m "Restore common example environment setup"
```

### Task 5: Perform Final End-to-End Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md`

- [ ] **Step 1: Run the five non-mix examples on xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash -lc "
source examples/env.sh
for ex in \
add-broadcast-concat \
broadcast-add-reduce \
gather-elementwise-fusion \
relu-broadcast-transpose \
split-relu-brc-add-mul; do
  echo ===== \$ex
  bash examples/\$ex/run.sh --log > /tmp/\$ex.final.log 2>&1 || true
  tail -n 40 /tmp/\$ex.final.log
done
"'
```

Expected:
- each example now has a final classified outcome

- [ ] **Step 2: Run the mix regression check**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && bash -lc "source examples/env.sh && bash examples/matmul-add-leakyrelu/run.sh --log"'
```

Expected:
- `max_abs_diff=0`
- `mean_abs_diff=0`
- `PASS`

- [ ] **Step 3: Finalize the results matrix**

For each example, fill:
- last good stage
- first failing stage
- compile/runtime/accuracy status
- final root cause
- whether the issue was fixed or remains environment-only

- [ ] **Step 4: Commit the final verification record**

```bash
git add docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md
git commit -m "Record non-mix pipeline regression recovery results"
```

### Task 6: Close Out with a Mechanism-Level Review

**Files:**
- Modify: `docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md`

- [ ] **Step 1: Review all functional fix commits for forbidden patterns**

Check that no accepted code change:
- branches on example directory names
- branches on known sample kernel symbols
- rewrites generated source only for one sample

- [ ] **Step 2: Add a short closing section to the results document**

Append:

```md
## Recovery Summary

- fixed shared translator routing: yes/no
- fixed shared runtime routing: yes/no
- fixed shared environment contract: yes/no
- sample-specific patches introduced: no
- mix regression check preserved: yes/no
```

- [ ] **Step 3: Commit the close-out review**

```bash
git add docs/superpowers/plans/2026-04-07-non-mix-pipeline-regression-results.md
git commit -m "Close out non-mix pipeline regression recovery review"
```
