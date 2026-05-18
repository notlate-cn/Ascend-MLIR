# Transformer Runtime Session E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `examples/transformer` from compile/artifact smoke into a runnable runtime-session simulation demo with concrete inputs, golden output, and LIT coverage.

**Architecture:** Keep the existing Normalize -> Kernelize -> Schedule -> Realize -> ComputeLower -> CANN artifact pipeline as the source of truth, then add an explicit runtime E2E mode to the transformer script. The first green path uses deterministic zero-valued model/input tensors so the whole network has a stable all-zero golden while still exercising the generated full graph, ABI, artifact compile, simulator run, output load, and validation path.

**Tech Stack:** Bash example script, Python/NumPy `.npy` generator, `afir-opt`, `afir-translate`, `runtime-session`, LLVM LIT/FileCheck, xvm CANN simulator.

---

### Task 1: Add Runtime E2E RED Test

**Files:**
- Create: `test/tools/examples/transformer-runtime-e2e.mlir`

- [x] **Step 1: Write the failing test**

```mlir
// REQUIRES: ascend_env
// RUN: bash %S/../../../examples/transformer/run-mainline.sh --runtime-e2e --log | FileCheck %s

// CHECK: transformer_dynamic.mainline_prefix=pass
// CHECK: transformer_dynamic.full_codegen=pass
// CHECK: transformer_dynamic.runtime_session=pass
// CHECK: session.result=success
// CHECK: session.validation=pass
// CHECK: transformer_dynamic.validation=pass
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/tools/examples/transformer-runtime-e2e.mlir'
```

Result: FAIL as expected because `run-mainline.sh` emitted the existing compile/artifact markers and did not emit `transformer_dynamic.runtime_session=pass`.

### Task 2: Add Deterministic Transformer Data Generator

**Files:**
- Create: `examples/transformer/gen_data.py`

- [x] **Step 1: Generate all-zero ABI tensors**

Implement a NumPy generator that writes:

```text
input0.npy: [B, S, 128] f32
input1.npy through input12.npy: promoted dense-resource weights/biases, all f32 zeros
output0.npy: [B, S, 128] f32 all zeros
```

The exact shapes are:

```python
WEIGHT_SHAPES = [
    (128,),
    (128,),
    (128,),
    (128, 512),
    (512,),
    (512, 128),
    (128,),
    (128,),
    (128,),
    (128, 128),
    (384,),
    (384, 128),
]
```

The generator must validate `batch >= 1` and `seq >= 1`, then print the tensor directory and shape summary.

- [x] **Step 2: Run generator locally under xvm**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && python3 examples/transformer/gen_data.py --batch 1 --seq 2 --out-dir /tmp/ascend-transformer-npy && ls /tmp/ascend-transformer-npy'
```

Result: xvm NumPy generator produces 13 input bindings, 43 output bindings,
and an all-zero `expected_arg55.npy` golden for `[batch, seq, 128]`.

### Task 3: Extend Transformer Script With Runtime Mode

**Files:**
- Modify: `examples/transformer/run-mainline.sh`

- [x] **Step 1: Add CLI options**

Support:

```text
--runtime-e2e
--batch <positive-int>
--seq <positive-int>
--block-dim <positive-int>
--soc <soc-version>
--log
```

Defaults: `runtime_e2e=false`, `batch=1`, `seq=1`, `block_dim=1`, `soc=${SOC_VERSION:-Ascend910B1}`.

- [x] **Step 2: Keep existing compile smoke unchanged**

When `--runtime-e2e` is absent, the script must still emit the current seven compile/artifact markers and exit after CANN artifact generation.

- [x] **Step 3: Compile runtime artifact when runtime mode is enabled**

Initial plan used:

```bash
runtime-session \
  --kernel "$BUILD_DIR/kernel.cpp" \
  --kernel-kind mix \
  --name transformer_dynamic \
  --cann-mlir "$BUILD_DIR/phase5_cann.mlir" \
  --npy-dir "$NPY_DIR" \
  --soc "$SOC" \
  --output "$ARTIFACT_ROOT"
```

Actual implementation uses `--kernel-kind vec` because the current transformer
kernel is a single 57-argument CANN kernel and the mix runner/tiling helper is
only shaped for the small matmul ABI. The slice now reaches CANN artifact
compilation and fails there, so the failure is kept visible as an evidence-backed
blocker rather than hidden behind the script.

- [ ] **Step 4: Run simulator and validate golden**

Create `run_manifest.json` with 13 inputs, one output, expected `output0.npy`, binary tiling from `artifact/out/tiling.bin`, workspace size `16777216`, profiling enabled, and tolerances `atol=1e-3`, `rtol=1e-3`.

Current blocker: runtime-session cannot yet compile the generated transformer
kernel. The first remaining CANN compile failures include unsupported
GlobalTensor->GlobalTensor `DataCopy`, residual GM pointer views, scalar
`AscendC::Exp(value)`, and double scalar casts inside the generated aicore
function. After runtime-session completes, print:

```text
transformer_dynamic.runtime_session=pass
transformer_dynamic.validation=pass
```

### Task 4: Wire The Demo Into Tracking

**Files:**
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [x] **Step 1: Add the plan reference**

Append this plan under the existing implementation-plan list:

```markdown
- Transformer runtime-session E2E 计划：`docs/superpowers/plans/2026-05-18-transformer-runtime-session-e2e.md`
```

- [x] **Step 2: Update transformer status truthfully**

Runtime E2E is recorded as blocked at CANN artifact compilation, after
Normalize -> Kernelize -> Schedule -> Realize -> ComputeLower -> Phase5 backend
-> translate/runtime-artifact smoke has passed.

### Task 5: Verify The Slice

**Files:**
- Test: `test/tools/examples/transformer-runtime-e2e.mlir`
- Test: `test/tools/examples/example-pipelines.mlir` only if transformer is added to the full suite.

- [x] **Step 1: Focused build and test**

Run:

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 afir-opt afir-translate runtime-session && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/tools/examples/transformer-runtime-e2e.mlir'
```

Result: focused transformer runtime E2E is marked `XFAIL` and records the
precise CANN compile blocker described above.

- [ ] **Step 2: Static guards**

Run:

```bash
git diff --check
./test/tools/check_ascend_no_v2_code_naming.sh
./test/tools/check_ascend_public_headers.sh
```

Expected: all pass.

- [ ] **Step 3: Commit only on explicit request**

This task changes demo/test/docs. Do not commit automatically unless the user asks for commit.
