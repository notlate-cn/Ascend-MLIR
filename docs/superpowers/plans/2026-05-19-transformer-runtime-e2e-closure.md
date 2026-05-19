# Transformer Runtime E2E Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `examples/transformer/run-mainline.sh --runtime-e2e` finish with `transformer_dynamic.validation=pass` on xvm simulator for bounded transformer shapes.

**Architecture:** Close the full-network gap by moving from black-box full-graph runs to a monotonic ladder: diagnostics, attention softmax, attention context, composed attention block, then full transformer E2E. Each task has a RED command, a focused implementation slice, and a GREEN command before moving wider.

**Tech Stack:** MLIR Linalg/Func/SCF/EmitAsc/AscendC dialects, CANN 9.1 xvm simulator, `afir-opt`, `afir-translate`, `runtime-session`, LLVM LIT, NumPy golden generation.

---

## File Map

- `examples/transformer/run-mainline.sh`: Full transformer pipeline driver; add bounded runtime diagnostics and make the runtime compile/run mode match generated kernel artifacts instead of hiding long-run failures.
- `examples/transformer/gen_data.py`: Deterministic full transformer input/output bindings; update only if ABI or final output golden contract changes.
- `test/tools/examples/transformer-runtime-e2e.mlir`: Full transformer E2E gate; starts as XFAIL and becomes PASS only after full validation closes.
- `examples/transformer-fragments/*.mlir`: Attention fragment ladder; add softmax, context, and attention block fragments.
- `examples/transformer-fragments/gen_data.py`: NumPy data/golden generator for new fragments.
- `examples/transformer-fragments/run-mainline.sh`: Fragment driver; add new fragments and keep `--runtime-e2e` stable.
- `test/tools/examples/transformer-fragments.mlir`: Focused fragment LIT gate.
- `lib/Conversion/...`: Only touch if a fragment exposes a lowering/classification gap.
- `lib/Target/CannKernel/CannTranslation.cpp`: Extend CANN emission only when a fragment proves a missing supported shape/type.
- `lib/Runtime/Mix/*`: Extend mix tiling/ABI only when a fragment proves a runtime metadata gap.
- `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`: Record each closed blocker and verification command.

## Task 1: Full Transformer Diagnostic Gate

**Files:**
- Modify: `examples/transformer/run-mainline.sh`
- Modify: `test/tools/examples/transformer-runtime-e2e.mlir`
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`

- [x] **Step 1: Run the current bounded RED**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && timeout 180s bash examples/transformer/run-mainline.sh --runtime-e2e --batch 1 --seq 1 --log'
```

Expected before fixes: reaches `transformer_dynamic.full_codegen=pass` and artifact compile, then either times out, lacks `session.validation=pass`, or records the first runtime failure in `examples/transformer/build_mainline/runtime_session.log`.

- [x] **Step 2: Add explicit runtime stage markers**

In `examples/transformer/run-mainline.sh`, print these markers around runtime compile/run:

```bash
echo "transformer_dynamic.artifact_compile=start"
# runtime-session compile command
echo "transformer_dynamic.artifact_compile=pass"
echo "transformer_dynamic.runtime_session=start"
# runtime-session run command
```

Keep existing final markers:

```bash
echo "transformer_dynamic.runtime_session=pass"
echo "transformer_dynamic.validation=pass"
```

- [x] **Step 3: Add bounded run timeout diagnostics**

Wrap the runtime run in a script-level timeout variable:

```bash
RUN_TIMEOUT="${RUN_TIMEOUT:-600s}"
if ! timeout "$RUN_TIMEOUT" "$RUNTIME_SESSION" \
    --run-manifest "$RUN_MANIFEST" \
    --run >"$VALIDATION_LOG" 2>&1; then
  echo "transformer_dynamic.runtime_session=timeout_or_fail"
  tail -n 200 "$VALIDATION_LOG" || true
  exit 1
fi
```

- [ ] **Step 4: Verify diagnostics do not regress normal smoke**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && bash examples/transformer/run-mainline.sh --batch 1 --seq 1 --log'
```

Expected: full codegen/backend/translate/runtime artifact smoke markers still pass.

- [x] **Step 5: Keep the LIT gate bounded**

Update `test/tools/examples/transformer-runtime-e2e.mlir` to use the new timeout env:

```mlir
// RUN: RUN_TIMEOUT=180s timeout 240s bash %S/../../../examples/transformer/run-mainline.sh --runtime-e2e --batch 1 --seq 1 --log | FileCheck %s
```

Leave `XFAIL: *` until Task 5 closes.

## Task 2: Attention Softmax Fragment

**Files:**
- Create: `examples/transformer-fragments/attn_softmax.mlir`
- Modify: `examples/transformer-fragments/gen_data.py`
- Modify: `examples/transformer-fragments/run-mainline.sh`
- Modify: `test/tools/examples/transformer-fragments.mlir`
- Modify implementation files only if RED exposes a compiler/runtime gap.

- [x] **Step 1: Add a RED softmax fragment gate**

Add a LIT line:

```mlir
// RUN: timeout 180s bash %S/../../../examples/transformer-fragments/run-mainline.sh --fragment attn_softmax --batch 1 --seq 16 --runtime-e2e --log | FileCheck --check-prefix=ATTNSOFTMAX %s
```

Expected checks:

```mlir
// ATTNSOFTMAX: transformer_fragment.attn_softmax.full_codegen=pass
// ATTNSOFTMAX: transformer_fragment.attn_softmax.phase5_translate=pass
// ATTNSOFTMAX: transformer_fragment.attn_softmax.artifact_compile=pass
// ATTNSOFTMAX: transformer_fragment.attn_softmax.runtime_session=pass
// ATTNSOFTMAX: transformer_fragment.attn_softmax.validation=pass
```

- [x] **Step 2: Create `attn_softmax.mlir` from the full-transformer softmax shape**

Use the same semantic shape as the full graph:

```mlir
func.func @kernel(%score: tensor<?x4x?x?xf32>) -> tensor<?x4x?x?xf32>
```

Include row max reduction, subtract, `math.exp`, row sum reduction, and divide over the last axis. If the imported form produces an auxiliary int64 argmax output, keep it as an output binding but do not require a golden for it.

- [x] **Step 3: Add NumPy golden**

In `examples/transformer-fragments/gen_data.py`, generate:

```python
score = rng.normal(0.0, 0.2, size=(args.batch, 4, args.seq, args.seq)).astype(np.float32)
shifted = score - score.max(axis=-1, keepdims=True)
exp = np.exp(shifted).astype(np.float32)
expected = exp / exp.sum(axis=-1, keepdims=True)
```

Set `atol=1.0e-2`, `rtol=1.0e-2` initially because simulator scalar math and vector math can differ slightly.

- [x] **Step 4: Implement only the failing compiler/runtime gap**

If RED fails in compute lowering, inspect `build_attn_softmax/step5_ascendc.mlir` and add the missing rank4 vector/reduction support in the existing generic lowering path. If RED fails in runtime validation only, inspect `runtime_session.log` and first fix ABI/output binding, not math code.

- [x] **Step 5: GREEN**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && timeout 180s bash examples/transformer-fragments/run-mainline.sh --fragment attn_softmax --batch 1 --seq 16 --runtime-e2e --log'
```

Expected: `transformer_fragment.attn_softmax.validation=pass`.

## Task 3: Attention Context Fragment

**Files:**
- Create: `examples/transformer-fragments/attn_context.mlir`
- Modify: `examples/transformer-fragments/gen_data.py`
- Modify: `examples/transformer-fragments/run-mainline.sh`
- Modify: `test/tools/examples/transformer-fragments.mlir`
- Likely modify: `lib/Target/CannKernel/CannTranslation.cpp`
- Likely modify: `lib/Runtime/Mix/MixTilingGenerator.cpp`

- [x] **Step 1: Add RED context gate**

Add LIT:

```mlir
// RUN: timeout 180s bash %S/../../../examples/transformer-fragments/run-mainline.sh --fragment attn_context --batch 1 --seq 16 --runtime-e2e --log | FileCheck --check-prefix=ATTNCONTEXT %s
```

Expected checks mirror `ATTNSOFTMAX`.

- [x] **Step 2: Create context fragment**

Use full-transformer context shape:

```mlir
func.func @kernel(
    %prob: tensor<?x?x?xf32>,
    %value: tensor<?x?x32xf32>) -> tensor<?x?x32xf32>
```

The body is:

```mlir
%init = tensor.empty(%batch_heads, %seq, %head_dim) : tensor<?x?x32xf32>
%context = linalg.batch_matmul
  ins(%prob, %value : tensor<?x?x?xf32>, tensor<?x?x32xf32>)
  outs(%init : tensor<?x?x32xf32>) -> tensor<?x?x32xf32>
return %context : tensor<?x?x32xf32>
```

- [x] **Step 3: Generate golden**

```python
prob = rng.random(size=(args.batch * 4, args.seq, args.seq)).astype(np.float32)
prob = prob / prob.sum(axis=-1, keepdims=True)
value = rng.normal(0.0, 0.2, size=(args.batch * 4, args.seq, 32)).astype(np.float32)
expected = np.matmul(prob, value).astype(np.float32)
```

- [x] **Step 4: Close the expected f32 batch-matmul gap**

If CANN translator rejects this because the current batch special shell requires f16 inputs, generalize the stable batch-matmul emission to choose `half` or `float` from memref element types:

```cpp
static StringRef getAscendCScalarType(Type type) {
  if (type.isF16()) return "half";
  if (type.isF32()) return "float";
  return "";
}
```

Then emit:

```cpp
MatmulType<TPosition::GM, CubeFormat::ND, float>
```

for f32 inputs/outputs, and keep f16 behavior unchanged for `attn_score`.

Observed closure: the fragment does not need the mix shell yet. Current f32
rank3 `batch_matmul` lowers through the existing GM scalar-loop fallback and
passes runtime-session validation for `--batch 1 --seq 16`; the script keeps
`attn_context` on the non-mix artifact path to avoid a mix wrapper ABI mismatch.

- [x] **Step 5: GREEN**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && timeout 180s bash examples/transformer-fragments/run-mainline.sh --fragment attn_context --batch 1 --seq 16 --runtime-e2e --log'
```

Expected: `transformer_fragment.attn_context.validation=pass`.

## Task 4: Composed Attention Block Fragment

**Files:**
- Create: `examples/transformer-fragments/attention_block.mlir`
- Modify: `examples/transformer-fragments/gen_data.py`
- Modify: `examples/transformer-fragments/run-mainline.sh`
- Modify: `test/tools/examples/transformer-fragments.mlir`
- Modify implementation files only for gaps proven by this composed fragment.

- [x] **Step 1: Add RED attention block gate**

```mlir
// RUN: timeout 240s bash %S/../../../examples/transformer-fragments/run-mainline.sh --fragment attention_block --batch 1 --seq 16 --runtime-e2e --log | FileCheck --check-prefix=ATTNBLOCK %s
```

Expected markers:

```mlir
// ATTNBLOCK: transformer_fragment.attention_block.full_codegen=pass
// ATTNBLOCK: transformer_fragment.attention_block.phase5_translate=pass
// ATTNBLOCK: transformer_fragment.attention_block.artifact_compile=pass
// ATTNBLOCK: transformer_fragment.attention_block.runtime_session=pass
// ATTNBLOCK: transformer_fragment.attention_block.validation=pass
```

- [x] **Step 2: Compose the block**

The fragment should cover:

```text
score = q @ key
scaled = score * 0.17677669529663687
prob = softmax(scaled)
context = prob @ value
out = reshape/transpose context to [seq, 1, 4, 32] or [batch, seq, 128]
```

Use the same reshape/collapse/transpose pattern as `examples/transformer/transformer_dynamic.mlir` around `%82` through `%121`.

- [x] **Step 3: Generate golden**

```python
score = np.matmul(q.astype(np.float32), key.astype(np.float32))
scaled = score * np.float32(0.17677669529663687)
shifted = scaled - scaled.max(axis=-1, keepdims=True)
prob = np.exp(shifted).astype(np.float32)
prob = prob / prob.sum(axis=-1, keepdims=True)
context = np.matmul(prob, value.astype(np.float32)).astype(np.float32)
expected = context.reshape(args.batch, 4, args.seq, 32).transpose(2, 0, 1, 3)
```

- [x] **Step 4: Fix only composed-boundary gaps**

Expected possible gaps are selected movement across view chains, rank4 reduction output ABI, or multi-kernel manifest ordering. Do not rewrite the full transformer driver here; make the fragment pass first.

Observed closure: the composed block exposed two generic gaps. First, the
single-kernel run manifest must bind all eight ABI outputs (`score`, `scaled`,
`row_max`, `exp_shifted`, `row_sum`, `prob`, `context`, `out`) while validating
only the final `out`. Second, GM scalar-loop `linalg.transpose` lowering must
use inverse permutation for input indices; rank2 swap had not caught this
because it is self-inverse. The fix is in the generic transpose fallback, not in
the fragment IR.

- [x] **Step 5: GREEN**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && timeout 240s bash examples/transformer-fragments/run-mainline.sh --fragment attention_block --batch 1 --seq 16 --runtime-e2e --log'
```

Expected: `transformer_fragment.attention_block.validation=pass`.

## Task 5: Full Transformer E2E Closure

**Files:**
- Modify: `examples/transformer/run-mainline.sh`
- Modify: `examples/transformer/gen_data.py`
- Modify: `test/tools/examples/transformer-runtime-e2e.mlir`
- Modify: `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
- Modify compiler/runtime files only for failures reproduced by full E2E after Tasks 2-4.

- [ ] **Step 1: Run full E2E with long enough timeout**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && RUN_TIMEOUT=900s timeout 960s bash examples/transformer/run-mainline.sh --runtime-e2e --batch 1 --seq 1 --log'
```

Expected after Tasks 2-4: either `transformer_dynamic.validation=pass` or a specific failure in `runtime_session.log`.

- [ ] **Step 2: If validation fails, compare final output contract**

Inspect:

```bash
python3 - <<'PY'
import numpy as np
from pathlib import Path
base = Path("examples/transformer/build_mainline")
print(np.load(base / "outputs" / "arg55.npy").shape)
print(np.load(base / "npy" / "expected_arg55.npy").shape)
PY
```

Fix `examples/transformer/gen_data.py` only if final output shape/name is wrong. Do not loosen validation to hide numeric mismatch.

- [ ] **Step 3: Convert XFAIL to PASS**

When the command prints:

```text
session.result=success
session.validation=pass
transformer_dynamic.validation=pass
```

remove `// XFAIL: *` from `test/tools/examples/transformer-runtime-e2e.mlir` and set timeout to a realistic bounded value:

```mlir
// RUN: RUN_TIMEOUT=900s timeout 960s bash %S/../../../examples/transformer/run-mainline.sh --runtime-e2e --batch 1 --seq 1 --log | FileCheck %s
```

- [ ] **Step 4: Run focused full transformer LIT**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/tools/examples/transformer-runtime-e2e.mlir'
```

Expected: 1/1 passed.

- [ ] **Step 5: Stability run**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && for i in 1 2 3; do RUN_TIMEOUT=900s timeout 960s bash examples/transformer/run-mainline.sh --runtime-e2e --batch 1 --seq 1 --log | tee /tmp/transformer-e2e-$i.log; grep -q "transformer_dynamic.validation=pass" /tmp/transformer-e2e-$i.log; done'
```

Expected: all three runs pass.

- [ ] **Step 6: Final verification**

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && ninja -C build -j6 afir-opt afir-translate runtime-session'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build/test/tools/examples/transformer-fragments.mlir build/test/tools/examples/transformer-runtime-e2e.mlir'
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && source examples/env.sh && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -q build/test/tools/examples'
git diff --check -- . ':!AGENTS.md'
./test/tools/check_ascend_no_v2_code_naming.sh
./test/tools/check_ascend_public_headers.sh
```

Expected: every command exits 0.

## Execution Notes

- Keep `AGENTS.md` out of scope unless explicitly requested.
- Do not make upper layers shape-customized for rank2/rank3. Shape/type specifics may live in backend/API handling and tests.
- If a full E2E failure can be reproduced by a smaller fragment, fix the smaller fragment first and only then rerun full transformer.
- Commit only after a task has fresh GREEN evidence. If the user asks for one final commit, keep intermediate commits local or squash before push.
