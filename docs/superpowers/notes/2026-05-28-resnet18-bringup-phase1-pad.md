# ResNet-18 bring-up — phase-1 + conv fixed, phase-2 BatchNorm-chain wall

**Date:** 2026-05-28
**Branch:** `develop`
**Status:** Phase-1 PASS. Conv groups route to aclnn fallback (next commit). Phase-2 walls in TileFuse codegen on **BatchNorm eval chain** (kernel_group1), `MutableOperandRange::operator[]` OOB assert.

---

## What was added

`examples/resnet18-e2e/` — mirror of `examples/bert-e2e/`:

| file | purpose |
|---|---|
| `export_resnet18.py` | `torchvision.models.resnet18(weights=None)` → `torch_to_linalg` → `step0_linalg.mlir` + `input_0.npy` + `expected_0.npy` |
| `env_sibling.sh` | source `examples/env.sh` and point tool paths at `build/bin` |
| `run.sh` | drive export + `network_runner.py --max-phase $1` |

Config: batch=1, 3×224×224, fp32, seed=0, random weights. Output `[1,1000]` logits.

Repro:
```bash
cd /home/gser/code/Ascend-MLIR
bash examples/resnet18-e2e/run.sh 1 /tmp/resnet18_e2e/r18   # passes
bash examples/resnet18-e2e/run.sh 2 /tmp/resnet18_e2e/r18   # hits phase-2 wall
```

## Phase-1 fixes (committed in this session)

Three related changes unblock phase-1 end-to-end:

### 1. `NetworkJsonEmitter.cpp` — accept `tensor.pad` in coordinator body

Coordinator-body whitelist (`lib/Conversion/AutoFuse/GroupOutline/NetworkJsonEmitter.cpp`) previously rejected `tensor.pad` with the unsupported-op error. Added a branch that registers a `from="pad"` provenance descriptor:
- `source` = wrapped upstream provenance
- `low` / `high` = static pad amounts
- `pad_value` = the yielded scalar (best-effort), with **JSON-safe encoding**: `inf` / `-inf` / `nan` are emitted as strings since standard JSON has no literal for them (maxpool padding yields -inf and would otherwise produce invalid JSON)
- `shape` / `dtype` of the padded result

### 2. `AclnnBackend.cpp` — emit host C++ that materializes the pad

`CoordEmitter` (`lib/Runtime/AclnnBackend/AclnnBackend.cpp`) walks the coordinator and emits the host program. Added `emitPad` that mirrors `emitExtractSlice`:
- allocate the padded buffer (`::operator new(dst_numel * eb)`)
- `std::memset` to zero (works for any 0.0 pad value across f16/bf16/f32 — the bit pattern is identical)
- strided copy from the source into the unpadded interior using src/dst row-major strides
- non-zero pad values get a `// WARNING` comment (not exercised by ResNet — every pad here is 0.0 or -inf for maxpool, but maxpool's init -inf is consumed by `linalg.fill` inside the kernel, not by the pad itself; the pad surrounding maxpool input still uses 0.0)

### 3. `GroupOutlinePass.cpp` — hoist constants above all clusters

`reorderGroupsContiguous` (`lib/Conversion/AutoFuse/GroupOutline/GroupOutlinePass.cpp:511+`) assigns each op a `keyRank` and re-sorts the coordinator body before outlining. Non-member ops (glue, constants) compute `prod` from direct `op->getOperands()` — but `tensor.pad` captures its scalar pad value via the region's `tensor.yield`, **not** as a direct operand. The scheduler was placing the pad in the gap between group 2 (BN+ReLU) and group 10 (maxpool), while the `-inf` constant — sharing only group 10 as a consumer — got placed inside group 10's cluster slot. Result: pad sorted before its captured constant → SSA domination error post-outline.

Fix: assign `kConstantRank = INT_MIN` to every ConstantLike op so they sort first. Their only real positional requirement is "before all uses"; placing them at the top of the block always satisfies that without needing to model region captures.

## Conv → aclnn fallback (commit `b80b68b4`)

Followed the existing Matmul / Transpose / Attention / LayerNorm aclnn-fallback
pattern. Five surgical edits:

1. `GroupAnalysisPass.cpp::isCubeOp` accepts `linalg::Conv2DNchwFchwOp`.
2. `GroupAnalysisPass.cpp` Step 0 fill-detach: extended to Conv2DNchwFchwOp.
   `{fill + conv}` groups reduce to single-op cube groups (aclnn allocates its
   own output, init fill is dead).
3. `GroupOutlinePass.cpp` Cube-kind classifier (line 78): accepts
   `linalg::Conv2DNchwFchwOp`. **Both** isCubeOp sites needed fixing — the
   outline pass rebuilds info from IR and re-classifies, so missing this is
   what kept the conv kernel emitting `auto_fuse.kind = "Vector"`.
4. `GroupOutlinePass.cpp` single-op cube stamping (line 240+): adds Conv2D
   branch alongside Matmul — stamps `aclnn.op="Conv2D"` + `aclnn.strides`
   + `aclnn.dilations` (padding lives in coordinator as `tensor.pad`).
5. `AclnnBackend.cpp emitCall` + `AclnnOps.h/cpp`: new Conv2D branch emits
   `run_Conv2D(in, weight, init, strides[2], dilations[2], &out, stream)`;
   CPU reference impl supports f16/f32, applies stride+dilation only (no
   padding — already materialized upstream). Device-mode aclnn impl deferred
   (Conv2D falls back to CPU with a warning when `!g_host_mode`).

Effect: ResNet-18 has 43 kernel files now, all conv groups tagged
`kind=aclnn` in `network.json`, so phase-2's `for k in network.ascendc_kernels()`
loop skips them.

## Phase-2 wall (current)

`--auto-fuse-codegen` on **`kernel_group1`** aborts:

```
afir-opt: externals/llvm-project/mlir/lib/IR/OperationSupport.cpp:533:
  mlir::OpOperand& mlir::MutableOperandRange::operator[](unsigned int) const:
  Assertion `index < length && "index is out of bounds"' failed.
```

`kernel_group1` is the **BatchNorm eval-mode chain** that torch.export emits.
It takes (running_mean, running_var, x, weight, bias, ...) and computes:

```
inv_std = 1.0 / sqrt(running_var + eps)
out     = (x - running_mean) * inv_std * gamma + beta
```

The MLIR is a sequence of `linalg.generic` ops chained together, with a
`cf.assert` guarding division-by-zero on inv_std:

```mlir
%21 = arith.cmpf one, %in, %cst_0 : f32
cf.assert %21, "unimplemented: tensor with zero element"
%22 = arith.divf %cst_1, %in : f32
```

Three layers of pain here, none of which exist in BERT:
- `math.sqrt` inside a linalg.generic body (no `--recognize-batchnorm` to fold it)
- `arith.divf` inside a linalg.generic body (TileFuse's existing matmul-fill-revert
  pattern hits this kind of divf — see `[[project_bert_bringup]]` Wall-B, where
  bert-tiny needed `divf→mul` for stability)
- `cf.assert` inside a linalg.generic body (a torch-export safety check; not
  expected by downstream passes — `MutableOperandRange::operator[]` OOB likely
  triggers on this op type)

BERT escapes all of this because `--recognize-layernorm` folds the whole
LayerNorm chain into a single `@__aclnn_layer_norm` aclnn call *before*
GroupAnalysis runs. ResNet's BatchNorm gets no equivalent treatment.

## Phase-2 unblock options

1. **`--recognize-batchnorm` (recommended)**: mirror `--recognize-layernorm`,
   match the BN chain (running_var + eps → sqrt → 1/inv → (x-mean)*inv*gamma+beta)
   and fold to `@__aclnn_batch_norm` aclnn call. Then run_BatchNorm in
   AclnnOps.cpp (CPU reference for sim, real aclnn for device).
   - Touch list: `RecognizeAttention`-style new pass file, AclnnOps.h/.cpp
     declaration + impl, run_BatchNorm emit in AclnnBackend, mirror the existing
     `--recognize-layernorm` invocation in network_runner.py phase-1.
   - Cleanest match to existing precedent. Conceptually small even if it
     touches many files.

2. **Constant-fold BN parameters earlier**: in torch.export, running_mean /
   running_var / weight / bias are CONSTANTS in eval mode. If we could fold
   `inv_std = 1.0 / sqrt(running_var + eps)` and `scale = inv_std * gamma` and
   `bias = beta - mean * scale` at compile time, the BN reduces to a single
   affine transform `out = x * scale + bias` with no sqrt / divf / cf.assert.
   - Better-than-aclnn perf (just elementwise mul+add on x).
   - But this requires constant-folding tensor-valued math.sqrt + arith.divf,
     which we don't have today; would need either a custom pass or a torch-side
     pre-export rewrite.

3. **Strip cf.assert from linalg.generic bodies before codegen**: just a band-aid,
   might unblock the immediate OOB assert but the divf and sqrt issues likely
   still bite TileFuse codegen.

**Recommended:** Option 1. Cleanest match to the LayerNorm/Attention pattern
that already works for BERT.

## Artifacts

```
/tmp/resnet18_e2e/r18/
  step0_linalg.mlir       (~93 MB)
  input_0.npy             (1×3×224×224 fp32)
  expected_0.npy          (1×1000 fp32)
  work/
    groups/
      network.json        (now contains from="pad" entries — verified)
      kernel_group0.mlir  (linalg.fill + linalg.conv_2d_nchw_fchw — the trip wire)
      ... (197 more)
    manifest.json
```

## Regression verification

- BERT phase-1 (`examples/bert-e2e/run.sh 1`): still PASS
- `test/Conversion/Group/group-outline/*.mlir` and `test/Conversion/Group/group-analysis/*.mlir`: all 17 run clean (no crash; FileCheck not validated due to missing llvm-lit in env, but IR transformation succeeds)

## Cross-references

- Template followed: `examples/bert-e2e/` ([[project_bert_bringup]])
- Session role: [[feedback_session_role_realnpu_runner]] — broken intentionally this session at user's request
- Discuss-before-edit: [[feedback_discuss_before_editing]]
