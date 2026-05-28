# ResNet-18 bring-up — phase-1 fixed, phase-2 conv codegen wall

**Date:** 2026-05-28
**Branch:** `develop`
**Status:** Phase-1 PASS (3 fixes below). Phase-2 walls in `--auto-fuse-codegen` on `linalg.conv_2d_nchw_fchw`.

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

## Phase-2 wall

`--auto-fuse-codegen` on `kernel_group0` (the stem 7×7 conv) aborts:

```
afir-opt: lib/Conversion/AutoFuse/TileFuse/GroupEmitter.cpp:299: Assertion
`allOuts.size() == iterArgs.size() && "allOuts / iterArgs count mismatch"' failed.
```

`kernel_group0` is exactly:
```mlir
func.func @kernel_group0(%out, %in, %weight)
  %0 = linalg.fill ins(%cst : f32) outs(%out : tensor<1x64x112x112xf32>) -> ...
  %1 = linalg.conv_2d_nchw_fchw {dilations=1, strides=2}
       ins(%in, %weight : tensor<1x3x230x230xf32>, tensor<64x3x7x7xf32>)
       outs(%0 : tensor<1x64x112x112xf32>) -> ...
  return %1
```

The TileFuse `GroupEmitter` is the loopnest builder for the kernel body — it expects each tiled body to produce one yielded result per iter_arg. For `linalg.conv_2d_nchw_fchw`, the existing emitter's `allOuts` / `iterArgs` accounting doesn't match.

This isn't a pad problem — it's the **conv kernel codegen path** (Vector / AscendC). BERT never tripped it because BERT has no conv. Three plausible directions:

1. **Route conv to aclnn fallback** (mirroring matmul / transpose / attention / layernorm).
   - `GroupOutlinePass.cpp:241-258` already does this for standalone Matmul / Transpose / Attention / LayerNorm — tag the kernel `aclnn.op="Conv2D"` (or `Convolution`), let aclnn-backend emit `run_Conv2D(...)`.
   - Easiest, cleanest, gets phase-2 unblocked. Cost: a real aclnn `Conv2D` op + perm/stride/dilation/padding plumbing in `run_Conv2D`.
   - Conv is cube-heavy → aclnn is the right home regardless.

2. **Fix GroupEmitter to handle conv**:
   - GroupEmitter assumes elementwise/reduce-style loopnest. Conv has 7 dims (N,C,H,W,F,KH,KW) with non-trivial indexing maps. Substantial work.

3. **Decompose conv into matmul + im2col earlier**:
   - Pre-pass to lower `linalg.conv_2d_nchw_fchw` to `linalg.matmul` over im2col. Matmul → aclnn already works. But the im2col tensor materialization is large (KH*KW*C × N*OH*OW) and would need its own host-side codegen.

**Recommended next step:** Option 1. Mirrors the existing matmul/attention pattern; conv is fundamentally a cube op so aclnn is the natural fallback. Plus host-staging matches what other cube ops do.

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
