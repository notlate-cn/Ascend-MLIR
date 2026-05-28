# ResNet-18 bring-up — phase-1 wall: `tensor.pad` in coordinator body

**Date:** 2026-05-28
**Branch:** `develop` (worktree at `/home/gser/code/Ascend-MLIR`)
**HEAD:** see `git rev-parse HEAD` at handoff time (no fix commits in this session)
**Status:** Front-end lower **PASS**. Phase-1 outline + emitNetworkJson **FAIL** on `tensor.pad`.
**Pipeline:** local CPU sim (`network_runner.py --max-phase 1`, no `--backend`)

---

## What was added

`examples/resnet18-e2e/` — mirror of `examples/bert-e2e/`:

| file | purpose |
|---|---|
| `export_resnet18.py` | `torchvision.models.resnet18(weights=None)` → `torch_to_linalg` → `step0_linalg.mlir` + `input_0.npy` + `expected_0.npy` |
| `env_sibling.sh` | source `examples/env.sh` and point tool paths at `build/bin` |
| `run.sh` | drive export + `network_runner.py --max-phase $1` |

Config: batch=1, 3×224×224, fp32, seed=0, random weights (no ckpt download). Output `[1,1000]` logits.

Repro:
```bash
cd /home/gser/code/Ascend-MLIR
bash examples/resnet18-e2e/run.sh 1 /tmp/resnet18_e2e/r18
```

## What worked

- `torch_to_linalg` produced `step0_linalg.mlir` cleanly.
- `afir-opt` folded constants and grouped into **198 `kernel_groupN` functions** (`kernel_group0`…`kernel_group197`); 46 group `.mlir` files written under `work/groups/` so far before the wall.
- Op coverage observed in groups (sample):
  - `kernel_group0` = `linalg.fill` + `linalg.conv_2d_nchw_fchw` (stride 2, the 7×7 stem)
  - BN folded into `linalg.generic` mul+add (eval-mode); ReLU as `arith.maximumf` generic
  - `linalg.matmul` + bias add (final FC `[1,512] × [512,1000]`)
  - `linalg.pooling_*` for maxpool / adaptive_avg_pool

## The wall

`afir-opt --auto-fuse-group-analysis --auto-fuse-group-outline` exits 1 with:

```
error: emitNetworkJson: unsupported op in coordinator body: tensor.pad
```

(precise message at `lib/Conversion/AutoFuse/GroupOutline/NetworkJsonEmitter.cpp:288-291`)

### Root cause

`torch.export` lowers conv `padding=K` as an **explicit `tensor.pad` ahead of `linalg.conv_2d_nchw_fchw`** instead of pushing pad into the conv op. ResNet-18 has **18 such pad sites** in `work/model_unit_folded.mlir` (one per padded conv). Examples:

```mlir
%9 = tensor.pad %8 low=[0,0,3,3] high=[0,0,3,3] {...}    // 224 → 230, stem 7×7 conv
%150 = tensor.pad %149 low=[0,1,1] high=[0,1,1] {...}    // 14 → 16,  3×3 conv
```

`tensor.pad` lands in the **coordinator (top-level) function body**, not inside the outlined kernel groups. `emitNetworkJson` walks coordinator ops and only accepts a whitelist:
- `tensor.expand_shape` / `tensor.collapse_shape` — propagate provenance (`NetworkJsonEmitter.cpp:79-129`)
- `tensor.empty` — `from=alloc` (lines 131-143)
- `tensor.extract_slice` — `from=slice` (lines 145-171)
- `arith.constant` — `from=const` (lines 173-197)
- `func.call` — kernel invocation (lines 219-284)
- `func.return` (lines 199-217)

Anything else trips the fallback at `NetworkJsonEmitter.cpp:287-291`. BERT never hits this because attention/FFN don't pad.

## Why this matters

The work in groups is exactly what we wanted to exercise — pure `linalg.conv_2d_*` + BN-fold-as-elementwise + matmul + pooling. We can't get past phase-1 to see how `auto-fuse-group-analysis` actually treats conv groups (the real bring-up question) until pad is handled.

## Three options to unblock

### Option 1 — extend `emitNetworkJson` whitelist (smallest LOC)

Treat `tensor.pad` in coordinator body the same way `tensor.expand_shape` / `tensor.extract_slice` are treated: propagate provenance, encode `low` / `high` / pad-value into the source descriptor (`from="pad"`), and let `network_runner` materialize the padded buffer host-side before the next kernel call.

- Edit point: `lib/Conversion/AutoFuse/GroupOutline/NetworkJsonEmitter.cpp` between line 171 (after `ExtractSliceOp`) and 173 (before `ConstantOp`).
- Mirror needed in `python/network_runner.py` to consume the new `from="pad"` provenance — find the dispatcher that handles `from=slice` and add a `pad` sibling.
- Caveat: 18 host-side pad-copy ops on activation tensors at 224² → 230² scale may be measurable; for sim it's fine, for real NPU we'd want to fold pad into conv eventually.
- **Recommended for unblocking phase-1 fast.**

### Option 2 — fold `tensor.pad` into the conv group (correct fix)

Make `GroupAnalysis` recognize `tensor.pad → linalg.conv_2d_*` and outline them as a single kernel_group. Coordinator never sees pad. This is the architecturally right answer (pad is conv-internal padding semantically).

- Edit point: `lib/Conversion/AutoFuse/GroupAnalysis/` — wherever the conv-anchor seeding lives.
- Downstream: phase-2 codegen for the conv kernel needs to know the pad geometry.
- Bigger change but cleaner and avoids host-side pad copies.

### Option 3 — rewrite `tensor.pad + conv` to conv-with-padding pre-grouping

A `torch2linalg` / `afir-opt` early pass that fuses explicit pad into the `linalg.conv_2d_*` op's `pads` attribute (if `linalg.conv_2d_nchw_fchw` supports nonzero pads — needs checking). Pure front-end fix, doesn't touch JsonEmitter or GroupAnalysis.

- Caveat: `linalg.conv_2d_nchw_fchw` in upstream MLIR does **not** carry a `pads` attribute; only strides/dilations. So this likely requires switching to a different conv variant or introducing a custom op. May not be feasible without dialect work.

## Suggested order

1. **Option 1 first** — quickest unblock; lets us discover the next wall (likely conv tiling in phase-2 or phase-3) and accumulate ResNet-18-shape evidence before deciding on the longer-term fix.
2. **Option 2 later** — once we know conv codegen actually works end-to-end, fold pad into the conv group to eliminate host-side pad copies.
3. Option 3 only if the dialect side ends up cheap.

## Artifacts for the next session

```
/tmp/resnet18_e2e/r18/
  step0_linalg.mlir       (~93 MB, full ResNet-18 linalg dump)
  input_0.npy             (1×3×224×224 fp32)
  expected_0.npy          (1×1000 fp32, PyTorch reference)
  work/
    model_unit_folded.mlir   (198 kernel_groupN function decls + coordinator with 18 tensor.pad)
    model_recognized.mlir    (post-recognize-attention; same in this case — no attention)
    manifest.json
    groups/                  (46 outlined .mlir files written before the wall hit)
```

Stderr capture at `/tmp/_r18_err.log` (only first line matters; rest is the dumped input MLIR).

## Things this session decided NOT to do

- **No fix code written** — per session role ([[feedback_session_role_realnpu_runner]]) this session does bring-up + diagnosis + handoff, not compiler fixes.
- **No `tensor.pad` workaround in `export_resnet18.py`** — tempting to `F.pad` the input by hand and feed pre-padded tensors, but it only sidesteps the stem and the inner 3×3 padded convs still need pad. Not worth the noise.
- **No fp16 attempted** — fp32 baseline first, matches BERT bring-up convention.

## Cross-references

- Template followed: `examples/bert-e2e/` (BertLayer bring-up, see [[project_bert_bringup]])
- Session role: [[feedback_session_role_realnpu_runner]]
- Discuss-before-edit: [[feedback_discuss_before_editing]]
