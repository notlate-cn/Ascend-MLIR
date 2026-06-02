# Dynamic-shape (from-torch export) compilation — handoff

Date: 2026-06-02
Branch: `develop`
Status: ✅ **RESOLVED — sim + real 910C** — a dynamic `?` seq dim now compiles +
runs end-to-end through `network_runner` on both camodel and real NPU (device 7).
See "Resolution" below; the original diagnosis (kept for context) is under "The
gap chain".

## Resolution (2026-06-02, commits `11cc77b0`, `74da4d26`, `d8bbad9f`)

Implemented the "Correct fix route" plus the predicted NEXT gaps. Repro
`gelu(x+b)` on `tensor<1x?x3072xf32>` → `network.output[0] max_diff=9.537e-07
PASS` (static twin identical → no regression; full lit 110/111, the 1 fail is
the pre-existing missing-example-script `example-pipelines.mlir`). Four layers:

1. **Eliminate the index dim-arg** (`GroupOutlinePass.cpp:eliminateDynamicDimArgs`,
   run after the outline loop, before `emitFiles`). Replaces each `index` kernel
   arg's uses with `tensor.dim %inputArg, %dynDim`, drops the arg + the matching
   coordinator call operand, re-stamps `auto_fuse.call_arg_index`. This is the
   route this doc proposed; `collectShapeDerivedFields` then turns the in-kernel
   `tensor.dim` into a `dim_arg<A>_<D>` ShapeDerived field as for hand-written
   dynamic kernels — so the phase-2 CCE `ptr*dim`/uint32 errors vanish.
2. **Host alloc for dynamic `tensor.empty`** (`AclnnBackend.cpp`): was bailing to
   an uninitialized `TensorInfo` (→ SIGSEGV). Now resolves each dynamic dim from
   its `tensor.dim` size operand → `inputs[N].shape[D]`, byte size at runtime.
3. **Runtime ShapeDerived tiling resolution** (`TilingSchema` + `HostLaunchHelper`):
   the `dim_arg0_0` param's tilings-JSON value is the `-1` placeholder; packing it
   into TilingData made the kernel loop on a garbage extent (camodel hang). Parse
   the schema `shape_key` ("arg<N>_dim<D>") and resolve the value from
   `inputs[N].shape[D]` at launch.
4. **Host codegen for dynamic reshape** (`AclnnBackend.cpp:emitReshapeView`): was
   aliasing the source on any dynamic collapse/expand (→ output rank wrong,
   `(1,3072)` not `(1,64,3072)`). Now expand reads its explicit `output_shape`
   operand; collapse multiplies the grouped source dims.

**Real-NPU validated** (2026-06-02, device 7, `examples/gelu-dyn-e2e` with
`BACKEND=npu`, seq=48): `network.output[0] max_diff=2.384e-07 PASS`. The
`[npu-launch]` trace shows `tiling.word[2]=48` — the ShapeDerived seq extent
resolved from the input npy shape at launch (not the `-1` placeholder), seq=48
≠ the GPT-2 static 64 → confirms runtime resolution on hardware. block_dim=192,
no hang. Committed example `examples/gelu-dyn-e2e/` (model.mlir + numpy-only
gen_inputs.py + run.sh; `SEQ=N` selectable) is the regression gate.

Remaining: full dynamic **GPT-2** additionally needs the causal mask generated
in-graph (`triu`) instead of a fixed `[64,64]` buffer. Multi-symbol kernels (>1
distinct torch `Dim`) use the "first input dynamic dim" shortcut — fine for
single-`Dim` exports, revisit if a model needs two independent dynamic axes.

---

## Original diagnosis (pre-resolution, kept for context)

Status (at handoff time): non-deterministic `tensor.dim` bug FIXED + committed;
full dynamic-shape compile still blocked, root cause diagnosed, fix route below.

## Goal

Compile a **dynamic-shape** kernel exported from torch (a `?` seq dim) through
`network_runner` and run it on sim/NPU — e.g. GPT-2 with a variable sequence
length instead of the static fixed-window `seq=64` we shipped. Motivation: avoid
recompiling per window size; step toward KV-cache-style variable lengths.

## TL;DR

- The pipeline **has** symbolic-shape machinery (`--afir-symbolize-shapes`,
  `afir.dim_symbols`, `ShapeDerived` schema fields, `axis_extent_expr`), built
  for **hand-written** dynamic kernels.
- The **from-torch** path produces an extra artifact the machinery doesn't
  expect: an explicit `tensor.dim` whose result is **plumbed as an `index`
  call-arg** (e.g. `tensor.empty(%dimArg)` for a `?xK` output).
- I fixed the first two gaps (committed). The remaining blocker is that this
  explicit `index` dim-arg cannot be *patched* — after bufferization the index
  arg is gone (the dim moved into `TilingData` / the input shape) and the
  codegen resolves the dim symbol to the **stale arg position = a GM pointer**.
  The dim-arg must be **eliminated** (rewritten to derive from the input shape),
  not annotated. That's a real rewrite-pass + coordinator change.

## Minimal repro

`/tmp/dyn/gelu_dyn.mlir` (regenerate with the snippet below): a `bias-add +
erf-GELU` on `tensor<1x?x3072xf32>` (seq dim dynamic). Inputs/expected in
`/tmp/dyn/{x,b,exp}.npy`. Run:

```bash
# torch export with a dynamic dim (produces tensor<1x?x3072xf32>):
python3 - <<'PY'
import sys, torch, torch.nn as nn
sys.path.insert(0,'python/torch'); from torch2linalg import torch_to_linalg
class M(nn.Module):
    def forward(self, x, b): return nn.functional.gelu(x + b)
S = torch.export.Dim('S', min=2, max=256)
txt = torch_to_linalg(M().eval(), [torch.randn(1,64,3072), torch.randn(3072)],
                      {'x': {1: S}, 'b': {}})
open('/tmp/dyn/gelu_dyn.mlir','w').write(txt)
PY

source examples/env.sh; export PATH=$PWD/build/bin:$PATH
NETWORK_RUNNER_SKIP_AUTOTUNE=1 PYTHONPATH=python python python/network_runner.py \
  --input-linalg /tmp/dyn/gelu_dyn.mlir --inputs /tmp/dyn/x.npy /tmp/dyn/b.npy \
  --expected /tmp/dyn/exp.npy --workdir /tmp/dyn_work --soc Ascend910B1 \
  --backend sim --atol 1e-2 --rtol 1e-2
```

## The gap chain (what I found, in order)

### 1. ✅ FIXED — phase-1 was missing `--afir-symbolize-shapes`
The `?` dim had no symbol, so the outliner choked. Added the pass in
`network_runner.py` phase-1, **right before outlining** (after the
fold/fuse-transpose/canonicalize rewrites — they create fresh ops and DROP the
`afir.symbolic_shapes` attrs, so running it earlier is useless). No-op for
static input. Committed `1e735d03`.

### 2. ✅ FIXED — `emitNetworkJson` flaky "unsupported op: tensor.dim"
The outliner **non-deterministically** leaves `tensor.dim` in the coordinator
vs sinks it into the kernel. `NetworkJsonEmitter.cpp`'s coordinator-body loop had
no `tensor.dim` case → flaky fail (same input passed once, then failed 3×).
Fix (committed `1e735d03`): handle `tensor.dim` (record an `"input_dim"`
provenance `{from, name, dim}`) + skip coordinator-resident `arith.constant`.
Now deterministic: 5/5 outline runs OK, phase-1 passes.

### 3. ⚠️ partial — phase-2 schema verifier rejects the `index` dim-arg
`VerifyTilingInfoSchemaPass` check 3: every `TileParam` arg needs a `Tunable`
field. The kernel's `%argN: index` (the dynamic dim value, `TilePlanGen.cpp:138`
classifies *any* index arg as `TileParam`) has no tunable field → error
`tile_param ... has no matching tunable field`.

**I tried a contained fix** (reverted — see below): in `TilePlanGen` bind the
index arg to a new `ShapeDerived` field (`sourceArg`/`sourceDim` = the first
input with a dynamic dim, plus `argIndex` = the holding arg), relax the verifier
to accept `Tunable OR ShapeDerived`, round-trip `arg_index` for ShapeDerived in
`TilePlanSchema.cpp`. This got phase-2 to **generate the kernel `.cpp`** (the
`TilingData` even gained a `dimval_argN` field) — but see gap 4.

### 4. 🔴 BLOCKER — codegen resolves the dim symbol to a STALE arg (a pointer)
`runtime-session` (CCE compile of the generated `kernel_group0.cpp`) fails:
```
error: invalid operands to binary expression ('__gm__ uint8_t *' and 'uint32_t')   // ptr * dim
error: cast from pointer to smaller type 'uint32_t' loses information               // 48-bit GM addr truncated
```
Root cause: the generated code uses `v4` (a `GM_ADDR` param) where it means the
dynamic dim. The dim symbol resolved to **mlir arg position 3 = the pre-bufferize
index arg**, but **bufferization removed that index arg** and moved the dim into
`TilingData` (`v7.dimval_argN`) and/or the input shape. So binding by
`argIndex=3` is invalid post-bufferize. **This is why gap-3's "annotate the
index arg" approach is fundamentally wrong** — proven, then reverted.

## Correct fix route (the real work)

**Eliminate the explicit `index` dim-arg** so the dynamic dim is derived from the
input tensor's shape (the stable `memref.dim %inputArg, %dim` mechanism that
`collectShapeDerivedFields` / `dim_arg<A>_<D>` already supports for hand-written
dynamic kernels). Concretely a new rewrite (a pass on the outlined module, run
between `--auto-fuse-group-outline` and `emitNetworkJson` so coordinator + kernel
+ network.json stay consistent):

1. For each kernel taking an `index` arg used as a dynamic extent (e.g. in
   `tensor.empty(%argN)`): resolve which `(inputArg, dim)` it represents via the
   using op's `afir.symbolic_shapes` + the func's `afir.dim_symbols`
   (`symshape::DimSymbolTable::fromAttr`). For elementwise all dynamic dims are
   equal (shape_equalities), so "first input dynamic dim" is a safe shortcut.
2. Replace the index arg's uses in the kernel body with
   `tensor.dim %inputArg, %dimConst`.
3. Drop the now-dead index arg from the kernel signature AND the corresponding
   operand from the coordinator's `call`. Drop the dead coordinator `tensor.dim`.
4. Re-emit network.json (the `input_dim` arg goes away; the runner resolves the
   dynamic extent from the input npy shape, same as hand-written dynamic kernels).

After this, expect possible NEXT gaps (not yet reached): dynamic-dim used in
pointer/offset arithmetic must stay int64 (the `uint32_t` truncation above hints
the codegen narrows extents); `axis_extent_expr` for the dynamic axis must
resolve from the input shape at runtime; and full dynamic **GPT-2** additionally
needs the causal mask generated in-graph (`triu`) instead of a fixed `[64,64]`
buffer.

## Files touched / relevant

- COMMITTED (`1e735d03`): `python/network_runner.py` (phase-1 symbolize step),
  `lib/Conversion/AutoFuse/GroupOutline/NetworkJsonEmitter.cpp` (tensor.dim case).
- The reverted contained attempt (gap 3) was in `TilePlanGen.cpp:138`,
  `VerifyTilingInfoSchemaPass.cpp` (check 3), `TilePlanSchema.cpp`
  (serialize/deserialize `arg_index` for ShapeDerived) — DON'T resurrect as-is;
  it's the wrong approach (gap 4). Kept only as a reference for what passes the
  verifier.
- Symbolic-shape machinery to reuse: `lib/Dialect/AFIR/Transforms/AFIRSymbolizeShapes.cpp`,
  `TilePlanGen.cpp:collectShapeDerivedFields` (`dim_arg<A>_<D>` ShapeDerived
  fields from `memref.dim`/`tensor.dim`), `Analysis/SymbolicShape/{SymExpr,DimSymbolTable}`.
- Design context: the shape-symbolization design doc
  `docs/.../2026-05-12-mlir-shape-symbolization-design.md`; this `auto-resolve-shapes`
  bridge is the proposed-but-unimplemented item noted there and in memory
  `project_shape_symbolization`.

## Where it sits relative to the rest

The static GPT-2 path is **fully working on real 910C** (forward `max_diff=0.004`,
live generation "The capital of France is the capital of the French Republic,
and"). Dynamic shape is a separate, optional enhancement — see memory
`project_gpt2_real_npu` for the full GPT-2 bring-up.
