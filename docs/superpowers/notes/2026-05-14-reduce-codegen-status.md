# Reduce path through the network runner — status report

Date: 2026-05-14
Branch: `llm-net`
Author: scout subagent (paired with Part 1: `examples/reduce-elewise-e2e/`)

## TL;DR

Bug catalogue + current status (2026-05-14 EOD):

| ID | Issue | Status |
|---|---|---|
| **R1** | Full-reduce-to-scalar: `--vector-plan-codegen` produces invalid IR | **Single-block FIXED** (`e3cf5a6` / `f6bacb5` / `6e59a19`); multi-block (R > autotuner's max XBLOCK ≈ 256) still wrong — partial→combine dual-kernel split pending **P3b-3**, currently blocked on the in-flight multi-plan retention framework. Empirical boundary in `2026-05-14-p3b-2-rcore-single-block-boundary.md`. |
| **R2** | CANN `ReduceSum<half, RA>` static_asserts at C++ compile | **FIXED** (`abe4536`): RA half input now upcasts to float, ReduceSum<float, RA>, downcasts result back to half. |
| **R3** | Reduce kernel writes init in-place; host alloc'd fresh output → all-zero | **FIXED** (`28c8ea6`): new `vector-plan-isolate-kernel-outputs` pass wraps the DPS init with `bufferization.alloc_tensor() copy(%init)` so the result bufferizes to a fresh GM buffer distinct from any input. |
| **R4** | `block_dim_expr` empty for static-shape funcs (known) | Partially addressed by C2 (`dd4593c`) — bare extent expression now emitted; verify whether `block_dim_expr` itself is still empty under static shapes. |
| **R5** | Greedy fuser merges past Rule 3 with `linalg.fill` | Open. Not on critical path; affects only the workaround that tried to replace function-arg inits with `tensor.empty + linalg.fill`. |

- **Group outline correctly splits two reductions on different axes** into two separate
  `kernel_group*.mlir` files (Rule 3: merged canonical axes lose every parallel axis).
- Part 1 demo `examples/reduce-elewise-e2e/` (referenced below) was a probe that
  pre-dated R3's fix; with R3 fixed, it is no longer needed.

## Bug catalogue

### R1 — `--vector-plan-codegen` produces invalid IR for full-reduce-to-scalar

**Status: SINGLE-BLOCK FIXED (commits `e3cf5a6` / `f6bacb5` / `6e59a19`, 2026-05-14);
MULTI-BLOCK PENDING P3b-3.** AF port's RCore template now selectable for true
full-reduce (no parallel axes); GroupEmitter emits R-as-block-axis with per-block
inner R loop + cross-space writeback via `bufferization.materialize_in_destination`.
**Correct only when block_dim = 1** (autotune picks XBLOCK ≥ R); for larger R every
block writes its partial to the same GM scalar → race. Empirical boundary documented
in `2026-05-14-p3b-2-rcore-single-block-boundary.md`. Production fix (partial→combine
dual kernels) is plan `2026-05-14-p3b-rcore-reduce-multicore.zh.md` §5 (P3b-3) and is
blocked on the in-flight multi-plan retention framework.

**Symptom**

```
error: 'func.return' op must be the last operation in the parent block
    return %0 : tensor<f16>
```

**Reproducer** (rank-1 input, full reduce → 0-D tensor):

```mlir
#mapId = affine_map<(d0) -> (d0)>
#mapScalar = affine_map<(d0) -> ()>
func.func private @kernel_group0(%x: tensor<16xf16>, %init: tensor<f16>) -> tensor<f16> {
  %0 = linalg.generic {
    indexing_maps = [#mapId, #mapScalar],
    iterator_types = ["reduction"]
  } ins(%x : tensor<16xf16>) outs(%init : tensor<f16>) {
  ^bb0(%a: f16, %acc: f16):
    %s = arith.addf %acc, %a : f16
    linalg.yield %s : f16
  } -> tensor<f16>
  return %0 : tensor<f16>
}
```

Same failure for 2D → 0-D (`reduction, reduction` iterators, scalar output).

**Phase**: 2 (`--vector-plan-codegen`).

**Root-cause guess**: the codegen lowering inserts ops AFTER `func.return` instead of
before it when the output tensor is rank 0. Likely in `lib/Conversion/VectorPlan/`
the rewriter materializes a writeback after the terminator. (Not localized to a
specific file:line — needs an MLIR-debug pass to confirm.)

**Workaround for Part 1**: avoid full reductions; keep at least one parallel axis
(rank ≥ 1 output).

---

### R2 — CANN `ReduceSum` API does not support `f16` for the `RA` pattern

**Status: FIXED (commit `abe4536`, 2026-05-14)** — `CannTranslation.cpp`'s
`ReduceSum2DL2Op` RA template now detects half input and emits an upcast
sequence: allocate VECCALC TBufs for `srcf: tensor<R*A × f32>` and
`dstf: tensor<A × f32>`, `Cast<float, half>(srcf, src)`,
`ReduceSum<float, Pattern::Reduce::RA>(dstf, srcf, ws, srcShape, false)`,
`Cast<half, float>(dst, dstf, RoundMode::CAST_RINT)`, with PIPE_V barriers
between Cast and ReduceSum. Workspace is sized in the op (float) type.
The non-half (float) path is unchanged.

Verified: axis-0 fp16 reduce reproducer (8x16 fp16 → 16 fp16) is now
`max_diff=0.002 PASS` (fp16↔fp32 quantization noise). The original
analysis below is preserved for context.


**Symptom** (compile-time error during `runtime-session`):

```
reduce_sum_v220_impl.h:287:5: error: static assertion failed due to requirement
'SupportType<half, float>()': failed to check the data type, current api supports
data type is float!
    static_assert(SupportType<T, float>(), ...);
```

**Reproducer**: 2D reduce, axis 0 (the "RA" / row-reduce-then-axis pattern), `f16`:

```mlir
#map2   = affine_map<(d0, d1) -> (d0, d1)>
#map_d1 = affine_map<(d0, d1) -> (d1)>
func.func private @k(%x: tensor<8x16xf16>, %i: tensor<16xf16>) -> tensor<16xf16> {
  %0 = linalg.generic {
    indexing_maps = [#map2, #map_d1],
    iterator_types = ["reduction", "parallel"]
  } ins(%x : tensor<8x16xf16>) outs(%i : tensor<16xf16>) {
  ^bb0(%a: f16, %acc: f16):
    %s = arith.addf %acc, %a : f16
    linalg.yield %s : f16
  } -> tensor<16xf16>
  return %0 : tensor<16xf16>
}
```

Axis-1 (`AR` pattern) on `f16` compiles fine; only the `RA` lowering hits the
static-assert.

**Phase**: 2 (`runtime-session` invokes the CANN cross-compiler on the emitted
`.cpp`).

**Root-cause guess**: vector-plan-codegen emits `AscendC::ReduceSum<T, Pattern::RA>`
with `T = half`, but the CANN tikcfw shipped in 9.0.0 only instantiates `RA` for
`float`. Either the codegen needs an upcast (load `half` → cast to `float` →
ReduceSum → cast back) for `RA`, or a lit-test gate.

**Workaround for Part 1**: change dtype to `f32`. (Done.)

---

### R3 — Reduce kernels bufferize in-place on the DPS init; `aclnn-backend` assumes a separate output → host launch passes wrong buffer to the kernel

**Status: FIXED (commit 28c8ea6, 2026-05-14)** — new pass
`vector-plan-isolate-kernel-outputs` runs after LinalgElementwiseOpFusion
in vector-plan-codegen. For each returned linalg result whose DPS init
traces back to a func BlockArgument, the init is wrapped:
`outs(bufferization.alloc_tensor() copy(%init))`. Bufferize then gives
the result a fresh GM buffer, init is read-only, the memref ABI matches
the tensor-level call signature, host buffer routing aligns. The original
analysis below is preserved for context.

**Symptom**: phase 3/5 produces all-zero outputs. No error, no hang, just a wrong-result
PASS/FAIL at the very end (`max_diff ≈ |sum(x)|`).

**Reproducer** (single reduce, dropped into the network runner):

```mlir
#map2   = affine_map<(d0, d1) -> (d0, d1)>
#map_d0 = affine_map<(d0, d1) -> (d0)>
func.func @single_red(%x: tensor<8x16xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
  %r = linalg.generic {
    indexing_maps = [#map2, #map_d0],
    iterator_types = ["parallel", "reduction"]
  } ins(%x : tensor<8x16xf32>) outs(%init : tensor<8xf32>) {
  ^bb0(%a: f32, %acc: f32):
    %s = arith.addf %acc, %a : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %r : tensor<8xf32>
}
```

```bash
network_runner.py --input-linalg single_red.mlir \
  --inputs x.npy init.npy --expected exp.npy --workdir w
# -> phase 5: max_diff=0.82  FAIL  (out is all zeros; expected = sum(x, axis=1))
```

**Phase**: 3, 5 (host-launch ABI mismatch).

**Root cause** (localized):

After `--vector-plan-codegen` + `-mlir-to-cann`, the CanonicalizeCannSignaturePass
(`lib/Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.cpp:100-155`)
sets `cann.num_inputs` based on the number of "real-IO" memref args. For our reduce
the lowered signature is

```
@kernel_group0(memref<8x16xf32>, memref<8xf32>, memref<ui8>, !struct)
                ^arg0=input        ^arg1=init/output reused in-place
                                                       ^workspace ^tiling
```

with `cann.num_inputs = 1`. So the kernel ABI is **1 input + 1 output**.

The outline pass emits `network.json` listing every coordinator-function arg as a
network input — including `%init`. The aclnn-backend host generator
(`network_host.cpp`) then synthesizes:

```cpp
TensorInfo t0[2] = {inputs[0], inputs[1]};   // (x, init)  -- treated as 2 inputs
TensorInfo t1[1] = {};                        // freshly-allocated output
hostLaunchAscendCKernel("kernel_group0", ..., t0, /*nIn=*/2, t1, /*nOut=*/1);
outputs[0] = t1[0];                           // <-- the freshly-allocated buffer
```

so it passes 2 inputs + 1 output = 3 buffers to a kernel whose ABI expects only 2.
The kernel writes into `arg1` (the `init` GM buffer), but the harness reads from
`t1[0]` (the freshly-allocated, never-written buffer) → all zeros.

**Why mixed-attn-e2e doesn't hit this**: its elementwise `kernel_group0` bufferizes
to `cann.num_inputs = 4` because vector-plan adds a separate strided output memref
(arg4) on top of `init_pre` (arg3). The reduce path does NOT add that extra output
arg; it writes back through the init memref directly. So mixed-attn's host-passed
`t1[0]` IS the buffer the kernel writes to; the reduce path's `t1[0]` is not.

Likely fix locations (do not patch in this branch per scope):
- `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp` — needs a way to
  classify coordinator-function args as input-vs-init-output (or the kernel must
  declare which of its args are written to so the JSON / host-gen can route them).
- OR the reduce vector-plan codegen should mirror elewise: allocate a fresh output
  buffer (strided) and write to it, leaving init untouched.

**Workaround for Part 1**: none found that doesn't require code changes. Both
alternatives explored failed:
- `arith.constant dense<0.0>` for the init: `NetworkJsonEmitter` rejects with
  `unsupported op in coordinator body: arith.constant`.
- `tensor.empty + linalg.fill` for the init: the fill ops change the fusion order
  and the two reduces are merged back into a single kernel by the greedy fuser
  (Rule 3 evaluated pairwise rather than transitively against the merged group).

So Part 1 ships as a "probe" demo: the group-outline split works (visible at the end
of phase 1: `kernel_group0.mlir`, `kernel_group1.mlir`); phases 2–5 run to completion
without errors or hangs but produce wrong outputs. README documents this. The example
exits non-zero.

---

### R4 — `block_dim_expr` is empty for static-shape funcs (known)

Re-confirmed for both reduce kernels in Part 1: `_block_dim` defaults to 1, so a
single core does all the work. Not a hang for shapes ≤ a few hundred elements; would
swamp performance for anything realistic. Already in the project memory note.

---

### R5 — Greedy fuser merges past Rule 3 when the group contains a `linalg.fill` of lower rank

When I tried to replace the function-arg inits with `tensor.empty + linalg.fill`,
the two axis-different reduces (which the bare model correctly splits into 2 kernels
under Rule 3) were merged back into a single `kernel_group0`. The fill ops are
rank-1 with iter=`[parallel]`, the reduces are rank-2; the per-axis lattice across
all 4 members SHOULD still come out reduction-on-both-axes (axis 0 = par|par|par|red,
axis 1 = absent|absent|red|par → red/red, no parallel). Either the greedy fuser
evaluates merges pairwise without re-checking the merged group, or `linalg.fill` is
treated specially and skipped.

Not a correctness bug per se (the merged kernel is still computable in principle),
but it removes the only easy lever for forcing a 2-kernel split with a "clean"
network.json (one without a function-arg init). Not investigated further.

---

## Coverage matrix

| Scenario                                          | Phase 1 outline | Phase 2 codegen+compile | Phase 3 default-build+dump | Phase 5 verify |
|---------------------------------------------------|:---------------:|:-----------------------:|:--------------------------:|:--------------:|
| Single reduce, axis=last, fp32, 2D                | OK (1 kernel)   | OK                      | runs, output all 0         | FAIL (R3)      |
| Single reduce, axis=last, fp16, 2D                | OK              | OK                      | runs                       | FAIL (R3)      |
| Single reduce, axis=0, fp16, 2D                   | OK              | FAIL (R2 static_assert) | —                          | —              |
| Single reduce, axis=0, fp32, 2D                   | OK              | OK                      | runs, output all 0         | FAIL (R3)      |
| Two reduces, different axes, fp32 (Part 1)        | OK (2 kernels)  | OK                      | runs, output all 0         | FAIL (R3)      |
| Reduce, full (rank-1 → 0-D)                       | n/a             | FAIL (R1 invalid IR)    | —                          | —              |
| Reduce, full (rank-2 → 0-D)                       | n/a             | FAIL (R1 invalid IR)    | —                          | —              |
| Reduce + elementwise on reduce result (would-fuse)| —               | not probed              | —                          | —              |

`linalg.reduce` (named op) was not separately probed in this pass. The greedy fuser
classifies it via `isa<linalg::ReduceOp>` (CanFuse.cpp:225) and `linalg.generic`
with a reduction iterator the same way (lines 230–237), so they should land in the
same group; correctness through codegen is likely the same as for `linalg.generic`.
Worth a follow-up.

## What works

- Group outline correctly splits axis-incompatible reductions (Rule 3).
- `--vector-plan-codegen` correctly handles 2D `reduce-axis-1` AND `reduce-axis-0`
  (no IR errors, valid CANN cpp emitted).
- Both kernels in the Part 1 demo launch on the camodel sim — no hangs, no
  segfaults, no infinite waits. The two-launch-after-aclnn hang from project memory
  is FA-specific; back-to-back AscendC reduce launches behave fine.

## Recommended next steps (out of scope for this branch)

1. Fix R3 first: it's the one that's silently corrupting reduce outputs and would
   block any reduce-bearing demo through the network runner. Either the outline
   pass needs to mark DPS-init operands and the host-gen must route them, or the
   reduce codegen needs to materialize a fresh output buffer like elewise does.
2. After R3 is fixed, re-run Part 1 → it should pass without further model changes.
3. Then look at R1 (full-reduce-to-scalar) — needed for any softmax/mean/var
   pattern that fully reduces an axis to a scalar.
4. R2 (axis-0 fp16 reduce) is a CANN-API limitation; either upcast in codegen or
   document as a hard constraint.
