# Bug Report: Multi-input dynamic-shape reduce produces wrong numerical output

**Reporter:** dyn-bucketed-e2e example (this directory)
**Branch / Commit:** `llm-net` @ `a479284` (HEAD before this report)
**Date:** 2026-05-14
**Status:** **FIXED (commit `b3d196e`, 2026-05-14).** Root cause in
`AFIRSymbolizeShapes::transfer` for `bufferization.alloc_tensor` — it
only consulted `dynamicSizes`, which is required to be empty when
`copy` is set (per the op verifier; that's exactly the drive-by fix in
§5).  With empty dynamicSizes and a dynamic result type, the symbolic
transfer failed, propagating `nullopt` to the downstream
linalg.generic, dropping `afir.iter_extents`, then Collapse's
`afir.axis_extents`, then TilePlanGen's `afir.axis_extent_expr`.  The
empty `axis_extent_expr` in `<kernel>__v0_space.json` left
network_runner phase 3 unable to cap XBLOCK candidates against the
runtime extent, so default tilings picked the largest value
(XBLOCK_SUB = 256 against an actual `d0*d1 = 16`-element parallel
axis).  In the kernel's overlap-tail block, the offset computation
`extent - XBLOCK_SUB` underflowed `uint32_t` to a huge value, the
writes landed at out-of-bounds GM addresses, and the host's freshly-
allocated output buffer stayed zero.

Fix: in `AFIRSymbolizeShapes::transfer` for `AllocTensorOp`, when
`copy` is set, take the source's symbolic ShapeVec as the result's.
No-op for the no-copy case.

Verified: `bash examples/dyn-bucketed-e2e/run.sh` →
`max_diff = 1.2e-07 / 2.4e-07 PASS`.

**Follow-up (2026-05-15):** at R ≥ 256 a second, distinct failure mode
appeared — same all-zero output, but this time *not* from underflowed
GM offsets.  Root cause: the phase-3 default picker chose the largest
extent-capped tunable (`XBLOCK_SUB = 128`) without any check against
UB capacity.  On 910B1 the TBuf pool is 184 KB
(`__NPU_ARCH__ 2201, TOTAL_VEC_LOCAL_SIZE`), but
`align32(XBLOCK_SUB * dim_arg0_2 * 4)` at R = 512 is 256 KB for a
single buffer — the TPipe bump-pointer allocator silently overflows
(ASCENDC_DEBUG_ASSERT only logs).

Fix landed in this series (llm-net):
  - `377b99d` SocSpec table for SoC UB capacity
  - `f687bad` UbCostExpr — lift `ascendc.pipe.init_buffer` size operand
              to a SymExpr keyed on TilingData field names
  - `892d63d` CannTranslation emits `ub_budget_bytes` (188416) and
              `ub_cost_bytes_expr` (= `2 * max(align32(size))`) into
              every `<kid>__v<i>_space.json`
  - `0ec9e47` network_runner picker reads them and filters tunable
              candidates whose evaluated cost exceeds the budget
  - `b4ac1c6` autotuner mirrors the same prune in its candidate loop

Picker math validated against the empirical sweep (see commit
messages): R=64 still picks XBLOCK_SUB=128; R=256 drops to 64;
R=512 drops to 32 — all within budget.

---

## Summary

A linalg kernel of the form
```text
out[d0,d1] = init[d0,d1] + sum_{d2}( elementwise(in0, in1, ..., inK)[d0,d1,d2] )
```
with all dimensions dynamic (`?x?x?xf32` inputs, `?x?xf32` output) and ≥ 4 input
operands feeding a parallel-then-reduction body **runs to completion on the
910B1 simulator but writes numerically wrong values to the output buffer**.
The failure is independent of tiling (XBLOCK / XBLOCK_SUB), so it is not the
known "XBLOCK > total elements → zero output" pattern from
[[network_runner_v1]].

The kernel itself is correct in IR (verifier-clean, `vector-plan-codegen` lowers
to AscendC without complaint). The numerical mismatch is consistent
(`max_abs_diff = 7.19`, identical across all 25 autotuned configs at d0=d1=4),
suggesting a deterministic codegen miscompile rather than scheduling
nondeterminism.

A second pre-existing bug was discovered along the way and fixed (see §5).

---

## 1. Reproduction

```bash
cd /home/gser/code/Ascend-MLIR
bash examples/dyn-bucketed-e2e/run.sh
# → network.output[0]: max_diff=2.891  FAIL
# → network.output[1]: max_diff=1.819  FAIL
```

Source: `examples/dyn-bucketed-e2e/model.mlir`
Driver: `examples/dyn-bucketed-e2e/run.sh` (uses `python/network_runner.py`)
Inputs: `examples/dyn-bucketed-e2e/gen_inputs.py` (defaults d0=2, d1=4, d2=16,
all-zero init tensors)

The minimal failing kernel (post-fold-unit-extent-dims, written to
`build_e2e/groups/kernel_group0.mlir`):
```mlir
func.func private @kernel_group0(
    %arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>,
    %arg2: tensor<?x?x?xf32>, %arg3: tensor<?x?x?xf32>,
    %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
      indexing_maps = [#map3, #map3, #map3, #map3, #map2],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3 : ...) outs(%arg4 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
    %1 = arith.addf %in, %in_0 : f32   // a + b
    %2 = arith.mulf %1, %in_1 : f32    // *c
    %3 = arith.addf %2, %in_2 : f32    // +d
    %4 = arith.addf %out, %3 : f32     // accumulate
    linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
```

## 2. Evidence the kernel is the culprit (not the host driver)

`python/network_runner.py` dumps the actual NPU input/output tensors per call
under `build_e2e/intermediates_default/`. With default shape d0=2, d1=4, d2=16:

| Tensor                       | Wiring check                    |
| ---------------------------- | ------------------------------- |
| `kernel_group0__v0_in_0.npy` | == `a.npy` ✓                    |
| `kernel_group0__v0_in_1.npy` | == `b.npy` ✓                    |
| `kernel_group0__v0_in_2.npy` | == `c.npy` ✓                    |
| `kernel_group0__v0_in_3.npy` | == `d.npy` ✓                    |
| `kernel_group0__v0_in_4.npy` | shape (2,4), all-zero (init0) ✓ |
| `kernel_group0__v0_out_0.npy`| shape (2,4), **all-zero** ✗     |

i.e. host wires inputs correctly; the kernel reads them and writes zeros.

Reference value `expected0 = sum_d2((a+b)*c+d) ≈ [-1.04, 1.18, …, 3.51]`.

## 3. Search-space interaction (not the root cause, but confounds debugging)

`tilings_default.json` candidates have `XBLOCK_SUB ∈ {16,32,64,128,256}`.
At d0=d1=2,4 the parallel work axis size is `d0*d1 = 8 < 16`, so in
`build_e2e/kernel_group0_lowered.mlir` line 68–74:

```mlir
%50 = arith.divsi %49, %8 : index   // 8 / XBLOCK_SUB(16) = 0
%51 = arith.muli %50, %8 : index    // 0 * 16 = 0
scf.for %arg8 = %c0 to %51 step %8  // never iterates
```

→ the main loop is skipped; only the tail `scf.if` (line 131+) runs.

Bumping to `--d0 4 --d1 4 --d2 16` (parallel work = 16, matches smallest
XBLOCK_SUB) **all 25 autotune candidates still fail** with a consistent
`max_abs_diff = 7.19`. So the bug reproduces when the main loop DOES execute;
the small-parallel-axis case just compounds it with the
"`d0*d1 < XBLOCK_SUB → zero output`" pattern.

The XBLOCK_SUB search-space pruning issue is tracked separately under
[[network_runner_v1]] memory.

## 4. Autotuner false-pass

`build_e2e/kernel_group0_best.json` reports `passed: true, max_abs_diff: 0`
even though the same kernel writes zeros to GM and fails end-to-end. The
autotuner only scores by cycle count and is not actually comparing against
`expected.npy` (or is comparing against itself). The `max_abs_diff: 0`
field appears to be uninitialized / default. This is a secondary bug; in
isolation it just makes the primary bug harder to spot.

## 5. Drive-by fix: `IsolateKernelOutputs` builds invalid `bufferization.alloc_tensor`

While bringing this example up, `vector-plan-codegen` failed with:
```
error: dynamic sizes not needed when copying a tensor
```

Root cause: `lib/Conversion/VectorPlan/GroupOutline/IsolateKernelOutputs.cpp`
constructed `bufferization::AllocTensorOp` with **both** explicit `dynSizes`
operands **and** a `copy` source — disallowed by the op verifier (dynamic
sizes are inferred from the copy source).

This regressed the (otherwise passing) `examples/combo-elewise-reduce-e2e`
example as well; that one ran the same path and would fail with the identical
verifier error. After the fix below, combo runs through `vector-plan-codegen`
again.

Fix (this commit): pass `ValueRange{}` for `dynSizes` when a copy source is
provided. The diff is one hunk:

```diff
-      builder.setInsertionPoint(defOp);
-      SmallVector<Value> dynSizes;
-      for (auto [i, d] : llvm::enumerate(tensorTy.getShape())) {
-        if (ShapedType::isDynamic(d))
-          dynSizes.push_back(
-              builder.create<tensor::DimOp>(defOp->getLoc(), initVal, i));
-      }
-      Value fresh = builder.create<bufferization::AllocTensorOp>(
-                            defOp->getLoc(), tensorTy, dynSizes,
-                            /*copy=*/initVal)
-                        .getResult();
+      builder.setInsertionPoint(defOp);
+      // When a copy source is provided, AllocTensorOp infers dynamic dim
+      // sizes from the source, and passing them again is a verifier error
+      // ("dynamic sizes not needed when copying a tensor").
+      Value fresh = builder.create<bufferization::AllocTensorOp>(
+                            defOp->getLoc(), tensorTy, ValueRange{},
+                            /*copy=*/initVal)
+                        .getResult();
```

File: `lib/Conversion/VectorPlan/GroupOutline/IsolateKernelOutputs.cpp:83-86`

## 6. What's known to work in this example

Despite the numerical bug, the rest of the pipeline behaves correctly:

- `--vector-plan-group-analysis` + `--vector-plan-group-outline` split the
  9-arg model into **two** kernel groups (chain 0 inputs `{a,b,c,d}` + init0;
  chain 1 inputs `{e,f,g}` + init1). Triggered by the H2 horizontal-fuse
  bound (`maxHorizontalExtraInputs=4`).
- `build_e2e/groups/network.json` is valid and emits `kind=ascendc` for both
  kernels (no aclnn fallback).
- `vector-plan-codegen` lowers both kernels to AscendC; `g++` compiles
  `network_host.cpp` and the kernel `.cpp` files; the simulator launches
  successfully (no crash, both `[block_end]` markers appear).
- All inputs are passed as fully-dynamic `?` tensors and resolved at runtime
  via the host's TensorInfo wiring.

This confirms the bug is localized to compute / DPS handling inside the
multi-input dynamic-shape reduction kernel itself, not anywhere in the
fusion / outline / host-driver chain.

## 7. Suggested next steps for whoever picks this up

1. Compare the lowered IR of this kernel against `combo-elewise-reduce-e2e`'s
   reduce kernel (which has the same shape but only 2 ins). Diff at the
   `ascendc.reduce_sum_2d_l2 / ascendc.add_l2` accumulator level
   (`kernel_group0_lowered.mlir:107–116`).
2. Verify that the `%out` (init) operand read is being routed to the correct
   `tbuf<vecout>` accumulator rather than being dropped or aliased to an
   `ascendc.duplicate_l2 %_, %cst` zero-fill (line 107).
3. Cross-check `vector_plan.tiling_infos` (line 1) only emits dim fields for
   `arg0` and `arg4`; whether `arg1..arg3` are assumed to share `arg0`'s
   dims via `afir.dim_symbols` and whether that holds at the lowering
   level for the reduce body.
4. Fix the autotuner so `passed`/`max_abs_diff` reflect real comparison to
   `--expected`; otherwise this class of miscompile will keep slipping
   through.

## Appendix: relevant artifact paths

```
examples/dyn-bucketed-e2e/
├── model.mlir                                   # source IR
├── gen_inputs.py                                # input generator
├── run.sh                                       # e2e driver
└── build_e2e/
    ├── model_unit_folded.mlir                   # after fold-unit-extent-dims
    ├── _outlined_combined.mlir                  # after group-outline (all funcs)
    ├── groups/
    │   ├── kernel_group0.mlir                   # outlined chain-0 kernel
    │   ├── kernel_group1.mlir                   # outlined chain-1 kernel
    │   ├── network.mlir                         # coordinator only
    │   └── network.json                         # call graph + tensor descs
    ├── kernel_group0_lowered.mlir               # after vector-plan-codegen
    ├── kernel_group0.cpp                        # generated AscendC C++
    ├── kernel_group0_best.json                  # autotuner result (false pass)
    ├── network_host.cpp                         # host wiring (HostLaunchHelper)
    ├── intermediates_default/                   # per-kernel I/O dumps
    └── outputs/                                 # final network output (mismatched)
```
