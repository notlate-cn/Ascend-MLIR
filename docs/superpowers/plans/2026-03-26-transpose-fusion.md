# Transpose Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `relu-broadcast-transpose` example with a new pipeline that uses named linalg ops in step0, fuses them into a single generic, and generalizes `ComputeConversion.cpp` to handle transpose semantics embedded in indexing maps.

**Architecture:** Named ops (`linalg.elemwise_unary` → `linalg.transpose` → `linalg.broadcast` → `linalg.add`) are generalized then fused into one `linalg.generic` with map `(d0,d1)->(d1,0)` for the transposed-broadcast input. `ComputeConversion.cpp` gains a new `analyzeIndexingMap()` function that classifies this map and emits `data_copy_l2` + `broadcast_l2` + `ascendc.transpose`. The fallback `isTransposeGeneric` is generalized to N dimensions.

**Tech Stack:** MLIR (linalg, affine, transform dialects), C++17, AscendC PyAsc dialect, lit/FileCheck testing

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `examples/relu-broadcast-transpose/step0_input.mlir` | Rewrite | New computation graph with named linalg ops |
| `examples/relu-broadcast-transpose/step0_input_out.mlir` | Rewrite | Fused generic IR (expected output after generalize+fuse) |
| `examples/relu-broadcast-transpose/step2_transform.mlir` | Rewrite | Single-generic tiling (no library_call matching) |
| `examples/relu-broadcast-transpose/step1_fused.mlir` | Update | Reflects new fused output |
| `examples/relu-broadcast-transpose/step[3-8]*.mlir` | Update | Regenerated from pipeline run |
| `examples/relu-broadcast-transpose/run.sh` | Modify | Add generalize+fuse step between stage 0 and 1 |
| `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp` | Modify | Add `analyzeIndexingMap()`, generalize `isTransposeGeneric` |

---

## Task 1: Write step0_input.mlir with named linalg ops

**Files:**
- Rewrite: `examples/relu-broadcast-transpose/step0_input.mlir`

**Context:** Current file has 3 `linalg.generic` ops with `library_call` tags. New file uses `linalg.elemwise_unary`, `linalg.transpose`, `linalg.broadcast`, `linalg.add` — the output of StableHLO/torch-MLIR lowering.

Computation: `data0[m,1] → relu → transpose[1,0] → broadcast dim[0] → add ← data1[n,m]`

- [ ] **Step 1: Write the new step0_input.mlir**

```mlir
// ============================================================
// STAGE 0: Named Linalg Op Source IR
//
// Computation graph:
//   data0[m,1] -> relu -> transpose[1,0] -> broadcast dim[0]
//                                              |
//                         data1[n,m] -------> add -> out[n,m]
//
// Named ops match StableHLO/torch-MLIR lowering output.
// Three fusion scenarios covered:
//   Elementwise+Transpose: relu -> transpose
//   Transpose+Broadcast:   transpose -> broadcast
//   Transpose+Elementwise: broadcast -> add (with data1)
//
// RUN: afir-opt --linalg-generalize-named-ops \
// RUN:          --linalg-fuse-elementwise-ops %s \
// RUN:          --canonicalize --cse | FileCheck %s
// CHECK: linalg.generic
// CHECK: affine_map<(d0, d1) -> (d1, 0)>
// CHECK: affine_map<(d0, d1) -> (d0, d1)>
// ============================================================

module {
  func.func @relu_transpose_broadcast_add(
      %data0: tensor<?x1xf16>,
      %data1: tensor<?x?xf16>) -> tensor<?x?xf16> {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %data0, %c0 : tensor<?x1xf16>
    %n = tensor.dim %data1, %c0 : tensor<?x?xf16>

    // relu: max(x, 0)
    %relu_init = tensor.empty(%m, %c1) : tensor<?x1xf16>
    %relu = linalg.elemwise_unary
        ins(%data0 : tensor<?x1xf16>)
        outs(%relu_init : tensor<?x1xf16>) -> tensor<?x1xf16>

    // transpose [1,0]: [m,1] -> [1,m]
    // tensor<1x?xf16> has one dynamic dim at position 1, so tensor.empty takes %m only
    %tr_init = tensor.empty(%m) : tensor<1x?xf16>
    %tr = linalg.transpose
        ins(%relu : tensor<?x1xf16>)
        outs(%tr_init : tensor<1x?xf16>)
        permutation = [1, 0]

    // broadcast dim[0]: [1,m] -> [n,m]
    %bc_init = tensor.empty(%n, %m) : tensor<?x?xf16>
    %bc = linalg.broadcast
        ins(%tr : tensor<1x?xf16>)
        outs(%bc_init : tensor<?x?xf16>)
        dimensions = [0]

    // add: out[n,m] = broadcast_result[n,m] + data1[n,m]
    %add_init = tensor.empty(%n, %m) : tensor<?x?xf16>
    %result = linalg.add
        ins(%bc, %data1 : tensor<?x?xf16>, tensor<?x?xf16>)
        outs(%add_init : tensor<?x?xf16>) -> tensor<?x?xf16>

    return %result : tensor<?x?xf16>
  }
}
```

- [ ] **Step 2: Verify it parses on xvm**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && \
  afir-opt examples/relu-broadcast-transpose/step0_input.mlir"
```
Expected: No errors, module printed.

- [ ] **Step 3: Commit**

```bash
git add examples/relu-broadcast-transpose/step0_input.mlir
git commit -m "feat(relu-broadcast-transpose): rewrite step0 with named linalg ops"
```

---

## Task 2: Write step0_input_out.mlir (expected fused generic)

**Files:**
- Rewrite: `examples/relu-broadcast-transpose/step0_input_out.mlir`

**Context:** This is the expected output after `--linalg-generalize-named-ops --linalg-fuse-elementwise-ops`. Iteration space `[n,m]`. data0 map `(d0,d1)->(d1,0)` encodes: d0(n) absent→broadcast, d1(m) at position 0→transpose.

Note: run.sh is updated in Task 3 to bake these passes in. Task 2 runs the command directly (not via run.sh), so Tasks 2 and 3 are independent and can be done in either order.

- [ ] **Step 1: Run the generalize+fuse pipeline directly to get actual output**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && \
  afir-opt --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
    --canonicalize --cse \
    examples/relu-broadcast-transpose/step0_input.mlir"
```
Capture the actual output — this is the ground truth for `step0_input_out.mlir`.

- [ ] **Step 2: Write step0_input_out.mlir from actual output**

The expected form (write whatever the actual output is):
```mlir
// ============================================================
// STAGE 0-out: Fused Generic IR
//
// After --linalg-generalize-named-ops --linalg-fuse-elementwise-ops
// from step0_input.mlir.
//
// Iteration space [n, m] (d0=n, d1=m).
// data0 map (d0,d1)->(d1,0):
//   - d0 absent: all n-values read same data (broadcast)
//   - d1 at pos 0: m-axis indexes input rows (transpose)
//   - 0 at pos 1: column always 0 (input is [m,1])
// ============================================================
// <actual output goes here>
```

- [ ] **Step 3: Verify FileCheck on step0_input.mlir**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && \
  afir-opt --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
    --canonicalize --cse \
    examples/relu-broadcast-transpose/step0_input.mlir | \
  /home/niu/code/llvm-project/build/bin/FileCheck \
    examples/relu-broadcast-transpose/step0_input.mlir"
```
Expected: FileCheck passes (all CHECK lines satisfied).

- [ ] **Step 4: Commit**

```bash
git add examples/relu-broadcast-transpose/step0_input_out.mlir
git commit -m "feat(relu-broadcast-transpose): add fused generic step0_input_out"
```

---

## Task 3: Update run.sh with generalize+fuse step

**Files:**
- Modify: `examples/relu-broadcast-transpose/run.sh`

**Context:** Current run.sh parses step0_input.mlir directly to step0_input_out.mlir (just `afir-opt` with no transforms). Stage 1 then does `--linalg-fuse-elementwise-ops` on step0_input.mlir. New flow: stage 0 runs `--linalg-generalize-named-ops --linalg-fuse-elementwise-ops` to produce step0_input_out.mlir; stage 1 just copies or is skipped since fusion already happened.

The Transform script (stage 2) will now use `step0_input_out.mlir` as source and match the single fused generic (no `library_call`).

- [ ] **Step 1: Update run.sh stage 0 to generalize+fuse**

Change the stage 0 command from:
```bash
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir"
```
To:
```bash
$AFIR_OPT --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  --canonicalize --cse \
  "$DIR/step0_input.mlir" \
  -o "$DIR/step0_input_out.mlir"
```

Also update the stage 1 command to operate on `step0_input_out.mlir` (already fused, stage 1 is now a no-op canonicalize):
```bash
$AFIR_OPT --canonicalize --cse "$DIR/step0_input_out.mlir" \
  -o "$DIR/step1_fused.mlir"
```

After this change `step1_fused.mlir` is identical to `step0_input_out.mlir` — it serves only as a reference checkpoint in the pipeline. Update comments/log messages to reflect the new computation graph (data0[m,1], data1[n,m]).

- [ ] **Step 2: Verify stage 0 and 1 run cleanly**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && \
  bash examples/relu-broadcast-transpose/run.sh 2>&1 | head -30"
```
Expected: Stage 0 and Stage 1 succeed without error.

- [ ] **Step 3: Commit**

```bash
git add examples/relu-broadcast-transpose/run.sh
git commit -m "feat(relu-broadcast-transpose): add generalize+fuse step to run.sh"
```

---

## Task 4: Rewrite step2_transform.mlir for single fused generic

**Files:**
- Rewrite: `examples/relu-broadcast-transpose/step2_transform.mlir`

**Context:** Current transform script matches 3 ops by `library_call` (relu_bias_add, transpose, scale_mul). After fusion there is exactly 1 `linalg.generic` with no `library_call`. The new script must match it by op type, add 2 index args (TB_N, Tb_N for the n-axis tile sizes), and tile the single op along d0 (n-axis) with two levels (TB/Tb).

The function signature changes: `(%data0: tensor<?x1xf16>, %data1: tensor<?x?xf16>) -> tensor<?x?xf16>`.

The fused generic has iterator space `[n, m]` (d0=n, d1=m). Tiling strategy: tile d0 (n-axis) at TB and Tb levels; leave d1 (m-axis) untiled (processed whole in UB, since data0 is a column vector of size 1 in that axis).

- [ ] **Step 1: Write the new step2_transform.mlir**

```mlir
// ============================================================
// STAGE 2: Transform Dialect Tiling — single fused generic
//
// Input: step0_input_out.mlir (one linalg.generic, no library_call)
// Computation: relu(data0[m,1]) + data1[n,m] (transpose+broadcast fused into map)
//
// Iteration space: [n, m] — d0=n, d1=m
// Tiling strategy: tile d0 (n-axis) at TB and Tb levels; d1 (m-axis) untiled
//   TB-level: inter-core parallelism (one AiCore per TB_N rows)
//   Tb-level: UB batch size (Tb_N rows per UB tile)
//
// Function params appended: TB_N (inter-core block), Tb_N (UB batch)
//
// RUN: afir-opt --transform-interpreter %s --canonicalize --cse | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

#map_data0 = affine_map<(d0, d1) -> (d1, 0)>
#map_identity = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  func.func @relu_transpose_broadcast_add(
      %data0: tensor<?x1xf16>,
      %data1: tensor<?x?xf16>) -> tensor<?x?xf16> {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %data0, %c0 : tensor<?x1xf16>
    %n = tensor.dim %data1, %c0 : tensor<?x?xf16>

    // Single fused generic: relu(data0[m,1]) broadcast+transpose+add with data1[n,m]
    %out_init = tensor.empty(%n, %m) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [#map_data0, #map_identity, #map_identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%data0, %data1 : tensor<?x1xf16>, tensor<?x?xf16>)
      outs(%out_init : tensor<?x?xf16>) {
    ^bb0(%v0: f16, %v1: f16, %vout: f16):
      %cst = arith.constant 0.000000e+00 : f16
      %relu = arith.maximumf %v0, %cst : f16
      %add  = arith.addf %relu, %v1 : f16
      linalg.yield %add : f16
    } -> tensor<?x?xf16>

    return %result : tensor<?x?xf16>
  }

  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    // ---- Step 1: Match func.func, append 2 index params (TB_N, Tb_N) ----
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %func_new, %tb_n_param, %tb_inner_n_param =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // ---- Step 2: Match the single fused linalg.generic ----
    %fused_op = transform.structured.match ops{["linalg.generic"]} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // ---- Common annotation params ----
    %true_param = transform.param.constant true -> !transform.any_param
    %prologue_param = transform.param.constant
        "src:GM->VECIN" -> !transform.any_param
    %epilogue_param = transform.param.constant
        "dst:VECOUT->GM" -> !transform.any_param
    %vector_unit_param = transform.param.constant
        "AiCore.Vector" -> !transform.any_param

    // ══════ Fused generic tiling: d0 (n-axis) at TB and Tb levels ══════
    // d1 (m-axis) untiled: data0 has size 1 in that axis, processed whole in UB

    %op_tb, %loop_tb =
        transform.structured.tile_using_for %fused_op
            tile_sizes [%tb_n_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    %op_inner, %loop_inner =
        transform.structured.tile_using_for %op_tb
            tile_sizes [%tb_inner_n_param, 0]
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.annotate %loop_inner "ascendc.prologue"
        = %prologue_param : !transform.any_op, !transform.any_param
    transform.annotate %loop_inner "ascendc.epilogue"
        = %epilogue_param : !transform.any_op, !transform.any_param
    transform.annotate %op_inner "ascendc.unit"
        = %vector_unit_param : !transform.any_op, !transform.any_param

    transform.loop.hoist_loop_invariant_subsets %loop_inner
        : !transform.any_op

    transform.yield
  }
}
```

- [ ] **Step 2: Test transform tiling on xvm**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && \
  afir-opt --transform-interpreter --canonicalize --cse \
    examples/relu-broadcast-transpose/step2_transform.mlir 2>&1 | head -40"
```
Expected: scf.for loops appear; `ascendc.parallel` attribute on outer loop; `ascendc.prologue/epilogue` on inner loop.

- [ ] **Step 3: Run FileCheck**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && \
  afir-opt --transform-interpreter --canonicalize --cse \
    examples/relu-broadcast-transpose/step2_transform.mlir | \
  /home/niu/code/llvm-project/build/bin/FileCheck \
    examples/relu-broadcast-transpose/step2_transform.mlir"
```
Expected: FileCheck passes.

- [ ] **Step 4: Commit**

```bash
git add examples/relu-broadcast-transpose/step2_transform.mlir
git commit -m "feat(relu-broadcast-transpose): rewrite transform script for single fused generic"
```

---

## Task 5: Run stages 0–4 to get step2_tiled through step4_buffer_placement

**Files:**
- Update: `step2_tiled.mlir`, `step3_bufferized.mlir`, `step4_buffer_placement.mlir`

**Context:** With step0, step2_transform and run.sh updated, run stages 0–4 to regenerate intermediate step files. This verifies the existing passes (bufferize, buffer-placement) handle the new IR without changes.

- [ ] **Step 1: Run stages 0–4**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && \
  bash examples/relu-broadcast-transpose/run.sh 2>&1"
```
Look for success at stage 4. If stage 3 or 4 fails, investigate the error — most likely a bufferize or memory_space inference issue.

- [ ] **Step 2: Check step4_buffer_placement.mlir for expected memory_space annotations**

```bash
ssh xvm@orb "grep -E 'memory_space|: i32' \
  /home/niu/code/Ascend-MLIR/examples/relu-broadcast-transpose/step4_buffer_placement.mlir | head -20"
```
Expected: `data0` subview has `memref<?x1xf16>` (GM, no annotation); output buffers have `9 : i32` (VECIN) or `10 : i32` (VECOUT).

- [ ] **Step 3: Save the generated files**

```bash
# The run.sh already writes these files — just verify they exist
ls examples/relu-broadcast-transpose/step{2_tiled,3_bufferized,4_buffer_placement}.mlir
```

- [ ] **Step 4: Commit**

```bash
git add examples/relu-broadcast-transpose/step{1_fused,2_tiled,3_bufferized,4_buffer_placement}.mlir
git commit -m "feat(relu-broadcast-transpose): update pipeline step1-4 for fused generic"
```

---

## Task 6: Add `analyzeIndexingMap()` to ComputeConversion.cpp

**Files:**
- Modify: `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`

**Context:** This is the core new function. It classifies an input indexing map into 4 cases. Add it as a free helper function near `isBroadcastMap` (around line 195). It will be used in Task 7.

The four cases (in order of precedence):
1. **Identity**: `map.isIdentity()`
2. **Pure broadcast**: all results are `AffineDimExpr`; some iteration dims absent (output rank < iterRank)
3. **Pure transpose**: all results are `AffineDimExpr`; output rank == iterRank; non-identity order
4. **Broadcast+transpose**: one or more results are `AffineConstantExpr` (e.g., `0`); remaining results are dim exprs possibly reordered

- [ ] **Step 1: Add struct definition and function before `isBroadcastMap`**

Add after the anonymous namespace opening (check exact location — it's before line 195 in the current file):

```cpp
// Analysis of a single input indexing map relative to the iteration space.
struct IndexingMapAnalysis {
  enum class Kind {
    Identity,           // (d0,d1)->(d0,d1): direct read
    PureBroadcast,      // (d0,d1)->(d0): some dims absent, no reordering
    PureTranspose,      // (d0,d1)->(d1,d0): all dims present, permuted
    BroadcastTranspose, // (d0,d1)->(d1,0): constants + reordering
  };
  Kind kind;
  SmallVector<int64_t> permutation;    // valid for PureTranspose, BroadcastTranspose
  SmallVector<int64_t> broadcastDims;  // iteration dims absent from output
};

// Analyze an input indexing map to classify how the input is accessed
// relative to the iteration space of rank `iterRank`.
static IndexingMapAnalysis analyzeIndexingMap(AffineMap map,
                                               unsigned iterRank) {
  IndexingMapAnalysis result;

  // Identity: fast path
  if (map.isIdentity()) {
    result.kind = IndexingMapAnalysis::Kind::Identity;
    return result;
  }

  // Collect which iteration dims appear in the map results (as dim exprs)
  // and which results are constants.
  SmallVector<int64_t> presentDims;  // iteration dim positions that appear
  bool hasConstant = false;
  for (AffineExpr expr : map.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      presentDims.push_back(static_cast<int64_t>(dimExpr.getPosition()));
    } else if (isa<AffineConstantExpr>(expr)) {
      hasConstant = true;
    } else {
      // Non-trivial affine expression: not handled.
      result.kind = IndexingMapAnalysis::Kind::Identity; // fallback: treat as identity
      return result;
    }
  }

  // Determine broadcast dims: iteration dims not in presentDims.
  for (unsigned d = 0; d < iterRank; ++d) {
    if (llvm::find(presentDims, static_cast<int64_t>(d)) == presentDims.end())
      result.broadcastDims.push_back(d);
  }

  bool hasBroadcast = !result.broadcastDims.empty() || hasConstant;
  bool hasTranspose = !llvm::is_sorted(presentDims);

  if (hasConstant || (hasBroadcast && hasTranspose)) {
    result.kind = IndexingMapAnalysis::Kind::BroadcastTranspose;
    result.permutation.assign(presentDims.begin(), presentDims.end());
    return result;
  }

  if (hasBroadcast) {
    result.kind = IndexingMapAnalysis::Kind::PureBroadcast;
    return result;
  }

  if (hasTranspose) {
    result.kind = IndexingMapAnalysis::Kind::PureTranspose;
    result.permutation.assign(presentDims.begin(), presentDims.end());
    return result;
  }

  // All dims present in identity order: effectively identity (map.isIdentity()
  // should have caught this, but guard anyway).
  result.kind = IndexingMapAnalysis::Kind::Identity;
  return result;
}
```

- [ ] **Step 2: Build to verify the new code compiles**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && \
  ./scripts/build.sh --build-project 2>&1 | tail -20"
```
Expected: Build succeeds (or only errors unrelated to this change).

- [ ] **Step 3: Commit**

```bash
git add lib/Conversion/LinalgToAscendC/ComputeConversion.cpp
git commit -m "feat(ComputeConversion): add analyzeIndexingMap() with 4-case classification"
```

---

## Task 7: Handle BroadcastTranspose case in the general parallel path

**Files:**
- Modify: `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`

**Context:** In the general parallel generic handler (around line 794–880), inputs are promoted to VECCALC via `isBroadcastMap()`. We need to add a new branch for the `BroadcastTranspose` case that:
1. Reads the GM tile (narrow: `[Tb_m, 1]` for map `(d0,d1)->(d1,0)`)
2. `data_copy_l2` → VECIN
3. `broadcast_l2([Tb_m, 1] → [Tb_m, Tb_n])` — expand the column to Tb_n
4. `ascendc.transpose([Tb_n, Tb_m])` — reorder to match iteration space

**Important:** Step 3 and 4 match the tentative plan from the spec. During implementation, verify the `broadcast_l2` API accepts these shapes by checking the existing `BroadcastL2Op` in `externals/pyasc/include/ascir/Dialect/Asc/IR/` and comparing with the existing broadcast_l2 call in `step5_ascendc.mlir` line 64:
```
ascendc.broadcast_l2 %dst, %src, %row:i32, %col:i32, %row_mult:i32, %dst_col:i32
```

Also handle `PureTranspose` case for full-rank transposed inputs.

- [ ] **Step 1: Replace the input promotion block (lines ~794–880) to use `analyzeIndexingMap`**

The new dispatch structure inside the input promotion loop:

```cpp
for (unsigned i = 0; i < numInputs; ++i) {
  Value inMemref = genOp.getDpsInputOperand(i)->get();
  AffineMap inMap = maps[i];
  int64_t inMs    = getMemorySpace(inMemref.getType());

  IndexingMapAnalysis analysis = analyzeIndexingMap(inMap, iterRank);

  switch (analysis.kind) {
  case IndexingMapAnalysis::Kind::Identity: {
    if (inMs == 0 /*GM*/) {
      // GM full-rank: data_copy_l2 via GlobalTensor + VECIN queue
      Value srcGt = builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref, /*size=*/Value{});
      inputLts[i] = copyGmToVecin(builder, loc, elemType, srcGt, totalElems);
    } else {
      inputLts[i] = readTensor(builder, loc, inMemref);
    }
    break;
  }
  case IndexingMapAnalysis::Kind::PureBroadcast: {
    // Existing broadcast_l2 path (same as before, refactored)
    if (inMs == 9 /*VECIN*/) {
      // existing VECIN broadcast code (copy from old branch)
    } else {
      // existing GM broadcast code (copy from old branch)
    }
    break;
  }
  case IndexingMapAnalysis::Kind::PureTranspose: {
    // Full-rank transposed input: data_copy_l2 + ascendc.transpose
    // The input tensor has shape permuted relative to iteration space.
    // iterDimSizes[d] gives the size of iteration dim d.
    // For map (d0,d1)->(d1,d0), input shape is [iterDimSizes[1], iterDimSizes[0]].
    // Read that shape from GM, then transpose to [iterDimSizes[0], iterDimSizes[1]].
    SmallVector<Value> srcDims;
    for (int64_t permDim : analysis.permutation)
      srcDims.push_back(iterDimSizes[static_cast<unsigned>(permDim)]);
    Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
    for (Value d : srcDims)
      srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);

    Value srcGt = builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
    builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref, /*size=*/Value{});
    Value srcVecinLt = copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount);

    // Allocate VECCALC for transposed output (iteration-space shape)
    auto [transpTbuf, transpLt] = allocVeccalc(builder, loc, elemType, iterDimSizes);
    builder.create<TransposeOp>(loc, transpLt, srcVecinLt);
    inputLts[i] = transpLt;
    break;
  }
  case IndexingMapAnalysis::Kind::BroadcastTranspose: {
    // Input has constant results (unit dims) and/or reordered dim exprs.
    // For map (d0,d1)->(d1,0), input[M,1]:
    //   presentDims = [1] (d1=m), broadcastDim = [0] (d0=n), constant col = 0
    // Step 1: read narrow tile from GM: shape [Tb_m, 1]
    // Step 2: broadcast_l2 to [Tb_m, Tb_n] (expand col 1->Tb_n)
    // Step 3: ascendc.transpose to [Tb_n, Tb_m] (match iteration space)
    auto srcMrt = cast<MemRefType>(inMemref.getType());
    unsigned srcRank = srcMrt.getRank();

    // Compute element count for narrow tile (actual input dims, not iter dims)
    SmallVector<Value> srcDimsVals;
    for (unsigned d = 0; d < srcRank; ++d)
      srcDimsVals.push_back(getDynDim(builder, loc, inMemref, d));
    Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
    for (Value d : srcDimsVals)
      srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);

    Value srcGt = builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
    builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref, /*size=*/Value{});
    Value srcVecinLt = copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount);

    // For the broadcast_l2 step: expand to intermediate shape before transpose.
    // The intermediate shape has the present dims' sizes at their result positions,
    // and broadcast dims get the corresponding iteration dim sizes.
    // For (d0,d1)->(d1,0): presentDim d1=m is at pos 0, constant at pos 1.
    // Intermediate shape before transpose: determine from inMap result positions.
    // Strategy: build dstShape as [iterDimSizes[presentDims[r]] for r in results]
    // where constants get their broadcast expansion size.
    //
    // For the specific (d0,d1)->(d1,0) case:
    //   - result[0] = d1 (m): iterDimSizes[1] = Tb_m
    //   - result[1] = 0  (const): expand to iterDimSizes[0] = Tb_n
    //   intermediate = [Tb_m, Tb_n]
    // Then transpose [Tb_m, Tb_n] -> [Tb_n, Tb_m] (match iteration space [d0=n, d1=m])
    //
    // General: the broadcast dims are the dims absent from presentDims.
    // For each constant result position, assign the corresponding broadcast dim size.

    // Build intermediate shape (before final transpose to iteration-space order)
    SmallVector<Value> intermediateShape;
    unsigned broadcastDimIdx = 0;
    unsigned srcDimIdx2 = 0;
    for (AffineExpr expr : inMap.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        intermediateShape.push_back(
            iterDimSizes[static_cast<unsigned>(dimExpr.getPosition())]);
        srcDimIdx2++;
      } else {
        // Constant: use the corresponding broadcast dim size
        if (broadcastDimIdx < analysis.broadcastDims.size())
          intermediateShape.push_back(
              iterDimSizes[static_cast<unsigned>(
                  analysis.broadcastDims[broadcastDimIdx++])]);
        else
          intermediateShape.push_back(
              builder.create<arith::ConstantIndexOp>(loc, 1));
      }
    }

    // Build src/dst shape args for broadcast_l2
    SmallVector<Value> bcastDstShape, bcastSrcShape;
    for (Value s : intermediateShape)
      bcastDstShape.push_back(
          builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), s));
    for (unsigned d = 0; d < srcRank; ++d)
      bcastSrcShape.push_back(
          builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), srcDimsVals[d]));

    auto [intermTbuf, intermLt] = allocVeccalc(builder, loc, elemType, intermediateShape);
    builder.create<BroadcastL2Op>(
        loc, intermLt, srcVecinLt,
        bcastDstShape, bcastSrcShape,
        builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));

    // Transpose intermediate [Tb_m, Tb_n] -> [Tb_n, Tb_m] (iteration-space order)
    auto [finalTbuf, finalLt] = allocVeccalc(builder, loc, elemType, iterDimSizes);
    builder.create<TransposeOp>(loc, finalLt, intermLt);
    inputLts[i] = finalLt;
    break;
  }
  } // end switch
}
```

**Note:** The `BroadcastTranspose` case uses a two-step approach: `broadcast_l2` to intermediate shape, then `ascendc.transpose`. During implementation, compare the intermediate shapes with the existing `broadcast_l2` usage (line 64 of step5_ascendc.mlir) to verify the API accepts the shape arguments. If `broadcast_l2` produces the final iteration-space shape directly (without needing transpose), remove the TransposeOp.

- [ ] **Step 2: Build**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && \
  ./scripts/build.sh --build-project 2>&1 | tail -30"
```
Expected: Clean build.

- [ ] **Step 3: Run stage 5 to see if linalg-to-ascendc handles the new generic**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && \
  afir-opt --linalg-to-ascendc --canonicalize --cse \
    examples/relu-broadcast-transpose/step4_buffer_placement.mlir 2>&1"
```
Expected: AscendC ops emitted: `data_copy_l2`, `broadcast_l2`, `ascendc.transpose`, `add_l2` / `max_l2`.

- [ ] **Step 4: Commit**

```bash
git add lib/Conversion/LinalgToAscendC/ComputeConversion.cpp
git commit -m "feat(ComputeConversion): handle BroadcastTranspose and PureTranspose input maps"
```

---

## Task 8: Generalize isTransposeGeneric to N dimensions

**Files:**
- Modify: `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp` (lines 605–627)

**Context:** The current `isTransposeGeneric` lambda is hardcoded to 2D. Generalize to any rank using `getPermutationFromMap` and keep the yield-only body check.

- [ ] **Step 1: Replace isTransposeGeneric lambda**

Replace lines 605–627:
```cpp
// Before (hardcoded 2D check):
auto isTransposeGeneric = [](linalg::GenericOp op) -> bool {
  ...
  unsigned rank = op.getIteratorTypesArray().size();
  if (rank != 2)
    return false;
  ...
  return r0.getPosition() == 1 && r1.getPosition() == 0;
};
```

With the N-dimensional version:
```cpp
// Helper: detect a standalone transpose generic (any rank).
// Pattern: 1 input with a non-identity permutation map, 1 output with identity
// map, body is a single linalg.yield of the input block argument (no computation).
auto isTransposeGeneric = [](linalg::GenericOp op) -> bool {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
    return false;
  auto maps = op.getIndexingMapsArray();
  if (maps.size() != 2)
    return false;
  AffineMap inMap  = maps[0];
  AffineMap outMap = maps[1];
  unsigned rank    = op.getIteratorTypesArray().size();
  if (rank == 0)
    return false;
  // Output must be identity
  if (!outMap.isIdentity())
    return false;
  // Input must have same rank as iteration space (no broadcast)
  if (inMap.getNumResults() != rank)
    return false;
  // All input map results must be distinct AffineDimExprs (no constants, no complex exprs)
  SmallVector<int64_t> perm(rank, -1);
  for (unsigned r = 0; r < rank; ++r) {
    auto dimExpr = dyn_cast<AffineDimExpr>(inMap.getResult(r));
    if (!dimExpr)
      return false;
    int64_t pos = static_cast<int64_t>(dimExpr.getPosition());
    if (pos < 0 || pos >= static_cast<int64_t>(rank))
      return false;
    perm[r] = pos;
  }
  // Must be a non-identity permutation
  bool isIdentityPerm = true;
  for (unsigned r = 0; r < rank; ++r)
    if (perm[r] != static_cast<int64_t>(r)) { isIdentityPerm = false; break; }
  if (isIdentityPerm)
    return false;
  // Body must be yield-only (single linalg.yield yielding the input block arg)
  Block &body = *op.getBody();
  if (body.getOperations().size() != 1)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(&body.front());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;
  auto ba = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
  return ba && ba.getArgNumber() == 0;
};
```

- [ ] **Step 2: Build and run existing test suite**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && \
  ./scripts/build.sh --build-tests 2>&1 | tail -30"
```
Expected: All existing tests still pass.

- [ ] **Step 3: Commit**

```bash
git add lib/Conversion/LinalgToAscendC/ComputeConversion.cpp
git commit -m "feat(ComputeConversion): generalize isTransposeGeneric to N dimensions"
```

---

## Task 9: Run full pipeline and regenerate step5–step8 files

**Files:**
- Update: `step5_ascendc.mlir`, `step6_parallelize.mlir`, `step7_kernel.mlir`, `step8_kernel.cpp`, `step8_no_transform.mlir`

**Context:** With all code changes in place, run the full `run.sh` pipeline and save the generated files as reference outputs.

- [ ] **Step 1: Run full pipeline**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && \
  bash examples/relu-broadcast-transpose/run.sh --log 2>&1"
```
Expected: All 8 stages succeed. If stage 5 (linalg-to-ascendc) fails, debug `ComputeConversion.cpp` — the most likely issues are:
- `broadcast_l2` shape argument mismatch (see Task 7 Note)
- `iterDimSizes` not populated for the fused generic (check the "compute iter dim sizes" section, line 764–778, which looks for a full-rank input; our `data0` map has rank 2 but both results count as non-trivial — may need to use output shape instead)

- [ ] **Step 2: Inspect step5_ascendc.mlir for expected ops**

```bash
ssh xvm@orb "grep -E 'ascendc\.(broadcast_l2|transpose|data_copy|add_l2|max_l2|duplicate_l2)' \
  /home/niu/code/Ascend-MLIR/examples/relu-broadcast-transpose/step5_ascendc.mlir"
```
Expected ops: `data_copy_l2` (GM→VECIN for data0 and data1), `broadcast_l2`, `ascendc.transpose`, `add_l2`, `max_l2` (for relu).

- [ ] **Step 3: Inspect step8_kernel.cpp for AscendC API calls**

```bash
ssh xvm@orb "head -60 /home/niu/code/Ascend-MLIR/examples/relu-broadcast-transpose/step8_kernel.cpp"
```
Expected: AscendC C++ with `Transpose(...)`, `BroadcastL2(...)`, `Add(...)`, `Maximum(...)` calls.

- [ ] **Step 4: Commit all regenerated step files**

```bash
git add examples/relu-broadcast-transpose/step{5_ascendc,6_parallelize,7_kernel,8_kernel.cpp,8_no_transform}.mlir
git commit -m "feat(relu-broadcast-transpose): regenerate step5-8 after ComputeConversion update"
```

---

## Task 10: Fix iterDimSizes population for BroadcastTranspose inputs (if needed)

**Files:**
- Modify: `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`

**Context:** The existing "compute iter dim sizes from the first full-rank input" block (lines 764–778) checks `inMap.getNumResults() == iterRank`. For our fused generic, `data0` map `(d0,d1)->(d1,0)` has 2 results but one is a constant — `getNumResults()` returns 2 (== iterRank), but the dims read from `inMemref` at positions 0,1 won't give the right `iterDimSizes` because data0 shape is `[m, 1]` not `[n, m]`.

If stage 5 failed in Task 9 due to wrong `iterDimSizes`, fix here. The fix: skip inputs whose map contains any constant results when choosing the reference input for `iterDimSizes`; prefer full-rank dim-only maps.

- [ ] **Step 1: Update the iterDimSizes computation to skip BroadcastTranspose inputs**

In the block at lines 764–778, change the condition from:
```cpp
if (inMap.getNumResults() == iterRank) {
```
To check that ALL results are `AffineDimExpr`:
```cpp
bool allDimExprs = llvm::all_of(inMap.getResults(),
    [](AffineExpr e) { return isa<AffineDimExpr>(e); });
if (inMap.getNumResults() == iterRank && allDimExprs) {
```

This ensures we only use a map like `(d0,d1)->(d0,d1)` or `(d0,d1)->(d1,d0)` (not `(d0,d1)->(d1,0)`) to populate `iterDimSizes`. For our generic, `data1` has the identity map and will be selected as the reference.

- [ ] **Step 2: Build and re-run stage 5**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && \
  ./scripts/build.sh --build-project 2>&1 | tail -10 && \
  source examples/env.sh && \
  afir-opt --linalg-to-ascendc --canonicalize --cse \
    examples/relu-broadcast-transpose/step4_buffer_placement.mlir 2>&1 | head -40"
```

- [ ] **Step 3: Commit**

```bash
git add lib/Conversion/LinalgToAscendC/ComputeConversion.cpp
git commit -m "fix(ComputeConversion): skip BroadcastTranspose inputs when computing iterDimSizes"
```

---

## Task 11: Final end-to-end verification and cleanup

**Files:**
- Run: `examples/relu-broadcast-transpose/run.sh`
- Run: `./scripts/build.sh --build-tests`

- [ ] **Step 1: Run full pipeline end-to-end**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && source examples/env.sh && \
  bash examples/relu-broadcast-transpose/run.sh --log 2>&1"
```
Expected: All 8 stages succeed, `step8_kernel.cpp` generated.

- [ ] **Step 2: Run full test suite to check no regressions**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && \
  ./scripts/build.sh --build-tests 2>&1 | tail -20"
```
Expected: All tests pass.

- [ ] **Step 3: Run clang-format on modified C++ files**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && \
  bash scripts/check_clang_format.sh -c -f"
```

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "feat(relu-broadcast-transpose): complete transpose fusion pipeline with named ops"
```
