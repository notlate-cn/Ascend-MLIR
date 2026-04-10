# Transpose Fusion with Elementwise/Broadcast Ops — Design Spec

**Date:** 2026-03-26
**Example:** `examples/relu-broadcast-transpose`
**Status:** Draft

---

## 1. Background

The existing `relu-broadcast-transpose` example pipeline handles transpose as a standalone
`linalg.generic` with a permutation indexing map, detected by `isTransposeGeneric()` in
`ComputeConversion.cpp` and lowered to `ascendc.transpose`. This approach works but prevents
fusion of transpose with adjacent elementwise/broadcast ops, leading to extra memory round-trips
(GM → UB → GM → UB).

This spec **replaces the existing `relu-broadcast-transpose` example** with a new computation
graph that:
- Uses named linalg ops in `step0_input.mlir` to match the output format of StableHLO/torch-MLIR
- Fuses all ops into a single `linalg.generic` at step 0-out
- Drives a generalization of `ComputeConversion.cpp` to handle transpose semantics embedded in
  indexing maps

This spec describes:
1. A new example `step0` input IR using **named linalg ops** to represent the computation graph
2. A **generalize + fuse** step that collapses the named ops into a single `linalg.generic`
3. A **generalized `analyzeIndexingMap()`** in `ComputeConversion.cpp` that recognizes
   transpose semantics embedded in indexing maps and generates correct AscendC code
4. A **fallback path** for standalone transpose ops (any dimensionality)

The existing three-op graph (`relu_bias_add` / `transpose` / `scale_mul`) and its associated
Transform tiling script are replaced in full. The new example has a single fused generic at
step 0-out, requiring a new Transform script in `step2_transform.mlir`.

---

## 2. Target Computation Graph and Scenario Coverage

```
data0[m,1]
    │
linalg.elemwise_unary (relu)   →  [m,1]
    │
linalg.transpose [1,0]         →  [1,m]
    │
linalg.broadcast dim[0]        →  [n,m]
    │                data1[n,m]
linalg.add                     →  [n,m]
```

This graph demonstrates two of the three target fusion scenarios directly:

- **Elementwise+Transpose**: `relu(data0[m,1]) → transpose → [1,m]` — an elementwise op feeds
  directly into a transpose
- **Transpose+Broadcast**: `transpose([1,m]) → broadcast → [n,m]` — a transpose feeds directly
  into a broadcast

The third scenario, **Transpose+Elementwise** (a transpose whose output feeds directly into an
elementwise op without an intervening broadcast), is handled by the same `analyzeIndexingMap`
logic but is not the focus of this demo. An example would be:
`transpose(X[m,n]) → add(Y[n,m]) → [n,m]`, where the fused generic has input map
`(d0,d1)->(d1,d0)` for X. This case is covered by the "pure transpose" analysis path in Section 6.

---

## 3. Step 0: Named Linalg Op IR

`step0_input.mlir` uses standard MLIR named ops:

```mlir
func.func @relu_transpose_broadcast_add(
    %data0: tensor<?x1xf16>,
    %data1: tensor<?x?xf16>) -> tensor<?x?xf16> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %data0, %c0
  %n = tensor.dim %data1, %c0

  // relu: elemwise unary (max with zero)
  %relu_init = tensor.empty(%m, %c1) : tensor<?x1xf16>
  %relu = linalg.elemwise_unary
      ins(%data0 : tensor<?x1xf16>)
      outs(%relu_init : tensor<?x1xf16>) -> tensor<?x1xf16>

  // transpose [1,0]: [m,1] -> [1,m]
  // tensor<1x?xf16> has one dynamic dim (?), so tensor.empty takes only %m
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

  // add
  %add_init = tensor.empty(%n, %m) : tensor<?x?xf16>
  %result = linalg.add
      ins(%bc, %data1 : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%add_init : tensor<?x?xf16>) -> tensor<?x?xf16>

  return %result : tensor<?x?xf16>
}
```

---

## 4. Step 0-out: Fused Generic IR

After `--linalg-generalize-named-ops | --linalg-fuse-elementwise-ops`, the four ops collapse
into a single `linalg.generic` with iteration space `[n, m]`:

```mlir
#map0 = affine_map<(d0, d1) -> (d1, 0)>   // data0: broadcast+transpose encoded
#map1 = affine_map<(d0, d1) -> (d0, d1)>  // data1: identity
#map2 = affine_map<(d0, d1) -> (d0, d1)>  // output: identity

linalg.generic {
  indexing_maps = [#map0, #map1, #map2],
  iterator_types = ["parallel", "parallel"]
} ins(%data0, %data1 : tensor<?x1xf16>, tensor<?x?xf16>)
  outs(%out : tensor<?x?xf16>) {
^bb0(%v0: f16, %v1: f16, %vout: f16):
  %cst = arith.constant 0.0 : f16
  %relu = arith.maximumf %v0, %cst : f16
  %add  = arith.addf %relu, %v1 : f16
  linalg.yield %add : f16
} -> tensor<?x?xf16>
```

**Reading the fused map `(d0,d1) -> (d1, 0)` for `data0[m,1]`:**
- Iteration space: `d0` = n-axis, `d1` = m-axis
- `d1` in position 0 of the result: for each `(n_idx, m_idx)` iteration point, we access
  `data0[m_idx, ...]` — the m-axis indexes the first (row) dimension of data0, encoding the
  **transpose** (rows and columns are swapped relative to the iteration space)
- `0` in position 1 of the result: the second dimension of data0 is always 0 — encoding the
  **broadcast** (`d0` / n-axis is absent from the map, so all n-values read the same row)

These two effects are independent:
- Absent dimension (`d0` not in map) → broadcast: all n-values share the same data
- Dim reordering (`d1` at result position 0) → transpose: m-axis indexes the input rows

---

## 5. Pipeline Changes

### 5.1 run.sh additions

```bash
# New step: named ops → fused generic (produces step0_input_out.mlir)
afir-opt step0_input.mlir \
  --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  -o step0_input_out.mlir

# Remaining steps operate on the single fused generic.
# step2_transform.mlir must be updated to match a single-op IR.
```

The Transform tiling script (`step2_transform.mlir`) currently matches three separate
`linalg.generic` ops by `library_call` attribute. After fusion there is one generic with no
`library_call`. The Transform script must be rewritten to match and tile the single fused
generic by its output type or by `named_sequence`.

### 5.2 Buffer placement note

After bufferization, `data0` becomes `memref<?x1xf16>`. The `AscendCBufferPlacementPass`
assigns it GM memory space (function argument). When the pass processes the tiled loop body,
it will see a subview of shape `[Tb_m, 1]` extracted from `data0`. This subview stays in GM
(not promoted to on-chip) because the `data_copy_l2` + `broadcast_l2` emission in
`ComputeConversion` handles the GM→VECIN→VECCALC copy explicitly. No changes to
`AscendCBufferPlacementPass` are expected; this assumption must be verified during implementation.

### 5.3 Files to update

| File | Change |
|------|--------|
| `examples/relu-broadcast-transpose/step0_input.mlir` | Rewrite with named linalg ops |
| `examples/relu-broadcast-transpose/step0_input_out.mlir` | Update to fused generic form |
| `examples/relu-broadcast-transpose/step[1-8]*.mlir` | Update to match new IR shape |
| `examples/relu-broadcast-transpose/run.sh` | Add generalize+fuse step |
| `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp` | See Section 6 |

---

## 6. ComputeConversion Generalization

### 6.1 `IndexingMapAnalysis` struct

```cpp
struct IndexingMapAnalysis {
  bool isIdentity;            // map is (d0,d1,...) -> (d0,d1,...)
  bool hasBroadcast;          // some iteration dims absent from output
  bool hasTranspose;          // present dims appear in non-identity order
  SmallVector<int64_t> permutation;    // permutation of the present dims (relative to iter order)
  SmallVector<int64_t> broadcastDims; // iteration dims absent from the output
};
```

### 6.2 `analyzeIndexingMap()` classification

```cpp
IndexingMapAnalysis analyzeIndexingMap(AffineMap map, unsigned iterRank);
```

Four cases, in order of precedence:

| Case | Condition | Example map (2D iter) | Scenario |
|------|-----------|-----------------------|----------|
| Identity | `map.isIdentity()` | `(d0,d1)->(d0,d1)` → input[n,m] | Plain elementwise |
| Pure broadcast | all results are dim exprs; some iteration dims absent (output rank < iterRank) | `(d0,d1)->(d0)` → input[n] | Broadcast+elementwise |
| Pure transpose | all results are dim exprs; output rank == iterRank; non-identity order | `(d0,d1)->(d1,d0)` → input[m,n] | Transpose+elementwise |
| Broadcast+transpose | one or more results are constants (unit dims); remaining dim results may be reordered | `(d0,d1)->(d1,0)` → input[m,1] | Elementwise+transpose+broadcast |

For **Broadcast+transpose**, constants in map results (e.g., `0`) indicate unit-size dimensions
in the input tensor. The corresponding output positions do not correspond to any iteration dim,
so the iteration dimension that would normally occupy that output axis is absent (broadcast).
The output rank of such a map equals iterRank (two result expressions for 2D iteration), but
one result is a constant — this is what distinguishes it from Case 2 (dimension absent from
results entirely) and Case 3 (all results are non-constant dim exprs).

### 6.3 AscendC code generation per case

#### Case 1: Identity
No change from current behavior: `dequeue` (VECIN) or `data_copy_l2` (GM→VECIN).

#### Case 2: Pure broadcast
Same as current behavior:
- GM: `data_copy_l2(src[narrow_tile], dst_vecin[narrow_tile])`
- Then: `broadcast_l2(dst_veccalc[full_tile], src_vecin[narrow_tile])`

Example: map `(d0,d1)->(d0)`, input `[M]`, tile `[Tb_n]`:
- GM tile shape: `[Tb_n]`
- `broadcast_l2`: `[Tb_n]` → `[Tb_n, Tb_m]` (expand along d1/m-axis)

#### Case 3: Pure transpose
Input tensor has shape permuted relative to the iteration space.

For map `(d0,d1)->(d1,d0)`, input `[M,N]`, iteration tile `[Tb_n, Tb_m]`:
- Read from GM: tile at `input[tb_m_offset:tb_m_offset+Tb_m, tb_n_offset:tb_n_offset+Tb_n]`
  → shape `[Tb_m, Tb_n]` (input's natural layout)
- `data_copy_l2(src_gm[Tb_m, Tb_n], dst_vecin[Tb_m, Tb_n])`
- `ascendc.transpose(dst_veccalc[Tb_n, Tb_m], src_vecin[Tb_m, Tb_n])` → produces `[Tb_n, Tb_m]`
  matching the iteration-space layout

#### Case 4: Broadcast+transpose
The input has a constant in one map result position (unit dim) AND the variable dim result
is reordered relative to its iteration axis.

For map `(d0,d1)->(d1,0)`, input `[M,1]`, iteration tile `[Tb_n, Tb_m]`:
- `d0` (n-axis) absent from variable results → broadcast: all n-values read the same row
- `d1` (m-axis) at result position 0 → m-indexes input rows (transpose effect)
- `0` at result position 1 → input column always 0, size 1 in input tensor

Target output tile in iteration space: `[Tb_n, Tb_m]`

AscendC emission (logical):
1. Read GM tile: `input[m_offset:+Tb_m, 0:1]` → shape `[Tb_m, 1]`
2. `data_copy_l2(src_gm[Tb_m, 1], dst_vecin[Tb_m, 1])`

The `broadcast_l2` API from existing usage in `step5_ascendc.mlir`:
```
broadcast_l2(dst, src, src_rows:i32, src_cols:i32, dst_rows_multiplier:i32, dst_cols:i32)
```
The existing usage broadcasts `[Tb_N, 1]` → `[Tb_N, Tb_M]` (column-vector to matrix, expanding
`src_cols=1` to `dst_cols=Tb_M` with `dst_rows_multiplier=1`).

For the new case `[Tb_m, 1]` → `[Tb_n, Tb_m]`, a direct single `broadcast_l2` call is
insufficient because the expansion requires both column-broadening (1→Tb_m) and row-broadening
(Tb_m→Tb_n), which changes the row-axis semantics. Two options at implementation time:

**Option A:** `broadcast_l2([Tb_m,1] → [Tb_m,Tb_n])` then `ascendc.transpose([Tb_n,Tb_m])`
- First expand cols: `src_rows=Tb_m, src_cols=1, dst_rows_multiplier=1, dst_cols=Tb_n`
- Then transpose to get `[Tb_n, Tb_m]`

**Option B:** `broadcast_l2([Tb_m,1] → [Tb_n*Tb_m,1])` then reshape — non-standard, avoid.

**Implementation must determine** which option is valid per the AscendC `broadcast_l2`
constraint (e.g., whether `dst_cols` must equal original `src_cols * multiplier`).
Option A is tentatively preferred.

### 6.4 Fallback: standalone transpose generic

A `linalg.generic` is a standalone transpose if:
- Exactly 1 input, 1 output
- Output map is identity
- Input map is a non-identity permutation of all iteration dims
- Body consists of a single `linalg.yield` with the input block argument (no computation)

**Generalization to N dimensions:**

```cpp
// Before: hardcoded 2D check
return r0.getPosition() == 1 && r1.getPosition() == 0;

// After: check input map is a valid non-identity N-dim permutation
SmallVector<int64_t> perm;
if (!getPermutationFromMap(inMap, perm)) return false;
if (llvm::equal(perm, llvm::seq<int64_t>(0, perm.size()))) return false; // identity → not transpose
// Also verify body is yield-only (existing check, retained)
return isYieldOnlyBody(op);
```

The downstream `ascendc.transpose` emission passes the permutation; no further changes needed.

---

## 7. Three Fusion Scenarios — Coverage Summary

| Scenario | Fused input map | `analyzeIndexingMap` case | AscendC op sequence | Covered by demo? |
|----------|----------------|--------------------------|---------------------|-----------------|
| Elementwise+Transpose | `(d0,d1)->(d1,d0)` | Case 3: Pure transpose | `data_copy_l2([Tb_m,Tb_n])` → `ascendc.transpose` → `[Tb_n,Tb_m]` | No (requires `data0[m,n]`) |
| Transpose+Broadcast | `(d0,d1)->(d1,0)` | Case 4: Broadcast+transpose | `data_copy_l2([Tb_m,1])` → `broadcast_l2` → `ascendc.transpose` → `[Tb_n,Tb_m]` (verify) | Yes (`data0[m,1]`) |
| Transpose+Elementwise | `(d0,d1)->(d1,d0)` on one input; identity on others | Case 3 + Case 1 | `data_copy_l2` + `ascendc.transpose` for transposed input; normal path for others | No (requires `data0[m,n]`) |

The demo example (`data0[m,1]` map = `(d0,d1)->(d1,0)`, `data1[n,m]` map = identity) directly
exercises scenario 2 (Transpose+Broadcast). Scenarios 1 and 3 share the same Case 3 analysis
code path and are validated by the `analyzeIndexingMap` logic, but a separate test case with
`data0[m,n]` would be needed to exercise the full `ascendc.transpose` path end-to-end.

Standalone (unfused) transpose falls back to the generalized `isTransposeGeneric` path (Section 6.4).

---

## 8. Out of Scope

- Transpose of rank > 2 in the demo example (supported by design, not demonstrated)
- Reduction ops fused with transpose (not supported; transpose must be pure parallel)
- Non-affine indexing (e.g., gather) fused with transpose (not supported)
- Auto-detection of fusability (driven by MLIR's existing fusion pass, not custom logic)
- Verifying `broadcast_l2` axis-ordering semantics for Case 4 (deferred to implementation)

---

## 9. Testing

- `examples/relu-broadcast-transpose/run.sh` — end-to-end pipeline smoke test
- Each step file (`step0` through `step8`) serves as a reference for the expected IR at that stage
- Existing lit tests in `test/Conversion/` are unaffected (no pass API changes)
