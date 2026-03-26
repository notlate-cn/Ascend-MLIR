# Gather + Elementwise Fusion Design

**Date:** 2026-03-26
**Scope:** `examples/gather-elementwise-fusion/` + pipeline infrastructure
**Goal:** Support fused Gather+Elementwise kernels on AscendNPU with data staying in UB, using standard torch-MLIR-compatible IR (no `library_call`).

---

## 1. Problem Statement

The existing `ewop-broadcast-gather` example uses `library_call = "gather_by_index"` to annotate gather semantics, which blocks direct integration with torch-MLIR or StableHLO frontends. The goal is to:

1. Accept standard torch-MLIR-lowered IR (body-level `tensor.extract`, no `library_call`)
2. Support two gather patterns: **index_select** (column selection) and **embedding** (row selection)
3. Fuse adjacent elementwise ops into the gather's row loop to keep data in UB
4. Fall back to single AscendC API for unsupported patterns

---

## 2. Computation Graph

Example: `relu → index_select → add`

```
data[M, N]  ──relu──→  relu_out[M, N]
                             │
indices[K]  ────────→  index_select(dim=1)  ──→  gathered[M, K]
                                                       │
bias[K]     ──────────────────────────────────→  add  ──→  out[M, K]
```

Expressed as three `linalg.generic` ops connected by SSA values (no `library_call`):

```mlir
// Op1: relu — arith.maximumf(x, 0) in body
// Op2: index_select — tensor.extract %relu_out[i, indices[j]] in body
// Op3: add — arith.addf in body
```

---

## 3. Two Gather Patterns

### 3.1 index_select (Column Selection)

**Semantics:** `out[i, j] = data[i, indices[j]]`
**torch-MLIR origin:** `aten.index_select(data, dim=1, indices)`

```mlir
%out = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d1)>,        // indices: col-broadcast
    affine_map<(d0, d1) -> (d0, d1)>     // out: full
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%indices : tensor<?xi64>)
  outs(%empty : tensor<?x?xf32>) {
^bb0(%idx: i64, %out: f32):
  %i = linalg.index 0
  %j = linalg.index 1
  %idx_cast = arith.index_cast %idx : i64 to index
  %val = tensor.extract %data[%i, %idx_cast] : tensor<?x?xf32>
  linalg.yield %val : f32
}
```

**UB data flow per tile row i:**
```
data_row[N] ──(pre-fused relu)──→ relu_row[N]
relu_row[N]  ──gather_l2(indices[K])──→ gathered_row[K]
gathered_row[K] + bias[K] ──add_l2──→ out_row[K]
```

### 3.2 embedding (Row Selection)

**Semantics:** `out[j, :] = weight[indices[j], :]`
**torch-MLIR origin:** `aten.embedding(weight, indices)`

```mlir
%out = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0)>,        // indices: row-broadcast
    affine_map<(d0, d1) -> (d0, d1)>     // out: full
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%indices : tensor<?xi64>)
  outs(%empty : tensor<?x?xf32>) {
^bb0(%idx: i64, %out: f32):
  %i = linalg.index 0
  %j = linalg.index 1
  %idx_cast = arith.index_cast %idx : i64 to index
  %val = tensor.extract %weight[%idx_cast, %j] : tensor<?x?xf32>
  linalg.yield %val : f32
}
```

**UB data flow per tile index j:**
```
weight_row[EmbDim] = weight[indices[j], :]  // gather_l2 取整行
weight_row[EmbDim] ──(fused relu/add)──→ out_row[EmbDim]
```

---

## 4. Gather Detection: Five-Condition Rule

Implemented in `--mark-structured-ops` pass (C++, not Transform script).

A `linalg.generic` is identified as a gather op iff ALL five conditions hold:

1. **body contains `tensor.extract`** — at least one extract op in the region
2. **extracted tensor is in `ins`** — the tensor operand of extract is a block argument mapped to an ins operand (not a captured external value)
3. **one ins operand acts as indices** — another ins block arg is used as an index into the extracted tensor at exactly one dimension
4. **indices ins has a lower-rank indexing map** — its affine map has fewer results than the iteration rank (broadcast map), identifying it as the index tensor
5. **remaining extract indices come from `linalg.index`** — all other dimensions of the extract use `linalg.index` ops, not block arguments

This rules out:
- Constant table lookups (extracted tensor not in ins)
- stablehlo.gather style (ins is empty, all tensors captured)
- transpose/copy ops (no tensor.extract in body)
- concat/split (not linalg.generic or body has no extract)

**Distinguishing index_select vs embedding:**

After the five conditions pass, inspect the indices indexing map:
- `(d0, d1) -> (d1)` — indices vary along d1 → **column selection** → `{gather_dim = 1 : i64}`
- `(d0, d1) -> (d0)` — indices vary along d0 → **row selection** → `{embedding_dim = 0 : i64}`

For N-D inputs: the dimension that appears in the indices map (and is used as the dynamic index into the data tensor) determines gather_dim / embedding_dim.

---

## 5. New Pass: `--mark-structured-ops`

**Location:** `lib/Conversion/MarkStructuredOps/MarkStructuredOpsPass.cpp`
**Registration:** added to `include/Conversion/Passes.td`, exposed via `afir-opt`

**Behavior:**
- Walks all `linalg.generic` ops in the function
- Applies five-condition check to detect gather patterns
- Sets `{gather_dim = N : i64}` or `{embedding_dim = N : i64}` attribute on matching ops
- Emits no diagnostics for non-matching ops (silent pass)
- Does **not** modify IR semantics — attribute-only annotation

**Extensibility:** The same pass will host detection for other structured op patterns in the future (e.g., transpose, reduction with special semantics). Each detector is a separate function returning an optional attribute set.

---

## 6. Pipeline

```
step0_input.mlir          torch-MLIR style, no library_call
    │ afir-opt --mark-structured-ops
step1_marked.mlir         gather_dim/embedding_dim attributes added
    │ afir-opt --transform-interpreter
step2_tiled.mlir          two-level tiling (TB + Tb=1 for gather rows)
    │ afir-opt --one-shot-bufferize
step3_bufferized.mlir
    │ afir-opt --ascendc-buffer-placement
step4_buffer_placement.mlir
    │ afir-opt --linalg-to-ascendc-compute   (extended)
step5_ascendc.mlir
    │ (existing: parallelize → prepare-for-emit → emit)
step8_kernel.cpp
```

---

## 7. Transform Script (step2)

Uses `attributes{gather_dim}` / `attributes{embedding_dim}` instead of `library_call`:

```mlir
// Match gather op by attribute
%gather_op = transform.structured.match ops{["linalg.generic"]}
    attributes{gather_dim} in %func

// Match elementwise ops (no gather_dim, no embedding_dim attribute)
%ew_ops = transform.structured.match ops{["linalg.generic"]} in %func
// (filter: ops without gather_dim/embedding_dim)

// Two-level tiling for gather: Tb_M = 1 (one row per UB batch)
%tb_loop, %gather_tiled = transform.structured.tile_using_for %gather_op
    tile_sizes = [TB_M, 0]
%inner_loop, %gather_inner = transform.structured.tile_using_for %gather_tiled
    tile_sizes = [1, 0]   // Tb_M = 1 for index_select

// Annotate for AscendC scheduling
transform.annotate %tb_loop "ascendc.parallel" = %true
transform.annotate %inner_loop "ascendc.prologue" = "src:GM->VECIN"
transform.annotate %inner_loop "ascendc.epilogue" = "dst:VECOUT->GM"
```

Elementwise ops adjacent to the gather (pre/post) are handled by the same tiling structure since they share the same iteration space and are fused into the row loop by ComputeConversion.

---

## 8. ComputeConversion Extension

### 8.1 Fusion Detection

ComputeConversion runs **after bufferization**, so tensor SSA values have become memrefs. The fusion detection uses memref alias analysis (the same `AscendCBufferContext` used by the existing pass) to find adjacent elementwise generics:

- **Pre-gather elementwise:** a `linalg.generic` whose output memref is the same as (or an alias of) the `data` memref input of the gather generic, and whose body contains only arith ops (no `tensor.extract`)
- **Post-gather elementwise:** a `linalg.generic` that writes to a memref that is the gather output's memref, same body condition

The `--mark-structured-ops` pass runs **before bufferization** and sets attributes on tensor-level ops; those attributes are preserved through bufferization on the resulting memref-level ops.

Each fused elementwise op is processed inline within the gather's row loop. Intermediate results stay in VECCALC buffers in UB.

**Fusion boundary condition:** An elementwise op is only fused if:
- It has a single producer or consumer relationship with the gather
- Its body contains only supported arith ops (`arith.maximumf`, `arith.addf`, `arith.mulf`)
- No other op in the function consumes its intermediate result (no fan-out)

Ops that fail these conditions are left as separate generics and fall back to individual AscendC API calls.

### 8.2 index_select Code Path (`isIndexSelectGeneric`)

**Detection:** `gather_dim` attribute present, indices indexing map rank < iteration rank, gather dimension is not 0.

**Code generation per tile (Tb_M=1, one row i):**

```
1. Pre-fused elementwise (e.g. relu):
   data_row[N] → relu_row[N]   via existing parallel elementwise path

2. gather_l2:
   relu_row[N] + indices[K] → gathered_row[K]
   gather_l2(dst=gathered_row, src=relu_row, srcOffset=indices, srcBase=0, count=K)

3. Post-fused elementwise (e.g. add):
   gathered_row[K] + bias[K] → out_row[K]   via add_l2
```

### 8.3 embedding Code Path (`isEmbeddingGeneric`)

**Detection:** `embedding_dim` attribute present, indices indexing map selects dimension 0.

**Code generation per tile (one index j):**

```
1. gather_l2 (row copy):
   weight_row[EmbDim] = weight[indices[j], :]
   gather_l2(dst=weight_row, src=weight_slice, srcOffset=indices_j, srcBase=0, count=EmbDim)

2. Pre/post-fused elementwise applied directly on weight_row in UB
   (no separate data copy needed — gather IS the data movement)
```

### 8.4 Fallback

Any `linalg.generic` without `gather_dim`/`embedding_dim` that contains `tensor.extract` in its body, or any gather op whose adjacent elementwise ops fail the fusion conditions, is emitted as a single AscendC API call (existing fallback path). A warning is emitted to help users identify unfused ops.

---

## 9. File Layout

```
examples/gather-elementwise-fusion/
    step0_input.mlir          relu → index_select → add (torch-MLIR style)
    step2_transform.mlir      Transform script using gather_dim attribute
    run.sh                    Full pipeline script

include/Conversion/MarkStructuredOps/
    MarkStructuredOpsPass.h

lib/Conversion/MarkStructuredOps/
    MarkStructuredOpsPass.cpp
    CMakeLists.txt

lib/Conversion/LinalgToAscendC/
    ComputeConversion.cpp     Extended with isIndexSelectGeneric, isEmbeddingGeneric
```

---

## 10. N-D Generalization

All detection and code generation is parameterized by `gather_dim` (the attribute value), not hardcoded for 2D. For an N-D input tensor:

- `gather_dim = k` means dimension k is the gather axis
- The row loop iterates over all other dimensions
- `gather_l2` processes a 1-D slice along dimension k

This allows the same pipeline to handle 3D+ inputs without changes to the transform script or detection logic.

---

## 11. Out of Scope

- `aten.gather` (element-level gather) — not supported, falls back to single API
- stablehlo.gather — not supported in this phase (no ins, body-only access)
- Multi-level gather (gather of gather) — not supported
- Reduction ops fused with gather — not supported in this phase
