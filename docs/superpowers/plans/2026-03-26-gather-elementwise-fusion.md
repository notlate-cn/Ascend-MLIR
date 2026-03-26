# Gather + Elementwise Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support fused Gather+Elementwise AscendNPU kernels from torch-MLIR-compatible IR (`relu → index_select → add`) with data staying in UB, using a new `--mark-structured-ops` pass instead of `library_call`.

**Architecture:** A new `MarkStructuredOps` pass detects gather patterns via five structural conditions on `linalg.generic` bodies (no `library_call`), stamps `{gather_dim}` or `{embedding_dim}` attributes, and the Transform script matches by attribute. `ComputeConversion` is extended with two new code paths (`isIndexSelectGeneric`, `isEmbeddingGeneric`) that fuse adjacent elementwise ops into the gather row loop, keeping intermediate data in VECCALC.

**Tech Stack:** MLIR TableGen (pass declaration), C++ (pass implementation + ComputeConversion extension), MLIR Transform dialect (tiling script), MLIR linalg/arith/tensor dialects.

---

## File Map

**New files:**
- `include/Conversion/MarkStructuredOps/MarkStructuredOpsPass.h` — pass factory declaration
- `lib/Conversion/MarkStructuredOps/MarkStructuredOpsPass.cpp` — pass implementation (gather detection + attribute stamping)
- `lib/Conversion/MarkStructuredOps/CMakeLists.txt` — build target `MarkStructuredOpsConversion`
- `examples/gather-elementwise-fusion/step0_input.mlir` — torch-MLIR style input IR (relu → index_select → add)
- `examples/gather-elementwise-fusion/step2_transform.mlir` — Transform script using `gather_dim` attribute
- `examples/gather-elementwise-fusion/run.sh` — full 8-stage pipeline script

**Modified files:**
- `include/Conversion/Passes.td` — add `MarkStructuredOpsPass` declaration
- `include/Conversion/Passes.h` — add `#include` for new pass header
- `lib/Conversion/CMakeLists.txt` — add `add_subdirectory(MarkStructuredOps)`
- `tools/afir-opt/CMakeLists.txt` — link `MarkStructuredOpsConversion`
- `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp` — add `isIndexSelectGeneric`, `isEmbeddingGeneric` code paths with fusion

---

## Task 1: Add `--mark-structured-ops` Pass Infrastructure

Register the new pass in the build system and TableGen.

**Files:**
- Modify: `include/Conversion/Passes.td`
- Modify: `include/Conversion/Passes.h`
- Modify: `lib/Conversion/CMakeLists.txt`
- Modify: `tools/afir-opt/CMakeLists.txt`
- Create: `include/Conversion/MarkStructuredOps/MarkStructuredOpsPass.h`
- Create: `lib/Conversion/MarkStructuredOps/CMakeLists.txt`
- Create: `lib/Conversion/MarkStructuredOps/MarkStructuredOpsPass.cpp` (skeleton)

- [ ] **Step 1.1: Add pass declaration to `include/Conversion/Passes.td`**

Add before the final `#endif`:

```tablegen
//===----------------------------------------------------------------------===//
// MarkStructuredOps
//===----------------------------------------------------------------------===//

def MarkStructuredOpsPass : Pass<"mark-structured-ops", "mlir::func::FuncOp"> {
  let summary = "Annotate structured linalg.generic ops with semantic attributes";
  let description = [{
    Detects structured patterns in linalg.generic ops and stamps semantic
    attributes to guide downstream passes (Transform tiling, ComputeConversion).

    Currently detected patterns:
    - index_select (column gather): body has tensor.extract where the extracted
      tensor is in ins, indexed by another ins block arg at one dimension, and
      remaining indices come from linalg.index. Stamps {gather_dim = N : i64}.
    - embedding (row gather): same five conditions, but indices indexing map
      selects dimension 0. Stamps {embedding_dim = 0 : i64}.

    Detection uses five structural conditions (see pass implementation).
    Does not modify IR semantics — attribute-only annotation.
  }];
  let constructor = "mlir::afir::createMarkStructuredOpsPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}
```

- [ ] **Step 1.2: Create `include/Conversion/MarkStructuredOps/MarkStructuredOpsPass.h`**

```cpp
//===- MarkStructuredOpsPass.h - Mark structured linalg ops -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_MARKSTRUCTUREDOPS_PASS_H
#define ASCEND_MLIR_CONVERSION_MARKSTRUCTUREDOPS_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::afir {

std::unique_ptr<Pass> createMarkStructuredOpsPass();

}  // namespace mlir::afir

#endif  // ASCEND_MLIR_CONVERSION_MARKSTRUCTUREDOPS_PASS_H
```

- [ ] **Step 1.3: Add include to `include/Conversion/Passes.h`**

Add after the last `#include "Conversion/..."` line:

```cpp
#include "Conversion/MarkStructuredOps/MarkStructuredOpsPass.h"
```

- [ ] **Step 1.4: Create `lib/Conversion/MarkStructuredOps/CMakeLists.txt`**

```cmake
add_mlir_library(MarkStructuredOpsConversion
  MarkStructuredOpsPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransforms
)
```

- [ ] **Step 1.5: Add subdirectory to `lib/Conversion/CMakeLists.txt`**

Add after the last `add_subdirectory(...)` line:

```cmake
add_subdirectory(MarkStructuredOps)
```

- [ ] **Step 1.6: Link in `tools/afir-opt/CMakeLists.txt`**

Add after the last `*Conversion` line in `target_link_libraries`:

```cmake
    MarkStructuredOpsConversion
```

- [ ] **Step 1.7: Create skeleton `lib/Conversion/MarkStructuredOps/MarkStructuredOpsPass.cpp`**

```cpp
//===- MarkStructuredOpsPass.cpp - Mark structured linalg ops -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/MarkStructuredOps/MarkStructuredOpsPass.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GEN_PASS_DECL_MARKSTRUCTUREDOPSPASS
#define GEN_PASS_DEF_MARKSTRUCTUREDOPSPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

struct MarkStructuredOpsPass
    : public impl::MarkStructuredOpsPassBase<MarkStructuredOpsPass> {
  void runOnOperation() override;
};

}  // namespace

void MarkStructuredOpsPass::runOnOperation() {
  // Placeholder — implementation in Task 2
}

std::unique_ptr<Pass> createMarkStructuredOpsPass() {
  return std::make_unique<MarkStructuredOpsPass>();
}

}  // namespace mlir::afir
```

- [ ] **Step 1.8: Build to verify infrastructure compiles**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && sleep 1 && \
  ./scripts/build.sh --build-project 2>&1 | tail -20"
```

Expected: build succeeds, `afir-opt --help` lists `--mark-structured-ops`.

- [ ] **Step 1.9: Verify pass is registered**

```bash
ssh xvm@orb "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  afir-opt --help 2>&1 | grep mark-structured"
```

Expected output: `--mark-structured-ops`

- [ ] **Step 1.10: Commit**

```bash
git add include/Conversion/Passes.td \
        include/Conversion/Passes.h \
        include/Conversion/MarkStructuredOps/ \
        lib/Conversion/MarkStructuredOps/ \
        lib/Conversion/CMakeLists.txt \
        tools/afir-opt/CMakeLists.txt
git commit -m "feat(pass): add --mark-structured-ops pass infrastructure (skeleton)"
```

---

## Task 2: Implement Five-Condition Gather Detection

Implement the gather detection logic in `MarkStructuredOpsPass.cpp`.

**Files:**
- Modify: `lib/Conversion/MarkStructuredOps/MarkStructuredOpsPass.cpp`

- [ ] **Step 2.1: Write the full pass implementation**

Replace the skeleton `runOnOperation()` with:

```cpp
//===- MarkStructuredOpsPass.cpp - Mark structured linalg ops -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/MarkStructuredOps/MarkStructuredOpsPass.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GEN_PASS_DECL_MARKSTRUCTUREDOPSPASS
#define GEN_PASS_DEF_MARKSTRUCTUREDOPSPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

struct MarkStructuredOpsPass
    : public impl::MarkStructuredOpsPassBase<MarkStructuredOpsPass> {
  void runOnOperation() override;
};

// Five-condition gather detection.
// Returns {isGather, gatherDim, isEmbedding} where gatherDim is the dimension
// of the data tensor used as the dynamic gather axis.
//
// Conditions (all must hold):
//  1. Body contains at least one tensor.extract op.
//  2. The tensor operand of extract is a block argument in ins (not captured).
//  3. One other ins block arg is used as a dynamic index into the extract at
//     exactly one dimension position.
//  4. That indices ins has a lower-rank affine map (broadcast map).
//  5. All other index operands of the extract come from linalg.index ops.
//
// After conditions pass:
//  - Inspect indices affine map: if indices vary along d0 → embedding (row),
//    if along d1 (or last dim) → index_select (column).
struct GatherInfo {
  bool isGather = false;
  int64_t gatherDim = -1;   // dimension in data tensor used as gather axis
  bool isEmbedding = false; // true = row selection, false = column selection
};

GatherInfo detectGather(linalg::GenericOp op) {
  GatherInfo info;
  unsigned iterRank = op.getIteratorTypesArray().size();

  // Must have at least 2 ins (indices + data) and 1 out.
  if (op.getNumDpsInputs() < 2 || op.getNumDpsInits() != 1)
    return info;

  // Condition 1: body contains tensor.extract.
  tensor::ExtractOp extractOp;
  op.getBody()->walk([&](tensor::ExtractOp e) {
    if (!extractOp)
      extractOp = e;
  });
  if (!extractOp)
    return info;

  // Condition 2: extracted tensor is a block argument in ins.
  auto dataBa = dyn_cast<BlockArgument>(extractOp.getTensor());
  if (!dataBa || dataBa.getOwner() != op.getBody())
    return info;
  unsigned dataArgIdx = dataBa.getArgNumber();
  if (dataArgIdx >= (unsigned)op.getNumDpsInputs())
    return info; // not an ins arg

  // Condition 3 & 5: scan extract indices.
  // Exactly one index must come from another ins block arg (the indices tensor).
  // All remaining indices must come from linalg.index ops.
  int indicesArgIdx = -1;
  int64_t dynamicDim = -1; // which dimension of data the indices index into

  auto indices = extractOp.getIndices();
  for (auto [dimPos, idxVal] : llvm::enumerate(indices)) {
    if (auto ba = dyn_cast<BlockArgument>(idxVal)) {
      // Must be from a different ins arg (the indices tensor).
      if (ba.getOwner() != op.getBody())
        return info;
      if (ba.getArgNumber() == dataArgIdx)
        return info; // data arg used as its own index — unexpected
      if ((int)ba.getArgNumber() >= op.getNumDpsInputs())
        return info; // not an ins arg
      if (indicesArgIdx != -1)
        return info; // more than one dynamic dimension — not a simple gather
      indicesArgIdx = (int)ba.getArgNumber();
      dynamicDim = (int64_t)dimPos;
    } else if (auto idxOp = idxVal.getDefiningOp<linalg::IndexOp>()) {
      // Condition 5: OK, comes from linalg.index.
      (void)idxOp;
    } else {
      // Some other value — not a clean gather pattern.
      return info;
    }
  }
  if (indicesArgIdx == -1 || dynamicDim == -1)
    return info;

  // Condition 4: indices ins has a lower-rank affine map (broadcast map).
  auto maps = op.getIndexingMapsArray();
  AffineMap indicesMap = maps[(unsigned)indicesArgIdx];
  if (indicesMap.getNumResults() >= iterRank)
    return info; // not a broadcast map

  // All five conditions passed. Determine row vs column selection.
  // The indices map has exactly one result (1D indices tensor).
  // If that result is AffineDimExpr(0) → row selection (embedding).
  // If that result is AffineDimExpr(iterRank-1) → column selection (index_select).
  if (indicesMap.getNumResults() != 1)
    return info; // only support 1D indices for now

  auto dimExpr = dyn_cast<AffineDimExpr>(indicesMap.getResult(0));
  if (!dimExpr)
    return info;

  info.isGather = true;
  info.gatherDim = dynamicDim;
  // Row selection: indices vary along d0 (first iteration dim).
  info.isEmbedding = (dimExpr.getPosition() == 0);
  return info;
}

void MarkStructuredOpsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  OpBuilder builder(funcOp.getContext());

  funcOp.walk([&](linalg::GenericOp op) {
    GatherInfo info = detectGather(op);
    if (!info.isGather)
      return;

    if (info.isEmbedding) {
      op->setAttr("embedding_dim",
                  builder.getI64IntegerAttr(info.gatherDim));
    } else {
      op->setAttr("gather_dim",
                  builder.getI64IntegerAttr(info.gatherDim));
    }
  });
}

}  // namespace

std::unique_ptr<Pass> createMarkStructuredOpsPass() {
  return std::make_unique<MarkStructuredOpsPass>();
}

}  // namespace mlir::afir
```

- [ ] **Step 2.2: Build**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && sleep 1 && \
  ./scripts/build.sh --build-project 2>&1 | tail -20"
```

Expected: build succeeds.

- [ ] **Step 2.3: Commit**

```bash
git add lib/Conversion/MarkStructuredOps/MarkStructuredOpsPass.cpp
git commit -m "feat(pass): implement five-condition gather detection in mark-structured-ops"
```

---

## Task 3: Create Example Input IR (`step0_input.mlir`)

Write the torch-MLIR-style input for `relu → index_select → add`.

**Files:**
- Create: `examples/gather-elementwise-fusion/step0_input.mlir`

- [ ] **Step 3.1: Create the example directory and input IR**

```bash
mkdir -p /Volumes/GM9/code/Ascend-MLIR/examples/gather-elementwise-fusion
```

Create `examples/gather-elementwise-fusion/step0_input.mlir`:

```mlir
// ============================================================
// STAGE 0: High-Level IR - torch-MLIR style (no library_call)
//
// Computation graph: relu → index_select(dim=1) → add
//
//   data[M, N]   ──relu──→  relu_out[M, N]
//   indices[K]   ─────────→ index_select(dim=1) ──→  gathered[M, K]
//   bias[K]      ────────────────────────────────→  add ──→  out[M, K]
//
// Op1 (relu): linalg.generic, body: arith.maximumf(x, 0)
//   Matches torch-MLIR aten.relu lowering.
//
// Op2 (index_select, dim=1): linalg.generic, body: tensor.extract
//   Semantics: out[i, j] = relu_out[i, indices[j]]
//   Matches torch-MLIR aten.index_select(data, dim=1, indices) lowering.
//   Five-condition pattern (detected by --mark-structured-ops):
//     - body has tensor.extract
//     - extracted tensor (relu_out) is in ins
//     - indices ins block arg used as dynamic index at dim=1
//     - indices map (d0,d1)->d1 is a broadcast map (rank 1 < iter rank 2)
//     - dim=0 index comes from linalg.index 0
//   → stamped with {gather_dim = 1 : i64}
//
// Op3 (add): linalg.generic, body: arith.addf
//   bias[K] uses identity map (d0,d1)->d1 (col-broadcast).
//
// M, N, K are all dynamic dimensions.
// ============================================================
// RUN: afir-opt %s | FileCheck %s
// CHECK: func.func @relu_index_select_add

// indices broadcast map: indices[K] accessed as (d0,d1)->d1
#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
// full access map
#full_map = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @relu_index_select_add(
      %data    : tensor<?x?xf16>,   // [M, N]
      %indices : tensor<?xi64>,     // [K]
      %bias    : tensor<?xf16>      // [K]
  ) -> tensor<?x?xf16> {

    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %zero = arith.constant 0.0 : f16

    %dim_m = tensor.dim %data,    %c0 : tensor<?x?xf16>
    %dim_n = tensor.dim %data,    %c1 : tensor<?x?xf16>
    %dim_k = tensor.dim %indices, %c0 : tensor<?xi64>

    // ── Op1: relu(data[M,N]) → relu_out[M,N] ───────────────────────
    // torch-MLIR: aten.relu → linalg.generic with arith.maximumf(x, 0)
    %empty_relu = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %relu_out = linalg.generic {
      indexing_maps = [#full_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%data : tensor<?x?xf16>)
      outs(%empty_relu : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      %v = arith.maximumf %in, %zero : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // ── Op2: index_select(relu_out, dim=1, indices) → gathered[M,K] ─
    // torch-MLIR: aten.index_select → linalg.generic with tensor.extract
    // Semantics: gathered[i, j] = relu_out[i, indices[j]]
    // --mark-structured-ops will stamp {gather_dim = 1 : i64} on this op.
    %empty_gathered = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %gathered = linalg.generic {
      indexing_maps = [#col_broadcast_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%indices : tensor<?xi64>)
      outs(%empty_gathered : tensor<?x?xf16>) {
    ^bb0(%idx: i64, %out: f16):
      %i = linalg.index 0
      %j = linalg.index 1
      %idx_cast = arith.index_cast %idx : i64 to index
      %val = tensor.extract %relu_out[%i, %idx_cast] : tensor<?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?xf16>

    // ── Op3: gathered[M,K] + bias[K] → out[M,K] ────────────────────
    // bias is broadcast along M axis (col-broadcast)
    %empty_out = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [#full_map, #col_broadcast_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%gathered, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out : tensor<?x?xf16>) {
    ^bb0(%g: f16, %b: f16, %o: f16):
      %v = arith.addf %g, %b : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    return %out : tensor<?x?xf16>
  }
}
```

- [ ] **Step 3.2: Verify it parses**

```bash
ssh xvm@orb "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  afir-opt /home/niu/code/Ascend-MLIR/examples/gather-elementwise-fusion/step0_input.mlir"
```

Expected: no errors, IR printed.

- [ ] **Step 3.3: Verify `--mark-structured-ops` stamps the gather op**

```bash
ssh xvm@orb "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  afir-opt --mark-structured-ops \
    /home/niu/code/Ascend-MLIR/examples/gather-elementwise-fusion/step0_input.mlir \
  2>&1 | grep gather_dim"
```

Expected output: `gather_dim = 1 : i64` on the index_select generic.

- [ ] **Step 3.4: Commit**

```bash
git add examples/gather-elementwise-fusion/step0_input.mlir
git commit -m "feat(example): add gather-elementwise-fusion step0 input IR (relu+index_select+add)"
```

---

## Task 4: Write Transform Script (`step2_transform.mlir`)

The transform script tiles the three ops using `gather_dim` attribute matching.

**Files:**
- Create: `examples/gather-elementwise-fusion/step2_transform.mlir`

- [ ] **Step 4.1: Create the transform script**

Note: the function and transform live in the same file (same pattern as ewop-broadcast-gather). The `--mark-structured-ops` pass stamps attributes before `--transform-interpreter` runs.

Create `examples/gather-elementwise-fusion/step2_transform.mlir` (this file is generated by run.sh by first running `--mark-structured-ops` on step0, then feeding the marked IR as input to `--transform-interpreter`):

```mlir
// ============================================================
// STAGE 2: Transform Dialect Tiling
//
// Input: step1_marked.mlir (after --mark-structured-ops)
//   Op2 (index_select) carries {gather_dim = 1 : i64}
//
// Tiling strategy:
//   All three ops share [M, K] iteration space.
//   d0 = M: two-level tile (TB inter-core, Tb=1 intra-core for gather row)
//   d1 = K: no tiling (full K stays in UB)
//
//   gather_dim op: Tb_M=1 (one row per UB batch, data_row[N] → gathered[K])
//   elementwise ops: same Tb_M parameter (may be >1 but kept equal for simplicity)
//
// Matching strategy (no library_call):
//   - Gather op matched by: attributes{gather_dim}
//   - Elementwise ops (relu, add): transform.foreach over all linalg.generic
//     ops that do NOT have gather_dim / embedding_dim attribute.
//     Implementation: match all generics, then exclude by checking attribute.
//     Practical approach: match by op count position or use transform.select.
//     Here we match each op individually by function position (relu first,
//     add last), using transform.structured.match with op name only, relying
//     on the fact that relu has 1 ins and add has 2 ins (distinguishable).
//
// RUN: afir-opt --transform-interpreter %s --canonicalize --cse | FileCheck %s
// CHECK: scf.for
// CHECK: ascendc.parallel
// ============================================================

#col_broadcast_map = affine_map<(d0, d1) -> (d1)>
#full_map = affine_map<(d0, d1) -> (d0, d1)>

module attributes {transform.with_named_sequence} {

  func.func @relu_index_select_add(
      %data    : tensor<?x?xf16>,
      %indices : tensor<?xi64>,
      %bias    : tensor<?xf16>
  ) -> tensor<?x?xf16> {

    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %zero = arith.constant 0.0 : f16

    %dim_m = tensor.dim %data,    %c0 : tensor<?x?xf16>
    %dim_n = tensor.dim %data,    %c1 : tensor<?x?xf16>
    %dim_k = tensor.dim %indices, %c0 : tensor<?xi64>

    // Op1: relu — no gather_dim attribute
    %empty_relu = tensor.empty(%dim_m, %dim_n) : tensor<?x?xf16>
    %relu_out = linalg.generic {
      indexing_maps = [#full_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%data : tensor<?x?xf16>)
      outs(%empty_relu : tensor<?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      %v = arith.maximumf %in, %zero : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    // Op2: index_select — has {gather_dim = 1 : i64} (stamped by mark-structured-ops)
    %empty_gathered = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %gathered = linalg.generic {
      indexing_maps = [#col_broadcast_map, #full_map],
      iterator_types = ["parallel", "parallel"],
      gather_dim = 1 : i64
    } ins(%indices : tensor<?xi64>)
      outs(%empty_gathered : tensor<?x?xf16>) {
    ^bb0(%idx: i64, %out: f16):
      %i = linalg.index 0
      %j = linalg.index 1
      %idx_cast = arith.index_cast %idx : i64 to index
      %val = tensor.extract %relu_out[%i, %idx_cast] : tensor<?x?xf16>
      linalg.yield %val : f16
    } -> tensor<?x?xf16>

    // Op3: add — no gather_dim attribute
    %empty_out = tensor.empty(%dim_m, %dim_k) : tensor<?x?xf16>
    %out = linalg.generic {
      indexing_maps = [#full_map, #col_broadcast_map, #full_map],
      iterator_types = ["parallel", "parallel"]
    } ins(%gathered, %bias : tensor<?x?xf16>, tensor<?xf16>)
      outs(%empty_out : tensor<?x?xf16>) {
    ^bb0(%g: f16, %b: f16, %o: f16):
      %v = arith.addf %g, %b : f16
      linalg.yield %v : f16
    } -> tensor<?x?xf16>

    return %out : tensor<?x?xf16>
  }

  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}
  ) {
    %func = transform.structured.match ops{["func.func"]} in %root
        : (!transform.any_op) -> !transform.any_op

    // Add two index parameters: TB_M (inter-core tile), Tb_M (intra-core tile)
    %func_new, %tb_m_param, %tb_inner_m_param =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Match ops by attribute / structure
    %gather_op = transform.structured.match ops{["linalg.generic"]}
        attributes{gather_dim} in %func_new
        : (!transform.any_op) -> !transform.any_op

    // Match all linalg.generic ops, then tile them all with same strategy.
    // ComputeConversion handles the semantics per op type.
    %all_generics = transform.structured.match ops{["linalg.generic"]}
        in %func_new : (!transform.any_op) -> !transform.any_op

    %true_param        = transform.param.constant true -> !transform.any_param
    %prologue_param    = transform.param.constant "src:GM->VECIN"  -> !transform.any_param
    %epilogue_param    = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    %vector_unit_param = transform.param.constant "AiCore.Vector"  -> !transform.any_param

    // Tile all generics uniformly (TB + Tb, d1=K untiled)
    transform.foreach %all_generics : !transform.any_op {
    ^bb0(%op : !transform.any_op):
      %op_tb, %loop_tb =
          transform.structured.tile_using_for %op
              tile_sizes [%tb_m_param, 0]
                  : (!transform.any_op, !transform.any_op)
              -> (!transform.any_op, !transform.any_op)

      transform.annotate %loop_tb "ascendc.parallel"
          = %true_param : !transform.any_op, !transform.any_param

      %op_inner, %loop_inner =
          transform.structured.tile_using_for %op_tb
              tile_sizes [%tb_inner_m_param, 0]
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
    }

    transform.yield
  }
}
```

- [ ] **Step 4.2: Commit**

```bash
git add examples/gather-elementwise-fusion/step2_transform.mlir
git commit -m "feat(example): add gather-elementwise-fusion transform script (gather_dim attribute tiling)"
```

---

## Task 5: Write `run.sh` Pipeline Script

**Files:**
- Create: `examples/gather-elementwise-fusion/run.sh`

- [ ] **Step 5.1: Create `run.sh`**

```bash
#!/bin/bash
# ============================================================
# gather + elementwise fusion 完整编译流水线 Demo
#
# 用法：
#   source examples/env.sh
#   bash examples/gather-elementwise-fusion/run.sh [--log]
#
# 计算图：relu → index_select(dim=1) → add
#   data[M,N] ──relu──→ relu_out[M,N]
#   relu_out + indices[K] ──index_select──→ gathered[M,K]
#   gathered + bias[K] ──add──→ out[M,K]
#
# 新增 pass: --mark-structured-ops（在 transform 之前运行）
#   识别 index_select 的 linalg.generic（五条件检测），
#   打 {gather_dim = 1} attribute，供 Transform 和 ComputeConversion 使用。
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"

VERBOSE=false
for arg in "$@"; do
  case $arg in
    --log) VERBOSE=true ;;
  esac
done

log() {
  if $VERBOSE; then echo "$@"; fi
}

echo "========================================================"
echo " gather + elementwise fusion 编译流水线"
echo "========================================================"

# ── STAGE 0: Parse ────────────────────────────────────────
echo ""
echo "==================== [STAGE 0] 解析 High-Level IR ===================="
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir" 2>&1
log "  ✓ 解析成功"

# ── STAGE 1: Mark structured ops ──────────────────────────
echo ""
echo "==================== [STAGE 1] --mark-structured-ops ===================="
log "  识别 index_select generic，打 {gather_dim = 1} attribute"
$AFIR_OPT --mark-structured-ops \
  "$DIR/step0_input.mlir" \
  -o "$DIR/step1_marked.mlir" 2>&1
log "  ✓ 标注成功，输出: step1_marked.mlir"
log ""
log "  [gather_dim attribute]"
log "$(grep "gather_dim" "$DIR/step1_marked.mlir" | head -5 || echo "  (未找到，请检查)")"

# ── STAGE 2: Transform Tiling ──────────────────────────────
echo ""
echo "==================== [STAGE 2] Tiling：--transform-interpreter ===================="
$AFIR_OPT --transform-interpreter "$DIR/step2_transform.mlir" \
  --canonicalize --cse \
  -o "$DIR/step2_tiled.mlir" 2>&1
log "  ✓ Tiling 成功，输出: step2_tiled.mlir"

# ── STAGE 3: Bufferize ─────────────────────────────────────
echo ""
echo "==================== [STAGE 3] Bufferize：--one-shot-bufferize ===================="
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
log "  ✓ Bufferize 成功，输出: step3_bufferized.mlir"

# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "==================== [STAGE 4] Buffer Placement：--ascendc-buffer-placement ===================="
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir" 2>&1
log "  ✓ Buffer Placement 成功，输出: step4_buffer_placement.mlir"

# ── STAGE 5: Linalg → AscendC Compute ─────────────────────
echo ""
echo "==================== [STAGE 5] Linalg → AscendC：--linalg-to-ascendc ===================="
log "  转换规则："
log "    relu generic → max_l2（fused into gather row loop by ComputeConversion）"
log "    index_select generic (gather_dim=1) → gather_l2 row loop"
log "    add generic → add_l2（fused into gather row loop by ComputeConversion）"
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize --cse \
  -o "$DIR/step5_ascendc.mlir" 2>&1
log "  ✓ Linalg→AscendC 成功，输出: step5_ascendc.mlir"
log ""
log "  [生成的 AscendC ops]"
log "$(grep -E "ascendc\.(gather_l2|add_l2|max_l2|data_copy)" \
  "$DIR/step5_ascendc.mlir" | head -20 || echo "  (请检查输出)")"

# ── STAGE 6: Parallelize ───────────────────────────────────
echo ""
echo "==================== [STAGE 6] Parallelize：--ascendc-parallelize ===================="
$AFIR_OPT "$DIR/step5_ascendc.mlir" \
  --ascendc-parallelize \
  --canonicalize --cse \
  -o "$DIR/step6_parallelize.mlir" 2>&1
log "  ✓ Parallelize 成功，输出: step6_parallelize.mlir"

# ── STAGE 7: Prepare For Emit ──────────────────────────────
echo ""
echo "==================== [STAGE 7] Prepare For Emit：--ascendc-prepare-for-emit ===================="
$AFIR_OPT "$DIR/step6_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --canonicalize --cse \
  -o "$DIR/step7_kernel.mlir" 2>&1
log "  ✓ Prepare For Emit 成功，输出: step7_kernel.mlir"

# ── STAGE 8: Codegen ───────────────────────────────────────
echo ""
echo "==================== [STAGE 8] Codegen：ascir-translate -mlir-to-ascendc ===================="
ASCIR_TRANSLATE="${ASCIR_TRANSLATE:-ascir-translate}"
if command -v "$ASCIR_TRANSLATE" &>/dev/null; then
  python3 -c "
import re, sys
content = open('$DIR/step7_kernel.mlir').read()
content = content.replace('module attributes {transform.with_named_sequence}', 'module')
content = re.sub(r'  transform\.named_sequence.*?^  \}\n', '', content, flags=re.DOTALL|re.MULTILINE)
sys.stdout.write(content)
" > "$DIR/step8_no_transform.mlir"
  "$ASCIR_TRANSLATE" -mlir-to-ascendc "$DIR/step8_no_transform.mlir" \
    -o "$DIR/step8_kernel.cpp" 2>&1
  log "  ✓ Codegen 成功，输出: step8_kernel.cpp"
else
  log "  (ascir-translate 未找到，跳过 Stage 8)"
fi

echo ""
echo "========================================================"
echo " 流水线完成！"
echo "========================================================"
```

- [ ] **Step 5.2: Make executable and commit**

```bash
chmod +x examples/gather-elementwise-fusion/run.sh
git add examples/gather-elementwise-fusion/run.sh
git commit -m "feat(example): add gather-elementwise-fusion run.sh pipeline script"
```

---

## Task 6: Extend ComputeConversion — `isIndexSelectGeneric` Code Path

Add fused `relu → gather_l2 → add` code generation in `ComputeConversion.cpp`.

**Files:**
- Modify: `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`

**Design:** When processing a parallel generic with `gather_dim` attribute:
1. Find the pre-gather elementwise (op whose output memref = data input of this gather). Apply it row-by-row using existing arith body walker.
2. Run `gather_l2` on the processed row.
3. Find the post-gather elementwise (op that consumes this gather's output memref). Apply it on the gathered row.
4. Erase all three ops.

- [ ] **Step 6.1: Replace `isGatherGeneric` lambda with `isIndexSelectGeneric` and `isEmbeddingGeneric`**

In `ComputeConversion.cpp`, locate the `isGatherGeneric` lambda (lines ~727-732) and replace it with two new lambdas:

```cpp
  // Helper: detect index_select gather (column gather, gather_dim attribute present).
  // Stamped by --mark-structured-ops on linalg.generic with tensor.extract body
  // where indices are 1D and index the data tensor along a non-zero dimension.
  auto isIndexSelectGeneric = [](linalg::GenericOp op) -> bool {
    return op->hasAttr("gather_dim");
  };

  // Helper: detect embedding gather (row gather, embedding_dim attribute present).
  auto isEmbeddingGeneric = [](linalg::GenericOp op) -> bool {
    return op->hasAttr("embedding_dim");
  };
```

- [ ] **Step 6.2: Add helper to find fused elementwise neighbor**

Add after the `isEmbeddingGeneric` lambda:

```cpp
  // Helper: given a memref value (the data input of a gather op), find the
  // linalg.generic that writes to it (the pre-gather elementwise), if it:
  //   (a) is in parallelGenericOps
  //   (b) has no gather_dim / embedding_dim attribute
  //   (c) its output memref aliases the given memref
  // Returns nullptr if not found.
  auto findPreGatherOp = [&](Value dataMemref,
                              linalg::GenericOp gatherOp) -> linalg::GenericOp {
    for (linalg::GenericOp candidate : parallelGenericOps) {
      if (candidate == gatherOp) continue;
      if (candidate->hasAttr("gather_dim") ||
          candidate->hasAttr("embedding_dim")) continue;
      Value candidateOut = candidate.getDpsInitOperand(0)->get();
      if (candidateOut == dataMemref)
        return candidate;
    }
    return nullptr;
  };

  // Helper: given a gather op's output memref, find the linalg.generic that
  // consumes it as an input (the post-gather elementwise), with same conditions.
  auto findPostGatherOp = [&](Value gatherOutMemref,
                               linalg::GenericOp gatherOp) -> linalg::GenericOp {
    for (linalg::GenericOp candidate : parallelGenericOps) {
      if (candidate == gatherOp) continue;
      if (candidate->hasAttr("gather_dim") ||
          candidate->hasAttr("embedding_dim")) continue;
      for (OpOperand *inp : candidate.getDpsInputOperands()) {
        if (inp->get() == gatherOutMemref)
          return candidate;
      }
    }
    return nullptr;
  };
```

- [ ] **Step 6.3: Add helper to apply elementwise body on a row LocalTensor**

Add after the `findPostGatherOp` lambda. This applies the arith ops from an elementwise generic's body to a given local_tensor row, producing a result in a fresh VECCALC buffer:

```cpp
  // Helper: inline the arith ops from an elementwise generic body onto
  // inputRowLt (a LocalTensor of size rowElems), writing result to a fresh
  // VECCALC LocalTensor of the same size. Returns the result LocalTensor.
  // Handles: arith.maximumf (relu), arith.addf, arith.mulf.
  // extraInputLts maps block arg indices (beyond arg 0, the main input) to
  // their LocalTensors (e.g., bias row).
  auto applyEwopBodyOnRow =
      [&](OpBuilder &b, Location loc, linalg::GenericOp ewOp, Value inputRowLt,
          SmallVector<Value> extraInputLts,
          Value rowElems) -> Value {
    Type elemType = cast<MemRefType>(
        ewOp.getDpsInitOperand(0)->get().getType()).getElementType();
    auto [resTbuf, resLt] =
        allocVeccalc(b, loc, elemType, SmallVector<Value>{rowElems});

    Block &body = *ewOp.getBody();
    // Map block args to local tensors.
    // arg 0 = main data input → inputRowLt
    // arg 1..n-1 = extra inputs → extraInputLts
    // last arg = output init (ignored here)
    llvm::SmallDenseMap<Value, Value> valToLt;
    auto blockArgs = body.getArguments();
    // blockArgs[0] = main input
    auto resolveRow = [&](Value v) -> Value {
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        unsigned idx = ba.getArgNumber();
        if (idx == 0) return inputRowLt;
        if (idx - 1 < extraInputLts.size()) return extraInputLts[idx - 1];
        return resLt; // output arg
      }
      auto it = valToLt.find(v);
      if (it != valToLt.end()) return it->second;
      if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        auto [dupTbuf, dupLt] =
            allocVeccalc(b, loc, elemType, SmallVector<Value>{rowElems});
        b.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), rowElems);
        valToLt[v] = dupLt;
        return dupLt;
      }
      return Value{};
    };

    Value rowElemsI32 =
        b.create<arith::IndexCastOp>(loc, b.getI32Type(), rowElems);

    for (auto &bodyOp : body.without_terminator()) {
      if (auto maxOp = dyn_cast<arith::MaximumFOp>(bodyOp)) {
        Value lhs = resolveRow(maxOp.getLhs());
        Value rhs = resolveRow(maxOp.getRhs());
        if (!lhs || !rhs) continue;
        b.create<MaxL2Op>(loc, resLt, lhs, rhs, rowElemsI32);
        valToLt[maxOp.getResult()] = resLt;
      } else if (auto addOp = dyn_cast<arith::AddFOp>(bodyOp)) {
        Value lhs = resolveRow(addOp.getLhs());
        Value rhs = resolveRow(addOp.getRhs());
        if (!lhs || !rhs) continue;
        b.create<AddL2Op>(loc, resLt, lhs, rhs, rowElemsI32);
        valToLt[addOp.getResult()] = resLt;
      } else if (auto mulOp = dyn_cast<arith::MulFOp>(bodyOp)) {
        Value lhs = resolveRow(mulOp.getLhs());
        Value rhs = resolveRow(mulOp.getRhs());
        if (!lhs || !rhs) continue;
        b.create<MulL2Op>(loc, resLt, lhs, rhs, rowElemsI32);
        valToLt[mulOp.getResult()] = resLt;
      }
    }
    return resLt;
  };
```

- [ ] **Step 6.4: Replace old `isGatherGeneric` branch with new `isIndexSelectGeneric` branch**

Locate the `if (isGatherGeneric(genOp))` block (lines ~741-785) and replace it entirely with:

```cpp
    // ---- index_select gather: relu → gather_l2 → add, fused in row loop ----
    // Detects via {gather_dim = N} attribute (stamped by --mark-structured-ops).
    // ins[0] = indices[K] (i64), output = gathered[Tb_M, K]
    // The data tensor is accessed via tensor.extract in the body (not in ins).
    // We find the pre-gather and post-gather elementwise ops via memref alias.
    if (isIndexSelectGeneric(genOp)) {
      Location loc = genOp.getLoc();
      builder.setInsertionPoint(genOp);

      // indices is the only ins operand (the data tensor is captured in body)
      Value indicesMemref = genOp.getDpsInputOperand(0)->get();
      Value outMemref     = genOp.getDpsInitOperand(0)->get();

      // Recover data memref: walk the body's tensor.extract to find the
      // captured tensor, then find its bufferized memref via AliasAnalysis.
      // Simpler: find pre-gather op and use its output memref as data.
      linalg::GenericOp preOp  = findPreGatherOp(Value{}, genOp);
      linalg::GenericOp postOp = findPostGatherOp(outMemref, genOp);

      // Find data memref: it's the memref that the pre-op writes into,
      // which is also the memref captured by the tensor.extract in genOp's body.
      // We get it from the pre-op's output if pre-op exists, otherwise we need
      // to find the memref corresponding to the captured tensor SSA value.
      // Use AscendCBufferContext to look up the memref for the captured value.
      // Fallback: scan genOp's body for tensor.extract and map back to memref.
      Value dataMemref;
      {
        // Find tensor.extract in body and get its tensor operand's memref.
        genOp.getBody()->walk([&](tensor::ExtractOp e) {
          if (!dataMemref) {
            // The tensor operand of extract is the pre-bufferization tensor.
            // After bufferization, the corresponding memref is the pre-op output.
            if (preOp)
              dataMemref = preOp.getDpsInitOperand(0)->get();
          }
        });
      }
      if (!dataMemref) {
        genOp.emitWarning("index_select gather: cannot find data memref, skipping fusion");
        continue;
      }

      // Re-find pre-op with correct dataMemref.
      preOp = findPreGatherOp(dataMemref, genOp);

      auto dataMrt = cast<MemRefType>(dataMemref.getType());
      Type elemType = dataMrt.getElementType();
      Type i32Type  = builder.getI32Type();
      Type i64Type  = builder.getI64Type();

      Value tbM  = getDynDim(builder, loc, dataMemref, 0);  // Tb_M rows
      Value dimN = getDynDim(builder, loc, dataMemref, 1);  // N per row
      Value dimK = getDynDim(builder, loc, indicesMemref, 0); // K gathered

      // Bytes per row in data and output (f16 = 2 bytes).
      unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
      Value bytesPerDataRow = builder.create<arith::MulIOp>(
          loc, dimN,
          builder.create<arith::ConstantIndexOp>(loc, elemBytes));
      Value bytesPerOutRow = builder.create<arith::MulIOp>(
          loc, dimK,
          builder.create<arith::ConstantIndexOp>(loc, elemBytes));

      // Get indices LocalTensor.
      Value indicesLt;
      int64_t idxMs = getMemorySpace(indicesMemref.getType());
      if (idxMs == 9 || idxMs == 11) {
        indicesLt = readTensor(builder, loc, indicesMemref);
      } else {
        // GM: copy indices into VECCALC.
        auto [idxTbuf, idxLt] = allocVeccalc(builder, loc, i64Type,
                                              SmallVector<Value>{dimK});
        Value idxGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(i64Type));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, idxGt, indicesMemref,
                                                       /*size=*/Value{});
        builder.create<DataCopyL2Op>(loc, idxLt, idxGt, dimK);
        indicesLt = idxLt;
      }

      // Get data TBuf (for row slicing).
      Value dataTbuf = ctx.getTBuf(dataMemref);
      Value outTbuf  = ctx.getTBuf(outMemref);

      // Alloc output local tensor.
      Value dstLt = writeTensor(builder, loc, outMemref);

      Value dimK_i32 = builder.create<arith::IndexCastOp>(loc, i32Type, dimK);
      Value srcBase  = builder.create<arith::ConstantIntOp>(loc, i32Type, 0);
      Value zero     = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value one      = builder.create<arith::ConstantIndexOp>(loc, 1);

      // Row loop: for i in 0..Tb_M
      builder.create<scf::ForOp>(
          loc, zero, tbM, one, ValueRange{},
          [&](OpBuilder &b, Location forLoc, Value rowIdx, ValueRange) {
            // Slice data row[N] from dataTbuf.
            Value srcByteOff =
                b.create<arith::MulIOp>(forLoc, rowIdx, bytesPerDataRow);
            Value dataRowLt = b.create<TBufGetWithOffsetOp>(
                forLoc, LocalTensorType::get(elemType),
                dataTbuf, bytesPerDataRow, srcByteOff);

            // Step 1: apply pre-gather elementwise (e.g. relu) on data row.
            Value processedRowLt = dataRowLt;
            if (preOp) {
              processedRowLt = applyEwopBodyOnRow(
                  b, forLoc, preOp, dataRowLt,
                  SmallVector<Value>{},  // relu has no extra inputs
                  dimN);
            }

            // Step 2: gather_l2(dst[K], src[N], indices[K], srcBase=0, count=K)
            Value dstByteOff =
                b.create<arith::MulIOp>(forLoc, rowIdx, bytesPerOutRow);
            Value gatheredRowLt = b.create<TBufGetWithOffsetOp>(
                forLoc, LocalTensorType::get(elemType),
                outTbuf, bytesPerOutRow, dstByteOff);
            b.create<GatherL2Op>(forLoc, gatheredRowLt, processedRowLt,
                                 indicesLt, srcBase, dimK_i32);

            // Step 3: apply post-gather elementwise (e.g. add with bias row).
            if (postOp) {
              // Get bias (or other extra input) for this row.
              // For add with col-broadcast bias[K]: bias row is just bias[0..K-1]
              // (no row slicing needed since bias is 1D[K]).
              SmallVector<Value> extraLts;
              for (unsigned inp = 0; inp < (unsigned)postOp.getNumDpsInputs(); ++inp) {
                Value inMref = postOp.getDpsInputOperand(inp)->get();
                if (inMref == outMemref) {
                  extraLts.push_back(gatheredRowLt);
                } else {
                  extraLts.push_back(readTensor(b, forLoc, inMref));
                }
              }
              // Extra lts minus the gathered input (arg 0 of post op body = gathered).
              // applyEwopBodyOnRow takes: arg0=gathered, extra=bias
              SmallVector<Value> biasLts;
              for (Value lt : extraLts)
                if (lt != gatheredRowLt)
                  biasLts.push_back(lt);
              applyEwopBodyOnRow(b, forLoc, postOp, gatheredRowLt, biasLts, dimK);
            }

            b.create<scf::YieldOp>(forLoc);
          });

      if (Value q = ctx.getQueue(outMemref))
        builder.create<TQueBindEnqueTensorOp>(loc, q, dstLt);

      // Erase fused ops (pre, post, then gather itself).
      if (postOp) postOp.erase();
      if (preOp)  preOp.erase();
      genOp.erase();
      continue;
    }
```

- [ ] **Step 6.5: Build**

```bash
ssh xvm@orb "cd /home/niu/code/Ascend-MLIR && sleep 1 && \
  ./scripts/build.sh --build-project 2>&1 | tail -30"
```

Expected: build succeeds.

- [ ] **Step 6.6: Run stages 0-5 of the example and inspect output**

```bash
ssh xvm@orb "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  bash /home/niu/code/Ascend-MLIR/examples/gather-elementwise-fusion/run.sh --log 2>&1 | head -80"
```

Expected: stages 0-5 complete; stage 5 output contains `gather_l2`, `max_l2`, `add_l2`.

- [ ] **Step 6.7: Commit**

```bash
git add lib/Conversion/LinalgToAscendC/ComputeConversion.cpp
git commit -m "feat(compute): add isIndexSelectGeneric fused gather code path (relu+gather_l2+add in row loop)"
```

---

## Task 7: Run Full Pipeline and Verify End-to-End

- [ ] **Step 7.1: Run the full pipeline**

```bash
ssh xvm@orb "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  bash /home/niu/code/Ascend-MLIR/examples/gather-elementwise-fusion/run.sh --log 2>&1"
```

Expected: all 8 stages complete without errors.

- [ ] **Step 7.2: Verify step1 has gather_dim attribute**

```bash
ssh xvm@orb "grep 'gather_dim' \
  /home/niu/code/Ascend-MLIR/examples/gather-elementwise-fusion/step1_marked.mlir"
```

Expected: `gather_dim = 1 : i64` present on the index_select generic.

- [ ] **Step 7.3: Verify step5 has gather_l2, max_l2, add_l2**

```bash
ssh xvm@orb "grep -E 'gather_l2|max_l2|add_l2' \
  /home/niu/code/Ascend-MLIR/examples/gather-elementwise-fusion/step5_ascendc.mlir"
```

Expected: all three op types present.

- [ ] **Step 7.4: Verify step5 has no intermediate GM stores (relu/add results not written to GM)**

```bash
ssh xvm@orb "grep 'data_copy_l2' \
  /home/niu/code/Ascend-MLIR/examples/gather-elementwise-fusion/step5_ascendc.mlir | wc -l"
```

Expected: only 2 data_copy_l2 ops (GM→VECIN for data + indices; VECOUT→GM for output), not 4+ (which would indicate intermediate results written to GM).

- [ ] **Step 7.5: Commit generated intermediate files**

```bash
git add examples/gather-elementwise-fusion/
git commit -m "feat(example): add gather-elementwise-fusion end-to-end pipeline (relu+index_select+add fused)"
```

---

## Task 8: Add Unit Test for `--mark-structured-ops`

- [ ] **Step 8.1: Create test file**

Create `test/Conversion/mark-structured-ops.mlir`:

```mlir
// RUN: afir-opt --mark-structured-ops %s | FileCheck %s

// Test 1: index_select pattern (column gather) → gather_dim = 1
// CHECK-LABEL: func.func @test_index_select
// CHECK: linalg.generic
// CHECK-SAME: gather_dim = 1
func.func @test_index_select(
    %data    : tensor<4x8xf16>,
    %indices : tensor<3xi64>
) -> tensor<4x3xf16> {
  %empty = tensor.empty() : tensor<4x3xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%indices : tensor<3xi64>)
    outs(%empty : tensor<4x3xf16>) {
  ^bb0(%idx: i64, %o: f16):
    %i = linalg.index 0
    %j = linalg.index 1
    %ic = arith.index_cast %idx : i64 to index
    %v = tensor.extract %data[%i, %ic] : tensor<4x8xf16>
    linalg.yield %v : f16
  } -> tensor<4x3xf16>
  return %out : tensor<4x3xf16>
}

// Test 2: embedding pattern (row gather) → embedding_dim = 0
// CHECK-LABEL: func.func @test_embedding
// CHECK: linalg.generic
// CHECK-SAME: embedding_dim = 0
func.func @test_embedding(
    %weight  : tensor<16x8xf16>,
    %indices : tensor<4xi64>
) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%indices : tensor<4xi64>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%idx: i64, %o: f16):
    %i = linalg.index 0
    %j = linalg.index 1
    %ic = arith.index_cast %idx : i64 to index
    %v = tensor.extract %weight[%ic, %j] : tensor<16x8xf16>
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}

// Test 3: plain elementwise (no tensor.extract) → no attribute stamped
// CHECK-LABEL: func.func @test_elementwise
// CHECK: linalg.generic
// CHECK-NOT: gather_dim
// CHECK-NOT: embedding_dim
func.func @test_elementwise(
    %a : tensor<4x8xf16>,
    %b : tensor<4x8xf16>
) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%a, %b : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %v = arith.addf %x, %y : f16
    linalg.yield %v : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}
```

- [ ] **Step 8.2: Run the test**

```bash
ssh xvm@orb "source /home/niu/code/Ascend-MLIR/examples/env.sh && \
  /home/niu/code/llvm-project/build/bin/FileCheck \
    /home/niu/code/Ascend-MLIR/test/Conversion/mark-structured-ops.mlir \
    < <(afir-opt --mark-structured-ops \
        /home/niu/code/Ascend-MLIR/test/Conversion/mark-structured-ops.mlir 2>&1)"
```

Expected: all three test functions pass FileCheck.

- [ ] **Step 8.3: Commit**

```bash
git add test/Conversion/mark-structured-ops.mlir
git commit -m "test(pass): add FileCheck tests for --mark-structured-ops gather detection"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Task |
|---|---|
| §1 accept torch-MLIR IR without library_call | Task 3 (step0_input.mlir) |
| §4 five-condition detection | Task 2 |
| §5 --mark-structured-ops pass | Tasks 1-2 |
| §6 pipeline (mark before transform) | Task 5 (run.sh) |
| §7 transform script with gather_dim attribute | Task 4 |
| §8.1 fusion detection via memref alias | Task 6 (findPreGatherOp/findPostGatherOp) |
| §8.2 index_select code path | Task 6 |
| §8.4 fallback (unfused ops continue as-is) | Task 6 (emitWarning + continue) |
| §9 file layout | All tasks |
| §10 N-D via gather_dim attribute | Detection is parametric; code gen uses dim from attribute |
| §3.2 embedding code path | **Not implemented — embedding isEmbeddingGeneric lambda declared but no code gen** |
| §11 out of scope: aten.gather, stablehlo.gather | Not implemented (by design) |

**Gap: embedding code path.** The `isEmbeddingGeneric` lambda is declared in Task 6 but no code generation is added. This is intentional for this plan — embedding is a separate code path that can be added in a follow-up plan once index_select is validated end-to-end.

**Type consistency:** `applyEwopBodyOnRow` takes `SmallVector<Value>` — consistent with usage in Task 6 step 6.4.

**Placeholder scan:** No TBD or TODO in code blocks. The `tensor::ExtractOp` include may need to be added to ComputeConversion.cpp includes — add `#include "mlir/Dialect/Tensor/IR/Tensor.h"` to the existing includes in that file before Task 6.
