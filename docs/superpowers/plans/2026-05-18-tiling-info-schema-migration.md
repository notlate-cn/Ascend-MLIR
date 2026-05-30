# Tiling-Info Schema Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the implicit, three-way tiling-info contract (PackTilingData heuristic / NetworkJsonEmitter input-only args / hand-written run.sh TILING_PARAMS) with a single canonical schema authored in `vector_plan.tiling_infos`. Removes `PackTilingData::argToOriginal` and makes `transpose-elementwise-e2e` PASS.

**Architecture:** TilePlanGen + GroupOutline write `schema_version=2` entries into `vector_plan.tiling_infos`, covering both `fields[]` (TilingData layout, with shape-derived sources by **MLIR** arg numbering) and `args[]` (per-MLIR-arg provenance). Downstream consumers (PackTilingData, CannTranslation, NetworkJsonEmitter, network_runner) switch to schema-driven behavior phase by phase. Old paths stay alive until phase S5 deletes them. See `docs/superpowers/specs/2026-05-18-tiling-info-schema-design.md`.

**Tech Stack:** MLIR (C++), Python (`network_runner`).

**Pre-req:** Plan `2026-05-18-bcast-bare-collapse-fix.md` should land first so bcast e2e gates are green before schema phases touch the same kernels.

---

## File Structure

**MLIR sources to modify:**
- `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` — extend `emitTilingInfos` (around line 818) to also emit shape-derived `fields[]` entries and the `args[]` sub-table; bump `schema_version`.
- `lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp` — annotate each outlined kernel with the call-site provenance needed by `args[]` (input network_index, output result_index).
- `include/Conversion/VectorPlan/TilePlan.h` — helper struct + serialization API for the schema.
- `lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp` — read schema when `schema_version=2`, else legacy path.
- `lib/Target/CannKernel/CannTranslation.cpp` — schema-driven `emitTilingSpaceJson` (translate `source_arg` MLIR→network indices when writing `shape_key`).
- `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp` — schema-driven kernel `args[]` (covers output/workspace/tile-param).

**Python:**
- `python/network_runner.py` — `_resolve_kernel_input_shape` + `_shape_key_values_for_kernel` resolve via schema args (or evaluate `shape_expr` for output-derived keys).

**Tests:**
- New: `test/Conversion/VectorPlan/TilePlanGen-schema.mlir` — assert v2 schema attrs.
- New: `test/Conversion/AscendCPrepareForEmit/pack-tiling-data-schema.mlir` — assert PackTilingData consumes schema correctly.
- Modify: `test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir` — keep one legacy v1 case until S5.
- e2e gates: `examples/transpose-elementwise-e2e/run.sh` (must transition from FAIL to PASS at S2), full e2e gate suite.

**Documentation:**
- The schema attr shape is documented inline in `TilePlan.h` and in the design spec.

---

## Phase S1: Author schema in TilePlanGen + GroupOutline

### Task 1.1: Define schema attr shape and helpers in `TilePlan.h`

**Files:**
- Modify: `include/Conversion/VectorPlan/TilePlan.h`

- [ ] **Step 1: Add schema data structures**

Append to `include/Conversion/VectorPlan/TilePlan.h` (before the final `}` of the namespace):

```cpp
// ──── Tiling-info schema v2 ──────────────────────────────────────────────
// One TilingInfoSchema per kernel func.  Serialized into the
// `vector_plan.tiling_infos` ModuleOp attribute (as a DictAttr), read by
// PackTilingData / CannTranslation / NetworkJsonEmitter / runner.
//
// See docs/superpowers/specs/2026-05-18-tiling-info-schema-design.md.

enum class SchemaFieldKind { Tunable, ShapeDerived };

struct SchemaField {
  std::string name;
  SchemaFieldKind kind;
  // Tunable: axis size (-1 dyn) + default + arg_index (the MLIR arg holding
  // the index-typed tile-param value injected by TilePlanGen).
  int64_t axisSize = -1;
  int64_t defaultValue = 0;
  int32_t argIndex = -1;  // MLIR arg index for the tile-param SSA value
  // ShapeDerived: which (MLIR arg, dim) this field's value comes from.
  int32_t sourceArg = -1;
  int32_t sourceDim = -1;
};

enum class SchemaArgRole {
  Input,
  Output,
  TileParam,
  Workspace,
  TilingDataStruct,
};

struct SchemaArg {
  int32_t mlirIndex;            // position in the kernel func signature
  SchemaArgRole role;
  // Input: network_index in coordinator-call operand list.
  int32_t networkIndex = -1;
  // Output: result_index in the kernel's `results` array + shape_expr per
  // output dim (each entry is a host-evaluable string like "arg0_dim1").
  int32_t resultIndex = -1;
  llvm::SmallVector<std::string, 4> shapeExpr;
  // TileParam: the field name this arg holds.
  std::string tileParamName;
};

struct TilingInfoSchema {
  static constexpr int kSchemaVersion = 2;
  std::string kernelId;
  std::string blockDimExpr;
  std::string axisExtentExpr;
  llvm::SmallVector<SchemaField, 8> fields;
  llvm::SmallVector<SchemaArg, 8> args;
  // Constraints: reuse the existing {kind, lhs, rhs} struct.  Carried through
  // unchanged.
};

// Serialize a TilingInfoSchema to a DictionaryAttr suitable for embedding in
// `vector_plan.tiling_infos`.  Round-trips with deserialize().
mlir::DictionaryAttr serializeTilingInfoSchema(
    mlir::MLIRContext *ctx, const TilingInfoSchema &s,
    mlir::ArrayAttr constraintsAttr);

// Decode a vector_plan.tiling_infos entry into a TilingInfoSchema.  Returns
// std::nullopt when the entry is not v2 (caller must fall back to legacy).
std::optional<TilingInfoSchema> deserializeTilingInfoSchema(
    mlir::DictionaryAttr entry);

// Look up the schema entry for a given kernel func from a module's
// vector_plan.tiling_infos attr.  Returns std::nullopt when absent or v1.
std::optional<TilingInfoSchema> lookupTilingInfoSchema(
    mlir::ModuleOp moduleOp, llvm::StringRef kernelName);
```

- [ ] **Step 2: Implement the three serializer/deserializer helpers in TilePlanGen.cpp**

Add to `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` near the top (after existing includes), before `emitTilingInfos`:

```cpp
namespace mlir::vector_plan {

static StringRef roleToString(SchemaArgRole r) {
  switch (r) {
    case SchemaArgRole::Input:            return "input";
    case SchemaArgRole::Output:           return "output";
    case SchemaArgRole::TileParam:        return "tile_param";
    case SchemaArgRole::Workspace:        return "workspace";
    case SchemaArgRole::TilingDataStruct: return "tiling_data_struct";
  }
  llvm_unreachable("unknown role");
}

static std::optional<SchemaArgRole> roleFromString(StringRef s) {
  if (s == "input")              return SchemaArgRole::Input;
  if (s == "output")             return SchemaArgRole::Output;
  if (s == "tile_param")         return SchemaArgRole::TileParam;
  if (s == "workspace")          return SchemaArgRole::Workspace;
  if (s == "tiling_data_struct") return SchemaArgRole::TilingDataStruct;
  return std::nullopt;
}

DictionaryAttr serializeTilingInfoSchema(MLIRContext *ctx,
                                          const TilingInfoSchema &s,
                                          ArrayAttr constraintsAttr) {
  Type i32Ty = IntegerType::get(ctx, 32);
  Type i64Ty = IntegerType::get(ctx, 64);

  SmallVector<Attribute> fieldsAttr;
  for (const auto &f : s.fields) {
    NamedAttrList d;
    d.append("name", StringAttr::get(ctx, f.name));
    d.append("kind",
             StringAttr::get(ctx, f.kind == SchemaFieldKind::Tunable
                                       ? "tunable" : "shape_derived"));
    if (f.kind == SchemaFieldKind::Tunable) {
      d.append("axis_size",     IntegerAttr::get(i64Ty, f.axisSize));
      d.append("default_value", IntegerAttr::get(i64Ty, f.defaultValue));
      d.append("arg_index",     IntegerAttr::get(i32Ty, f.argIndex));
    } else {
      d.append("source_arg", IntegerAttr::get(i32Ty, f.sourceArg));
      d.append("source_dim", IntegerAttr::get(i32Ty, f.sourceDim));
    }
    fieldsAttr.push_back(d.getDictionary(ctx));
  }

  SmallVector<Attribute> argsAttr;
  for (const auto &a : s.args) {
    NamedAttrList d;
    d.append("mlir_index", IntegerAttr::get(i32Ty, a.mlirIndex));
    d.append("role",       StringAttr::get(ctx, roleToString(a.role)));
    if (a.role == SchemaArgRole::Input)
      d.append("network_index", IntegerAttr::get(i32Ty, a.networkIndex));
    if (a.role == SchemaArgRole::Output) {
      d.append("result_index", IntegerAttr::get(i32Ty, a.resultIndex));
      SmallVector<Attribute> exprs;
      for (auto &e : a.shapeExpr) exprs.push_back(StringAttr::get(ctx, e));
      d.append("shape_expr", ArrayAttr::get(ctx, exprs));
    }
    if (a.role == SchemaArgRole::TileParam)
      d.append("name", StringAttr::get(ctx, a.tileParamName));
    argsAttr.push_back(d.getDictionary(ctx));
  }

  NamedAttrList entry;
  entry.append("kernel_id",      StringAttr::get(ctx, s.kernelId));
  entry.append("schema_version", IntegerAttr::get(i32Ty,
                                                  TilingInfoSchema::kSchemaVersion));
  entry.append("fields", ArrayAttr::get(ctx, fieldsAttr));
  entry.append("args",   ArrayAttr::get(ctx, argsAttr));
  if (!s.blockDimExpr.empty())
    entry.append("block_dim_expr", StringAttr::get(ctx, s.blockDimExpr));
  if (!s.axisExtentExpr.empty())
    entry.append("axis_extent_expr", StringAttr::get(ctx, s.axisExtentExpr));
  if (constraintsAttr)
    entry.append("constraints", constraintsAttr);
  return entry.getDictionary(ctx);
}

std::optional<TilingInfoSchema> deserializeTilingInfoSchema(
    DictionaryAttr entry) {
  auto verAttr = entry.getAs<IntegerAttr>("schema_version");
  if (!verAttr || verAttr.getInt() != TilingInfoSchema::kSchemaVersion)
    return std::nullopt;
  TilingInfoSchema s;
  if (auto a = entry.getAs<StringAttr>("kernel_id"))      s.kernelId      = a.str();
  if (auto a = entry.getAs<StringAttr>("block_dim_expr")) s.blockDimExpr  = a.str();
  if (auto a = entry.getAs<StringAttr>("axis_extent_expr")) s.axisExtentExpr = a.str();

  if (auto arr = entry.getAs<ArrayAttr>("fields")) {
    for (Attribute fa : arr) {
      auto d = dyn_cast<DictionaryAttr>(fa);
      if (!d) continue;
      SchemaField f;
      f.name = d.getAs<StringAttr>("name").str();
      auto k = d.getAs<StringAttr>("kind").getValue();
      f.kind = (k == "tunable") ? SchemaFieldKind::Tunable
                                : SchemaFieldKind::ShapeDerived;
      if (f.kind == SchemaFieldKind::Tunable) {
        f.axisSize     = d.getAs<IntegerAttr>("axis_size").getInt();
        f.defaultValue = d.getAs<IntegerAttr>("default_value").getInt();
        f.argIndex     = (int32_t)d.getAs<IntegerAttr>("arg_index").getInt();
      } else {
        f.sourceArg = (int32_t)d.getAs<IntegerAttr>("source_arg").getInt();
        f.sourceDim = (int32_t)d.getAs<IntegerAttr>("source_dim").getInt();
      }
      s.fields.push_back(std::move(f));
    }
  }
  if (auto arr = entry.getAs<ArrayAttr>("args")) {
    for (Attribute aa : arr) {
      auto d = dyn_cast<DictionaryAttr>(aa);
      if (!d) continue;
      SchemaArg a;
      a.mlirIndex = (int32_t)d.getAs<IntegerAttr>("mlir_index").getInt();
      auto r = roleFromString(d.getAs<StringAttr>("role").getValue());
      if (!r) continue;
      a.role = *r;
      if (a.role == SchemaArgRole::Input)
        a.networkIndex = (int32_t)d.getAs<IntegerAttr>("network_index").getInt();
      if (a.role == SchemaArgRole::Output) {
        a.resultIndex = (int32_t)d.getAs<IntegerAttr>("result_index").getInt();
        if (auto se = d.getAs<ArrayAttr>("shape_expr"))
          for (Attribute e : se)
            a.shapeExpr.push_back(cast<StringAttr>(e).str());
      }
      if (a.role == SchemaArgRole::TileParam)
        a.tileParamName = d.getAs<StringAttr>("name").str();
      s.args.push_back(std::move(a));
    }
  }
  return s;
}

std::optional<TilingInfoSchema>
lookupTilingInfoSchema(ModuleOp moduleOp, StringRef kernelName) {
  auto arr = moduleOp->getAttrOfType<ArrayAttr>("vector_plan.tiling_infos");
  if (!arr) return std::nullopt;
  for (Attribute a : arr) {
    auto d = dyn_cast<DictionaryAttr>(a);
    if (!d) continue;
    auto kid = d.getAs<StringAttr>("kernel_id");
    if (!kid || kid.getValue() != kernelName) continue;
    return deserializeTilingInfoSchema(d);
  }
  return std::nullopt;
}

} // namespace mlir::vector_plan
```

- [ ] **Step 3: Build and confirm helpers compile**

```bash
cd build && ninja MLIRVectorPlanTileFuse 2>&1 | tail -3
```

Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
cd /home/gser/code/Ascend-MLIR
git add include/Conversion/VectorPlan/TilePlan.h \
        lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp
git commit -m "feat(vector-plan): add TilingInfoSchema v2 serializer/deserializer helpers

No behavior change.  Schema attrs are not emitted yet; this commit just
adds the API surface that the next task will call.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 1.2: Extend `emitTilingInfos` to write schema v2

**Files:**
- Modify: `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp:818-985` (the body of `emitTilingInfos`)

- [ ] **Step 1: Add a lit test that asserts v2 attrs are emitted (failing)**

Create `test/Conversion/VectorPlanCodegen/tiling-info-schema-v2.mlir`:

```mlir
// RUN: afir-opt %s --vector-plan-codegen 2>&1 | FileCheck %s

// Simple elementwise: shape-equal output, single input, single tunable axis.
// CHECK: vector_plan.tiling_infos
// CHECK-SAME: schema_version = 2
// CHECK-SAME: kind = "tunable", name = "XBLOCK"
// CHECK-SAME: kind = "shape_derived"
// CHECK-SAME: role = "input"
// CHECK-SAME: role = "output"
// CHECK-SAME: role = "tile_param"

func.func @add_1d(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
  %o = tensor.empty() : tensor<128xf32>
  %r = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>,
                     affine_map<(i) -> (i)>,
                     affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]}
    ins(%a, %b : tensor<128xf32>, tensor<128xf32>)
    outs(%o : tensor<128xf32>) {
  ^bb0(%x: f32, %y: f32, %_: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<128xf32>
  return %r : tensor<128xf32>
}
```

- [ ] **Step 2: Run the new test — expect FAIL**

```bash
cd build && ninja afir-opt && \
  ./bin/afir-opt ../test/Conversion/VectorPlanCodegen/tiling-info-schema-v2.mlir \
    --vector-plan-codegen 2>&1 | grep -E "schema_version|kind|role" | head
```

Expected: `schema_version` does not appear.

- [ ] **Step 3: Extend `emitTilingInfos` to populate the schema**

Replace the body of `emitTilingInfos` in `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` starting at line 818 (keep the function signature). The new body keeps existing block_dim_expr / axis_extent_expr / constraints derivation, then builds a `TilingInfoSchema` and serializes via the helper. Specifically:

1. Compute tunable fields exactly as today (lines 826–863).
2. Compute existing `blockDimExpr` block (lines 871–940) unchanged.
3. **New**: walk the kernel func args to build `SchemaArg` entries. The kernel func at this point in the pipeline has the signature documented in `pre_pack.mlir` exploration: `(input memref... , tile-param index... , output memref strided, [workspace memref])`. Identify each:
   - Identity-layout MemRef at head → `Input`; `networkIndex` = position in the input run (0, 1, 2, ...).
   - `index` type → `TileParam`; `tileParamName` = the matching `tp.name` (cross-reference via `arg_index`).
   - Non-identity-layout MemRef → `Output`; `resultIndex = 0` for now (DPS kernels return one result). `shapeExpr` computed below.
   - `memref<ui8>` → `Workspace`.
   - `emitasc.py_struct` → `TilingDataStruct` (only present after PackTilingData runs; absent here).
4. **New**: walk `memref.dim %argN, %cD` ops in the kernel body to find every shape-derived field this kernel needs. For each unique `(argN, dimIdx)` not already a tunable, push a `SchemaField{ kind=ShapeDerived, sourceArg=argN, sourceDim=dimIdx, name="dim_arg<argN>_<dimIdx>" }`.
5. **New**: compute `shapeExpr` for `Output` args. For each output dim `d`, find the input arg+dim with the same SymExpr root (via `afir.dim_symbols` + `afir.symbolic_shape`). Emit `"arg<inputNetworkIndex>_dim<inputDim>"`. If no symbolic info is available (legacy kernel), fall back to literal `std::to_string(staticDim)`.
6. Call `serializeTilingInfoSchema` and append to the `vector_plan.tiling_infos` array.

Concretely, after the existing block at line 940 (`func->setAttr("afir.block_dim_expr", ...)`), replace the `NamedAttrList entryAttrs; ... infos.push_back(entryAttrs.getDictionary(ctx));` tail (lines 955–985) with:

```cpp
  TilingInfoSchema schema;
  schema.kernelId       = func.getName().str();
  schema.blockDimExpr   = blockDimExpr;
  if (auto a = func->getAttrOfType<StringAttr>("afir.axis_extent_expr"))
    schema.axisExtentExpr = a.getValue().str();

  // Build tunable fields: re-walk same source as the old loop above so the
  // ordering matches the existing TilingData struct layout.
  for (auto &group : plan.tileable) {
    for (const auto &tp : group) {
      auto ba = dyn_cast<BlockArgument>(tp.ssa);
      if (!ba) continue;
      int64_t defaultVal = 0;
      if (auto attr = func.getArgAttrOfType<IntegerAttr>(
              ba.getArgNumber(), "vector_plan.default_tile_size"))
        defaultVal = attr.getInt();
      int64_t axisSize = -1;
      if (plan.group && tp.axisIdx >= 0 &&
          tp.axisIdx < (int)plan.group->collapsedAxes.size()) {
        int64_t s = plan.group->collapsedAxes[tp.axisIdx].staticSize;
        if (s != ShapedType::kDynamic) axisSize = s;
      }
      SchemaField f;
      f.name         = tp.name;
      f.kind         = SchemaFieldKind::Tunable;
      f.axisSize     = axisSize;
      f.defaultValue = defaultVal;
      f.argIndex     = (int32_t)ba.getArgNumber();
      schema.fields.push_back(std::move(f));
    }
  }

  // Build args[].
  unsigned numNetworkInputs = 0;
  Block &entry = func.getBody().front();
  for (BlockArgument ba : entry.getArguments()) {
    SchemaArg a;
    a.mlirIndex = (int32_t)ba.getArgNumber();
    auto ty = ba.getType();
    if (auto mt = dyn_cast<MemRefType>(ty)) {
      if (mt.getElementType().isInteger(8)) {
        a.role = SchemaArgRole::Workspace;
      } else if (mt.getLayout().isIdentity()) {
        a.role = SchemaArgRole::Input;
        a.networkIndex = (int32_t)numNetworkInputs++;
      } else {
        a.role = SchemaArgRole::Output;
        a.resultIndex = 0; // DPS: one result per kernel; revisit for multi-result.
      }
    } else if (isa<IndexType>(ty)) {
      a.role = SchemaArgRole::TileParam;
      // Match by arg index against the tunable fields we just built.
      for (auto &f : schema.fields)
        if (f.kind == SchemaFieldKind::Tunable && f.argIndex == a.mlirIndex)
          a.tileParamName = f.name;
    } else {
      continue; // unknown arg type — skip
    }
    schema.args.push_back(std::move(a));
  }

  // Compute output shape_expr using afir.dim_symbols + afir.symbolic_shape.
  auto dimSymsAttr2 = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
  std::optional<symshape::DimSymbolTable> symTable2;
  if (dimSymsAttr2)
    symTable2 = symshape::DimSymbolTable::fromAttr(dimSymsAttr2);
  for (auto &a : schema.args) {
    if (a.role != SchemaArgRole::Output) continue;
    auto mt = cast<MemRefType>(entry.getArgument(a.mlirIndex).getType());
    auto symAttr = func.getArgAttrOfType<StringAttr>(
        a.mlirIndex, "afir.symbolic_shape");
    auto symList = symAttr ? symshape::parseSymExprList(symAttr.getValue())
                           : std::nullopt;
    for (int64_t d = 0; d < mt.getRank(); ++d) {
      std::string expr;
      if (symList && (size_t)d < symList->size() && symTable2) {
        const auto &e = (*symList)[d];
        if (e.getKind() == symshape::SymExpr::Kind::Sym &&
            e.getSym() < symTable2->numRoots()) {
          auto src = symTable2->sourceOf(e.getSym());
          // Translate MLIR arg → network input index.
          for (auto &ia : schema.args)
            if (ia.role == SchemaArgRole::Input &&
                (unsigned)ia.mlirIndex == src.first) {
              expr = "arg" + std::to_string(ia.networkIndex) +
                     "_dim" + std::to_string(src.second);
              break;
            }
        }
      }
      if (expr.empty() && !ShapedType::isDynamic(mt.getShape()[d]))
        expr = std::to_string(mt.getShape()[d]);
      a.shapeExpr.push_back(expr);
    }
  }

  // Walk memref.dim ops to add shape-derived fields the kernel actually uses.
  llvm::DenseSet<std::pair<int32_t, int32_t>> seenDims;
  func.walk([&](memref::DimOp dimOp) {
    auto ba = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!ba || ba.getOwner() != &entry) return;
    auto cst = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
    if (!cst) return;
    auto intAttr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!intAttr) return;
    int32_t argN = (int32_t)ba.getArgNumber();
    int32_t dimI = (int32_t)intAttr.getValue().getSExtValue();
    if (!seenDims.insert({argN, dimI}).second) return;
    SchemaField f;
    f.name      = "dim_arg" + std::to_string(argN) + "_" + std::to_string(dimI);
    f.kind      = SchemaFieldKind::ShapeDerived;
    f.sourceArg = argN;
    f.sourceDim = dimI;
    schema.fields.push_back(std::move(f));
  });

  // Constraints (preserve existing serialization).
  ArrayAttr constraintsAttr;
  if (!plan.constraints.empty()) {
    SmallVector<Attribute> cs;
    for (auto &c : plan.constraints) {
      NamedAttrList ca;
      ca.append("kind",
                StringAttr::get(ctx, c.kind == TileConstraint::Divides
                                          ? "divides" : "le_bytes"));
      ca.append("lhs", StringAttr::get(ctx, c.lhs));
      ca.append("rhs", StringAttr::get(ctx, c.rhs));
      cs.push_back(ca.getDictionary(ctx));
    }
    constraintsAttr = ArrayAttr::get(ctx, cs);
  }

  DictionaryAttr entryDict = serializeTilingInfoSchema(ctx, schema,
                                                        constraintsAttr);
  StringRef attrName = "vector_plan.tiling_infos";
  SmallVector<Attribute> infos;
  if (auto existing = moduleOp->getAttrOfType<ArrayAttr>(attrName))
    llvm::append_range(infos, existing.getValue());
  infos.push_back(entryDict);
  moduleOp->setAttr(attrName, ArrayAttr::get(ctx, infos));
}
```

Note: this **replaces** the existing legacy `entryAttrs` block. Existing P6a v1 consumers (autotuner) need to continue reading v2 — see Task 1.3 for the format mapping `abi_index`/`arg_index`/`axis_size`/`default_value`/`name` is preserved in the tunable fields by design.

- [ ] **Step 4: Rebuild and run the lit test**

```bash
cd build && ninja afir-opt && cd .. && \
  build/bin/afir-opt test/Conversion/VectorPlanCodegen/tiling-info-schema-v2.mlir \
    --vector-plan-codegen 2>&1 | FileCheck test/Conversion/VectorPlanCodegen/tiling-info-schema-v2.mlir
```

Expected: PASS.

- [ ] **Step 5: Confirm existing P6a lit tests still pass**

```bash
cd build && ninja check-afir 2>&1 | tail -20
```

Expected: zero regressions. If anything breaks because of the renamed attr fields, those tests need migration — note them and update inline (this is mechanical; the v2 attr is a superset of v1's tunable fields).

- [ ] **Step 6: Commit**

```bash
git add lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp \
        test/Conversion/VectorPlanCodegen/tiling-info-schema-v2.mlir
git commit -m "feat(vector-plan): TilePlanGen emits schema_version=2 tiling info

Schema covers (a) tunable fields (unchanged from v1), (b) shape-derived
fields collected from memref.dim ops, (c) per-MLIR-arg provenance with
roles (input/output/tile_param/workspace) and, for outputs, a
SymExpr-derived shape_expr.

Downstream consumers still use the legacy path; this commit only writes
the attrs.  Consumer changes follow.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 1.3: GroupOutline — stamp coordinator-call provenance

**Background:** TilePlanGen runs *inside* the outlined kernel; it doesn't see the coordinator-level call. The `networkIndex` for input args is set by counting (good enough today since coordinator-call operand order matches the kernel func arg order). But for safety against future GroupOutline reorders we want to also stamp the coordinator-call mapping as an arg attr so the schema author isn't relying on positional invariance.

**Files:**
- Modify: `lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp` — when outlining a group, stamp `vector_plan.call_arg_index = N` on each outlined kernel func arg that maps to the N-th operand of the coordinator call.
- Modify: `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp` — in the schema-build loop, prefer the `vector_plan.call_arg_index` arg attr over the positional count when present.

- [ ] **Step 1: Locate where GroupOutline builds the outlined func signature**

```bash
grep -n "FuncOp\|addArgument\|setArgAttr\|create<.*FuncOp" \
  lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp | head -20
```

- [ ] **Step 2: After the outlined func is created and call operands are matched to args, add an arg attr per input arg**

In `lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp`, find the loop that maps `callOp.getOperand(i)` to `outlinedFunc.getArgument(i)` (you will need to inspect the file; the loop iterates over the kernel call's operands in order). Inside that loop, add:

```cpp
outlinedFunc.setArgAttr(i, "vector_plan.call_arg_index",
                        IntegerAttr::get(IntegerType::get(ctx, 32),
                                          (int32_t)i));
```

This stamps the coordinator-call operand index onto each outlined kernel input arg.

- [ ] **Step 3: In TilePlanGen schema build, honor the attr when present**

In `lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp`, inside the "Build args[]" loop you wrote in Task 1.2 Step 3, replace the line:

```cpp
        a.networkIndex = (int32_t)numNetworkInputs++;
```

with:

```cpp
        if (auto attr = func.getArgAttrOfType<IntegerAttr>(
                ba.getArgNumber(), "vector_plan.call_arg_index"))
          a.networkIndex = (int32_t)attr.getInt();
        else
          a.networkIndex = (int32_t)numNetworkInputs;
        ++numNetworkInputs;
```

- [ ] **Step 4: Rebuild + run lit suite**

```bash
cd build && ninja check-afir 2>&1 | tail -10
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/VectorPlan/GroupOutline/GroupOutlinePass.cpp \
        lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp
git commit -m "feat(vector-plan): stamp call_arg_index on outlined kernel input args

Eliminates the positional-equality assumption between coordinator-call
operand order and outlined kernel func arg order.  Schema build prefers
the explicit attr when present.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase S2: PackTilingData consumes schema

### Task 2.1: PackTilingData reads schema when present

**Files:**
- Modify: `lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp`
- Modify: `test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir` (add v2 case)

- [ ] **Step 1: Add a v2-case lit test (failing)**

Append to `test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir`:

```mlir
// v2 schema path: PackTilingData should consume vector_plan.tiling_infos
// directly and emit field names verbatim — no argToOriginal redirect.
// CHECK-LABEL: func.func @v2_transpose
// CHECK: emitasc.py_struct<"TilingData", [i64, i64], ["XBLOCK", "dim_arg2_1"]>
module attributes {vector_plan.tiling_infos = [{
    kernel_id = "v2_transpose",
    schema_version = 2 : i32,
    fields = [
      {name = "XBLOCK", kind = "tunable", axis_size = 32 : i64,
       default_value = 16 : i64, arg_index = 1 : i32},
      {name = "dim_arg2_1", kind = "shape_derived",
       source_arg = 2 : i32, source_dim = 1 : i32}
    ],
    args = [
      {mlir_index = 0 : i32, role = "input",  network_index = 0 : i32},
      {mlir_index = 1 : i32, role = "tile_param", name = "XBLOCK"},
      {mlir_index = 2 : i32, role = "output", result_index = 0 : i32,
       shape_expr = ["arg0_dim1", "arg0_dim0"]}
    ]
  }]} {
  func.func @v2_transpose(%arg0: memref<16x32xf16>, %arg1: index,
                          %arg2: memref<32x16xf16, strided<[?, 1], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %d = memref.dim %arg2, %c1 : memref<32x16xf16, strided<[?, 1], offset: ?>>
    return
  }
}
```

Run lit — expect FAIL (TilingData struct field name will currently be `dim_arg0_1` due to argToOriginal).

- [ ] **Step 2: Modify `packTilingData` to take the schema path when v2 is present**

In `lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp`, at the start of `static LogicalResult packTilingData(func::FuncOp func)` (around line 30 — check current code), add:

```cpp
  auto moduleOp = func->getParentOfType<ModuleOp>();
  std::optional<mlir::vector_plan::TilingInfoSchema> schema;
  if (moduleOp)
    schema = mlir::vector_plan::lookupTilingInfoSchema(moduleOp, func.getName());
  if (schema)
    return packTilingDataFromSchema(func, *schema);
  // Legacy v1 path below.
  ...existing body...
```

Then add a new helper `packTilingDataFromSchema` (above or below the existing fn). It mirrors the existing pack flow but:
- field names come directly from `schema->fields[i].name`
- field types are all i64 (matches today)
- the `memref.dim %argN, %cD` rewrite uses `schema->fields` to look up which TilingData member to extract — match by `(sourceArg, sourceDim)`
- no `argToOriginal`, no `symTable` canonicalization

```cpp
static LogicalResult
packTilingDataFromSchema(func::FuncOp func,
                          const mlir::vector_plan::TilingInfoSchema &schema) {
  using namespace mlir::vector_plan;
  MLIRContext *ctx = func.getContext();
  Block &entry = func.getBody().front();
  Type i64Ty = IntegerType::get(ctx, 64);
  Type indexTy = IndexType::get(ctx);

  // Build TilingData struct from schema.fields in order.
  SmallVector<StringRef> tilingNames;
  SmallVector<std::string> nameStorage;
  for (auto &f : schema.fields) nameStorage.push_back(f.name);
  for (auto &s : nameStorage) tilingNames.push_back(s);
  if (tilingNames.empty()) return success();

  auto tilingStructTy = buildTilingDataType(ctx, tilingNames);
  auto tilingArgTy = MemRefType::get(
      {ShapedType::kDynamic}, tilingStructTy, MemRefLayoutAttrInterface{},
      IntegerAttr::get(IntegerType::get(ctx, 32), kGMSpace));

  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&entry);
  BlockArgument tilingArg = entry.addArgument(tilingArgTy, func.getLoc());
  Value localStruct = builder.create<emitasc::CopyStructOp>(
      func.getLoc(), tilingStructTy, tilingArg);

  SmallVector<Value> fieldVals;
  for (auto &n : nameStorage)
    fieldVals.push_back(builder.create<emitasc::MemberOp>(
        func.getLoc(), i64Ty, localStruct, builder.getStringAttr(n)));

  // For each tunable field, replace uses of the corresponding tile-param arg
  // (index type) with the i64→index cast of the field value.
  SmallVector<unsigned> argsToErase;
  for (size_t i = 0; i < schema.fields.size(); ++i) {
    auto &f = schema.fields[i];
    if (f.kind != SchemaFieldKind::Tunable) continue;
    BlockArgument tileArg = entry.getArgument(f.argIndex);
    Value val = builder.create<arith::IndexCastOp>(
        func.getLoc(), indexTy, fieldVals[i]);
    tileArg.replaceAllUsesWith(val);
    argsToErase.push_back(f.argIndex);
  }

  // For shape-derived fields, rewrite memref.dim %argN, %cD where
  // (N, D) matches.
  SmallVector<memref::DimOp> dimOps;
  func.walk([&](memref::DimOp d) { dimOps.push_back(d); });
  for (memref::DimOp dimOp : dimOps) {
    auto ba = dyn_cast<BlockArgument>(dimOp.getSource());
    if (!ba || ba.getOwner() != &entry) continue;
    auto cst = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
    if (!cst) continue;
    int64_t d = cast<IntegerAttr>(cst.getValue()).getInt();
    for (size_t i = 0; i < schema.fields.size(); ++i) {
      auto &f = schema.fields[i];
      if (f.kind != SchemaFieldKind::ShapeDerived) continue;
      if (f.sourceArg != (int32_t)ba.getArgNumber()) continue;
      if (f.sourceDim != (int32_t)d) continue;
      OpBuilder b(dimOp);
      Value idxVal = b.create<arith::IndexCastOp>(
          dimOp.getLoc(), indexTy, fieldVals[i]);
      dimOp.replaceAllUsesWith(idxVal);
      dimOp.erase();
      break;
    }
  }

  // Erase tile-param args in reverse order.
  llvm::sort(argsToErase, std::greater<unsigned>());
  for (unsigned idx : argsToErase) entry.eraseArgument(idx);
  if (func->getAttr("arg_attrs")) func->removeAttr("arg_attrs");

  SmallVector<Type> newArgTypes;
  for (BlockArgument a : entry.getArguments()) newArgTypes.push_back(a.getType());
  func.setFunctionType(FunctionType::get(ctx, newArgTypes,
                                          func.getFunctionType().getResults()));

  OpBuilder modBuilder(ctx);
  modBuilder.setInsertionPointToStart(
      func->getParentOfType<ModuleOp>().getBody());
  modBuilder.create<emitasc::DeclarePyStructOp>(func.getLoc(),
                                                 TypeAttr::get(tilingStructTy));
  return success();
}
```

Include `Conversion/VectorPlan/TilePlan.h` at top of file for the schema helpers.

- [ ] **Step 3: Rebuild and run lit**

```bash
cd build && ninja afir-opt && ninja check-afir 2>&1 | tail -10
```

Expected: all lit PASS, including the new v2 case.

- [ ] **Step 4: Re-run full e2e gate suite to confirm v2 + v1 still both work**

```bash
cd /home/gser/code/Ascend-MLIR && source examples/env.sh
export PATH=$PWD/build/bin:$PATH
for d in examples/add-mul-relu-e2e examples/two-elewise-e2e \
         examples/bcast-leading-e2e examples/bcast-trailing-e2e \
         examples/dyn-bucketed-e2e examples/transpose-elementwise-e2e \
         examples/reduce-big-r-e2e examples/mixed-attn-e2e; do
  echo "=== $d ==="
  (cd "$d" && bash run.sh 2>&1 | tail -3)
done
```

Expected: **transpose-elementwise-e2e ends with `session.validation=pass`** (the bug is now fixed via the schema path). All other gates remain PASS.

- [ ] **Step 5: Commit**

```bash
git add lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp \
        test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir
git commit -m "feat(pack-tiling-data): consume tiling_infos schema v2

When the module carries schema_version=2, PackTilingData reads field
names + (source_arg, source_dim) directly from the schema and skips the
argToOriginal heuristic entirely.  Legacy v1 path remains for kernels
that don't yet emit schema (none in tree, but the fallback is kept until
S5).

Fixes transpose-elementwise-e2e sim crash.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase S3: CannTranslation + NetworkJsonEmitter consume schema

### Task 3.1: CannTranslation writes shape_key via schema args[]

**Files:**
- Modify: `lib/Target/CannKernel/CannTranslation.cpp:emitTilingSpaceJson` (around line 1988)

- [ ] **Step 1: Replace the `dim_arg<N>_<D> → arg<N>_dim<D>` translator**

In `lib/Target/CannKernel/CannTranslation.cpp`, find `emitTilingSpaceJson`. Today it derives `shape_key` from the field name by string surgery (`"dim_arg2_1" → "arg2_dim1"`). Replace this with a schema-driven mapping that translates MLIR `source_arg` to network `args[source_arg].network_index`:

```cpp
  auto schema = mlir::vector_plan::lookupTilingInfoSchema(
      funcOp->getParentOfType<ModuleOp>(), funcOp.getName());

  auto makeShapeKey = [&](StringRef fieldName) -> std::string {
    if (!schema) {
      // Legacy v1 fallback (kept until S5).
      StringRef rest = fieldName.drop_front(4);
      auto pos = rest.rfind('_');
      if (pos == StringRef::npos) return rest.str();
      return rest.substr(0, pos).str() + "_dim" + rest.substr(pos + 1).str();
    }
    // v2 path: locate the field in the schema, translate source_arg through
    // the args[] table.
    for (auto &f : schema->fields) {
      if (f.kind != mlir::vector_plan::SchemaFieldKind::ShapeDerived) continue;
      if (f.name != fieldName) continue;
      // Output-arg-sourced fields: emit the corresponding shape_expr entry.
      for (auto &a : schema->args) {
        if (a.mlirIndex != f.sourceArg) continue;
        if (a.role == mlir::vector_plan::SchemaArgRole::Input)
          return "arg" + std::to_string(a.networkIndex) +
                 "_dim" + std::to_string(f.sourceDim);
        if (a.role == mlir::vector_plan::SchemaArgRole::Output &&
            (size_t)f.sourceDim < a.shapeExpr.size())
          return a.shapeExpr[f.sourceDim];
        break;
      }
      return std::string{}; // unresolvable — emitter will warn
    }
    return std::string{};
  };
```

- [ ] **Step 2: Build and inspect transpose-elementwise's tiling_space.json**

```bash
cd build && ninja afir-translate && cd .. && \
  source examples/env.sh && export PATH=$PWD/build/bin:$PATH && \
  cd examples/transpose-elementwise-e2e && bash run.sh 2>&1 | tail -5 && \
  cat transpose_relu__v0_space.json
```

Expected: `transpose_relu__v0_space.json` has a `dim_arg2_1` field (output stride) with `shape_key = "arg0_dim0"` (i.e., the shape_expr's resolved name = input's d0 = M = 16). The corresponding shape_key in v0 reflects the actual output stride.

- [ ] **Step 3: Commit**

```bash
git add lib/Target/CannKernel/CannTranslation.cpp
git commit -m "feat(cann-translation): write shape_key via schema args[] when v2 present

Output-derived TilingData fields now emit the shape_expr literal as
shape_key, so the runner resolves them against actual input dims rather
than an unresolvable output-arg shape_key.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3.2: NetworkJsonEmitter writes args[] from schema

**Files:**
- Modify: `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp`

NetworkJsonEmitter today only writes coordinator-call operands into kernel `args[]`. With v2 schema present we want to write per-MLIR-arg entries so the runner can resolve `arg<N>_dim<D>` keys for any role.

- [ ] **Step 1: Extend the kernel entry to add `schema_args` when v2 is present**

In `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp`, in the per-call kernel-entry builder (around line 192), before assigning `kernelEntry["args"] = std::move(argsArr);`, add:

```cpp
      // v2 schema: also emit a full per-MLIR-arg table for the runner.
      if (callee) {
        if (auto schema = mlir::vector_plan::lookupTilingInfoSchema(
                module, calleeName)) {
          llvm::json::Array schArgs;
          for (auto &a : schema->args) {
            llvm::json::Object e;
            e["mlir_index"] = static_cast<int64_t>(a.mlirIndex);
            switch (a.role) {
              case mlir::vector_plan::SchemaArgRole::Input:
                e["role"] = "input";
                e["network_index"] = static_cast<int64_t>(a.networkIndex);
                break;
              case mlir::vector_plan::SchemaArgRole::Output: {
                e["role"] = "output";
                e["result_index"] = static_cast<int64_t>(a.resultIndex);
                llvm::json::Array se;
                for (auto &s : a.shapeExpr) se.push_back(s);
                e["shape_expr"] = std::move(se);
                break;
              }
              case mlir::vector_plan::SchemaArgRole::TileParam:
                e["role"] = "tile_param";
                e["name"] = a.tileParamName;
                break;
              case mlir::vector_plan::SchemaArgRole::Workspace:
                e["role"] = "workspace";
                break;
              case mlir::vector_plan::SchemaArgRole::TilingDataStruct:
                e["role"] = "tiling_data_struct";
                break;
            }
            schArgs.push_back(std::move(e));
          }
          kernelEntry["schema_args"] = std::move(schArgs);
        }
      }
```

- [ ] **Step 2: Verify network.json contains schema_args**

```bash
cd /home/gser/code/Ascend-MLIR && source examples/env.sh
export PATH=$PWD/build/bin:$PATH
cd examples/transpose-elementwise-e2e && bash run.sh >/dev/null 2>&1; \
  find . -name network.json -newer transpose_relu.mlir -exec head -80 {} \;
```

Expected: kernel entry has both `args` (legacy) and `schema_args` (v2). Spec compliance: `schema_args[0]` has role=input, network_index=0; `schema_args[i]` for the output has role=output and shape_expr.

- [ ] **Step 3: Commit**

```bash
git add lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp
git commit -m "feat(network-json): emit schema_args[] from tiling_infos v2

Adds a per-MLIR-arg provenance table to each kernel entry (input/output/
tile_param/workspace), letting downstream tools resolve shape_keys for
any kernel arg, not just coordinator-call operands.  Legacy 'args' field
preserved.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase S4: network_runner consumes schema

### Task 4.1: Resolve shape_key via schema_args

**Files:**
- Modify: `python/network_runner.py`

- [ ] **Step 1: Add `_resolve_shape_via_schema`**

In `python/network_runner.py`, just above `_resolve_kernel_input_shape` (around line 246), add:

```python
def _resolve_shape_via_schema(network, kid, shape_key, runner_inputs):
    """Resolve a shape_key string against a kernel's schema_args.

    shape_key formats:
      - "arg<N>_dim<D>"            → look up schema_args[mlir_index=N] in kernel
      - any other (already an output shape_expr) → recurse on its components

    Returns int (resolved dim size) or None if not v2.
    """
    import re
    k = network.kernel_by_id(kid)
    sch = k.get("schema_args")
    if not sch:
        return None
    m = re.match(r"^arg(\d+)_dim(\d+)$", shape_key)
    if not m:
        # The runner currently doesn't evaluate complex shape_expr (a*b+c);
        # those are emitted only for kernels we haven't introduced yet
        # (reduce, matmul).  Fail loud so the implementer notices.
        raise RuntimeError(
            f"network_runner: shape_key {shape_key!r} for kernel {kid!r} "
            "is not a simple argN_dimD form; complex shape_expr not yet "
            "supported in runner.")
    target_arg, dim_idx = int(m.group(1)), int(m.group(2))
    # The 'argN' in a shape_key is a NETWORK arg index (input ordinal).
    # Find the schema arg with that network_index.
    for sa in sch:
        if sa.get("role") == "input" and sa.get("network_index") == target_arg:
            mlir_idx = sa["mlir_index"]
            # Use legacy 'args' resolver (it's indexed by network-call order,
            # which matches network_index for inputs).
            shape = _resolve_kernel_input_shape(
                network, kid, target_arg, runner_inputs)
            return int(shape[dim_idx])
    raise RuntimeError(
        f"network_runner: shape_key {shape_key!r} does not match any input "
        f"in kernel {kid!r}'s schema_args.")
```

- [ ] **Step 2: Update `_shape_key_values_for_kernel` to prefer the schema path**

Replace the body of `_shape_key_values_for_kernel` (around line 271) with:

```python
def _shape_key_values_for_kernel(space, network, kid, runner_inputs):
    import re
    out = {}
    for p in space.get("tiling_params", []):
        sk = p.get("shape_key")
        if not sk or sk in out:
            continue
        val = _resolve_shape_via_schema(network, kid, sk, runner_inputs)
        if val is not None:
            out[sk] = val
            continue
        # Legacy path: shape_key like "arg<i>_dim<j>" against kernel.args[i].
        m = re.match(r"^arg(\d+)_dim(\d+)$", sk)
        if not m:
            continue
        arg_idx, dim_idx = int(m.group(1)), int(m.group(2))
        shape = _resolve_kernel_input_shape(network, kid, arg_idx, runner_inputs)
        out[sk] = int(shape[dim_idx])
    return out
```

- [ ] **Step 3: Run network-runner-backed e2e**

```bash
cd /home/gser/code/Ascend-MLIR && source examples/env.sh
export PATH=$PWD/build/bin:$PATH
cd examples/mixed-attn-e2e && bash run.sh 2>&1 | tail -5
cd ../dyn-bucketed-e2e && bash run.sh 2>&1 | tail -5
```

Expected: both end in `session.validation=pass`.

- [ ] **Step 4: Commit**

```bash
git add python/network_runner.py
git commit -m "feat(network-runner): resolve shape_key via schema_args when present

For kernels whose network.json entry carries schema_args (v2), resolve
shape_key 'argN_dimD' by mapping N (network input index) through
schema_args; legacy path retained for kernels without schema_args.
Complex shape_expr (a*b+c) currently raises — no such kernels in tree.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4.2: Update example run.sh to drop hand-written shape-derived TILING_PARAMS

**Files:**
- Modify: `examples/transpose-elementwise-e2e/run.sh`
- Modify: `examples/bcast-leading-e2e/run.sh`
- Modify: `examples/bcast-trailing-e2e/run.sh`

These run.sh files invoke `runtime-session --run-manifest ...` directly, not through `network_runner.py`. They hand-build the TILING_PARAMS string. With v2 schema in tiling_space.json, the host-facing JSON already carries the right shape_keys; the run.sh just needs to send the correct values keyed by the new field names.

- [ ] **Step 1: For transpose-elementwise, update TILING_PARAMS to use new field names**

Edit `examples/transpose-elementwise-e2e/run.sh` line that builds `TILING_PARAMS`:

```bash
# Old:
TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TILING_PARAMS+=",dim_arg0_1=${N},dim_arg3_1=${M}"

# New (post-S2 the kernel uses dim_arg2_1 for output stride; verify against the
# generated tiling_space.json):
TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TILING_PARAMS+=",dim_arg2_1=${M}"
```

The actual field name set is `XBLOCK, XBLOCK_SUB, dim_arg<output_mlir_idx>_1`. Confirm by reading `transpose_relu__v0_space.json` after running translate.

- [ ] **Step 2: Similarly verify the bcast run scripts**

For bcast-leading and bcast-trailing, inspect post-translate `<kernel>__v0_space.json`. Update each TILING_PARAMS string to match the schema field names exactly. Most likely no change is needed for bcast (output is shape-equal to input, so v1 and v2 field naming coincides) — verify rather than blindly editing.

- [ ] **Step 3: Re-run all three to confirm green**

```bash
cd /home/gser/code/Ascend-MLIR && source examples/env.sh
export PATH=$PWD/build/bin:$PATH
for d in examples/transpose-elementwise-e2e examples/bcast-leading-e2e examples/bcast-trailing-e2e; do
  echo "=== $d ==="; (cd "$d" && bash run.sh 2>&1 | tail -3)
done
```

Expected: all three `session.validation=pass`.

- [ ] **Step 4: Commit**

```bash
git add examples/transpose-elementwise-e2e/run.sh \
        examples/bcast-leading-e2e/run.sh \
        examples/bcast-trailing-e2e/run.sh
git commit -m "test(examples): align run.sh TILING_PARAMS with schema v2 field names

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase S5: Cleanup — delete legacy paths

### Task 5.1: Remove argToOriginal and v1 fallbacks

**Files:**
- Modify: `lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp` (remove `argToOriginal`, `symTable`-only canonicalize branch, and the v1 `packTilingData` body — keep only the schema entrypoint)
- Modify: `lib/Target/CannKernel/CannTranslation.cpp` (remove the legacy `makeShapeKey` string-surgery branch — schema is now required)
- Modify: `lib/Conversion/VectorPlan/GroupOutline/NetworkJsonEmitter.cpp` (collapse `args` and `schema_args` into a single canonical field; or drop `schema_args` and replace `args` semantics — see Step 2)
- Modify: `python/network_runner.py` (remove `_resolve_kernel_input_shape` legacy path that handles missing schema_args)
- Update lit tests: remove v1-only `pack-tiling-data.mlir` cases, keep v2 only.

- [ ] **Step 1: Remove `argToOriginal` and dead canonicalize branches in PackTilingData**

In `lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp`, delete the legacy `packTilingData` body (everything that runs when `schema` is empty). Replace `packTilingData` body with:

```cpp
static LogicalResult packTilingData(func::FuncOp func) {
  auto moduleOp = func->getParentOfType<ModuleOp>();
  if (!moduleOp) return success();
  auto schema = mlir::vector_plan::lookupTilingInfoSchema(moduleOp, func.getName());
  if (!schema)
    return func.emitError("PackTilingData: missing schema_version=2 "
                          "vector_plan.tiling_infos entry for kernel ")
           << func.getName();
  return packTilingDataFromSchema(func, *schema);
}
```

- [ ] **Step 2: Remove legacy shape_key fallback in CannTranslation**

In `CannTranslation.cpp::emitTilingSpaceJson`, delete the `if (!schema) { /* string surgery */ }` branch — error out if schema is missing.

- [ ] **Step 3: NetworkJsonEmitter — unify args**

Decide here whether to keep both `args` (legacy) and `schema_args`, or rename `schema_args` to `args` and drop the old shape. Recommended: keep both, since `args` (coordinator-call-only) is also useful for non-shape-key purposes (e.g., debug). No change in S5; just document.

- [ ] **Step 4: Python runner — remove the legacy fallback in `_shape_key_values_for_kernel`**

Replace the legacy fallback path with a hard error when `_resolve_shape_via_schema` returns None.

- [ ] **Step 5: Update or delete v1 lit cases**

`test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir` currently contains v1 cases. Either retrofit them with v2 attrs or delete the v1-only ones.

- [ ] **Step 6: Full regression**

```bash
cd build && ninja check-afir
cd /home/gser/code/Ascend-MLIR && source examples/env.sh
export PATH=$PWD/build/bin:$PATH
# Run the full e2e gate suite.
for d in examples/*-e2e; do
  echo "=== $d ==="; (cd "$d" && bash run.sh 2>&1 | tail -3) || { echo "FAILED: $d"; break; }
done
```

Expected: lit green, all e2e PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/Conversion/AscendCPrepareForEmit/PackTilingDataPass.cpp \
        lib/Target/CannKernel/CannTranslation.cpp \
        python/network_runner.py \
        test/Conversion/AscendCPrepareForEmit/pack-tiling-data.mlir
git commit -m "refactor: remove tiling-info v1 fallback paths

Schema v2 is now required.  Deletes:
  - PackTilingData::argToOriginal heuristic
  - dead symTable-only canonicalize branch
  - CannTranslation legacy dim_argN_D string-surgery shape_key path
  - network_runner pre-schema_args fallback
  - v1-only pack-tiling-data lit cases

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Final acceptance

- [ ] **Step 1: Confirm transpose-elementwise PASS**
- [ ] **Step 2: Confirm full e2e gate suite (14 kernels) PASS**
- [ ] **Step 3: Confirm lit (78 tests) PASS**
- [ ] **Step 4: Grep clean — no remaining references to `argToOriginal`, no `dim_arg5_1` style stale references**

```bash
git grep -n "argToOriginal" lib/ python/ && echo "REGRESSION: argToOriginal still present"
```

Expected: no matches.

---

## Notes on deferred work

- **Complex `shape_expr` (a*b+c)** — needed if/when a future kernel emits a non-trivial output shape (reduce splitting an axis into N*K, matmul). Runner currently raises a clear error; extend `_resolve_shape_via_schema` with a small expression evaluator at that time.
- **`kind="const"` schema fields** — deferred per design spec §3.3 (Q4 brainstorm). Retrofit when needed.
- **autotuner cross-variant** — `tools/autotuner/autotuner_main.cpp` may need updates if it reads tiling_space JSON in a way assuming v1 field naming. Inspect during S4 dry-run.
