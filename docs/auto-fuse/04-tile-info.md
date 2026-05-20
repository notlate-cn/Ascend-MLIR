# impl-04: TileInfo 接入层

**设计依据**: [00-architecture.md](./00-architecture.md) §6–8, [00-data-model.md](./00-data-model.md) §8  
**前置**: impl-03（Pass 2 已生成 TilePlan，loop nest 已建立）

---

## 定位

TileInfo 是 TilePlan 与 TilingData / AutoTuner 之间的**稳定可序列化中间层**：

```
TilePlan（含 MLIR Value/OpFoldResult）
  → TileInfo（稳定序列化对象，无 MLIR 内部指针）
      → tiling.infos（module-level MLIR attribute，权威表示）
      → PrepareForEmit（读 TileInfo 构造 TilingData struct）
      → AutoTuner（读 search.candidates 搜索）
```

**迁移路径**（§11）：

| Phase | 内容 |
|-------|------|
| A | Pass 2 写 `tiling.infos`（本模块） |
| B | PrepareForEmit 改为消费 TileInfo（本模块） |
| C | AutoTuner 改为消费 TileInfo；hard-fail 若缺失 |
| D | `tiling.tiles` / `tiling.shapes` 降级为兼容视图 |

本文档实现 Phase A + Phase B。

---

## Phase 1: TilePlanToTileInfo — 语义降维

文件：`lib/Conversion/AutoFuse/TileInfo/TilePlanToTileInfo.cpp`

### ValueExpr 工厂实现

```cpp
ValueExpr ValueExpr::makeConst(int64_t v) {
  return {.kind = Kind::Const, .constValue = v};
}
ValueExpr ValueExpr::makeShapeDim(int32_t arg, int32_t dim) {
  return {.kind = Kind::ShapeDim, .argIndex = arg, .dimIndex = dim};
}
ValueExpr ValueExpr::makeFieldRef(StringRef id) {
  return {.kind = Kind::FieldRef, .fieldId = id.str()};
}
ValueExpr ValueExpr::makeMul(ValueExpr lhs, ValueExpr rhs) {
  return {.kind = Kind::Mul,
          .lhs = std::make_shared<ValueExpr>(lhs),
          .rhs = std::make_shared<ValueExpr>(rhs)};
}
ValueExpr ValueExpr::makeCeilDiv(ValueExpr lhs, ValueExpr rhs) {
  return {.kind = Kind::CeilDiv,
          .lhs = std::make_shared<ValueExpr>(lhs),
          .rhs = std::make_shared<ValueExpr>(rhs)};
}
```

### OpFoldResult → ValueExpr 转换

```cpp
ValueExpr opFoldResultToValueExpr(OpFoldResult ofr,
                                   const ShapeBindingMap &shapeMap) {
  // 静态常量
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(ofr.dyn_cast<Attribute>()))
    return ValueExpr::makeConst(intAttr.getInt());

  // 运行时 Value：查 shapeMap（func arg dim → ShapeDim）
  if (auto val = dyn_cast_or_null<Value>(ofr.dyn_cast<Value>())) {
    if (auto it = shapeMap.find(val); it != shapeMap.end())
      return ValueExpr::makeShapeDim(it->second.argIndex, it->second.dimIndex);
    // ceildiv / mul：递归处理
    if (auto ceilOp = val.getDefiningOp<arith::CeilDivSIOp>())
      return ValueExpr::makeCeilDiv(
          opFoldResultToValueExpr(ceilOp.getLhs(), shapeMap),
          opFoldResultToValueExpr(ceilOp.getRhs(), shapeMap));
    if (auto mulOp = val.getDefiningOp<arith::MulIOp>())
      return ValueExpr::makeMul(
          opFoldResultToValueExpr(mulOp.getLhs(), shapeMap),
          opFoldResultToValueExpr(mulOp.getRhs(), shapeMap));
    // 兜底：FieldRef（tile param SSA value）
    if (auto fieldId = getTileParamFieldId(val))
      return ValueExpr::makeFieldRef(*fieldId);
  }
  llvm_unreachable("unsupported OpFoldResult in TileInfo conversion");
}
```

### 主转换函数

```cpp
TileInfo buildTileInfo(const CollapsedGroupInfo &info,
                        const TilePlan &plan,
                        func::FuncOp kernelFunc,
                        int32_t groupId, int32_t planId) {
  TileInfo ti;
  ti.kernelId = formatv("group{0}_plan{1}", groupId, planId);
  ti.groupId  = groupId;
  ti.planId   = planId;

  // 建立 func arg → ShapeRef 的映射（用于 OpFoldResult 转换）
  ShapeBindingMap shapeMap;
  for (auto [argIdx, arg] : enumerate(kernelFunc.getArguments())) {
    auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
    if (!tensorType) continue;
    for (int dimIdx = 0; dimIdx < tensorType.getRank(); ++dimIdx) {
      // 找所有使用该 dim 的 tensor.dim op
      Value dimVal = findDimValue(arg, dimIdx);
      if (dimVal) shapeMap[dimVal] = ShapeRef{(int32_t)argIdx, dimIdx};
    }
  }

  // axes：每个 collapsedAxis → TileAxisInfo
  for (auto [i, ax] : enumerate(info.collapsedAxes)) {
    TileAxisInfo axInfo;
    axInfo.axisIndex  = i;
    axInfo.axisName   = ax.name.str();
    axInfo.role       = ax.role;
    axInfo.extentExpr = opFoldResultToValueExpr(
        getAxisExtentOFR(ax, kernelFunc), shapeMap);
    ti.axes.push_back(axInfo);
  }

  // fields：tile params → TileFieldSpec
  int32_t abiIdx = 0;
  auto addParam = [&](const TileParam &tp, TileFieldKind kind) {
    TileFieldSpec spec;
    spec.fieldId    = formatv("tile.{0}", llvm::StringRef(tp.name).lower());
    spec.abiName    = tp.name.str();
    spec.abiType    = "i64";
    spec.abiIndex   = abiIdx++;
    spec.kind       = kind;
    spec.axisIndex  = tp.axisIdx;
    spec.level      = tp.level;
    if (tp.defaultValue)
      spec.defaultExpr = opFoldResultToValueExpr(tp.defaultValue, shapeMap);
    if (kind == TileFieldKind::TunableTile)
      spec.search = buildSearchSpace(tp);
    ti.fields.push_back(spec);
  };

  // tileable params（TunableTile）
  for (auto &tileVec : plan.tileable)
    for (auto &tp : tileVec)
      addParam(tp, TileFieldKind::TunableTile);

  // full / reduction params
  for (auto &tp : plan.full)
    addParam(tp, tp.level == TileLevel::Full
                   ? TileFieldKind::FixedTile : TileFieldKind::TunableTile);

  // shape fields：func arg 的每个 primitive dim
  for (auto [argIdx, arg] : enumerate(kernelFunc.getArguments())) {
    auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
    if (!tensorType) continue;
    for (int dimIdx = 0; dimIdx < tensorType.getRank(); ++dimIdx) {
      TileFieldSpec spec;
      spec.fieldId      = formatv("shape.{0}.{1}", argIdx, dimIdx);
      spec.abiName      = formatv("dim_{0}_{1}", argIdx, dimIdx);
      spec.abiType      = "i64";
      spec.abiIndex     = abiIdx++;
      spec.kind         = TileFieldKind::ShapeDim;
      spec.shapeBinding = ShapeRef{(int32_t)argIdx, dimIdx};
      ti.fields.push_back(spec);
    }
  }

  // blockDimExprs
  for (auto &expr : plan.blockDimExprs)
    ti.blockDimExprs.push_back(opFoldResultToValueExpr(expr, shapeMap));

  return ti;
}
```

---

## Phase 2: TileInfoSerializer — TileInfo → tiling.infos attribute

文件：`lib/Conversion/AutoFuse/TileInfo/TileInfoSerializer.cpp`

### ValueExpr → MLIR DictionaryAttr

```cpp
Attribute valueExprToAttr(const ValueExpr &expr, MLIRContext *ctx) {
  SmallVector<NamedAttribute> fields;
  auto s = [&](StringRef k, Attribute v) {
    fields.push_back({StringAttr::get(ctx, k), v});
  };
  auto i64 = [&](int64_t v) { return IntegerAttr::get(IntegerType::get(ctx, 64), v); };

  switch (expr.kind) {
  case ValueExpr::Kind::Const:
    s("op", StringAttr::get(ctx, "const"));
    s("value", i64(expr.constValue));
    break;
  case ValueExpr::Kind::ShapeDim:
    s("op", StringAttr::get(ctx, "shape_dim"));
    s("arg", i64(expr.argIndex));
    s("dim", i64(expr.dimIndex));
    break;
  case ValueExpr::Kind::FieldRef:
    s("op", StringAttr::get(ctx, "field_ref"));
    s("id", StringAttr::get(ctx, expr.fieldId));
    break;
  case ValueExpr::Kind::Mul:
    s("op", StringAttr::get(ctx, "mul"));
    s("lhs", valueExprToAttr(*expr.lhs, ctx));
    s("rhs", valueExprToAttr(*expr.rhs, ctx));
    break;
  case ValueExpr::Kind::CeilDiv:
    s("op", StringAttr::get(ctx, "ceildiv"));
    s("lhs", valueExprToAttr(*expr.lhs, ctx));
    s("rhs", valueExprToAttr(*expr.rhs, ctx));
    break;
  default:
    llvm_unreachable("unsupported ValueExpr kind in serializer");
  }
  return DictionaryAttr::get(ctx, fields);
}
```

### TileInfo → ArrayAttr 并写入 module

```cpp
void serializeTileInfos(ModuleOp module,
                         ArrayRef<TileInfo> tileInfos) {
  MLIRContext *ctx = module.getContext();
  SmallVector<Attribute> infoAttrs;

  for (const TileInfo &ti : tileInfos) {
    SmallVector<NamedAttribute> tiFields;
    auto s = [&](StringRef k, Attribute v) {
      tiFields.push_back({StringAttr::get(ctx, k), v});
    };
    auto i64 = [&](int64_t v) {
      return IntegerAttr::get(IntegerType::get(ctx, 64), v);
    };

    s("kernel", StringAttr::get(ctx, ti.kernelId));
    s("group",  i64(ti.groupId));
    s("plan",   i64(ti.planId));

    // axes
    SmallVector<Attribute> axisAttrs;
    for (const TileAxisInfo &ax : ti.axes) {
      SmallVector<NamedAttribute> axF;
      axF.push_back({StringAttr::get(ctx, "axis"),   i64(ax.axisIndex)});
      axF.push_back({StringAttr::get(ctx, "name"),   StringAttr::get(ctx, ax.axisName)});
      axF.push_back({StringAttr::get(ctx, "role"),
                     StringAttr::get(ctx, ax.role == AxisRole::Parallel
                                          ? "parallel" : "reduction")});
      axF.push_back({StringAttr::get(ctx, "extent"), valueExprToAttr(ax.extentExpr, ctx)});
      axisAttrs.push_back(DictionaryAttr::get(ctx, axF));
    }
    s("axes", ArrayAttr::get(ctx, axisAttrs));

    // fields
    SmallVector<Attribute> fieldAttrs;
    for (const TileFieldSpec &f : ti.fields) {
      SmallVector<NamedAttribute> fF;
      fF.push_back({StringAttr::get(ctx, "id"),       StringAttr::get(ctx, f.fieldId)});
      fF.push_back({StringAttr::get(ctx, "abi_name"), StringAttr::get(ctx, f.abiName)});
      if (f.abiIndex)
        fF.push_back({StringAttr::get(ctx, "abi_index"), i64(*f.abiIndex)});
      fF.push_back({StringAttr::get(ctx, "kind"),
                    StringAttr::get(ctx, tileFieldKindStr(f.kind))});
      if (f.axisIndex)
        fF.push_back({StringAttr::get(ctx, "axis"), i64(*f.axisIndex)});
      if (f.level)
        fF.push_back({StringAttr::get(ctx, "level"),
                      StringAttr::get(ctx, tileLevelStr(*f.level))});
      if (f.defaultExpr)
        fF.push_back({StringAttr::get(ctx, "default"),
                      valueExprToAttr(*f.defaultExpr, ctx)});
      if (f.search) {
        SmallVector<Attribute> cands;
        for (int64_t c : f.search->candidates)
          cands.push_back(i64(c));
        SmallVector<NamedAttribute> searchF;
        searchF.push_back({StringAttr::get(ctx, "candidates"),
                           ArrayAttr::get(ctx, cands)});
        if (f.search->alignment)
          searchF.push_back({StringAttr::get(ctx, "alignment"),
                             i64(*f.search->alignment)});
        if (f.search->upperBound)
          searchF.push_back({StringAttr::get(ctx, "upper_bound"),
                             valueExprToAttr(*f.search->upperBound, ctx)});
        fF.push_back({StringAttr::get(ctx, "search"),
                      DictionaryAttr::get(ctx, searchF)});
      }
      if (f.shapeBinding)
        fF.push_back({StringAttr::get(ctx, "shape_binding"),
                      DictionaryAttr::get(ctx, {
                          {StringAttr::get(ctx, "arg"), i64(f.shapeBinding->argIndex)},
                          {StringAttr::get(ctx, "dim"), i64(f.shapeBinding->dimIndex)},
                      })});
      fieldAttrs.push_back(DictionaryAttr::get(ctx, fF));
    }
    s("fields", ArrayAttr::get(ctx, fieldAttrs));

    // block_dim
    SmallVector<Attribute> bdAttrs;
    for (const ValueExpr &bd : ti.blockDimExprs)
      bdAttrs.push_back(valueExprToAttr(bd, ctx));
    s("block_dim", ArrayAttr::get(ctx, bdAttrs));

    infoAttrs.push_back(DictionaryAttr::get(ctx, tiFields));
  }

  module->setAttr("tiling.infos", ArrayAttr::get(ctx, infoAttrs));
}
```

### 测试用例（Phase 1 + 2）

```mlir
// test/Conversion/AutoFuse/tile-info-vector-layernorm.mlir
// RUN: mlir-opt --auto-fuse-tile-fuse %s | FileCheck %s

func.func @kernel_group0(%input: tensor<?x?x?xf16>,
                          %scale: tensor<?xf16>,
                          %bias: tensor<?xf16>) -> tensor<?x?x?xf16> {
  // LayerNorm [B, S, H]，collapse 后 [B*S, H]
  // tileable axis 0: BS (Parallel), reduction axis 1: H
}

// CHECK: module attributes
// CHECK-SAME: tiling.infos = [{
// CHECK-SAME:   kernel = "group0_plan0"
// CHECK-SAME:   axes = [
// CHECK-SAME:     {axis = 0 : i64, name = "BS", role = "parallel"
// CHECK-SAME:      extent = {op = "mul"
// CHECK-SAME:     {axis = 1 : i64, name = "H",  role = "reduction"
// CHECK-SAME:   fields = [
// CHECK-SAME:     {id = "tile.xblock",    abi_name = "XBLOCK",    abi_index = 0 : i64
// CHECK-SAME:     {id = "tile.xblock_sub",abi_name = "XBLOCK_SUB",abi_index = 1 : i64
// CHECK-SAME:     {id = "tile.rblock_0",  abi_name = "RBLOCK_0",  abi_index = 2 : i64, kind = "fixed"
// CHECK-SAME:   block_dim = [{op = "ceildiv"

// test/Conversion/AutoFuse/tile-info-cube-matmul.mlir
// RUN: mlir-opt --auto-fuse-tile-fuse %s | FileCheck %s
func.func @kernel_group1(%A: tensor<?x?xf16>, %B: tensor<?x?xf16>,
                          %bias: tensor<?xf16>) -> tensor<?x?xf16> {
  %mm  = linalg.matmul ins(%A, %B) outs(...)
  %add = linalg.generic { ... } ins(%mm, %bias) outs(...)
  return %add
}
// CHECK: tiling.infos = [{
// CHECK-SAME:   kernel = "group1_plan0"
// CHECK-SAME:   {id = "tile.bm",   abi_name = "BM",   abi_index = 0
// CHECK-SAME:   {id = "tile.bn",   abi_name = "BN",   abi_index = 1
// CHECK-SAME:   {id = "tile.tb_m", abi_name = "Tb_M", abi_index = 2
// CHECK-SAME:   {id = "tile.tb_n", abi_name = "Tb_N", abi_index = 3
// CHECK-SAME:   {id = "tile.tk",   abi_name = "t_K",  abi_index = 4
// CHECK-SAME:   block_dim = [{op = "ceildiv" {{.*}} {op = "ceildiv"
// （2D block_dim）
```

---

## Phase 3: PrepareForEmit 适配（Phase B 迁移）

文件：`lib/...AscendCPrepareForEmitPass.cpp`（修改，非新建）

### 改造前（旧逻辑，删除）

```cpp
// 旧：扫描 func args 中的 i64 类型，按位置顺序推断字段
for (auto arg : func.getArguments())
  if (arg.getType().isInteger(64))
    tilingArgs.push_back(arg);
```

### 改造后（从 tiling.infos 读取）

```cpp
void adaptPrepareForEmit(func::FuncOp func, ModuleOp module) {
  // 找到该 func 对应的 TileInfo
  auto tilingInfosAttr = module->getAttrOfType<ArrayAttr>("tiling.infos");
  if (!tilingInfosAttr) return; // Phase A 未完成时降级为旧行为

  DictionaryAttr tileInfo = findTileInfoForFunc(tilingInfosAttr, func);
  if (!tileInfo) return;

  // 按 abiIndex 排序的 fields（只取非 Derived）
  SmallVector<DictionaryAttr> packedFields;
  for (auto fieldAttr : tileInfo.getAs<ArrayAttr>("fields")) {
    auto f = cast<DictionaryAttr>(fieldAttr);
    if (!f.get("abi_index")) continue; // Derived → skip
    packedFields.push_back(f);
  }
  llvm::stable_sort(packedFields, [](DictionaryAttr a, DictionaryAttr b) {
    return cast<IntegerAttr>(a.get("abi_index")).getInt() <
           cast<IntegerAttr>(b.get("abi_index")).getInt();
  });

  // 构造 TilingData struct 字段列表
  SmallVector<std::pair<StringRef, Type>> structFields;
  for (auto f : packedFields) {
    StringRef abiName = cast<StringAttr>(f.get("abi_name")).getValue();
    // v1 统一 i64
    structFields.push_back({abiName, IntegerType::get(func.getContext(), 64)});
  }

  // 生成 emitasc.py_struct<"TilingData", ...>
  emitTilingDataStruct(func, structFields);
}
```

### 测试用例（Phase 3）

```mlir
// test/Conversion/AutoFuse/tile-info-prepare-emit.mlir
// RUN: mlir-opt --auto-fuse-tile-fuse --ascendc-prepare-for-emit %s \
// RUN:   | FileCheck %s

func.func @kernel_group0(...) -> ... { ... }

// CHECK: tiling.infos = [{
// CHECK-SAME:   abi_name = "XBLOCK",    abi_index = 0

// TilingData struct 字段顺序与 abi_index 一致
// CHECK: emitasc.py_struct<"TilingData",
// CHECK-SAME: i64, "XBLOCK"
// CHECK-SAME: i64, "XBLOCK_SUB"
// CHECK-SAME: i64, "RBLOCK_0"
// （字段顺序不再依赖 func arg 扫描）
```

---

## 不变量验证（自动检查）

在 `serializeTileInfos` 完成后，可选插入 assert 验证 §7.4 不变量：

```cpp
void verifyTileInfoInvariants(const TileInfo &ti) {
  // 不变量 2：abiIndex 连续且权威
  int32_t expectedIdx = 0;
  for (auto &f : ti.fields) {
    if (!f.abiIndex) {
      assert(f.kind == TileFieldKind::Derived && "only Derived fields may omit abiIndex");
      continue;
    }
    assert(*f.abiIndex == expectedIdx++ && "abiIndex must be contiguous and authoritative");
  }

  // 不变量 5：无 MLIR Value 外流（TileInfo 只持有 ValueExpr）
  // → 由类型系统保证（TileInfo 无 Value 成员）

  // 不变量 7：blockDimExprs 必须显式写出
  assert(!ti.blockDimExprs.empty() && "blockDimExprs must be explicit");
  // CubeGroup 通过 blockDimExprs 元素数量识别（2D dispatch），而非 groupId
  // VectorGroup: 1 个元素; CubeGroup: 2 个元素
  // （groupId 是连续序号，无法区分 Vector/Cube）
}
```

---

## 端到端验收（全流水线）

```bash
# Phase A + B：tile-fuse 生成 tiling.infos，prepare-emit 从中读取字段顺序
mlir-opt \
  --auto-fuse-tile-fuse \
  --ascendc-prepare-for-emit \
  test/Conversion/AutoFuse/tile-info-vector-layernorm.mlir \
  | FileCheck test/Conversion/AutoFuse/tile-info-vector-layernorm.mlir

# 验证 tiling.infos 权威性：abi_index 顺序与 TilingData struct 完全对应
# TilingData struct 中字段不再依赖 arg 扫描顺序
```
