#include "Conversion/AutoFuse/TilePlan.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::auto_fuse;

namespace mlir::auto_fuse {

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
      d.append("call_arg_index", IntegerAttr::get(i32Ty, a.callArgIndex));
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

  SmallVector<Attribute> shapeEqsAttr;
  for (const auto &g : s.shapeEqualities) {
    SmallVector<Attribute> pairs;
    for (auto [callIdx, dim] : g) {
      SmallVector<Attribute> pair = {
          IntegerAttr::get(i32Ty, callIdx),
          IntegerAttr::get(i32Ty, dim),
      };
      pairs.push_back(ArrayAttr::get(ctx, pair));
    }
    shapeEqsAttr.push_back(ArrayAttr::get(ctx, pairs));
  }

  NamedAttrList entry;
  entry.append("kernel_id",      StringAttr::get(ctx, s.kernelId));
  entry.append("schema_version", IntegerAttr::get(i32Ty,
                                                  TilingInfoSchema::kSchemaVersion));
  entry.append("fields", ArrayAttr::get(ctx, fieldsAttr));
  entry.append("args",   ArrayAttr::get(ctx, argsAttr));
  if (!shapeEqsAttr.empty())
    entry.append("shape_equalities", ArrayAttr::get(ctx, shapeEqsAttr));
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
  auto getStr = [](DictionaryAttr d, StringRef k) -> std::optional<std::string> {
    if (auto a = d.getAs<StringAttr>(k)) return a.str();
    return std::nullopt;
  };
  auto getInt = [](DictionaryAttr d, StringRef k) -> std::optional<int64_t> {
    if (auto a = d.getAs<IntegerAttr>(k)) return a.getInt();
    return std::nullopt;
  };

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
      auto name = getStr(d, "name");
      auto kind = getStr(d, "kind");
      if (!name || !kind) continue;
      f.name = *name;
      f.kind = (*kind == "tunable") ? SchemaFieldKind::Tunable
                                    : SchemaFieldKind::ShapeDerived;
      if (f.kind == SchemaFieldKind::Tunable) {
        f.axisSize     = getInt(d, "axis_size").value_or(-1);
        f.defaultValue = getInt(d, "default_value").value_or(0);
        f.argIndex     = (int32_t)getInt(d, "arg_index").value_or(-1);
      } else {
        f.sourceArg = (int32_t)getInt(d, "source_arg").value_or(-1);
        f.sourceDim = (int32_t)getInt(d, "source_dim").value_or(-1);
      }
      s.fields.push_back(std::move(f));
    }
  }
  if (auto arr = entry.getAs<ArrayAttr>("args")) {
    for (Attribute aa : arr) {
      auto d = dyn_cast<DictionaryAttr>(aa);
      if (!d) continue;
      SchemaArg a;
      auto mlirIdx = getInt(d, "mlir_index");
      auto roleStr = getStr(d, "role");
      if (!mlirIdx || !roleStr) continue;
      auto r = roleFromString(*roleStr);
      if (!r) continue;
      a.mlirIndex = (int32_t)*mlirIdx;
      a.role = *r;
      if (a.role == SchemaArgRole::Input)
        a.callArgIndex = (int32_t)getInt(d, "call_arg_index").value_or(-1);
      if (a.role == SchemaArgRole::Output) {
        a.resultIndex = (int32_t)getInt(d, "result_index").value_or(-1);
        if (auto se = d.getAs<ArrayAttr>("shape_expr"))
          for (Attribute e : se)
            if (auto sa = dyn_cast<StringAttr>(e))
              a.shapeExpr.push_back(sa.str());
      }
      if (a.role == SchemaArgRole::TileParam)
        a.tileParamName = getStr(d, "name").value_or(std::string());
      s.args.push_back(std::move(a));
    }
  }
  if (auto groups = entry.getAs<ArrayAttr>("shape_equalities")) {
    for (Attribute ga : groups) {
      auto gAttr = dyn_cast<ArrayAttr>(ga);
      if (!gAttr) continue;
      llvm::SmallVector<std::pair<int32_t, int32_t>, 4> group;
      for (Attribute pa : gAttr) {
        auto pAttr = dyn_cast<ArrayAttr>(pa);
        if (!pAttr || pAttr.size() != 2) continue;
        auto callIdx = dyn_cast<IntegerAttr>(pAttr[0]);
        auto dim     = dyn_cast<IntegerAttr>(pAttr[1]);
        if (!callIdx || !dim) continue;
        group.push_back({(int32_t)callIdx.getInt(), (int32_t)dim.getInt()});
      }
      if (group.size() >= 2)
        s.shapeEqualities.push_back(std::move(group));
    }
  }
  return s;
}

std::optional<TilingInfoSchema>
lookupTilingInfoSchema(ModuleOp moduleOp, StringRef kernelName) {
  auto arr = moduleOp->getAttrOfType<ArrayAttr>("auto_fuse.tiling_infos");
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

} // namespace mlir::auto_fuse
