#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
#include <string>

namespace mlir::vector_plan {

struct ValueExpr {
  enum class Kind { Const, ShapeDim, FieldRef, Mul, Add, CeilDiv, Min, Max } kind = Kind::Const;
  int64_t     constValue = 0;
  int32_t     argIndex   = -1;
  int32_t     dimIndex   = -1;
  std::string fieldId;
  std::shared_ptr<ValueExpr> lhs, rhs;

  static ValueExpr makeConst(int64_t v);
  static ValueExpr makeShapeDim(int32_t arg, int32_t dim);
  static ValueExpr makeFieldRef(llvm::StringRef id);
  static ValueExpr makeMul(ValueExpr lhs, ValueExpr rhs);
  static ValueExpr makeCeilDiv(ValueExpr lhs, ValueExpr rhs);
};

struct ShapeRef { int32_t argIndex, dimIndex; };

struct TileAxisInfo {
  int32_t     axisIndex;
  std::string axisName;
  AxisRole    role;
  ValueExpr   extentExpr;
};

struct SearchSpace {
  bool                       enabled = false;
  llvm::SmallVector<int64_t> candidates;
  std::optional<int64_t>     alignment;
  std::optional<ValueExpr>   upperBound;
};

struct TileFieldSpec {
  std::string              fieldId;
  std::string              abiName;
  std::string              abiType;
  std::optional<int32_t>   abiIndex;
  TileFieldKind            kind;
  std::optional<int32_t>   axisIndex;
  std::optional<TileLevel> level;
  std::optional<ShapeRef>  shapeBinding;
  std::optional<ValueExpr> defaultExpr;
  std::optional<SearchSpace> search;
};

struct TileInfo {
  std::string kernelId;
  int32_t     groupId;
  int32_t     planId;
  llvm::SmallVector<TileAxisInfo, 8>  axes;
  llvm::SmallVector<TileFieldSpec, 4> fields;
  llvm::SmallVector<ValueExpr, 2>     blockDimExprs;
};

} // namespace mlir::vector_plan
