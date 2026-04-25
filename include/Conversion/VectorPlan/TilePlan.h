#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::vector_plan {

enum class TileLevel : uint8_t { Outer, Inner, Full };

enum class TileFieldKind : uint8_t {
  TunableTile,
  FixedTile,
  ShapeDim,
  Derived,
};

struct TileParam {
  llvm::StringRef name;
  Value           ssa;
  OpFoldResult    defaultValue;
  int32_t         axisIdx;
  TileLevel       level;
  AxisRole        role;
};

struct TilePlan {
  const CollapsedGroupInfo                        *group;
  llvm::SmallVector<llvm::SmallVector<TileParam>>  tileable;
  llvm::SmallVector<TileParam>                     full;
  llvm::SmallVector<OpFoldResult, 2>               blockDimExprs;
};

} // namespace mlir::vector_plan
