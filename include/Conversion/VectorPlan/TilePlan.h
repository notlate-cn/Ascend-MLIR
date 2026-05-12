#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir::vector_plan {

enum class TileLevel : uint8_t { Outer, Inner, Full };

enum class TileFieldKind : uint8_t {
  TunableTile,
  FixedTile,
  ShapeDim,
  Derived,
};

// AxisKind / AxisClass / AxisGrouping live in GroupInfo.h (computed by Collapse,
// carried in CollapsedGroupInfo::grouping).

// One enumerated tiling-case draft (≈ AutoFuse `TilingCase`).  P1 produces a
// single default draft; full y×x×r enumeration + RCore variant comes later.
struct TilePlanDraft {
  int  ubTilingAxisY = -1;
  int  ubTilingAxisX = -1;
  int  ubTilingAxisR = -1;
  int  blockTilingId = 0;
  bool reduceIsBlock = false;
};

// A schedulability/legality constraint over the tunable params (≈ ATT's
// tiling-data constraints).  Emitted into `vector_plan.tiling_infos` in P6.
struct TileConstraint {
  enum Kind : uint8_t { Divides, LeBytes } kind;
  Value lhs;
  Value rhs;
};

struct TileParam {
  std::string  name;
  Value        ssa;
  OpFoldResult defaultValue;
  int32_t      axisIdx;
  TileLevel    level;
  AxisRole     role;

  // Flags carried from the axis's AxisClass (consulted by downstream passes;
  // default-initialized so existing aggregate-initializing call sites compile).
  bool bindMultiCore    = false;
  bool enableTail       = true;
  bool enablePad        = false;
  bool isReduceSplit    = false;
  bool isBroadcastSplit = false;
};

struct TilePlan {
  enum class ReduceTemplate : uint8_t { None, Common, FullLoad, RCore };

  const CollapsedGroupInfo                        *group = nullptr;
  llvm::SmallVector<llvm::SmallVector<TileParam>>  tileable;
  llvm::SmallVector<TileParam>                     full;
  llvm::SmallVector<OpFoldResult, 2>               blockDimExprs;

  ReduceTemplate            reduceTemplate = ReduceTemplate::None;
  bool                      reduceIsBlock  = false;
  llvm::SmallVector<int>    blockFusedAxes;        // collapsed-axis ids fused into the block axis (in order)
  int                       ubTilingAxisY = -1;
  int                       ubTilingAxisX = -1;
  int                       ubTilingAxisR = -1;
  // Per-operand vectorized iteration dims (the dims that stay whole inside the
  // tiled linalg op, ≈ tensor.attr.vectorized_axis).  Filled by P3.
  llvm::DenseMap<Value, llvm::SmallVector<int>> vectorizedDims;
  llvm::SmallVector<TileConstraint>             constraints;
};

} // namespace mlir::vector_plan
