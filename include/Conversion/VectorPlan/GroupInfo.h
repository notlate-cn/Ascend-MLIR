#pragma once
#include "Analysis/SymbolicShape/SymExpr.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::vector_plan {

enum class AxisRole : uint8_t {
  Parallel,
  Reduction,
};

struct AxisInfo {
  llvm::StringRef name;
  int64_t         staticSize; // ShapedType::kDynamic if dynamic
  AxisRole        role;
  // Symbolic extent in terms of afir.dim_symbols ids (serialized).  Invalid
  // (!isValid()) when unknown -- e.g. multi-op funcs, or before
  // afir-symbolize-shapes ran.  For a collapsed axis this is the product of the
  // grouped canonical-axis extents.
  mlir::afir::symshape::SymExpr extent;
};

// ---------------------------------------------------------------------------
// Axis classification (mirrors AutoFuse's `AxisGroup` x/y/r/n_group; see
// docs/superpowers/plans/2026-05-11-port-af-scheduler-to-vector-plan.zh.md).
// Computed once, on the post-collapse iteration axes, by the Collapse pass
// (≈ AF MergeContinuousAxis → GenTilingGroup/NormGroup) and carried in
// CollapsedGroupInfo::grouping; TilePlanGen consumes it directly.
// ---------------------------------------------------------------------------
enum class AxisKind : uint8_t {
  Y, // ordinary elementwise / injective ("y_group")
  R, // reduction ("r_group")
  X, // transpose-divergent input-side axes ("x_group")
  N, // non-tileable / vectorize-only ("n_group")
};

struct AxisClass {
  AxisKind kind          = AxisKind::Y;
  bool     bindMultiCore = false; // candidate for block dispatch (≈ SubAxis::is_bind_multi_core)
  bool     enableTail    = true;
  bool     enablePad     = false; // unaligned DataCopy → DataCopyPad — future
  bool     isReduceSplit    = false;
  // True when some operand's indexing map projects this iteration axis away —
  // i.e. that operand is constant along it ("broadcast").  Informational only:
  // the schedule treats a broadcast axis as an ordinary parallel axis; the
  // lowering (ComputeConversion) replicates the projecting operand on-chip.
  bool     isBroadcastConst = false;
  int      origPos = -1; // position in the original (pre-reorder) loop order
};

struct AxisGrouping {
  llvm::SmallVector<AxisClass> axes; // one per CollapsedGroupInfo::collapsedAxes
  llvm::SmallVector<int> yAxes, rAxes, xAxes, nAxes; // index lists, in axesOrder
  llvm::SmallVector<int> axesOrder;
};

struct GroupInfo {
  enum class Kind : uint8_t { Vector, Cube };
  Kind                                kind;
  llvm::SmallVector<linalg::LinalgOp> topoMembers;
  llvm::SmallVector<linalg::LinalgOp> sinks;
  llvm::SmallVector<AxisInfo>         canonicalAxes;
  llvm::SmallVector<Value>            boundaryIn;
  llvm::SmallVector<Value>            boundaryOut;
};

struct CollapsedGroupInfo : GroupInfo {
  llvm::SmallVector<AxisInfo> collapsedAxes;
  llvm::SmallVector<int>      axisMap;   // original axis idx -> collapsed axis idx; -1 if absorbed
  bool                        hasB2      = false;
  bool                        noCollapse = false;
  // Post-collapse broadcast axis indices (ascending). Filled by Phase 1 Collapse.
  llvm::SmallVector<int>            broadcastAxes;
  // Extent SSA values for broadcast axes (needed for dynamic shapes in Phase 2).
  llvm::DenseMap<int, mlir::Value>  broadcastAxisExtents;
  // X/Y/R/N classification of the post-collapse axes (≈ AF AxisGroup).  Filled
  // by Collapse; consumed by TilePlanGen.  (Filled in commit 2 of the
  // unify-axis-classification refactor; until then it is empty.)
  AxisGrouping                     grouping;
};

struct CubeGroupInfo : GroupInfo {
  linalg::LinalgOp matmul;
  linalg::LinalgOp epilogueAnchor;
};

} // namespace mlir::vector_plan
