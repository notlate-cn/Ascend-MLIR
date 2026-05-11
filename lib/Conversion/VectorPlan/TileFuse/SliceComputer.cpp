#include "SliceComputer.h"
#include "TileFuseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

static Value getTileSizeForAxis(const TilePlan &plan, int axisIdx,
                                 OpBuilder &b, Location loc) {
  for (const auto &tp : plan.full) {
    if (tp.axisIdx != axisIdx) continue;
    if (tp.role == AxisRole::Parallel)
      return b.create<arith::ConstantIndexOp>(loc, 1); // BCast Full step=1
    return tp.ssa; // Reduction Full: full extent
  }
  for (auto &group : plan.tileable)
    for (const auto &tp : group)
      if (tp.axisIdx == axisIdx && tp.level == TileLevel::Inner)
        return tp.ssa;
  return b.create<arith::ConstantIndexOp>(loc, 0); // unreachable
}

// Returns the outer (coarse) tile size for an axis. For tileable axes with
// both Outer and Inner levels, this returns the Outer ssa (e.g. XBLOCK).
// Falls back to getTileSizeForAxis for axes without an explicit outer level.
static Value getOuterTileSizeForAxis(const TilePlan &plan, int axisIdx,
                                      OpBuilder &b, Location loc) {
  for (auto &group : plan.tileable)
    for (const auto &tp : group)
      if (tp.axisIdx == axisIdx && tp.level == TileLevel::Outer)
        return tp.ssa;
  return getTileSizeForAxis(plan, axisIdx, b, loc);
}

// Compute a coarser (outer-level) slice using outer IVs and outer tile sizes.
// Used for BCast-independent inputs that can be hoisted before BCast loops.
SliceParams computeOuterSlice(AffineMap indexingMap,
                               const DenseMap<int, Value> &outerLoopIVs,
                               const TilePlan &plan,
                               Value tensor,
                               OpBuilder &builder, Location loc) {
  SliceParams sp;
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  for (int dimPos = 0; dimPos < (int)indexingMap.getNumResults(); ++dimPos) {
    AffineExpr expr = indexingMap.getResult(dimPos);
    auto d = dyn_cast<AffineDimExpr>(expr);
    if (d && outerLoopIVs.count((int)d.getPosition())) {
      int axisIdx = (int)d.getPosition();
      sp.offsets.push_back(OpFoldResult(outerLoopIVs.lookup(axisIdx)));
      sp.sizes.push_back(
          OpFoldResult(getOuterTileSizeForAxis(plan, axisIdx, builder, loc)));
    } else {
      sp.offsets.push_back(OpFoldResult(c0));
      Value dimSize =
          builder.create<tensor::DimOp>(loc, tensor, (int64_t)dimPos);
      sp.sizes.push_back(OpFoldResult(dimSize));
    }
    sp.strides.push_back(OpFoldResult(c1));
  }
  return sp;
}

SliceParams computeSlice(AffineMap indexingMap,
                          const DenseMap<int, Value> &loopIVs,
                          const TilePlan &plan,
                          Value tensor,
                          OpBuilder &builder, Location loc,
                          const DenseMap<int, Value> *sizeOverride) {
  SliceParams sp;
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  for (int dimPos = 0; dimPos < (int)indexingMap.getNumResults(); ++dimPos) {
    AffineExpr expr = indexingMap.getResult(dimPos);
    auto d = dyn_cast<AffineDimExpr>(expr);
    if (d && loopIVs.count((int)d.getPosition())) {
      int axisIdx = (int)d.getPosition();
      sp.offsets.push_back(OpFoldResult(loopIVs.lookup(axisIdx)));
      // Tail emit passes {tile_axis -> tail_size} so the tail body shrinks the
      // slice on the peeled axis only; other axes keep their planned size.
      Value sz;
      if (sizeOverride) {
        auto it = sizeOverride->find(axisIdx);
        if (it != sizeOverride->end()) sz = it->second;
      }
      if (!sz) sz = getTileSizeForAxis(plan, axisIdx, builder, loc);
      sp.sizes.push_back(OpFoldResult(sz));
    } else {
      sp.offsets.push_back(OpFoldResult(c0));
      Value dimSize =
          builder.create<tensor::DimOp>(loc, tensor, (int64_t)dimPos);
      sp.sizes.push_back(OpFoldResult(dimSize));
    }
    sp.strides.push_back(OpFoldResult(c1));
  }
  return sp;
}

} // namespace mlir::afir
