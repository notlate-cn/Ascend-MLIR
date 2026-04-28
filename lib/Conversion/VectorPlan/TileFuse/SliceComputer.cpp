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

SliceParams computeSlice(AffineMap indexingMap,
                          const DenseMap<int, Value> &loopIVs,
                          const TilePlan &plan,
                          Value tensor,
                          OpBuilder &builder, Location loc) {
  SliceParams sp;
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  for (int dimPos = 0; dimPos < (int)indexingMap.getNumResults(); ++dimPos) {
    AffineExpr expr = indexingMap.getResult(dimPos);
    auto d = dyn_cast<AffineDimExpr>(expr);
    if (d && loopIVs.count((int)d.getPosition())) {
      int axisIdx = (int)d.getPosition();
      sp.offsets.push_back(OpFoldResult(loopIVs.lookup(axisIdx)));
      sp.sizes.push_back(
          OpFoldResult(getTileSizeForAxis(plan, axisIdx, builder, loc)));
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
