#include "TileFuseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

Value castToIndex(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex()) return v;
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

Value getAxisExtentValue(OpBuilder &b, Location loc,
                          const CollapsedGroupInfo &info, int axisIdx) {
  int64_t staticSize = info.collapsedAxes[axisIdx].staticSize;
  if (staticSize != ShapedType::kDynamic)
    return b.create<arith::ConstantIndexOp>(loc, staticSize);

  // Dynamic: find first operand that has this axis in its indexing map.
  for (linalg::LinalgOp op : info.topoMembers) {
    auto maps     = op.getIndexingMapsArray();
    auto operands = op->getOperands();
    for (auto [operand, map] : llvm::zip(operands, maps)) {
      if (!isa<RankedTensorType>(operand.getType())) continue;
      for (auto [dimPos, expr] : llvm::enumerate(map.getResults())) {
        auto d = dyn_cast<AffineDimExpr>(expr);
        if (d && (int)d.getPosition() == axisIdx)
          return b.create<tensor::DimOp>(loc, operand, (int64_t)dimPos);
      }
    }
  }
  llvm_unreachable("axis not found in any operand map");
}

} // namespace mlir::afir
