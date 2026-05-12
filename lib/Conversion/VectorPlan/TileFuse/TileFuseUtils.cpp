#include "TileFuseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

bool resultUsedOnlyByGroupMembers(linalg::LinalgOp op,
                                   const CollapsedGroupInfo &info) {
  llvm::SmallPtrSet<Operation *, 8> members;
  for (linalg::LinalgOp m : info.topoMembers)
    members.insert(m.getOperation());
  for (Value r : op->getResults())
    for (Operation *user : r.getUsers())
      if (!members.contains(user))
        return false;
  return true;
}

Value castToIndex(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex()) return v;
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

Value getAxisExtentValue(OpBuilder &b, Location loc,
                          const CollapsedGroupInfo &info, int axisIdx) {
  int64_t staticSize = info.collapsedAxes[axisIdx].staticSize;
  if (staticSize != ShapedType::kDynamic)
    return b.create<arith::ConstantIndexOp>(loc, staticSize);

  // Dynamic: search operands of the post-collapse generics for a value that
  // maps to axisIdx and dominates the current insertion point.
  //
  // After multi-op collapse, operands are tensor.collapse_shape results, not
  // function arguments. We must trace through collapse_shape to its source
  // (which should be a block argument) and compute the product of original
  // dimensions to recover the collapsed axis extent.
  for (linalg::LinalgOp op : info.topoMembers) {
    auto maps     = op.getIndexingMapsArray();
    auto operands = op->getOperands();
    for (auto [operand, map] : llvm::zip(operands, maps)) {
      if (!isa<RankedTensorType>(operand.getType())) continue;
      for (auto [dimPos, expr] : llvm::enumerate(map.getResults())) {
        auto d = dyn_cast<AffineDimExpr>(expr);
        if (!d || (int)d.getPosition() != axisIdx) continue;

        // Prefer block arguments: they dominate everywhere.
        if (isa<BlockArgument>(operand))
          return b.create<tensor::DimOp>(loc, operand, (int64_t)dimPos);

        // If operand is tensor.collapse_shape whose src is a block argument,
        // recover the collapsed extent as the product of the original dims.
        if (auto colOp = operand.getDefiningOp<tensor::CollapseShapeOp>()) {
          Value src = colOp.getSrc();
          if (isa<BlockArgument>(src)) {
            auto grps = colOp.getReassociationIndices();
            if ((size_t)dimPos < grps.size()) {
              Value prod = b.create<arith::ConstantIndexOp>(loc, 1);
              for (int64_t origDim : grps[dimPos])
                prod = b.create<arith::MulIOp>(
                    loc, prod,
                    b.create<tensor::DimOp>(loc, src, origDim).getResult());
              return prod;
            }
          }
        }
        // Operand does not dominate the insertion point; keep searching.
      }
    }
  }
  llvm_unreachable("axis not found in any operand map");
}

} // namespace mlir::afir
