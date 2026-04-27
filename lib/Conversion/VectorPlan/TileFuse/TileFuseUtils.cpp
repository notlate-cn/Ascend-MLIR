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
  // TODO: implemented in Task 2
  return b.create<arith::ConstantIndexOp>(loc, 0);
}

} // namespace mlir::afir
