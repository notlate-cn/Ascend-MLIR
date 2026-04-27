#include "SliceComputer.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

SliceParams computeSlice(AffineMap indexingMap,
                          const DenseMap<int, Value> &loopIVs,
                          const TilePlan &plan,
                          Value tensor,
                          OpBuilder &builder, Location loc) {
  return {}; // TODO Task 3
}

} // namespace mlir::afir
