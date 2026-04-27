#include "GroupEmitter.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

SmallVector<Value> emitGroup(OpBuilder &builder, Location loc,
                              const CollapsedGroupInfo &info,
                              const TilePlan &plan,
                              const LoopNestResult &loopNest) {
  return {}; // TODO Task 3
}

} // namespace mlir::afir
