#include "LoopNestBuilder.h"
#include "TileFuseUtils.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

LoopNestResult buildLoopNest(OpBuilder &builder, Location loc,
                              const TilePlan &plan, ValueRange initTensors) {
  LoopNestResult result;
  result.innermostBody = builder.getInsertionBlock();
  result.iterArgs = SmallVector<Value>(initTensors);
  return result; // TODO Task 3
}

} // namespace mlir::afir
