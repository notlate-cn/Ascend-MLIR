#include "TilePlanGen.h"
#include "TileFuseUtils.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

TilePlan genVectorTilePlan(func::FuncOp func,
                            const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc,
                            bool enableReductionSplit,
                            int64_t maxFullLoopIters) {
  TilePlan plan;
  plan.group = &info;
  return plan; // TODO Task 2
}

} // namespace mlir::afir
