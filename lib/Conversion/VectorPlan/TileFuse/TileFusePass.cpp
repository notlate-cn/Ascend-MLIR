#include "Collapse.h"
#include "Conversion/VectorPlan/GroupInfo.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "Conversion/VectorPlan/VectorPlanPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DECL_VECTORPLANTILEFUSE
#define GEN_PASS_DEF_VECTORPLANTILEFUSE
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

namespace {
struct VectorPlanTileFusePass
    : public ::impl::VectorPlanTileFuseBase<VectorPlanTileFusePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());

    // Phase 0: absorb linalg.broadcast into downstream linalg.generic.
    {
      RewritePatternSet patterns(&getContext());
      populateBroadcastAbsorbPatterns(patterns);
      (void)applyPatternsGreedily(func, std::move(patterns));
    }

    // Phase 1: Collapse — transforms IR, returns CollapsedGroupInfo for Phase 2/3.
    auto collapsedInfo = mlir::afir::collapseGroup(builder, func);
    (void)collapsedInfo; // consumed by Phase 2/3 — TODO

    // Phase 2: TilePlanGen   — TODO
    // Phase 3: LoopNestBuilder + GroupEmitter — TODO
  }
};
} // namespace

std::unique_ptr<Pass> createVectorPlanTileFusePass() {
  return std::make_unique<VectorPlanTileFusePass>();
}

} // namespace mlir::afir
