#include "Collapse.h"
#include "GroupEmitter.h"
#include "LoopNestBuilder.h"
#include "SliceComputer.h"
#include "TilePlanGen.h"
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
#include "llvm/ADT/DenseSet.h"

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
    if (collapsedInfo.topoMembers.empty()) return;

    // Phase 2: TilePlanGen.
    builder.setInsertionPointToStart(&func.getBody().front());
    auto plan = genVectorTilePlan(func, collapsedInfo, builder, func.getLoc(),
                                  enableReductionSplit, maxFullLoopIters);
    // Emit tiling.infos module attribute (consumed by PrepareForEmit Phase B).
    emitTilingInfos(func, plan);

    // Collect init tensors and original results BEFORE modification.
    SmallVector<Value> originalResults;
    SmallVector<Value> initTensors;
    DenseSet<Value> seenInits;
    for (linalg::LinalgOp op : collapsedInfo.topoMembers) {
      for (Value r : op->getResults())
        originalResults.push_back(r);
      for (Value out : op.getDpsInits())
        if (seenInits.insert(out).second)
          initTensors.push_back(out);
    }

    // Phase 3a: LoopNestBuilder.
    builder.setInsertionPoint(collapsedInfo.topoMembers.front());
    auto loopNest =
        buildLoopNest(builder, func.getLoc(), plan, initTensors);

    // Phase 3b+c: GroupEmitter.
    builder.setInsertionPointToEnd(loopNest.innermostBody);
    auto loopResults =
        emitGroup(builder, func.getLoc(), collapsedInfo, plan, loopNest);

    // Replace original results with loop results and erase original ops.
    for (auto [origRes, loopRes] :
         llvm::zip(originalResults, loopResults))
      origRes.replaceAllUsesWith(loopRes);
    for (linalg::LinalgOp op : collapsedInfo.topoMembers)
      op->erase();
  }
};
} // namespace

std::unique_ptr<Pass> createVectorPlanTileFusePass() {
  return std::make_unique<VectorPlanTileFusePass>();
}

} // namespace mlir::afir
