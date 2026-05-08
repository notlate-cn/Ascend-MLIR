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
    // Ensure every initTensor's defining op precedes the first linalg op so
    // that scf.for iter_args dominate the loop.  When ops between the first
    // and last linalg op produce an initTensor (e.g. a tensor.empty inserted
    // by linalg-generalize-named-ops between two named ops), move them before
    // the first member.
    {
      Operation *insertBefore = collapsedInfo.topoMembers.front();
      // Recursively hoist op and its pure operand-defining ops before
      // insertBefore. Handles multi-op collapse where intermediate
      // tensor.empty / collapse_shape ops appear between linalg members.
      DenseSet<Operation *> memberSet;
      for (linalg::LinalgOp m : collapsedInfo.topoMembers)
        memberSet.insert(m.getOperation());
      std::function<void(Operation *)> hoistBefore = [&](Operation *op) {
        if (!op) return;
        if (op->getBlock() != insertBefore->getBlock()) return;
        if (!insertBefore->isBeforeInBlock(op)) return; // already before
        if (memberSet.count(op)) return; // never hoist another member
        // Recurse into operands first so SSA order is preserved.
        for (Value v : op->getOperands())
          hoistBefore(v.getDefiningOp());
        op->moveBefore(insertBefore);
      };
      // Hoist init tensors (needed as iter_args) and all non-member inputs
      // (e.g. collapse_shape of function args inserted for non-first members
      // by applyMultiOpIRTransform — these must precede the loop).
      for (Value init : initTensors) {
        Operation *defOp = init.getDefiningOp();
        if (!defOp) continue; // block argument — always dominates
        hoistBefore(defOp);
      }
      for (linalg::LinalgOp m : collapsedInfo.topoMembers) {
        for (Value inp : m.getDpsInputs()) {
          Operation *defOp = inp.getDefiningOp();
          if (!defOp || memberSet.count(defOp)) continue;
          hoistBefore(defOp);
        }
      }
    }
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
