#include "Conversion/VectorPlan/VectorPlanPasses.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DECL_VECTORPLANBROADCASTABSORB
#define GEN_PASS_DEF_VECTORPLANBROADCASTABSORB
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir::afir {

namespace {

// Drop results at the given sorted positions from an AffineMap.
static AffineMap dropResultsAt(AffineMap map,
                                ArrayRef<int64_t> sortedPositions) {
  DenseSet<int64_t> posSet(sortedPositions.begin(), sortedPositions.end());
  SmallVector<AffineExpr> kept;
  for (auto [i, expr] : llvm::enumerate(map.getResults()))
    if (!posSet.count((int64_t)i))
      kept.push_back(expr);
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                        kept, map.getContext());
}

struct AbsorbBroadcastIntoGeneric : OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp bcast,
                                PatternRewriter &rewriter) const override {
    // BroadcastOp has a variadic result; grab the single tensor result.
    if (bcast->getNumResults() != 1)
      return failure();
    Value bcastResult = bcast->getResult(0);
    if (!bcastResult.hasOneUse())
      return failure();
    auto *user = *bcastResult.getUsers().begin();
    auto generic = dyn_cast<GenericOp>(user);
    if (!generic)
      return failure();

    int32_t opIdx = -1;
    for (auto [i, inp] : llvm::enumerate(generic.getInputs())) {
      if (inp == bcastResult) {
        opIdx = (int32_t)i;
        break;
      }
    }
    if (opIdx < 0)
      return failure();

    // Get broadcast dimensions as sorted int64_t list.
    SmallVector<int64_t> dims(bcast.getDimensions().begin(),
                              bcast.getDimensions().end());
    llvm::sort(dims);

    SmallVector<AffineMap> maps(generic.getIndexingMapsArray());
    maps[opIdx] = dropResultsAt(maps[opIdx], dims);

    rewriter.modifyOpInPlace(generic, [&]() {
      generic->setOperand(opIdx, bcast.getInput());
      generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    });

    if (bcastResult.use_empty())
      rewriter.eraseOp(bcast);

    return success();
  }
};

struct VectorPlanBroadcastAbsorbPass
    : public ::impl::VectorPlanBroadcastAbsorbBase<
          VectorPlanBroadcastAbsorbPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AbsorbBroadcastIntoGeneric>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void populateBroadcastAbsorbPatterns(RewritePatternSet &patterns) {
  patterns.add<AbsorbBroadcastIntoGeneric>(patterns.getContext());
}

std::unique_ptr<Pass> createVectorPlanBroadcastAbsorbPass() {
  return std::make_unique<VectorPlanBroadcastAbsorbPass>();
}

} // namespace mlir::afir
