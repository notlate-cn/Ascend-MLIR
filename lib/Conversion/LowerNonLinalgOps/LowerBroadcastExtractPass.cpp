#include "Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DECL_LOWERBROADCASTEXTRACT
#define GEN_PASS_DEF_LOWERBROADCASTEXTRACT
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

// torch lowers a *dynamic* broadcast (broadcasting a tensor whose dim might be
// 1) as an extract-based generic rather than a clean affine-map broadcast:
//
//   %g = linalg.generic outs(%init : tensor<?x64xf32>) {
//     ^bb0(%out):
//       %i  = linalg.index 0
//       %c  = arith.cmpi eq, %d, %c1          // %d = the source's dim extent
//       %s  = arith.select %c, %c0, %i         // dim==1 ? 0 : i   (broadcast guard)
//       %e  = tensor.extract %src[%c0, %s, %c0]
//       linalg.yield %e
//   }
//
// The `dim==1 ? 0 : i` guard is spurious when %d equals the loop-i extent (then
// i in [0,%d): %d==1 => i==0 anyway), which holds for LayerNorm's mean/rstd
// broadcasts (the source seq dim equals the output seq dim — same SSA value).
// This extract form defeats every pattern that expects an affine-map broadcast
// (RecognizeLayerNorm's isBroadcastCopy; the vector codegen).  Raise it back to
// a clean `ins`-broadcast generic so the rest of the pipeline sees the canonical
// form.  Soundness: only fires when each guard's %d is the SAME SSA value as the
// output init's matching loop extent.
struct LowerBroadcastExtractPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp g,
                                PatternRewriter &rewriter) const override {
    // Shape: 0 inputs, 1 DPS init, 1 result, all-parallel, identity out map.
    if (g.getInputs().size() != 0 || g.getNumResults() != 1 ||
        g.getOutputs().size() != 1)
      return failure();
    for (utils::IteratorType it : g.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();
    SmallVector<AffineMap> maps = g.getIndexingMapsArray();
    if (maps.size() != 1 || !maps[0].isIdentity())
      return failure();
    AffineMap outMap = maps[0];

    Block *body = g.getBody();
    auto yield = cast<linalg::YieldOp>(body->getTerminator());
    if (yield.getNumOperands() != 1)
      return failure();
    auto extract = yield.getOperand(0).getDefiningOp<tensor::ExtractOp>();
    if (!extract)
      return failure();
    Value src = extract.getTensor();
    if (src.getParentBlock() == body) // must be captured from outside
      return failure();
    auto srcType = dyn_cast<RankedTensorType>(src.getType());
    if (!srcType || (int64_t)extract.getIndices().size() != srcType.getRank())
      return failure();

    Value init = g.getOutputs()[0];
    auto outType = dyn_cast<RankedTensorType>(init.getType());
    if (!outType)
      return failure();
    auto initEmpty = init.getDefiningOp<tensor::EmptyOp>();

    // SSA value of loop-k extent (== output init dim k).  Only needed for the
    // dynamic guard case; null when unavailable.
    auto loopExtentValue = [&](unsigned k) -> Value {
      if (!outType.isDynamicDim(k) || !initEmpty)
        return Value();
      unsigned dynIdx = 0;
      for (unsigned d = 0; d < k; ++d)
        if (outType.isDynamicDim(d))
          ++dynIdx;
      if (dynIdx >= initEmpty.getDynamicSizes().size())
        return Value();
      return initEmpty.getDynamicSizes()[dynIdx];
    };

    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineExpr> srcExprs;
    for (Value idx : extract.getIndices()) {
      // Constant index -> constant affine expr (broadcast of a size-1 dim).
      if (auto c = getConstantIntValue(idx)) {
        srcExprs.push_back(getAffineConstantExpr(*c, ctx));
        continue;
      }
      // linalg.index K -> dim expr K.
      if (auto li = idx.getDefiningOp<linalg::IndexOp>()) {
        srcExprs.push_back(getAffineDimExpr(li.getDim(), ctx));
        continue;
      }
      // select(cmpi eq %d 1, 0, linalg.index K) -> dim K iff %d == extent(K).
      if (auto sel = idx.getDefiningOp<arith::SelectOp>()) {
        auto li = sel.getFalseValue().getDefiningOp<linalg::IndexOp>();
        auto trueC = getConstantIntValue(sel.getTrueValue());
        auto cmp = sel.getCondition().getDefiningOp<arith::CmpIOp>();
        if (li && trueC && *trueC == 0 && cmp &&
            cmp.getPredicate() == arith::CmpIPredicate::eq) {
          Value d;
          if (auto one = getConstantIntValue(cmp.getRhs()); one && *one == 1)
            d = cmp.getLhs();
          else if (auto one = getConstantIntValue(cmp.getLhs()); one && *one == 1)
            d = cmp.getRhs();
          unsigned k = li.getDim();
          Value ext = loopExtentValue(k);
          if (d && ext && d == ext) {
            srcExprs.push_back(getAffineDimExpr(k, ctx));
            continue;
          }
        }
      }
      return failure(); // unrecognized index form -> leave untouched
    }

    AffineMap bcastMap =
        AffineMap::get(g.getNumLoops(), /*symbolCount=*/0, srcExprs, ctx);
    SmallVector<AffineMap> newMaps = {bcastMap, outMap};

    auto newGeneric = rewriter.create<linalg::GenericOp>(
        g.getLoc(), TypeRange{outType}, ValueRange{src}, ValueRange{init},
        newMaps, g.getIteratorTypesArray(),
        [](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args[0]);
        });
    // Preserve the afir.* shape annotations (iter_extents / symbolic_shapes).
    if (auto a = g->getAttr("afir.iter_extents"))
      newGeneric->setAttr("afir.iter_extents", a);
    if (auto a = g->getAttr("afir.symbolic_shapes"))
      newGeneric->setAttr("afir.symbolic_shapes", a);
    rewriter.replaceOp(g, newGeneric.getResults());
    return success();
  }
};

struct LowerBroadcastExtractPass
    : public ::impl::LowerBroadcastExtractBase<LowerBroadcastExtractPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerBroadcastExtractPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerBroadcastExtractPass() {
  return std::make_unique<LowerBroadcastExtractPass>();
}

} // namespace mlir::afir
