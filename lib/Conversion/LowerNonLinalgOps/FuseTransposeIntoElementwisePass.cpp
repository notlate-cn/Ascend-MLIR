#include "Conversion/LinalgToAscendC/ComputeConversionHelpers.h"
#include "Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DECL_FUSETRANSPOSEINTOELEMENTWISE
#define GEN_PASS_DEF_FUSETRANSPOSEINTOELEMENTWISE
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// FuseTransposeIntoElementwisePattern
//
// Phase-1 "eliminate transpose" fold for the network path.  A standalone
// `linalg.transpose` is otherwise kept as its own group and routed to the aclnn
// single-op fallback (CanFuse). When the transpose's SOLE consumer is a
// pure-elementwise `linalg.generic`, we can instead absorb the permutation into
// that consumer's input indexing map and delete the transpose op — the same
// "eliminate" form `--linalg-fuse-elementwise-ops` produces, but GATED to the
// cases the on-chip transpose template can actually codegen.
//
// Gate (all required):
//   1. transpose result has exactly one use,
//   2. that use is an *input* of an all-parallel (elementwise) linalg.generic,
//   3. transposeSupportedByConfusion(elemType, perm) — i.e. f16/f32 ∧ rank-2
//      [1,0] (codegen::AfirConfusionTranspose2D handles either dtype at any
//      size; anything else stays a separate transpose → aclnn, unchanged).
//
// Rewrite: t = transpose(x, perm) feeding the generic via map M becomes the
// generic reading x via M' with perm composed in.  Because
//   t[idx] = x[idx']  where  idx'[perm[d]] = idx[d],
// the new operand map satisfies  M'.result[perm[d]] = M.result[d].
//===----------------------------------------------------------------------===//

struct FuseTransposeIntoElementwisePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp gen,
                                PatternRewriter &rewriter) const override {
    // Consumer must be pure-elementwise: every iterator parallel.
    for (utils::IteratorType it : gen.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();

    SmallVector<AffineMap> maps = gen.getIndexingMapsArray();
    ValueRange inputs = gen.getInputs();
    for (unsigned i = 0; i < inputs.size(); ++i) {
      auto tp = inputs[i].getDefiningOp<linalg::TransposeOp>();
      if (!tp)
        continue;
      // Sole consumer: folding here must not strand the transpose for other
      // users (that would duplicate work / diverge layouts).
      if (!tp->getResult(0).hasOneUse())
        continue;

      ArrayRef<int64_t> perm = tp.getPermutation();
      auto inTy = cast<ShapedType>(tp.getInput().getType());
      if (!transposeSupportedByConfusion(inTy.getElementType(), perm))
        continue;

      // The consumer must read the full transpose result (no broadcast on this
      // operand) so the permutation composes cleanly.
      AffineMap M = maps[i];
      if (M.getNumResults() != perm.size())
        continue;

      SmallVector<AffineExpr> newResults(perm.size());
      for (unsigned d = 0; d < perm.size(); ++d)
        newResults[perm[d]] = M.getResult(d);
      AffineMap newM = AffineMap::get(M.getNumDims(), M.getNumSymbols(),
                                      newResults, getContext());

      SmallVector<AffineMap> newMaps = maps;
      newMaps[i] = newM;
      rewriter.modifyOpInPlace(gen, [&]() {
        gen->setOperand(i, tp.getInput());
        gen.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newMaps));
      });
      rewriter.eraseOp(tp);
      return success();
    }
    return failure();
  }
};

struct FuseTransposeIntoElementwisePass
    : public ::impl::FuseTransposeIntoElementwiseBase<
          FuseTransposeIntoElementwisePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseTransposeIntoElementwisePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFuseTransposeIntoElementwisePass() {
  return std::make_unique<FuseTransposeIntoElementwisePass>();
}

} // namespace mlir::afir
