#include "Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DECL_RECOGNIZEEMBEDDING
#define GEN_PASS_DEF_RECOGNIZEEMBEDDING
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// RecognizeEmbeddingPattern
//
// torch.export lowers nn.Embedding(input_ids) to a linalg.generic with body:
//
//   %5 = arith.index_cast %in : i64 to index            // indices[i] → index
//   %6 = linalg.index 1   : index                        // column index
//   %7 = arith.cmpi slt, %5, <vocab> : index
//   cf.assert %7, "index must be smaller than dim size"
//   %8 = arith.cmpi sge, %in, %c0_i64 : i64
//   cf.assert %8, "index must be larger or equal to 0"
//   %extracted = tensor.extract %table[%5, %6] : tensor<VxFxf32>
//   linalg.yield %extracted : f32
//
// TileFuse codegen can't lower this pattern (the tensor.extract inside a
// fused multi-generic group breaks SSA dominance during loopnest emit). We
// recognize the pattern and replace the linalg.generic with
//   func.call @__aclnn_embedding(table, indices) → out
// where `table` is the [V, F] embedding matrix and `indices` is the rank-N
// integer tensor of token IDs (the output rank is indices.rank + 1).
//===----------------------------------------------------------------------===//

static constexpr StringRef kAclnnFuncName = "__aclnn_embedding";

template <typename OpT>
static bool bodyHas(linalg::GenericOp g) {
  bool found = false;
  g.getBody()->walk([&](OpT) { found = true; });
  return found;
}

static func::FuncOp getOrCreateDecl(ModuleOp module,
                                    RankedTensorType tableType,
                                    RankedTensorType indicesType,
                                    RankedTensorType outType) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kAclnnFuncName))
    return existing;

  // Dynamic-shape signature: table [?, ?] / indices [?...] / out [?...].
  Type tabT = RankedTensorType::get(
      SmallVector<int64_t>(tableType.getRank(), ShapedType::kDynamic),
      tableType.getElementType());
  Type idxT = RankedTensorType::get(
      SmallVector<int64_t>(indicesType.getRank(), ShapedType::kDynamic),
      indicesType.getElementType());
  Type outT = RankedTensorType::get(
      SmallVector<int64_t>(outType.getRank(), ShapedType::kDynamic),
      outType.getElementType());

  OpBuilder b(module.getContext());
  b.setInsertionPointToStart(module.getBody());
  auto funcType = b.getFunctionType(TypeRange{tabT, idxT}, TypeRange{outT});
  auto funcOp =
      b.create<func::FuncOp>(b.getUnknownLoc(), kAclnnFuncName, funcType);
  funcOp.setPrivate();
  funcOp->setAttr("aclnn.kind", b.getStringAttr("embedding"));
  return funcOp;
}

struct RecognizeEmbeddingPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp gOp,
                                PatternRewriter &rewriter) const override {
    // Quick filter: the embedding-lookup generic has exactly one input (the
    // indices tensor), one output, and a body containing tensor.extract.
    if (gOp.getInputs().size() != 1 || gOp.getNumResults() != 1)
      return failure();
    if (!bodyHas<tensor::ExtractOp>(gOp))
      return failure();

    // Walk the body to find the tensor.extract; the table tensor is the
    // extract's source (which is captured from outside the linalg.generic).
    tensor::ExtractOp extractOp;
    gOp.getBody()->walk(
        [&](tensor::ExtractOp e) { extractOp = e; });
    if (!extractOp)
      return failure();
    Value table = extractOp.getTensor();

    // table and indices must be ranked tensors.
    auto tableType = dyn_cast<RankedTensorType>(table.getType());
    Value indices = gOp.getInputs()[0];
    auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
    auto outType = dyn_cast<RankedTensorType>(gOp.getResult(0).getType());
    if (!tableType || !indicesType || !outType)
      return failure();
    if (tableType.getRank() != 2)
      return rewriter.notifyMatchFailure(gOp, "embedding table must be rank-2");
    if (outType.getRank() != indicesType.getRank() + 1)
      return rewriter.notifyMatchFailure(
          gOp, "embedding out.rank must == indices.rank + 1");

    Location loc = gOp.getLoc();
    auto module = gOp->getParentOfType<ModuleOp>();
    func::FuncOp decl =
        getOrCreateDecl(module, tableType, indicesType, outType);

    auto castTo = [&](Value v, Type t) -> Value {
      return v.getType() == t ? v : rewriter.create<tensor::CastOp>(loc, t, v);
    };
    SmallVector<Value> args = {
        castTo(table, decl.getFunctionType().getInput(0)),
        castTo(indices, decl.getFunctionType().getInput(1))};
    auto call = rewriter.create<func::CallOp>(loc, decl, args);
    Value result = call.getResult(0);
    if (result.getType() != gOp.getResult(0).getType())
      result = rewriter.create<tensor::CastOp>(
          loc, gOp.getResult(0).getType(), result);
    rewriter.replaceOp(gOp, result);
    return success();
  }
};

struct RecognizeEmbeddingPass
    : public ::impl::RecognizeEmbeddingBase<RecognizeEmbeddingPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RecognizeEmbeddingPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRecognizeEmbeddingPass() {
  return std::make_unique<RecognizeEmbeddingPass>();
}

} // namespace mlir::afir
