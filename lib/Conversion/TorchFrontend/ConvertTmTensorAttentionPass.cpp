#include "Conversion/TorchFrontend/TorchFrontendPasses.h"
#include "Dialect/TmTensor/IR/TmTensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DECL_CONVERTTMTENSORATTENTION
#define GEN_PASS_DEF_CONVERTTMTENSORATTENTION
#include "Conversion/TorchFrontend/TorchFrontendPasses.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// AttentionToAclnnCallPattern
//
// tm_tensor.attention ins(Q3D, K3D, V3D, mask3D) outs(init3D) -> T3D
//   where Q3D etc. are produced by tensor.collapse_shape from 4D [B,N,S,D]
//
// => bypass the collapse, use the pre-collapse 4D sources directly:
//
//   func.call @__aclnn_flash_attention(Q4D, K4D, V4D, mask4D, init4D)
//   tensor.collapse_shape %result4D → T3D   (restore original 3D type)
//
// The 4D layout is [B, N, S, D] (BNSD), directly usable by aclnn.
//===----------------------------------------------------------------------===//

static constexpr StringRef kAclnnFuncName = "__aclnn_flash_attention";

// If `val` comes from a tensor.collapse_shape that collapses a 4D tensor to
// 3D, return the pre-collapse 4D source; otherwise return null.
static Value getPreCollapseSrc(Value val) {
  auto collapse = dyn_cast_or_null<tensor::CollapseShapeOp>(val.getDefiningOp());
  if (!collapse || collapse.getSrcType().getRank() != 4)
    return {};
  return collapse.getSrc();
}

// Get or insert the @__aclnn_flash_attention private func declaration.
// Uses fully-dynamic 4D tensor types so one declaration covers all shapes.
static func::FuncOp getOrCreateAclnnDecl(ModuleOp module, Type elemType,
                                          MLIRContext *ctx) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kAclnnFuncName))
    return existing;

  auto dynT = RankedTensorType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic,
       ShapedType::kDynamic, ShapedType::kDynamic},
      elemType);

  OpBuilder b(ctx);
  b.setInsertionPointToStart(module.getBody());

  auto funcType = b.getFunctionType(
      TypeRange{dynT, dynT, dynT, dynT, dynT}, // Q, K, V, mask, init
      TypeRange{dynT});

  auto funcOp = b.create<func::FuncOp>(b.getUnknownLoc(), kAclnnFuncName, funcType);
  funcOp.setPrivate();
  funcOp->setAttr("aclnn.kind", b.getStringAttr("flash_attention"));
  return funcOp;
}

struct AttentionToAclnnCallPattern
    : public OpRewritePattern<tm_tensor::AttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tm_tensor::AttentionOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Retrieve pre-collapse 4D sources: [B, N, S, D] and [B, N, S, S]
    Value q4d    = getPreCollapseSrc(op.getInputs()[0]);
    Value k4d    = getPreCollapseSrc(op.getInputs()[1]);
    Value v4d    = getPreCollapseSrc(op.getInputs()[2]);
    Value mask4d = getPreCollapseSrc(op.getInputs()[3]);

    if (!q4d || !k4d || !v4d || !mask4d)
      return rewriter.notifyMatchFailure(
          op, "attention inputs not produced by 4D→3D collapse_shape");

    auto q4dType = cast<RankedTensorType>(q4d.getType());
    Type elemType = q4dType.getElementType();

    // Build 4D init tensor for the output.
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < 4; ++i)
      if (q4dType.isDynamicDim(i))
        dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, q4d, i));
    Value init4d = rewriter.create<tensor::EmptyOp>(loc, q4dType, dynSizes);

    // Ensure the declaration exists at module level.
    auto module = op->getParentOfType<ModuleOp>();
    auto declFuncOp = getOrCreateAclnnDecl(module, elemType, rewriter.getContext());

    // Cast concrete-shaped 4D tensors to the fully-dynamic declared type.
    auto dynT = cast<RankedTensorType>(declFuncOp.getFunctionType().getInput(0));
    auto castDyn = [&](Value v) -> Value {
      if (v.getType() == dynT)
        return v;
      return rewriter.create<tensor::CastOp>(loc, dynT, v);
    };

    SmallVector<Value> args = {
        castDyn(q4d), castDyn(k4d), castDyn(v4d), castDyn(mask4d), castDyn(init4d)
    };

    auto callOp = rewriter.create<func::CallOp>(loc, declFuncOp, args);
    Value result4D = callOp.getResult(0);

    // Cast result back to the concrete 4D type.
    if (result4D.getType() != q4dType)
      result4D = rewriter.create<tensor::CastOp>(loc, q4dType, result4D);

    // Collapse [B, N, S, D] → [B*N, S, D] to restore the original 3D result type.
    auto origResultType = cast<RankedTensorType>(op.getResult(0).getType());
    auto qCollapse = cast<tensor::CollapseShapeOp>(op.getInputs()[0].getDefiningOp());
    Value result3D = rewriter.create<tensor::CollapseShapeOp>(
        loc, origResultType, result4D, qCollapse.getReassociationIndices());

    rewriter.replaceOp(op, result3D);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertTmTensorAttentionPass
    : public ::impl::ConvertTmTensorAttentionBase<ConvertTmTensorAttentionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AttentionToAclnnCallPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertTmTensorAttentionPass() {
  return std::make_unique<ConvertTmTensorAttentionPass>();
}

} // namespace mlir::afir