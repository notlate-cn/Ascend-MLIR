#include "Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h"
#include "RecognizeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DECL_RECOGNIZEBATCHNORM
#define GEN_PASS_DEF_RECOGNIZEBATCHNORM
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// RecognizeBatchNormPattern
//
// Anchors on `math.sqrt` inside a linalg.generic, then walks the
// torch BatchNorm-eval-mode lowering:
//
//   add_eps = running_var + eps   (rank-1, [C])
//   sqrt    = sqrt(add_eps)       (rank-1, anchor)
//   inv_std = 1.0 / sqrt          (rank-1, with a cf.assert non-zero guard)
//   cen     = x - running_mean    (rank-N, broadcast over C dim)
//   norm    = cen * inv_std       (rank-N, broadcast)
//   scaled  = norm * weight       (rank-N, broadcast weight=gamma)
//   out     = scaled + bias       (rank-N, broadcast bias=beta)
//
// Replaces `out` with @__aclnn_batch_norm(x, weight, bias, running_mean,
// running_var). eps is recomputed inside the aclnn op (torch default 1e-5).
// The intermediate inv_std / centered / etc. become dead and DCE picks them up.
//===----------------------------------------------------------------------===//

static constexpr StringRef kAclnnFuncName = "__aclnn_batch_norm";

// bodyHas / userGenericWithBody / otherInput: see RecognizeUtils.h.

// Walk backward through tensor.expand_shape (the broadcast-prep torch.export
// emits to lift rank-1 affine params [C] up to rank-N for the elementwise
// broadcast).  Returns the underlying rank-1 source if found, else the original
// value.  Used to recover the 1-D mean/var/gamma/beta SSA values from their
// expanded [C, 1, 1] views.
static Value stripExpandShape(Value v) {
  while (auto exp = v.getDefiningOp<tensor::ExpandShapeOp>())
    v = exp.getSrc();
  return v;
}

static func::FuncOp getOrCreateDecl(ModuleOp module, Type elemType,
                                    int64_t rank) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kAclnnFuncName))
    return existing;

  SmallVector<int64_t> dynN(rank, ShapedType::kDynamic);
  auto xT = RankedTensorType::get(dynN, elemType);
  auto vecT = RankedTensorType::get({ShapedType::kDynamic}, elemType);

  OpBuilder b(module.getContext());
  b.setInsertionPointToStart(module.getBody());
  // (x, weight, bias, running_mean, running_var) -> x
  auto funcType =
      b.getFunctionType(TypeRange{xT, vecT, vecT, vecT, vecT}, TypeRange{xT});
  auto funcOp =
      b.create<func::FuncOp>(b.getUnknownLoc(), kAclnnFuncName, funcType);
  funcOp.setPrivate();
  funcOp->setAttr("aclnn.kind", b.getStringAttr("batch_norm"));
  return funcOp;
}

struct RecognizeBatchNormPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp gSqrt,
                                PatternRewriter &rewriter) const override {
    if (!bodyHas<math::SqrtOp>(gSqrt))
      return failure();
    if (gSqrt.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(gSqrt, "sqrt expects 1 input");

    Location loc = gSqrt.getLoc();

    // sqrt input = (running_var + eps).
    auto gAddEps = gSqrt.getInputs()[0].getDefiningOp<linalg::GenericOp>();
    if (!gAddEps || !bodyHas<arith::AddFOp>(gAddEps) ||
        gAddEps.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(gSqrt, "no running_var+eps producer");
    Value running_var = gAddEps.getInputs()[0];

    // sqrt result → divf-generic (1.0 / sqrt). The divf body also contains
    // arith.cmpf + cf.assert (torch's division-by-zero guard) — bodyHas<DivFOp>
    // is enough to anchor.
    linalg::GenericOp gDiv =
        userGenericWithBody<arith::DivFOp>(gSqrt.getResult(0));
    if (!gDiv || gDiv.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(gSqrt, "no 1/sqrt divf consumer");
    Value inv_std = gDiv.getResult(0);

    // inv_std → tensor.expand_shape (rank-1 [C] → rank-N [C,1,...,1] for the
    // broadcast over [N, C, H, W]) → mulf-generic.  The expand is what feeds
    // the mulf, not inv_std directly.
    Value invStdBcast = inv_std;
    for (Operation *u : inv_std.getUsers())
      if (auto exp = dyn_cast<tensor::ExpandShapeOp>(u)) {
        invStdBcast = exp.getResult();
        break;
      }
    linalg::GenericOp gNorm =
        userGenericWithBody<arith::MulFOp>(invStdBcast);
    if (!gNorm || gNorm.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(gSqrt, "no normalize mulf");
    Value centered = otherInput(gNorm, invStdBcast);

    // centered = subf(x, running_mean).
    auto gSub = centered ? centered.getDefiningOp<linalg::GenericOp>() : nullptr;
    if (!gSub || !bodyHas<arith::SubFOp>(gSub) || gSub.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(gSqrt, "centered not x - mean");
    Value x = gSub.getInputs()[0];
    Value running_mean = stripExpandShape(gSub.getInputs()[1]);

    // scaled = mulf(norm, gamma) ; out = addf(scaled, beta).
    linalg::GenericOp gGamma =
        userGenericWithBody<arith::MulFOp>(gNorm.getResult(0));
    if (!gGamma || gGamma.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(gSqrt, "no gamma scale");
    Value gamma = stripExpandShape(otherInput(gGamma, gNorm.getResult(0)));

    linalg::GenericOp gBeta =
        userGenericWithBody<arith::AddFOp>(gGamma.getResult(0));
    if (!gBeta || gBeta.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(gSqrt, "no beta add");
    Value beta = stripExpandShape(otherInput(gBeta, gGamma.getResult(0)));
    Value output = gBeta.getResult(0);

    auto xType = dyn_cast<RankedTensorType>(x.getType());
    auto wType = dyn_cast<RankedTensorType>(gamma.getType());
    auto bType = dyn_cast<RankedTensorType>(beta.getType());
    auto mType = dyn_cast<RankedTensorType>(running_mean.getType());
    auto vType = dyn_cast<RankedTensorType>(running_var.getType());
    if (!xType || !wType || !bType || !mType || !vType ||
        wType.getRank() != 1 || bType.getRank() != 1 ||
        mType.getRank() != 1 || vType.getRank() != 1)
      return rewriter.notifyMatchFailure(gSqrt, "unexpected operand ranks");

    Type elemType = xType.getElementType();
    auto module = gSqrt->getParentOfType<ModuleOp>();
    func::FuncOp decl = getOrCreateDecl(module, elemType, xType.getRank());

    auto castTo = [&](Value v, Type t) -> Value {
      return v.getType() == t ? v : rewriter.create<tensor::CastOp>(loc, t, v);
    };
    SmallVector<Value> args = {
        castTo(x, decl.getFunctionType().getInput(0)),
        castTo(gamma, decl.getFunctionType().getInput(1)),
        castTo(beta, decl.getFunctionType().getInput(2)),
        castTo(running_mean, decl.getFunctionType().getInput(3)),
        castTo(running_var, decl.getFunctionType().getInput(4))};
    auto call = rewriter.create<func::CallOp>(loc, decl, args);

    Value result = call.getResult(0);
    if (result.getType() != output.getType())
      result = rewriter.create<tensor::CastOp>(loc, output.getType(), result);

    // Capture gBeta's producer chain roots BEFORE replaceOp — afterwards we
    // lose the connection. linalg.generic ops aren't DCE'd by the greedy
    // pattern driver here (memory-effects conservatism), and the BN chain
    // has ~6 dead generics + 4 dead expand_shapes per BN. Without this the
    // 20-BN model leaks ~200 dead ops into kernel_group1.
    SmallVector<Operation *> deadRoots;
    for (Value v : gBeta.getOperands())
      if (Operation *def = v.getDefiningOp())
        deadRoots.push_back(def);
    rewriter.replaceOp(gBeta, result);

    // Walk each dead chain depth-first until fixed point — erase ops with no
    // remaining users, then re-check their operands.
    for (Operation *root : deadRoots) {
      SmallVector<Operation *> stack = {root};
      while (!stack.empty()) {
        Operation *op = stack.pop_back_val();
        if (!op || !op->use_empty())
          continue;
        SmallVector<Value, 4> opOperands(op->operand_begin(),
                                          op->operand_end());
        rewriter.eraseOp(op);
        for (Value o : opOperands)
          if (Operation *d = o.getDefiningOp())
            stack.push_back(d);
      }
    }
    return success();
  }
};

struct RecognizeBatchNormPass
    : public ::impl::RecognizeBatchNormBase<RecognizeBatchNormPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RecognizeBatchNormPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRecognizeBatchNormPass() {
  return std::make_unique<RecognizeBatchNormPass>();
}

} // namespace mlir::afir
