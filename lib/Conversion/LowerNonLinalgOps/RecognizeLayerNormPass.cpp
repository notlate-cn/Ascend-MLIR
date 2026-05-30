#include "Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h"
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

#define GEN_PASS_DECL_RECOGNIZELAYERNORM
#define GEN_PASS_DEF_RECOGNIZELAYERNORM
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// RecognizeLayerNormPattern
//
// Anchors on the linalg.generic containing the `math.rsqrt`, then walks the
// torch LayerNorm lowering:
//
//   rstd-generic -> [collapse] -> broadcast-generic -> norm = mulf(cen, rstd)
//   cen = subf(x, mean)                      (x = the centered input)
//   norm -> scaled = mulf(norm, gamma)       (gamma rank-1 affine weight)
//   scaled -> out = addf(scaled, beta)       (beta  rank-1 affine bias)
//
// and replaces `out` with @__aclnn_layer_norm(x, gamma, beta).  eps and the
// mean/variance are recomputed inside the aclnn op.
//===----------------------------------------------------------------------===//

static constexpr StringRef kAclnnFuncName = "__aclnn_layer_norm";

template <typename OpT>
static bool bodyHas(linalg::GenericOp g) {
  bool found = false;
  g.getBody()->walk([&](OpT) { found = true; });
  return found;
}

// A pure broadcast/copy generic: one input, body just yields that input.
static bool isBroadcastCopy(linalg::GenericOp g) {
  if (g.getInputs().size() != 1 || g.getNumResults() != 1)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(g.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  return yield.getOperand(0) == g.getBody()->getArgument(0);
}

// The UNIQUE linalg.generic user of `v` whose body contains an OpT.  Returns
// null if there is no such user OR if there is more than one: with multiple
// candidates (e.g. the gamma-scaled value also feeds a residual/second-bias
// add) we cannot tell which generic is the real layernorm affine op, and
// guessing would silently fold with the wrong gamma/beta.  The caller bails on
// null rather than guess.
template <typename OpT>
static linalg::GenericOp userGenericWithBody(Value v) {
  linalg::GenericOp found;
  for (Operation *u : v.getUsers())
    if (auto g = dyn_cast<linalg::GenericOp>(u))
      if (bodyHas<OpT>(g)) {
        if (found)
          return {}; // ambiguous
        found = g;
      }
  return found;
}

// The (single) other tensor input of a 2-input generic.
static Value otherInput(linalg::GenericOp g, Value known) {
  for (Value in : g.getInputs())
    if (in != known)
      return in;
  return {};
}

static func::FuncOp getOrCreateDecl(ModuleOp module, Type elemType,
                                    int64_t rank) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kAclnnFuncName))
    return existing;

  SmallVector<int64_t> dyn(rank, ShapedType::kDynamic);
  auto xT = RankedTensorType::get(dyn, elemType);
  auto vecT = RankedTensorType::get({ShapedType::kDynamic}, elemType);

  OpBuilder b(module.getContext());
  b.setInsertionPointToStart(module.getBody());
  auto funcType = b.getFunctionType(TypeRange{xT, vecT, vecT}, // x, gamma, beta
                                    TypeRange{xT});
  auto funcOp =
      b.create<func::FuncOp>(b.getUnknownLoc(), kAclnnFuncName, funcType);
  funcOp.setPrivate();
  funcOp->setAttr("aclnn.kind", b.getStringAttr("layer_norm"));
  return funcOp;
}

struct RecognizeLayerNormPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp gRstd,
                                PatternRewriter &rewriter) const override {
    if (!bodyHas<math::RsqrtOp>(gRstd))
      return failure();

    Location loc = gRstd.getLoc();
    Value rstd = gRstd.getResult(0);

    // rstd -> (optional collapse_shape) -> broadcast generic.
    linalg::GenericOp gBcast;
    auto findBcast = [&](Value v) {
      for (Operation *u : v.getUsers())
        if (auto g = dyn_cast<linalg::GenericOp>(u))
          if (isBroadcastCopy(g)) {
            gBcast = g;
            return;
          }
    };
    findBcast(rstd);
    if (!gBcast)
      for (Operation *u : rstd.getUsers())
        if (auto c = dyn_cast<tensor::CollapseShapeOp>(u))
          findBcast(c.getResult());
    if (!gBcast)
      return rewriter.notifyMatchFailure(gRstd, "no rstd broadcast");

    // norm = mulf(cen, rstdFull)
    linalg::GenericOp gNorm =
        userGenericWithBody<arith::MulFOp>(gBcast.getResult(0));
    if (!gNorm || gNorm.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(gRstd, "no normalize mul");
    Value cen = otherInput(gNorm, gBcast.getResult(0));

    // cen = subf(x, mean)
    auto gSub = cen ? cen.getDefiningOp<linalg::GenericOp>() : nullptr;
    if (!gSub || !bodyHas<arith::SubFOp>(gSub) || gSub.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(gRstd, "centered not x - mean");
    Value x = gSub.getInputs()[0];

    // scaled = mulf(norm, gamma) ; out = addf(scaled, beta)
    linalg::GenericOp gGamma =
        userGenericWithBody<arith::MulFOp>(gNorm.getResult(0));
    if (!gGamma || gGamma.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(gRstd, "no gamma scale");
    Value gamma = otherInput(gGamma, gNorm.getResult(0));

    linalg::GenericOp gBeta =
        userGenericWithBody<arith::AddFOp>(gGamma.getResult(0));
    if (!gBeta || gBeta.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(gRstd, "no beta add");
    Value beta = otherInput(gBeta, gGamma.getResult(0));
    Value output = gBeta.getResult(0);

    auto xType = dyn_cast<RankedTensorType>(x.getType());
    auto gType = dyn_cast<RankedTensorType>(gamma.getType());
    auto bType = dyn_cast<RankedTensorType>(beta.getType());
    if (!xType || !gType || !bType || gType.getRank() != 1 ||
        bType.getRank() != 1)
      return rewriter.notifyMatchFailure(gRstd, "unexpected operand ranks");

    Type elemType = xType.getElementType();
    auto module = gRstd->getParentOfType<ModuleOp>();
    func::FuncOp decl = getOrCreateDecl(module, elemType, xType.getRank());

    auto castTo = [&](Value v, Type t) -> Value {
      return v.getType() == t ? v : rewriter.create<tensor::CastOp>(loc, t, v);
    };
    SmallVector<Value> args = {
        castTo(x, decl.getFunctionType().getInput(0)),
        castTo(gamma, decl.getFunctionType().getInput(1)),
        castTo(beta, decl.getFunctionType().getInput(2))};
    auto call = rewriter.create<func::CallOp>(loc, decl, args);

    Value result = call.getResult(0);
    if (result.getType() != output.getType())
      result = rewriter.create<tensor::CastOp>(loc, output.getType(), result);
    rewriter.replaceOp(gBeta, result);
    return success();
  }
};

struct RecognizeLayerNormPass
    : public ::impl::RecognizeLayerNormBase<RecognizeLayerNormPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RecognizeLayerNormPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRecognizeLayerNormPass() {
  return std::make_unique<RecognizeLayerNormPass>();
}

} // namespace mlir::afir
