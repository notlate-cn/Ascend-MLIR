#include "Conversion/LowerNonLinalgOps/LowerNonLinalgOpsPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

#define GEN_PASS_DECL_RECOGNIZEATTENTION
#define GEN_PASS_DEF_RECOGNIZEATTENTION
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

//===----------------------------------------------------------------------===//
// RecognizeAttentionPattern
//
// Anchors on the 2nd batch_matmul (bmm2 = probs @ V).  Walks back from
// bmm2.lhs through the softmax glue (generic/collapse/expand/fill) until it
// reaches the 1st batch_matmul (bmm1 = Q @ K^T).  The chain must contain a
// `math.exp` — that is what distinguishes attention from two arbitrary chained
// matmuls.  On a match:
//
//   Q   = bmm1.lhs  [BH,S,D]
//   K^T = bmm1.rhs  [BH,D,S]   ->  K = transpose(K^T, last two) [BH,S,D]
//   V   = bmm2.rhs  [BH,S,D]
//
// each expanded to BNSD [BH,1,S,D], passed to @__aclnn_flash_attention, and the
// result collapsed back to [BH,S,D] to replace bmm2.  The scale (1/sqrt(D)) and
// softmax are recomputed inside FlashAttentionScore, so the matched region is
// discarded (left dead for DCE).
//===----------------------------------------------------------------------===//

static constexpr StringRef kAclnnFuncName = "__aclnn_flash_attention";

// Backward search from `start` over the softmax glue.  Returns the unique
// upstream batch_matmul (bmm1) iff a single one is reachable AND a math.exp was
// seen along the way; otherwise null.
static linalg::BatchMatmulOp traceToBmm1(Value start) {
  SmallVector<Value> worklist{start};
  DenseSet<Operation *> visited;
  linalg::BatchMatmulOp bmm1;
  bool sawExp = false;

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def)
      continue;
    if (auto bm = dyn_cast<linalg::BatchMatmulOp>(def)) {
      if (bmm1 && bmm1 != bm)
        return {}; // ambiguous: more than one upstream matmul
      bmm1 = bm;
      continue; // boundary — do not traverse into Q / K^T
    }
    if (!visited.insert(def).second)
      continue;
    def->walk([&](math::ExpOp) { sawExp = true; });
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }

  if (bmm1 && sawExp)
    return bmm1;
  return {};
}

// Get or insert the @__aclnn_flash_attention private decl (fully-dynamic 4D
// tensors).  Shares the contract with convert-tm-tensor-attention.
static func::FuncOp getOrCreateAclnnDecl(ModuleOp module, Type elemType) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(kAclnnFuncName))
    return existing;

  auto dynT = RankedTensorType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic,
       ShapedType::kDynamic},
      elemType);

  OpBuilder b(module.getContext());
  b.setInsertionPointToStart(module.getBody());
  auto funcType = b.getFunctionType(
      TypeRange{dynT, dynT, dynT, dynT, dynT}, // Q, K, V, mask, init
      TypeRange{dynT});
  auto funcOp =
      b.create<func::FuncOp>(b.getUnknownLoc(), kAclnnFuncName, funcType);
  funcOp.setPrivate();
  funcOp->setAttr("aclnn.kind", b.getStringAttr("flash_attention"));
  return funcOp;
}

struct RecognizeAttentionPattern
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp bmm2,
                                PatternRewriter &rewriter) const override {
    Location loc = bmm2.getLoc();

    Value probs = bmm2.getInputs()[0];
    Value v = bmm2.getInputs()[1];
    linalg::BatchMatmulOp bmm1 = traceToBmm1(probs);
    if (!bmm1)
      return rewriter.notifyMatchFailure(bmm2, "no upstream attention bmm1");

    Value q = bmm1.getInputs()[0];   // [BH,S,D]
    Value kt = bmm1.getInputs()[1];  // [BH,D,S]

    auto qType = dyn_cast<RankedTensorType>(q.getType());
    auto ktType = dyn_cast<RankedTensorType>(kt.getType());
    auto vType = dyn_cast<RankedTensorType>(v.getType());
    if (!qType || !ktType || !vType || !qType.hasStaticShape() ||
        qType.getRank() != 3)
      return rewriter.notifyMatchFailure(bmm2, "non-static rank-3 operands");

    int64_t BH = qType.getDimSize(0);
    int64_t S = qType.getDimSize(1);
    int64_t D = qType.getDimSize(2);
    Type elemType = qType.getElementType();

    // K = transpose(K^T, [0,2,1]) : [BH,D,S] -> [BH,S,D]
    Value kInit = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{BH, S, D}, elemType);
    Operation *kT = rewriter.create<linalg::TransposeOp>(
        loc, kt, kInit, ArrayRef<int64_t>{0, 2, 1});
    Value k = kT->getResult(0);

    // Expand each [BH,S,D] -> BNSD [BH,1,S,D].
    auto bnsd = RankedTensorType::get({BH, 1, S, D}, elemType);
    SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}, {3}};
    auto expand = [&](Value src) -> Value {
      return rewriter.create<tensor::ExpandShapeOp>(loc, bnsd, src, reassoc);
    };
    Value q4 = expand(q);
    Value k4 = expand(k);
    Value v4 = expand(v);

    // Cast to the fully-dynamic declared type and call.
    auto module = bmm2->getParentOfType<ModuleOp>();
    func::FuncOp decl = getOrCreateAclnnDecl(module, elemType);
    auto dynT = cast<RankedTensorType>(decl.getFunctionType().getInput(0));
    auto castDyn = [&](Value val) -> Value {
      if (val.getType() == dynT)
        return val;
      return rewriter.create<tensor::CastOp>(loc, dynT, val);
    };
    // mask/init are dummy slots: FlashAttentionScore here is bidirectional
    // (no mask) and the host CPU-reference run_FlashAttentionScore ignores both
    // (sdpa_cpu allocates its own output). Reuse Q's casted value rather than
    // fresh tensor.empty's — a standalone empty that feeds only this call has no
    // kernel-group affiliation and the group-outline reorder can sink it past
    // the call (invalid SSA / no network-json provenance).
    Value q4dyn = castDyn(q4);
    SmallVector<Value> args = {q4dyn, castDyn(k4), castDyn(v4), q4dyn, q4dyn};
    auto call = rewriter.create<func::CallOp>(loc, decl, args);

    // Cast result back to BNSD, collapse to [BH,S,D], replace bmm2.
    Value result4 = call.getResult(0);
    if (result4.getType() != bnsd)
      result4 = rewriter.create<tensor::CastOp>(loc, bnsd, result4);
    auto outType = cast<RankedTensorType>(bmm2.getResult(0).getType());
    Value out3 = rewriter.create<tensor::CollapseShapeOp>(loc, outType, result4,
                                                          reassoc);
    rewriter.replaceOp(bmm2, out3);
    return success();
  }
};

struct RecognizeAttentionPass
    : public ::impl::RecognizeAttentionBase<RecognizeAttentionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RecognizeAttentionPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRecognizeAttentionPass() {
  return std::make_unique<RecognizeAttentionPass>();
}

} // namespace mlir::afir
