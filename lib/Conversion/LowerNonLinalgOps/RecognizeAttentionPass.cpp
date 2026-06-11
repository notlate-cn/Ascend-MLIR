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
// upstream batch_matmul (bmm1) iff a single one is reachable AND a full softmax
// was seen along the way; otherwise null.  Softmax is identified by BOTH a
// math.exp AND a reduction (the normalization sum over the key dim) between the
// two bmms -- requiring the reduction distinguishes real softmax attention from
// a bare `exp(Q@K^T) @ V` (kernelized/linear attention, or two GEMMs with an
// exp activation), which must NOT be folded to FlashAttentionScore.
static linalg::BatchMatmulOp traceToBmm1(Value start) {
  SmallVector<Value> worklist{start};
  DenseSet<Operation *> visited;
  linalg::BatchMatmulOp bmm1;
  bool sawExp = false;
  bool sawReduction = false;

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
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(def))
      if (llvm::is_contained(linalgOp.getIteratorTypesArray(),
                             utils::IteratorType::reduction))
        sawReduction = true;
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }

  if (bmm1 && sawExp && sawReduction)
    return bmm1;
  return {};
}

// Walk forward from bmm1's result through reshape and scale glue, looking for
// an addf-only generic with 2 ranked-tensor inputs.  GPT-style causal/padding
// masks appear here as `(Q @ K^T) * scale + mask` — the second input (the one
// that's NOT the "main" attention path) is the mask we want to thread through.
// Returns the mask value (potentially rank-2 [S, S]) if found, null otherwise.
static Value findMaskFromBmm1(linalg::BatchMatmulOp bmm1) {
  auto bodyHas = [](linalg::GenericOp g, auto pred) {
    bool found = false;
    g.getBody()->walk([&](Operation *op) {
      if (pred(op))
        found = true;
    });
    return found;
  };
  Value v = bmm1.getResult(0);
  while (true) {
    if (!v.hasOneUse())
      return {};
    Operation *user = *v.getUsers().begin();
    if (auto ex = dyn_cast<tensor::ExpandShapeOp>(user)) {
      v = ex.getResult();
      continue;
    }
    if (auto cl = dyn_cast<tensor::CollapseShapeOp>(user)) {
      v = cl.getResult();
      continue;
    }
    auto gen = dyn_cast<linalg::GenericOp>(user);
    if (!gen)
      return {};
    bool hasMul =
        bodyHas(gen, [](Operation *op) { return isa<arith::MulFOp>(op); });
    bool hasAddf =
        bodyHas(gen, [](Operation *op) { return isa<arith::AddFOp>(op); });
    if (hasMul && !hasAddf && gen.getInputs().size() == 1) {
      // scale generic — descend.
      v = gen.getResult(0);
      continue;
    }
    if (hasAddf && !hasMul && gen.getInputs().size() == 2) {
      Value a = gen.getInputs()[0], b = gen.getInputs()[1];
      return (a == v) ? b : a;
    }
    return {};
  }
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
    if (!qType || !ktType || !vType || qType.getRank() != 3)
      return rewriter.notifyMatchFailure(bmm2, "non rank-3 operands");

    int64_t BH = qType.getDimSize(0);
    int64_t S = qType.getDimSize(1); // may be dynamic (variable seq length)
    int64_t D = qType.getDimSize(2);
    Type elemType = qType.getElementType();
    // BH (batch*heads) and D (head dim) must be static; only the sequence dim S
    // may be dynamic.  A dynamic BH/D would need runtime head counts the BNSD
    // construction below cannot express.
    if (ShapedType::isDynamic(BH) || ShapedType::isDynamic(D))
      return rewriter.notifyMatchFailure(bmm2, "dynamic batch*heads or head dim");

    // Dynamic sequence length: a Value for S (Q's dim 1), used to build the
    // dynamic tensor.empty / expand_shape sizes below.  sOFR() yields S as an
    // OpFoldResult (the dynamic Value, or a static index attr).
    bool dynS = ShapedType::isDynamic(S);
    Value sVal;
    if (dynS)
      sVal = rewriter.create<tensor::DimOp>(loc, q, 1);
    auto sOFR = [&]() -> OpFoldResult {
      return dynS ? OpFoldResult(sVal) : OpFoldResult(rewriter.getIndexAttr(S));
    };

    // Determine the additive mask up front -- before any IR mutation -- so we
    // can bail cleanly when a mask is present but cannot be represented in the
    // FlashAttentionScore [1,1,S,S] contract (only a rank-2 [S,S] mask is
    // supported; sdpa_cpu tells a real mask from the Q-placeholder by
    // shape[3] == S vs == D).  Folding such a region anyway would silently drop
    // the mask -- Q would be passed as the no-mask placeholder -- turning a
    // causal/padding-masked attention into a bidirectional one.  notifyMatchFailure
    // instead keeps the explicit mask add + bmms in place for the generic path.
    Value maskRank2; // rank-2 [S,S] mask to thread through, or null = no mask
    if (Value mask = findMaskFromBmm1(bmm1)) {
      auto mType = dyn_cast<RankedTensorType>(mask.getType());
      // Accept a rank-2 [S,S] mask whose dims match the (possibly dynamic) S:
      // getDimSize == S compares kDynamic==kDynamic when S is dynamic, or the
      // literal extents when static.
      if (mType && mType.getRank() == 2 && mType.getDimSize(0) == S &&
          mType.getDimSize(1) == S && mType.getElementType() == elemType)
        maskRank2 = mask;
      else
        return rewriter.notifyMatchFailure(
            bmm2, "attention mask present but unsupported by the "
                  "FlashAttentionScore [1,1,S,S] contract; refusing to fold "
                  "(would silently drop the mask)");
    }

    // K = transpose(K^T, [0,2,1]) : [BH,D,S] -> [BH,S,D]
    Value kInit = rewriter.create<tensor::EmptyOp>(
        loc,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(BH), sOFR(),
                               rewriter.getIndexAttr(D)},
        elemType);
    Operation *kT = rewriter.create<linalg::TransposeOp>(
        loc, kt, kInit, ArrayRef<int64_t>{0, 2, 1});
    Value k = kT->getResult(0);

    // Expand each [BH,S,D] -> BNSD [BH,1,S,D].
    auto bnsd = RankedTensorType::get({BH, 1, S, D}, elemType);
    SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}, {3}};
    SmallVector<OpFoldResult> bnsdSizes = {rewriter.getIndexAttr(BH),
                                           rewriter.getIndexAttr(1), sOFR(),
                                           rewriter.getIndexAttr(D)};
    auto expand = [&](Value src) -> Value {
      return rewriter.create<tensor::ExpandShapeOp>(loc, bnsd, src, reassoc,
                                                    bnsdSizes);
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
    // Thread the mask (determined above) through to FlashAttentionScore: expand
    // the rank-2 [S,S] mask to rank-4 [1,1,S,S] so sdpa_cpu can distinguish a
    // real mask from the Q-placeholder by checking shape[3] == S (vs Q's
    // shape[3] == D).  No mask -> keep the bidirectional path (Q as placeholder).
    Value q4dyn = castDyn(q4);
    Value maskArg = q4dyn;
    if (maskRank2) {
      auto mask4 = RankedTensorType::get({1, 1, S, S}, elemType);
      Value mask4Val;
      if (dynS) {
        // Dynamic causal mask: thread a distinct [1,1,S,S] buffer as a causal
        // *sentinel*.  FlashAttentionScore regenerates the triangular mask from
        // the runtime seq length (on both device and sdpa_cpu), so the buffer's
        // values are unused — only "a mask is present, shaped [1,1,S,S]" matters.
        // This lets the in-graph `triu` mask DCE instead of forcing an AscendC
        // kernel for its index/compare/select chain (which the vector codegen
        // can't tile).  The static path keeps threading the real mask.
        mask4Val = rewriter.create<tensor::EmptyOp>(
            loc,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                                   rewriter.getIndexAttr(1), sOFR(), sOFR()},
            elemType);
      } else {
        // Reassociation maps [S, S] -> [1, 1, S, S]: output dims {0,1,2}
        // collapse to input dim 0 (extents 1*1*S = S) and dim {3} → input
        // dim 1 (extent S).
        SmallVector<ReassociationIndices> maskReassoc = {{0, 1, 2}, {3}};
        mask4Val = rewriter.create<tensor::ExpandShapeOp>(loc, mask4, maskRank2,
                                                          maskReassoc);
      }
      maskArg = castDyn(mask4Val);
    }
    SmallVector<Value> args = {q4dyn, castDyn(k4), castDyn(v4), maskArg, q4dyn};
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
