#include "Collapse.h"
#include "GroupEmitter.h"
#include "LoopNestBuilder.h"
#include "SliceComputer.h"
#include "TileFuseUtils.h"
#include "TilePlanGen.h"
#include "Conversion/VectorPlan/GroupInfo.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "Conversion/VectorPlan/VectorPlanPasses.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

/// Apply the full collapse → buildPlan → loop-nest → emit-group pipeline to
/// `func`, using the caller-provided `draft`. `func` is assumed to be fresh
/// (broadcast-absorbed but not yet collapsed). All IR mutations happen here.
static void runVariantPipeline(func::FuncOp func,
                                const TilePlanDraft &draft) {
  OpBuilder builder(func.getContext());

  // Phase 1: Collapse.
  auto collapsedInfo = collapseGroup(builder, func);
  if (collapsedInfo.topoMembers.empty()) return;

  // Phase 2: materialize the assigned draft into a TilePlan.
  builder.setInsertionPointToStart(&func.getBody().front());
  TilePlan plan = buildPlanForDraft(func, collapsedInfo, draft, builder,
                                     func.getLoc());
  emitTilingInfos(func, plan);

  // Collect init tensors and original results BEFORE modification. Same
  // logic as the legacy single-plan path — an intra-group intermediate
  // (only consumed by other members) needs no scf.for iter_arg.
  SmallVector<Value> originalResults;
  SmallVector<Value> initTensors;
  DenseSet<Value> seenInits;
  for (linalg::LinalgOp op : collapsedInfo.topoMembers) {
    if (resultUsedOnlyByGroupMembers(op, collapsedInfo))
      continue;
    for (Value r : op->getResults())
      originalResults.push_back(r);
    for (Value out : op.getDpsInits())
      if (seenInits.insert(out).second)
        initTensors.push_back(out);
  }

  // Phase 3a: LoopNestBuilder (hoist initTensors / non-member inputs first).
  {
    Operation *insertBefore = collapsedInfo.topoMembers.front();
    DenseSet<Operation *> memberSet;
    for (linalg::LinalgOp m : collapsedInfo.topoMembers)
      memberSet.insert(m.getOperation());
    std::function<void(Operation *)> hoistBefore = [&](Operation *op) {
      if (!op) return;
      if (op->getBlock() != insertBefore->getBlock()) return;
      if (!insertBefore->isBeforeInBlock(op)) return;
      if (memberSet.count(op)) return;
      for (Value v : op->getOperands())
        hoistBefore(v.getDefiningOp());
      op->moveBefore(insertBefore);
    };
    for (Value init : initTensors) {
      Operation *defOp = init.getDefiningOp();
      if (!defOp) continue;
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
  auto loopNest = buildLoopNest(builder, func.getLoc(), plan, initTensors);

  // Phase 3b+c: GroupEmitter.
  builder.setInsertionPointToEnd(loopNest.innermostBody);
  auto loopResults =
      emitGroup(builder, func.getLoc(), collapsedInfo, plan, loopNest);

  for (auto [origRes, loopRes] : llvm::zip(originalResults, loopResults))
    origRes.replaceAllUsesWith(loopRes);
  for (linalg::LinalgOp op : llvm::reverse(collapsedInfo.topoMembers))
    op->erase();
}

/// True iff `func` carries any linalg.generic op (after group-outline, the
/// only funcs of interest to TileFuse).
static bool funcHasLinalgContent(func::FuncOp func) {
  bool found = false;
  func.walk([&](linalg::LinalgOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

struct VectorPlanTileFusePass
    : public ::impl::VectorPlanTileFuseBase<VectorPlanTileFusePass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Collect candidate funcs up-front (we'll be inserting siblings).
    SmallVector<func::FuncOp> originals;
    for (auto func : mod.getOps<func::FuncOp>())
      if (funcHasLinalgContent(func) && !func.getName().contains("__v"))
        originals.push_back(func);

    for (func::FuncOp original : originals)
      processFunc(original, mod);
  }

  void processFunc(func::FuncOp func, ModuleOp mod) {
    std::string originalName = func.getName().str();

    // Phase 0: broadcast absorb (shared across all variants — same input IR).
    {
      RewritePatternSet patterns(&getContext());
      populateBroadcastAbsorbPatterns(patterns);
      (void)applyPatternsGreedily(func, std::move(patterns));
    }

    // Discovery: clone the (broadcast-absorbed) func, collapse it, enumerate
    // feasible drafts, then erase. This is the only way to learn N without
    // committing to a tiling — collapse and buildPlan both mutate IR.
    func::FuncOp discoveryClone = cast<func::FuncOp>(func->clone());
    discoveryClone.setName("__tilefuse_discovery_" + originalName);
    mod.push_back(discoveryClone);

    SmallVector<TilePlanDraft> feasible;
    {
      OpBuilder b(func.getContext());
      auto info = collapseGroup(b, discoveryClone);
      if (!info.topoMembers.empty()) {
        feasible = enumerateFeasibleDrafts(info, enableReductionSplit,
                                            enableTilingVariants &&
                                                relaxNonBlockUbY);
      }
    }
    discoveryClone.erase();

    if (feasible.empty()) {
      // Defensive: no feasible draft (e.g. linalg func with no parallel /
      // reduce axis). Leave the func alone — no rename, no codegen — so the
      // downstream emitter can still see the original linalg form.
      return;
    }

    // When the user opts out of multi-variant codegen, keep the legacy
    // behavior: still rename to __v0 (so the rest of the pipeline P2-P5
    // sees a uniform `<original>__v<idx>` naming), but only ever build one
    // variant (the first feasible draft = pickBest's choice).
    if (!enableTilingVariants && feasible.size() > 1)
      feasible.resize(1);

    // Fan-out: clone the original N-1 times. variantFuncs[0] is the original;
    // variantFuncs[i] for i>0 is a fresh clone. Rename each to <original>__v<i>.
    SmallVector<func::FuncOp> variantFuncs = {func};
    for (size_t i = 1; i < feasible.size(); ++i) {
      auto clone = cast<func::FuncOp>(func->clone());
      mod.push_back(clone);
      variantFuncs.push_back(clone);
    }
    for (size_t i = 0; i < variantFuncs.size(); ++i)
      variantFuncs[i].setName(originalName + "__v" + std::to_string(i));

    // Process each variant independently. The clones share the same
    // (broadcast-absorbed, pre-collapse) IR as the original, so re-running
    // collapse per variant is deterministic.
    for (size_t i = 0; i < variantFuncs.size(); ++i)
      runVariantPipeline(variantFuncs[i], feasible[i]);
  }
};

} // namespace

std::unique_ptr<Pass> createVectorPlanTileFusePass() {
  return std::make_unique<VectorPlanTileFusePass>();
}

} // namespace mlir::afir
