#include "LoopNestBuilder.h"
#include "TileFuseUtils.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

LoopNestResult buildLoopNest(OpBuilder &builder, Location loc,
                              const TilePlan &plan, ValueRange initTensors) {
  LoopNestResult result;
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> iterArgs(initTensors);
  DenseMap<int, Value> outerIVs; // axisIdx → outer loop IV

  auto emitFor = [&](Value lb, Value ub, Value step,
                     bool isParallelOuter) -> scf::ForOp {
    auto forOp = builder.create<scf::ForOp>(loc, lb, ub, step, iterArgs);
    if (isParallelOuter)
      forOp->setAttr("ascendc.parallel", builder.getUnitAttr());
    result.allForOps.push_back(forOp);
    // scf.ForOp builder auto-inserts a default scf.yield terminator; remove it
    // so subsequent ops are appended naturally and we emit our own yield later.
    if (!forOp.getBody()->empty()) {
      Operation &term = forOp.getBody()->back();
      if (isa<scf::YieldOp>(term)) term.erase();
    }
    builder.setInsertionPointToEnd(forOp.getBody());
    iterArgs = SmallVector<Value>(forOp.getRegionIterArgs());
    return forOp;
  };

  // 1. Outer loops (TileLevel::Outer).
  for (auto &group : plan.tileable)
    for (const auto &tp : group)
      if (tp.level == TileLevel::Outer) {
        Value ub = getAxisExtentValue(builder, loc, *plan.group, tp.axisIdx);
        auto forOp = emitFor(c0, ub, tp.ssa, /*isParallelOuter=*/true);
        outerIVs[tp.axisIdx] = forOp.getInductionVar();
        result.outerLoopIVs[tp.axisIdx] = forOp.getInductionVar();
      }

  // 2. BCast Full loops (plan.full entries with AxisRole::Parallel).
  for (const auto &tp : plan.full) {
    if (tp.role != AxisRole::Parallel) continue;
    Value ub = getAxisExtentValue(builder, loc, *plan.group, tp.axisIdx);
    auto forOp = emitFor(c0, ub, tp.ssa, /*isParallelOuter=*/false);
    result.bcastForOps.push_back(forOp);
    result.loopIVs[tp.axisIdx] = forOp.getInductionVar();
    result.outerLoopIVs[tp.axisIdx] = forOp.getInductionVar();
  }

  // 3. Inner loops (TileLevel::Inner), sorted by axisIdx.
  SmallVector<const TileParam *> innerParams;
  for (auto &group : plan.tileable)
    for (const auto &tp : group)
      if (tp.level == TileLevel::Inner)
        innerParams.push_back(&tp);
  llvm::sort(innerParams, [](const TileParam *a, const TileParam *b) {
    return a->axisIdx < b->axisIdx;
  });

  // Peel only the innermost (last) inner tile axis so the main body sees a
  // static slice size while the tail handles `extent % step != 0`.  Other
  // inner-tile axes (rare in current workloads) keep their original ub.
  const TileParam *innermostInner = innerParams.empty() ? nullptr
                                                          : innerParams.back();

  for (const TileParam *tp : innerParams) {
    // parentStep is this inner axis's "tile budget":
    //   - outer XBLOCK when an Outer level exists for this axis,
    //   - full axis extent otherwise.
    Value parentStep;
    Value parentIV = c0;
    if (outerIVs.count(tp->axisIdx)) {
      for (auto &group : plan.tileable)
        for (const auto &op2 : group)
          if (op2.level == TileLevel::Outer && op2.axisIdx == tp->axisIdx)
            parentStep = op2.ssa;
      parentIV = outerIVs[tp->axisIdx];
    }
    if (!parentStep)
      parentStep = getAxisExtentValue(builder, loc, *plan.group, tp->axisIdx);

    Value ub = parentStep;
    Value remaining, mainInnerUb;

    if (tp == innermostInner) {
      // remaining = min(parentStep, extent - parentIV)
      Value extent = getAxisExtentValue(builder, loc, *plan.group, tp->axisIdx);
      Value extMinusParent =
          builder.create<arith::SubIOp>(loc, extent, parentIV);
      remaining = builder.create<arith::MinSIOp>(loc, parentStep, extMinusParent);
      Value q = builder.create<arith::DivSIOp>(loc, remaining, tp->ssa);
      mainInnerUb = builder.create<arith::MulIOp>(loc, q, tp->ssa);
      ub = mainInnerUb;
    }

    auto forOp = emitFor(c0, ub, tp->ssa, /*isParallelOuter=*/false);

    if (outerIVs.count(tp->axisIdx)) {
      Value composed = builder.create<arith::AddIOp>(
          loc, outerIVs[tp->axisIdx], forOp.getInductionVar());
      result.loopIVs[tp->axisIdx] = composed;
      // outerLoopIVs[axisIdx] already set to the outer IV above.
    } else {
      result.loopIVs[tp->axisIdx] = forOp.getInductionVar();
      result.outerLoopIVs[tp->axisIdx] = forOp.getInductionVar();
    }

    if (tp == innermostInner) {
      result.hasTail = true;
      result.innerTileAxisIdx = tp->axisIdx;
      result.remaining = remaining;
      result.mainInnerUb = mainInnerUb;
      result.outerOfTailIV = parentIV;
    }
  }

  result.innermostBody = builder.getInsertionBlock();
  result.iterArgs = iterArgs;
  return result;
}

} // namespace mlir::afir
