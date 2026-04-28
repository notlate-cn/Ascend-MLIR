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
      }

  // 2. BCast Full loops (plan.full entries with AxisRole::Parallel).
  for (const auto &tp : plan.full) {
    if (tp.role != AxisRole::Parallel) continue;
    Value ub = getAxisExtentValue(builder, loc, *plan.group, tp.axisIdx);
    auto forOp = emitFor(c0, ub, tp.ssa, /*isParallelOuter=*/false);
    result.bcastForOps.push_back(forOp);
    result.loopIVs[tp.axisIdx] = forOp.getInductionVar();
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

  for (const TileParam *tp : innerParams) {
    Value ub;
    if (outerIVs.count(tp->axisIdx)) {
      for (auto &group : plan.tileable)
        for (const auto &op2 : group)
          if (op2.level == TileLevel::Outer && op2.axisIdx == tp->axisIdx)
            ub = op2.ssa;
    }
    if (!ub)
      ub = getAxisExtentValue(builder, loc, *plan.group, tp->axisIdx);
    auto forOp = emitFor(c0, ub, tp->ssa, /*isParallelOuter=*/false);

    if (outerIVs.count(tp->axisIdx)) {
      Value composed = builder.create<arith::AddIOp>(
          loc, outerIVs[tp->axisIdx], forOp.getInductionVar());
      result.loopIVs[tp->axisIdx] = composed;
    } else {
      result.loopIVs[tp->axisIdx] = forOp.getInductionVar();
    }
  }

  result.innermostBody = builder.getInsertionBlock();
  result.iterArgs = iterArgs;
  return result;
}

} // namespace mlir::afir
