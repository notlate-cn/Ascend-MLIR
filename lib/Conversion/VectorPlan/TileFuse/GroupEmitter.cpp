#include "GroupEmitter.h"
#include "SliceComputer.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::vector_plan;
using mlir::linalg::LinalgOp;

namespace mlir::afir {

// Returns true if `val` is defined at or before `targetBlock` (i.e., it
// dominates any insertion in `targetBlock`). A value dominates `targetBlock`
// if its defining block is an ancestor of `targetBlock` (strictly outside it)
// or if it is a block argument of that ancestor.
static bool dominatesBlock(Value val, Block *targetBlock) {
  Block *defBlock = val.getParentBlock();
  // Walk up the region tree from targetBlock's parent; if we reach defBlock,
  // then val is defined outside targetBlock.
  Block *cur = targetBlock;
  while (cur) {
    if (cur == defBlock)
      return true;
    // Move to the parent block of the region that contains `cur`.
    Region *parentRegion = cur->getParent();
    if (!parentRegion)
      break;
    Operation *parentOp = parentRegion->getParentOp();
    if (!parentOp)
      break;
    cur = parentOp->getBlock();
  }
  return false;
}


SmallVector<Value> emitGroup(OpBuilder &builder, Location loc,
                              const CollapsedGroupInfo &info,
                              const TilePlan &plan,
                              const LoopNestResult &loopNest) {
  // --- Collect boundary outs and map to iter args ---
  SmallVector<Value> allOuts;
  DenseSet<Value> seenOuts;
  for (LinalgOp op : info.topoMembers)
    for (Value out : op.getDpsInits())
      if (seenOuts.insert(out).second)
        allOuts.push_back(out);

  DenseMap<Value, Value> outToIterArg;
  assert(allOuts.size() == loopNest.iterArgs.size() &&
         "allOuts / iterArgs count mismatch");
  for (auto [out, iterArg] : llvm::zip(allOuts, loopNest.iterArgs))
    outToIterArg[out] = iterArg;

  DenseMap<Value, Value> tiledValues;

  builder.setInsertionPointToEnd(loopNest.innermostBody);

  // --- Emit tiled ops ---
  for (LinalgOp op : info.topoMembers) {
    auto maps     = op.getIndexingMapsArray();
    int numInputs = op.getNumDpsInputs();
    SmallVector<Value> newOperands;

    // Inputs.
    for (int idx = 0; idx < numInputs; ++idx) {
      Value operand = op->getOperand(idx);
      if (tiledValues.count(operand)) {
        newOperands.push_back(tiledValues[operand]);
        continue;
      }
      AffineMap map = maps[idx];

      // Try to hoist: first determine the hoist point using outer IVs.
      // We need to call computeOuterSlice at the hoist location so that
      // constants (c0, c1, DimOps) are created there and dominate later uses.
      //
      // Step 1: probe whether hoist is possible by checking that outer IVs
      // dominate the BCast loop's containing block.
      // Identify which bcastForOp is the hoistable target.
      scf::ForOp hoistBeforeOp;
      if (!loopNest.bcastForOps.empty()) {
        // Check from innermost to outermost BCast loop.
        for (int i = (int)loopNest.bcastForOps.size() - 1; i >= 0; --i) {
          scf::ForOp bcastFor = loopNest.bcastForOps[i];
          // Check that bcastFor IV is NOT in the map's results (i.e., the
          // operand does not use the BCast axis).
          bool bcastUsed = false;
          for (int r = 0; r < (int)map.getNumResults(); ++r) {
            auto d = dyn_cast<AffineDimExpr>(map.getResult(r));
            if (d && loopNest.loopIVs.lookup(d.getPosition()) ==
                         bcastFor.getInductionVar()) {
              bcastUsed = true;
              break;
            }
          }
          if (bcastUsed) break;
          // Check that all outer IVs used in the map dominate bcastFor's block.
          Block *candidateBlock = bcastFor->getBlock();
          bool outersDominate = true;
          for (int r = 0; r < (int)map.getNumResults(); ++r) {
            auto d = dyn_cast<AffineDimExpr>(map.getResult(r));
            if (!d) continue;
            Value outerIV = loopNest.outerLoopIVs.lookup(d.getPosition());
            if (outerIV && !dominatesBlock(outerIV, candidateBlock)) {
              outersDominate = false;
              break;
            }
          }
          if (!outersDominate) break;
          hoistBeforeOp = bcastFor;
        }
      }

      if (hoistBeforeOp) {
        // Emit the outer (coarse) extract_slice before the BCast loop.
        // The builder is set to the hoist point so constants are created there.
        Value outerSliced;
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(hoistBeforeOp->getBlock(),
                                    Block::iterator(hoistBeforeOp));
          auto outerSp = computeOuterSlice(map, loopNest.outerLoopIVs, plan,
                                            operand, builder, loc);
          auto outerSlicedType = tensor::ExtractSliceOp::inferResultType(
              cast<RankedTensorType>(operand.getType()),
              outerSp.offsets, outerSp.sizes, outerSp.strides);
          outerSliced = builder.create<tensor::ExtractSliceOp>(
              loc, outerSlicedType, operand,
              outerSp.offsets, outerSp.sizes, outerSp.strides);
        }
        // Emit the inner (fine) extract_slice in the innermost body,
        // using the hoisted coarse slice as the source.
        // Inner offset = composedIV - outerIV (relative within outer tile).
        DenseMap<int, Value> innerOnlyIVs;
        for (auto &[axisIdx, composedIV] : loopNest.loopIVs) {
          Value outerIV = loopNest.outerLoopIVs.lookup(axisIdx);
          if (outerIV && composedIV != outerIV)
            innerOnlyIVs[axisIdx] =
                builder.create<arith::SubIOp>(loc, composedIV, outerIV);
          else
            innerOnlyIVs[axisIdx] = composedIV;
        }
        auto innerSp = computeSlice(map, innerOnlyIVs, plan, outerSliced,
                                     builder, loc);
        auto innerSlicedType = tensor::ExtractSliceOp::inferResultType(
            cast<RankedTensorType>(outerSliced.getType()),
            innerSp.offsets, innerSp.sizes, innerSp.strides);
        Value sliced = builder.create<tensor::ExtractSliceOp>(
            loc, innerSlicedType, outerSliced,
            innerSp.offsets, innerSp.sizes, innerSp.strides);
        newOperands.push_back(sliced);
      } else {
        // No hoist possible; emit extract_slice in the innermost body as usual.
        auto sp = computeSlice(map, loopNest.loopIVs, plan, operand, builder, loc);
        auto slicedType = tensor::ExtractSliceOp::inferResultType(
            cast<RankedTensorType>(operand.getType()),
            sp.offsets, sp.sizes, sp.strides);
        Value sliced = builder.create<tensor::ExtractSliceOp>(
            loc, slicedType, operand, sp.offsets, sp.sizes, sp.strides);
        newOperands.push_back(sliced);
      }
    }

    // DPS inits (outs).
    for (int idx = 0; idx < op.getNumDpsInits(); ++idx) {
      Value outOperand = op.getDpsInits()[idx];
      Value iterArg = outToIterArg.lookup(outOperand);
      if (!iterArg) iterArg = outOperand;
      AffineMap outMap = maps[numInputs + idx];
      auto sp = computeSlice(outMap, loopNest.loopIVs, plan, iterArg,
                              builder, loc);
      auto slicedType = tensor::ExtractSliceOp::inferResultType(
          cast<RankedTensorType>(iterArg.getType()),
          sp.offsets, sp.sizes, sp.strides);
      Value sliced = builder.create<tensor::ExtractSliceOp>(
          loc, slicedType, iterArg, sp.offsets, sp.sizes, sp.strides);
      newOperands.push_back(sliced);
    }

    // Clone op and update operand and result types.
    Operation *cloned = builder.clone(*op.getOperation());
    for (auto [i, val] : llvm::enumerate(newOperands))
      cloned->setOperand((unsigned)i, val);
    // Update result types to match new init operand types (DPS semantics).
    int numIn = op.getNumDpsInputs();
    for (int idx = 0; idx < op.getNumDpsInits(); ++idx)
      cloned->getResult(idx).setType(newOperands[numIn + idx].getType());
    for (auto [origRes, newRes] :
         llvm::zip(op->getResults(), cloned->getResults()))
      tiledValues[origRes] = newRes;
  }

  // --- Emit insert_slice for each boundary out ---
  SmallVector<Value> yieldVals;
  for (auto [origOut, iterArg] : llvm::zip(allOuts, loopNest.iterArgs)) {
    Value tiledResult;
    for (LinalgOp op : info.topoMembers) {
      auto inits   = op.getDpsInits();
      auto results = op->getResults();
      for (auto [init, result] : llvm::zip(inits, results)) {
        if (init == origOut && tiledValues.count(result)) {
          tiledResult = tiledValues[result];
          break;
        }
      }
      if (tiledResult) break;
    }
    if (!tiledResult) { yieldVals.push_back(iterArg); continue; }

    AffineMap outMap;
    for (LinalgOp op : info.topoMembers) {
      auto maps    = op.getIndexingMapsArray();
      int numIn    = op.getNumDpsInputs();
      auto inits   = op.getDpsInits();
      for (auto [i, init] : llvm::enumerate(inits)) {
        if (init == origOut) { outMap = maps[numIn + (int)i]; break; }
      }
      if (outMap) break;
    }
    auto sp = computeSlice(outMap, loopNest.loopIVs, plan, iterArg,
                            builder, loc);
    Value inserted = builder.create<tensor::InsertSliceOp>(
        loc, tiledResult, iterArg, sp.offsets, sp.sizes, sp.strides);
    yieldVals.push_back(inserted);
  }

  builder.create<scf::YieldOp>(loc, yieldVals);

  for (int i = (int)loopNest.allForOps.size() - 2; i >= 0; --i) {
    scf::ForOp inner = loopNest.allForOps[i + 1];
    scf::ForOp outer = loopNest.allForOps[i];
    builder.setInsertionPointToEnd(outer.getBody());
    builder.create<scf::YieldOp>(loc, inner.getResults());
  }

  if (loopNest.allForOps.empty()) return yieldVals;
  scf::ForOp outermost = loopNest.allForOps.front();
  return SmallVector<Value>(outermost.getResults());
}

} // namespace mlir::afir
