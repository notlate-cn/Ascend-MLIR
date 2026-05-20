#include "GroupEmitter.h"
#include "SliceComputer.h"
#include "TileFuseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::auto_fuse;
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


static const TileParam *findReductionInner(const TilePlan &plan) {
  for (auto &group : plan.tileable)
    for (const auto &tp : group)
      if (tp.level == TileLevel::Inner && tp.role == AxisRole::Reduction)
        return &tp;
  return nullptr;
}

static SmallVector<Value>
emitGroupWithReductionSplit(OpBuilder &builder, Location loc,
                             const CollapsedGroupInfo &info,
                             const TilePlan &plan,
                             LoopNestResult loopNest,
                             const TileParam *rblockParam) {
  assert(info.topoMembers.size() == 1 && "reduction split: single-op only");
  LinalgOp op = info.topoMembers[0];

  // The LoopNestBuilder placed RBLOCK as the innermost Inner loop. Detach it:
  // its parent body becomes the new innermost (parallel) body.
  assert(!loopNest.allForOps.empty());
  scf::ForOp rblockFor = loopNest.allForOps.back();
  // Defensive: the innermost for must be the reduction axis.
  Value rblockIV = rblockFor.getInductionVar();
  // Pop it.
  loopNest.allForOps.pop_back();
  // Parent body becomes the new innermost.
  Block *parallelBody = nullptr;
  SmallVector<Value> parallelIterArgs;
  if (!loopNest.allForOps.empty()) {
    scf::ForOp parent = loopNest.allForOps.back();
    parallelBody = parent.getBody();
    parallelIterArgs =
        SmallVector<Value>(parent.getRegionIterArgs());
  } else {
    parallelBody = rblockFor->getBlock();
    parallelIterArgs = SmallVector<Value>(loopNest.iterArgs);
  }
  // The rblockFor's iter_args used the parent's iter_args directly (since it
  // was the innermost and the chain just forwarded). Take the iter args we
  // need to write into.
  // Move insertion point out of the doomed RBLOCK body before erasing.
  builder.setInsertionPointToEnd(parallelBody);
  // Erase the RBLOCK loop (its body is empty: LoopNestBuilder removed yield).
  rblockFor->erase();

  loopNest.innermostBody = parallelBody;
  loopNest.iterArgs = parallelIterArgs;

  // parallelIVs = loopIVs minus the reduction axis IV (which was
  // composed using rblockIV). We strip it and rebuild within RBLOCK below.
  DenseMap<int, Value> parallelIVs;
  for (auto &kv : loopNest.loopIVs)
    if (kv.first != rblockParam->axisIdx)
      parallelIVs[kv.first] = kv.second;

  // Collect outs and map to iter args.
  SmallVector<Value> allOuts;
  DenseSet<Value> seenOuts;
  for (Value out : op.getDpsInits())
    if (seenOuts.insert(out).second)
      allOuts.push_back(out);

  builder.setInsertionPointToEnd(parallelBody);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);

  auto maps     = op.getIndexingMapsArray();
  int numInputs = op.getNumDpsInputs();

  SmallVector<Value> yieldVals;
  for (auto [origOut, iterArg] : llvm::zip(allOuts, parallelIterArgs)) {
    // Find outMap for this init.
    AffineMap outMap;
    auto inits = op.getDpsInits();
    for (auto [i, init] : llvm::enumerate(inits))
      if (init == origOut) { outMap = maps[numInputs + (int)i]; break; }

    // Slice params for output (only parallel IVs).
    auto outSp = computeSlice(outMap, parallelIVs, plan, iterArg,
                               builder, loc);

    auto iterArgType = cast<RankedTensorType>(iterArg.getType());
    Type elemTy = iterArgType.getElementType();

    // Init accumulator: bufferization.alloc_tensor (memory_space=11/VECCALC)
    // + linalg.fill(0). The explicit memory space makes it bufferize to a
    // memref.alloc in VECCALC, so the downstream tile-buffer-insertion pass
    // skips the inner reduce generic (its output isn't a GM target) and the
    // trailing acc->GM copy survives self-copy cleanup.
    SmallVector<Value> dynSizes;
    SmallVector<int64_t> staticShape;
    for (OpFoldResult ofr : outSp.sizes) {
      if (auto v = dyn_cast<Value>(ofr)) {
        dynSizes.push_back(v);
        staticShape.push_back(ShapedType::kDynamic);
      } else {
        auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
        staticShape.push_back(attr.getInt());
      }
    }
    Value accEmpty = builder.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get(staticShape, elemTy), dynSizes,
        /*copy=*/Value{},
        /*memory_space=*/builder.getI64IntegerAttr(11));
    Value zeroAttr = builder.create<arith::ConstantOp>(
        loc, builder.getZeroAttr(elemTy));
    Value accZero = builder.create<linalg::FillOp>(
                        loc, ValueRange{zeroAttr}, ValueRange{accEmpty})
                        .getResult(0);

    // RBLOCK scf.for.
    // Common: span the full R extent (each parallel block does the whole
    // reduction over R).
    // RCore (P3b-2c/d): R itself is the block axis, so each core only sees
    // a slice of R of size parentStep == XBLOCK; span 0..min(XBLOCK,
    // R - outerR_IV) so the per-core inner reduce stays within its slice.
    bool rcore = (plan.reduceTemplate == TilePlan::ReduceTemplate::RCore);
    Value reductionExtent;
    if (rcore) {
      Value rExt =
          getAxisExtentValue(builder, loc, info, rblockParam->axisIdx);
      // Find the Outer-R TileParam's ssa (XBLOCK) and the outer IV.
      Value outerStep, outerIV;
      for (auto &grp : plan.tileable)
        for (const TileParam &tp : grp)
          if (tp.axisIdx == rblockParam->axisIdx &&
              tp.level == TileLevel::Outer) {
            outerStep = tp.ssa;
            break;
          }
      auto outerIVIt = loopNest.outerLoopIVs.find(rblockParam->axisIdx);
      assert(outerStep && outerIVIt != loopNest.outerLoopIVs.end() &&
             "RCore: missing Outer-R TileParam or outerLoopIV");
      outerIV = outerIVIt->second;
      Value remaining =
          builder.create<arith::SubIOp>(loc, rExt, outerIV);
      reductionExtent =
          builder.create<arith::MinSIOp>(loc, outerStep, remaining);
    } else {
      reductionExtent =
          getAxisExtentValue(builder, loc, info, rblockParam->axisIdx);
    }
    auto rFor = builder.create<scf::ForOp>(
        loc, c0, reductionExtent, rblockParam->ssa,
        SmallVector<Value>{accZero});
    // ForOp with iter_args does NOT auto-insert a yield (MLIR leaves it to the
    // caller). Just point the builder at the end of the empty body.
    builder.setInsertionPointToEnd(rFor.getBody());

    // IVs inside RBLOCK.  For RCore the per-core inner IV is local to the
    // R slice [outerR_IV, outerR_IV+parentStep); compose with the outer IV
    // so input slicing addresses the absolute R position.
    DenseMap<int, Value> allIVs(parallelIVs);
    Value innerR = rFor.getInductionVar();
    if (rcore) {
      Value outerIV = loopNest.outerLoopIVs[rblockParam->axisIdx];
      innerR = builder.create<arith::AddIOp>(loc, outerIV, innerR);
    }
    allIVs[rblockParam->axisIdx] = innerR;
    (void)rblockIV;

    SmallVector<Value> newOperands;
    for (int idx = 0; idx < numInputs; ++idx) {
      Value operand = op->getOperand(idx);
      AffineMap m = maps[idx];
      auto sp = computeSlice(m, allIVs, plan, operand, builder, loc);
      auto slicedType = tensor::ExtractSliceOp::inferResultType(
          cast<RankedTensorType>(operand.getType()),
          sp.offsets, sp.sizes, sp.strides);
      Value sliced = builder.create<tensor::ExtractSliceOp>(
          loc, slicedType, operand, sp.offsets, sp.sizes, sp.strides);
      newOperands.push_back(sliced);
    }
    // DPS init = RBLOCK iter arg accumulator.
    Value accIter = rFor.getRegionIterArgs().front();
    newOperands.push_back(accIter);

    Operation *cloned = builder.clone(*op.getOperation());
    for (auto [i, val] : llvm::enumerate(newOperands))
      cloned->setOperand((unsigned)i, val);
    cloned->getResult(0).setType(accIter.getType());

    builder.create<scf::YieldOp>(loc, cloned->getResult(0));

    // After RBLOCK: write the partial back to the outer (parallel/block) iter
    // arg's space.  For Common, tensor.insert_slice into a non-empty slice of
    // a parallel-axis-indexed iterArg bufferizes to subview + memref.copy.
    // For RCore (full-reduce, rank-0 → rank-0), that insert_slice has empty
    // offsets/sizes/strides and canonicalize folds it to identity — the cross-
    // space copy disappears and bufferize complains about inconsistent memory
    // spaces between the outer iter_arg (GM) and yield (VECCALC).  Use
    // bufferization.materialize_in_destination, which always emits memref.copy.
    builder.setInsertionPointAfter(rFor);
    Value writtenBack;
    if (rcore) {
      writtenBack = builder.create<bufferization::MaterializeInDestinationOp>(
                            loc, /*resultType=*/iterArg.getType(),
                            /*source=*/rFor.getResult(0), /*dest=*/iterArg)
                        .getResult();
    } else {
      writtenBack = builder.create<tensor::InsertSliceOp>(
          loc, rFor.getResult(0), iterArg,
          outSp.offsets, outSp.sizes, outSp.strides);
    }
    yieldVals.push_back(writtenBack);
  }

  builder.create<scf::YieldOp>(loc, yieldVals);

  // Propagate yields up.
  for (int i = (int)loopNest.allForOps.size() - 2; i >= 0; --i) {
    scf::ForOp inner = loopNest.allForOps[i + 1];
    scf::ForOp outer = loopNest.allForOps[i];
    builder.setInsertionPointToEnd(outer.getBody());
    builder.create<scf::YieldOp>(loc, inner.getResults());
  }

  if (loopNest.allForOps.empty()) return yieldVals;
  return SmallVector<Value>(loopNest.allForOps.front().getResults());
}

// Emit one round of the tile body (extract_slice for each input/init, clone
// the linalg op, insert_slice back).  Builder must be positioned where the
// new ops should be inserted (inside an scf.for body or scf.if then-block).
// Returns the values to yield (one per boundary output).
//
//   `iterArgs`     — input tensors for this iteration (loop iter_args, or
//                    inner-for results when emitting the tail body).
//   `bcastForOps`  — used by the hoist path; pass `{}` in tail mode to disable
//                    hoisting (the tail body lives inside any BCast for and
//                    must not re-hoist).
//   `sizeOverride` — per-axis size override map (tail emit passes
//                    `{innerTileAxisIdx: tailSize}`).
static SmallVector<Value>
emitGroupBodyOnce(OpBuilder &builder, Location loc,
                  const CollapsedGroupInfo &info, const TilePlan &plan,
                  const DenseMap<int, Value> &loopIVs,
                  const DenseMap<int, Value> &outerLoopIVs,
                  ArrayRef<Value> iterArgs,
                  ArrayRef<scf::ForOp> bcastForOps,
                  const DenseMap<int, Value> *sizeOverride) {
  // --- Collect boundary outs and map to iter args ---
  // A member whose result is consumed only by other members (an intra-group
  // intermediate — e.g. a multi-use linalg.transpose) gets a fresh per-
  // iteration VECCALC tile below; it is not an scf.for iter_arg.
  SmallVector<Value> allOuts;
  DenseSet<Value> seenOuts;
  for (LinalgOp op : info.topoMembers) {
    if (resultUsedOnlyByGroupMembers(op, info))
      continue;
    for (Value out : op.getDpsInits())
      if (seenOuts.insert(out).second)
        allOuts.push_back(out);
  }

  DenseMap<Value, Value> outToIterArg;
  assert(allOuts.size() == iterArgs.size() &&
         "allOuts / iterArgs count mismatch");
  for (auto [out, iterArg] : llvm::zip(allOuts, iterArgs))
    outToIterArg[out] = iterArg;

  DenseMap<Value, Value> tiledValues;

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
      if (!bcastForOps.empty()) {
        // Check from innermost to outermost BCast loop.
        for (int i = (int)bcastForOps.size() - 1; i >= 0; --i) {
          scf::ForOp bcastFor = bcastForOps[i];
          // Check that bcastFor IV is NOT in the map's results (i.e., the
          // operand does not use the BCast axis).
          bool bcastUsed = false;
          for (int r = 0; r < (int)map.getNumResults(); ++r) {
            auto d = dyn_cast<AffineDimExpr>(map.getResult(r));
            if (d && loopIVs.lookup(d.getPosition()) ==
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
            Value outerIV = outerLoopIVs.lookup(d.getPosition());
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
          auto outerSp = computeOuterSlice(map, outerLoopIVs, plan,
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
        for (auto &[axisIdx, composedIV] : loopIVs) {
          Value outerIV = outerLoopIVs.lookup(axisIdx);
          if (outerIV && composedIV != outerIV)
            innerOnlyIVs[axisIdx] =
                builder.create<arith::SubIOp>(loc, composedIV, outerIV);
          else
            innerOnlyIVs[axisIdx] = composedIV;
        }
        auto innerSp = computeSlice(map, innerOnlyIVs, plan, outerSliced,
                                     builder, loc, sizeOverride);
        auto innerSlicedType = tensor::ExtractSliceOp::inferResultType(
            cast<RankedTensorType>(outerSliced.getType()),
            innerSp.offsets, innerSp.sizes, innerSp.strides);
        Value sliced = builder.create<tensor::ExtractSliceOp>(
            loc, innerSlicedType, outerSliced,
            innerSp.offsets, innerSp.sizes, innerSp.strides);
        newOperands.push_back(sliced);
      } else {
        // No hoist possible; emit extract_slice in the innermost body as usual.
        auto sp = computeSlice(map, loopIVs, plan, operand, builder, loc,
                                sizeOverride);
        auto slicedType = tensor::ExtractSliceOp::inferResultType(
            cast<RankedTensorType>(operand.getType()),
            sp.offsets, sp.sizes, sp.strides);
        Value sliced = builder.create<tensor::ExtractSliceOp>(
            loc, slicedType, operand, sp.offsets, sp.sizes, sp.strides);
        newOperands.push_back(sliced);
      }
    }

    // DPS inits (outs).
    bool isIntraGroup = resultUsedOnlyByGroupMembers(op, info);
    for (int idx = 0; idx < op.getNumDpsInits(); ++idx) {
      Value outOperand = op.getDpsInits()[idx];
      AffineMap outMap = maps[numInputs + idx];
      if (isIntraGroup) {
        // Intra-group intermediate: a fresh per-iteration VECCALC tile (no
        // iter_arg, no extract/insert into a full-shape tensor that would
        // bufferize to a GM buffer); consumers read the cloned op's result.
        auto sp = computeSlice(outMap, loopIVs, plan, outOperand,
                                builder, loc, sizeOverride);
        Type elemTy =
            cast<RankedTensorType>(outOperand.getType()).getElementType();
        SmallVector<Value> dynSizes;
        SmallVector<int64_t> staticShape;
        for (OpFoldResult ofr : sp.sizes) {
          if (auto v = dyn_cast<Value>(ofr)) {
            dynSizes.push_back(v);
            staticShape.push_back(ShapedType::kDynamic);
          } else {
            staticShape.push_back(cast<IntegerAttr>(cast<Attribute>(ofr)).getInt());
          }
        }
        newOperands.push_back(builder.create<bufferization::AllocTensorOp>(
            loc, RankedTensorType::get(staticShape, elemTy), dynSizes,
            /*copy=*/Value{}, /*memory_space=*/builder.getI64IntegerAttr(11)));
        continue;
      }
      Value iterArg = outToIterArg.lookup(outOperand);
      if (!iterArg) iterArg = outOperand;
      auto sp = computeSlice(outMap, loopIVs, plan, iterArg,
                              builder, loc, sizeOverride);
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
  for (auto [origOut, iterArg] : llvm::zip(allOuts, iterArgs)) {
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
    auto sp = computeSlice(outMap, loopIVs, plan, iterArg,
                            builder, loc, sizeOverride);
    Value inserted = builder.create<tensor::InsertSliceOp>(
        loc, tiledResult, iterArg, sp.offsets, sp.sizes, sp.strides);
    yieldVals.push_back(inserted);
  }

  return yieldVals;
}

// Emit the displaced-multi-R case (≈ AF `IsNeedMultiReduce`,
// reduce_api_call.cpp:96): for `out[a] = sum_{r1,r2} x[r1,a,r2]` with
// iter [reduction, parallel, reduction], peel the outermost displaced R as a
// step=1 outer scf.for around the parallel inner body's accumulator, and
// rank-reduce that axis out of the inner linalg.generic so what reaches
// ComputeConversion is the well-tested 2-D [parallel, reduction] shape.
//
// Single-op groups only (the only displaced-R shape we have seen in practice
// is a bare reduce — fused-with-elementwise extensions are a separate effort).
static SmallVector<Value>
emitGroupWithPeeledReduce(OpBuilder &builder, Location loc,
                           const CollapsedGroupInfo &info,
                           const TilePlan &plan,
                           LoopNestResult loopNest) {
  assert(plan.peelOuterR >= 0 && "peel-outer-R requires plan.peelOuterR");
  assert(info.topoMembers.size() == 1 &&
         "peel-outer-R: single-op groups only");
  LinalgOp op = info.topoMembers[0];

  int peelAxis = plan.peelOuterR;
  MLIRContext *ctx = builder.getContext();

  // Affine-map helper: drop the peel iter dim from `m`'s results and renumber
  // remaining dim positions to fit the new (peeled) iter space.
  auto dropDim = [&](AffineMap m, unsigned drop) -> AffineMap {
    SmallVector<AffineExpr> nr;
    for (AffineExpr r : m.getResults()) {
      auto de = dyn_cast<AffineDimExpr>(r);
      assert(de && "non-AffineDimExpr in indexing map");
      unsigned p = de.getPosition();
      if (p == drop) continue;
      nr.push_back(getAffineDimExpr(p > drop ? p - 1 : p, ctx));
    }
    return AffineMap::get(m.getNumDims() - 1, /*symbols=*/0, nr, ctx);
  };

  auto origIterTypes = op.getIteratorTypesArray();
  auto origMaps      = op.getIndexingMapsArray();
  SmallVector<Attribute> newIterTypeAttrs;
  for (int i = 0; i < (int)origIterTypes.size(); ++i)
    if (i != peelAxis)
      newIterTypeAttrs.push_back(
          linalg::IteratorTypeAttr::get(ctx, origIterTypes[i]));

  builder.setInsertionPointToEnd(loopNest.innermostBody);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> allOuts;
  DenseSet<Value> seenOuts;
  for (Value out : op.getDpsInits())
    if (seenOuts.insert(out).second) allOuts.push_back(out);
  assert(allOuts.size() == loopNest.iterArgs.size() &&
         "peel-outer-R: outs / iterArgs mismatch");

  Value peelExt = getAxisExtentValue(builder, loc, info, peelAxis);

  int numInputs = op.getNumDpsInputs();

  SmallVector<Value> yieldVals;
  for (auto [origOut, iterArg] : llvm::zip(allOuts, loopNest.iterArgs)) {
    AffineMap outMap;
    {
      auto inits = op.getDpsInits();
      for (auto [i, in] : llvm::enumerate(inits))
        if (in == origOut) { outMap = origMaps[numInputs + (int)i]; break; }
    }

    // Output tile in the parallel body (the peel axis is projected away from
    // outMap for a reduce, so computeSlice trivially yields the parallel
    // slice).
    auto outSp =
        computeSlice(outMap, loopNest.loopIVs, plan, iterArg, builder, loc);

    auto iterArgTy = cast<RankedTensorType>(iterArg.getType());
    Type elemTy    = iterArgTy.getElementType();

    SmallVector<Value> dynSizes;
    SmallVector<int64_t> staticShape;
    for (OpFoldResult ofr : outSp.sizes) {
      if (auto v = dyn_cast<Value>(ofr)) {
        dynSizes.push_back(v);
        staticShape.push_back(ShapedType::kDynamic);
      } else {
        staticShape.push_back(
            cast<IntegerAttr>(cast<Attribute>(ofr)).getInt());
      }
    }
    Value accEmpty = builder.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get(staticShape, elemTy), dynSizes,
        /*copy=*/Value{},
        /*memory_space=*/builder.getI64IntegerAttr(11));
    Value zeroAttr = builder.create<arith::ConstantOp>(
        loc, builder.getZeroAttr(elemTy));
    Value accZero = builder.create<linalg::FillOp>(
                        loc, ValueRange{zeroAttr}, ValueRange{accEmpty})
                        .getResult(0);

    // Outer scf.for over the displaced R, step=1, accumulator iter_arg.  Each
    // iteration processes a r1-size-1 chunk through a peeled 2-D inner generic.
    auto rFor = builder.create<scf::ForOp>(loc, c0, peelExt, c1,
                                            SmallVector<Value>{accZero});
    builder.setInsertionPointToEnd(rFor.getBody());
    Value accIter = rFor.getRegionIterArgs().front();
    Value peelIV  = rFor.getInductionVar();

    DenseMap<int, Value> allIVs(loopNest.loopIVs);
    allIVs[peelAxis] = peelIV;
    DenseMap<int, Value> sizeOverride;
    sizeOverride[peelAxis] = c1;

    // Build rank-reduced inputs and the dropped indexing maps.
    SmallVector<Value>     newInputs;
    SmallVector<AffineMap> newMaps;
    for (int idx = 0; idx < numInputs; ++idx) {
      Value operand = op->getOperand(idx);
      AffineMap m   = origMaps[idx];
      auto sp = computeSlice(m, allIVs, plan, operand, builder, loc,
                              &sizeOverride);
      // The operand-layout dim position that corresponds to the peel iter dim
      // gets dropped from the result type (its slice size is 1).  Some operands
      // (e.g. layernorm-style y[a] with map (r1,a,r2)->(a)) do not have the
      // peel axis in their map at all — dropDimPos stays -1 and the result
      // shape is unchanged.
      auto srcTy = cast<RankedTensorType>(operand.getType());
      int dropDimPos = -1;
      for (int dp = 0; dp < (int)m.getNumResults(); ++dp) {
        auto de = dyn_cast<AffineDimExpr>(m.getResult(dp));
        if (de && (int)de.getPosition() == peelAxis) {
          dropDimPos = dp;
          break;
        }
      }
      // Force the peel-axis size to a *static* `1` so the extract_slice
      // verifier accepts rank-reduction (its `static_sizes` array must have a
      // literal 1 in the dropped position; an SSA `arith.constant 1` is still
      // dynamic from the verifier's perspective).
      if (dropDimPos >= 0)
        sp.sizes[dropDimPos] = OpFoldResult(builder.getI64IntegerAttr(1));
      auto inferredTy = cast<RankedTensorType>(
          tensor::ExtractSliceOp::inferResultType(
              srcTy, sp.offsets, sp.sizes, sp.strides));
      RankedTensorType rrTy = inferredTy;
      if (dropDimPos >= 0) {
        SmallVector<int64_t> rrShape;
        for (int dp = 0; dp < (int)inferredTy.getRank(); ++dp)
          if (dp != dropDimPos) rrShape.push_back(inferredTy.getShape()[dp]);
        rrTy = RankedTensorType::get(rrShape, srcTy.getElementType());
      }
      Value sliced = builder.create<tensor::ExtractSliceOp>(
          loc, rrTy, operand, sp.offsets, sp.sizes, sp.strides);
      newInputs.push_back(sliced);
      newMaps.push_back(dropDim(m, peelAxis));
    }
    newMaps.push_back(dropDim(outMap, peelAxis));

    // Clone the original linalg.generic and rewrite operands + structural
    // attrs to the peeled shape.  The body block (scalar arith ops) is
    // unchanged — same f32 block args, same yield.  This avoids reconstructing
    // the body manually and works for any reduce-style body.
    SmallVector<Value> newOperands(newInputs);
    newOperands.push_back(accIter);
    IRMapping noMap;
    Operation *cloned = builder.clone(*op.getOperation(), noMap);
    for (auto [i, val] : llvm::enumerate(newOperands))
      cloned->setOperand((unsigned)i, val);
    cloned->getResult(0).setType(accIter.getType());
    cloned->setAttr("iterator_types",
                    ArrayAttr::get(ctx, newIterTypeAttrs));
    cloned->setAttr("indexing_maps", builder.getAffineMapArrayAttr(newMaps));

    builder.create<scf::YieldOp>(loc, cloned->getResult(0));

    builder.setInsertionPointAfter(rFor);
    Value writtenBack = builder.create<tensor::InsertSliceOp>(
        loc, rFor.getResult(0), iterArg,
        outSp.offsets, outSp.sizes, outSp.strides);
    yieldVals.push_back(writtenBack);
  }

  builder.create<scf::YieldOp>(loc, yieldVals);

  // Propagate yields up the outer loop nest (mirrors emitGroupWithReductionSplit).
  for (int i = (int)loopNest.allForOps.size() - 2; i >= 0; --i) {
    scf::ForOp inner = loopNest.allForOps[i + 1];
    scf::ForOp outer = loopNest.allForOps[i];
    builder.setInsertionPointToEnd(outer.getBody());
    builder.create<scf::YieldOp>(loc, inner.getResults());
  }
  if (loopNest.allForOps.empty()) return yieldVals;
  return SmallVector<Value>(loopNest.allForOps.front().getResults());
}

SmallVector<Value> emitGroup(OpBuilder &builder, Location loc,
                              const CollapsedGroupInfo &info,
                              const TilePlan &plan,
                              const LoopNestResult &loopNest) {
  if (plan.peelOuterR >= 0)
    return emitGroupWithPeeledReduce(builder, loc, info, plan, loopNest);
  if (const TileParam *rblockParam = findReductionInner(plan))
    return emitGroupWithReductionSplit(builder, loc, info, plan, loopNest,
                                        rblockParam);

  // --- Main body in innermost for ---
  builder.setInsertionPointToEnd(loopNest.innermostBody);
  SmallVector<Value> mainYieldVals =
      emitGroupBodyOnce(builder, loc, info, plan,
                         loopNest.loopIVs, loopNest.outerLoopIVs,
                         loopNest.iterArgs, loopNest.bcastForOps,
                         /*sizeOverride=*/nullptr);
  builder.create<scf::YieldOp>(loc, mainYieldVals);

  // --- Tail peel: scf.if after innermost for ---
  scf::ForOp innermostFor;
  scf::IfOp tailIf;
  if (loopNest.hasTail && !loopNest.allForOps.empty()) {
    innermostFor = loopNest.allForOps.back();
    builder.setInsertionPointAfter(innermostFor);

    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt,
        loopNest.mainInnerUb, loopNest.remaining);

    SmallVector<Type> resultTypes(innermostFor.getResultTypes().begin(),
                                   innermostFor.getResultTypes().end());
    tailIf = builder.create<scf::IfOp>(loc, resultTypes, cond,
                                        /*withElseRegion=*/true);

    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(tailIf.thenBlock());

      // Overlap-tail: process a STATIC slice of size T at offset
      // `innerTileExtent - innerTileStep`, instead of a dynamic-size slice
      // clamped to `remaining - mainInnerUb`.  Benefits:
      //   - slice size is a compile-time T, matches the main loop's tile,
      //     so bufferize does NOT wrap the linalg in a shadow-alloc
      //     sandwich and DataCopy counts trivially meet dtype alignment.
      //   - the only cost is that this tail re-computes the last
      //     `T - actual_tail_size` rows that the prior main iter (or the
      //     previous block's main loop) already produced; since both
      //     elementwise and parallel-axis reduce are deterministic per
      //     output row, the duplicate writes converge to the same value.
      //
      // Precondition: `innerTileExtent >= innerTileStep`.  The scf.if
      // condition (mainInnerUb < remaining) only fires on the tail core,
      // and in current workloads the tail core's enclosing axis always
      // has `extent >= XBLOCK_SUB` (the tiling space enforces this).  If
      // a future workload violates the precondition, the offset would
      // underflow — TODO: scalar fallback or DataCopyPad for that edge.
      Value tailComposed = builder.create<arith::SubIOp>(
          loc, loopNest.innerTileExtent, loopNest.innerTileStep);

      DenseMap<int, Value> tailLoopIVs = loopNest.loopIVs;
      tailLoopIVs[loopNest.innerTileAxisIdx] = tailComposed;
      // No sizeOverride — tail uses the planned static tile size.

      SmallVector<Value> tailIterArgs(innermostFor.getResults().begin(),
                                       innermostFor.getResults().end());

      // Tail body is inside any BCast for(s); disable hoisting (already
      // applied to the main body).
      SmallVector<Value> tailYieldVals =
          emitGroupBodyOnce(builder, loc, info, plan,
                             tailLoopIVs, loopNest.outerLoopIVs,
                             tailIterArgs, /*bcastForOps=*/{},
                             /*sizeOverride=*/nullptr);
      builder.create<scf::YieldOp>(loc, tailYieldVals);
    }

    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(tailIf.elseBlock());
      builder.create<scf::YieldOp>(
          loc, SmallVector<Value>(innermostFor.getResults().begin(),
                                   innermostFor.getResults().end()));
    }
  }

  // --- Propagate yields up the loop nest ---
  // The immediate parent of innermost for yields scf.if.results (when peeled),
  // not innermost.results.
  for (int i = (int)loopNest.allForOps.size() - 2; i >= 0; --i) {
    scf::ForOp inner = loopNest.allForOps[i + 1];
    scf::ForOp outer = loopNest.allForOps[i];
    builder.setInsertionPointToEnd(outer.getBody());
    ValueRange yieldVR =
        (tailIf && inner == innermostFor) ? tailIf.getResults()
                                          : inner.getResults();
    builder.create<scf::YieldOp>(loc, yieldVR);
  }

  if (loopNest.allForOps.empty()) return mainYieldVals;
  if (loopNest.allForOps.size() == 1 && tailIf)
    return SmallVector<Value>(tailIf.getResults());
  scf::ForOp outermost = loopNest.allForOps.front();
  return SmallVector<Value>(outermost.getResults());
}

} // namespace mlir::afir
