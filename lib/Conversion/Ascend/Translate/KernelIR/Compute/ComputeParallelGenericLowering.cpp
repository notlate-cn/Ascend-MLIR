/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under terms and conditions of the CANN Open Software License Agreement
 * Version 2.0 (the "License"). Please refer to LICENSE in the root of the
 * software repository for the full text of the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

#include "ComputeLoweringInternal.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Translate/KernelIR/Capabilities/ElementwiseBodyOpRegistry.h"
#include "Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include <algorithm>
#include <limits>
#include <optional>

#define DEBUG_TYPE "ascend-compute-lower-compute"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace ascend {

LogicalResult lowerParallelGenericComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  OpBuilder &builder = lowering.builder;
  // --- linalg.generic {all-parallel, on-chip output} ---
  //
  // Pure-parallel generic lowering (e.g. broadcast+add, broadcast+mul).
  // These have iterator_types = ["parallel", "parallel", ...] with no reduction.
  //
  // Strategy mirrors the reduction path (Steps 1-2) but skips Step 3:
  //   GM/VECIN inputs → promote to VECCALC local_tensors (broadcast_l2 or copy)
  //   Body arith ops → inline as AscendC vector ops on VECCALC accumulator
  //   Final result   → write directly to VECOUT (writeTensor handles alloc)
  //   Enqueue VECOUT for downstream data-move epilogue copy
  //
  // Concat semantics are implicitly handled: the VECOUT->GM copy op from
  // memory realization targets a memref subview of the output buffer with the
  // correct byte offset, so Op1 and Op2 results land at the right positions in
  // the concatenated output without any asc.concat op.
  SmallVector<linalg::GenericOp> parallelGenericOps;
  funcOp.walk([&](linalg::GenericOp op) {
    auto iterTypes = op.getIteratorTypesArray();
    bool allParallel = llvm::all_of(iterTypes, [](utils::IteratorType t) {
      return t == utils::IteratorType::parallel;
    });
    if (allParallel && op.getNumDpsInits() == 1)
      parallelGenericOps.push_back(op);
  });

  // Helper: detect a standalone generic transpose supported by the backend.
  auto isTransposeGeneric = [](linalg::GenericOp op) -> bool {
    FailureOr<TransposeLoweringSpec> spec = buildTransposeLoweringSpec(op);
    return succeeded(spec) &&
           planTransposeLowering(*spec).kind ==
               TransposeLoweringKind::AscendCSimple2D;
  };


  for (linalg::GenericOp genOp : parallelGenericOps) {
    Value outMemref = genOp.getDpsInitOperand(0)->get();
    int64_t outMs   = getMemorySpace(outMemref.getType());
    if (outMs == 0 && isPureYieldGeneric(genOp)) {
      builder.setInsertionPoint(genOp);
      if (failed(lowerPureYieldGenericToLoops(builder, genOp))) {
        genOp.emitError("failed to lower pure-yield generic copy");
        return failure();
      }
      genOp.erase();
      continue;
    }
    if (outMs == 0 && isGmAllParallelGeneric(genOp)) {
      builder.setInsertionPoint(genOp);
      if (failed(lowerAllParallelGenericToLoops(builder, genOp))) {
        genOp.emitError("failed to lower GM all-parallel generic");
        return failure();
      }
      genOp.erase();
      continue;
    }
    if (outMs <= 0)
      continue; // output must be on-chip (VECOUT or VECCALC)

    if (genOp->hasAttr(ascend::kGatherDimAttr)) {
      if (failed(lowerGatherCompute(lowering, genOp, parallelGenericOps)))
        return failure();
      continue;
    }

    // ---- Transpose generic: emit ascendc.transpose ----
    if (isTransposeGeneric(genOp)) {
      Value inMemref = genOp.getDpsInputOperand(0)->get();
      Location loc = genOp.getLoc();
      builder.setInsertionPoint(genOp);

      Value srcLt = lowering.readTensor(builder, loc, inMemref);
      Value dstLt = lowering.writeTensor(builder, loc, outMemref);
      builder.create<TransposeOp>(loc, dstLt, srcLt);

      if (Value q = lowering.ctx.getQueue(outMemref))
        builder.create<TQueBindEnqueTensorOp>(loc, q, dstLt);

      genOp.erase();
      continue;
    }

    unsigned numInputs = genOp.getNumDpsInputs();
    auto iterTypes     = genOp.getIteratorTypesArray();
    unsigned iterRank  = iterTypes.size();
    auto maps          = genOp.getIndexingMapsArray();

    Location loc = genOp.getLoc();
    builder.setInsertionPoint(genOp);
    Type elemType = cast<MemRefType>(outMemref.getType()).getElementType();

    // ---- Compute iteration dim sizes from the first full-rank input ----
    SmallVector<Value> iterDimSizes(iterRank);
    for (unsigned i = 0; i < numInputs; ++i) {
      Value inMemref = genOp.getDpsInputOperand(i)->get();
      AffineMap inMap = maps[i];
      bool allDimExprs = llvm::all_of(inMap.getResults(),
          [](AffineExpr e) { return isa<AffineDimExpr>(e); });
      if (inMap.getNumResults() == iterRank && allDimExprs) {
        for (unsigned d = 0; d < iterRank; ++d)
          iterDimSizes[d] = getDynDim(lowering, builder, loc, inMemref, d);
        break;
      }
    }
    // Fall back: fill remaining dims from output (all parallel, same rank).
    for (unsigned d = 0; d < iterRank; ++d)
      if (!iterDimSizes[d])
        iterDimSizes[d] = getDynDim(lowering, builder, loc, outMemref, d);

    // totalElems is the actual element count for this tile.  Buffer
    // allocation uses the enclosing loop-step upper bound so tail iterations
    // reuse one max-sized queue/tbuf instead of repeatedly InitBuffer-ing.
    Value totalElems = lowering.computeProduct(builder, loc, iterDimSizes);
    SmallVector<Value> bufferDimSizes =
        getBufferDimSizes(lowering, iterDimSizes, genOp.getOperation());
    Value bufferTotalElems = lowering.computeProduct(builder, loc, bufferDimSizes);

    Value outQueue = lowering.ctx.getQueue(outMemref);
    Value accumLt;
    if (!outQueue) {
      // Allocate the shared VECCALC accumulator for intermediate results.
      auto [accumTbuf, veccalcAccumLt] =
          allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
      accumLt = veccalcAccumLt;
    }

    // ---- Step 1: Promote each input to a VECCALC local_tensor ----
    SmallVector<OwnedQueueTensor> ownedInputTensors;
    SmallVector<Value> inputLts(numInputs);
    for (unsigned i = 0; i < numInputs; ++i) {
      Value inMemref = genOp.getDpsInputOperand(i)->get();
      AffineMap inMap = maps[i];
      int64_t inMs    = getMemorySpace(inMemref.getType());
      IndexingMapAnalysis analysis = analyzeIndexingMap(inMap, iterRank);

      switch (analysis.kind) {
      case IndexingMapAnalysis::Kind::Identity: {
        if (inMs == 0 /*GM*/) {
          // GM rank-2 subviews with partial inner tiles are not contiguous in GM.
          // Copy them row-by-row into the compact local tile used by vector ops.
          if (isRank2GmSubview(inMemref) &&
              !isContiguousRank2GmSubview(inMemref)) {
            inputLts[i] = copyRank2GmSubviewRowsToVecin(lowering,
                builder, loc, elemType, inMemref, bufferTotalElems,
                &ownedInputTensors);
            if (!inputLts[i]) {
              genOp.emitError("failed to lower rank-2 GM subview input copy");
              return failure();
            }
          } else {
            // GM input at full rank: copy via VECIN TQue.
            Value srcGt = builder.create<GlobalTensorOp>(
                loc, GlobalTensorType::get(elemType));
            builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                           /*size=*/Value{});
            inputLts[i] =
                copyGmToVecin(lowering, builder, loc, elemType, srcGt, totalElems,
                              bufferTotalElems, &ownedInputTensors);
          }
        } else {
          inputLts[i] = lowering.readTensor(builder, loc, inMemref);
          if (Value q = lowering.ctx.getQueue(inMemref))
            if (!lowering.ctx.getLiveTensor(inMemref))
              rememberQueueRead(ownedInputTensors, q, inputLts[i]);
        }
        break;
      }
      case IndexingMapAnalysis::Kind::PureBroadcast: {
        auto srcMrt = cast<MemRefType>(inMemref.getType());
        unsigned srcRank = srcMrt.getRank();
        if (inMs == 9 /*VECIN*/) {
          // broadcast_l2: expand narrow VECIN tile into full-shape VECCALC.
          SmallVector<Value> dstShapeVals, srcShapeVals;
          for (Value s : iterDimSizes)
            dstShapeVals.push_back(
                builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), s));
          unsigned srcDimIdx = 0;
          for (unsigned d = 0; d < iterRank; ++d) {
            bool inResult = false;
            for (AffineExpr result : inMap.getResults())
              if (auto dimExpr = dyn_cast<AffineDimExpr>(result))
                if (dimExpr.getPosition() == d) { inResult = true; break; }
            if (inResult && srcDimIdx < srcRank)
              srcShapeVals.push_back(builder.create<arith::IndexCastOp>(
                  loc, builder.getI32Type(),
                  getDynDim(lowering, builder, loc, inMemref, srcDimIdx++)));
            else
              srcShapeVals.push_back(
                  builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
          }
          Value srcLt = lowering.readTensor(builder, loc, inMemref);
          if (Value q = lowering.ctx.getQueue(inMemref))
            if (!lowering.ctx.getLiveTensor(inMemref))
              rememberQueueRead(ownedInputTensors, q, srcLt);
          auto [bcastTbuf, bcastLt] =
              allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
          auto bcastOp = builder.create<BroadcastL2Op>(
              loc, bcastLt, srcLt,
              dstShapeVals, srcShapeVals,
              builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
          lowering.copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
          inputLts[i] = bcastLt;
        } else {
          // broadcast from GM: copy via VECIN TQue first, then broadcast_l2.
          SmallVector<Value> srcDims;
          for (unsigned d = 0; d < srcRank; ++d)
            srcDims.push_back(getDynDim(lowering, builder, loc, inMemref, d));
          Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (Value d : srcDims)
            srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
          SmallVector<Value> srcBufferDims =
              getBufferDimSizes(lowering, srcDims, genOp.getOperation());
          Value srcBufferElemCount =
              lowering.computeProduct(builder, loc, srcBufferDims);
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          Value srcLt =
              copyGmToVecinScalar(lowering, builder, loc, elemType, srcGt,
                                  srcElemCount, srcBufferElemCount,
                                  &ownedInputTensors);
          SmallVector<Value> dstShapeVals, srcShapeVals;
          for (Value s : iterDimSizes)
            dstShapeVals.push_back(
                builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), s));
          unsigned srcDimIdx = 0;
          for (unsigned d = 0; d < iterRank; ++d) {
            bool inResult = false;
            for (AffineExpr result : inMap.getResults())
              if (auto dimExpr = dyn_cast<AffineDimExpr>(result))
                if (dimExpr.getPosition() == d) { inResult = true; break; }
            if (inResult && srcDimIdx < srcRank)
              srcShapeVals.push_back(builder.create<arith::IndexCastOp>(
                  loc, builder.getI32Type(), srcDims[srcDimIdx++]));
            else
              srcShapeVals.push_back(
                  builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
          }
          auto [bcastTbuf, bcastLt] =
              allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
          auto bcastOp = builder.create<BroadcastL2Op>(
              loc, bcastLt, srcLt,
              dstShapeVals, srcShapeVals,
              builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
          lowering.copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
          inputLts[i] = bcastLt;
        }
        break;
      }
      case IndexingMapAnalysis::Kind::PureTranspose: {
        // data_copy from GM into VECIN, then transpose to VECCALC.
        SmallVector<Value> srcDims;
        for (int64_t permDim : analysis.permutation)
          srcDims.push_back(iterDimSizes[static_cast<unsigned>(permDim)]);
        Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
        for (Value d : srcDims)
          srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);

        Value srcGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(elemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                       /*size=*/Value{});
        Value srcVecinLt =
            copyGmToVecin(lowering, builder, loc, elemType, srcGt, srcElemCount,
                          srcElemCount, &ownedInputTensors);

        auto [transpTbuf, transpLt] =
            allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
        auto transposeOp = builder.create<TransposeOp>(loc, transpLt, srcVecinLt);
        lowering.copyAscendCUnitAttr(genOp.getOperation(), transposeOp.getOperation());
        inputLts[i] = transpLt;
        break;
      }
      case IndexingMapAnalysis::Kind::BroadcastTranspose: {
        // Step 1: Get the input as a local_tensor in VECIN.
        // After buffer-placement, data0 may already be in VECIN (inMs==9) via
        // a memref.copy placeholder; use readTensor directly. Otherwise copy
        // from GM.
        auto srcMrt = cast<MemRefType>(inMemref.getType());
        unsigned srcRank = srcMrt.getRank();
        SmallVector<Value> srcDimsVals;
        for (unsigned d = 0; d < srcRank; ++d)
          srcDimsVals.push_back(getDynDim(lowering, builder, loc, inMemref, d));

        Value srcVecinLt;
        if (inMs == 9 /*VECIN*/) {
          srcVecinLt = lowering.readTensor(builder, loc, inMemref);
          if (Value q = lowering.ctx.getQueue(inMemref))
            if (!lowering.ctx.getLiveTensor(inMemref))
              rememberQueueRead(ownedInputTensors, q, srcVecinLt);
        } else {
          Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (Value d : srcDimsVals)
            srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          srcVecinLt =
              copyGmToVecin(lowering, builder, loc, elemType, srcGt, srcElemCount,
                            srcElemCount, &ownedInputTensors);
        }

        // Step 2: Broadcast directly into iteration-space order [iterDimSizes]
        // without a Transpose. AscendC::Transpose(dst, src) only works
        // correctly for square matrices; non-square cases produce wrong results
        // in the simulator. Instead, build dstShape = iterDimSizes and srcShape
        // with broadcast dims set to 1 and present dims set to their sizes.
        //
        // Example: map (d0,d1)->(d1,0), iterDimSizes=[Tb_N, M], src=[M,1]
        //   dstShape = [Tb_N, M]
        //   srcShape = [1,    M]   (d0=broadcast→1, d1=present→M)
        //   axis=0 (first src dim is 1, i.e. broadcast along first axis)
        unsigned iterRank = iterDimSizes.size();
        SmallVector<Value> bcastDstShape, bcastSrcShape;
        for (unsigned d = 0; d < iterRank; ++d) {
          bcastDstShape.push_back(builder.create<arith::IndexCastOp>(
              loc, builder.getI32Type(), iterDimSizes[d]));
        }
        // srcShape: for each iteration dim, if it appears in presentDims of the
        // map put the actual src size, otherwise put 1 (broadcast dim).
        // We need to map iter-dim → src-dim via the src's indexing map results.
        // Build a lookup: iter dim position → src dim index (or -1 if broadcast).
        SmallVector<int64_t> iterDimToSrcDim(iterRank, -1);
        for (unsigned r = 0; r < inMap.getNumResults(); ++r) {
          AffineExpr expr = inMap.getResult(r);
          if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
            // iter dim dimExpr.getPosition() maps to src dimension r
            iterDimToSrcDim[dimExpr.getPosition()] = static_cast<int64_t>(r);
          }
        }
        for (unsigned d = 0; d < iterRank; ++d) {
          int64_t srcDimIdx = iterDimToSrcDim[d];
          if (srcDimIdx >= 0) {
            bcastSrcShape.push_back(builder.create<arith::IndexCastOp>(
                loc, builder.getI32Type(), srcDimsVals[srcDimIdx]));
          } else {
            bcastSrcShape.push_back(
                builder.create<arith::ConstantOp>(
                    loc, builder.getI32IntegerAttr(1)));
          }
        }

        auto [finalTbuf, finalLt] =
            allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, finalLt, srcVecinLt,
            bcastDstShape, bcastSrcShape,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        lowering.copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
        inputLts[i] = finalLt;
        break;
      }
      } // end switch
    }

    if (outQueue)
      accumLt = lowering.allocTensor(builder, loc, outQueue, elemType);

    // ---- Step 2: Walk body and inline arith ops onto VECCALC tensors ----
    Block &bodyBlock = *genOp.getBody();
    unsigned numBodyArgs = bodyBlock.getNumArguments();
    SmallVector<Value> argToLt(numBodyArgs);
    for (unsigned i = 0; i < numInputs; ++i)
      argToLt[i] = inputLts[i];
    argToLt[numInputs] = accumLt;

    llvm::SmallDenseMap<Value, Value> valToLt;
    bool accumLtHasBodyValue = false;
    auto emitAccumReadAfterWriteBarrier = [&](bool readsAccumLt) {
      if (!accumLtHasBodyValue || !readsAccumLt)
        return;
      builder.create<PipeBarrierOp>(
          loc, PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));
    };
    for (auto &bodyOp : bodyBlock.without_terminator()) {
      auto resolve = [&](Value v) -> Value {
        if (auto ba = dyn_cast<BlockArgument>(v))
          return argToLt[ba.getArgNumber()];
        auto it = valToLt.find(v);
        if (it != valToLt.end()) return it->second;
        // Scalar constant? Fill a fresh VECCALC with duplicate_l2.
        if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
          auto [dupTbuf, dupLt] =
              allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
          auto dupOp = builder.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), totalElems);
          lowering.copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
          builder.create<PipeBarrierOp>(
              loc, PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));
          valToLt[v] = dupLt;
          return dupLt;
        }
        return Value{};
      };

      using namespace mlir::ascend::backend;
      const ElementwiseBodyOpEntry *entry =
          lookupElementwiseBodyOp(bodyOp.getName().getStringRef());
      if (!entry) continue;

      if (entry->unaryEmitter) {
        Value src = resolve(bodyOp.getOperand(0));
        if (!src) continue;
        emitAccumReadAfterWriteBarrier(src == accumLt);
        entry->unaryEmitter(builder, loc, accumLt, src, totalElems);
        lowering.copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        valToLt[bodyOp.getResult(0)] = accumLt;
        accumLtHasBodyValue = true;
      } else if (entry->binaryEmitter) {
        Value lhs = resolve(bodyOp.getOperand(0));
        Value rhs = resolve(bodyOp.getOperand(1));
        if (!lhs || !rhs) continue;
        emitAccumReadAfterWriteBarrier(lhs == accumLt || rhs == accumLt);
        entry->binaryEmitter(builder, loc, accumLt, lhs, rhs, totalElems);
        lowering.copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        valToLt[bodyOp.getResult(0)] = accumLt;
        accumLtHasBodyValue = true;
      }
    }

    if (!ownedInputTensors.empty())
      builder.create<PipeBarrierOp>(loc, PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));
    freeOwnedQueueTensors(builder, loc, ownedInputTensors);

    // ---- Step 3: Write accumulator to output buffer ----
    // No reduction needed (all-parallel). The compute result is in accumLt
    // (a VECCALC tbuf). We need to deliver it to the output buffer:
    //
    //   VECOUT (ms=10): alloc from queue, use AddL2 to copy accumLt→vecoutLt
    //                   (add_l2(dst, src, zero_tbuf, count) would need a zero
    //                    tensor; instead use the queue alloc tensor directly and
    //                    simply enqueue accumLt if the queue accepts VECCALC).
    //                   Simplest: treat the VECCALC accumLt as the enqueue source
    //                   and let the downstream DataMovementConversion handle writeback.
    //   VECCALC (ms=11): accumLt already holds the result; no copy needed.
    //
    // Key insight: the epilogue memref.copy (VECOUT->GM) from memory
    // realization is converted by DataMovementConversion into a data_copy_l2 with
    // the correct subview offset, so the concat position is preserved
    // automatically. We just need to enqueue the result tensor.
    if (outQueue) {
      // The queue expects a tensor allocated from the same queue.  Real
      // hardware is stricter than the simulator here; enqueueing a VECCALC
      // tbuf tensor into a VECOUT queue can surface as UB/MTE faults.
      builder.create<TQueBindEnqueTensorOp>(loc, outQueue, accumLt);
    }
    // If outMemref has no queue (VECCALC alloc without a queue), the result
    // already resides in the VECCALC tbuf and will be consumed by the next op.

    genOp.erase();
  }

  return success();
}

} // namespace ascend
} // namespace mlir
