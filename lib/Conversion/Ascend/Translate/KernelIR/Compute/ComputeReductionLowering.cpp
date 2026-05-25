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
namespace afir {

LogicalResult lowerReductionComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  OpBuilder &builder = lowering.builder;
  SmallVector<linalg::GenericOp> genericOps;
  funcOp.walk([&](linalg::GenericOp op) { genericOps.push_back(op); });

  for (linalg::GenericOp genOp : genericOps) {
    // Only handle generics that contain at least one reduction iterator.
    auto iterTypes = genOp.getIteratorTypesArray();
    bool hasReduction = llvm::any_of(iterTypes, [](utils::IteratorType t) {
      return t == utils::IteratorType::reduction;
    });
    if (!hasReduction)
      continue;

    // Require exactly one init (output) for now.
    if (genOp.getNumDpsInits() != 1)
      continue;

    unsigned numInputs = genOp.getNumDpsInputs();
    unsigned iterRank  = iterTypes.size();
    auto maps          = genOp.getIndexingMapsArray();
    Value outMemref    = genOp.getDpsInitOperand(0)->get();
    int64_t outMs      = getMemorySpace(outMemref.getType());
    if (outMs <= 0)
      continue; // output must be on-chip

    Location loc = genOp.getLoc();
    builder.setInsertionPoint(genOp);
    Type elemType = cast<MemRefType>(outMemref.getType()).getElementType();
    ascend::backend::AscendBackendSupportMatrix matrix;
    ascend::backend::ComputeKind reductionKind =
        ascend::backend::classifyBackendReductionBody(genOp, matrix);
    if (reductionKind == ascend::backend::ComputeKind::Unknown)
      continue;

    auto findPrecedingFillInit = [](Value output, Operation *writer) -> Value {
      Operation *bestFill = nullptr;
      Value bestInit;
      for (Operation *user : output.getUsers()) {
        auto fillOp = dyn_cast<linalg::FillOp>(user);
        if (!fillOp || fillOp.getOutputs()[0] != output)
          continue;
        if (fillOp->getBlock() != writer->getBlock() ||
            !fillOp->isBeforeInBlock(writer))
          continue;
        if (!bestFill || bestFill->isBeforeInBlock(fillOp)) {
          bestFill = fillOp.getOperation();
          bestInit = fillOp.getInputs()[0];
        }
      }
      return bestInit;
    };

    auto buildNeutralIdentity =
        [&](ascend::backend::ComputeKind kind) -> Value {
      if (!isa<FloatType>(elemType))
        return Value{};

      double identity = 0.0;
      switch (kind) {
      case ascend::backend::ComputeKind::ReductionAdd:
        identity = 0.0;
        break;
      case ascend::backend::ComputeKind::ReductionMul:
        identity = 1.0;
        break;
      case ascend::backend::ComputeKind::ReductionMax:
        identity = -std::numeric_limits<double>::infinity();
        break;
      case ascend::backend::ComputeKind::ReductionMin:
        identity = std::numeric_limits<double>::infinity();
        break;
      default:
        return Value{};
      }
      return builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(elemType, identity));
    };

    // ------------------------------------------------------------------
    // Step 1: For each input, promote it to a VECCALC local_tensor.
    //   - "broadcast" input (VECIN, ms==9): use broadcast_l2 to expand
    //     the 1-D tile into the full 2-D iteration shape.
    //   - "full" input (GM, ms==0): data_copy_l2 into a fresh VECCALC.
    //   - "full" input (VECIN, ms==9): already a local_tensor; use readTensor.
    // The result is a SmallVector of VECCALC local_tensors, one per input.
    // ------------------------------------------------------------------

    // Compute the parallel and reduction dim sizes from the output memref
    // and the 2D input (if present).  We derive the full [M, N] iteration
    // shape from the first "full" input (rank == iterRank).
    SmallVector<Value> iterDimSizes(iterRank);
    for (unsigned i = 0; i < numInputs; ++i) {
      Value inMemref = genOp.getDpsInputOperand(i)->get();
      AffineMap inMap = maps[i];
      if (inMap.getNumResults() == iterRank) {
        // Full map — use this operand to fill iterDimSizes.
        for (unsigned d = 0; d < iterRank; ++d)
          iterDimSizes[d] = getDynDim(lowering, builder, loc, inMemref, d);
        break;
      }
    }
    // Fall back: fill remaining parallel dims from output (output only covers
    // parallel dims, so only use it when the iterator type is parallel).
    {
      unsigned outDim = 0;
      for (unsigned d = 0; d < iterRank; ++d) {
        if (!iterDimSizes[d] && iterTypes[d] == utils::IteratorType::parallel)
          iterDimSizes[d] = getDynDim(lowering, builder, loc, outMemref, outDim++);
      }
    }

    // Collect parallel and reduction dim sizes.
    SmallVector<Value> parallelDims, reductionDims;
    for (unsigned d = 0; d < iterRank; ++d) {
      if (iterTypes[d] == utils::IteratorType::parallel)
        parallelDims.push_back(iterDimSizes[d]);
      else
        reductionDims.push_back(iterDimSizes[d]);
    }

    // Full shape = parallelDims ++ reductionDims (for 2D: [M, N]).
    SmallVector<Value> fullShape;
    llvm::append_range(fullShape, parallelDims);
    llvm::append_range(fullShape, reductionDims);
    Value totalElems = lowering.computeProduct(builder, loc, fullShape);
    // Build a VECCALC accumulator for the full shape.  This is the tensor
    // that will hold the element-wise intermediate results before reduction.
    Value accumLt = allocVeccalc(lowering, builder, loc, elemType, fullShape).second;

    // Initialize the expanded accumulator with the reduction identity. Prefer
    // the producer fill value when present, because linalg outs carries the
    // semantic init. Fall back to the neutral identity for legacy reductions
    // that arrive without an explicit fill in the same block.
    Value initVal = findPrecedingFillInit(outMemref, genOp.getOperation());
    if (!initVal)
      initVal = buildNeutralIdentity(reductionKind);
    if (initVal) {
      auto initDup =
          builder.create<DuplicateL2Op>(loc, accumLt, initVal, totalElems);
      lowering.copyAscendCUnitAttr(genOp.getOperation(), initDup.getOperation());
      builder.create<PipeBarrierOp>(loc,
                                    PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));
    }

    // Promote each input to a local_tensor of shape `fullShape`.
    SmallVector<OwnedQueueTensor> ownedInputTensors;
    SmallVector<Value> inputLts(numInputs);
    for (unsigned i = 0; i < numInputs; ++i) {
      Value inMemref = genOp.getDpsInputOperand(i)->get();
      AffineMap inMap = maps[i];
      int64_t inMs    = getMemorySpace(inMemref.getType());
      bool isBcast    = isBroadcastMap(inMap, iterRank);

      if (isBcast && inMs == 9 /*VECIN*/) {
        // broadcast_l2: expand the narrow VECIN tile into the full 2D VECCALC.
        // src shape follows the map results; dst shape is fullShape.
        // Determine src shape values from the operand's memref dims.
        auto srcMrt = cast<MemRefType>(inMemref.getType());
        unsigned srcRank = srcMrt.getRank();
        // Build i32 shape arrays expected by broadcast_l2.
        SmallVector<Value> dstShapeVals, srcShapeVals;
        // dstShape = fullShape cast to i32
        for (Value s : fullShape)
          dstShapeVals.push_back(
              builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), s));
        // srcShape: dims present in inMap result, others are 1.
        // For a map (d0,d1)->(d0): srcShape=[M, 1] for 2D iteration.
        unsigned srcDimIdx = 0;
        for (unsigned d = 0; d < iterRank; ++d) {
          // Check if dim d appears in inMap results.
          bool inResult = false;
          for (AffineExpr result : inMap.getResults()) {
            if (auto dimExpr = dyn_cast<AffineDimExpr>(result))
              if (dimExpr.getPosition() == d) { inResult = true; break; }
          }
          if (inResult && srcDimIdx < srcRank)
            srcShapeVals.push_back(
                builder.create<arith::IndexCastOp>(
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
            allocVeccalc(lowering, builder, loc, elemType, fullShape);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, bcastLt, srcLt,
            dstShapeVals, srcShapeVals,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        lowering.copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
        inputLts[i] = bcastLt;
      } else if (isBcast && inMs == 0 /*GM*/) {
        // broadcast from GM: copy the small src tensor into VECIN via TQue
        // first (GM→VECIN DataCopy), then broadcast_l2 VECIN→VECCALC.
        // Using TQue is required because the AscendC simulator does not
        // support DataCopy directly from GM to VECCALC TBuf.
        auto srcMrt = cast<MemRefType>(inMemref.getType());
        unsigned srcRank = srcMrt.getRank();
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
        for (Value s : fullShape)
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
            allocVeccalc(lowering, builder, loc, elemType, fullShape);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, bcastLt, srcLt,
            dstShapeVals, srcShapeVals,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        lowering.copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
        inputLts[i] = bcastLt;
      } else if (inMs == 0 /*GM*/) {
        // GM input at full rank: copy via VECIN TQue (simulator requires
        // DataCopy to go through TQue, not directly to VECCALC TBuf).
        Value srcGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(elemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                       /*size=*/Value{});
        inputLts[i] =
            copyGmToVecin(lowering, builder, loc, elemType, srcGt, totalElems,
                          totalElems, &ownedInputTensors);
      } else {
        // Already VECIN or VECCALC — use readTensor as-is.
        inputLts[i] = lowering.readTensor(builder, loc, inMemref);
        if (Value q = lowering.ctx.getQueue(inMemref))
          if (!lowering.ctx.getLiveTensor(inMemref))
            rememberQueueRead(ownedInputTensors, q, inputLts[i]);
      }
    }

    // ------------------------------------------------------------------
    // Step 2: Walk the body and inline each arith op onto VECCALC tensors.
    //
    // The body block args map to: [ins..., outs...].
    // We maintain a map from block arg index → current VECCALC local_tensor.
    // For each arith op, we emit the corresponding AscendC vector op and
    // record the result local_tensor for the op's SSA result.
    //
    // Supported body ops:
    //   arith.addf(x, y)  →  add_l2(accumLt, lt[x], lt[y], totalElems)
    //   arith.maxf(x, y)  →  max_l2(accumLt, lt[x], lt[y], totalElems)
    //   linalg.yield      →  (terminal, skipped)
    //
    // We use accumLt as the destination for all intermediate results
    // (in-place style, reusing the single VECCALC buffer).
    // ------------------------------------------------------------------
    Block &bodyBlock = *genOp.getBody();
    // bodyBlock.getArguments(): [in0, in1, ..., out0]
    unsigned numBodyArgs = bodyBlock.getNumArguments();
    SmallVector<Value> argToLt(numBodyArgs);
    for (unsigned i = 0; i < numInputs; ++i)
      argToLt[i] = inputLts[i];
    // Output block arg starts life as accumLt (the running accumulator).
    argToLt[numInputs] = accumLt;

    // Walk body ops in order (excluding linalg.yield).
    // Each arith op produces one SSA value; we map it to a VECCALC local_tensor.
    // Determine which SSA value is yielded (the final accumulator result).
    // Only the op that produces this value writes to accumLt; intermediate
    // ops get fresh VECCALC buffers so that accumLt is never aliased with a
    // temporary, avoiding the add(acc,acc) doubling bug.
    Value yieldedVal;
    if (auto yield = dyn_cast<linalg::YieldOp>(bodyBlock.getTerminator()))
      if (!yield.getValues().empty())
        yieldedVal = yield.getValues()[0];

    llvm::SmallDenseMap<Value, Value> valToLt;
    for (auto &bodyOp : bodyBlock.without_terminator()) {
      // Resolve an SSA value to its corresponding local_tensor.
      // Handles block args, prior body results, and scalar constants
      // (via duplicate_l2 into a fresh VECCALC tensor).
      auto resolve = [&](Value v) -> Value {
        // Block argument?
        if (auto ba = dyn_cast<BlockArgument>(v))
          return argToLt[ba.getArgNumber()];
        // Result of a previous body op?
        auto it = valToLt.find(v);
        if (it != valToLt.end()) return it->second;
        // Scalar constant? Fill a fresh VECCALC with duplicate_l2.
        if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
          auto [dupTbuf, dupLt] =
              allocVeccalc(lowering, builder, loc, elemType, fullShape);
          auto dupOp = builder.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), totalElems);
          lowering.copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
          builder.create<PipeBarrierOp>(
              loc, PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));
          valToLt[v] = dupLt;
          return dupLt;
        }
        return Value{};
      };

      // Choose destination: accumLt for the final yielded op, fresh buffer otherwise.
      auto chooseDst = [&](Value result) -> Value {
        if (result == yieldedVal)
          return accumLt;
        auto [tmpTbuf, tmpLt] =
            allocVeccalc(lowering, builder, loc, elemType, fullShape);
        valToLt[result] = tmpLt;
        return tmpLt;
      };

      if (isa<arith::ConstantOp>(bodyOp))
        continue;

      using namespace mlir::afir::ascend::backend;
      const ElementwiseBodyOpEntry *entry =
          lookupElementwiseBodyOp(bodyOp.getName().getStringRef());
      if (!entry)
        continue;

      if (entry->unaryEmitter) {
        Value src = resolve(bodyOp.getOperand(0));
        if (!src) continue;
        Value dst = chooseDst(bodyOp.getResult(0));
        entry->unaryEmitter(builder, loc, dst, src, totalElems);
        lowering.copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        if (dst == accumLt) valToLt[bodyOp.getResult(0)] = accumLt;
      } else if (entry->binaryEmitter) {
        Value lhs = resolve(bodyOp.getOperand(0));
        Value rhs = resolve(bodyOp.getOperand(1));
        if (!lhs || !rhs) continue;
        Value dst = chooseDst(bodyOp.getResult(0));
        entry->binaryEmitter(builder, loc, dst, lhs, rhs, totalElems);
        lowering.copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        if (dst == accumLt) valToLt[bodyOp.getResult(0)] = accumLt;
      }
    }

    freeOwnedQueueTensors(builder, loc, ownedInputTensors);

    // ------------------------------------------------------------------
    // Step 3: Reduce the accumulated VECCALC to the output VECOUT tensor.
    //
    // For a 2D iteration [parallel_dim, reduction_dim] with AR layout:
    //   reduce_sum_2d_l2(vecoutLt, accumLt, AR, no_tmp)
    // ------------------------------------------------------------------
    Value vecoutLt = lowering.writeTensor(builder, loc, outMemref);
    auto layoutAttr = ReduceLayoutAttr::get(lowering.mlirCtx, ReduceLayout::AR);
    if (reductionKind == ascend::backend::ComputeKind::ReductionMax) {
      auto reduceOp = builder.create<ReduceMax2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                      /*sharedTmpBuffer=*/Value{});
      lowering.copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    } else if (reductionKind == ascend::backend::ComputeKind::ReductionMin) {
      auto reduceOp = builder.create<ReduceMin2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                      /*sharedTmpBuffer=*/Value{});
      lowering.copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    } else if (reductionKind == ascend::backend::ComputeKind::ReductionMul) {
      auto reduceOp = builder.create<ReduceProd2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                       /*sharedTmpBuffer=*/Value{});
      lowering.copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    } else {
      auto reduceOp = builder.create<ReduceSum2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                      /*sharedTmpBuffer=*/Value{});
      lowering.copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    }

    // Enqueue vecout if it has a queue (VECOUT path).
    if (Value q = lowering.ctx.getQueue(outMemref))
      builder.create<TQueBindEnqueTensorOp>(loc, q, vecoutLt);

    genOp.erase();
  }

  return success();
}

} // namespace afir
} // namespace mlir
