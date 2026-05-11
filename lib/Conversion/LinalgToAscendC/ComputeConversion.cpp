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

#include "Conversion/LinalgToAscendC/LinalgToAscendCUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#include <algorithm>

#define DEBUG_TYPE "linalg-to-ascendc-compute"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace afir {

LogicalResult convertCompute(func::FuncOp funcOp, AscendCBufferContext &ctx) {
  MLIRContext *mlirCtx = funcOp.getContext();
  OpBuilder builder(mlirCtx);

  auto copyAscendCUnitAttr = [](Operation *src, Operation *dst) {
    if (!src || !dst)
      return;
    if (auto unitAttr = src->getAttrOfType<StringAttr>("ascendc.unit"))
      dst->setAttr("ascendc.unit", unitAttr);
  };

  auto getEnclosingLoopStepBound = [](Value value,
                                      Operation *anchor) -> Value {
    auto matchesEnclosingStep = [&](Value candidate) -> bool {
      for (Operation *parent = anchor; parent; parent = parent->getParentOp()) {
        auto forOp = dyn_cast<scf::ForOp>(parent);
        if (forOp && candidate == forOp.getStep())
          return true;
      }
      return false;
    };

    if (auto minOp = value.getDefiningOp<arith::MinSIOp>()) {
      if (matchesEnclosingStep(minOp.getLhs()))
        return minOp.getLhs();
      if (matchesEnclosingStep(minOp.getRhs()))
        return minOp.getRhs();
    }
    if (auto minOp = value.getDefiningOp<arith::MinUIOp>()) {
      if (matchesEnclosingStep(minOp.getLhs()))
        return minOp.getLhs();
      if (matchesEnclosingStep(minOp.getRhs()))
        return minOp.getRhs();
    }
    if (auto minOp = value.getDefiningOp<affine::AffineMinOp>()) {
      for (Value operand : minOp.getOperands())
        if (matchesEnclosingStep(operand))
          return operand;
    }
    return value;
  };

  auto getSubviewSizeValue = [&](OpBuilder &b, Location loc, Value memref,
                                 unsigned dim) -> Value {
    auto subviewOp = memref.getDefiningOp<memref::SubViewOp>();
    if (!subviewOp)
      return Value{};
    SmallVector<OpFoldResult> mixedSizes = subviewOp.getMixedSizes();
    if (dim >= mixedSizes.size())
      return Value{};
    OpFoldResult size = mixedSizes[dim];
    if (auto attr = size.dyn_cast<Attribute>())
      return b.create<arith::ConstantIndexOp>(loc,
                                              cast<IntegerAttr>(attr).getInt());
    return size.get<Value>();
  };

  auto computeProduct = [&](OpBuilder &b, Location loc,
                            ArrayRef<Value> dims) -> Value {
    Value totalElems;
    for (Value s : dims)
      totalElems = totalElems ? b.create<arith::MulIOp>(loc, totalElems, s) : s;
    if (!totalElems)
      totalElems = b.create<arith::ConstantIndexOp>(loc, 1);
    return totalElems;
  };

  // Helper: get a local_tensor by deque from a queue.
  auto dequeTensor = [&](OpBuilder &b, Location loc, Value queue,
                         Type elemType) -> Value {
    return b.create<TQueBindDequeTensorOp>(loc, LocalTensorType::get(elemType),
                                           queue);
  };

  // Helper: alloc a local_tensor from a queue.
  auto allocTensor = [&](OpBuilder &b, Location loc, Value queue,
                         Type elemType) -> Value {
    return b.create<TQueBindAllocTensorOp>(loc, LocalTensorType::get(elemType),
                                           queue);
  };

  // Helper: get_tensor from a fresh TBuf (for VECCALC temporaries with no
  // queue, or buffers without an alloc-based queue).
  auto tbufTensor = [&](OpBuilder &b, Location loc, int64_t ms,
                        Type elemType) -> Value {
    auto pos = static_cast<TPosition>(ms > 0 ? ms : 0);
    Value tbuf = b.create<TBufOp>(loc, TBufType::get(mlirCtx, pos));
    return b.create<TBufGetTensorOp>(loc, LocalTensorType::get(elemType), tbuf,
                                     /*len=*/Value{});
  };

  // Helper: compute the linear byte offset for a subview into its parent alloc.
  // For a 2D row-major parent with shape [D0 x D1]:
  //   linear_offset_bytes = (offsets[0] * D1 + offsets[1]) * elem_bytes
  // Returns null Value if `memref` is not a subview.
  auto subviewByteOffset = [&](OpBuilder &b, Location loc,
                                Value memref) -> Value {
    auto subviewOp = memref.getDefiningOp<memref::SubViewOp>();
    if (!subviewOp)
      return Value{};
    Value parent = subviewOp.getSource();
    auto parentType = cast<MemRefType>(parent.getType());
    if (parentType.getRank() != 2)
      return Value{};

    SmallVector<OpFoldResult> mixedOffsets = subviewOp.getMixedOffsets();
    // Row stride = dim[1] of parent alloc.
    Value rowStride;
    if (!ShapedType::isDynamic(parentType.getShape()[1]))
      rowStride =
          b.create<arith::ConstantIndexOp>(loc, parentType.getShape()[1]);
    else
      rowStride = b.create<memref::DimOp>(
          loc, parent, b.create<arith::ConstantIndexOp>(loc, 1));

    auto toIndex = [&](OpFoldResult ofr) -> Value {
      if (auto attr = ofr.dyn_cast<Attribute>())
        return b.create<arith::ConstantIndexOp>(
            loc, cast<IntegerAttr>(attr).getInt());
      return ofr.get<Value>();
    };
    Value off0 = toIndex(mixedOffsets[0]);
    Value off1 = toIndex(mixedOffsets[1]);

    Value linearElems = b.create<arith::MulIOp>(loc, off0, rowStride);
    linearElems = b.create<arith::AddIOp>(loc, linearElems, off1);
    unsigned elemBytes = parentType.getElementTypeBitWidth() / 8;
    Value bytesVal = b.create<arith::ConstantIndexOp>(loc, elemBytes);
    return b.create<arith::MulIOp>(loc, linearElems, bytesVal);
  };

  // Helper: obtain a local_tensor slice via tbuf.get_with_offset.
  // Returns null if no tbuf registered for `memref`.
  auto tbufSlice = [&](OpBuilder &b, Location loc, Value memref,
                        Value sizeBytes, Value offsetBytes) -> Value {
    Value tbuf = ctx.getTBuf(memref);
    if (!tbuf)
      return Value{};
    auto mrt = cast<MemRefType>(memref.getType());
    return b.create<TBufGetWithOffsetOp>(
        loc, LocalTensorType::get(mrt.getElementType()), tbuf, sizeBytes,
        offsetBytes);
  };

  // Helper: get a read-side local_tensor for a compute operand.
  // If `memref` is a subview of a live-tensor buffer, use tbuf.get_with_offset
  // to obtain the correctly-offset slice (avoids returning the whole tensor).
  // Otherwise deque from queue or fall back to fresh tbuf.
  auto readTensor = [&](OpBuilder &b, Location loc, Value memref) -> Value {
    auto mrt = cast<MemRefType>(memref.getType());
    if (ctx.getLiveTensor(memref)) {
      if (Value byteOff = subviewByteOffset(b, loc, memref)) {
        Value sizeBytes = computeByteCount(b, loc, memref);
        if (Value t = tbufSlice(b, loc, memref, sizeBytes, byteOff))
          return t;
      }
      return ctx.getLiveTensor(memref);
    }
    if (Value q = ctx.getQueue(memref))
      return dequeTensor(b, loc, q, mrt.getElementType());
    return tbufTensor(b, loc, getMemorySpace(mrt), mrt.getElementType());
  };

  // Helper: get a write-side local_tensor for a compute output.
  // If `memref` is a subview, use tbuf.get_with_offset so the write lands at
  // the correct offset inside the parent buffer (e.g. VECOUT sub-tile).
  // Otherwise prefer alloc_tensor from queue, else fresh tbuf.
  auto writeTensor = [&](OpBuilder &b, Location loc, Value memref) -> Value {
    auto mrt = cast<MemRefType>(memref.getType());
    if (Value byteOff = subviewByteOffset(b, loc, memref)) {
      Value sizeBytes = computeByteCount(b, loc, memref);
      if (Value t = tbufSlice(b, loc, memref, sizeBytes, byteOff))
        return t;
    }
    if (Value q = ctx.getQueue(memref))
      return allocTensor(b, loc, q, mrt.getElementType());
    return tbufTensor(b, loc, getMemorySpace(mrt), mrt.getElementType());
  };

  // Helper: return the nearest enclosing scf::ForOp of `op`, or nullptr.
  auto getEnclosingFor = [](Operation *op) -> scf::ForOp {
    for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
      if (auto f = dyn_cast<scf::ForOp>(p))
        return f;
    return nullptr;
  };

  // Helper: allocate a write-side tensor before the nearest enclosing for-loop
  // and enqueue it after.  This ensures the queue slot stays valid across all
  // iterations of that loop (e.g. CO1 accumulating across K, VECOUT across
  // Tb_M/Tb_N).  Returns {localTensor, hoistFor} where hoistFor may be null.
  auto allocHoisted =
      [&](Operation *op, Value queue, Type elemType,
          Location loc) -> std::pair<Value, scf::ForOp> {
    scf::ForOp forOp = getEnclosingFor(op);
    if (!forOp)
      return {allocTensor(builder, loc, queue, elemType), nullptr};
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(forOp);
    Value tensor = allocTensor(builder, loc, queue, elemType);
    return {tensor, forOp};
  };

  // --- linalg.generic {iterator_types contains "reduction"} ---
  //
  // Generic lowering for reduction generics (e.g. broadcast+add+reducesum).
  // The strategy follows the AscendNPU vector memory hierarchy:
  //   GM → VECIN (via data_copy_l2)
  //   VECIN → VECCALC (via broadcast_l2 / add_l2 / etc., inlined from body)
  //   VECCALC → VECOUT (via reduce_sum_2d_l2 for reduction dims)
  //   VECOUT → GM (via data_copy_l2, handled by data-move pass)
  //
  // Body inlining rules:
  //   - Each input is classified by its indexing map:
  //       * "broadcast" input: map results < loop dims (some dims absent) → broadcast_l2
  //       * "full" input: map results == loop dims → direct copy into VECCALC via data_copy_l2
  //   - GM inputs (memory_space == 0) are dynamically copied into a fresh VECCALC.
  //   - VECIN inputs (memory_space == 9) that are broadcast get broadcast_l2'd into VECCALC.
  //   - Body arith ops are walked in order; each arith.addf / arith.maxf maps to add_l2 / max_l2
  //     operating on the accumulated VECCALC tensors.
  //   - The final accumulated VECCALC (over parallel dims) is reduced via reduce_sum_2d_l2
  //     with ReduceLayout::AR (A=parallel rows, R=reduction cols).
  //
  // Analysis of a single input indexing map relative to the iteration space.
  struct IndexingMapAnalysis {
    enum class Kind {
      Identity,           // (d0,d1)->(d0,d1): direct read
      PureBroadcast,      // (d0,d1)->(d0): some dims absent, no reordering
      PureTranspose,      // (d0,d1)->(d1,d0): all dims present, permuted
      BroadcastTranspose, // (d0,d1)->(d1,0): constants + reordering
    };
    Kind kind;
    SmallVector<int64_t> permutation;    // valid for PureTranspose, BroadcastTranspose
    SmallVector<int64_t> broadcastDims;  // iteration dims absent from output
  };

  // Analyze an input indexing map to classify how the input is accessed
  // relative to the iteration space of rank `iterRank`.
  auto analyzeIndexingMap = [](AffineMap map,
                                unsigned iterRank) -> IndexingMapAnalysis {
    IndexingMapAnalysis result;

    // Identity: fast path
    if (map.isIdentity()) {
      result.kind = IndexingMapAnalysis::Kind::Identity;
      return result;
    }

    // Collect which iteration dims appear in the map results (as dim exprs)
    // and which results are constants.
    SmallVector<int64_t> presentDims;  // iteration dim positions that appear
    bool hasConstant = false;
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        presentDims.push_back(static_cast<int64_t>(dimExpr.getPosition()));
      } else if (isa<AffineConstantExpr>(expr)) {
        hasConstant = true;
      } else {
        // Non-trivial affine expression: not handled.
        result.kind = IndexingMapAnalysis::Kind::Identity; // fallback: treat as identity
        return result;
      }
    }

    // Determine broadcast dims: iteration dims not in presentDims.
    for (unsigned d = 0; d < iterRank; ++d) {
      if (llvm::find(presentDims, static_cast<int64_t>(d)) == presentDims.end())
        result.broadcastDims.push_back(d);
    }

    bool hasBroadcast = !result.broadcastDims.empty() || hasConstant;
    bool hasTranspose = !llvm::is_sorted(presentDims);

    if (hasConstant || (hasBroadcast && hasTranspose)) {
      result.kind = IndexingMapAnalysis::Kind::BroadcastTranspose;
      result.permutation.assign(presentDims.begin(), presentDims.end());
      return result;
    }

    if (hasBroadcast) {
      result.kind = IndexingMapAnalysis::Kind::PureBroadcast;
      return result;
    }

    if (hasTranspose) {
      result.kind = IndexingMapAnalysis::Kind::PureTranspose;
      result.permutation.assign(presentDims.begin(), presentDims.end());
      return result;
    }

    result.kind = IndexingMapAnalysis::Kind::Identity;
    return result;
  };

  // Helper: return true when an AffineMap is a "broadcast" map for the given
  // iterator rank — i.e., it projects away at least one dimension (a dim whose
  // axis does not appear in the map's result expressions).
  auto isBroadcastMap = [](AffineMap map, unsigned iterRank) -> bool {
    if (map.getNumResults() >= iterRank)
      return false;
    return true;
  };

  // Helper: allocate a fresh on-chip VECCALC buffer matching the given dynamic
  // sizes, insert tbuf + init_buffer, and return {tbufVal, localTensorVal}.
  // On-chip buffers do not use memref.alloc; lifetime is managed by TPipe.
  auto allocVeccalc =
      [&](OpBuilder &b, Location loc, Type elemType,
          SmallVector<Value> dynSizes) -> std::pair<Value, Value> {
    Value tbuf = b.create<TBufOp>(loc, TBufType::get(mlirCtx, TPosition::VECCALC));
    // Byte size = product(dynSizes) * elemBytes
    Value totalElems;
    for (Value s : dynSizes)
      totalElems = totalElems ? b.create<arith::MulIOp>(loc, totalElems, s) : s;
    if (!totalElems)
      totalElems = b.create<arith::ConstantIndexOp>(loc, 1);
    unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    Value byteSize = b.create<arith::MulIOp>(
        loc, totalElems, b.create<arith::ConstantIndexOp>(loc, elemBytes));
    b.create<TPipeInitBufferOp>(loc, ctx.pipe, tbuf, byteSize);

    Value lt = b.create<TBufGetTensorOp>(
        loc, LocalTensorType::get(elemType), tbuf, /*len=*/Value{});
    return {tbuf, lt};
  };

  // Helper: copy `elemCount` elements from a GM GlobalTensor into a fresh
  // VECIN TQue (AllocTensor → DataCopy → EnQue → DeQue) and return the
  // dequeued VECIN LocalTensor.  The AscendC simulator only supports
  // DataCopy from GM → VECIN TQue (not directly to VECCALC TBuf).
  auto copyGmToVecin =
      [&](OpBuilder &b, Location loc, Type elemType, Value srcGt,
          Value elemCount, Value bufferElemCount,
          SmallVectorImpl<std::pair<Value, Value>> *tempVecinTensors) -> Value {
    if (!bufferElemCount)
      bufferElemCount = elemCount;
    unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    Value byteSize = b.create<arith::MulIOp>(
        loc, bufferElemCount, b.create<arith::ConstantIndexOp>(loc, elemBytes));
    Value vecinTbuf =
        b.create<TBufOp>(loc, TBufType::get(mlirCtx, TPosition::VECIN));
    b.create<TPipeInitBufferOp>(loc, ctx.pipe, vecinTbuf, byteSize);
    Value vecinQue =
        b.create<QueueOp>(loc, QueueType::get(mlirCtx, TPosition::VECIN, 1));
    Value depth = b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(1));
    b.create<TPipeInitQueueOp>(loc, ctx.pipe, vecinQue, depth, byteSize);
    Value lt = b.create<TQueBindAllocTensorOp>(
        loc, LocalTensorType::get(elemType), vecinQue);
    b.create<DataCopyL2Op>(loc, lt, srcGt, elemCount);
    b.create<TQueBindEnqueTensorOp>(loc, vecinQue, lt);
    Value dequeued = b.create<TQueBindDequeTensorOp>(
        loc, LocalTensorType::get(elemType), vecinQue);
    if (tempVecinTensors)
      tempVecinTensors->push_back({vecinQue, dequeued});
    return dequeued;
  };

  // Helper: get a runtime Value for dimension `dim` of a memref.
  auto getDynDim = [&](OpBuilder &b, Location loc, Value memref,
                        unsigned dim) -> Value {
    if (Value subviewSize = getSubviewSizeValue(b, loc, memref, dim))
      return subviewSize;
    auto mrt = cast<MemRefType>(memref.getType());
    if (!ShapedType::isDynamic(mrt.getShape()[dim]))
      return b.create<arith::ConstantIndexOp>(loc, mrt.getShape()[dim]);
    return b.create<memref::DimOp>(loc, memref, dim);
  };

  auto getBufferDimSizes = [&](ArrayRef<Value> dims,
                               Operation *anchor) -> SmallVector<Value> {
    SmallVector<Value> bufferDims;
    bufferDims.reserve(dims.size());
    for (Value dim : dims)
      bufferDims.push_back(getEnclosingLoopStepBound(dim, anchor));
    return bufferDims;
  };

  auto freeTempVecinTensors =
      [&](OpBuilder &b, Location loc,
          ArrayRef<std::pair<Value, Value>> tempVecinTensors) {
    for (auto [queue, tensor] : tempVecinTensors)
      b.create<TQueBindFreeTensorOp>(loc, queue, tensor);
  };

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
        auto mrt = cast<MemRefType>(inMemref.getType());
        for (unsigned d = 0; d < iterRank; ++d)
          iterDimSizes[d] = getDynDim(builder, loc, inMemref, d);
        break;
      }
    }
    // Fall back: fill remaining parallel dims from output (output only covers
    // parallel dims, so only use it when the iterator type is parallel).
    {
      unsigned outDim = 0;
      for (unsigned d = 0; d < iterRank; ++d) {
        if (!iterDimSizes[d] && iterTypes[d] == utils::IteratorType::parallel)
          iterDimSizes[d] = getDynDim(builder, loc, outMemref, outDim++);
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
    Value totalElems = computeProduct(builder, loc, fullShape);
    SmallVector<std::pair<Value, Value>> tempVecinTensors;

    // Build a VECCALC accumulator for the full shape.  This is the tensor
    // that will hold the element-wise intermediate results before reduction.
    Value accumLt = allocVeccalc(builder, loc, elemType, fullShape).second;

    // Zero-initialize the accumulator.  The linalg.generic outs operand
    // provides the initial accumulator value, which is 0.0 (set by the
    // linalg.fill that precedes this kernel in the pipeline).  The on-chip
    // VECCALC TBuf is uninitialized by default, so we must fill it here.
    Value zeroVal;
    if (elemType.isF16())
      zeroVal = builder.create<arith::ConstantOp>(
          loc, builder.getF16FloatAttr(0.0f));
    else if (elemType.isF32())
      zeroVal = builder.create<arith::ConstantOp>(
          loc, builder.getF32FloatAttr(0.0f));
    if (zeroVal) {
      auto zeroDup = builder.create<DuplicateL2Op>(loc, accumLt, zeroVal, totalElems);
      copyAscendCUnitAttr(genOp.getOperation(), zeroDup.getOperation());
    }

    // Promote each input to a local_tensor of shape `fullShape`.
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
                    getDynDim(builder, loc, inMemref, srcDimIdx++)));
          else
            srcShapeVals.push_back(
                builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
        }
        Value srcLt = readTensor(builder, loc, inMemref);
        auto [bcastTbuf, bcastLt] =
            allocVeccalc(builder, loc, elemType, fullShape);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, bcastLt, srcLt,
            dstShapeVals, srcShapeVals,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
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
          srcDims.push_back(getDynDim(builder, loc, inMemref, d));
        Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
        for (Value d : srcDims)
          srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
        Value srcGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(elemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                       /*size=*/Value{});
        Value srcLt =
            copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
                          srcElemCount, &tempVecinTensors);
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
            allocVeccalc(builder, loc, elemType, fullShape);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, bcastLt, srcLt,
            dstShapeVals, srcShapeVals,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
        inputLts[i] = bcastLt;
      } else if (inMs == 0 /*GM*/) {
        // GM input at full rank: copy via VECIN TQue (simulator requires
        // DataCopy to go through TQue, not directly to VECCALC TBuf).
        Value srcGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(elemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                       /*size=*/Value{});
        inputLts[i] =
            copyGmToVecin(builder, loc, elemType, srcGt, totalElems,
                          totalElems, &tempVecinTensors);
      } else {
        // Already VECIN or VECCALC — use readTensor as-is.
        inputLts[i] = readTensor(builder, loc, inMemref);
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
              allocVeccalc(builder, loc, elemType, fullShape);
          auto dupOp = builder.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), totalElems);
          copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
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
            allocVeccalc(builder, loc, elemType, fullShape);
        valToLt[result] = tmpLt;
        return tmpLt;
      };

      if (auto addOp = dyn_cast<arith::AddFOp>(bodyOp)) {
        Value lhs = resolve(addOp.getLhs());
        Value rhs = resolve(addOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = chooseDst(addOp.getResult());
        auto addL2Op = builder.create<AddL2Op>(loc, dst, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), addL2Op.getOperation());
        if (dst == accumLt) valToLt[addOp.getResult()] = accumLt;
      } else if (auto mulOp = dyn_cast<arith::MulFOp>(bodyOp)) {
        Value lhs = resolve(mulOp.getLhs());
        Value rhs = resolve(mulOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = chooseDst(mulOp.getResult());
        auto mulOp2 = builder.create<MulL2Op>(loc, dst, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), mulOp2.getOperation());
        if (dst == accumLt) valToLt[mulOp.getResult()] = accumLt;
      } else if (auto maxOp = dyn_cast<arith::MaximumFOp>(bodyOp)) {
        Value lhs = resolve(maxOp.getLhs());
        Value rhs = resolve(maxOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = chooseDst(maxOp.getResult());
        auto maxOp2 = builder.create<MaxL2Op>(loc, dst, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), maxOp2.getOperation());
        if (dst == accumLt) valToLt[maxOp.getResult()] = accumLt;
      }
      // Other arith ops can be added here as needed.
    }

    // ------------------------------------------------------------------
    // Step 3: Reduce the accumulated VECCALC to the output VECOUT tensor.
    //
    // The VECCALC accumulator's physical layout follows the iteration
    // order (the GM operand is data_copy'd verbatim).  When the reduction
    // iterator is the innermost non-unit operand dim the buffer is
    // [A_rows, R_cols] → AscendC reduces the contiguous (R) axis →
    // ReduceLayout::AR.  When a parallel (non-unit) operand dim comes
    // *after* the reduction dim — e.g. out[d0,d2] = sum_{d1} x[d0,d1,d2]
    // once d0 is sliced to 1 — the buffer is [R_rows, A_cols] and we need
    // ReduceLayout::RA (result[a] = sum_r src[r*A + a]).
    // ------------------------------------------------------------------
    ReduceLayout layout = ReduceLayout::AR;
    {
      int redDim = -1;
      for (unsigned d = 0; d < iterRank; ++d)
        if (iterTypes[d] == utils::IteratorType::reduction) {
          redDim = (int)d;
          break;
        }
      for (unsigned i = 0; i < numInputs && layout == ReduceLayout::AR; ++i) {
        AffineMap m = maps[i];
        if (m.getNumResults() != iterRank)
          continue; // not a full-rank operand
        auto mrt =
            cast<MemRefType>(genOp.getDpsInputOperand(i)->get().getType());
        for (unsigned rp = 0; rp < m.getNumResults(); ++rp) {
          auto de = dyn_cast<AffineDimExpr>(m.getResult(rp));
          if (!de || (int)de.getPosition() != redDim)
            continue;
          for (unsigned d = rp + 1; d < mrt.getRank(); ++d)
            if (mrt.getDimSize(d) != 1) {
              layout = ReduceLayout::RA;
              break;
            }
          break;
        }
      }
    }
    Value vecoutLt = writeTensor(builder, loc, outMemref);
    auto layoutAttr = ReduceLayoutAttr::get(mlirCtx, layout);
    auto reduceOp = builder.create<ReduceSum2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                    /*sharedTmpBuffer=*/Value{});
    copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());

    // Enqueue vecout if it has a queue (VECOUT path).
    if (Value q = ctx.getQueue(outMemref))
      builder.create<TQueBindEnqueTensorOp>(loc, q, vecoutLt);

    freeTempVecinTensors(builder, loc, tempVecinTensors);
    genOp.erase();
  }

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
  // Concat semantics are implicitly handled: the VECOUT→GM copy op (inserted by
  // AscendCBufferPlacement + DataMoveConversion) targets a memref subview of the
  // output buffer with the correct byte offset, so Op1 and Op2 results land at
  // the right positions in the concatenated output without any asc.concat op.
  SmallVector<linalg::GenericOp> parallelGenericOps;
  funcOp.walk([&](linalg::GenericOp op) {
    auto iterTypes = op.getIteratorTypesArray();
    bool allParallel = llvm::all_of(iterTypes, [](utils::IteratorType t) {
      return t == utils::IteratorType::parallel;
    });
    if (allParallel && op.getNumDpsInits() == 1)
      parallelGenericOps.push_back(op);
  });

  // Helper: detect a standalone transpose generic (any rank).
  // Pattern: 1 input with a non-identity permutation map, 1 output with identity
  // map, body is a single linalg.yield of the input block argument (no computation).
  auto isTransposeGeneric = [](linalg::GenericOp op) -> bool {
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
      return false;
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 2)
      return false;
    AffineMap inMap  = maps[0];
    AffineMap outMap = maps[1];
    unsigned rank    = op.getIteratorTypesArray().size();
    if (rank == 0)
      return false;
    // Output must be identity
    if (!outMap.isIdentity())
      return false;
    // Input must have same rank as iteration space (no broadcast)
    if (inMap.getNumResults() != rank)
      return false;
    // All input map results must be distinct AffineDimExprs (no constants, no complex exprs)
    SmallVector<int64_t> perm(rank, -1);
    for (unsigned r = 0; r < rank; ++r) {
      auto dimExpr = dyn_cast<AffineDimExpr>(inMap.getResult(r));
      if (!dimExpr)
        return false;
      int64_t pos = static_cast<int64_t>(dimExpr.getPosition());
      if (pos < 0 || pos >= static_cast<int64_t>(rank))
        return false;
      perm[r] = pos;
    }
    // All positions must be distinct (no repeated dim in permutation)
    llvm::SmallDenseSet<int64_t> seen;
    for (unsigned r = 0; r < rank; ++r)
      if (!seen.insert(perm[r]).second)
        return false;
    // Must be a non-identity permutation
    bool isIdentityPerm = true;
    for (unsigned r = 0; r < rank; ++r)
      if (perm[r] != static_cast<int64_t>(r)) { isIdentityPerm = false; break; }
    if (isIdentityPerm)
      return false;
    // Body must be yield-only (single linalg.yield yielding the input block arg)
    Block &body = *op.getBody();
    if (body.getOperations().size() != 1)
      return false;
    auto yieldOp = dyn_cast<linalg::YieldOp>(&body.front());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
      return false;
    auto ba = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
    return ba && ba.getArgNumber() == 0;
  };

  // Helper: detect index_select gather (column gather).
  // Stamped by --mark-structured-ops: {gather_dim = N : i64} attribute.
  auto isIndexSelectGeneric = [](linalg::GenericOp op) -> bool {
    return op->hasAttr("gather_dim");
  };

  // Helper: detect embedding gather (row gather).
  // Stamped by --mark-structured-ops: {embedding_dim = N : i64} attribute.
  auto isEmbeddingGeneric = [](linalg::GenericOp op) -> bool {
    return op->hasAttr("embedding_dim");
  };
  (void)isEmbeddingGeneric; // reserved for future use

  for (linalg::GenericOp genOp : parallelGenericOps) {
    Value outMemref = genOp.getDpsInitOperand(0)->get();
    int64_t outMs   = getMemorySpace(outMemref.getType());
    if (outMs <= 0)
      continue; // output must be on-chip (VECOUT or VECCALC)

    // ---- Index-select gather: emit gather_l2 row by row ----
    // New pattern: {gather_dim = 1} attribute, 1 input (indices),
    // data accessed via memref.load in the body (captures a GM memref).
    // For each row i in 0..Tb_M:
    //   copy data row from GM → VECCALC
    //   gather_l2(dst_row[K], src_row[N], indices[K], 0, K)
    if (isIndexSelectGeneric(genOp)) {
      Value indicesMemref = genOp.getDpsInputOperand(0)->get();
      Location loc = genOp.getLoc();
      builder.setInsertionPoint(genOp);

      // Find data memref from memref.load in the body (bufferized tensor.extract).
      Value dataMemref;
      genOp.getBody()->walk([&](memref::LoadOp loadOp) {
        if (!dataMemref)
          dataMemref = loadOp.getMemref();
      });
      if (!dataMemref) {
        LLVM_DEBUG(llvm::dbgs()
                   << "isIndexSelectGeneric: no memref.load found in body\n");
        continue;
      }

      // Find pre-gather op: parallel generic whose output memref == dataMemref
      // (i.e., the op that wrote the data we're gathering from)
      linalg::GenericOp preOp;
      for (linalg::GenericOp candidate : parallelGenericOps) {
        if (candidate == genOp) continue;
        if (candidate->hasAttr("gather_dim") || candidate->hasAttr("embedding_dim")) continue;
        if (candidate.getDpsInitOperand(0)->get() == dataMemref) {
          preOp = candidate;
          break;
        }
      }

      // Find post-gather op: parallel generic that has outMemref as one of its inputs
      linalg::GenericOp postOp;
      for (linalg::GenericOp candidate : parallelGenericOps) {
        if (candidate == genOp) continue;
        if (candidate->hasAttr("gather_dim") || candidate->hasAttr("embedding_dim")) continue;
        for (OpOperand *inp : candidate.getDpsInputOperands()) {
          if (inp->get() == outMemref) {
            postOp = candidate;
            break;
          }
        }
        if (postOp) break;
      }

      auto outMrt  = cast<MemRefType>(outMemref.getType());
      Type elemType = outMrt.getElementType();
      Type i32Type  = builder.getI32Type();

      // Tb_M = dim[0] of output, N = dim[1] of data, K = dim[1] of output
      Value tbM  = getDynDim(builder, loc, outMemref, 0);
      Value dimN = getDynDim(builder, loc, dataMemref, 1);
      Value dimK = getDynDim(builder, loc, outMemref, 1);

      unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;

      // Get indices as a local_tensor.
      // If indices are in VECIN (ms=9), deque from queue.
      // If indices are in GM (ms=0), copy into VECCALC first.
      auto idxMrt = cast<MemRefType>(indicesMemref.getType());
      Type idxElemType = idxMrt.getElementType();
      Value idxCount = getDynDim(builder, loc, indicesMemref, 0);
      Value indicesLt;
      int64_t idxMs = getMemorySpace(indicesMemref.getType());
      if (idxMs == 9 /*VECIN*/ || idxMs == 11 /*VECCALC*/) {
        indicesLt = readTensor(builder, loc, indicesMemref);
      } else {
        // GM: copy indices into a fresh VECCALC buffer.
        SmallVector<Value> idxDims = {idxCount};
        auto [idxTbuf, idxLt] =
            allocVeccalc(builder, loc, idxElemType, idxDims);
        Value idxGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(idxElemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, idxGt, indicesMemref,
                                                       /*size=*/Value{});
        builder.create<DataCopyL2Op>(loc, idxLt, idxGt, idxCount);
        indicesLt = idxLt;
      }

      Value dimK_i32 =
          builder.create<arith::IndexCastOp>(loc, i32Type, dimK);
      Value srcBaseAddr =
          builder.create<arith::ConstantIntOp>(loc, i32Type, 0);

      // Alloc the VECOUT output tensor.
      Value dstLt = writeTensor(builder, loc, outMemref);

      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value one  = builder.create<arith::ConstantIndexOp>(loc, 1);
      // Element offset for a single output row.
      Value outElemsPerRow = dimK;

      // Collect enclosing scf.for induction variables to compute the global
      // row offset into the data memref.  The generic sits inside nested
      // tiling loops whose IVs sum to the tile origin in the data tensor.
      SmallVector<Value> enclosingIVs;
      for (Operation *p = genOp->getParentOp(); p; p = p->getParentOp())
        if (auto f = dyn_cast<scf::ForOp>(p))
          enclosingIVs.push_back(f.getInductionVar());

      // Data is in GM: set up a GlobalTensor for row-by-row copy.
      Value dataGt = builder.create<GlobalTensorOp>(
          loc, GlobalTensorType::get(elemType));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, dataGt, dataMemref,
                                                     /*size=*/Value{});

      // Allocate a VECCALC buffer for one data row plus one 32B datablock of
      // padding.  AscendC Gather offsets are byte offsets into UB; on hardware
      // the vector instruction may touch the tail datablock even when the
      // logical element is within [0, N).  Padding keeps max-index gathers from
      // reading past the registered UB buffer.
      unsigned gatherPadElems = std::max<unsigned>(1, 32 / elemBytes);
      Value gatherPadElemsVal =
          builder.create<arith::ConstantIndexOp>(loc, gatherPadElems);
      Value paddedDimN =
          builder.create<arith::AddIOp>(loc, dimN, gatherPadElemsVal);
      Value dataRowQueue = builder.create<QueueOp>(
          loc, QueueType::get(mlirCtx, TPosition::VECIN, 1));
      Value rowBytes = builder.create<arith::MulIOp>(
          loc, paddedDimN,
          builder.create<arith::ConstantIndexOp>(loc, elemBytes));
      Value dataRowQueueDepth =
          builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
      builder.create<TPipeInitQueueOp>(loc, ctx.pipe, dataRowQueue,
                                       dataRowQueueDepth, rowBytes);

      builder.create<scf::ForOp>(
          loc, zero, tbM, one, ValueRange{},
          [&](OpBuilder &b, Location forLoc, Value rowIdx, ValueRange) {
            // Global row = sum(enclosing IVs) + rowIdx (tile-local row)
            Value globalRow = rowIdx;
            for (Value iv : enclosingIVs)
              globalRow = b.create<arith::AddIOp>(forLoc, globalRow, iv);

            // Step 1: Copy one data row from GM → VECCALC.
            // Offset the GlobalTensor by globalRow * N elements, then DataCopy.
            Value rowElemOff =
                b.create<arith::MulIOp>(forLoc, globalRow, dimN);
            Value dataRowGt = b.create<GlobalTensorBracketOp>(
                forLoc, GlobalTensorType::get(elemType), dataGt,
                rowElemOff);
            Value dataRowAllocLt = b.create<TQueBindAllocTensorOp>(
                forLoc, LocalTensorType::get(elemType), dataRowQueue);
            b.create<DataCopyL2Op>(forLoc, dataRowAllocLt, dataRowGt, dimN);
            b.create<TQueBindEnqueTensorOp>(forLoc, dataRowQueue,
                                            dataRowAllocLt);
            Value dataRowLt = b.create<TQueBindDequeTensorOp>(
                forLoc, LocalTensorType::get(elemType), dataRowQueue);

            // Step 1b: If pre-op exists (e.g. relu), apply it on dataRowLt
            Value processedRowLt = dataRowLt;
            if (preOp) {
              Value procLt =
                  allocVeccalc(b, forLoc, elemType, SmallVector<Value>{dimN})
                      .second;
              Value dimN_i32 = b.create<arith::IndexCastOp>(forLoc, b.getI32Type(), dimN);

              Block &preBody = *preOp.getBody();
              llvm::SmallDenseMap<Value, Value> preValToLt;

              auto preResolve = [&](Value v) -> Value {
                if (auto ba = dyn_cast<BlockArgument>(v)) {
                  if (ba.getArgNumber() == 0) return dataRowLt;
                  return procLt;
                }
                auto it = preValToLt.find(v);
                if (it != preValToLt.end()) return it->second;
                if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                  auto [dupTbuf2, dupLt] = allocVeccalc(b, forLoc, elemType,
                                                         SmallVector<Value>{dimN});
                  auto dupOp2 = b.create<DuplicateL2Op>(forLoc, dupLt, constOp.getResult(), dimN_i32);
                  copyAscendCUnitAttr(preOp.getOperation(), dupOp2.getOperation());
                  preValToLt[v] = dupLt;
                  return dupLt;
                }
                return Value{};
              };

              for (auto &bodyOp : preBody.without_terminator()) {
                if (auto maxOp = dyn_cast<arith::MaximumFOp>(bodyOp)) {
                  Value lhs = preResolve(maxOp.getLhs()), rhs = preResolve(maxOp.getRhs());
                  if (lhs && rhs) {
                    auto maxOp2 = b.create<MaxL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                    copyAscendCUnitAttr(preOp.getOperation(), maxOp2.getOperation());
                    preValToLt[maxOp.getResult()] = procLt;
                  }
                } else if (auto addOp2 = dyn_cast<arith::AddFOp>(bodyOp)) {
                  Value lhs = preResolve(addOp2.getLhs()), rhs = preResolve(addOp2.getRhs());
                  if (lhs && rhs) {
                    auto addOp3 = b.create<AddL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                    copyAscendCUnitAttr(preOp.getOperation(), addOp3.getOperation());
                    preValToLt[addOp2.getResult()] = procLt;
                  }
                } else if (auto mulOp2 = dyn_cast<arith::MulFOp>(bodyOp)) {
                  Value lhs = preResolve(mulOp2.getLhs()), rhs = preResolve(mulOp2.getRhs());
                  if (lhs && rhs) {
                    auto mulOp3 = b.create<MulL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                    copyAscendCUnitAttr(preOp.getOperation(), mulOp3.getOperation());
                    preValToLt[mulOp2.getResult()] = procLt;
                  }
                }
              }
              processedRowLt = procLt;
            }

            // Step 2: gather_l2(dst[K], src[N], indices, srcBase=0, count=K)
            Value dstElemOff =
                b.create<arith::MulIOp>(forLoc, rowIdx, outElemsPerRow);
            Value dstRowLt = b.create<LocalTensorSubIndexOp>(
                forLoc, LocalTensorType::get(elemType), dstLt, dstElemOff);
            Value gatheredRowLt =
                allocVeccalc(b, forLoc, elemType, SmallVector<Value>{dimK})
                    .second;
            b.create<GatherL2Op>(forLoc, gatheredRowLt, processedRowLt,
                                 indicesLt, srcBaseAddr, dimK_i32);

            // Step 3: If post-op exists (e.g. add bias), apply it on gatheredRowLt.
            // If no post-op, walk the gather body itself for any arith ops that
            // appear after the memref.load (from upstream fusion). This handles
            // the case where relu + add were fused into the gather body by
            // --fuse-gather-elementwise before bufferization.
            if (postOp) {
              Value dimK_i32v = b.create<arith::IndexCastOp>(forLoc, b.getI32Type(), dimK);
              Block &postBody = *postOp.getBody();
              llvm::SmallDenseMap<Value, Value> postValToLt;

              auto postResolve = [&](Value v) -> Value {
                if (auto ba = dyn_cast<BlockArgument>(v)) {
                  unsigned argNum = ba.getArgNumber();
                  unsigned numIns = (unsigned)postOp.getNumDpsInputs();
                  if (argNum >= numIns) return gatheredRowLt; // output init arg
                  Value argMemref = postOp.getDpsInputOperand(argNum)->get();
                  if (argMemref == outMemref) return gatheredRowLt;
                  // Other inputs (bias, etc.) — read their tensor
                  return readTensor(b, forLoc, argMemref);
                }
                auto it = postValToLt.find(v);
                if (it != postValToLt.end()) return it->second;
                if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                  auto [dupTbuf3, dupLt] = allocVeccalc(b, forLoc, elemType,
                                                         SmallVector<Value>{dimK});
                  auto dupOp3 = b.create<DuplicateL2Op>(forLoc, dupLt, constOp.getResult(), dimK_i32v);
                  copyAscendCUnitAttr(postOp.getOperation(), dupOp3.getOperation());
                  postValToLt[v] = dupLt;
                  return dupLt;
                }
                return Value{};
              };

              for (auto &bodyOp : postBody.without_terminator()) {
                if (auto addOp3 = dyn_cast<arith::AddFOp>(bodyOp)) {
                  Value lhs = postResolve(addOp3.getLhs()), rhs = postResolve(addOp3.getRhs());
                  if (lhs && rhs) {
                    auto addOp4 = b.create<AddL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(postOp.getOperation(), addOp4.getOperation());
                    postValToLt[addOp3.getResult()] = gatheredRowLt;
                  }
                } else if (auto mulOp3 = dyn_cast<arith::MulFOp>(bodyOp)) {
                  Value lhs = postResolve(mulOp3.getLhs()), rhs = postResolve(mulOp3.getRhs());
                  if (lhs && rhs) {
                    auto mulOp4 = b.create<MulL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(postOp.getOperation(), mulOp4.getOperation());
                    postValToLt[mulOp3.getResult()] = gatheredRowLt;
                  }
                } else if (auto maxOp3 = dyn_cast<arith::MaximumFOp>(bodyOp)) {
                  Value lhs = postResolve(maxOp3.getLhs()), rhs = postResolve(maxOp3.getRhs());
                  if (lhs && rhs) {
                    auto maxOp4 = b.create<MaxL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(postOp.getOperation(), maxOp4.getOperation());
                    postValToLt[maxOp3.getResult()] = gatheredRowLt;
                  }
                }
              }
            } else {
              // Walk the fused gather body for arith ops that appear after
              // the memref.load (these were inlined by --fuse-gather-elementwise).
              // Block args:
              //   arg0 = indices element (i64, skip)
              //   arg1..argN-2 = extra ins (bias etc.)
              //   argN-1 = out init (skip, use gatheredRowLt instead)
              Value dimK_i32v = b.create<arith::IndexCastOp>(forLoc, b.getI32Type(), dimK);
              Block &gatherBody = *genOp.getBody();
              unsigned numBodyIns = (unsigned)genOp.getNumDpsInputs();
              llvm::SmallDenseMap<Value, Value> bodyValToLt;

              // Helper: find the memref.load result in the body.
              Value loadResult;
              for (auto &op : gatherBody.without_terminator()) {
                if (isa<memref::LoadOp>(op)) {
                  loadResult = op.getResult(0);
                  break;
                }
              }

              SmallVector<std::pair<Value, Value>> bodyTempQueueTensors;

              auto bodyResolve = [&](Value v) -> Value {
                // The "gathered row" value — the memref.load result maps to
                // gatheredRowLt (post-gather result).
                if (loadResult && v == loadResult) return gatheredRowLt;
                auto it = bodyValToLt.find(v);
                if (it != bodyValToLt.end()) return it->second;
                if (auto ba = dyn_cast<BlockArgument>(v)) {
                  unsigned argNum = ba.getArgNumber();
                  if (argNum == 0) return Value{}; // indices arg, skip
                  if (argNum >= numBodyIns) return gatheredRowLt; // out init
                  // Extra ins (bias, etc.) at argNum=1..numBodyIns-1
                  Value argMemref = genOp.getDpsInputOperand(argNum)->get();
                  int64_t argMs = getMemorySpace(argMemref.getType());
                  if (argMs > 0) {
                    // On-chip: use readTensor directly.
                    Value lt = readTensor(b, forLoc, argMemref);
                    bodyValToLt[v] = lt;
                    return lt;
                  }
                  // GM: copy through a VECIN queue so the vector op observes
                  // a synchronized local tensor on real hardware.
                  Value argCount = getDynDim(b, forLoc, argMemref, 0);
                  auto argMrt = cast<MemRefType>(argMemref.getType());
                  Type argElem = argMrt.getElementType();
                  Value argQueue = b.create<QueueOp>(
                      forLoc, QueueType::get(mlirCtx, TPosition::VECIN, 1));
                  unsigned argElemBytes =
                      argElem.getIntOrFloatBitWidth() / 8;
                  Value argBytes = b.create<arith::MulIOp>(
                      forLoc, argCount,
                      b.create<arith::ConstantIndexOp>(forLoc, argElemBytes));
                  Value depth = b.create<arith::ConstantOp>(
                      forLoc, b.getI32IntegerAttr(1));
                  b.create<TPipeInitQueueOp>(forLoc, ctx.pipe, argQueue, depth,
                                             argBytes);
                  Value argAllocLt = b.create<TQueBindAllocTensorOp>(
                      forLoc, LocalTensorType::get(argElem), argQueue);
                  Value argGt = b.create<GlobalTensorOp>(forLoc, GlobalTensorType::get(argElem));
                  b.create<GlobalTensorSetGlobalBufferOp>(forLoc, argGt, argMemref,
                                                           /*size=*/Value{});
                  b.create<DataCopyL2Op>(forLoc, argAllocLt, argGt, argCount);
                  b.create<TQueBindEnqueTensorOp>(forLoc, argQueue, argAllocLt);
                  Value argLt = b.create<TQueBindDequeTensorOp>(
                      forLoc, LocalTensorType::get(argElem), argQueue);
                  bodyTempQueueTensors.push_back({argQueue, argLt});
                  bodyValToLt[v] = argLt;
                  return argLt;
                }
                if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                  auto [dupTbuf4, dupLt] = allocVeccalc(b, forLoc, elemType,
                                                         SmallVector<Value>{dimK});
                  auto dupOp4 = b.create<DuplicateL2Op>(forLoc, dupLt, constOp.getResult(), dimK_i32v);
                  copyAscendCUnitAttr(genOp.getOperation(), dupOp4.getOperation());
                  bodyValToLt[v] = dupLt;
                  return dupLt;
                }
                return Value{};
              };

              bool pastLoad = false;
              for (auto &op : gatherBody.without_terminator()) {
                if (isa<memref::LoadOp>(op)) {
                  pastLoad = true;
                  continue;
                }
                if (!pastLoad) continue;
                if (auto addOp4 = dyn_cast<arith::AddFOp>(op)) {
                  Value lhs = bodyResolve(addOp4.getLhs()),
                        rhs = bodyResolve(addOp4.getRhs());
                  if (lhs && rhs) {
                    auto addOp5 = b.create<AddL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(genOp.getOperation(), addOp5.getOperation());
                    bodyValToLt[addOp4.getResult()] = gatheredRowLt;
                  }
                } else if (auto maxOp4 = dyn_cast<arith::MaximumFOp>(op)) {
                  Value lhs = bodyResolve(maxOp4.getLhs()),
                        rhs = bodyResolve(maxOp4.getRhs());
                  if (lhs && rhs) {
                    auto maxOp5 = b.create<MaxL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(genOp.getOperation(), maxOp5.getOperation());
                    bodyValToLt[maxOp4.getResult()] = gatheredRowLt;
                  }
                } else if (auto mulOp4 = dyn_cast<arith::MulFOp>(op)) {
                  Value lhs = bodyResolve(mulOp4.getLhs()),
                        rhs = bodyResolve(mulOp4.getRhs());
                  if (lhs && rhs) {
                    auto mulOp5 = b.create<MulL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(genOp.getOperation(), mulOp5.getOperation());
                    bodyValToLt[mulOp4.getResult()] = gatheredRowLt;
                  }
                }
              }
              for (auto [queue, tensor] : bodyTempQueueTensors)
                b.create<TQueBindFreeTensorOp>(forLoc, queue, tensor);
            }

            Value zeroVal;
            if (elemType.isF16())
              zeroVal = b.create<arith::ConstantOp>(
                  forLoc, b.getF16FloatAttr(0.0f));
            else if (elemType.isF32())
              zeroVal = b.create<arith::ConstantOp>(
                  forLoc, b.getF32FloatAttr(0.0f));
            if (zeroVal) {
              Value zeroLt =
                  allocVeccalc(b, forLoc, elemType, SmallVector<Value>{dimK})
                      .second;
              auto dupOp =
                  b.create<DuplicateL2Op>(forLoc, zeroLt, zeroVal, dimK_i32);
              copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
              auto copyOp = b.create<AddL2Op>(
                  forLoc, dstRowLt, gatheredRowLt, zeroLt, dimK_i32);
              copyAscendCUnitAttr(genOp.getOperation(), copyOp.getOperation());
            }

            b.create<TQueBindFreeTensorOp>(forLoc, dataRowQueue, dataRowLt);
            b.create<scf::YieldOp>(forLoc);
          });

      if (Value q = ctx.getQueue(outMemref))
        builder.create<TQueBindEnqueTensorOp>(loc, q, dstLt);

      if (postOp) postOp.erase();
      if (preOp) preOp.erase();
      genOp.erase();
      continue;
    }

    // ---- Transpose generic: emit ascendc.transpose ----
    if (isTransposeGeneric(genOp)) {
      Value inMemref = genOp.getDpsInputOperand(0)->get();
      Location loc = genOp.getLoc();
      builder.setInsertionPoint(genOp);

      Value srcLt = readTensor(builder, loc, inMemref);
      Value dstLt = writeTensor(builder, loc, outMemref);
      builder.create<TransposeOp>(loc, dstLt, srcLt);

      if (Value q = ctx.getQueue(outMemref))
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
          iterDimSizes[d] = getDynDim(builder, loc, inMemref, d);
        break;
      }
    }
    // Fall back: fill remaining dims from output (all parallel, same rank).
    for (unsigned d = 0; d < iterRank; ++d)
      if (!iterDimSizes[d])
        iterDimSizes[d] = getDynDim(builder, loc, outMemref, d);

    // totalElems is the actual element count for this tile.  Buffer
    // allocation uses the enclosing loop-step upper bound so tail iterations
    // reuse one max-sized queue/tbuf instead of repeatedly InitBuffer-ing.
    Value totalElems = computeProduct(builder, loc, iterDimSizes);
    SmallVector<Value> bufferDimSizes =
        getBufferDimSizes(iterDimSizes, genOp.getOperation());
    Value bufferTotalElems = computeProduct(builder, loc, bufferDimSizes);
    SmallVector<std::pair<Value, Value>> tempVecinTensors;

    Value outQueue = ctx.getQueue(outMemref);
    Value accumLt;
    if (!outQueue) {
      // Allocate the shared VECCALC accumulator for intermediate results.
      auto [accumTbuf, veccalcAccumLt] =
          allocVeccalc(builder, loc, elemType, bufferDimSizes);
      accumLt = veccalcAccumLt;
    }

    // ---- Step 1: Promote each input to a VECCALC local_tensor ----
    SmallVector<Value> inputLts(numInputs);
    for (unsigned i = 0; i < numInputs; ++i) {
      Value inMemref = genOp.getDpsInputOperand(i)->get();
      AffineMap inMap = maps[i];
      int64_t inMs    = getMemorySpace(inMemref.getType());
      IndexingMapAnalysis analysis = analyzeIndexingMap(inMap, iterRank);

      switch (analysis.kind) {
      case IndexingMapAnalysis::Kind::Identity: {
        if (inMs == 0 /*GM*/) {
          // GM input at full rank: copy via VECIN TQue.
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          inputLts[i] =
              copyGmToVecin(builder, loc, elemType, srcGt, totalElems,
                            bufferTotalElems, &tempVecinTensors);
        } else {
          inputLts[i] = readTensor(builder, loc, inMemref);
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
                  getDynDim(builder, loc, inMemref, srcDimIdx++)));
            else
              srcShapeVals.push_back(
                  builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
          }
          Value srcLt = readTensor(builder, loc, inMemref);
          auto [bcastTbuf, bcastLt] =
              allocVeccalc(builder, loc, elemType, bufferDimSizes);
          auto bcastOp = builder.create<BroadcastL2Op>(
              loc, bcastLt, srcLt,
              dstShapeVals, srcShapeVals,
              builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
          copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
          inputLts[i] = bcastLt;
        } else {
          // broadcast from GM: copy via VECIN TQue first, then broadcast_l2.
          SmallVector<Value> srcDims;
          for (unsigned d = 0; d < srcRank; ++d)
            srcDims.push_back(getDynDim(builder, loc, inMemref, d));
          Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (Value d : srcDims)
            srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          Value srcLt =
              copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
                            srcElemCount, &tempVecinTensors);
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
              allocVeccalc(builder, loc, elemType, bufferDimSizes);
          auto bcastOp = builder.create<BroadcastL2Op>(
              loc, bcastLt, srcLt,
              dstShapeVals, srcShapeVals,
              builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
          copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
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
            copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
                          srcElemCount, &tempVecinTensors);

        auto [transpTbuf, transpLt] =
            allocVeccalc(builder, loc, elemType, bufferDimSizes);
        auto transposeOp = builder.create<TransposeOp>(loc, transpLt, srcVecinLt);
        copyAscendCUnitAttr(genOp.getOperation(), transposeOp.getOperation());
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
          srcDimsVals.push_back(getDynDim(builder, loc, inMemref, d));

        Value srcVecinLt;
        if (inMs == 9 /*VECIN*/) {
          srcVecinLt = readTensor(builder, loc, inMemref);
        } else {
          Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (Value d : srcDimsVals)
            srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          srcVecinLt =
              copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
                            srcElemCount, &tempVecinTensors);
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
            allocVeccalc(builder, loc, elemType, bufferDimSizes);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, finalLt, srcVecinLt,
            bcastDstShape, bcastSrcShape,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
        inputLts[i] = finalLt;
        break;
      }
      } // end switch
    }

    if (outQueue)
      accumLt = allocTensor(builder, loc, outQueue, elemType);

    // ---- Step 2: Walk body and inline arith ops onto VECCALC tensors ----
    Block &bodyBlock = *genOp.getBody();
    unsigned numBodyArgs = bodyBlock.getNumArguments();
    SmallVector<Value> argToLt(numBodyArgs);
    for (unsigned i = 0; i < numInputs; ++i)
      argToLt[i] = inputLts[i];
    argToLt[numInputs] = accumLt;

    llvm::SmallDenseMap<Value, Value> valToLt;
    for (auto &bodyOp : bodyBlock.without_terminator()) {
      auto resolve = [&](Value v) -> Value {
        if (auto ba = dyn_cast<BlockArgument>(v))
          return argToLt[ba.getArgNumber()];
        auto it = valToLt.find(v);
        if (it != valToLt.end()) return it->second;
        // Scalar constant? Fill a fresh VECCALC with duplicate_l2.
        if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
          auto [dupTbuf, dupLt] =
              allocVeccalc(builder, loc, elemType, bufferDimSizes);
          auto dupOp = builder.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), totalElems);
          copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
          valToLt[v] = dupLt;
          return dupLt;
        }
        return Value{};
      };

      if (auto addOp = dyn_cast<arith::AddFOp>(bodyOp)) {
        Value lhs = resolve(addOp.getLhs());
        Value rhs = resolve(addOp.getRhs());
        if (!lhs || !rhs) continue;
        auto addL2Op =
            builder.create<AddL2Op>(loc, accumLt, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), addL2Op.getOperation());
        valToLt[addOp.getResult()] = accumLt;
      } else if (auto mulOp = dyn_cast<arith::MulFOp>(bodyOp)) {
        Value lhs = resolve(mulOp.getLhs());
        Value rhs = resolve(mulOp.getRhs());
        if (!lhs || !rhs) continue;
        auto mulL2Op =
            builder.create<MulL2Op>(loc, accumLt, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), mulL2Op.getOperation());
        valToLt[mulOp.getResult()] = accumLt;
      } else if (auto maxOp = dyn_cast<arith::MaximumFOp>(bodyOp)) {
        Value lhs = resolve(maxOp.getLhs());
        Value rhs = resolve(maxOp.getRhs());
        if (!lhs || !rhs) continue;
        auto maxL2Op =
            builder.create<MaxL2Op>(loc, accumLt, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(), maxL2Op.getOperation());
        valToLt[maxOp.getResult()] = accumLt;
      }
    }

    // ---- Step 3: Write accumulator to output buffer ----
    // No reduction needed (all-parallel). The compute result is in accumLt
    // (a VECCALC tbuf). We need to deliver it to the output buffer:
    //
    //   VECOUT (ms=10): alloc from queue, use AddL2 to copy accumLt→vecoutLt
    //                   (add_l2(dst, src, zero_tbuf, count) would need a zero
    //                    tensor; instead use the queue alloc tensor directly and
    //                    simply enqueue accumLt if the queue accepts VECCALC).
    //                   Simplest: treat the VECCALC accumLt as the enqueue source
    //                   and let the downstream DataMoveConversion handle writeback.
    //   VECCALC (ms=11): accumLt already holds the result; no copy needed.
    //
    // Key insight: the epilogue memref.copy (VECOUT→GM) inserted by
    // AscendCBufferPlacement is converted by DataMoveConversion into a
    // data_copy_l2 with the correct subview offset, so the Concat position
    // is preserved automatically. We just need to enqueue the result tensor.
    if (outQueue) {
      // The queue expects a tensor allocated from the same queue.  Real
      // hardware is stricter than the simulator here; enqueueing a VECCALC
      // tbuf tensor into a VECOUT queue can surface as UB/MTE faults.
      builder.create<TQueBindEnqueTensorOp>(loc, outQueue, accumLt);
    }
    freeTempVecinTensors(builder, loc, tempVecinTensors);
    // If outMemref has no queue (VECCALC alloc without a queue), the result
    // already resides in the VECCALC tbuf and will be consumed by the next op.

    genOp.erase();
  }

  // --- linalg.matmul → mmad ---
  SmallVector<linalg::MatmulOp> matmulOps;
  funcOp.walk([&](linalg::MatmulOp op) { matmulOps.push_back(op); });

  for (linalg::MatmulOp matmulOp : matmulOps) {
    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];
    if (getMemorySpace(A.getType()) != 2) continue;
    if (getMemorySpace(B.getType()) != 4) continue;
    if (getMemorySpace(C.getType()) != 7) continue;

    Value qA = ctx.getQueue(A), qB = ctx.getQueue(B), qC = ctx.getQueue(C);
    if (!qA || !qB || !qC) {
      matmulOp.emitError("missing queue for matmul A2/B2/CO1 buffer");
      return failure();
    }

    Location loc = matmulOp.getLoc();
    builder.setInsertionPoint(matmulOp);
    Type elemTypeA = cast<MemRefType>(A.getType()).getElementType();
    Type elemTypeC = cast<MemRefType>(C.getType()).getElementType();

    Value tensorA = dequeTensor(builder, loc, qA, elemTypeA);
    Value tensorB = dequeTensor(builder, loc, qB, elemTypeA);

    // CO1 accumulates across the K-loop: alloc before the enclosing for-loop,
    // enque after it, so the queue slot is held for all K iterations.
    // CO1 uses its own element type (f32 for half-precision matmul accumulation).
    auto [tensorC, cHoistFor] =
        allocHoisted(matmulOp, qC, elemTypeC, loc);

    // Build MmadParams with runtime m/n/k values.
    // A: [m x k], B: [k x n]
    auto toI16 = [&](Value idx) -> Value {
      return builder.create<arith::IndexCastOp>(loc, builder.getI16Type(), idx);
    };
    auto getDim = [&](Value mem, int64_t d) -> Value {
      auto mrt = cast<MemRefType>(mem.getType());
      if (!ShapedType::isDynamic(mrt.getShape()[d]))
        return builder.create<arith::ConstantIndexOp>(loc, mrt.getShape()[d]);
      return builder.create<memref::DimOp>(loc, mem, d);
    };

    Value mVal = toI16(getDim(A, 0)); // A rows = m
    Value kVal = toI16(getDim(A, 1)); // A cols = k
    Value nVal = toI16(getDim(B, 1)); // B cols = n

    // unit_flag / fm_offset / filter_offset default to 0 (i8)
    Value zero8 =
        builder.create<arith::ConstantIntOp>(loc, builder.getI8Type(), 0);

    SmallVector<Value> mmadOperands = {mVal, nVal, kVal, zero8, zero8, zero8};
    // MmadParams fields are uint16_t/uint8_t — use Unsigned IntegerType so
    // the CodeEmitter emits static_cast<uint16_t> rather than <int16_t>.
    auto ui16 = IntegerType::get(mlirCtx, 16, IntegerType::Unsigned);
    auto ui8  = IntegerType::get(mlirCtx, 8,  IntegerType::Unsigned);
    SmallVector<Type> mmadTypes = {ui16, ui16, ui16, ui8, ui8, ui8};
    Value mmadParams = builder.create<ConstructOp>(
        loc, MmadParamsType::get(mlirCtx), mmadOperands,
        builder.getTypeArrayAttr(mmadTypes));
    auto mmadOp = builder.create<MmadOp>(loc, tensorC, tensorA, tensorB, mmadParams);
    copyAscendCUnitAttr(matmulOp.getOperation(), mmadOp.getOperation());

    if (cHoistFor) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(cHoistFor);
      builder.create<TQueBindEnqueTensorOp>(cHoistFor.getLoc(), qC, tensorC);
    } else {
      builder.create<TQueBindEnqueTensorOp>(loc, qC, tensorC);
    }
    builder.create<TQueBindFreeTensorOp>(loc, qA, tensorA);
    builder.create<TQueBindFreeTensorOp>(loc, qB, tensorB);
    matmulOp.erase();
  }

  // --- linalg.elementwise (add / max_signed) ---
  SmallVector<linalg::ElementwiseOp> ewOps;
  funcOp.walk([&](linalg::ElementwiseOp op) { ewOps.push_back(op); });

  for (linalg::ElementwiseOp ewOp : ewOps) {
    auto kind = ewOp.getKind();
    if (kind != linalg::ElementwiseKind::add &&
        kind != linalg::ElementwiseKind::mul &&
        kind != linalg::ElementwiseKind::max_signed)
      continue;

    Value src0 = ewOp.getInputs()[0];
    Value src1 = ewOp.getInputs()[1];
    Value dst = ewOp.getOutputs()[0];
    if (getMemorySpace(dst.getType()) <= 0) continue;

    Location loc = ewOp.getLoc();
    builder.setInsertionPoint(ewOp);

    Value localSrc0 = readTensor(builder, loc, src0);
    Value localSrc1 = readTensor(builder, loc, src1);

    // For VECOUT (ms=10) where dst is a subview of the whole VECOUT alloc:
    //   - Hoist alloc_tensor for the full VECOUT buffer before the enclosing
    //     for-loop and enque it after, so it remains live for one enque/deque
    //     cycle spanning all Tb_M/Tb_N iterations.
    //   - Use tbuf.get_with_offset to write each sub-tile at the correct byte
    //     offset into the underlying tbuf (the alloc_tensor and the tbuf share
    //     the same on-chip memory region).
    // For other outputs (e.g. VECCALC ms=11): writeTensor handles it directly.
    Value localDst;    // tensor used for the enque (may be null for subview)
    Value writeTarget; // tensor actually passed to the compute op
    scf::ForOp dstHoistFor = nullptr;
    int64_t dstMs = getMemorySpace(dst.getType());
    bool isDstSubview = dst.getDefiningOp<memref::SubViewOp>() != nullptr;
    if (dstMs == 10 && ctx.getQueue(dst)) {
      Value q = ctx.getQueue(dst);
      auto mrt = cast<MemRefType>(dst.getType());
      auto [t, f] = allocHoisted(ewOp, q, mrt.getElementType(), loc);
      localDst = t;
      dstHoistFor = f;
      // If dst is a subview, write through an offset slice of the tbuf rather
      // than to the start of the alloc_tensor.
      if (isDstSubview) {
        if (Value byteOff = subviewByteOffset(builder, loc, dst)) {
          Value sizeBytes = computeByteCount(builder, loc, dst);
          writeTarget = tbufSlice(builder, loc, dst, sizeBytes, byteOff);
        }
      }
      if (!writeTarget)
        writeTarget = localDst;
    } else {
      localDst = writeTensor(builder, loc, dst);
      writeTarget = localDst;
    }

    Value count = computeElementCount(builder, loc, dst);

    if (kind == linalg::ElementwiseKind::add) {
      auto addOp =
          builder.create<AddL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      copyAscendCUnitAttr(ewOp.getOperation(), addOp.getOperation());
    } else if (kind == linalg::ElementwiseKind::mul) {
      auto mulOp =
          builder.create<MulL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      copyAscendCUnitAttr(ewOp.getOperation(), mulOp.getOperation());
    } else {
      auto maxOp =
          builder.create<MaxL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      copyAscendCUnitAttr(ewOp.getOperation(), maxOp.getOperation());
    }

    if (Value q = ctx.getQueue(dst)) {
      if (dstHoistFor) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(dstHoistFor);
        builder.create<TQueBindEnqueTensorOp>(dstHoistFor.getLoc(), q,
                                              localDst);
      } else {
        builder.create<TQueBindEnqueTensorOp>(loc, q, localDst);
      }
    }
    // Only free a src tensor if it was freshly dequeued (not a live tensor
    // that is being reused across loop iterations and freed elsewhere).
    if (Value q = ctx.getQueue(src0))
      if (!ctx.getLiveTensor(src0))
        builder.create<TQueBindFreeTensorOp>(loc, q, localSrc0);
    if (Value q = ctx.getQueue(src1))
      if (!ctx.getLiveTensor(src1))
        builder.create<TQueBindFreeTensorOp>(loc, q, localSrc1);

    ewOp.erase();
  }

  // --- linalg.fill → duplicate_l2 ---
  //
  // Erased cases (no AscendC op emitted):
  //   ms=0  (GM):    fill initializes a GM accumulator that is fully overwritten
  //                  by subsequent data_copy from on-chip; redundant after
  //                  linalg.generic→reduce_sum_2d_l2 lowering.
  //   ms=7  (CO1):   mmad hardware zeroes CO1 automatically (cmatrixInitVal=false
  //                  default), so a separate duplicate_l2 is redundant and would
  //                  also cause a double-alloc on the CO1 queue.
  //   ms=10 (VECOUT): max_l2 writes the output directly; a prior fill(0) is
  //                   redundant and causes a double-alloc on the VECOUT queue.
  SmallVector<linalg::FillOp> fillOps;
  funcOp.walk([&](linalg::FillOp op) { fillOps.push_back(op); });

  for (linalg::FillOp fillOp : fillOps) {
    Value dst = fillOp.getOutputs()[0];
    int64_t ms = getMemorySpace(dst.getType());
    if (ms <= 0) {
      // GM fill: only erase if the destination is an alloc within the function
      // (accumulator pattern — fully overwritten by data_copy after reduce).
      // Fills into function-argument GM memrefs are NOT converted and must be
      // left in place.
      bool dstIsAlloc = dst.getDefiningOp<memref::AllocOp>() != nullptr;
      if (!dstIsAlloc)
        continue; // preserve fill on func arg
      fillOp.erase();
      continue;
    }

    // Skip fills that would cause a double-alloc or are otherwise redundant.
    if (ms == 7 || ms == 10) {
      fillOp.erase();
      continue;
    }

    Location loc = fillOp.getLoc();
    builder.setInsertionPoint(fillOp);

    Value localDst = writeTensor(builder, loc, dst);
    Value count = computeElementCount(builder, loc, dst);
    auto dupOp =
        builder.create<DuplicateL2Op>(loc, localDst, fillOp.getInputs()[0], count);
    copyAscendCUnitAttr(fillOp.getOperation(), dupOp.getOperation());

    if (Value q = ctx.getQueue(dst))
      builder.create<TQueBindEnqueTensorOp>(loc, q, localDst);

    fillOp.erase();
  }

  return success();
}

} // namespace afir
} // namespace mlir
