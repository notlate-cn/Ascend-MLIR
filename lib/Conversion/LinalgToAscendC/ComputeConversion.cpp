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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#define DEBUG_TYPE "linalg-to-ascendc-compute"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace afir {

LogicalResult convertCompute(func::FuncOp funcOp, AscendCBufferContext &ctx) {
  MLIRContext *mlirCtx = funcOp.getContext();
  OpBuilder builder(mlirCtx);

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

  // Helper: get a runtime Value for dimension `dim` of a memref.
  auto getDynDim = [&](OpBuilder &b, Location loc, Value memref,
                        unsigned dim) -> Value {
    auto mrt = cast<MemRefType>(memref.getType());
    if (!ShapedType::isDynamic(mrt.getShape()[dim]))
      return b.create<arith::ConstantIndexOp>(loc, mrt.getShape()[dim]);
    return b.create<memref::DimOp>(loc, memref, dim);
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
    Value totalElems;
    for (Value s : fullShape)
      totalElems = totalElems ? builder.create<arith::MulIOp>(loc, totalElems, s) : s;
    if (!totalElems)
      totalElems = builder.create<arith::ConstantIndexOp>(loc, 1);

    // Build a VECCALC accumulator for the full shape.  This is the tensor
    // that will hold the element-wise intermediate results before reduction.
    auto [accumTbuf, accumLt] =
        allocVeccalc(builder, loc, elemType, fullShape);

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
        builder.create<BroadcastL2Op>(
            loc, bcastLt, srcLt,
            dstShapeVals, srcShapeVals,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        inputLts[i] = bcastLt;
      } else if (inMs == 0 /*GM*/) {
        // data_copy_l2: GM subview → fresh VECCALC.
        auto [copyTbuf, copyLt] =
            allocVeccalc(builder, loc, elemType, fullShape);
        Value srcGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(elemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                       /*size=*/Value{});
        builder.create<DataCopyL2Op>(loc, copyLt, srcGt, totalElems);
        inputLts[i] = copyLt;
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
    // We reuse accumLt as the dst for all intermediate ops.
    llvm::SmallDenseMap<Value, Value> valToLt;
    for (auto &bodyOp : bodyBlock.without_terminator()) {
      // Resolve an SSA value to its corresponding local_tensor.
      auto resolve = [&](Value v) -> Value {
        // Block argument?
        if (auto ba = dyn_cast<BlockArgument>(v))
          return argToLt[ba.getArgNumber()];
        // Result of a previous body op?
        auto it = valToLt.find(v);
        if (it != valToLt.end()) return it->second;
        return Value{};
      };

      if (auto addOp = dyn_cast<arith::AddFOp>(bodyOp)) {
        Value lhs = resolve(addOp.getLhs());
        Value rhs = resolve(addOp.getRhs());
        if (!lhs || !rhs) continue;
        builder.create<AddL2Op>(loc, accumLt, lhs, rhs, totalElems);
        valToLt[addOp.getResult()] = accumLt;
      } else if (auto maxOp = dyn_cast<arith::MaximumFOp>(bodyOp)) {
        Value lhs = resolve(maxOp.getLhs());
        Value rhs = resolve(maxOp.getRhs());
        if (!lhs || !rhs) continue;
        builder.create<MaxL2Op>(loc, accumLt, lhs, rhs, totalElems);
        valToLt[maxOp.getResult()] = accumLt;
      }
      // Other arith ops can be added here as needed.
    }

    // ------------------------------------------------------------------
    // Step 3: Reduce the accumulated VECCALC to the output VECOUT tensor.
    //
    // For a 2D iteration [parallel_dim, reduction_dim] with AR layout:
    //   reduce_sum_2d_l2(vecoutLt, accumLt, AR, no_tmp)
    // ------------------------------------------------------------------
    Value vecoutLt = writeTensor(builder, loc, outMemref);
    auto layoutAttr = ReduceLayoutAttr::get(mlirCtx, ReduceLayout::AR);
    builder.create<ReduceSum2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                     /*sharedTmpBuffer=*/Value{});

    // Enqueue vecout if it has a queue (VECOUT path).
    if (Value q = ctx.getQueue(outMemref))
      builder.create<TQueBindEnqueTensorOp>(loc, q, vecoutLt);

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
    Type elemType = cast<MemRefType>(A.getType()).getElementType();

    Value tensorA = dequeTensor(builder, loc, qA, elemType);
    Value tensorB = dequeTensor(builder, loc, qB, elemType);

    // CO1 accumulates across the K-loop: alloc before the enclosing for-loop,
    // enque after it, so the queue slot is held for all K iterations.
    auto [tensorC, cHoistFor] =
        allocHoisted(matmulOp, qC, elemType, loc);

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
    SmallVector<Type> mmadTypes = {builder.getI16Type(), builder.getI16Type(),
                                    builder.getI16Type(), builder.getI8Type(),
                                    builder.getI8Type(), builder.getI8Type()};
    Value mmadParams = builder.create<ConstructOp>(
        loc, MmadParamsType::get(mlirCtx), mmadOperands,
        builder.getTypeArrayAttr(mmadTypes));
    builder.create<MmadOp>(loc, tensorC, tensorA, tensorB, mmadParams);

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

    if (kind == linalg::ElementwiseKind::add)
      builder.create<AddL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
    else
      builder.create<MaxL2Op>(loc, writeTarget, localSrc0, localSrc1, count);

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
      // GM fill: fully overwritten by AscendC data_copy after reduce; erase.
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
    builder.create<DuplicateL2Op>(loc, localDst, fillOp.getInputs()[0], count);

    if (Value q = ctx.getQueue(dst))
      builder.create<TQueBindEnqueTensorOp>(loc, q, localDst);

    fillOp.erase();
  }

  return success();
}

} // namespace afir
} // namespace mlir
