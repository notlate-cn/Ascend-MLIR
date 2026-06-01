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

#include "Conversion/LinalgToAscendC/ComputeConversionContext.h"
#include "Conversion/LinalgToAscendC/ComputeConversionHelpers.h"
#include "Conversion/LinalgToAscendC/LinalgToAscendCUtils.h"

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
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include <algorithm>

#define DEBUG_TYPE "linalg-to-ascendc-compute"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace afir {

LogicalResult convertCompute(func::FuncOp funcOp, AscendCBufferContext &ctx) {
  MLIRContext *mlirCtx = funcOp.getContext();
  OpBuilder builder(mlirCtx);
  ComputeCtx cc{builder, ctx, mlirCtx};

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

  // ------------------------------------------------------------------
  // Fill PRE-pass: any linalg.fill writing a VECCALC (ms==11) buffer is an
  // accumulator initializer (e.g. the loop-carried accumulator in the RBLOCK
  // reduction-split path).  Lower it to Duplicate + register the resulting
  // local_tensor as the live tensor for that alloc, so later steps
  // (writeTensor, the reduce step, the trailing acc→GM copy) reuse it instead
  // of re-allocating / failing to resolve the on-chip source.
  {
    SmallVector<linalg::FillOp> preFills;
    funcOp.walk([&](linalg::FillOp op) {
      if (getMemorySpace(op.getOutputs()[0].getType()) == 11)
        preFills.push_back(op);
    });
    for (linalg::FillOp fillOp : preFills) {
      Value dst = fillOp.getOutputs()[0];
      // Skip if some earlier pass already registered a live tensor.
      if (ctx.getLiveTensor(dst))
        continue;
      Location loc = fillOp.getLoc();
      builder.setInsertionPoint(fillOp);
      Type elemTy = cast<MemRefType>(dst.getType()).getElementType();
      Value localDst;
      if (Value tbuf = ctx.getTBuf(dst))
        localDst = builder.create<TBufGetTensorOp>(
            loc, LocalTensorType::get(elemTy), tbuf, /*len=*/Value{});
      else
        localDst = cc.writeTensor(builder, loc, dst);
      Value count = computeElementCount(builder, loc, dst);
      auto dupOp = builder.create<DuplicateL2Op>(loc, localDst,
                                                 fillOp.getInputs()[0], count);
      copyAscendCUnitAttr(fillOp.getOperation(), dupOp.getOperation());
      ctx.setLiveTensor(dst, localDst);
      fillOp.erase();
    }
  }

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
    memref::SubViewOp fullRankSv;
    for (unsigned i = 0; i < numInputs; ++i) {
      Value inMemref = genOp.getDpsInputOperand(i)->get();
      AffineMap inMap = maps[i];
      if (inMap.getNumResults() == iterRank) {
        // Full map — use this operand to fill iterDimSizes.
        auto mrt = cast<MemRefType>(inMemref.getType());
        for (unsigned d = 0; d < iterRank; ++d)
          iterDimSizes[d] = cc.getDynDim(builder, loc, inMemref, d);
        if (inMap.isIdentity())
          fullRankSv = inMemref.getDefiningOp<memref::SubViewOp>();
        break;
      }
    }
    // Fall back: fill remaining parallel dims from output (output only covers
    // parallel dims, so only use it when the iterator type is parallel).
    {
      unsigned outDim = 0;
      for (unsigned d = 0; d < iterRank; ++d) {
        if (!iterDimSizes[d] && iterTypes[d] == utils::IteratorType::parallel)
          iterDimSizes[d] = cc.getDynDim(builder, loc, outMemref, outDim++);
      }
    }

    // ------------------------------------------------------------------
    // Hoisting: when this reduce generic sits inside the RBLOCK reduction-
    // split scf.for (input is a 2-D strided subview x[a_tile, r_chunk] whose
    // size operands are kernel args), allocate all its UB scratch ONCE at
    // function entry rather than once per loop trip — re-running
    // InitBuffer/InitQueue ~R/RBLOCK_0 times overruns the UB pool.  Detect
    // exactly that case (so the existing few-trip reduce kernels are left
    // byte-for-byte unchanged).
    // ------------------------------------------------------------------
    Block &funcEntry = funcOp.getBody().front();
    OpBuilder entryBuilder(mlirCtx);
    entryBuilder.setInsertionPointAfter(ctx.pipe.getDefiningOp());
    Value strideRowsEntry, strideColsEntry, strideRowStrideEntry;
    bool hoist = false;
    if (iterRank == 2 && fullRankSv &&
        genOp->getParentOfType<scf::ForOp>() != nullptr) {
      auto svTy = cast<MemRefType>(fullRankSv.getType());
      if (auto sl = dyn_cast<StridedLayoutAttr>(svTy.getLayout())) {
        int64_t s0 = sl.getStrides()[0], s1 = sl.getStrides()[1];
        int64_t shape1 = svTy.getShape()[1];
        if (s1 == 1 && !ShapedType::isDynamic(s0) &&
            (ShapedType::isDynamic(shape1) || s0 != shape1)) {
          // Both subview sizes must be kernel args (defined at function entry).
          auto svSizes = fullRankSv.getMixedSizes();
          Value v0, v1;
          if (auto a = dyn_cast<Attribute>(svSizes[0]))
            v0 = entryBuilder.create<arith::ConstantIndexOp>(
                loc, cast<IntegerAttr>(a).getInt());
          else if (auto ba = dyn_cast<BlockArgument>(cast<Value>(svSizes[0]));
                   ba && ba.getOwner() == &funcEntry)
            v0 = cast<Value>(svSizes[0]);
          if (auto a = dyn_cast<Attribute>(svSizes[1]))
            v1 = entryBuilder.create<arith::ConstantIndexOp>(
                loc, cast<IntegerAttr>(a).getInt());
          else if (auto ba = dyn_cast<BlockArgument>(cast<Value>(svSizes[1]));
                   ba && ba.getOwner() == &funcEntry)
            v1 = cast<Value>(svSizes[1]);
          if (v0 && v1) {
            hoist = true;
            strideRowsEntry = v0;
            strideColsEntry = v1;
            strideRowStrideEntry =
                entryBuilder.create<arith::ConstantIndexOp>(loc, s0);
          }
        }
      }
    }
    // Builder used for all *allocations* (TBuf/InitBuffer/Queue/InitQueue) — at
    // function entry when hoisting, else at the generic (in the loop, as before).
    OpBuilder &initB = hoist ? entryBuilder : builder;
    // Queue tensors that must be FreeTensor'd after use (only when the queue is
    // hoisted out of the loop and therefore shared across trips).
    SmallVector<std::pair<Value, Value>> vecinFrees;

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
    // When hoisting, all allocation sizes / counts use this loop-invariant
    // value (built at function entry) instead of the per-trip `totalElems`.
    // For the reduction-split shape iterDimSizes = [a_tile(dim0), r_chunk(dim1)]
    // so totalElems == strideRowsEntry * strideColsEntry.
    Value totalElemsEntry = totalElems;
    SmallVector<Value> allocShape = fullShape;
    if (hoist) {
      totalElemsEntry = entryBuilder.create<arith::MulIOp>(
          loc, strideRowsEntry, strideColsEntry);
      allocShape = {totalElemsEntry};
    }
    Value count = hoist ? totalElemsEntry : totalElems;

    // Build a VECCALC accumulator for the full shape.  This is the tensor
    // that will hold the element-wise intermediate results before reduction.
    // NOTE: plain Value (not a structured binding) so the chooseDst lambda
    // below can capture it; .first (tbuf) is unused here.
    Value accumLt = cc.allocVeccalc(initB, loc, elemType, allocShape).second;

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
      auto zeroDup = builder.create<DuplicateL2Op>(loc, accumLt, zeroVal, count);
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
                    cc.getDynDim(builder, loc, inMemref, srcDimIdx++)));
          else
            srcShapeVals.push_back(
                builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
        }
        Value srcLt = cc.readTensor(builder, loc, inMemref);
        auto [bcastTbuf, bcastLt] =
            cc.allocVeccalc(builder, loc, elemType, fullShape);
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
          srcDims.push_back(cc.getDynDim(builder, loc, inMemref, d));
        Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
        for (Value d : srcDims)
          srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
        Value srcGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(elemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                       /*size=*/Value{});
        Value srcLt =
            cc.copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
                          /*bufferElemCount=*/Value{}, &vecinFrees);
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
            cc.allocVeccalc(builder, loc, elemType, fullShape);
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
        // If the operand is a rank-2 subview whose row stride is a static
        // value that doesn't (provably) match its column extent — i.e. the
        // rows aren't packed — a flat DataCopy of rows*cols elements would
        // read the wrong memory.  This is the [XBLOCK_SUB x RBLOCK_0] chunk
        // of x[A,R] in the RBLOCK reduction-split path.  Use a strided copy.
        bool strided = false;
        if (auto mrt = dyn_cast<MemRefType>(inMemref.getType()))
          if (mrt.getRank() == 2)
            if (auto sl = dyn_cast<StridedLayoutAttr>(mrt.getLayout())) {
              int64_t s0 = sl.getStrides()[0], s1 = sl.getStrides()[1];
              int64_t shape1 = mrt.getShape()[1];
              strided = s1 == 1 && !ShapedType::isDynamic(s0) &&
                        (ShapedType::isDynamic(shape1) || s0 != shape1);
            }
        if (strided) {
          // Sizes/stride used for the (hoisted) buffer alloc must be loop-
          // invariant — use the function-entry copies when hoisting.
          Value rows = hoist ? strideRowsEntry
                             : cc.getDynDim(builder, loc, inMemref, 0);
          Value cols = hoist ? strideColsEntry
                             : cc.getDynDim(builder, loc, inMemref, 1);
          Value rowStride =
              hoist ? strideRowStrideEntry
                    : initB.create<arith::ConstantIndexOp>(
                          loc, cast<StridedLayoutAttr>(
                                   cast<MemRefType>(inMemref.getType())
                                       .getLayout())
                                   .getStrides()[0]);
          auto [deq, q] = cc.copyGmToVecinStrided(initB, builder, loc, elemType,
                                               srcGt, rows, cols, rowStride);
          inputLts[i] = deq;
          // The per-trip AllocTensor comes from a depth-1 VECIN queue, so it
          // MUST be freed after use regardless of whether InitBuffer/InitQueue
          // were hoisted — otherwise the next loop trip's AllocTensor dead-locks
          // on the unfreed buffer (camodel sim hangs indefinitely; real HW too).
          // This is the multi-r-noncontig / full-reduce peeled-outer-R path,
          // where `hoist` is false.  emitVecinFrees() emits the free in-loop
          // after the reduce consumes the tile.
          vecinFrees.push_back({q, deq});
        } else {
          inputLts[i] =
              cc.copyGmToVecin(builder, loc, elemType, srcGt, totalElems,
                            /*bufferElemCount=*/Value{}, &vecinFrees);
        }
      } else {
        // Already VECIN or VECCALC — use readTensor as-is.
        inputLts[i] = cc.readTensor(builder, loc, inMemref);
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
    // TBufs written by a vector op in this body carry no queue (EnQue/DeQue)
    // sync, so a later vector op reading one has a vector->vector RAW that the
    // hardware does not auto-order across multi-repeat tiles: it races on the
    // real NPU while the simulator hides it (combo-elewise-reduce-e2e produced
    // ~0 on device for the `acc += (a+b)` chain — the accumulate add read its
    // TBuf inputs before the preceding add's write had landed).  Track such
    // TBufs and emit a PipeBarrier before any vector op that reads one.  Mirrors
    // the TBuf-accumulator->GM barrier in DataMoveConversion: a TBuf has no
    // queue, so its write->read order must be made explicit.
    llvm::SmallDenseSet<Value> vectorWrittenTBufs;
    if (zeroVal)
      vectorWrittenTBufs.insert(accumLt); // written by the zero-init Duplicate
    auto syncTBufRAW = [&](Value lhs, Value rhs) {
      if (vectorWrittenTBufs.contains(lhs) || vectorWrittenTBufs.contains(rhs))
        builder.create<PipeBarrierOp>(loc, Pipe::PIPE_ALL);
    };
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
          auto [dupTbuf, dupLt] = cc.allocVeccalc(initB, loc, elemType, allocShape);
          auto dupOp = builder.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), count);
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
        auto [tmpTbuf, tmpLt] = cc.allocVeccalc(initB, loc, elemType, allocShape);
        valToLt[result] = tmpLt;
        return tmpLt;
      };

      if (auto addOp = dyn_cast<arith::AddFOp>(bodyOp)) {
        Value lhs = resolve(addOp.getLhs());
        Value rhs = resolve(addOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = chooseDst(addOp.getResult());
        syncTBufRAW(lhs, rhs);
        auto addL2Op = builder.create<AddL2Op>(loc, dst, lhs, rhs, count);
        copyAscendCUnitAttr(genOp.getOperation(), addL2Op.getOperation());
        vectorWrittenTBufs.insert(dst);
        if (dst == accumLt) valToLt[addOp.getResult()] = accumLt;
      } else if (auto mulOp = dyn_cast<arith::MulFOp>(bodyOp)) {
        Value lhs = resolve(mulOp.getLhs());
        Value rhs = resolve(mulOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = chooseDst(mulOp.getResult());
        syncTBufRAW(lhs, rhs);
        auto mulOp2 = builder.create<MulL2Op>(loc, dst, lhs, rhs, count);
        copyAscendCUnitAttr(genOp.getOperation(), mulOp2.getOperation());
        vectorWrittenTBufs.insert(dst);
        if (dst == accumLt) valToLt[mulOp.getResult()] = accumLt;
      } else if (auto maxOp = dyn_cast<arith::MaximumFOp>(bodyOp)) {
        Value lhs = resolve(maxOp.getLhs());
        Value rhs = resolve(maxOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = chooseDst(maxOp.getResult());
        syncTBufRAW(lhs, rhs);
        auto maxOp2 = builder.create<MaxL2Op>(loc, dst, lhs, rhs, count);
        copyAscendCUnitAttr(genOp.getOperation(), maxOp2.getOperation());
        vectorWrittenTBufs.insert(dst);
        if (dst == accumLt) valToLt[maxOp.getResult()] = accumLt;
      } else if (auto minOp = dyn_cast<arith::MinimumFOp>(bodyOp)) {
        Value lhs = resolve(minOp.getLhs());
        Value rhs = resolve(minOp.getRhs());
        if (!lhs || !rhs) continue;
        Value dst = chooseDst(minOp.getResult());
        syncTBufRAW(lhs, rhs);
        auto minOp2 = builder.create<MinL2Op>(loc, dst, lhs, rhs, count);
        copyAscendCUnitAttr(genOp.getOperation(), minOp2.getOperation());
        vectorWrittenTBufs.insert(dst);
        if (dst == accumLt) valToLt[minOp.getResult()] = accumLt;
      } else if (!isa<arith::ConstantOp>(bodyOp)) {
        // Fail loudly rather than silently emitting a kernel that drops this op.
        genOp.emitError("LinalgToAscendC: unsupported op in linalg.generic "
                        "body: ")
            << bodyOp.getName();
        return failure();
      }
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
    Value vecoutLt = cc.writeTensor(builder, loc, outMemref);
    auto layoutAttr = ReduceLayoutAttr::get(mlirCtx, layout);

    // When hoisting (RBLOCK reduction-split), give reduce_sum_2d_l2 a
    // pre-allocated scratch tensor so CannTranslation doesn't InitBuffer one
    // inside the loop.  Size it generously: rows*cols + 64 elements covers the
    // AR layout's [8-elem result slot | cols-elem ReduceSum workspace] and the
    // RA layout's input-sized workspace.
    Value reduceScratch;
    if (hoist && layout == ReduceLayout::AR) {
      Value scratchElems = entryBuilder.create<arith::AddIOp>(
          loc, totalElemsEntry,
          entryBuilder.create<arith::ConstantIndexOp>(loc, 64));
      reduceScratch =
          cc.allocVeccalc(entryBuilder, loc, elemType, {scratchElems}).second;
    }

    auto emitVecinFrees = [&] {
      for (auto &qlt : vecinFrees)
        builder.create<TQueBindFreeTensorOp>(loc, qlt.first, qlt.second);
    };

    // Detect the RBLOCK reduction-split accumulator: outMemref is a non-init
    // iter_arg of an scf.for (the RBLOCK loop).  Each RBLOCK iteration reduces
    // its chunk into a temporary, then accumulates: acc += reduce(chunk).
    bool isAccumulating = false;
    if (auto ba = dyn_cast<BlockArgument>(outMemref))
      isAccumulating = ba.getArgNumber() > 0 &&
                       isa<scf::ForOp>(ba.getOwner()->getParentOp());

    if (isAccumulating) {
      SmallVector<Value> outDyn;
      if (hoist) {
        outDyn.push_back(strideRowsEntry);
      } else {
        for (unsigned d = 0;
             d < cast<MemRefType>(outMemref.getType()).getRank(); ++d)
          outDyn.push_back(cc.getDynDim(builder, loc, outMemref, d));
      }
      auto [tmpTbuf, tmpLt] = cc.allocVeccalc(initB, loc, elemType, outDyn);
      auto reduceOp = builder.create<ReduceSum2DL2Op>(
          loc, tmpLt, accumLt, layoutAttr, /*sharedTmpBuffer=*/reduceScratch);
      copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
      Value cnt = hoist ? strideRowsEntry
                        : computeElementCount(builder, loc, outMemref);
      auto addOp = builder.create<AddL2Op>(loc, vecoutLt, vecoutLt, tmpLt, cnt);
      copyAscendCUnitAttr(genOp.getOperation(), addOp.getOperation());
      emitVecinFrees();
      genOp.erase();
      continue;
    }

    auto reduceOp = builder.create<ReduceSum2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                    /*sharedTmpBuffer=*/Value{});
    copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());

    // Register the reduce result as the live tensor for the output memref so
    // downstream group members (e.g. an elementwise op that broadcasts s[a]
    // back over the reduced axis) read it via getLiveTensor instead of treating
    // the intra-group VECCALC alloc as a GM source.  Mirrors isTransposeGeneric.
    if (outMs == 11 /*VECCALC*/ && !ctx.getLiveTensor(outMemref))
      ctx.setLiveTensor(outMemref, vecoutLt);

    // Enqueue vecout if it has a queue (VECOUT path).
    if (Value q = ctx.getQueue(outMemref))
      builder.create<TQueBindEnqueTensorOp>(loc, q, vecoutLt);

    emitVecinFrees();
    genOp.erase();
  }

  if (failed(convertParallelGenerics(funcOp, cc)))
    return failure();

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

    Value tensorA = cc.dequeTensor(builder, loc, qA, elemTypeA);
    Value tensorB = cc.dequeTensor(builder, loc, qB, elemTypeA);

    // CO1 accumulates across the K-loop: alloc before the enclosing for-loop,
    // enque after it, so the queue slot is held for all K iterations.
    // CO1 uses its own element type (f32 for half-precision matmul accumulation).
    auto [tensorC, cHoistFor] =
        cc.allocHoisted(matmulOp, qC, elemTypeC, loc);

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

    Value localSrc0 = cc.readTensor(builder, loc, src0);
    Value localSrc1 = cc.readTensor(builder, loc, src1);

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
      auto [t, f] = cc.allocHoisted(ewOp, q, mrt.getElementType(), loc);
      localDst = t;
      dstHoistFor = f;
      // If dst is a subview, write through an offset slice of the tbuf rather
      // than to the start of the alloc_tensor.
      if (isDstSubview) {
        if (Value byteOff = cc.subviewByteOffset(builder, loc, dst)) {
          Value sizeBytes = computeByteCount(builder, loc, dst);
          writeTarget = cc.tbufSlice(builder, loc, dst, sizeBytes, byteOff);
        }
      }
      if (!writeTarget)
        writeTarget = localDst;
    } else {
      localDst = cc.writeTensor(builder, loc, dst);
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
    // ms=7 (CO1) mmad zeroes automatically.  ms=10 (VECOUT) is only redundant
    // when another op overwrites the same buffer (max_l2 precursor); when the
    // fill IS the producer (standalone fill→output), emit Duplicate so its 0
    // write isn't dropped (multi-output buffers are now shared/serialized).
    if (ms == 7) {
      fillOp.erase();
      continue;
    }

    Location loc = fillOp.getLoc();
    builder.setInsertionPoint(fillOp);

    Value localDst = cc.writeTensor(builder, loc, dst);
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
