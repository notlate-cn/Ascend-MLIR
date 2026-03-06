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
  // Skipped cases:
  //   ms=7 (CO1): mmad hardware zeroes CO1 automatically (cmatrixInitVal=false
  //               default), so a separate duplicate_l2 is redundant and would
  //               also cause a double-alloc on the CO1 queue.
  //   ms=10 (VECOUT): max_l2 writes the output directly; a prior fill(0) is
  //                   redundant and causes a double-alloc on the VECOUT queue.
  SmallVector<linalg::FillOp> fillOps;
  funcOp.walk([&](linalg::FillOp op) { fillOps.push_back(op); });

  for (linalg::FillOp fillOp : fillOps) {
    Value dst = fillOp.getOutputs()[0];
    int64_t ms = getMemorySpace(dst.getType());
    if (ms <= 0) continue;

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
