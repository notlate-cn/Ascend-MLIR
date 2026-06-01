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

LogicalResult convertFillPrePass(func::FuncOp funcOp, ComputeCtx &cc) {
  [[maybe_unused]] OpBuilder &builder = cc.builder;
  [[maybe_unused]] AscendCBufferContext &ctx = cc.ctx;
  [[maybe_unused]] MLIRContext *mlirCtx = cc.mlirCtx;

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

  return success();
}

LogicalResult convertMatmuls(func::FuncOp funcOp, ComputeCtx &cc) {
  [[maybe_unused]] OpBuilder &builder = cc.builder;
  [[maybe_unused]] AscendCBufferContext &ctx = cc.ctx;
  [[maybe_unused]] MLIRContext *mlirCtx = cc.mlirCtx;

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

  return success();
}

LogicalResult convertElementwise(func::FuncOp funcOp, ComputeCtx &cc) {
  [[maybe_unused]] OpBuilder &builder = cc.builder;
  [[maybe_unused]] AscendCBufferContext &ctx = cc.ctx;
  [[maybe_unused]] MLIRContext *mlirCtx = cc.mlirCtx;

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

  return success();
}

LogicalResult convertFills(func::FuncOp funcOp, ComputeCtx &cc) {
  [[maybe_unused]] OpBuilder &builder = cc.builder;
  [[maybe_unused]] AscendCBufferContext &ctx = cc.ctx;
  [[maybe_unused]] MLIRContext *mlirCtx = cc.mlirCtx;

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
