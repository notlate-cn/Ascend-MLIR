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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#define DEBUG_TYPE "linalg-to-ascendc-datamove"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace afir {

// Helper: emit a runtime dim value for dimension d of a memref.
static Value emitDim(OpBuilder &b, Location loc, Value memref, int64_t d) {
  auto mrt = cast<MemRefType>(memref.getType());
  if (!ShapedType::isDynamic(mrt.getShape()[d]))
    return b.create<arith::ConstantIndexOp>(loc, mrt.getShape()[d]);
  return b.create<memref::DimOp>(loc, memref, d);
}

// Helper: cast an index value to i16 (signless, compatible with ui16 field).
static Value toI16(OpBuilder &b, Location loc, Value idx) {
  return b.create<arith::IndexCastOp>(loc, b.getI16Type(), idx);
}

// Helper: create an i16 constant (used for ui16 fields).
static Value constI16(OpBuilder &b, Location loc, int64_t v) {
  return b.create<arith::ConstantIntOp>(loc, b.getI16Type(), v);
}

// Helper: create an i8 constant (used for ui8 fields).
static Value constI8(OpBuilder &b, Location loc, int64_t v) {
  return b.create<arith::ConstantIntOp>(loc, b.getI8Type(), v);
}

// Helper: cast an index value to i8 via i64 truncation (for ui8 fields).
static Value toI8(OpBuilder &b, Location loc, Value idx) {
  Value i64Val = b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
  return b.create<arith::TruncIOp>(loc, b.getI8Type(), i64Val);
}

// Helper: create a i1 constant.
static Value constI1(OpBuilder &b, Location loc, bool v) {
  return b.create<arith::ConstantIntOp>(loc, b.getI1Type(), v ? 1 : 0);
}

// Helper: return the outermost enclosing scf::ForOp of `op` such that
// `allocOp` is NOT an ancestor of (i.e., lives outside) that for-loop.
// This is the loop at whose boundary the deque/free should be hoisted to
// match the enque level of `allocOp`.
// Returns nullptr if no such for-loop exists (allocOp and op are at the same
// loop level or allocOp is inside a loop that contains op).
static scf::ForOp getOutermostEnclosingForOutside(Operation *op,
                                                   Operation *allocOp) {
  scf::ForOp result = nullptr;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp)
      continue;
    // Keep this for-loop as a candidate if allocOp is outside it.
    if (!forOp->isProperAncestor(allocOp))
      result = forOp;
  }
  return result;
}

// Helper: find the root alloc of a memref value (walk through subviews).
static Value getRootAlloc(Value v) {
  while (auto subview = v.getDefiningOp<memref::SubViewOp>())
    v = subview.getSource();
  return v;
}

LogicalResult convertDataMove(func::FuncOp funcOp,
                               AscendCBufferContext &ctx) {
  MLIRContext *mlirCtx = funcOp.getContext();
  OpBuilder builder(mlirCtx);

  // Collect all copies first to avoid iterator invalidation.
  SmallVector<memref::CopyOp> copies;
  funcOp.walk([&](memref::CopyOp op) { copies.push_back(op); });

  for (memref::CopyOp copyOp : copies) {
    Value src = copyOp.getSource();
    Value dst = copyOp.getTarget();
    int64_t srcMs = getMemorySpace(src.getType());
    int64_t dstMs = getMemorySpace(dst.getType());
    Location loc = copyOp.getLoc();
    builder.setInsertionPoint(copyOp);

    // GM(0) → A1(1) / B1(3): alloc_tensor → data_copy_nd2nz → enque_tensor
    if (srcMs == 0 && (dstMs == 1 || dstMs == 3)) {
      Value dstQueue = ctx.getQueue(dst);
      if (!dstQueue) {
        copyOp.emitError("missing queue for A1/B1 buffer");
        return failure();
      }
      auto dstLtType =
          LocalTensorType::get(cast<MemRefType>(dst.getType()).getElementType());
      Value dstLt =
          builder.create<TQueBindAllocTensorOp>(loc, dstLtType, dstQueue);
      Value srcGt = builder.create<GlobalTensorOp>(
          loc,
          GlobalTensorType::get(cast<MemRefType>(src.getType()).getElementType()));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, src,
                                                     /*size=*/Value{});

      // Nd2NzParams for a 2D [height x width] tile:
      //   nd_num            = height (number of rows)
      //   n_value           = width / 16 (C0 = 16)
      //   d_value           = height
      //   src_nd_matrix_stride = width
      //   src_d_value       = height
      //   dst_nz_c0_stride  = 0
      //   dst_nz_n_stride   = height
      //   dst_nz_matrix_stride = 0
      Value height = emitDim(builder, loc, dst, 0);
      Value width  = emitDim(builder, loc, dst, 1);
      Value c0     = builder.create<arith::ConstantIndexOp>(loc, 16);
      Value nValue = builder.create<arith::DivUIOp>(loc, width, c0);

      SmallVector<Value> nd2nzOperands = {
          toI16(builder, loc, height),  // nd_num
          toI16(builder, loc, nValue),  // n_value
          toI16(builder, loc, height),  // d_value
          toI16(builder, loc, width),   // src_nd_matrix_stride
          toI16(builder, loc, height),  // src_d_value
          constI16(builder, loc, 0),    // dst_nz_c0_stride
          toI16(builder, loc, height),  // dst_nz_n_stride
          constI16(builder, loc, 0),    // dst_nz_matrix_stride
      };
      SmallVector<Type> nd2nzTypes(8, builder.getI16Type());
      Value params = builder.create<ConstructOp>(
          loc, Nd2NzParamsType::get(mlirCtx), nd2nzOperands,
          builder.getTypeArrayAttr(nd2nzTypes));
      builder.create<DataCopyNd2NzOp>(loc, dstLt, srcGt, params);
      builder.create<TQueBindEnqueTensorOp>(loc, dstQueue, dstLt);
      copyOp.erase();
      continue;
    }

    // GM(0) → VECIN(9): alloc_tensor → data_copy_l2 → enque_tensor
    //
    // The resulting VECIN local_tensor may be consumed (via subview) across
    // multiple inner loop iterations.  To avoid repeated deque/free pairs
    // (which would drain the queue after the first iteration), we immediately
    // deque the tensor right after enque and record it as a "live tensor" in
    // ctx.  convertCompute will read this live tensor directly rather than
    // deque-ing again.  The matching free is placed after the innermost
    // enclosing scf::ForOp so it runs once when the loop completes.
    if (srcMs == 0 && dstMs == 9) {
      Value dstQueue = ctx.getQueue(dst);
      if (!dstQueue) {
        copyOp.emitError("missing queue for VECIN buffer");
        return failure();
      }
      auto dstLtType =
          LocalTensorType::get(cast<MemRefType>(dst.getType()).getElementType());
      Value dstLt =
          builder.create<TQueBindAllocTensorOp>(loc, dstLtType, dstQueue);
      Value srcGt = builder.create<GlobalTensorOp>(
          loc,
          GlobalTensorType::get(cast<MemRefType>(src.getType()).getElementType()));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, src,
                                                     /*size=*/Value{});
      Value count = computeElementCount(builder, loc, dst);
      builder.create<DataCopyL2Op>(loc, dstLt, srcGt, count);
      builder.create<TQueBindEnqueTensorOp>(loc, dstQueue, dstLt);

      // Immediately deque so the local_tensor is available as a live value
      // that can be reused across inner loop iterations.
      Value liveLt =
          builder.create<TQueBindDequeTensorOp>(loc, dstLtType, dstQueue);
      ctx.allocToLiveTensor[getRootAlloc(dst)] = liveLt;

      // Place free_tensor at the end of copyOp's parent block (just before the
      // block terminator).  This keeps liveLt in scope across any inner loops
      // that follow the copy in the same block, while ensuring dominance: the
      // free runs once after all inner-loop uses complete.
      {
        OpBuilder::InsertionGuard guard(builder);
        Block *parentBlock = copyOp->getBlock();
        builder.setInsertionPoint(parentBlock->getTerminator());
        builder.create<TQueBindFreeTensorOp>(loc, dstQueue, liveLt);
      }

      copyOp.erase();
      continue;
    }

    // A1(1) → A2(2): deque → alloc → load_data_l0 → enque → free
    //
    // A1 is loaded (GM→A1) at an outer loop level; the A1→A2 copies that
    // consume A1 live inside an inner K-loop.  To keep the deque/free pair at
    // the same loop level as the enque (outer), we hoist the deque to just
    // before the enclosing for-loop and the free to just after it.
    if (srcMs == 1 && dstMs == 2) {
      Value srcQueue = ctx.getQueue(src);
      Value dstQueue = ctx.getQueue(dst);
      if (!srcQueue || !dstQueue) {
        copyOp.emitError("missing queue for A1/A2 buffer");
        return failure();
      }
      auto srcLtType =
          LocalTensorType::get(cast<MemRefType>(src.getType()).getElementType());
      auto dstLtType =
          LocalTensorType::get(cast<MemRefType>(dst.getType()).getElementType());

      // Hoist deque(A1) to before the outermost for-loop that contains copyOp
      // but does NOT contain the A1 alloc.  This matches the enque level so
      // each deque/free pair is executed exactly once per enque.
      Value srcRootAlloc = getRootAlloc(src);
      scf::ForOp hoistFor = getOutermostEnclosingForOutside(
          copyOp, srcRootAlloc.getDefiningOp());

      Value srcLt;
      if (hoistFor) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(hoistFor);
        srcLt = builder.create<TQueBindDequeTensorOp>(loc, srcLtType, srcQueue);
      } else {
        srcLt =
            builder.create<TQueBindDequeTensorOp>(loc, srcLtType, srcQueue);
      }

      Value dstLt =
          builder.create<TQueBindAllocTensorOp>(loc, dstLtType, dstQueue);

      // LoadData2DParams for A1→A2 (NZ→ZZ, no transpose):
      //   start_index = 0, repeat_times = k/16 (kBlocks),
      //   src_stride  = m/16 (mBlocks, stride between C0 groups),
      //   if_transpose = false
      // dst is [m x k]
      Value mBlocks = toI16(builder, loc,
          builder.create<arith::DivUIOp>(
              loc, emitDim(builder, loc, dst, 0),
              builder.create<arith::ConstantIndexOp>(loc, 16)));
      Value kBlocks = toI8(builder, loc,
          builder.create<arith::DivUIOp>(
              loc, emitDim(builder, loc, dst, 1),
              builder.create<arith::ConstantIndexOp>(loc, 16)));

      SmallVector<Value> ldOperands = {
          constI16(builder, loc, 0),    // start_index
          kBlocks,                       // repeat_times (k/16)
          mBlocks,                       // src_stride (m/16)
          constI16(builder, loc, 0),    // sid
          constI16(builder, loc, 0),    // dst_gap
          constI1(builder, loc, false), // if_transpose
          constI8(builder, loc, 0),     // addr_mode
      };
      SmallVector<Type> ldTypes = {builder.getI16Type(), builder.getI8Type(),
                                    builder.getI16Type(), builder.getI16Type(),
                                    builder.getI16Type(), builder.getI1Type(),
                                    builder.getI8Type()};
      Value params = builder.create<ConstructOp>(
          loc, LoadData2DParamsType::get(mlirCtx), ldOperands,
          builder.getTypeArrayAttr(ldTypes));
      builder.create<LoadDataL0Op>(loc, dstLt, srcLt, params);
      builder.create<TQueBindEnqueTensorOp>(loc, dstQueue, dstLt);
      if (hoistFor) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(hoistFor);
        builder.create<TQueBindFreeTensorOp>(hoistFor.getLoc(), srcQueue,
                                             srcLt);
      } else {
        builder.create<TQueBindFreeTensorOp>(loc, srcQueue, srcLt);
      }
      copyOp.erase();
      continue;
    }

    // B1(3) → B2(4): deque → alloc → load_data_with_transpose → enque → free
    if (srcMs == 3 && dstMs == 4) {
      Value srcQueue = ctx.getQueue(src);
      Value dstQueue = ctx.getQueue(dst);
      if (!srcQueue || !dstQueue) {
        copyOp.emitError("missing queue for B1/B2 buffer");
        return failure();
      }
      auto srcLtType =
          LocalTensorType::get(cast<MemRefType>(src.getType()).getElementType());
      auto dstLtType =
          LocalTensorType::get(cast<MemRefType>(dst.getType()).getElementType());

      // Same hoist logic as A1→A2.
      Value srcRootAllocB = getRootAlloc(src);
      scf::ForOp hoistForB = getOutermostEnclosingForOutside(
          copyOp, srcRootAllocB.getDefiningOp());

      Value srcLt;
      if (hoistForB) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(hoistForB);
        srcLt = builder.create<TQueBindDequeTensorOp>(loc, srcLtType, srcQueue);
      } else {
        srcLt =
            builder.create<TQueBindDequeTensorOp>(loc, srcLtType, srcQueue);
      }

      Value dstLt =
          builder.create<TQueBindAllocTensorOp>(loc, dstLtType, dstQueue);

      // LoadData2DParams for B1→B2 (NZ→ZN, transpose=true):
      //   repeat_times = k/16, src_stride = 1, if_transpose = true
      // src is [k x n]
      Value kBlocks = toI8(builder, loc,
          builder.create<arith::DivUIOp>(
              loc, emitDim(builder, loc, src, 0),
              builder.create<arith::ConstantIndexOp>(loc, 16)));

      SmallVector<Value> ldOperands = {
          constI16(builder, loc, 0),   // start_index
          kBlocks,                      // repeat_times (k/16)
          constI16(builder, loc, 1),   // src_stride = 1
          constI16(builder, loc, 0),   // sid
          constI16(builder, loc, 0),   // dst_gap
          constI1(builder, loc, true), // if_transpose = true
          constI8(builder, loc, 0),    // addr_mode
      };
      SmallVector<Type> ldTypes = {builder.getI16Type(), builder.getI8Type(),
                                    builder.getI16Type(), builder.getI16Type(),
                                    builder.getI16Type(), builder.getI1Type(),
                                    builder.getI8Type()};
      Value params = builder.create<ConstructOp>(
          loc, LoadData2dTransposeParamsType::get(mlirCtx), ldOperands,
          builder.getTypeArrayAttr(ldTypes));
      builder.create<LoadDataWithTransposeOp>(loc, dstLt, srcLt, params);
      builder.create<TQueBindEnqueTensorOp>(loc, dstQueue, dstLt);
      if (hoistForB) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(hoistForB);
        builder.create<TQueBindFreeTensorOp>(hoistForB.getLoc(), srcQueue,
                                             srcLt);
      } else {
        builder.create<TQueBindFreeTensorOp>(loc, srcQueue, srcLt);
      }
      copyOp.erase();
      continue;
    }

    // CO1(7) → VECIN(9): deque → alloc → data_copy_co12dst → enque → free
    if (srcMs == 7 && dstMs == 9) {
      Value srcQueue = ctx.getQueue(src);
      Value dstQueue = ctx.getQueue(dst);
      if (!srcQueue || !dstQueue) {
        copyOp.emitError("missing queue for CO1/VECIN buffer");
        return failure();
      }
      Type elemType = cast<MemRefType>(src.getType()).getElementType();
      auto srcLtType = LocalTensorType::get(elemType);
      Value srcLt =
          builder.create<TQueBindDequeTensorOp>(loc, srcLtType, srcQueue);
      Value dstLt =
          builder.create<TQueBindAllocTensorOp>(loc, srcLtType, dstQueue);
      
      // Create DataCopyCO12DstParams for CO1 → VECIN transfer
      Value params = builder.create<ConstructOp>(
          loc, DataCopyCO12DstParamsType::get(mlirCtx));
      
      builder.create<DataCopyCO12DstOp>(loc, dstLt, srcLt, params);
      builder.create<TQueBindEnqueTensorOp>(loc, dstQueue, dstLt);
      builder.create<TQueBindFreeTensorOp>(loc, srcQueue, srcLt);
      copyOp.erase();
      continue;
    }

    // VECOUT(10) → GM(0): deque → data_copy_l2 → free
    if (srcMs == 10 && dstMs == 0) {
      Value srcQueue = ctx.getQueue(src);
      if (!srcQueue) {
        copyOp.emitError("missing queue for VECOUT buffer");
        return failure();
      }
      auto srcLtType =
          LocalTensorType::get(cast<MemRefType>(src.getType()).getElementType());
      Value srcLt =
          builder.create<TQueBindDequeTensorOp>(loc, srcLtType, srcQueue);
      Value dstGt = builder.create<GlobalTensorOp>(
          loc,
          GlobalTensorType::get(cast<MemRefType>(dst.getType()).getElementType()));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, dst,
                                                     /*size=*/Value{});
      Value count = computeElementCount(builder, loc, src);
      builder.create<DataCopyL2Op>(loc, dstGt, srcLt, count);
      builder.create<TQueBindFreeTensorOp>(loc, srcQueue, srcLt);
      copyOp.erase();
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "[datamove] unrecognized copy: srcMs=" << srcMs
                             << " dstMs=" << dstMs << "\n");
  }

  return success();
}

} // namespace afir
} // namespace mlir
