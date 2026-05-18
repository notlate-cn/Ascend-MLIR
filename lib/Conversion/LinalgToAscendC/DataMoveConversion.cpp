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
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

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

// Helper: C++ scalar type name for a verbatim DataCopy template.
static std::string cppScalarName(Type t) {
  if (t.isF32()) return "float";
  if (t.isF16()) return "half";
  if (t.isBF16()) return "bfloat16_t";
  if (auto it = dyn_cast<IntegerType>(t))
    return "int" + std::to_string(it.getWidth()) + "_t";
  return "float";
}

// Helper: detect a 2-D memref whose rows may be non-contiguous in memory — a
// "strided" subview such as the transposed-operand tile x[r0:r0+R, c0:c0+C] of
// a wider buffer (row stride = the wider buffer's inner dim, not C).  Returns
// true and sets `rowStride` (in elements) when the row stride is statically
// known and a row gap is possible; a plain contiguous memref returns false (the
// flat DataCopy fast path applies).  Only the 2-D case is handled here.
// Returns true when the memref is a 2-D row-strided source/dest that needs
// per-row copies (rows not packed in memory).  Two flavors hit this:
//   - static stride that's known to differ from the column extent
//     (e.g. an RBLOCK reduction-split x[A,R] tile, or a transposed operand);
//   - dynamic stride (e.g. a FullLoad leading-reduce slice of x[D0,D1] where
//     shape symbolization hasn't pinned D1 — the runtime stride may exceed
//     cols).  Caller must extract the runtime stride via
//     memref.extract_strided_metadata.
//
// A plain contiguous memref (stride[0] statically == cols) returns false so
// the flat-DataCopy fast path applies.  Only the 2-D case is handled here.
static bool isMaybeRowStrided2D(MemRefType mrt) {
  if (mrt.getRank() != 2)
    return false;
  // Only fire when the type carries an explicit StridedLayoutAttr — that's
  // the marker for "this came from a subview / cast and the rows may not be
  // packed".  An identity-layout memref<?x?xf32> is contiguous at runtime
  // (stride[0] = dim 1), even though its dim-0 stride looks dynamic.
  auto sl = dyn_cast<StridedLayoutAttr>(mrt.getLayout());
  if (!sl)
    return false;
  ArrayRef<int64_t> strides = sl.getStrides();
  if (strides[1] != 1)
    return false;
  if (ShapedType::isDynamic(strides[0]))
    return true; // conservatively assume strided
  int64_t cols = mrt.getDimSize(1);
  if (!ShapedType::isDynamic(cols) && cols == strides[0])
    return false; // provably contiguous
  return true;
}

// Get an index Value for the row stride of a (possibly dynamic-stride) 2-D
// strided memref.  Static stride → ConstantIndexOp.  Dynamic stride: the
// producer is a memref.subview with unit strides over a row-major source
// memref; the row stride is the source's dim 1.  Pulling from the parent
// directly (instead of memref.extract_strided_metadata on the subview)
// keeps the subview from being kept alive past LinalgToAscendC — the
// downstream afir-translate has no printer for memref.subview.
static Value materializeRowStride(OpBuilder &b, Location loc, Value memref) {
  auto mrt = cast<MemRefType>(memref.getType());
  auto sl = cast<StridedLayoutAttr>(mrt.getLayout());
  ArrayRef<int64_t> strides = sl.getStrides();
  if (!ShapedType::isDynamic(strides[0]))
    return b.create<arith::ConstantIndexOp>(loc, strides[0]);
  auto subview = memref.getDefiningOp<memref::SubViewOp>();
  assert(subview && "dynamic-stride 2-D strided memref must come from a "
                    "memref.subview (materializeRowStride extension needed)");
  for (OpFoldResult s : subview.getMixedStrides()) {
    auto attr = dyn_cast<Attribute>(s);
    assert(attr && cast<IntegerAttr>(attr).getInt() == 1 &&
           "non-unit subview stride: materializeRowStride needs extension");
    (void)attr;
  }
  Value src = subview.getSource();
  auto srcMrt = cast<MemRefType>(src.getType());
  assert(srcMrt.getRank() == 2 &&
         "rank>2 source: materializeRowStride needs extension");
  (void)srcMrt;
  return b.create<memref::DimOp>(loc, src, 1);
}

// Get an index Value for `dim` of a possibly-dynamic 2-D strided subview.
// Prefers the SubViewOp's mixed size at that index (an Attribute constant or
// the SSA value passed to the subview) over `memref.dim` on the subview
// result — the latter keeps the subview alive past LinalgToAscendC.
static Value materializeSubviewDim(OpBuilder &b, Location loc, Value memref,
                                    unsigned dim) {
  auto mrt = cast<MemRefType>(memref.getType());
  if (!ShapedType::isDynamic(mrt.getShape()[dim]))
    return b.create<arith::ConstantIndexOp>(loc, mrt.getShape()[dim]);
  if (auto subview = memref.getDefiningOp<memref::SubViewOp>()) {
    OpFoldResult s = subview.getMixedSizes()[dim];
    if (auto attr = dyn_cast<Attribute>(s))
      return b.create<arith::ConstantIndexOp>(loc,
                                              cast<IntegerAttr>(attr).getInt());
    return cast<Value>(s);
  }
  return b.create<memref::DimOp>(loc, memref, dim);
}

// Emit a GM → VECIN copy of a 2-D row-strided source: one plain DataCopy per
// row (src row i at srcGt[i*rowStride], dst row i packed at dstLt[i*cols]).
// Uses only the proven GetPhyAddr / DataCopy path (no strided DataCopyPad,
// which this AscendC/sim build does not handle for GM→UB).  Requires
// cols*sizeof(elem) % 32 == 0 — the tiling-space generator is expected to
// honour that for inner tile sizes feeding a transposed operand.
static void emitStridedGmToVecinDataCopy(OpBuilder &b, Location loc, Type elemTy,
                                          Value dstLt, Value srcGt, Value rows,
                                          Value cols, Value rowStride) {
  std::string ets = cppScalarName(elemTy);
  std::string tmpl =
      "{\n"
      "  for (uint32_t _afir_i = 0; _afir_i < (uint32_t)$2; _afir_i++) {\n"
      "    AscendC::GlobalTensor<" + ets + "> _afir_gt;\n"
      "    _afir_gt.SetGlobalBuffer($1.GetPhyAddr(_afir_i * (uint32_t)$4));\n"
      "    AscendC::DataCopy($0[_afir_i * (uint32_t)$3], _afir_gt, (uint32_t)$3);\n"
      "  }\n"
      "}";
  b.create<emitasc::VerbatimOp>(loc, b.getStringAttr(tmpl),
                                ValueRange({dstLt, srcGt, rows, cols,
                                            rowStride}));
}

// Emit a VECOUT/VECCALC → GM store of a 2-D row-strided destination: one plain
// DataCopy per row (src row i packed at srcLt[i*cols], dst row i at
// dstGt[i*rowStride]).  Symmetric to emitStridedGmToVecinDataCopy.  Same
// cols*sizeof(elem) % 32 == 0 requirement.
static void emitStridedVecToGmDataCopy(OpBuilder &b, Location loc, Type elemTy,
                                        Value dstGt, Value srcLt, Value rows,
                                        Value cols, Value rowStride) {
  std::string ets = cppScalarName(elemTy);
  std::string tmpl =
      "{\n"
      "  for (uint32_t _afir_i = 0; _afir_i < (uint32_t)$2; _afir_i++) {\n"
      "    AscendC::GlobalTensor<" + ets + "> _afir_gt;\n"
      "    _afir_gt.SetGlobalBuffer($0.GetPhyAddr(_afir_i * (uint32_t)$4));\n"
      "    AscendC::DataCopy(_afir_gt, $1[_afir_i * (uint32_t)$3], (uint32_t)$3);\n"
      "  }\n"
      "}";
  b.create<emitasc::VerbatimOp>(loc, b.getStringAttr(tmpl),
                                ValueRange({dstGt, srcLt, rows, cols,
                                            rowStride}));
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
      // Nd2NzParams fields are all uint16_t — use Unsigned so CodeEmitter emits
      // static_cast<uint16_t>() instead of static_cast<int16_t>().
      auto ui16 = IntegerType::get(mlirCtx, 16, IntegerType::Unsigned);
      SmallVector<Type> nd2nzTypes(8, ui16);
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
      auto srcMrt = cast<MemRefType>(src.getType());
      Value srcGt = builder.create<GlobalTensorOp>(
          loc, GlobalTensorType::get(srcMrt.getElementType()));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, src,
                                                     /*size=*/Value{});
      if (isMaybeRowStrided2D(srcMrt)) {
        // Strided 2-D source — copy row by row, packing into the VECIN tile.
        // Triggers for absorbed-transpose operand tiles, RBLOCK reduction-
        // split chunks, and FullLoad leading-reduce slices.  A flat DataCopy
        // of rows*cols elements would read contiguous GM (the wrong rows).
        Value rowStride = materializeRowStride(builder, loc, src);
        Value rows = materializeSubviewDim(builder, loc, src, 0);
        Value cols = materializeSubviewDim(builder, loc, src, 1);
        emitStridedGmToVecinDataCopy(builder, loc, srcMrt.getElementType(),
                                     dstLt, srcGt, rows, cols, rowStride);
      } else {
        Value count = computeElementCount(builder, loc, dst);
        builder.create<DataCopyL2Op>(loc, dstLt, srcGt, count);
      }
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
      // LoadData2DParams fields: uint16, uint8, uint16, uint8, uint16, bool, uint8
      auto ui16 = IntegerType::get(mlirCtx, 16, IntegerType::Unsigned);
      auto ui8  = IntegerType::get(mlirCtx, 8,  IntegerType::Unsigned);
      SmallVector<Type> ldTypes = {ui16, ui8, ui16, ui8, ui16,
                                   builder.getI1Type(), ui8};
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

      // LoadData2dTransposeParams(startIndex, repeatTimes, srcStride,
      //                           dstGap, dstFracGap, addrMode)
      // No sid or ifTranspose fields in this struct.
      SmallVector<Value> ldOperands = {
          constI16(builder, loc, 0),   // startIndex
          kBlocks,                      // repeatTimes (k/16)
          constI16(builder, loc, 1),   // srcStride = 1
          constI16(builder, loc, 0),   // dstGap = 0
          constI16(builder, loc, 0),   // dstFracGap = 0
          constI8(builder, loc, 0),    // addrMode = 0
      };
      // LoadData2dTransposeParams fields: uint16, uint8, uint16, uint16, uint16, uint8
      auto ui16 = IntegerType::get(mlirCtx, 16, IntegerType::Unsigned);
      auto ui8  = IntegerType::get(mlirCtx, 8,  IntegerType::Unsigned);
      SmallVector<Type> ldTypes = {ui16, ui8, ui16, ui16, ui16, ui8};
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
      auto dstMrt = cast<MemRefType>(dst.getType());
      auto srcLtType =
          LocalTensorType::get(cast<MemRefType>(src.getType()).getElementType());
      Value srcLt =
          builder.create<TQueBindDequeTensorOp>(loc, srcLtType, srcQueue);
      Value dstGt = builder.create<GlobalTensorOp>(
          loc, GlobalTensorType::get(dstMrt.getElementType()));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, dst,
                                                     /*size=*/Value{});
      if (isMaybeRowStrided2D(dstMrt)) {
        // Strided 2-D destination — store row by row; a flat DataCopy would
        // write contiguous GM (the wrong rows).
        Value rowStride = materializeRowStride(builder, loc, dst);
        Value rows = materializeSubviewDim(builder, loc, dst, 0);
        Value cols = materializeSubviewDim(builder, loc, dst, 1);
        emitStridedVecToGmDataCopy(builder, loc, dstMrt.getElementType(), dstGt,
                                   srcLt, rows, cols, rowStride);
      } else {
        Value count = computeElementCount(builder, loc, src);
        builder.create<DataCopyL2Op>(loc, dstGt, srcLt, count);
      }
      builder.create<TQueBindFreeTensorOp>(loc, srcQueue, srcLt);
      copyOp.erase();
      continue;
    }

    // VECCALC(11) → GM(0): the result-store of a reduction-split accumulator.
    // The accumulator is a TBuf-backed local tensor (not queued); the compute
    // conversion's fill pre-pass registers it as `src`'s live tensor and writes
    // it via the same TBuf, so a fresh TBufGetTensor on that same TBuf is the
    // correct source here (data_copy reads it after the reduction loop).
    if (srcMs == 11 && dstMs == 0) {
      auto elemTy = cast<MemRefType>(src.getType()).getElementType();
      Value srcLt = ctx.getLiveTensor(src);
      if (!srcLt) {
        Value tbuf = ctx.getTBuf(src);
        if (!tbuf) {
          copyOp.emitError("missing TBuf for VECCALC accumulator");
          return failure();
        }
        srcLt = builder.create<TBufGetTensorOp>(
            loc, LocalTensorType::get(elemTy), tbuf, /*len=*/Value{});
      }
      Value dstGt =
          builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemTy));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, dst,
                                                     /*size=*/Value{});
      Value count = computeElementCount(builder, loc, src);
      // The accumulator was just written by vector ops in the RBLOCK loop; the
      // DataCopy below runs on the MTE3 pipe and would otherwise race ahead of
      // those writes (the queued VECOUT path gets this sync from EnQue/DeQue,
      // but the TBuf accumulator has no queue).  Barrier all pipes first.
      builder.create<PipeBarrierOp>(loc, Pipe::PIPE_ALL);
      builder.create<DataCopyL2Op>(loc, dstGt, srcLt, count);
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
