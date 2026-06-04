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

#include "Conversion/Ascend/Translate/KernelIR/KernelIRUtils.h"

#include "Conversion/Ascend/Common/Attributes.h"
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

#include <optional>

#define DEBUG_TYPE "ascend-compute-lower-datamove"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace ascend {

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

static Value ceilDivIndex(OpBuilder &b, Location loc, Value value,
                          int64_t divisor) {
  Value divisorValue = b.create<arith::ConstantIndexOp>(loc, divisor);
  Value bias = b.create<arith::ConstantIndexOp>(loc, divisor - 1);
  Value numerator = b.create<arith::AddIOp>(loc, value, bias);
  return b.create<arith::DivUIOp>(loc, numerator, divisorValue);
}

static Value ceilToMultipleIndex(OpBuilder &b, Location loc, Value value,
                                 int64_t divisor) {
  return b.create<arith::MulIOp>(
      loc, ceilDivIndex(b, loc, value, divisor),
      b.create<arith::ConstantIndexOp>(loc, divisor));
}

// Helper: create a i1 constant.
static Value constI1(OpBuilder &b, Location loc, bool v) {
  return b.create<arith::ConstantIntOp>(loc, b.getI1Type(), v ? 1 : 0);
}

static bool shouldUseScalarVecoutWriteback(Value src) {
  auto srcType = dyn_cast<MemRefType>(src.getType());
  return srcType && srcType.getRank() == 1 &&
         ShapedType::isDynamic(srcType.getShape()[0]);
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

static bool isRank2Subview(Value v) {
  auto type = dyn_cast<MemRefType>(v.getType());
  return type && type.getRank() == 2 && v.getDefiningOp<memref::SubViewOp>();
}

static bool isContiguousRank2Subview(Value v) {
  auto type = dyn_cast<MemRefType>(v.getType());
  auto subview = v.getDefiningOp<memref::SubViewOp>();
  if (!type || type.getRank() != 2 || !subview)
    return false;
  auto [strides, offset] = type.getStridesAndOffset();
  (void)offset;
  if (strides.size() != 2 || strides[0] == ShapedType::kDynamic)
    return false;

  auto getStaticIndex = [](OpFoldResult ofr) -> std::optional<int64_t> {
    if (auto attr = ofr.dyn_cast<Attribute>())
      return cast<IntegerAttr>(attr).getInt();
    if (auto value = ofr.dyn_cast<Value>())
      if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
        return constant.value();
    return std::nullopt;
  };
  SmallVector<OpFoldResult> sizes = subview.getMixedSizes();
  if (sizes.size() < 2)
    return false;
  std::optional<int64_t> innerSize = getStaticIndex(sizes[1]);
  return innerSize && *innerSize == strides[0];
}

static Value getRank2RowStride(OpBuilder &b, Location loc, Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || type.getRank() != 2)
    return Value{};

  auto [strides, offset] = type.getStridesAndOffset();
  (void)offset;
  if (strides.size() == 2 && strides[0] != ShapedType::kDynamic)
    return b.create<arith::ConstantIndexOp>(loc, strides[0]);

  Value root = memref;
  while (auto subview = root.getDefiningOp<memref::SubViewOp>())
    root = subview.getSource();

  auto rootType = dyn_cast<MemRefType>(root.getType());
  if (rootType && rootType.getRank() == 2)
    return emitDim(b, loc, root, 1);

  return emitDim(b, loc, memref, 1);
}

static std::string getVerbatimScalarTypeName(Type elemType) {
  if (elemType.isF16())
    return "half";
  if (elemType.isF32())
    return "float";
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    if (intType.getWidth() == 8)
      return intType.isUnsigned() ? "uint8_t" : "int8_t";
    if (intType.getWidth() == 16)
      return intType.isUnsigned() ? "uint16_t" : "int16_t";
    if (intType.getWidth() == 32)
      return intType.isUnsigned() ? "uint32_t" : "int32_t";
  }
  return "auto";
}

static void emitStridedLocalToGmCopy(OpBuilder &b, Location loc, Type elemType,
                                     Value dstGt, Value srcLt, Value rows,
                                     Value cols, Value dstRowStride) {
  std::string elemTypeStr = getVerbatimScalarTypeName(elemType);
  std::string body = "{\n";
  body += "  uint32_t _ascend_rows = (uint32_t)$2;\n";
  body += "  uint32_t _ascend_cols = (uint32_t)$3;\n";
  body += "  uint32_t _ascend_row_stride = (uint32_t)$4;\n";
  body += "  uint32_t _ascend_gap_bytes = (_ascend_row_stride - _ascend_cols) * "
          "sizeof(" + elemTypeStr + ");\n";
  body += "  uint32_t _ascend_count = _ascend_rows * _ascend_cols;\n";
  body += "  if (_ascend_gap_bytes == 0u) {\n";
  body += "    if ((_ascend_count * sizeof(" + elemTypeStr +
          ")) % 32u == 0u) {\n";
  body += "      AscendC::DataCopy($0, $1, _ascend_count);\n";
  body += "    } else {\n";
  body += "      for (uint32_t _ascend_i = 0; _ascend_i < _ascend_count; "
          "++_ascend_i)\n";
  body += "        $0.SetValue(_ascend_i, static_cast<" + elemTypeStr +
          ">($1.GetValue(_ascend_i)));\n";
  body += "    }\n";
  body += "  } else {\n";
  body += "    for (uint32_t _ascend_r = 0; _ascend_r < _ascend_rows; ++_ascend_r) {\n";
  body += "      for (uint32_t _ascend_c = 0; _ascend_c < _ascend_cols; ++_ascend_c) "
          "{\n";
  body += "        uint32_t _ascend_local = _ascend_r * _ascend_cols + _ascend_c;\n";
  body += "        uint64_t _ascend_gm = (uint64_t)_ascend_r * _ascend_row_stride + "
          "_ascend_c;\n";
  body += "        $0.SetValue(_ascend_gm, static_cast<" + elemTypeStr +
          ">($1.GetValue(_ascend_local)));\n";
  body += "      }\n";
  body += "    }\n";
  body += "  }\n";
  body += "}";
  b.create<emitasc::VerbatimOp>(loc, b.getStringAttr(body),
                                ValueRange{dstGt, srcLt, rows, cols,
                                           dstRowStride});
}

static bool isValueOffset(OpFoldResult ofr, Value value) {
  if (auto offsetValue = dyn_cast<Value>(ofr))
    return offsetValue == value;
  return false;
}

static bool genericHasReductionIterator(linalg::GenericOp genericOp) {
  return llvm::any_of(genericOp.getIteratorTypesArray(),
                      [](utils::IteratorType iteratorType) {
                        return iteratorType == utils::IteratorType::reduction;
                      });
}

static linalg::GenericOp findReductionGenericWriting(Value memref) {
  Value root = getRootAlloc(memref);
  for (Operation *user : root.getUsers()) {
    auto genericOp = dyn_cast<linalg::GenericOp>(user);
    if (!genericOp || !genericHasReductionIterator(genericOp))
      continue;
    for (Value init : genericOp.getDpsInits()) {
      if (getRootAlloc(init) == root)
        return genericOp;
    }
  }
  return nullptr;
}

static scf::ForOp findReductionTileLoop(linalg::GenericOp genericOp) {
  if (!genericOp)
    return nullptr;

  auto iterTypes = genericOp.getIteratorTypesArray();
  auto maps = genericOp.getIndexingMapsArray();
  unsigned iterRank = iterTypes.size();
  SmallVector<unsigned> reductionDims;
  for (unsigned d = 0; d < iterRank; ++d)
    if (iterTypes[d] == utils::IteratorType::reduction)
      reductionDims.push_back(d);
  if (reductionDims.empty())
    return nullptr;

  SmallVector<scf::ForOp> enclosingLoops;
  for (Operation *parent = genericOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent))
      enclosingLoops.push_back(forOp);
  }

  for (scf::ForOp forOp : enclosingLoops) {
    Value iv = forOp.getInductionVar();
    for (unsigned inputIdx = 0; inputIdx < genericOp.getNumDpsInputs();
         ++inputIdx) {
      AffineMap map = maps[inputIdx];
      if (map.getNumResults() != iterRank)
        continue;
      Value input = genericOp.getDpsInputOperand(inputIdx)->get();
      auto subview = input.getDefiningOp<memref::SubViewOp>();
      if (!subview)
        continue;
      SmallVector<OpFoldResult> offsets = subview.getMixedOffsets();
      for (unsigned reductionDim : reductionDims) {
        if (reductionDim < offsets.size() &&
            isValueOffset(offsets[reductionDim], iv))
          return forOp;
      }
    }
  }

  return nullptr;
}

static void emitAddPreviousReductionPartial(OpBuilder &builder, Location loc,
                                            MLIRContext *mlirCtx,
                                            AscendCBufferContext &ctx,
                                            scf::ForOp reductionLoop,
                                            Value dst, Value srcLt,
                                            Value count, Type elemType) {
  if (!reductionLoop || !count)
    return;

  Value isNotFirst = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, reductionLoop.getInductionVar(),
      reductionLoop.getLowerBound());
  auto ifOp = builder.create<scf::IfOp>(loc, isNotFirst, /*withElseRegion=*/false);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  Value dstGt =
      builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
  builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, dst,
                                                 /*size=*/Value{});

  unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  Value byteSize = builder.create<arith::MulIOp>(
      loc, count, builder.create<arith::ConstantIndexOp>(loc, elemBytes));
  Value oldTbuf =
      builder.create<TBufOp>(loc, TBufType::get(mlirCtx, TPosition::VECIN));
  builder.create<TPipeInitBufferOp>(loc, ctx.pipe, oldTbuf, byteSize);
  Value oldQueue =
      builder.create<QueueOp>(loc, QueueType::get(mlirCtx, TPosition::VECIN, 1));
  Value depth = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(1));
  builder.create<TPipeInitQueueOp>(loc, ctx.pipe, oldQueue, depth, byteSize);

  Value oldLt = builder.create<TQueBindAllocTensorOp>(
      loc, LocalTensorType::get(elemType), oldQueue);
  builder.create<DataCopyL2Op>(loc, oldLt, dstGt, count);
  builder.create<TQueBindEnqueTensorOp>(loc, oldQueue, oldLt);
  Value oldDequeued = builder.create<TQueBindDequeTensorOp>(
      loc, LocalTensorType::get(elemType), oldQueue);
  builder.create<AddL2Op>(loc, srcLt, srcLt, oldDequeued, count);
  builder.create<TQueBindFreeTensorOp>(loc, oldQueue, oldDequeued);
}

static bool isGatherTBufBackedVecout(Value src) {
  Value root = getRootAlloc(src);
  for (Operation *user : root.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic || !generic->hasAttr(ascend::kGatherDimAttr))
      continue;
    for (Value init : generic.getDpsInits())
      if (getRootAlloc(init) == root)
        return true;
  }
  return false;
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
      Value dstNzNStride = ceilToMultipleIndex(builder, loc, height, 16);

      SmallVector<Value> nd2nzOperands = {
          toI16(builder, loc, height),  // nd_num
          toI16(builder, loc, nValue),  // n_value
          toI16(builder, loc, height),  // d_value
          toI16(builder, loc, width),   // src_nd_matrix_stride
          toI16(builder, loc, height),  // src_d_value
          constI16(builder, loc, 0),    // dst_nz_c0_stride
          toI16(builder, loc, dstNzNStride), // dst_nz_n_stride
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

    // GM(0) -> GM(0): plain global-to-global copy.
    if (srcMs == 0 && dstMs == 0) {
      Type elemType = cast<MemRefType>(dst.getType()).getElementType();
      Value dstGt = builder.create<GlobalTensorOp>(
          loc, GlobalTensorType::get(elemType));
      Value srcGt = builder.create<GlobalTensorOp>(
          loc, GlobalTensorType::get(
                   cast<MemRefType>(src.getType()).getElementType()));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, dst,
                                                     /*size=*/Value{});
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, src,
                                                     /*size=*/Value{});
      Value count = computeElementCount(builder, loc, dst);
      builder.create<DataCopyL2Op>(loc, dstGt, srcGt, count);
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
      //   src_stride  = ceil(m/16) (mBlocks, stride between C0 groups),
      //   if_transpose = false
      // dst is [m x k]
      Value mBlocks = toI16(builder, loc,
          ceilDivIndex(builder, loc, emitDim(builder, loc, dst, 0), 16));
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
      Value liveLt =
          builder.create<TQueBindDequeTensorOp>(loc, srcLtType, dstQueue);
      ctx.allocToLiveTensor[getRootAlloc(dst)] = liveLt;
      {
        OpBuilder::InsertionGuard guard(builder);
        Block *parentBlock = copyOp->getBlock();
        builder.setInsertionPoint(parentBlock->getTerminator());
        builder.create<TQueBindFreeTensorOp>(loc, dstQueue, liveLt);
      }
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
      if (isGatherTBufBackedVecout(src)) {
        Value srcTbuf = ctx.getTBuf(src);
        if (!srcTbuf) {
          copyOp.emitError("missing tbuf for gather VECOUT buffer");
          return failure();
        }
        Value srcLt = builder.create<TBufGetTensorOp>(
            loc, srcLtType, srcTbuf, /*len=*/Value{});
        Value dstGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(
                     cast<MemRefType>(dst.getType()).getElementType()));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, dst,
                                                       /*size=*/Value{});
        Value count = computeElementCount(builder, loc, src);
        if (isRank2Subview(dst) && !isContiguousRank2Subview(dst)) {
          Value rows = emitDim(builder, loc, dst, 0);
          Value cols = emitDim(builder, loc, dst, 1);
          Value dstRowStride = getRank2RowStride(builder, loc, dst);
          if (!dstRowStride) {
            copyOp.emitError("failed to compute rank-2 GM subview row stride");
            return failure();
          }
          emitStridedLocalToGmCopy(
              builder, loc, cast<MemRefType>(dst.getType()).getElementType(),
              dstGt, srcLt, rows, cols, dstRowStride);
        } else {
          builder.create<DataCopyL2Op>(loc, dstGt, srcLt, count);
        }
        copyOp.erase();
        continue;
      }
      Value srcLt =
          builder.create<TQueBindDequeTensorOp>(loc, srcLtType, srcQueue);
      Value dstGt = builder.create<GlobalTensorOp>(
          loc,
          GlobalTensorType::get(cast<MemRefType>(dst.getType()).getElementType()));
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, dst,
                                                     /*size=*/Value{});
      Value count = computeElementCount(builder, loc, src);
      linalg::GenericOp producer = findReductionGenericWriting(src);
      if (isRank2Subview(dst) && !isContiguousRank2Subview(dst) && !producer) {
        Value rows = emitDim(builder, loc, dst, 0);
        Value cols = emitDim(builder, loc, dst, 1);
        Value dstRowStride = getRank2RowStride(builder, loc, dst);
        if (!dstRowStride) {
          copyOp.emitError("failed to compute rank-2 GM subview row stride");
          return failure();
        }
        emitStridedLocalToGmCopy(
            builder, loc, cast<MemRefType>(dst.getType()).getElementType(),
            dstGt, srcLt, rows, cols, dstRowStride);
      } else if (producer) {
        if (scf::ForOp reductionLoop = findReductionTileLoop(producer)) {
          emitAddPreviousReductionPartial(
              builder, loc, mlirCtx, ctx, reductionLoop, dst, srcLt, count,
              cast<MemRefType>(src.getType()).getElementType());
        }
        builder.create<DataCopyL2Op>(loc, dstGt, srcLt, count);
      } else if (shouldUseScalarVecoutWriteback(src)) {
        builder.create<emitasc::VerbatimOp>(
            loc,
            builder.getStringAttr(
                "{\n"
                "  for (uint32_t _ascend_i = 0; _ascend_i < (uint32_t)$2; "
                "++_ascend_i) {\n"
                "    $0.SetValue(_ascend_i, $1.GetValue(_ascend_i));\n"
                "  }\n"
                "}"),
            ValueRange{dstGt, srcLt, count});
      } else {
        builder.create<DataCopyL2Op>(loc, dstGt, srcLt, count);
      }
      builder.create<TQueBindFreeTensorOp>(loc, srcQueue, srcLt);
      copyOp.erase();
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "[datamove] unrecognized copy: srcMs=" << srcMs
                             << " dstMs=" << dstMs << "\n");
  }

  return success();
}

} // namespace ascend
} // namespace mlir
