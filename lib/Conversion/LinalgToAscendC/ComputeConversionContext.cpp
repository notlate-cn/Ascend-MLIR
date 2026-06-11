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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallBitVector.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace afir {

Value ComputeCtx::getSubviewSizeValue(OpBuilder &b, Location loc, Value memref,
                                      unsigned dim) {
  auto subviewOp = memref.getDefiningOp<memref::SubViewOp>();
  if (!subviewOp)
    return Value{};
  SmallVector<OpFoldResult> mixedSizes = subviewOp.getMixedSizes();
  // `dim` indexes the (possibly rank-reduced) RESULT; mixedSizes is indexed by
  // SOURCE rank.  For a rank-reducing subview (e.g. peel-outer-R's
  // x[1, a_tile, R2] -> [a_tile, R2]) the dropped unit dims must be skipped so
  // result dim d maps to the d-th kept source dim — otherwise the reduction
  // extent (R2) is read as the dropped unit size and the reduce degenerates.
  llvm::SmallBitVector dropped = subviewOp.getDroppedDims();
  unsigned srcDim = 0, kept = 0;
  for (; srcDim < mixedSizes.size(); ++srcDim) {
    if (dropped[srcDim])
      continue;
    if (kept == dim)
      break;
    ++kept;
  }
  if (srcDim >= mixedSizes.size())
    return Value{};
  OpFoldResult size = mixedSizes[srcDim];
  if (auto attr = size.dyn_cast<Attribute>())
    return b.create<arith::ConstantIndexOp>(loc,
                                            cast<IntegerAttr>(attr).getInt());
  return size.get<Value>();
}

Value ComputeCtx::computeProduct(OpBuilder &b, Location loc,
                                 ArrayRef<Value> dims) {
  Value totalElems;
  for (Value s : dims)
    totalElems = totalElems ? b.create<arith::MulIOp>(loc, totalElems, s) : s;
  if (!totalElems)
    totalElems = b.create<arith::ConstantIndexOp>(loc, 1);
  return totalElems;
}

Value ComputeCtx::dequeTensor(OpBuilder &b, Location loc, Value queue,
                              Type elemType) {
  return b.create<TQueBindDequeTensorOp>(loc, LocalTensorType::get(elemType),
                                         queue);
}

Value ComputeCtx::allocTensor(OpBuilder &b, Location loc, Value queue,
                              Type elemType) {
  return b.create<TQueBindAllocTensorOp>(loc, LocalTensorType::get(elemType),
                                         queue);
}

Value ComputeCtx::tbufTensor(OpBuilder &b, Location loc, int64_t ms,
                             Type elemType) {
  auto pos = static_cast<TPosition>(ms > 0 ? ms : 0);
  Value tbuf = b.create<TBufOp>(loc, TBufType::get(mlirCtx, pos));
  return b.create<TBufGetTensorOp>(loc, LocalTensorType::get(elemType), tbuf,
                                   /*len=*/Value{});
}

Value ComputeCtx::subviewByteOffset(OpBuilder &b, Location loc, Value memref) {
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
}

Value ComputeCtx::tbufSlice(OpBuilder &b, Location loc, Value memref,
                            Value sizeBytes, Value offsetBytes) {
  Value tbuf = ctx.getTBuf(memref);
  if (!tbuf)
    return Value{};
  auto mrt = cast<MemRefType>(memref.getType());
  return b.create<TBufGetWithOffsetOp>(
      loc, LocalTensorType::get(mrt.getElementType()), tbuf, sizeBytes,
      offsetBytes);
}

Value ComputeCtx::readTensor(OpBuilder &b, Location loc, Value memref) {
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
}

Value ComputeCtx::writeTensor(OpBuilder &b, Location loc, Value memref) {
  auto mrt = cast<MemRefType>(memref.getType());
  if (Value byteOff = subviewByteOffset(b, loc, memref)) {
    Value sizeBytes = computeByteCount(b, loc, memref);
    if (Value t = tbufSlice(b, loc, memref, sizeBytes, byteOff))
      return t;
  }
  if (Value live = ctx.getLiveTensor(memref))
    return live;
  if (Value q = ctx.getQueue(memref))
    return allocTensor(b, loc, q, mrt.getElementType());
  return tbufTensor(b, loc, getMemorySpace(mrt), mrt.getElementType());
}

std::pair<Value, scf::ForOp> ComputeCtx::allocHoisted(Operation *op,
                                                      Value queue,
                                                      Type elemType,
                                                      Location loc) {
  scf::ForOp forOp = getEnclosingFor(op);
  if (!forOp)
    return {allocTensor(builder, loc, queue, elemType), nullptr};
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(forOp);
  Value tensor = allocTensor(builder, loc, queue, elemType);
  return {tensor, forOp};
}

std::pair<Value, Value> ComputeCtx::allocVeccalc(OpBuilder &b, Location loc,
                                                 Type elemType,
                                                 SmallVector<Value> dynSizes) {
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
}

Value ComputeCtx::copyGmToVecin(
    OpBuilder &b, Location loc, Type elemType, Value srcGt, Value elemCount,
    Value bufferElemCount,
    SmallVectorImpl<std::pair<Value, Value>> *tempVecinTensors) {
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
}

std::pair<Value, Value> ComputeCtx::copyGmToVecinStrided(
    OpBuilder &initB, OpBuilder &b, Location loc, Type elemType, Value srcGt,
    Value rows, Value cols, Value rowStride) {
  unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  Value nElems = initB.create<arith::MulIOp>(loc, rows, cols);
  Value byteSize = initB.create<arith::MulIOp>(
      loc, nElems, initB.create<arith::ConstantIndexOp>(loc, elemBytes));
  Value vecinTbuf =
      initB.create<TBufOp>(loc, TBufType::get(mlirCtx, TPosition::VECIN));
  initB.create<TPipeInitBufferOp>(loc, ctx.pipe, vecinTbuf, byteSize);
  Value vecinQue =
      initB.create<QueueOp>(loc, QueueType::get(mlirCtx, TPosition::VECIN, 1));
  Value depth = initB.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(1));
  initB.create<TPipeInitQueueOp>(loc, ctx.pipe, vecinQue, depth, byteSize);
  Value lt = b.create<TQueBindAllocTensorOp>(
      loc, LocalTensorType::get(elemType), vecinQue);
  std::string ets = cppScalarName(elemType);
  // One plain DataCopy per row: src row `i` lives at $1[i*rowStride] in GM
  // (GetPhyAddr → fresh GlobalTensor), dst row `i` at $0[i*cols] in UB.
  // Uses only the proven DataCopy / GetPhyAddr / LocalTensor::operator[]
  // path (rather than a strided DataCopyPad, which this AscendC/sim build
  // does not handle for GM→UB).  Requires cols*sizeof(elem) % 32 == 0.
  std::string tmpl =
      "{\n"
      "  for (uint32_t _afir_i = 0; _afir_i < (uint32_t)$2; _afir_i++) {\n"
      "    AscendC::GlobalTensor<" + ets + "> _afir_gt;\n"
      "    _afir_gt.SetGlobalBuffer($1.GetPhyAddr(_afir_i * (uint32_t)$4));\n"
      "    AscendC::DataCopy($0[_afir_i * (uint32_t)$3], _afir_gt, (uint32_t)$3);\n"
      "  }\n"
      "}";
  b.create<emitasc::VerbatimOp>(loc, b.getStringAttr(tmpl),
                                ValueRange({lt, srcGt, rows, cols, rowStride}));
  b.create<TQueBindEnqueTensorOp>(loc, vecinQue, lt);
  Value deq = b.create<TQueBindDequeTensorOp>(
      loc, LocalTensorType::get(elemType), vecinQue);
  return {deq, vecinQue};
}

Value ComputeCtx::getDynDim(OpBuilder &b, Location loc, Value memref,
                            unsigned dim) {
  if (Value subviewSize = getSubviewSizeValue(b, loc, memref, dim))
    return subviewSize;
  auto mrt = cast<MemRefType>(memref.getType());
  if (!ShapedType::isDynamic(mrt.getShape()[dim]))
    return b.create<arith::ConstantIndexOp>(loc, mrt.getShape()[dim]);
  return b.create<memref::DimOp>(loc, memref, dim);
}

SmallVector<Value> ComputeCtx::getBufferDimSizes(ArrayRef<Value> dims,
                                                 Operation *anchor) {
  SmallVector<Value> bufferDims;
  bufferDims.reserve(dims.size());
  for (Value dim : dims)
    bufferDims.push_back(getEnclosingLoopStepBound(dim, anchor));
  return bufferDims;
}

void ComputeCtx::freeTempVecinTensors(
    OpBuilder &b, Location loc,
    ArrayRef<std::pair<Value, Value>> tempVecinTensors) {
  for (auto [queue, tensor] : tempVecinTensors)
    b.create<TQueBindFreeTensorOp>(loc, queue, tensor);
}

} // namespace afir
} // namespace mlir
