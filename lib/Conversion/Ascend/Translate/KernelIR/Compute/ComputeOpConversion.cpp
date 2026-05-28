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
namespace ascend {

namespace {

std::string getVerbatimScalarTypeName(Type elemType) {
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

} // namespace

void emitLocalToLocalScalarCopy(OpBuilder &builder, Location loc, Type elemType,
                                Value dstLt, Value srcLt, Value count) {
  std::string elemTypeStr = getVerbatimScalarTypeName(elemType);
  std::string body = "{\n";
  body += "  AscendC::PipeBarrier<PIPE_ALL>();\n";
  body += "  uint32_t _afir_count = static_cast<uint32_t>($2);\n";
  body += "  for (uint32_t _afir_i = 0; _afir_i < _afir_count; ++_afir_i)\n";
  body += "    $0.SetValue(_afir_i, static_cast<" + elemTypeStr +
          ">($1.GetValue(_afir_i)));\n";
  body += "  $0.SetSize(_afir_count);\n";
  body += "  AscendC::PipeBarrier<PIPE_ALL>();\n";
  body += "}";
  builder.create<emitasc::VerbatimOp>(
      loc, builder.getStringAttr(body), ValueRange{dstLt, srcLt, count});
}

void emitLocalTensorZeroPad(OpBuilder &builder, Location loc, Type elemType,
                            Value tensor, Value begin, Value end) {
  std::string elemTypeStr = getVerbatimScalarTypeName(elemType);
  std::string body = "{\n";
  body += "  AscendC::PipeBarrier<PIPE_ALL>();\n";
  body += "  uint32_t _afir_begin = static_cast<uint32_t>($1);\n";
  body += "  uint32_t _afir_end = static_cast<uint32_t>($2);\n";
  body += "  $0.SetSize(_afir_end);\n";
  body += "  for (uint32_t _afir_i = _afir_begin; _afir_i < _afir_end; "
          "++_afir_i)\n";
  body += "    $0.SetValue(_afir_i, static_cast<" + elemTypeStr + ">(0));\n";
  body += "  AscendC::PipeBarrier<PIPE_ALL>();\n";
  body += "}";
  builder.create<emitasc::VerbatimOp>(
      loc, builder.getStringAttr(body), ValueRange{tensor, begin, end});
}

Value ceilToMultipleIndex(OpBuilder &builder, Location loc, Value value,
                          int64_t divisor) {
  Value divisorValue = builder.create<arith::ConstantIndexOp>(loc, divisor);
  Value bias = builder.create<arith::ConstantIndexOp>(loc, divisor - 1);
  Value numerator = builder.create<arith::AddIOp>(loc, value, bias);
  Value quotient = builder.create<arith::DivUIOp>(loc, numerator, divisorValue);
  return builder.create<arith::MulIOp>(loc, quotient, divisorValue);
}

namespace {

struct QueuedLocalTensor {
  Value queue;
  Value tensor;
};

QueuedLocalTensor copyGlobalToVecinQueue(OpBuilder &builder, Location loc,
                                         Value pipe, Type elemType,
                                         Value srcGt, Value elemCount) {
  unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  Value byteCount = builder.create<arith::MulIOp>(
      loc, elemCount, builder.create<arith::ConstantIndexOp>(loc, elemBytes));
  Value vecinTbuf =
      builder.create<TBufOp>(loc, TBufType::get(builder.getContext(),
                                               TPosition::VECIN));
  builder.create<TPipeInitBufferOp>(loc, pipe, vecinTbuf, byteCount);
  Value queue = builder.create<QueueOp>(
      loc, QueueType::get(builder.getContext(), TPosition::VECIN, 1));
  Value depth =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
  builder.create<TPipeInitQueueOp>(loc, pipe, queue, depth, byteCount);
  Value allocated = builder.create<TQueBindAllocTensorOp>(
      loc, LocalTensorType::get(elemType), queue);
  builder.create<DataCopyL2Op>(loc, allocated, srcGt, elemCount);
  builder.create<TQueBindEnqueTensorOp>(loc, queue, allocated);
  Value dequeued = builder.create<TQueBindDequeTensorOp>(
      loc, LocalTensorType::get(elemType), queue);
  return {queue, dequeued};
}

LogicalResult lowerProjectedSuffixCopyToSegmentDataCopy(OpBuilder &builder,
                                                        linalg::GenericOp op,
                                                        Value pipe) {
  if (!isPureYieldGeneric(op))
    return failure();

  Location loc = op.getLoc();
  Value inMemref = op.getDpsInputOperand(0)->get();
  Value outMemref = op.getDpsInitOperand(0)->get();
  auto inType = dyn_cast<MemRefType>(inMemref.getType());
  auto outType = dyn_cast<MemRefType>(outMemref.getType());
  if (!inType || !outType || getMemorySpace(inType) != 0 ||
      getMemorySpace(outType) != 0)
    return failure();
  if (inType.getElementType() != outType.getElementType())
    return failure();

  unsigned inRank = static_cast<unsigned>(inType.getRank());
  unsigned outRank = static_cast<unsigned>(outType.getRank());
  if (inRank > outRank)
    return failure();

  AffineMap inMap = op.getIndexingMapsArray()[0];
  if (inMap.getNumResults() != inRank)
    return failure();

  unsigned prefixRank = outRank - inRank;
  for (unsigned dim = 0; dim < inRank; ++dim) {
    auto dimExpr = dyn_cast<AffineDimExpr>(inMap.getResult(dim));
    if (!dimExpr || dimExpr.getPosition() != prefixRank + dim)
      return failure();
  }

  Type elemType = inType.getElementType();
  Value copyCount = computeElementCount(builder, loc, inMemref);
  Value srcGt =
      builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
  builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                /*size=*/Value{});
  if (prefixRank == 0) {
    Value dstGt =
        builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
    builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, outMemref,
                                                  /*size=*/Value{});
    builder.create<DataCopyL2Op>(loc, dstGt, srcGt, copyCount);
    return success();
  }

  QueuedLocalTensor local =
      copyGlobalToVecinQueue(builder, loc, pipe, elemType, srcGt, copyCount);
  Value dstGt =
      builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 4> prefixUpperBounds;
  prefixUpperBounds.reserve(prefixRank);
  for (unsigned dim = 0; dim < prefixRank; ++dim)
    prefixUpperBounds.push_back(getDimValue(builder, loc, outMemref, dim));

  SmallVector<Value, 4> prefixIndices;
  auto buildNest = [&](auto &self, unsigned depth) -> LogicalResult {
    if (depth == prefixRank) {
      Value dstOffset;
      if (prefixRank > 0) {
        Value flatPrefix = prefixIndices.front();
        for (unsigned dim = 1; dim < prefixRank; ++dim) {
          flatPrefix =
              builder.create<arith::MulIOp>(loc, flatPrefix,
                                            prefixUpperBounds[dim]);
          flatPrefix =
              builder.create<arith::AddIOp>(loc, flatPrefix,
                                            prefixIndices[dim]);
        }
        dstOffset = builder.create<arith::MulIOp>(loc, flatPrefix, copyCount);
      }
      Value dstOffsetI32;
      if (dstOffset)
        dstOffsetI32 =
            builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                               dstOffset);
      builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, outMemref,
                                                    dstOffsetI32);
      builder.create<DataCopyL2Op>(loc, dstGt, local.tensor, copyCount);
      return success();
    }

    auto forOp =
        builder.create<scf::ForOp>(loc, c0, prefixUpperBounds[depth], c1);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    prefixIndices.push_back(forOp.getInductionVar());
    LogicalResult result = self(self, depth + 1);
    prefixIndices.pop_back();
    return result;
  };

  if (failed(buildNest(buildNest, 0)))
    return failure();
  builder.create<TQueBindFreeTensorOp>(loc, local.queue, local.tensor);
  return success();
}

} // namespace

IndexingMapAnalysis analyzeIndexingMap(AffineMap map, unsigned iterRank) {
  IndexingMapAnalysis result;

  if (map.isIdentity()) {
    result.kind = IndexingMapAnalysis::Kind::Identity;
    return result;
  }

  SmallVector<int64_t> presentDims;
  bool hasConstant = false;
  for (AffineExpr expr : map.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      presentDims.push_back(static_cast<int64_t>(dimExpr.getPosition()));
    } else if (isa<AffineConstantExpr>(expr)) {
      hasConstant = true;
    } else {
      result.kind = IndexingMapAnalysis::Kind::Identity;
      return result;
    }
  }

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
}

bool isBroadcastMap(AffineMap map, unsigned iterRank) {
  if (map.getNumResults() >= iterRank)
    return false;
  return true;
}

std::pair<Value, Value> allocVeccalc(ComputeLoweringContext &lowering,
                                     OpBuilder &builder, Location loc,
                                     Type elemType, ArrayRef<Value> dynSizes) {
  Value tbuf = builder.create<TBufOp>(
      loc, TBufType::get(lowering.mlirCtx, TPosition::VECCALC));
  Value totalElems = lowering.computeProduct(builder, loc, dynSizes);
  unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  Value byteSize = builder.create<arith::MulIOp>(
      loc, totalElems, builder.create<arith::ConstantIndexOp>(loc, elemBytes));
  builder.create<TPipeInitBufferOp>(loc, lowering.ctx.pipe, tbuf, byteSize);

  Value lt = builder.create<TBufGetTensorOp>(
      loc, LocalTensorType::get(elemType), tbuf, /*len=*/Value{});
  return {tbuf, lt};
}

void freeOwnedQueueTensors(OpBuilder &builder, Location loc,
                           ArrayRef<OwnedQueueTensor> ownedTensors) {
  for (const auto &[queue, tensor] : ownedTensors)
    builder.create<TQueBindFreeTensorOp>(loc, queue, tensor);
}

void rememberQueueRead(SmallVectorImpl<OwnedQueueTensor> &ownedTensors,
                       Value queue, Value tensor) {
  if (queue && tensor)
    ownedTensors.push_back({queue, tensor});
}

Value copyGmToVecin(ComputeLoweringContext &lowering, OpBuilder &builder,
                    Location loc, Type elemType, Value srcGt, Value elemCount,
                    Value bufferElemCount,
                    SmallVectorImpl<OwnedQueueTensor> *ownedTensors) {
  if (!bufferElemCount)
    bufferElemCount = elemCount;
  unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  Value byteSize = builder.create<arith::MulIOp>(
      loc, bufferElemCount,
      builder.create<arith::ConstantIndexOp>(loc, elemBytes));
  Value vecinTbuf = builder.create<TBufOp>(
      loc, TBufType::get(lowering.mlirCtx, TPosition::VECIN));
  builder.create<TPipeInitBufferOp>(loc, lowering.ctx.pipe, vecinTbuf,
                                    byteSize);
  Value vecinQue = builder.create<QueueOp>(
      loc, QueueType::get(lowering.mlirCtx, TPosition::VECIN, 1));
  Value depth = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(1));
  builder.create<TPipeInitQueueOp>(loc, lowering.ctx.pipe, vecinQue, depth,
                                   byteSize);
  Value lt = builder.create<TQueBindAllocTensorOp>(
      loc, LocalTensorType::get(elemType), vecinQue);
  builder.create<DataCopyL2Op>(loc, lt, srcGt, elemCount);
  builder.create<TQueBindEnqueTensorOp>(loc, vecinQue, lt);
  Value dequeued = builder.create<TQueBindDequeTensorOp>(
      loc, LocalTensorType::get(elemType), vecinQue);
  if (ownedTensors)
    rememberQueueRead(*ownedTensors, vecinQue, dequeued);
  return dequeued;
}

Value copyGmToVecinScalar(ComputeLoweringContext &lowering, OpBuilder &builder,
                          Location loc, Type elemType, Value srcGt,
                          Value elemCount, Value bufferElemCount,
                          SmallVectorImpl<OwnedQueueTensor> *ownedTensors) {
  if (!bufferElemCount)
    bufferElemCount = elemCount;
  unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  Value byteSize = builder.create<arith::MulIOp>(
      loc, bufferElemCount,
      builder.create<arith::ConstantIndexOp>(loc, elemBytes));
  Value vecinTbuf = builder.create<TBufOp>(
      loc, TBufType::get(lowering.mlirCtx, TPosition::VECIN));
  builder.create<TPipeInitBufferOp>(loc, lowering.ctx.pipe, vecinTbuf,
                                    byteSize);
  Value vecinQue = builder.create<QueueOp>(
      loc, QueueType::get(lowering.mlirCtx, TPosition::VECIN, 1));
  Value depth =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
  builder.create<TPipeInitQueueOp>(loc, lowering.ctx.pipe, vecinQue, depth,
                                   byteSize);
  Value lt = builder.create<TQueBindAllocTensorOp>(
      loc, LocalTensorType::get(elemType), vecinQue);

  std::string elemTypeStr = getVerbatimScalarTypeName(elemType);
  std::string body = "{\n";
  body += "  uint32_t _afir_count = static_cast<uint32_t>($2);\n";
  body += "  for (uint32_t _afir_i = 0; _afir_i < _afir_count; ++_afir_i)\n";
  body += "    $0.SetValue(_afir_i, static_cast<" + elemTypeStr +
          ">($1.GetValue(_afir_i)));\n";
  body += "  $0.SetSize(_afir_count);\n";
  body += "}";
  builder.create<emitasc::VerbatimOp>(loc, builder.getStringAttr(body),
                                      ValueRange{lt, srcGt, elemCount});

  builder.create<TQueBindEnqueTensorOp>(loc, vecinQue, lt);
  Value dequeued = builder.create<TQueBindDequeTensorOp>(
      loc, LocalTensorType::get(elemType), vecinQue);
  if (ownedTensors)
    rememberQueueRead(*ownedTensors, vecinQue, dequeued);
  return dequeued;
}

Value copyGmToVeccalc(ComputeLoweringContext &lowering, OpBuilder &builder,
                      Location loc, Type elemType, Value srcGt,
                      Value elemCount) {
  auto [veccalcTbuf, veccalcLt] =
      allocVeccalc(lowering, builder, loc, elemType,
                   SmallVector<Value>{elemCount});
  (void)veccalcTbuf;
  builder.create<DataCopyL2Op>(loc, veccalcLt, srcGt, elemCount);
  return veccalcLt;
}

Value getDynDim(ComputeLoweringContext &lowering, OpBuilder &builder,
                Location loc, Value memref, unsigned dim) {
  if (Value subviewSize = lowering.getSubviewSizeValue(builder, loc, memref, dim))
    return subviewSize;
  auto mrt = cast<MemRefType>(memref.getType());
  if (!ShapedType::isDynamic(mrt.getShape()[dim]))
    return builder.create<arith::ConstantIndexOp>(loc, mrt.getShape()[dim]);
  return builder.create<memref::DimOp>(loc, memref, dim);
}

SmallVector<Value> getBufferDimSizes(ComputeLoweringContext &lowering,
                                     ArrayRef<Value> dims,
                                     Operation *anchor) {
  SmallVector<Value> bufferDims;
  bufferDims.reserve(dims.size());
  for (Value dim : dims)
    bufferDims.push_back(lowering.getEnclosingLoopStepBound(dim, anchor));
  return bufferDims;
}

bool isRank2GmSubview(Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  return type && type.getRank() == 2 && getMemorySpace(type) == 0 &&
         memref.getDefiningOp<memref::SubViewOp>();
}

bool isContiguousRank2GmSubview(Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  auto subview = memref.getDefiningOp<memref::SubViewOp>();
  if (!type || type.getRank() != 2 || getMemorySpace(type) != 0 || !subview)
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

Value getRank2RowStride(ComputeLoweringContext &lowering, OpBuilder &builder,
                        Location loc, Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || type.getRank() != 2)
    return Value{};

  auto [strides, offset] = type.getStridesAndOffset();
  (void)offset;
  if (strides.size() == 2 && strides[0] != ShapedType::kDynamic)
    return builder.create<arith::ConstantIndexOp>(loc, strides[0]);

  Value root = memref;
  while (auto subview = root.getDefiningOp<memref::SubViewOp>())
    root = subview.getSource();

  auto rootType = dyn_cast<MemRefType>(root.getType());
  if (rootType && rootType.getRank() == 2)
    return getDynDim(lowering, builder, loc, root, 1);

  return getDynDim(lowering, builder, loc, memref, 1);
}

Value copyRank2GmSubviewRowsToVecin(
    ComputeLoweringContext &lowering, OpBuilder &builder, Location loc,
    Type elemType, Value srcMemref, Value bufferElemCount,
    SmallVectorImpl<OwnedQueueTensor> *ownedTensors) {
  unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  Value byteSize = builder.create<arith::MulIOp>(
      loc, bufferElemCount,
      builder.create<arith::ConstantIndexOp>(loc, elemBytes));
  Value vecinTbuf = builder.create<TBufOp>(
      loc, TBufType::get(lowering.mlirCtx, TPosition::VECIN));
  builder.create<TPipeInitBufferOp>(loc, lowering.ctx.pipe, vecinTbuf,
                                    byteSize);
  Value vecinQue = builder.create<QueueOp>(
      loc, QueueType::get(lowering.mlirCtx, TPosition::VECIN, 1));
  Value depth = builder.create<arith::ConstantOp>(
      loc, builder.getI32IntegerAttr(1));
  builder.create<TPipeInitQueueOp>(loc, lowering.ctx.pipe, vecinQue, depth,
                                   byteSize);
  Value lt = builder.create<TQueBindAllocTensorOp>(
      loc, LocalTensorType::get(elemType), vecinQue);

  Value srcGt = builder.create<GlobalTensorOp>(
      loc, GlobalTensorType::get(elemType));
  builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, srcMemref,
                                                /*size=*/Value{});
  Value rows = getDynDim(lowering, builder, loc, srcMemref, 0);
  Value cols = getDynDim(lowering, builder, loc, srcMemref, 1);
  Value srcRowStride = getRank2RowStride(lowering, builder, loc, srcMemref);
  if (!srcRowStride)
    return Value{};
  emitStridedGmToLocalCopy(builder, loc, elemType, lt, srcGt, rows, cols,
                           srcRowStride);

  builder.create<TQueBindEnqueTensorOp>(loc, vecinQue, lt);
  Value dequeued = builder.create<TQueBindDequeTensorOp>(
      loc, LocalTensorType::get(elemType), vecinQue);
  if (ownedTensors)
    rememberQueueRead(*ownedTensors, vecinQue, dequeued);
  return dequeued;
}

LogicalResult lowerScalarFallbackComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  OpBuilder &builder = lowering.builder;
  SmallVector<linalg::GenericOp> genericOps;
  funcOp.walk([&](linalg::GenericOp op) { genericOps.push_back(op); });

  for (linalg::GenericOp genOp : genericOps) {
    if (!isGmScalarLoopGeneric(genOp))
      continue;

    builder.setInsertionPoint(genOp);
    if (succeeded(
            lowerProjectedSuffixCopyToSegmentDataCopy(builder, genOp,
                                                      lowering.ctx.pipe))) {
      genOp.erase();
      continue;
    }

    if (failed(lowerGmGenericToScalarLoops(builder, genOp))) {
      genOp.emitError("failed to lower GM generic scalar loop");
      return failure();
    }
    genOp.erase();
  }

  return success();
}

} // namespace ascend
} // namespace mlir
