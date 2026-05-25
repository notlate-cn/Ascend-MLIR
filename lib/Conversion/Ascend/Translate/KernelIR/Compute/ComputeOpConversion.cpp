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
namespace afir {

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



struct IndexingMapAnalysis {
  enum class Kind {
    Identity,
    PureBroadcast,
    PureTranspose,
    BroadcastTranspose,
  };
  Kind kind;
  SmallVector<int64_t> permutation;
  SmallVector<int64_t> broadcastDims;
};

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

using OwnedQueueTensor = std::pair<Value, Value>;

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
                    SmallVectorImpl<OwnedQueueTensor> *ownedTensors = nullptr) {
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
                          SmallVectorImpl<OwnedQueueTensor> *ownedTensors =
                              nullptr) {
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
    SmallVectorImpl<OwnedQueueTensor> *ownedTensors = nullptr) {
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

} // namespace



void ComputeLoweringContext::copyAscendCUnitAttr(Operation *src,
                                                 Operation *dst) const {
  if (!src || !dst)
    return;
  if (auto unitAttr = src->getAttrOfType<StringAttr>(ascend::kAscendCUnitAttr))
    dst->setAttr(ascend::kAscendCUnitAttr, unitAttr);
}

Value ComputeLoweringContext::getEnclosingLoopStepBound(
    Value value, Operation *anchor) const {
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
}

Value ComputeLoweringContext::getSubviewSizeValue(OpBuilder &builder,
                                                  Location loc, Value memref,
                                                  unsigned dim) const {
  auto subviewOp = memref.getDefiningOp<memref::SubViewOp>();
  if (!subviewOp)
    return Value{};
  SmallVector<OpFoldResult> mixedSizes = subviewOp.getMixedSizes();
  if (dim >= mixedSizes.size())
    return Value{};
  OpFoldResult size = mixedSizes[dim];
  if (auto attr = size.dyn_cast<Attribute>())
    return builder.create<arith::ConstantIndexOp>(
        loc, cast<IntegerAttr>(attr).getInt());
  return size.get<Value>();
}

Value ComputeLoweringContext::computeProduct(OpBuilder &builder, Location loc,
                                             ArrayRef<Value> dims) const {
  Value totalElems;
  for (Value s : dims)
    totalElems = totalElems ? builder.create<arith::MulIOp>(loc, totalElems, s)
                            : s;
  if (!totalElems)
    totalElems = builder.create<arith::ConstantIndexOp>(loc, 1);
  return totalElems;
}

Value ComputeLoweringContext::dequeTensor(OpBuilder &builder, Location loc,
                                          Value queue, Type elemType) const {
  return builder.create<TQueBindDequeTensorOp>(loc,
                                               LocalTensorType::get(elemType),
                                               queue);
}

Value ComputeLoweringContext::allocTensor(OpBuilder &builder, Location loc,
                                          Value queue, Type elemType) const {
  return builder.create<TQueBindAllocTensorOp>(loc,
                                               LocalTensorType::get(elemType),
                                               queue);
}

Value ComputeLoweringContext::tbufTensor(OpBuilder &builder, Location loc,
                                         int64_t memorySpace,
                                         Type elemType) const {
  auto pos = static_cast<TPosition>(memorySpace > 0 ? memorySpace : 0);
  Value tbuf = builder.create<TBufOp>(loc, TBufType::get(this->mlirCtx, pos));
  return builder.create<TBufGetTensorOp>(loc, LocalTensorType::get(elemType),
                                         tbuf, /*len=*/Value{});
}

Value ComputeLoweringContext::subviewByteOffset(OpBuilder &builder,
                                                Location loc,
                                                Value memref) const {
  auto subviewOp = memref.getDefiningOp<memref::SubViewOp>();
  if (!subviewOp)
    return Value{};
  Value parent = subviewOp.getSource();
  auto parentType = cast<MemRefType>(parent.getType());
  if (parentType.getRank() != 2)
    return Value{};

  SmallVector<OpFoldResult> mixedOffsets = subviewOp.getMixedOffsets();
  Value rowStride;
  if (!ShapedType::isDynamic(parentType.getShape()[1]))
    rowStride = builder.create<arith::ConstantIndexOp>(
        loc, parentType.getShape()[1]);
  else
    rowStride = builder.create<memref::DimOp>(
        loc, parent, builder.create<arith::ConstantIndexOp>(loc, 1));

  auto toIndex = [&](OpFoldResult ofr) -> Value {
    if (auto attr = ofr.dyn_cast<Attribute>())
      return builder.create<arith::ConstantIndexOp>(
          loc, cast<IntegerAttr>(attr).getInt());
    return ofr.get<Value>();
  };
  Value off0 = toIndex(mixedOffsets[0]);
  Value off1 = toIndex(mixedOffsets[1]);

  Value linearElems = builder.create<arith::MulIOp>(loc, off0, rowStride);
  linearElems = builder.create<arith::AddIOp>(loc, linearElems, off1);
  unsigned elemBytes = parentType.getElementTypeBitWidth() / 8;
  Value bytesVal = builder.create<arith::ConstantIndexOp>(loc, elemBytes);
  return builder.create<arith::MulIOp>(loc, linearElems, bytesVal);
}

Value ComputeLoweringContext::tbufSlice(OpBuilder &builder, Location loc,
                                        Value memref, Value sizeElems,
                                        Value offsetBytes) const {
  Value tbuf = this->ctx.getTBuf(memref);
  if (!tbuf)
    return Value{};
  auto mrt = cast<MemRefType>(memref.getType());
  return builder.create<TBufGetWithOffsetOp>(
      loc, LocalTensorType::get(mrt.getElementType()), tbuf, sizeElems,
      offsetBytes);
}

Value ComputeLoweringContext::readTensor(OpBuilder &builder, Location loc,
                                         Value memref) const {
  auto mrt = cast<MemRefType>(memref.getType());
  if (this->ctx.getLiveTensor(memref)) {
    if (Value byteOff = subviewByteOffset(builder, loc, memref)) {
      Value sizeElems = computeElementCount(builder, loc, memref);
      if (Value t = tbufSlice(builder, loc, memref, sizeElems, byteOff))
        return t;
    }
    return this->ctx.getLiveTensor(memref);
  }
  if (Value q = this->ctx.getQueue(memref))
    return dequeTensor(builder, loc, q, mrt.getElementType());
  return tbufTensor(builder, loc, getMemorySpace(mrt), mrt.getElementType());
}

Value ComputeLoweringContext::writeTensor(OpBuilder &builder, Location loc,
                                          Value memref) const {
  auto mrt = cast<MemRefType>(memref.getType());
  if (Value byteOff = subviewByteOffset(builder, loc, memref)) {
    Value sizeElems = computeElementCount(builder, loc, memref);
    if (Value t = tbufSlice(builder, loc, memref, sizeElems, byteOff))
      return t;
  }
  if (Value q = this->ctx.getQueue(memref))
    return allocTensor(builder, loc, q, mrt.getElementType());
  return tbufTensor(builder, loc, getMemorySpace(mrt), mrt.getElementType());
}

scf::ForOp ComputeLoweringContext::getEnclosingFor(Operation *op) const {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (auto f = dyn_cast<scf::ForOp>(p))
      return f;
  return nullptr;
}

std::pair<Value, scf::ForOp>
ComputeLoweringContext::allocHoisted(Operation *op, Value queue, Type elemType,
                                      Location loc) {
  scf::ForOp forOp = getEnclosingFor(op);
  if (!forOp)
    return {allocTensor(builder, loc, queue, elemType), nullptr};
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(forOp);
  Value tensor = allocTensor(builder, loc, queue, elemType);
  return {tensor, forOp};
}

LogicalResult lowerTransposeComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  OpBuilder &builder = lowering.builder;
  // --- linalg.transpose ---
  SmallVector<linalg::TransposeOp> transposeOps;
  funcOp.walk([&](linalg::TransposeOp op) { transposeOps.push_back(op); });
  for (linalg::TransposeOp transposeOp : transposeOps) {
    FailureOr<TransposeLoweringSpec> spec =
        buildTransposeLoweringSpec(transposeOp);
    if (failed(spec))
      continue;
    TransposeLoweringPlan plan = planTransposeLowering(*spec);
    if (plan.kind == TransposeLoweringKind::Unsupported)
      continue;

    Value inMemref = transposeOp.getDpsInputOperand(0)->get();
    Value outMemref = transposeOp.getDpsInitOperand(0)->get();
    if (plan.kind == TransposeLoweringKind::ScalarMemRefLoop) {
      Location loc = transposeOp.getLoc();
      builder.setInsertionPoint(transposeOp);
      if (succeeded(lowerRank2GmTransposeToLocalDataCopy(
              builder, loc, inMemref, outMemref, spec->permutation,
              lowering.ctx.pipe))) {
        transposeOp.erase();
        continue;
      }
      if (failed(lowerTransposeToLoops(builder, loc, inMemref, outMemref,
                                       spec->permutation))) {
        transposeOp.emitError("failed to lower transpose scalar fallback");
        return failure();
      }
      transposeOp.erase();
      continue;
    }

    if (getMemorySpace(outMemref.getType()) <= 0)
      continue;

    Location loc = transposeOp.getLoc();
    builder.setInsertionPoint(transposeOp);
    Value srcLt = lowering.readTensor(builder, loc, inMemref);
    Value dstLt = lowering.writeTensor(builder, loc, outMemref);
    auto lowered = builder.create<TransposeOp>(loc, dstLt, srcLt);
    lowering.copyAscendCUnitAttr(transposeOp.getOperation(), lowered.getOperation());
    if (Value queue = lowering.ctx.getQueue(outMemref))
      builder.create<TQueBindEnqueTensorOp>(loc, queue, dstLt);
    transposeOp.erase();
  }

  return success();
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

LogicalResult lowerReductionComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  OpBuilder &builder = lowering.builder;
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
    ascend::backend::AscendBackendSupportMatrix matrix;
    ascend::backend::ComputeKind reductionKind =
        ascend::backend::classifyBackendReductionBody(genOp, matrix);
    if (reductionKind == ascend::backend::ComputeKind::Unknown)
      continue;

    auto findPrecedingFillInit = [](Value output, Operation *writer) -> Value {
      Operation *bestFill = nullptr;
      Value bestInit;
      for (Operation *user : output.getUsers()) {
        auto fillOp = dyn_cast<linalg::FillOp>(user);
        if (!fillOp || fillOp.getOutputs()[0] != output)
          continue;
        if (fillOp->getBlock() != writer->getBlock() ||
            !fillOp->isBeforeInBlock(writer))
          continue;
        if (!bestFill || bestFill->isBeforeInBlock(fillOp)) {
          bestFill = fillOp.getOperation();
          bestInit = fillOp.getInputs()[0];
        }
      }
      return bestInit;
    };

    auto buildNeutralIdentity =
        [&](ascend::backend::ComputeKind kind) -> Value {
      if (!isa<FloatType>(elemType))
        return Value{};

      double identity = 0.0;
      switch (kind) {
      case ascend::backend::ComputeKind::ReductionAdd:
        identity = 0.0;
        break;
      case ascend::backend::ComputeKind::ReductionMul:
        identity = 1.0;
        break;
      case ascend::backend::ComputeKind::ReductionMax:
        identity = -std::numeric_limits<double>::infinity();
        break;
      case ascend::backend::ComputeKind::ReductionMin:
        identity = std::numeric_limits<double>::infinity();
        break;
      default:
        return Value{};
      }
      return builder.create<arith::ConstantOp>(
          loc, builder.getFloatAttr(elemType, identity));
    };

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
        for (unsigned d = 0; d < iterRank; ++d)
          iterDimSizes[d] = getDynDim(lowering, builder, loc, inMemref, d);
        break;
      }
    }
    // Fall back: fill remaining parallel dims from output (output only covers
    // parallel dims, so only use it when the iterator type is parallel).
    {
      unsigned outDim = 0;
      for (unsigned d = 0; d < iterRank; ++d) {
        if (!iterDimSizes[d] && iterTypes[d] == utils::IteratorType::parallel)
          iterDimSizes[d] = getDynDim(lowering, builder, loc, outMemref, outDim++);
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
    Value totalElems = lowering.computeProduct(builder, loc, fullShape);
    // Build a VECCALC accumulator for the full shape.  This is the tensor
    // that will hold the element-wise intermediate results before reduction.
    Value accumLt = allocVeccalc(lowering, builder, loc, elemType, fullShape).second;

    // Initialize the expanded accumulator with the reduction identity. Prefer
    // the producer fill value when present, because linalg outs carries the
    // semantic init. Fall back to the neutral identity for legacy reductions
    // that arrive without an explicit fill in the same block.
    Value initVal = findPrecedingFillInit(outMemref, genOp.getOperation());
    if (!initVal)
      initVal = buildNeutralIdentity(reductionKind);
    if (initVal) {
      auto initDup =
          builder.create<DuplicateL2Op>(loc, accumLt, initVal, totalElems);
      lowering.copyAscendCUnitAttr(genOp.getOperation(), initDup.getOperation());
      builder.create<PipeBarrierOp>(loc,
                                    PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));
    }

    // Promote each input to a local_tensor of shape `fullShape`.
    SmallVector<OwnedQueueTensor> ownedInputTensors;
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
                    getDynDim(lowering, builder, loc, inMemref, srcDimIdx++)));
          else
            srcShapeVals.push_back(
                builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
        }
        Value srcLt = lowering.readTensor(builder, loc, inMemref);
        if (Value q = lowering.ctx.getQueue(inMemref))
          if (!lowering.ctx.getLiveTensor(inMemref))
            rememberQueueRead(ownedInputTensors, q, srcLt);
        auto [bcastTbuf, bcastLt] =
            allocVeccalc(lowering, builder, loc, elemType, fullShape);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, bcastLt, srcLt,
            dstShapeVals, srcShapeVals,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        lowering.copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
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
          srcDims.push_back(getDynDim(lowering, builder, loc, inMemref, d));
        Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
        for (Value d : srcDims)
          srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
        SmallVector<Value> srcBufferDims =
            getBufferDimSizes(lowering, srcDims, genOp.getOperation());
        Value srcBufferElemCount =
            lowering.computeProduct(builder, loc, srcBufferDims);
        Value srcGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(elemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                       /*size=*/Value{});
        Value srcLt =
            copyGmToVecinScalar(lowering, builder, loc, elemType, srcGt,
                                srcElemCount, srcBufferElemCount,
                                &ownedInputTensors);
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
            allocVeccalc(lowering, builder, loc, elemType, fullShape);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, bcastLt, srcLt,
            dstShapeVals, srcShapeVals,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        lowering.copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
        inputLts[i] = bcastLt;
      } else if (inMs == 0 /*GM*/) {
        // GM input at full rank: copy via VECIN TQue (simulator requires
        // DataCopy to go through TQue, not directly to VECCALC TBuf).
        Value srcGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(elemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                       /*size=*/Value{});
        inputLts[i] =
            copyGmToVecin(lowering, builder, loc, elemType, srcGt, totalElems,
                          totalElems, &ownedInputTensors);
      } else {
        // Already VECIN or VECCALC — use readTensor as-is.
        inputLts[i] = lowering.readTensor(builder, loc, inMemref);
        if (Value q = lowering.ctx.getQueue(inMemref))
          if (!lowering.ctx.getLiveTensor(inMemref))
            rememberQueueRead(ownedInputTensors, q, inputLts[i]);
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
              allocVeccalc(lowering, builder, loc, elemType, fullShape);
          auto dupOp = builder.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), totalElems);
          lowering.copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
          builder.create<PipeBarrierOp>(
              loc, PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));
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
            allocVeccalc(lowering, builder, loc, elemType, fullShape);
        valToLt[result] = tmpLt;
        return tmpLt;
      };

      if (isa<arith::ConstantOp>(bodyOp))
        continue;

      using namespace mlir::afir::ascend::backend;
      const ElementwiseBodyOpEntry *entry =
          lookupElementwiseBodyOp(bodyOp.getName().getStringRef());
      if (!entry)
        continue;

      if (entry->unaryEmitter) {
        Value src = resolve(bodyOp.getOperand(0));
        if (!src) continue;
        Value dst = chooseDst(bodyOp.getResult(0));
        entry->unaryEmitter(builder, loc, dst, src, totalElems);
        lowering.copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        if (dst == accumLt) valToLt[bodyOp.getResult(0)] = accumLt;
      } else if (entry->binaryEmitter) {
        Value lhs = resolve(bodyOp.getOperand(0));
        Value rhs = resolve(bodyOp.getOperand(1));
        if (!lhs || !rhs) continue;
        Value dst = chooseDst(bodyOp.getResult(0));
        entry->binaryEmitter(builder, loc, dst, lhs, rhs, totalElems);
        lowering.copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        if (dst == accumLt) valToLt[bodyOp.getResult(0)] = accumLt;
      }
    }

    freeOwnedQueueTensors(builder, loc, ownedInputTensors);

    // ------------------------------------------------------------------
    // Step 3: Reduce the accumulated VECCALC to the output VECOUT tensor.
    //
    // For a 2D iteration [parallel_dim, reduction_dim] with AR layout:
    //   reduce_sum_2d_l2(vecoutLt, accumLt, AR, no_tmp)
    // ------------------------------------------------------------------
    Value vecoutLt = lowering.writeTensor(builder, loc, outMemref);
    auto layoutAttr = ReduceLayoutAttr::get(lowering.mlirCtx, ReduceLayout::AR);
    if (reductionKind == ascend::backend::ComputeKind::ReductionMax) {
      auto reduceOp = builder.create<ReduceMax2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                      /*sharedTmpBuffer=*/Value{});
      lowering.copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    } else if (reductionKind == ascend::backend::ComputeKind::ReductionMin) {
      auto reduceOp = builder.create<ReduceMin2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                      /*sharedTmpBuffer=*/Value{});
      lowering.copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    } else if (reductionKind == ascend::backend::ComputeKind::ReductionMul) {
      auto reduceOp = builder.create<ReduceProd2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                       /*sharedTmpBuffer=*/Value{});
      lowering.copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    } else {
      auto reduceOp = builder.create<ReduceSum2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                      /*sharedTmpBuffer=*/Value{});
      lowering.copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    }

    // Enqueue vecout if it has a queue (VECOUT path).
    if (Value q = lowering.ctx.getQueue(outMemref))
      builder.create<TQueBindEnqueTensorOp>(loc, q, vecoutLt);

    genOp.erase();
  }

  return success();
}

LogicalResult lowerParallelGenericComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  OpBuilder &builder = lowering.builder;
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
  // Concat semantics are implicitly handled: the VECOUT->GM copy op from
  // memory realization targets a memref subview of the output buffer with the
  // correct byte offset, so Op1 and Op2 results land at the right positions in
  // the concatenated output without any asc.concat op.
  SmallVector<linalg::GenericOp> parallelGenericOps;
  funcOp.walk([&](linalg::GenericOp op) {
    auto iterTypes = op.getIteratorTypesArray();
    bool allParallel = llvm::all_of(iterTypes, [](utils::IteratorType t) {
      return t == utils::IteratorType::parallel;
    });
    if (allParallel && op.getNumDpsInits() == 1)
      parallelGenericOps.push_back(op);
  });

  // Helper: detect a standalone generic transpose supported by the backend.
  auto isTransposeGeneric = [](linalg::GenericOp op) -> bool {
    FailureOr<TransposeLoweringSpec> spec = buildTransposeLoweringSpec(op);
    return succeeded(spec) &&
           planTransposeLowering(*spec).kind ==
               TransposeLoweringKind::AscendCSimple2D;
  };

  // Helper: detect index_select gather (column gather).
  // Stamped by ascend-kernelize: {gather_dim = N : i64} attribute.
  auto isIndexSelectGeneric = [](linalg::GenericOp op) -> bool {
    return op->hasAttr(ascend::kGatherDimAttr);
  };

  // Helper: detect embedding gather (row gather).
  // Stamped by ascend-kernelize: {embedding_dim = N : i64} attribute.
  auto isEmbeddingGeneric = [](linalg::GenericOp op) -> bool {
    return op->hasAttr(ascend::kEmbeddingDimAttr);
  };
  (void)isEmbeddingGeneric; // reserved for future use

  for (linalg::GenericOp genOp : parallelGenericOps) {
    Value outMemref = genOp.getDpsInitOperand(0)->get();
    int64_t outMs   = getMemorySpace(outMemref.getType());
    if (outMs == 0 && isPureYieldGeneric(genOp)) {
      builder.setInsertionPoint(genOp);
      if (failed(lowerPureYieldGenericToLoops(builder, genOp))) {
        genOp.emitError("failed to lower pure-yield generic copy");
        return failure();
      }
      genOp.erase();
      continue;
    }
    if (outMs == 0 && isGmAllParallelGeneric(genOp)) {
      builder.setInsertionPoint(genOp);
      if (failed(lowerAllParallelGenericToLoops(builder, genOp))) {
        genOp.emitError("failed to lower GM all-parallel generic");
        return failure();
      }
      genOp.erase();
      continue;
    }
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
        if (candidate->hasAttr(ascend::kGatherDimAttr) ||
            candidate->hasAttr(ascend::kEmbeddingDimAttr))
          continue;
        if (candidate.getDpsInitOperand(0)->get() == dataMemref) {
          preOp = candidate;
          break;
        }
      }

      // Find post-gather op: parallel generic that has outMemref as one of its inputs
      linalg::GenericOp postOp;
      for (linalg::GenericOp candidate : parallelGenericOps) {
        if (candidate == genOp) continue;
        if (candidate->hasAttr(ascend::kGatherDimAttr) ||
            candidate->hasAttr(ascend::kEmbeddingDimAttr))
          continue;
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
      Value tbM  = getDynDim(lowering, builder, loc, outMemref, 0);
      Value dimN = getDynDim(lowering, builder, loc, dataMemref, 1);
      Value dimK = getDynDim(lowering, builder, loc, outMemref, 1);

      unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;

      // Get indices as a local_tensor.
      // If indices are in VECIN (ms=9), deque from queue.
      // If indices are in GM (ms=0), copy into VECCALC first.
      auto idxMrt = cast<MemRefType>(indicesMemref.getType());
      Type idxElemType = idxMrt.getElementType();
      Value idxCount = getDynDim(lowering, builder, loc, indicesMemref, 0);
      Value indicesLt;
      int64_t idxMs = getMemorySpace(indicesMemref.getType());
      if (idxMs == 9 /*VECIN*/ || idxMs == 11 /*VECCALC*/) {
        indicesLt = lowering.readTensor(builder, loc, indicesMemref);
      } else {
        // GM: copy indices into a fresh VECCALC buffer.
        SmallVector<Value> idxDims = {idxCount};
        auto [idxTbuf, idxLt] =
            allocVeccalc(lowering, builder, loc, idxElemType, idxDims);
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

      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value one  = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value outTbuf = lowering.ctx.getTBuf(outMemref);
      if (!outTbuf) {
        genOp.emitError("missing TBuf for gather output buffer");
        return failure();
      }

      // Byte size for a single output row (K * elemBytes).
      Value elemBytesVal =
          builder.create<arith::ConstantIndexOp>(loc, elemBytes);
      Value outBytesPerRow =
          builder.create<arith::MulIOp>(loc, dimK, elemBytesVal);

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

      // Allocate a VECCALC buffer for one data row plus one Gather vector
      // window of padding. AscendC Gather offsets are byte offsets into UB; on
      // hardware the vector instruction can still bounds-check a full element
      // window past the logical index. Padding keeps max-index gathers inside
      // the registered UB buffer.
      unsigned gatherPadElems =
          elemBytes <= 2 ? 256 : (elemBytes <= 4 ? 128 : 64);
      unsigned gatherChunkElems =
          elemBytes <= 2 ? 128 : (elemBytes <= 4 ? 64 : 32);
      Value gatherPadElemsVal =
          builder.create<arith::ConstantIndexOp>(loc, gatherPadElems);
      Value paddedDimN =
          builder.create<arith::AddIOp>(loc, dimN, gatherPadElemsVal);
      Value paddedDimK =
          ceilToMultipleIndex(builder, loc, dimK, gatherChunkElems);
      Value dataRowQueue = builder.create<QueueOp>(
          loc, QueueType::get(lowering.mlirCtx, TPosition::VECIN, 1));
      Value rowBytes = builder.create<arith::MulIOp>(
          loc, paddedDimN,
          builder.create<arith::ConstantIndexOp>(loc, elemBytes));
      Value dataRowQueueDepth =
          builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
      builder.create<TPipeInitQueueOp>(loc, lowering.ctx.pipe, dataRowQueue,
                                       dataRowQueueDepth, rowBytes);

      // Gather and post-gather vector ops must run in VECCALC.  Real hardware
      // rejects some VEC reads/writes against VECOUT TBuf slices that the
      // simulator accepts, so rows are copied to VECOUT only after vector work.
      auto gatheredRowAlloc =
          allocVeccalc(lowering, builder, loc, elemType, SmallVector<Value>{paddedDimK});
      Value gatheredRowLt = gatheredRowAlloc.second;
      auto gatherSourceRowAlloc =
          allocVeccalc(lowering, builder, loc, elemType, SmallVector<Value>{paddedDimN});
      Value gatherSourceRowLt = gatherSourceRowAlloc.second;

      // Pre-op temporaries are reused for every row. Initializing these TPipe
      // buffers inside the row loop exhausts simulator buffer bookkeeping for
      // larger M even though the loop is sequential.
      Value preProcessedRowLt;
      Value preDimN_i32;
      llvm::SmallDenseMap<Value, Value> preInvariantConstLt;
      if (preOp) {
        auto [procTbuf, procLt] =
            allocVeccalc(lowering, builder, loc, elemType, SmallVector<Value>{paddedDimN});
        (void)procTbuf;
        preProcessedRowLt = procLt;
        preDimN_i32 =
            builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), dimN);

        Block &preBody = *preOp.getBody();
        for (auto &bodyOp : preBody.without_terminator()) {
          for (Value operand : bodyOp.getOperands()) {
            auto constOp = operand.getDefiningOp<arith::ConstantOp>();
            if (!constOp || preInvariantConstLt.contains(operand))
              continue;
            auto [dupTbuf, dupLt] =
                allocVeccalc(lowering, builder, loc, elemType, SmallVector<Value>{dimN});
            (void)dupTbuf;
            auto dupOp = builder.create<DuplicateL2Op>(
                loc, dupLt, constOp.getResult(), preDimN_i32);
            lowering.copyAscendCUnitAttr(preOp.getOperation(), dupOp.getOperation());
            preInvariantConstLt[operand] = dupLt;
          }
        }
      }

      Value fusedBodyDimK_i32;
      llvm::SmallDenseMap<Value, Value> fusedBodyInvariantConstLt;
      llvm::SmallDenseMap<Value, Value> fusedBodyInvariantInputLt;
      if (!postOp) {
        fusedBodyDimK_i32 =
            builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), dimK);
        Block &gatherBody = *genOp.getBody();
        unsigned numBodyIns = static_cast<unsigned>(genOp.getNumDpsInputs());

        for (unsigned argNum = 1; argNum < numBodyIns; ++argNum) {
          BlockArgument blockArg = gatherBody.getArgument(argNum);
          Value argMemref = genOp.getDpsInputOperand(argNum)->get();
          if (argMemref == outMemref)
            continue;
          int64_t argMs = getMemorySpace(argMemref.getType());
          if (argMs == 0) {
            auto argMrt = cast<MemRefType>(argMemref.getType());
            Type argElem = argMrt.getElementType();
            Value argGt = builder.create<GlobalTensorOp>(
                loc, GlobalTensorType::get(argElem));
            builder.create<GlobalTensorSetGlobalBufferOp>(
                loc, argGt, argMemref, /*size=*/Value{});
            Value argCount = computeElementCount(builder, loc, argMemref);
            fusedBodyInvariantInputLt[blockArg] =
                copyGmToVeccalc(lowering, builder, loc, argElem, argGt, argCount);
          }
        }

        bool pastLoad = false;
        for (auto &bodyOp : gatherBody.without_terminator()) {
          if (isa<memref::LoadOp>(bodyOp)) {
            pastLoad = true;
            continue;
          }
          if (!pastLoad)
            continue;
          for (Value operand : bodyOp.getOperands()) {
            auto constOp = operand.getDefiningOp<arith::ConstantOp>();
            if (!constOp || fusedBodyInvariantConstLt.contains(operand))
              continue;
            auto [dupTbuf, dupLt] =
                allocVeccalc(lowering, builder, loc, elemType, SmallVector<Value>{dimK});
            (void)dupTbuf;
            auto dupOp = builder.create<DuplicateL2Op>(
                loc, dupLt, constOp.getResult(), fusedBodyDimK_i32);
            lowering.copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
            fusedBodyInvariantConstLt[operand] = dupLt;
          }
        }
      }

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
              Value procLt = preProcessedRowLt;
              Value dimN_i32 = preDimN_i32;

              Block &preBody = *preOp.getBody();
              llvm::SmallDenseMap<Value, Value> preValToLt;

              auto preResolve = [&](Value v) -> Value {
                if (auto ba = dyn_cast<BlockArgument>(v)) {
                  if (ba.getArgNumber() == 0) return dataRowLt;
                  return procLt;
                }
                auto it = preValToLt.find(v);
                if (it != preValToLt.end()) return it->second;
                if (v.getDefiningOp<arith::ConstantOp>()) {
                  auto constIt = preInvariantConstLt.find(v);
                  if (constIt == preInvariantConstLt.end())
                    return Value{};
                  Value dupLt = constIt->second;
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
                    lowering.copyAscendCUnitAttr(preOp.getOperation(), maxOp2.getOperation());
                    preValToLt[maxOp.getResult()] = procLt;
                  }
                } else if (auto addOp2 = dyn_cast<arith::AddFOp>(bodyOp)) {
                  Value lhs = preResolve(addOp2.getLhs()), rhs = preResolve(addOp2.getRhs());
                  if (lhs && rhs) {
                    auto addOp3 = b.create<AddL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                    lowering.copyAscendCUnitAttr(preOp.getOperation(), addOp3.getOperation());
                    preValToLt[addOp2.getResult()] = procLt;
                  }
                } else if (auto mulOp2 = dyn_cast<arith::MulFOp>(bodyOp)) {
                  Value lhs = preResolve(mulOp2.getLhs()), rhs = preResolve(mulOp2.getRhs());
                  if (lhs && rhs) {
                    auto mulOp3 = b.create<MulL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                    lowering.copyAscendCUnitAttr(preOp.getOperation(), mulOp3.getOperation());
                    preValToLt[mulOp2.getResult()] = procLt;
                  }
                }
              }
              processedRowLt = procLt;
            } else {
              emitLocalToLocalScalarCopy(b, forLoc, elemType,
                                         gatherSourceRowLt, dataRowLt, dimN);
              processedRowLt = gatherSourceRowLt;
            }
            emitLocalTensorZeroPad(b, forLoc, elemType, processedRowLt, dimN,
                                   paddedDimN);

            // Step 2: gather_l2(dst[K], src[N], indices, srcBase=0, count=K)
            Value dstByteOff =
                b.create<arith::MulIOp>(forLoc, rowIdx, outBytesPerRow);
            Value dstRowLt = b.create<TBufGetWithOffsetOp>(
                forLoc, LocalTensorType::get(elemType), outTbuf,
                dimK, dstByteOff);
            b.create<GatherL2Op>(forLoc, gatheredRowLt, processedRowLt,
                                 indicesLt, srcBaseAddr, dimK_i32);

            // Step 3: If post-op exists (e.g. add bias), apply it on gatheredRowLt.
            // If no post-op, walk the gather body itself for any arith ops that
            // appear after the memref.load (from upstream fusion). This handles
            // the case where relu + add were fused into the gather body by
            // ascend-kernelize gather elementwise fusion before bufferization.
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
                  int64_t argMs = getMemorySpace(argMemref.getType());
                  if (argMs == 0) {
                    auto argMrt = cast<MemRefType>(argMemref.getType());
                    Type argElem = argMrt.getElementType();
                    Value argGt = b.create<GlobalTensorOp>(
                        forLoc, GlobalTensorType::get(argElem));
                    b.create<GlobalTensorSetGlobalBufferOp>(
                        forLoc, argGt, argMemref, /*size=*/Value{});
                    Value argCount = computeElementCount(b, forLoc, argMemref);
                    return copyGmToVeccalc(lowering, b, forLoc, argElem, argGt, argCount);
                  }
                  // Other on-chip inputs (bias, etc.) — read their tensor.
                  return lowering.readTensor(b, forLoc, argMemref);
                }
                auto it = postValToLt.find(v);
                if (it != postValToLt.end()) return it->second;
                if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                  auto [dupTbuf3, dupLt] = allocVeccalc(lowering, b, forLoc, elemType,
                                                         SmallVector<Value>{dimK});
                  auto dupOp3 = b.create<DuplicateL2Op>(forLoc, dupLt, constOp.getResult(), dimK_i32v);
                  lowering.copyAscendCUnitAttr(postOp.getOperation(), dupOp3.getOperation());
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
                    lowering.copyAscendCUnitAttr(postOp.getOperation(), addOp4.getOperation());
                    postValToLt[addOp3.getResult()] = gatheredRowLt;
                  }
                } else if (auto mulOp3 = dyn_cast<arith::MulFOp>(bodyOp)) {
                  Value lhs = postResolve(mulOp3.getLhs()), rhs = postResolve(mulOp3.getRhs());
                  if (lhs && rhs) {
                    auto mulOp4 = b.create<MulL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    lowering.copyAscendCUnitAttr(postOp.getOperation(), mulOp4.getOperation());
                    postValToLt[mulOp3.getResult()] = gatheredRowLt;
                  }
                } else if (auto maxOp3 = dyn_cast<arith::MaximumFOp>(bodyOp)) {
                  Value lhs = postResolve(maxOp3.getLhs()), rhs = postResolve(maxOp3.getRhs());
                  if (lhs && rhs) {
                    auto maxOp4 = b.create<MaxL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    lowering.copyAscendCUnitAttr(postOp.getOperation(), maxOp4.getOperation());
                    postValToLt[maxOp3.getResult()] = gatheredRowLt;
                  }
                }
              }
            } else {
              // Walk the fused gather body for arith ops that appear after
              // the memref.load (these were inlined by gather elementwise fusion).
              // Block args:
              //   arg0 = indices element (i64, skip)
              //   arg1..argN-2 = extra ins (bias etc.)
              //   argN-1 = out init (skip, use gatheredRowLt instead)
              Value dimK_i32v = fusedBodyDimK_i32;
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
                  if (auto inputIt = fusedBodyInvariantInputLt.find(ba);
                      inputIt != fusedBodyInvariantInputLt.end()) {
                    bodyValToLt[v] = inputIt->second;
                    return inputIt->second;
                  }
                  // Extra ins (bias, etc.) at argNum=1..numBodyIns-1
                  Value argMemref = genOp.getDpsInputOperand(argNum)->get();
                  int64_t argMs = getMemorySpace(argMemref.getType());
                  if (argMs > 0) {
                    // On-chip: use readTensor directly.
                    Value lt = lowering.readTensor(b, forLoc, argMemref);
                    bodyValToLt[v] = lt;
                    return lt;
                  }
                  // GM: copy to VECCALC for vector ops. The CANN translation
                  // rewrites the narrow gather+bias Add to read the copied
                  // scalar source directly from GM in the simulator.
                  Value argCount = computeElementCount(b, forLoc, argMemref);
                  auto argMrt = cast<MemRefType>(argMemref.getType());
                  Type argElem = argMrt.getElementType();
                  Value argGt = b.create<GlobalTensorOp>(forLoc, GlobalTensorType::get(argElem));
                  b.create<GlobalTensorSetGlobalBufferOp>(forLoc, argGt, argMemref,
                                                           /*size=*/Value{});
                  Value argLt =
                      copyGmToVeccalc(lowering, b, forLoc, argElem, argGt, argCount);
                  bodyValToLt[v] = argLt;
                  return argLt;
                }
                if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                  auto constIt = fusedBodyInvariantConstLt.find(v);
                  if (constIt == fusedBodyInvariantConstLt.end())
                    return Value{};
                  Value dupLt = constIt->second;
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
                    lowering.copyAscendCUnitAttr(genOp.getOperation(), addOp5.getOperation());
                    bodyValToLt[addOp4.getResult()] = gatheredRowLt;
                  }
                } else if (auto maxOp4 = dyn_cast<arith::MaximumFOp>(op)) {
                  Value lhs = bodyResolve(maxOp4.getLhs()),
                        rhs = bodyResolve(maxOp4.getRhs());
                  if (lhs && rhs) {
                    auto maxOp5 = b.create<MaxL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    lowering.copyAscendCUnitAttr(genOp.getOperation(), maxOp5.getOperation());
                    bodyValToLt[maxOp4.getResult()] = gatheredRowLt;
                  }
                } else if (auto mulOp4 = dyn_cast<arith::MulFOp>(op)) {
                  Value lhs = bodyResolve(mulOp4.getLhs()),
                        rhs = bodyResolve(mulOp4.getRhs());
                  if (lhs && rhs) {
                    auto mulOp5 = b.create<MulL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    lowering.copyAscendCUnitAttr(genOp.getOperation(), mulOp5.getOperation());
                    bodyValToLt[mulOp4.getResult()] = gatheredRowLt;
                  }
                }
              }
            }

            emitLocalToLocalScalarCopy(b, forLoc, elemType, dstRowLt,
                                       gatheredRowLt, dimK);
            b.create<TQueBindFreeTensorOp>(forLoc, dataRowQueue, dataRowLt);
            b.create<scf::YieldOp>(forLoc);
          });

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

      Value srcLt = lowering.readTensor(builder, loc, inMemref);
      Value dstLt = lowering.writeTensor(builder, loc, outMemref);
      builder.create<TransposeOp>(loc, dstLt, srcLt);

      if (Value q = lowering.ctx.getQueue(outMemref))
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
          iterDimSizes[d] = getDynDim(lowering, builder, loc, inMemref, d);
        break;
      }
    }
    // Fall back: fill remaining dims from output (all parallel, same rank).
    for (unsigned d = 0; d < iterRank; ++d)
      if (!iterDimSizes[d])
        iterDimSizes[d] = getDynDim(lowering, builder, loc, outMemref, d);

    // totalElems is the actual element count for this tile.  Buffer
    // allocation uses the enclosing loop-step upper bound so tail iterations
    // reuse one max-sized queue/tbuf instead of repeatedly InitBuffer-ing.
    Value totalElems = lowering.computeProduct(builder, loc, iterDimSizes);
    SmallVector<Value> bufferDimSizes =
        getBufferDimSizes(lowering, iterDimSizes, genOp.getOperation());
    Value bufferTotalElems = lowering.computeProduct(builder, loc, bufferDimSizes);

    Value outQueue = lowering.ctx.getQueue(outMemref);
    Value accumLt;
    if (!outQueue) {
      // Allocate the shared VECCALC accumulator for intermediate results.
      auto [accumTbuf, veccalcAccumLt] =
          allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
      accumLt = veccalcAccumLt;
    }

    // ---- Step 1: Promote each input to a VECCALC local_tensor ----
    SmallVector<OwnedQueueTensor> ownedInputTensors;
    SmallVector<Value> inputLts(numInputs);
    for (unsigned i = 0; i < numInputs; ++i) {
      Value inMemref = genOp.getDpsInputOperand(i)->get();
      AffineMap inMap = maps[i];
      int64_t inMs    = getMemorySpace(inMemref.getType());
      IndexingMapAnalysis analysis = analyzeIndexingMap(inMap, iterRank);

      switch (analysis.kind) {
      case IndexingMapAnalysis::Kind::Identity: {
        if (inMs == 0 /*GM*/) {
          // GM rank-2 subviews with partial inner tiles are not contiguous in GM.
          // Copy them row-by-row into the compact local tile used by vector ops.
          if (isRank2GmSubview(inMemref) &&
              !isContiguousRank2GmSubview(inMemref)) {
            inputLts[i] = copyRank2GmSubviewRowsToVecin(lowering,
                builder, loc, elemType, inMemref, bufferTotalElems,
                &ownedInputTensors);
            if (!inputLts[i]) {
              genOp.emitError("failed to lower rank-2 GM subview input copy");
              return failure();
            }
          } else {
            // GM input at full rank: copy via VECIN TQue.
            Value srcGt = builder.create<GlobalTensorOp>(
                loc, GlobalTensorType::get(elemType));
            builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                           /*size=*/Value{});
            inputLts[i] =
                copyGmToVecin(lowering, builder, loc, elemType, srcGt, totalElems,
                              bufferTotalElems, &ownedInputTensors);
          }
        } else {
          inputLts[i] = lowering.readTensor(builder, loc, inMemref);
          if (Value q = lowering.ctx.getQueue(inMemref))
            if (!lowering.ctx.getLiveTensor(inMemref))
              rememberQueueRead(ownedInputTensors, q, inputLts[i]);
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
                  getDynDim(lowering, builder, loc, inMemref, srcDimIdx++)));
            else
              srcShapeVals.push_back(
                  builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
          }
          Value srcLt = lowering.readTensor(builder, loc, inMemref);
          if (Value q = lowering.ctx.getQueue(inMemref))
            if (!lowering.ctx.getLiveTensor(inMemref))
              rememberQueueRead(ownedInputTensors, q, srcLt);
          auto [bcastTbuf, bcastLt] =
              allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
          auto bcastOp = builder.create<BroadcastL2Op>(
              loc, bcastLt, srcLt,
              dstShapeVals, srcShapeVals,
              builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
          lowering.copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
          inputLts[i] = bcastLt;
        } else {
          // broadcast from GM: copy via VECIN TQue first, then broadcast_l2.
          SmallVector<Value> srcDims;
          for (unsigned d = 0; d < srcRank; ++d)
            srcDims.push_back(getDynDim(lowering, builder, loc, inMemref, d));
          Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (Value d : srcDims)
            srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
          SmallVector<Value> srcBufferDims =
              getBufferDimSizes(lowering, srcDims, genOp.getOperation());
          Value srcBufferElemCount =
              lowering.computeProduct(builder, loc, srcBufferDims);
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          Value srcLt =
              copyGmToVecinScalar(lowering, builder, loc, elemType, srcGt,
                                  srcElemCount, srcBufferElemCount,
                                  &ownedInputTensors);
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
              allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
          auto bcastOp = builder.create<BroadcastL2Op>(
              loc, bcastLt, srcLt,
              dstShapeVals, srcShapeVals,
              builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
          lowering.copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
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
            copyGmToVecin(lowering, builder, loc, elemType, srcGt, srcElemCount,
                          srcElemCount, &ownedInputTensors);

        auto [transpTbuf, transpLt] =
            allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
        auto transposeOp = builder.create<TransposeOp>(loc, transpLt, srcVecinLt);
        lowering.copyAscendCUnitAttr(genOp.getOperation(), transposeOp.getOperation());
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
          srcDimsVals.push_back(getDynDim(lowering, builder, loc, inMemref, d));

        Value srcVecinLt;
        if (inMs == 9 /*VECIN*/) {
          srcVecinLt = lowering.readTensor(builder, loc, inMemref);
          if (Value q = lowering.ctx.getQueue(inMemref))
            if (!lowering.ctx.getLiveTensor(inMemref))
              rememberQueueRead(ownedInputTensors, q, srcVecinLt);
        } else {
          Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (Value d : srcDimsVals)
            srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          srcVecinLt =
              copyGmToVecin(lowering, builder, loc, elemType, srcGt, srcElemCount,
                            srcElemCount, &ownedInputTensors);
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
            allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, finalLt, srcVecinLt,
            bcastDstShape, bcastSrcShape,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        lowering.copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
        inputLts[i] = finalLt;
        break;
      }
      } // end switch
    }

    if (outQueue)
      accumLt = lowering.allocTensor(builder, loc, outQueue, elemType);

    // ---- Step 2: Walk body and inline arith ops onto VECCALC tensors ----
    Block &bodyBlock = *genOp.getBody();
    unsigned numBodyArgs = bodyBlock.getNumArguments();
    SmallVector<Value> argToLt(numBodyArgs);
    for (unsigned i = 0; i < numInputs; ++i)
      argToLt[i] = inputLts[i];
    argToLt[numInputs] = accumLt;

    llvm::SmallDenseMap<Value, Value> valToLt;
    bool accumLtHasBodyValue = false;
    auto emitAccumReadAfterWriteBarrier = [&](bool readsAccumLt) {
      if (!accumLtHasBodyValue || !readsAccumLt)
        return;
      builder.create<PipeBarrierOp>(
          loc, PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));
    };
    for (auto &bodyOp : bodyBlock.without_terminator()) {
      auto resolve = [&](Value v) -> Value {
        if (auto ba = dyn_cast<BlockArgument>(v))
          return argToLt[ba.getArgNumber()];
        auto it = valToLt.find(v);
        if (it != valToLt.end()) return it->second;
        // Scalar constant? Fill a fresh VECCALC with duplicate_l2.
        if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
          auto [dupTbuf, dupLt] =
              allocVeccalc(lowering, builder, loc, elemType, bufferDimSizes);
          auto dupOp = builder.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), totalElems);
          lowering.copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
          builder.create<PipeBarrierOp>(
              loc, PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));
          valToLt[v] = dupLt;
          return dupLt;
        }
        return Value{};
      };

      using namespace mlir::afir::ascend::backend;
      const ElementwiseBodyOpEntry *entry =
          lookupElementwiseBodyOp(bodyOp.getName().getStringRef());
      if (!entry) continue;

      if (entry->unaryEmitter) {
        Value src = resolve(bodyOp.getOperand(0));
        if (!src) continue;
        emitAccumReadAfterWriteBarrier(src == accumLt);
        entry->unaryEmitter(builder, loc, accumLt, src, totalElems);
        lowering.copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        valToLt[bodyOp.getResult(0)] = accumLt;
        accumLtHasBodyValue = true;
      } else if (entry->binaryEmitter) {
        Value lhs = resolve(bodyOp.getOperand(0));
        Value rhs = resolve(bodyOp.getOperand(1));
        if (!lhs || !rhs) continue;
        emitAccumReadAfterWriteBarrier(lhs == accumLt || rhs == accumLt);
        entry->binaryEmitter(builder, loc, accumLt, lhs, rhs, totalElems);
        lowering.copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        valToLt[bodyOp.getResult(0)] = accumLt;
        accumLtHasBodyValue = true;
      }
    }

    if (!ownedInputTensors.empty())
      builder.create<PipeBarrierOp>(loc, PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));
    freeOwnedQueueTensors(builder, loc, ownedInputTensors);

    // ---- Step 3: Write accumulator to output buffer ----
    // No reduction needed (all-parallel). The compute result is in accumLt
    // (a VECCALC tbuf). We need to deliver it to the output buffer:
    //
    //   VECOUT (ms=10): alloc from queue, use AddL2 to copy accumLt→vecoutLt
    //                   (add_l2(dst, src, zero_tbuf, count) would need a zero
    //                    tensor; instead use the queue alloc tensor directly and
    //                    simply enqueue accumLt if the queue accepts VECCALC).
    //                   Simplest: treat the VECCALC accumLt as the enqueue source
    //                   and let the downstream DataMovementConversion handle writeback.
    //   VECCALC (ms=11): accumLt already holds the result; no copy needed.
    //
    // Key insight: the epilogue memref.copy (VECOUT->GM) from memory
    // realization is converted by DataMovementConversion into a data_copy_l2 with
    // the correct subview offset, so the concat position is preserved
    // automatically. We just need to enqueue the result tensor.
    if (outQueue) {
      // The queue expects a tensor allocated from the same queue.  Real
      // hardware is stricter than the simulator here; enqueueing a VECCALC
      // tbuf tensor into a VECOUT queue can surface as UB/MTE faults.
      builder.create<TQueBindEnqueTensorOp>(loc, outQueue, accumLt);
    }
    // If outMemref has no queue (VECCALC alloc without a queue), the result
    // already resides in the VECCALC tbuf and will be consumed by the next op.

    genOp.erase();
  }

  return success();
}

LogicalResult lowerGatherCompute(ComputeLoweringContext &lowering,
                                 linalg::GenericOp genOp,
                                 ArrayRef<linalg::GenericOp> parallelGenericOps) {
  (void)lowering;
  (void)genOp;
  (void)parallelGenericOps;
  return failure();
}

} // namespace afir
} // namespace mlir
