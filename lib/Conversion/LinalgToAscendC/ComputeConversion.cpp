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

#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Backend/ElementwiseBodyOpRegistry.h"
#include "Conversion/Ascend/Backend/LinalgBodyClassifier.h"

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

#define DEBUG_TYPE "linalg-to-ascendc-compute"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace afir {

namespace {

Value getDimValue(OpBuilder &builder, Location loc, Value memref,
                  unsigned dim) {
  auto memrefType = cast<MemRefType>(memref.getType());
  if (!ShapedType::isDynamic(memrefType.getShape()[dim]))
    return builder.create<arith::ConstantIndexOp>(loc,
                                                  memrefType.getShape()[dim]);
  return builder.create<memref::DimOp>(loc, memref, dim);
}

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

void emitStridedGmToLocalCopy(OpBuilder &builder, Location loc, Type elemType,
                              Value dstLt, Value srcGt, Value rows,
                              Value cols, Value srcRowStride) {
  std::string elemTypeStr = getVerbatimScalarTypeName(elemType);
  std::string body = "{\n";
  body += "  uint32_t _afir_rows = (uint32_t)$2;\n";
  body += "  uint32_t _afir_cols = (uint32_t)$3;\n";
  body += "  uint32_t _afir_row_stride = (uint32_t)$4;\n";
  body += "  uint32_t _afir_block_bytes = _afir_cols * sizeof(" +
          elemTypeStr + ");\n";
  body += "  uint32_t _afir_gap_bytes = (_afir_row_stride - _afir_cols) * "
          "sizeof(" + elemTypeStr + ");\n";
  body += "  uint32_t _afir_count = _afir_rows * _afir_cols;\n";
  body += "  if (_afir_gap_bytes == 0u) {\n";
  body += "    if ((_afir_count * sizeof(" + elemTypeStr +
          ")) % 32u == 0u) {\n";
  body += "      AscendC::DataCopy($0, $1, _afir_count);\n";
  body += "    } else {\n";
  body += "      for (uint32_t _afir_i = 0; _afir_i < _afir_count; "
          "++_afir_i)\n";
  body += "        $0.SetValue(_afir_i, $1.GetValue(_afir_i));\n";
  body += "    }\n";
  body += "  } else if ((_afir_block_bytes % 32u) == 0u && "
          "(_afir_gap_bytes % 32u) == 0u) {\n";
  body += "    AscendC::DataCopyExtParams _afir_params{"
          "static_cast<uint16_t>(_afir_rows), _afir_block_bytes, "
          "_afir_gap_bytes, 0u, 0u};\n";
  body += "    AscendC::DataCopyPadExtParams<" + elemTypeStr +
          "> _afir_pad{false, 0, 0, static_cast<" + elemTypeStr + ">(0)};\n";
  body += "    AscendC::DataCopyPad($0, $1, _afir_params, _afir_pad);\n";
  body += "  } else {\n";
  body += "    for (uint32_t _afir_r = 0; _afir_r < _afir_rows; ++_afir_r) {\n";
  body += "      for (uint32_t _afir_c = 0; _afir_c < _afir_cols; ++_afir_c) "
          "{\n";
  body += "        uint32_t _afir_local = _afir_r * _afir_cols + _afir_c;\n";
  body += "        uint64_t _afir_gm = (uint64_t)_afir_r * _afir_row_stride + "
          "_afir_c;\n";
  body += "        $0.SetValue(_afir_local, $1.GetValue(_afir_gm));\n";
  body += "      }\n";
  body += "    }\n";
  body += "  }\n";
  body += "  $0.SetSize(_afir_count);\n";
  body += "}";
  builder.create<emitasc::VerbatimOp>(
      loc, builder.getStringAttr(body),
      ValueRange{dstLt, srcGt, rows, cols, srcRowStride});
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

static memref::AllocOp getRootAllocOp(Value value) {
  while (true) {
    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }
    if (auto castOp = value.getDefiningOp<memref::CastOp>()) {
      value = castOp.getSource();
      continue;
    }
    return value.getDefiningOp<memref::AllocOp>();
  }
}

static Value getMemRefDimWithoutDimOp(OpBuilder &builder, Location loc,
                                      Value memref, unsigned dim) {
  auto memrefType = cast<MemRefType>(memref.getType());
  if (!ShapedType::isDynamic(memrefType.getShape()[dim]))
    return builder.create<arith::ConstantIndexOp>(loc,
                                                  memrefType.getShape()[dim]);

  if (auto allocOp = getRootAllocOp(memref)) {
    unsigned dynamicOrdinal = 0;
    for (unsigned i = 0; i < dim; ++i)
      if (ShapedType::isDynamic(allocOp.getType().getShape()[i]))
        ++dynamicOrdinal;
    if (dynamicOrdinal < allocOp.getDynamicSizes().size())
      return allocOp.getDynamicSizes()[dynamicOrdinal];
  }

  return {};
}

static Value computeContiguousFlatIndex(OpBuilder &builder, Location loc,
                                        Value memref, ValueRange indices) {
  auto memrefType = cast<MemRefType>(memref.getType());
  if (memrefType.getRank() != static_cast<int64_t>(indices.size()))
    return {};
  if (indices.empty())
    return builder.create<arith::ConstantIndexOp>(loc, 0);

  Value flat = indices.front();
  for (unsigned dim = 1; dim < indices.size(); ++dim) {
    Value extent = getMemRefDimWithoutDimOp(builder, loc, memref, dim);
    if (!extent)
      return {};
    flat = builder.create<arith::MulIOp>(loc, flat, extent);
    flat = builder.create<arith::AddIOp>(loc, flat, indices[dim]);
  }
  return flat;
}

bool isSupportedRank2Reduction(linalg::GenericOp op) {
  if (op.getNumDpsInits() != 1)
    return false;
  auto iterTypes = op.getIteratorTypesArray();
  return iterTypes.size() == 2 &&
         iterTypes[0] == utils::IteratorType::parallel &&
         iterTypes[1] == utils::IteratorType::reduction;
}

bool isSupportedRank2AllParallel(linalg::GenericOp op) {
  if (op.getNumDpsInits() != 1)
    return false;
  auto iterTypes = op.getIteratorTypesArray();
  return iterTypes.size() == 2 &&
         iterTypes[0] == utils::IteratorType::parallel &&
         iterTypes[1] == utils::IteratorType::parallel;
}

LogicalResult lowerTransposeToLoops(OpBuilder &builder, Location loc,
                                    Value inMemref, Value outMemref,
                                    ArrayRef<int64_t> permutation) {
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  auto inType = cast<MemRefType>(inMemref.getType());
  auto outType = cast<MemRefType>(outMemref.getType());
  unsigned rank = outType.getRank();
  if (inType.getRank() != static_cast<int64_t>(rank) ||
      permutation.size() != rank)
    return failure();

  SmallVector<bool, 8> seen(rank, false);
  SmallVector<unsigned, 8> inversePermutation(rank, 0);
  unsigned outputDim = 0;
  for (int64_t position : permutation) {
    if (position < 0 || position >= static_cast<int64_t>(rank))
      return failure();
    if (seen[position])
      return failure();
    seen[position] = true;
    inversePermutation[static_cast<unsigned>(position)] = outputDim++;
  }

  SmallVector<Value, 8> upperBounds;
  upperBounds.reserve(rank);
  for (unsigned dim = 0; dim < rank; ++dim)
    upperBounds.push_back(getDimValue(builder, loc, outMemref, dim));

  SmallVector<Value, 8> loopIndices;
  auto buildNest = [&](auto &self, unsigned depth) -> LogicalResult {
    if (depth == rank) {
      SmallVector<Value, 8> inputIndices;
      inputIndices.reserve(rank);
      for (unsigned inputDim = 0; inputDim < rank; ++inputDim)
        inputIndices.push_back(loopIndices[inversePermutation[inputDim]]);
      Value value =
          builder.create<memref::LoadOp>(loc, inMemref, inputIndices);
      builder.create<memref::StoreOp>(loc, value, outMemref, loopIndices);
      return success();
    }

    auto forOp = builder.create<scf::ForOp>(loc, c0, upperBounds[depth], c1);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    loopIndices.push_back(forOp.getInductionVar());
    LogicalResult result = self(self, depth + 1);
    loopIndices.pop_back();
    return result;
  };

  return buildNest(buildNest, 0);
}

bool isPureYieldGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
    return false;
  auto iteratorTypes = op.getIteratorTypesArray();
  if (!llvm::all_of(iteratorTypes, [](utils::IteratorType iteratorType) {
        return iteratorType == utils::IteratorType::parallel;
      }))
    return false;
  auto maps = op.getIndexingMapsArray();
  unsigned outputMapIndex = op.getNumDpsInputs();
  if (maps.size() <= outputMapIndex || !maps[outputMapIndex].isIdentity())
    return false;
  unsigned lastDim = 0;
  bool hasLastDim = false;
  for (AffineExpr expr : maps[0].getResults()) {
    if (isa<AffineConstantExpr>(expr))
      continue;
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      return false;
    unsigned position = dimExpr.getPosition();
    if (hasLastDim && position <= lastDim)
      return false;
    lastDim = position;
    hasLastDim = true;
  }
  Block *body = op.getBody();
  if (body->getOperations().size() != 1)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(&body->front());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;
  auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
  return blockArg && blockArg.getArgNumber() == 0;
}

bool isGmAllParallelGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInits() != 1)
    return false;

  Value outMemref = op.getDpsInitOperand(0)->get();
  auto outType = dyn_cast<MemRefType>(outMemref.getType());
  if (!outType || getMemorySpace(outType) != 0)
    return false;

  auto iteratorTypes = op.getIteratorTypesArray();
  if (!llvm::all_of(iteratorTypes, [](utils::IteratorType iteratorType) {
        return iteratorType == utils::IteratorType::parallel;
      }))
    return false;

  auto maps = op.getIndexingMapsArray();
  unsigned firstOutputMap = op.getNumDpsInputs();
  size_t expectedMapCount = static_cast<size_t>(firstOutputMap) +
                            static_cast<size_t>(op.getNumDpsInits());
  if (maps.size() != expectedMapCount)
    return false;

  AffineMap outMap = maps[firstOutputMap];
  return outMap.isIdentity() &&
         outMap.getNumResults() == static_cast<unsigned>(outType.getRank());
}

bool isSimpleDimOrConstantMap(AffineMap map, unsigned rank) {
  for (AffineExpr expr : map.getResults()) {
    if (isa<AffineConstantExpr>(expr))
      continue;
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr || dimExpr.getPosition() >= rank)
      return false;
  }
  return true;
}

bool isGmScalarLoopGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInits() == 0)
    return false;

  auto iteratorTypes = op.getIteratorTypesArray();
  if (!llvm::all_of(iteratorTypes, [](utils::IteratorType iteratorType) {
        return iteratorType == utils::IteratorType::parallel ||
               iteratorType == utils::IteratorType::reduction;
      }))
    return false;

  unsigned rank = iteratorTypes.size();
  auto maps = op.getIndexingMapsArray();
  unsigned firstOutputMap = op.getNumDpsInputs();
  size_t expectedMapCount = static_cast<size_t>(firstOutputMap) +
                            static_cast<size_t>(op.getNumDpsInits());
  if (maps.size() != expectedMapCount)
    return false;

  for (unsigned i = 0, e = op.getNumDpsInputs(); i < e; ++i)
    if (!isSimpleDimOrConstantMap(maps[i], rank))
      return false;

  for (unsigned i = 0, e = op.getNumDpsInits(); i < e; ++i) {
    Value output = op.getDpsInitOperand(i)->get();
    auto outputType = dyn_cast<MemRefType>(output.getType());
    if (!outputType || getMemorySpace(output.getType()) != 0)
      return false;
    AffineMap outputMap = maps[firstOutputMap + i];
    if (!isSimpleDimOrConstantMap(outputMap, rank) ||
        outputMap.getNumResults() != static_cast<unsigned>(outputType.getRank()))
      return false;
  }

  Block *body = op.getBody();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != op.getNumDpsInits())
    return false;
  for (Operation &bodyOp : body->without_terminator()) {
    if (isa<linalg::IndexOp>(bodyOp))
      continue;
    if (bodyOp.getNumRegions() != 0)
      return false;
  }
  return true;
}

FailureOr<SmallVector<Value, 4>>
buildMappedIndices(OpBuilder &builder, Location loc, AffineMap map,
                   ArrayRef<Value> loopIndices) {
  SmallVector<Value, 4> indices;
  for (AffineExpr expr : map.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      unsigned position = dimExpr.getPosition();
      if (position >= loopIndices.size())
        return failure();
      indices.push_back(loopIndices[position]);
      continue;
    }
    if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
      indices.push_back(
          builder.create<arith::ConstantIndexOp>(loc, constExpr.getValue()));
      continue;
    }
    return failure();
  }
  return indices;
}

LogicalResult lowerPureYieldGenericToLoops(OpBuilder &builder,
                                           linalg::GenericOp op) {
  if (!isPureYieldGeneric(op))
    return failure();

  Location loc = op.getLoc();
  Value inMemref = op.getDpsInputOperand(0)->get();
  Value outMemref = op.getDpsInitOperand(0)->get();
  AffineMap inMap = op.getIndexingMapsArray()[0];
  unsigned rank = op.getIteratorTypesArray().size();

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 4> upperBounds;
  upperBounds.reserve(rank);
  for (unsigned dim = 0; dim < rank; ++dim)
    upperBounds.push_back(getDimValue(builder, loc, outMemref, dim));

  SmallVector<Value, 4> loopIndices;
  auto buildNest = [&](auto &self, unsigned depth) -> LogicalResult {
    if (depth == rank) {
      FailureOr<SmallVector<Value, 4>> inputIndices =
          buildMappedIndices(builder, loc, inMap, loopIndices);
      if (failed(inputIndices))
        return failure();
      Value value =
          builder.create<memref::LoadOp>(loc, inMemref, *inputIndices);
      builder.create<memref::StoreOp>(loc, value, outMemref, loopIndices);
      return success();
    }

    auto forOp = builder.create<scf::ForOp>(loc, c0, upperBounds[depth], c1);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    loopIndices.push_back(forOp.getInductionVar());
    LogicalResult result = self(self, depth + 1);
    loopIndices.pop_back();
    return result;
  };

  return buildNest(buildNest, 0);
}

Value createLocalTensorBuffer(OpBuilder &builder, Location loc, Value pipe,
                              TPosition position, Type elemType,
                              Value byteCount) {
  Value tbuf =
      builder.create<TBufOp>(loc, TBufType::get(builder.getContext(), position));
  builder.create<TPipeInitBufferOp>(loc, pipe, tbuf, byteCount);
  return builder.create<TBufGetTensorOp>(
      loc, LocalTensorType::get(elemType), tbuf, /*len=*/Value{});
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

LogicalResult lowerAllParallelGenericToLoops(OpBuilder &builder,
                                             linalg::GenericOp op) {
  if (!isGmAllParallelGeneric(op))
    return failure();

  Location loc = op.getLoc();
  Block &body = *op.getBody();
  Value outMemref = op.getDpsInitOperand(0)->get();
  auto maps = op.getIndexingMapsArray();
  unsigned inputCount = op.getNumDpsInputs();
  unsigned rank = op.getIteratorTypesArray().size();

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 4> upperBounds;
  upperBounds.reserve(rank);
  for (unsigned dim = 0; dim < rank; ++dim)
    upperBounds.push_back(getDimValue(builder, loc, outMemref, dim));

  SmallVector<Value, 4> loopIndices;
  auto buildNest = [&](auto &self, unsigned depth) -> LogicalResult {
    if (depth == rank) {
      IRMapping mapper;
      for (unsigned i = 0; i < inputCount; ++i) {
        FailureOr<SmallVector<Value, 4>> inputIndices =
            buildMappedIndices(builder, loc, maps[i], loopIndices);
        if (failed(inputIndices))
          return failure();
        Value input = op.getDpsInputOperand(i)->get();
        Value value =
            builder.create<memref::LoadOp>(loc, input, *inputIndices);
        mapper.map(body.getArgument(i), value);
      }

      FailureOr<SmallVector<Value, 4>> outputIndices =
          buildMappedIndices(builder, loc, maps[inputCount], loopIndices);
      if (failed(outputIndices))
        return failure();
      Value currentOut =
          builder.create<memref::LoadOp>(loc, outMemref, *outputIndices);
      mapper.map(body.getArgument(inputCount), currentOut);

      Value yieldedValue;
      for (Operation &bodyOp : body.getOperations()) {
        if (auto yieldOp = dyn_cast<linalg::YieldOp>(bodyOp)) {
          if (yieldOp.getNumOperands() != 1)
            return failure();
          yieldedValue = mapper.lookupOrDefault(yieldOp.getOperand(0));
          break;
        }

        if (auto indexOp = dyn_cast<linalg::IndexOp>(bodyOp)) {
          unsigned dim = static_cast<unsigned>(indexOp.getDim());
          if (dim >= loopIndices.size())
            return failure();
          mapper.map(indexOp.getResult(), loopIndices[dim]);
          continue;
        }

        if (bodyOp.getNumRegions() != 0)
          return failure();
        builder.clone(bodyOp, mapper);
      }

      if (!yieldedValue)
        return failure();
      builder.create<memref::StoreOp>(loc, yieldedValue, outMemref,
                                      *outputIndices);
      return success();
    }

    auto forOp = builder.create<scf::ForOp>(loc, c0, upperBounds[depth], c1);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    loopIndices.push_back(forOp.getInductionVar());
    LogicalResult result = self(self, depth + 1);
    loopIndices.pop_back();
    return result;
  };

  return buildNest(buildNest, 0);
}

FailureOr<SmallVector<Value, 4>>
buildLoopUpperBounds(OpBuilder &builder, Location loc, linalg::GenericOp op) {
  unsigned rank = op.getIteratorTypesArray().size();
  SmallVector<Value, 4> upperBounds(rank);
  auto maps = op.getIndexingMapsArray();

  auto bindOperandDims = [&](Value operand, AffineMap map) -> LogicalResult {
    auto memrefType = dyn_cast<MemRefType>(operand.getType());
    if (!memrefType)
      return failure();
    for (unsigned resultIndex = 0, e = map.getNumResults(); resultIndex < e;
         ++resultIndex) {
      auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(resultIndex));
      if (!dimExpr)
        continue;
      unsigned loopDim = dimExpr.getPosition();
      if (loopDim >= rank ||
          resultIndex >= static_cast<unsigned>(memrefType.getRank()))
        return failure();
      if (!upperBounds[loopDim])
        upperBounds[loopDim] =
            getDimValue(builder, loc, operand, resultIndex);
    }
    return success();
  };

  for (unsigned i = 0, e = op.getNumDpsInputs(); i < e; ++i)
    if (failed(bindOperandDims(op.getDpsInputOperand(i)->get(), maps[i])))
      return failure();

  unsigned firstOutputMap = op.getNumDpsInputs();
  for (unsigned i = 0, e = op.getNumDpsInits(); i < e; ++i)
    if (failed(bindOperandDims(op.getDpsInitOperand(i)->get(),
                               maps[firstOutputMap + i])))
      return failure();

  for (Value bound : upperBounds)
    if (!bound)
      return failure();
  return upperBounds;
}

LogicalResult lowerGmGenericToScalarLoops(OpBuilder &builder,
                                          linalg::GenericOp op) {
  if (!isGmScalarLoopGeneric(op))
    return failure();

  Location loc = op.getLoc();
  Block &body = *op.getBody();
  auto maps = op.getIndexingMapsArray();
  unsigned inputCount = op.getNumDpsInputs();
  unsigned outputCount = op.getNumDpsInits();
  unsigned rank = op.getIteratorTypesArray().size();

  FailureOr<SmallVector<Value, 4>> upperBounds =
      buildLoopUpperBounds(builder, loc, op);
  if (failed(upperBounds))
    return failure();

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 4> loopIndices;
  auto buildNest = [&](auto &self, unsigned depth) -> LogicalResult {
    if (depth == rank) {
      IRMapping mapper;
      for (unsigned i = 0; i < inputCount; ++i) {
        FailureOr<SmallVector<Value, 4>> inputIndices =
            buildMappedIndices(builder, loc, maps[i], loopIndices);
        if (failed(inputIndices))
          return failure();
        Value input = op.getDpsInputOperand(i)->get();
        Value value =
            builder.create<memref::LoadOp>(loc, input, *inputIndices);
        mapper.map(body.getArgument(i), value);
      }

      SmallVector<Value, 4> outputs;
      SmallVector<SmallVector<Value, 4>, 4> outputIndicesList;
      unsigned firstOutputMap = inputCount;
      for (unsigned i = 0; i < outputCount; ++i) {
        FailureOr<SmallVector<Value, 4>> outputIndices =
            buildMappedIndices(builder, loc, maps[firstOutputMap + i],
                               loopIndices);
        if (failed(outputIndices))
          return failure();
        Value output = op.getDpsInitOperand(i)->get();
        Value current =
            builder.create<memref::LoadOp>(loc, output, *outputIndices);
        mapper.map(body.getArgument(inputCount + i), current);
        outputs.push_back(output);
        outputIndicesList.push_back(*outputIndices);
      }

      bool sawYield = false;
      for (Operation &bodyOp : body.getOperations()) {
        if (auto yieldOp = dyn_cast<linalg::YieldOp>(bodyOp)) {
          if (yieldOp.getNumOperands() != outputCount)
            return failure();
          for (unsigned i = 0; i < outputCount; ++i) {
            Value yielded = mapper.lookupOrDefault(yieldOp.getOperand(i));
            builder.create<memref::StoreOp>(loc, yielded, outputs[i],
                                            outputIndicesList[i]);
          }
          sawYield = true;
          break;
        }

        if (auto indexOp = dyn_cast<linalg::IndexOp>(bodyOp)) {
          unsigned dim = static_cast<unsigned>(indexOp.getDim());
          if (dim >= loopIndices.size())
            return failure();
          mapper.map(indexOp.getResult(), loopIndices[dim]);
          continue;
        }

        if (bodyOp.getNumRegions() != 0)
          return failure();
        builder.clone(bodyOp, mapper);
      }

      return success(sawYield);
    }

    auto forOp =
        builder.create<scf::ForOp>(loc, c0, (*upperBounds)[depth], c1);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    loopIndices.push_back(forOp.getInductionVar());
    LogicalResult result = self(self, depth + 1);
    loopIndices.pop_back();
    return result;
  };

  return buildNest(buildNest, 0);
}

LogicalResult lowerRank2GmTransposeToLocalDataCopy(
    OpBuilder &builder, Location loc, Value inMemref, Value outMemref,
    ArrayRef<int64_t> permutation, Value pipe) {
  auto inType = dyn_cast<MemRefType>(inMemref.getType());
  auto outType = dyn_cast<MemRefType>(outMemref.getType());
  if (!inType || !outType || inType.getRank() != 2 || outType.getRank() != 2)
    return failure();
  if (!inType.getLayout().isIdentity() || !outType.getLayout().isIdentity())
    return failure();
  if (getMemorySpace(inType) != 0 || getMemorySpace(outType) != 0)
    return failure();
  if (inType.getElementType() != outType.getElementType())
    return failure();
  if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0)
    return failure();

  Type elemType = inType.getElementType();
  Value elemCount = computeElementCount(builder, loc, inMemref);
  Value byteCount = computeByteCount(builder, loc, inMemref);

  Value srcGt =
      builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
  builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                /*size=*/Value{});
  Value srcLt = createLocalTensorBuffer(builder, loc, pipe, TPosition::VECIN,
                                        elemType, byteCount);
  builder.create<DataCopyL2Op>(loc, srcLt, srcGt, elemCount);

  Value dstLt = createLocalTensorBuffer(builder, loc, pipe, TPosition::VECCALC,
                                        elemType, byteCount);
  builder.create<TransposeOp>(loc, dstLt, srcLt);

  Value dstGt =
      builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
  builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, outMemref,
                                                /*size=*/Value{});
  builder.create<DataCopyL2Op>(loc, dstGt, dstLt, elemCount);
  return success();
}

LogicalResult lowerFillToScalarLoops(OpBuilder &builder, linalg::FillOp op) {
  Location loc = op.getLoc();
  Value fillValue = op.getInputs()[0];
  Value outMemref = op.getOutputs()[0];
  auto outType = dyn_cast<MemRefType>(outMemref.getType());
  if (!outType)
    return failure();

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 4> upperBounds;
  upperBounds.reserve(outType.getRank());
  for (unsigned dim = 0, e = outType.getRank(); dim < e; ++dim)
    upperBounds.push_back(getDimValue(builder, loc, outMemref, dim));

  SmallVector<Value, 4> loopIndices;
  auto buildNest = [&](auto &self, unsigned depth) -> LogicalResult {
    if (depth == static_cast<unsigned>(outType.getRank())) {
      builder.create<memref::StoreOp>(loc, fillValue, outMemref, loopIndices);
      return success();
    }

    auto forOp =
        builder.create<scf::ForOp>(loc, c0, upperBounds[depth], c1);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    loopIndices.push_back(forOp.getInductionVar());
    LogicalResult result = self(self, depth + 1);
    loopIndices.pop_back();
    return result;
  };

  return buildNest(buildNest, 0);
}

void lowerBatchMatmulToLoops(OpBuilder &builder, linalg::BatchMatmulOp op) {
  Location loc = op.getLoc();
  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value out = op.getDpsInitOperand(0)->get();

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value batch = getDimValue(builder, loc, out, 0);
  Value mSize = getDimValue(builder, loc, out, 1);
  Value nSize = getDimValue(builder, loc, out, 2);
  Value kSize = getDimValue(builder, loc, lhs, 2);

  auto forB = builder.create<scf::ForOp>(loc, c0, batch, c1);
  {
    OpBuilder::InsertionGuard guardB(builder);
    builder.setInsertionPointToStart(forB.getBody());
    Value b = forB.getInductionVar();
    auto forM = builder.create<scf::ForOp>(loc, c0, mSize, c1);
    {
      OpBuilder::InsertionGuard guardM(builder);
      builder.setInsertionPointToStart(forM.getBody());
      Value m = forM.getInductionVar();
      auto forN = builder.create<scf::ForOp>(loc, c0, nSize, c1);
      {
        OpBuilder::InsertionGuard guardN(builder);
        builder.setInsertionPointToStart(forN.getBody());
        Value n = forN.getInductionVar();
        Value init =
            builder.create<memref::LoadOp>(loc, out, ValueRange{b, m, n});
        auto forK =
            builder.create<scf::ForOp>(loc, c0, kSize, c1, ValueRange{init});
        {
          OpBuilder::InsertionGuard guardK(builder);
          builder.setInsertionPointToStart(forK.getBody());
          Value k = forK.getInductionVar();
          Value acc = forK.getRegionIterArgs().front();
          Value lhsValue =
              builder.create<memref::LoadOp>(loc, lhs, ValueRange{b, m, k});
          Value rhsValue =
              builder.create<memref::LoadOp>(loc, rhs, ValueRange{b, k, n});
          Value product =
              builder.create<arith::MulFOp>(loc, lhsValue, rhsValue);
          Value sum = builder.create<arith::AddFOp>(loc, acc, product);
          builder.create<scf::YieldOp>(loc, sum);
        }
        builder.create<memref::StoreOp>(loc, forK.getResult(0), out,
                                        ValueRange{b, m, n});
      }
    }
  }
}

void lowerMatmulToLoops(OpBuilder &builder, linalg::MatmulOp op) {
  Location loc = op.getLoc();
  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value out = op.getDpsInitOperand(0)->get();

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value mSize = getDimValue(builder, loc, out, 0);
  Value nSize = getDimValue(builder, loc, out, 1);
  Value kSize = getDimValue(builder, loc, lhs, 1);

  auto forM = builder.create<scf::ForOp>(loc, c0, mSize, c1);
  {
    OpBuilder::InsertionGuard guardM(builder);
    builder.setInsertionPointToStart(forM.getBody());
    Value m = forM.getInductionVar();
    auto forN = builder.create<scf::ForOp>(loc, c0, nSize, c1);
    {
      OpBuilder::InsertionGuard guardN(builder);
      builder.setInsertionPointToStart(forN.getBody());
      Value n = forN.getInductionVar();
      Value init = builder.create<memref::LoadOp>(loc, out, ValueRange{m, n});
      auto forK =
          builder.create<scf::ForOp>(loc, c0, kSize, c1, ValueRange{init});
      {
        OpBuilder::InsertionGuard guardK(builder);
        builder.setInsertionPointToStart(forK.getBody());
        Value k = forK.getInductionVar();
        Value acc = forK.getRegionIterArgs().front();
        Value lhsValue =
            builder.create<memref::LoadOp>(loc, lhs, ValueRange{m, k});
        Value rhsValue =
            builder.create<memref::LoadOp>(loc, rhs, ValueRange{k, n});
        Value product = builder.create<arith::MulFOp>(loc, lhsValue, rhsValue);
        Value sum = builder.create<arith::AddFOp>(loc, acc, product);
        builder.create<scf::YieldOp>(loc, sum);
      }
      builder.create<memref::StoreOp>(loc, forK.getResult(0), out,
                                      ValueRange{m, n});
    }
  }
}

FailureOr<unsigned> getSingleDimProjection(AffineMap map) {
  if (map.getNumDims() != 2 || map.getNumResults() != 1)
    return failure();
  auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(0));
  if (!dimExpr)
    return failure();
  unsigned position = dimExpr.getPosition();
  if (position >= 2)
    return failure();
  return position;
}

bool isRank2IdentityMap(AffineMap map) {
  return map.getNumDims() == 2 && map.getNumResults() == 2 &&
         map.isIdentity();
}

bool isRank2BroadcastTransposeMap(AffineMap map) {
  if (map.getNumDims() != 2 || map.getNumResults() != 2)
    return false;

  auto first = dyn_cast<AffineDimExpr>(map.getResult(0));
  auto second = dyn_cast<AffineConstantExpr>(map.getResult(1));
  return first && first.getPosition() == 1 && second &&
         second.getValue() == 0;
}

bool isSupportedSelectedTileMap(Value operand, AffineMap map) {
  auto memrefType = dyn_cast<MemRefType>(operand.getType());
  if (!memrefType)
    return false;
  if (memrefType.getRank() == 1)
    return succeeded(getSingleDimProjection(map));
  if (memrefType.getRank() == 2)
    return isRank2IdentityMap(map);
  return false;
}

bool isSupportedSelectedAllParallelTileMap(Value operand, AffineMap map) {
  auto memrefType = dyn_cast<MemRefType>(operand.getType());
  if (!memrefType)
    return false;
  if (memrefType.getRank() == 1)
    return succeeded(getSingleDimProjection(map));
  if (memrefType.getRank() == 2)
    return isRank2IdentityMap(map) || isRank2BroadcastTransposeMap(map);
  return false;
}

bool hasSupportedSelectedAllParallelTileMaps(linalg::GenericOp op,
                                             ArrayRef<AffineMap> maps,
                                             Value writebackTarget) {
  if (maps.size() !=
      static_cast<size_t>(op.getNumDpsInputs() + op.getNumDpsInits()))
    return false;

  for (unsigned i = 0, e = op.getNumDpsInputs(); i < e; ++i)
    if (!isSupportedSelectedAllParallelTileMap(
            op.getDpsInputOperand(i)->get(), maps[i]))
      return false;

  return isSupportedSelectedAllParallelTileMap(writebackTarget, maps.back());
}

FailureOr<int64_t> getStaticReductionExtent(linalg::GenericOp op,
                                            ArrayRef<AffineMap> maps) {
  for (unsigned i = 0, e = op.getNumDpsInputs(); i < e; ++i) {
    Value input = op.getDpsInputOperand(i)->get();
    auto inputType = dyn_cast<MemRefType>(input.getType());
    if (!inputType || inputType.getRank() != 2 ||
        !isRank2IdentityMap(maps[i]))
      continue;
    return inputType.getShape()[1];
  }
  return failure();
}

LogicalResult validateSelectedReductionTile(linalg::GenericOp op,
                                            ArrayRef<AffineMap> maps,
                                            int64_t reductionTile) {
  if (ShapedType::isDynamic(reductionTile))
    return success();

  FailureOr<int64_t> reductionExtent = getStaticReductionExtent(op, maps);
  if (failed(reductionExtent) || ShapedType::isDynamic(*reductionExtent))
    return op.emitError("selected reduction tile requires full reduction axis");
  if (reductionTile != *reductionExtent)
    return op.emitError("selected reduction tile requires full reduction axis");

  return success();
}

LogicalResult validateSelectedAllParallelTile(linalg::GenericOp op,
                                              Value outMemref,
                                              int64_t innerTile) {
  if (ShapedType::isDynamic(innerTile) || innerTile <= 0)
    return op.emitError("selected all-parallel tile requires a static "
                        "positive inner tile");

  auto outType = dyn_cast<MemRefType>(outMemref.getType());
  if (!outType || outType.getRank() != 2)
    return op.emitError("selected all-parallel tile requires a rank-2 output");

  return success();
}

LogicalResult validateSelectedTileMaps(linalg::GenericOp op,
                                       ArrayRef<AffineMap> maps,
                                       Value writebackTarget) {
  if (maps.size() !=
      static_cast<size_t>(op.getNumDpsInputs() + op.getNumDpsInits()))
    return op.emitError("unsupported selected-tile indexing map");

  for (unsigned i = 0, e = op.getNumDpsInputs(); i < e; ++i)
    if (!isSupportedSelectedTileMap(op.getDpsInputOperand(i)->get(), maps[i]))
      return op.emitError("unsupported selected-tile indexing map");

  if (!isSupportedSelectedTileMap(writebackTarget, maps.back()))
    return op.emitError("unsupported selected-tile indexing map");

  return success();
}

Value createRank2TileAlloc(OpBuilder &builder, Location loc,
                           MemRefType sourceType, Value tileRows,
                           Value tileCols) {
  SmallVector<int64_t> tileShape{ShapedType::kDynamic,
                                 ShapedType::kDynamic};
  SmallVector<Value> dynamicSizes{tileRows, tileCols};

  auto tileType = MemRefType::get(tileShape, sourceType.getElementType(),
                                  MemRefLayoutAttrInterface{},
                                  sourceType.getMemorySpace());
  return builder.create<memref::AllocOp>(loc, tileType, dynamicSizes);
}

FailureOr<Value> buildTiledOperandSubview(OpBuilder &builder, Location loc,
                                          Value operand, AffineMap map,
                                          Value rowOffset, Value tileRows,
                                          Value reductionExtent) {
  auto memrefType = dyn_cast<MemRefType>(operand.getType());
  if (!memrefType)
    return failure();

  auto one = builder.getIndexAttr(1);
  if (memrefType.getRank() == 1) {
    FailureOr<unsigned> projection = getSingleDimProjection(map);
    if (failed(projection))
      return failure();

    SmallVector<OpFoldResult> offsets{
        *projection == 0 ? OpFoldResult(rowOffset)
                         : OpFoldResult(builder.getIndexAttr(0))};
    SmallVector<OpFoldResult> sizes{
        *projection == 0 ? OpFoldResult(tileRows)
                         : OpFoldResult(reductionExtent)};
    SmallVector<OpFoldResult> strides{one};
    return builder
        .create<memref::SubViewOp>(loc, operand, offsets, sizes, strides)
        .getResult();
  }

  if (memrefType.getRank() == 2 && isRank2IdentityMap(map)) {
    SmallVector<OpFoldResult> offsets{rowOffset, builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{tileRows, reductionExtent};
    SmallVector<OpFoldResult> strides{one, one};
    return builder
        .create<memref::SubViewOp>(loc, operand, offsets, sizes, strides)
        .getResult();
  }

  return failure();
}

FailureOr<Value> buildTiledAllParallelOperandSubview(
    OpBuilder &builder, Location loc, Value operand, AffineMap map,
    Value rowOffset, Value colOffset, Value tileRows, Value tileCols) {
  auto memrefType = dyn_cast<MemRefType>(operand.getType());
  if (!memrefType)
    return failure();

  auto one = builder.getIndexAttr(1);
  if (memrefType.getRank() == 1) {
    FailureOr<unsigned> projection = getSingleDimProjection(map);
    if (failed(projection))
      return failure();

    SmallVector<OpFoldResult> offsets{*projection == 0
                                          ? OpFoldResult(rowOffset)
                                          : OpFoldResult(colOffset)};
    SmallVector<OpFoldResult> sizes{*projection == 0
                                        ? OpFoldResult(tileRows)
                                        : OpFoldResult(tileCols)};
    SmallVector<OpFoldResult> strides{one};
    return builder
        .create<memref::SubViewOp>(loc, operand, offsets, sizes, strides)
        .getResult();
  }

  if (memrefType.getRank() == 2 && isRank2IdentityMap(map)) {
    SmallVector<OpFoldResult> offsets{rowOffset, colOffset};
    SmallVector<OpFoldResult> sizes{tileRows, tileCols};
    SmallVector<OpFoldResult> strides{one, one};
    return builder
        .create<memref::SubViewOp>(loc, operand, offsets, sizes, strides)
        .getResult();
  }

  if (memrefType.getRank() == 2 && isRank2BroadcastTransposeMap(map)) {
    SmallVector<OpFoldResult> offsets{colOffset, builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{tileCols, builder.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides{one, one};
    return builder
        .create<memref::SubViewOp>(loc, operand, offsets, sizes, strides)
        .getResult();
  }

  return failure();
}

memref::CopyOp findSingleWritebackCopy(Value source) {
  memref::CopyOp result;
  for (Operation *user : llvm::make_early_inc_range(source.getUsers())) {
    auto copyOp = dyn_cast<memref::CopyOp>(user);
    if (!copyOp || copyOp.getSource() != source ||
        getMemorySpace(copyOp.getTarget().getType()) != 0)
      continue;
    if (result)
      return {};
    result = copyOp;
  }
  return result;
}

Value rootMemref(Value value) {
  while (true) {
    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    return value;
  }
}

SmallVector<Value, 4> inputMemrefRoots(linalg::GenericOp op) {
  SmallVector<Value, 4> roots;
  auto appendRoot = [&](Value value) {
    if (!isa<MemRefType>(value.getType()))
      return;
    Value root = rootMemref(value);
    if (!llvm::is_contained(roots, root))
      roots.push_back(root);
  };

  for (OpOperand *operand : op.getDpsInputOperands())
    appendRoot(operand->get());
  return roots;
}

bool rootIntersects(Value value, ArrayRef<Value> roots) {
  return isa<MemRefType>(value.getType()) &&
         llvm::is_contained(roots, rootMemref(value));
}

bool rootEquals(Value value, Value root) {
  return isa<MemRefType>(value.getType()) && rootMemref(value) == root;
}

bool isBenignShapeOrViewOp(Operation *op) {
  return isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
             arith::MinSIOp, arith::MaxSIOp, arith::IndexCastOp,
             affine::AffineApplyOp, memref::AllocOp, memref::DimOp,
             memref::SubViewOp, memref::CastOp>(op);
}

bool mayWriteAnyRoot(Operation *op, ArrayRef<Value> roots) {
  if (auto copyOp = dyn_cast<memref::CopyOp>(op))
    return rootIntersects(copyOp.getTarget(), roots);
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return rootIntersects(storeOp.getMemref(), roots);
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    return llvm::any_of(linalgOp.getDpsInits(), [&](Value init) {
      return rootIntersects(init, roots);
    });
  return false;
}

bool mayReadRoot(Operation *op, Value root) {
  if (auto copyOp = dyn_cast<memref::CopyOp>(op))
    return rootEquals(copyOp.getSource(), root);
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return rootEquals(loadOp.getMemref(), root);
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    return llvm::any_of(linalgOp.getDpsInputs(), [&](Value input) {
      return rootEquals(input, root);
    });
  return false;
}

bool touchesAnyRoot(Operation *op, ArrayRef<Value> roots) {
  return llvm::any_of(op->getOperands(), [&](Value operand) {
    return rootIntersects(operand, roots);
  });
}

bool canMoveSelectedTileLoopBeforeWriteback(linalg::GenericOp op,
                                            memref::CopyOp writeback,
                                            Value outMemref) {
  SmallVector<Value, 4> inputRoots = inputMemrefRoots(op);
  Value outputRoot = rootMemref(outMemref);
  SmallVector<Value, 1> outputRoots{outputRoot};
  for (Operation *it = op->getNextNode(); it && it != writeback.getOperation();
       it = it->getNextNode()) {
    if (isBenignShapeOrViewOp(it))
      continue;

    if (mayWriteAnyRoot(it, inputRoots) || mayWriteAnyRoot(it, outputRoots) ||
        mayReadRoot(it, outputRoot))
      return false;

    if (!isa<linalg::LinalgOp, memref::CopyOp, memref::LoadOp,
             memref::StoreOp>(it) &&
        (touchesAnyRoot(it, inputRoots) || touchesAnyRoot(it, outputRoots)))
      return false;
  }
  return true;
}

Operation *selectedTileInsertionPoint(linalg::GenericOp op,
                                      memref::CopyOp writeback,
                                      Value outMemref) {
  if (!canMoveSelectedTileLoopBeforeWriteback(op, writeback, outMemref))
    return nullptr;

  Operation *targetDef = writeback.getTarget().getDefiningOp();
  if (targetDef && targetDef->getBlock() == op->getBlock() &&
      op->isBeforeInBlock(targetDef))
    return writeback.getOperation();
  return op.getOperation();
}

bool isZeroScalarConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;

  Attribute attr = constant.getValue();
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  return false;
}

linalg::FillOp findRedundantZeroFill(Value target, Operation *anchor) {
  if (!target || !anchor || anchor->getBlock() == nullptr)
    return {};

  linalg::FillOp latestFill;
  for (Operation *user : target.getUsers()) {
    auto fillOp = dyn_cast<linalg::FillOp>(user);
    if (!fillOp || fillOp.getOutputs()[0] != target ||
        fillOp->getBlock() != anchor->getBlock() ||
        !fillOp->isBeforeInBlock(anchor) ||
        !isZeroScalarConstant(fillOp.getInputs()[0]))
      continue;
    if (!latestFill || latestFill->isBeforeInBlock(fillOp))
      latestFill = fillOp;
  }
  if (!latestFill)
    return {};

  Value targetRoot = rootMemref(target);
  SmallVector<Value, 1> targetRoots{targetRoot};
  for (Operation *it = latestFill->getNextNode(); it && it != anchor;
       it = it->getNextNode()) {
    if (isBenignShapeOrViewOp(it))
      continue;
    if (mayReadRoot(it, targetRoot) || mayWriteAnyRoot(it, targetRoots))
      return {};
    if (!isa<linalg::LinalgOp, memref::CopyOp, memref::LoadOp,
             memref::StoreOp>(it) &&
        touchesAnyRoot(it, targetRoots))
      return {};
  }

  return latestFill;
}

} // namespace

LogicalResult materializeSelectedReductionTiles(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  SmallVector<linalg::GenericOp> candidates;
  funcOp.walk([&](linalg::GenericOp op) {
    if (op->getParentOfType<scf::ForOp>())
      return;
    if (op->getAttrOfType<DenseI64ArrayAttr>(
            ascend::kScheduleSelectedTileShapeAttr))
      candidates.push_back(op);
  });

  for (linalg::GenericOp genOp : candidates) {
    if (!isSupportedRank2Reduction(genOp))
      continue;

    auto selectedTile = genOp->getAttrOfType<DenseI64ArrayAttr>(
        ascend::kScheduleSelectedTileShapeAttr);
    if (!selectedTile || selectedTile.asArrayRef().size() < 2)
      return genOp.emitError(
          "selected rank-2 reduction tile requires at least two dimensions");
    int64_t tileRows = selectedTile.asArrayRef()[0];
    if (ShapedType::isDynamic(tileRows) || tileRows <= 0)
      return genOp.emitError(
          "selected reduction tile requires a static positive parallel tile");

    Value outMemref = genOp.getDpsInitOperand(0)->get();
    auto outType = dyn_cast<MemRefType>(outMemref.getType());
    if (!outType || outType.getRank() != 1)
      continue;

    memref::CopyOp writeback = findSingleWritebackCopy(outMemref);
    if (!writeback)
      continue;

    auto maps = genOp.getIndexingMapsArray();
    if (failed(validateSelectedTileMaps(genOp, maps, writeback.getTarget())))
      return failure();
    if (failed(validateSelectedReductionTile(
            genOp, maps, selectedTile.asArrayRef()[1])))
      return failure();

    Location loc = genOp.getLoc();
    Operation *insertionPoint =
        selectedTileInsertionPoint(genOp, writeback, outMemref);
    if (!insertionPoint)
      continue;

    Value dstMemref = writeback.getTarget();
    linalg::FillOp redundantZeroFill =
        findRedundantZeroFill(dstMemref, writeback.getOperation());
    builder.setInsertionPoint(insertionPoint);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value step = builder.create<arith::ConstantIndexOp>(loc, tileRows);
    Value rows = getDimValue(builder, loc, dstMemref, 0);

    auto forOp = builder.create<scf::ForOp>(loc, zero, rows, step);
    forOp->setAttr("ascendc.parallel", builder.getBoolAttr(true));

    OpBuilder bodyBuilder(funcOp.getContext());
    bodyBuilder.setInsertionPointToStart(forOp.getBody());
    Value remaining =
        bodyBuilder.create<arith::SubIOp>(loc, rows, forOp.getInductionVar());
    Value tileRowsValue =
        bodyBuilder.create<arith::MinSIOp>(loc, step, remaining);

    Value reductionExtent;
    for (unsigned i = 0, e = genOp.getNumDpsInputs(); i < e; ++i) {
      Value input = genOp.getDpsInputOperand(i)->get();
      auto inputType = dyn_cast<MemRefType>(input.getType());
      if (!inputType || inputType.getRank() != 2 ||
          !isRank2IdentityMap(maps[i]))
        continue;
      reductionExtent = getDimValue(bodyBuilder, loc, input, 1);
      break;
    }
    if (!reductionExtent)
      return genOp.emitError("selected reduction tile requires a rank-2 input");

    IRMapping mapper;
    for (unsigned i = 0, e = genOp.getNumDpsInputs(); i < e; ++i) {
      Value input = genOp.getDpsInputOperand(i)->get();
      FailureOr<Value> tiledInput = buildTiledOperandSubview(
          bodyBuilder, loc, input, maps[i], forOp.getInductionVar(),
          tileRowsValue, reductionExtent);
      if (failed(tiledInput))
        return genOp.emitError("unsupported selected-tile indexing map");
      mapper.map(input, *tiledInput);
    }

    auto tileOutType =
        MemRefType::get({ShapedType::kDynamic}, outType.getElementType(),
                        MemRefLayoutAttrInterface{}, outType.getMemorySpace());
    Value tiledOut = bodyBuilder.create<memref::AllocOp>(
        loc, tileOutType, ValueRange{tileRowsValue});
    mapper.map(outMemref, tiledOut);

    bodyBuilder.clone(*genOp, mapper);

    FailureOr<Value> tiledDst = buildTiledOperandSubview(
        bodyBuilder, loc, dstMemref, maps.back(), forOp.getInductionVar(),
        tileRowsValue, reductionExtent);
    if (failed(tiledDst))
      return writeback.emitError("unsupported selected-tile indexing map");
    bodyBuilder.create<memref::CopyOp>(loc, tiledOut, *tiledDst);

    genOp.erase();
    writeback.erase();
    if (redundantZeroFill)
      redundantZeroFill.erase();
    if (auto allocOp = outMemref.getDefiningOp<memref::AllocOp>())
      if (allocOp->use_empty())
        allocOp.erase();
  }

  return success();
}

LogicalResult materializeSelectedAllParallelTiles(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  SmallVector<linalg::GenericOp> candidates;
  funcOp.walk([&](linalg::GenericOp op) {
    if (op->getParentOfType<scf::ForOp>())
      return;
    if (op->getAttrOfType<DenseI64ArrayAttr>(
            ascend::kScheduleSelectedTileShapeAttr))
      candidates.push_back(op);
  });

  for (linalg::GenericOp genOp : candidates) {
    if (!isSupportedRank2AllParallel(genOp))
      continue;

    auto selectedTile = genOp->getAttrOfType<DenseI64ArrayAttr>(
        ascend::kScheduleSelectedTileShapeAttr);
    if (!selectedTile || selectedTile.asArrayRef().size() < 2)
      return genOp.emitError("selected rank-2 all-parallel tile requires "
                             "at least two dimensions");
    int64_t tileRows = selectedTile.asArrayRef()[0];
    if (ShapedType::isDynamic(tileRows) || tileRows <= 0)
      return genOp.emitError("selected all-parallel tile requires a static "
                             "positive outer tile");
    int64_t tileCols = selectedTile.asArrayRef()[1];
    if (ShapedType::isDynamic(tileCols) || tileCols <= 0)
      return genOp.emitError("selected all-parallel tile requires a static "
                             "positive inner tile");

    Value outMemref = genOp.getDpsInitOperand(0)->get();
    auto outType = dyn_cast<MemRefType>(outMemref.getType());
    if (!outType || outType.getRank() != 2)
      continue;

    memref::CopyOp writeback = findSingleWritebackCopy(outMemref);
    if (!writeback)
      continue;

    auto maps = genOp.getIndexingMapsArray();
    if (!hasSupportedSelectedAllParallelTileMaps(genOp, maps,
                                                 writeback.getTarget()))
      continue;
    if (failed(validateSelectedAllParallelTile(
            genOp, outMemref, selectedTile.asArrayRef()[1])))
      return failure();

    Location loc = genOp.getLoc();
    Operation *insertionPoint =
        selectedTileInsertionPoint(genOp, writeback, outMemref);
    if (!insertionPoint)
      continue;

    Value dstMemref = writeback.getTarget();
    builder.setInsertionPoint(insertionPoint);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value step = builder.create<arith::ConstantIndexOp>(loc, tileRows);
    Value rows = getDimValue(builder, loc, dstMemref, 0);
    Value innerExtent = getDimValue(builder, loc, dstMemref, 1);
    bool dynamicInner = ShapedType::isDynamic(outType.getShape()[1]);
    bool hasRank2SubviewInput = false;
    for (unsigned i = 0, e = genOp.getNumDpsInputs(); i < e; ++i) {
      auto inputType =
          dyn_cast<MemRefType>(genOp.getDpsInputOperand(i)->get().getType());
      if (inputType && inputType.getRank() == 2 &&
          genOp.getDpsInputOperand(i)->get().getDefiningOp<memref::SubViewOp>()) {
        hasRank2SubviewInput = true;
        break;
      }
    }
    bool needsInnerLoop =
        (!dynamicInner && tileCols < outType.getShape()[1]) ||
        (dynamicInner && hasRank2SubviewInput);

    auto forOp = builder.create<scf::ForOp>(loc, zero, rows, step);
    forOp->setAttr("ascendc.parallel", builder.getBoolAttr(true));

    OpBuilder bodyBuilder(funcOp.getContext());
    bodyBuilder.setInsertionPointToStart(forOp.getBody());
    Value remaining =
        bodyBuilder.create<arith::SubIOp>(loc, rows, forOp.getInductionVar());
    Value tileRowsValue =
        bodyBuilder.create<arith::MinSIOp>(loc, step, remaining);

    auto emitTileBody = [&](OpBuilder &tileBuilder, Value colOffset,
                            Value tileColsValue) -> LogicalResult {
      IRMapping mapper;
      for (unsigned i = 0, e = genOp.getNumDpsInputs(); i < e; ++i) {
        Value input = genOp.getDpsInputOperand(i)->get();
        FailureOr<Value> tiledInput = buildTiledAllParallelOperandSubview(
            tileBuilder, loc, input, maps[i], forOp.getInductionVar(),
            colOffset, tileRowsValue, tileColsValue);
        if (failed(tiledInput))
          return genOp.emitError("unsupported selected-tile indexing map");
        mapper.map(input, *tiledInput);
      }

      Value tiledOut = createRank2TileAlloc(tileBuilder, loc, outType,
                                            tileRowsValue, tileColsValue);
      mapper.map(outMemref, tiledOut);
      tileBuilder.clone(*genOp, mapper);

      FailureOr<Value> tiledDst = buildTiledAllParallelOperandSubview(
          tileBuilder, loc, dstMemref, maps.back(), forOp.getInductionVar(),
          colOffset, tileRowsValue, tileColsValue);
      if (failed(tiledDst))
        return writeback.emitError("unsupported selected-tile indexing map");
      tileBuilder.create<memref::CopyOp>(loc, tiledOut, *tiledDst);
      return success();
    };

    if (needsInnerLoop) {
      Value colStep = bodyBuilder.create<arith::ConstantIndexOp>(loc, tileCols);
      auto colFor = bodyBuilder.create<scf::ForOp>(loc, zero, innerExtent,
                                                   colStep);
      OpBuilder innerBuilder(funcOp.getContext());
      innerBuilder.setInsertionPointToStart(colFor.getBody());
      Value remainingCols = innerBuilder.create<arith::SubIOp>(
          loc, innerExtent, colFor.getInductionVar());
      Value tileColsValue =
          innerBuilder.create<arith::MinSIOp>(loc, colStep, remainingCols);
      if (failed(emitTileBody(innerBuilder, colFor.getInductionVar(),
                              tileColsValue)))
        return failure();
    } else if (failed(emitTileBody(bodyBuilder, zero, innerExtent))) {
      return failure();
    }

    genOp.erase();
    writeback.erase();
    if (auto allocOp = outMemref.getDefiningOp<memref::AllocOp>())
      if (allocOp->use_empty())
        allocOp.erase();
  }

  return success();
}

LogicalResult convertCompute(func::FuncOp funcOp, AscendCBufferContext &ctx) {
  MLIRContext *mlirCtx = funcOp.getContext();
  OpBuilder builder(mlirCtx);

  auto copyAscendCUnitAttr = [](Operation *src, Operation *dst) {
    if (!src || !dst)
      return;
    if (auto unitAttr = src->getAttrOfType<StringAttr>(ascend::kAscendCUnitAttr))
      dst->setAttr(ascend::kAscendCUnitAttr, unitAttr);
  };

  auto getEnclosingLoopStepBound = [](Value value,
                                      Operation *anchor) -> Value {
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
  };

  auto getSubviewSizeValue = [&](OpBuilder &b, Location loc, Value memref,
                                 unsigned dim) -> Value {
    auto subviewOp = memref.getDefiningOp<memref::SubViewOp>();
    if (!subviewOp)
      return Value{};
    SmallVector<OpFoldResult> mixedSizes = subviewOp.getMixedSizes();
    if (dim >= mixedSizes.size())
      return Value{};
    OpFoldResult size = mixedSizes[dim];
    if (auto attr = size.dyn_cast<Attribute>())
      return b.create<arith::ConstantIndexOp>(loc,
                                              cast<IntegerAttr>(attr).getInt());
    return size.get<Value>();
  };

  auto computeProduct = [&](OpBuilder &b, Location loc,
                            ArrayRef<Value> dims) -> Value {
    Value totalElems;
    for (Value s : dims)
      totalElems = totalElems ? b.create<arith::MulIOp>(loc, totalElems, s) : s;
    if (!totalElems)
      totalElems = b.create<arith::ConstantIndexOp>(loc, 1);
    return totalElems;
  };

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
  // AscendC GetWithOffset(size, bufOffset) takes `size` in elements and
  // `bufOffset` in bytes.
  // Returns null if no tbuf registered for `memref`.
  auto tbufSlice = [&](OpBuilder &b, Location loc, Value memref,
                        Value sizeElems, Value offsetBytes) -> Value {
    Value tbuf = ctx.getTBuf(memref);
    if (!tbuf)
      return Value{};
    auto mrt = cast<MemRefType>(memref.getType());
    return b.create<TBufGetWithOffsetOp>(
        loc, LocalTensorType::get(mrt.getElementType()), tbuf, sizeElems,
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
        Value sizeElems = computeElementCount(b, loc, memref);
        if (Value t = tbufSlice(b, loc, memref, sizeElems, byteOff))
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
      Value sizeElems = computeElementCount(b, loc, memref);
      if (Value t = tbufSlice(b, loc, memref, sizeElems, byteOff))
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
              ctx.pipe))) {
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
    Value srcLt = readTensor(builder, loc, inMemref);
    Value dstLt = writeTensor(builder, loc, outMemref);
    auto lowered = builder.create<TransposeOp>(loc, dstLt, srcLt);
    copyAscendCUnitAttr(transposeOp.getOperation(), lowered.getOperation());
    if (Value queue = ctx.getQueue(outMemref))
      builder.create<TQueBindEnqueTensorOp>(loc, queue, dstLt);
    transposeOp.erase();
  }

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
  // Analysis of a single input indexing map relative to the iteration space.
  struct IndexingMapAnalysis {
    enum class Kind {
      Identity,           // (d0,d1)->(d0,d1): direct read
      PureBroadcast,      // (d0,d1)->(d0): some dims absent, no reordering
      PureTranspose,      // (d0,d1)->(d1,d0): all dims present, permuted
      BroadcastTranspose, // (d0,d1)->(d1,0): constants + reordering
    };
    Kind kind;
    SmallVector<int64_t> permutation;    // valid for PureTranspose, BroadcastTranspose
    SmallVector<int64_t> broadcastDims;  // iteration dims absent from output
  };

  // Analyze an input indexing map to classify how the input is accessed
  // relative to the iteration space of rank `iterRank`.
  auto analyzeIndexingMap = [](AffineMap map,
                                unsigned iterRank) -> IndexingMapAnalysis {
    IndexingMapAnalysis result;

    // Identity: fast path
    if (map.isIdentity()) {
      result.kind = IndexingMapAnalysis::Kind::Identity;
      return result;
    }

    // Collect which iteration dims appear in the map results (as dim exprs)
    // and which results are constants.
    SmallVector<int64_t> presentDims;  // iteration dim positions that appear
    bool hasConstant = false;
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        presentDims.push_back(static_cast<int64_t>(dimExpr.getPosition()));
      } else if (isa<AffineConstantExpr>(expr)) {
        hasConstant = true;
      } else {
        // Non-trivial affine expression: not handled.
        result.kind = IndexingMapAnalysis::Kind::Identity; // fallback: treat as identity
        return result;
      }
    }

    // Determine broadcast dims: iteration dims not in presentDims.
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
  };

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

  auto isDuplicateL2FillSupportedType = [](Type elemType) {
    if (elemType.isF16() || elemType.isF32() || elemType.isBF16())
      return true;
    if (auto intType = dyn_cast<IntegerType>(elemType))
      return intType.getWidth() == 16 || intType.getWidth() == 32;
    return false;
  };

  // Helper: copy `elemCount` elements from a GM GlobalTensor into a fresh
  // VECIN TQue (AllocTensor → DataCopy → EnQue → DeQue) and return the
  // dequeued VECIN LocalTensor.  The AscendC simulator only supports
  // DataCopy from GM → VECIN TQue (not directly to VECCALC TBuf).
  using OwnedQueueTensor = std::pair<Value, Value>;
  auto freeOwnedQueueTensors =
      [&](OpBuilder &b, Location loc, ArrayRef<OwnedQueueTensor> ownedTensors) {
        for (const auto &[queue, tensor] : ownedTensors)
          b.create<TQueBindFreeTensorOp>(loc, queue, tensor);
      };
  auto rememberQueueRead = [&](SmallVectorImpl<OwnedQueueTensor> &ownedTensors,
                               Value queue, Value tensor) {
    if (queue && tensor)
      ownedTensors.push_back({queue, tensor});
  };

  auto copyGmToVecin =
      [&](OpBuilder &b, Location loc, Type elemType, Value srcGt,
          Value elemCount, Value bufferElemCount,
          SmallVectorImpl<OwnedQueueTensor> *ownedTensors = nullptr) -> Value {
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
    if (ownedTensors)
      rememberQueueRead(*ownedTensors, vecinQue, dequeued);
    return dequeued;
  };

  auto copyGmToVeccalc =
      [&](OpBuilder &b, Location loc, Type elemType, Value srcGt,
          Value elemCount) -> Value {
    auto [veccalcTbuf, veccalcLt] =
        allocVeccalc(b, loc, elemType, SmallVector<Value>{elemCount});
    b.create<DataCopyL2Op>(loc, veccalcLt, srcGt, elemCount);
    return veccalcLt;
  };

  // Helper: get a runtime Value for dimension `dim` of a memref.
  auto getDynDim = [&](OpBuilder &b, Location loc, Value memref,
                        unsigned dim) -> Value {
    if (Value subviewSize = getSubviewSizeValue(b, loc, memref, dim))
      return subviewSize;
    auto mrt = cast<MemRefType>(memref.getType());
    if (!ShapedType::isDynamic(mrt.getShape()[dim]))
      return b.create<arith::ConstantIndexOp>(loc, mrt.getShape()[dim]);
    return b.create<memref::DimOp>(loc, memref, dim);
  };

  auto getBufferDimSizes = [&](ArrayRef<Value> dims,
                               Operation *anchor) -> SmallVector<Value> {
    SmallVector<Value> bufferDims;
    bufferDims.reserve(dims.size());
    for (Value dim : dims)
      bufferDims.push_back(getEnclosingLoopStepBound(dim, anchor));
    return bufferDims;
  };

  auto isRank2GmSubview = [](Value memref) -> bool {
    auto type = dyn_cast<MemRefType>(memref.getType());
    return type && type.getRank() == 2 && getMemorySpace(type) == 0 &&
           memref.getDefiningOp<memref::SubViewOp>();
  };

  auto isContiguousRank2GmSubview = [](Value memref) -> bool {
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
  };

  auto getRank2RowStride = [&](OpBuilder &b, Location loc,
                               Value memref) -> Value {
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
      return getDynDim(b, loc, root, 1);

    return getDynDim(b, loc, memref, 1);
  };

  auto copyRank2GmSubviewRowsToVecin =
      [&](OpBuilder &b, Location loc, Type elemType, Value srcMemref,
          Value bufferElemCount,
          SmallVectorImpl<OwnedQueueTensor> *ownedTensors = nullptr) -> Value {
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

    Value srcGt =
        b.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
    b.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, srcMemref,
                                            /*size=*/Value{});
    Value rows = getDynDim(b, loc, srcMemref, 0);
    Value cols = getDynDim(b, loc, srcMemref, 1);
    Value srcRowStride = getRank2RowStride(b, loc, srcMemref);
    if (!srcRowStride)
      return Value{};
    emitStridedGmToLocalCopy(b, loc, elemType, lt, srcGt, rows, cols,
                             srcRowStride);

    b.create<TQueBindEnqueTensorOp>(loc, vecinQue, lt);
    Value dequeued = b.create<TQueBindDequeTensorOp>(
        loc, LocalTensorType::get(elemType), vecinQue);
    if (ownedTensors)
      rememberQueueRead(*ownedTensors, vecinQue, dequeued);
    return dequeued;
  };

  SmallVector<linalg::GenericOp> genericOps;
  funcOp.walk([&](linalg::GenericOp op) { genericOps.push_back(op); });

  for (linalg::GenericOp genOp : genericOps) {
    if (!isGmScalarLoopGeneric(genOp))
      continue;

    builder.setInsertionPoint(genOp);
    if (succeeded(
            lowerProjectedSuffixCopyToSegmentDataCopy(builder, genOp,
                                                      ctx.pipe))) {
      genOp.erase();
      continue;
    }

    if (failed(lowerGmGenericToScalarLoops(builder, genOp))) {
      genOp.emitError("failed to lower GM generic scalar loop");
      return failure();
    }
    genOp.erase();
  }

  genericOps.clear();
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
        ascend::backend::classifyPhase5ReductionBody(genOp, matrix);
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
    Value totalElems = computeProduct(builder, loc, fullShape);
    // Build a VECCALC accumulator for the full shape.  This is the tensor
    // that will hold the element-wise intermediate results before reduction.
    Value accumLt = allocVeccalc(builder, loc, elemType, fullShape).second;

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
      copyAscendCUnitAttr(genOp.getOperation(), initDup.getOperation());
      builder.create<PipeBarrierOp>(loc,
                                    PipeAttr::get(mlirCtx, Pipe::PIPE_ALL));
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
                    getDynDim(builder, loc, inMemref, srcDimIdx++)));
          else
            srcShapeVals.push_back(
                builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
        }
        Value srcLt = readTensor(builder, loc, inMemref);
        if (Value q = ctx.getQueue(inMemref))
          if (!ctx.getLiveTensor(inMemref))
            rememberQueueRead(ownedInputTensors, q, srcLt);
        auto [bcastTbuf, bcastLt] =
            allocVeccalc(builder, loc, elemType, fullShape);
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
          srcDims.push_back(getDynDim(builder, loc, inMemref, d));
        Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
        for (Value d : srcDims)
          srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
        Value srcGt = builder.create<GlobalTensorOp>(
            loc, GlobalTensorType::get(elemType));
        builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                       /*size=*/Value{});
        Value srcLt =
            copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
                          srcElemCount, &ownedInputTensors);
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
            allocVeccalc(builder, loc, elemType, fullShape);
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
        inputLts[i] =
            copyGmToVecin(builder, loc, elemType, srcGt, totalElems,
                          totalElems, &ownedInputTensors);
      } else {
        // Already VECIN or VECCALC — use readTensor as-is.
        inputLts[i] = readTensor(builder, loc, inMemref);
        if (Value q = ctx.getQueue(inMemref))
          if (!ctx.getLiveTensor(inMemref))
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
              allocVeccalc(builder, loc, elemType, fullShape);
          auto dupOp = builder.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), totalElems);
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
        auto [tmpTbuf, tmpLt] =
            allocVeccalc(builder, loc, elemType, fullShape);
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
        copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        if (dst == accumLt) valToLt[bodyOp.getResult(0)] = accumLt;
      } else if (entry->binaryEmitter) {
        Value lhs = resolve(bodyOp.getOperand(0));
        Value rhs = resolve(bodyOp.getOperand(1));
        if (!lhs || !rhs) continue;
        Value dst = chooseDst(bodyOp.getResult(0));
        entry->binaryEmitter(builder, loc, dst, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(),
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
    Value vecoutLt = writeTensor(builder, loc, outMemref);
    auto layoutAttr = ReduceLayoutAttr::get(mlirCtx, ReduceLayout::AR);
    if (reductionKind == ascend::backend::ComputeKind::ReductionMax) {
      auto reduceOp = builder.create<ReduceMax2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                      /*sharedTmpBuffer=*/Value{});
      copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    } else if (reductionKind == ascend::backend::ComputeKind::ReductionMin) {
      auto reduceOp = builder.create<ReduceMin2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                      /*sharedTmpBuffer=*/Value{});
      copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    } else if (reductionKind == ascend::backend::ComputeKind::ReductionMul) {
      auto reduceOp = builder.create<ReduceProd2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                       /*sharedTmpBuffer=*/Value{});
      copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    } else {
      auto reduceOp = builder.create<ReduceSum2DL2Op>(loc, vecoutLt, accumLt, layoutAttr,
                                                      /*sharedTmpBuffer=*/Value{});
      copyAscendCUnitAttr(genOp.getOperation(), reduceOp.getOperation());
    }

    // Enqueue vecout if it has a queue (VECOUT path).
    if (Value q = ctx.getQueue(outMemref))
      builder.create<TQueBindEnqueTensorOp>(loc, q, vecoutLt);

    genOp.erase();
  }

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
  // Concat semantics are implicitly handled: the VECOUT→GM copy op (inserted by
  // AscendCBufferPlacement + DataMoveConversion) targets a memref subview of the
  // output buffer with the correct byte offset, so Op1 and Op2 results land at
  // the right positions in the concatenated output without any asc.concat op.
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
  // Stamped by --mark-structured-ops: {gather_dim = N : i64} attribute.
  auto isIndexSelectGeneric = [](linalg::GenericOp op) -> bool {
    return op->hasAttr(ascend::kGatherDimAttr);
  };

  // Helper: detect embedding gather (row gather).
  // Stamped by --mark-structured-ops: {embedding_dim = N : i64} attribute.
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
      Value tbM  = getDynDim(builder, loc, outMemref, 0);
      Value dimN = getDynDim(builder, loc, dataMemref, 1);
      Value dimK = getDynDim(builder, loc, outMemref, 1);

      unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;

      // Get indices as a local_tensor.
      // If indices are in VECIN (ms=9), deque from queue.
      // If indices are in GM (ms=0), copy into VECCALC first.
      auto idxMrt = cast<MemRefType>(indicesMemref.getType());
      Type idxElemType = idxMrt.getElementType();
      Value idxCount = getDynDim(builder, loc, indicesMemref, 0);
      Value indicesLt;
      int64_t idxMs = getMemorySpace(indicesMemref.getType());
      if (idxMs == 9 /*VECIN*/ || idxMs == 11 /*VECCALC*/) {
        indicesLt = readTensor(builder, loc, indicesMemref);
      } else {
        // GM: copy indices into a fresh VECCALC buffer.
        SmallVector<Value> idxDims = {idxCount};
        auto [idxTbuf, idxLt] =
            allocVeccalc(builder, loc, idxElemType, idxDims);
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
      Value outTbuf = ctx.getTBuf(outMemref);
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
          loc, QueueType::get(mlirCtx, TPosition::VECIN, 1));
      Value rowBytes = builder.create<arith::MulIOp>(
          loc, paddedDimN,
          builder.create<arith::ConstantIndexOp>(loc, elemBytes));
      Value dataRowQueueDepth =
          builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
      builder.create<TPipeInitQueueOp>(loc, ctx.pipe, dataRowQueue,
                                       dataRowQueueDepth, rowBytes);

      // Gather and post-gather vector ops must run in VECCALC.  Real hardware
      // rejects some VEC reads/writes against VECOUT TBuf slices that the
      // simulator accepts, so rows are copied to VECOUT only after vector work.
      auto gatheredRowAlloc =
          allocVeccalc(builder, loc, elemType, SmallVector<Value>{paddedDimK});
      Value gatheredRowLt = gatheredRowAlloc.second;
      auto gatherSourceRowAlloc =
          allocVeccalc(builder, loc, elemType, SmallVector<Value>{paddedDimN});
      Value gatherSourceRowLt = gatherSourceRowAlloc.second;

      // Pre-op temporaries are reused for every row. Initializing these TPipe
      // buffers inside the row loop exhausts simulator buffer bookkeeping for
      // larger M even though the loop is sequential.
      Value preProcessedRowLt;
      Value preDimN_i32;
      llvm::SmallDenseMap<Value, Value> preInvariantConstLt;
      if (preOp) {
        auto [procTbuf, procLt] =
            allocVeccalc(builder, loc, elemType, SmallVector<Value>{paddedDimN});
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
                allocVeccalc(builder, loc, elemType, SmallVector<Value>{dimN});
            (void)dupTbuf;
            auto dupOp = builder.create<DuplicateL2Op>(
                loc, dupLt, constOp.getResult(), preDimN_i32);
            copyAscendCUnitAttr(preOp.getOperation(), dupOp.getOperation());
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
                copyGmToVeccalc(builder, loc, argElem, argGt, argCount);
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
                allocVeccalc(builder, loc, elemType, SmallVector<Value>{dimK});
            (void)dupTbuf;
            auto dupOp = builder.create<DuplicateL2Op>(
                loc, dupLt, constOp.getResult(), fusedBodyDimK_i32);
            copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
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
                    copyAscendCUnitAttr(preOp.getOperation(), maxOp2.getOperation());
                    preValToLt[maxOp.getResult()] = procLt;
                  }
                } else if (auto addOp2 = dyn_cast<arith::AddFOp>(bodyOp)) {
                  Value lhs = preResolve(addOp2.getLhs()), rhs = preResolve(addOp2.getRhs());
                  if (lhs && rhs) {
                    auto addOp3 = b.create<AddL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                    copyAscendCUnitAttr(preOp.getOperation(), addOp3.getOperation());
                    preValToLt[addOp2.getResult()] = procLt;
                  }
                } else if (auto mulOp2 = dyn_cast<arith::MulFOp>(bodyOp)) {
                  Value lhs = preResolve(mulOp2.getLhs()), rhs = preResolve(mulOp2.getRhs());
                  if (lhs && rhs) {
                    auto mulOp3 = b.create<MulL2Op>(forLoc, procLt, lhs, rhs, dimN_i32);
                    copyAscendCUnitAttr(preOp.getOperation(), mulOp3.getOperation());
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
            // --fuse-gather-elementwise before bufferization.
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
                    return copyGmToVeccalc(b, forLoc, argElem, argGt, argCount);
                  }
                  // Other on-chip inputs (bias, etc.) — read their tensor.
                  return readTensor(b, forLoc, argMemref);
                }
                auto it = postValToLt.find(v);
                if (it != postValToLt.end()) return it->second;
                if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
                  auto [dupTbuf3, dupLt] = allocVeccalc(b, forLoc, elemType,
                                                         SmallVector<Value>{dimK});
                  auto dupOp3 = b.create<DuplicateL2Op>(forLoc, dupLt, constOp.getResult(), dimK_i32v);
                  copyAscendCUnitAttr(postOp.getOperation(), dupOp3.getOperation());
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
                    copyAscendCUnitAttr(postOp.getOperation(), addOp4.getOperation());
                    postValToLt[addOp3.getResult()] = gatheredRowLt;
                  }
                } else if (auto mulOp3 = dyn_cast<arith::MulFOp>(bodyOp)) {
                  Value lhs = postResolve(mulOp3.getLhs()), rhs = postResolve(mulOp3.getRhs());
                  if (lhs && rhs) {
                    auto mulOp4 = b.create<MulL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(postOp.getOperation(), mulOp4.getOperation());
                    postValToLt[mulOp3.getResult()] = gatheredRowLt;
                  }
                } else if (auto maxOp3 = dyn_cast<arith::MaximumFOp>(bodyOp)) {
                  Value lhs = postResolve(maxOp3.getLhs()), rhs = postResolve(maxOp3.getRhs());
                  if (lhs && rhs) {
                    auto maxOp4 = b.create<MaxL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(postOp.getOperation(), maxOp4.getOperation());
                    postValToLt[maxOp3.getResult()] = gatheredRowLt;
                  }
                }
              }
            } else {
              // Walk the fused gather body for arith ops that appear after
              // the memref.load (these were inlined by --fuse-gather-elementwise).
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
                    Value lt = readTensor(b, forLoc, argMemref);
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
                      copyGmToVeccalc(b, forLoc, argElem, argGt, argCount);
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
                    copyAscendCUnitAttr(genOp.getOperation(), addOp5.getOperation());
                    bodyValToLt[addOp4.getResult()] = gatheredRowLt;
                  }
                } else if (auto maxOp4 = dyn_cast<arith::MaximumFOp>(op)) {
                  Value lhs = bodyResolve(maxOp4.getLhs()),
                        rhs = bodyResolve(maxOp4.getRhs());
                  if (lhs && rhs) {
                    auto maxOp5 = b.create<MaxL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(genOp.getOperation(), maxOp5.getOperation());
                    bodyValToLt[maxOp4.getResult()] = gatheredRowLt;
                  }
                } else if (auto mulOp4 = dyn_cast<arith::MulFOp>(op)) {
                  Value lhs = bodyResolve(mulOp4.getLhs()),
                        rhs = bodyResolve(mulOp4.getRhs());
                  if (lhs && rhs) {
                    auto mulOp5 = b.create<MulL2Op>(forLoc, gatheredRowLt, lhs, rhs, dimK_i32v);
                    copyAscendCUnitAttr(genOp.getOperation(), mulOp5.getOperation());
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

      Value srcLt = readTensor(builder, loc, inMemref);
      Value dstLt = writeTensor(builder, loc, outMemref);
      builder.create<TransposeOp>(loc, dstLt, srcLt);

      if (Value q = ctx.getQueue(outMemref))
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
          iterDimSizes[d] = getDynDim(builder, loc, inMemref, d);
        break;
      }
    }
    // Fall back: fill remaining dims from output (all parallel, same rank).
    for (unsigned d = 0; d < iterRank; ++d)
      if (!iterDimSizes[d])
        iterDimSizes[d] = getDynDim(builder, loc, outMemref, d);

    // totalElems is the actual element count for this tile.  Buffer
    // allocation uses the enclosing loop-step upper bound so tail iterations
    // reuse one max-sized queue/tbuf instead of repeatedly InitBuffer-ing.
    Value totalElems = computeProduct(builder, loc, iterDimSizes);
    SmallVector<Value> bufferDimSizes =
        getBufferDimSizes(iterDimSizes, genOp.getOperation());
    Value bufferTotalElems = computeProduct(builder, loc, bufferDimSizes);

    Value outQueue = ctx.getQueue(outMemref);
    Value accumLt;
    if (!outQueue) {
      // Allocate the shared VECCALC accumulator for intermediate results.
      auto [accumTbuf, veccalcAccumLt] =
          allocVeccalc(builder, loc, elemType, bufferDimSizes);
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
            inputLts[i] = copyRank2GmSubviewRowsToVecin(
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
                copyGmToVecin(builder, loc, elemType, srcGt, totalElems,
                              bufferTotalElems, &ownedInputTensors);
          }
        } else {
          inputLts[i] = readTensor(builder, loc, inMemref);
          if (Value q = ctx.getQueue(inMemref))
            if (!ctx.getLiveTensor(inMemref))
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
                  getDynDim(builder, loc, inMemref, srcDimIdx++)));
            else
              srcShapeVals.push_back(
                  builder.create<arith::ConstantIntOp>(loc, builder.getI32Type(), 1));
          }
          Value srcLt = readTensor(builder, loc, inMemref);
          if (Value q = ctx.getQueue(inMemref))
            if (!ctx.getLiveTensor(inMemref))
              rememberQueueRead(ownedInputTensors, q, srcLt);
          auto [bcastTbuf, bcastLt] =
              allocVeccalc(builder, loc, elemType, bufferDimSizes);
          auto bcastOp = builder.create<BroadcastL2Op>(
              loc, bcastLt, srcLt,
              dstShapeVals, srcShapeVals,
              builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
          copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
          inputLts[i] = bcastLt;
        } else {
          // broadcast from GM: copy via VECIN TQue first, then broadcast_l2.
          SmallVector<Value> srcDims;
          for (unsigned d = 0; d < srcRank; ++d)
            srcDims.push_back(getDynDim(builder, loc, inMemref, d));
          Value srcElemCount = builder.create<arith::ConstantIndexOp>(loc, 1);
          for (Value d : srcDims)
            srcElemCount = builder.create<arith::MulIOp>(loc, srcElemCount, d);
          Value srcGt = builder.create<GlobalTensorOp>(
              loc, GlobalTensorType::get(elemType));
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt, inMemref,
                                                         /*size=*/Value{});
          Value srcLt =
              copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
                            srcElemCount, &ownedInputTensors);
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
              allocVeccalc(builder, loc, elemType, bufferDimSizes);
          auto bcastOp = builder.create<BroadcastL2Op>(
              loc, bcastLt, srcLt,
              dstShapeVals, srcShapeVals,
              builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
          copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
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
            copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
                          srcElemCount, &ownedInputTensors);

        auto [transpTbuf, transpLt] =
            allocVeccalc(builder, loc, elemType, bufferDimSizes);
        auto transposeOp = builder.create<TransposeOp>(loc, transpLt, srcVecinLt);
        copyAscendCUnitAttr(genOp.getOperation(), transposeOp.getOperation());
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
          srcDimsVals.push_back(getDynDim(builder, loc, inMemref, d));

        Value srcVecinLt;
        if (inMs == 9 /*VECIN*/) {
          srcVecinLt = readTensor(builder, loc, inMemref);
          if (Value q = ctx.getQueue(inMemref))
            if (!ctx.getLiveTensor(inMemref))
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
              copyGmToVecin(builder, loc, elemType, srcGt, srcElemCount,
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
            allocVeccalc(builder, loc, elemType, bufferDimSizes);
        auto bcastOp = builder.create<BroadcastL2Op>(
            loc, finalLt, srcVecinLt,
            bcastDstShape, bcastSrcShape,
            builder.getI32IntegerAttr(static_cast<int32_t>(iterRank)));
        copyAscendCUnitAttr(genOp.getOperation(), bcastOp.getOperation());
        inputLts[i] = finalLt;
        break;
      }
      } // end switch
    }

    if (outQueue)
      accumLt = allocTensor(builder, loc, outQueue, elemType);

    // ---- Step 2: Walk body and inline arith ops onto VECCALC tensors ----
    Block &bodyBlock = *genOp.getBody();
    unsigned numBodyArgs = bodyBlock.getNumArguments();
    SmallVector<Value> argToLt(numBodyArgs);
    for (unsigned i = 0; i < numInputs; ++i)
      argToLt[i] = inputLts[i];
    argToLt[numInputs] = accumLt;

    llvm::SmallDenseMap<Value, Value> valToLt;
    for (auto &bodyOp : bodyBlock.without_terminator()) {
      auto resolve = [&](Value v) -> Value {
        if (auto ba = dyn_cast<BlockArgument>(v))
          return argToLt[ba.getArgNumber()];
        auto it = valToLt.find(v);
        if (it != valToLt.end()) return it->second;
        // Scalar constant? Fill a fresh VECCALC with duplicate_l2.
        if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
          auto [dupTbuf, dupLt] =
              allocVeccalc(builder, loc, elemType, bufferDimSizes);
          auto dupOp = builder.create<DuplicateL2Op>(loc, dupLt, constOp.getResult(), totalElems);
          copyAscendCUnitAttr(genOp.getOperation(), dupOp.getOperation());
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
        entry->unaryEmitter(builder, loc, accumLt, src, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        valToLt[bodyOp.getResult(0)] = accumLt;
      } else if (entry->binaryEmitter) {
        Value lhs = resolve(bodyOp.getOperand(0));
        Value rhs = resolve(bodyOp.getOperand(1));
        if (!lhs || !rhs) continue;
        entry->binaryEmitter(builder, loc, accumLt, lhs, rhs, totalElems);
        copyAscendCUnitAttr(genOp.getOperation(),
                            &*std::prev(builder.getInsertionPoint()));
        valToLt[bodyOp.getResult(0)] = accumLt;
      }
    }

    if (!ownedInputTensors.empty())
      builder.create<PipeBarrierOp>(loc, PipeAttr::get(mlirCtx, Pipe::PIPE_ALL));
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
    //                   and let the downstream DataMoveConversion handle writeback.
    //   VECCALC (ms=11): accumLt already holds the result; no copy needed.
    //
    // Key insight: the epilogue memref.copy (VECOUT→GM) inserted by
    // AscendCBufferPlacement is converted by DataMoveConversion into a
    // data_copy_l2 with the correct subview offset, so the Concat position
    // is preserved automatically. We just need to enqueue the result tensor.
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

  // --- linalg.matmul → mmad ---
  // --- linalg.batch_matmul GM fallback ---
  SmallVector<linalg::BatchMatmulOp> batchMatmulOps;
  funcOp.walk([&](linalg::BatchMatmulOp op) { batchMatmulOps.push_back(op); });

  for (linalg::BatchMatmulOp batchMatmulOp : batchMatmulOps) {
    Value out = batchMatmulOp.getDpsInitOperand(0)->get();
    if (getMemorySpace(out.getType()) != 0)
      continue;

    builder.setInsertionPoint(batchMatmulOp);
    lowerBatchMatmulToLoops(builder, batchMatmulOp);
    batchMatmulOp.erase();
  }

  batchMatmulOps.clear();
  funcOp.walk([&](linalg::BatchMatmulOp op) { batchMatmulOps.push_back(op); });

  auto getDim = [&](OpBuilder &b, Location loc, Value mem, int64_t d) -> Value {
    auto mrt = cast<MemRefType>(mem.getType());
    if (!ShapedType::isDynamic(mrt.getShape()[d]))
      return b.create<arith::ConstantIndexOp>(loc, mrt.getShape()[d]);
    return b.create<memref::DimOp>(loc, mem, d);
  };

  auto buildMmadParams = [&](OpBuilder &b, Location loc, Value m, Value n,
                             Value k) -> Value {
    auto toI16 = [&](Value idx) -> Value {
      return b.create<arith::IndexCastOp>(loc, b.getI16Type(), idx);
    };
    Value zero8 = b.create<arith::ConstantIntOp>(loc, b.getI8Type(), 0);

    SmallVector<Value> operands = {toI16(m), toI16(n), toI16(k), zero8, zero8,
                                   zero8};
    auto ui16 = IntegerType::get(mlirCtx, 16, IntegerType::Unsigned);
    auto ui8 = IntegerType::get(mlirCtx, 8, IntegerType::Unsigned);
    SmallVector<Type> types = {ui16, ui16, ui16, ui8, ui8, ui8};
    return b.create<ConstructOp>(loc, MmadParamsType::get(mlirCtx), operands,
                                 b.getTypeArrayAttr(types));
  };

  auto matrixElementCount = [&](OpBuilder &b, Location loc, Value mem,
                                int64_t rowDim, int64_t colDim) -> Value {
    Value rows = getDim(b, loc, mem, rowDim);
    Value cols = getDim(b, loc, mem, colDim);
    return b.create<arith::MulIOp>(loc, rows, cols);
  };

  auto batchMatrixByteOffset = [&](OpBuilder &b, Location loc, Value mem,
                                   Value batchIndex, int64_t rowDim,
                                   int64_t colDim) -> Value {
    auto memType = cast<MemRefType>(mem.getType());
    Value rows = getDim(b, loc, mem, rowDim);
    Value cols = getDim(b, loc, mem, colDim);
    Value elems = b.create<arith::MulIOp>(loc, batchIndex, rows);
    elems = b.create<arith::MulIOp>(loc, elems, cols);
    unsigned elemBytes = memType.getElementTypeBitWidth() / 8;
    return b.create<arith::MulIOp>(
        loc, elems, b.create<arith::ConstantIndexOp>(loc, elemBytes));
  };

  // --- linalg.batch_matmul -> batched mmads ---
  for (linalg::BatchMatmulOp batchMatmulOp : batchMatmulOps) {
    Value A = batchMatmulOp.getInputs()[0];
    Value B = batchMatmulOp.getInputs()[1];
    Value C = batchMatmulOp.getOutputs()[0];
    if (getMemorySpace(A.getType()) != 2)
      continue;
    if (getMemorySpace(B.getType()) != 4)
      continue;
    if (getMemorySpace(C.getType()) != 7)
      continue;

    auto aType = dyn_cast<MemRefType>(A.getType());
    auto bType = dyn_cast<MemRefType>(B.getType());
    auto cType = dyn_cast<MemRefType>(C.getType());
    if (!aType || !bType || !cType || aType.getRank() != 3 ||
        bType.getRank() != 3 || cType.getRank() != 3)
      continue;

    Value qA = ctx.getQueue(A), qB = ctx.getQueue(B), qC = ctx.getQueue(C);
    Value tbufA = ctx.getTBuf(A), tbufB = ctx.getTBuf(B);
    Value tbufC = ctx.getTBuf(C);
    if (!qA || !qB || !qC || !tbufA || !tbufB || !tbufC) {
      batchMatmulOp.emitError(
          "missing queue/tbuf for batch_matmul A2/B2/CO1 buffer");
      return failure();
    }

    Location loc = batchMatmulOp.getLoc();
    builder.setInsertionPoint(batchMatmulOp);
    Type elemTypeA = aType.getElementType();
    Type elemTypeB = bType.getElementType();
    Type elemTypeC = cType.getElementType();

    Value tensorA;
    if (!ctx.getLiveTensor(A))
      tensorA = dequeTensor(builder, loc, qA, elemTypeA);
    Value tensorB;
    if (!ctx.getLiveTensor(B))
      tensorB = dequeTensor(builder, loc, qB, elemTypeB);
    Value tensorC = allocTensor(builder, loc, qC, elemTypeC);

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value batch = getDim(builder, loc, C, 0);
    auto forOp = builder.create<scf::ForOp>(loc, zero, batch, one);

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(forOp.getBody());
      Value batchIndex = forOp.getInductionVar();
      Value aSize = matrixElementCount(builder, loc, A, 1, 2);
      Value bSize = matrixElementCount(builder, loc, B, 1, 2);
      Value cSize = matrixElementCount(builder, loc, C, 1, 2);
      Value aOffset = batchMatrixByteOffset(builder, loc, A, batchIndex, 1, 2);
      Value bOffset = batchMatrixByteOffset(builder, loc, B, batchIndex, 1, 2);
      Value cOffset = batchMatrixByteOffset(builder, loc, C, batchIndex, 1, 2);

      Value aSlice = builder.create<TBufGetWithOffsetOp>(
          loc, LocalTensorType::get(elemTypeA), tbufA, aSize, aOffset);
      Value bSlice = builder.create<TBufGetWithOffsetOp>(
          loc, LocalTensorType::get(elemTypeB), tbufB, bSize, bOffset);
      Value cSlice = builder.create<TBufGetWithOffsetOp>(
          loc, LocalTensorType::get(elemTypeC), tbufC, cSize, cOffset);
      Value params = buildMmadParams(builder, loc,
                                     getDim(builder, loc, A, 1),
                                     getDim(builder, loc, B, 2),
                                     getDim(builder, loc, A, 2));
      auto mmadOp = builder.create<MmadOp>(loc, cSlice, aSlice, bSlice, params);
      copyAscendCUnitAttr(batchMatmulOp.getOperation(), mmadOp.getOperation());
    }

    builder.setInsertionPointAfter(forOp);
    builder.create<TQueBindEnqueTensorOp>(loc, qC, tensorC);
    if (tensorA)
      builder.create<TQueBindFreeTensorOp>(loc, qA, tensorA);
    if (tensorB)
      builder.create<TQueBindFreeTensorOp>(loc, qB, tensorB);
    batchMatmulOp.erase();
  }

  // --- linalg.matmul -> mmad ---
  SmallVector<linalg::MatmulOp> matmulOps;
  funcOp.walk([&](linalg::MatmulOp op) { matmulOps.push_back(op); });

  for (linalg::MatmulOp matmulOp : matmulOps) {
    Value A = matmulOp.getInputs()[0];
    Value B = matmulOp.getInputs()[1];
    Value C = matmulOp.getOutputs()[0];
    if (getMemorySpace(A.getType()) == 0 && getMemorySpace(B.getType()) == 0 &&
        getMemorySpace(C.getType()) == 0) {
      builder.setInsertionPoint(matmulOp);
      lowerMatmulToLoops(builder, matmulOp);
      matmulOp.erase();
      continue;
    }
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

    Value tensorA = dequeTensor(builder, loc, qA, elemTypeA);
    Value tensorB = dequeTensor(builder, loc, qB, elemTypeA);

    // CO1 accumulates across the K-loop: alloc before the enclosing for-loop,
    // enque after it, so the queue slot is held for all K iterations.
    // CO1 uses its own element type (f32 for half-precision matmul accumulation).
    auto [tensorC, cHoistFor] =
        allocHoisted(matmulOp, qC, elemTypeC, loc);

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
        kind != linalg::ElementwiseKind::max_signed &&
        kind != linalg::ElementwiseKind::sub &&
        kind != linalg::ElementwiseKind::div &&
        kind != linalg::ElementwiseKind::min_signed)
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
          Value sizeElems = computeElementCount(builder, loc, dst);
          writeTarget = tbufSlice(builder, loc, dst, sizeElems, byteOff);
        }
      }
      if (!writeTarget)
        writeTarget = localDst;
    } else {
      localDst = writeTensor(builder, loc, dst);
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
    } else if (kind == linalg::ElementwiseKind::max_signed) {
      auto maxOp =
          builder.create<MaxL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      copyAscendCUnitAttr(ewOp.getOperation(), maxOp.getOperation());
    } else if (kind == linalg::ElementwiseKind::sub) {
      auto subOp =
          builder.create<SubL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      copyAscendCUnitAttr(ewOp.getOperation(), subOp.getOperation());
    } else if (kind == linalg::ElementwiseKind::div) {
      auto divOp =
          builder.create<DivL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      copyAscendCUnitAttr(ewOp.getOperation(), divOp.getOperation());
    } else if (kind == linalg::ElementwiseKind::min_signed) {
      auto minOp =
          builder.create<MinL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      copyAscendCUnitAttr(ewOp.getOperation(), minOp.getOperation());
    }

    builder.create<PipeBarrierOp>(loc, PipeAttr::get(mlirCtx, Pipe::PIPE_ALL));

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

  // --- linalg.fill -> duplicate_l2 / segment GM writes ---
  //
  // Erased cases (no AscendC op emitted):
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
      auto dstType = dyn_cast<MemRefType>(dst.getType());
      if (!dstType || !dstType.getLayout().isIdentity()) {
        fillOp.emitError("unsupported GM fill layout");
        return failure();
      }

      Location loc = fillOp.getLoc();
      builder.setInsertionPoint(fillOp);

      unsigned rank = static_cast<unsigned>(dstType.getRank());
      Type elemType = dstType.getElementType();
      if (!isDuplicateL2FillSupportedType(elemType)) {
        if (failed(lowerFillToScalarLoops(builder, fillOp))) {
          fillOp.emitError("failed to lower unsupported-dtype GM fill scalar loop");
          return failure();
        }
        fillOp.erase();
        continue;
      }

      Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value segmentCount =
          rank == 0 ? c1 : getDimValue(builder, loc, dst, rank - 1);
      auto fillAlloc =
          allocVeccalc(builder, loc, elemType, SmallVector<Value>{segmentCount});
      Value fillTbuf = fillAlloc.first;
      Value fillLt = fillAlloc.second;
      (void)fillTbuf;
      auto dupOp = builder.create<DuplicateL2Op>(
          loc, fillLt, fillOp.getInputs()[0], segmentCount);
      copyAscendCUnitAttr(fillOp.getOperation(), dupOp.getOperation());

      Value dstGt =
          builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
      unsigned prefixRank = rank == 0 ? 0 : rank - 1;
      SmallVector<Value, 4> prefixUpperBounds;
      prefixUpperBounds.reserve(prefixRank);
      for (unsigned dim = 0; dim < prefixRank; ++dim)
        prefixUpperBounds.push_back(getDimValue(builder, loc, dst, dim));

      SmallVector<Value, 4> prefixIndices;
      auto buildNest = [&](auto &self, unsigned depth) -> LogicalResult {
        if (depth == prefixRank) {
          Value dstOffsetI32;
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
            Value dstOffset =
                builder.create<arith::MulIOp>(loc, flatPrefix, segmentCount);
            dstOffsetI32 = builder.create<arith::IndexCastOp>(
                loc, builder.getI32Type(), dstOffset);
          }
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, dst,
                                                        dstOffsetI32);
          builder.create<DataCopyL2Op>(loc, dstGt, fillLt, segmentCount);
          return success();
        }

        auto forOp = builder.create<scf::ForOp>(
            loc, c0, prefixUpperBounds[depth], c1);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(forOp.getBody());
        prefixIndices.push_back(forOp.getInductionVar());
        LogicalResult result = self(self, depth + 1);
        prefixIndices.pop_back();
        return result;
      };
      if (failed(buildNest(buildNest, 0))) {
        fillOp.emitError("failed to lower GM fill segment copy");
        return failure();
      }
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
    auto dupOp =
        builder.create<DuplicateL2Op>(loc, localDst, fillOp.getInputs()[0], count);
    copyAscendCUnitAttr(fillOp.getOperation(), dupOp.getOperation());

    if (Value q = ctx.getQueue(dst))
      builder.create<TQueBindEnqueTensorOp>(loc, q, localDst);

    fillOp.erase();
  }

  // Lower scalar loops that still touch live on-chip buffers. These loops are
  // produced by conservative fallback paths around cube/vector boundaries; the
  // logical memref has already been materialized as an AscendC local tensor.
  SmallVector<memref::LoadOp> localLoads;
  funcOp.walk([&](memref::LoadOp loadOp) {
    if (getMemorySpace(loadOp.getMemRef().getType()) > 0)
      localLoads.push_back(loadOp);
  });
  for (memref::LoadOp loadOp : localLoads) {
    Value tensor = ctx.getLiveTensor(loadOp.getMemRef());
    if (!tensor)
      continue;
    OpBuilder b(loadOp);
    Value flatIndex = computeContiguousFlatIndex(
        b, loadOp.getLoc(), loadOp.getMemRef(), loadOp.getIndices());
    if (!flatIndex)
      continue;
    Value value = b.create<LocalTensorGetValueOp>(
        loadOp.getLoc(), loadOp.getType(), tensor, flatIndex);
    loadOp.replaceAllUsesWith(value);
    loadOp.erase();
  }

  SmallVector<memref::StoreOp> localStores;
  funcOp.walk([&](memref::StoreOp storeOp) {
    if (getMemorySpace(storeOp.getMemRef().getType()) > 0)
      localStores.push_back(storeOp);
  });
  for (memref::StoreOp storeOp : localStores) {
    Value tensor = ctx.getLiveTensor(storeOp.getMemRef());
    if (!tensor)
      continue;
    OpBuilder b(storeOp);
    Value flatIndex = computeContiguousFlatIndex(
        b, storeOp.getLoc(), storeOp.getMemRef(), storeOp.getIndices());
    if (!flatIndex)
      continue;
    b.create<LocalTensorSetValueOp>(storeOp.getLoc(), tensor, flatIndex,
                                    storeOp.getValue());
    storeOp.erase();
  }

  SmallVector<memref::AllocOp> deadOnChipAllocs;
  funcOp.walk([&](memref::AllocOp allocOp) {
    if (getMemorySpace(allocOp.getType()) > 0 && allocOp->use_empty())
      deadOnChipAllocs.push_back(allocOp);
  });
  for (memref::AllocOp allocOp : deadOnChipAllocs)
    allocOp.erase();

  return success();
}

} // namespace afir
} // namespace mlir
