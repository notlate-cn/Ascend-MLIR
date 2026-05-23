#include "ComputeLoweringInternal.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include <algorithm>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir {
namespace afir {

Value getDimValue(OpBuilder &builder, Location loc, Value memref,
                  unsigned dim) {
  auto memrefType = cast<MemRefType>(memref.getType());
  if (!ShapedType::isDynamic(memrefType.getShape()[dim]))
    return builder.create<arith::ConstantIndexOp>(loc,
                                                  memrefType.getShape()[dim]);
  return builder.create<memref::DimOp>(loc, memref, dim);
}

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

void emitStridedGmToLocalCopy(OpBuilder &builder, Location loc, Type elemType,
                              Value dstLt, Value srcGt, Value rows,
                              Value cols, Value srcRowStride,
                              Value srcBaseOffset) {
  bool useBaseOffset = false;
  if (srcBaseOffset) {
    auto constant = srcBaseOffset.getDefiningOp<arith::ConstantIndexOp>();
    useBaseOffset = !constant || constant.value() != 0;
  }
  std::string srcTensor = useBaseOffset ? "_afir_src" : "$1";
  std::string elemTypeStr = getVerbatimScalarTypeName(elemType);
  std::string body = "{\n";
  body += "  uint32_t _afir_rows = (uint32_t)$2;\n";
  body += "  uint32_t _afir_cols = (uint32_t)$3;\n";
  body += "  uint32_t _afir_row_stride = (uint32_t)$4;\n";
  if (useBaseOffset) {
    body += "  uint64_t _afir_base = (uint64_t)$5;\n";
    body += "  AscendC::GlobalTensor<" + elemTypeStr + "> _afir_src;\n";
    body += "  _afir_src.SetGlobalBuffer($1.GetPhyAddr(_afir_base));\n";
  }
  body += "  uint32_t _afir_block_bytes = _afir_cols * sizeof(" +
          elemTypeStr + ");\n";
  body += "  uint32_t _afir_gap_bytes = (_afir_row_stride - _afir_cols) * "
          "sizeof(" + elemTypeStr + ");\n";
  body += "  uint32_t _afir_count = _afir_rows * _afir_cols;\n";
  body += "  if (_afir_gap_bytes == 0u) {\n";
  body += "    if ((_afir_count * sizeof(" + elemTypeStr +
          ")) % 32u == 0u) {\n";
  body += "      AscendC::DataCopy($0, " + srcTensor + ", _afir_count);\n";
  body += "    } else {\n";
  body += "      for (uint32_t _afir_i = 0; _afir_i < _afir_count; "
          "++_afir_i)\n";
  body += "        $0.SetValue(_afir_i, " + srcTensor +
          ".GetValue(_afir_i));\n";
  body += "    }\n";
  body += "  } else if ((_afir_block_bytes % 32u) == 0u && "
          "(_afir_gap_bytes % 32u) == 0u) {\n";
  body += "    AscendC::DataCopyExtParams _afir_params{"
          "static_cast<uint16_t>(_afir_rows), _afir_block_bytes, "
          "_afir_gap_bytes, 0u, 0u};\n";
  body += "    AscendC::DataCopyPadExtParams<" + elemTypeStr +
          "> _afir_pad{false, 0, 0, static_cast<" + elemTypeStr + ">(0)};\n";
  body += "    AscendC::DataCopyPad($0, " + srcTensor +
          ", _afir_params, _afir_pad);\n";
  body += "  } else {\n";
  body += "    for (uint32_t _afir_r = 0; _afir_r < _afir_rows; ++_afir_r) {\n";
  body += "      for (uint32_t _afir_c = 0; _afir_c < _afir_cols; ++_afir_c) "
          "{\n";
  body += "        uint32_t _afir_local = _afir_r * _afir_cols + _afir_c;\n";
  body += "        uint64_t _afir_gm = (uint64_t)_afir_r * _afir_row_stride + "
          "_afir_c;\n";
  body += "        $0.SetValue(_afir_local, " + srcTensor +
          ".GetValue(_afir_gm));\n";
  body += "      }\n";
  body += "    }\n";
  body += "  }\n";
  body += "  $0.SetSize(_afir_count);\n";
  body += "}";
  SmallVector<Value> operands{dstLt, srcGt, rows, cols, srcRowStride};
  if (useBaseOffset)
    operands.push_back(srcBaseOffset);
  builder.create<emitasc::VerbatimOp>(loc, builder.getStringAttr(body),
                                      operands);
}

namespace {

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

} // namespace

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

namespace {

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

} // namespace

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

namespace {

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

} // namespace

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

namespace {

Value createLocalTensorBuffer(OpBuilder &builder, Location loc, Value pipe,
                              TPosition position, Type elemType,
                              Value byteCount) {
  Value tbuf =
      builder.create<TBufOp>(loc, TBufType::get(builder.getContext(), position));
  builder.create<TPipeInitBufferOp>(loc, pipe, tbuf, byteCount);
  return builder.create<TBufGetTensorOp>(
      loc, LocalTensorType::get(elemType), tbuf, /*len=*/Value{});
}

} // namespace

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

namespace {

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

} // namespace

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


namespace {

bool isContiguousRank2View(Value memref);
Value getRank2RowStrideValue(OpBuilder &builder, Location loc, Value memref);

Value materializeIndexValue(OpBuilder &builder, Location loc,
                            OpFoldResult value) {
  if (auto attr = value.dyn_cast<Attribute>())
    return builder.create<arith::ConstantIndexOp>(
        loc, cast<IntegerAttr>(attr).getInt());
  return value.get<Value>();
}

bool isKnownZeroIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value() == 0;
  return false;
}

FailureOr<std::pair<Value, Value>>
getRank2BaseAndOffset(OpBuilder &builder, Location loc, Value memref) {
  Value current = memref;
  Value totalOffset = builder.create<arith::ConstantIndexOp>(loc, 0);

  while (true) {
    if (auto castOp = current.getDefiningOp<memref::CastOp>()) {
      current = castOp.getSource();
      continue;
    }

    auto subview = current.getDefiningOp<memref::SubViewOp>();
    if (!subview)
      break;

    SmallVector<OpFoldResult> offsets = subview.getMixedOffsets();
    if (offsets.size() < 2)
      return failure();

    Value rowOffset = materializeIndexValue(builder, loc, offsets[0]);
    Value colOffset = materializeIndexValue(builder, loc, offsets[1]);
    Value rowStride =
        getRank2RowStrideValue(builder, loc, subview.getResult());
    if (!rowStride)
      return failure();

    Value linearOffset = colOffset;
    if (!isKnownZeroIndex(rowOffset)) {
      Value rowPart = builder.create<arith::MulIOp>(loc, rowOffset, rowStride);
      linearOffset = isKnownZeroIndex(colOffset)
                         ? rowPart
                         : builder.create<arith::AddIOp>(loc, rowPart,
                                                         colOffset);
    }
    if (!isKnownZeroIndex(linearOffset))
      totalOffset = isKnownZeroIndex(totalOffset)
                        ? linearOffset
                        : builder.create<arith::AddIOp>(loc, totalOffset,
                                                        linearOffset);

    current = subview.getSource();
  }

  return std::make_pair(current, totalOffset);
}

} // namespace

LogicalResult lowerRank2GmTransposeToLocalDataCopy(
    OpBuilder &builder, Location loc, Value inMemref, Value outMemref,
    ArrayRef<int64_t> permutation, Value pipe) {
  auto inType = dyn_cast<MemRefType>(inMemref.getType());
  auto outType = dyn_cast<MemRefType>(outMemref.getType());
  if (!inType || !outType || inType.getRank() != 2 || outType.getRank() != 2)
    return failure();
  if (!isContiguousRank2View(outMemref))
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
  FailureOr<std::pair<Value, Value>> srcBaseAndOffset =
      getRank2BaseAndOffset(builder, loc, inMemref);
  FailureOr<std::pair<Value, Value>> dstBaseAndOffset =
      getRank2BaseAndOffset(builder, loc, outMemref);
  if (failed(srcBaseAndOffset) || failed(dstBaseAndOffset))
    return failure();

  Value srcGt =
      builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
  builder.create<GlobalTensorSetGlobalBufferOp>(loc, srcGt,
                                                srcBaseAndOffset->first,
                                                /*size=*/Value{});
  Value srcLt = createLocalTensorBuffer(builder, loc, pipe, TPosition::VECIN,
                                        elemType, byteCount);
  if (isContiguousRank2View(inMemref)) {
    Value src = srcGt;
    if (!isKnownZeroIndex(srcBaseAndOffset->second))
      src = builder.create<GlobalTensorBracketOp>(
          loc, GlobalTensorType::get(elemType), srcGt,
          srcBaseAndOffset->second);
    builder.create<DataCopyL2Op>(loc, srcLt, src, elemCount);
  } else {
    Value rows = getDimValue(builder, loc, inMemref, 0);
    Value cols = getDimValue(builder, loc, inMemref, 1);
    Value rowStride = getRank2RowStrideValue(builder, loc, inMemref);
    if (!rowStride)
      return failure();
    emitStridedGmToLocalCopy(builder, loc, elemType, srcLt, srcGt, rows, cols,
                             rowStride, srcBaseAndOffset->second);
  }

  Value dstLt = createLocalTensorBuffer(builder, loc, pipe, TPosition::VECCALC,
                                        elemType, byteCount);
  builder.create<TransposeOp>(loc, dstLt, srcLt);

  Value dstGt =
      builder.create<GlobalTensorOp>(loc, GlobalTensorType::get(elemType));
  builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt,
                                                dstBaseAndOffset->first,
                                                /*size=*/Value{});
  Value dst = dstGt;
  if (!isKnownZeroIndex(dstBaseAndOffset->second))
    dst = builder.create<GlobalTensorBracketOp>(
        loc, GlobalTensorType::get(elemType), dstGt,
        dstBaseAndOffset->second);
  builder.create<DataCopyL2Op>(loc, dst, dstLt, elemCount);
  return success();
}

namespace {

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

bool isRank2SwapPermutation(ArrayRef<int64_t> permutation) {
  return permutation.size() == 2 && permutation[0] == 1 &&
         permutation[1] == 0;
}

std::optional<int64_t> getStaticIndexValue(OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>())
    return cast<IntegerAttr>(attr).getInt();
  if (auto value = ofr.dyn_cast<Value>())
    if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
      return constant.value();
  return std::nullopt;
}

bool isContiguousRank2View(Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || type.getRank() != 2)
    return false;
  if (type.getLayout().isIdentity())
    return true;

  auto [strides, offset] = type.getStridesAndOffset();
  (void)offset;
  if (strides.size() != 2 || strides[0] == ShapedType::kDynamic ||
      strides[1] != 1)
    return false;

  if (!ShapedType::isDynamic(type.getShape()[1]))
    return type.getShape()[1] == strides[0];

  auto subview = memref.getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return false;
  SmallVector<OpFoldResult> sizes = subview.getMixedSizes();
  if (sizes.size() < 2)
    return false;
  std::optional<int64_t> innerSize = getStaticIndexValue(sizes[1]);
  return innerSize && *innerSize == strides[0];
}

Value getRank2RowStrideValue(OpBuilder &builder, Location loc, Value memref) {
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
    return getDimValue(builder, loc, root, 1);

  return getDimValue(builder, loc, memref, 1);
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

LogicalResult materializeSelectedTransposeTiles(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  SmallVector<linalg::TransposeOp> candidates;
  funcOp.walk([&](linalg::TransposeOp op) {
    if (op->getParentOfType<scf::ForOp>())
      return;
    candidates.push_back(op);
  });

  for (linalg::TransposeOp transposeOp : candidates) {
    FailureOr<TransposeLoweringSpec> spec =
        buildTransposeLoweringSpec(transposeOp);
    if (failed(spec) || !isRank2SwapPermutation(spec->permutation))
      continue;

    Value inMemref = transposeOp.getDpsInputOperand(0)->get();
    Value outMemref = transposeOp.getDpsInitOperand(0)->get();
    auto inType = dyn_cast<MemRefType>(inMemref.getType());
    auto outType = dyn_cast<MemRefType>(outMemref.getType());
    if (!inType || !outType || inType.getRank() != 2 ||
        outType.getRank() != 2)
      continue;
    if (getMemorySpace(inType) != 0 || getMemorySpace(outType) != 0)
      continue;
    if (inType.getElementType() != outType.getElementType())
      continue;

    int64_t tileRows = ShapedType::kDynamic;
    int64_t tileCols = ShapedType::kDynamic;
    ArrayRef<int64_t> outShape = outType.getShape();
    auto deriveStaticFullInnerTile = [&]() -> bool {
      if (ShapedType::isDynamic(outShape[0]) ||
          ShapedType::isDynamic(outShape[1]))
        return false;
      unsigned elemBits = outType.getElementTypeBitWidth();
      if (elemBits == 0 || elemBits % 8 != 0)
        return false;
      int64_t elemBytes = static_cast<int64_t>(elemBits / 8);
      constexpr int64_t kTransposeBufferBudgetBytes = 48 * 1024;
      int64_t fullBytes = outShape[0] * outShape[1] * elemBytes;
      if (fullBytes <= kTransposeBufferBudgetBytes)
        return false;
      tileCols = outShape[1];
      tileRows =
          std::max<int64_t>(1, kTransposeBufferBudgetBytes /
                                   std::max<int64_t>(1, tileCols * elemBytes));
      tileRows = std::min(tileRows, outShape[0]);
      return true;
    };

    auto selectedTile = transposeOp->getAttrOfType<DenseI64ArrayAttr>(
        ascend::kScheduleSelectedTileShapeAttr);
    if (selectedTile) {
      if (selectedTile.asArrayRef().size() < 2)
        return transposeOp.emitError("selected rank-2 transpose tile requires "
                                     "at least two dimensions");
      tileRows = selectedTile.asArrayRef()[0];
      tileCols = selectedTile.asArrayRef()[1];
      if (ShapedType::isDynamic(tileRows) || tileRows <= 0)
        return transposeOp.emitError(
            "selected transpose tile requires a static positive outer tile");
      if (ShapedType::isDynamic(tileCols) || tileCols <= 0)
        return transposeOp.emitError(
            "selected transpose tile requires a static positive inner tile");
      if (!ShapedType::isDynamic(outShape[1]) && tileCols != outShape[1]) {
        // GM rank-2 swap lowering consumes full output rows. A generic UB tile
        // may cap the wrong dimension, so fall back to a full-inner row slice.
        if (!deriveStaticFullInnerTile())
          continue;
      }
    } else {
      if (!deriveStaticFullInnerTile())
        continue;
    }

    if (ShapedType::isDynamic(outType.getShape()[1]) ||
        tileCols != outType.getShape()[1])
      continue;

    Location loc = transposeOp.getLoc();
    builder.setInsertionPoint(transposeOp);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value step = builder.create<arith::ConstantIndexOp>(loc, tileRows);
    Value rows = getDimValue(builder, loc, outMemref, 0);
    auto forOp = builder.create<scf::ForOp>(loc, zero, rows, step);
    forOp->setAttr("ascendc.parallel", builder.getBoolAttr(true));

    OpBuilder bodyBuilder(funcOp.getContext());
    bodyBuilder.setInsertionPointToStart(forOp.getBody());
    Value remaining =
        bodyBuilder.create<arith::SubIOp>(loc, rows, forOp.getInductionVar());
    Value tileRowsValue =
        bodyBuilder.create<arith::MinSIOp>(loc, step, remaining);

    OpFoldResult zeroAttr = bodyBuilder.getIndexAttr(0);
    OpFoldResult oneAttr = bodyBuilder.getIndexAttr(1);
    OpFoldResult fullInnerAttr = bodyBuilder.getIndexAttr(tileCols);
    OpFoldResult rowOffset = forOp.getInductionVar();
    OpFoldResult tileRowsSize = tileRowsValue;

    Value inputTile =
        bodyBuilder
            .create<memref::SubViewOp>(
                loc, inMemref,
                SmallVector<OpFoldResult>{zeroAttr, rowOffset},
                SmallVector<OpFoldResult>{fullInnerAttr, tileRowsSize},
                SmallVector<OpFoldResult>{oneAttr, oneAttr})
            .getResult();
    Value outputTile =
        bodyBuilder
            .create<memref::SubViewOp>(
                loc, outMemref,
                SmallVector<OpFoldResult>{rowOffset, zeroAttr},
                SmallVector<OpFoldResult>{tileRowsSize, fullInnerAttr},
                SmallVector<OpFoldResult>{oneAttr, oneAttr})
            .getResult();

    IRMapping mapper;
    mapper.map(inMemref, inputTile);
    mapper.map(outMemref, outputTile);
    Operation *cloned = bodyBuilder.clone(*transposeOp, mapper);
    if (selectedTile)
      cloned->removeAttr(ascend::kScheduleSelectedTileShapeAttr);
    transposeOp.erase();
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

} // namespace afir
} // namespace mlir
