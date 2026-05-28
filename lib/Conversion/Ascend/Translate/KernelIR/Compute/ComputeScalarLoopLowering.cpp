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
namespace ascend {

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
  std::string srcTensor = useBaseOffset ? "_ascend_src" : "$1";
  std::string elemTypeStr = getVerbatimScalarTypeName(elemType);
  std::string body = "{\n";
  body += "  uint32_t _ascend_rows = (uint32_t)$2;\n";
  body += "  uint32_t _ascend_cols = (uint32_t)$3;\n";
  body += "  uint32_t _ascend_row_stride = (uint32_t)$4;\n";
  if (useBaseOffset) {
    body += "  uint64_t _ascend_base = (uint64_t)$5;\n";
    body += "  AscendC::GlobalTensor<" + elemTypeStr + "> _ascend_src;\n";
    body += "  _ascend_src.SetGlobalBuffer($1.GetPhyAddr(_ascend_base));\n";
  }
  body += "  uint32_t _ascend_block_bytes = _ascend_cols * sizeof(" +
          elemTypeStr + ");\n";
  body += "  uint32_t _ascend_gap_bytes = (_ascend_row_stride - _ascend_cols) * "
          "sizeof(" + elemTypeStr + ");\n";
  body += "  uint32_t _ascend_count = _ascend_rows * _ascend_cols;\n";
  body += "  if (_ascend_gap_bytes == 0u) {\n";
  body += "    if ((_ascend_count * sizeof(" + elemTypeStr +
          ")) % 32u == 0u) {\n";
  body += "      AscendC::DataCopy($0, " + srcTensor + ", _ascend_count);\n";
  body += "    } else {\n";
  body += "      for (uint32_t _ascend_i = 0; _ascend_i < _ascend_count; "
          "++_ascend_i)\n";
  body += "        $0.SetValue(_ascend_i, " + srcTensor +
          ".GetValue(_ascend_i));\n";
  body += "    }\n";
  body += "  } else if ((_ascend_block_bytes % 32u) == 0u && "
          "(_ascend_gap_bytes % 32u) == 0u) {\n";
  body += "    AscendC::DataCopyExtParams _ascend_params{"
          "static_cast<uint16_t>(_ascend_rows), _ascend_block_bytes, "
          "_ascend_gap_bytes, 0u, 0u};\n";
  body += "    AscendC::DataCopyPadExtParams<" + elemTypeStr +
          "> _ascend_pad{false, 0, 0, static_cast<" + elemTypeStr + ">(0)};\n";
  body += "    AscendC::DataCopyPad($0, " + srcTensor +
          ", _ascend_params, _ascend_pad);\n";
  body += "  } else {\n";
  body += "    for (uint32_t _ascend_r = 0; _ascend_r < _ascend_rows; ++_ascend_r) {\n";
  body += "      for (uint32_t _ascend_c = 0; _ascend_c < _ascend_cols; ++_ascend_c) "
          "{\n";
  body += "        uint32_t _ascend_local = _ascend_r * _ascend_cols + _ascend_c;\n";
  body += "        uint64_t _ascend_gm = (uint64_t)_ascend_r * _ascend_row_stride + "
          "_ascend_c;\n";
  body += "        $0.SetValue(_ascend_local, " + srcTensor +
          ".GetValue(_ascend_gm));\n";
  body += "      }\n";
  body += "    }\n";
  body += "  }\n";
  body += "  $0.SetSize(_ascend_count);\n";
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

Value materializeIndexValue(OpBuilder &builder, Location loc,
                            OpFoldResult value) {
  if (auto attr = value.dyn_cast<Attribute>())
    return builder.create<arith::ConstantIndexOp>(
        loc, cast<IntegerAttr>(attr).getInt());
  return cast<Value>(value);
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

} // namespace ascend
} // namespace mlir
