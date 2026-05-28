//===- ComputeFillLowering.cpp - Fill compute lowering --------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ComputeLoweringInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir::ascend {
namespace {

std::pair<Value, Value> allocVeccalcForFill(ComputeLoweringContext &lowering,
                                            OpBuilder &builder, Location loc,
                                            Type elemType,
                                            ArrayRef<Value> dynSizes) {
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

bool isDuplicateL2FillSupportedType(Type elemType) {
  if (elemType.isF16() || elemType.isF32() || elemType.isBF16())
    return true;
  if (auto intType = dyn_cast<IntegerType>(elemType))
    return intType.getWidth() == 16 || intType.getWidth() == 32;
  return false;
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

} // namespace

LogicalResult lowerFillComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  OpBuilder &builder = lowering.builder;
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
      auto fillAlloc = allocVeccalcForFill(
          lowering, builder, loc, elemType, SmallVector<Value>{segmentCount});
      Value fillTbuf = fillAlloc.first;
      Value fillLt = fillAlloc.second;
      (void)fillTbuf;
      auto dupOp = builder.create<DuplicateL2Op>(
          loc, fillLt, fillOp.getInputs()[0], segmentCount);
      lowering.copyAscendCUnitAttr(fillOp.getOperation(), dupOp.getOperation());

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
            dstOffsetI32 =
                builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                                   dstOffset);
          }
          builder.create<GlobalTensorSetGlobalBufferOp>(loc, dstGt, dst,
                                                        dstOffsetI32);
          builder.create<DataCopyL2Op>(loc, dstGt, fillLt, segmentCount);
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
      if (failed(buildNest(buildNest, 0))) {
        fillOp.emitError("failed to lower GM fill segment copy");
        return failure();
      }
      fillOp.erase();
      continue;
    }

    if (ms == 7 || ms == 10) {
      fillOp.erase();
      continue;
    }

    Location loc = fillOp.getLoc();
    builder.setInsertionPoint(fillOp);

    Value localDst = lowering.writeTensor(builder, loc, dst);
    Value count = computeElementCount(builder, loc, dst);
    auto dupOp =
        builder.create<DuplicateL2Op>(loc, localDst, fillOp.getInputs()[0], count);
    lowering.copyAscendCUnitAttr(fillOp.getOperation(), dupOp.getOperation());

    if (Value q = lowering.ctx.getQueue(dst))
      builder.create<TQueBindEnqueTensorOp>(loc, q, localDst);

    fillOp.erase();
  }

  return success();
}

} // namespace mlir::ascend
