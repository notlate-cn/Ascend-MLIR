//===- ComputeMatmulLowering.cpp - Matmul compute lowering ----------------===//
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

} // namespace

LogicalResult lowerMatmulComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  OpBuilder &builder = lowering.builder;
  SmallVector<linalg::BatchMatmulOp> batchMatmulOps;
  funcOp.walk([&](linalg::BatchMatmulOp op) { batchMatmulOps.push_back(op); });

  for (linalg::BatchMatmulOp batchMatmulOp : batchMatmulOps) {
    Value out = batchMatmulOp.getDpsInitOperand(0)->get();
    if (getMemorySpace(out.getType()) == 0) {
      batchMatmulOp.emitError(
          "unsupported GM-output batch_matmul lowering: materialize GM tensors "
          "through cube/local buffers before lowering; scalar loop fallback is "
          "disabled");
      return failure();
    }
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
    auto ui16 = IntegerType::get(lowering.mlirCtx, 16, IntegerType::Unsigned);
    auto ui8 = IntegerType::get(lowering.mlirCtx, 8, IntegerType::Unsigned);
    SmallVector<Type> types = {ui16, ui16, ui16, ui8, ui8, ui8};
    return b.create<ConstructOp>(loc, MmadParamsType::get(lowering.mlirCtx),
                                 operands, b.getTypeArrayAttr(types));
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

    Value qA = lowering.ctx.getQueue(A), qB = lowering.ctx.getQueue(B),
          qC = lowering.ctx.getQueue(C);
    Value tbufA = lowering.ctx.getTBuf(A), tbufB = lowering.ctx.getTBuf(B);
    Value tbufC = lowering.ctx.getTBuf(C);
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
    if (!lowering.ctx.getLiveTensor(A))
      tensorA = lowering.dequeTensor(builder, loc, qA, elemTypeA);
    Value tensorB;
    if (!lowering.ctx.getLiveTensor(B))
      tensorB = lowering.dequeTensor(builder, loc, qB, elemTypeB);
    Value tensorC = lowering.allocTensor(builder, loc, qC, elemTypeC);

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
      Value params = buildMmadParams(builder, loc, getDim(builder, loc, A, 1),
                                     getDim(builder, loc, B, 2),
                                     getDim(builder, loc, A, 2));
      auto mmol = builder.create<MmadOp>(loc, cSlice, aSlice, bSlice, params);
      lowering.copyAscendCUnitAttr(batchMatmulOp.getOperation(),
                                   mmol.getOperation());
    }

    builder.setInsertionPointAfter(forOp);
    builder.create<TQueBindEnqueTensorOp>(loc, qC, tensorC);
    if (tensorA)
      builder.create<TQueBindFreeTensorOp>(loc, qA, tensorA);
    if (tensorB)
      builder.create<TQueBindFreeTensorOp>(loc, qB, tensorB);
    batchMatmulOp.erase();
  }

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
    if (getMemorySpace(A.getType()) != 2)
      continue;
    if (getMemorySpace(B.getType()) != 4)
      continue;
    if (getMemorySpace(C.getType()) != 7)
      continue;

    Value qA = lowering.ctx.getQueue(A), qB = lowering.ctx.getQueue(B),
          qC = lowering.ctx.getQueue(C);
    if (!qA || !qB || !qC) {
      matmulOp.emitError("missing queue for matmul A2/B2/CO1 buffer");
      return failure();
    }

    Location loc = matmulOp.getLoc();
    builder.setInsertionPoint(matmulOp);
    Type elemTypeA = cast<MemRefType>(A.getType()).getElementType();
    Type elemTypeC = cast<MemRefType>(C.getType()).getElementType();

    Value tensorA = lowering.dequeTensor(builder, loc, qA, elemTypeA);
    Value tensorB = lowering.dequeTensor(builder, loc, qB, elemTypeA);

    auto [tensorC, cHoistFor] =
        lowering.allocHoisted(matmulOp, qC, elemTypeC, loc);

    auto toI16 = [&](Value idx) -> Value {
      return builder.create<arith::IndexCastOp>(loc, builder.getI16Type(), idx);
    };
    auto getDim = [&](Value mem, int64_t d) -> Value {
      auto mrt = cast<MemRefType>(mem.getType());
      if (!ShapedType::isDynamic(mrt.getShape()[d]))
        return builder.create<arith::ConstantIndexOp>(loc, mrt.getShape()[d]);
      return builder.create<memref::DimOp>(loc, mem, d);
    };

    Value mVal = toI16(getDim(A, 0));
    Value kVal = toI16(getDim(A, 1));
    Value nVal = toI16(getDim(B, 1));
    Value zero8 =
        builder.create<arith::ConstantIntOp>(loc, builder.getI8Type(), 0);

    SmallVector<Value> mmolOperands = {mVal, nVal, kVal, zero8, zero8, zero8};
    auto ui16 = IntegerType::get(lowering.mlirCtx, 16, IntegerType::Unsigned);
    auto ui8 = IntegerType::get(lowering.mlirCtx, 8, IntegerType::Unsigned);
    SmallVector<Type> mmolTypes = {ui16, ui16, ui16, ui8, ui8, ui8};
    Value mmolParams = builder.create<ConstructOp>(
        loc, MmadParamsType::get(lowering.mlirCtx), mmolOperands,
        builder.getTypeArrayAttr(mmolTypes));
    auto mmolOp = builder.create<MmadOp>(loc, tensorC, tensorA, tensorB, mmolParams);
    lowering.copyAscendCUnitAttr(matmulOp.getOperation(), mmolOp.getOperation());

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

  return success();
}

} // namespace mlir::ascend
