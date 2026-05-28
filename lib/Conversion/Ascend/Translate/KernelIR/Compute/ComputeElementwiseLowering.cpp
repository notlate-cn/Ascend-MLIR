//===- ComputeElementwiseLowering.cpp - Elementwise compute lowering ------===//
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

LogicalResult lowerElementwiseComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  OpBuilder &builder = lowering.builder;
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
    if (getMemorySpace(dst.getType()) <= 0)
      continue;

    Location loc = ewOp.getLoc();
    builder.setInsertionPoint(ewOp);

    Value localSrc0 = lowering.readTensor(builder, loc, src0);
    Value localSrc1 = lowering.readTensor(builder, loc, src1);

    Value localDst;
    Value writeTarget;
    scf::ForOp dstHoistFor = nullptr;
    int64_t dstMs = getMemorySpace(dst.getType());
    bool isDstSubview = dst.getDefiningOp<memref::SubViewOp>() != nullptr;
    if (dstMs == 10 && lowering.ctx.getQueue(dst)) {
      Value q = lowering.ctx.getQueue(dst);
      auto mrt = cast<MemRefType>(dst.getType());
      auto [t, f] = lowering.allocHoisted(ewOp, q, mrt.getElementType(), loc);
      localDst = t;
      dstHoistFor = f;
      if (isDstSubview) {
        if (Value byteOff = lowering.subviewByteOffset(builder, loc, dst)) {
          Value sizeElems = computeElementCount(builder, loc, dst);
          writeTarget =
              lowering.tbufSlice(builder, loc, dst, sizeElems, byteOff);
        }
      }
      if (!writeTarget)
        writeTarget = localDst;
    } else {
      localDst = lowering.writeTensor(builder, loc, dst);
      writeTarget = localDst;
    }

    Value count = computeElementCount(builder, loc, dst);

    if (kind == linalg::ElementwiseKind::add) {
      auto addOp =
          builder.create<AddL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      lowering.copyAscendCUnitAttr(ewOp.getOperation(), addOp.getOperation());
    } else if (kind == linalg::ElementwiseKind::mul) {
      auto mulOp =
          builder.create<MulL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      lowering.copyAscendCUnitAttr(ewOp.getOperation(), mulOp.getOperation());
    } else if (kind == linalg::ElementwiseKind::max_signed) {
      auto maxOp =
          builder.create<MaxL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      lowering.copyAscendCUnitAttr(ewOp.getOperation(), maxOp.getOperation());
    } else if (kind == linalg::ElementwiseKind::sub) {
      auto subOp =
          builder.create<SubL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      lowering.copyAscendCUnitAttr(ewOp.getOperation(), subOp.getOperation());
    } else if (kind == linalg::ElementwiseKind::div) {
      auto divOp =
          builder.create<DivL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      lowering.copyAscendCUnitAttr(ewOp.getOperation(), divOp.getOperation());
    } else if (kind == linalg::ElementwiseKind::min_signed) {
      auto minOp =
          builder.create<MinL2Op>(loc, writeTarget, localSrc0, localSrc1, count);
      lowering.copyAscendCUnitAttr(ewOp.getOperation(), minOp.getOperation());
    }

    builder.create<PipeBarrierOp>(
        loc, PipeAttr::get(lowering.mlirCtx, Pipe::PIPE_ALL));

    if (Value q = lowering.ctx.getQueue(dst)) {
      if (dstHoistFor) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(dstHoistFor);
        builder.create<TQueBindEnqueTensorOp>(dstHoistFor.getLoc(), q,
                                              localDst);
      } else {
        builder.create<TQueBindEnqueTensorOp>(loc, q, localDst);
      }
    }
    if (Value q = lowering.ctx.getQueue(src0))
      if (!lowering.ctx.getLiveTensor(src0))
        builder.create<TQueBindFreeTensorOp>(loc, q, localSrc0);
    if (Value q = lowering.ctx.getQueue(src1))
      if (!lowering.ctx.getLiveTensor(src1))
        builder.create<TQueBindFreeTensorOp>(loc, q, localSrc1);

    ewOp.erase();
  }

  return success();
}

} // namespace mlir::ascend
