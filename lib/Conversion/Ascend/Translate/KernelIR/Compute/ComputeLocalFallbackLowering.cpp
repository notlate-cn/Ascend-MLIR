//===- ComputeLocalFallbackLowering.cpp - Local scalar fallback lowering --===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ComputeLoweringInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

using namespace mlir;
using namespace mlir::ascendc;

namespace mlir::ascend {
namespace {

memref::AllocOp getRootAllocOp(Value value) {
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

Value getMemRefDimWithoutDimOp(OpBuilder &builder, Location loc, Value memref,
                               unsigned dim) {
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

Value computeContiguousFlatIndex(OpBuilder &builder, Location loc, Value memref,
                                 ValueRange indices) {
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

} // namespace

LogicalResult lowerLocalScalarFallbackComputes(ComputeLoweringContext &lowering) {
  func::FuncOp funcOp = lowering.funcOp;
  SmallVector<memref::LoadOp> localLoads;
  funcOp.walk([&](memref::LoadOp loadOp) {
    if (getMemorySpace(loadOp.getMemRef().getType()) > 0)
      localLoads.push_back(loadOp);
  });
  for (memref::LoadOp loadOp : localLoads) {
    Value tensor = lowering.ctx.getLiveTensor(loadOp.getMemRef());
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
    Value tensor = lowering.ctx.getLiveTensor(storeOp.getMemRef());
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

} // namespace mlir::ascend
