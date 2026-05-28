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

} // namespace ascend
} // namespace mlir
