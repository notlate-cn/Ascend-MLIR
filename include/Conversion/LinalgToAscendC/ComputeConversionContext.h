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

#ifndef CONVERSION_LINALGTOASCENDC_COMPUTECONVERSIONCONTEXT_H
#define CONVERSION_LINALGTOASCENDC_COMPUTECONVERSIONCONTEXT_H

#include "Conversion/LinalgToAscendC/LinalgToAscendCUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace mlir {
namespace afir {

// Bundles the convertCompute() state (`builder`, the buffer `ctx`, the MLIR
// context) shared by the stateful linalg→AscendC compute helpers, so those
// helpers live outside the 2000+ line convertCompute body as methods rather
// than `[&]` lambdas.  Construct once per convertCompute via aggregate init:
//   ComputeCtx cc{builder, ctx, mlirCtx};
// then call e.g. cc.readTensor(b, loc, memref).
struct ComputeCtx {
  OpBuilder &builder;
  AscendCBufferContext &ctx;
  MLIRContext *mlirCtx;

  // Runtime size Value for result `dim` of a (possibly rank-reducing) subview,
  // or null if `memref` is not a subview.
  Value getSubviewSizeValue(OpBuilder &b, Location loc, Value memref,
                            unsigned dim);

  // Product of `dims` (1 if empty).
  Value computeProduct(OpBuilder &b, Location loc, ArrayRef<Value> dims);

  // local_tensor by deque from a queue.
  Value dequeTensor(OpBuilder &b, Location loc, Value queue, Type elemType);

  // local_tensor by alloc from a queue.
  Value allocTensor(OpBuilder &b, Location loc, Value queue, Type elemType);

  // get_tensor from a fresh TBuf (VECCALC temporaries / queue-less buffers).
  Value tbufTensor(OpBuilder &b, Location loc, int64_t ms, Type elemType);

  // Linear byte offset of a 2-D row-major subview into its parent alloc, or
  // null if `memref` is not a subview.
  Value subviewByteOffset(OpBuilder &b, Location loc, Value memref);

  // local_tensor slice via tbuf.get_with_offset, or null if no tbuf registered.
  Value tbufSlice(OpBuilder &b, Location loc, Value memref, Value sizeBytes,
                  Value offsetBytes);

  // Read-side local_tensor for a compute operand (slice / deque / fresh tbuf).
  Value readTensor(OpBuilder &b, Location loc, Value memref);

  // Write-side local_tensor for a compute output (slice / live / alloc / tbuf).
  Value writeTensor(OpBuilder &b, Location loc, Value memref);

  // Allocate a write-side tensor before the nearest enclosing for-loop so the
  // queue slot stays valid across all iterations.  Returns {tensor, hoistFor}
  // (hoistFor null when not inside a loop).
  std::pair<Value, scf::ForOp> allocHoisted(Operation *op, Value queue,
                                            Type elemType, Location loc);

  // Allocate a fresh on-chip VECCALC buffer (tbuf + init_buffer) matching the
  // dynamic sizes.  Returns {tbufVal, localTensorVal}.
  std::pair<Value, Value> allocVeccalc(OpBuilder &b, Location loc, Type elemType,
                                       SmallVector<Value> dynSizes);

  // Copy `elemCount` elements GM→fresh VECIN TQue and return the dequeued
  // VECIN local_tensor.
  Value copyGmToVecin(OpBuilder &b, Location loc, Type elemType, Value srcGt,
                      Value elemCount, Value bufferElemCount = Value{},
                      SmallVectorImpl<std::pair<Value, Value>>
                          *tempVecinTensors = nullptr);

  // Copy a row-strided 2-D tile GM→packed VECIN TQue (one DataCopy per row).
  // `initB` allocates the buffer/queue (hoisted), `b` does the per-trip work.
  // Returns {dequeued tensor, queue} so the caller can FreeTensor it.
  std::pair<Value, Value> copyGmToVecinStrided(OpBuilder &initB, OpBuilder &b,
                                               Location loc, Type elemType,
                                               Value srcGt, Value rows,
                                               Value cols, Value rowStride);

  // Runtime Value for dimension `dim` of a memref.
  Value getDynDim(OpBuilder &b, Location loc, Value memref, unsigned dim);

  // For each of `dims`, the enclosing loop's step bound (full tile size).
  SmallVector<Value> getBufferDimSizes(ArrayRef<Value> dims, Operation *anchor);

  // FreeTensor every (queue, tensor) in `tempVecinTensors`.
  void freeTempVecinTensors(
      OpBuilder &b, Location loc,
      ArrayRef<std::pair<Value, Value>> tempVecinTensors);
};

// convertCompute() phase handlers — each lowers one op category in `funcOp`
// using `cc`, in the order convertCompute calls them (ctx state threads
// through).  Defined in their own TUs (ComputeReduce/Parallel/Ops.cpp).
LogicalResult convertParallelGenerics(func::FuncOp funcOp, ComputeCtx &cc);

} // namespace afir
} // namespace mlir

#endif // CONVERSION_LINALGTOASCENDC_COMPUTECONVERSIONCONTEXT_H
