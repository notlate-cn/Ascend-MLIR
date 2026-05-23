//===- ComputeLoweringInternal.h - Ascend compute lowering internals -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTELOWERINGINTERNAL_H
#define ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTELOWERINGINTERNAL_H

#include "Conversion/Ascend/Translate/KernelIR/KernelIRUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include <utility>

namespace mlir::afir {

struct ComputeLoweringContext {
  func::FuncOp funcOp;
  AscendCBufferContext &ctx;
  MLIRContext *mlirCtx;
  OpBuilder builder;

  ComputeLoweringContext(func::FuncOp funcOp, AscendCBufferContext &ctx)
      : funcOp(funcOp), ctx(ctx), mlirCtx(funcOp.getContext()),
        builder(mlirCtx) {}

  void copyAscendCUnitAttr(Operation *src, Operation *dst) const;
  Value getEnclosingLoopStepBound(Value value, Operation *anchor) const;
  Value getSubviewSizeValue(OpBuilder &builder, Location loc, Value memref,
                            unsigned dim) const;
  Value computeProduct(OpBuilder &builder, Location loc,
                       ArrayRef<Value> dims) const;
  Value dequeTensor(OpBuilder &builder, Location loc, Value queue,
                    Type elemType) const;
  Value allocTensor(OpBuilder &builder, Location loc, Value queue,
                    Type elemType) const;
  Value tbufTensor(OpBuilder &builder, Location loc, int64_t memorySpace,
                   Type elemType) const;
  Value subviewByteOffset(OpBuilder &builder, Location loc, Value memref) const;
  Value tbufSlice(OpBuilder &builder, Location loc, Value memref,
                  Value sizeElems, Value offsetBytes) const;
  Value readTensor(OpBuilder &builder, Location loc, Value memref) const;
  Value writeTensor(OpBuilder &builder, Location loc, Value memref) const;
  scf::ForOp getEnclosingFor(Operation *op) const;
  std::pair<Value, scf::ForOp> allocHoisted(Operation *op, Value queue,
                                            Type elemType, Location loc);
};

Value getDimValue(OpBuilder &builder, Location loc, Value memref,
                  unsigned dim);
void emitStridedGmToLocalCopy(OpBuilder &builder, Location loc, Type elemType,
                              Value dstLt, Value srcGt, Value rows,
                              Value cols, Value srcRowStride,
                              Value srcBaseOffset = Value{});
bool isPureYieldGeneric(linalg::GenericOp op);
bool isGmAllParallelGeneric(linalg::GenericOp op);
bool isGmScalarLoopGeneric(linalg::GenericOp op);
LogicalResult lowerTransposeToLoops(OpBuilder &builder, Location loc,
                                    Value inMemref, Value outMemref,
                                    ArrayRef<int64_t> permutation);
LogicalResult lowerPureYieldGenericToLoops(OpBuilder &builder,
                                           linalg::GenericOp op);
LogicalResult lowerAllParallelGenericToLoops(OpBuilder &builder,
                                             linalg::GenericOp op);
LogicalResult lowerGmGenericToScalarLoops(OpBuilder &builder,
                                          linalg::GenericOp op);
LogicalResult lowerRank2GmTransposeToLocalDataCopy(
    OpBuilder &builder, Location loc, Value inMemref, Value outMemref,
    ArrayRef<int64_t> permutation, Value pipe);

LogicalResult lowerScalarFallbackComputes(ComputeLoweringContext &lowering);
LogicalResult lowerTransposeComputes(ComputeLoweringContext &lowering);
LogicalResult lowerReductionComputes(ComputeLoweringContext &lowering);
LogicalResult lowerParallelGenericComputes(ComputeLoweringContext &lowering);
LogicalResult lowerGatherCompute(ComputeLoweringContext &lowering,
                                 linalg::GenericOp genOp,
                                 ArrayRef<linalg::GenericOp> parallelGenericOps);
LogicalResult lowerMatmulComputes(ComputeLoweringContext &lowering);
LogicalResult lowerElementwiseComputes(ComputeLoweringContext &lowering);
LogicalResult lowerFillComputes(ComputeLoweringContext &lowering);
LogicalResult lowerLocalScalarFallbackComputes(ComputeLoweringContext &lowering);

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTELOWERINGINTERNAL_H
