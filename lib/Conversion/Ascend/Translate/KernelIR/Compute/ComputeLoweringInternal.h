//===- ComputeLoweringInternal.h - Ascend compute lowering internals -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTE_COMPUTELOWERINGINTERNAL_H
#define ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTE_COMPUTELOWERINGINTERNAL_H

#include "Conversion/Ascend/Translate/KernelIR/KernelIRUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
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

struct IndexingMapAnalysis {
  enum class Kind {
    Identity,
    PureBroadcast,
    PureTranspose,
    BroadcastTranspose,
  };
  Kind kind;
  SmallVector<int64_t> permutation;
  SmallVector<int64_t> broadcastDims;
};

using OwnedQueueTensor = std::pair<Value, Value>;

LogicalResult prepareComputeLoweringPreconditions(func::FuncOp funcOp);

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

Value ceilToMultipleIndex(OpBuilder &builder, Location loc, Value value,
                          int64_t divisor);
void emitLocalToLocalScalarCopy(OpBuilder &builder, Location loc, Type elemType,
                                Value dstLt, Value srcLt, Value count);
void emitLocalTensorZeroPad(OpBuilder &builder, Location loc, Type elemType,
                            Value tensor, Value begin, Value end);
IndexingMapAnalysis analyzeIndexingMap(AffineMap map, unsigned iterRank);
bool isBroadcastMap(AffineMap map, unsigned iterRank);
std::pair<Value, Value> allocVeccalc(ComputeLoweringContext &lowering,
                                     OpBuilder &builder, Location loc,
                                     Type elemType, ArrayRef<Value> dynSizes);
void freeOwnedQueueTensors(OpBuilder &builder, Location loc,
                           ArrayRef<OwnedQueueTensor> ownedTensors);
void rememberQueueRead(SmallVectorImpl<OwnedQueueTensor> &ownedTensors,
                       Value queue, Value tensor);
Value copyGmToVecin(ComputeLoweringContext &lowering, OpBuilder &builder,
                    Location loc, Type elemType, Value srcGt, Value elemCount,
                    Value bufferElemCount,
                    SmallVectorImpl<OwnedQueueTensor> *ownedTensors = nullptr);
Value copyGmToVecinScalar(
    ComputeLoweringContext &lowering, OpBuilder &builder, Location loc,
    Type elemType, Value srcGt, Value elemCount, Value bufferElemCount,
    SmallVectorImpl<OwnedQueueTensor> *ownedTensors = nullptr);
Value copyGmToVeccalc(ComputeLoweringContext &lowering, OpBuilder &builder,
                      Location loc, Type elemType, Value srcGt,
                      Value elemCount);
Value getDynDim(ComputeLoweringContext &lowering, OpBuilder &builder,
                Location loc, Value memref, unsigned dim);
SmallVector<Value> getBufferDimSizes(ComputeLoweringContext &lowering,
                                     ArrayRef<Value> dims, Operation *anchor);
bool isRank2GmSubview(Value memref);
bool isContiguousRank2GmSubview(Value memref);
Value getRank2RowStride(ComputeLoweringContext &lowering, OpBuilder &builder,
                        Location loc, Value memref);
Value copyRank2GmSubviewRowsToVecin(
    ComputeLoweringContext &lowering, OpBuilder &builder, Location loc,
    Type elemType, Value srcMemref, Value bufferElemCount,
    SmallVectorImpl<OwnedQueueTensor> *ownedTensors = nullptr);

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

#endif // ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_COMPUTE_COMPUTELOWERINGINTERNAL_H
