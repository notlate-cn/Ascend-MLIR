//===- LinalgToAscendCUtils.h - Linalg to AscendC helpers -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_LOWERING_LINALG_TO_ASCENDC_UTILS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_LOWERING_LINALG_TO_ASCENDC_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

/// Shared context for the LinalgToAscendC pass.
/// Holds the single pipe and the per-alloc queue map, built once in the pass
/// entry and passed to both the data-move and compute conversion steps.
struct AscendCBufferContext {
  /// The single TPipe for the function (one per kernel).
  Value pipe;

  /// Maps each on-chip memref.alloc result → its QueueOp result.
  /// Built from memref.alloc ops with memory_space > 0.
  llvm::DenseMap<Value, Value> allocToQueue;

  /// Maps each on-chip memref.alloc result → its TBufOp result.
  /// Built from memref.alloc ops with memory_space > 0.
  llvm::DenseMap<Value, Value> allocToTBuf;

  /// Maps each on-chip memref.alloc result → its live local_tensor Value.
  ///
  /// Populated by convertDataMove when a copy produces a local_tensor that
  /// must remain live across multiple loop iterations (e.g. a bias buffer
  /// loaded once in an outer loop and consumed via subviews in inner loops).
  /// convertCompute uses this to re-use the live tensor instead of deque-ing
  /// again (which would underflow the queue).
  llvm::DenseMap<Value, Value> allocToLiveTensor;

  /// Look up the queue for a memref value, walking through subviews/casts to
  /// find the defining alloc.
  Value getQueue(Value memref) const;

  /// Look up the TBuf for a memref value, walking through subviews/casts to
  /// find the defining alloc. Returns {} if not found.
  Value getTBuf(Value memref) const;

  /// Look up a pre-dequeued live local_tensor for a memref value, walking
  /// through subviews to find the defining alloc.  Returns {} if not found.
  Value getLiveTensor(Value memref) const;
};

/// Return the integer memory_space of a memref type, or -1 if unavailable.
/// GM (default, no attribute) returns 0.
int64_t getMemorySpace(mlir::Type type);

enum class TransposeLoweringKind {
  Unsupported,
  AscendCSimple2D,
  ScalarMemRefLoop,
};

struct TransposeLoweringSpec {
  unsigned rank = 0;
  llvm::SmallVector<int64_t, 8> permutation;
  bool hasOnChipOutput = false;
};

struct TransposeLoweringPlan {
  TransposeLoweringKind kind = TransposeLoweringKind::Unsupported;
};

FailureOr<TransposeLoweringSpec>
buildTransposeLoweringSpec(linalg::TransposeOp transpose);

FailureOr<TransposeLoweringSpec>
buildTransposeLoweringSpec(linalg::GenericOp generic);

TransposeLoweringPlan
planTransposeLowering(const TransposeLoweringSpec &spec);

/// Compute total element count of a memref as an index-typed Value.
/// Inserts arith.constant / memref.dim / arith.muli ops at the current
/// insertion point of the builder.
Value computeElementCount(OpBuilder &b, Location loc, Value memrefVal);

/// Compute total byte size of a memref as an index-typed Value.
/// Equivalent to computeElementCount * (elementTypeBitWidth / 8).
/// This is the value expected by ascendc.pipe.init_buffer.
Value computeByteCount(OpBuilder &b, Location loc, Value memrefVal);

/// Run the data-move conversion step (memref.copy → AscendC data-move ops).
/// Requires ctx.pipe and ctx.allocToQueue to be populated.
/// Returns failure() if any required queue mapping is missing.
LogicalResult convertDataMove(func::FuncOp funcOp, AscendCBufferContext &ctx);

/// Run the compute conversion step (linalg ops → AscendC compute ops).
/// Requires ctx.pipe and ctx.allocToQueue to be populated.
/// Returns failure() if any required queue mapping is missing.
LogicalResult convertCompute(func::FuncOp funcOp, AscendCBufferContext &ctx);

/// Materialize selected rank-2 reduction tiles as explicit scf loops before
/// data-move and compute conversion consume the linalg/memref surface.
LogicalResult materializeSelectedReductionTiles(func::FuncOp funcOp);

/// Materialize selected rank-2 all-parallel tiles as explicit scf loops before
/// data-move and compute conversion consume the linalg/memref surface.
LogicalResult materializeSelectedAllParallelTiles(func::FuncOp funcOp);

/// Materialize selected rank-2 GM transpose tiles as explicit scf loops before
/// compute conversion lowers each tile.
LogicalResult materializeSelectedTransposeTiles(func::FuncOp funcOp);

/// Run the existing LinalgToAscendC lowering implementation on one function.
/// This is shared by the legacy --linalg-to-ascendc pass and the Phase 5
/// --ascend-compute-lower wrapper.
LogicalResult lowerLinalgToAscendC(func::FuncOp funcOp);

} // namespace mlir::afir

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_LOWERING_LINALG_TO_ASCENDC_UTILS_H
