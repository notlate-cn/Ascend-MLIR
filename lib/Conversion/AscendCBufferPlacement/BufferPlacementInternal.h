//===- BufferPlacementInternal.h - AscendCBufferPlacement internal --------===//
//
// Module-internal contract shared between AscendCBufferPlacementPass.cpp and
// BufferPlacementCopyInsert.cpp (the data-copy-insertion half split out of the
// former 1650-line file).  Holds the cross-file data types and the handful of
// functions that cross the split boundary.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir::afir::buffer_placement {

/// Copy task: a single data transfer between two TPositions.
struct CopyTask {
  StringRef role; // "lhs", "rhs", "bias", "acc", "result"
  uint8_t srcPos; // source TPosition enum value
  uint8_t dstPos; // destination TPosition enum value
};

/// prologue/epilogue copy tasks for a loop.
struct LoopAnnotation {
  SmallVector<CopyTask> prologue;
  SmallVector<CopyTask> epilogue;
  bool isParallel = false;
};

/// Map from a memref alloc result Value to its TPosition value.
using BufferPosMap = DenseMap<Value, uint8_t>;

// Defined in AscendCBufferPlacementPass.cpp.
Attribute createMemorySpace(MLIRContext *ctx, uint8_t tposValue);
void updateAllocMemorySpace(BufferPosMap &posMap, RewriterBase &rewriter);

// Defined in BufferPlacementCopyInsert.cpp.
Value traceToAlloc(Value value);
void insertCopiesForLoops(func::FuncOp funcOp,
                          const DenseMap<Operation *, LoopAnnotation> &loopAnns,
                          BufferPosMap &posMap, OpBuilder &builder);

} // namespace mlir::afir::buffer_placement
