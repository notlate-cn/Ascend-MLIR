#pragma once
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

struct SliceParams {
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;
};

SliceParams computeSlice(mlir::AffineMap indexingMap,
                          const llvm::DenseMap<int, mlir::Value> &loopIVs,
                          const mlir::vector_plan::TilePlan &plan,
                          mlir::Value tensor,
                          mlir::OpBuilder &builder,
                          mlir::Location loc);

} // namespace mlir::afir
