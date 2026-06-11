#pragma once
#include "Conversion/AutoFuse/TilePlan.h"
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
                          const mlir::auto_fuse::TilePlan &plan,
                          mlir::Value tensor,
                          mlir::OpBuilder &builder,
                          mlir::Location loc,
                          const llvm::DenseMap<int, mlir::Value> *sizeOverride = nullptr);

/// Like computeSlice but uses outer IVs and outer (coarse) tile sizes.
/// Used for hoisting BCast-independent extracts before BCast loops.
SliceParams computeOuterSlice(mlir::AffineMap indexingMap,
                               const llvm::DenseMap<int, mlir::Value> &outerLoopIVs,
                               const mlir::auto_fuse::TilePlan &plan,
                               mlir::Value tensor,
                               mlir::OpBuilder &builder,
                               mlir::Location loc);

} // namespace mlir::afir
