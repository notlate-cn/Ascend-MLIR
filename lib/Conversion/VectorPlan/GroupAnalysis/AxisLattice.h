#pragma once

#include "FusionGroup.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::afir {

/// Compute canonical axes for a set of linalg ops.
/// Returns axes sorted by position index, skipping positions where all members
/// are absent.
llvm::SmallVector<mlir::vector_plan::AxisInfo>
computeCanonicalAxes(llvm::ArrayRef<mlir::linalg::LinalgOp> members);

/// Convenience wrapper for FusionGroup.
llvm::SmallVector<mlir::vector_plan::AxisInfo>
computeCanonicalAxes(const FusionGroup &g);

/// Returns all values used by the ops' operands that are defined outside
/// (block arguments of the enclosing func, or results of non-linalg ops not
/// in the members set, or results of linalg ops not in the members set).
llvm::DenseSet<mlir::Value>
collectBoundaryIn(llvm::ArrayRef<mlir::linalg::LinalgOp> members,
                  mlir::func::FuncOp func);

} // namespace mlir::afir
