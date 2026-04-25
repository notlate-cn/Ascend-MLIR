#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::afir {

/// Analyse all linalg ops in `func`, compute collapse groups, and transform
/// the IR (insert tensor.collapse_shape / expand_shape, rewrite indexing maps).
/// Returns a CollapsedGroupInfo for Phase 2/3 consumption.
/// v1: handles single-generic funcs only; multi-op → identity (no collapse).
mlir::vector_plan::CollapsedGroupInfo
collapseGroup(mlir::OpBuilder &builder, mlir::func::FuncOp func);

} // namespace mlir::afir
