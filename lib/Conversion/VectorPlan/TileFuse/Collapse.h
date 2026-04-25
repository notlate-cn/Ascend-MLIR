#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::afir {

/// Analyse all linalg ops in `func`, compute BCast-pruned collapse groups, and
/// (for single-generic funcs) transform the IR — inserting tensor.collapse_shape
/// for C-type operands, rewriting indexing maps, and inserting tensor.expand_shape
/// to restore original return types.
///
/// Returns a CollapsedGroupInfo describing the post-collapse iteration space.
/// When no collapse is profitable (BCast pruning leaves no group ≥ 2, B2 detected,
/// or empty func), returns an identity mapping (collapsedAxes == canonicalAxes,
/// axisMap[i] == i).
///
/// v1 limitation: IR transform is applied only when exactly one linalg.generic
/// is present. For multi-op funcs the analysis (axisMap, collapsedAxes) is still
/// computed and returned; callers must not assume the IR was transformed in that
/// case (check members.size() == 1 or track this separately).
mlir::vector_plan::CollapsedGroupInfo
collapseGroup(mlir::OpBuilder &builder, mlir::func::FuncOp func);

} // namespace mlir::afir
