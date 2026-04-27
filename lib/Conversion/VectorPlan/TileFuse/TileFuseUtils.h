#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir::afir {

mlir::Value castToIndex(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v);

mlir::Value getAxisExtentValue(mlir::OpBuilder &b, mlir::Location loc,
                                const mlir::vector_plan::CollapsedGroupInfo &info,
                                int axisIdx);

} // namespace mlir::afir
