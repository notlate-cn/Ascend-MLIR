#pragma once
#include "Conversion/VectorPlan/GroupInfo.h"
#include "Conversion/VectorPlan/TilePlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::afir {

mlir::vector_plan::TilePlan
genVectorTilePlan(mlir::func::FuncOp func,
                  const mlir::vector_plan::CollapsedGroupInfo &info,
                  mlir::OpBuilder &builder,
                  mlir::Location loc,
                  bool enableReductionSplit,
                  int64_t maxFullLoopIters);

} // namespace mlir::afir
