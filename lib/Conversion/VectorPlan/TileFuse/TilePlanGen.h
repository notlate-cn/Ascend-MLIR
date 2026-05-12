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
                  bool enableReductionSplit);

/// Write a `vector_plan.tiling_infos` entry for `func` to the parent ModuleOp.
/// Must be called after genVectorTilePlan so all tiling args already exist on func.
/// Only processes tileable params (those backed by a func BlockArgument).
/// Full/fixed params (Reduction Full, whole-loaded parallel/broadcast axes) are
/// silently skipped.
void emitTilingInfos(mlir::func::FuncOp func,
                     const mlir::vector_plan::TilePlan &plan);

} // namespace mlir::afir
