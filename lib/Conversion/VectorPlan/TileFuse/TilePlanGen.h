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
                  llvm::StringRef socName = "Ascend910B1");

/// Write a `vector_plan.tiling_infos` entry for `func` to the parent ModuleOp.
/// Must be called after genVectorTilePlan so all tiling args already exist on func.
/// Only processes tileable params (those backed by a func BlockArgument).
/// Full/fixed params (Reduction Full, whole-loaded parallel/broadcast axes) are
/// silently skipped.
void emitTilingInfos(mlir::func::FuncOp func,
                     const mlir::vector_plan::TilePlan &plan);

/// P1b: enumerate every feasible TilePlanDraft for `info`. When
/// `relaxNonBlockUbY=true` (P6 default) the non-block ub-Y drafts survive
/// the feasibility filter with a soft penalty, so the autotuner can profile
/// them as separate variants alongside the block-axis ub-Y pick.
llvm::SmallVector<mlir::vector_plan::TilePlanDraft>
enumerateFeasibleDrafts(const mlir::vector_plan::CollapsedGroupInfo &info,
                        bool enableReductionSplit,
                        bool relaxNonBlockUbY,
                        llvm::StringRef socName = "Ascend910B1");

/// P1b: materialize a specific draft into a TilePlan, mutating `func` (adds
/// the tile-size BlockArguments). Caller owns the scheduling choice instead
/// of letting `genVectorTilePlan`'s feasibility-only pickBest decide.
mlir::vector_plan::TilePlan
buildPlanForDraft(mlir::func::FuncOp func,
                  const mlir::vector_plan::CollapsedGroupInfo &info,
                  const mlir::vector_plan::TilePlanDraft &draft,
                  mlir::OpBuilder &builder,
                  mlir::Location loc,
                  llvm::StringRef socName = "Ascend910B1");

} // namespace mlir::afir
