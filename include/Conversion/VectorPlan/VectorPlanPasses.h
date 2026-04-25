#ifndef ASCEND_MLIR_CONVERSION_VECTORPLAN_PASSES_H
#define ASCEND_MLIR_CONVERSION_VECTORPLAN_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::afir {

std::unique_ptr<Pass> createVectorPlanGroupAnalysisPass();
std::unique_ptr<Pass> createVectorPlanGroupOutlinePass();
std::unique_ptr<Pass> createVectorPlanTileFusePass();
std::unique_ptr<Pass> createVectorPlanBroadcastAbsorbPass();

void registerVectorPlanPipeline();

}  // namespace mlir::afir

#endif
