#ifndef ASCEND_MLIR_CONVERSION_VECTORPLAN_PASSES_H
#define ASCEND_MLIR_CONVERSION_VECTORPLAN_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::afir {

std::unique_ptr<Pass> createVectorPlanGroupAnalysisPass();
std::unique_ptr<Pass> createVectorPlanGroupOutlinePass();
std::unique_ptr<Pass> createVectorPlanTileFusePass();
std::unique_ptr<Pass> createVectorPlanBroadcastAbsorbPass();
std::unique_ptr<Pass> createVectorPlanInsertTileBuffersPass();
std::unique_ptr<Pass> createVectorPlanFoldShadowAllocPass();
std::unique_ptr<Pass> createVectorPlanIsolateKernelOutputsPass();
std::unique_ptr<Pass> createVectorPlanSplitRCoreGroupPass();
std::unique_ptr<Pass> createVectorPlanRestoreMatmulPass();

// Split every full-reduce private kernel func in `module` into a partial +
// combine pair (RCore template), rewriting coordinator call sites
// accordingly.  Same logic as the standalone pass; exposed so GroupOutline
// can invoke it between the structural transform and file emit.
// Returns the number of kernels that were actually split (≥ 0).
unsigned splitRCoreGroupsInPlace(mlir::ModuleOp module, int64_t parallelSlots);
void populateBroadcastAbsorbPatterns(mlir::RewritePatternSet &patterns);

void registerVectorPlanPipeline();

}  // namespace mlir::afir

#endif
