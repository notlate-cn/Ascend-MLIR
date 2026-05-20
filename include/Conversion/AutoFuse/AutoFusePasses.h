#ifndef ASCEND_MLIR_CONVERSION_AUTOFUSE_PASSES_H
#define ASCEND_MLIR_CONVERSION_AUTOFUSE_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::afir {

std::unique_ptr<Pass> createAutoFuseGroupAnalysisPass();
std::unique_ptr<Pass> createAutoFuseGroupOutlinePass();
std::unique_ptr<Pass> createAutoFuseTileFusePass();
std::unique_ptr<Pass> createAutoFuseBroadcastAbsorbPass();
std::unique_ptr<Pass> createAutoFuseInsertTileBuffersPass();
std::unique_ptr<Pass> createAutoFuseFoldShadowAllocPass();
std::unique_ptr<Pass> createAutoFuseIsolateKernelOutputsPass();
std::unique_ptr<Pass> createAutoFuseSplitRCoreGroupPass();
std::unique_ptr<Pass> createAutoFuseRestoreMatmulPass();

// Split every full-reduce private kernel func in `module` into a partial +
// combine pair (RCore template), rewriting coordinator call sites
// accordingly.  Same logic as the standalone pass; exposed so GroupOutline
// can invoke it between the structural transform and file emit.
// Returns the number of kernels that were actually split (≥ 0).
unsigned splitRCoreGroupsInPlace(mlir::ModuleOp module, int64_t parallelSlots);
void populateBroadcastAbsorbPatterns(mlir::RewritePatternSet &patterns);

void registerAutoFusePipeline();

}  // namespace mlir::afir

#endif
