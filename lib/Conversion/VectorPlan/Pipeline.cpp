#include "Conversion/AscendCBufferPlacement/AscendCBufferPlacementPass.h"
#include "Conversion/AscendCParallelize/AscendCParallelizePass.h"
#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"
#include "Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h"
#include "Conversion/LinalgToAscendC/LinalgToAscendCPass.h"
#include "Conversion/MarkStructuredOps/MarkStructuredOpsPass.h"
#include "Conversion/VectorPlan/VectorPlanPasses.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir::afir {

namespace {

struct VectorPlanPipelineOptions
    : PassPipelineOptions<VectorPlanPipelineOptions> {
  Option<std::string> outputDir{
      *this, "output-dir",
      llvm::cl::desc("Write split kernel_groupN.mlir + network.mlir here"),
      llvm::cl::init("")};
};

void buildVectorPlanPipeline(OpPassManager &pm,
                              const VectorPlanPipelineOptions &opts) {
  // Step 1: Fuse adjacent elementwise linalg ops so that reshape/collapse ops
  // land at group boundaries rather than inside groups.
  pm.addNestedPass<func::FuncOp>(mlir::createLinalgElementwiseOpFusionPass());

  // Step 2: Classify linalg ops into fusion groups (assigns group_id /
  // topo_index attributes).
  pm.addNestedPass<func::FuncOp>(createVectorPlanGroupAnalysisPass());

  // Step 3: Outline each group into a private kernel func; optionally split
  // into per-kernel MLIR files.
  std::string outlineSpec = "vector-plan-group-outline";
  if (!opts.outputDir.getValue().empty())
    outlineSpec += "{output-dir=" + opts.outputDir.getValue() + "}";
  if (failed(parsePassPipeline(outlineSpec, pm)))
    llvm::report_fatal_error("VectorPlan pipeline: failed to add outline pass");
}

} // namespace

void registerVectorPlanPipeline() {
  PassPipelineRegistration<VectorPlanPipelineOptions>(
      "vector-plan",
      "VectorPlan pipeline: elewise-fusion → group-analysis → group-outline",
      buildVectorPlanPipeline);
  PassPipelineRegistration<>(
      "vector-plan-codegen",
      "VectorPlan codegen: tile-fuse → bufferize → linalg-to-ascendc → "
      "parallelize → prepare-for-emit → canonicalize-cann-signature",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createVectorPlanTileFusePass());
        if (failed(parsePassPipeline(
                "one-shot-bufferize{"
                "bufferize-function-boundaries=true "
                "allow-return-allocs-from-loops=true "
                "function-boundary-type-conversion=identity-layout-map}",
                pm)))
          llvm::report_fatal_error(
              "vector-plan-codegen: failed to add bufferize pass");
        pm.addNestedPass<func::FuncOp>(createAnnotateAscendCKernelKindPass());
        pm.addPass(createCSEPass());
        pm.addNestedPass<func::FuncOp>(createVectorPlanInsertTileBuffersPass());
        pm.addNestedPass<func::FuncOp>(createAscendCBufferPlacementPass());
        pm.addNestedPass<func::FuncOp>(createLinalgToAscendCPass());
        pm.addNestedPass<func::FuncOp>(createAscendCParallelizePass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addNestedPass<func::FuncOp>(createAscendCFlattenGMPtrPass());
        pm.addNestedPass<func::FuncOp>(createAscendCPackTilingDataPass());
        pm.addNestedPass<func::FuncOp>(createAscendCFinalizeKernelPass());
        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizeCannSignaturePass());
      });
}

} // namespace mlir::afir