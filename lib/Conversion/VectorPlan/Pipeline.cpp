#include "Conversion/AscendCBufferPlacement/AscendCBufferPlacementPass.h"
#include "Conversion/AscendCParallelize/AscendCParallelizePass.h"
#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"
#include "Conversion/AscendCRCoreCombine/AscendCRCoreCombinePass.h"
#include "Conversion/CanonicalizeCannSignature/CanonicalizeCannSignaturePass.h"
#include "Conversion/LinalgToAscendC/LinalgToAscendCPass.h"
#include "Conversion/MarkStructuredOps/MarkStructuredOpsPass.h"
#include "Conversion/VectorPlan/VectorPlanPasses.h"
#include "Dialect/AFIR/Transforms/Passes.h"
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
        pm.addNestedPass<func::FuncOp>(mlir::createLinalgGeneralizeNamedOpsPass());
        // Fuse adjacent elementwise/broadcast linalg ops into downstream
        // consumers (e.g. a reduce) so tile-fuse always sees a single op per
        // kernel group.  Without this, multi-op groups that mix parallel and
        // reduction iterators trip the tile-fuse assertions / IR domination.
        pm.addNestedPass<func::FuncOp>(mlir::createLinalgElementwiseOpFusionPass());
        // Make every returned tensor bufferize to a fresh buffer distinct from
        // any input init. Without this, reduce kernels alias result→init and
        // the host-launch ABI breaks (see R3 in reduce-codegen-status notes).
        pm.addNestedPass<func::FuncOp>(createVectorPlanIsolateKernelOutputsPass());
        // Symbolize the kernel's dynamic dims (afir.dim_symbols on the func,
        // afir.symbolic_shapes / afir.iter_extents on the ops) so tile-fuse can
        // carry the symbolic axis extents through to the AscendC kernel.
        pm.addNestedPass<func::FuncOp>(mlir::createAFIRSymbolizeShapesPass());
        // P1b: TileFuse is now a ModuleOp pass (it may emit multiple
        // <name>__v<i> sibling funcs from one outlined group).
        pm.addPass(createVectorPlanTileFusePass());
        // Fold tensor.dim on statically-known dimensions (e.g. the size-1
        // broadcast axis) before bufferization so that subview size operands
        // become constants rather than dynamic memref.dim values.
        pm.addPass(createCanonicalizerPass());
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
        // Fold bufferize-inserted shadow allocs (dynamic-size GM alloc +
        // GM->GM copy sandwich around a linalg.generic) before InsertTileBuffers,
        // which doesn't know how to handle them.  This only matters when a
        // tail block produces dynamic-size DPS init operands.
        pm.addNestedPass<func::FuncOp>(createVectorPlanFoldShadowAllocPass());
        pm.addNestedPass<func::FuncOp>(createVectorPlanInsertTileBuffersPass());
        pm.addNestedPass<func::FuncOp>(createAscendCBufferPlacementPass());
        pm.addNestedPass<func::FuncOp>(createLinalgToAscendCPass());
        // Lower multi-axis broadcasts (e.g. [1,D,1]→[D0,D,D2]) into a chain
        // of single-axis broadcasts before code emission, since AscendC's
        // Broadcast intrinsic supports only one broadcast axis per call.
        pm.addNestedPass<func::FuncOp>(
            createAscendCDecomposeMultiAxisBroadcastPass());
        pm.addNestedPass<func::FuncOp>(createAscendCParallelizePass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addNestedPass<func::FuncOp>(createAscendCFlattenGMPtrPass());
        pm.addNestedPass<func::FuncOp>(createAscendCPackTilingDataPass());
        pm.addNestedPass<func::FuncOp>(createAscendCFinalizeKernelPass());
        pm.addPass(createCSEPass());
        // Canonicalize DCEs dead memref view ops (collapse_shape, expand_shape)
        // left over after finalize-kernel removes the function return.
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCanonicalizeCannSignaturePass());
        // RCore (full-reduce multi-core): redirect each core's partial write
        // to workspace[block_idx], insert SyncAll + block-0 combine.  No-op
        // for non-RCore funcs.  Runs last because it needs the canonicalized
        // signature (workspace `memref<ui8>` arg in place).
        pm.addPass(createAscendCRCoreCombinePass());
      });
}

} // namespace mlir::afir