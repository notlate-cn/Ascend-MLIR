//===- Passes.td - Ascend conversion passes --------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_CONVERSION_PASSES
#define ASCEND_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Ascend MLIR Pipeline
//===----------------------------------------------------------------------===//

def AscendPrintTargetProfilePass
    : Pass<"ascend-print-target-profile", "mlir::ModuleOp"> {
  let summary = "Load and print the Ascend CANN target profile";
  let description = [{
    Loads a CANN platform_config target profile and prints a stable MVP summary
    for target model smoke tests and early Ascend pipeline development.
  }];
  let constructor = "mlir::afir::createAscendPrintTargetProfilePass()";
  let options = [
    Option<"soc", "soc", "std::string", /*default=*/"\"Ascend910B2\"",
           "CANN SoC version">,
    Option<"cannRoot", "cann-root", "std::string", /*default=*/"\"\"",
           "CANN root path">,
  ];
}

def AscendNormalizePass : Pass<"ascend-normalize", "mlir::ModuleOp"> {
  let summary = "Normalize supported entry IR for the Ascend MLIR pipeline";
  let description = [{
    Skeleton pass for the Ascend Normalize stage. It preserves IR unchanged in this
    MVP task and only emits an optional debug stage header.
  }];
  let constructor = "mlir::afir::createAscendNormalizePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::cf::ControlFlowDialect"
  ];
  let options = [
    Option<"debugStage", "debug-stage", "std::string", /*default=*/"\"none\"",
           "Ascend debug stage: none, normalize, kernelize, schedule, all">,
    Option<"dumpReport", "dump-report", "bool", /*default=*/"false",
           "Dump Ascend analysis report to stderr">,
  ];
}

def AscendKernelizePass : Pass<"ascend-kernelize", "mlir::ModuleOp"> {
  let summary = "Build Ascend MLIR kernel candidates";
  let description = [{
    Analyzes linalg ops in a normalized module, fuses them into kernel candidates
    via dependency, horizontal fusion, and merge analysis, then marks kernel
    boundaries and assigns op roles (cube, vector, reduction, memory) for
    downstream scheduling. Fails closed on any unsupported op semantics.
  }];
  let constructor = "mlir::afir::createAscendKernelizePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect"
  ];
  let options = [
    Option<"debugStage", "debug-stage", "std::string", /*default=*/"\"none\"",
           "Ascend debug stage: none, normalize, kernelize, schedule, all">,
    Option<"dumpReport", "dump-report", "bool", /*default=*/"false",
           "Dump Ascend analysis report to stderr">,
  ];
}

def AscendSchedulePass : Pass<"ascend-schedule", "mlir::ModuleOp"> {
  let summary = "Construct Ascend MLIR schedule decisions";
  let description = [{
    Assigns tile schedules to kernelize-marked kernel candidates using a
    template registry, axis coalescing, and a tuning database. Supports
    persistent cache IO and target-aware tiling policy derivation.
  }];
  let constructor = "mlir::afir::createAscendSchedulePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect"
  ];
  let options = [
    Option<"debugStage", "debug-stage", "std::string", /*default=*/"\"none\"",
           "Ascend debug stage: none, normalize, kernelize, schedule, all">,
    Option<"dumpReport", "dump-report", "bool", /*default=*/"false",
           "Dump Ascend analysis report to stderr">,
    Option<"runtimeTopK", "runtime-top-k", "unsigned", /*default=*/"1",
           "Maximum number of schedule decisions kept selectable at runtime">,
    Option<"targetTilePolicy", "target-tile-policy", "std::string",
           /*default=*/"\"require-explicit\"",
           "Schedule tile policy mode: require-explicit, legacy-default, or target-aware">,
    Option<"soc", "soc", "std::string", /*default=*/"\"Ascend910B2\"",
           "CANN SoC version for target-aware schedule tiling">,
    Option<"cannRoot", "cann-root", "std::string", /*default=*/"\"\"",
           "CANN root path for target-aware schedule tiling">,
    Option<"tuningCacheIn", "tuning-cache-in", "std::string",
           /*default=*/"\"\"",
           "Line-based persistent tuning signature cache input path">,
    Option<"tuningCacheOut", "tuning-cache-out", "std::string",
           /*default=*/"\"\"",
           "Line-based persistent tuning signature cache output path">,
    Option<"tuningDbIn", "tuning-db-in", "std::string",
           /*default=*/"\"\"",
           "Versioned persistent schedule tuning database input path">,
    Option<"tuningDbOut", "tuning-db-out", "std::string",
           /*default=*/"\"\"",
           "Versioned persistent schedule tuning database output path">,
    Option<"maxSearchBudget", "max-search-budget", "unsigned", /*default=*/"64",
           "Maximum number of axis-product tile shape candidates per kernel">,
  ];
}

def AscendKernelSplitPass : Pass<"ascend-kernel-split", "mlir::ModuleOp"> {
  let summary = "Outline logical Ascend kernels into separate functions";
  let description = [{
    Splits a scheduled single-function module that contains multiple
    ascend.kernel logical groups into one func.func per kernel. Cross-kernel SSA
    values become function arguments/results, while pure local helper ops are
    cloned into the consuming kernel function. Existing single-kernel functions
    are preserved unchanged.
  }];
  let constructor = "mlir::afir::createAscendKernelSplitPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def AscendRealizePass : Pass<"ascend-realize", "mlir::ModuleOp"> {
  let summary = "Construct Ascend MLIR memory realization plans";
  let description = [{
    Pass for the Ascend Realize stage. The default mode preserves IR unchanged,
    validates that Schedule/StructuredLowering attributes are present, and emits
    an optional debug report with GM-only plan objects. The opt-in
    one-shot-bufferize materialization mode runs upstream One-Shot Bufferize as
    the first Phase 3B IR mutation slice. The opt-in memory-space-annotate mode
    runs One-Shot Bufferize first and annotates proven local vector temporaries
    with on-chip memory spaces.
  }];
  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect"
  ];
  let constructor = "mlir::afir::createAscendRealizePass()";
  let options = [
    Option<"debugStage", "debug-stage", "std::string", /*default=*/"\"none\"",
           "Ascend debug stage: none, normalize, kernelize, schedule, realize, all">,
    Option<"dumpReport", "dump-report", "bool", /*default=*/"false",
           "Dump Ascend analysis report to stderr">,
    Option<"placementMode", "placement-mode", "std::string",
           /*default=*/"\"gm-default\"",
           "Realize placement mode: gm-default or target-aware">,
    Option<"materializationMode", "materialization-mode", "std::string",
           /*default=*/"\"plan-only\"",
           "Realize materialization mode: plan-only, one-shot-bufferize, memory-space-annotate, tiled-linalg, or full-realize">,
    Option<"soc", "soc", "std::string", /*default=*/"\"Ascend910B2\"",
           "CANN SoC version for target-aware placement">,
    Option<"cannRoot", "cann-root", "std::string", /*default=*/"\"\"",
           "CANN root path for target-aware placement">,
  ];
}

def AscendComputeLowerPass : Pass<"ascend-compute-lower", "mlir::func::FuncOp"> {
  let summary = "Lower supported memory-realized linalg and copies to AscendC";
  let description = [{
    Phase 5 compute lowering wrapper. It reuses the existing LinalgToAscendC
    lowering implementation and adds fail-closed support-matrix diagnostics.
  }];
  let constructor = "mlir::afir::createAscendComputeLowerPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def AscendParallelizePass : Pass<"ascend-parallelize", "mlir::ModuleOp"> {
  let summary = "Run Phase 5 Ascend kernel dispatch lowering";
  let description = [{
    Versionless Phase 5 wrapper around the existing AscendC parallelization
    implementation.
  }];
  let constructor = "mlir::afir::createAscendParallelizePass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
}

def AscendPrepareForEmitPass
    : Pass<"ascend-prepare-for-emit", "mlir::ModuleOp"> {
  let summary = "Prepare Phase 5 AscendC kernel ABI data";
  let description = [{
    Versionless Phase 5 wrapper around the existing AscendC prepare-for-emit
    implementation. It also derives ascendc.kernel_kind after prepare-for-emit
    so downstream artifact generation can select vec, cube, or mix compile
    paths per kernel.
  }];
  let constructor = "mlir::afir::createAscendPrepareForEmitPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def AscendCanonicalizeCannSignaturePass
    : Pass<"ascend-canonicalize-cann-signature", "mlir::ModuleOp"> {
  let summary = "Canonicalize Phase 5 Ascend kernel signatures to CANN ABI";
  let description = [{
    Versionless Phase 5 wrapper around the existing CANN signature
    canonicalization implementation.
  }];
  let constructor = "mlir::afir::createAscendCanonicalizeCannSignaturePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Legacy-compatible Ascend entry points hosted by the five-layer Ascend tree.
//===----------------------------------------------------------------------===//

def AscendCFoldConcatAllocPass
    : Pass<"ascendc-fold-concat-alloc", "mlir::func::FuncOp"> {
  let summary = "Eliminate redundant GM-to-GM copies introduced by tensor.concat bufferization";
  let description = [{
    After one-shot-bufferize of decomposed tensor.concat, this pass detects
    intermediate result allocs that are copied into slices of the concat output
    and rewrites users to write directly into the output subviews.
  }];
  let constructor = "mlir::afir::createAscendCFoldConcatAllocPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def AscendCBufferPlacementPass
    : Pass<"ascendc-buffer-placement", "mlir::func::FuncOp"> {
  let summary = "Annotate memref allocs with TPosition memory spaces and materialize data movement";
  let description = [{
    Legacy AscendC buffer-placement entry point retained for direct tests and
    legacy scripts. It now lives under the Ascend Realize layer.
  }];
  let constructor = "mlir::afir::createAscendCBufferPlacementPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def MarkStructuredOpsPass : Pass<"mark-structured-ops", "mlir::func::FuncOp"> {
  let summary = "Annotate structured linalg.generic ops with semantic attributes";
  let description = [{
    Detects structured linalg.generic patterns such as gather and embedding and
    stamps semantic attributes consumed by downstream Ascend kernelization and
    lowering stages.
  }];
  let constructor = "mlir::afir::createMarkStructuredOpsPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def FuseGatherElementwisePass
    : Pass<"fuse-gather-elementwise", "mlir::func::FuncOp"> {
  let summary = "Fuse elementwise ops into adjacent gather generics at tensor level";
  let description = [{
    Runs after structured-op marking and before transform interpretation.
    Detects pre/post elementwise chains around gather generics and folds the
    supported chain into a single gather generic.
  }];
  let constructor = "mlir::afir::createFuseGatherElementwisePass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];
}

def AnnotateMixMatmulSemanticsPass
    : Pass<"annotate-mix-matmul-semantics", "mlir::func::FuncOp"> {
  let summary = "Annotate func.func with stable abi_matmul_* semantics for simple mix matmul patterns";
  let description = [{
    Scans bufferized mix matmul kernels while linalg.matmul and fused vector
    epilogue ops are still present, then stamps stable function-level
    abi_matmul_* attributes for downstream ABI extraction.
  }];
  let constructor = "mlir::afir::createAnnotateMixMatmulSemanticsPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def LinalgToAscendCPass
    : Pass<"linalg-to-ascendc", "mlir::func::FuncOp"> {
  let summary = "Lower linalg compute ops and memref.copy to AscendC dialect ops";
  let description = [{
    Legacy direct entry point for the Ascend backend lowering engine. It lowers
    supported linalg compute operations and explicit memref.copy data movement
    to AscendC dialect operations.
  }];
  let constructor = "mlir::afir::createLinalgToAscendCPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def AscendCParallelizePass
    : Pass<"ascendc-parallelize", "mlir::func::FuncOp"> {
  let summary = "Replace outermost parallel scf.for loops with ascendc.get_block_idx dispatch";
  let description = [{
    Converts annotated top-level scf.for tile loops into multi-core AiCore
    dispatch using ascendc.get_block_idx.
  }];
  let constructor = "mlir::afir::createAscendCParallelizePass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
  let options = [
    Option<"coreNum", "core-num", "int64_t", /*default=*/"20",
           "Number of AiCores available for parallel dispatch">,
  ];
}

def AscendCPrepareForEmitPass
    : Pass<"ascendc-prepare-for-emit", "mlir::ModuleOp"> {
  let summary = "Prepare AscendC func signature and attributes for ascir-translate emission";
  let description = [{
    Transforms AscendC functions to the kernel interface expected by
    ascir-translate, including global/aicore attributes, flattened GM memrefs,
    TilingData consolidation, and void kernel returns.
  }];
  let constructor = "mlir::afir::createAscendCPrepareForEmitPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def AnnotateAscendCKernelKindPass
    : Pass<"annotate-ascendc-kernel-kind", "mlir::func::FuncOp"> {
  let summary = "Annotate func.func with ascendc.kernel_kind derived from ascendc.unit";
  let description = [{
    Scans descendant operations for ascendc.unit and stamps the enclosing
    function with ascendc.kernel_kind = vec, cube, or mix.
  }];
  let constructor = "mlir::afir::createAnnotateAscendCKernelKindPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}

def CanonicalizeCannSignaturePass
    : Pass<"canonicalize-cann-signature", "mlir::ModuleOp"> {
  let summary = "Canonicalize aicore kernel signature to CANN calling convention";
  let description = [{
    Transforms AscendC aicore kernel signatures from the internal format to the
    CANN runtime ABI and stamps cann.num_inputs metadata.
  }];
  let constructor = "mlir::afir::createCanonicalizeCannSignaturePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // ASCEND_CONVERSION_PASSES
