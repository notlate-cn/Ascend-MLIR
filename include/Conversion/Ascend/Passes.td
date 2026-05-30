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
    Loads a CANN platform_config target profile and prints a stable summary
    for target model smoke tests and Ascend pipeline development.
  }];
  let constructor = "mlir::ascend::createAscendPrintTargetProfilePass()";
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
    Validates supported entry dialects for the Ascend pipeline, preserves the
    normalized IR shape, stamps the module with the normalized marker, and emits
    optional debug-stage reporting.
  }];
  let constructor = "mlir::ascend::createAscendNormalizePass()";
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
    downstream scheduling. It also owns structured gather marking and gather
    elementwise fusion before candidate analysis. Fails closed on any
    unsupported op semantics.
  }];
  let constructor = "mlir::ascend::createAscendKernelizePass()";
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
    Option<"debugDumpDir", "debug-dump-dir", "std::string", /*default=*/"\"\"",
           "Directory for Ascend debug checkpoint MLIR dumps">,
  ];
}

def AscendSchedulePass : Pass<"ascend-schedule", "mlir::ModuleOp"> {
  let summary = "Construct Ascend MLIR schedule decisions";
  let description = [{
    Assigns tile schedules to kernelize-marked kernel candidates using a
    template registry, axis coalescing, and a tuning database. Supports
    persistent cache IO and target-aware tiling policy derivation.
  }];
  let constructor = "mlir::ascend::createAscendSchedulePass()";
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
    Option<"debugDumpDir", "debug-dump-dir", "std::string", /*default=*/"\"\"",
           "Directory for Ascend debug checkpoint MLIR dumps">,
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
  let constructor = "mlir::ascend::createAscendKernelSplitPass()";
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
    one-shot-bufferize materialization mode runs upstream One-Shot Bufferize.
    The opt-in memory-space-annotate mode runs One-Shot Bufferize first and
    annotates proven local vector temporaries with on-chip memory spaces.
  }];
  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect"
  ];
  let constructor = "mlir::ascend::createAscendRealizePass()";
  let options = [
    Option<"debugStage", "debug-stage", "std::string", /*default=*/"\"none\"",
           "Ascend debug stage: none, normalize, kernelize, schedule, realize, all">,
    Option<"dumpReport", "dump-report", "bool", /*default=*/"false",
           "Dump Ascend analysis report to stderr">,
    Option<"debugDumpDir", "debug-dump-dir", "std::string", /*default=*/"\"\"",
           "Directory for Ascend debug checkpoint MLIR dumps">,
    Option<"placementMode", "placement-mode", "std::string",
           /*default=*/"\"gm-default\"",
           "Realize placement mode: gm-default or target-aware">,
    Option<"materializationMode", "materialization-mode", "std::string",
           /*default=*/"\"plan-only\"",
           "Realize materialization mode: plan-only, one-shot-bufferize, or memory-space-annotate">,
    Option<"soc", "soc", "std::string", /*default=*/"\"Ascend910B2\"",
           "CANN SoC version for target-aware placement">,
    Option<"cannRoot", "cann-root", "std::string", /*default=*/"\"\"",
           "CANN root path for target-aware placement">,
  ];
}

def AscendComputeLowerPass : Pass<"ascend-compute-lower", "mlir::func::FuncOp"> {
  let summary = "Lower supported memory-realized linalg and copies to AscendC";
  let description = [{
    Translate compute lowering. It annotates kernel kind and mix matmul ABI
    semantics when needed, lowers supported linalg compute operations and
    memref.copy movement to AscendC, and fails closed on support-matrix
    diagnostics.
  }];
  let constructor = "mlir::ascend::createAscendComputeLowerPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def AscendParallelizePass : Pass<"ascend-parallelize", "mlir::ModuleOp"> {
  let summary = "Lower Ascend kernel dispatch to explicit parallel loops";
  let description = [{
    Lowers Ascend kernel dispatch structure to explicit parallel loops for the
    internal AscendC kernel construction surface.
  }];
  let constructor = "mlir::ascend::createAscendParallelizePass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
}

def AscendPrepareForEmitPass
    : Pass<"ascend-prepare-for-emit", "mlir::ModuleOp"> {
  let summary = "Prepare AscendC kernel ABI data for emission";
  let description = [{
    Prepares AscendC functions for emission. It also derives
    ascendc.kernel_kind after prepare-for-emit so downstream artifact
    generation can select vec, cube, or mix compile paths per kernel.
  }];
  let constructor = "mlir::ascend::createAscendPrepareForEmitPass()";
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
  let summary = "Canonicalize Ascend kernel signatures to CANN ABI";
  let description = [{
    Canonicalizes Ascend kernel signatures to the CANN runtime ABI.
  }];
  let constructor = "mlir::ascend::createAscendCanonicalizeCannSignaturePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // ASCEND_CONVERSION_PASSES
