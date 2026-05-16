//===- Passes.td - Conversion passes -----------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_PASSES
#define AFIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Ascend MLIR Pipeline Skeleton
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
    Skeleton pass for the Ascend Kernelize stage. It preserves IR unchanged in this
    MVP task and only emits an optional debug stage header.
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
    Skeleton pass for the Ascend Schedule stage. It preserves IR unchanged in this
    MVP task and only emits an optional debug stage header.
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
    implementation.
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
// StableHLO to AFIR Conversion
//===----------------------------------------------------------------------===//

//def ConvertStableHLOToAFIR : Pass<"convert-stablehlo-to-afir", "mlir::ModuleOp"> {
//  let summary = "Convert StableHLO dialect to AFIR dialect";
//  let description = [{
//    This pass converts supported StableHLO operations to AFIR operations.
//    Currently supports: add, subtract, multiply, divide.
//  }];
//  let dependentDialects = [
//    "mlir::afir::AFIRDialect"
//  ];
//}

//===----------------------------------------------------------------------===//
// AFIR to ASC-IR (PyAsc) Conversion
//===----------------------------------------------------------------------===//

def ConvertAFIRToASCIRPass : Pass<"convert-afir-to-ascir", "mlir::ModuleOp"> {
  let summary = "Convert AFIR dialect to ASC-IR dialect";
  let description = [{
    This pass converts AFIR operations to ASC-IR operations from PyAsc.
    This is the final lowering step for Ascend hardware.
    It outputs the converted graph to stderr or to a file if --ascir-path is specified.
  }];
  let constructor = "mlir::afir::createConvertAFIRToASCIRPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect"
  ];
  let options = [
    Option<"ascirPath", "ascir-path", "std::string", /*default=*/"\"\"",
           "Path to output ASCIR text file">,
  ];
}

//===----------------------------------------------------------------------===//
// AFIR to ASCIR Text Conversion
//===----------------------------------------------------------------------===//

def ConvertAFIRToASCIRTextPass : Pass<"convert-afir-to-ascir-text", "mlir::ModuleOp"> {
  let summary = "Convert AFIR dialect to ASCIR text format";
  let description = [{
    This pass converts AFIR operations to ASCIR protobuf text format.
    It outputs the converted graph to stderr or to a file if --ascir-path is specified.
  }];
  let constructor = "mlir::afir::createConvertAFIRToASCIRTextPass()";
  let dependentDialects = [
    "mlir::afir::AFIRDialect"
  ];
  let options = [
    Option<"ascirPath", "ascir-path", "std::string", /*default=*/"\"\"",
           "Path to output ASCIR text file">,
  ];
}

//===----------------------------------------------------------------------===//
// AscendC Fold Concat Alloc
//===----------------------------------------------------------------------===//

def AscendCFoldConcatAllocPass : Pass<"ascendc-fold-concat-alloc", "mlir::func::FuncOp"> {
  let summary = "Eliminate redundant GM→GM memref.copy ops introduced by tensor.concat bufferization";
  let description = [{
    After one-shot-bufferize of decomposed tensor.concat, the IR contains:

      %alloc_c   = memref.alloc(%M, %N)        // Op1 result buffer
      %alloc_d   = memref.alloc(%M, %N)        // Op2 result buffer
      ... loops writing into %alloc_c / %alloc_d ...
      %alloc_out = memref.alloc(%2M, %N)       // concat output
      memref.copy %alloc_c → subview %alloc_out[0,0]
      memref.copy %alloc_d → subview %alloc_out[M,0]
      return %alloc_out

    This pass detects the pattern and replaces each intermediate alloc with
    the corresponding subview of the output alloc, eliminating the copies:

      %alloc_out = memref.alloc(%2M, %N)
      %sub0 = memref.subview %alloc_out[0,0][M,N]
      %sub1 = memref.subview %alloc_out[M,0][M,N]
      ... loops writing directly into %sub0 / %sub1 ...
      return %alloc_out

    The epilogue data_copy (VECOUT→GM) in each loop now writes directly to the
    correct slice of %alloc_out, so no GM→GM memmove is needed.
  }];
  let constructor = "mlir::afir::createAscendCFoldConcatAllocPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}


//===----------------------------------------------------------------------===//
// AscendC Buffer Placement
//===----------------------------------------------------------------------===//

def AscendCBufferPlacementPass : Pass<"ascendc-buffer-placement", "mlir::func::FuncOp"> {
  let summary = "Annotate memref allocs with TPosition memory_space and insert explicit DataCopy/Fixpipe ops";
  let description = [{
    This pass implements Phase 1 of AscendC buffer placement:
    - Annotates memref allocs with TPosition memory_space attributes
    - Inserts explicit DataCopy/Fixpipe operations based on ascendc annotations
    - Removes transient ascendc.* annotations but preserves ascendc.kernel_kind

    TPosition values:
    - GM (0): Global memory
    - A1 (1), B1 (3): L1 buffers
    - A2 (2), B2 (4): L0 buffers
    - CO1 (7): Cube output buffer
    - VECIN (9): Vector input buffer
    - VECOUT (10): Vector output buffer
    - VECCALC (11): Vector calculation buffer

    The pass processes ascendc.prologue/epilogue annotations on scf.for loops
    and ascendc.unit annotations to determine appropriate buffer placement.
  }];
  let constructor = "mlir::afir::createAscendCBufferPlacementPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}


//===----------------------------------------------------------------------===//
// Linalg + memref.copy to AscendC Conversion (unified pass)
//===----------------------------------------------------------------------===//

def LinalgToAscendCPass : Pass<"linalg-to-ascendc", "mlir::func::FuncOp"> {
  let summary = "Lower linalg compute ops and memref.copy to AscendC dialect ops";
  let description = [{
    Unified pass that lowers both linalg compute ops and memref.copy data-move
    ops to AscendC dialect ops. A single TPipe and one TQueue per on-chip buffer
    are created once per function; all ops share the same queue objects so data
    flow is correctly connected.

    Data-move lowering (memref.copy):
      GM(0) → A1(1)/B1(3)  : alloc_tensor → data_copy_nd2nz → enque_tensor
      GM(0) → VECIN(9)      : alloc_tensor → data_copy_l2    → enque_tensor
      A1(1) → A2(2)         : deque → alloc → load_data_l0              → enque → free
      B1(3) → B2(4)         : deque → alloc → load_data_with_transpose  → enque → free
      CO1(7) → VECIN(9)     : deque → alloc → fixpipe                   → enque → free
      VECOUT(10) → GM(0)    : deque → data_copy_l2/scalar tail writeback → free

    Compute lowering (linalg ops):
      linalg.matmul (A2/B2/CO1)       : deque A2+B2, alloc CO1 → mmad → enque CO1
      linalg.fill (on-chip dst)        : alloc → duplicate_l2 → enque
      linalg.elementwise add           : deque srcs, alloc dst → add_l2 → enque dst
      linalg.elementwise max_signed    : deque srcs, alloc dst → max_l2 → enque dst
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


//===----------------------------------------------------------------------===//
// AscendC Parallelize
//===----------------------------------------------------------------------===//

def AscendCParallelizePass : Pass<"ascendc-parallelize", "mlir::func::FuncOp"> {
  let summary = "Replace outermost parallel scf.for loops with ascendc.get_block_idx dispatch";
  let description = [{
    Converts annotated top-level scf.for tile loops into multi-core AiCore
    dispatch using get_block_idx.

    The pass emits:
      %block_idx = ascendc.get_block_idx
      %i         = arith.muli %block_idx, %step
    and wraps the body in an scf.if guard for the final partial tile.
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

//===----------------------------------------------------------------------===//
// AscendC Prepare For Emit
//===----------------------------------------------------------------------===//

def AscendCPrepareForEmitPass : Pass<"ascendc-prepare-for-emit", "mlir::func::FuncOp"> {
  let summary = "Prepare AscendC func signature and attributes for ascir-translate emission";
  let description = [{
    Transforms the function to match the AscendC C++ kernel interface expected
    by ascir-translate:
      1. Adds {ascendc.aicore, ascendc.global} attributes to the function.
      2. Replaces strided memref args with flat memref<?, 22:i32> (GM memory_space=22).
      3. Consolidates i64 tiling parameters into a TilingData struct arg read
         from a GM pointer using emitasc.copy_struct.
      4. Removes the function return value (kernels return void).
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

//===----------------------------------------------------------------------===//
// CanonicalizeCannSignature
//===----------------------------------------------------------------------===//

def CanonicalizeCannSignaturePass : Pass<"canonicalize-cann-signature", "mlir::ModuleOp"> {
  let summary = "Canonicalize aicore kernel signature to CANN calling convention";
  let description = [{
    Transforms func.func ops with {ascendc.aicore, ascendc.global} attributes from
    PyAsc internal format (inputs..., tiling_memref, outputs...) to CANN standard:
    (inputs..., outputs..., workspace:memref<ui8>, tiling:PyStructType).

    Also:
    - Removes the emitasc.copy_struct op (tiling is now by-value).
    - Adds cann.num_inputs = N : i32 attribute to the function.
  }];
  let constructor = "mlir::afir::createCanonicalizeCannSignaturePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// FuseGatherElementwise
//===----------------------------------------------------------------------===//

def FuseGatherElementwisePass : Pass<"fuse-gather-elementwise", "mlir::func::FuncOp"> {
  let summary = "Fuse elementwise ops into adjacent gather generics at tensor level";
  let description = [{
    Runs at tensor level (after --mark-structured-ops, before --transform-interpreter).
    Detects chains of: pre-op (e.g. relu) -> gather (gather_dim/embedding_dim attr) -> post-op (e.g. add)
    and fuses them into a single gather generic that inlines all ops in its body.

    The fused op:
      ins  = gather's ins + post-op's extra ins (excluding gather result)
      outs = post-op's outs (or gather's outs if no post-op)
      body = pre-op arith ops + tensor.extract + post-op arith ops
  }];
  let constructor = "mlir::afir::createFuseGatherElementwisePass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// MarkStructuredOps
//===----------------------------------------------------------------------===//

def MarkStructuredOpsPass : Pass<"mark-structured-ops", "mlir::func::FuncOp"> {
  let summary = "Annotate structured linalg.generic ops with semantic attributes";
  let description = [{
    Detects structured patterns in linalg.generic ops and stamps semantic
    attributes to guide downstream passes (Transform tiling, ComputeConversion).

    Currently detected patterns:
    - index_select (column gather): body has tensor.extract where the extracted
      tensor is in ins, indexed by another ins block arg at one dimension, and
      remaining indices come from linalg.index. Stamps {gather_dim = N : i64}.
    - embedding (row gather): same five conditions, but indices indexing map
      selects dimension 0. Stamps {embedding_dim = 0 : i64}.

    Detection uses five structural conditions (see pass implementation).
    Does not modify IR semantics — attribute-only annotation.
  }];
  let constructor = "mlir::afir::createMarkStructuredOpsPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def AnnotateAscendCKernelKindPass : Pass<"annotate-ascendc-kernel-kind", "mlir::func::FuncOp"> {
  let summary = "Annotate func.func with ascendc.kernel_kind derived from ascendc.unit";
  let description = [{
    Scans descendant linalg ops for ascendc.unit and stamps the enclosing
    function with ascendc.kernel_kind = vec|cube|mix.

    - AiCore.Vector only -> vec
    - AiCore.Cube only   -> cube
    - both kinds present  -> mix

    The attribute is function-level metadata and is preserved by later passes
    that do not explicitly clear it.
  }];
  let constructor = "mlir::afir::createAnnotateAscendCKernelKindPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}

def AnnotateMixMatmulSemanticsPass : Pass<"annotate-mix-matmul-semantics", "mlir::func::FuncOp"> {
  let summary = "Annotate func.func with stable abi_matmul_* semantics for simple mix matmul patterns";
  let description = [{
    Scans bufferized mix matmul kernels while linalg.matmul and fused vector
    epilogue ops are still present, then stamps stable function-level
    abi_matmul_* attributes for downstream ABI extraction.

    First-pass coverage:
    - 2D linalg.matmul
    - optional bias add epilogue
    - optional leaky_relu epilogue

    Emitted attributes:
    - abi_matmul_op_kind = "matmul"
    - abi_matmul_trans_a = false
    - abi_matmul_trans_b = false
    - abi_matmul_has_bias = true|false
    - abi_matmul_layout_a = "ND"
    - abi_matmul_layout_b = "ND"
    - abi_matmul_layout_c = "ND"
    - abi_matmul_epilogue_kind = "None" | "BiasAdd" | "BiasAddLeakyRelu"

    The pass is conservative: unsupported patterns are left unannotated.
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

#endif // AFIR_CONVERSION_PASSES
