//===- Passes.td - Conversion passes -----------------------*- tablegen -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef AFIR_CONVERSION_PASSES
#define AFIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

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
      VECOUT(10) → GM(0)    : deque → data_copy_l2 → free

    Compute lowering (linalg ops):
      linalg.matmul (A2/B2/CO1)       : deque A2+B2, alloc CO1 → mmad → enque CO1
      linalg.fill (on-chip dst)        : alloc → duplicate_l2 → enque
      linalg.elementwise add           : deque srcs, alloc dst → add_l2 → enque dst
      linalg.elementwise max_signed    : deque srcs, alloc dst → max_l2 → enque dst
  }];
  let constructor = "mlir::afir::createLinalgToAscendCPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
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
    Converts the outermost two scf.for loops (TB_M × TB_N) that were annotated
    with ascendc.parallel into multi-core AiCore dispatch using get_block_idx.

    The 2D block index is linearized as: block_idx = i_TB_M * num_blocks_N + i_TB_N.
    The pass emits:
      %block_idx = ascendc.get_block_idx
      %i_TB_M    = arith.divui  %block_idx, %num_blocks_N
      %i_TB_N    = arith.remui  %block_idx, %num_blocks_N
    and wraps the body in an scf.if guard for out-of-bounds blocks.

    Options:
      - core-num: total number of AiCores (default 20)
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
// AscendC Prepare For Emit — refactored sub-passes
//===----------------------------------------------------------------------===//

def AscendCFlattenGMPtrPass : Pass<"ascendc-flatten-gm-ptr", "mlir::func::FuncOp"> {
  let summary = "Promote GM allocs to func args and flatten subview chains for set_global_buffer";
  let description = [{
    Runs before PackTilingData so memref.dim values are still available as
    plain IR ops (not yet packed into TilingData).

    Transformations:
      1. Promote every top-level memref.alloc with memory_space=0 (GM) to a
         new function block argument.
      2. Replace each ascendc.global_tensor.set_global_buffer whose buffer is a
         memref.subview chain with a flat-pointer + integer offset form using
         emitasc.reinterpret_cast + emitasc.ptr_offset.
      3. Replace GM->GM memref.copy ops with a verbatim memmove call.
  }];
  let constructor = "mlir::afir::createAscendCFlattenGMPtrPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def AscendCPackTilingDataPass : Pass<"ascendc-pack-tiling-data", "mlir::func::FuncOp"> {
  let summary = "Pack index/i64 tiling args and memref.dim uses into a TilingData GM struct";
  let description = [{
    Reads vector_plan.tiling_infos on the parent ModuleOp to discover named
    tiling parameters (e.g. XBLOCK, XBLOCK_SUB).  Fails if tiling_infos is
    absent — no Phase A positional fallback.

    Transformations:
      1. Collect tiling arg names/indices from vector_plan.tiling_infos.
      2. Scan for memref.dim %blockArg, %cI ops; each unique (arg, dim) pair
         becomes a dim_argN_D field in TilingData.
      3. Inject a new TilingData GM pointer block argument, emit
         emitasc.copy_struct + emitasc.member to extract every field.
      4. Replace all tiling arg uses and memref.dim uses with the extracted
         values; erase the original args.
      5. Emit emitasc.declare_py_struct at module scope.
  }];
  let constructor = "mlir::afir::createAscendCPackTilingDataPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::emitasc::EmitAscDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect"
  ];
}

def AscendCDecomposeMultiAxisBroadcastPass
    : Pass<"ascendc-decompose-multi-axis-broadcast", "mlir::func::FuncOp"> {
  let summary = "Decompose multi-axis ascendc.broadcast_l2 into a chain of single-axis ones";
  let description = [{
    AscendC::Broadcast template only supports a single broadcast axis per call.
    A linalg::generic with an indexing map that drops multiple iteration dims
    (e.g. (d0,d1,d2) -> (d1) projecting away d0 and d2) lowers to a single
    BroadcastL2Op whose srcShape contains a constant 1 in every dropped slot.

    This pass walks each BroadcastL2Op, identifies the broadcast axes (where
    srcShape[i] is statically 1 and dstShape[i] is not), and if there are 2+
    axes, replaces the op with a chain of single-axis BroadcastL2Ops separated
    by intermediate VECCALC tiles allocated via TBufOp + TPipeInitBufferOp.
    The last broadcast in the chain writes to the original op's dst so that
    downstream users observe the same final tensor.
  }];
  let constructor = "mlir::afir::createAscendCDecomposeMultiAxisBroadcastPass()";
  let dependentDialects = [
    "mlir::ascendc::AscendCDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}

def AscendCFinalizeKernelPass : Pass<"ascendc-finalize-kernel", "mlir::func::FuncOp"> {
  let summary = "Stamp kernel attributes, strip return values, lower affine.min";
  let description = [{
    Final cleanup before ascir-translate:
      1. Set {ascendc.aicore, ascendc.global} attributes.
      2. Replace func.return with operands by a void func.return.
      3. Lower affine.min -> arith.minsi (ascir-translate has no affine support).
  }];
  let constructor = "mlir::afir::createAscendCFinalizeKernelPass()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
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

//===----------------------------------------------------------------------===//
// VectorPlan Passes
//===----------------------------------------------------------------------===//

def VectorPlanGroupAnalysis : Pass<"vector-plan-group-analysis", "mlir::func::FuncOp"> {
  let summary = "Analyze linalg ops and annotate fusion groups";
  let constructor = "mlir::afir::createVectorPlanGroupAnalysisPass()";
  let options = [
    Option<"enableReductionSplit",     "enable-reduction-split",      "bool",    "false", "">,
    Option<"maxReduceEpilogueOps",     "max-reduce-epilogue-ops",     "int32_t", "3",     "">,
    Option<"maxHorizontalExtraInputs", "max-horizontal-extra-inputs", "int32_t", "4",     "">,
  ];
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::func::FuncDialect"];
}

def VectorPlanGroupOutline : Pass<"vector-plan-group-outline", "mlir::ModuleOp"> {
  let summary = "Outline annotated fusion groups into kernel funcs";
  let constructor = "mlir::afir::createVectorPlanGroupOutlinePass()";
  let options = [
    Option<"kernelFuncPrefix", "kernel-func-prefix", "std::string", "\"kernel_group\"", "">,
    Option<"outputDir",        "output-dir",         "std::string", "\"\"",             "">,
  ];
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::linalg::LinalgDialect"];
}

def VectorPlanTileFuse : Pass<"vector-plan-tile-fuse", "mlir::func::FuncOp"> {
  let summary = "Tile and fuse kernel group func";
  let constructor = "mlir::afir::createVectorPlanTileFusePass()";
  let options = [
    Option<"enableCollapse",       "enable-collapse",        "bool",    "true",  "">,
    Option<"enableReductionSplit", "enable-reduction-split", "bool",    "false", "">,
    Option<"maxFullLoopIters",     "max-full-loop-iters",    "int64_t", "2048",
           "BCast axes larger than this threshold escape to tileable Inner">,
    Option<"hwMatmulMAlign",       "hw-matmul-m-align",      "int64_t", "16",    "">,
    Option<"hwMatmulNAlign",       "hw-matmul-n-align",      "int64_t", "16",    "">,
    Option<"hwMatmulKAlign",       "hw-matmul-k-align",      "int64_t", "16",    "">,
    Option<"l0aCapacityBytes",     "l0a-capacity-bytes",     "int64_t", "65536", "">,
  ];
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::scf::SCFDialect", "mlir::tensor::TensorDialect", "mlir::func::FuncDialect"];
}

def VectorPlanInsertTileBuffers : Pass<"vector-plan-insert-tile-buffers", "mlir::func::FuncOp"> {
  let summary = "Promote inner-tile linalg.generic operands from GM to on-chip VECIN/VECOUT";
  let description = [{
    Runs after one-shot-bufferize and before ascendc-buffer-placement.

    For each linalg.generic whose output memref is in global memory (memory_space=0):
      - Allocates a VECIN (memory_space=9) buffer for each GM input and inserts
        memref.copy(gm_input -> vecin_alloc) before the op.
      - Allocates a VECOUT (memory_space=10) buffer for the output and inserts
        memref.copy(vecout_alloc -> gm_output) after the op.
      - Rewrites the generic to use the on-chip buffers.

    After promotion, remaining GM->GM memref.copy ops (bufferization artifacts
    from one-shot-bufferize's yield-copy lowering) are erased.
  }];
  let constructor = "mlir::afir::createVectorPlanInsertTileBuffersPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
  ];
}

//===----------------------------------------------------------------------===//
// AclnnFinalizeDecl
//===----------------------------------------------------------------------===//

def AclnnFinalizeDecl : Pass<"aclnn-finalize-decl", "mlir::ModuleOp"> {
  let summary = "Finalize aclnn func.func private declarations: replace aclnn.kind with aclnn.op + aclnn.layout";
  let description = [{
    Runs after vector-plan-group-outline. Scans func.func private declarations
    that carry {aclnn.kind = "..."} and replaces that marker with the final
    aclnn metadata attributes consumed by AclnnBackend:

      aclnn.kind = "flash_attention"
        ->  aclnn.op = "FlashAttentionScore", aclnn.layout = "BNSD"

    aclnn.kind is removed after finalization. Unknown kind values trigger
    signalPassFailure (fail-fast).

    Registered kinds (Phase 1):
      "flash_attention" -> FlashAttentionScore, BNSD
  }];
  let constructor = "mlir::afir::createAclnnFinalizeDeclPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// BroadcastAbsorb
//===----------------------------------------------------------------------===//

def VectorPlanBroadcastAbsorb : Pass<"vector-plan-broadcast-absorb", "mlir::func::FuncOp"> {
  let summary = "Absorb linalg.broadcast into downstream linalg.generic indexing maps";
  let description = [{
    For each linalg.broadcast whose result is consumed by exactly one
    linalg.generic, replaces the generic's operand with the broadcast's
    original input and drops the broadcast dimensions from the operand's
    indexing map. Broadcasts with other consumers are left unchanged.
  }];
  let constructor = "mlir::afir::createVectorPlanBroadcastAbsorbPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

#endif // AFIR_CONVERSION_PASSES
