set(LLVM_LINK_COMPONENTS
  Core
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(afir-opt
  afir-opt.cpp
)

target_link_libraries(afir-opt
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}

    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRAnalysis
    MLIRIR
    MLIRTransformUtils
    MLIRCastInterfaces

    AFIRDialect
    AFIRTransforms
    AFIRToASCIRConversion
    AFIRToASCIRTextConversion
    AscendCBufferPlacementConversion
    AscendCFoldConcatAllocConversion
    LinalgToAscendCConversion
    AscendCParallelizeConversion
    AscendCPrepareForEmitConversion
    CanonicalizeCannSignatureConversion
    MarkStructuredOpsConversion
    MLIREmitAsc
    AFIRUtils
)

mlir_check_all_link_libraries(afir-opt)