set(LLVM_LINK_COMPONENTS
  Core
  Support
)

add_llvm_executable(ascend-mlir-opt
  ascend-mlir-opt.cpp
)

target_link_libraries(ascend-mlir-opt
  PRIVATE
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRAnalysis
    MLIRIR
    MLIRTransformUtils
    MLIRCastInterfaces

    MLIRAffineDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect

    MLIRAsc
    MLIRAscUtils
    MLIREmitAsc
    AscendTargetProfile
    AscendConversion
)

mlir_check_all_link_libraries(ascend-mlir-opt)
