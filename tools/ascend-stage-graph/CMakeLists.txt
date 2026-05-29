set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(ascend-stage-graph
  ascend-stage-graph.cpp
)

target_link_libraries(ascend-stage-graph
  PRIVATE
    MLIRParser
    MLIRSupport
    MLIRIR

    MLIRAffineDialect
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRVectorDialect

    MLIRAsc
    MLIRAscUtils
    MLIREmitAsc
)

mlir_check_all_link_libraries(ascend-stage-graph)
