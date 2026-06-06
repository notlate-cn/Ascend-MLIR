add_mlir_library(CannKernelTranslation
  CannTranslation.cpp
  CannRuntimeArtifacts.cpp
  AscendOpEmitters.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include

  LINK_LIBS PUBLIC
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIREmitAsc
  MLIRAsc
  MLIRTargetAsc
  MLIRIR
  LLVMSupport
)
