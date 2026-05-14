add_mlir_library(CannKernelTranslation
  CannTranslation.cpp
  AfirOpEmitters.cpp
  UbCostExpr.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIREmitAsc
  MLIRAsc
  MLIRTargetAsc
  MLIRIR
  AFIRSymbolicShape
  LLVMSupport
)
