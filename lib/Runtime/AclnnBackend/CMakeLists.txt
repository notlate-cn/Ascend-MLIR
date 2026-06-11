add_mlir_library(AclnnBackend
  AclnnBackend.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Runtime

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRParser
  MLIRIR
)
