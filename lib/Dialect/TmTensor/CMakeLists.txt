add_mlir_dialect_library(TmTensorDialect
  IR/TmTensorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Dialect/TmTensor

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)