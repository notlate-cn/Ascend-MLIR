add_mlir_library(MarkStructuredOpsConversion
  MarkStructuredOpsPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransforms
)
