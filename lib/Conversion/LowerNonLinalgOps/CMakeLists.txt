add_mlir_library(LowerNonLinalgOpsConversion
  AclnnFinalizeDeclPass.cpp
  RecognizeAttentionPass.cpp
  RecognizeLayerNormPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRPass
)
