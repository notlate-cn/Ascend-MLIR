add_mlir_library(LowerNonLinalgOpsConversion
  AclnnFinalizeDeclPass.cpp
  FuseTransposeIntoElementwisePass.cpp
  RecognizeAttentionPass.cpp
  RecognizeBatchNormPass.cpp
  RecognizeEmbeddingPass.cpp
  RecognizeLayerNormPass.cpp
  LowerBroadcastExtractPass.cpp

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
  # transposeSupportedByIntrinsic — the on-chip transpose capability gate.
  LinalgToAscendCConversion
)
