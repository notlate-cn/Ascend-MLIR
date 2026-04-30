add_mlir_public_c_api_library(AFIRCAPI
  PARTIAL_SOURCES_INTENDED
  AFIR.cpp
  AFIRPasses.cpp

  DEPENDS
  AFIRConversionPassIncGen
  AFIRTransformsPassIncGen

  LINK_LIBS PUBLIC
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
  FuseGatherElementwiseConversion
  MarkStructuredOpsConversion
)