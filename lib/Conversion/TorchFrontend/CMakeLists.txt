add_mlir_library(TorchFrontendConversion
  ConvertTmTensorAttentionPass.cpp
  RemoveCfAssertPass.cpp
  Pipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion/TorchFrontend

  DEPENDS
  TorchFrontendPassIncGen

  LINK_LIBS PUBLIC
  TmTensorDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRControlFlowDialect
  MLIRPass
  MLIRTransforms
)
