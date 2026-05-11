add_mlir_library(AscendCPrepareForEmitConversion
  AscendCPrepareForEmitPass.cpp
  FlattenGMPtrPass.cpp
  PackTilingDataPass.cpp
  FinalizeKernelPass.cpp
  DecomposeMultiAxisBroadcastPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRAsc
  MLIREmitAsc
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTransforms
)
