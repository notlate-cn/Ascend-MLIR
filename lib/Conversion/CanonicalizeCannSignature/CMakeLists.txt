add_mlir_library(CanonicalizeCannSignatureConversion
  CanonicalizeCannSignaturePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIREmitAsc
  MLIRMemRefDialect
  MLIRTransforms
  MLIRIR
)
