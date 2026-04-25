add_mlir_library(LowerNonLinalgOpsConversion
  AclnnFinalizeDeclPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
)
