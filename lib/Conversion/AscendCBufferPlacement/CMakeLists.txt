add_mlir_library(AscendCBufferPlacementConversion
  AscendCBufferPlacementPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRAsc
  MLIRTransforms
)
