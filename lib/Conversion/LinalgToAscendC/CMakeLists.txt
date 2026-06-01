add_mlir_library(LinalgToAscendCConversion
  LinalgToAscendCPass.cpp
  DataMoveConversion.cpp
  ComputeConversion.cpp
  ComputeConversionContext.cpp
  ComputeConversionHelpers.cpp
  ComputeParallel.cpp
  ComputeReduce.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Conversion

  DEPENDS
  AFIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRAsc
  MLIRAscUtils
  MLIREmitAsc
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTransforms
)
