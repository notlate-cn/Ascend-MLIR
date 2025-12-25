add_mlir_dialect_library(AFIRTransforms
  AFIRCanonicalize.cpp
  AFIRShapeInference.cpp

  ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include/Dialect/AFIR/Transforms

  DEPENDS
  AFIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRPass
  MLIRTransforms
)
