add_mlir_dialect_library(AFIRTransformOps
  AFIRTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include/Dialect/AFIR/TransformOps

  DEPENDS
  AFIRTransformOpsIncGen

  LINK_LIBS PUBLIC
  AFIRDialect
  MLIRFuncDialect
  MLIRTransformDialect
)

target_include_directories(AFIRTransformOps PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
