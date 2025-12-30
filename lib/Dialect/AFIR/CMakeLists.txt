add_mlir_dialect_library(AFIRDialect
  AFIRDialect.cpp
  AFIROps.cpp
  AFIROps/Math/Elementwise.cpp
  ShapeHelper.cpp

  ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include/Dialect/AFIR

  DEPENDS
  AFIRDialectIncGen
  AFIROpsIncGen
  AFIRShapeHelperOpInterfaceIncGen
  AFIRElementwiseOpsIncGen
  MLIRAscIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRCastInterfaces
  AFIRInterface
  MLIRAsc
  MLIRCastInterfaces
)

target_include_directories(AFIRDialect PRIVATE
  ${CMAKE_BINARY_DIR}/externals/pyasc/include
)

add_subdirectory(Transforms)
