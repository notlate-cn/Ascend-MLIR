add_mlir_dialect_library(AFIRDialect
  AFIRDialect.cpp
  AFIROps.cpp

  ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include/Dialect/AFIR

  DEPENDS
  AFIRDialectIncGen
  AFIROpsIncGen
  AFIRShapeHelperOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  AFIRInterface
)

add_subdirectory(Transforms)
