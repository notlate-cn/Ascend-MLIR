add_mlir_library(AFIRInterface
  ShapeHelperOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include/Interface

  DEPENDS
  AFIRShapeHelperOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
