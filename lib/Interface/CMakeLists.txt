add_mlir_library(AFIRInterface
  ShapeHelperOpInterface.cpp
  ShapeInferenceOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include/Interface

  DEPENDS
  AFIRShapeHelperOpInterfaceIncGen
  AFIRShapeInferenceOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
