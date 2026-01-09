add_mlir_library(AFIRUtils
  AFIRUtils.cpp

  ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include/Utils

  DEPENDS
  MLIRAscIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
