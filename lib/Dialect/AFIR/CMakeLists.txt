add_mlir_dialect_library(AFIRDialect
  Dialect.cpp
  Ops/Basic/math.cpp

  ADDITIONAL_HEADER_DIRS
        ${CMAKE_SOURCE_DIR}/include/Dialect/AFIR

  DEPENDS
  AFIRDialectIncGen
  AFIROpsIncGen
  MLIRAscIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRCastInterfaces
  MLIRAsc
  MLIRCastInterfaces
)

target_include_directories(AFIRDialect PRIVATE
  ${CMAKE_BINARY_DIR}/externals/pyasc/include
)

add_subdirectory(Transforms)
