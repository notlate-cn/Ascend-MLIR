add_mlir_dialect_library(AFIRDialect
  Dialect.cpp
  Ops/Basic/math.cpp
  Ops/Basic/data.cpp
  Ops/Basic/broadcast.cpp
  Ops/Basic/condition.cpp
  Ops/Basic/cast.cpp
  Ops/Adv/concat.cpp
  Ops/Adv/reduce.cpp

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
