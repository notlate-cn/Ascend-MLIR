add_mlir_library(AFIRSymbolicShape
  SymExpr.cpp
  DimSymbolTable.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Analysis/SymbolicShape

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
