# lib/Runtime/CMakeLists.txt
add_mlir_library(AscendCRuntime
  Compiler.cpp
  Executor.cpp
  HostRunnerGen.cpp
  SimValidator.cpp
  NpyIO.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Runtime

  LINK_LIBS PUBLIC
  LLVMSupport
)
