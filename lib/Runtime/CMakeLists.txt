# lib/Runtime/CMakeLists.txt
add_mlir_library(AscendCRuntime
  AscendCMixCompiler.cpp
  Compiler.cpp
  Executor.cpp
  HostRunnerGen.cpp
  MixCommandBuilder.cpp
  MixAbiExtractor.cpp
  MixDirectBackend.cpp
  MixAbi.cpp
  MixSourceAnalyzer.cpp
  MixStubTemplate.cpp
  SimValidator.cpp
  NpyIO.cpp
  TilingSchema.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Runtime

  LINK_LIBS PUBLIC
  LLVMSupport
)

foreach(_dir
    /home/niu/Ascend/latest/include
    /usr/local/Ascend/latest/include
    /usr/local/Ascend/ascend-toolkit/latest/include)
  if(EXISTS ${_dir})
    target_include_directories(AscendCRuntime PRIVATE ${_dir})
  endif()
endforeach()

target_link_libraries(AscendCRuntime PRIVATE
  ${CMAKE_DL_LIBS}
)
