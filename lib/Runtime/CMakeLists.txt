# lib/Runtime/CMakeLists.txt
add_mlir_library(AscendCRuntime
  AscendCMixCompiler.cpp
  Compiler.cpp
  Executor.cpp
  HostRunnerGen.cpp
  PathUtils.cpp
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

set(ASCEND_CANN_ARCH "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(ASCEND_CANN_ARCH "x86_64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(ASCEND_CANN_ARCH "aarch64-linux")
endif()

set(ASCEND_INCLUDE_DIRS
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_TOOLKIT_HOME}/include)
if(NOT ASCEND_CANN_ARCH STREQUAL "")
  list(APPEND ASCEND_INCLUDE_DIRS
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/include
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/include)
endif()

foreach(_dir ${ASCEND_INCLUDE_DIRS})
  if(EXISTS ${_dir})
    target_include_directories(AscendCRuntime PRIVATE ${_dir})
  endif()
endforeach()

target_link_libraries(AscendCRuntime PRIVATE
  ${CMAKE_DL_LIBS}
)
