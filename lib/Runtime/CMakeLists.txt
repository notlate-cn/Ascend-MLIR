# lib/Runtime/CMakeLists.txt
add_mlir_library(AscendCRuntime
  Artifact/ArtifactCompiler.cpp
  Artifact/RunManifest.cpp
  Execution/ExecutionBackend.cpp
  Execution/ExecutionSession.cpp
  Execution/NpuBackend.cpp
  Execution/SimBackend.cpp
  Execution/TaskGraph.cpp
  Legacy/CompatRuntime.cpp
  Legacy/Compiler.cpp
  Legacy/Executor.cpp
  Legacy/HostRunnerGen.cpp
  Legacy/SimValidator.cpp
  Mix/AscendCMixCompiler.cpp
  Mix/MixAbi.cpp
  Mix/MixAbiExtractor.cpp
  Mix/MixCommandBuilder.cpp
  Mix/MixDirectBackend.cpp
  Mix/MixSourceAnalyzer.cpp
  Mix/MixStubTemplate.cpp
  Profile/ProfileTrace.cpp
  Profile/ProfileUtils.cpp
  Support/NpyIO.cpp
  Support/PathUtils.cpp
  Support/TilingPack.cpp
  Support/TilingSchema.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Runtime

  LINK_LIBS PUBLIC
  LLVMSupport
)

include(${CMAKE_CURRENT_LIST_DIR}/Mix/AscendCannPaths.cmake)
resolve_ascend_cann_arch_dir(ASCEND_CANN_ARCH)

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
