# lib/Runtime/CMakeLists.txt
option(ASCEND_RUNTIME_SESSION_RUN_ONLY
  "Build runtime-session with only --run-manifest planning/execution support"
  OFF)

add_mlir_library(AscendCRuntime
  Artifact/ArtifactCompiler.cpp
  Artifact/VecCubeArtifactBackend.cpp
  Artifact/RunManifest.cpp
  Artifact/RuntimeSessionArtifactCompile.cpp
  Artifact/RuntimeSessionRequestBuilder.cpp
  Execution/DefaultExecutionRunner.cpp
  Execution/DynamicLibraryArtifactEnv.cpp
  Execution/BackendCapabilities.cpp
  Execution/ExecutionBackend.cpp
  Execution/OutputComparator.cpp
  Execution/RuntimeFrontendCore.cpp
  Execution/ExecutionSession.cpp
  Execution/GlobalScheduler.cpp
  Execution/LaunchTrace.cpp
  Execution/NpuBackend.cpp
  Execution/NativeExecutionRunner.cpp
  Execution/ResourceScheduler.cpp
  Execution/SessionHandle.cpp
  Execution/SimBackend.cpp
  Execution/TaskGraph.cpp
  Mix/AscendCMixCompiler.cpp
  Mix/MixAbi.cpp
  Mix/MixAbiExtractor.cpp
  Mix/MixCompileMetadata.cpp
  Mix/MixCommandBuilder.cpp
  Mix/MixDirectBackend.cpp
  Mix/MixDirectBinaryBuild.cpp
  Mix/MixDirectCompileLayout.cpp
  Mix/MixDirectCompilePipeline.cpp
  Mix/MixDirectCompileSupport.cpp
  Mix/MixDirectPreprocess.cpp
  Mix/MixDirectRuntimeAbi.cpp
  Mix/MixDirectTilingArtifacts.cpp
  Mix/MatmulApiTilingBackend.cpp
  Mix/MatmulTilingDispatcher.cpp
  Mix/NativeMatmulPlanner.cpp
  Mix/NativeMatmulTilingBackend.cpp
  Mix/MixTilingGenerator.cpp
  Mix/MixSourceAnalyzer.cpp
  Mix/MixStubTemplate.cpp
  Profile/ProfileTrace.cpp
  Profile/ProfileUtils.cpp
  Support/NpyIO.cpp
  Support/PathUtils.cpp
  Support/ToolDiscovery.cpp
  Support/TilingPack.cpp
  Support/TilingSchema.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_SOURCE_DIR}/include/Runtime

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  LLVMSupport
)

include(${CMAKE_CURRENT_LIST_DIR}/Mix/AscendCannPaths.cmake)
resolve_ascend_cann_arch_dir(ASCEND_CANN_ARCH)

set(ASCEND_RUNTIME_SOC_VERSION "$ENV{SOC_VERSION}")
if(ASCEND_RUNTIME_SOC_VERSION STREQUAL "")
  set(ASCEND_RUNTIME_SOC_VERSION "Ascend910B1")
endif()

set(ASCEND_INCLUDE_DIRS
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_TOOLKIT_HOME}/include)
if(NOT ASCEND_CANN_ARCH STREQUAL "")
  list(APPEND ASCEND_INCLUDE_DIRS
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/include
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/include
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/asc/include
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/asc/include
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/ascendc/include
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/ascendc/include
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/include/ascendc/host_api
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/include/ascendc/host_api
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/tikcpp
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/tikcpp
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/tikcpp/tikcfw
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/tikcpp/tikcfw
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/tikcpp/tikcfw/include
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/tikcpp/tikcfw/include)
endif()

foreach(_dir ${ASCEND_INCLUDE_DIRS})
  if(EXISTS ${_dir})
    target_include_directories(AscendCRuntime PRIVATE ${_dir})
    if(TARGET obj.AscendCRuntime)
      target_include_directories(obj.AscendCRuntime PRIVATE ${_dir})
    endif()
  endif()
endforeach()

set(ASCEND_LINK_DIRS)
if(NOT ASCEND_CANN_ARCH STREQUAL "")
  list(APPEND ASCEND_LINK_DIRS
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/lib64
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/lib64
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/simulator/${ASCEND_RUNTIME_SOC_VERSION}/lib
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/simulator/${ASCEND_RUNTIME_SOC_VERSION}/lib
      $ENV{ASCEND_HOME_PATH}/${ASCEND_CANN_ARCH}/lib64/device/lib64
      $ENV{ASCEND_TOOLKIT_HOME}/${ASCEND_CANN_ARCH}/lib64/device/lib64)
endif()

if(NOT ASCEND_RUNTIME_SESSION_RUN_ONLY)
  foreach(_dir ${ASCEND_LINK_DIRS})
    if(EXISTS ${_dir})
      target_link_directories(AscendCRuntime PUBLIC ${_dir})
    endif()
  endforeach()
endif()

if(ASCEND_RUNTIME_SESSION_RUN_ONLY)
  target_link_libraries(AscendCRuntime PUBLIC
    ${CMAKE_DL_LIBS}
  )
else()
  target_link_libraries(AscendCRuntime PUBLIC
    tiling_api
    register
    platform
    ascendalog
    unified_dlog
    runtime_camodel
    npu_drv
    stars
    model_top
    ascendcl
    error_manager
    profapi
    ge_common_base
    ascend_dump
    mmpa
    c_sec
    ${CMAKE_DL_LIBS}
  )
endif()
