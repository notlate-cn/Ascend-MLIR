set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(mix-tiling-helper
  mix_tiling_helper_main.cpp
)

target_include_directories(mix-tiling-helper PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

include(${CMAKE_SOURCE_DIR}/lib/Runtime/Mix/AscendCannPaths.cmake)
resolve_ascend_cann_arch_dir(ASCEND_CANN_ARCH)

set(MIX_TILING_SOC_VERSION "$ENV{SOC_VERSION}")
if(MIX_TILING_SOC_VERSION STREQUAL "")
  set(MIX_TILING_SOC_VERSION "Ascend910B1")
endif()

set(MIX_TILING_ASCEND_HOME "")
set(MIX_TILING_ASCEND_HOME_CANDIDATES
  "$ENV{ASCEND_HOME_PATH}"
  "$ENV{ASCEND_TOOLKIT_HOME}"
  "$ENV{HOME}/Ascend/latest"
  "$ENV{HOME}/Ascend/20260323_newest/ascend-toolkit/latest"
  "$ENV{HOME}/Ascend/20260323_newest-full/ascend-toolkit/latest"
  "/usr/local/Ascend/ascend-toolkit/latest"
)

foreach(_candidate ${MIX_TILING_ASCEND_HOME_CANDIDATES})
  if(NOT _candidate STREQUAL "" AND EXISTS "${_candidate}")
    set(MIX_TILING_ASCEND_HOME "${_candidate}")
    break()
  endif()
endforeach()

set(MIX_TILING_INCLUDE_DIRS)
if(NOT MIX_TILING_ASCEND_HOME STREQUAL "")
  list(APPEND MIX_TILING_INCLUDE_DIRS
    ${MIX_TILING_ASCEND_HOME}/include
  )
  if(NOT ASCEND_CANN_ARCH STREQUAL "")
    list(APPEND MIX_TILING_INCLUDE_DIRS
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/include
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/asc/include
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/ascendc/include
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/include/ascendc/host_api
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/asc/include
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/tikcpp
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/tikcpp/tikcfw
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/tikcpp/tikcfw/include
    )
  endif()
endif()

set(MIX_TILING_FOUND_PLATFORM_HEADER FALSE)
foreach(_dir ${MIX_TILING_INCLUDE_DIRS})
  if(EXISTS ${_dir})
    target_include_directories(mix-tiling-helper PRIVATE ${_dir})
    if(EXISTS "${_dir}/tiling/platform/platform_ascendc.h")
      set(MIX_TILING_FOUND_PLATFORM_HEADER TRUE)
    endif()
  endif()
endforeach()

if(NOT MIX_TILING_FOUND_PLATFORM_HEADER)
  message(FATAL_ERROR
    "mix-tiling-helper could not find tiling/platform/platform_ascendc.h. "
    "Set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME, or install Ascend toolkit under a default path.")
endif()

set(MIX_TILING_LINK_DIRS)
if(NOT MIX_TILING_ASCEND_HOME STREQUAL "")
  if(NOT ASCEND_CANN_ARCH STREQUAL "")
    list(APPEND MIX_TILING_LINK_DIRS
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/lib64
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/simulator/${MIX_TILING_SOC_VERSION}/lib
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/lib64
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/lib64/device/lib64
    )
  endif()
endif()

foreach(_dir ${MIX_TILING_LINK_DIRS})
  if(EXISTS ${_dir})
    target_link_directories(mix-tiling-helper PRIVATE ${_dir})
  endif()
endforeach()

target_link_libraries(mix-tiling-helper PRIVATE
  AscendCRuntime
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
