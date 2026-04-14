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

set(MIX_TILING_ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
if(MIX_TILING_ASCEND_HOME STREQUAL "")
  set(MIX_TILING_ASCEND_HOME "$ENV{ASCEND_TOOLKIT_HOME}")
endif()

set(MIX_TILING_INCLUDE_DIRS)
if(NOT MIX_TILING_ASCEND_HOME STREQUAL "")
  list(APPEND MIX_TILING_INCLUDE_DIRS
    ${MIX_TILING_ASCEND_HOME}/include
  )
  if(NOT ASCEND_CANN_ARCH STREQUAL "")
    list(APPEND MIX_TILING_INCLUDE_DIRS
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/include
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/asc/include
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/tikcpp
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/tikcpp/tikcfw
      ${MIX_TILING_ASCEND_HOME}/${ASCEND_CANN_ARCH}/tikcpp/tikcfw/include
    )
  endif()
endif()

foreach(_dir ${MIX_TILING_INCLUDE_DIRS})
  if(EXISTS ${_dir})
    target_include_directories(mix-tiling-helper PRIVATE ${_dir})
  endif()
endforeach()

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
