cmake_minimum_required(VERSION 3.20)
project(Ascend-MLIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# LLVM/MLIR配置
set(LLVM_DIR "${CMAKE_SOURCE_DIR}/externals/llvm-project/build/lib/cmake/llvm")
set(MLIR_DIR "${CMAKE_SOURCE_DIR}/externals/llvm-project/build/lib/cmake/mlir")
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

# StableHLO配置
set(STABLEHLO_DIR "${CMAKE_SOURCE_DIR}/externals/stablehlo")
include_directories(${STABLEHLO_DIR}/include)

set(PYASC_DIR "${CMAKE_SOURCE_DIR}/externals/pyasc")
set(PYASC_BIN_DIR "${CMAKE_BINARY_DIR}/externals/pyasc")
include_directories(${PYASC_DIR}/include)
include_directories(${PYASC_BIN_DIR}/include)

# 添加LLVM定义
add_definitions(${LLVM_DEFINITIONS})

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加子目录
add_subdirectory(include/Dialect/AFIR)
add_subdirectory(include/Interface)
add_subdirectory(include/Conversion)
add_subdirectory(lib/Dialect/AFIR)
add_subdirectory(lib/Interface)
add_subdirectory(lib/Conversion)
add_subdirectory(lib/Utils)
add_subdirectory(tools/afir-opt)
add_subdirectory(tools/afir-tool)
add_subdirectory(test)
add_subdirectory(${PYASC_DIR} EXCLUDE_FROM_ALL)

# 测试配置
enable_testing()
