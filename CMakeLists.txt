cmake_minimum_required(VERSION 3.20)
project(Ascend-MLIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 生成compile_commands.json文件
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# LLVM/MLIR配置
# 用户通过 -DLLVM_BUILD_DIR 指定 LLVM 构建根目录
# 如果未指定，则使用默认的 externals/llvm-project/build
# 注意：必须 unset LLVM_DIR 缓存，否则当 MLIR_DIR 改变时，LLVM_DIR 不会相应改变
unset(LLVM_DIR CACHE)

# 如果用户指定了 LLVM_BUILD_DIR，自动推导 MLIR_DIR
if(LLVM_BUILD_DIR AND NOT MLIR_DIR)
  set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir")
  message(STATUS "LLVM_BUILD_DIR: ${LLVM_BUILD_DIR}")
endif()

# 如果 MLIR_DIR 仍未设置，使用默认路径
if(NOT MLIR_DIR)
  set(MLIR_DIR "${CMAKE_SOURCE_DIR}/externals/llvm-project/build/lib/cmake/mlir")
endif()

message(STATUS "MLIR_DIR: ${MLIR_DIR}")

# 检查 MLIR_DIR 是否有效
if(NOT EXISTS "${MLIR_DIR}/MLIRConfig.cmake")
  message(FATAL_ERROR
    "MLIRConfig.cmake not found at: ${MLIR_DIR}\n"
    "\n"
    "Please specify LLVM build directory:\n"
    "  cmake -DLLVM_BUILD_DIR=/path/to/llvm-project/build ..\n"
    "\n"
    "Example:\n"
    "  cmake -DLLVM_BUILD_DIR=../llvm-project/build ..\n"
    "\n"
    "If you haven't built LLVM yet, run: ./scripts/build_llvm.sh"
  )
endif()

# find_package(MLIR) 会自动加载 LLVM
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
add_subdirectory(test)

# 只构建必要的pyasc，注意需要先构建其tablegen
add_subdirectory(${PYASC_DIR}/lib/TableGen EXCLUDE_FROM_ALL)
add_subdirectory(${PYASC_DIR}/include EXCLUDE_FROM_ALL)
add_subdirectory(${PYASC_DIR}/lib EXCLUDE_FROM_ALL)

# 测试配置
enable_testing()
