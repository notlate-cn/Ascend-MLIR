set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(compiler
  compiler_main.cpp
)

target_link_libraries(compiler PRIVATE
  AscendCRuntime
)

target_include_directories(compiler PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
