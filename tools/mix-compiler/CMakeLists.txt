set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(mix-compiler
  mix_compiler_main.cpp
)

target_link_libraries(mix-compiler PRIVATE
  AscendCRuntime
)

target_include_directories(mix-compiler PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

add_dependencies(mix-compiler mix-tiling-helper)
