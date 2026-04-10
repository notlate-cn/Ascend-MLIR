# tools/autotuner/CMakeLists.txt
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(autotuner
  autotuner_main.cpp
)

target_link_libraries(autotuner PRIVATE
  AscendCRuntime
)

target_include_directories(autotuner PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
