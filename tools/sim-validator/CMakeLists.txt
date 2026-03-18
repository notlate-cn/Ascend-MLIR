# tools/sim-validator/CMakeLists.txt
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(sim-validator
  sim_validator_main.cpp
)

target_link_libraries(sim-validator PRIVATE
  AscendCRuntime
)

target_include_directories(sim-validator PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
