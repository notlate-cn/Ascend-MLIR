set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(mix-validator
  mix_validator_main.cpp
)

target_link_libraries(mix-validator PRIVATE
  AscendCRuntime
)

target_include_directories(mix-validator PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
