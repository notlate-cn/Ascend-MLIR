set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(validator
  validator_main.cpp
)

target_link_libraries(validator PRIVATE
  AscendCRuntime
)

target_include_directories(validator PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(validator PRIVATE
  ${CMAKE_DL_LIBS}
)
