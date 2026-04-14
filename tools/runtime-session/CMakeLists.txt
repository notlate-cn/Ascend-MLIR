set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(runtime-session
  runtime_session_main.cpp
)

target_link_libraries(runtime-session PRIVATE
  AscendCRuntime
)

target_include_directories(runtime-session PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

add_dependencies(runtime-session mix-tiling-helper)
