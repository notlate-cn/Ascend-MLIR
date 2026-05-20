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

if(ASCEND_RUNTIME_SESSION_RUN_ONLY)
  target_compile_definitions(runtime-session PRIVATE
    ASCEND_RUNTIME_SESSION_RUN_ONLY=1)
  target_link_options(runtime-session PRIVATE "LINKER:--as-needed")
endif()

if(NOT ASCEND_RUNTIME_SESSION_RUN_ONLY)
  add_dependencies(runtime-session mix-tiling-helper)
endif()
