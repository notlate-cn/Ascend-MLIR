# tools/pipeline-analyzer/CMakeLists.txt
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(pipeline-analyzer
  pipeline_analyzer_main.cpp
)

target_include_directories(pipeline-analyzer PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
