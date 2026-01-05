add_executable(elementwise_ops_demo
  ElementwiseOpsDemo.cpp
)

target_include_directories(elementwise_ops_demo PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(elementwise_ops_demo PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRLinalgDialect
  AFIRDialect
)
