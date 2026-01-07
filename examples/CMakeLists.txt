add_executable(demo
  demo.cpp
)

target_include_directories(demo PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(demo PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRLinalgDialect
  AFIRDialect
)
