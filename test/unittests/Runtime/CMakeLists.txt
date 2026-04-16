get_filename_component(LLVM_PROJECT_DIR "${LLVM_BUILD_MAIN_SRC_DIR}/.." ABSOLUTE)
set(GOOGLETEST_DIR "${LLVM_PROJECT_DIR}/third-party/unittest/googletest")

add_executable(MatmulTilingTypesTest
  MatmulTilingTypesTest.cpp
  ${GOOGLETEST_DIR}/src/gtest-all.cc
  ${GOOGLETEST_DIR}/src/gtest_main.cc
)

target_include_directories(MatmulTilingTypesTest PRIVATE
  ${GOOGLETEST_DIR}/include
  ${GOOGLETEST_DIR}
)

target_compile_definitions(MatmulTilingTypesTest PRIVATE
  GTEST_NO_LLVM_SUPPORT=1
)

add_dependencies(RuntimeUnitTests MatmulTilingTypesTest)

add_test(NAME MatmulTilingTypesTest
  COMMAND $<TARGET_FILE:MatmulTilingTypesTest>
)
