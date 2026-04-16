add_executable(MatmulTilingTypesTest
  MatmulTilingTypesTest.cpp
  ../gtest/gtest_main.cpp
)

target_include_directories(MatmulTilingTypesTest PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_custom_target(RuntimeUnitTests)
set_target_properties(RuntimeUnitTests PROPERTIES FOLDER "Tests")
add_dependencies(RuntimeUnitTests MatmulTilingTypesTest)

add_test(NAME MatmulTilingTypesTest
  COMMAND $<TARGET_FILE:MatmulTilingTypesTest>
)
