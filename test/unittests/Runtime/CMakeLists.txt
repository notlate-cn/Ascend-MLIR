add_executable(MatmulTilingTypesTest
  MatmulTilingTypesTest.cpp
)

target_link_libraries(MatmulTilingTypesTest PRIVATE
  RuntimeUnitTestSupport
)

add_dependencies(RuntimeUnitTests MatmulTilingTypesTest)

add_test(NAME MatmulTilingTypesTest
  COMMAND $<TARGET_FILE:MatmulTilingTypesTest>
)
