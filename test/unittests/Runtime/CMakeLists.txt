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

add_executable(MatmulTilingDispatcherTest
  MatmulTilingDispatcherTest.cpp
)

target_link_libraries(MatmulTilingDispatcherTest PRIVATE
  RuntimeUnitTestSupport
  AscendCRuntime
)

add_dependencies(RuntimeUnitTests MatmulTilingDispatcherTest)

add_test(NAME MatmulTilingDispatcherTest
  COMMAND $<TARGET_FILE:MatmulTilingDispatcherTest>
)
