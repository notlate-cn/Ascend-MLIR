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

add_executable(MatmulApiTilingBackendTest
  MatmulApiTilingBackendTest.cpp
)

target_link_libraries(MatmulApiTilingBackendTest PRIVATE
  RuntimeUnitTestSupport
  AscendCRuntime
)

add_dependencies(RuntimeUnitTests MatmulApiTilingBackendTest)

add_test(NAME MatmulApiTilingBackendTest
  COMMAND $<TARGET_FILE:MatmulApiTilingBackendTest>
)

add_executable(MixTilingGeneratorTest
  MixTilingGeneratorTest.cpp
)

target_link_libraries(MixTilingGeneratorTest PRIVATE
  RuntimeUnitTestSupport
  AscendCRuntime
)

add_dependencies(RuntimeUnitTests MixTilingGeneratorTest)

add_test(NAME MixTilingGeneratorTest
  COMMAND $<TARGET_FILE:MixTilingGeneratorTest>
)
