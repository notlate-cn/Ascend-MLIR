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

target_include_directories(MatmulApiTilingBackendTest PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/Runtime/Mix
)

target_link_libraries(MatmulApiTilingBackendTest PRIVATE
  RuntimeUnitTestSupport
  AscendCRuntime
)

add_dependencies(RuntimeUnitTests MatmulApiTilingBackendTest)

add_test(NAME MatmulApiTilingBackendTest
  COMMAND $<TARGET_FILE:MatmulApiTilingBackendTest>
)

add_executable(MixCommandBuilderTest
  MixCommandBuilderTest.cpp
)

target_link_libraries(MixCommandBuilderTest PRIVATE
  RuntimeUnitTestSupport
  AscendCRuntime
)

add_dependencies(RuntimeUnitTests MixCommandBuilderTest)

add_test(NAME MixCommandBuilderTest
  COMMAND $<TARGET_FILE:MixCommandBuilderTest>
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

add_executable(NativeMatmulTilingBackendTest
  NativeMatmulTilingBackendTest.cpp
)

target_link_libraries(NativeMatmulTilingBackendTest PRIVATE
  RuntimeUnitTestSupport
  AscendCRuntime
)

add_dependencies(RuntimeUnitTests NativeMatmulTilingBackendTest)

add_test(NAME NativeMatmulTilingBackendTest
  COMMAND $<TARGET_FILE:NativeMatmulTilingBackendTest>
)

add_executable(NativeMatmulPlannerTest
  NativeMatmulPlannerTest.cpp
)

target_link_libraries(NativeMatmulPlannerTest PRIVATE
  RuntimeUnitTestSupport
  AscendCRuntime
)

add_dependencies(RuntimeUnitTests NativeMatmulPlannerTest)

add_test(NAME NativeMatmulPlannerTest
  COMMAND $<TARGET_FILE:NativeMatmulPlannerTest>
)

add_executable(MixAbiTest
  MixAbiTest.cpp
)

target_link_libraries(MixAbiTest PRIVATE
  RuntimeUnitTestSupport
  AscendCRuntime
)

add_dependencies(RuntimeUnitTests MixAbiTest)

add_test(NAME MixAbiTest
  COMMAND $<TARGET_FILE:MixAbiTest>
)

add_executable(MixDirectTilingArtifactsTest
  MixDirectTilingArtifactsTest.cpp
)

target_include_directories(MixDirectTilingArtifactsTest PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/Runtime/Mix
)

target_link_libraries(MixDirectTilingArtifactsTest PRIVATE
  RuntimeUnitTestSupport
  AscendCRuntime
)

add_dependencies(RuntimeUnitTests MixDirectTilingArtifactsTest)

add_test(NAME MixDirectTilingArtifactsTest
  COMMAND $<TARGET_FILE:MixDirectTilingArtifactsTest>
)

add_executable(NativeLaunchTraceTest
  NativeLaunchTraceTest.cpp
)

target_include_directories(NativeLaunchTraceTest PRIVATE
  ${CMAKE_SOURCE_DIR}/lib/Runtime/Execution
)

target_link_libraries(NativeLaunchTraceTest PRIVATE
  RuntimeUnitTestSupport
  AscendCRuntime
)

add_dependencies(RuntimeUnitTests NativeLaunchTraceTest)

add_test(NAME NativeLaunchTraceTest
  COMMAND $<TARGET_FILE:NativeLaunchTraceTest>
)
