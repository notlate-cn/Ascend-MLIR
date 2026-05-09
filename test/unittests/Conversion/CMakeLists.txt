add_executable(AscendKernelPatternTest
  AscendKernelPatternTest.cpp
)

target_link_libraries(AscendKernelPatternTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendKernelPatternTest)

add_test(NAME AscendKernelPatternTest
  COMMAND $<TARGET_FILE:AscendKernelPatternTest>
)

add_executable(AscendCommonAttributesTest
  AscendCommonAttributesTest.cpp
)

target_link_libraries(AscendCommonAttributesTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendCommonAttributesTest)

add_test(NAME AscendCommonAttributesTest
  COMMAND $<TARGET_FILE:AscendCommonAttributesTest>
)
