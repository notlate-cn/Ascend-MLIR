add_executable(AscendV2KernelPatternTest
  AscendV2KernelPatternTest.cpp
)

target_link_libraries(AscendV2KernelPatternTest PRIVATE
  RuntimeUnitTestSupport
  AscendV2Conversion
)

add_dependencies(RuntimeUnitTests AscendV2KernelPatternTest)

add_test(NAME AscendV2KernelPatternTest
  COMMAND $<TARGET_FILE:AscendV2KernelPatternTest>
)

add_executable(AscendV2CommonAttributesTest
  AscendV2CommonAttributesTest.cpp
)

target_link_libraries(AscendV2CommonAttributesTest PRIVATE
  RuntimeUnitTestSupport
  AscendV2Conversion
)

add_dependencies(RuntimeUnitTests AscendV2CommonAttributesTest)

add_test(NAME AscendV2CommonAttributesTest
  COMMAND $<TARGET_FILE:AscendV2CommonAttributesTest>
)
