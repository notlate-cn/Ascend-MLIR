add_executable(AscendTargetMemoryModelTest
  AscendTargetMemoryModelTest.cpp
)

target_link_libraries(AscendTargetMemoryModelTest PRIVATE
  RuntimeUnitTestSupport
  AscendTargetProfile
)

add_dependencies(RuntimeUnitTests AscendTargetMemoryModelTest)

add_test(NAME AscendTargetMemoryModelTest
  COMMAND $<TARGET_FILE:AscendTargetMemoryModelTest>
)
