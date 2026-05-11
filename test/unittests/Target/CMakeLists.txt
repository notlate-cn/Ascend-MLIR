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

add_executable(AscendTargetIntrinsicModelTest
  AscendTargetIntrinsicModelTest.cpp
)

target_link_libraries(AscendTargetIntrinsicModelTest PRIVATE
  RuntimeUnitTestSupport
  AscendTargetProfile
)

add_dependencies(RuntimeUnitTests AscendTargetIntrinsicModelTest)

add_test(NAME AscendTargetIntrinsicModelTest
  COMMAND $<TARGET_FILE:AscendTargetIntrinsicModelTest>
)

add_executable(AscendTargetCostModelTest
  AscendTargetCostModelTest.cpp
)

target_link_libraries(AscendTargetCostModelTest PRIVATE
  RuntimeUnitTestSupport
  AscendTargetProfile
)

add_dependencies(RuntimeUnitTests AscendTargetCostModelTest)

add_test(NAME AscendTargetCostModelTest
  COMMAND $<TARGET_FILE:AscendTargetCostModelTest>
)
