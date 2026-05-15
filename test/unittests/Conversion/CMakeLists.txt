set(ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/lib)

add_executable(AscendKernelPatternTest
  AscendKernelPatternTest.cpp
)

target_include_directories(AscendKernelPatternTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
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

target_include_directories(AscendCommonAttributesTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendCommonAttributesTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendCommonAttributesTest)

add_test(NAME AscendCommonAttributesTest
  COMMAND $<TARGET_FILE:AscendCommonAttributesTest>
)

add_executable(AscendScheduleDecisionTest
  AscendScheduleDecisionTest.cpp
)

target_include_directories(AscendScheduleDecisionTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendScheduleDecisionTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendScheduleDecisionTest)

add_test(NAME AscendScheduleDecisionTest
  COMMAND $<TARGET_FILE:AscendScheduleDecisionTest>
)

add_executable(AscendRealizePlannerTest
  AscendRealizePlannerTest.cpp
)

target_include_directories(AscendRealizePlannerTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendRealizePlannerTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
  MLIRParser
)

add_dependencies(RuntimeUnitTests AscendRealizePlannerTest)

add_test(NAME AscendRealizePlannerTest
  COMMAND $<TARGET_FILE:AscendRealizePlannerTest>
)

add_executable(AscendBackendSupportMatrixTest
  AscendBackendSupportMatrixTest.cpp
)

target_link_libraries(AscendBackendSupportMatrixTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendBackendSupportMatrixTest)

add_test(NAME AscendBackendSupportMatrixTest
  COMMAND $<TARGET_FILE:AscendBackendSupportMatrixTest>
)

add_executable(AscendLinalgBodyClassifierTest
  AscendLinalgBodyClassifierTest.cpp
)

target_link_libraries(AscendLinalgBodyClassifierTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
  MLIRParser
)

add_dependencies(RuntimeUnitTests AscendLinalgBodyClassifierTest)

add_test(NAME AscendLinalgBodyClassifierTest
  COMMAND $<TARGET_FILE:AscendLinalgBodyClassifierTest>
)

add_executable(AscendKernelizeOpInterfaceTest
  AscendKernelizeOpInterfaceTest.cpp
)

target_link_libraries(AscendKernelizeOpInterfaceTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendKernelizeOpInterfaceTest)

add_test(NAME AscendKernelizeOpInterfaceTest
  COMMAND $<TARGET_FILE:AscendKernelizeOpInterfaceTest>
)
