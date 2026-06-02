set_directory_properties(PROPERTIES
  LABELS "runtime-unittest;conversion-unittest"
)

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

add_executable(AscendScheduleTemplateRegistryTest
  AscendScheduleTemplateRegistryTest.cpp
)

target_include_directories(AscendScheduleTemplateRegistryTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendScheduleTemplateRegistryTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendScheduleTemplateRegistryTest)

add_test(NAME AscendScheduleTemplateRegistryTest
  COMMAND $<TARGET_FILE:AscendScheduleTemplateRegistryTest>
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

target_include_directories(AscendKernelizeOpInterfaceTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendKernelizeOpInterfaceTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendKernelizeOpInterfaceTest)

add_test(NAME AscendKernelizeOpInterfaceTest
  COMMAND $<TARGET_FILE:AscendKernelizeOpInterfaceTest>
)

add_executable(AscendHandwrittenContractRegistryTest
  AscendHandwrittenContractRegistryTest.cpp
)

target_include_directories(AscendHandwrittenContractRegistryTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendHandwrittenContractRegistryTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendHandwrittenContractRegistryTest)

add_test(NAME AscendHandwrittenContractRegistryTest
  COMMAND $<TARGET_FILE:AscendHandwrittenContractRegistryTest>
)

add_executable(AscendElementwiseBodyOpRegistryTest
  AscendElementwiseBodyOpRegistryTest.cpp
)

target_include_directories(AscendElementwiseBodyOpRegistryTest PRIVATE
  ${ASCEND_CONVERSION_INTERNAL_INCLUDE_DIR}
)

target_link_libraries(AscendElementwiseBodyOpRegistryTest PRIVATE
  RuntimeUnitTestSupport
  AscendConversion
)

add_dependencies(RuntimeUnitTests AscendElementwiseBodyOpRegistryTest)

add_test(NAME AscendElementwiseBodyOpRegistryTest
  COMMAND $<TARGET_FILE:AscendElementwiseBodyOpRegistryTest>
)
