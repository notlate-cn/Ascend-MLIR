add_custom_target(RuntimeUnitTests)
set_target_properties(RuntimeUnitTests PROPERTIES FOLDER "Tests")

add_library(RuntimeUnitTestSupport INTERFACE)
set(_runtime_unittest_support_ready OFF)

if(TARGET GTest::gtest_main)
  target_link_libraries(RuntimeUnitTestSupport INTERFACE GTest::gtest_main)
  set(_runtime_unittest_support_ready ON)
elseif(TARGET GTest::gtest)
  target_link_libraries(RuntimeUnitTestSupport INTERFACE GTest::gtest)
  set(_runtime_unittest_support_ready ON)
endif()

if(NOT _runtime_unittest_support_ready)
  find_package(GTest CONFIG QUIET)
  if(TARGET GTest::gtest_main)
    target_link_libraries(RuntimeUnitTestSupport INTERFACE GTest::gtest_main)
    set(_runtime_unittest_support_ready ON)
  elseif(TARGET GTest::gtest)
    target_link_libraries(RuntimeUnitTestSupport INTERFACE GTest::gtest)
    set(_runtime_unittest_support_ready ON)
  endif()
endif()

if(NOT _runtime_unittest_support_ready)
  if(DEFINED LLVM_BUILD_MAIN_SRC_DIR AND
     EXISTS "${LLVM_BUILD_MAIN_SRC_DIR}/third-party/unittest/googletest")
    set(_runtime_gtest_dir
        "${LLVM_BUILD_MAIN_SRC_DIR}/third-party/unittest/googletest")
    add_library(RuntimeGTestMain STATIC
      "${_runtime_gtest_dir}/src/gtest-all.cc"
      "${_runtime_gtest_dir}/src/gtest_main.cc")
    target_include_directories(RuntimeGTestMain PUBLIC
      "${_runtime_gtest_dir}/include"
      "${_runtime_gtest_dir}")
    target_compile_definitions(RuntimeGTestMain PUBLIC
      GTEST_NO_LLVM_SUPPORT=1)
    target_link_libraries(RuntimeUnitTestSupport INTERFACE RuntimeGTestMain)
    set(_runtime_unittest_support_ready ON)
  else()
    message(FATAL_ERROR
      "Runtime unit tests need googletest. Tried imported GTest::gtest_main / "
      "GTest::gtest targets, GTest CONFIG package discovery, and "
      "LLVM_BUILD_MAIN_SRC_DIR/third-party/unittest/googletest, but none were "
      "available.")
  endif()
endif()

add_subdirectory(Runtime)
