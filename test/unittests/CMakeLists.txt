add_custom_target(RuntimeUnitTests)
set_target_properties(RuntimeUnitTests PROPERTIES FOLDER "Tests")

add_library(RuntimeUnitTestSupport INTERFACE)
set(_runtime_unittest_support_ready OFF)

function(_runtime_unit_test_add_imported_rpath imported_target)
  get_target_property(_runtime_unit_test_location ${imported_target} IMPORTED_LOCATION)
  if(NOT _runtime_unit_test_location)
    get_target_property(_runtime_unit_test_location ${imported_target} IMPORTED_LOCATION_RELEASE)
  endif()
  if(_runtime_unit_test_location)
    get_filename_component(_runtime_unit_test_dir
      "${_runtime_unit_test_location}" DIRECTORY)
    target_link_options(RuntimeUnitTestSupport INTERFACE
      "-Wl,-rpath,${_runtime_unit_test_dir}")
  endif()
endfunction()

if(TARGET GTest::gtest_main)
  target_link_libraries(RuntimeUnitTestSupport INTERFACE GTest::gtest_main)
  _runtime_unit_test_add_imported_rpath(GTest::gtest_main)
  set(_runtime_unittest_support_ready ON)
elseif(TARGET GTest::gtest)
  add_library(RuntimeUnitTestMain STATIC
    RuntimeUnitTestMain.cpp
  )
  target_link_libraries(RuntimeUnitTestMain PUBLIC GTest::gtest)
  target_link_libraries(RuntimeUnitTestSupport INTERFACE RuntimeUnitTestMain)
  _runtime_unit_test_add_imported_rpath(GTest::gtest)
  set(_runtime_unittest_support_ready ON)
endif()

if(NOT _runtime_unittest_support_ready)
  find_package(GTest CONFIG QUIET)
  if(TARGET GTest::gtest_main)
    target_link_libraries(RuntimeUnitTestSupport INTERFACE GTest::gtest_main)
    _runtime_unit_test_add_imported_rpath(GTest::gtest_main)
    set(_runtime_unittest_support_ready ON)
  elseif(TARGET GTest::gtest)
    add_library(RuntimeUnitTestMain STATIC
      RuntimeUnitTestMain.cpp
    )
    target_link_libraries(RuntimeUnitTestMain PUBLIC GTest::gtest)
    target_link_libraries(RuntimeUnitTestSupport INTERFACE RuntimeUnitTestMain)
    _runtime_unit_test_add_imported_rpath(GTest::gtest)
    set(_runtime_unittest_support_ready ON)
  endif()
endif()

if(NOT _runtime_unittest_support_ready)
  set(_runtime_gtest_candidates
    "${LLVM_BUILD_MAIN_SRC_DIR}/third-party/unittest/googletest"
    "${LLVM_BUILD_MAIN_SRC_DIR}/../third-party/unittest/googletest")
  set(_runtime_gtest_dir "")
  foreach(_candidate IN LISTS _runtime_gtest_candidates)
    if(EXISTS "${_candidate}")
      set(_runtime_gtest_dir "${_candidate}")
      break()
    endif()
  endforeach()

  if(NOT _runtime_gtest_dir STREQUAL "")
    add_library(RuntimeUnitTestMain STATIC
      RuntimeUnitTestMain.cpp
    )
    add_library(RuntimeGTestMain STATIC
      "${_runtime_gtest_dir}/src/gtest-all.cc"
    )
    target_include_directories(RuntimeGTestMain PUBLIC
      "${_runtime_gtest_dir}/include"
      "${_runtime_gtest_dir}")
    target_compile_definitions(RuntimeGTestMain PUBLIC
      GTEST_NO_LLVM_SUPPORT=1)
    target_link_libraries(RuntimeUnitTestMain PUBLIC RuntimeGTestMain)
    target_link_libraries(RuntimeUnitTestSupport INTERFACE RuntimeUnitTestMain)
    set(_runtime_unittest_support_ready ON)
  else()
    message(FATAL_ERROR
      "Runtime unit tests need googletest. Tried imported GTest::gtest_main / "
      "GTest::gtest targets, GTest CONFIG package discovery, and "
      "LLVM_BUILD_MAIN_SRC_DIR/third-party/unittest/googletest or "
      "LLVM_BUILD_MAIN_SRC_DIR/../third-party/unittest/googletest, but none "
      "were available.")
  endif()
endif()

add_subdirectory(Runtime)
