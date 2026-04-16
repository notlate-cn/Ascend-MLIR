function(_runtime_unit_test_set_imported_rpath imported_target)
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

set(_runtime_unittest_support_ready OFF)
set(_runtime_gtest_dir "")

set_directory_properties(PROPERTIES
  LABELS "runtime-unittest"
)

if(NOT TARGET GTest::gtest_main AND NOT TARGET GTest::gtest)
  find_package(GTest CONFIG QUIET)
endif()

if(NOT TARGET GTest::gtest_main AND NOT TARGET GTest::gtest)
  find_package(GTest QUIET)
endif()

if(TARGET GTest::gtest_main)
  add_library(RuntimeUnitTestSupport INTERFACE)
  target_link_libraries(RuntimeUnitTestSupport INTERFACE GTest::gtest_main)
  _runtime_unit_test_set_imported_rpath(GTest::gtest_main)
  set(_runtime_unittest_support_ready ON)
elseif(TARGET GTest::gtest)
  add_library(RuntimeUnitTestMain STATIC RuntimeUnitTestMain.cpp)
  target_link_libraries(RuntimeUnitTestMain PUBLIC GTest::gtest)
  add_library(RuntimeUnitTestSupport INTERFACE)
  target_link_libraries(RuntimeUnitTestSupport INTERFACE RuntimeUnitTestMain)
  _runtime_unit_test_set_imported_rpath(GTest::gtest)
  set(_runtime_unittest_support_ready ON)
elseif(DEFINED LLVM_BUILD_MAIN_SRC_DIR)
  set(_runtime_gtest_candidates
    "${LLVM_BUILD_MAIN_SRC_DIR}/third-party/unittest/googletest"
    "${LLVM_BUILD_MAIN_SRC_DIR}/../third-party/unittest/googletest")
  foreach(_candidate IN LISTS _runtime_gtest_candidates)
    if(EXISTS "${_candidate}")
      set(_runtime_gtest_dir "${_candidate}")
      break()
    endif()
  endforeach()

  if(NOT _runtime_gtest_dir STREQUAL "")
    find_package(Threads QUIET)
    add_library(RuntimeUnitTestMain STATIC RuntimeUnitTestMain.cpp)
    add_library(RuntimeGTestMain STATIC
      "${_runtime_gtest_dir}/src/gtest-all.cc")
    target_include_directories(RuntimeGTestMain PUBLIC
      "${_runtime_gtest_dir}/include"
      "${_runtime_gtest_dir}")
    target_compile_definitions(RuntimeGTestMain PUBLIC
      GTEST_NO_LLVM_SUPPORT=1)
    if(TARGET Threads::Threads)
      target_link_libraries(RuntimeGTestMain PUBLIC Threads::Threads)
    endif()
    target_link_libraries(RuntimeUnitTestMain PUBLIC RuntimeGTestMain)
    add_library(RuntimeUnitTestSupport INTERFACE)
    target_link_libraries(RuntimeUnitTestSupport INTERFACE RuntimeUnitTestMain)
    set(_runtime_unittest_support_ready ON)
  endif()
endif()

if(_runtime_unittest_support_ready)
  add_custom_target(RuntimeUnitTests)
  set_target_properties(RuntimeUnitTests PROPERTIES FOLDER "Tests")
  add_subdirectory(Runtime)
else()
  message(STATUS "Skipping runtime unit tests because googletest is unavailable.")
endif()
