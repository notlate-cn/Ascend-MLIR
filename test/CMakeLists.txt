# AFIR Test Suite
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(AFIR_TEST_DEPENDS
  afir-opt
  afir-translate
  FileCheck
  count
  not
)

add_lit_testsuite(check-afir "Running AFIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${AFIR_TEST_DEPENDS}
)
set_target_properties(check-afir PROPERTIES FOLDER "Tests")

add_lit_testsuites(AFIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${AFIR_TEST_DEPENDS})

add_subdirectory(unittests)

if(TARGET RuntimeUnitTests)
  add_custom_target(check-unittests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "runtime-unittest"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  add_dependencies(check-unittests RuntimeUnitTests)
else()
  add_custom_target(check-unittests
    COMMAND ${CMAKE_COMMAND} -E echo "Skipping runtime unit tests because googletest is unavailable."
  )
endif()
set_target_properties(check-unittests PROPERTIES FOLDER "Tests")
