# AFIR Test Suite
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(AFIR_TEST_DEPENDS
  afir-opt
  FileCheck
  count
  not
)

add_lit_testsuite(check-afir "Running AFIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${AFIR_TEST_DEPENDS}
)
set_target_properties(check-afir PROPERTIES FOLDER "Tests")

# Create a custom target for coverage collection
if(CMAKE_CXX_FLAGS MATCHES "-fprofile-instr-generate")
  add_custom_target(check-afir-coverage
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/coverage/%p.profraw
      ${LLVM_BUILD_DIR}/bin/llvm-lit -sv ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${AFIR_TEST_DEPENDS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running AFIR tests with coverage"
    USES_TERMINAL
  )
  set_target_properties(check-afir-coverage PROPERTIES FOLDER "Tests")
endif()

add_lit_testsuites(AFIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${AFIR_TEST_DEPENDS})
