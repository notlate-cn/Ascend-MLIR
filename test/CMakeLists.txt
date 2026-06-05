# Ascend-MLIR test suite
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ASCEND_TEST_DEPENDS
  ascend-mlir-opt
  FileCheck
  count
  not
)

set(ASCEND_LIT "${LLVM_EXTERNAL_LIT}")
if(NOT ASCEND_LIT)
  set(ASCEND_LIT "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
endif()

add_custom_target(check-ascend-conversion
  COMMAND "${ASCEND_LIT}" -v --filter "ascend-"
          --filter-out "phase5-transformer-dynamic-smoke"
          "${CMAKE_CURRENT_BINARY_DIR}/Conversion"
  DEPENDS ${ASCEND_TEST_DEPENDS}
)
set_target_properties(check-ascend-conversion PROPERTIES FOLDER "Tests")

function(add_afir_lit_directory relative_dir)
  set(lit_suite "${CMAKE_CURRENT_BINARY_DIR}/${relative_dir}")
  string(REPLACE "/" "-" name_dashes "${relative_dir}")
  string(TOLOWER "AFIR-${name_dashes}" name_var)
  add_lit_target("check-${name_var}" "Running lit suite ${lit_suite}"
    ${lit_suite}
    DEPENDS ${AFIR_TEST_DEPENDS}
  )
  set_target_properties("check-${name_var}" PROPERTIES FOLDER "Tests")
endfunction()

if(ASCEND_ENABLE_AFIR)
  set(AFIR_TEST_DEPENDS
    afir-opt
    ascend-mlir-opt
    ascend-mlir-translate
    ascend-debug
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

  foreach(lit_dir
      Conversion
      Dialect
      Dialect/AFIR
      Dialect/AFIR/Transforms
      Target
      tools
      tools/diagnostics
      tools/examples
      tools/longrun
      tools/runtime)
    add_afir_lit_directory(${lit_dir})
  endforeach()

  if(AFIR_ENABLE_BINDING_PYTHON)
    add_afir_lit_directory(python)
  endif()
endif()

function(add_ascend_lit_alias alias_target legacy_target)
  if(TARGET ${legacy_target} AND NOT TARGET ${alias_target})
    add_custom_target(${alias_target})
    add_dependencies(${alias_target} ${legacy_target})
    set_target_properties(${alias_target} PROPERTIES FOLDER "Tests")
  endif()
endfunction()

if(TARGET check-afir)
  add_ascend_lit_alias(check-ascend check-afir)
else()
  add_ascend_lit_alias(check-ascend check-ascend-conversion)
endif()
add_ascend_lit_alias(check-ascend-tools check-afir-tools)
add_ascend_lit_alias(check-ascend-tools-diagnostics check-afir-tools-diagnostics)
add_ascend_lit_alias(check-ascend-tools-examples check-afir-tools-examples)
add_ascend_lit_alias(check-ascend-tools-longrun check-afir-tools-longrun)
add_ascend_lit_alias(check-ascend-tools-runtime check-afir-tools-runtime)

foreach(lit_tool_target
    check-afir
    check-afir-tools-examples
    check-afir-tools-runtime)
  if(TARGET ${lit_tool_target})
    add_dependencies(${lit_tool_target}
      mix-compiler
      runtime-session)
  endif()
endforeach()

add_subdirectory(unittests)

function(add_afir_ctest_label_target target_name label)
  if(TARGET RuntimeUnitTests)
    add_custom_target(${target_name}
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L "${label}"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    add_dependencies(${target_name} RuntimeUnitTests)
  else()
    add_custom_target(${target_name}
      COMMAND ${CMAKE_COMMAND} -E echo "Skipping runtime unit tests because googletest is unavailable."
    )
  endif()
  set_target_properties(${target_name} PROPERTIES FOLDER "Tests")
endfunction()

add_afir_ctest_label_target(check-unittests runtime-unittest)
add_afir_ctest_label_target(check-afir-unittests runtime-unittest)
add_afir_ctest_label_target(check-afir-unittests-conversion conversion-unittest)
add_afir_ctest_label_target(check-afir-unittests-runtime runtime-component-unittest)
add_afir_ctest_label_target(check-afir-unittests-target target-unittest)
