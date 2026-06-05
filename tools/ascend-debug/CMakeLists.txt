set(ASCEND_DEBUG_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ascend-debug.py")
set(ASCEND_DEBUG_OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ascend-debug")
set(ASCEND_DEBUG_PACKAGE_OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ascend_debug")
set(ASCEND_DEBUG_STAMP "${CMAKE_CURRENT_BINARY_DIR}/ascend-debug.stamp")
set(ASCEND_DEBUG_PACKAGE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/__init__.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/collect.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/debug_graph.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/diff.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/kernel_dag.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/layout.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/locate.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/memory.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/open_view.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/runner.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/run_case.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/serve.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/stage_graph.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/ui_text.py"
)

add_custom_command(
  OUTPUT "${ASCEND_DEBUG_STAMP}"
  BYPRODUCTS "${ASCEND_DEBUG_OUTPUT}" "${ASCEND_DEBUG_PACKAGE_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  COMMAND "${CMAKE_COMMAND}" -E copy "${ASCEND_DEBUG_SOURCE}" "${ASCEND_DEBUG_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${ASCEND_DEBUG_PACKAGE_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug"
          "${ASCEND_DEBUG_PACKAGE_OUTPUT}"
  COMMAND chmod +x "${ASCEND_DEBUG_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${ASCEND_DEBUG_STAMP}"
  DEPENDS "${ASCEND_DEBUG_SOURCE}"
          ${ASCEND_DEBUG_PACKAGE_SOURCES}
  COMMENT "Installing ascend-debug Python CLI"
  VERBATIM
)

add_custom_target(ascend-debug ALL
  DEPENDS "${ASCEND_DEBUG_STAMP}"
)
