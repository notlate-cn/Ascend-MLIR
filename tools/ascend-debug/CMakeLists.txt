set(ASCEND_DEBUG_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ascend-debug.py")
set(ASCEND_DEBUG_OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ascend-debug")
set(ASCEND_DEBUG_PACKAGE_OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ascend_debug")
set(ASCEND_DEBUG_PACKAGE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/__init__.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/collect.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/diff.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/layout.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/locate.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/memory.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/open_view.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug/runner.py"
)

add_custom_command(
  OUTPUT "${ASCEND_DEBUG_OUTPUT}" "${ASCEND_DEBUG_PACKAGE_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  COMMAND "${CMAKE_COMMAND}" -E copy "${ASCEND_DEBUG_SOURCE}" "${ASCEND_DEBUG_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${ASCEND_DEBUG_PACKAGE_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/ascend_debug"
          "${ASCEND_DEBUG_PACKAGE_OUTPUT}"
  COMMAND chmod +x "${ASCEND_DEBUG_OUTPUT}"
  DEPENDS "${ASCEND_DEBUG_SOURCE}"
          ${ASCEND_DEBUG_PACKAGE_SOURCES}
  COMMENT "Installing ascend-debug Python CLI"
  VERBATIM
)

add_custom_target(ascend-debug ALL
  DEPENDS "${ASCEND_DEBUG_OUTPUT}" "${ASCEND_DEBUG_PACKAGE_OUTPUT}"
)
