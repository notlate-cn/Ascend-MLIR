# Version configuration

# Read version from VERSION file
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VERSION_CONTENTS)

foreach(line ${VERSION_CONTENTS})
  if(line MATCHES "^ASCEND_MLIR_VERSION=(.*)$")
    set(ASCEND_MLIR_VERSION ${CMAKE_MATCH_1})
  elseif(line MATCHES "^ASCEND_MLIR_VERSION_MAJOR=(.*)$")
    set(ASCEND_MLIR_VERSION_MAJOR ${CMAKE_MATCH_1})
  elseif(line MATCHES "^ASCEND_MLIR_VERSION_MINOR=(.*)$")
    set(ASCEND_MLIR_VERSION_MINOR ${CMAKE_MATCH_1})
  elseif(line MATCHES "^ASCEND_MLIR_VERSION_PATCH=(.*)$")
    set(ASCEND_MLIR_VERSION_PATCH ${CMAKE_MATCH_1})
  endif()
endforeach()

message(STATUS "Ascend-MLIR Version: ${ASCEND_MLIR_VERSION}")

# Generate version header
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/version.h.in"
  "${CMAKE_BINARY_DIR}/include/ascend-mlir/version.h"
  @ONLY
)
