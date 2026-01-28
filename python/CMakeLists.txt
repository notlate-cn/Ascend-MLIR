include(AddMLIRPython)

declare_mlir_python_sources(AFIRPythonSources)

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT AFIRPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    TD_FILE dialects/afir.td
    SOURCES 
      dialects/afir.py
    SOURCES_GLOB dialects/afir_dialect/*.py
    DIALECT_NAME afir
    GEN_ENUM_BINDINGS_TD_FILE
    ../include/Dialect/AFIR/Core/Enums.td
)

declare_mlir_python_extension(AFIRPythonSources.NanobindExtension
  MODULE_NAME _afir
  ADD_TO_PARENT AFIRPythonSources
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    AFIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    AFIRCAPI
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(AFIRPythonCAPI
  INSTALL_COMPONENT AFIRPythonModules
  INSTALL_DESTINATION python_packages/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/mlir/_mlir_libs"
  DECLARED_SOURCES
    AFIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)


################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(AFIRPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/mlir"
  INSTALL_PREFIX "python_packages/mlir/"
  DECLARED_SOURCES
    AFIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    AFIRPythonCAPI
  )
