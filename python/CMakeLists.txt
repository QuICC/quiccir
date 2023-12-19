include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_quiccir`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_quiccir.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(QuiccirPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QuiccirPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_quiccir"
  TD_FILE dialects/QuiccirOps.td
  SOURCES
    dialects/quiccir.py
  DIALECT_NAME quiccir)

declare_mlir_python_extension(QuiccirPythonSources.Extension
  MODULE_NAME _quiccirDialects
  ADD_TO_PARENT QuiccirPythonSources
  SOURCES
    QuiccirExtension.cpp
  EMBED_CAPI_LINK_LIBS
    QuiccirCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(QuiccirPythonCAPI
  INSTALL_COMPONENT QuiccirPythonModules
  INSTALL_DESTINATION python_packages/quiccir/mlir_quiccir/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/quiccir/mlir_quiccir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    QuiccirPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(QuiccirPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/quiccir/mlir_quiccir"
  INSTALL_PREFIX "python_packages/quiccir/mlir_quiccir"
  DECLARED_SOURCES
    QuiccirPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    QuiccirPythonCAPI
  )
