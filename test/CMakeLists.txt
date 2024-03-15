option(QUICCIR_BUILD_TEST "Build quiccir tests." ON)
if(NOT QUICCIR_BUILD_TEST)
  return()
endif()
message(STATUS "Building quiccir tests")

llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

set(MLIR_LIB_DIR "${MLIR_INSTALL_PREFIX}/lib")

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(QUICCIR_TEST_DEPENDS
        FileCheck count not
        quiccir-capi-test
        quiccir-opt
        quiccir-translate
        )

# The native target may not be enabled, in this case we won't
# run tests that involves executing on the host: do not build
# useless binaries.
if(TARGET ${LLVM_NATIVE_ARCH})
  list(APPEND MLIR_TEST_DEPENDS
    mlir-cpu-runner
    mlir_c_runner_utils
    mlir_runner_utils
  )
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND QUICCIR_TEST_DEPENDS QuiccirPythonModules)
endif()

add_lit_testsuite(check-quiccir "Running the quiccir regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${QUICCIR_TEST_DEPENDS}
        )
set_target_properties(check-quiccir PROPERTIES FOLDER "Tests")

add_lit_testsuites(QUICCIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${QUICCIR_TEST_DEPENDS})

add_subdirectory(CAPI)
