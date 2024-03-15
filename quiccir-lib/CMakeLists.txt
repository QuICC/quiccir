get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
    MLIRExecutionEngine
)

add_library(QuiccirJITLib INTERFACE)
target_link_libraries(QuiccirJITLib
    INTERFACE
        ${LIBS}
        MLIRQuiccirDialect
)

# Attach include folders
target_include_directories(QuiccirJITLib
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_include_directories(QuiccirJITLib INTERFACE ${LLVM_INCLUDE_DIRS})
target_include_directories(QuiccirJITLib INTERFACE ${MLIR_INCLUDE_DIRS})
target_link_directories(QuiccirJITLib INTERFACE ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(QuiccirJITLib INTERFACE ${LLVM_DEFINITIONS})

# Export targets
include(AddQuiccir)
add_quiccir_library_install(QuiccirJITLib)
add_quiccir_library_install(MLIRQuiccirDialect)
add_quiccir_library_install(MLIRQuiccirPasses)
add_quiccir_library_install(MLIRQuiccirPipelines)
