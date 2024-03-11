get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIROptLib
        MLIRQuiccir
        MLIRExecutionEngine
        quiccir_external
        )
add_llvm_executable(quiccir-miniapp quiccir-miniapp.cpp)

llvm_update_compile_flags(quiccir-miniapp)
target_link_libraries(quiccir-miniapp PRIVATE ${LIBS})

mlir_check_all_link_libraries(quiccir-miniapp)
