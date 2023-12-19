get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRQuiccir
        # MLIRQuiccirExternalTransforms
        )
add_llvm_executable(quiccir-opt quiccir-opt.cpp)

llvm_update_compile_flags(quiccir-opt)
target_link_libraries(quiccir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(quiccir-opt)
