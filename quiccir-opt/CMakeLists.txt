option(QUICCIR_BUILD_OPT "Build quiccir-opt." ON)
if(NOT QUICCIR_BUILD_OPT)
  return()
endif()
message(STATUS "Building quiccir-opt")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIROptLib
        MLIRQuiccirDialect
        MLIRQuiccirPipelines
        # MLIRQuiccirExternalTransforms
        )
add_llvm_executable(quiccir-opt quiccir-opt.cpp)

llvm_update_compile_flags(quiccir-opt)
target_link_libraries(quiccir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(quiccir-opt)
