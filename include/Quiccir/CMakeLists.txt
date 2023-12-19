add_mlir_dialect(QuiccirOps quiccir)
add_mlir_doc(QuiccirDialect QuiccirDialect Quiccir/ -gen-dialect-doc)
add_mlir_doc(QuiccirOps QuiccirOps Quiccir/ -gen-op-doc)


# Declare a dialect in the include directory
function(add_mlir_pass pass name)
  set(LLVM_TARGET_DEFINITIONS ${pass}.td)
  mlir_tablegen(${pass}.h.inc -gen-pass-decls -name ${name})
  # mlir_tablegen(${pass}.capi.h.inc -gen-pass-capi-header --prefix ${name})
  # mlir_tablegen(${pass}.capi.cpp.inc -gen-pass-capi-impl --prefix ${name})
  add_public_tablegen_target(MLIR${pass}IncGen)
  add_dependencies(mlir-headers MLIR${pass}IncGen)
endfunction()

add_mlir_pass(QuiccirPasses Quiccir)

add_mlir_doc(QuiccirPasses QuiccirPasses Quiccir/ -gen-pass-doc)
