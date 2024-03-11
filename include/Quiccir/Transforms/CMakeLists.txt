set(LLVM_TARGET_DEFINITIONS QuiccirPasses.td)
mlir_tablegen(QuiccirPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRQuiccirPassesIncGen)