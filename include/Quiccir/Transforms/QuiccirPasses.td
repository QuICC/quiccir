//===-- QuiccirPasses.td - Quiccir pass definition file ------------*- tablegen -*-===//

#ifndef QUICCIR_QUICCIRPASSES
#define QUICCIR_QUICCIRPASSES

include "mlir/Pass/PassBase.td"

def QuiccirViewDeallocation : Pass<"quiccir-view-deallocation", "func::FuncOp"> {
  let summary = "Adds all required dealloc operations for all allocations in "
                "the input program";
  let description = [{
    This pass implements an algorithm to automatically introduce all required
    deallocation operations for all buffers in the input program. This ensures
    that the resulting program does not have any memory leaks.
  }];
  let constructor = "mlir::quiccir::createViewDeallocationPass()";
  // let dependentDialects = [
  //   "func::FuncDialect"];
}

def QuiccirLowerToCall : Pass<"convert-quiccir-to-call", "ModuleOp"> {
  let summary = "Lower the operations from the quiccir dialect into call ops"
                "external function call";
  let constructor = "mlir::quiccir::createLowerToCallPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect"];
}

def QuiccirLowerAlloc : Pass<"lower-quiccir-alloc", "ModuleOp"> {
  let summary = "Lower the alloc and dealloc operations from the quiccir "
                "dialect into call ops external function call";
  let constructor = "mlir::quiccir::createLowerAllocPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect"];
}

def QuiccirFinalizeViewToLLVM : Pass<"finalize-quiccir-view", "ModuleOp"> {
  let summary = "Lower View to llvm.struct for func/call/return ops";
  let constructor = "mlir::quiccir::createFinalizeViewToLLVMPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect"];
}

def QuiccirSetViewLayout : Pass<"set-quiccir-view-lay", "ModuleOp"> {
  let summary = "Set View layout based on options and ops";
  let constructor = "mlir::quiccir::createSetViewLayoutPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect"];
   let options = [
    Option<"setViewLayout", "integrator0", "std::string", /*default=*/"\"Layout0\"",
           "Set View Layout attribute">
  ];
}

// def QuiccirLowerToLinalg : Pass<"convert-quiccir-to-linalg", "func::FuncOp"> {
//   let summary = "Lower the operations from the quiccir dialect into linalg "
//                 "loops";
//   let constructor = "mlir::quiccir::createLowerToLinalgPass()";
//   let dependentDialects = [
//     "tosa::TosaDialect",
//     "AffineDialect",
//     "linalg::LinalgDialect",
//     "func::FuncDialect",
//     "memref::MemRefDialect",
//     "bufferization::BufferizationDialect"];
// }

#endif // QUICCIR_QUICCIRPASSES
