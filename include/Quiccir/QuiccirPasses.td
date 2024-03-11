//===-- QuiccirPasses.td - Quiccir pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
