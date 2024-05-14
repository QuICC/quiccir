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

def QuiccirSetViewLayout : Pass<"set-quiccir-view-lay", "func::FuncOp"> {
  let summary = "Set View layout based on options and ops";
  let constructor = "mlir::quiccir::createSetViewLayoutPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect"];
   let options = [
    ListOption<"layZero", "lay-ppp2mpp", "std::string",
           "Set View Layout pair for first integrator">,
    ListOption<"layOne", "lay-pmp2mmp", "std::string",
           "Set View Layout pair for second integrator">,
    ListOption<"layTwo", "lay-pmm2mmm", "std::string",
           "Set View Layout pair for second integrator">
  ];
}

def QuiccirSetDimensions : Pass<"set-quiccir-dims", "func::FuncOp"> {
  let summary = "Set View layout based on options and ops";
  let constructor = "mlir::quiccir::createSetDimensionsPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect"];
   let options = [
    ListOption<"physDim", "phys", "int64_t",
           "Set Dimensions for physical space">,
    ListOption<"modsDim", "mods", "int64_t",
           "Set Dimensions for modal space">,
  ];
}

def QuiccirViewWrapper : Pass<"quiccir-view-wrapper", "ModuleOp"> {
  let summary = "Insert view wrapper";
  let constructor = "mlir::quiccir::createViewWrapperPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect"];
   let options = [
    Option<"args", "args", "int64_t", /*default=*/"",
           "Set number of input arguments">,
    Option<"rets", "rets", "int64_t", /*default=*/"",
           "Set number of return values">,
    ListOption<"argsDim", "dim-args", "int64_t",
           "Set Dimensions for arguments">,
    ListOption<"retsDim", "dim-rets", "int64_t",
           "Set Dimensions for returns">,
    Option<"layArgs", "lay-args", "std::string", /*default=*/"",
           "Set View Layout for arguments">,
    Option<"laRets", "lay-rets", "std::string", /*default=*/"",
           "Set View Layout for returns">
  ];
}

#endif // QUICCIR_QUICCIRPASSES
