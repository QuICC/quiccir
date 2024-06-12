//===- Passes.cpp - Quiccir dialect pipelines -----------------*- C++ -*-===//
//
//   Copyright (c) 2024,
//   Earth and Planetary Magnetism group, ETH Zurich
//
//===---------------------------------------------------------------------===//

#include "Quiccir/Pipelines/Passes.h"
#include "Quiccir/Transforms/QuiccirPasses.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::quiccir;

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void mlir::quiccir::quiccLibCallPipelineBuilder(OpPassManager &pm){
  // Lower to view rapresentation
  pm.addPass(mlir::quiccir::createLowerToCallPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Convert meta ops
  mlir::OpPassManager &nestedFuncPm = pm.nest<mlir::func::FuncOp>();
  nestedFuncPm.addPass(mlir::quiccir::createConvertToLLVMPass());

  // Insert dealloc
  nestedFuncPm.addPass(mlir::quiccir::createViewDeallocationPass());

  // Lower to llvm
  pm.addPass(mlir::quiccir::createLowerAllocPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::quiccir::createFinalizeViewToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void mlir::quiccir::registerQuiccirPipelines() {
  PassPipelineRegistration<>(
      "quiccirToQuICC",
      "The standard pipeline for lowering quiccir to QuICC library",
      quiccLibCallPipelineBuilder);
}
