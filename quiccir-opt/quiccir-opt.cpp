//===- quiccir-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Quiccir/QuiccirDialect.h"
#include "Quiccir/QuiccirPasses.h"

int main(int argc, char **argv) {
  // MLIR passes
  mlir::registerAllPasses();
  // Quiccir passes
  mlir::quiccir::registerQuiccirLowerToCallPass();
  mlir::quiccir::registerQuiccirLowerAllocPass();
  mlir::quiccir::registerQuiccirViewDeallocationPass();
  mlir::quiccir::registerQuiccirFinalizeViewToLLVMPass();
  // mlir::quiccir::registerQuiccirLowerToAffinePass();
  // mlir::quiccir::registerQuiccirLowerToLinalgPass();
  /// Custom non-Quiccir passes
  // mlir::registerMergeAffineParallelLoopPass();

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);

  registry.insert<mlir::quiccir::QuiccirDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Quiccir optimizer driver\n", registry));
}
