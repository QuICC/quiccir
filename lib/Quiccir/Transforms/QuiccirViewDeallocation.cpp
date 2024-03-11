//====- QuiccirViewDeallocation.cpp - Inserting Quiccir Dealloc -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that inserts DeallocOps.
// At the moment, for simplicity, we assume that the temporary buffers are not
// passed as arguments to other blocks (i.e. no control flow allowed).
// Otherwise a similar analysis as in BufferizationDeallocation would be needed.
//
//===----------------------------------------------------------------------===//

#include "Quiccir/QuiccirPassDetail.h"

#include "Quiccir/QuiccirDialect.h"
#include "Quiccir/QuiccirOps.h"
#include "Quiccir/QuiccirPasses.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::quiccir;

namespace {

//===----------------------------------------------------------------------===//
// QuiccirViewDeallocation
//===----------------------------------------------------------------------===//

LogicalResult deallocateBuffers(Operation *op) {
  OpBuilder builder(op);
  auto loc = op->getLoc();
  // ret val
  Value view = *op->result_begin();
  // find pos last user (assumes no copies, i.e. no control flow)
  if (view.getUsers().empty()) {
    op->emitWarning() << "trying to deallocate unused view";
    return success();
  }
  Operation *lastUser = *(view.user_begin()); // single link list
  builder.setInsertionPointAfter(lastUser);
  builder.create<quiccir::DeallocOp>(loc, view);
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
//  QuiccirViewDeallocationPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the quiccir operations
namespace {
struct  QuiccirViewDeallocationPass
    : public QuiccirViewDeallocationBase<QuiccirViewDeallocationPass> {
  void runOnOperation() override;
};
} // namespace

void  QuiccirViewDeallocationPass::runOnOperation() {

  // Walk from root func
  WalkResult result = getOperation()->walk([&](quiccir::AllocOp op) {
      // llvm::errs() << op.getOperationName() << '\n';
      // return deallocateBuffers(op);
      if (failed(deallocateBuffers(op)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

  if (result.wasInterrupted())
    signalPassFailure();
}

/// Create a pass for lowering operations to library calls
std::unique_ptr<Pass> mlir::quiccir::createViewDeallocationPass() {
  return std::make_unique<QuiccirViewDeallocationPass>();
}
